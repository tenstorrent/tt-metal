# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host only: the whole-model golden cache round-trips. A second run loads from disk instead of
recomputing, a changed ReferenceCacheKey field (seed, depth, width, length) forces a miss, the key is
frozen, and ``MISTRAL_REF_CACHE_REQUIRED=1`` turns a miss into an error instead of a CPU recompute."""

import dataclasses

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.config import MistralMediumConfig
from models.demos.mistral_medium_3_5_128b.reference import golden
from models.demos.mistral_medium_3_5_128b.reference.model import random_state_dict

CFG = MistralMediumConfig().reduced(
    num_hidden_layers=3,
    hidden_size=256,
    intermediate_size=512,
    num_attention_heads=4,
    num_key_value_heads=2,
    vocab_size=512,
)


def test_golden_cache_round_trip(tmp_path, monkeypatch, expect_error):
    monkeypatch.setenv("MISTRAL_REF_CACHE", str(tmp_path))
    monkeypatch.delenv("MISTRAL_REF_CACHE_REQUIRED", raising=False)
    key, ids, snaps, kv = golden.golden_forward(CFG, 96, seed=0)
    assert golden.cache_path(key).is_file()
    assert len(snaps) == CFG.num_hidden_layers + 3 and len(kv) == CFG.num_hidden_layers

    # The cached golden is exactly a fresh reference run.
    fresh_snaps, fresh_kv = golden.run_reference(CFG, random_state_dict(CFG, 0), ids)
    assert all(torch.equal(a, b) for a, b in zip(snaps, fresh_snaps))
    assert all(torch.equal(a, b) for a, b in zip(kv, fresh_kv))

    # Second run: served from disk, never recomputed.
    def no_recompute(*args, **kwargs):
        raise AssertionError("golden recomputed on a cache hit")

    monkeypatch.setattr(golden, "run_reference", no_recompute)
    key2, _, snaps2, kv2 = golden.golden_forward(CFG, 96, seed=0)
    assert key2 == key
    assert all(torch.equal(a, b) for a, b in zip(snaps, snaps2)) and all(torch.equal(a, b) for a, b in zip(kv, kv2))

    # Every output-changing field yields a different file, so nothing stale is reused.
    variants = [
        golden.make_key(CFG, 96, seed=1),
        golden.make_key(CFG, 128, seed=0),
        golden.make_key(CFG.reduced(num_hidden_layers=4), 96, seed=0),
        golden.make_key(CFG.reduced(hidden_size=512), 96, seed=0),
        golden.make_key(CFG, 96, seed=0, weight_type="pretrained"),
    ]
    names = {str(k) for k in variants}
    assert str(key) not in names and len(names) == len(variants)
    assert not any(golden.cache_path(k).exists() for k in variants)
    with expect_error(dataclasses.FrozenInstanceError, "seed"):
        key.seed = 3

    # CI mode: a miss fails loudly instead of burning CPU time.
    monkeypatch.setenv("MISTRAL_REF_CACHE_REQUIRED", "1")
    with expect_error(FileNotFoundError, "Reference cache not found"):
        golden.golden_forward(CFG, 96, seed=7)


@pytest.mark.parametrize("seq_len", [64])
def test_golden_snapshots_are_bf16(tmp_path, monkeypatch, seq_len):
    monkeypatch.setenv("MISTRAL_REF_CACHE", str(tmp_path))
    _, _, snaps, kv = golden.golden_forward(CFG, seq_len, seed=0)
    assert all(t.dtype == torch.bfloat16 for t in snaps + kv)
