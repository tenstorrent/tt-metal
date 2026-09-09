# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M1 — the whole-model golden cache round-trips, and a changed key forces a miss.

Host only. No pattern in the repo to copy; authored for this bring-up.

A full-depth CPU forward of an 8B model costs minutes, so several tests share one cached result.
That makes the cache load-bearing in a way the cheap per-module goldens are not, and gives it two
properties worth testing rather than assuming:

1. **A second run LOADS instead of recomputing.** Otherwise the cache is decoration and every test
   pays full price.
2. **A changed key field MISSES rather than silently reusing a stale result.** This is the property
   `ReferenceCacheKey` is frozen to guarantee. It is the one that turns a wrong number into a
   failure instead of a plausible-looking measurement — change the ISL, the layer count or the
   weight type and you must not get the previous run's tensors back.

There is also a third behaviour: on a miss with `allow_recompute=False` the load must FAIL LOUDLY
rather than burn an hour regenerating. That is what CI wants — the pattern
`deepseek_v3_d_p/tests/test_mla.py:293` uses.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from models.demos.llama_3_1_8b_d_p.reference.config import LlamaConfigConstants
from models.demos.llama_3_1_8b_d_p.reference.model_golden import (
    N_ROUTED_EXPERTS,
    VARIANT,
    cache_key,
    get_reference,
)

# Tiny: this file tests the CACHE, not the model. A real-dims forward here would defeat the point.
TINY = LlamaConfigConstants(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=256,
    vocab_size=64,
)
ISL = 32


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv(VARIANT.ref_cache_env, str(tmp_path))


def _ids():
    torch.manual_seed(0)
    return torch.randint(0, TINY.vocab_size, (1, ISL))


def test_reference_round_trips_through_the_cache(tmp_path):
    """First call computes and saves; the second returns the same tensors from disk."""
    ids = _ids()
    snapshots, kv = get_reference(TINY, ids, weight_type="random", input_source="random")
    files = list(tmp_path.glob("*.pt"))
    assert len(files) == 1, f"expected exactly one cache file, got {[f.name for f in files]}"

    again_snapshots, again_kv = get_reference(TINY, ids, weight_type="random", input_source="random")
    assert len(again_snapshots) == len(snapshots) == TINY.num_hidden_layers
    for a, b in zip(snapshots, again_snapshots):
        assert torch.equal(a, b), "cached hidden-state snapshots differ from the originals"
    for (ka, va), (kb, vb) in zip(kv, again_kv):
        assert torch.equal(ka, kb) and torch.equal(va, vb), "cached KV differs from the original"
    assert list(tmp_path.glob("*.pt")) == files, "the second call wrote a new file instead of loading"


def test_a_miss_can_be_made_fatal(tmp_path):
    """`allow_recompute=False` must raise rather than silently pay for a full forward.

    In CI a cache miss should be a clear failure naming the missing artifact, not a timeout.
    """
    with pytest.raises(FileNotFoundError, match="recompute is disabled"):
        get_reference(TINY, _ids(), weight_type="random", input_source="random", allow_recompute=False)


@pytest.mark.parametrize(
    "field,value",
    [
        ("weight_type", "pretrained"),
        ("input_source", "abc_1k"),
        ("isl_total", ISL * 2),
        ("num_layers", TINY.num_hidden_layers + 1),
        ("padding_side", "left"),
    ],
)
def test_changed_key_field_forces_a_miss(field, value):
    """Every field of the key must change the filename, or a stale result is reused silently."""
    base = cache_key(
        weight_type="random", input_source="random", isl_total=ISL, num_layers=TINY.num_hidden_layers
    )
    assert str(replace(base, **{field: value})) != str(base), f"{field} does not change the cache filename"


def test_key_is_frozen_and_dense():
    """The key is immutable, and this dense model records 0 experts as a real value."""
    base = cache_key(weight_type="random", input_source="random", isl_total=ISL, num_layers=2)
    assert base.n_routed_experts == N_ROUTED_EXPERTS == 0
    with pytest.raises((AttributeError, TypeError)):
        base.isl_total = 1  # type: ignore[misc]


def test_cache_is_keyed_apart_from_a_moe_sibling():
    """The expert count is in the key, so this model's cache cannot collide with a MoE model's."""
    dense = cache_key(weight_type="random", input_source="random", isl_total=ISL, num_layers=2)
    moe = replace(dense, n_routed_experts=256)
    assert str(dense) != str(moe)
