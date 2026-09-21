# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""D1/M1: the golden cache round-trips, and a changed key forces a miss.

No pattern in the repo for this — authored here. It is the test that makes "compute once" safe:
the failure it guards against is not a crash but a *silent* reuse of a stale reference after a
config, seed or chunking change, which turns every downstream PCC number into a measurement of
the wrong thing.

Host only.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from models.demos.qwen_3_8_27b_d_p.reference.golden import (
    GoldenCacheKey,
    build_random_reference,
    cached_reference,
    run_reference,
)
from models.demos.qwen_3_8_27b_d_p.reference.modeling import AttentionCapture, GdnCapture
from models.demos.qwen_3_8_27b_d_p.tests.host.hf_bridge import reduced_config

ISL = 128


@pytest.fixture
def cfg():
    return reduced_config(num_hidden_layers=4)


@pytest.fixture
def input_ids(cfg):
    torch.manual_seed(31)
    return torch.randint(0, cfg.vocab_size, (1, ISL))


@pytest.fixture
def key(cfg):
    return GoldenCacheKey(
        weight_type="random",
        weight_seed=0,
        input_source="random_ids",
        input_seed=31,
        isl_total=ISL,
        num_layers=cfg.num_hidden_layers,
        hidden_size=cfg.hidden_size,
        vocab_size=cfg.vocab_size,
        chunking="one_shot",
        dtype="float16",
    )


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("QWEN35_REF_CACHE", str(tmp_path / "ref_cache"))


def test_round_trip_is_exact(cfg, input_ids, key):
    """Second call loads from disk; the loaded tensors are bit-identical to the computed ones."""
    build = lambda: build_random_reference(cfg, seed=0)  # noqa: E731
    first = cached_reference(cfg, input_ids, key, build=build)

    def _fail() -> None:
        raise AssertionError("the second call recomputed instead of loading from the cache")

    second = cached_reference(cfg, input_ids, key, build=_fail)

    assert torch.equal(first.final_hidden, second.final_hidden)
    assert len(first.hidden_per_layer) == len(second.hidden_per_layer) == cfg.num_hidden_layers
    for a, b in zip(first.hidden_per_layer, second.hidden_per_layer):
        assert torch.equal(a, b)


def test_round_trip_preserves_both_state_kinds(cfg, input_ids, key):
    """A hybrid model's cache carries two shapes of carried state; both must survive the trip."""
    build = lambda: build_random_reference(cfg, seed=0)  # noqa: E731
    first = cached_reference(cfg, input_ids, key, build=build)
    second = cached_reference(cfg, input_ids, key, build=lambda: pytest.fail("recomputed"))

    for i in cfg.full_attention_layers:
        assert isinstance(second.states[i], AttentionCapture)
        assert torch.equal(first.layer_kv(i)[0], second.layer_kv(i)[0])
        assert torch.equal(first.layer_kv(i)[1], second.layer_kv(i)[1])
    for i in cfg.linear_attention_layers:
        assert isinstance(second.states[i], GdnCapture)
        assert torch.equal(first.layer_gdn_state(i)[0], second.layer_gdn_state(i)[0])
        assert torch.equal(first.layer_gdn_state(i)[1], second.layer_gdn_state(i)[1])


@pytest.mark.parametrize(
    "field, value",
    [
        ("weight_seed", 1),
        ("input_seed", 99),
        ("isl_total", 256),
        ("num_layers", 8),
        ("chunking", "chunked64"),
        ("dtype", "bfloat16"),
        ("weight_type", "pretrained"),
    ],
)
def test_changed_key_field_forces_a_miss(cfg, input_ids, key, field, value, expect_error):
    """Every field must change the filename. A field that does not is a silent stale-hit bug."""
    cached_reference(cfg, input_ids, key, build=lambda: build_random_reference(cfg, seed=0))
    changed = dataclasses.replace(key, **{field: value})
    assert str(changed) != str(key)
    with expect_error(AssertionError, "no cached reference"):
        cached_reference(cfg, input_ids, changed, build=lambda: pytest.fail("should not build"), require_cached=True)


def test_key_is_frozen(key, expect_error):
    with expect_error(dataclasses.FrozenInstanceError, "cannot assign to field"):
        key.isl_total = 1  # type: ignore[misc]


def test_require_cached_asserts_rather_than_recomputing(cfg, input_ids, key, expect_error):
    with expect_error(AssertionError, "no cached reference"):
        cached_reference(cfg, input_ids, key, build=lambda: pytest.fail("should not build"), require_cached=True)


def test_chunked_reference_matches_one_shot(cfg, input_ids):
    """``run_reference`` with a chunk size threads both carried-state kinds; the concatenated
    per-layer hidden states must equal the one-shot run. The host-side statement of P2's goal."""
    from models.common.utility_functions import comp_pcc

    model = build_random_reference(cfg, seed=0)
    one_shot = run_reference(model, input_ids)
    chunked = run_reference(model, input_ids, chunk_size=ISL // 2)
    passing, pcc = comp_pcc(one_shot.final_hidden.float(), chunked.final_hidden.float(), 0.999)
    assert passing, f"chunked reference diverged from one-shot: {pcc}"
    for i in cfg.full_attention_layers:
        # The chunked run must have accumulated the WHOLE prefix, not just the last chunk.
        assert one_shot.layer_kv(i)[0].shape == chunked.layer_kv(i)[0].shape
