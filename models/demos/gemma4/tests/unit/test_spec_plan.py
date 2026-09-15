# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the model side of the plugin speculative contract (``spec_plan``).

See vllm-tt-plugin#110 s.2. ``spec_plan`` is a config-time query: the plugin
calls it several times per launch (engine process, the engine-core re-run of
``VllmConfig.__post_init__``, and ``TTWorker.init_device``) before any device
exists, so these are host-only tests and must stay that way.

What is pinned here is the CONTRACT, not the numbers: that concurrency above
one request is refused, that K is a set rather than a range, that a missing
drafter checkpoint is refused at config time rather than at the first request,
and that the reported fixed cost actually tracks the decoder's allocations.
"""

import json
import os

import pytest
from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

from models.demos.gemma4.tt.generator_vllm import Gemma4DFlashForCausalLM

_VERIFY = 5


@pytest.fixture
def drafter_snapshot(tmp_path, monkeypatch):
    """A minimal drafter checkpoint dir — only ``config.json`` is read."""
    cfg = {
        "num_hidden_layers": 5,
        "hidden_size": 5376,
        "head_dim": 128,
        "num_key_value_heads": 8,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("GEMMA4_DFLASH_DRAFTER", str(tmp_path))
    monkeypatch.setenv("GEMMA4_DFLASH_VERIFY", str(_VERIFY))
    monkeypatch.setenv("MESH_DEVICE", "P150x8")
    monkeypatch.delenv("GEMMA4_DFLASH_REPLICATED", raising=False)
    monkeypatch.delenv("GEMMA4_DFLASH_CTX_CAP", raising=False)
    return tmp_path, cfg


def test_rejects_concurrency_above_one(drafter_snapshot):
    """dFlash captures its fused verify at B=1, so B>1 cannot speculate."""
    out = Gemma4DFlashForCausalLM.spec_plan(None, max_num_seqs=2, requested_k=_VERIFY)
    assert isinstance(out, SpecReject)
    assert "single-stream" in out.reason
    # Nothing is speculable at this concurrency, so no K is offered.
    assert out.supported_k == ()


def test_k_is_a_set_not_a_range(drafter_snapshot):
    """A narrower K would need its own packed-verify capture, so it is refused
    and the one supported width is named."""
    out = Gemma4DFlashForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY - 2)
    assert isinstance(out, SpecReject)
    assert out.supported_k == (_VERIFY,)


def test_missing_drafter_is_refused_at_config_time(drafter_snapshot, monkeypatch):
    """Without this the drafter resolves lazily and a missing checkpoint kills
    the engine at the FIRST REQUEST instead (EngineDeadError, no traceback)."""
    monkeypatch.setenv("GEMMA4_DFLASH_DRAFTER", "/nonexistent/dflash-drafter")
    out = Gemma4DFlashForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY)
    assert isinstance(out, SpecReject)
    assert out.supported_k == (_VERIFY,)
    # The message has to be actionable: it is surfaced verbatim to the operator.
    assert "GEMMA4_DFLASH_DRAFTER" in out.reason


def test_plan_fields(drafter_snapshot):
    out = Gemma4DFlashForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY)
    assert isinstance(out, SpecPlan)
    assert out.effective_k == _VERIFY
    assert out.lanes_per_request == 1
    # The drafter's context cache is a fixed window, not a per-token paged KV.
    assert out.extra_bytes_per_token == 0
    assert out.accept_modes == ("argmax_ids",)
    assert out.drafter_state == "internal"
    assert out.drafter_target_cache_requires == ()


def test_effective_k_never_exceeds_requested(drafter_snapshot):
    """#110: ``effective_k <= requested_k``. The plugin publishes it back to
    ``speculative_config.num_speculative_tokens``, which upstream's scheduler
    reads to size its lookahead reservation."""
    out = Gemma4DFlashForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY + 6)
    assert isinstance(out, SpecPlan)
    assert out.effective_k <= _VERIFY + 6


def test_fixed_cost_tracks_the_decoder_allocations(drafter_snapshot):
    """``extra_bytes_per_seq`` must be the real per-chip cost: under-reporting
    under-reserves by the whole drafter cache (#110 s.2).

    Mirrors ``DFlashFusedDecoder.__init__``: ctx_dev, ctx_k/ctx_v, ctx_pos,
    fc_prev, commit_pos — bf16 except the int64 position vectors.
    """
    _, cfg = drafter_snapshot
    cap, p_v, tp = 2048, _VERIFY + 1, 8
    local_kv = cfg["num_key_value_heads"] // tp
    expected = (
        cap * cfg["hidden_size"] * 2
        + 2 * cfg["num_hidden_layers"] * local_kv * cap * cfg["head_dim"] * 2
        + cap * 8
        + p_v * cfg["hidden_size"] * 2
        + p_v * 8
    )
    out = Gemma4DFlashForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY)
    assert out.extra_bytes_per_seq == expected


def test_unparseable_mesh_over_reserves_rather_than_under(drafter_snapshot, monkeypatch):
    """Galaxy DP entries set no MESH_DEVICE. Falling back to tp=1 yields the
    LARGEST per-chip cache, so the plugin over-reserves; the failure the
    contract warns about is under-reserving."""
    sharded = Gemma4DFlashForCausalLM.spec_plan(None, 1, _VERIFY).extra_bytes_per_seq
    monkeypatch.delenv("MESH_DEVICE", raising=False)
    unknown_mesh = Gemma4DFlashForCausalLM.spec_plan(None, 1, _VERIFY).extra_bytes_per_seq
    assert unknown_mesh > sharded


def test_is_pure_and_frozen(drafter_snapshot, expect_error):
    """Called several times per launch; must be side-effect free and stable."""
    env_before = dict(os.environ)
    first = Gemma4DFlashForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY)
    second = Gemma4DFlashForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY)
    assert first == second
    assert dict(os.environ) == env_before
    with expect_error(AttributeError, "effective_k"):
        first.effective_k = 99  # frozen dataclass
