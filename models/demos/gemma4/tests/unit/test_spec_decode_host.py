# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host tests for the Gemma 4 speculative decoder (``tt/spec_decode.py``) and the pure helpers around it.

Host-only: every decoder is a stub or a pure function, no device.
"""

import pytest
import torch

# The gemma4 vLLM generator imports vllm at module scope (through
# tt_transformers.generator_vllm), so COLLECTING this file fails outright on a
# runner without vLLM installed -- which is the tt-metal unit-test job. Skip
# before the import rather than inside the tests: the failure is at import.
pytest.importorskip("vllm")
import ast
import builtins
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import ttnn
from models.demos.gemma4.tests.unit.dflash_contract_harness import make_expect_error
from models.demos.gemma4.tt.attention import SPEC_RING_HEADROOM_ENV, bounded_ring_modulo
from models.demos.gemma4.tt.attention.decode import packed_verify_mask_span
from models.demos.gemma4.tt.dflash_drafter import DFlashFusedDecoder
from models.demos.gemma4.tt.spec_decode import (
    SPEC_K_BY_ROWS_DEFAULT,
    SPEC_MASK_NEG,
    SPEC_RECORD_MB_DEFAULT,
    ServingStepResult,
    SpeculativeDecoder,
    auto_draft_len,
    auto_draft_len_batched,
    packed_verify_kv_groups,
    parse_k_by_rows,
    serving_mask_padded_bytes,
    serving_mask_plan,
    serving_mask_rows,
    serving_masks_from_plan,
    serving_masks_host,
    serving_pad_rows,
    serving_plan,
    serving_seed_index,
    serving_select_rows,
    serving_trace_projection,
    serving_verify_table_widths,
    serving_width_for,
    serving_width_ladder,
    spec_capture_shapes,
    spec_check_shape,
)

try:
    import vllm  # noqa: F401

    _HAS_VLLM = True
except ImportError:  # the CI unit runners have no vLLM: those sections skip
    _HAS_VLLM = False


# ===== merged from tests/unit/test_dflash_bucket_ladder.py: Tests for the enumerable packed-verify bucket ladder.
try:
    from models.demos.gemma4.tt.generator_vllm import dflash_bucket_for, dflash_pv_bucket_ladder
except ImportError:  # vLLM absent: this section's tests skip
    pass


_H, _V = 2048, 5


_MAX_CTX = 262144


def _exact_bucket(start, horizon=_H, verify=_V):
    """Mirror of DFlashFusedDecoder.pv_bucket -- the value capture would use."""
    return ((start + horizon + (verify + 1) + 64 + 1023) // 1024) * 1024


@pytest.fixture()
def _pin_env(monkeypatch):
    monkeypatch.setenv("GEMMA4_DFLASH_SERVE_HORIZON", str(_H))
    monkeypatch.setenv("GEMMA4_DFLASH_VERIFY", str(_V))


@pytest.mark.usefixtures("_pin_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_ladder_is_enumerable_and_small():
    """Logarithmic in context: a 256K server must not need ~258 captures."""
    ladder = dflash_pv_bucket_ladder(_MAX_CTX)
    assert len(ladder) <= 12
    assert ladder == sorted(set(ladder))
    assert all(r % 1024 == 0 for r in ladder)


@pytest.mark.usefixtures("_pin_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
@pytest.mark.parametrize("start", [0, 1, 127, 128, 1000, 4096, 8192, 16384, 32768, 65536, 131072, 200000, _MAX_CTX])
def test_every_prompt_length_rounds_up_to_a_rung(start):
    """The whole point: no prompt inside max_context may be uncovered, and the
    chosen rung must be >= the bucket capture would have used (never smaller,
    or the verify would attend past what it captured)."""
    ladder = dflash_pv_bucket_ladder(_MAX_CTX)
    rung = dflash_bucket_for(start, ladder)
    assert rung is not None
    assert rung >= _exact_bucket(start)


@pytest.mark.usefixtures("_pin_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_rounding_waste_is_bounded():
    """Rounding up costs verify width, so the ladder may not be arbitrarily
    coarse. Doubling rungs bound it below 2x the exact bucket."""
    ladder = dflash_pv_bucket_ladder(_MAX_CTX)
    worst = max(dflash_bucket_for(s, ladder) / _exact_bucket(s) for s in range(0, 65536, 97))
    assert worst < 2.0


@pytest.mark.usefixtures("_pin_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_beyond_ladder_is_none_not_a_wrong_rung():
    """A request past the ladder must signal fallback, NOT silently get the
    largest rung -- that would attend past the captured width."""
    ladder = dflash_pv_bucket_ladder(_MAX_CTX)
    assert dflash_bucket_for(_MAX_CTX * 2, ladder) is None


@pytest.mark.usefixtures("_pin_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_cap_keeps_the_largest_rungs():
    """Capping trades wasted width on SHORT prompts for a bounded capture count.
    Dropping the largest rungs instead would leave long prompts uncovered."""
    full = dflash_pv_bucket_ladder(_MAX_CTX)
    capped = dflash_pv_bucket_ladder(_MAX_CTX, max_rungs=3)
    assert len(capped) == 3
    assert capped == full[-3:]
    # Long prompts stay covered; that is the invariant the cap must not break.
    assert dflash_bucket_for(_MAX_CTX, capped) is not None


@pytest.mark.usefixtures("_pin_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_horizon_and_verify_widen_the_smallest_rung():
    """The bucket carries the generation horizon, so a bigger horizon must not
    silently leave a short prompt under-covered."""
    small = dflash_pv_bucket_ladder(_MAX_CTX, horizon=512, verify=_V)
    large = dflash_pv_bucket_ladder(_MAX_CTX, horizon=8192, verify=_V)
    assert large[0] > small[0]
    assert dflash_bucket_for(0, large) >= _exact_bucket(0, horizon=8192)


# ===== merged from tests/unit/test_dflash_release_runtime.py: ``release_persistent_capture`` on the base ``Generator`` and DiffusionGemma.
pytest.importorskip("ttnn.device", reason="needs the TT runtime")


def _base_generator():
    from models.tt_transformers.tt.generator import Generator

    generator = Generator.__new__(Generator)
    released = []
    generator.model_args = [SimpleNamespace(mesh_device="mesh")]
    generator.model = [SimpleNamespace(sampling=SimpleNamespace(reset_trace=lambda: released.append("sampling")))]
    generator.trace_id_prefill = {"128_0": 11}
    generator.trace_id_prefill_sampling = {}
    generator.trace_ids_decode = {}
    generator.data_parallel = 1
    return generator, released


def test_base_release_runs_once_and_the_destructor_only_runs_the_base(monkeypatch):
    generator, released = _base_generator()
    monkeypatch.setattr(ttnn, "release_trace", lambda mesh, tid: released.append(tid))
    generator.release_persistent_capture()
    generator.release_persistent_capture()
    assert released.count(11) == 1 and released.count("sampling") == 1
    generator.__del__()
    assert released.count(11) == 1


@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="DiffusionGemmaForCausalLM derives from the vLLM generator base in models.tt_transformers.tt.generator_vllm, "
    "which imports vllm at module level; the models unit-test image does not ship vllm",
)
def test_diffusion_gemma_release_reaches_the_base():
    from models.experimental.diffusion_gemma.tt.generator_vllm import DiffusionGemmaForCausalLM

    model = DiffusionGemmaForCausalLM.__new__(DiffusionGemmaForCausalLM)
    model._sessions = {}
    model._persistent_adapter = None
    model.model_args = []
    model.model = []
    model.release_persistent_capture()
    assert getattr(model, "_generator_capture_released", False) is True


# ===== merged from tests/unit/test_dflash_tap_keep.py: ``Gemma4Model.dflash_capture_taps`` stores the ``keep_last`` cap.
def _model_dflash_tap_keep():
    module = pytest.importorskip("models.demos.gemma4.tt.model", reason="needs the TT runtime")
    return module.Gemma4Model.__new__(module.Gemma4Model)


def test_keep_last_is_stored_in_tap_entries():
    model = _model_dflash_tap_keep()
    model.dflash_capture_taps([1, 2, 3], keep_last=4)
    assert model._dflash_tap_layers == {1, 2, 3}
    assert model._dflash_tap_keep == 12
    assert model._dflash_taps == [] and model._dflash_tap_idx == 0


def test_no_keep_last_means_no_cap():
    model = _model_dflash_tap_keep()
    model.dflash_capture_taps([1, 2], buffers=["a", "b"])
    assert model._dflash_tap_keep is None
    assert model._dflash_tap_buffers == ["a", "b"]


def test_disarming_clears_the_cap():
    model = _model_dflash_tap_keep()
    model.dflash_capture_taps([1], keep_last=2)
    model.dflash_capture_taps(None)
    assert model._dflash_tap_layers is None and model._dflash_tap_keep is None


# ===== merged from tests/unit/test_dflash_warmup_prefill_trace.py: dFlash warms prefill eagerly: it never replays a prefill trace (tt-metal#57853).
try:
    from models.demos.gemma4.tt.generator_vllm import (
        Gemma4DFlashBase,
        Gemma4DFlashContractForCausalLM,
        Gemma4ForCausalLM,
        Gemma4MTPContractForCausalLM,
    )
except ImportError:  # vLLM absent: this section's tests skip
    pass


def _captured_enable_trace(monkeypatch, cls, *, bounded):
    """Call ``cls``'s warmup override and report the enable_trace it passed up.

    The base ``Gemma4ForCausalLM.warmup_model_prefill`` needs a live model and a
    device, so it is replaced with a recorder: what matters here is the value the
    subclass decides to pass, not what the base then does with it.
    """
    seen = {}

    def _record(self, kv_cache, enable_trace, can_sample_on_device, greedy_only=False):
        seen["enable_trace"] = enable_trace

    monkeypatch.setattr(Gemma4ForCausalLM, "warmup_model_prefill", _record, raising=True)
    instance = object.__new__(cls)
    instance._bounded_sliding_kv_cache = bounded
    cls.warmup_model_prefill(instance, None, True, True)
    return seen.get("enable_trace")


if _HAS_VLLM:

    @pytest.mark.parametrize("bounded", [False, True], ids=["unbounded", "bounded"])
    def test_dflash_warmup_never_captures_a_prefill_trace(monkeypatch, bounded):
        """Both bounded settings: the capture is always off.

        The unbounded case is the regression -- that is the one the inherited
        warmup left at True while runtime forced it False.
        """
        assert _captured_enable_trace(monkeypatch, Gemma4DFlashContractForCausalLM, bounded=bounded) is False


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_the_override_is_scoped_to_the_dflash_rails():
    """It must not leak to classes that DO replay their prefill traces.

    ``Gemma4ForCausalLM`` replays them, and capturing them at warmup is the
    #49083 fix -- forcing the capture off there would undo it. MTP inherits the
    base warmup for the same reason. Asserting the identity of the bound
    function keeps this honest without re-stubbing the base's internals.
    """
    assert Gemma4DFlashContractForCausalLM.warmup_model_prefill is not Gemma4ForCausalLM.warmup_model_prefill
    assert Gemma4DFlashContractForCausalLM.warmup_model_prefill is Gemma4DFlashBase.warmup_model_prefill
    assert Gemma4MTPContractForCausalLM.warmup_model_prefill is Gemma4ForCausalLM.warmup_model_prefill


# ===== merged from tests/unit/test_dflash_width_set.py: Tests for the dFlash packed-verify WIDTH SET selection and migration.
try:
    from models.demos.gemma4.tt.generator_vllm import dflash_pv_bucket_ladder
except ImportError:  # vLLM absent: this section's tests skip
    pass


def _decoder(widths, captured=None, P_v=6):
    """A decoder carrying a captured width set and nothing else."""
    dec = DFlashFusedDecoder.__new__(DFlashFusedDecoder)
    dec.P_v = P_v
    dec.use_packed = True
    captured = set(widths if captured is None else captured)
    dec._pv_widths = {int(w): {"pv_sk": int(w), "trace": (object() if w in captured else None)} for w in widths}
    dec.pv_sk = min(int(w) for w in widths)
    return dec


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_narrowest_covering_width_is_selected():
    dec = _decoder([2048, 4096, 8192])
    assert dec.width_for(0) == 2048
    assert dec.width_for(2048 - 6 - 64) == 2048
    assert dec.width_for(2048 - 6 - 64 + 1) == 4096
    assert dec.width_for(4096 - 6 - 64 + 1) == 8192


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_uncaptured_widths_are_never_selected():
    """A width whose trace was not captured cannot serve a step: replaying
    another width's trace would read the wrong pv_iota length."""
    dec = _decoder([2048, 4096, 8192], captured=[2048, 8192])
    assert dec.width_for(2048) == 8192


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_past_the_widest_width_is_none_not_a_wrong_width():
    dec = _decoder([2048, 4096])
    assert dec.width_for(4096) is None


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_ladder_covers_every_position_up_to_max_model_len():
    """The migration guarantee: the largest rung covers max_model_len, so a
    request never runs out of widths before vLLM's own length limit."""
    max_model_len = 262144
    ladder = dflash_pv_bucket_ladder(max_model_len, horizon=2048, verify=5)
    dec = _decoder(ladder)
    for pos in (0, 1, 2047, 4096, 131072, max_model_len - 1, max_model_len):
        assert dec.width_for(pos) is not None, pos


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
@pytest.mark.parametrize("pos", [0, 1000, 5000, 100000])
def test_selection_is_monotonic_in_position(pos):
    """Width never shrinks as a request generates: a migration only ever moves
    to a wider trace, so the mask and page table only ever grow."""
    ladder = dflash_pv_bucket_ladder(262144, horizon=2048, verify=5)
    dec = _decoder(ladder)
    w = dec.width_for(pos)
    assert dec.width_for(pos + 1) >= w
    assert dec.width_for(pos + 1000) >= w


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_top_width_is_the_block_budget_not_the_rounded_context():
    """Reproduces the warmup kill: the ladder's top rung rounds UP past
    max_model_len (265216 for a 262144 server = 4144 blocks), and
    paged_update_cache requires max_num_blocks_per_seq < max_num_blocks, so
    warmup died with 'max_num_blocks_per_seq=4144, max_num_blocks=4128'."""
    from models.demos.gemma4.tt.generator_vllm import dflash_width_set

    widths = dflash_width_set(262144, num_blocks=4096, horizon=2048, verify=5)
    assert max(widths) == 262144
    assert max(widths) // 64 == 4096
    assert all(w // 64 <= 4096 for w in widths)


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_width_set_drops_rungs_past_the_budget_and_keeps_the_budget():
    from models.demos.gemma4.tt.generator_vllm import dflash_width_set

    widths = dflash_width_set(262144, num_blocks=1024, horizon=2048, verify=5)
    assert max(widths) == 65536  # 1024 blocks x 64
    assert all(w <= 65536 for w in widths)
    assert widths == sorted(widths)


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_width_set_without_a_block_budget_is_the_plain_ladder():
    from models.demos.gemma4.tt.generator_vllm import dflash_pv_bucket_ladder, dflash_width_set

    assert dflash_width_set(262144, num_blocks=0, horizon=2048, verify=5) == dflash_pv_bucket_ladder(
        262144, horizon=2048, verify=5
    )


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_width_set_covers_every_position_the_pool_can_hold():
    from models.demos.gemma4.tt.generator_vllm import dflash_width_set

    widths = dflash_width_set(262144, num_blocks=4096, horizon=2048, verify=5)
    dec = _decoder(widths)
    # Everything up to the last verify block's worth of the context is covered;
    # past that vLLM's own max_model_len stop arrives first.
    for pos in (0, 4096, 131072, 262144 - 6 - 64):
        assert dec.width_for(pos) is not None, pos


# ===== merged from tests/unit/test_drafter_snapshot_resolution.py: Drafter checkpoints resolve from the HF cache the RUNNER configured.
try:
    from models.demos.gemma4.tt import generator_vllm as gv
except ImportError:  # vLLM absent: this section's tests skip
    pass


def _mk(root, repo):
    snap = os.path.join(root, repo, "snapshots", "abc123")
    os.makedirs(snap, exist_ok=True)
    return snap


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_hf_hub_cache_is_searched_before_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "mlperf" / "hub"))
    roots = gv._hf_hub_cache_dirs()
    assert roots[0] == str(tmp_path / "mlperf" / "hub")
    assert roots[-1] == os.path.expanduser("~/.cache/huggingface/hub")


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_hf_home_contributes_its_hub_subdirectory(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))
    assert str(tmp_path / "hfhome" / "hub") in gv._hf_hub_cache_dirs()


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_roots_are_deduplicated(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))
    roots = gv._hf_hub_cache_dirs()
    assert len(roots) == len(set(roots))


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_dflash_drafter_is_found_in_the_configured_cache(monkeypatch, tmp_path):
    root = str(tmp_path / "mlperf" / "hub")
    snap = _mk(root, "models--z-lab--gemma-4-31B-it-DFlash")
    monkeypatch.setenv("HF_HUB_CACHE", root)
    got = gv._dflash_default_snapshot()
    assert got is not None and got.rstrip("/") == snap


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_mtp_assistant_is_found_in_the_configured_cache(monkeypatch, tmp_path):
    root = str(tmp_path / "mlperf" / "hub")
    snap = _mk(root, "models--google--gemma-4-12B-it-assistant")
    monkeypatch.setenv("HF_HUB_CACHE", root)
    got = gv._assistant_default_snapshot("google/gemma-4-12B-it")
    assert got.rstrip("/") == snap


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_missing_drafter_still_reports_absence(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty2"))
    # No assertion on the default root: a developer box may legitimately have
    # the drafter cached there. Absence is only asserted for the configured
    # roots, which is what the resolver adds.
    assert gv._hf_snapshot_glob("models--z-lab--does-not-exist") is None


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_read_only_cache_does_not_attempt_a_fetch(monkeypatch, tmp_path):
    """A :ro weights mount is cache-only.

    HF writes metadata on every HEAD call, so a fetch into a read-only mount
    fails with "Read-only file system" rather than populating anything. The
    caller reports the miss instead.
    """
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        monkeypatch.setattr(gv, "_hf_hub_cache_dirs", lambda: [str(ro)])
        called = []
        monkeypatch.setattr(
            "huggingface_hub.snapshot_download",
            lambda *a, **k: called.append(k) or "/never",
        )
        assert gv._hf_resolve_repo("z-lab/nope", "models--z-lab--nope") is None
        assert called == []
    finally:
        ro.chmod(0o755)


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_writable_cache_fetches_even_with_offline_set(monkeypatch, tmp_path):
    """Write access is the operator asking for the fetch.

    HF_HUB_OFFLINE is set for the read-only case; it must not veto a fetch into
    a cache the operator deliberately made writable, and it is restored after.

    Asserting the ENV VAR alone is not enough, and that is what let the real bug
    ship: huggingface_hub reads HF_HUB_OFFLINE once at import into
    constants.HF_HUB_OFFLINE, and every offline gate goes through
    constants.is_offline_mode(), which returns that module global. Clearing only
    the env var left the hub offline, so a :rw CI run still raised
    LocalEntryNotFoundError and the drafter never downloaded. Assert the gate the
    hub actually consults.
    """
    from huggingface_hub import constants as hf_constants

    root = tmp_path / "rw"
    root.mkdir()
    monkeypatch.setattr(gv, "_hf_hub_cache_dirs", lambda: [str(root)])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    seen = {}

    def fake_download(*a, **k):
        seen["env_during_call"] = os.environ.get("HF_HUB_OFFLINE")
        seen["offline_mode_during_call"] = hf_constants.is_offline_mode()
        seen.update(k)
        return str(root / "fetched")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    got = gv._hf_resolve_repo("z-lab/x", "models--z-lab--x")
    assert got == str(root / "fetched")
    assert seen["env_during_call"] is None, "the env var is cleared for the call"
    assert seen["offline_mode_during_call"] is False, (
        "the hub's own offline gate must be off for the call, or the fetch raises "
        "LocalEntryNotFoundError however the env var is set"
    )
    assert os.environ.get("HF_HUB_OFFLINE") == "1", "env restored afterwards"
    assert hf_constants.HF_HUB_OFFLINE is True, "and the hub global restored too"


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_cache_hit_never_reaches_the_hub(monkeypatch, tmp_path):
    root = str(tmp_path / "hub")
    snap = _mk(root, "models--z-lab--gemma-4-31B-it-DFlash")
    monkeypatch.setenv("HF_HUB_CACHE", root)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    called = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *a, **k: called.append(k) or "/never",
    )
    assert gv._dflash_default_snapshot().rstrip("/") == snap
    assert called == []


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_an_online_miss_fetches_the_repo(monkeypatch, tmp_path):
    """Cache-then-hub, the way transformers resolves the target model."""
    # Pin the search roots: ~/.cache is the final fallback and a dev box may
    # genuinely have the drafter cached there, which would mask the fetch.
    (tmp_path / "empty").mkdir(exist_ok=True)
    monkeypatch.setattr(gv, "_hf_hub_cache_dirs", lambda: [str(tmp_path / "empty")])
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    asked = {}

    def fake_download(*a, **k):
        asked.update(k)
        return str(tmp_path / "fetched")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    got = gv._dflash_default_snapshot()
    assert got == str(tmp_path / "fetched")
    assert asked.get("repo_id") == "z-lab/gemma-4-31B-it-DFlash"


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_failed_fetch_is_reported_as_a_miss_not_an_exception(monkeypatch, tmp_path):
    (tmp_path / "empty").mkdir(exist_ok=True)
    monkeypatch.setattr(gv, "_hf_hub_cache_dirs", lambda: [str(tmp_path / "empty")])
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    def boom(*a, **k):
        raise OSError("no route to host")

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
    assert gv._dflash_default_snapshot() is None


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_model_weights_dir_is_used_when_it_exists(monkeypatch, tmp_path):
    """tt-inference-server exports MODEL_WEIGHTS_DIR; prefer it."""
    d = tmp_path / "weights"
    d.mkdir()
    monkeypatch.setenv("MODEL_WEIGHTS_DIR", str(d))
    monkeypatch.delenv("HF_MODEL", raising=False)
    assert gv._target_weights_dir() == str(d)


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_repo_id_in_hf_model_resolves_through_the_hub_cache(monkeypatch, tmp_path):
    """tt-metal's vLLM CI sets HF_MODEL to a REPO ID and never sets
    MODEL_WEIGHTS_DIR, so the only way to the checkpoint is the hub cache.
    Relying on the env var alone built 'None/model.safetensors.index.json' and
    killed the engine during warmup after the drafter had downloaded fine."""
    root = tmp_path / "hub"
    snap = root / "models--google--gemma-4-31B-it" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    monkeypatch.setattr(gv, "_hf_hub_cache_dirs", lambda: [str(root)])
    monkeypatch.delenv("MODEL_WEIGHTS_DIR", raising=False)
    monkeypatch.delenv("GEMMA4_MODEL_PATH", raising=False)
    monkeypatch.setenv("HF_MODEL", "google/gemma-4-31B-it")
    got = gv._target_weights_dir()
    assert got is not None, "a repo id must resolve through the hub cache"
    assert got.rstrip("/") == str(snap)


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_local_hf_model_directory_is_used_directly(monkeypatch, tmp_path):
    d = tmp_path / "local-ckpt"
    d.mkdir()
    monkeypatch.delenv("MODEL_WEIGHTS_DIR", raising=False)
    monkeypatch.setenv("HF_MODEL", str(d))
    assert gv._target_weights_dir() == str(d)


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_nothing_resolvable_reports_absence(monkeypatch, tmp_path):
    """None must be returned so the caller can raise something readable,
    rather than formatting it into a path."""
    monkeypatch.setattr(gv, "_hf_hub_cache_dirs", lambda: [str(tmp_path / "empty")])
    monkeypatch.delenv("MODEL_WEIGHTS_DIR", raising=False)
    monkeypatch.delenv("GEMMA4_MODEL_PATH", raising=False)
    monkeypatch.setenv("HF_MODEL", "google/gemma-4-31B-it")
    assert gv._target_weights_dir() is None


# ===== merged from tests/unit/test_mtp_width_ladder.py: Tests for the MTP serving verify-width ladder and its per-step selection.
try:
    from models.demos.gemma4.tt.generator_vllm import mtp_pv_width_ladder
except ImportError:  # vLLM absent: this section's tests skip
    pass


def _records(shapes_widths):
    """A decoder carrying prepared serving widths and nothing else (no device)."""
    spec = SpeculativeDecoder.__new__(SpeculativeDecoder)
    spec._srv_records = {}
    spec._srv_captured = False
    spec._srv_shape_bufs = {shape: {} for shape in shapes_widths}
    spec._srv_widths = {shape: sorted(ws) for shape, ws in shapes_widths.items()}
    return spec


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_ladder_doubles_and_covers_the_whole_context():
    ladder = mtp_pv_width_ladder(262144, draft_len=5)
    assert ladder == sorted(ladder)
    assert ladder[0] == 1024
    assert ladder[-1] >= 262144 + 7
    # doubling -> logarithmic rung count
    assert len(ladder) <= 12
    for a, b in zip(ladder, ladder[1:]):
        assert b >= 2 * a or b == ladder[-1]


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_every_position_in_the_context_has_a_rung_per_shape():
    for shape in [(1, 5), (5, 5), (8, 3), (16, 1)]:
        k = shape[1]
        ladder = mtp_pv_width_ladder(262144, draft_len=k)
        spec = _records({shape: ladder})
        for pos in (0, 1, 1000, 2048, 4096, 131072, 262143):
            assert spec.serving_width_for(shape, pos) is not None, (shape, pos)


def test_selection_is_the_narrowest_covering_width():
    """Width is per-iteration BANDWIDTH -- the masks are rebuilt on the host
    and uploaded at [B, 1, P, S_k] every replay -- so a step must not sit on a
    wider rung than its highest position needs: need = pos + K + 2."""
    assert serving_width_for([1024, 2048, 4096], 0, 5) == 1024
    assert serving_width_for([1024, 2048, 4096], 1024 - 7, 5) == 1024
    assert serving_width_for([1024, 2048, 4096], 1024 - 6, 5) == 2048
    assert serving_width_for([1024, 2048, 4096], 2048 - 6, 5) == 4096
    # a shorter draft needs less tail: (16, 1) stays on 1024 two positions longer
    assert serving_width_for([1024, 2048], 1024 - 3, 1) == 1024
    assert serving_width_for([1024, 2048], 1024 - 2, 1) == 2048
    # order of the list is irrelevant
    assert serving_width_for([4096, 1024, 2048], 0, 5) == 1024


def test_position_past_the_prepared_set_is_none_so_the_adapter_declines():
    """None is the signal to propose nothing for that step (plain decode):
    nothing is captured during serving, so there is no next rung to take."""
    assert serving_width_for([1024], 2000, 5) is None
    spec = _records({(1, 5): [1024]})
    assert spec.serving_width_for((1, 5), 2000) is None
    assert spec.serving_width_for((1, 5), 1024 - 7) == 1024


def test_width_query_rejects_unknown_shapes_and_unprepared_decoders():
    spec = _records({(1, 5): [1024]})
    with pytest.raises(ValueError, match="was not prepared"):  # allow-pytest.raises (pure value check)
        spec.serving_width_for((5, 5), 0)
    bare = SpeculativeDecoder.__new__(SpeculativeDecoder)
    bare._srv_records = None
    with pytest.raises(RuntimeError, match="serving_prepare has not run"):  # allow-pytest.raises (pure)
        bare.serving_width_for((1, 5), 0)


@pytest.mark.parametrize("draft_len", [1, 3, 5, 7])
def test_tail_follows_the_draft_length(draft_len):
    """A verify block is P = K + 1 rows at pos..pos+K, so the width has to
    cover K + 2 past the position, not just the position."""
    widths = [1024, 2048, 8192]
    top = max(widths)
    assert serving_width_for(widths, top - draft_len - 2, draft_len) == top
    assert serving_width_for(widths, top - draft_len - 1, draft_len) is None


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_cap_keeps_the_largest_rungs_mtp_width_ladder():
    ladder = mtp_pv_width_ladder(262144, draft_len=5, max_rungs=4)
    assert len(ladder) == 4
    assert ladder[-1] == max(mtp_pv_width_ladder(262144, draft_len=5))


# ===== merged from tests/unit/test_spec_decode_no_unbound_names.py: No method in the speculative serving path reads a name it never binds.
SOURCE = Path(__file__).resolve().parents[2] / "tt" / "spec_decode.py"


def _module():
    return ast.parse(SOURCE.read_text())


def _bound_at_module_level(tree):
    names = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _free_names(fn, module_names):
    """Names a function reads without binding them anywhere in its own body."""
    bound = {a.arg for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs}
    if fn.args.vararg:
        bound.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        bound.add(fn.args.kwarg.arg)
    loads = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name):
            (bound if isinstance(node.ctx, (ast.Store, ast.Del)) else loads).add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    return loads - bound - module_names


def _functions(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def test_srv_upload_binds_every_name_it_reads():
    """The per-replay upload (the method whose B=1 ancestor regressed), named so the failure is unmistakable."""
    tree = _module()
    module_names = _bound_at_module_level(tree)
    target = [fn for fn in _functions(tree) if fn.name == "_srv_upload"]
    assert target, "_srv_upload not found in spec_decode.py"
    assert _free_names(target[0], module_names) == set()


_SERVING_METHODS = [
    "serving_prepare",
    "serving_capture",
    "serving_width_for",
    "serving_refresh_page_tables",
    "serving_reseed",
    "serving_step",
    "serving_release",
    "_srv_alloc_shape",
    "_srv_alloc_width",
    "_srv_publish_hidden",
    "_srv_publish_logits",
    "_srv_trace_region_mb",
    "_srv_trace_region_size_mb",
    "_srv_iota_for",
    "_srv_mask_chunks",
    "_srv_masks_device",
    "_srv_seed_rows",
    "_srv_seed_gather",
    "_srv_assemble_verify_x",
    "_srv_body",
    "_srv_upload",
    "_srv_upload_tables",
    "_srv_slot_table",
    "_srv_logits_to_host",
    "refresh_page_tables",
]


def test_serving_path_methods_bind_every_name_they_read():
    tree = _module()
    module_names = _bound_at_module_level(tree)
    defined = {fn.name: fn for fn in _functions(tree)}
    for name in _SERVING_METHODS:
        assert name in defined, f"{name} is not defined in spec_decode.py"
        free = _free_names(defined[name], module_names)
        assert free == set(), f"{name} reads unbound name(s): {sorted(free)}"


# ===== merged from tests/unit/test_spec_draft_len.py: Host-only tests for the pure speculative draft-length policies.
@pytest.fixture()
def _no_draft_len_env(monkeypatch):
    monkeypatch.delenv("GEMMA4_SPEC_DRAFT_LEN", raising=False)


@pytest.mark.parametrize(
    ("prompt_len", "expected"),
    [
        (None, 3),  # unknown prompt -> short default
        (128, 3),
        (1023, 3),  # boundary: strictly below the knee
        (1024, 5),  # boundary: at the knee
        (131072, 5),
    ],
)
@pytest.mark.usefixtures("_no_draft_len_env")
def test_auto_draft_len_prompt_boundary(prompt_len, expected):
    assert auto_draft_len(prompt_len) == expected


@pytest.mark.usefixtures("_no_draft_len_env")
def test_auto_draft_len_env_override_wins(monkeypatch, expect_error):
    monkeypatch.setenv("GEMMA4_SPEC_DRAFT_LEN", "7")
    assert auto_draft_len(128) == 7
    assert auto_draft_len(131072) == 7

    # -- the acceptance-adaptive policy (tt/spec_policy.py): its knobs read from the env
    _policy_policy_config_defaults_are_on_with_the_documented_knobs()
    _policy_policy_config_reads_every_knob_from_the_env()
    _policy_bad_policy_knobs_are_refused_naming_the_knob(expect_error)


@pytest.mark.usefixtures("_no_draft_len_env")
def test_auto_draft_len_env_auto_defers_to_policy(monkeypatch, expect_error):
    monkeypatch.setenv("GEMMA4_SPEC_DRAFT_LEN", "auto")
    assert auto_draft_len(128) == 3
    assert auto_draft_len(4096) == 5

    # -- the acceptance-adaptive policy: the record choice by expected tokens per cost, with hysteresis
    _policy_rank_records_scores_expected_tokens_per_cost_relative_to_the_plain_step(expect_error)
    _policy_the_policy_picks_k5_at_high_acceptance_and_drops_to_k3_k1_and_plain_as_it_falls()
    _policy_the_margin_decides_between_a_marginal_record_and_the_plain_step()


@pytest.mark.usefixtures("_no_draft_len_env")
def test_auto_draft_len_custom_default():
    assert auto_draft_len(None, default=2) == 2
    assert auto_draft_len(64, default=2) == 2

    # -- the acceptance-adaptive policy: the per-row acceptance statistics and the batch prior
    _policy_acceptance_stats_update_only_the_positions_the_verify_reached()
    _policy_expected_tokens_is_one_plus_the_running_product_of_the_conditionals()
    _policy_a_new_row_starts_from_the_batch_prior_and_the_prior_learns_from_every_row()


@pytest.mark.parametrize(
    ("batch", "prompt_len", "expected"),
    [
        (None, 4096, 5),  # no batch -> plain policy
        (1, 4096, 5),  # B=1 -> plain policy
        (8, 128, 3),  # short prompt K=3 within the 32-row cap (8*(3+1)=32)
        (8, 4096, 3),  # context K=5 CLIPPED to the row cap 32//8-1=3
        (16, 4096, 1),  # row cap 32//16-1=1
        (32, 4096, 0),  # past the compute knee: speculation OFF
        (33, 128, 0),  # row cap goes negative -> OFF, never negative K
    ],
)
@pytest.mark.usefixtures("_no_draft_len_env")
def test_auto_draft_len_batched_row_cap(batch, prompt_len, expected):
    assert auto_draft_len_batched(prompt_len, batch) == expected


@pytest.mark.usefixtures("_no_draft_len_env")
def test_auto_draft_len_batched_env_override_wins(monkeypatch, expect_error):
    monkeypatch.setenv("GEMMA4_SPEC_DRAFT_LEN", "2")
    assert auto_draft_len_batched(4096, 32) == 2  # explicit K bypasses the cap

    # -- the acceptance-adaptive policy: the measured cost table, the entry cost and the amortized retry
    _policy_the_cost_table_is_consumed_as_measured(expect_error)
    _policy_after_n_declined_steps_the_policy_retries_at_the_smallest_draft_length()


@pytest.mark.usefixtures("_no_draft_len_env")
def test_auto_draft_len_batched_env_auto_defers_to_policy(monkeypatch):
    monkeypatch.setenv("GEMMA4_SPEC_DRAFT_LEN", "auto")
    assert auto_draft_len_batched(4096, 8) == 3
    assert auto_draft_len_batched(4096, 32) == 0

    # -- the acceptance-adaptive policy: the probe, the measured positions and the warmup line
    _policy_below_the_cap_the_policy_probes_the_next_larger_draft_length_every_n_steps()
    _policy_measure_positions_time_each_rung_at_the_first_position_it_serves()
    _policy_describe_costs_lists_the_plain_step_then_every_record()


# ===== merged from tests/unit/test_spec_plan.py: Tests for the model side of the plugin speculative contract (``spec_plan``).
try:
    from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

    from models.demos.gemma4.tt.generator_vllm import Gemma4DFlashContractForCausalLM
except ImportError:  # vLLM absent: this section's tests skip
    pass


_VERIFY = 5


@pytest.fixture
def drafter_snapshot(tmp_path, monkeypatch):
    """A minimal drafter checkpoint dir — only ``config.json`` is read."""
    cfg = {
        "num_hidden_layers": 5,
        "hidden_size": 5376,
        "head_dim": 128,
        "num_key_value_heads": 8,
        "block_size": 16,  # the checkpoint's block geometry; spec_plan reads it (GEMMA4_DFLASH_BLOCK overrides)
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("GEMMA4_DFLASH_DRAFTER", str(tmp_path))
    monkeypatch.setenv("GEMMA4_DFLASH_VERIFY", str(_VERIFY))
    monkeypatch.setenv("MESH_DEVICE", "P150x8")
    monkeypatch.delenv("GEMMA4_DFLASH_REPLICATED", raising=False)
    monkeypatch.delenv("GEMMA4_DFLASH_CTX_CAP", raising=False)
    return tmp_path, cfg


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_k_is_a_set_not_a_range(drafter_snapshot):
    """A narrower K would need its own packed-verify capture, so it is refused
    and the one supported width is named."""
    out = Gemma4DFlashContractForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY - 2)
    assert isinstance(out, SpecReject)
    assert out.supported_k == (_VERIFY,)


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_missing_drafter_is_refused_at_config_time(drafter_snapshot, monkeypatch):
    """Without this the drafter resolves lazily and a missing checkpoint kills
    the engine at the FIRST REQUEST instead (EngineDeadError, no traceback)."""
    monkeypatch.setenv("GEMMA4_DFLASH_DRAFTER", "/nonexistent/dflash-drafter")
    out = Gemma4DFlashContractForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY)
    assert isinstance(out, SpecReject)
    assert out.supported_k == (_VERIFY,)
    # The message has to be actionable: it is surfaced verbatim to the operator.
    assert "GEMMA4_DFLASH_DRAFTER" in out.reason


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_plan_fields(drafter_snapshot):
    out = Gemma4DFlashContractForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY)
    assert isinstance(out, SpecPlan)
    assert out.effective_k == _VERIFY
    assert out.lanes_per_request == 1
    # The drafter's context cache is a fixed window, not a per-token paged KV.
    assert out.extra_bytes_per_token == 0
    assert out.accept_modes == ("argmax_ids",)
    assert out.drafter_state == "internal"
    assert out.drafter_target_cache_requires == ()


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_effective_k_never_exceeds_requested(drafter_snapshot):
    """#110: ``effective_k <= requested_k``. The plugin publishes it back to
    ``speculative_config.num_speculative_tokens``, which upstream's scheduler
    reads to size its lookahead reservation."""
    out = Gemma4DFlashContractForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY + 6)
    assert isinstance(out, SpecPlan)
    assert out.effective_k <= _VERIFY + 6


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
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
    out = Gemma4DFlashContractForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY)
    assert out.extra_bytes_per_seq == expected


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_unparseable_mesh_over_reserves_rather_than_under(drafter_snapshot, monkeypatch):
    """Galaxy DP entries set no MESH_DEVICE. Falling back to tp=1 yields the
    LARGEST per-chip cache, so the plugin over-reserves; the failure the
    contract warns about is under-reserving."""
    sharded = Gemma4DFlashContractForCausalLM.spec_plan(None, 1, _VERIFY).extra_bytes_per_seq
    monkeypatch.delenv("MESH_DEVICE", raising=False)
    unknown_mesh = Gemma4DFlashContractForCausalLM.spec_plan(None, 1, _VERIFY).extra_bytes_per_seq
    assert unknown_mesh > sharded


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_is_pure_and_frozen(drafter_snapshot, expect_error):
    """Called several times per launch; must be side-effect free and stable."""
    env_before = dict(os.environ)
    first = Gemma4DFlashContractForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY)
    second = Gemma4DFlashContractForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY)
    assert first == second
    assert dict(os.environ) == env_before
    with expect_error(AttributeError, "effective_k"):
        first.effective_k = 99  # frozen dataclass


# ===== merged from tests/unit/test_spec_ring_guard.py: The bounded ring is sized for the verify, and refused when it cannot hold it.
try:
    from models.demos.gemma4.tt.generator_vllm import (
        _auto_size_spec_ring,
        _reserve_spec_ring_headroom,
        _spec_ring_blocks_for,
    )
except ImportError:  # vLLM absent: this section's tests skip
    pass


WINDOW = 1024


SHIPPED_WIDTH = 6  # GEMMA4_DFLASH_VERIFY=5 -> P_v = K + 1


@pytest.fixture()
def _clean_env():
    """Restore the headroom env around every test.

    ``_auto_size_spec_ring`` writes ``os.environ`` directly, and monkeypatch
    only restores what monkeypatch itself changed -- ``delenv`` on an unset var
    records nothing. Without this, a sizing test leaks the headroom into every
    later test in the process and silently changes ``bounded_ring_modulo`` for
    the device suites.
    """
    saved = os.environ.get(SPEC_RING_HEADROOM_ENV)
    os.environ.pop(SPEC_RING_HEADROOM_ENV, None)
    yield
    if saved is None:
        os.environ.pop(SPEC_RING_HEADROOM_ENV, None)
    else:
        os.environ[SPEC_RING_HEADROOM_ENV] = saved


def _cls(width, *, declines=False):
    return SimpleNamespace(_SPEC_N=width, __name__="FakeSpecClass", _SPEC_DECLINES_AT_RING_WRAP=declines)


def _cfg(window=WINDOW):
    return SimpleNamespace(text_config=SimpleNamespace(sliding_window=window))


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_speculating_class_gets_a_ring_that_holds_its_drafts():
    """The shipped 31B dFlash / 12B MTP case: 1024 window, K=5 -> ring 2048."""
    _auto_size_spec_ring(_cls(SHIPPED_WIDTH), _cfg(), bounded_sliding=True)
    assert bounded_ring_modulo(WINDOW) == 2048
    _reserve_spec_ring_headroom(WINDOW, SHIPPED_WIDTH, "Gemma4DFlash")  # must not raise


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_the_ring_is_sized_from_drafts_not_from_the_anchor():
    """K drafts need K slots, not K + 1.

    The anchor write at p lands in slot p % ring, which held p - ring -- already
    outside the window -- so only the drafts evict live history. This is the same
    test the contract rail applies per request (``_dflash_ring_advice``).

    The shipped 1024 window cannot show the difference (1024 + 5 and 1024 + 6
    both round to 2048), so this uses a window where counting the anchor crosses
    a power of two: 3776 + 320 IS 4096, while 3776 + 321 forces 8192. Counting
    one row too many would double the bounded pool for every sliding layer.
    """
    assert _spec_ring_blocks_for(3776, 320) == (4096 - 3776) // 64
    assert _spec_ring_blocks_for(3776, 321) == (8192 - 3776) // 64
    # The shipped case, for the record: 1024 window, K=5 -> 2048 ring -> 16 blocks.
    assert _spec_ring_blocks_for(WINDOW, 5) == 16


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_baseline_class_is_left_alone():
    """No _SPEC_N: nothing speculates, so the exact-window ring is correct."""
    _auto_size_spec_ring(SimpleNamespace(__name__="Gemma4ForCausalLM"), _cfg(), bounded_sliding=True)
    assert bounded_ring_modulo(WINDOW) == WINDOW


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_unbounded_serving_is_left_alone():
    """Without bounded sliding there is no ring to size."""
    _auto_size_spec_ring(_cls(SHIPPED_WIDTH), _cfg(), bounded_sliding=False)
    assert SPEC_RING_HEADROOM_ENV not in os.environ


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_full_attention_models_are_left_alone():
    _auto_size_spec_ring(_cls(SHIPPED_WIDTH), _cfg(window=None), bounded_sliding=True)
    assert SPEC_RING_HEADROOM_ENV not in os.environ


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_an_operator_value_wins(monkeypatch):
    """A pinned value is not overwritten -- it is checked instead."""
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, "16")
    _auto_size_spec_ring(_cls(SHIPPED_WIDTH), _cfg(), bounded_sliding=True)
    assert os.environ[SPEC_RING_HEADROOM_ENV] == "16"


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_an_exact_window_ring_is_refused(expect_error):
    with expect_error(RuntimeError, "cannot hold a verify"):
        _reserve_spec_ring_headroom(WINDOW, SHIPPED_WIDTH, "Gemma4DFlash")


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_the_refusal_names_the_remedy_and_the_issue():
    try:
        _reserve_spec_ring_headroom(WINDOW, SHIPPED_WIDTH, "Gemma4DFlash")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("an exact-window ring must not be accepted")
    assert SPEC_RING_HEADROOM_ENV in message, "the remedy must be named"
    assert "57701" in message, "the issue must be findable from the failure"
    assert "16" in message, "the working value must be given, not just described"


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_class_that_declines_at_the_wrap_is_not_refused():
    """The contract rail stops proposing before the wrap and carries on as plain
    decode, so an exact ring costs it speed, not correctness."""
    _reserve_spec_ring_headroom(WINDOW, SHIPPED_WIDTH, "Gemma4DFlashContract", declines_at_wrap=True)


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_width_one_needs_no_headroom():
    """Width 1 writes only the anchor, so it evicts nothing in the live window."""
    _reserve_spec_ring_headroom(WINDOW, 1, "Gemma4DFlash")


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_an_invalid_ring_is_reported_whatever_the_width(monkeypatch, expect_error):
    """The ring is validated before the width check.

    Width 1 needs no headroom, but a ring that is not a power of two is wrong for
    any width. Validating after the width check would have made this guard's
    error path depend on verify_width, leaving a bad env value to surface later
    from the model or trace path instead of at config time.
    """
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, "32")  # 1024 + 32*64 = 3072
    with expect_error(ValueError, "power of two"):
        _reserve_spec_ring_headroom(WINDOW, 1, "Gemma4DFlash")


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_full_attention_is_unaffected_by_the_guard():
    _reserve_spec_ring_headroom(None, SHIPPED_WIDTH, "Gemma4DFlash")


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_the_check_is_on_the_ring_not_on_the_env_var(monkeypatch, expect_error):
    """Headroom set too small is still unsafe.

    The first version of this guard returned as soon as the env var was present,
    so an under-sized headroom passed. A 1024 window with 32 blocks is a 3072
    ring, not a power of two, so the ring builder rejects it rather than the
    guard silently accepting a value that only LOOKS configured.
    """
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, "32")
    with expect_error(ValueError, "power of two"):
        _reserve_spec_ring_headroom(WINDOW, SHIPPED_WIDTH, "Gemma4DFlash")


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_every_real_speculating_class_gets_sized():
    """The shipped speculating classes, by their real _SPEC_N.

    The fakes above pin the arithmetic; this pins the wiring. A new serving
    class that forgets ``_SPEC_N``, or one whose K changes, silently falls back
    to the exact-window ring -- which is the #57701 defect, and is invisible
    until someone reads generated text past the ring.
    """
    from models.demos.gemma4.tt.generator_vllm import Gemma4DFlashContractForCausalLM, Gemma4MTPContractForCausalLM

    for cls in (Gemma4DFlashContractForCausalLM, Gemma4MTPContractForCausalLM):
        os.environ.pop(SPEC_RING_HEADROOM_ENV, None)
        assert int(cls._SPEC_N) > 1, f"{cls.__name__} must declare a verify width"
        _auto_size_spec_ring(cls, _cfg(), bounded_sliding=True)
        ring = bounded_ring_modulo(WINDOW)
        assert ring - WINDOW >= cls._SPEC_N - 1, f"{cls.__name__}: ring {ring} cannot hold its drafts"
        # and the guard agrees, including for the rail that declines at the wrap
        _reserve_spec_ring_headroom(WINDOW, cls._SPEC_N, cls.__name__, declines_at_wrap=cls._SPEC_DECLINES_AT_RING_WRAP)


# ===== merged from tests/unit/test_native_context_guard.py: Native-context guard: the serving templates waive vLLM's max_model_len check, so gemma4 must refuse a
try:
    from models.demos.gemma4.tt.generator_vllm import _assert_within_native_context
except ImportError:  # vLLM absent: this section's tests skip
    pass


expect_error_native_context_guard = pytest.fixture(lambda: make_expect_error())


NATIVE = 262_144


def _args(native=NATIVE):
    return SimpleNamespace(_hf_text_config=SimpleNamespace(max_position_embeddings=native))


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_at_native_passes(monkeypatch):
    monkeypatch.delenv("GEMMA4_ALLOW_BEYOND_NATIVE", raising=False)
    _assert_within_native_context(NATIVE, _args())


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_below_native_passes(monkeypatch):
    monkeypatch.delenv("GEMMA4_ALLOW_BEYOND_NATIVE", raising=False)
    _assert_within_native_context(131_072, _args())


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_one_past_native_raises(monkeypatch, expect_error_native_context_guard):
    monkeypatch.delenv("GEMMA4_ALLOW_BEYOND_NATIVE", raising=False)
    with expect_error_native_context_guard(ValueError, "native max_position_embeddings"):
        _assert_within_native_context(NATIVE + 1, _args())


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_the_scenario_from_the_report_raises(monkeypatch, expect_error_native_context_guard):
    # An operator writes max_context: 300000; the pool convention follows it.
    monkeypatch.delenv("GEMMA4_ALLOW_BEYOND_NATIVE", raising=False)
    with expect_error_native_context_guard(ValueError, "VLLM_ALLOW_LONG_MAX_MODEL_LEN"):
        _assert_within_native_context(300_000, _args())


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_explicit_env_waiver_passes(monkeypatch):
    monkeypatch.setenv("GEMMA4_ALLOW_BEYOND_NATIVE", "1")
    _assert_within_native_context(300_000, _args())


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_unknown_native_stays_permissive(monkeypatch):
    # A model_args without the attribute (older adaptors) must not brick boots.
    monkeypatch.delenv("GEMMA4_ALLOW_BEYOND_NATIVE", raising=False)
    _assert_within_native_context(300_000, SimpleNamespace())


# ===== Serving records (batched fused traces): the pure parts of the SpeculativeDecoder serving API.
# Shape table -------------------------------------------------------------------------------------
def test_default_k_by_rows_table_parses_to_the_catalog_entries():
    assert parse_k_by_rows(SPEC_K_BY_ROWS_DEFAULT) == ((5, 5), (8, 3), (16, 1))
    assert parse_k_by_rows(" 5:5 , 8:3,16:1, ") == ((5, 5), (8, 3), (16, 1))
    # the default capture shapes are one (1, k_max) record plus one per entry
    table = parse_k_by_rows(SPEC_K_BY_ROWS_DEFAULT)
    assert spec_capture_shapes(table, 5) == [(1, 5), (5, 5), (8, 3), (16, 1)]
    # a table that already names (1, k_max) does not duplicate it
    assert spec_capture_shapes(((1, 5), (8, 3)), 5) == [(1, 5), (8, 3)]
    # bad tables are refused with the reason
    for text, message in [
        ("5:5,4:3", "max_rows must increase"),
        ("5-5", "is not max_rows:k"),
        ("5:x", "is not max_rows:k"),
        ("", "has no entries"),
        ("16:2", "above the 32-row argmax tile"),  # 16 * 3 = 48 rows
        ("0:5", "must be >= 1"),
        ("5:0", "must be >= 1"),
    ]:
        with pytest.raises(ValueError, match=message):  # allow-pytest.raises (pure parser)
            parse_k_by_rows(text)
    # the tile rule is the only shape rule
    assert spec_check_shape((16, 1)) == (16, 1)  # 32 rows exactly
    assert spec_check_shape([1, 31]) == (1, 31)  # any (rows, drafts) pair, returned as an int tuple
    for shape in [(32, 1), (11, 2), (1, 32)]:
        with pytest.raises(ValueError, match="argmax tile"):  # allow-pytest.raises (pure)
            spec_check_shape(shape)
    with pytest.raises(ValueError, match="is \\(rows, drafts\\)"):  # allow-pytest.raises (pure)
        spec_check_shape(5)


# Plan ---------------------------------------------------------------------------------------------
def test_a_plan_sorts_shapes_and_widths_and_drops_duplicates():
    shapes, widths = serving_plan([(8, 3), (1, 5), (8, 3)], {(1, 5): [2048, 1024, 1024], (8, 3): (4096,)}, max_rows=16)
    assert shapes == [(1, 5), (8, 3)]
    assert widths == {(1, 5): [1024, 2048], (8, 3): [4096]}
    # bad plans are refused
    for shapes, widths, max_rows, message in [
        ([(1, 5)], {}, 16, "no verify widths for shape"),
        ([(1, 5)], {(1, 5): [1024], (5, 5): [1024]}, 16, "not prepared"),
        ([(1, 5)], {(1, 5): [1000]}, 16, "multiples of 64"),
        ([(1, 5)], {(1, 5): [0]}, 16, "multiples of 64"),
        ([(16, 1)], {(16, 1): [1024]}, 8, "more rows than max_rows"),
        ([], {}, 16, "at least one shape"),
        ([(1, 5)], {(1, 5): [1024]}, 0, "max_rows must be >= 1"),
    ]:
        with pytest.raises(ValueError, match=message):  # allow-pytest.raises (pure)
            serving_plan(shapes, widths, max_rows)


# Seed index ---------------------------------------------------------------------------------------
def test_seed_index_gathers_reseed_pool_rows_and_previous_verify_rows_per_record():
    # the first step after a reseed gathers the reseed-pool rows
    idx = serving_seed_index((5, 5), rows=[7, 3], seed_cols=[None, None], previous=None, seeded={7: 0, 3: 1})
    assert idx == [32, 33, 32, 32, 32]  # padding rows mirror live row 0 (serving_pad_rows)
    # a continuation in the same record gathers verify row r * P + col
    prev = (5, [7, 3, 9])
    idx = serving_seed_index((5, 5), rows=[7, 3, 9], seed_cols=[0, 5, 2], previous=prev, seeded={})
    assert idx == [0 * 6 + 0, 1 * 6 + 5, 2 * 6 + 2, 0, 0]
    # a continuation across shapes uses the previous record's geometry: (5,5) replayed rows [a, b, c]; the
    # next step is (8,3) with rows [c, a]: the gather still indexes the 6-column layout of the replay that
    # produced the hidden.
    prev = (5, [10, 11, 12])
    idx = serving_seed_index((8, 3), rows=[12, 10], seed_cols=[2, 5], previous=prev, seeded={})
    assert idx[:2] == [2 * 6 + 2, 0 * 6 + 5]
    assert idx[2:] == [idx[0]] * 6  # padding rows mirror row 0
    # a reseeded row and a continuing row mix in one step
    prev = (3, [1, 2])
    idx = serving_seed_index((8, 3), rows=[2, 5], seed_cols=[3, None], previous=prev, seeded={5: 1})
    assert idx[:2] == [1 * 4 + 3, 32 + 1]
    # padding rows take row zero whatever its source: row 0 continues from the previous replay, so the
    # mirrors gather that same verify row
    idx = serving_seed_index((8, 3), rows=[5, 6], seed_cols=[2, None], previous=(3, [5]), seeded={6: 0})
    assert idx == [0 * 4 + 2, 32, 2, 2, 2, 2, 2, 2]
    # row 0 was reseeded: the mirrors gather its reseed-pool row
    idx = serving_seed_index((5, 5), rows=[6, 5], seed_cols=[None, 2], previous=(3, [5]), seeded={6: 1})
    assert idx == [33, 2, 33, 33, 33]
    # padding rows mirror the first live row (the bounded ring pool has no null block)
    assert serving_pad_rows([7, 3], 5) == [7, 3, 7, 7, 7]
    assert serving_pad_rows([9], 1) == [9]
    assert serving_pad_rows([4, 2, 1], 3) == [4, 2, 1]
    with pytest.raises(ValueError, match="at least one live row"):  # allow-pytest.raises (pure)
        serving_pad_rows([], 5)
    with pytest.raises(ValueError, match="3 live rows for a 2-row record"):  # allow-pytest.raises (pure)
        serving_pad_rows([1, 2, 3], 2)


def test_seed_index_refuses_inconsistent_histories():
    for rows, seed_cols, previous, seeded, message in [
        ([1], [6], (5, [1]), {}, "outside 0..5"),
        ([1], [-1], (5, [1]), {}, "outside 0..5"),
        ([4], [0], (5, [1, 2]), {}, "last replay served rows"),
        ([4], [None], (5, [4]), {}, "not in the last serving_reseed"),
        ([4], [0], None, {}, "none ran"),
        ([4, 4], [0, 0], (5, [4]), {}, "duplicate rows"),
        ([1, 2, 3, 4, 5, 6], [0] * 6, (5, [1, 2, 3, 4, 5, 6]), {}, "expected 1..5"),
        ([1], [0, 0], (5, [1]), {}, "seed_cols for 1 live rows"),
    ]:
        with pytest.raises(ValueError, match=message):  # allow-pytest.raises (pure)
            serving_seed_index((5, 5), rows, seed_cols, previous, seeded)


# Masks --------------------------------------------------------------------------------------------
_W = 1024


def test_serving_masks_are_p_row_per_batch_and_bf16():
    mf, ms = serving_masks_host([100, 7, 0], P=6, window=_W, S_k=2048)
    assert mf.shape == (3, 1, 6, 2048) and ms.shape == (3, 1, 6, 2048)
    assert mf.dtype == torch.bfloat16 and ms.dtype == torch.bfloat16
    neg = torch.tensor(SPEC_MASK_NEG, dtype=torch.bfloat16)
    for m in (mf, ms):
        assert set(m.float().unique().tolist()) <= {0.0, float(neg)}
    # They match the batched host builder row for row: the old batched builder made [B, 1, H*P, S_k] on
    # the host; the new one makes the P-row block and the in-trace H repeat tiles it. With H=1 they must
    # agree exactly.
    cs, B, P, S_k = [100, 7, 2000], 3, 6, 2048
    ref_f, ref_s = SpeculativeDecoder._packed_mask_host(None, cs, B, P, 1, _W, S_k)
    mf, ms = serving_masks_host(cs, P, _W, S_k)
    assert torch.equal(mf, ref_f)
    assert torch.equal(ms, ref_s)
    # The head repeat reproduces the head-major layout: ttnn.repeat([1, 1, H, 1]) on [B, 1, P, S_k] tiles
    # the whole P block per head, i.e. row h*P+p -- the packed SDPA layout the host builder produced directly.
    cs, B, P, H, S_k = [5, 300], 2, 4, 3, 1024
    ref_f, ref_s = SpeculativeDecoder._packed_mask_host(None, cs, B, P, H, _W, S_k)
    mf, ms = serving_masks_host(cs, P, _W, S_k)
    assert torch.equal(mf.repeat(1, 1, H, 1), ref_f)
    assert torch.equal(ms.repeat(1, 1, H, 1), ref_s)
    # the full mask is causal per row and the sliding mask adds the window
    mf, ms = serving_masks_host([1500], P=3, window=_W, S_k=2048)
    for p in range(3):
        upper = 1500 + p
        full_ok = (mf[0, 0, p] == 0).nonzero().reshape(-1)
        assert full_ok.min() == 0 and full_ok.max() == upper and full_ok.numel() == upper + 1
        slide_ok = (ms[0, 0, p] == 0).nonzero().reshape(-1)
        assert slide_ok.min() == upper - _W + 1 and slide_ok.max() == upper and slide_ok.numel() == _W
    # The padded bytes count the 32-row tile, not the logical rows: a [B, 1, P, S_k] TILE tensor pads P to
    # 32 rows, so the device footprint of one mask (or one fp32 intermediate of the in-trace builder) is
    # B * 32 * S_k * elem, whatever P is.
    top = 263168
    assert serving_mask_padded_bytes((1, 5), top) == 32 * top * 2
    assert serving_mask_padded_bytes((16, 1), top) == 16 * 32 * top * 2
    assert serving_mask_padded_bytes((16, 1), 32768, elem_bytes=4) == 16 * 32 * 32768 * 4  # one fp32 chunk, 67 MB
    # P does not enter: (8, 1) and (8, 3) cost the same tile-padded bytes
    assert serving_mask_padded_bytes((8, 1), top) == serving_mask_padded_bytes((8, 3), top) == 8 * 32 * top * 2
    # the helper applies the tile rule like every other shape consumer
    with pytest.raises(ValueError, match="argmax tile"):  # allow-pytest.raises (pure)
        serving_mask_padded_bytes((16, 2), top)


def test_the_ring_mask_spans_ring_slots_with_the_pv_host_inputs_rule():
    """Bounded sliding: slot j holds position top - ((top - j) mod ring); query p may attend
    it iff (P-1-p) <= d < (P-1-p)+W and d <= top, with d = (top - j) mod ring (_pv_host_inputs)."""
    ring, P, W = 2048, 6, _W
    cs = [3000, 10]
    mf, ms = serving_masks_host(cs, P, W, S_k=4096, ring=ring)
    assert ms.shape == (2, 1, P, ring)
    assert mf.shape == (2, 1, P, 4096)
    neg = torch.tensor(SPEC_MASK_NEG, dtype=torch.bfloat16)
    jr = torch.arange(ring)
    for b, c in enumerate(cs):
        top = c + P - 1
        d = torch.remainder(top - jr, ring)
        for p in range(P):
            lo = P - 1 - p
            ok = (d >= lo) & (d < lo + W) & (d <= top)
            ref = torch.where(ok, torch.tensor(0.0, dtype=torch.bfloat16), neg)
            assert torch.equal(ms[b, 0, p], ref), (b, p)


# In-trace mask builder: its plan + op-for-op emulation must equal the host reference --------------
def test_device_mask_plan_equals_the_host_masks_unbounded():
    for positions, P, S_k in [
        ([100, 7, 2000], 6, 2048),
        ([0], 6, 1024),  # anchor at position 0: only key 0 visible on row 0
        ([1023, 1018], 6, 2048),  # rows straddle the window and the first rung
        ([5000, 11, 300], 2, 8192),
        ([40000], 4, 65536 + 1024),  # more than one SPEC_MASK_KEY_CHUNK of keys
    ]:
        plan = serving_mask_plan(positions, P)
        assert plan["q"].tolist() == [[c + p for p in range(P)] for c in positions]
        assert "tm" not in plan and "base" not in plan
        got_f, got_s = serving_masks_from_plan(plan, S_k, _W)
        ref_f, ref_s = serving_masks_host(positions, P, _W, S_k)
        assert got_f.shape == ref_f.shape and got_s.shape == ref_s.shape
        assert got_f.dtype == torch.bfloat16 and got_s.dtype == torch.bfloat16
        assert torch.equal(got_f, ref_f), (positions, P, S_k)
        assert torch.equal(got_s, ref_s), (positions, P, S_k)
        # the in-trace build works at the packed height from the uploaded packed-row positions (row h*P+p of
        # batch b = c_b + p): the twin at h local heads is the host mask repeated per head, no wide row repeat
        for h in (4, 8):
            rows = serving_mask_rows(positions, P, h)
            assert rows.shape == (1, len(positions) * h * P) and rows.dtype == torch.int64
            assert rows.tolist()[0] == [c + p for c in positions for _ in range(h) for p in range(P)]
            assert rows.reshape(len(positions), h, P)[:, 0, :].tolist() == plan["q"].tolist()
            hf, hs = serving_masks_from_plan(plan, S_k, _W, h=h)
            assert hf.shape == (len(positions), 1, h * P, S_k) and hs.shape == hf.shape
            assert torch.equal(hf, ref_f.repeat(1, 1, h, 1)) and torch.equal(hs, ref_s.repeat(1, 1, h, 1))


def test_device_mask_plan_equals_the_host_masks_under_a_ring():
    for positions, P, S_k, ring in [
        ([3000, 10], 6, 4096, 2048),
        ([2047, 2048, 2049], 6, 4096, 2048),  # the wrap boundary
        ([0, 1], 2, 1024, 2048),  # early context: the ring's untouched tail is masked
        ([70000, 1500], 4, 71680, 2048),
        ([5, 4100], 6, 8192, 4096),
    ]:
        plan = serving_mask_plan(positions, P, ring=ring)
        for b, c in enumerate(positions):
            top = c + P - 1
            assert int(plan["tm"][b]) == top % ring
            assert int(plan["base"][b]) == top - top % ring
        got_f, got_s = serving_masks_from_plan(plan, S_k, _W, ring=ring)
        ref_f, ref_s = serving_masks_host(positions, P, _W, S_k, ring=ring)
        assert got_s.shape == (len(positions), 1, P, ring)
        assert torch.equal(got_f, ref_f), (positions, P, S_k, ring)
        assert torch.equal(got_s, ref_s), (positions, P, S_k, ring)
        hf, hs = serving_masks_from_plan(plan, S_k, _W, ring=ring, h=8)  # the packed height under a ring
        assert torch.equal(hf, ref_f.repeat(1, 1, 8, 1)) and torch.equal(hs, ref_s.repeat(1, 1, 8, 1))
    # The ring-slot position rule matches the modulo form: base + j - ring * [j > tm] is
    # top - ((top - j) mod ring) for every slot, so the device form needs no modulo.
    ring, P = 2048, 6
    for c in (0, 7, 2042, 2043, 2048, 5000, 100000):
        plan = serving_mask_plan([c], P, ring=ring)
        top = c + P - 1
        j = torch.arange(ring)
        device_form = int(plan["base"][0]) + j - ring * (j > int(plan["tm"][0])).long()
        modulo_form = top - torch.remainder(top - j, ring)
        assert torch.equal(device_form, modulo_form), c


# Page-table rows ----------------------------------------------------------------------------------
def test_select_rows_picks_rows_zero_pads_padding_rows_and_narrow_sources():
    table = torch.arange(1, 13).reshape(3, 4)  # rows 0..2, 4 blocks each
    out = serving_select_rows(table, [2, 0], B=4, width=6, allow_clip=False)
    assert out.dtype == torch.int64 and out.shape == (4, 6)
    assert out[0].tolist() == [9, 10, 11, 12, 0, 0]
    assert out[1].tolist() == [1, 2, 3, 4, 0, 0]
    assert out[2:].abs().sum() == 0
    # a wider source is clipped only when allowed
    table = torch.tensor([[1, 2, 3, 0], [5, 6, 7, 8]])
    # zeros past the width are not live blocks: fine either way
    assert serving_select_rows(table, [0], 1, 3, allow_clip=False).tolist() == [[1, 2, 3]]
    assert serving_select_rows(table, [1], 1, 3, allow_clip=True).tolist() == [[5, 6, 7]]
    with pytest.raises(ValueError, match="live blocks past column 3"):  # allow-pytest.raises (pure)
        serving_select_rows(table, [1], 1, 3, allow_clip=False)
    # bad rows and shapes are refused
    table = torch.tensor([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="outside the 2-row page table"):  # allow-pytest.raises (pure)
        serving_select_rows(table, [2], 1, 2, allow_clip=True)
    with pytest.raises(ValueError, match="3 rows for a 2-row table"):  # allow-pytest.raises (pure)
        serving_select_rows(table, [0, 1, 0], 2, 2, allow_clip=True)
    with pytest.raises(ValueError, match="must be \\[rows, blocks\\]"):  # allow-pytest.raises (pure)
        serving_select_rows(torch.zeros(1, 2, 2), [0], 1, 2, allow_clip=True)
    # a 1-D row is a one-row table
    assert serving_select_rows(torch.tensor([7, 8]), [0], 1, 2, allow_clip=True).tolist() == [[7, 8]]


# Record bookkeeping on a device-free decoder ----------------------------------------------------
def _bare_decoder(n_layers=3):
    spec = SpeculativeDecoder.__new__(SpeculativeDecoder)
    spec._srv_records = None
    spec._srv_captured = False
    spec._srv_shape_bufs = {}
    spec._srv_per_layer = None
    spec._srv_pool_v = spec._srv_pool_s = None
    spec._srv_uploaded = {}
    spec._srv_last = None
    spec._srv_seeded = {}
    spec.target = SimpleNamespace(layers=list(range(n_layers)))
    return spec


def _rail_allocates_nothing_after_capture(monkeypatch, expect_error):
    """The serving rail on a recording ttnn: every device buffer a refresh, a reseed or a replay touches exists
    after serving_prepare, for every record and rung, first-time row sets, seed-column patterns and positions
    included; a buffer one of them would have to create is an error, never an allocation (the decode-trace
    guard releases and re-captures every decode trace when the allocator grows after a capture, 0.7 s each)."""
    import weakref

    import models.demos.gemma4.tt.spec_decode as spec_decode_module

    class _Buf:
        made = [0]

        def __init__(self, host, dtype=None, on_device=False):
            _Buf.made[0] += 1
            self.address = _Buf.made[0]
            self.host, self.dtype, self.on_device, self.freed = host, dtype, on_device, False

        @property
        def shape(self):
            return tuple(self.host.shape)

        def buffer_address(self):
            return self.address

        def deallocate(self, force=False):
            self.freed = True

    allocated, ops, h2d, replays, forwards = [], [], [], [], []

    def forbidden(name):
        def call(*args, **kwargs):
            raise AssertionError(f"ttnn.{name} would allocate on the device after serving_capture")

        return call

    def from_torch(t, device=None, layout=None, dtype=None, mesh_mapper=None):
        buf = _Buf(t.clone(), dtype, on_device=device is not None)
        if device is not None:
            allocated.append(buf)  # the only device constructor the rail uses
        return buf

    def copy_host_to_device_tensor(host, dev):
        assert dev.on_device and not dev.freed and not host.on_device
        dev.host = host.host.clone()
        h2d.append(dev.address)

    def op(host, dtype=None, address=None):
        out = _Buf(host, dtype, on_device=True)
        if address is not None:
            out.address = address  # a view: ttnn's same-volume reshape shares the source's buffer
        ops.append(weakref.ref(out))  # an op output: freed explicitly or dropped (collected) before a call returns
        return out

    def gone(ref):
        return ref() is None or ref().freed

    def pad(t, pads, value=0.0):
        return op(torch.zeros(1, 1, t.host.shape[2] + pads[2][1], t.host.shape[-1]), t.dtype)

    def assign(src, dst):
        dst.host = src.host

    def copy(src, dst):
        assert dst.on_device and not dst.freed and dst.address in prepared, "ttnn.copy writes a prepared buffer"
        assert tuple(src.host.shape) == tuple(dst.host.shape), (src.host.shape, dst.host.shape)
        dst.host = src.host.clone()
        copies.append(dst.address)

    traces = []

    fake = SimpleNamespace(
        ROW_MAJOR_LAYOUT=object(),
        TILE_LAYOUT=object(),
        DRAM_MEMORY_CONFIG=object(),
        uint32=object(),
        int32=object(),
        bfloat16=object(),
        float32=object(),
        Tensor=type("Tensor", (), {}),
        from_torch=from_torch,
        copy_host_to_device_tensor=copy_host_to_device_tensor,
        execute_trace=lambda mesh, tid, cq_id=0, blocking=False: replays.append(tid),
        release_trace=lambda mesh, tid: None,
        synchronize_device=lambda mesh: None,
        get_device_tensors=lambda t: [t],
        to_torch=lambda t: t.host,
        to_memory_config=lambda t, cfg: t,
        pad=pad,
        assign=assign,
        copy=copy,
        reshape=lambda t, shape: op(t.host.reshape(*shape), t.dtype, address=t.address),
        concat=lambda ts, dim: op(torch.cat([t.host for t in ts], dim=dim), ts[0].dtype),
        permute=lambda t, dims: op(t.host.permute(*dims).contiguous(), t.dtype),
        begin_trace_capture=lambda mesh, cq_id=0: traces.append(len(traces) + 1) or traces[-1],
        end_trace_capture=lambda mesh, tid, cq_id=0: None,
        BufferType=SimpleNamespace(TRACE=object()),
        **{
            name: forbidden(name)
            for name in ("to_device", "zeros", "ones", "full", "empty", "arange", "allocate_tensor")
        },
    )
    monkeypatch.setattr(spec_decode_module, "ttnn", fake)
    prepared, copies = set(), []
    shapes = [(1, 5), (5, 5), (8, 3), (16, 1)]
    ladder = [2048, 8192, 32768, 131072, 263168]  # the 12B ladder
    layer_types = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention"]
    backbone, vocab, wire_rows = 64, 96, 32

    def verify_forward(x, current_pos, current_pos_cache, page_table, kv_cache, page_tables_per_layer):
        rows = int(x.host.shape[-1])
        forwards.append((x.address, current_pos.address, current_pos_cache.address, rows))
        return op(torch.zeros(1, 1, rows, vocab), fake.bfloat16), op(torch.zeros(1, 1, rows, backbone), fake.bfloat16)

    spec = SpeculativeDecoder.__new__(SpeculativeDecoder)
    spec.mesh_device, spec._mapper, spec._tp, spec.tt_kv_cache = object(), None, 8, []
    spec.target = SimpleNamespace(
        layers=list(range(len(layer_types))),
        hf_config=SimpleNamespace(layer_types=layer_types),
        last_kv_layer_by_type={"sliding_attention": 3, "full_attention": 4},
        vocab_size=vocab,
        ttnn_verify_forward=verify_forward,
    )
    spec._pv_ring = {"sliding_attention": None, "full_attention": None}
    spec._pv_bs_t = {"sliding_attention": 128, "full_attention": 64}
    spec._pv_ring_rep = {"sliding_attention": 3, "full_attention": 4}
    spec._packed_H = lambda: 4  # local query heads: the packed-row positions are B x 4 x P per shape
    spec._srv_blocks = ladder[-1] // 64
    spec._srv_slots, spec._srv_n_slots = serving_kv_group_slots(layer_types)
    spec._srv_flat_type = "full_attention"
    spec._srv_per_layer, spec._srv_uploaded, spec._srv_last, spec._srv_seeded = None, {}, None, {}
    spec._srv_check_steps, spec._srv_checked, spec._srv_check_last = 1, {}, []  # GEMMA4_SPEC_VERIFY_CHECK=1
    spec._srv_measuring = False
    spec._srv_pool_v = spec._srv_pool_s = spec._srv_logits = spec._srv_backbone = None
    spec._srv_iota, spec._srv_records, spec._srv_shape_bufs = {}, {}, {}
    spec._srv_widths, spec._srv_shapes = {shape: list(ladder) for shape in shapes}, shapes
    spec._srv_captured = spec._srv_frozen = False
    spec._srv_shapes, spec._srv_projection = shapes, SimpleNamespace(describe=lambda: "fake plan")
    spec._srv_trace_region_mb = lambda: 0.0
    spec._srv_trace_region_size_mb = lambda: None
    # the REAL body runs over fakes of the drafter step, the packed verify, the seed gather, the masks and the
    # argmax (their outputs are op outputs like any other), so the assembly of the record's outputs is the real one
    spec._shared_kv = object()
    spec.assistant = SimpleNamespace(
        step=lambda tok, h, kv, d_ptl, d_pu, d_pi: (
            op(torch.zeros(1, 1, tok.host.shape[-1], vocab), fake.bfloat16),
            op(h.host.clone(), fake.bfloat16),
        )
    )
    spec.target.ttnn_packed_verify_forward = lambda x, **kw: (
        op(torch.zeros(1, 1, x.host.shape[-1], vocab), fake.bfloat16),
        op(torch.zeros(1, 1, x.host.shape[-1], backbone), fake.bfloat16),
    )
    spec._srv_seed_gather = lambda S: op(torch.zeros(1, 1, S["B"], backbone), fake.bfloat16)
    spec._srv_masks_device = lambda S, rec: (
        op(torch.zeros(1, 1, 32, rec["S_k"]), fake.bfloat16),
        op(torch.zeros(1, 1, 32, rec["S_k"]), fake.bfloat16),
    )
    spec._argmax_last = lambda logits, rows: op(torch.arange(rows).reshape(1, 1, rows), fake.uint32)
    # serving_prepare's device set through the real allocators and the real eager bodies (the iotas are made
    # here the way the real masks make them)
    for shape in shapes:
        B, K = shape
        S = spec._srv_alloc_shape(shape)
        assert S["v_pos"].shape == (1, B * (K + 1)) and S["v_pos_rows"].shape == (1, B * 4 * (K + 1))
        spec._srv_shape_bufs[shape] = S
        spec._srv_seed_rows(shape, [0] * B, [0] * B)
        for w in ladder:
            rec = spec._srv_alloc_width(shape, S, w)
            prepared.update(buf.address for buf in allocated)
            for _j0, n_keys in spec._srv_mask_chunks(w):
                spec._srv_iota_for(n_keys)
            outs = spec._srv_body(rec)
            assert outs == (rec["verify_x"], rec["vidx"]), "the body returns the record's persistent outputs"
            assert rec["verify_x"].shape == (1, B * (K + 1)) and rec["vidx"].shape == (1, 1, B * (K + 1))
            spec._srv_records[(shape, w)] = rec
    assert sorted(spec._srv_iota) == [1024, 2048, 8192, 32768]  # 263168 = 8 x 32768 + 1024
    prepared = {buf.address for buf in allocated}
    assert all(gone(ref) for ref in ops), "the eager bodies' transients are freed before they return"
    outputs = {key: (id(rec["verify_x"]), id(rec["vidx"])) for key, rec in spec._srv_records.items()}
    # the real capture binds only the record's own handles; an allocation inside it would be the one the
    # decode-trace guard counted at the first served request (30 x 128 B + 10 x 64 B on the 12B Galaxy)
    allocated.clear(), ops.clear()
    spec.serving_capture()
    assert spec._srv_captured and allocated == [] and all(gone(ref) for ref in ops)
    assert {key: (id(rec["verify_x"]), id(rec["vidx"])) for key, rec in spec._srv_records.items()} == outputs
    assert [rec["id"] for rec in spec._srv_records.values()] == traces and len(traces) == 20
    spec._srv_frozen = True
    ops.clear(), h2d.clear(), copies.clear(), forwards.clear()
    # every record and rung: a first-time refresh, row set and position, a reseed, a seeded replay, then a
    # continuing replay (seed columns from the last replay's verify hidden)
    group_tables, used = {}, []
    for shape in shapes:
        B, K = shape
        live = min(B, 3)
        for w, first, last in SpeculativeDecoder.serving_measure_positions(ladder, K, None):
            rec = spec._srv_records[(shape, w)]
            for pos in sorted({first, last}):
                tables = []
                for slot in spec._srv_slots:
                    t = group_tables.get((slot, pos))
                    if t is None:
                        t = torch.arange(wire_rows * spec._srv_blocks, dtype=torch.int32).reshape(wire_rows, -1) % 977
                        t = group_tables[(slot, pos)] = t + pos % 5 + slot
                    tables.append(t)
                spec.serving_refresh_page_tables(tables)
                rows = [(pos + 7 * i) % wire_rows for i in range(live)]
                positions = [max(0, pos - i) for i in range(live)]
                table = torch.zeros(B, spec._srv_blocks, dtype=torch.int32)
                for i, r in enumerate(rows):
                    table[i] = tables[0][r]
                spec.serving_reseed(shape, rows, [1 + i for i in range(live)], positions, table)
                result = spec.serving_step(shape, rows, [1 + i for i in range(live)], positions, table, [None] * live)
                assert (result.width, result.live) == (w, live) and result.logits_to_host().shape == (
                    live,
                    K + 1,
                    vocab,
                )
                assert result.drafts.shape == (live, K) and result.argmax_ids.shape == (live, K + 1)
                cols = [min(K, i) for i in range(live)]
                result = spec.serving_step(shape, rows, [2 + i for i in range(live)], positions, table, cols)
                assert result.width == w
                used.extend([rec["id"], rec["id"]])
    assert allocated == [], [buf.shape for buf in allocated]
    assert all(gone(ref) for ref in ops) and set(h2d) <= prepared and replays == used and copies == []
    # the verify check (one served replay per shape and rung here) ran its plain forward on the shape's persistent
    # seed inputs and compared row 0 of every live row: the recording argmax answers arange(rows), so the plain
    # ids are 0..live-1 and the packed row-0 ids r * P -- rows past the first disagree, the first agrees
    assert spec._srv_checked == {(shape, w): 1 for shape in shapes for w in ladder}
    assert spec._srv_check_last == [] or all(plain != packed for _r, _p, plain, packed in spec._srv_check_last)
    # a measurement replay (serving_measure_costs raises the flag around its zero-table steps) is never checked
    spec._srv_measuring = True
    spec.serving_step(shape, rows, [3 + i for i in range(live)], positions, table, cols)
    spec._srv_measuring = False
    used.append(rec["id"])
    assert spec._srv_checked == {(shape, w): 1 for shape in shapes for w in ladder}
    # the cost table itself: every row of the shape live for the reseed and the step lines (the covering record's
    # cost), a one-row step beside it kept as the partial line only where cheaper; the flag is down afterwards and
    # the measurement replays left the check counters alone
    before = dict(spec._srv_checked)
    costs = spec.serving_measure_costs(ladder[-1] - 8, repeats=1)
    assert not spec._srv_measuring and spec._srv_checked == before
    assert {key[:2] for key in costs if key[0] == "reseed"} == {("reseed", shape) for shape in shapes}
    assert {key for key in costs if key[0] == "step"} == {("step", shape, w) for shape in shapes for w in ladder}
    assert all(key[0] in ("reseed", "step", "partial") for key in costs)
    partials = [key for key in costs if key[0] == "partial"]
    assert all(costs[key][1] < costs[("step", key[1], key[2])][1] for key in partials)
    assert all(key[1][0] > 1 for key in partials), "a one-row shape has no partial replay"
    forwards.clear()
    assert {key: (id(rec["verify_x"]), id(rec["vidx"])) for key, rec in spec._srv_records.items()} == outputs
    assert all(
        buf.address in prepared and not buf.freed
        for rec in spec._srv_records.values()
        for buf in (rec["verify_x"], rec["vidx"])
    ), "the readbacks come from the records' persistent outputs"
    seeds = {
        (S["seed_tok"].address, S["seed_pu"].address, S["seed_pi"].address, S["B"])
        for S in spec._srv_shape_bufs.values()
    }
    assert set(forwards) <= seeds, "every reseed forward ran on its shape's persistent inputs"
    # a buffer a call would have to create is an error, never an allocation
    with expect_error(RuntimeError, "a replay never allocates"):
        spec._srv_iota_for(4096)
    logits = spec._srv_logits
    spec._srv_logits = None
    with expect_error(RuntimeError, "a replay never allocates"):
        spec._srv_publish_logits(_Buf(torch.zeros(1, 1, 32, vocab), fake.bfloat16, on_device=True))
    spec._srv_logits = logits
    pools = spec._srv_pool_v, spec._srv_pool_s
    spec._srv_pool_v = None
    with expect_error(RuntimeError, "a reseed never allocates"):
        spec.serving_reseed((1, 5), [0], [1], [0], torch.zeros(1, spec._srv_blocks, dtype=torch.int32))
    spec._srv_pool_v, spec._srv_pool_s = pools
    assert allocated == []
    # the release frees the reseed inputs with everything else and lifts the freeze
    bufs, recs = list(spec._srv_shape_bufs.values()), list(spec._srv_records.values())
    spec.serving_release()
    assert all(S[key].freed for S in bufs for key in ("seed_tok", "seed_pu", "seed_pi", "v_pos", "v_pos_rows"))
    assert not spec._srv_frozen
    assert all(rec[key].freed for rec in recs for key in ("verify_x", "vidx"))


def test_capture_and_step_refuse_to_run_out_of_order(monkeypatch, expect_error):
    spec = _bare_decoder()
    spec.serving_release()  # a no-op before prepare
    assert spec._srv_records is None
    with pytest.raises(RuntimeError, match="serving_prepare has not run"):  # allow-pytest.raises (pure)
        spec.serving_capture()
    spec._srv_records = {}
    spec._srv_shape_bufs = {(1, 5): {}}
    spec._srv_widths = {(1, 5): [1024]}
    with pytest.raises(RuntimeError, match="serving_capture has not run"):  # allow-pytest.raises (pure)
        spec.serving_step((1, 5), [0], [1], [0], torch.zeros(1, 4, dtype=torch.int32), [None])
    spec._srv_captured = True
    with pytest.raises(RuntimeError, match="serving_capture already ran"):  # allow-pytest.raises (pure)
        spec.serving_capture()
    _rail_allocates_nothing_after_capture(monkeypatch, expect_error)


def test_refresh_installs_one_host_table_per_layer_as_int64_rows():
    spec = _bare_decoder(n_layers=3)
    spec.serving_refresh_page_tables(None)  # no-op, like refresh_page_tables
    assert spec._srv_per_layer is None
    tables = [torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), None, torch.tensor([5, 6])]
    spec.serving_refresh_page_tables(tables)
    assert spec._srv_per_layer[0].dtype == torch.int64 and spec._srv_per_layer[0].shape == (2, 2)
    assert spec._srv_per_layer[1] is None
    assert spec._srv_per_layer[2].shape == (1, 2)  # a 1-D row is a one-row table
    # the layers of a KV group hand the same host table: it is converted once and kept shared (the group check
    # then compares identities; converting and comparing every layer cost ~10 ms a refresh at 32 rows x 4096)
    shared = torch.tensor([[7, 8], [9, 10]], dtype=torch.int32)
    spec.serving_refresh_page_tables([shared, shared, shared])
    assert spec._srv_per_layer[0] is spec._srv_per_layer[1] is spec._srv_per_layer[2]
    assert spec._srv_per_layer[0].dtype == torch.int64 and spec._srv_per_layer[0].tolist() == [[7, 8], [9, 10]]
    # tables that do not match the target are refused
    for tables, error, message in [
        ([torch.zeros(1, 2)] * 2, ValueError, "one table per target layer \\(3\\)"),
        (torch.zeros(1, 2), ValueError, "one table per target layer"),
        ([torch.zeros(1, 2), "x", None], TypeError, "must be a host torch tensor"),
        ([torch.zeros(1, 2, 2), None, None], ValueError, "must be \\[rows, blocks\\]"),
    ]:
        spec = _bare_decoder(n_layers=3)
        with pytest.raises(error, match=message):  # allow-pytest.raises (pure)
            spec.serving_refresh_page_tables(tables)
    # the step inputs are validated per row
    for rows, anchors, positions, message in [
        ([], [], [], "expected 1..5"),
        ([0, 1, 2, 3, 4, 5], [1] * 6, [0] * 6, "expected 1..5"),
        ([0, 1], [1], [0, 0], "lengths differ"),
        ([0, 0], [1, 1], [0, 0], "duplicate rows"),
        ([-1], [1], [0], "negative row"),
        ([0], [1], [-3], "negative anchor/position"),
    ]:
        with pytest.raises(ValueError, match=message):  # allow-pytest.raises (pure)
            SpeculativeDecoder._srv_check_rows((5, 5), rows, anchors, positions)
    # the step result reads its logits through its reader
    seen = []
    result = ServingStepResult(
        shape=(5, 5),
        width=1024,
        live=2,
        drafts=torch.zeros(2, 5, dtype=torch.int64),
        argmax_ids=torch.zeros(2, 6, dtype=torch.int64),
        logits=object(),
        _reader=lambda: seen.append(1) or torch.zeros(2, 6, 8, dtype=torch.bfloat16),
    )
    assert result.logits_to_host().shape == (2, 6, 8)
    assert seen == [1]
    bare = ServingStepResult((1, 5), 1024, 1, torch.zeros(1, 5), torch.zeros(1, 6), None)
    with pytest.raises(RuntimeError, match="no logits reader"):  # allow-pytest.raises (pure)
        bare.logits_to_host()


# --- packed verify KV write: the sharded-layout probe must run whenever the per-position loop runs ---


def test_the_packed_write_probe_follows_the_fallback_loop():
    from models.demos.gemma4.tt.attention.decode import packed_write_takes_fallback_loop

    for kv_staging, embed_idx, is_kv_shared, expected in [
        (None, None, False, True),  # no staging: the loop, probe needed
        ("staging", None, False, True),  # staging attached but a positional write: still the loop
        ("staging", "embed", False, False),  # loop-free staging fill: no probe
        (None, "embed", False, True),  # embed index without staging cannot fill loop-free
        ("staging", None, True, False),  # shared-KV layers write nothing
    ]:
        assert packed_write_takes_fallback_loop(kv_staging, embed_idx, is_kv_shared) is expected, (
            kv_staging,
            embed_idx,
            is_kv_shared,
        )


# Default width ladder (x4 from 2048) and the trace-region projection -------------------------------
_DEFAULT_SHAPES = [(1, 5), (5, 5), (8, 3), (16, 1)]


def test_default_ladder_is_x4_from_2048_up_to_the_top_rung():
    assert serving_width_ladder(262144, 5) == [2048, 8192, 32768, 131072, 263168]
    assert serving_width_ladder(131072, 5) == [2048, 8192, 32768, 131072, 132096]
    assert serving_width_ladder(8192, 1) == [2048, 8192, 9216]
    assert serving_width_ladder(1024, 5) == [2048]  # the top rung alone when it is the smallest
    # the top rung is independent of K across the default table (every tail rounds to 263168)
    assert {serving_width_ladder(262144, k)[-1] for _rows, k in _DEFAULT_SHAPES} == {263168}
    # below the smallest rung the record still serves
    ladder = serving_width_ladder(262144, 5)
    assert serving_width_for(ladder, 0, 5) == 2048
    assert serving_width_for(ladder, 2048 - 7, 5) == 2048
    assert serving_width_for(ladder, 2048 - 6, 5) == 8192
    # the rung override keeps the largest rungs
    assert serving_width_ladder(262144, 5, max_rungs=2) == ladder[-2:] == [131072, 263168]
    assert serving_width_ladder(262144, 5, max_rungs=1) == [263168]
    assert serving_width_ladder(262144, 5, max_rungs=99) == ladder
    with pytest.raises(ValueError, match="max_rungs must be >= 1"):  # allow-pytest.raises (pure)
        serving_width_ladder(262144, 5, max_rungs=0)
    # bad ladder parameters are refused
    for kwargs, message in [
        ({"max_context": 0, "draft_len": 5}, "must be >= 1"),
        ({"max_context": 1024, "draft_len": 0}, "must be >= 1"),
        ({"max_context": 1024, "draft_len": 5, "bucket": 100}, "multiple of 64"),
        ({"max_context": 1024, "draft_len": 5, "factor": 1}, "factor >= 2"),
        ({"max_context": 1024, "draft_len": 5, "smallest": 64}, "smallest >= bucket"),
    ]:
        with pytest.raises(ValueError, match=message):  # allow-pytest.raises (pure)
            serving_width_ladder(**kwargs)


def test_every_ladder_is_x4_spaced_and_ends_at_a_covering_top_rung():
    for max_context in (1024, 4096, 8192, 65536, 131072, 262144, 300000):
        for k in (1, 3, 5):
            ladder = serving_width_ladder(max_context, k)
            assert ladder == sorted(set(ladder)), (max_context, k)
            for a, b in zip(ladder, ladder[1:]):
                assert b == 4 * a or b == ladder[-1], (max_context, k, a, b)
            assert all(w % 1024 == 0 for w in ladder), (max_context, k)
            # the last position of the context gets a rung, and the rung count stays logarithmic
            assert serving_width_for(ladder, max_context - 1, k) == ladder[-1], (max_context, k)
            assert len(ladder) <= 6, (max_context, k)


def test_the_plan_takes_the_default_ladder_per_shape_and_the_projection_counts_its_records():
    shapes, widths = serving_plan(_DEFAULT_SHAPES, None, 16, max_context=262144)
    assert shapes == _DEFAULT_SHAPES
    assert all(len(ws) == 5 for ws in widths.values())
    assert widths[(16, 1)] == [2048, 8192, 32768, 131072, 263168]
    _shapes, capped = serving_plan(_DEFAULT_SHAPES, None, 16, max_context=262144, max_rungs=2)
    assert all(ws == [131072, 263168] for ws in capped.values())
    with pytest.raises(ValueError, match="needs max_context"):  # allow-pytest.raises (pure)
        serving_plan(_DEFAULT_SHAPES, None, 16)
    # the projection counts records per shape and adds the base
    proj = serving_trace_projection(shapes, widths, record_mb=32.0, region_mb=512, base_mb=87)
    assert proj.records == 20
    assert proj.per_shape == {shape: 5 for shape in _DEFAULT_SHAPES}
    assert proj.projected_mb == 87 + 20 * 32.0
    assert proj.fits is False
    assert "EXCEEDS" in proj.describe()
    assert serving_trace_projection(shapes, widths, record_mb=32.0, region_mb=960, base_mb=87).fits is True
    unknown = serving_trace_projection(shapes, widths, record_mb=32.0, base_mb=87)
    assert unknown.fits is None and "not readable" in unknown.describe()
    assert SPEC_RECORD_MB_DEFAULT == 32.0
    with pytest.raises(ValueError, match="record_mb must be > 0"):  # allow-pytest.raises (pure)
        serving_trace_projection(shapes, widths, record_mb=0)
    # The 12B measurement fits the Galaxy but not QB2 at the full ladder: measured per-record trace sizes
    # (12B TP8, width-independent) on top of the 87 MB plain traces.
    measured = {(1, 5): 27.1, (5, 5): 31.9, (8, 3): 25.5, (16, 1): 21.9}
    total = 87 + sum(len(widths[s]) * mb for s, mb in measured.items())
    assert round(total, 1) == 619.0
    assert total > 512 and total < 960
    _shapes, three = serving_plan(_DEFAULT_SHAPES, None, 16, max_context=262144, max_rungs=3)
    assert 87 + sum(len(three[s]) * mb for s, mb in measured.items()) < 512


# Alias guards and the seed-column convention -------------------------------------------------------
def _calls_in(fn_name):
    tree = _module()
    fn = next(f for f in _functions(tree) if f.name == fn_name)
    return [n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]


def test_argmax_frees_its_slice_sources_only_when_they_are_copies():
    """Both partial-slice frees in _argmax_last (the >32-row tile slices and the trimmed ids) are guarded by
    _same_device_buffer, like the pad above them: a ttnn that serves a slice as a view must not lose its source.
    The body stages are separate methods the device test can run eagerly, with the guarded frees inside them."""
    calls = _calls_in("_argmax_last")
    assert calls.count("_same_device_buffer") >= 3, calls
    tree = _module()
    names = {fn.name for fn in _functions(tree)}
    assert {"_srv_seed_gather", "_srv_assemble_verify_x", "_srv_masks_device"} <= names
    body_calls = _calls_in("_srv_body")
    assert "_same_device_buffer" not in body_calls  # the guarded frees moved into the stage methods
    body = next(f for f in _functions(tree) if f.name == "_srv_body")
    method_calls = {
        n.func.attr for n in ast.walk(body) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert {"_srv_seed_gather", "_srv_assemble_verify_x", "_srv_masks_device"} <= method_calls, method_calls


def test_the_seed_column_is_the_pre_contract_current_mode():
    """The adapter passes seed_cols = committed count - 1 = accepted drafts; the pre-contract B=1 production
    mode picked the same verify row (_fused_shift_seed_row("current") = min(accepted, K)), and the gather
    index lands on that row of the previous replay's hidden layout."""
    spec = SpeculativeDecoder.__new__(SpeculativeDecoder)
    spec._fused_shift_seed = "current"
    K = 5
    for accepted in range(K + 1):
        committed = accepted + 1
        assert committed - 1 == spec._fused_shift_seed_row(accepted, K)
        idx = serving_seed_index((5, K), [3], [committed - 1], (K, [9, 3]), {})
        assert idx[0] == 1 * (K + 1) + accepted


# Packed-verify GQA groups at nkv_local >= 2 (the TP4 column) --------------------------------------
def test_packed_kv_groups_are_contiguous_head_blocks_in_the_head_major_order():
    """Row h*P+p of the packed Q (and of the [B,1,H*P,S_k] mask the body H-repeats) lands in the KV group the
    head-sharded GQA map gives its head, so the kernel's row-range grouping by num_q_heads/num_kv_heads on the
    logical head count is the right one for every TP (nkv_local = 1 on TP8, 2 on TP4 for the 12B)."""
    for h_local, nkv_local in [(2, 1), (4, 1), (4, 2), (8, 4), (8, 2)]:
        for P in (2, 4, 6):
            groups = packed_verify_kv_groups(h_local, nkv_local, P)
            assert [g for g, _a, _b in groups] == list(range(nkv_local)), (h_local, nkv_local, P)
            assert groups[0][1] == 0 and groups[-1][2] == h_local * P, (h_local, nkv_local, P)
            heads_per_kv = h_local // nkv_local
            for g, first, last in groups:
                assert last - first == (h_local * P) // nkv_local  # num_q_heads / num_kv_heads, rows per KV head
                for row in range(first, last):
                    head, _p = divmod(row, P)
                    assert head // heads_per_kv == g, (row, head, g)
    # uneven head splits are refused
    with pytest.raises(ValueError, match="do not split evenly"):  # allow-pytest.raises (pure)
        packed_verify_kv_groups(6, 4, 2)
    with pytest.raises(ValueError, match="must be >= 1"):  # allow-pytest.raises (pure)
        packed_verify_kv_groups(4, 0, 2)


# Width-matched verify tables per layer type (hybrid kv-cache groups) ------------------------------
_HYBRID_12B = {"sliding_attention": 256, "full_attention": 128}  # tokens per block, HMA byte-unified pools
_SINGLE_POOL = {"sliding_attention": 64, "full_attention": 64}
_NO_RING = {"sliding_attention": None, "full_attention": None}


def test_hybrid_pools_get_one_width_per_type_spanning_the_same_mask():
    """Each type's table times its own block size spans the rung: one block size for both types would
    give the full-attention table half the mask (the packed SDPA then attends 1024 tokens under a
    2048-wide mask and reads the mask rows at the wrong pitch for every batch row past the first)."""
    for s_k, sliding, full in [(2048, 8, 16), (8192, 32, 64), (263168, 1028, 2056)]:
        widths = serving_verify_table_widths(s_k, _HYBRID_12B, _NO_RING, blocks=4096)
        assert widths == {"sliding_attention": sliding, "full_attention": full}, s_k
        assert all(widths[lt] * _HYBRID_12B[lt] == s_k for lt in widths), s_k
    # a single pool is the control
    assert serving_verify_table_widths(2048, _SINGLE_POOL, _NO_RING, blocks=4096) == {
        "sliding_attention": 32,
        "full_attention": 32,
    }
    # a bounded ring sizes the sliding table from the ring
    ring = {"sliding_attention": 2048, "full_attention": None}
    widths = serving_verify_table_widths(8192, _SINGLE_POOL, ring, blocks=4096)
    assert widths == {"sliding_attention": 32, "full_attention": 128}
    # a hybrid ring pool: the ring in tokens over the sliding type's own block size
    widths = serving_verify_table_widths(8192, _HYBRID_12B, ring, blocks=4096)
    assert widths == {"sliding_attention": 8, "full_attention": 64}
    # the adapter's table width never caps a type: the top rung rounds max_context + K + 2 up, so a type at the
    # plugin's own block size always needs more columns than the adapter's table (zero-padded, block 0). The 31B
    # at TP4 (sliding block 64, full 128 under hybrid groups; a 4096-column plugin table at 262144 context): a
    # 4096-column sliding table under the 263168-wide mask made the kernel read the mask's second row tile
    # (packed rows 32-47) 32 tiles early, 1024 keys ahead, and its 256K verify accepted drafts the target would
    # not have produced; 128K (2048 columns) and the one-row-tile geometries (12B TP4, 31B TP8) were untouched.
    widths = serving_verify_table_widths(263168, _HYBRID_12B, _NO_RING, blocks=1500)
    assert widths == {"sliding_attention": 1028, "full_attention": 2056}
    _31B_TP4 = {"sliding_attention": 64, "full_attention": 128}
    widths = serving_verify_table_widths(263168, _31B_TP4, _NO_RING, blocks=4096)
    assert widths == {"sliding_attention": 4112, "full_attention": 2056}
    assert all(widths[lt] * _31B_TP4[lt] == 263168 for lt in widths)
    assert serving_verify_table_widths(131072, _31B_TP4, _NO_RING, blocks=4096) == {
        "sliding_attention": 2048,
        "full_attention": 1024,
    }
    # the paged packed verify refuses a table that does not span its mask (the kernel's mask pitch)
    assert packed_verify_mask_span(4112, 64, 263168) == 263168 and packed_verify_mask_span(2056, 128, 263168) == 263168
    with pytest.raises(ValueError, match="262144 keys but the mask is 263168"):  # allow-pytest.raises (pure)
        packed_verify_mask_span(4096, 64, 263168)
    # misaligned geometries are refused
    for s_k, bs, ring, blocks, message in [
        (2048, {"full_attention": 96}, {}, 64, "rung 2048 is not a multiple"),
        (8192, {"sliding_attention": 96}, {"sliding_attention": 2048}, 64, "ring 2048 is not a multiple"),
        (2048, {"full_attention": 0}, {}, 64, "block size must be >= 1"),
        (0, {"full_attention": 64}, {}, 64, "must be >= 1"),
    ]:
        with pytest.raises(ValueError, match=message):  # allow-pytest.raises (pure)
            serving_verify_table_widths(s_k, bs, ring, blocks)


# ===== KV group slots: the serving records follow vLLM's hybrid KV-cache grouping, not the layer type
from models.demos.gemma4.tt.spec_decode import serving_check_group_tables, serving_kv_group_slots  # noqa: E402


def _layer_types(pattern, repeats):
    return list(pattern) * repeats


def test_kv_group_slots_follow_vllm_strided_groups_for_the_12b_and_31b_patterns():
    """vLLM takes group_size = min(count per type) and gives group g the layers
    layers_of_type[g::num_groups]: five sliding groups (layers i, i+6, i+12, ...)
    and one full group for both Gemma 4 dense patterns."""
    for repeats in (8, 10):  # 12B: 40 sliding + 8 full; 31B: 50 sliding + 10 full
        types = _layer_types(["sliding_attention"] * 5 + ["full_attention"], repeats)
        slots, n_slots = serving_kv_group_slots(types)
        assert n_slots == 6
        assert slots == [i % 6 for i in range(len(types))]
        for g in range(5):
            members = [i for i, s in enumerate(slots) if s == g]
            assert members == [6 * k + g for k in range(repeats)]
        assert [i for i, s in enumerate(slots) if s == 5] == [6 * k + 5 for k in range(repeats)]
    # a uniform model collapses to one slot
    slots, n_slots = serving_kv_group_slots(["full_attention"] * 12)
    assert (slots, n_slots) == ([0] * 12, 1)
    assert serving_kv_group_slots([]) == ([], 0)
    # 7 sliding + 3 full: 7 is not a multiple of 3, so the sliding layers get a slot each (always correct)
    types = ["sliding_attention"] * 7 + ["full_attention"] * 3
    slots, n_slots = serving_kv_group_slots(types)
    assert slots[:7] == list(range(7))
    assert slots[7:] == [7, 7, 7]
    assert n_slots == 8


def test_group_table_check_accepts_identical_tables_per_slot_and_rejects_a_mismatch(expect_error):
    types = _layer_types(["sliding_attention"] * 5 + ["full_attention"], 2)
    slots, n_slots = serving_kv_group_slots(types)
    tables = [torch.arange(8, dtype=torch.int64).reshape(2, 4) + 100 * slots[i] for i in range(len(types))]
    serving_check_group_tables(tables, slots)  # identical per slot
    serving_check_group_tables([None] * len(types), slots)  # every layer flat
    shared = [tables[slots[i]] for i in range(len(types))]
    serving_check_group_tables(shared, slots)  # the same object per slot
    bad = list(tables)
    bad[6] = bad[6] + 1  # layer 6 is slot 0 like layer 0, but differs
    with expect_error(ValueError, "layers 0 and 6 differ inside KV group slot 0"):
        serving_check_group_tables(bad, slots)
    mixed = list(tables)
    mixed[6] = None
    with expect_error(ValueError, "slot 0"):
        serving_check_group_tables(mixed, slots)


def test_every_layer_verifies_and_reseeds_through_its_own_group_table(monkeypatch):
    """vLLM-grouped per-layer tables with DISTINCT block ids per group: after the
    upload, the table bound to every layer (its slot) holds that layer's own
    installed rows, for the drafter/reseed tables and the width-clipped verify
    tables alike; padding rows mirror row 0. The per-type records of the old
    code gave 32 of the 40 sliding layers another group's block ids."""
    types = _layer_types(["sliding_attention"] * 5 + ["full_attention"], 8)
    slots, n_slots = serving_kv_group_slots(types)
    B, blocks, s_k = 4, 16, 2048
    bs = {"sliding_attention": 256, "full_attention": 128}
    widths = {lt: s_k // b for lt, b in bs.items()}  # 8 / 16 columns
    installed = [torch.arange(8 * blocks, dtype=torch.int64).reshape(8, blocks) + 1000 * slots[i] for i in range(48)]

    spec = _bare_decoder(n_layers=48)
    spec._srv_slots, spec._srv_n_slots = slots, n_slots
    spec._srv_blocks = blocks
    spec._pv_ring = {"sliding_attention": None, "full_attention": None}
    spec._pv_ring_rep = {"sliding_attention": 0, "full_attention": 5}
    spec.target.last_kv_layer_by_type = {"sliding_attention": 46, "full_attention": 47}
    spec.target.hf_config = SimpleNamespace(layer_types=types)
    spec.serving_refresh_page_tables(installed)  # passes the slot check
    shape = (B, 5)
    d_slots = [SimpleNamespace(shape=(B, blocks), tag=("d", s)) for s in range(n_slots)]
    v_slots = [SimpleNamespace(shape=(B, widths[types[slots.index(s)]]), tag=("v", s)) for s in range(n_slots)]
    spec._srv_shape_bufs[shape] = {"B": B, "d_ptl_slots": d_slots, "d_ptl_list": [d_slots[s] for s in slots]}
    rec = {"S_k": s_k, "v_ptl_slots": v_slots, "v_ptl_list": [v_slots[s] for s in slots]}
    uploads = {}
    monkeypatch.setattr(
        spec,
        "_srv_copy_if_changed",
        lambda key, host, dev, dtype, layout=None: uploads.__setitem__(dev.tag, host.clone()),
    )

    rows, live = [2, 7], 2
    spec._srv_upload_tables(shape, rows, live, torch.zeros(B, blocks, dtype=torch.int64), rec)

    assert len(uploads) == 2 * n_slots
    for i in range(48):
        want = installed[i][rows]
        d = uploads[spec._srv_shape_bufs[shape]["d_ptl_list"][i].tag]
        assert torch.equal(d[:live], want.to(torch.int32)), f"layer {i} drafter table"
        assert torch.equal(d[live:], d[0].expand(B - live, -1)), f"layer {i} padding rows"
        v = uploads[rec["v_ptl_list"][i].tag]
        w = widths[types[i]]
        assert v.shape == (B, w), f"layer {i} verify width"
        assert torch.equal(v[:live], want[:, :w].to(torch.int32)), f"layer {i} verify table"
    # the old per-type collapse would have given layer 1 (slot 1) layer 0's (slot 0) ids:
    assert not torch.equal(
        uploads[rec["v_ptl_list"][1].tag][:live], installed[0][rows][:, : widths["sliding_attention"]].to(torch.int32)
    )


# ===== Acceptance-adaptive speculation (tt/spec_policy.py): the pure policy, exercised from the draft-length
# policy tests above (helpers, not collected).
# ===== Acceptance-adaptive draft length (tt/spec_policy.py): the pure policy functions.
from models.demos.gemma4.tt.spec_policy import (  # noqa: E402
    SPEC_ADAPTIVE_ENV,
    SPEC_CALIB_MIN_ENV,
    SPEC_CALIB_PROBE_STEPS_ENV,
    SPEC_ENTRY_HORIZON_ENV,
    SPEC_EWMA_ENV,
    SPEC_EXIT_GAIN_ENV,
    SPEC_MIN_GAIN_ENV,
    SPEC_PLAIN_STEP_MS_ENV,
    SPEC_PROBE_STEPS_ENV,
    SPEC_RETRY_OVERHEAD_ENV,
    SPEC_RETRY_STEPS_ENV,
    SPEC_STATS_STEPS_ENV,
    SPEC_TRIAL_GAIN_ENV,
    AcceptanceStats,
    Calibration,
    CostLine,
    ModeStats,
    RecordCandidate,
    SpecPolicy,
    SpecPolicyConfig,
    cost_at,
    describe_costs,
    mode_name,
    rank_records,
)

_PLAIN = 20.0
# Step costs per record: 2.5x / 2.5x / 1.75x / 1.5x the plain step.
_COSTS = {(1, 5): 50.0, (5, 5): 50.0, (8, 3): 35.0, (16, 1): 30.0}


def _cands(costs, n=1, width=2048):
    return [RecordCandidate(shape, width, ms) for shape, ms in costs.items() if shape[0] >= n]


def _row(cond, alpha=0.2):
    stats = AcceptanceStats(5, alpha)
    stats.cond = list(cond)
    return stats


def _policy_policy_config_defaults_are_on_with_the_documented_knobs(expect_error=None):
    assert SpecPolicyConfig.from_env({}) == SpecPolicyConfig(
        adaptive=True,
        min_gain=1.10,
        exit_gain=0.95,
        entry_horizon=64,
        retry_steps=8,
        retry_overhead=0.03,
        probe_steps=32,
        ewma=0.1,
        plain_step_ms=None,
    )


def _policy_policy_config_reads_every_knob_from_the_env(expect_error=None):
    env = {
        SPEC_ADAPTIVE_ENV: "0",
        SPEC_MIN_GAIN_ENV: "1.25",
        SPEC_EXIT_GAIN_ENV: "0.9",
        SPEC_ENTRY_HORIZON_ENV: "16",
        SPEC_RETRY_STEPS_ENV: "4",
        SPEC_RETRY_OVERHEAD_ENV: "0.05",
        SPEC_PROBE_STEPS_ENV: "0",
        SPEC_EWMA_ENV: "0.5",
        SPEC_PLAIN_STEP_MS_ENV: "22.5",
    }
    assert SpecPolicyConfig.from_env(env) == SpecPolicyConfig(False, 1.25, 0.9, 16, 4, 0.05, 0, 0.5, (22.5,))
    env.update({SPEC_STATS_STEPS_ENV: "64", SPEC_CALIB_MIN_ENV: "3", SPEC_CALIB_PROBE_STEPS_ENV: "0"})
    assert SpecPolicyConfig.from_env(env) == SpecPolicyConfig(False, 1.25, 0.9, 16, 4, 0.05, 0, 0.5, (22.5,), 64, 3, 0)
    env.update({SPEC_STATS_STEPS_ENV: "64", SPEC_CALIB_MIN_ENV: "3", SPEC_CALIB_PROBE_STEPS_ENV: "0"})
    assert SpecPolicyConfig.from_env(env) == SpecPolicyConfig(False, 1.25, 0.9, 16, 4, 0.05, 0, 0.5, (22.5,), 64, 3, 0)
    assert SpecPolicyConfig.from_env({SPEC_PLAIN_STEP_MS_ENV: " "}).plain_step_ms is None
    assert SpecPolicyConfig.from_env({SPEC_PLAIN_STEP_MS_ENV: "31,83"}).plain_step_ms == (31.0, 83.0)
    assert SpecPolicyConfig.from_env({}).trial_gain == 0.6 and SpecPolicyConfig.from_env(env).trial_gain == 0.6
    assert SpecPolicyConfig.from_env({SPEC_TRIAL_GAIN_ENV: "0.8"}).trial_gain == 0.8


_BAD_POLICY_KNOBS = [
    (SPEC_ADAPTIVE_ENV, "maybe", "must be 1 or 0"),
    (SPEC_MIN_GAIN_ENV, "0.9", "must be at least 1.0"),
    (SPEC_EXIT_GAIN_ENV, "1.5", r"must be in \[0.0, 1.1\]"),
    (SPEC_RETRY_STEPS_ENV, "-1", "must be at least 0"),
    (SPEC_PROBE_STEPS_ENV, "2.5", "must be an integer"),
    (SPEC_EWMA_ENV, "1.5", r"must be in \[0.0, 1.0\]"),
    (SPEC_PLAIN_STEP_MS_ENV, "-3", "must be positive"),
    (SPEC_TRIAL_GAIN_ENV, "-0.1", "must be at least 0.0"),
]


def _policy_bad_policy_knobs_are_refused_naming_the_knob(expect_error):
    for name, value, message in _BAD_POLICY_KNOBS:
        with expect_error(ValueError, f"{name} {message}"):
            SpecPolicyConfig.from_env({name: value})


def _policy_acceptance_stats_update_only_the_positions_the_verify_reached(expect_error=None):
    stats = AcceptanceStats(5, 0.5)
    stats.observe(accepted=2, verified=5)  # drafts 1 and 2 accepted, draft 3 rejected, 4 and 5 never reached
    assert stats.cond == [1.0, 1.0, 0.5, 1.0, 1.0] and stats.observed == [1, 1, 1, 0, 0]
    assert stats.ewma_accepted == 2.0 and stats.spec_steps == 1
    stats.observe(accepted=3, verified=3)  # a shorter verify, every carried draft accepted: no rejection seen
    assert stats.cond == [1.0, 1.0, 0.75, 1.0, 1.0] and stats.observed == [2, 2, 2, 0, 0]
    assert stats.ewma_accepted == 2.5 and stats.spec_steps == 2
    stats.observe(accepted=0, verified=0)  # a draftless verify observes nothing
    assert stats.spec_steps == 2
    stats.observe(accepted=9, verified=9)  # clamped to the draft length
    assert stats.observed == [3, 3, 3, 1, 1] and stats.ewma_accepted == 3.75


def _policy_expected_tokens_is_one_plus_the_running_product_of_the_conditionals(expect_error=None):
    stats = _row([0.9, 0.5, 0.5, 0.0, 1.0])
    assert stats.accept_probs(5) == pytest.approx([0.9, 0.45, 0.225, 0.0, 0.0])
    assert stats.expected_tokens(1) == pytest.approx(1.9)
    assert stats.expected_tokens(3) == pytest.approx(2.575)
    assert stats.expected_tokens(5) == pytest.approx(2.575)
    assert stats.expected_tokens(0) == 1.0


def _policy_a_new_row_starts_from_the_batch_prior_and_the_prior_learns_from_every_row(expect_error=None):
    policy = SpecPolicy(SpecPolicyConfig(ewma=0.5), 5)
    first = policy.new_row()
    assert first.cond == [1.0] * 5, "optimistic until the first verify"
    policy.observe(first, accepted=0, verified=5)
    assert policy.prior.cond[0] == 0.5 and first.cond[0] == 0.5
    second = policy.new_row()
    assert second.cond == [0.5, 1.0, 1.0, 1.0, 1.0] and second.observed == [0] * 5 and second.ewma_accepted is None
    policy.observe(second, accepted=5, verified=5)
    assert first.cond[0] == 0.5 and second.cond[0] == 0.75 and policy.prior.cond[0] == 0.75
    # a row's own observations count at once: at a position with n < ROW_MIN_OBSERVATIONS of its own it reads
    # n / (n + 3) of its own mean outcome and the rest of the prior (one verify at 5 accepted moves it most of
    # the way), from three on its own EWMA as before; a request that stays plain observes nothing
    informed = SpecPolicy(SpecPolicyConfig(ewma=0.5), 5)
    informed.prior.cond, informed.prior.informed = [0.2] * 5, [True] * 5
    row = informed.new_row()
    assert row.effective_cond(5) == pytest.approx([0.2] * 5) and row.own_sum == [0.0] * 5
    informed.observe(row, accepted=5, verified=5)
    assert informed.prior.cond[0] == pytest.approx(0.6) and row.cond[0] == pytest.approx(0.6)
    assert row.effective_cond(5) == pytest.approx([0.7] * 5), "0.25 x own 1.0 + 0.75 x prior 0.6"
    assert row.expected_tokens(5) == pytest.approx(1 + 0.7 + 0.49 + 0.343 + 0.2401 + 0.16807)
    informed.observe(row, accepted=0, verified=5)
    assert row.effective_cond(2) == pytest.approx(
        [0.38, 0.7]
    ), "n=2 at draft 1: 0.4 x 0.5 + 0.6 x 0.3; draft 2 unreached"
    informed.observe(row, accepted=0, verified=5)
    assert row.observed[0] == 3 and row.cond[0] == pytest.approx(0.15)
    assert row.effective_cond(1) == pytest.approx([0.15]), "three own observations: the row's own EWMA, as before"
    untouched = informed.new_row()
    assert untouched.effective_cond(1) == pytest.approx([informed.prior.cond[0]]) and untouched.spec_steps == 0


def _policy_rank_records_scores_expected_tokens_per_cost_relative_to_the_plain_step(expect_error=None):
    rows = [_row([1.0] * 5), _row([0.5] * 5)]
    ranked = rank_records(_cands(_COSTS), rows, _PLAIN)
    # (8,3): 4 + 1.875 tokens over two rows at 1.75x; (1,5) and (5,5): 6 + 1.96875 at 2.5x; (16,1): 2 + 1.5 at 1.5x
    assert [c.shape for c in ranked] == [(8, 3), (1, 5), (5, 5), (16, 1)]
    assert ranked[0].expected_tokens == pytest.approx(5.875) and ranked[0].gain == pytest.approx(5.875 / 2 / 1.75)
    assert ranked[1].gain == pytest.approx(7.96875 / 2 / 2.5) and ranked[3].gain == pytest.approx(3.5 / 2 / 1.5)
    assert all(c.reason == "gain" and c.rows == 2 and c.entry_ms == 0.0 for c in ranked)
    assert ranked[0].tokens_per_row == pytest.approx(5.875 / 2)
    with expect_error(ValueError, "negative entry cost"):
        rank_records([RecordCandidate((1, 5), 2048, 50.0, -1.0)], rows, _PLAIN)
    with expect_error(ValueError, "positive plain step cost"):
        rank_records(_cands(_COSTS), rows, 0.0)
    with expect_error(ValueError, "non-positive cost"):
        rank_records([RecordCandidate((1, 5), 2048, 0.0)], rows, _PLAIN)
    with expect_error(ValueError, "at least one drafting row"):
        rank_records(_cands(_COSTS), [], _PLAIN)
    # a partial replay drafts for its members only: a row left out commits one token from the plain decode
    partial = rank_records([RecordCandidate((8, 3), 2048, 35.0 + 20.0, 0.0, members=(0,))], rows, _PLAIN)[0]
    assert partial.members == (0,) and partial.expected_tokens == pytest.approx(4.0 + 1.0)
    assert partial.gain == pytest.approx(2.5 / 2.75)
    with expect_error(ValueError, "members"):
        rank_records([RecordCandidate((8, 3), 2048, 35.0, 0.0, members=(0, 2))], rows, _PLAIN)
    with expect_error(ValueError, "members"):
        rank_records([RecordCandidate((8, 3), 2048, 35.0, 0.0, members=())], rows, _PLAIN)


def _policy_the_policy_picks_k5_at_high_acceptance_and_drops_to_k3_k1_and_plain_as_it_falls(expect_error=None):
    policy = SpecPolicy(SpecPolicyConfig(probe_steps=0), 5)

    def choice(cond):
        return policy.decide(_cands(_COSTS), [_row(cond)], _PLAIN, 5)

    assert choice([1.0] * 5).shape == (1, 5)  # 6 tokens at 2.5x: 2.4x
    assert choice([0.7] * 5).shape == (8, 3)  # 2.53 at 1.75x = 1.45x beats 2.94 at 2.5x = 1.18x
    assert choice([0.8, 0.2, 0.2, 0.2, 0.2]).shape == (16, 1)  # only the first draft lands: 1.8 at 1.5x = 1.20x
    # 1.4 at 1.5x = 0.93x: below the exit gain, but a speculating batch never leaves on one step
    held = choice([0.4] * 5)
    assert held.shape == (16, 1) and held.reason == "hold" and policy.below_exit == 1
    assert choice([0.4] * 5) is None and policy.below_exit == 0 and policy.last_k == 0
    # the entry cooldown: a batch that just left waits retry_steps plain steps before any reseeding entry (the
    # 12B Galaxy at 32 users ran enter-exit-enter cycles at a 250 ms reseed each, the arrival waves re-trialing
    # between); the retry's cadence counts the same steps, note_plain_steps too, an entry-free record waits for
    # nothing
    cooled = SpecPolicy(SpecPolicyConfig(probe_steps=0, retry_steps=3, retry_overhead=1.0), 5)
    good, weak = [_row([0.7] * 5)], [_row([0.4] * 5)]
    entering = [RecordCandidate((8, 3), 2048, 35.0, 230.0)]  # 2.53 tokens at 1.75x: 1.45x, the reseed pays
    assert cooled.decide(entering, good, _PLAIN, 3).reason == "enter" and cooled.since_exit is None
    assert cooled.decide(_cands(_COSTS), weak, _PLAIN, 3).reason == "hold"
    assert cooled.decide(_cands(_COSTS), weak, _PLAIN, 3) is None and cooled.since_exit == 0 and cooled.cooling()
    assert cooled.decide(entering, good, _PLAIN, 3) is None and cooled.last_reason == "cooldown"
    assert not cooled.would_speculate(entering, good, _PLAIN) and cooled.since_exit == 1
    assert cooled.decide(entering, good, _PLAIN, 3, fresh=True) is None and cooled.last_reason == "cooldown"
    assert not cooled.would_speculate(entering, good, _PLAIN, fresh=True) and cooled.since_exit == 2
    cooled.note_plain_steps()
    assert cooled.since_exit == 3 and not cooled.cooling()
    assert cooled.decide(entering, good, _PLAIN, 3).reason == "enter" and cooled.since_exit is None
    free = SpecPolicy(SpecPolicyConfig(probe_steps=0, retry_steps=3), 5)
    assert free.decide(_cands(_COSTS), good, _PLAIN, 3).reason == "gain"
    assert (
        free.decide(_cands(_COSTS), weak, _PLAIN, 3).reason == "hold"
        and free.decide(_cands(_COSTS), weak, _PLAIN, 3) is None
    )
    assert free.decide(_cands(_COSTS), good, _PLAIN, 3).reason == "gain" and free.since_exit is None
    assert SpecPolicy(SpecPolicyConfig(), 5).decide(_cands(_COSTS), weak, _PLAIN, 3) is None, "declined, not an exit"


def _policy_the_margin_decides_between_a_marginal_record_and_the_plain_step(expect_error=None):
    row = _row([0.8, 0.2, 0.2, 0.2, 0.2])  # (16,1): 1.8 tokens at 1.5x the plain step = 1.20x
    assert SpecPolicy(SpecPolicyConfig(min_gain=1.15), 5).decide(_cands(_COSTS), [row], _PLAIN, 5).shape == (16, 1)
    assert SpecPolicy(SpecPolicyConfig(min_gain=1.25), 5).decide(_cands(_COSTS), [row], _PLAIN, 5) is None
    # hysteresis: once in, the same row is held down to the exit gain, and leaves after EXIT_STEPS below it
    policy = SpecPolicy(SpecPolicyConfig(min_gain=1.15, exit_gain=1.0), 5)
    assert policy.decide(_cands(_COSTS), [row], _PLAIN, 5).reason == "gain"
    weaker = _row([0.6, 0.2, 0.2, 0.2, 0.2])  # 1.6 tokens: 1.07x, between the exit and the entry gain
    assert policy.decide(_cands(_COSTS), [weaker], _PLAIN, 5).reason == "hold"
    weakest = _row([0.4, 0.2, 0.2, 0.2, 0.2])  # 1.4 tokens: 0.93x, below the exit gain
    assert policy.decide(_cands(_COSTS), [weakest], _PLAIN, 5).reason == "hold" and policy.below_exit == 1
    assert policy.decide(_cands(_COSTS), [weakest], _PLAIN, 5) is None
    assert policy.decide(_cands(_COSTS), [weaker], _PLAIN, 5) is None, "1.07x does not re-enter: no hold from plain"


def _policy_the_cost_table_is_consumed_as_measured(expect_error=None):
    """The same acceptance picks another record when the measured costs change."""
    row = _row([0.7] * 5)
    cheap_k5 = {**_COSTS, (1, 5): 36.0}  # 1.8x: 2.94 tokens -> 1.63x, above (8,3)'s 1.45x
    assert SpecPolicy(SpecPolicyConfig(), 5).decide(_cands(cheap_k5), [row], _PLAIN, 5).shape == (1, 5)
    dear = {shape: ms * 2 for shape, ms in _COSTS.items()}
    assert SpecPolicy(SpecPolicyConfig(), 5).decide(_cands(dear), [row], _PLAIN, 5) is None
    # entering costs the reseed: the saving over the entry horizon, 64 x (2.53 x 20 - 35) = 1000 ms, must cover it
    enter = SpecPolicy(SpecPolicyConfig(), 5).decide([RecordCandidate((8, 3), 2048, 35.0, 900.0)], [row], _PLAIN, 3)
    assert enter.reason == "enter" and enter.entry_ms == 900.0
    assert (
        SpecPolicy(SpecPolicyConfig(), 5).decide([RecordCandidate((8, 3), 2048, 35.0, 1100.0)], [row], _PLAIN, 3)
        is None
    )
    assert (
        SpecPolicy(SpecPolicyConfig(entry_horizon=8), 5).decide(
            [RecordCandidate((8, 3), 2048, 35.0, 200.0)], [row], _PLAIN, 3
        )
        is None
    )
    # every cost is a line in the position, read at the step's top position
    line = CostLine(0, 30.0, 1000, 80.0)
    assert [line.at(p) for p in (-5, 0, 500, 1000, 5000)] == [30.0, 30.0, 55.0, 80.0, 80.0]
    assert cost_at(30.0, 700) == 30.0 and cost_at((0, 30.0, 1000, 80.0), 250) == 42.5 and cost_at(line, 250) == 42.5
    assert CostLine.of(line) is line and CostLine.flat(7.0, 100) == CostLine(0, 7.0, 100, 7.0)
    assert CostLine(0, 30.0, 0, 80.0).at(10) == 30.0, "a line without a span is flat"
    # online calibration: a mode reads the warmup table until it has min_samples counted steps, then the table
    # scaled by the median ratio of its walls to the table at each step's own position; a dirty step (a prefill
    # ran inside it) counts like a clean one: the mode in force served that wall (the batch points lost to plain
    # on arrival stalls, joins and switches each clean step looked profitable through)
    calib = Calibration(2)
    mode = ("step", (8, 3), 2048)
    assert calib.step_ms(mode, 35.0) == 35.0 and not calib.calibrated(mode)
    calib.note_step(mode, 80.0, 3, True, 40.0, 1)  # ratio 2.0
    calib.note_step(mode, 90.0, 3, False, 40.0, 1, cause="prefill")  # dirty: counted, calibrates, not clean
    assert calib.calibrated(mode) and calib.step_ms(mode, 35.0) == pytest.approx(35.0 * 2.125)
    assert calib.modes[mode].steps == 2 and calib.modes[mode].clean == 1 and calib.modes[mode].busy == 2
    calib.note_step(mode, 100.0, 3, True, 40.0, 1)  # ratio 2.5
    assert calib.step_ms(mode, 35.0) == pytest.approx(35.0 * 2.25)
    assert calib.step_ms(mode, 136.0) == pytest.approx(136.0 * 2.25), "the position stays in the table"
    assert calib.tokens_per_row(mode) == 3.0 and calib.modes[mode].tokens == 9
    assert calib.dirty == {"prefill": [1, 90.0]} and calib.total == [3, 270.0, 9]
    # the measured tokens per row are the window's MEAN (a throughput: a row taking 1,1,1,5,6 tokens on five
    # verifies commits 2.8 per step, where the median 1 read the record as a loss and the policy left a 1.18x
    # record at 256K), pooled over the records of one draft length at the rung: the drafter's acceptance does
    # not depend on the record's rows, so an unreplayed sibling reads the calibrated one's instead of the
    # model (whose deep positions sit at the prior) and cannot outrank it on the model alone
    skew = Calibration(5)
    for tokens in (1, 1, 1, 5, 6):
        skew.note_step(("step", (1, 5), 263168), 150.0, tokens, True, 150.0, 1)
    assert skew.tokens_per_row(("step", (1, 5), 263168)) == pytest.approx(2.8)
    assert skew.tokens_per_row(("step", (5, 5), 263168)) == pytest.approx(2.8), "same draft length, same rung"
    assert skew.tokens_per_row(("step", (8, 3), 263168)) is None, "another draft length is its own"
    assert skew.tokens_per_row(("step", (1, 5), 2048)) is None, "another rung is its own"
    assert not skew.calibrated(("step", (5, 5), 263168)), "its step cost still reads the table"
    skew.note_step(("step", (5, 5), 263168), 200.0, 10, True, 200.0, 5)  # 2 per row joins the pool
    assert skew.tokens_per_row(("step", (1, 5), 263168)) == pytest.approx(16.0 / 6)
    skew.note_step(("plain", 1), 80.0, 1, True, 80.0, 1)
    assert skew.tokens_per_row(("plain", 1)) is None, "the plain step pools nothing and needs its own samples"
    # an outlier: a clean sample above 3x the running median once four samples exist; reported, never averaged in
    for wall in (84.0, 88.0):
        calib.note_step(mode, wall, 3, True, 40.0, 1)
    calib.note_step(mode, 800.0, 3, True, 40.0, 1)
    assert calib.modes[mode].outliers == [1, 800.0] and calib.modes[mode].clean == 4
    # an idle interval (every row of the previous step ended before this one: the gap between requests) is
    # reported, never counted; a dirty stall of any size the mode was in force for is counted
    calib.note_step(mode, 16000.0, 1, False, 40.0, 1, cause="prefill", idle=True)
    assert calib.dirty["idle"] == [1, 16000.0] and calib.modes[mode].idle == [1, 16000.0]
    assert calib.modes[mode].busy == 5 and calib.step_ms(mode, 40.0) == pytest.approx(40.0 * 2.2)
    calib.note_step(mode, 400.0, 3, False, 40.0, 1, cause="prefill")  # 10x the table: an arrival's prefill
    assert calib.modes[mode].busy == 6 and calib.step_ms(mode, 40.0) == pytest.approx(40.0 * 2.225)
    assert calib.dirty == {"prefill": [2, 490.0], "idle": [1, 16000.0]} and calib.tokens_per_row(mode) == 3.0
    assert calib.reseed_ms((8, 3), 250.0) == 250.0
    calib.note_reseed((8, 3), 300.0, 250.0)
    calib.note_reseed((8, 3), 350.0, 250.0)
    assert calib.reseed_ms((8, 3), 250.0) == pytest.approx(250.0 * 1.3) and calib.reseeds[(8, 3)][0] == 2
    assert calib.reseed_ms((8, 3), 500.0) == pytest.approx(650.0), "a ratio too: the position stays in the line"
    assert calib.steps_without("plain") == 8 and calib.steps_without("step") == 2  # a dirty or idle step resets nothing
    calib.note_step(("plain", 8), 31.0, 8, True, 30.0, 8)
    assert calib.steps_without("plain") == 0 and calib.steps_without("step") == 3
    with expect_error(ValueError, "min_samples"):
        Calibration(0)
    # the entry horizon follows the measured request length (steps), capped by GEMMA4_SPEC_ENTRY_HORIZON
    policy = SpecPolicy(SpecPolicyConfig(entry_horizon=64), 5)
    assert policy.entry_horizon() == 64 and policy.entry_horizon(10) == 10
    policy.note_request_steps(34)
    assert policy.request_steps == 34.0 and policy.entry_horizon() == 34
    policy.note_request_steps(10)
    assert policy.request_steps == pytest.approx(28.0) and policy.entry_horizon() == 28
    assert policy.entry_horizon(500) == 64
    # a saving of 64 x 15.7 = 1002 ms covers a 500 ms reseed over the configured horizon, not over 20 steps
    entering = [RecordCandidate((8, 3), 2048, 35.0, 500.0)]
    row = _row([0.7] * 5)  # 2.53 tokens: 15.7 ms saved per step at plain 20
    assert SpecPolicy(SpecPolicyConfig(), 5).decide(entering, [row], _PLAIN, 3).reason == "enter"
    assert SpecPolicy(SpecPolicyConfig(), 5).decide(entering, [row], _PLAIN, 3, remaining=20) is None
    short = SpecPolicy(SpecPolicyConfig(), 5)
    short.note_request_steps(20)
    assert short.decide(entering, [row], _PLAIN, 3) is None, "the measured request length caps the horizon"
    assert not short.would_speculate(entering, [row], _PLAIN) and short.would_speculate(entering, [row], _PLAIN, 64)
    # the shape's measured run length caps it too: steps served on the shape's records per reseed of the shape,
    # from the second reseed on. Two reseeds and 20 steps (a run of 10: 157 ms of saving) refuse the 500 ms
    # entry, 80 steps (a run of 40: 628 ms) take it; one reseed measures nothing
    churn = SpecPolicy(SpecPolicyConfig(), 5)
    churn.calibration.note_reseed((8, 3), 250.0, 250.0)
    assert churn.calibration.run_length((8, 3)) is None and churn.decide(entering, [row], _PLAIN, 3).reason == "enter"
    churn.calibration.note_reseed((8, 3), 250.0, 250.0)
    for _ in range(20):
        churn.calibration.note_step(("step", (8, 3), 2048), 60.0, 16, True, 57.0, 8)
    assert churn.calibration.run_length((8, 3)) == 10.0
    assert churn.decide(entering, [row], _PLAIN, 3) is None and not churn.would_speculate(entering, [row], _PLAIN)
    assert churn.entry_pays(rank_records([RecordCandidate((8, 3), 2048, 35.0, 150.0)], [row], _PLAIN)[0], _PLAIN)
    for _ in range(60):
        churn.calibration.note_step(("step", (8, 3), 8192), 60.0, 16, False, 57.0, 8, cause="rows")
    assert churn.calibration.run_length((8, 3)) == 40.0 and churn.decide(entering, [row], _PLAIN, 3).reason == "enter"
    assert churn.calibration.run_length((16, 1)) is None, "another shape's reseeds are its own"
    # the measured tokens per row of a calibrated record replace the acceptance model
    measured = [RecordCandidate((8, 3), 2048, 35.0, 0.0, tokens_per_row=1.2)]
    assert rank_records(measured, [row], _PLAIN)[0].expected_tokens == pytest.approx(1.2)
    assert SpecPolicy(SpecPolicyConfig(), 5).decide(measured, [row], _PLAIN, 3) is None
    partial = [RecordCandidate((8, 3), 2048, 55.0, 0.0, members=(0,), tokens_per_row=3.0)]
    assert rank_records(partial, [row, row], _PLAIN)[0].expected_tokens == pytest.approx(3.0 + 1.0)


def _policy_after_n_declined_steps_the_policy_retries_at_the_smallest_draft_length(expect_error=None):
    policy = SpecPolicy(SpecPolicyConfig(retry_steps=3, retry_overhead=0.2, probe_steps=0), 5)
    row = _row([0.4] * 5)  # (16,1): 1.4 tokens at 1.5x = 0.93x, the cheapest record and above min_gain / 4
    kinds = []
    for _ in range(8):
        choice = policy.decide(_cands(_COSTS), [row], _PLAIN, 5)
        kinds.append(None if choice is None else (choice.shape, choice.reason))
    assert kinds == [None, None, None, ((16, 1), "retry"), None, None, None, ((16, 1), "retry")]
    assert policy.declined == 0 and policy.last_k == 1 and policy.trial
    assert policy.decide(_cands(_COSTS), [row], _PLAIN, 5) is None, "a trial below the entry gain is not held"
    # the overhead gate: (30 - 20) / (N x 20) <= 0.03 needs 17 plain steps, whatever retry_steps says, so the
    # retry is the 18th decision
    policy = SpecPolicy(SpecPolicyConfig(retry_steps=3, probe_steps=0), 5)
    first = next(i for i in range(1, 40) if policy.decide(_cands(_COSTS), [row], _PLAIN, 5) is not None)
    assert first == 18
    # a reseed is part of the overhead: (900 + 30 - 20) / (N x 20) <= 0.2 needs 228 plain steps
    policy = SpecPolicy(SpecPolicyConfig(retry_steps=3, retry_overhead=0.2, probe_steps=0), 5)
    reseeding = [RecordCandidate(s, 1024, ms, 900.0) for s, ms in _COSTS.items()]
    assert all(policy.decide(reseeding, [row], _PLAIN, 5) is None for _ in range(228))
    assert policy.decide(reseeding, [row], _PLAIN, 5).reason == "retry"
    # a record the statistics rule out (gain below min_gain / 4) is never retried
    hopeless = SpecPolicy(SpecPolicyConfig(retry_steps=1, retry_overhead=1.0, probe_steps=0), 5)
    assert all(hopeless.decide([RecordCandidate((16, 1), 1024, 120.0)], [row], _PLAIN, 5) is None for _ in range(10))
    never = SpecPolicy(SpecPolicyConfig(retry_steps=0, retry_overhead=1.0), 5)
    assert all(never.decide(_cands(_COSTS), [row], _PLAIN, 5) is None for _ in range(20))
    # a new request's trial: declined otherwise, a fresh step replays the cheapest record (entry plus step) once
    # when the prior's gain clears trial_gain, whatever the retry cadence says; never held, reason "trial"
    fresh = SpecPolicy(SpecPolicyConfig(retry_steps=0, retry_overhead=1.0, probe_steps=0), 5)
    trial = fresh.decide(_cands(_COSTS), [row], _PLAIN, 5, fresh=True)
    assert (trial.shape, trial.reason, trial.gain) == ((16, 1), "trial", pytest.approx(1.4 * 20 / 30)) and fresh.trial
    assert fresh.decide(_cands(_COSTS), [row], _PLAIN, 5) is None, "a trial below the entry gain is not held"
    assert fresh.would_speculate(_cands(_COSTS), [row], _PLAIN, fresh=True)
    assert not fresh.would_speculate(_cands(_COSTS), [row], _PLAIN)
    dear = [RecordCandidate((16, 1), 2048, 50.0)]  # 1.4 tokens at 2.5x plain: 0.56x, below the trial gain
    assert fresh.decide(dear, [row], _PLAIN, 5, fresh=True) is None and fresh.last_reason == "declined"
    assert SpecPolicy(SpecPolicyConfig(trial_gain=0.5), 5).decide(dear, [row], _PLAIN, 5, fresh=True).reason == "trial"
    with_entry = [RecordCandidate((1, 5), 2048, 50.0, 170.0), RecordCandidate((16, 1), 2048, 30.0, 300.0)]
    assert fresh.decide(with_entry, [row], _PLAIN, 5, fresh=True).shape == (1, 5), "cheapest entry plus step"
    # a batch that is not all new: its trial needs the hypothesis that the new rows accept every draft to clear
    # the entry bar, since the trial reseeds every row. Eight rows at a 0.3 conditional (1.42 tokens a step at
    # K=3) on the (8,3) at 35 ms (1.75x plain): the prior's gain 0.81 clears trial_gain, but one new row at 4
    # tokens makes 1.74 tokens/row = 0.99x < min_gain (declined); three new rows make 2.39 = 1.36x (trial), and
    # the (8,3)'s 230 ms reseed must pay over the joining rows' horizon (not over 10 steps: 12.8 ms x 10)
    batch = [_row([0.3] * 5) for _ in range(8)]
    eight = SpecPolicy(SpecPolicyConfig(retry_steps=0, retry_overhead=1.0, probe_steps=0), 5)
    only83 = [c for c in _cands(_COSTS, n=8) if c.shape == (8, 3)]
    assert rank_records(only83, batch, _PLAIN)[0].gain == pytest.approx(1.417 * 20 / 35, abs=1e-3)
    hyped = rank_records(only83, batch, _PLAIN, optimistic=(0,))
    assert hyped[0].expected_tokens == pytest.approx(7 * 1.417 + 4, abs=1e-2)
    assert eight.decide(only83, batch, _PLAIN, 3, fresh=(0,)) is None and eight.last_reason == "declined"
    assert not eight.would_speculate(only83, batch, _PLAIN, fresh=(0,))
    assert eight.decide(only83, batch, _PLAIN, 3, fresh=(0, 1, 2)).reason == "trial"
    assert eight.would_speculate(only83, batch, _PLAIN, fresh=(0, 1, 2))
    # with the (16,1) at 30 ms beside it (1.3 tokens at 1.5x: 0.87x, the cheapest) one new row makes 0.93x and
    # three 1.04x: still declined; five new rows of eight make 1.14x
    both = _cands(_COSTS, n=8)
    assert rank_records(both, batch, _PLAIN)[0].shape == (16, 1)
    assert eight.decide(both, batch, _PLAIN, 3, fresh=(0, 1, 2)) is None
    assert eight.decide(both, batch, _PLAIN, 3, fresh=(0, 1, 2, 3, 4)).shape == (16, 1)
    reseeding8 = [RecordCandidate((8, 3), 2048, 35.0, 230.0), RecordCandidate((16, 1), 2048, 30.0, 300.0)]
    eight = SpecPolicy(SpecPolicyConfig(retry_steps=0, retry_overhead=1.0, probe_steps=0), 5)
    assert eight.decide(reseeding8, batch, _PLAIN, 3, fresh=(0, 1, 2)).reason == "trial"
    assert eight.decide(reseeding8, batch, _PLAIN, 3, remaining=10, fresh=(0, 1, 2)) is None
    assert eight.decide(reseeding8, batch, _PLAIN, 3, fresh=True).reason == "trial", "a batch all new: the floor"
    # a record serving measured at the rung keeps its measured tokens per row for the new rows too: seven new
    # rows of eight (the 26B's arrival waves) no longer re-trial an (8,3) measured at 1.46 tokens a row (0.83x),
    # while the same record unmeasured is their trial; the floor still refuses a measured record below it
    measured8 = [RecordCandidate((8, 3), 2048, 35.0, 230.0, tokens_per_row=1.46)]
    wave = tuple(range(7))
    assert rank_records(measured8, batch, _PLAIN, optimistic=wave)[0].expected_tokens == pytest.approx(8 * 1.46)
    assert eight.decide(measured8, batch, _PLAIN, 3, fresh=wave) is None and eight.last_reason == "declined"
    assert eight.decide([RecordCandidate((8, 3), 2048, 35.0, 230.0)], batch, _PLAIN, 3, fresh=wave).reason == "trial"
    good = [RecordCandidate((8, 3), 2048, 35.0, 230.0, tokens_per_row=2.4)]  # measured above the bar: an entry
    assert eight.decide(good, batch, _PLAIN, 3, fresh=wave).reason == "enter"
    assert (
        eight.decide(measured8, batch, _PLAIN, 3, fresh=True) is None and eight.last_reason == "declined"
    ), "all new: a measured record below break-even (0.83x) gets no trial either"
    even = [RecordCandidate((8, 3), 2048, 35.0, 230.0, tokens_per_row=1.75)]  # measured at break-even: 1.00x
    assert eight.decide(even, batch, _PLAIN, 3, fresh=True).reason == "trial"
    assert (
        SpecPolicy(SpecPolicyConfig(trial_gain=1.2, retry_steps=0), 5).decide(even, batch, _PLAIN, 3, fresh=True)
        is None
    )
    assert rank_records(even, batch, _PLAIN)[0].measured and not rank_records(reseeding8, batch, _PLAIN)[0].measured
    with pytest.raises(ValueError, match="fresh rows .* outside the 8 rows"):  # allow-pytest.raises (pure)
        eight.decide(reseeding8, batch, _PLAIN, 3, fresh=(8,))
    # the calibration probe: a speculating batch serves one plain step once calib_probe_steps passed without
    # a clean plain sample and the probe's cost (the reseed to resume, the tokens forgone) is amortized
    probing = SpecPolicy(SpecPolicyConfig(calib_probe_steps=3, retry_overhead=2.0, probe_steps=0), 5)
    rows = [_row([1.0] * 5)]
    kinds = []
    for _ in range(5):
        choice = probing.decide(_cands(_COSTS), rows, _PLAIN, 5)
        kinds.append(None if choice is None else choice.reason)
        if choice is not None:
            probing.calibration.note_step(("step", choice.shape, choice.width), 50.0, 6, True, 50.0, 1)
        else:
            probing.calibration.note_step(("plain", 1), 20.0, 1, True, 20.0, 1)
    assert kinds == ["gain", "gain", "gain", None, "gain"] and probing.last_reason == "gain"
    assert probing.calibration.steps_without("plain") == 1
    never = SpecPolicy(SpecPolicyConfig(calib_probe_steps=0, probe_steps=0), 5)
    for _ in range(6):
        assert never.decide(_cands(_COSTS), rows, _PLAIN, 5) is not None
        never.calibration.note_step(("step", (1, 5), 1024), 50.0, 6, True, 50.0, 1)
    costly = SpecPolicy(SpecPolicyConfig(calib_probe_steps=3, retry_overhead=0.03, probe_steps=0), 5)
    for _ in range(6):  # 5 tokens forgone x 20 ms over 3+ steps is far above 3 % of them: no probe
        assert costly.decide(_cands(_COSTS), rows, _PLAIN, 5) is not None
        costly.calibration.note_step(("step", (1, 5), 1024), 50.0, 6, True, 50.0, 1)


def _policy_below_the_cap_the_policy_probes_the_next_larger_draft_length_every_n_steps(expect_error=None):
    policy = SpecPolicy(SpecPolicyConfig(probe_steps=3), 5)
    rows = [_row([0.7] * 5) for _ in range(5)]
    out = [(c.shape, c.reason) for c in (policy.decide(_cands(_COSTS, n=5), rows, _PLAIN, 5) for _ in range(7))]
    assert out == [
        ((8, 3), "gain"),
        ((8, 3), "gain"),
        ((5, 5), "probe"),
        ((8, 3), "gain"),
        ((8, 3), "gain"),
        ((5, 5), "probe"),
        ((8, 3), "gain"),
    ]
    # at the cap, and on a plain step, the count restarts
    policy.decide(_cands(_COSTS, n=5), rows, _PLAIN, 5)
    assert policy.below_cap == 2
    policy.decide(_cands(_COSTS, n=5), [_row([1.0] * 5) for _ in range(5)], _PLAIN, 5)
    assert policy.below_cap == 0 and policy.last_k == 5
    policy.decide(_cands(_COSTS, n=5), rows, _PLAIN, 5)
    for _ in range(2):  # two decisions below the exit gain leave for the plain step
        policy.decide(_cands(_COSTS, n=5), [_row([0.1] * 5) for _ in range(5)], _PLAIN, 5)
    assert policy.below_cap == 0 and policy.last_k == 0
    # entering (a reseed this step) is not a step below the cap: no probe starts from it
    entering = SpecPolicy(SpecPolicyConfig(probe_steps=1), 5)
    choice = entering.decide([RecordCandidate((8, 3), 2048, 35.0, 1.0)], rows, _PLAIN, 5)
    assert choice.reason == "enter" and entering.below_cap == 0
    # without a larger record the probe step keeps the choice
    only = SpecPolicy(SpecPolicyConfig(probe_steps=1), 5)
    assert only.decide([RecordCandidate((16, 1), 2048, 30.0)], rows, _PLAIN, 5).reason == "gain"


def _policy_measure_positions_time_each_rung_at_the_first_position_it_serves(expect_error=None):
    widths = [2048, 8192, 32768]
    positions = SpeculativeDecoder.serving_measure_positions(widths, 5)
    assert positions == [(2048, 0, 2041), (8192, 2042, 8185), (32768, 8186, 32761)]
    for w, first, last in positions:
        assert serving_width_for(widths, first, 5) == w and serving_width_for(widths, last, 5) == w
    assert SpeculativeDecoder.serving_measure_positions(widths, 5, top_position=10000) == [
        (2048, 0, 2041),
        (8192, 2042, 8185),
        (32768, 8186, 10000),
    ]
    assert SpeculativeDecoder.serving_measure_positions([4096, 1024, 1024], 1) == [(1024, 0, 1021), (4096, 1022, 4093)]
    assert SpeculativeDecoder.serving_measure_positions([4096], 1, top_position=0) == [(4096, 0, 0)]


def _policy_describe_costs_lists_the_plain_step_then_every_record(expect_error=None):
    costs = {
        ("step", (1, 5), 2048): (0, 50.0, 2041, 52.0),
        ("step", (1, 5), 8192): 52.5,
        ("reseed", (1, 5)): CostLine(0, 18.0, 1018, 30.0),
        ("step", (16, 1), 2048): 30.0,
        ("partial", (16, 1), 2048): (0, 24.0, 2045, 24.5),  # a one-row replay the warmup found cheaper
    }
    assert describe_costs(costs, CostLine(0, 20.0, 1018, 60.0)) == (
        "plain 20.0@0..60.0@1018 | (1, 5): S_k 2048 50.0@0..52.0@2041, S_k 8192 52.5; reseed 18.0@0..30.0@1018 | "
        "(16, 1): S_k 2048 30.0 [1 row 24.0@0..24.5@2045]"
    )
    assert describe_costs({}, 20.0) == "plain 20.0" and describe_costs({}, None) == "plain unmeasured"
    # the plain step per decode bucket, and the stats line's pieces
    assert describe_costs({}, {1: 22.7, 32: CostLine(0, 30.7, 262138, 217.1)}) == (
        "plain b1 22.7, b32 30.7@0..217.1@262138"
    )
    assert mode_name(("plain", 8)) == "plain[b=8]" and mode_name(("step", (8, 3), 2048)) == "(8,3)@2048"
    stats = ModeStats()
    assert stats.describe("plain[b=8]") == "plain[b=8]: 0 steps, 0 tokens, 0 clean"
    stats.note(31.0, 8, True, 30.0, 8)
    stats.note(33.0, 8, True, 30.0, 8)
    stats.note(99.0, 8, False, 30.0, 8)
    assert stats.describe("plain[b=8]") == (
        "plain[b=8]: 3 steps, 24 tokens, 2 clean, 32.0 ms/step, 8.00 tokens/step, 4.0 ms/token, "
        "total 54.3 ms/step 6.8 ms/token, ratio 1.10x"
    )
    for wall in (30.0, 32.0, 900.0):
        stats.note(wall, 8, True, 30.0, 8)
    assert stats.describe("plain[b=8]").endswith(
        "3.9 ms/token, total 45.0 ms/step 5.6 ms/token, ratio 1.07x, outliers 1 at 900.0 ms"
    )
    assert stats.tokens_per_row() == 1.0 and stats.clean == 4 and stats.busy == 5
    stats.note(5000.0, 8, False, 30.0, 8, idle=True)
    assert stats.describe("plain[b=8]").endswith("outliers 1 at 900.0 ms, idle 1 at 5000.0 ms") and stats.busy == 5
    calib = Calibration(8)
    calib.note_step(("plain", 8), 31.0, 8, True, 30.0, 8)
    calib.note_step(("plain", 8), 250.0, 8, False, 30.0, 8, cause="switch")
    calib.note_reseed((8, 3), 248.0, 240.0)
    calib.note_phase("replay", 41.5)
    calib.note_phase("replay", 42.5)
    calib.note_prefill()
    assert calib.describe() == (
        "plain[b=8]: 2 steps, 16 tokens, 1 clean, 31.0 ms/step, 8.00 tokens/step, 3.9 ms/token, "
        "total 140.5 ms/step 17.6 ms/token, ratio 4.68x | "
        "reseeds 1 at 248.0 ms | dirty 1 steps, 250 ms: switch 1/250 | total 2 steps, 281 ms, 16 tokens, "
        "17.6 ms/token | phases ms/call: replay 42.0 | prefills 1 | calibrated: none"
    )
    calib.flush_window()
    assert calib.describe() == "reseeds 0 | prefills 0 | calibrated: none"
    assert calib.describe(window=False).startswith("plain[b=8]: 2 steps, 16 tokens, 1 clean")
