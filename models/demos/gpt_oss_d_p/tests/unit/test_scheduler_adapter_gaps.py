# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TDD tests for the common/prefill scheduler-adapter gaps in ``gpt_oss_d_p``.

Each test pins ONE gap from ``tt/runners/adapters/SCHEDULER_INTEGRATION.md``. Every test in this file
is expected to FAIL against today's code (documenting the current gap) and PASS after that gap is
closed. Run the file to snapshot today's state; watch a test flip green as you land the fix.

Two tiers:

* **Source-level tests** (no device) — cheap; verify the runtime/adapter API surface. These fail
  right now because the methods / behaviors do not exist yet.
* **Galaxy integration tests** — driven by ``models.demos.common.prefill.runners.prefill_runner``
  via subprocess. Marked ``@pytest.mark.requires_mesh_topology(...)``; skipped locally, run on a
  full galaxy. Each corresponds to a T1..T5 scenario in SCHEDULER_INTEGRATION.md.

Env for the galaxy tests (same as the standalone harness — see galaxy_prefill_kv_pcc.py):
  PREFILL_HF_MODEL, PREFILL_TTNN_CACHE, PREFILL_TRACE_DIR
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROWS, COLS = 4, 8  # galaxy mesh shape


# ---------------------------------------------------------------------------
# Gap 1 — set_layer_ack_channel on the runtime (LayerAck → scheduler channel)
# ---------------------------------------------------------------------------


def test_gap1_runtime_has_set_layer_ack_channel():
    """Runtime must expose ``set_layer_ack_channel(channel)`` so the engine can register the
    scheduler's per-layer counter channel in single-rank request mode
    (prefill_runner.py:1097). Fails today: method does not exist."""
    from models.demos.gpt_oss_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    assert hasattr(TtPrefillRuntime, "set_layer_ack_channel"), (
        "TtPrefillRuntime missing set_layer_ack_channel; engine cannot register the LayerAck channel "
        "(gap 1 in SCHEDULER_INTEGRATION.md)"
    )
    sig = inspect.signature(TtPrefillRuntime.set_layer_ack_channel)
    # one positional (self) + one channel arg
    assert len(sig.parameters) == 2, f"set_layer_ack_channel signature unexpected: {sig}"


def test_gap1_runtime_init_reserves_layer_ack_slot():
    """Runtime's ``__init__`` must initialize a ``_on_layer_complete`` attr (starts as None) so that
    ``prefill_chunk`` can forward it to the model without a NameError when no channel is registered.
    Fails today: attribute not set."""
    from models.demos.gpt_oss_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    src = inspect.getsource(TtPrefillRuntime.__init__)
    assert "_on_layer_complete" in src, (
        "TtPrefillRuntime.__init__ must initialize self._on_layer_complete (gap 1)"
    )


def test_gap1_prefill_chunk_forwards_on_layer_complete():
    """``prefill_chunk`` must pass ``on_layer_complete=self._on_layer_complete`` into
    ``model.prefill_forward``. Without this the model's layer-loop seam (model.py:210–211) never
    fires under scheduler control. Fails today: not passed."""
    from models.demos.gpt_oss_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    src = inspect.getsource(TtPrefillRuntime.prefill_chunk)
    assert "on_layer_complete" in src, (
        "TtPrefillRuntime.prefill_chunk does not forward on_layer_complete to model.prefill_forward "
        "(gap 1)"
    )


# ---------------------------------------------------------------------------
# Gap 2 — request-mode input path: make_chunk_input returns uint32 tokens,
#                                  prefill_chunk embeds internally
# ---------------------------------------------------------------------------


def test_gap2_make_chunk_input_returns_uint32_tokens():
    """The engine's request loop pushes SP-sharded uint32 token tensors from the H2D socket straight
    into ``runtime.prefill_chunk`` (prefill_runner.py:543). ``make_chunk_input`` must build the SAME
    per-chip layout — uint32 token IDs, NOT an already-embedded bf16 activation — so standalone and
    request modes feed one code path. Fails today: make_chunk_input calls
    ``model.prepare_inputs_prefill`` which embeds + returns bf16.

    This is a source-level check (avoids opening a mesh). See the M3 reference at
    minimax_m3/tt_prefill_runtime.py:192–220 for the target shape."""
    from models.demos.gpt_oss_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    src = inspect.getsource(TtPrefillRuntime.make_chunk_input)
    assert "prepare_inputs_prefill" not in src, (
        "make_chunk_input still returns an embedded activation via model.prepare_inputs_prefill; "
        "request-mode H2D delivers raw uint32 tokens — return those instead (gap 2)"
    )
    # Explicit uint32 mesh-mapped tensor construction, like M3's make_chunk_input.
    assert "uint32" in src, "make_chunk_input must build a uint32 token tensor (gap 2)"


def test_gap2_prefill_chunk_embeds_internally():
    """With ``make_chunk_input`` returning uint32 tokens, ``prefill_chunk`` must embed inside the
    runtime before calling ``prefill_forward``. Fails today: prefill_chunk hands the input straight
    to prefill_forward assuming it is already embedded."""
    from models.demos.gpt_oss_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    src = inspect.getsource(TtPrefillRuntime.prefill_chunk)
    embeds = ("embedding" in src) or ("_embed_tokens" in src) or ("prepare_inputs_prefill" in src)
    assert embeds, (
        "prefill_chunk must embed uint32 tokens internally (call an ``_embed_tokens`` helper or "
        "model.embedding / prepare_inputs_prefill) once make_chunk_input stops embedding (gap 2)"
    )


# ---------------------------------------------------------------------------
# Gap 3 — bf16 weight-load fallback in the adapter
# ---------------------------------------------------------------------------


def test_gap3_adapter_does_not_unconditionally_pass_empty_state_dict():
    """``GptOssPrefillAdapter.build_runtime`` currently builds ``TtPrefillRuntime(..., state_dict={})``
    unconditionally. If the tilized weight cache is stale/incomplete this produces a silent-garbage
    runtime under the engine. Either fall back to a real bf16 load (mirroring
    minimax_m3.py:142–177) OR hard-assert cache completeness. Fails today: literal
    ``state_dict={}`` with no completeness check and no ``load_state_dict`` fallback."""
    from models.demos.gpt_oss_d_p.tt.runners.adapters import gpt_oss as adapter_mod

    src = inspect.getsource(adapter_mod.GptOssPrefillAdapter.build_runtime)
    unconditional_empty = "state_dict={}" in src and "load_state_dict" not in src and "cache_is_complete" not in src
    # Also allow a loud assertion path as an acceptable fallback per SCHEDULER_INTEGRATION.md.
    has_hard_assert = "assert" in src and (".tensorbin" in src or "cache" in src.lower())
    assert not unconditional_empty or has_hard_assert, (
        "build_runtime hands TtPrefillRuntime state_dict={} unconditionally with no cache-completeness "
        "check and no bf16 fallback — a stale/incomplete tilized cache will silently produce garbage "
        "under the engine (gap 3)"
    )


def test_gap3_adapter_todo_removed():
    """The gap-3 TODO block at adapters/gpt_oss.py:137–145 should be removed once the fallback (or a
    hard-assert) is wired. Fails today: TODO present."""
    from models.demos.gpt_oss_d_p.tt.runners.adapters import gpt_oss as adapter_mod

    src = inspect.getsource(adapter_mod.GptOssPrefillAdapter.build_runtime)
    assert "TODO(P5, engine integration)" not in src, (
        "gap-3 TODO still present in build_runtime — remove once the weight-load fallback (or "
        "cache-completeness assert) is in place"
    )


# ---------------------------------------------------------------------------
# Gap 4 — (optional) migration hooks. These are xfail by default; flip to
# expected-pass only if you decide to wire migration for GPT-OSS.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="Gap 4 is P2 / optional; unmark once migration hooks are wired.", strict=False)
def test_gap4_runtime_has_build_kv_chunk_table():
    from models.demos.gpt_oss_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    assert hasattr(TtPrefillRuntime, "build_kv_chunk_table"), "gap 4: no build_kv_chunk_table on runtime"


@pytest.mark.xfail(reason="Gap 4 is P2 / optional; unmark once migration hooks are wired.", strict=False)
def test_gap4_runtime_has_kv_migration_base_address():
    from models.demos.gpt_oss_d_p.tt.tt_prefill_runtime import TtPrefillRuntime

    assert hasattr(TtPrefillRuntime, "kv_migration_base_address"), (
        "gap 4: no kv_migration_base_address on runtime"
    )


# ---------------------------------------------------------------------------
# Galaxy integration tests (T1..T5 from SCHEDULER_INTEGRATION.md).
#
# These invoke prefill_runner (and prefill_producer where needed) as subprocesses and grep the log
# for the pass markers each scenario produces. They require a full galaxy + the standalone-harness
# environment. Skipped when PREFILL_GAP_TESTS_GALAXY=1 is unset so the source-level tests can run
# stand-alone on any host.
# ---------------------------------------------------------------------------

_GALAXY_GATE = pytest.mark.skipif(
    os.environ.get("PREFILL_GAP_TESTS_GALAXY", "0") != "1",
    reason="Set PREFILL_GAP_TESTS_GALAXY=1 on a full galaxy to run the T1..T5 scenarios",
)


def _required_env(*, needs_trace: bool):
    """Fail fast if the harness env isn't set; makes it obvious what's missing.

    ``PREFILL_TRACE_DIR`` is ONLY needed for PCC validation (T1/T2). Prefill compute itself never
    reads the trace — the request-mode / LayerAck / multi-user tests (T3/T4/T5) run without it and
    the producer synthesizes tokens."""
    required = ["PREFILL_HF_MODEL", "PREFILL_TTNN_CACHE"]
    if needs_trace:
        required.append("PREFILL_TRACE_DIR")
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        pytest.fail(f"missing required env for galaxy integration: {missing}")


def _run(cmd, extra_env, timeout_s):
    env = os.environ.copy()
    env.update(extra_env)
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)


def _assert_pcc_ok(stdout: str, floor: float = 0.99):
    """Parse the runtime's ``[kv-pcc] min PCC across N layers: K=... V=...`` line and assert it is
    above ``floor``. This is the same marker the standalone harness emits."""
    import re

    m = re.search(r"\[kv-pcc\] min PCC across \d+ layers: K=([0-9.]+) V=([0-9.]+)", stdout)
    assert m, f"no [kv-pcc] summary line in runner stdout:\n{stdout[-2000:]}"
    k, v = float(m.group(1)), float(m.group(2))
    assert k >= floor and v >= floor, f"PCC below floor ({floor}): K={k} V={v}"


_RUNNER = [sys.executable, "-m", "models.demos.common.prefill.runners.prefill_runner"]
_PRODUCER = [sys.executable, "-m", "models.demos.common.prefill.runners.prefill_producer"]

_COMMON_ENV = {
    "PREFILL_MODEL": "gpt_oss_d_p",
    "PREFILL_SP": str(ROWS),
    "PREFILL_TP": str(COLS),
    "PREFILL_NUM_LAYERS": "36",
    "PREFILL_CHUNK_SIZE": "5120",
}


@_GALAXY_GATE
@pytest.mark.requires_mesh_topology(mesh_shape=(ROWS, COLS), topology=f"mesh-{ROWS}x{COLS}")
def test_T1_standalone_pcc_via_engine():
    """T1 — Standalone KV PCC through the common engine matches the standalone harness (~≥0.99).
    Regression here → gap 3 (weights)."""
    _required_env(needs_trace=True)
    r = _run(_RUNNER, {**_COMMON_ENV, "PREFILL_STANDALONE": "1", "PREFILL_STANDALONE_PCC": "1"}, timeout_s=1800)
    assert r.returncode == 0, f"prefill_runner exited {r.returncode}\nstderr:\n{r.stderr[-2000:]}"
    _assert_pcc_ok(r.stdout)


@_GALAXY_GATE
@pytest.mark.requires_mesh_topology(mesh_shape=(ROWS, COLS), topology=f"mesh-{ROWS}x{COLS}")
def test_T2_multi_chunk_pcc_via_engine():
    """T2 — Multi-chunk (cached_len>0, SP ring cache-read) under engine control. Same PCC bar."""
    _required_env(needs_trace=True)
    r = _run(
        _RUNNER,
        {**_COMMON_ENV, "PREFILL_STANDALONE": "1", "PREFILL_STANDALONE_PCC": "1", "PREFILL_STANDALONE_NCHUNKS": "11"},
        timeout_s=2400,
    )
    assert r.returncode == 0, f"prefill_runner exited {r.returncode}\nstderr:\n{r.stderr[-2000:]}"
    _assert_pcc_ok(r.stdout)


@_GALAXY_GATE
@pytest.mark.requires_mesh_topology(mesh_shape=(ROWS, COLS), topology=f"mesh-{ROWS}x{COLS}")
def test_T3_request_mode_with_producer():
    """T3 — Request mode + producer over the H2D socket. 11 chunks, no shape/dtype error inside
    prefill_chunk. Failure on chunk 0 → gap 2 (input path)."""
    _required_env(needs_trace=False)
    service_id = "gpt_oss_gap_t3"
    env = {**_COMMON_ENV, "PREFILL_H2D_SERVICE_ID": service_id}

    runner = subprocess.Popen(_RUNNER, env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        prod = _run(_PRODUCER, {**env, "PREFILL_STANDALONE_NCHUNKS": "11"}, timeout_s=1800)
        assert prod.returncode == 0, f"prefill_producer exited {prod.returncode}\n{prod.stdout[-2000:]}"
        try:
            out, _ = runner.communicate(timeout=1800)
        except subprocess.TimeoutExpired:
            runner.kill()
            out, _ = runner.communicate()
            pytest.fail(f"runner did not exit after producer finished:\n{out[-2000:]}")
        assert runner.returncode == 0, f"prefill_runner exited {runner.returncode}\n{out[-2000:]}"
        assert "CHUNK_START c=10" in out, "runner did not process all 11 chunks (gap 2 suspected)"
    finally:
        if runner.poll() is None:
            runner.kill()


@_GALAXY_GATE
@pytest.mark.requires_mesh_topology(mesh_shape=(ROWS, COLS), topology=f"mesh-{ROWS}x{COLS}")
def test_T4_layer_ack_completions():
    """T4 — LayerAck emission verified by scheduler_standins.CompletionCheckConsumer.
    Fails today: runtime has no set_layer_ack_channel (gap 1) so PREFILL_ENABLE_LAYER_ACK=1 errors
    during setup at prefill_runner.py:1097."""
    _required_env(needs_trace=False)
    service_id = "gpt_oss_gap_t4"
    env = {
        **_COMMON_ENV,
        "PREFILL_H2D_SERVICE_ID": service_id,
        "PREFILL_ENABLE_LAYER_ACK": "1",
        "PREFILL_CHECK_COMPLETIONS": "1",
    }
    runner = subprocess.Popen(_RUNNER, env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        prod = _run(_PRODUCER, {**env, "PREFILL_STANDALONE_NCHUNKS": "11"}, timeout_s=1800)
        assert prod.returncode == 0, f"prefill_producer exited {prod.returncode}\n{prod.stdout[-2000:]}"
        try:
            out, _ = runner.communicate(timeout=1800)
        except subprocess.TimeoutExpired:
            runner.kill()
            out, _ = runner.communicate()
            pytest.fail(f"runner did not exit:\n{out[-2000:]}")
        assert runner.returncode == 0, f"runner exited {runner.returncode}\n{out[-2000:]}"
        # CompletionCheckConsumer prints its verdict on shutdown; look for either the "OK" marker or
        # the absence of the "missing completions" failure. Adjust if the stand-in's log wording
        # differs on your build.
        assert "missing" not in out.lower() or "PASS" in out, f"LayerAck completion check flagged an issue:\n{out[-2000:]}"
    finally:
        if runner.poll() is None:
            runner.kill()


@_GALAXY_GATE
@pytest.mark.requires_mesh_topology(mesh_shape=(ROWS, COLS), topology=f"mesh-{ROWS}x{COLS}")
def test_T5_multi_user():
    """T5 — Producer alternates slot_ids across chunks; per-user cache-slot indexing works under real
    scheduling (harness only exercises slot_id=0). Runs on top of the T4 setup."""
    _required_env(needs_trace=False)
    service_id = "gpt_oss_gap_t5"
    env = {
        **_COMMON_ENV,
        "PREFILL_H2D_SERVICE_ID": service_id,
        "PREFILL_NUM_USERS": "2",
        "PREFILL_ENABLE_LAYER_ACK": "1",
        "PREFILL_CHECK_COMPLETIONS": "1",
    }
    runner = subprocess.Popen(_RUNNER, env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        # Producer must alternate slot_ids. If your producer needs a specific flag to do that, add it
        # here; the default trace producer today only fills slot 0, so this test may need a small
        # producer patch (or a scheduler stand-in) to exercise slot 1 as well.
        prod = _run(
            _PRODUCER,
            {**env, "PREFILL_STANDALONE_NCHUNKS": "4", "PREFILL_PRODUCER_MULTI_USER": "1"},
            timeout_s=1800,
        )
        assert prod.returncode == 0, f"prefill_producer exited {prod.returncode}\n{prod.stdout[-2000:]}"
        try:
            out, _ = runner.communicate(timeout=1800)
        except subprocess.TimeoutExpired:
            runner.kill()
            out, _ = runner.communicate()
            pytest.fail(f"runner did not exit:\n{out[-2000:]}")
        assert runner.returncode == 0, f"runner exited {runner.returncode}\n{out[-2000:]}"
        # Both slots must appear in the CHUNK_START logs.
        assert "slot=0" in out and "slot=1" in out, f"only one slot exercised; producer did not alternate:\n{out[-2000:]}"
    finally:
        if runner.poll() is None:
            runner.kill()
