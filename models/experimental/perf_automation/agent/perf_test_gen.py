# SPDX-License-Identifier: Apache-2.0
"""Generate a bounded, profiler-safe perf test for a pipeline FROM its demo, when none exists.

emit-e2e emits demos (demo/demo_<task>.py) but no perf test; some tt-metal demos lack one too.
Discovery calls generate_perf_test() for any pipeline whose perf_test resolved to None: an LLM lifts
the build+run from the demo and wraps it in a fixed profiler-safe skeleton (bounded work + periodic
ttnn.ReadDeviceProfiler drain, NO PCC asserts). The perf test is an OUTPUT we manufacture from the
demo (the reliable input), not something we require to pre-exist. Idempotent.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

# Structural reference handed to the LLM (the seamless bounded-perf pattern, generic-ized).
_SKELETON_REF = """
import os
import time
import pytest
import ttnn
# from <model>.tt.<generator> import <Generator>   # lift the import from the demo

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
if _PERF_TRACE:
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "23887872"))
    _DEV_PARAMS["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "1"))

@pytest.mark.parametrize("device_params", [_DEV_PARAMS], indirect=True)
def test_<task>_perf(device_params, device):
    # 1) build the pipeline EXACTLY as demo/demo_<task>.py does
    # 2) drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY ttnn
    #    operation (type 'FastOperation') across ttnn + its op submodules, so the flush counter
    #    tracks TOTAL device dispatch for ANY op mix. A curated op list under-counts (sdpa/eltwise/
    #    transpose/reduction slip through) and the 12000-marker buffer overflows on some device,
    #    dropping ops -> non-reproducible device_ms. Wrapping by TYPE never misses an op.
    counter = [0]
    _orig = []
    def _draining(fn):
        def inner(*a, **k):
            r = fn(*a, **k); counter[0] += 1
            if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                try: ttnn.ReadDeviceProfiler(device)   # 'device' = mesh_device on multi-chip
                except Exception: pass
            return r
        return inner
    _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
    for _mod in [_m for _m in _mods if _m is not None]:
        for _n in dir(_mod):
            _op = getattr(_mod, _n, None)
            if type(_op).__name__ == "FastOperation":     # every dispatched ttnn op, by type
                _orig.append((_mod, _n, _op)); setattr(_mod, _n, _draining(_op))
    _fw0 = time.monotonic()
    try:
        out = ...  # run the pipeline BOUNDED (cap decode via PERF_MAX_NEW_TOKENS, or one forward)
        try: ttnn.ReadDeviceProfiler(device)
        except Exception: pass
    finally:
        for _mod, _n, _f in _orig: setattr(_mod, _n, _f)
    print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
    assert out is not None   # perf only — NO PCC

    if _PERF_TRACE:
        try:
            from models.experimental.perf_automation.agent.trace_replay import measure_adapter
            from models.experimental.perf_automation.agent.perf_adapter import PipelineDecodeAdapter

            def _build_for_perf(dev):
                ...
            _prompt_ids = ...
            _adapter = PipelineDecodeAdapter(_build_for_perf, _prompt_ids, batch=1)
            measure_adapter(_adapter, device, mode="auto")
        except Exception as _te:  # noqa: BLE001
            print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
"""


def _inline_inprocess_sources(src_text: str, root: Path) -> str:
    """When a source orchestrates the forward by launching pytest node-ids in SUBPROCESSES (a
    union gate), tracy cannot see those device ops — profiling yields an empty CSV. Pull the REAL
    in-process forwards: find the referenced `<path>.py::test_*` node-ids, resolve each file
    (under the model root), and return their bodies so the LLM can lift the build+forward directly.
    Model-agnostic: any model whose gate/demo shells out to per-module nodes gets them inlined."""
    if not any(tok in src_text for tok in ("subprocess", "Popen", "os.system", "os.popen")):
        return ""
    blocks, seen = [], set()
    for m in re.finditer(r"([\w./\-]+\.py)::\w+", src_text):
        rel = m.group(1)
        if rel in seen:
            continue
        seen.add(rel)
        cand = None
        candidates = [root / rel, root / Path(rel).name]
        marker = f"{root.name}/"
        if marker in rel:
            candidates.append(root / rel.split(marker, 1)[1])
        for c in candidates:
            if c.is_file():
                cand = c
                break
        if cand is None:
            hits = list(root.rglob(Path(rel).name))
            cand = hits[0] if hits else None
        if cand and cand.is_file():
            blocks.append(f"<inprocess_source path='{rel}'>\n{cand.read_text(errors='ignore')}\n</inprocess_source>")
    return "\n\n".join(blocks)


def _strip_fence(text: str) -> str:
    t = (text or "").strip()
    # The model sometimes wraps the file in PROSE + a ```python fence (e.g. "here is the file: ```python
    # ..."). Extract the first fenced code block when present, so the prose preamble never reaches disk.
    m = re.search(r"```(?:python|py)?[^\n]*\n(.*?)```", t, re.DOTALL)
    if m:
        return m.group(1).strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t


def _claude(prompt: str, timeout_s: int = 600) -> str | None:
    """One headless `claude` CLI call returning the generated file text (None on failure). Uses the
    CLI's native auth (real key or login), not the perf_automation LiteLLM proxy."""
    env = dict(os.environ)
    # native auth: drop proxy vars; restore the native key stashed by config (else fall back to login)
    for _k in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
        env.pop(_k, None)
    _native = env.pop("PERF_NATIVE_ANTHROPIC_API_KEY", "")
    if _native:
        env["ANTHROPIC_API_KEY"] = _native
    else:
        env.pop("ANTHROPIC_API_KEY", None)
    try:
        r = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except Exception:  # noqa: BLE001
        return None
    return r.stdout if r.returncode == 0 else None


def generate_perf_test(
    model_root: str | Path,
    task: str,
    demo_rel: str | None,
    *,
    runner=None,
    force: bool = False,
    source_abs: str | Path | None = None,
    source_kind: str = "demo",
) -> str | None:
    """Write tests/e2e/test_<task>_perf.py by lifting build+run from a source — the WHOLE pipeline
    forward (prefill + a capped decode loop when the source has one). Returns the node id
    'tests/e2e/test_<task>_perf.py::test_<task>_perf' on success, else None. `runner` (prompt->str)
    overrides the default claude call (for tests).

    Source: source_kind='demo' (default) lifts from `demo_rel` (under model_root); source_kind='pcc'
    lifts from `source_abs` (the e2e PCC test, which may live outside model_root) and DROPS the
    reference build + correctness asserts, keeping only the TTNN forward.

    force=False keeps the old idempotent behavior (return an existing file unchanged). force=True
    REGENERATES from scratch every time and overwrites — used by discovery so a stale/partial
    (e.g. prefill-only) perf test is NEVER reused; the pipeline's perf workload is recomputed each run."""
    root = Path(model_root)
    out_rel = f"tests/e2e/test_{task}_perf.py"
    out_path = root / out_rel
    node = f"{out_rel}::test_{task}_perf"
    if out_path.exists() and not force:
        return node
    if source_kind == "pcc":
        src_file = Path(source_abs) if source_abs else None
        if src_file is None or not src_file.is_file():
            return None
        src_label = str(src_file)
    else:
        src_file = root / demo_rel if demo_rel else None
        if src_file is None or not src_file.is_file():
            return None
        src_label = demo_rel
    demo_src = src_file.read_text(errors="ignore")
    if source_kind == "pcc":
        prompt = (
            f"Write a pytest PERFORMANCE test file `{out_rel}` for the '{task}' pipeline of this TTNN model.\n"
            f"This source is a CORRECTNESS (PCC) test — build and run the TTNN model EXACTLY as it does, but "
            f"KEEP ONLY the on-device TTNN forward: DROP the reference/torch model construction and DROP every "
            f"PCC / comp_pcc / allclose / assert_with_pcc correctness comparison.\n"
            f"<pcc_test path='{src_label}'>\n{demo_src}\n</pcc_test>\n\n"
            "Requirements:\n"
        )
    else:
        prompt = (
            f"Write a pytest PERFORMANCE test file `{out_rel}` for the '{task}' pipeline of this TTNN model.\n"
            f"Build and run the pipeline EXACTLY as this demo does:\n<demo path='{src_label}'>\n{demo_src}\n</demo>\n\n"
            "Requirements:\n"
        )
    prompt += (
        "- CRITICAL — run the device forward IN-PROCESS inside the test. NEVER shell out: no "
        "subprocess, os.system, os.popen, Popen, or launching `pytest` / `python -m`. Tracy profiles "
        "ONLY the current process, so any TTNN op executed in a child process is INVISIBLE to the "
        "profiler and produces an EMPTY ops-perf CSV (the run aborts with TracyRunError). If the "
        "source orchestrates work by launching pytest node-ids in subprocesses, do NOT replicate "
        "that — inline / call those modules' build+forward directly so every device op runs here.\n"
        f"- a pytest function named `test_{task}_perf`.\n"
        "- DEVICE OPEN — MATCH THE SOURCE'S TOPOLOGY EXACTLY (this is critical for sharded models). If the "
        "source SELF-OPENS its device (calls open_pipeline_mesh / open_mesh_device / ttnn.open_mesh_device, "
        "or builds a MeshShape), your test MUST open + close the device the SAME way — lift that exact "
        "open call into the test body, close it in a finally — and pass that device object to "
        "build_pipeline / the forward. Do NOT substitute a pytest `device` / `device_params` fixture: a "
        "single `device` fixture silently DISABLES the pipeline's sharding (shard_active becomes False) and "
        "profiles the WRONG single-chip config for a model built to run tensor-parallel on a mesh. Use the "
        "pytest `device`/`device_params` fixture ONLY when the source itself uses that fixture (genuine "
        "single-device pipelines). When TT_PERF_TRACE is set and the source's open function accepts "
        "trace_region_size / num_command_queues, pass them through that open; otherwise open exactly as the "
        "source does (the trace block stays guarded and simply falls back).\n"
        "- BOUNDED + profiler-safe so tracy's 12000-marker buffer never overflows: cap the work (decode "
        "loop via env TT_PERF_MAX_NEW_TOKENS default 4, or a SINGLE forward if there's no loop), AND drain "
        "the profiler every TT_PERF_FLUSH_EVERY ops (default 32) + a final ttnn.ReadDeviceProfiler. DRAIN "
        "MUST BE MODEL-AGNOSTIC — wrap EVERY ttnn op by TYPE, not a curated list: iterate ttnn (and its op "
        "submodules ttnn.transformer / ttnn.experimental) and wrap every attribute whose "
        "type(obj).__name__ == 'FastOperation' with a counter that drains every TT_PERF_FLUSH_EVERY calls. "
        "A curated list (matmul/linear/conv only) UNDER-counts — sdpa/eltwise/transpose/reduction slip "
        "through, the buffer overflows on some device, ops get dropped, and device_ms becomes "
        "non-reproducible. Wrapping by type can never miss an op. Restore all originals in a finally. "
        "(Use the generic wrap loop from the skeleton below verbatim — do NOT hand-pick op names.)\n"
        "- CAP THE INPUT SIZE SMALL: use a SMALL fixed sequence length / token count (e.g. 128) for every "
        "forward. Do NOT reuse the model's production / maximum shapes (max_position_embeddings, max_seq, "
        "max_enc_seq, etc.) even if the source/PCC test does — those are correctness stress sizes. Under "
        "tracy EVERY device op is instrumented, so one max-seq forward runs orders of magnitude slower and "
        "the host blocks in ttnn.synchronize_device for many minutes, stalling the run. If the source "
        "defines a large seq constant, OVERRIDE it with a small value here (env-overridable, small default). "
        "A perf profile only needs a representative dispatch-dense pass, not the max shape.\n"
        '- KEEP the skeleton\'s `os.environ.setdefault("TT_PERF_LAYERS", ...)` line VERBATIM near the top. '
        "It caps profiled depth for deep (many-layer) models so the device profiler's marker buffer does "
        "not overflow (worse on a multi-chip mesh, where markers scale x chips). It is set in-process so "
        "ONLY this perf run is capped; a pipeline that does not read TT_PERF_LAYERS simply ignores it. Do "
        "NOT hard-require it and do NOT gate on it — just carry it through.\n"
        "- NO PCC / correctness assertions (this is perf only) — just assert the pipeline produced output.\n"
        "- TIME THE FORWARD: keep the skeleton's time.monotonic() bracket around the bounded forward and "
        'the final print("FORWARD_WALL_MS=...") VERBATIM — the harness reads it as an independent '
        "end-to-end check on the profiler capture. Do not remove or rename it.\n"
        "- KEEP the skeleton's trace-replay block VERBATIM in structure: the `_PERF_TRACE`/`_DEV_PARAMS` "
        "device-param gate near the top AND the trailing `if _PERF_TRACE:` measure_adapter block. This is a "
        "MODEL-AGNOSTIC, GPU-comparable per-token latency (TRACE_PER_TOKEN_MS). Do NOT write a per-model "
        "adapter class — the tool ships the generic PipelineDecodeAdapter. Your ONLY job in that block is to "
        "fill `_build_for_perf(dev)` so it builds the pipeline EXACTLY as this test/demo builds it (lift the "
        "same imports + build args, using `dev`), and set `_prompt_ids` to a SMALL prompt. Leave everything "
        "else in the block verbatim. The clean number is emitted automatically IFF the built pipeline exposes "
        "a trace-capturable `decode_step(state)`; if its decode is repeat-prefill (re-runs the growing "
        "sequence / host argmax), the adapter raises, the guard falls back to FORWARD_WALL_MS, and that is "
        "fine. Never delete the block, never let it fail the test.\n"
        "- TRACE BLOCK + SELF-OPEN: if (per the DEVICE OPEN rule) the test self-opens a mesh, pass the "
        "device the test actually opened to `measure_adapter(...)` (NOT a fixture `device`), and put "
        "`trace_region_size`/`num_command_queues` on that self-open call when TT_PERF_TRACE (drop the "
        "`_DEV_PARAMS`/`device_params` fixture entirely). Keep `_build_for_perf(dev)` building the pipeline "
        "on the passed-in `dev` so both the eager forward and the trace run the SAME sharded topology.\n"
        "- Lift the imports + build args straight from the demo above.\n\n"
        f"Use this structural skeleton (adapt the build+run to the demo):\n{_SKELETON_REF}\n"
    )
    inproc_ctx = _inline_inprocess_sources(demo_src, root)
    if inproc_ctx:
        prompt += (
            "\n\nNOTE: the source above is a SUBPROCESS-UNION — it launches per-module pytest node-ids "
            "in child processes, whose device ops tracy CANNOT profile. Below are those modules' ACTUAL "
            f"in-process build+forward bodies. Lift the build and on-device TTNN forward from THESE and run "
            f"them directly in test_{task}_perf (one process, same modules covered), dropping every "
            "PCC/correctness assert. CRITICAL: use the SAME pytest device fixtures and the SAME "
            "@pytest.mark.parametrize decorator these modules use (e.g. `mesh_device`, `device_params`, "
            "`reset_seeds` with their MESH_DEVICE_PARAMETRIZE_* marker) — do NOT substitute a plain single "
            "`device` fixture; the lifted builds run on whatever device/mesh object these modules take. "
            "Reuse their imports/constants (config builders, MESH_DEVICE_PARAMETRIZE_*, helpers) verbatim:\n"
            f"{inproc_ctx}\n"
        )
    prompt += (
        "\n\nDo NOT use any tools and do NOT try to write the file yourself — the caller writes it. "
        "Respond with ONLY the complete python file content as your message text — no prose, no markdown fences."
    )
    # A generative demo's perf test must exercise the (capped) decode loop, not a prefill-only slice.
    demo_is_generative = any(
        k in demo_src.lower()
        for k in ("max_new_tokens", "generate(", ".generate", "next_token", "decode_step", "for _ in range")
    )
    gen = runner or _claude
    # retry so a transient bad/partial draft is recomputed rather than dropped; never reuse existing.
    for _attempt in range(3):
        content = _strip_fence(gen(prompt) or "")
        if "def test_" not in content or "ttnn" not in content:
            continue
        if re.search(
            r"import\s+subprocess|subprocess\.|\bPopen\s*\(|os\.system\s*\(|os\.popen\s*\(|"
            r"-m['\"]\s*,\s*['\"]pytest|python\s+-m\s+pytest",
            content,
        ):
            continue
        if demo_is_generative and "TT_PERF_MAX_NEW_TOKENS" not in content:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)
        return node
    return None
