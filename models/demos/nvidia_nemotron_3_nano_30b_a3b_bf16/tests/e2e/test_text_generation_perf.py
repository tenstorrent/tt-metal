import os
import time

import torch

import ttnn
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt import pipeline as pl
from models.demos.nvidia_nemotron_3_nano_30b_a3b_bf16.tt._hf_compat import install_hf_compat

install_hf_compat()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

PERF_MAX_NEW_TOKENS = int(os.environ.get("TT_PERF_MAX_NEW_TOKENS", "4"))
PERF_FLUSH_EVERY = int(os.environ.get("TT_PERF_FLUSH_EVERY", "32"))
# small fixed prompt length for the perf pass; NOT the model's production/max shape
PERF_SEQ_LEN = int(os.environ.get("TT_PERF_SEQ_LEN", "128"))
# perf-only depth cap: profile a few blocks so a deep model's marker stream (x mesh chips) does not
# overflow / bloat the profiler; pipelines that read TT_PERF_LAYERS honor it, others ignore it. This
# is set in-process here so ONLY the perf run is capped (the correctness/e2e gate runs the full model).
os.environ.setdefault("TT_PERF_LAYERS", "2")

# Trace-replay per-token latency (GPU-comparable T/S/U). MODEL-AGNOSTIC + OFF-BY-DEFAULT-SAFE:
# TT_PERF_TRACE=1 (default) adds trace_region_size + num_command_queues to the device open so the
# per-token block below CAN capture a device trace; TT_PERF_TRACE=0 restores the plain eager open
# (exactly the old behavior -> guaranteed non-breaking escape hatch for tight-memory models).
_PERF_TRACE = os.environ.get("TT_PERF_TRACE", "1") == "1"
_DEV_PARAMS = {"l1_small_size": 24576}
if _PERF_TRACE:
    _DEV_PARAMS["trace_region_size"] = int(os.environ.get("TT_PERF_TRACE_REGION", "120000000"))
    _DEV_PARAMS["num_command_queues"] = int(os.environ.get("TT_PERF_NUM_CQ", "2"))  # 2 = trace+2CQ overlap path


def test_text_generation_perf():
    # ---- open the mesh EXACTLY as demo/demo_text_generation.py does (self-open, NOT a fixture) ----
    # The demo calls pl.open_pipeline_mesh(l1_small_size=24576); a single pytest `device` fixture would
    # silently disable the pipeline's tensor-parallel sharding (shard_active -> False) and profile the
    # wrong single-chip config. When TT_PERF_TRACE is set, thread trace_region_size / num_command_queues
    # through that same open if it accepts them; otherwise fall back to the plain open the demo uses.
    try:
        device, is_mesh = pl.open_pipeline_mesh(**_DEV_PARAMS)
    except TypeError:
        device, is_mesh = pl.open_pipeline_mesh(l1_small_size=24576)
    try:
        # ---- build the pipeline EXACTLY as the demo does ----
        tok = AutoTokenizer.from_pretrained(pl.HF_MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            pl.HF_MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        model.eval()
        eos = int(getattr(model.config, "eos_token_id", 2))

        input_ids = tok("The capital of France is", return_tensors="pt")["input_ids"]
        # CAP the input size small: never profile a max-shape forward under tracy.
        if input_ids.shape[-1] > PERF_SEQ_LEN:
            input_ids = input_ids[:, :PERF_SEQ_LEN]

        pipe = pl.build_pipeline(device, model, compose=True)
        print(f"[perf] mesh={is_mesh} shard_active={pipe.shard_active}", flush=True)

        # ---- drain the device profiler every PERF_FLUSH_EVERY ops. MODEL-AGNOSTIC: wrap EVERY ttnn
        # operation (type 'FastOperation') across ttnn + its op submodules, so the flush counter tracks
        # TOTAL device dispatch for ANY op mix. A curated op list under-counts (sdpa/eltwise/transpose/
        # reduction slip through) and the 12000-marker buffer overflows on some device, dropping ops ->
        # non-reproducible device_ms. Wrapping by TYPE never misses an op. ----
        counter = [0]
        _orig = []

        def _draining(fn):
            def inner(*a, **k):
                r = fn(*a, **k)
                counter[0] += 1
                if PERF_FLUSH_EVERY and counter[0] % PERF_FLUSH_EVERY == 0:
                    try:
                        ttnn.ReadDeviceProfiler(device)  # 'device' = mesh_device on multi-chip
                    except Exception:
                        pass
                return r

            return inner

        _mods = [ttnn] + [getattr(ttnn, _m, None) for _m in ("transformer", "experimental")]
        for _mod in [_m for _m in _mods if _m is not None]:
            for _n in dir(_mod):
                _op = getattr(_mod, _n, None)
                if type(_op).__name__ == "FastOperation":  # every dispatched ttnn op, by type
                    _orig.append((_mod, _n, _op))
                    setattr(_mod, _n, _draining(_op))

        _fw0 = time.monotonic()
        try:
            out, _ = pipe.generate(input_ids, PERF_MAX_NEW_TOKENS, eos_token_id=eos)
            try:
                ttnn.ReadDeviceProfiler(device)
            except Exception:
                pass
        finally:
            for _mod, _n, _f in _orig:
                setattr(_mod, _n, _f)
        print("FORWARD_WALL_MS=%.4f" % ((time.monotonic() - _fw0) * 1000.0))
        assert out is not None  # perf only — NO PCC

        # ---- clean, GPU-comparable per-token latency via trace-replay (GENERIC + guarded) ----
        # ONE generic adapter (agent/perf_adapter.PipelineDecodeAdapter) wraps the SAME pipeline build:
        # measure_adapter captures one decode step as a device trace + replays it -> prints
        # TRACE_PER_TOKEN_MS (parsed by the tool into per_token_ms + tokens_per_sec_per_user for a GPU
        # side-by-side). There is NO per-model adapter here. The clean number appears only when the built
        # pipeline exposes a trace-capturable `decode_step(state)` (fixed shape, on-device sample, no host
        # reads). A repeat-prefill pipeline has no decode_step, so setup raises, the guard swallows it,
        # and FORWARD_WALL_MS stands.
        if _PERF_TRACE:
            try:
                from models.experimental.perf_automation.agent.perf_adapter import PipelineDecodeAdapter
                from models.experimental.perf_automation.agent.trace_replay import measure_adapter

                os.environ["TT_PERF_LAYERS"] = "0"  # trace the FULL model (Section above may have capped it)

                def _build_for_perf(dev):
                    # REUSE the pipeline built above (return that same object). Do NOT build a second copy:
                    # a 2nd full build of a large resident model OOMs the mesh, and its layer children are
                    # already resident.
                    return pipe

                _prompt_ids = input_ids  # SMALL prompt fed to decode_prefill
                _adapter = PipelineDecodeAdapter(_build_for_perf, _prompt_ids, batch=1)
                measure_adapter(_adapter, device, mode="auto")  # prints TRACE_PER_TOKEN_MS=<ms>
                # PREFILL clean-trace bookend (availability-gated): if the pipeline exposes a
                # trace-capturable prefill hook, measure it too so BOTH phases have a before/after
                # (prompt pre-uploaded outside the trace; 2CQ overlaps the prompt H2D when available).
                if hasattr(pipe, "prefill_trace_step") and os.environ.get("TT_PERF_PREFILL_TRACE") == "1":
                    # Prefill trace is a FULL-sequence deep capture — it floods tracy's marker stream if
                    # run inside the profiled baseline (decode's single-step trace is fine). Gate it OFF
                    # by default; set TT_PERF_PREFILL_TRACE=1 to measure prefill standalone. Decode
                    # trace+2CQ always runs above.
                    from models.experimental.perf_automation.agent.trace_replay import measure_prefill

                    pipe.prefill_trace_setup(_prompt_ids)
                    measure_prefill(
                        device,
                        pipe.prefill_trace_step,
                        write_inputs=getattr(pipe, "prefill_write_inputs", None),
                        mode="auto",
                    )
            except Exception as _te:  # noqa: BLE001
                print("TRACE_REPLAY_SKIPPED=%r" % (_te,), flush=True)
    finally:
        pl.close_pipeline_mesh(device, is_mesh)
