# Gemma 4 Perf Log

## 2026-05-03 Full 1x8 Strict Device-Feedback Trace

Status: accepted strict decode milestone. Token feedback, token embedding, sampled-token handoff, RoPE position, and KV-cache position stay on device across 128 TTNN trace replays. The default text demo remains a host-feedback path for collecting generated text.

Environment:

- Host: `bh-lb-01-special-moconnor-for-reservation-67920`
- Hardware: 8x Blackhole `p150b`, 16G GDDR per board
- KMD: `2.7.0`
- Firmware bundle: `19.7.0.0`
- Python: `/proj_sw/user_dev/moconnor/tt-metal/python_env/bin/python`
- Transformers: `4.57.1`
- HF checkpoint: `google/gemma-4-26B-A4B@64143b04706fadeb2f8ac198f7ecab57b94b1e0b`
- Built CWD for hardware run: `/proj_sw/user_dev/moconnor/tt-metal`
- Local bringup repo script: `/localdev/moconnor/tt-metal-gemma-4-26b-a4b/models/demos/gemma4/demo/strict_device_feedback_demo.py`
- AICLK warning observed during run: expected `1350`, last observed `800`; post-run `tt-smi` also reported `800 MHz`.

Primary command/log:

```bash
cd /proj_sw/user_dev/moconnor/tt-metal
TT_METAL_CACHE=/tmp/tt-metal-cache-gemma4-strict-full-1x8-0503 \
TT_CACHE_PATH=/proj_sw/user_dev/moconnor/hf-cache/tt_cache/google--gemma-4-26B-A4B \
HF_MODEL=google/gemma-4-26B-A4B \
/proj_sw/user_dev/moconnor/tt-metal/python_env/bin/python -u \
  /localdev/moconnor/tt-metal-gemma-4-26b-a4b/models/demos/gemma4/demo/strict_device_feedback_demo.py \
  --model-path google/gemma-4-26B-A4B \
  --max-new-tokens 128 \
  --max-seq-len 512 \
  --mesh-rows 1 \
  --mesh-cols 8 \
  --trace-region-size 50000000 \
  | tee /tmp/gemma4_strict_device_feedback_full_1x8_128_script_warm.log
```

Evidence log: `/tmp/gemma4_strict_device_feedback_full_1x8_128_script_warm.log`

Measured results:

| Metric | Value |
| --- | ---: |
| Model creation from BF16 tensor cache | `13.02 s` |
| Strict prefill / TTFT | `5368.943 ms` |
| Strict decode compile | `3.940 s` |
| Strict trace capture | `0.174 s` |
| 1st strict traced replay token | `64.014 ms` |
| 128th strict traced replay token | `64.299 ms` |
| Average strict traced replay | `64.121 ms/token` |
| Strict decode throughput | `15.595 tokens/sec/user` |
| Strict traced replay tokens | `128` |
| Full strict harness wall time | `35.78 s` |

Evidence:

- Initial device token was `496` on all 8 devices.
- Strict compile produced token `3207` and advanced RoPE/cache positions to `7`.
- Trace capture preserved positions at `7`.
- After 128 trace replays, final token was `236858` on all 8 devices.
- Final RoPE and KV-cache positions were `135` on all 8 devices, matching `prompt_len 6 + compile step 1 + 128 replay steps`.

Notes:

- The strict harness writes sampling into a padded `[32]` device token buffer, slices lane 0, and copies it into the next-token input buffer inside the trace. A prior `[1]` output-buffer prototype corrupted an adjacent sentinel because sampling/argmax writes a padded result.
- RoPE position uses a `uint32` `[1]` device tensor; KV cache/SDPA position uses a separate `int32` `[1]` device tensor.
- Decode MoE uses sparse matmul with `nnz=top_k=8`, so decode computes active experts only.
- The harness validates final token/position state, but does not reconstruct the generated text stream.

## 2026-05-03 Full 1x8 Traced Decode

Status: accepted traced-core text-output milestone. Superseded for strict decode acceptance by the device-feedback trace above.

Environment:

- Host: `bh-lb-01-special-moconnor-for-reservation-67920`
- Hardware: 8x Blackhole `p150b`, 16G GDDR per board
- KMD: `2.7.0`
- Firmware bundle: `19.7.0.0`
- Python: `/proj_sw/user_dev/moconnor/tt-metal/python_env/bin/python`
- Transformers: `4.57.1`
- HF checkpoint: `google/gemma-4-26B-A4B@64143b04706fadeb2f8ac198f7ecab57b94b1e0b`
- Built CWD for hardware run: `/proj_sw/user_dev/moconnor/tt-metal`
- Local bringup repo inserted into `sys.path`: `/localdev/moconnor/tt-metal-gemma-4-26b-a4b`

Primary command/log:

```bash
cd /proj_sw/user_dev/moconnor/tt-metal
TT_METAL_CACHE=/tmp/tt-metal-cache-gemma4-full-1x8-50153 \
TT_CACHE_PATH=/proj_sw/user_dev/moconnor/hf-cache/tt_cache/google--gemma-4-26B-A4B \
HF_MODEL=google/gemma-4-26B-A4B \
/proj_sw/user_dev/moconnor/tt-metal/python_env/bin/python - <<'PY'
import sys, ttnn
sys.path.insert(0, "/localdev/moconnor/tt-metal-gemma-4-26b-a4b")
from models.demos.gemma4.demo.text_demo import run_generation

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 8), trace_region_size=50_000_000)
try:
    run_generation(
        mesh_device=mesh_device,
        model_path="google/gemma-4-26B-A4B",
        prompts=["The capital of France is"],
        max_new_tokens=128,
        num_layers=None,
        max_seq_len=512,
        enable_decode_trace=True,
    )
finally:
    ttnn.close_mesh_device(mesh_device)
PY
```

Evidence log: `/tmp/gemma4_full_1x8_long.log`

Measured results:

| Metric | Value |
| --- | ---: |
| Model creation from BF16 tensor cache | `12.5 s` |
| Prefill / TTFT | `4853.59 ms` |
| Decode trace capture | yes |
| 1st traced decode token | `64.01 ms`, `15.62 t/s/u` |
| 128th traced decode token | `64.78 ms`, `15.44 t/s/u` |
| Average traced decode | `64.8 ms/token` |
| Average decode throughput | `15.43 tokens/sec/user` |
| Generated tokens | `128` |
| Full demo runtime | `33.49 s` |
| Wrapper wall time | `35.49 s` |

Notes:

- Decode path used TTNN trace replay with host token feedback plus embedding/position preparation copied into trace-owned device buffers each token.
- On-device sampling was active for TP=8, but the sampled token was read back to host to seed the next decode step.
- Decode MoE uses sparse matmul with `nnz=top_k=8`, so decode computes active experts only.
- Prefill still computes all experts.
- No BFP4/packed expert weights were used.

## 2026-05-03 Sample Prompt Pass

Command/log: `/tmp/gemma4_full_1x8_samples.log`

Settings: same full 1x8 traced path, `max_new_tokens=48`, three prompts.

Summary metrics:

- TTFT shown by benchmark: `4988.9 ms`
- Average decode: `64.81 ms/token`
- Throughput: `15.43 tokens/sec/user`
- Generated tokens per prompt setting: `48`
- Full demo runtime: `39.71 s`

The sample outputs are recorded in top-level `RESULTS.md`.

## 2026-05-03 Cold Full 1x8 Short Smoke

Status: superseded by warm long run, retained as cache-warmup context.

Settings: full 30 layers, 1x8, real checkpoint, `max_new_tokens=4`, trace enabled.

Measured results:

- Model creation while generating tensor cache: `332.5 s`
- Prefill / TTFT: `222705.38 ms`
- Decode compile time: `215.05 s`
- Average decode: `100.64 ms/token`
- Decode throughput: `9.94 tokens/sec/user`
- Output: `The capital of France is a city of romance,`

## 2026-05-03 One-Layer 1x1 Trace Smoke

Status: passed for plumbing only; output quality not meaningful with one layer.

Settings: one layer, 1x1 mesh, real checkpoint slice, `max_new_tokens=4`, trace enabled.

Measured results:

- Prefill / TTFT: `173732.85 ms`
- Decode compile time: `151.98 s`
- Average decode: `18.01 ms/token`
- Decode throughput: `55.52 tokens/sec/user`
- Output contained an unused token because only one layer was active.
