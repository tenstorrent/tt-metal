# TTFT / t·s across context lengths (#47465 functional perf)

Full **30-layer** DiffusionGemma 26B-A4B-it, QB2 `P150x4` `(1,4)` mesh, **`build_Release` (Tracy OFF,
production build)**, `demo/serving_smoke.py`, `--num-blocks 2 --canvas-length 256 --max-denoising-steps 48`
(adaptive early-stop), `--max-seq-len 2048`. Raw per-run metrics in `ttft_sweep/metrics_*.json`.

Repro:
```
DG_CKPT=… python -u models/experimental/diffusion_gemma/demo/serving_smoke.py \
  --num-blocks 2 --canvas-length 256 --max-denoising-steps 48 --max-seq-len 2048 \
  --prompt "<prompt>" --metrics-json out.json
```

| context | prompt_len | denoise steps/block | **TTFT** (prefill+block0) | per-block latency | **t/s** (tok/block/s) | ~s / denoise step |
|---|---|---|---|---|---|---|
| short | 18 | [27] | **152.5 s** | 150.9 s | **1.70** | 5.6 |
| medium | 66 | [36, 18] | **197.9 s** | 109.3 s (mean) | **2.34** | 4.1 |
| long | 373 | [38] | **211.4 s** | 197.3 s | **1.30** | 5.2 |

## Findings
1. **TTFT rises with context** (152→198→211 s), but **prefill itself is small** — solving
   `TTFT = prefill + block0_steps × per_step`: prefill ≈ **1.6 s @18 tok**, ≈ **14 s @373 tok**.
   TTFT is dominated by **block-0's denoise steps**, not prefill.
2. **Per-denoise-step time is ~4–5.6 s and essentially context-independent** (short prefix=32 and
   long prefix=384 give the same per-step cost). The decode step is **MoE weight-traffic bound
   (30 layers)**, not attention/prefix bound — context barely changes decode speed per step. This
   matches the op-level analysis: making the attention/reorder faster is a 0% step win (the step is
   MoE-compute bound, see `moe_transpose_investigation.md`).
3. **t/s (1.3–2.3 tok/s) tracks the adaptive denoise-step count, not context.** Medium's 2nd block
   stopped at 18 steps → lower mean block latency → highest t/s. So t/s is content/step-count
   dependent, not monotonic in context.

**Bottom line:** TTFT ≈ 2.5–3.5 min, throughput ≈ 1.3–2.3 tok/s; the lever is the per-step MoE cost
(~4–5 s/step, ~85–110× the ~49 ms/step weight-traffic roofline), not context/attention handling.
(Short-prompt run emits degenerate text — `text_chars=4` — the expected #48291 argmax-fidelity effect.)
