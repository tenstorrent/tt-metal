# Qwen2-VL-7B-Instruct — performance & accuracy

Measured on a single **Blackhole p150** (`device_id=0`), best-of-3 after warm-up.
Precision: vision tower float32; text tower bf16 weights with HiFi4 / fp32
accumulation; KV cache float32. Reproduce with the bench command in each section.

## Accuracy

| Metric | Value | Gate |
|--------|-------|------|
| e2e next-token logits PCC (vs HF golden, `N=16`) | **0.970** | ≥ 0.95 ✅ |
| greedy token match (vs HF `generate`) | **16/16** | exact ✅ |
| vision `image_embeds` PCC | 0.992 | — |
| Tier-2 KV-cache decode PCC (fp32 cached attention) | **0.9987** | ≥ 0.95 ✅ |

## Decode performance

`image-text-to-text`, prompt length `S=44` (image + text), greedy decode of `N`
tokens at fixed KV capacity `C`. Three paths timed:

- **full-seq eager** — no cache; recomputes the whole 28-layer tower over the
  entire context every token (`generate`).
- **KV-cache eager** — prefill once, then `seq=1` steps attending a
  fixed-capacity K/V cache, O(1) compute per step (`generate_kv`).
- **KV traced+2CQ** — the same `seq=1` step captured once into a replayable
  trace bound to persistent buffers, replayed with `ttnn.execute_trace` on a
  2-command-queue device.

### C = 64, N = 16 (validated demo point)

| path | latency | per-token | throughput | vs full-seq |
|------|--------:|----------:|-----------:|:-----------:|
| full-seq eager | 1476.9 ms | 92.3 ms/tok | 10.83 tok/s | 1.00x |
| KV-cache eager | 1399.1 ms | 87.4 ms/tok | 11.44 tok/s | 1.06x |
| KV traced+2CQ  | 1353.7 ms | 84.6 ms/tok | 11.82 tok/s | 1.09x |

```bash
QV_CAP=64 QV_NTOK=16 ./python_env/bin/python -m \
    models.demos.qwen2_vl.qwen2_vl_7b_instruct.tests.e2e._bench_kv_trace2cq
```

### C = 512, N = 32 (longer context)

| path | latency | per-token | throughput | vs full-seq |
|------|--------:|----------:|-----------:|:-----------:|
| full-seq eager | 6507.1 ms | 203.3 ms/tok | 4.92 tok/s | 1.00x |
| KV-cache eager | 2986.2 ms | 93.3 ms/tok | 10.72 tok/s | **2.18x** |
| KV traced+2CQ  | 3239.5 ms | 101.2 ms/tok | 9.88 tok/s | 2.01x |

```bash
QV_CAP=512 QV_NTOK=32 ./python_env/bin/python -m \
    models.demos.qwen2_vl.qwen2_vl_7b_instruct.tests.e2e._bench_kv_trace2cq
```

One-time trace capture ≈ 63 ms (amortized). In all runs the traced token stream
equals the KV-eager stream (correctness gate PASS).

## Takeaways

- **KV-cache decode scales with context.** Per-token latency stays ~flat
  (87 → 93 ms/tok as context grows 64 → 512), while full-seq recompute balloons
  (92 → 203 ms/tok). The KV speedup therefore grows with context: **1.06x @ C=64
  → 2.18x @ C=512**, and keeps widening with longer prompts.
- **trace+2CQ does not yet pay off for this decode.** At long context it is
  slightly *slower* than KV-cache eager (0.92x): the `seq=1` step is
  **compute-bound at HiFi4 (fp32)**, so replacing host dispatch with a trace
  saves little, and the per-step buffer staging adds overhead. Unlocking a
  trace+2CQ win needs a **precision-drop lever** (e.g. bf16/bfp8 KV attention)
  to make the step dispatch-bound — the intended next optimization step.
- **Best decode path today: KV-cache eager** (`generate_kv`).
