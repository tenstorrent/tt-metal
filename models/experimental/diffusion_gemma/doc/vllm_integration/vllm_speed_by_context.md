# vLLM serving speed vs context (model-faithful, #47488/#47465)

Live vLLM OpenAI server, DiffusionGemma 26B-A4B-it, QB2 (1,4) mesh, #47488 fork patches applied,
`DG_SPARSE_MOE=1 DG_DEDUP_ARGMAX=1 DG_SPARSE_MOE_TUNED=1`, `--generation-config vllm`, on-device
sampling. **DG_DENOISE_TRACED / DG_COMMIT_BATCHED are OFF on the vLLM paged path** (expected) — so
vLLM runs the eager-sparse generator path, WITHOUT the traced-loop/batched-commit wins the direct
path has. All blocks ran the FULL 48 denoise steps (`halted=False` — HF early-halt is a no-op under
#48291), so these are model-faithful.

| context | prompt_len | TTFT (prefill+block0) | steady block-1 latency | **t/s** (256/block) | steps |
|---|---:|---:|---:|---:|---:|
| short  | 10  | 34.60 s | 35.13 s | **7.29** | 48 |
| medium | 61  | 37.19 s | 36.23 s | **7.07** | 48 |
| long   | 265 | 35.73 s | 34.46 s | **7.43** | 48 |

**Findings**
- vLLM serving throughput ≈ **7.1–7.4 tok/s, flat across context** (block latency 34–37 s independent
  of prompt_len 10→265) ⇒ per-block cost is MoE-compute bound (48 steps × 30L), not attention/prefix.
- TTFT ≈ 35–37 s (prefill is a small part of the block; also context-flat).
- **vs the direct traced path: 17.92 t/s @48** — vLLM is ~2.4× slower because the traced serving loop
  (2.72×) + batched commit are not wired into the vLLM paged path (future: #47488 paged-cache
  ownership + #47557). vLLM ≈ the eager-sparse @48 rate (~6.6 t/s) + on-device sampling.
- ~6.8× over the original dense-128 baseline (~1.08 t/s).
