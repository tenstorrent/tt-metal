# Qwen/Qwen3.8-27B TTNN autoport

The Stage8 precision sweep selects BFP4/LoFi decoder and LM-head projections,
FP32 destination accumulation, BF16 activations/residuals/CCL, and BFP8 KV cache.
The [selected precision artifact](doc/datatype_sweep/selected_precision_config.json)
is consumed by default by `QwenModel` and `build_generator`.
`QWEN_PRECISION_CONFIG=baseline` restores the Stage7 policy.

Measured on four Blackhole p300c devices, TP4 Ring, warmed batch1:

| Measurement regime | TTFT | Decode tokens/s/user |
| --- | ---: | ---: |
| Post-selection token-out, no readback, S128/G128 | **58.418 ms** | **41.095** |
| Deferred complete token delivery, S128/G128 | 58.986 ms | 41.075 |
| Traced teacher-forcing selection, AIME24 S203/G100 | 76.607 ms | 40.878 |

Use the **post-selection token-out** row for later token-out/serving comparisons.
Its timing excludes final-token readback and includes one final synchronization.
Deferred complete delivery includes the final history transfer and output-list
construction. Teacher forcing uploads reference feedback and is a separate
selection metric; the normal-default confirmation reproduces 40.849 t/s/user.

Pinned AIME24 chat-template readiness (100 generated tokens) gives prefill
**100%/100%/100%** and decode **99%/100%/100%** top-1/top-5/top-100. The public
context remains **262,144 tokens**. Full64 non-aligned prompts and batch32 fixed
slots/cache ownership pass; maximum context is separately tested at batch1.

See the [Stage8 report](doc/datatype_sweep/README.md) for all 15 measured policies,
Pareto charts, exact commands, trace/policy propagation evidence and limitations.
All numeric, context, non-aligned, batch32 and watcher checks pass. Stage8 has an
independent [clean-pass review](doc/datatype_sweep/stage_review.md). One controlled
quality limitation remains: the selected haiku scans 5/7/6 while the HF/BFP8-head
controls scan 5/7/5. The [focused investigation](doc/datatype_sweep/AUTOFIX_haiku.md)
reproduces it with native and host-greedy sampling; it is not claimed repaired.

The [Stage7 report](doc/optimized_full_model/README.md) records the completed
optimized full-model baseline and historical performance. The
[Stage5 decoder geometry](doc/optimized_multichip_decoder/README.md) and residual
layout are preserved. [Stage6](doc/full_model/README.md) records the starting full
model. No vLLM integration is included.
