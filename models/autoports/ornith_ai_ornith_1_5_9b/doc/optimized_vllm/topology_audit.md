# Serving operation audit before tuning

Selected policy: datatype_sweep/configs/bfp8_all_lofi.json. Prior device topology and geometry evidence is retained in optimized_full_model/checklist.md and optimized_multichip_decoder/topology_decisions.md. This stage measures real serving; no new device profiles.

| Boundary | Current sequence | Candidate / disposition |
|---|---|---|
| Decoder | Packed attention, separate decode MLP gate/up, selected BFP8/LoFi weights, BF16 width-sharded residual, BFP8 CCL/KV | Preserve selected arithmetic; geometry trials must use LoFi, not historical HiFi2 head evidence |
| Terminal | Distributed norm, DRAM vocabulary-sharded head, common split greedy sampler, persistent token feedback | Already avoids full-vocabulary gather/readback and host argmax |
| Steady decode | Nonblocking model + sampler replay, first-chip token read, event, host formatting | Move read enqueue from forward into deferred-read callback; prove ownership across two outstanding results |
| Admission | Helper history reset, then canonical prompt/output-aware history reset | Remove duplicate helper reset; preserve terminal canonical reset and unrelated slots |
| Page admission | CPU torch prefix scan each step over unchanged host table | Cache first unmapped page on scheduler table refresh; scalar steady bounds check |
| Prefill | Serial rows, persistent prepared buckets, discarded intermediate logits | Sweep scheduler budget; optional dense batching requires exact logits/KV equality before adoption |
| Masked sampler | Preserve inactive seed words around common sampling | Optional all-active epilogue candidate, not yet measured |

Independent CPU-only source audit returned these candidates. It found no model math regression or output-handle retention leak. Historical serving8.700078ms versus selected datatype resident8.610224ms and token-readback8.638849ms suggests a small decode gap, but these used different context/sampler widths; final comparison needs a current full-model run. No runtime attribution is inferred from source inspection.
