# Fused Decoder Stage Review

Final verdict: `clean-pass`

Review agent: `01a01765-316a-7a43-9376-98fd8a383e3e`

The final review found no blocking work. It verified that `tt/fused_decoder.py` is a direct fused runtime path rather than a functional-decoder wrapper; correctness covers synthetic and real-weight linear/full layers, batch 1 and 2, traced decode, non-aligned lengths, and repeated decode; fallback audit and watcher runs passed; fused warmed prefill and traced decode beat the functional baseline; `tt-perf-report` artifacts exist; and the prior graph-fusing review findings were fixed.

Previously rejected findings:

| Finding | Resolution |
| --- | --- |
| `generalized_moe_gate` rejection was not earned | `graph_fusion_candidate_probe.py` now adapts Qwen router logits to the op's required L1 height-sharded layout and dense scatter rebuild. The adapted candidate reaches Blackhole JIT and is rejected because this checkout lacks `experimental/llk_sfpu/llk_math_generalized_moe_gate_topk_single_face.h` in the Blackhole LLK path. The header exists only under the Wormhole path, so the candidate is unavailable on the current hardware. |
| padded `nlp_concat_heads_decode` shape compatibility was overstated | The probe now records raw padded output `(1, 1, 32, 4096)`, the required logical slice back to `(1, 1, 1, 4096)`, and the slower repaired timing `0.0476 ms/iter` versus current reshape `0.0119 ms/iter`. |

Residual risk accepted by review: fused-specific 262k context probes were not rerun because no cache layout, sharding, dtype, or capacity-affecting contract changed; fused tests cover non-aligned logical lengths and paged decode semantics.
