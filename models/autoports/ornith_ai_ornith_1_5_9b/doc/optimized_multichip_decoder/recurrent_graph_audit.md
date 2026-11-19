# Batched recurrent graph audit before the next rewrite

All rows refer to TP4 local heads, B=1..32, real Ornith weights. This extends
the initial operation-topology audit; no new primitive kernel is proposed.

| Subgraph | Present sequence / movement | Candidate | Classification / check |
|---|---|---|---|
| Dense input | replicated L1 norm -> packed DRAM-sharded matmul -> L1 interleaved | already grouped over B | measured prior stack comparisons |
| Four-tap convolution | pack histories -> FP32 multiply/add four taps -> SiLU -> BF16; scatter per request | setup-owned pooled RM history, one copy | structural movement reduction; whole state PCC |
| Q/K normalization | reshape -> FP32 -> RMSNorm -> scalar multiply, twice | chunk-GDN qk_norm=True | existing dedicated fused op, same epsilon/scale checked by model PCC |
| Fused recurrent wrapper | query scale; head split; GQA repeat; g exp; beta/g RM round-trips; three groups of <=12; state concat/scatter | direct chunk-GDN over B independent users, one live token plus internally masked tail | dedicated fused op; all tail gates neutral, state exact ownership |
| Recurrent state | private states concatenate; subgroup slices; final concatenate; per-request copy | persistent pooled FP32 state, explicit per-row prefill writes; chunk-GDN initial/final alias | structural reduction; compare every user state and varied batch counts |
| Chunk-GDN geometry | one-token B1 uses existing num_receivers=1 | batched native B axis, wave-head geometry32/48 plus default, receivers1 | match real shape and numerical gate; API errors require adaptation |
| Output | BF16 -> per-value-head RMSNorm -> flattened local hidden -> fused SiLU(z)*out -> WO -> async AR | retained pending profiler evidence | no inter-layer CCL introduced |

Relevant source: fused_recurrent_gated_delta_rule.cpp documents wrapper ops;
chunk_gated_delta_rule_nanobind.cpp exposes qk_norm, native B, head waves,
preallocated o_out/final_state_out. The existing production B1 uses the same
chunk-GDN direct op and neutral g/beta padding. Every new candidate remains
unselected until outputs and recurrent/conv state pass >=0.995 and latency improves.
