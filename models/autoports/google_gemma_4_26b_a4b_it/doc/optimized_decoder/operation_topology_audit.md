# Operation-topology audit

This audit preceded candidate integration. Detailed commands are in
`work_log.md`, machine-readable results in `candidate_matrix.json`, and final
device rows in `tracy/current_fused_final/`.

| Path | Starting operations | Candidate | Final action | Evidence |
| --- | --- | --- | --- | --- |
| Cache | separate paged K/V updates | fused composite update | fuse native/non-modulo batch 1; retain measured full-batch-32 and bounded paths | one fused profiler row; cache-tail and trace tests |
| Attention | packed QKV, head split, norms, RoPE, paged SDPA, O | per-kind grids, DRAM weights, fidelity, persistent O | batch-1 sliding w1/full w2 DRAM QKV; batch-32 sliding w2; persistent L1 O; BF16 | per-kind real/trace PCC and 50-replay timing |
| Dense MLP | same-input gate/up, GELU/mul, down | pack gate/up, tune K blocks and DRAM placement | load-time packed gate/up; dense-down w3; batch-32 DRAM packed w4 | packed/down sweeps and trace correctness |
| Router | RMSNorm, two static scales, FP32 projection, TopK/softmax/scatter | fold static scales and composite routing | fold scales; retain FP32 row-major routing boundary | PCC, trace, and source audit |
| Decode MoE | gate/up/down sparse matmuls | exact grids, K blocks, placement, dtype/fidelity | 11x2 gate/up w11; portable down; BFP8, gate HiFi4 | profiler rows, BFP4 length-31 failure |
| Prefill MoE | portable separate same-input gate/up plus down dominate 96.5% of seq256 | packed gate/up, exact grids, chunk sizes, L1/DRAM | internal chunk32; packed gate/up 11x4 w11 L1; down 11x8 w11 L1 | 168.959 -> 32.093 separate -> 21.490 packed; chunk64/128 rejected |
| Norm/residual | one-core norms and DRAM residuals | persistent L1 and 88-core sharded chains | L1-interleaved O boundary; reject sharded norms | 0.993390/0.990832 PCC failure |
| Host/layout | device reshapes and conversions | remove host and redundant movement | no host transfer/Torch conversion; retain consumer-required device layouts | hot-path audit and zero host ops |

DRAM-sharded decode was tested per role, batch, and meaningful layer kind rather than dismissed at a
first API error. Batch-1 QKV required layer-kind specialization: width 1 is
correct for sliding but fails full, while width 2 is correct for full but fails
the stricter sliding case. Batch-32 QKV width 2 likewise fails full PCC but is
trace-correct and selected for sliding. Packed dense width 4 is selected for
both batch-32 kinds. Dense-down DRAM configurations
are correct but not consistently faster, and batch-1 packed/down DRAM variants
regress. Sparse weights use their separate 128-expert batched contract.

Prefill candidates preserve the inherited outer 1024-token chunking and public
logical slicing. Only the internal MoE tile loop changes, so non-aligned public
lengths and paged-cache semantics are unchanged. The adapted 128-token DRAM
candidate demonstrates that its initial L1 validation error was not used as a
rejection; measured latency rejects it.

The packed sparse candidate consumes the original HF fused gate/up tensor as
`[1,128,2816,1408]`, runs the unchanged all-expert sparsity mask once, and
splits at the tile-aligned 704 boundary before unchanged GeGLU/down math. It is
not a dense surrogate: final Tracy contains one sparse packed projection per
internal tile, and real PCC/non-aligned gates prove semantic equivalence.
