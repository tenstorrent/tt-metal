# Measured-path topology audit

The inherited selected policy is `head4_inner_all4_shared_down4`; this pass
preserves it while isolating orchestration overhead. Earlier rejected device
families are recorded in `../optimized_multichip_decoder/candidate_comparisons.md`,
`residual_contract.md`, and `final_perf_findings.md`, and in
`../optimized_full_model/terminal_lm_head_plan.md`. They are prior evidence,
not fresh measurements on the current image.

| Boundary | Current sequence / contract | Candidate and constraint | Action |
|---|---|---|---|
| Request prefill → decode | Short B1 greedy exact-length prefill trace; longer prefill eager; decode released and recaptured for each long request | Retain decode only across warmed exact-signature eager prefill, preserving cache identity, logical length, all page columns and mode invalidation | Primary experiment; local4K setup costs293–302ms/request; long-prefill tracing rejected for capacity |
| Steady decode | Nonblocking model and common-sampler traces; persistent token feedback; device position increment | Avoid extra host token/position/page copies on unchanged scheduler state | Count refreshes and compare page-growth transitions |
| Attention projections | Packed QKV, explicit per-role geometry; per-kind accepted precision | DRAM reader/geometry and BFP4 choices already extensively compared in inherited stage | Reprofile reduced representative path before reopening candidates |
| Attention/cache | Native paged decode; BFP8 KV,32-token pages;25 sliding and5 global layers | Context-dependent SDPA tuning must retain262144 capacity and correct logical positions | Compare4K with short context in reduced non-serving profiles |
| Residual/CCL | Replicated BF16 layer boundary, FP32 post-attention residual; persistent async collective outputs; attention CCL BF16/sliding and BFP8/full, MoE CCL BFP8/sliding and BF16/full | Sharded carry-forward and fused CCL/matmul previously adapted and measured slower | Selected precision JSON agrees with observed BFP8 reduce-scatter; no policy drift |
| MoE | Eight selected experts, indexed active path; per-kind reduced-weight policy | Legal sparse matmul geometry/precision and memory placement | Prior candidate tables exist; reopen only measured remaining bottlenecks |
| Terminal path | Vocab-sharded BFP4 head, K4 on11x10 grid; power-of-two local vocab; split common sampler and on-device feedback | LM-head readers and geometry, sampler overhead | Include in reduced profile; no host full-logits readback or greedy argmax |
| Serving | vLLM0.26, synchronous scheduler in catalog, async decode API supported | Same-harness async comparison, deduplicate truly shared page tables, remove avoidable host work | Report TPOT alongside TTFT/E2EL to reject mere timing-phase shifts |

The reduced profiler must include one real layer of each kind, full-sized
weights/terminal path and representative page-table/cache shapes. No live
server or all-layer stack is profiled.

Fresh report advice: the509us gap on the first Slice is the interval from the
previous, out-of-window warmup operation. Native CSV confirms that Slice is the
first operation on all four ranks; it is not an internal replay dispatch gap.
The whole-window summary deliberately starts at its firmware start and includes
all subsequent gaps (2548.31us maximum rank). Already-traced replay therefore
does not have the report's suggested538us tracing opportunity. Router BF16
weights also invalidate the generic BFP8-fidelity advice. Accepted precision
and prior actual-input sparse/subblock/CCL controls are linked above.

The full-grid N20/subblock4 follow-up is now measured and loses to N19.
All35 full-grid block/width candidates,20 conversion-inclusive L1 candidates,
32 grid/block candidates and30 small-chunk DRAM reader candidates were tried
(including exact allocation rejections). Native DRAM K1/N19 wins the isolated
path,375.951us versus~389.85us K4. Paired full-model K4/K2/K1/K4 controls
retain identical128-token outputs and confirm only11–13us total-decode savings.
K1 is selected provisionally for final serving/qualitative qualification; the
host retention/scheduling changes dominate the practical TSU gain.
# Serving readback accounting

The synchronous plugin calls `submit_decode(read_from_device=False,
async_read=False)` and then `process_decode_output_host` through finalization.
That helper's `ttnn.to_torch` performs one blocking token-only device read, but
the inherited `token_readbacks` counter is incremented only by
`read_decode_output`. Thus baseline synchronous logs showing zero readbacks do
**not** mean zero device-to-host transfers. Async mode uses `read_decode_output`
and an event wait. Both modes retain nonblocking model and sampler trace replay;
neither baseline counter alone can prove their full read/synchronization count.

## Context-dependent device cost

Fresh128-token reduced profile (same262144 cache, real layers0/5, K1 head)
has a2474.757us maximum-rank complete replay window. The earlier4K profile
uses K4 head and2548.310us, so the whole-window delta includes the separate
~11–14us head improvement and must not all be attributed to context.
Per-op SDPA isolates the context effect: sliding18.271→35.864us and
global17.539→57.556us at128→4096 input, with unchanged SDPA program/layout.
TopK is unchanged278.955/279.054us. Extrapolating these two representative
SDPA differences over25 sliding plus5 global layers gives~0.64ms; this is
an estimate, not an all-layer device profile. It is consistent with the
remaining~0.5ms short-versus4K async serving TPOT difference after recapture
is removed. The original~3ms gap was not all attention cost.
