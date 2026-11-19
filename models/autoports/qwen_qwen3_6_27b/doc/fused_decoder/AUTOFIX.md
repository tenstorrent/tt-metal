# AutoFix: close the graph-fusing performance gate

The failing contract was the explicit stopping rule: at least 70% of the
weight/state roofline and two final kept gains below 3%, or eight hours of
device optimization. The earlier default was faster than functional but
reached only 64.11%/65.75%. Completing the named checklist did not waive this
gate. [AutoDebug](AUTODEBUG.md) independently confirmed the accounting and
identified missed native fusions. No hardware fault or correctness failure
was used to justify a capability reduction.

## Discriminating experiments

All rows use real checkpoint weights, BF16 dense weights/activations, HiFi4,
FP32 accumulation, the existing HF and direct functional 0.995 gates, and
100 warmed traced replays at B1/prefill 512. Commands are the run_decoder
family in work_log.md with QWEN_DECODER=fused and the candidate stem selected
through QWEN_FUSION_CANDIDATE. Source hashes and exact argv are in each JSON.

| Hypothesis | Result / decision |
|---|---|
| Native packed matmul+SwiGLU removes intermediate gate/up outputs | PCC passes, but default 2.648/2.513ms loses to 2.205/2.068ms control. Adapted K blocks 8/16/20, N block 10 remains 2.692/2.643/2.639ms GDN. Rejected. |
| Native RMSNorm absorbs explicit widen/narrow casts | PCC passes; general prefill improves. BF16 input retains FP32 intermediate/gamma formats under the selected config. Kept for general shapes; decode unsharded control is slower. |
| Native width-sharded RMSNorm plus dtype-converting transfers removes separate cast ops | B1/s1/width5120:2.205/2.068ms. Eight cores;40-core control has no meaningful gain. Kept. This is a fused norm adapter inside fused_decoder.py, not a new optimized-decoder stage. |
| GDN softplus and FP32 bias intake can merge with binary ops | Initial two-parameter binding syntax corrected; adapted 2.201ms, PCC passes. Kept. |
| Retain padded convolution Q/K/V rather than slice then pad again |2.192ms with logical convolution end unchanged; output/state PCC passes. Kept. Broader padded-Z chain was slower. |
| Move head concatenation before elementwise Z gating |2.168ms; eliminates Z reshape/transpose, per-head norm unchanged. Kept. |
| Native output logical/padded shape metadata avoids output copy | Initial view overload corrected to reshape with explicit logical/padded Shape.2.162ms, PCC passes. Kept. |
| Native prep gate masking removes explicit beta/g pads | Independent read-only supporting agent verified exact native signatures, tail masking and head/core ceiling. B1/s1 only, same phased topology:2.132ms. Kept. Larger batches use the public wave-capable native fused path. |
| Native sigmoid-gated norm followed by Z multiplication expresses SiLU gating | Adapted packer flag and rank1 weight requirements; PCC passes but2.167ms loses to the2.162ms flat-Z control. Rejected. |
| Fuse paged K and V updates | Required equal-sized disjoint input shard grids implemented;2.061ms vs2.067ms, PCC passes. Kept. |
| Native HF rotary preserves logical shape |2.058ms vs2.061ms, PCC passes; removes repair slices. Kept. |

The combined candidate reports 2.131844ms GDN and2.041396ms full attention,
with warmed prefill 5.594715/5.771844ms. The numerator remains 773062656 and
746590208 bytes divided by 512 GB/s: floor 1.509888/1.458184ms. These candidate
results exceed 70%; the delivered default must reproduce them. Final-source
results, profiler reconciliation and all correctness gates are indexed by
acceptance.json and roofline.json after validate_final.sh completes.

Last incremental kept GDN changes were flat-Z output view (~0.28%) and
logical gates (~1.40%). Full attention's fused-cache (~0.29%) and HF-rotary
(~0.13%) changes were also below 3%; the subsequent combined general-norm
cast absorption was below 3% in decode. No idle time is counted as device
optimization. Native state-output aliasing/ring mode and residual epilogues
remain untried after the stopping condition, recorded as hand-off notes.

## Changed files and validation

Only tt/fused_decoder.py, test runners/evidence tools, and stage docs changed.
The functional runtime remains untouched. Tests select FusedDecoder
explicitly, and the functional class runs only as a numerical control outside
measured windows. Native source audit is in native_runtime_audit.md.

The final bounded, serialized driver is validate_final.sh. It runs both
layer-kind contracts, batch2 awkward lengths, batch32, mixed positions,
retained trace reuse,262143/262144 contexts, synthetic tests, separate
Watcher runs, 100-replay timing and separate Tracy/tt-perf-report runs. The
driver fails on any failed job. No device reset, process termination,
precision lowering, context reduction or next-stage implementation occurred.

Residual risk: native internal GDN prep/scan bindings are checkout-specific;
future upstream changes require revalidation. Per-layer persistent convolution
scratch requires serialized calls on its owning queue. The roofline is a
nominal hardware estimate, not a measured-bandwidth substitute. This report
does not itself replace the independent stage review.

## Subsequent context-gate repair

The first delivered combined graph exposed an L1 lifetime defect at the
262143-token full-attention continuation: retained K/V update input tensors
clashed with SDPA's large-context circular buffers. The trace and old source
are preserved under before_lifetime_fix_* and source_archives. Releasing the
local input list immediately after the ordered fused update removes the
unneeded lifetime; persistent KV caches are unaffected. The exact failing
context is rerun before repeating final gates. This is required correctness
work, not a waived capacity limit.

Final outcome: the repaired source passed the full serialized suite (34 final-source results), four synthetic cases, four clean Watcher logs and host/profiler reconciliation. Delivered decode medians are2.132236/2.041422ms, reaching70.8124%/71.4298% of the unchanged roofline. The L1 lifetime finding is closed by delivered_context_3_262143.json and delivered_context_3_262144.json; context262144 remains supported. Independent stage review is the next gate.
