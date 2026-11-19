# Physical-batch prefill candidate

This is a source-audited, CPU-checked experimental family, with no hardware result or default recommendation. It tests whether avoiding compact/repad transformations compensates for additional padded matmul and collective work. It keeps logical `[B,S,C]` through prefill dense projections, distributed norm and residuals. The motivating profiles are `perf/interim_profile4_l0_b32_s33/prefill_perf_report.csv` and `perf/interim_profile4_l3_b32_s33/prefill_perf_report.csv`; the parent's measured four-reshape totals are approximately 742us and 534us respectively. Those totals are an opportunity ceiling, not a predicted speedup, and this family does not remove every reshape in attention/GDN head handling.

## Source legality

- `ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp`, `validate_matmul_mcast_fuse_batch_preamble`: fused-batch native matmul permits broadcast weights with batch dimensions equal to one. This family does not transpose A.
- `.../utilities/matmul_utilities.hpp`, `get_M_dim`: fused M is derived from **padded** input volume divided by padded width. Thus `[B,S,K]` uses `B*ceil32(S)`, not `ceil32(B*S)`.
- `.../factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp`: fused batch sets the outer batch loop to one and uses that padded M. The prototype programs `per_core_M=ceil(B*ceil32(S)/(32*grid_y))`.
- `.../utilities/matmul_utilities.cpp`, `compute_matmul_output_shape`, and `MatmulDeviceOperation::compute_output_specs`: logical output copies A's batch/sequence dimensions and B's output width; fuse_batch does not change the public logical output to `[1,B*S,N]`.
- The joint correction combine already iterates `B*(padded_S/32)*130` tiles and reads the primary/correction columns through the same physical stride. It can consume a batched FP32 joint output without compacting.

This establishes a plausible legal native path, not proof of successful program compilation, native numerical equivalence, peak allocator capacity or speed. Native kernel validation remains authoritative.

## Exact opt-in behavior

New module: `tt/tiled_batch_prefill_candidate.py`. New runner: `tests/probe_tiled_batch_prefill.py`; it substitutes a real subclass only during runner setup, then delegates the unchanged correctness, timing, trace, stress and output machinery to `run_multichip_decoder.main`. No per-instance method binding, old-owner closure, tensor host conversion or model source edit is introduced.

Setup forces `flat_prefill=False`, `flat_finish=False`, `flat_norm=False`, `flat_ccl=False`, `flat_prefill_correction=False`, `chunk_add_prefill_correction=False`, `prefill_minimal=False`, and `minimal_swiglu=False`. This is a coherent family, not an isolated reshape ablation: it uses the existing contiguous packed gate/up weights and kind-specific elementwise epilogue instead of the paired native minimal-matmul SwiGLU path. Existing attention/MLP weights, dtypes, compute fidelity, correction values and activation-rounding choice remain in effect. Selected-policy metadata records the family.

For B>1,S>1 only, `_linear` bypasses the compact MMRS branch and calls ordinary local matmul followed by the existing row reduction. Joint GDN uses the same packed joint matrix and exact existing FP32-add/BF16 combine. Other corrected projections retain inherited correction dispatch. Native 2D program options use each role's existing geometry; joint options also inherit the joint candidate's prefill overrides. The new `tiled_prefill_row_cap` defaults to4096 physical rows and splits **whole users**, then concatenates on batch dimension. Joint pieces are combined toBF16 before concatenation. The cap cannot subdivide one user's sequence; inherited temporal chunks remain512 by default. `row_cap` inside joint prefill options does not control this family; use `tiled_prefill_row_cap` explicitly.

Compact decode and B1/raw single-token projection dispatch remain inherited. Global setup toggles can still alter B1 prefill scheduling and collective layout; this is not a claim that B1 performance is unchanged. Existing MMRS weights are retained when requested, because inherited B1/S1 paths can use them. The paired SwiGLU matrix is not created, saving47,349,760 bytes per device/layer at the current BFP8 physical shape. No new weight or persistent runtime scratch allocation is added.

## Paid padding and resource cost

Let `Rflat=32*ceil(B*S/32)` and `Rbatch=B*32*ceil(S/32)`. For fixed physical K/N and dtype, dense tile work and noncompact activation/CCL storage scale as `Rbatch/Rflat`, before extra per-core/block padding. This is real extra work, not useful model FLOPs and not extra useful roofline bytes. Collective payload below means tensor payload, not measured network/link traffic or a full collective traffic multiplier.

| B | S | Compact rows | Physical-batch rows | Ratio |
|---:|---:|---:|---:|---:|
| 1 | 33 | 64 | 64 | 1.0000 |
| 32 | 33 | 1056 | 2048 | 1.9394 |
| 1 | 513 | 544 | 544 | 1.0000 |
| 32 | 513 unsplit | 16416 | 17408 | 1.0604 |
| 32 | 512 | 16384 | 16384 | 1.0000 |

Actual S513 scheduling is512+1, not one S513 matmul. The S512 chunk has no extra per-user tile padding; the S1 tail retains inherited compact projection dispatch. Its nonflat norm/CCL layout can nevertheless carry padded per-user rows and must be measured. These formulas do not claim the whole S513 forward incurs the unsplit ratio.

For B32S33 BF16 hidden tensors, compact versus batched payload is10,813,440 versus20,971,520 bytes at width5120, and2,703,360 versus5,242,880 at width1280. A width128 BF16 norm-stats tensor is270,336 versus524,288 bytes. A joint width7168 FP32 output is30,277,632 versus58,720,256 bytes. The latter combines to width4160 BF16. Gate/up width8704 BF16 similarly becomes35,651,584 bytes rather than18,382,848. Each full-batch output and concat can remain live independently; the row cap bounds a matmul piece, not total layer activation residency. At cap4096 the joint FP32 piece alone is117,440,512 bytes, plus its input, corrected output and any live concat pieces. Peak allocator/trace reservations must be measured rather than inferred from this one tensor.

No logical padded tokens are inserted into recurrent state or KV cache: `[B,S]` is unchanged, the inherited temporal scheduler still uses logical S, convolution's `actual_end` still receives logical S, and existing inactive padded recurrence rows are preserved. Public context remains262144. Head reshape/permute, RoPE, GDN convolution/chunk processing and cache operations are inherited and can still move data. Extra collective padding or slower separate MM+RS can outweigh saved compact/repad movement.

## Invocation and validation boundary

CPU only:

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_tiled_batch_prefill --cpu-plan
```

Hardware invocation for the parent, using its selected concrete policy file and reference:

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_tiled_batch_prefill \
  --layer 0 --batch 32 --length 33 --policy-file /path/to/current-policy.json \
  --reference models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder/delivery_perf_ref_l0_b32_s33.pt \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/tiled_batch_l0_b32_s33.json
```

Repeat with layer3 and its reference, then normal nonaligned S513 and B1 cases if short-prefill results justify continuing. Keep the same controls, correctness thresholds and final evidence obligations. No hardware invocation was performed for this proposal. CPU checks covered128 M/K/N geometries, padded-M coverage, N coverage, K/subblock divisibility, and an AST scan confirming no reshape or tensor host-conversion calls in the new runtime methods. Module import and `--cpu-plan` did not import TTNN. The checks do not replace device correctness, runtime sentinel, tracing or profiling.
