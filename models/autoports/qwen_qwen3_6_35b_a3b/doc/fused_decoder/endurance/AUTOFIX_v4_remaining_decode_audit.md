# Remaining decode pattern audit on selected v4

Inspection base: [selected v4](../selected_fused_decoder_v4.py.snapshot), SHA256 `b1633d51621b4c45d9514a66fad21faa57ad562185502342c3d87bd5ebc2701c`. Production remains frozen at that source. This audit rereads the graph-fusing skill's dedicated-op, structural-rewrite and adjacent-op-folding patterns, [pattern assessment](../pattern_assessment.json), the current decoder, [qualified-v3 topology](../profile_v3/op_topology.md), and the completed v4 control evidence. Historical profile costs identify work, not the frozen-v4 latency of that work. Root owns current validation and hardware; this investigation executed CPU setup only.

One additional structural boundary is supported by the current APIs: persistent row-major convolution carry, optionally written directly by the slice writer. This is newly useful because selected v4 constructs history directly in row-major layout. The previously prepared v4 controls are measured; they are not proposed again here. Neither the new controls nor this audit establish the stop rule, stage completion, or an optimal graph. At assignment the reported conservative device time was about 3.3 hours and useful-byte roofline remained below 70%; final selected-source validation was in progress.

## New boundary: keep the convolution cache row-major

Current `tt/fused_decoder.py:461–473` allocates convolution cache `[1,3,8192]` as BF16 TILE. `_linear_attention:532–545` immediately untilizes it to join current projected rows in RM, uses that RM history for all tap slices, then tilizes the final three raw projected rows before copying them into the same TILE cache. Both sides of the persistent boundary are now RM. No arithmetic consumes the persistent cache in TILE layout.

The conservative control changes only that cache allocation to BF16 RM, passes it directly into RM history concat, and copies the existing RM carry slice into the stable RM destination. It leaves all tap tilization/widening, four-tap accumulation, BF16 preactivation/FP32 SiLU/BF16 output boundaries, output slices, and recurrent state unchanged. It does not replace the cache tensor object during runtime.

Native contracts, relative to repository root:

- `ttnn/cpp/ttnn/operations/data_movement/copy/copy.cpp:15–17` passes the destination's memory configuration and dtype into the preallocated copy operation.
- `copy/device/copy_device_operation.cpp:65–74` supports a row-major factory; `:105–136` requires matching logical shape/layout and forbids dtype conversion only for RM. Both operands here are BF16 RM DRAM `[1,3,8192]`, so no dtype conversion is requested.
- `copy/device/copy_same_memory_config_program_factory.cpp:57–87` explicitly supports RM stick pages and partitions work by rows. This is also the performance caveat: only three logical rows can limit parallelism. Removing conversions is not proof that the different copy kernel is faster.
- Existing selected RM concat and RM slice already operate on this history geometry. No new unaligned TILE view or ring-buffer indexing assumption is introduced.

The allocated convolution payload shrinks from `32*8192*2 = 524288` bytes to `3*8192*2 = 49152` bytes: **475136 bytes (464 KiB) per state**, excluding allocator/bank rounding. The probe records native padded shape and derived payload rather than calling this total device-memory usage. Recurrent state remains the same 2 MiB FP32 buffer. A 32-state caller would remove 14.5 MiB of convolution payload if every state used this internal representation; this is a capacity bound, not a measured context-capacity pass.

The old [native causal-convolution RM carry controls](../native_conv_carry_rm_0.json) changed both the native arithmetic path and its cache adapter. They subsequently failed stronger endurance checks; [native convolution reassessment](AUTOFIX_native_conv_reassessment.md) identifies the missing BF16 preactivation boundary. They do not reject RM storage independently. Conversely, [v3 RM carry reuse](AUTOFIX_v3_data_movement.md) and [v4 direct RM history](AUTOFIX_v4_decode_opportunities.md) explicitly retained TILE persistent state. Neither measures this new persistent boundary.

## Separate second control: slice writes the stable RM cache

After the conservative RM-cache control, a second snapshot replaces `RM slice -> temporary -> copy` with:

```python
ttnn.slice(history_rm, (0, s, 0), (1, s + 3, 8192), output_tensor=state["conv"])
```

The destination is separate from the newly concatenated history input, even though the old cache was one input to concat. Keep this statement at the original carry-update location after all tap consumers. The dictionary entry and buffer address remain unchanged.

Source support is explicit: `data_movement/slice/slice_nanobind.cpp:99,121,133` exposes `output_tensor`; `slice.hpp:13–23` accepts it; `slice.cpp:135–138,165–182,424–435` derives the output configuration from the destination and passes a matching-layout, nonaliasing destination to `prim::slice`. `slice/device/slice_device_operation.cpp:160–196` requires matching padded shape, exact RM logical shape and tensor layout, and `:413–416` returns the preallocated tensor. For interleaved BF16 RM `[1,S+3,8192] -> [1,3,8192]`, source and destination page formats agree. No dtype conversion or layout repair is required. Installed runtime behavior still needs the root-run probe; the source proof is not a device claim.

This optional writer fusion is isolated from the first control so an RM-copy performance regression cannot be confused with the writer's effect. No shared-native changes are proposed.

## Prepared isolated artifacts and discriminator

| Artifact | Purpose | SHA256 |
| --- | --- | --- |
| [v4_persistent_rm_cache_baseline.py.snapshot](v4_persistent_rm_cache_baseline.py.snapshot) | Byte-identical selected b163 control | `b1633d51621b4c45d9514a66fad21faa57ad562185502342c3d87bd5ebc2701c` |
| [v4_persistent_rm_cache.py.snapshot](v4_persistent_rm_cache.py.snapshot) | RM allocation, direct RM consumption, RM carry copy | `eeff5ed09d96709c0ad2223e7a879386bb02e9be93add6c63abc0d4cfb352eea` |
| [v4_persistent_rm_cache_slice.py.snapshot](v4_persistent_rm_cache_slice.py.snapshot) | Same RM allocation plus direct slice writer | `ef2b0d13cd1217144ccc7de087d4b43631144ffee6a99f445577e46e63c9251e` |

[tests/v4_persistent_rm_cache_probe.py](../../../tests/v4_persistent_rm_cache_probe.py) extracts each snapshot's actual `new_recurrent_state` method and complete convolution block through the carry update. It uses real layer-0 checkpoint taps and explicitly labeled synthetic BF16 projected rows; it does not claim those rows are captured activations. S1/33/128/129 plus exhaustive finite BF16 history cover single-call adapters. It compares all three FP32 tap operands, FP32 preactivation, BF16 output and carry. Only the specified source substitutions are permitted by CPU assertions.

Allocation checks verify zero initialization, BF16 cache dtype/layout/shape, unchanged FP32 recurrent allocation, cache readback, clone/copy reset, output repeatability, and the persistent Python tensor object plus device address. Sequential chains cover `128 -> 1 -> 1 -> 1 -> 33 -> 1` and `33 -> 1 -> 1 -> 129 -> 1`; every eager step checks carry against independent CPU chronology and compares outputs across all variants. The cache cloned immediately after prefill is also restored and its decode/continuation suffix rerun. Captured chains are replayed twice after restoring the original stable buffers, comparing every retained output and final carry. These are convolution-cache contract tests, not full-prefill/full-decoder acceptance.

Single-call warmed trace timings include the entire adapter: input conversion/concat, every tap slice and conversion, arithmetic, activation, Q/K/V output slices, carry slicing, and copy or direct write. State allocation, clone/reset and readback remain outside timing as setup/diagnostics. S1 history is first stabilized through at least four calls; first/last outputs, carry and stable addresses are checked after timed replays. No layout adapter is removed from the measured function. Exhaustive-code checks are correctness-only. Differences remain recorded; no global bitwise model-acceptance gate or threshold relaxation is introduced.

[CPU setup report](v4_persistent_rm_cache_probe_cpu.json) passes five cases, two chain mappings, source-scope assertions and AST extraction; TTNN was not imported. Device status is unset. Earlier measured scripts/results are untouched.

Root-only commands after the active frozen-source queue:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v4_persistent_rm_cache_probe \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v4_persistent_rm_cache_probe.json

for variant in baseline '' slice; do
  if [ -z "$variant" ]; then suffix=''; else suffix="_$variant"; fi
  python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.benchmark_fused_snapshot \
    --snapshot "models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v4_persistent_rm_cache${suffix}.py.snapshot" \
    --layer 0 --length 256 --context 512 --prefill-repeats 10 --decode-repeats 60 \
    --output "models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v4_persistent_rm_cache${suffix}_s256.json"
done
```

Any retained candidate still needs the normal seeded full-decoder continuation, determinism, trace, capacity and final-reference gates. Whole-model tests must allocate state from the selected candidate class. `tests/stress_fused_decoder.py:94–99,111–118` already does this and clones/copies the actual state without asserting TILE cache. One diagnostic helper is **not** layout-agnostic: `linear_frontend_stream_probe.py:336–341` guesses conv layout from the presence of `to_layout(state["conv"])` and can choose the wrong layout for this new source. Do not use that heuristic to qualify the candidate; the new probe calls the actual allocation method. No helper or core harness was edited.

## Other patterns: concrete exclusions rather than an optimality claim

| Area | Current boundary / reason not to repeat | Source and prior evidence |
| --- | --- | --- |
| Dedicated causal convolution | No activation-off or preactivation-output option; native SiLU runs before BF16 pack. Stock exact FP32 depthwise adapter was slower: decode .871028 vs .155060 ms, prefill 1.221654 vs .627377 ms. | `experimental/kda/qkv_causal_conv1d_silu/qkv_causal_conv1d_silu.hpp:21–59`, device operation `:190–201`, compute kernel `:66–77`; [native reassessment](AUTOFIX_native_conv_reassessment.md), pattern `conv+activation`. |
| More history-cast removal | First tap uses accurate mixed multiply; remaining addcmul inputs must share dtype. Selected v4 folds RM tilization/widening instead. Tiny synthetic differences remain explicitly documented; model PCC, not a silently stronger all-input bitwise rule, governs selection. | `eltwise/ternary/ternary.cpp:348–353`; [widening findings](AUTOFIX_v4_conv_widen_boundary.md), [adapter report](v4_conv_adapter_probe.json). |
| Mixed recurrent V subtraction | Accurate mixed BF16/FP32 subtract with FP32 output is not exposed by the current Python wrapper: explicit false is rejected for FP32 ADD/SUB, while true selects the narrower mixed path. Existing FP32/FP32 stays SFPU. | `eltwise/binary_ng/device/binary_ng_device_operation.cpp:25–58`, `eltwise/binary/binary_nanobind.cpp:1314–1396,2185–2195`; [v4 blockers](AUTOFIX_v4_decode_opportunities.md). |
| Recurrent native kernel, matmul or FMA | Native state narrowing and altered FP32 reduction/update association already produce model-visible failures. Output-only native adapters were measured slower. The current separate multiply/add and ordered sum are deliberate. | [native output controls](AUTOFIX_native_output_only_candidates.md), [recurrence diagnosis](../AUTOFIX_recurrence_hypothesis.md), `stress_addcmul_0.json` and `recurrent_memory_matmul_0.json` in parent directory. |
| Final recurrent copy | New state is also the L1 input to the Q contraction before copying to stable DRAM. Writing the add directly to the persistent DRAM output is not free copy elision: it changes the output-contraction producer placement and source aliasing/lifetime. No measured win inferred, no new control prepared. | Current F `:510–523,629`; [recurrent placement rationale](AUTOFIX_recurrent_boundary_fusion.md). |
| Projection geometry | Compact L1 singleton cumsum, serial K1024 prefill and RM input/output hoists already selected. Smaller tiles are blocked by 32×32 binary validation and fixed FastReduce tile geometry. Earlier packed A/B reduces payload but full decode regresses 1.458950 -> 1.686174 ms. | `binary_ng_device_operation.cpp:480–501`; `experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_program_factory.cpp:48–64,101–112`; [padding contracts/measurements](AUTOFIX_state_projection_padding_contract.md), [serial/RM controls](AUTOFIX_v4_remaining_patterns.md). |
| Input/output norm fusion | Current exact input sum/inverse refinement and output BF16 core/norm/gamma boundaries repair actual stream failures. Native mean or weighted gated norm changes those boundaries. Generic BF16 identity casts are already removed. Final four/eight-lane reduction control was exact but not consistently faster. | Current F `:355–450,643–660`; [seed154 diagnosis](AUTOFIX_seed154_recurrence_and_gate.md), [last-reduce control](AUTOFIX_v3_norm_last_reduce.md), [identity probe](v3_norm_identity_probe.json). |
| Decode routing movement | Direct sharded native slice and original TILE score geometry are selected. Delaying score tilization to eight separate routing planes was measured slower. Repeating it would not be a new boundary. | Current F `:683–700`; [router direct slice](AUTOFIX_v4_router_direct_slice.md), [RM score geometry](AUTOFIX_v3_data_movement.md). |
| Expert fusion | Native MoE compute hardcodes BFP4 weight CB, with no BF16 option. Indexed sparse prefetch is excluded by current capability/firmware. Native postcombine, split fused-activation projections, minimal shared SwiGLU and residual-output projection all have adapted measured losses. | `experimental/ccl/moe_compute/device/moe_compute_program_factory.cpp:839`; [pattern assessment](../pattern_assessment.json) entries `Native fused MoE compute`, `DRISC sparse-weight transport fusion`, `Native post-combine reduce in decode`, `matmul+activation`, `native minimal-matmul SwiGLU for shared expert`, `native output projection plus residual add`. |
| Padding/consumer rewrites | MoE/concat-head logical crops are now metadata-only. Full-only grouped-down RM output is selected; unconditional linear adaptation regresses longer prefill. These newly measured choices are already in b163. | [padded view proof](AUTOFIX_v4_padded_views.md), [grouped output controls](AUTOFIX_v4_grouped_rm_padding.md), [full-only selection](AUTOFIX_v4_full_only_grouped_rm.md). |
| Attention and unrelated patterns | Native SDPA, RoPE, QKV/head creation, head concatenation and generalized topk/softmax are already used. No BatchNorm, parallel spatial conv, spatial pooling or distributed all-gather exists in this single-chip graph. | Current full-attention methods and [pattern assessment](../pattern_assessment.json); these are applicability statements, not performance optimality claims. |

No additional arithmetic rewrite with a demonstrated matching existing-op contract and a new likely material decode saving emerged from this audit. The two RM-cache controls are bounded structural tests justified by the changed producer/consumer layout, not by a need to consume the remaining stop-rule time.

## Root-owned RM-cache device result

The completed [device probe](v4_persistent_rm_cache_probe.json) reports `status=exact`, `device_exact=true`. Its [original provenance](v4_persistent_rm_cache_probe.json.provenance.json) records execution at `2026-11-13T06:05:33Z`; production was b163. The result binds probe source SHA256 `62f1e8d70ac0710e4ae46fb22ad51e34727f014a81e78a9f0c819634b50745c6` and all three immutable snapshot hashes in the table above. A read-only recheck verified those hashes and both imported helper hashes against the files, and checked the detailed equality records rather than relying only on the top-level flag. [Review manifest](v4_persistent_rm_cache_probe.review.json) binds the original result SHA256 `471f0a0ec58a8edc6326de1cafe21725bf2fcf3f74369da639d56f6a322fae2b` and its original provenance. Neither original artifact was rewritten.

Both candidates match baseline with zero code differences for all five single-call cases: S1/33/128/129 and exhaustive finite BF16 history. All recorded FP32 tap inputs, FP32 preactivation, BF16 output and independent CPU carry chronology agree. Allocation/reset/readback checks pass; native padded shapes confirm 524288-byte TILE versus 49152-byte RM payloads. Persistent object and buffer address stay unchanged.

Both continuation chains pass all eleven eager step comparisons. The six post-prefill restore/suffix runs (three variants across two chains) and twelve captured-chain replays (two per variant per chain) preserve every retained output and final carry. First/last timed replay outputs and carry are exact for all twelve timed variant/shape combinations. These checks validate the actual cache adapter and stable ownership; they do not substitute for full-decoder seeded endurance or capacity gates.

Complete-adapter median latency, milliseconds; each entry has 15 samples with 10 trace replays per sample:

| Rows | b163 TILE cache | RM cache + copy | RM cache + slice writer | RM copy saving | Slice minus copy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.142646814 | 0.125709409 | 0.125630200 | 11.87% | −0.079 us |
| 33 | 0.251861708 | 0.234526489 | 0.234577572 | 6.88% | +0.051 us |
| 128 | 0.522891013 | 0.505903596 | 0.505439611 | 3.25% | −0.464 us |
| 129 | 0.617149426 | 0.598957622 | 0.599519629 | 2.95% | +0.562 us |

The clear measured effect is persistent RM storage: the complete S1 adapter saves about 16.94 us with copy and 17.02 us with the slice writer. The extra direct-writer difference is only 0.079 us at S1 (about 0.063% of the RM-copy adapter) and changes sign by shape. This single probe does not establish a decisive copy-versus-slice winner. All layout, carry, conversion and activation costs remain inside these adapter timings; there is no convolution-arithmetic or intermediate-rounding change between these controls.

Root's immutable-baseline/RM-copy/RM-slice whole-decoder benchmark queue is still running at this update. Final selection, whole-path speedup and renewed model gates remain pending. No production or harness change was made in this result review.
