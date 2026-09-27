# Native TP4 performance accounting audit

CPU-only audit of `tests/summarize_multichip_perf.py`, the actual historical `profile_v0` native CSV, the selected hybrid/grouped runtime sources and native collective/SDPA contracts. Only the multichip summarizer was changed. `tests/summarize_perf.py`, model runtime and hardware were untouched. This audit prepares final-profile accounting; the only currently inspected native whole-layer CSV remains the historical sliding 4096/1 profile, not the final selected 4096/128 path.

## Concrete correction and regression

The native minimal reduce-scatter returns `[intermediate_buffer, output_buffer]`, with an optional third penultimate scratch buffer. In the historical CSV, `OUTPUT_0` has shape `[2,1,32,2816]`; it is the Linear topology scratch allocation. `OUTPUT_1` is the actual reduced output `[1,1,32,704]`. The generic operand estimator was charging the entire scratch allocation as one output write while its assumptions excluded internal CCL passes. An allocation size does not establish the number of bytes written/read internally.

The multichip wrapper now counts the operation's declared input and actual `OUTPUT_1`, excluding scratch `OUTPUT_0`/`OUTPUT_2` from the nominal collective-I/O estimate. Excluded allocation bytes are reported separately. This does not estimate internal CCL scratch transactions or fabric/NoC traffic; they remain outside the byte numerator and inside the whole-layer timing denominator.

Source: `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op_device_operation.cpp:172-270` constructs/returns the two or three buffers; `:423` identifies output index1. Linear topology doubles the scratch's leading dimension. The correction is local to the multichip report and does not modify the baseline estimator.

| Actual historical sliding 4096/1 CSV | Previous report | Audited report |
| --- | ---: | ---: |
| Prefill whole-layer device span, us | 825575.946667 | 825575.946667 |
| Decode whole-layer device span, us | 999.845185 | 999.845185 |
| Estimated decode DRAM operand bytes | 199607360 | 193840192 |
| Estimated bytes / four-ASIC DRAM peak / time | 9.747962% | 9.466319% |

The difference is exactly 5,767,168 scratch-allocation bytes across four ASICs and three reduce-scatter calls per ASIC. No latency or useful-FLOP value changed. The original report is preserved; the corrected report is [profile_v0/whole_layer_audited.json](profile_v0/whole_layer_audited.json), with [per-device windows](profile_v0/whole_layer_audited.windows.csv).

[CPU regression evidence](native_accounting_cpu_checks.json) records the actual CSV SHA256, summarizer hashes, the old/new byte totals and source-derived expectations. [Preserved test source](native_accounting_cpu_checks.py.txt) imports no TTNN. It checks actual active-expert metadata, dtype scaling, grouped payload scaling, 128 SDPA positions for both kinds and whole-window preservation. The synthetic full/grouped checks validate arithmetic and metadata handling; they are not newly measured native profiles.

## Sparse and grouped communication accounting

The historical native CSV exposes exactly two sparse matmuls per device. Each records `use_indices=true`, no numeric `nnz`, weight groups 128, compact logical output groups 8 and eight UINT16 indices. The existing baseline helper correctly computes weight reads as 8/128 of the local padded expert-weight allocation. The multichip summarizer retains that helper, checks active 8 and emits the first replay's sparse evidence. It never charges all 128 experts during indexed decode. Hybrid EP prefill does not change this TP decode accounting; prefill union/extra expert work is excluded from useful FLOPs, not from the measured time.

For replicated `[1,1,1,2816]` decode with physical M32, the nominal mesh I/O for one reduce-scatter plus all-gather is:

| Recorded payload dtype | One branch, bytes across four ASICs | Paired/grouped branch tensor, bytes |
| --- | ---: | ---: |
| FP32 | 3,604,480 | 7,208,960 |
| BF16 | 1,802,240 | 3,604,480 |
| BFP8 | 957,440 | 1,914,880 |

The formula is four devices times two collectives times `(32*2816 + 32*704)` elements times storage bytes. Grouping shared and routed tensors changes their Z extent from 1 to 2, so one grouped collective pair carries the same nominal bytes as two separate same-dtype pairs. Its performance benefit can come from fewer launches/barriers and other execution effects; the estimator does not invent a halved payload. Actual concat/slice/conversion operations retain their own recorded DRAM operand traffic. Attention BF16/BFP8 follows its recorded dtype independently of the BF16 grouped MoE payload. BFP8 uses 1088/1024 bytes per element, including tile exponents; BFP4 uses 576/1024.

The final report now emits per-collective example shapes, dtypes, nominal I/O and excluded scratch allocations so grouped and attention payload choices can be checked directly against native metadata. No new selected-path native CSV was available for this audit.

## Native SDPA and whole-layer windows

The previous multichip code hard-coded 128-token rounding without checking native metadata. It now validates TP4 local query/cache geometry and reuses the existing metadata-based chunk/window solver. A temporary host-side metadata view scales only the head axes to that solver's TP1 contract; returned K/V counts are divided back to this rank's actual heads. No tensor or device work is involved.

Local query heads are 4 on each rank. Sliding cache heads are 2 at width 256; full cache heads are 1 at width 512. For positions 4096 through 4223 and current dynamic chunk 128, the CPU regression finds mean mesh K/V reads 5,009,152 bytes sliding and 18,382,848 bytes full. Full TP4 has four total local KV heads because the two logical full-attention KV heads are replicated; the byte estimate counts those real reads across all four ASICs. It does not divide away actual replication. The full useful-FLOP numerator remains logical model work.

`NativePagedAttention` requests `k_chunk_size=0`, FP32 destination accumulation and unsharded DRAM query. Source/metadata resolve its chunk to 128 here. Missing or contradictory chunk/window/cache metadata now fails explicitly; an optional `--native-sdpa-read-chunk-size` supports a documented explicit fallback when effective metadata is absent. `--decode-start-position` defaults 4096. Exactly one native SDPA per device/replay, four device windows, nonempty replay IDs and consistent native op counts are checked.

For every prefill or trace replay and each ASIC, timing remains first native firmware start to final native firmware end, using the median firmware duration/cycle ratio from that device. All intervening native operations and gaps are included. Mesh duration is the maximum of the four per-device spans, avoiding any assumption that device clock origins are synchronized. Decode reports the mean of complete replay spans. Host input refresh, output reads and gaps between separate replays are excluded; this is not end-to-end serving latency. The target-workload marker requires 128 steps starting at 4096; the useful prefill formulas are specifically for 4096 logical input tokens.

## Useful work and theoretical peak

The denominator is the theoretical total of four P300 Blackhole ASICs: `4 * 120 * 4096 * 1.35e9 = 2.654208e15 FLOP/s` at the declared LoFi reference, and `4 * 512e9 = 2.048e12 bytes/s` DRAM bandwidth. It uses 120 theoretical Tensix cores per ASIC, not the 110 workers exposed by this runtime. The FLOP/cycle and fidelity convention is documented in `tech_reports/GEMM_FLOPS/GEMM_FLOPS.md:50-70`; the per-ASIC/card basis was already verified in the optimized stage's roofline audits. This is a common mixed-fidelity useful-work reference, not measured FPU utilization or controller bandwidth.

Useful work counts a multiply-add as two FLOPs and includes one logical model layer, not four copies of it. Let `S=4096`, `H=2816`, `Q=16`, shared width 2112, expert width 704 and top-k 8. Shared MLP is `6*S*H*2112`; selected experts are `6*S*H*704*8`; router is `2*S*H*128`. Attention counts QK and PV with the logical causal/sliding pair count and all 16 logical query heads. Sliding includes distinct K/V projections; the full useful-work convention counts their shared projection once. `FusedAttention`'s common-KV global path derives V normalization from K. Duplicated/padded physical TP work is not silently added to the useful numerator.

| Useful term, FLOPs | Sliding | Full |
| --- | ---: | ---: |
| QKV and output projections | 283467841536 | 401579442176 |
| Shared MLP | 146163105792 | 146163105792 |
| Active top 8 experts | 389768282112 | 389768282112 |
| Router projection | 2952790016 | 2952790016 |
| Causal/sliding QK and PV | 60137930752 | 274945015808 |
| Total | 882489950208 | 1215408635904 |

Padding, inactive/union expert work, duplicated projections, masked attention work, normalization, scalar/transcendental operations and data movement are excluded only from this useful numerator. Their operation time remains in the whole-layer denominator. No percentage is clamped. `--precision-policy` records the actual mixed policy in the final summary.

## Validation and remaining evidence

The Python-only summarizer passes compilation, Black checking and the actual-CSV CPU regression. Baseline `summarize_perf.py` remains SHA256 `8beb5082231274b8a8b8f1a7ba6212471b37f0bc1e615a9b484b54b43706381a`. No C++ build or hardware run was required. Final selected native profiles must still be collected separately for sliding and full prefill/decode and passed through the corrected summarizer with 128 traced steps. The historical numbers above must not be promoted to final selected-path performance.
