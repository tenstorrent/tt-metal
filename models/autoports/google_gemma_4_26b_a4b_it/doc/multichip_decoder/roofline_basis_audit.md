# Roofline basis audit

Source/CPU audit, 2026-09-26. No device command, runtime change, new timing, or
performance improvement is claimed. The numerical examples below reuse existing
native profiles; `profile_v0` is historical and does not measure the final TP4
or EP4 candidate.

## Recommended denominator

Keep the current **physical ASIC, LoFi reference** for the whole-layer headline:

```text
TP1 peak = 120 * 4096 * 1.35e9 =   663,552,000,000,000 FLOP/s
TP4 peak = 4 * TP1 peak       = 2,654,208,000,000,000 FLOP/s
TP1 DRAM peak                =   512,000,000,000 bytes/s
TP4 DRAM peak                = 2,048,000,000,000 bytes/s
prefill percentage = 100 * useful logical layer FLOPs / (peak * whole-layer seconds)
decode percentage  = 100 * estimated mesh DRAM bytes / (DRAM peak * whole-layer seconds)
```

The 120-core value is correct for the advertised theoretical silicon peak of
this hardware. The observed 110 available workers describe the schedulable
compute grid. Changing to 110 is an alternative reference, not a correction to
the physical specification. Name the basis explicitly and apply it identically
to TP1 and TP4. An optional second `available_worker_lofi_reference` can use
110 cores/ASIC; never silently substitute it into historical headline values.
Neither reference is a measured FPU utilization counter.

Keep the complete-layer latency denominator unchanged: first native firmware
start through last firmware end on each ASIC, including every operation and
intra-layer gap; mesh duration is the maximum of those per-device spans. Decode
uses the mean of complete replay windows. Do not use a sum of selected matmul
times, average per-operation percentages, or subtract zero-fill, routing,
collectives, idle workers, or fidelity overhead. Useful FLOPs remain one logical
model layer, not four copies of that workload.

## Hardware, frequency, and arithmetic evidence

- [Recorded device discovery](final_device_list.log) identifies four PCI device
  IDs as `p300c`, grouped into two board serial numbers. The official
  [p300c specification](https://docs.tenstorrent.com/aibs/blackhole/p300.html),
  checked on 2026-09-26, specifies two ASICs per card, 120 Tensix cores and 32 GB
  per ASIC, a 1.35 GHz AI clock, and 1,024 GB/s per card. Thus four participating
  ASICs means two cards; do not multiply the card bandwidth by four.
- The native CSV
  `profile_v0/reports/tp4/2026_09_26_20_28_12/ops_perf_results_tp4_2026_09_26_20_28_12.csv`
  contains 37,632 device rows with `AVAILABLE WORKER CORE COUNT=110` and
  `DEVICE ARCH=blackhole`; there are 9,408 duration-bearing rows per device.
  Every device has median `DEVICE FW DURATION [ns] / (END CYCLE - START CYCLE)`
  equal to `0.7407407407407407`, corresponding to the profiler's 1,350 MHz
  conversion. This is recorded clock-conversion evidence, not an independent
  measurement of instantaneous DVFS behavior.
- `tt_metal/impl/profiler/profiler_analysis.cpp:92` obtains the available worker
  count from program/subdevice availability or the compute-grid area. At line
  351, Blackhole duration conversion uses `get_device_aiclk(device_id)`.
  `tt_metal/core_descriptors/blackhole_140_arch.yaml:64` describes the
  two-column-harvested compute grid `[0,0]..[10,9]`, with separate dispatch-core
  coordinates. A 110-worker application grid is consistent with reserving a
  Tensix column from 120 functional cores; the profiler field does not purport
  to count all silicon cores. The optimized TP1 layer-0 v8 native CSV also
  reports 110 workers for all 38,944 device rows.
- `ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2773`
  derives the matrix arithmetic rate as `2 * 8 * 16 * 16 = 4096` operations per
  core per cycle and counts an FMA as two operations. Its ideal-cycle formula
  multiplies by math-fidelity phases. The independent SDPA model at
  `ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_perf_model.cpp:13`
  uses the same convention.
- The installed `tt-perf-report` 1.3.0 source at
  `python_env/lib/python3.10/site-packages/tt_perf_report/perf_report.py:398`
  uses Blackhole phase divisors LoFi/HiFi2/HiFi3/HiFi4 = 1/2/3/4, a 110-worker
  default, 512 GB/s, and 1.35 GHz. Its per-operation arithmetic uses the
  operation's selected core count and fidelity. Such an operation percentage
  has a different denominator from the complete-layer physical-ASIC reference.

| Reference | One ASIC, TFLOP/s | Four ASICs, TFLOP/s |
| --- | ---: | ---: |
| Physical 120 cores, LoFi | 663.552 | 2,654.208 |
| Available 110 workers, LoFi | 608.256 | 2,433.024 |
| Physical 120 cores, HiFi2 | 331.776 | 1,327.104 |
| Physical 120 cores, HiFi3 | 221.184 | 884.736 |
| Physical 120 cores, HiFi4 | 165.888 | 663.552 |

BFP8 and BFP4 are storage formats here, not separate double/quadruple matrix
rates. Under the same LoFi program both use the 4,096-operation reference.
The Blackhole LLK matmul MOP in
`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h:434` selects the
number of inner fidelity phases from `math_fidelity`; it does not double that
rate because a weight occupies four bits. Storage does change traffic:
`tt_metal/impl/data_format/tile.cpp:70` gives a 32x32 BFP8 tile as 1,088 bytes
and BFP4 as 576 bytes, including 64 exponent bytes; BF16 is 2,048 bytes.

The runtime mixes LoFi experts with attention, router, projection and SFPU
work. A common LoFi roof is a deliberately stringent, comparable useful-work
reference. Applying HiFi4's divisor four to the whole layer would raise the
percentage fourfold without making its LoFi expert workload faster. Any future
phase-weighted efficiency estimate must be separately named and source each
term's fidelity; it must retain the complete-layer elapsed time.

## Comparison with the baseline summarizer

`tests/summarize_multichip_perf.py:109` hardcodes the physical 120-core LoFi
reference above. `tests/summarize_perf.py:350` uses the same per-ASIC formula
divided by `--peak-fidelity-cycles`; that argument defaults to **4** at line
245. The selected optimized v8 layer-0 and layer-5 `whole_layer.json` artifacts
both record divisor **1**, so those selected profiles and the TP4 summarizer
already share the same reference. Invoking the baseline script with defaults
would produce an incompatible HiFi4 reference; explicitly use
`--peak-fidelity-cycles 1` for this comparison. Older fused artifacts using
divisor four must be identified by their stored basis, not assumed comparable.

Switching both reports to 110 available workers would multiply each compute
percentage by `120/110 = 1.090909...`, leaving latency, useful FLOPs, DRAM
percentages, and any TP1/TP4 speed ratio unchanged. Examples from existing
artifacts, with no new measurement:

| Existing profile | Physical-core prefill % | Available-worker reference % |
| --- | ---: | ---: |
| Historical TP4 `profile_v0` sliding | 0.0402733525 | 0.0439345664 |
| Optimized TP1 v8 sliding | 0.6007866160 | 0.6554035811 |
| Optimized TP1 v8 full | 0.9817981770 | 1.0710525568 |

Core-count choice therefore does not explain the candidate's performance gap.
These examples are accounting illustrations across existing source versions,
not a matched final-candidate performance comparison.

## Decode DRAM traffic: TP4 versus EP4

For batch-one decode with eight distinct active experts, expert weight traffic
can be calculated exactly under the summarizer's one-read-per-selected-weight
convention. This is an operand-byte estimate, not DRAM controller traffic.

`models/demos/gemma4/tt/experts/weights.py:73` pads expert width 704 to 768 for
TP4, giving 192 per rank. Each rank holds all 128 expert shards and executes the
same eight selected IDs. `tt/optimized_decoder.py:194` uses compact indexed
decode with eight output slots. The EP class in `tt/multichip_decoder.py:97`
owns 32 full-width 704 experts per rank and gathers that rank's routing mask.
At line 194 it omits both `nnz` and indices, allowing 0..8 local active experts.
The non-indexed weight reader skips inactive groups before loading weights:
`ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp:300`.

Let `g` be gate/up tile bytes (1,088 sliding, 576 full), with down tile bytes
576. Hidden width is 88 tiles, TP local width six tiles, and EP width 22 tiles:

```text
TP4 mesh selected weight bytes = 4 * 8 * 88 * 6  * (2*g + 576)
EP4 mesh selected weight bytes =     8 * 88 * 22 * (2*g + 576)
EP4 rank r selected bytes      =   n_r * 88 * 22 * (2*g + 576)
0 <= n_r <= 8; sum_r(n_r) = 8
```

| Layer kind | TP4 mesh bytes | EP4 mesh bytes | Reduction | TP4 per-rank bytes |
| --- | ---: | ---: | ---: | ---: |
| Sliding, BFP8 gate/up + BFP4 down | 46,497,792 | 42,622,976 | 3,874,816 (8.3333%) | 11,624,448 |
| Full, BFP4 gate/up + BFP4 down | 29,196,288 | 26,763,264 | 2,433,024 (8.3333%) | 7,299,072 |

EP has the same aggregate useful active-expert weight bytes as unpadded TP1.
Executing roughly two experts per EP rank instead of eight TP shards does
**not** reduce mesh weight bytes fourfold: each EP expert is much wider. The
sole aggregate saving in this term is removal of 768-to-704 padding.

| EP local active count | Sliding rank bytes | Full rank bytes |
| --- | ---: | ---: |
| 0 | 0 | 0 |
| 1 | 5,327,872 | 3,345,408 |
| 2 | 10,655,744 | 6,690,816 |
| 8 | 42,622,976 | 26,763,264 |

Mixed ownership preserves the mesh sum but changes the critical rank. For
example, `[1,1,1,5]` and `[8,0,0,0]` put more expert weight reads on one rank
than balanced `[2,2,2,2]`. At theoretical 512 GB/s per rank, sliding weight-only
transfer floors are 22.704 us for TP4, 20.812 us for balanced EP4 and 83.248 us
for all-eight-on-one-rank EP4. These are arithmetic lower bounds for this one
traffic term, not observed times or a prediction of complete-layer latency.
The aggregate whole-layer roofline must still use four ASICs and the measured
maximum rank duration; idle ranks are part of the selected topology's cost.

Do not present the weight-byte delta as the final whole-layer traffic delta.
EP adds local route gathering and processes expanded 32-slot outputs versus
eight TP compact slots. Both candidate sparse decode outputs and the mix are
explicitly L1. The expanded physical BF16 gate/up and down output buffers per
rank are respectively 2,883,584 and 5,767,168 bytes (TP: 196,608 and 1,441,792).
Those sizes are not DRAM transfers merely because the buffers are larger.
`sparse_matmul_device_operation.cpp:451` zero-fills auto-created outputs in
both modes; EP's larger zero-fill/SFPU/reduction work remains in elapsed time.
Conversions, actual memory placement, duplicated inputs/metadata, native
attention cache reads, and collective operands require the final native CSV.
The current traffic convention also excludes extra per-core rereads, internal
CCL/NoC passes and profiler traffic. No final TP4-versus-EP4 complete-layer
DRAM byte total can be inferred from `profile_v0`.

The concurrently added `_HybridExperts` wrapper dispatches single-token decode
to indexed TP experts and prefill to EP experts. Its decode weight-read estimate
therefore uses the TP4 column above; keeping EP weights resident does not imply
reading those weights during a TP decode. This source observation does not
establish any hybrid performance or correctness result.

## Requirements for future target-profile accounting

`tests/summarize_perf.py:137` currently rejects non-indexed sparse metadata
without a numeric `nnz`; the dynamic EP candidate correctly supplies
`nnz=None`. Therefore the existing summarizer cannot yet produce a valid EP
traffic report. Preserve that fail-closed behavior until an explicit accounting
method is added; do not substitute 8 or 32 on every EP rank. The current helper
also rejects zero, which a per-rank EP accounting override must allow.

For batch-one, top-eight decode, either provide actual local counts for each
replay and rank, or label a mesh-aggregate invariant estimate that substitutes
exactly eight full-width expert weight groups across the four ranks for each
gate/up and down operation. Check all four ownership partitions and dimensions;
retain all other native DRAM operand counts unchanged. A global invariant does
not recover a per-rank traffic breakdown or prove load balance. Numeric
nonzero counts should reflect the actual mask if selected router weights can
become zero; indexed TP executes its eight listed IDs regardless. Report that
assumption rather than claiming to have read dynamic masks from the CSV.

Prefill is different: sparse work uses each 32-token chunk's union of selected
experts. Neither local count eight nor global count eight is a valid prefill
union count. Useful prefill FLOPs continue to count eight experts per logical
token, while all extra union work remains in the complete-layer denominator.

Audited source SHA-256:

```text
tt/multichip_decoder.py
3f3fc2ed3ad8ff21bcc19cb37519438187dbac5fd1e4cfdcfcf6aacd66504a20
tests/summarize_multichip_perf.py
ebedbf90fff42733a4dd6843db3f1e09ffe39fa34d27eda6430b615307119de1
tests/summarize_perf.py
8beb5082231274b8a8b8f1a7ba6212471b37f0bc1e615a9b484b54b43706381a
```

Validation was limited to source inspection, native CSV counts, stored JSON
bases, and independently evaluated byte/peak arithmetic. Only this report was
written for this audit.
