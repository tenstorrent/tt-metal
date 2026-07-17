# Llama 3.1 70B optimized multichip decoder

## Result

`tt/multichip_decoder.py` is the decoder-layer stack baseline for
`meta-llama/Llama-3.1-70B-Instruct` on the complete local four-Blackhole p300c
mesh. `MultiChipDecoder` subclasses the completed `OptimizedDecoder`, exposes
it as `single_chip_baseline`, and contains a real TP4 implementation with no
PyTorch/host runtime fallback. This stage does not construct a full model or a
vLLM path.

The selected hardware view is a logical `1x4` ring over the discovered
four-chip physical cycle, with `FABRIC_1D_RING`, two links, BF16 collectives,
and mesh-shared trace-safe CCL semaphores. Projection weights use the optimized
baseline's BFP4/LoFi policy; activations, norms, collectives, and KV cache are
BF16. Llama 3.1 70B has one dense decoder-layer kind and no MoE experts.

## Selected tensor and kernel plan

Global dimensions are hidden 8192, 64 Q heads, 8 KV heads, head dimension 128,
and intermediate 28672. All model dimensions divide by TP4, so semantic
weight/cache padding is unnecessary. Batch-1 decode is physically tile-padded
to 32 rows inside TTNN; prefill MLP work is internally chunked at 4096 logical
tokens with an arbitrary final tail. Neither padding is visible in the public
sequence contract.

| Tensor | Global logical shape | TP4 mapping | Per-device logical shape |
| --- | --- | --- | --- |
| residual/input/output | `[1,B,S,8192]` | replicated | `[1,B,S,8192]` |
| RMSNorm weights | `[8192]` | replicated | `[8192]` |
| packed QKV weight | `[8192,10240]` | output columns, grouped Q/K/V | `[8192,2560]` |
| Q/K/V activation | `[B,64/8/8,S,128]` | heads | `[B,16/2/2,S,128]` |
| output weight | `[8192,8192]` | input rows | `[2048,8192]` |
| packed gate/up weight | `[8192,57344]` | output columns, local gate/up pairs | `[8192,14336]` |
| down weight | `[28672,8192]` | input rows | `[7168,8192]` |
| paged K/V cache | `[blocks,8,64,128]` | KV heads | `[blocks,2,64,128]` |
| page table/current position/RoPE index | sequence dependent | replicated | unchanged |

Weights remain DRAM width-sharded across eight banks, and the `8x8` prefill
grid lets prefill and decode share the same physical weight tensors. Selected
batch-1 decode activation shards and program geometry are:

| Projection | Padded activation | Width shards | `in0_block_w` | `per_core_N` |
| --- | --- | ---: | ---: | ---: |
| QKV | `[1,1,32,8192]` | 16 cores, 512 values/core | 16 | 5 |
| output | `[1,1,32,2048]` | 8 cores, 256 values/core | 8 | 8 |
| packed gate/up | `[1,1,32,8192]` | 32 cores, 256 values/core | 8 | 14 |
| down | `[1,1,32,7168]` | 32 cores, 224 values/core | 7 | 8 |

The output and gate/up settings are the measured O2 and G1 winners. Ten
one-variable real-weight candidates are retained as `geometry_*.xml`; all
passed PCC >= 0.999855. O2 improved eager decode from 1.521315 to 1.451859 ms,
and G1 from 1.537236 to 1.470207 ms in their paired runs. Other apparent trace
wins either regressed eager latency or were within run-to-run noise.

Packed QKV and gate/up are column parallel and require no reduction. Output
and down are row parallel. Each decode reduction is explicitly decomposed into
`reduce_scatter_minimal_async` plus `all_gather_async` using persistent L1
full/intermediate/shard buffers; the output returns to the replicated
`[1,B,1,8192]` stack boundary. Prefill uses the composite all-reduce because
its logical sequence and final chunk sizes vary.

The persistent candidate improved paired trace replay from 0.620704 to
0.603138 ms and eager decode from 1.471944 to 1.451630 ms at PCC 1.0. BF8,
one-link, and linear variants were rejected: BF8 regressed eager time;
one-link and linear regressed trace time. Persistent BF8 reached 0.601112 ms,
only 0.34% beyond persistent BF16 in that pair, while regressing eager decode
to 1.614094 ms, so BF16 was retained.

## Alternatives and topology audit

`Provenance2DDecoder` retains the compiler-derived `2x2` implementation. It
measured 1.410 ms prefill and 1.859 ms decode, versus 3.187/1.853 ms for its
paired optimized baseline. Its hidden-sharded residual needs projection
reductions and distributed-norm statistic gathers, making it unsuitable for
single-token stack latency. A flat collective on the logical `2x2` ordering
was also rejected: row-major ranks 1 and 2 are not direct fabric neighbors.
The logical `1x4` ordering follows the actual physical cycle.

A non-fused hidden-sharded flat path does not reduce raw large-payload wire
volume: it replaces the current two all-reduces with two activation
all-gathers plus two result reduce-scatters and adds two norm-statistic
all-gathers. It is therefore source-dominated by the selected persistent
replicated boundary. The only credible lower-movement form fuses those
collectives into the neighboring matmuls. The current fused
all-gather-matmul validator does not accept this path's DRAM-sharded matmul
program configuration; the matmul-reduce-scatter family has incompatible
layout/program constraints, and the minimal fused Blackhole path has an open
nondeterminism blocker (#46181). Repacking weights and changing the kernel
family without a correctness-stable fused primitive was rejected for this
stage.

Pipeline parallelism, sequence parallelism for one-token decode, KV-head
replication, and smaller meshes were rejected. Dense all-expert execution is
inapplicable because this model is not MoE. The full decision record and the
pre-code frozen plan are in `mesh_plan.md`.

## Correctness, cache, stack, and trace evidence

The canonical real-weight run is `final.xml`/`final.log`. It uses layer 39,
batch 1, nonaligned logical prefill length 39, internal MLP chunk 32 plus a
seven-token tail, positions 39/40 and 64, and the completed single-chip TTNN
optimized decoder as reference.

| Check | PCC versus optimized TTNN baseline |
| --- | ---: |
| nonaligned prefill output | 0.9999978879 |
| decode output | 0.9996807609 |
| contiguous key/value cache | 0.9999931706 / 0.9999924131 |
| nonidentity paged decode | 0.9996794316 |
| two-layer direct stack composition | 0.9984936517 |
| advancing trace outputs, positions 39/40 | 0.9996876997 / 0.9997024905 |
| advancing trace K/V writes | >= 0.9998138782 |
| dynamic page-table position-64 trace | 0.9999930420 |
| position-64 physical K/V writes | 0.9998256458 / 0.9998141212 |

The page table is first `[[1,0]]`, so logical page zero occupies physical block
one, then is refreshed in place to `[[0,1]]` before a captured replay at the
64-token boundary. Hidden input, current-position tensor, RoPE-index tensor,
and page-table contents all change without recapture. Both position tensors
advance on device. Cache validation reconstructs the global eight heads from
the four local two-head shards.

The stack check runs the real layer twice with independent K/V caches and
feeds layer-zero TP4 output directly into layer one, with no host reshape,
gather, or conversion. The output remains replicated `[1,B,1,8192]`. The
capacity test also allocates the exact advertised-context local K and V shapes
`[2048,2,64,128]` on every device.

`watcher_final.xml`/`watcher_final.log` contain a clean mesh-only Tensix
watcher run over 17 eager positions, the page boundary, and 100 advancing
trace replays with persistent CCL buffers. ETH watcher instrumentation is
disabled because the instrumented Blackhole fabric-router binary exceeds its
kernel-config buffer (27,920 versus 25,600 bytes); watcher remains attached to
and checks Tensix cores on devices 0--3.

## Context and peak-memory contract

`doc/context_contract.json` preserves the full Hugging Face 131072-token
context at batch 1. Per-device full-stack accounting is:

| Component | Bytes/device |
| --- | ---: |
| 80 decoder layers, one shared BFP4 weight copy | 9,628,549,120 |
| sharded embedding plus LM-head reserve | 295,501,824 |
| 80-layer BF16 K/V cache, two local heads | 10,737,418,240 |
| persistent subtotal | 20,661,469,184 |
| conservative peak live explicit activations | 10,737,418,240 |
| shared constants/metadata reserve | 134,217,728 |
| trace region reserve | 100,000,000 |
| persistent CCL/RS reserve | 536,870,912 |
| allocator/fragmentation reserve | 1,073,741,824 |
| total reserved peak | 33,243,717,888 |
| observed allocator capacity | 34,178,731,008 |
| remaining peak margin | 935,013,120 |

An unchunked full-context MLP has a conservative 23,689,428,992-byte peak,
which exceeds the 13,517,261,824-byte transient headroom. The 4096-token
internal chunks bound MLP temporaries; explicit deallocation after each last
consumer and final concatenation bound the conservative explicit peak to five
full hidden BF16 tensors. This is a peak lifetime proof, not an additive sum
of every operation output. The maximum validated advertised-context batch is
one; larger batches trade batch for context rather than reducing batch-1
capability.

## Warmed performance and profiler review

Canonical wall timings from `final.xml` use the same real layer, batch 1,
logical length 39, dtype policy, warmups, and synchronized timing. The prefill
test intentionally forces a 32+7 chunk split; production default is 4096.

| Path | Optimized single chip | TP4 | Speedup | Four-device efficiency |
| --- | ---: | ---: | ---: | ---: |
| prefill eager | 3.710810 ms | 1.653101 ms | 2.244757x | 56.1189% |
| decode eager | 1.850049 ms | 1.556849 ms | 1.188330x | 29.7082% |
| decode trace replay | 1.844675 ms | 0.598285 ms | 3.083272x | 77.0818% |

The single-chip baseline trace is its unchanged fixed-position graph. The
TP4 trace is the production advancing-position graph and includes on-device
RoPE lookup, K/V update, and position increments. Trace replay is the layer
stack headline; eager timing is retained separately rather than mixed into
the trace result.

The clean final profiler capture is under `tracy_final/`. Its trace-decode
signpost contains five replays and 230 device ops: 2,938 us device time, 257 us
aggregate op-to-op gap, 182 GB/s aggregate DRAM observation (35.5% roofline),
or 587.6 us device time per replay. Only 5 us (0.2%) is identified as
trace-removable gap. Per replay, gate/up is about 204 us, down 108 us, QKV
41 us, output 35 us, each reduce-scatter 21--23 us, and each all-gather
12--14 us. The two RS+AG pairs are about 12.3% of device time; matmuls remain
the dominant work.

The reduced prefill signpost has 26 device ops and 1,018 us device time, with
109 GB/s aggregate DRAM observation (21.2%). Its host gaps include profiler
harness/capture setup, so they are not used as latency; the synchronized
`final.xml` wall timing is authoritative. `tt-perf-report` marks the four
matmuls slow and suggests L1 inputs/output subblocks. The retained geometry
sweep tested legal DRAM-sharded alternatives and selected O2/G1; moving
full-context inputs to L1 is not physically viable.

## Reproduction

```bash
export LLAMA_31_70B_REAL_WEIGHT_FILE=/home/mvasiljevic/hf-cache/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b/model-00015-of-00030.safetensors

flock /tmp/tt-device.lock env TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
  pytest -sv models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_multichip_decoder.py \
  --junitxml=models/autoports/meta_llama_llama_3_1_70b_instruct/doc/multichip_decoder/final.xml

flock /tmp/tt-device.lock env RUN_MULTICHIP_DECODER_WATCHER=1 \
  TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
  TT_METAL_OPERATION_TIMEOUT_SECONDS=10 TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
  pytest -sv models/autoports/meta_llama_llama_3_1_70b_instruct/tests/test_multichip_decoder.py \
  -k watcher_stress \
  --junitxml=models/autoports/meta_llama_llama_3_1_70b_instruct/doc/multichip_decoder/watcher_final.xml
```

The profiler command and exact `tt-perf-report` render commands are recorded
in `work_log.md`.

## Canonical artifacts and limitations

- `final.xml`, `final.log`: final correctness, PCC, stack, paging, trace,
  capacity allocation, eager latency, and trace latency.
- `watcher_final.xml`, `watcher_final.log`: clean 100-replay watcher stress.
- `geometry_*.xml`, `topology_*.xml`: isolated real-weight selection evidence.
- `tracy_final/reports/2026_07_17_15_18_41/ops_perf_results_2026_07_17_15_18_41.csv.gz`:
  losslessly compressed merged operation-level profiler provenance. Expand
  with `gzip -dk` before rerendering reports.
- `tracy_final/final_{prefill,trace_decode}_table.txt`: human-readable reports
  with advice; adjacent report and summary CSV/PNG files are retained.
- `AUTODEBUG.md`, `STAGE_REVIEW_INITIAL.md`: failed fresh-debug bootstrap and
  initial independent review findings that drove remediation.
- `STAGE_REVIEW_FINAL.md`: fresh independent `clean-pass`, with no required
  work and a controlled anomaly ledger.
- `mesh_plan.md`, `work_log.md`: plan, alternatives, commands, recovery, and
  commit record.

The supported target is exactly the four local Blackhole devices as a logical
1x4 ring. Smaller/differently wired meshes are outside the contract. The
root-owned `generated/inspector` directory prevents Inspector logging, so
commands use its documented nonfatal mode. Firmware 19.8.0 is newer than the
runtime's latest fully tested 19.5.0, the B850M-C board uses the bus-ID topology
fallback, and `/dev/shm` reports low headroom; none caused a final gate failure.
