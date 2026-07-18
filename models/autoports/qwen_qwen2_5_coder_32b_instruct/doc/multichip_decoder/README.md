# Qwen2.5-Coder-32B multichip decoder

This stage targets the complete four-device mesh available on this machine and
stops at one production-shaped decoder layer. It does not assemble a full model
or add vLLM integration.

## Pre-implementation mesh decision

- Baseline: `tt/optimized_decoder.py`, specifically its selected batch-32
  BFP8/HiFi2 packed-projection, BFP8-cache, traced-decode contract.
- Hardware: four Blackhole p300c devices, fixed `MeshShape(1,4)`, TP=4 on mesh
  axis 1, `FABRIC_1D_RING`, Ring topology, and two links.
- Model: 64 structurally identical dense Qwen2 decoder layers. There is no MoE
  router or expert tensor, so expert parallelism and active-expert execution are
  not applicable.
- Layer boundary: BF16 hidden-width fracture, `[1,32,L,1280]` per rank. A
  following decoder consumes this directly. Gathering for comparison belongs
  only in the test fixture.
- Attention: full hidden input is gathered for RMSNorm/QKV, then each rank owns
  10 query heads and 2 KV heads. O is row parallel and reduce-scatters back to
  the local residual.
- MLP: full hidden input is gathered for RMSNorm and packed gate/up; each rank
  owns one padded intermediate shard. Down is row parallel and reduce-scatters
  back to the local residual.
- KV cache: local-head contiguous or paged cache on each rank. Page tables,
  positions, RoPE tables, and persistent CCL scratch are stable replicated
  device tensors; layer-independent state is shared by the sequential stack.
- Prefill and decode accept logical lengths independently of tile, page, CCL,
  and internal chunk alignment. Padding is internal and sliced or made
  mathematically inert before a model-visible boundary.

The source compiler provenance used TP=4 with replicated residuals and two
all-reduces. That is the correctness prior. The selected final family replaces
each all-reduce boundary with a stack-compatible hidden fracture: two hidden
all-gathers and two row-parallel reduce-scatters per layer, with no layer-boundary
gather. The provenance family, fused CCL/matmul families, and CCL dtype variants
remain measurement candidates rather than assumptions.

## Calculated tensor and padding plan

For prefill, `M=32*logical_sequence_length`; for decode, `M=32`.

| Tensor/activation | Dense logical shape | Per-device physical shape | Mesh/layout plan |
|---|---:|---:|---|
| Layer input/output | `[M,5120]` | `[M,1280]` | hidden-width TP fracture, BF16 |
| RMSNorm weights | `[5120]` | `[5120]` | replicated BF16 |
| Packed QKV weight | `[5120,7168]` | `[5120,2048]` | rank-packed Q10/K2/V2 width `1792`, then 256 zero channels; column TP |
| Packed QKV bias | `[7168]` | `[2048]` | same rank packing and zero tail as QKV |
| O weight | `[5120,5120]` | `[1280,5120]` | row TP over local query heads |
| Packed gate/up weight | two `[5120,27648]` | `[5120,14336]` | per rank gate `6912→7168` plus up `6912→7168`; column TP |
| Down weight | `[27648,5120]` | `[7168,5120]` | matching 256-row zero tail per TP rank; row TP |
| Q heads | `[32,40,1,128]` | logical `[32,10,1,128]` | local heads; device-only padding for decode SDPA |
| K/V cache, each | `[32,8,S,128]` | `[32,2,S,128]` | KV-head TP, BFP8, contiguous |
| Paged K/V, each | `[blocks,8,64,128]` | `[blocks,2,64,128]` | KV-head TP; replicated page table |
| RoPE tables | four `[1,1,S,128]` tables | one stack-shared set/device | TILE + ROW_MAJOR cos/sin, replicated |
| Positions/page table | model logical | one stack-shared set/device | replicated int32/uint32 stable trace inputs |

The 256-channel QKV and MLP tails are chosen before final tuning because they
change the viable program geometry. Without padding, local QKV/MLP tile counts
`56` and `216` share only eight exact workers with hidden width `160` tiles.
The first construction probe rejected `1920/7040`: those 20-core-friendly widths
are not divisible across eight Blackhole DRAM banks. Padding to `64` and `224`
tiles (`2048/7168`) is the smallest jointly DRAM-bank-legal choice and makes
16-core QKV, packed gate/up, and down programs exact.
The MLP tail remains present through packed gate/up, SiLU/multiply, and a
zero-padded down weight, so it contributes exactly zero to the residual. The
QKV tail is sliced before head creation.

## Row-parallel topology candidates fixed for measurement

| Family | Residual before/after | Collective bytes at decode, per rank | Next consumer | Persistent plan | Pre-coding expectation |
|---|---|---:|---|---|---|
| Local matmul + provenance all-reduce | full/full | RS `327680→81920`, AG `81920→327680` twice | local RMSNorm | non-persistent control | correctness prior; extra gather at both sublayers |
| Local matmul + RS, delayed AG | local/local | AG `81920→327680`, RS reverse, twice | next distributed boundary | ping-pong AG/RS buffers | selected starting family; no stack boundary conversion |
| Fused matmul + RS | local/local | fused row projection/RS | residual add | preallocated fused intermediates | measure after shape/rank adaptation |
| Fused AG + matmul | local to partial/full | fused AG consumes local input | O/down or next projection | persistent output where supported | measure with locally packed output weights |
| Residual-sharded distributed norm | local/local | distributed-stat reduction rather than hidden AG | QKV/gate-up needs full K | common RMSNorm primitives | likely loses because both column projections require full hidden K, but requires a shape-faithful probe |

CCL payload starts at BF16 to match the optimized residual contract. BFP8 CCL
is a coupled topology/precision candidate. All repeated decode collectives will
use common semaphore ownership and rotating persistent buffers where the API
supports them.

## Rejected strategy classes before implementation

- TP=2 leaves half the detected hardware idle and doubles per-rank weights and
  KV ownership.
- Replicated layers duplicate the 32B model and do not reduce single-layer
  latency.
- Data parallelism optimizes throughput, not this single-model latency goal.
- Sequence parallelism does not accelerate single-token decode and does not
  solve projection weight ownership.
- KV/head-only sharding leaves the dense MLP and projection weights dominant.
- 2D TP/EP is inapplicable to a physical 1x4 mesh and a dense non-MoE layer.

## Selected implementation

`tt/multichip_decoder.py` subclasses `OptimizedDecoder` and owns its complete
TP4 forward path. The selected fixed configuration is:

- attention and KV cache BFP8, HiFi2; MLP gate/up/down BFP4, LoFi; BF16
  residuals, norms, CCL payloads, biases, and RoPE;
- 16-core QKV, gate, up, and down decode programs, an independently tuned
  8-core O program, and a 20-core local residual layout;
- one stack-shared persistent ping-pong decode workspace for two all-gathers
  and two reduce-scatters, with two Ring links;
- one stack-shared `SharedRotaryTables` set and public fixed-address position
  workspace; and
- packed prefill QKV and gate/up. Decode gate/up remains split because the
  packed output requested 2,094,848 bytes of static CB space versus the
  1,572,864-byte Blackhole L1 limit.

There is no host or single-chip forward fallback. Torch is used only while
loading/padding weights, building RoPE/cache/control tensors, and in tests.
Prefill/decode math, cache mutation, head transforms, and collectives remain on
device. Fixed-address current-position, RoPE, and page-table buffers are
updated in place for trace replay.

## Correctness and layout contracts

The real-weight gate uses an independently captured
`OptimizedDecoder/advisor_packed_bfp8_hifi2_1d` result for homogeneous dense
layer 32. Qwen2.5-Coder-32B has no MoE layer kind.

| Real layer-32 comparison | PCC |
| --- | ---: |
| prefill output, length 17 | 0.993392 |
| prefill K / V cache | 0.999838 / 0.999662 |
| decode output, position 17 | 0.994006 |
| decode K / V update | 0.999910 / 0.999814 |

The synthetic HF-reference test exercises logical length 31, deliberately not
aligned to tile or page size. Selected-policy PCC is 0.996552 for prefill,
0.996447 for decode, and 0.991185 after passing the fractured output directly
into a second decoder. A separate two-instance sequential decode reuses the
same RoPE, position, and CCL workspace and reaches output PCC 0.990739 with
second-layer K/V PCC above 0.9962. First-layer K/V PCC is at least 0.999782.
Paged and contiguous
outputs and logical caches are PCC 1.0. The public contract accepts every
logical length in range; tile/page/chunk padding remains internal.

Trace capture/replay passes for contiguous and paged decode. Ten identical
replays are bitwise equal. In-place position refresh from 32 to 33 retains
output PCC at least 0.996442 and cache PCC at least 0.999832 against HF. A
separate paged test remaps every user's unused second physical page while
retaining the page-table address, then advances position 64 to 65; eager versus
traced output and K/V PCC are all 1.0. These checks validate local cache shape
`[32,2,S,128]`, page-table semantics, current positions, and stacked
`[1,32,L,1280]` boundaries. Evidence is in `synthetic_correctness.json`,
`paged_trace_refresh.json`, and `real_baseline_pcc.json`.

## Topology, precision, and graph-rewrite measurements

All timing candidates use seven warmed prefill trials and seven groups of 100
trace replays unless stated otherwise.

| Candidate | Prefill ms | Decode ms | Decision |
| --- | ---: | ---: | --- |
| all BFP8/HiFi2, 16 cores | 3.408 | 0.972 | slower |
| mixed MLP BFP4/LoFi, 16 cores | 3.346 | 0.796 | retained family |
| mixed, O on 8 cores | 3.342 | 0.792 | selected geometry |
| mixed, all 32-core role grid | 3.374 | 0.796 | no improvement |
| mixed, down on 32 cores | 3.237 | 0.797 | prefill improves, decode regresses |
| BFP8 CCL | 3.355 | 0.808 | slower and lower PCC |
| non-persistent decode CCL | 3.405 | 0.805 | slower |
| fused matmul + reduce-scatter | 3.354 | 0.912 | correct, decode slower |
| all BFP4/LoFi | 3.309 | 0.766 | rejected: prefill PCC 0.989796 is below 0.99 |
| all BFP4, attention HiFi2 | 3.280 | 0.790 | rejected: prefill PCC 0.989919 is below 0.99 |

The final reproduced selected run is 3.344 ms prefill and 0.791826 ms traced
decode, with PCC 0.993392/0.994006 and bitwise-equal repeated trace output.
Candidate JSON files retain every sample, mesh plan, and PCC. All ten were
regenerated after the final `2048/7168` physical padding decisions, with the
same seven prefill trials and seven groups of 100 decode replays.

The independent rereview then identified prefill program geometry as a missing
sweep dimension. Controls for the prefill grid and K block were added and the
selected BFP4/LoFi graph was isolated without changing decode. The primary
protocol remained seven synchronized warmed prefill samples; the closest two
10x10 candidates were then repeated with 21 samples:

| Prefill grid / block limit | Primary median ms | 21-trial median ms | Decision |
| --- | ---: | ---: | --- |
| 10x10 / 10 | 3.323 | 3.234 | selected |
| 10x10 / 16 | 3.289 | 3.295 | rejected: 1.90% slower in confirmation |
| 10x10 / 20 | 3.381 | — | rejected: slower |
| 10x10 / 32 | — | — | rejected: packed gate/up CB 2,216,704 B > 1,572,864 B L1 |
| 8x10 / 10 | 3.490 | — | rejected: slower despite exact N tiling |
| 8x10 / 16 | 3.561 | — | rejected: slower |
| 8x10 / 20 | — | — | rejected: packed gate/up CB 1,794,816 B > L1 |
| 8x10 / 32 | — | — | rejected: packed gate/up CB 2,667,264 B > L1 |

The shape-aligned 8x10 grid does produce 1x4 QKV and packed-gate/up output
subblocks, versus 1x1 and 1x3 on 10x10, but losing 20 workers outweighs that
tiling improvement. The selected 10x10/block-10 program therefore remains the
same path captured by the retained profiler. Raw and aggregate evidence is in
`candidate_prefill_*.json` and `prefill_geometry_sweep.json`.

The compiler-provenance replicated-boundary/two-all-reduce family is
numerically identical to the selected fractured family for the measured
decode (output and K/V PCC 1.0). Under the final fair 7x100 protocol, warmed
eager measures 2.0533 ms selected versus 2.0375 ms provenance. Both families
also use persistent buffers and are trace-captured: provenance measures
0.869899 ms versus 0.791554 ms selected, so the selected stack contract is
1.099x faster under the accepted replay regime. Both traces are bitwise stable
and PCC 1.0 against their eager paths. The fused all-gather+column-sharded-O
shape was also executed. All
24 global K-rank weight permutations were tested; the best PCC was 0.24946,
because the Ring gather order is output-rank-relative while the shared mesh
weight permits only one global K packing. It was rejected before latency
acceptance. See `topology_family_benchmark.json` and
`fused_all_gather_o_probe.json`.

Distributed RMSNorm cannot remove a selected collective in this graph: both
QKV and gate/up keep dense K=5120, so each still needs the full-hidden gather.
It would replace the profiled 7-us local norm with pre/post norm plus a
statistics collective while leaving the 11/12-us hidden gathers in place.
That topology is rejected by graph dependency and the measured cost bound.

The local shard advisor was attempted before hand tuning. Its current
environment cannot import because `libTTMLIRRuntime.so` references an undefined
`moe_compute` symbol. The existing optimized-decoder advisor report remains
the single-chip baseline; the directly measured TP4 candidates above are
authoritative. Details are in `shard_advisor_status.md`.

## Performance and profiler acceptance

Both baselines are serialized host-wall measurements on the same p300c
machine. Efficiency is speedup divided by four devices.

| Mode | Single-chip optimized | TP4 selected | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: |
| prefill, batch 32 x length 17 | 9.997 ms | 3.344 ms | 2.990x | 74.7% |
| traced decode, batch 32 | 1.9308 ms | 0.7918 ms | 2.438x | 61.0% |

`tt-perf-report` was captured separately from Watcher. Decode contains 70
device operations and 630 us of merged device work. The main rows are QKV 41
us, O 27 us, gate/up 68 us each, down 67 us, SDPA 22 us, all-gathers 11/12 us,
and reduce-scatters 24/21 us. Modeled DRAM bandwidth is 114 GB/s (22.4% of the
tool's roofline). Prefill contains 222 operations and 2,006 us of merged device
work: QKV 76 us, O 60 us, packed gate/up 356 us, down 185 us, all-gathers
83/82 us, and reduce-scatters 89/90 us. Modeled DRAM bandwidth is 61 GB/s
(11.8%). The compact human tables, raw op CSVs, summary CSV/PNG files, command,
hashes, and provenance are under `tracy/layer32/`.

The merged raw decode table has an 8.25-ms cross-device timestamp skew. It is
not dispatch time: it contradicts both the 630-us merged device work and the
independent 791.7-us trace wall measurement. Communication and the explicit
decode head padding/slicing are the remaining material non-matmul costs.

## Full-stack capacity and context contract

The allocator-faithful 64-layer reservation creates five independent
width-sharded decode projections and four independent interleaved prefill
tensors per layer, including the real packed gate/up allocation. It creates
all 128 K/V buffers and separately padded norm/bias objects at runtime
granularity. Layer-independent RoPE, position/page-table state, and CCL scratch
use the tested stack-sharing contracts; the CCL L1 workspace leaves 1,453,312
contiguous bytes/core. The reservation also includes the 1.6-GB/device trace
region and ownership-aware prefill peak. To avoid advertising a decoder-only
capacity that cannot become a full model, it additionally reserves the future
TP4 BFP8 embedding, untied LM head, and final norm (`438,691,840 B/device`)
without implementing those operations in this stage. Logical length 12,224
fits with 16,593,920 bytes/device free. Length 12,225 pads the cache to 12,288
and fails the adjacent prefill allocation: 200,802,304 bytes/bank are needed
and 191,272,064 are free. At the advertised
32,768 length, local BFP8 K/V alone require 36.507 GB/device, beyond the
34.179-GB physical device capacity.

`doc/context_contract.json` therefore advertises 12,224 as the largest
physically feasible batch-32 context on this fixed mesh. This is an increase
from the prior 3,999-token single-chip contract. It is a hard DRAM limit, not
an alignment restriction; every logical length from 1 through 12,224 remains
valid. The adjacent pass/fail evidence is retained in
`capacity_seq12224.json` and `capacity_seq12225.json`.

## Safety, reproducibility, and limitations

The real layer-32 prefill/decode PCC test is worker/dispatch Watcher-clean.
This is a documented partial-coverage gate exception: full Ethernet Watcher
instrumentation was attempted first and rejected by the runtime because its
27,920-byte active-Ethernet program exceeds the 25,600-byte kernel-config
buffer. The accepted separate run uses `TT_METAL_WATCHER_DISABLE_ETH=1` and
reports no `error`, `assert`, `hang`, `stuck`, or `timeout` signatures. The
exact coverage status, log, and SHA256 are in `watcher_clean.log/json`.
Seven hundred final decode replays plus the explicit bitwise and page-remap
tests provide stress/determinism coverage for the persistent-buffer risk.

This path intentionally supports only the detected four-device 1x4 Blackhole
mesh. It is a decoder-layer stack baseline only: embeddings, final norm,
logits, generation, full-model assembly, and vLLM are outside this stage.
