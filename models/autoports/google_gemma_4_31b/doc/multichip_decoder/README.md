# Gemma 4 31B multichip decoder (Stage 04)

Status: complete; independent stage review verdict `clean-pass`.

## Pre-implementation mesh decision

The target is the four-device Blackhole P150b mesh present on this host,
opened as `MeshShape(1, 4)`.  The selected design is one-dimensional tensor
parallelism with TP=4 and a replicated BF16 residual stream at decoder-layer
input/output boundaries.  QKV and MLP gate/up weights are column parallel;
attention output and MLP down weights are row parallel and reduce across the
four devices.  Attention, RoPE, Q/K norms, paged-cache update, and SDPA operate
on local heads.  Page tables, current positions, RoPE tables, learned RMSNorm
weights, and the layer-boundary residual are replicated.

This design uses `OptimizedDecoder` as the single-chip baseline and preserves
its BFP8 attention, BFP4 MLP, BF16 residual/norm, BFP8 KV-cache, paged-cache,
logical-length, and traced-decode contracts.  Gemma 4 31B is dense
(`enable_moe_block=false`), so expert parallelism and routed-expert placement
are not applicable.

### Calculated tensor plan

| Tensor / operation | Global logical shape | Per-device TP=4 shape | Mesh/layout plan | Padding |
|---|---:|---:|---|---|
| residual input/output | `[1,1,S,5376]` | `[1,1,S,5376]` | replicated BF16; stack-compatible | sequence tile padding stays internal |
| sliding Q | `[5376,8192]` | `[5376,2048]`, 8 heads | column parallel BFP8 | none |
| sliding K/V | `[5376,4096]` each | `[5376,1024]` each, 4 KV heads | packed with local Q | none |
| sliding packed QKV | `[5376,16384]` | `[5376,4096]` | one local packed projection | none |
| full Q | `[5376,16384]` | `[5376,4096]`, 8 heads | column parallel BFP8 | none |
| full K/V (tied source) | `[5376,2048]` each | `[5376,512]` each, 1 KV head | K source duplicated into local K/V lanes | none |
| full packed QKV | `[5376,20480]` | `[5376,5120]` | one local packed projection | none |
| sliding O | `[8192,5376]` | `[2048,5376]` | row parallel, sum collective | none |
| full O | `[16384,5376]` | `[4096,5376]` | row parallel, sum collective | none |
| gate / up | `[5376,21504]` each | `[5376,5376]` each | column parallel BFP4 | none |
| down | `[21504,5376]` | `[5376,5376]` | row parallel BFP4, sum collective | none |
| sliding K or V cache | `[blocks,16,64,256]` global heads | `[blocks,4,64,256]` | local KV heads; one mesh tensor | none |
| full K or V cache | `[blocks,4,64,512]` global heads | `[blocks,1,64,512]` | local KV heads; one mesh tensor | none |
| page table / positions | logical batch contract | same on every device | replicated int32/uint32 | existing padded decode lanes only |

The full-attention packed width includes distinct K and V lanes even though
the HF checkpoint ties their source weight.  This keeps the device head-split
contract unchanged.

### Full-stack memory expectation

Nominal stored-payload accounting (BFP tile/container metadata excluded):

- Sliding layer weights per device: 33,030,144 B attention + 43,352,064 B MLP
  = 76,382,208 B.
- Full layer weights per device: 49,545,216 B attention + 43,352,064 B MLP
  = 92,897,280 B.
- Sixty-layer stack (50 sliding, 10 full), including replicated learned norm
  vectors: approximately 4,750,735,360 B per device.
- Sliding K+V cache per layer at physical context 1,024: 2,097,152 B per
  device; 50 layers use 104,857,600 B.
- Full K+V cache per layer at context 262,144: 268,435,456 B per device;
  10 layers use 2,684,354,560 B.
- The production layer keeps separate large-M interleaved and M=1
  DRAM-sharded weight placements. Duplicate MLP decode copies add
  2,601,123,840 B/device across 60 layers; duplicate attention decode copies
  add 2,146,959,360 B/device. Physical full-stack weights are therefore
  approximately 9,498,818,560 B/device, not merely the 4,750,735,360 B unique
  payload.
- Physical weights plus batch-1 advertised-context KV are 12,288,030,720
  B/device. A conservative 12 GiB reserve covers trace, allocator
  fragmentation, exact-context residuals (2,818,572,288 B each), full-layer
  local Q/QKV intermediates (2,147,483,648 / 2,684,354,560 B), and shared
  full-RoPE tables (536,870,912 B). The accounted total is 25,172,932,608
  B/device, so the advertised 262,144-token context is retained.

### Alternatives to validate or reject

| Topology family | Residual before/after | Collective payload/placement | Next consumer | Initial disposition |
|---|---|---|---|---|
| local row matmul + all-reduce | replicated -> replicated | two hidden-width reductions per layer; BF16 | local RMSNorm | selected correctness path; measure |
| reduce-scatter + delayed all-gather | replicated -> width-fractured | RS after O/down, AG only at next column-parallel input | distributed RMSNorm then QKV/gate-up | measured 1.69x/1.26x slower decode/prefill; rejected |
| fused all-gather + matmul | fractured -> local column output | fuse delayed AG with QKV/gate-up where API supports exact ranks/layouts | local-head attention / local MLP | adapted boundary measured; existing fused API requires incompatible eight-device Ring |
| fused matmul + reduce-scatter | local row input -> fractured residual | remove separate row-output collective launch | distributed norm/residual | Blackhole kernel disabled in-tree for nondeterministic synchronization race |
| 2-D TP on 2x2 | fractured over both axes | two-axis collectives | every projection | rejected provisionally: four-chip non-Galaxy mesh and clean TP=4 divisibility |

TP=2/DP=2 leaves half of the mesh out of a single model instance and doubles
local weight/cache payload, so it is rejected for single-user latency.  KV
replication is rejected because both layer kinds divide their KV heads exactly
over TP=4.  Sequence parallel prefill is not selected because it changes the
causal/cache ownership path without helping decode and would make prefill and
decode layer contracts differ.

## Final measured path

The final decode path adds TP-local DRAM-width-sharded packed QKV and O
placements to the correctness topology. QKV projection uses 32 output cores;
attention residual/projection inputs remain width-sharded over eight cores,
while each square local MLP projection uses the selected 24-core geometry.
Both row outputs use a linear TP all-reduce with two fabric links. Prefill
retains interleaved local weights; for M<=128 its MLP uses an explicit 24-core
1-D program, while larger chunks retain auto-programming to avoid oversized
per-core L1 output blocks.

### Correctness and contracts

| Gate | Sliding layer 0 | Full layer 5 |
|---|---:|---:|
| prefill-33 PCC vs optimized single-chip | 0.999849 | 0.999759 |
| traced paged decode PCC vs optimized single-chip | 0.999954 | 0.999886 |
| batch-32 non-aligned prefill/decode PCC | 0.999907 | 0.999890 |
| sliding wrap decode PCC at 1,025 / 1,057 | 0.999678 / 0.999504 | n/a |
| advertised absolute position 262,143 | traced, finite, deterministic | traced, finite, deterministic |

The decode suite uses a non-identity page table, local BFP8 cache heads, and
mutable stable token/uint32-position/int32-cache-position buffers. Eight
identical trace replays are bitwise deterministic; changing those buffers
changes output, and restoring them restores the original output bit-for-bit.
Batch-32 uses flattened `[1,1,B*S,H]` prefill and `[1,1,B,H]` decode inputs,
and every layer output is replicated `[1,1,*,5376]`, suitable for stacking.

### Warmed latency

Medians are from 12 same-process, blocking samples after compilation and warmup.
Decode numbers are TTNN trace replay; efficiency is speedup divided by four.

| Layer kind / mode | single P150 ms | 4x P150 ms | speedup | TP4 efficiency |
|---|---:|---:|---:|---:|
| sliding prefill-128 | 2.6721 | 2.4044 | 1.111x | 27.8% |
| full prefill-128 | 3.4453 | 2.4387 | 1.413x | 35.3% |
| sliding traced decode | 1.1862 | 0.5268 | 2.252x | 56.3% |
| full traced decode | 1.3520 | 0.5768 | 2.344x | 58.6% |

### Measured and rejected topology alternatives

- Eight QKV output cores measured 0.5318 / 0.5821 ms sliding/full decode,
  versus 0.5302 / 0.5824 ms for 32 cores. The 32-core geometry wins the
  sliding case and is neutral in the full case, so it is final.
- One fabric link measured 0.5558 ms sliding decode versus 0.5302 ms with two;
  two links are final.
- The initial interleaved/default attention path measured 0.5999 / 0.6702 ms.
  TP-local DRAM-sharded attention is 11.6% / 13.1% faster.
- Full traced-layer MLP decode geometry measured 0.5324, 0.5294, 0.5413, and
  0.5261 ms for 8, 12, 21, and 24 cores. All candidates passed PCC; 24 cores
  with gate/up/down `in0_block_w=7` is final.
- At prefill M=128, the auto-programmed local MLP boundary measured 0.7253 ms.
  Explicit interleaved 1-D programs on 8/12/21/24 cores measured 0.7937,
  0.6241, 0.6226, and 0.5830 ms, all above PCC 0.99993. The 24-core program
  with per-core M/N 4/7 and output subblock 1x7 is final for M<=128. The
  DRAM-sharded program family was executed and rejected by its hard `M == 1`
  kernel constraint, so it cannot be a prefill placement.
- The adapted fractured boundary was implemented with reduce-scatter, two
  distributed RMSNorms, fractured residual add, and delayed all-gather into
  the real next gate projection. It passed PCC 0.99987 or better but measured
  0.2734 versus 0.1616 ms under decode trace replay (1.69x slower), and 0.8545
  versus 0.6796 ms for prefill-128 (1.26x slower). The replicated boundary is
  therefore retained from direct evidence. Exact H=5,376 gives 42 TP-local
  tiles; padding before distributed RMSNorm would change its denominator.
- Persistent async all-reduce was not substituted blindly: the Gemma
  `CCLManager` owns only one rotating barrier slot, whereas the async composite
  requires independent barrier, reduce-scatter, and all-gather semaphore
  handles plus stable persistent buffers. The current synchronous two-link
  composite is trace-safe and measured; introducing a second ownership model
  in one decoder layer would not be a coherent stack baseline.
- TP=2/DP=2 and KV replication both waste usable devices or duplicate exact-
  divisible KV heads. TP8 is physically unavailable. Gemma 4 31B is dense, so
  no expert-parallel alternative applies.

Profiler tables and CSVs live under `tracy/final`; acceptance XML/logs,
candidate timings, exact-context, mutable-trace, and watcher artifacts live
under `evidence`. The report tables explicitly expose
reduce-scatter/all-gather CCL rows, BFP8/BFP4 matmuls, BFP8 cache operations,
SDPA, DRAM utilization, and layout movement.

### Final profiler audit

Each mode has a signpost-filtered human-readable `report.txt`, `filtered.csv`,
summary CSV/PNG, and the source Tracy op CSV. The reports merge all four device
rows and show the following critical path:

| Mode | device ops | summed device time | modeled DRAM | CCL observation |
|---|---:|---:|---:|---|
| sliding decode | 51 | 491 us | 156 GB/s (30.4%) | each RS/AG phase is 21.5/20.8 us |
| full decode | 50 | 539 us | 172 GB/s (33.6%) | each RS/AG phase is 21.2/20.2 us |
| sliding prefill-128 | 33 | 1,155 us | 77 GB/s (15.0%) | each RS/AG phase is 44.6/40.7 us |
| full prefill-128 | 30 | 1,424 us | 75 GB/s (14.6%) | each RS/AG phase is 44.9/39.9 us |

Decode QKV/O BFP8 matmuls reach 75-83% of the modeled DRAM roofline. Decode
MLP BFP4 matmuls reach roughly 48-50% of modeled DRAM and compute. The selected
prefill MLP rows are now compute-bound at roughly 69-71% modeled FLOPs;
attention projection rows remain at roughly 20-25%. Collectives and
host-visible gaps are a larger fraction of prefill. The
profiled direct-call windows intentionally measure kernels and CCL rather than
trace-dispatch latency; the separate warmed benchmark is the trace-replay
latency authority. No dense all-expert or MoE rows exist because this is a
dense decoder.

### Watcher and stress gate

The final acceptance suite is `11 passed, 15 skipped` (the skips are the
separately gated exact-context, benchmark, geometry, fractured-boundary, and
profiler tests). A separate
watcher run executes eight deterministic trace replays plus token/current-
position/cache-position mutation for both layer kinds. Worker and NoC watcher
checks are enabled; `TT_METAL_WATCHER_DISABLE_ETH=1` is required only for
watcher instrumentation of the Blackhole fabric router. The clean run passes
both layer kinds and exits normally, and
`evidence/watcher_device.log` contains no error, assert,
NoC failure, timeout, hang, or mismatch.

The limitation was established rather than assumed: fully inlined watcher
firmware exceeded the active-Ethernet 25,600-byte kernel-config buffer (27,792
bytes). No-inline watcher then ran all four mutable-trace/batch-32 stress tests
successfully but timed out restoring one instrumented Ethernet router during
teardown. A coordinated four-board reset restored the mesh; disabling only
Ethernet watcher instrumentation produced the clean gate. The failed commands,
errors, recovery, and passing XML/logs remain in `evidence`.
