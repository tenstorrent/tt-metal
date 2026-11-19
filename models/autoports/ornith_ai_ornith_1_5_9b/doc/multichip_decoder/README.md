# Ornith-1.5-9B multichip decoder — stage 4

`tt/multichip_decoder.py` implements TP4 for all four local P300c Blackhole chips
in a 1x4 FABRIC_1D mesh, Linear collectives using both available links. The
single-chip reference is the completed `tt/optimized_decoder.py` (unchanged,
SHA256 `b678803e78b74ede66d61e7609c6ca9147b2977dbba702f8431eb621ef997c4e`).
Real checkpoint weights use HF revision `489cb97981b8654bcfcf30ce1f94ed1b62e07b53`.
The model is dense: 24 linear-attention and 8 full-attention layers, with no experts.
This stage supplies the layer-stack baseline; it does not implement a full model
or serving adapter. Independent [stage review](stage_review_round1.md) returned **clean-pass**, round1.
All required gates pass.

## Target and interface

The [mesh plan](mesh_plan.md) records the strategy chosen before implementation,
calculated per-chip tensor shapes, shard specs, internal padding, and alternatives.
Each chip owns 4 full-attention Q heads and 1 KV head, or 4 linear key heads and
8 value heads. MLP intermediate width is 3072 per chip. Projection weights and KV
use BFP8; activations and collectives use BF16; recurrent state remains FP32.
Measured matmuls retain HiFi2 attention and LoFi MLP with FP32 destination accumulation.

The default layer boundary is replicated device BF16 `[1,1,S,4096]`, directly
consumable by either layer kind. Decode keeps residual/norm/MLP intermediates in
local width-sharded L1 and uses direct asynchronous all-reduce. Prefill uses L1
matmul inputs, minimal matmul, reduce-scatter, then all-gather. Hidden-sharded
residuals with distributed norms, fused gather/matmul, fused matmul/scatter, and
combined fusion were adapted and measured before selecting this path.
See [topology comparisons](topology_comparison.csv) and
[312-case geometry sweep](geometry_comparison.csv).

Construct one `MultichipCollectives(mesh)` and pass `collectives=workspace` to
every `MultichipDecoder.from_state_dict(...)` in a serial stack. This shares
2 MiB/chip of persistent L1 receive buffers (64 KiB/core) and CCL semaphores.
Concurrent stacks require separate workspaces; validated execution uses one
command queue. Each request owns its `DecoderState`, replicated page table and
persistent int32 device position tensor. The caller advances positions and
restores state at request boundaries. Allocate persistent buffers and warm each
execution signature before capture; keep backing allocations alive for every
trace that references them. The tests demonstrate this lifecycle explicitly.

Local full-attention cache layout is `[physical_pages,1,32,256]`; logical positions
index the replicated page table and each chip stores its own KV head. State
capacity is page-aligned, while public logical lengths need no alignment.
Prefill chunks are at most 1024 tokens; padding is internal. Unaligned full-attention
continuations decode their leading rows until the next page boundary. Linear
attention never advances its recurrence through padded rows.

`prefill_batch` accepts ragged rows and independent state/page tables.
`decode_batch` traces a fixed number of independent rows, serialized within the
batch on all four chips. Batch32 evidence establishes this contract and its
measured latency; it is not a vectorized batch matmul implementation.

## Capacity and correctness

The advertised context remains **262144 tokens**, with no reduction.
[Context contract](../context_contract.json) and
[memory plan](memory_capacity_plan.json) account for all 32 layers' projection
copies, auxiliary tensors, conservative BF16 sharded embedding/head storage,
all full-context KV caches and unshared position/RoPE tables. The DRAM budget is
**16.340 GiB/chip**, below **31.875 GiB** usable. It includes 8 GiB for activation,
trace and CCL temporaries plus 1 GiB allocator margin; full-context input, chunk
outputs and concatenation alone can occupy 6 GiB. The shared L1 workspace is
accounted separately. This is an analytic whole-stack capacity plan, not a
measured allocator peak or a claim that all 32 layers were loaded in this stage.
Batch32 uses short independent contexts; cache memory grows with resident requests.

Final artifacts and numerical results are summarized in `final_audit.json`,
[accuracy.csv](accuracy.csv) and [performance.csv](performance.csv). The acceptance
runner checks both meaningful layer kinds against the single-chip TTNN reference:

- Every output row for logical lengths 1,2,3,4,31,32,33,1023,1024,1025,2051;
  eight changing-input traced decode steps per length; all four device outputs
  and reconstructed local KV/conv/recurrent state.
- Complete prefill at 262139, 262143 and 262144 tokens, comparing first/middle/end
  windows against complete optimized TTNN execution. Decode at position262143
  is replayed three times from restored state with bit-identical results.
- 128 evolving decode steps without state restore for each layer kind.
- Real layer0→layer3 stacks at batch1 and batch32, ragged31/32/33 prefill,
  unaligned17/18-token continuation, shuffled physical page tables, distinct
  positions, shared workspace and four changing-input trace steps.
- Trace allocation tracking enabled, including program-cache allocations;
  deliberately unsafe allocation positive controls fail as expected. Each of
  11 signatures reuses one captured trace for three fresh request-state replays.
- Dynamic torch/host-transfer guards plus static runtime fallback audit.
  Separate watcher10/NOINLINE runs cover both kinds and all11 logical lengths.

## Performance and profiler evidence

Timing tables below are generated from the final source's serial, warmed runs.
Prefill is a complete untraced1024-token layer call with synchronization; decode
is the median of seven groups of50 nonblocking trace replays. Stack timing uses
seven groups of20 replays. Reset/restore is outside timed regions. Profiler,
watcher and allocation tracking are disabled for latency measurements.
Efficiency means speedup/4; these are layer timings, not full-model TTFT or tokens/s.

| Path | Single chip ms | TP4 ms | Speedup | Efficiency |
|---|---:|---:|---:|---:|
| Linear prefill | 3.1796 | 2.1632 | 1.47× | 36.7% |
| Linear decode | 0.6437 | 0.3079 | 2.09× | 52.3% |
| Full prefill | 3.6146 | 2.2194 | 1.63× | 40.7% |
| Full decode | 0.6241 | 0.2885 | 2.16× | 54.1% |
| Layer0→3 stack decode batch1 | 1.2476 | 0.5782 | 2.16× | 53.9% |
| Layer0→3 stack decode batch32 | 39.7756 | 18.6352 | 2.13× | 53.4% |

Minimum PCC across final output, cache/state, stress, long-context, stack,
watcher and profiler comparisons: **0.99970448** (required0.995).

Final profiler category sums (µs; maximum device duration for noncollectives,
mean device duration for collectives):

| Kind/mode | Matmul | Collectives | Movement | Other compute |
|---|---:|---:|---:|---:|
| 0/prefill | 701.7 | 660.5 | 159.8 | 477.8 |
| 0/decode | 130.8 | 27.5 | 55.3 | 83.5 |
| 3/prefill | 636.8 | 663.5 | 235.9 | 575.4 |
| 3/decode | 124.0 | 27.8 | 52.0 | 73.0 |

These instrumented sums are not unprofiled critical-path estimates. Two links
reduce prefill collective totals from roughly1213–1228 to661–664 µs, and decode
from44–45 to28 µs. Decode matmuls report72–85% DRAM utilization; no matmul advice
remains. Prefill matmuls report18–26% DRAM utilization and reuse weights over1024
rows; compute and collective costs dominate this regime. Remaining movement is
159.8/235.9 µs in prefill and55.3/52.0 µs in decode. Packed attention must convert
for local head/conv consumers; MLP intermediates stay sharded in L1.


Human-readable advice tables, CSVs and source/raw-CSV provenance are under
`tracy/{0,3}/{prefill,decode}_perf_report.{txt,csv}` and
`{prefill,decode}_provenance.json`; [perf_summary.json](perf_summary.json) groups
communication, matmul and data-movement costs and verifies actual runtime precision.
The report's generic missing-program-config advice does not recognize the explicit
minimal-matmul config M4/K16/N8, subblock1x2. DRAM decode rows can report compute
utilization above100% because the report counts the eight DRAM reader cores;
this is a reporting denominator limitation, not physical compute utilization.

Profiler advice led to L1 prefill input placement and use of both fabric links.
The final path reproduces the selected two-link candidate. Direct all-reduce of
an entire1024-token prefill was rejected by its exact worker-L1 scratch contract:
`all_reduce_async_program_factory.cpp` reserves all output pages per link,
requiring8 MiB at one link or4 MiB at two, above1.5 MiB worker L1. Chunked direct
AR would be a separate future experiment; no measured rejection is claimed.

## Investigations and limits

[Anomaly ledger](anomalies.md) classifies observed errors and warnings. Two
AutoFix investigations closed stage failures without modifying shared kernels:

- [Geometry AutoTriage](AUTOTRIAGE.md) and [repair](AUTOFIX.md): over-covered K3072
  activation shards broke the single-reader sender-coordinate contract. The
  exact-K planner chooses48 input cores independently of64 output cores.
- [Watcher AutoTriage](watcher_triage/AUTOTRIAGE.md) and
  [repair](watcher_triage/AUTOFIX.md): resident fused-prefill issued reads across
  dedicated NoC owners. Public-API64/96-KiB plan constraints were validated but
  slower than separate operations. Default fusion is disabled; optional fusion
  is constrained to64 KiB. Watcher checks remain enabled. The shared resident
  kernel is not claimed repaired.

Only this 1x4 target is supported. Tested B1/B32 row-serial batching, one command
queue, and explicit shared-workspace lifetime are the stack contracts. Precision
selection is inherited from the optimized baseline; no later datatype/full-model
or vLLM stage is started here. Full-stack output accuracy and vectorized batching
remain subsequent-stage work, not claims of this decoder stage.

## Reproduction and evidence

Run from the checkout with its built TTNN environment and the pinned HF weights
available. Device jobs must be serialized and bounded. The complete runner embeds
600/900-second timeouts, separates watcher/profiling, and supports a `STOP` file
between jobs. It skips only artifacts whose implementation SHA matches source.

```bash
R=models/autoports/ornith_ai_ornith_1_5_9b
python "$R/tests/test_multichip_planner.py"
bash "$R/tests/run_multichip_final.sh" > "$R/doc/multichip_decoder/final_queue.log" 2>&1
python "$R/tests/summarize_multichip.py"
python "$R/tests/audit_multichip.py"
```

Detailed invocation history, recovery commands, evidence archive locations,
independent review and local commit SHAs are in [work_log.md](work_log.md).
`environment_provenance.json` pins hardware, source, model and tool versions.
Generated `.pt` references/weights remain under the ignored `generated/multichip/`;
raw evidence is preserved locally. Compact tables and lossless archives of the
relevant operation CSVs and large logs accompany this stage.
