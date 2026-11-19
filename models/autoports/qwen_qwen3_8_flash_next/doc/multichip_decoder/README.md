# Multichip decoder — accepted

The target is all four Blackhole p300c chips, a logical1x4 ring with two links
per edge. [multichip_decoder.py](../../tt/multichip_decoder.py) uses
`OptimizedDecoder` as its single-chip reference. Baseline commit:
`fb462731c5e32e50a3d2efadb8df3067d35dbaed`; checkpoint revision:
`812d96e10a0dd82b69d749e68693698cf6c36daa`. This stage implements decoder
layers; full-model and vLLM work are outside its scope.

The tables below measure the final default source. Long-context reservation
checks, all three full watcher runs and all final segmented profiler captures
passed. [Independent review round2](stage_review_round2.md) returned
`clean-pass`; source and documentation checks passed. Local commit SHAs are
recorded in [work_log.md](work_log.md). [Evidence manifest](evidence_manifest.json)
indexes accepted runs, CPU checks, exact artifact paths and excluded captures.
All 11 final short/long/watcher/profile run provenances match implementation SHA
`af4699739ad869466bbc825909daf868e8a2f01ed64e0af149602fe6128349c1`.

## Implementation and stack contract

The preimplementation [mesh plan](mesh_plan.md) records tensor shapes, shard
specifications, padding, topology and alternatives. Each rank owns one complete
2560-feature residual stream. A layer consumes and returns local
`[1,1,T,2560]` BF16 shards; consecutive layer kinds need no boundary conversion.
GR normalization is local to complete streams. Exact GR projections gather
normalized inputs and shard output columns. Collectives gather rank values,
widen BF16 operands to FP32, then accumulate in rank order.

GDN owns4 key and12 value heads/rank, including local recurrence and convolution
state. QSA owns6 query heads and one BF16 KV head/rank; ranks0/1 duplicate KV
head0 and ranks2/3 duplicate head1. Local caches are
`[pages,1,32,256]`, with page size32 and absolute device positions. The ordered
indexer and compressed-key selection remain replicated.

MoE uses EP4,128 complete experts/rank, and TP4 shared experts. Native grouped
dispatch/fused SwiGLU/combine executes gate-selected local experts. B32 batches
MLP rows across independent request states; its expected expert-union fraction
is46.8%, above the25% grouped-execution decision boundary. No dense all-expert
execution is used. Routed experts use BFP4 storage and LoFi native compute,
with BFP8 intermediates inside the fused kernel and BF16 external outputs.
Shared experts use BFP8 weights. Exact sensitive projections retain their
baseline precision and reduction contracts.

PLE preserves the packed group10/scale8 representation. Rows are assigned by
`rank=id%4`, `local_row=id//4`; each rank stores80000384 rows. Per-rank striped
code/scale widths are tile padded, adding4096 payload bytes/rank. The key
projection is stream-sharded and the full2560-wide value projection replicated.
Bounded host repacking uses cached memmaps outside git. PLE correctness passes;
the earlier lookup/combine failures are historical, documented in
[AUTOFIX.md](AUTOFIX.md).

Pass the **same `MeshWorkspace(mesh)` to serial layers**. It shares expert
output/reduction buffers, ownership bounds, RoPE and immutable context-index arrays; mutable
request state stays independent. Public prefill accepts nonaligned logical
lengths, internally chunking at32 rows with bounded output assembly. Padding
does not advance positions or history. Decode accepts one row/request and
list/tuple batches up to32 in the tested short-context profile.

## Capacity and context

[context_contract.json](../context_contract.json) preserves262144 advertised
positions. The current [memory ledger](memory_capacity_plan.json), including
bank rounding and concurrency assumptions, supersedes the initial estimates in
[capacity_audit.md](capacity_audit.md).

BFP4 is a physical fit requirement: all48 layers' EP4 experts occupy
16,986,931,200 bytes/rank; BFP8 experts plus packed PLE alone would require
39,766,462,464 bytes/rank, exceeding the observed34,178,731,008-byte budget
before dense weights or KV cache. Accuracy remains a separate gate.

The reservation target is31,203,443,520 bytes/rank for planned all-layer
weights/static workspace and one maximum-context request. It assumes shared
workspace/RoPE/index arrays, BF16 vocabulary-sharded embedding and independent
head, and an allowance for the future final GR mixer. Those future components
are budgeted, not instantiated. Runtime envelopes of1GiB and2GiB are estimates,
not measured full-stack trace allocations. Maximum-context input/output must
stream in bounded chunks: two fully resident262144-row residual slabs do not
fit with the1GiB envelope. B32 at capacity128 is tested; concurrent B32 requests
each at262144 are **not** promised. Arithmetic admission frontiers are in the
ledger and are not measured full-model capacities.

From-empty32769-token public prefill plus three decode steps passed with the
full reservation on layer1 and layer3. The layer1 PLE/GDN path exercises the
shared GDN implementation. Minimum whole-output PCC is0.99984349 and0.99917488,
respectively. Actual reserved allocation was31,203,453,440 /31,203,453,952
bytes/rank. See [PLE/GDN long smoke](final_mesh_l1_long32769.json) and
[QSA long smoke](final_mesh_l3_long32769.json). These are layer-plus-reservation
tests, not a running48-layer model or maximum-context accuracy run.

## Correctness and performance

Reports use real weights, logical33 prefill, reversed page tables, capacity128,
three following decode inputs, every tail length1..32, B32, and two-layer stack
checks. Runtime Torch/conversion fallback guards are active. Each B1 and B32
trace has16 replays with state restored. B1 alternates three prepared inputs;
B32 replays32 diverse request rows. Outputs and mutable state match the
corresponding eager calls exactly. Cross-baseline
cache/state comparisons are numerical where appropriate; QSA page tables and
positions, and PLE token history, match exactly. Whole-output PCC is the gate;
per-row PCC is diagnostic. These are TTNN-to-TTNN results, not new HF validation.

Sources: `baseline_l{0,1,3}_contract.json` paired with
[mesh layer0](final_mesh_l0.json),
[mesh layer1](final_mesh_l1.json), and
[mesh layer3](final_mesh_l3.json). Every report has adjacent console and
`*_provenance.json` files recording argv, environment, source hashes and exit.

| Final-source layer | Prefill + decode min PCC | Tail min PCC | Whole B32 PCC | B32 request min (diagnostic) | Stack PCC |
|---|---:|---:|---:|---:|---:|
| [0 GDN](final_mesh_l0.json) |0.99974340|0.99954444|0.99889278|0.99690312|0.99630183 (0→3)|
| [1 PLE/GDN](final_mesh_l1.json) |0.99973357|0.99958992|0.99969923|0.99886227|0.99877685 (1→0)|
| [3 QSA](final_mesh_l3.json) |0.99961370|0.99956435|0.99981844|0.99716353|0.99970829 (3→0)|

Latency is milliseconds; pairs are single-chip baseline / four-chip mesh.
The table uses separate `clean_timing_{baseline,mesh}_l{0,1,3}.json` runs with
trace allocation tracking, runtime fallback audit, watcher and profiler disabled.
The tracked final contracts above remain the correctness evidence. Speedup is
baseline/mesh and efficiency is speedup/4. Prefill uses the median of five warmed
eager calls. Decode uses the lower median of 30 samples after excluding two
warmups from 32 trace replays; state restoration and input transfer are outside
timing. B32 repeats one prepared diverse batch and is a serial request API with
grouped MLP work, not a concurrent-serving throughput measurement.

| Clean-timing layer | Prefill ms | Speedup / efficiency | B1 decode ms | Speedup / efficiency | B32 decode ms | Speedup / efficiency |
|---|---:|---:|---:|---:|---:|---:|
| [0 GDN](clean_timing_mesh_l0.json)|61.8633 /21.4153|2.8887x /72.22%|3.0770 /2.2645|1.3589x /33.97%|96.7154 /57.8731|1.6712x /41.78%|
| [1 PLE/GDN](clean_timing_mesh_l1.json)|67.5492 /26.8302|2.5177x /62.94%|4.2459 /3.4315|1.2373x /30.93%|135.4960 /95.1445|1.4241x /35.60%|
| [3 QSA](clean_timing_mesh_l3.json)|103.3456 /80.1295|1.2897x /32.24%|4.9553 /4.7477|1.0437x /26.09%|152.5764 /131.1138|1.1637x /29.09%|

These measurements use ordered gather and the selected n1 ownership-masked
matmul reduction; larger token groups retain native reduction. Exact derived
numbers and separate timing/correctness provenance are in
[performance_summary.json](performance_summary.json). Earlier tracked contract
and candidate timings are historical diagnostics, not final performance values.
The persistent native-all-reduce candidate was not selected after two B32
request PCC regressions. In diagnostic JSON, `max_absolute` means the maximum
absolute observed value, not maximum error; PCC and exactness are separate fields.

[Performance analysis](performance_analysis.md) links tt-perf-report human
readable tables, CSVs and provenance, and examines communication, compute,
DRAM and data movement. It records the four-policy producer/next-consumer
residual experiment and indexed-decode constraints without claiming unmeasured
alternatives are slower. [Profiler audit](profiler_audit.md) distinguishes
buffer overflow from the layer3 unmatched-marker/clock-rollover hypothesis.
Final [layer0](tracy/final_mesh_l0/provenance.json),
[layer1](tracy/final_mesh_l1/provenance.json), and
[layer3](tracy/final_mesh_l3/provenance.json) captures completed the ordinary
profiler parser and human-readable/CSV reporting successfully, without overflow
or fatal-marker diagnostics. Earlier incomplete captures are explicitly excluded
in the manifest. These segmented reports attribute device operations; profiler
drains inside signpost windows make those windows unsuitable as wall latency.

Watcher evidence is indexed in [watcher_summary.json](watcher_summary.json):
all four devices were watched with no exclusions, and all three full-contract
runs passed. These logs are separate from profiling and latency measurements.

CPU admission tests passed six tests and 12 subtests; packed PLE repartitioning
passed synthetic and sampled real-checkpoint byte equality. Final source
precommit checks also passed. Their logs and provenance are linked in the
manifest. Final documentation checks passed, and independent round2 returned
`clean-pass`. Nonblocking full-stack and performance hand-offs are recorded
under `Carried to next stage` in the work log.

## Reproduction

Run from this checkout in its configured Python environment, with devices
exclusively owned by the stage runner. Exact historical commands are the
`argv` and `environment` fields in adjacent provenance. The following examples
use fresh output names, so they do not overwrite recorded evidence:

```bash
export TT_VISIBLE_DEVICES=0,1,2,3
export TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto
export TT_METAL_TRACE_ALLOC_TRACKING=1
export TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0

TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  timeout 900 python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_multichip_decoder \
  --mode baseline --layer 0 --length 33 --capacity 128 \
  --reference /home/ttuser/qwen38-flash/multichip_l0_repro.pt \
  --output /home/ttuser/qwen38-flash/baseline_l0_repro.json \
  --trace --repeats 16 --tails --batch 32 --stack
timeout 900 python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_multichip_decoder \
  --mode mesh --layer 0 --length 33 --capacity 128 \
  --reference /home/ttuser/qwen38-flash/multichip_l0_repro.pt \
  --output /home/ttuser/qwen38-flash/mesh_l0_repro.json \
  --trace --repeats 16 --tails --batch 32 --stack
```

Repeat the paired commands with layer1 and layer3 and distinct reference/output
paths. For long smoke, generate the corresponding baseline first, then mesh:

```bash
TT_VISIBLE_DEVICES=0 TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  timeout 1800 python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_multichip_long_context \
  --mode baseline --layer 1 --length 32769 --capacity 32896 \
  --reference /home/ttuser/qwen38-flash/multichip_l1_long_repro.pt \
  --output /home/ttuser/qwen38-flash/baseline_l1_long_repro.json
timeout 1800 python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_multichip_long_context \
  --mode mesh --layer 1 --length 32769 --capacity 32896 \
  --reference /home/ttuser/qwen38-flash/multichip_l1_long_repro.pt \
  --output /home/ttuser/qwen38-flash/mesh_l1_long_repro.json \
  --reserve-bytes 31203443520
```

Use layer3 for the QSA long smoke. Profiler runs use the short probe's
`--profile` mode under Tracy; preserve segmented operation attribution and
keep latency runs separate. Exact capture and tt-perf-report commands are in
profile provenance linked by the performance analysis. Watcher runs must also
be separate from profiling. [work_log.md](work_log.md) records execution,
failures/recovery, remaining gates and eventual local commit SHAs.
