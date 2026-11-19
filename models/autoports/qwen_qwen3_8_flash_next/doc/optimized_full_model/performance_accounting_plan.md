# Performance accounting plan

CPU source/artifact audit for stage7, 2026-11-14. No device measurements were
performed for this document. Inspected checkout HEAD:
`de432e04c21c3ea53ffe13d9c6e78a96f7acb704`, with a live stage7 worktree.
This is a derivation and measurement plan, not a FUNCTIONAL or stage pass.
Updated against selected MeshQSA native causal scoring and head20/3/2. No
new sampler CCL selection is assumed while its policy probe is in progress.

## Boundaries that must remain separate

The primary workload is all48 layers, TP4/EP4 on the1x4 Blackhole mesh,
B1, 128 input/128 output, capacity262144,127 steady decode steps.

1. Resident token-out: model and split-sampling traces, device token feedback,
   position advancement, no interior readback or synchronization. This is the
   primary optimized decode measurement. Include both traces and terminal work.
2. Caller-read token-out: the public generator reads one token every step.
   Keep it as an additional caller-visible comparison.
3. Readiness teacher forcing: installed `run_teacher_forcing` replaces the next
   input but still samples/reads the prediction. This is an accuracy gate and a
   distinct measured boundary, not logits-only decode.
4. Logits-only replay: a fixed resident token with advancing positions excludes
   sampling and feedback. Label it explicitly; it is neither autoregression nor
   teacher forcing.

Stage6 `../full_model/acceptance_perf_final_v1.json` reported138.754269ms
caller-read token-out,138.152610ms logits-only, and7038.672064ms warmed TTFT.
Stage7 `baseline_resident.json` is the before source for the same-harness final
comparison. Its current README headline is138.618793ms resident token-out,
7.214029tokens/s/user and7035.653ms TTFT. Re-extract final numbers from JSON
rather than treating this planning document as an authoritative benchmark.

## Hardware and byte constants

Nominal bandwidth is512,000,000,000bytes/s per ASIC, so the four-ASIC aggregate
ceiling is2,048,000,000,000bytes/s. This is a specification ceiling, not achieved
bandwidth. Sources are `tech_reports/CCLs/CCL_bandwidth.md`, section "How this
compares to local memory", and the preserved
`../fused_decoder/hardware_roofline_source.json`, which cites the Blackhole
p300c specification at
https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html.

Use total traffic over four ranks divided by aggregate bandwidth, or
mean per-rank traffic divided by512GB/s. Do not divide a per-rank byte number
by the four-chip bandwidth. EP imbalance requires an additional critical-rank
bound; aggregate bandwidth is deliberately optimistic.

Model constants come from `../functional_decoder/target_text_config.json`:
48layers,36GDN/12QSA, one PLE layer, H2560,512experts/top10,I640,
four hyperconnection streams, indexer four heads/D128/compress4,
vocab248320 and context262144. Runtime ownership/dtypes are in
`tt/multichip_decoder.py`, `tt/model.py`, and the stage5 precision/layout
artifacts. BFP4 tiles occupy576bytes, BFP8 tiles1088bytes and BF16 tiles2048bytes
per1024elements, including shared exponents (`tt/dram_projection.py`,
`weight_memory_plan`). Do not use a bare0.5byte coefficient for BFP4.

### Active payload versus resident capacity

For B1, routed expert weight bytes across the complete mesh are

```text
48 layers * 10 active experts * 3 matrices * 2560 * 640 * (576/1024)
= 1,327,104,000 bytes/token
```

This is331,776,000bytes/rank on average, assuming ideal balance. Each layer
actually executes only the selected experts; `MeshMoe` routes into
`moe_fused_swiglu`. Do not charge512experts/token or the persistent worst-case
dispatch buffer as weights read. Record measured per-rank selected expert counts
when refining the imbalance bound: `sum_layer max_rank(bytes_layer_rank/BW)`
can exceed the aggregate average.

PLE hashes16table rows per token,160decoded values per row. Each packed row is
20uint32 code words plus4uint32 scale words =96bytes. Ideal useful table payload
is16*96=1536bytes across the mesh. The current implementation calls the local
lookup for all16IDs on every rank and masks ownership afterwards; it does not
only look up the four owned IDs. `pipelined_striped_gather.py` issues one64-byte
read per selected uint32 element. Hence current code/scale table requests are
16*(20+4)*64=24,576bytes/rank/token, before index tensors, lookup tables, output
writes, duplicate-cacheline effects and CCL. This is a kernel-request estimate,
not measured DRAM transactions. The30.72GB global packed table is resident
capacity, never a per-token scan.

The token embedding similarly reads one row per rank, not the complete table.
The vocabulary projection does scan its matrix: logical BF16 head bytes are
2560*248320*2=1,271,398,400 across the mesh. Eight-bank N padding makes the
prepared head318,504,960bytes/rank (local62080columns padded to62208).
Head20/3/2 changes L1/core geometry, not this eight-bank weight storage.
The selected input shard is[32,128] on20cores and output shard[32,3104];
per-bank weights remain[2560,7776]. No second interleaved head is retained
in the full model. Do not copy the isolated sweep's retained-native comparison
allocation into the full-model weight ledger.

QSA reads selected KV rows. At decode positions128..254 all causal tokens fit
within the2048-token selection budget: mean192valid token rows, each rank
holding one D256 BF16 K/V head. Minimum KV input payload is therefore
12*192*2*256*2=2,359,296bytes/rank/token averaged over this window. The selected
gather nevertheless writes2080physical rows and SDPA consumes their fixed
geometry. Those additional writes/reads belong to graph traffic, not to
"full-capacity KV reads". At long context, use the actual selected rows and
partial-group tail; never substitute all262144positions into this KV formula.

### Auditable major-weight inventory

This table counts one scan of the currently prepared, active major matrices.
It intentionally omits small norms, taps, LUTs, recurrent state, intermediate
traffic and collectives; it is a subtotal. All entries are bytes/rank/token;
experts use ideal mean rank balance. Unused legacy duplicate weights are omitted.

| Role | Bytes/rank/token | Source or calculation |
|---|---:|---|
| GDN exact QKV | 471859200 | 36*2560*2560*2 |
| GDN prepared ZAB | 330301440 | stage5 final memory plan, gdn-compact |
| GDN exact output | 283115520 | 36*640*6144*2 |
| GR down | 377487360 | 96*96*10240*4, physical80→96 |
| GR up | 314572800 | 96*2560*320*4 |
| Prepared GR injection | 503316480 | stage5 memory plan, physical bank padding |
| Router | 31457280 | 48*128*2560*2 |
| Prepared shared gate/up | 66846720 | stage5 final memory plan, shared |
| Shared down | 20889600 | 48*2560*160*1088/1024 |
| Padded scalar shared gate | 4177920 | 48*2560*32*1088/1024 |
| QSA QGate/K/V | 116981760 | 12*2560*3584*1088/1024 |
| QSA output | 50135040 | 12*1536*2560*1088/1024 |
| Active ordered indexer | 39321600 | 12*2560*640*2; one active copy |
| PLE key/value | 13926400 | 2*2560*2560*1088/1024 |
| Final mixer down/up | 7208960 | (96*10240+2560*320)*4 |
| Prepared vocabulary head | 318504960 | 2560*62208*2 |
| Routed experts, mean | 331776000 | active-expert formula above /4 |
| **Subtotal** | **3281879040** | **6.409920ms at512GB/s** |

The stage5 inventory sources are
`../multichip_decoder/memory_capacity_plan.json::static_ledger` and
`../optimized_multichip_decoder/final_context_memory_plan.json::roles`.
The latter replaces the former's old GR injection, GDN ZAB and shared gate/up
storage; do not add both. The head/final mixer are taken from the current model,
not stage4's future-full-model memory allowance. Small fields and physical
padding must be audited against the final measured source before promoting
this subtotal into `perf_summary.json`.

Removing the largest avoidable padding terms (GR injection logical one-column
output, GR down logical80, logical1560ZAB, shared width160, and head62080)
gives a more optimistic major-weight subtotal of2649128960bytes/rank,
5.174080ms, or a10%-of-peak throughput of19.3271tokens/s/user. This remains a
partial architectural payload estimate, not the implemented-graph denominator.
The physical-matrix subtotal gives a10% threshold of15.6008tokens/s/user.
Neither subtotal alone establishes a final FUNCTIONAL verdict.

### Necessary scans in the selected exact graph

The selected `MeshExactProjection` still materializes FP32 products in DRAM
and passes them to the accurate SFPU sum in `FusedPreciseProjection`. Count at
least one product write and one subsequent read. The current graph's major
product materializations for GR down/up, GDN output, router and final mixer are

```text
2 * [96*(96*10240 + 2560*320)*4
     + 36*640*6144*4
     + 48*128*2560*4
     + (96*10240 + 2560*320)*4]
= 2,656,829,440 bytes/rank/token
```

These are necessary for those unfused selected expressions, not an intrinsic
model requirement. The fused exact GDN QKV native kernel is deliberately not
charged a DRAM product tensor. Matrix subtotal plus these products is
5,938,708,480bytes/rank,11.599040ms at nominal bandwidth. This is an
implementation-traffic subtotal, **not a FUNCTIONAL acceptance denominator**;
no10% acceptance threshold may be derived from it. It still omits other graph
traffic named above and is useful for explaining materialization overhead.
The strict useful-weight plus selected-KV denominator remains6.414528ms.

### Selected causal-scorer traffic; old scorer excluded

`MeshQSA._selected_ids` now calls `causal_indexer_scores(..., packed_decode=True)`.
`causal_packed_v4.json` contains36 passing real-operand cases, Q1/3/17/32,
short prefixes through near-max context, with exact live scores and ordered
IDs. It is component evidence (`tracking=false`); final full-model gates are
pending. The selected kernel uses runtime device positions, hardware query ROW
broadcast and the unchanged SFPU multiply/sequential-four-tile-fold/row-sum
order. There is no allocated-capacity BF16→FP32 key cast or DRAM product tensor.

**Exclude the old4,227,858,432bytes/rank/token (8.257536ms) key-cast/product
scans from every final-graph denominator.** That number describes the old
full-capacity scorer only. It is not a remaining unavoidable cost or an
optimization credit to add to the new graph. Preserve it only in the historical
before/after explanation.

For selected B1 packed-head decode, define `K=65536`,
`L(p)=floor((p+1)/4)`, `A(p)=ceil(L(p)/32)` and
`R=min(compute_grid.x*compute_grid.y, K/32)`. Source is
`tt/kernels/causal_score_packed_q1_{reader,compute,writer}.cpp` plus
`tt/causal_indexer_scores.py`. Counts below are **per QSA layer, per rank**:

| Item | Source-derived bytes / actions | Interpretation |
|---|---:|---|
| Unique useful key payload | `L*128*2 = 256L` | Architectural useful bytes; one compressed key reused for all four heads. |
| Actual tiled key read requests | `A*4*2048 = 8192A` | Each active key tile loaded once; four heads share its L1 CB. |
| Query read requests | `A*4 heads*4 feature tiles*2048 = 32768A` | Current reader reloads four query tiles per head per active key tile. Unique query payload is only1024 B, but physical query tiles total32768 B. No claim that all repeated requests become unique DRAM transactions. |
| Position read requests | `2*R*4096` | Reader and writer each load a position tile on every worker, including workers with no live key jobs. |
| Query L1 scalar stores | `A*4*4*32` BF16 words | ROW broadcast replaces the former1024-word full-tile construction; these are L1 stores, not DRAM traffic. |
| Dot output write | `(K/32)*4096 = 8 MiB` | Every output tile is initialized, causal values in four head rows, remaining rows/columns zero. This remains capacity-dependent. |
| ReLU input/output | `8+8 = 16 MiB` | Separate FP32 tiled unary op. |
| Head reduction input/output | `8+8 = 16 MiB` | Reduce four logical rows, preserving physical32-row output allocation. Additional implementation scratch is excluded. |
| Scale input/output | `8+8 = 16 MiB` | Separate selected FP32 multiply. |
| **Public scorer materialization subtotal** | **56 MiB** | Dot write plus separate terminal input/output tensor passes; no old full-capacity product/cast traffic. |

Across primary positions128..254, mean `L=47.6220472441` and
`A=1.97637795276`. Thus unique keys average12191.2441 B/layer/rank;
combined query/key read requests average80952.4409 B, plus `8192R` position
requests. For R110 that position term is901120 B. Record the actual R from
final source/provenance rather than silently using the profiled worker count
of another operation. Physical padding and repeated NoC requests are a
**selected implementation request ledger**, not the ideal payload denominator.
At long context recompute A using the actual advancing positions.

For Q2..32 the selected standard kernel retains four-head physical output
`[1,4,32,K]` (32 MiB). Let `Amax=max_q ceil(L(p_q)/32)`.
It requests `4*Amax*8192` key bytes and the same query bytes, sharing each
head's key block across its active query rows. Each query/feature tile writes
32 BF16 row0 words and compute broadcasts them. Dot arithmetic executes
`4*sum_q ceil(L(p_q)/32)` jobs, each with four feature-tile products and three
sequential additions. Output writes remain32 MiB per invocation. The dim1
head-reduction path can add transpose/reshape materializations, so do not
reuse the Q1 terminal-byte total for prefill or Q32.

Adding only the selected B1 scorer's56 MiB ×12 layers to the previous
major-matrix plus exact-product subtotal gives

```text
5,938,708,480 + 12*56*1048576 = 6,643,351,552 bytes/rank/token
nominal selected-graph subtotal floor = 12.975296 ms at512 GB/s
```

This is **not the theoretical functional payload floor or an acceptance
denominator**. Keep the separate
ideal major-weight5.174080ms and prepared-weight6.409920ms values visible.
The12.975296ms estimate is a conservative subtotal for explicit materialization
in the selected graph, excluding metadata/query/key requests, KV gather/SDPA,
recurrent state, residual operations, embedding/PLE, collectives, sampler and
other graph traffic. It is unavoidable **within the current unfused graph**,
not intrinsic to the model or a reason to stop fusing avoidable intermediates.
Validate terminal physical transfers against the final reduced profile;
allocator size alone does not prove external-memory transactions. Never sum
this subtotal with an already-inclusive measured or inferred full-graph byte
estimate. Do not infer FLOPs/bytes from the known-inconsistent historical
BinaryNg shape annotations.

## Device-time reconstruction and host reconciliation

The installed optimize `references/perf-accounting.md` requires a reduced
one-layer-per-kind profile for complete models, with weighted layer time plus
terminal time. Use one source-matched reduced profile, real shapes, capacity,
dtype and representative history. Split its model region into L0GDN,
L1PLE/GDN, L3QSA and terminal/entry operations, then calculate

```text
device_estimate_ms = 35*L0_ms + 1*L1_ms + 12*L3_ms
                    + token_entry_ms + embedding_ms
                    + final_mixer_ms + head_ms + row_assembly_ms + split_sampler_ms
reconstruction_gap_ms = full48_resident_e2e_ms - device_estimate_ms
```

Keep source-order row ranges and global-call IDs supporting each boundary.
Do not multiply the entire three-layer model sum by16: there is one PLE layer,
not16, and terminal work executes once. Current stage6 signposts cover the
combined reduced model and each sampler; existing `terminal_final_analysis.json`
does not supply all three individual complete-layer ranges. Those ranges need
to be resolved in the final profile before claiming this reconstruction.

`tt-perf-report`1.5 merges noncollectives by max-rank duration and collectives by
mean-rank duration. That sum is neither one rank's wall time nor all-rank time.
Raw device durations are ns; report device time is microseconds. Preserve both
kernel sums and available operation-gap columns, and distinguish eager profile
host gaps from traced decode. The reduced-to-full residency/history projection
also makes `reconstruction_gap_ms` an estimated residual, not a pure measured
host-overhead term. A negative gap is an accounting anomaly to investigate.

Final `perf_summary.json` derivation should preserve three named regimes:
`ideal_active_payload`, `prepared_weight_scan`, and
`selected_graph_materialization_subtotal`, each with its byte formula, omissions,
nominal bandwidth, computed floor, and `is_complete=false` until audited.
A recommended source manifest includes hashes of model.py, multichip_decoder.py,
causal_indexer_scores.py, all six causal native kernels, dram_projection.py and
final generator/common sampler sources. Preserve the actual mesh/grid, head
20/3/2, BF16/HiFi4/FP32 DEST/packer-off, selected scorer and measured CCL branch.

Extract full48 end-to-end values directly from paired resident JSON using the
same per-token boundary; calculate `tps=1000/latency_ms`,
`speedup=before_ms/after_ms` and `latency_reduction=1-after_ms/before_ms`.
For weighted layer data record35GDN+1PLE/GDN+12QSA and terminal work separately.
Compute `avoidable_gap_fraction=resident_ms/(stack_ms+terminal_ms)-1` only when
source, capacity, live-history and boundary scopes are compatible; otherwise
leave the gate null with the mismatch reason. Do not use10geometry head trials
as final accuracy: `head_sweep_v2.json` uses real weights/synthetic activations
and is explicitly not acceptance-eligible. The8/1/1 L1 rejection lives in
`head_sweep_v1.json`; final20/3/2 integrated quality is still pending.

`perf_summary.json` must record exact source artifacts and boundaries next to
TTFT, resident end-to-end latency, reconstructed device latency, roofline
estimate, graph/payload byte regimes, nominal bandwidth and named limitations.
Use null plus an explicit pending reason while a required measurement is absent;
do not fill it with a historical incompatible profile. All48 profiling remains
prohibited. No estimated-layer reconstruction is an all48 hardware timestamp.

## Matching-capacity layer measurement commands

The existing harness is `tests/probe_multichip_decoder.py`. Its `--trace
--clean-timing --batch 0` runs the B1 trace and disables the additional B32
probe; `--batch 1` is unnecessary. It restores prefix state before each replay,
changes among three prepared input tokens, enqueues a nonblocking trace and
synchronizes inside the measured window. This is synchronized layer latency,
not an advancing127-token resident stream.

Reference metadata includes layer, snapshot, length and capacity. Existing
stage5 C128 references therefore **cannot** be reused at C262144. Generate
new single-chip references once with the matching real embedding fixture, then
reuse them for both mesh revisions. These are later-layer real-embedding routing
stress inputs, not captured full-stack activations. The fixture already has1027
sequence rows (`tests/prepare_multichip_prompt_fixture.py`).

The following commands are a recipe, not executed evidence. Run each job
serially under `run_recorded` with distinct output/log/provenance paths. Confirm
the devices are unowned before starting. Shell variables below are task-local.

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/environment.sh
export TT_METAL_TRACE_ALLOC_TRACKING=0
stage7_dir=models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model
stage7_fixture=/home/ttuser/qwen38-flash/stage5_real_prompt_fixture.pt

# Run L=0,1,3 serially. The single-chip reference uses its own accepted descriptor.
stage7_layer=0
TT_VISIBLE_DEVICES=0 \
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
timeout 1800 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_multichip_decoder \
  --mode baseline --snapshot "$QWEN_CHECKPOINT" --layer "$stage7_layer" \
  --reference "/home/ttuser/qwen38-flash/stage7_c262144_n128_l${stage7_layer}.pt" \
  --output "${stage7_dir}/reference_c262144_n128_l${stage7_layer}.json" \
  --length 128 --capacity 262144 --batch 0 --expert-dtype bfloat4_b \
  --batch-fixture "$stage7_fixture" --cache-dir "$QWEN_PLE_CACHE"

# Run once on the baseline revision, then on the final selected revision;
# preserve source/command provenance for each. Do not overwrite the first run.
stage7_revision_label=before
TT_VISIBLE_DEVICES=0,1,2,3 \
TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto \
timeout 1800 python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_multichip_decoder \
  --mode mesh --snapshot "$QWEN_CHECKPOINT" --layer "$stage7_layer" \
  --reference "/home/ttuser/qwen38-flash/stage7_c262144_n128_l${stage7_layer}.pt" \
  --output "${stage7_dir}/${stage7_revision_label}_c262144_n128_l${stage7_layer}.json" \
  --length 128 --capacity 262144 --batch 0 --expert-dtype bfloat4_b \
  --batch-fixture "$stage7_fixture" --cache-dir "$QWEN_PLE_CACHE" \
  --trace --repeats 32 --clean-timing
```

The baseline revision is the stage6 accepted source recorded in
`baseline_resident.json`; use an isolated matching checkout if the production
worktree already contains the new default. Do not silently call the changed
default "before". `run_recorded.py --record ... --log ... -- <command>` captures
Python and native-kernel sources. Optional `--reserve-bytes` is supported, but
filler only approximates allocation pressure; it does not recreate all48
weights or their DRAM bank behavior. Keep with/without-filler runs separate.

The historical weighted sum is
35*2.081642866 + 1*3.136728048 + 12*2.858865023 =110.300608635ms,
from `../optimized_multichip_decoder/real_mesh_final_l{0,1,3}.json`.
Those runs explicitly used C128/length33. The new C262144/length128 three-layer
sum corrects capacity and starting-history mismatches but still is not a proof
of an all48 lower bound or its growing-position average. State these limits.

## Claims not available yet

- A passing10%-theoretical verdict: the strict prepared useful-weight plus
  selected-KV denominator is settled at6.414528ms for this measured window.
  `functional_floor_audit_v1.json` records113.881454ms and5.632636% achieved,
  below10%; the required latency is at most64.145280ms. Implemented-graph
  intermediate traffic is diagnostic only and is never an interchangeable
  acceptance denominator. A future pass requires a faster correctly validated
  measured path, not counting product materializations, padded scorer outputs,
  all-resident table capacity or eliminated traffic as useful model work.
- A source-matched weighted device-time reconstruction with all complete layer
  boundaries, and its reconciliation against the final full48 resident window.
- Matched C262144 representative-layer before/after latency. Historical110.301ms
  remains a useful comparison with explicit caveats.
- Exact DRAM transactions or achieved bandwidth. The estimates above are byte
  arithmetic and kernel request reasoning; no hardware bandwidth counters were
  collected for this audit.
- Pure host overhead from full48 time minus reduced kernel sums. Residency,
  profiler perturbation, rank merging, token routing and history differences
  must remain visible rather than being relabeled dispatch or host work.
