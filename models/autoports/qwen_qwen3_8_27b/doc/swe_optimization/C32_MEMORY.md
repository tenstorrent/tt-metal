# Qwen C32 memory and concurrency investigation

Status: analysis and measured baseline reconciliation, 28 October 2026.
The user requests B32 and32 or more agent tasks for SWE100, SWE500 and
Terminal89. C8 partitions are prepared fallback configurations, not the final
concurrency decision. No large suite has been dispatched.

Distinguish resident model rows B, vLLM admission, HTTP requests and Harbor task
permits A. Target B32/A32, then A40 if the measured working set and reuse survive.
The current native batched implementation covers at most32 rows; A40 means
more tool-running/queued tasks, not an unvalidated64-row model.

## Measured per-device baseline

Flat serving smoke40599697092, immutable image677859aa1eba574ffb95214247ed1849ad17cddfe1faf1884c0dd73a5bf53d07,
Metal6faa07fb5320284381ca1a9ab4651489366d7436, B16/block512. Four TP devices
must each fit these bytes; their memory cannot be treated as one allocator.

| Component | GiB per device | Evidence scope |
|---|---:|---|
| Recurrent prefix snapshots | 16.344 | Exact16 shared pools,1368 IDs, FP32 state/BF16 convolution |
| Attention K/V | 5.678 | Exact selected BFP8 physical backing |
| Active recurrent state | 0.573 | All48 recurrent layers,16 live rows |
| Weights, runtime and other remainder | 5.982 | Allocator total minus the three identified components |
| Allocated total | 28.578 | Allocation checkpoint |
| Free | 3.129 | Same checkpoint, not additional safe cache budget |

Preparation's warmed-before-capture checkpoint is higher than the allocation
checkpoint. These samples are not continuous peaks or fragmentation proofs.
The configured trace arena is128 MiB; its size is not a measured used payload.
The historical same-policy tensor inventory reports6,224,093,184 payload bytes
per device including duplicate MLP owners; that is context, not a fresh weight
inventory in this serving lifetime. The unitemized remainder is not relabeled
as an exact measured weight subtotal.

## Actual bytes per user

Attention K/V costs8704 bytes per token per device:16 full-attention layers,
one local KV head, head dimension256, K and V, eight1088-byte BFP8 tiles per32
tokens. This includes tile format overhead. Active recurrent state adds
48*(3*2560*2 +12*128*128*4)=38,486,016 payload bytes, or36.70 MiB per user.
Each retained all-layer recurrent checkpoint also needs36.70 MiB, before
pool alignment and the allocation policy's current/previous/cache history.
BF16 convolution pages round independently across eight DRAM banks. At B1,
active-state padding adds1,228,800bytes/device across48layers; B8/B16/B32 align.
The proposed771-ID/block2048 shape adds573,440bytes/device across16 snapshot
pools. The historical1571-ID/block256 B8 metadata also omitted573,440bytes of
convolution padding. Its measured allocator total is unchanged; the corrected
layout-derived split moves these bytes out of the unitemized remainder.

| History including generated tokens | Attention K/V per user/device |
|---|---:|
| 16K | 136 MiB |
| 32K | 272 MiB |
| 64K | 544 MiB |
| 128K | 1088 MiB |
| Native256K | 2176 MiB |

Thus32 simultaneous native256K histories need68 GiB/device for attention alone.
Preserving the262144 per-request context does not promise32 simultaneous maximum
histories. The audited25-task control's per-request prompt median22592,
p9593891 and maximum137737 are workload observations, not measured simultaneous
demand. Tool gaps retain useful histories even after request ownership ends.

## Checkpoint geometry and planner units

With current shared snapshot pools, physical bytes per global ID are
`block_tokens*8704 +16*801792`. Live recurrent state is separate. At the
23,661,297,664-byte budget, changing checkpoint spacing redistributes the same
approximately22-GiB backing:

| Checkpoint tokens | Global IDs | Attention GiB | Snapshot GiB |
|---:|---:|---:|---:|
| 512 | 1368 | 5.678 | 16.344 |
| 1024 | 1088 | 9.031 | 12.999 |
| 2048 | 771 | 12.800 | 9.212 |
| 4096 | 488 | 16.203 | 5.830 |

These are physical shape forecasts; larger spacing can require more prefix
recomputation. The virtual token hook subtracts `7*B+1` worker IDs. Its output
is a pool-sizing inversion, **not guaranteed conversation-token capacity**.
All four cache groups compete for the global ID pool. At B32/block2048,
32 private32K attention histories need512 IDs; adding225 worker reservations
screens737 of771 IDs, before extra idle/checkpoint history. This is a diagnostic
screen, not a scheduler simulation, retention guarantee or scored result.

## Memory reductions to test

1. **Remove duplicate MLP owners.** The source retains separate BF4 gate/up and
   packed BF4 gateup weights. The duplicate packed payload is
   `64*(5120/32)*(8704/32)*576 =1,604,321,280 bytes/device`, or1.494 GiB.
   Test a default-off shared-owner path with unchanged dtype, K blocking and
   activation rounding. Measure real allocation deltas and latency; do not
   exchange memory savings for an unmeasured decode slowdown.
2. **Checkpoint1024/2048 with bounded physical prefill chunks.** Keep the native
   logical context and exact scheduler save/restore boundaries. Test scratch
   and trace preparation independently of larger checkpoint spacing. Current
   B32 warmup can use1024-token physical chunks; retaining512 may avoid extra
   scratch and reuse the validated flat geometry. The optional
   `QWEN38_PREFILL_CHUNK_CAP=512` now prepares this path without changing default
   policy or logical context. It requires a new source overlay; the existing
   immutable image does not implement this flag. Only CPU policy checks have
   run: native and exact0.30 serving save/restore, lifecycle, quality and latency
   validation remain required before selection.
3. **Group-exclusive union backing.** Existing shared pools already reuse storage
   across the three recurrent groups. Investigate including attention in the
   same ownership rule. Two page-aligned raw K/V carriers would reserve the
   maximum of attention versus snapshot bytes per global ID, rather than their
   sum. At block2048 this screens17,825,792 versus30,654,464 bytes/ID:41.85%
   less, or1327 rather than771 IDs at the same budget. This is unimplemented.
   It requires translated per-layer attention tables, exact raw FP32/BF16 DMA
   into mixed-type carriers, ownership/fence proofs, null-block handling and
   warmup changes that never clone the complete shared pool as scratch.

Gemma reference: agentic-research commit ecc743d72580ce7e5cb973d4ffef8670b3ab52f7,
`experiment-29/reports/kv-tight-packing/DESIGN.md` and `PREFIX_CACHE_MECHANICS.md`.
Its packing exploits exclusive group ownership, while weight sharing removes
duplicate phase owners. Qwen's FP32 recurrent snapshots make the physical layout
different. Preserve exact bytes/ownership; do not copy Gemma's numeric pool size.
Zero preemptions or low sampled KV usage do not exclude idle prefix eviction.

## Validation and next runs

Reconcile a fresh native weight/request/workspace inventory with allocator
checkpoints, then use standalone allocation-event diagnostics for transients.
Do not attach a profiler to a live serving run. Keep all raw evidence local/CI.
Test reduced and full64-layer state/logit/token correctness, ragged rows,
continuations, cancellation, eviction/reuse and trace lifetimes. Measure warmed
decode and realistic16K/32K/64K request work before selecting B32/A32 or A40.
Then use the frozen small scored sets with unchanged individual budgets before
SWE100/500 and the selected Terminal89 roster. Memory fit is necessary but does
not establish useful throughput or accuracy.

The combined C8 pilot40607188492 completed18/25 with two agent timeouts and
10933.654s trial envelope; control16/25 had no exceptions and9323.502s. Isolation,
stable archive and response reconciliation pass; retrieval review is pending.
This is not a measured SWE speedup or a reason to skip the C32 investigation.
Gemma Terminal89 is2.0; Qwen's old frozen89 roster is2.1. Version clarification
is pending; no new Terminal run is dispatched from an ambiguous dataset name.

Reproduce the CPU ledger from the compact prepared observations:
`python3 models/autoports/qwen_qwen3_8_27b/tests/plan_serving_memory.py --prepared-observations <prepared-observations.json> --output <ledger.json>`.
The planner reads arithmetic constants from the actual cache source, validates
typed backing against collected checkpoints and never imports TTNN/vLLM.
It retains all three collected B8/B16 allocator totals and distinguishes their
reported payloads from source-layout-derived bank-padded backing. Those derived
components are not fresh independent allocator measurements. A separate
legacy cache-spec test run on the shared host produced five passes and one
schema failure: installed vLLM0.26 lacks the0.30 MambaSpec checkpoint argument
used by the real CI image. This is not a measured model failure or a passed
0.30 schema test. The host environment was not changed to mask it.
See [compact numeric plan](C32_MEMORY_PLAN.json). The optional shared MLP path
now has a [measured full64 native result](SHARED_MLP_FULL64.md):1.483GiB/device
prepared saving, exact paired outputs, and a3.67% decode regression. It remains
disabled by default. Actual serving retention and C32 selection remain open.
