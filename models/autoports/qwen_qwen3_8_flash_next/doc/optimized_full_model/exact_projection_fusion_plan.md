# Exact Q1 projection fusion: CPU feasibility plan

Status: source-only proposal, 2026-11-14. No implementation change, device execution, speed claim, or precision-policy change accompanies this document. Wait for the independent AutoDebug diagnosis before implementing. The candidate preserves the selected accurate-SFPU path, including its explicit final BF16 typecast. It does not target the balanced QKV projection.

The [functional floor audit](functional_floor_audit_v1.json) records 113.881454 ms and a weights-plus-selected-KV floor of 6.414528 ms. Acceptance needs at most 64.14528 ms, a saving of at least 49.736174 ms (43.67%). Removing product intermediates must not increase that denominator. This proposal is a bounded experiment toward that gate, not evidence that the gate can already be met.

## Source contract and eligible projections

`tt/fused_decoder.py:26–129`, `FusedPreciseProjection.__call__`, widens the input to FP32, multiplies against each output-row weight block, invokes `ttnn.sum(... dim=-1, fast_and_approximate_mode=False)`, reshapes/concatenates the reduced columns, and explicitly typecasts to BF16. The `balanced_tree` branch has a different arithmetic order and is excluded. FP32-resident GR weights bypass an allocating weight typecast; mesh BF16 weights use `widen_in_unpack=True`.

`tt/multichip_decoder.py:124–142`, `MeshExactProjection`, retains one padded output-sharded weight tensor per projector. `GatheredProjection` at lines 144–159 owns the surrounding gathers and GR-down padding removal. The replacement goes inside this wrapper; no CCL, residual placement, padded rank ordering, or weight setup changes are needed.

| Eligible role, per rank | Existing resident weight | K tiles | R=8 jobs | Required output contract |
| --- | --- | ---: | ---: | --- |
| GR down, including terminal mixer | FP32 `[96,10240]`, 80 valid rows | 320 | 12 | BF16 width96; existing gathered rank compaction keeps 80/rank |
| GR up | FP32 `[2560,320]` | 10 | 320 | BF16 width2560, local stream |
| GDN out | BF16 `[640,6144]` | 192 | 80 | Existing input gather to6144, output gather from640 |
| MoE router | BF16 `[128,2560]` | 80 | 16 | BF16 width128 before existing gather and FP32 widening |

Shapes/dtypes follow `MeshResidual` at `tt/multichip_decoder.py:196–228`, `MeshGDN.out` at 277–279, and router construction at 647; also see [operation topology audit](operation_topology_audit.md). All K and padded O values are tile aligned. Q1 is the only initial dispatch condition; Q>1 calls the unchanged selected implementation, including awkward prompt lengths and factored MoE batches. A factored attention slot may independently use Q1, but a batched router must keep its Q>1 path.

## Exact arithmetic to retain

The authoritative reduction is `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl:111–124,486–574`. It seeds an accumulator with the first width tile, folds subsequent width tiles in increasing order with `add_binary_tile`, then calls the intra-tile `sfpu_reduce<SUM,Float32,REDUCE_ROW>` once. For each output row o and lane j=0..31:

```
p[t,j] = FP32_multiply(FP32(x[32*t+j]), FP32(w[o,32*t+j]))
a[j] = p[0,j]
for t = 1 .. K/32-1:
    a[j] = FP32_add(a[j], p[t,j])
y[o] = existing_SFPU_row_reduce(a[0:32])
output[o] = existing_TTNN_typecast_to_BF16(y[o])
```

Native code must retain separate multiply and add instructions, seed from the first product instead of adding to zero, and use the exact existing SFPU row reducer. Do not partition K into independently reduced chunks, accumulate scalar K elements, use a balanced tree, introduce FMA, use tensor-core dots, or pack directly to BF16. BF16 products fit FP32 for normal finite operands; FP32 GR weights still require the selected FP32 product rounding. Bitwise comparisons against the device reference decide acceptance, including signed zeros and cancellation cases; CPU `torch.sum` is not an exact oracle for this tree.

`tt/kernels/parallel_ordered_projection_compute.cpp:12–68` provides masked multiply/add wrappers with `VectorMode::R`, using two iterations for four rows and four for eight rows. Reuse only these wrappers, not that kernel's scalar-K accumulation order. The current [causal scorer](causal_scorer_contract.md) provides the proven multiply/width-fold/SFPU-reduce register lifecycle, but its BF16-only inputs do not establish FP32 GR correctness.

FP32 copy/broadcast must bypass the reduced-width source registers. Configure `UnpackToDestFp32` for FP32 copy CBs, following `tt/parallel_ordered_projection.py:93–105`. `tt_metal/hw/inc/api/compute/bcast.h:37–103` explicitly routes FP32 ROW broadcast through unpack-to-destination on Wormhole/Blackhole. After `unary_bcast_uninit<ROW>`, restore ordinary key/weight copy initialization and data format before copying weights, as the causal scorer does. Explicitly exercise mixed BF16-input/FP32-weight and FP32-input/FP32-weight variants. Do not silently route FP32 weights through BF16 or TF32. Quasar is outside this initial candidate's supported architecture.

## Minimal hardware-feasible candidate

Proposed API: `exact_projection_q1(value, weight, *, rows_per_worker=8, num_cores=None)` returns FP32 reduced outputs for diagnostic comparison, or an explicitly named wrapper applies the existing TTNN BF16 typecast. A production opt-in can eventually dispatch inside `MeshExactProjection.__call__` only for Q1 and `balanced_tree=False`. It must keep `weights`, `output_width`, dtype, and mesh placement unchanged. Initially support one existing parent tensor, exactly the selected mesh shape; do not generalize the original multi-parent projector before evidence requires it.

Use two native programs plus the existing output typecast. The first computes independent output-row groups of R=8 (also benchmark R=4); the second assembles compact reduced values into a normal tiled projection output. This avoids both huge product tensors and competing partial writes to a shared output tile.

**Reader and work ownership.** Static job g owns output rows `[g*R,(g+1)*R)`. Job assignment is `g=core_index; g<ceil(O/R); g+=num_cores`. R divides16 so a job never crosses a 16-row face boundary. For each width tile t, source weight tile index is `(g*R//32)*(K//32)+t`. For source row offset s=`g*R%32`, face-half h=`s//16`, and in-face row r=`s%16`, read exactly two contiguous fragments: source element offsets `h*512+r*16` and `h*512+256+r*16`, each `R*16` elements. Place them at local offsets0 and256 so owned output rows occupy rows0..R-1. No setup repacking or duplicate full matrix is needed. Both BF16 and FP32 fragments have naturally aligned, substantial burst lengths (R=4 gives128 or256 bytes/face).

Read activation row0's two 16-element faces from its tile t into row0 of an activation CB tile. Hardware ROW broadcast supplies the output rows. The initial implementation may reread the activation for each job; its total useful payload is `ceil(O/R)*K*sizeof(x)`, versus `O*K*sizeof(w)` for weights. Reuse/multicast is a later, separately validated optimization. Do not cache all K weights in L1. Query streaming, double-buffered activation and weight CBs, three FP32 DST tiles, and a small result queue keep the working set bounded independently of K.

**Compute.** For each job acquire DST once. For each t, wait one activation and one weight tile, ROW-broadcast activation to DST0, copy weight to DST1, multiply valid rows into DST2 on t=0 and DST0 thereafter. On t>0 add DST0 to DST2 over the same valid rows. Pop the two input tiles once consumed. After all K tiles, apply the standard full SFPU row reduction to DST2. Reserve one FP32 result tile, commit/wait/pack/release, then push it. No CB wait is conditioned on tensor values; reader and compute share the same static job and K loops.

Initialize masked-out DST2 rows to zero at each job before the first masked product (valid rows are then overwritten by that product, never added to zero). Zero non-owned weight rows on every CB page use before partial reads, or establish and prove permanent per-page zeros with a source audit. The first version should prefer simple full zeroing. The ROW broadcast consumes only the input row, so other activation rows do not enter arithmetic. Masked SFPU execution and the row reducer must be checked on all physical faces: repeated use must not leak a previous job's rows. Padding of final output is always zeroed by the assembler independently.

**Compact scratch and packing.** Allocate scratch as interleaved DRAM FP32 ROW_MAJOR `[1,1,groups,32]`: one 128-byte page per group, with slots0..R-1 containing reduced FP32 row values and all remaining slots zero. The first writer waits for a packed FP32 result tile, copies the row-reduced column0 words into a zeroed 128-byte staging page, writes the entire page to its uniquely owned scratch page, waits for the NOC write barrier, then pops its result tile. For R<=8, these reduced row words are at tile element offsets `16*r`, r=0..R-1. Keep this assumption covered by an explicit row-ramp probe. A raw staging CB page is never unpacked as a tile; its byte/page geometry must be specified separately from the 4096-byte compute result CB.

The assembly program gives each output column tile (32 outputs) one owner. It reads the corresponding `32/R` scratch pages, starts with a completely zeroed4096-byte output tile, and writes the32 reduced values into physical row0, face offsets0..15 and256..271. One owner writes the entire output tile and barriers before reusing staging storage. The result has logical `[1,1,1,O]`, padded `[1,1,32,O]`, FP32 TILE. Finally call `ttnn.typecast(...,ttnn.bfloat16)` exactly as today. No FP32 arithmetic is added by scratch/assembly.

This deliberately avoids direct 16/32-byte fragments from many workers into the same tiled final buffer, initialization-versus-write races, uncertain narrow NOC alignment, and inter-core completion semaphores. The second program is ordered after the first on the same CQ, so it sees complete scratch pages. Its additional launch is measured, not assumed free. A single-program semaphore fan-in is a later option only if assembly dispatch dominates a demonstrated win.

| Per first-program job | Reader | Compute | Writer |
| --- | --- | --- | --- |
| Activation CB | Reserve/push K/32 tiles total | Wait/pop K/32 | None |
| Weight CB | Reserve/push K/32 tiles total | Wait/pop K/32 | None |
| FP32 result CB | None | Reserve/push1 | Wait/pop1, after completed write |
| Scratch staging page | None | None | Fully reset, write once, barrier before reuse |

CB input depth2, result depth2, and one staging page suffice; readers may run ahead only to CB capacity. Every active core has at least one job. No cross-core CB counts, dynamic live bounds, mailbox protocol, or CCL semaphores are needed. Exact L1 allocation and all format descriptors still need compile-time validation in the eventual implementation.

**Lifetime and replay.** Each call's output and scratch are distinct from inputs/weights and each other. Weights remain setup-owned and unchanged. Inputs may change through the generator's existing persistent buffers; no host inspection of values or per-token allocation is introduced by trace replay. Prepare both programs and the typecast before capture; capture-time scratch lifetime and allocator reuse must remain protected by the trace ownership/allocation-tracker contract. Do not add an unbounded Python scratch cache or retain a transient tensor across unrelated traces. Stress multiple projector instances and back-to-back replays with changed inputs; output storage and padding must be overwritten each invocation. Q>1 delegates to the selected implementation without preparing a different precision policy.

## Performance feasibility and honest budget

R=8 increases GR-down parallelism from three32-row jobs to12 jobs; R=4 gives24. Router increases four32-row jobs to16 or32. Larger GDN-out and GR-up matrices provide80 and320 R=8 jobs. This is output parallelism only; K remains sequential in each job. The cost is more input rereads and SFPU setup calls. GR-down's320 sequential K steps and partial DRAM reads can still dominate. R=4 is not automatically faster than R=8.

Useful product-materialization elimination, per rank and Q1 invocation:

| Role | FP32 product bytes | Avoided product write+read bytes | R=8 scratch bytes |
| --- | ---: | ---: | ---: |
| GR down | 3,932,160 | 7,864,320 | 1,536 |
| GR up | 3,276,800 | 6,553,600 | 40,960 |
| GDN out | 15,728,640 | 31,457,280 | 10,240 |
| Router | 1,310,720 | 2,621,440 | 2,048 |

These are traffic removed, not extra useful bytes for the functional denominator. Output tiling/typecast, the retained CCL, and all weight reads remain. At512GB/s the pure traffic saving is modest; a large benefit must come from higher output parallelism, avoiding separate product/reduction work and their layouts/launches, and reducing SFPU work on unnecessary rows. Measure this rather than claiming the removed bytes explain49.7ms.

The historical terminal profile cited in the pending AutoDebug prompt gives two GR reads about407.5/400.7us and GDN-out including gathers about200.65us in layer0. Those are entire segments, not removable projector costs, and are not the current113.88ms trace. Even zero-cost replacement of those whole segments would be an invalid savings budget. First collect or attribute the current selected trace per role; let `T_role` be the measured retained-baseline projector cost and `C_role` the measured candidate including assembly/typecast. Project `sum(count_role*(T_role-C_role))`, then remeasure the complete token-out path. Keep the remaining gap explicit. This candidate alone has no source-based guarantee of reaching64.14528ms.

## Matched real-weight experiment and promotion gates

1. Obtain immutable checkpoint-shard weights and actual inputs captured at each eligible projection from the selected generator, across several layers and tokens including router cutoffs. Use heap-backed `.clone()` before upload of mmap-loaded fixtures; see [upload pinning diagnosis](upload_pinning_diagnosis.md). Preserve source hashes, exact shard order, logical/padded dimensions and dtypes. Compare on mesh1x4 with the current weight objects, not re-quantized replacements or a mesh1x1 timing surrogate.
2. First bounded smoke: small O32/K32 and O96/K320 with BF16 and FP32 inputs/weights, R4/R8, one core then several; include distinct output-row ramps, FP32 mantissa-sensitive values, cancellation, zeros, and changing inputs. Save/flush phases before uploads, native submit, synchronize, and readback. Warm native execution first so a stall cannot be attributed ambiguously to the baseline. Do not change precision or arithmetic to make it run.
3. Compare native FP32 outputs against the selected device `multiply` plus accurate `sum` **before** BF16 conversion, and compare final BF16 bitwise. Cover real per-rank shapes96x10240,2560x320,640x6144,128x2560. Check physical output padding with a dedicated raw read if ordinary `to_torch` removes it. Preserve diagnostics for every mismatched row/bit; do not settle for PCC or rounded BF16 equality alone.
4. Compare Q>1 untouched dispatch at Q2,3,17,31,32; confirm a balanced projector stays on its old branch. Check gathered GR-down80-per-rank compaction and exact router logits, selected expert IDs, and scores. Full integrated accuracy is still mandatory after component equality.
5. Prepare all signatures, then capture separate selected and candidate traces with persistent input/output handles. Change input before every replay and check exact output; include back-to-back different inputs and alternating projector instances. Run trace allocation tracking including program-cache allocations and positive control, then a separate Watcher run. Audit no host conversion, host decision from tensor values, stale scratch, unsafe alias, or runtime fallback.
6. Time warmed repeated nonblocking CQ0 replay with one final synchronization, excluding host readback and fixture upload. Report standalone raw fusion, assembly, explicit typecast, and end-to-end projector medians separately; include unchanged CCL only when measuring the wrapper. Use identical input/weight allocation, mesh, clock/runtime environment, repetitions and comparison ordering. Limit knobs initially to R4/R8 and a small occupancy set such as min(jobs,24/48/available); avoid a dtype frontier.
7. Keep native only where the whole eligible projector wins with exactness. Refresh full-model AIME prefill and teacher-forcing accuracy, greedy/top-k/top-p token feedback, mixed prompts, inactive/fixed slots, non-aligned prompts, and trace replay. Remeasure full48 token-out throughput and the strict unchanged functional denominator. Leave separate teacher-forcing and token-out numbers. A component win is not stage acceptance.

A future focused `tests/probe_exact_projection_fusion.py` should require a fresh JSON output and external timeout and record command/source hashes plus phases. It does not exist as part of this plan. Reuse patterns, not assumptions, from [probe_fused_projection_cast.py](../../tests/probe_fused_projection_cast.py) and [probe_parallel_ordered_projection.py](../../tests/probe_parallel_ordered_projection.py). Device runs remain serialized by the parent; this document does not authorize concurrent hardware execution.

## Follow-up reader variant: initialize each CB page once

The implemented conservative v1 reader clears both full input pages at every K step and remains the default through the first component hardware result. The same reader now has a separate compile-time zero-once branch selected by `exact_projection_q1(..., zero_once=True)` and probe `--zero-once`; no hardware result is claimed. It removes repeated stores without changing arithmetic or kernel counts:

1. Before the job loop, reserve the entire two-page capacity of CB2 and CB3. Zero exactly both physical pages of each allocation, using its own BF16/FP32 byte size. This is initialization of reader-owned free storage, not a push or a second queue consumer.
2. Retain the existing per-step `reserve_back(1)` calls. `cb_reserve_back` waits for free capacity without advancing the write pointer; subsequent `push_back(1)` advances it. Do not push initialization pages. No compute thread sees either page before the initial full zeroing finishes.
3. At every K step overwrite all32 activation values in physical row0 and all R-by32 weight values in rows0..R-1, with the same four NOC reads and read barrier. Remove only the per-step clearing loops. Job changes reuse the same R and same destination row footprint.
4. Leave compute's per-job DST2 zero fill, masked multiply/add, SFPU finalizer, compact scratch zeroing, output tile zeroing, and all barriers unchanged.

Why the non-owned cells remain zero: only the reader writes CB2/CB3 memory. Its later NOC destinations cover exactly the activation row0 or R weight rows. Compute only unpacks those CBs; it packs into distinct CB4. Writer uses distinct CB5. Circular reuse changes the page pointer, not the fixed footprint. Both pages were initialized, and no actor writes the remaining cells thereafter. Consequently every consumed page has the same physical contents as v1, not merely the same valid rows. ROW broadcast receives all required row0 values every step. Reader buffers must remain disjoint from output/scratch, as in v1.

For BF16 activation plus FP32 weight this changes1536 scalar32-bit zero stores per K step into3072 stores per participating core per program invocation. With one GR-down R8 job/core and320 K tiles, that is160 times fewer zero stores. This is an instruction-count observation, not a latency prediction. Repeated programs/replays must initialize again; never rely on L1 contents surviving dispatch.

The existing `parallel_ordered_projection_reader.cpp` already initializes both ping-pong pages before pushing its first live tile, providing an API precedent. The new flag defaults to v1 and appears in the probe report and reader compile-time arguments (index3), so it has a distinct program-cache key. Source hashes include the reader and wrapper. CPU byte-level simulation checked5208 ping-pong states across both R values and all four BF16/FP32 input/weight combinations, including repeated page wraps: every physical byte matched conservative clearing. This is an address/ownership check, not a substitute for kernel compilation or device arithmetic validation. Required comparison: raw FP32 v1/v2/selected equality for both R sizes and all four dtype combinations in the small controls, then real role weights/inputs; use one core with many jobs to force repeated page wrap, multiple cores with unequal job counts, changed-input back-to-back replay, allocation tracking, and separate Watcher. Benchmark only after these pass.

Independent AutoDebug also identifies a Blackhole BinaryNg row-broadcast workaround (`binary_ng_program_factory.cpp:1450–1465`). The custom ROW broadcast therefore needs raw product/reduction bit checks for both BF16 and FP32 inputs; BF16 output equality alone is insufficient. The candidate already exposes FP32 output and reinitializes SFPU reduction immediately before its finalizer. If FP32 discrepancies appear, first isolate native broadcast/copy/product bits and compare with explicit reader row replication before changing arithmetic.

## Optional R2: legal with the existing four-row arithmetic band

R2 is now accepted explicitly by the candidate and `--rows 2`; defaults remain module R8 and probe `8,4`. The naive `ITERATIONS=rows/2=1` is **incorrect**, even though the C++ template accepts it. Both architectures' `hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_bcast.h:30–43` document each SFPLOAD as four physical rows by eight columns. Raw destination offsets0 and2 cover the two eight-column halves of that same four-row band. `ckernel_sfpu_add_top_row.h:40–48` confirms these correspond to SFPI indices0 and1. `tt-llk/.../llk_math_eltwise_sfpu_common.h:58–70` invokes VectorMode::R on top faces0 and1, so a one-iteration callback would leave half the columns in each face uncomputed.

The implemented R2 uses the unchanged R4 two-iteration multiply/add footprint: real weight rows0/1, initialized-zero rows2/3. DST2 starts zero, both real rows see every lane and exactly the original K-tile sequence, and the standard final reducer still reduces each row independently. Only the two valid reduced words enter scratch. No new SFPU predicate or R1 mask is introduced. Both normal and zero-once readers already preserve the two unused weight rows byte-for-byte. Nonfinite payload behavior remains outside the validated contract; all future real-input checks still require finite exact raw outputs.

R2 source row groups are even and never straddle a16-row face. The two BF16 fragments are64 bytes each; FP32 fragments128 bytes. Both source face offsets and destination offsets are aligned to64 bytes, satisfying Blackhole DRAM-read alignment64 (`hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`) and Wormhole32 (`.../wormhole/noc/noc_parameters.h:295`). Assembly reads16 disjoint scratch pages per32-output tile; a sole owner writes that full tile, so output packing and zeroing remain race-free. GR-down now has48 jobs for padded O96, compared with24 for R4 and12 for R8. There is no reduction in per-K SFPU instructions relative to R4; extra activation reads, scratch pages, and full-tile finalizers can offset the lower per-worker weight traffic and increased occupancy. Only hardware comparison can establish a gain.

## Probe-only input widening and optional R16

`probe_exact_projection_q1.py --precast-input` widens the persistent BF16 query to FP32 inside the candidate callable only when its resident weights are FP32. Both native programs and the final explicit BF16 output cast remain in the callable. Consequently warm, exact raw comparison, changed-input capture/replay, and timing include the input widening and its transient allocation. The selected reference is unchanged. The report records request/application flags, actual native input dtype, and the expanded candidate timing scope. This tests exact widening and the FP32 ROW-broadcast route; it does not establish that reconfiguration instructions disappeared. Current `reconfig_data_format(2,2)` and `reconfig_data_format_srca(3)` are unconditional new-operand overloads. A later conditional old/new-operand optimization requires its own source-state proof and matched run.

Optional `--rows 16` is now exposed with default `8,4` unchanged. Eight SFPI iterations cover the complete top16 rows in each top face. The reader's16-row group begins at physical row0 or16 and reads exactly one full256-element fragment per face; it never crosses a face. The writer's16 reduced values reside at offsets `16*r`, r0..15, all in face0 column0. They fit the existing32-word scratch page. Two scratch pages assemble each32-output tile under one writer. All CB counts, page initialization, sequential K arithmetic, explicit cast and race protections remain unchanged. GR-up has160 jobs at R16, enough to occupy the target mesh cores while reducing input rereads and per-job initialization relative to R8; per-job SFPU work doubles. GR-down has only6 jobs and may lose occupancy. Hardware evidence, not that tradeoff alone, determines per-role acceptance.
