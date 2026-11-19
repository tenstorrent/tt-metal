# AutoDebug: stage-7 full-model FUNCTIONAL speed floor

Inspection: 2026-11-16, HEAD `ece17d228671dbc1375e45c6e73f90156707aa30`. `M/` below denotes `models/autoports/qwen_qwen3_6_35b_a3b/`; other paths are repository-relative. This investigation opened no device, ran no target workload, changed no source, and did not relaunch AutoDebug. Only this report was written. Three xhigh subinvestigators inspected exact frontend arithmetic, recurrence, and collective lowering independently.

## Findings

**The stage-7 floor remains unmet. There is no demonstrated single flag or previously qualified substitution that closes it. The least invasive plausible repair sequence is to remove redundant collective/recurrent movement, then optimize the existing exact projection scans and exact recurrent contractions. Hardware evidence is still needed to establish a combined 3.45 ms saving.** This is a repair investigation, not a waiver recommendation.

Two concrete, newly actionable discrepancies are present in the selected path:

1. Each singleton direct all-reduce pads an already physically tiled row to logical 32 rows and slices it back. The selected C++ direct operation does not require logical row 32. The surrounding operations account for about **0.494 ms** of extrapolated reduced-profile kernel work across all 40. Removing them is a strong narrow probe, not a proven runtime win.
2. The output recurrence contraction performs an HC permutation that its immediate caller reverses. Removing only that output pair preserves the mathematical result and exposes about **0.142 ms** of extrapolated kernel work across 30 linear layers. Padding and the next norm/SiLU consumer must still be checked.

The larger remaining opportunities are substantive but policy preserving:

- The exact projection reader fetches full padded activation tiles while using only 64 bytes, and reconstructs coefficients repeatedly across neighboring output-column runs.
- The exact projection scan stores every prefix, while its selected consumer reads only the final tile's final row. A terminal-only store variant can retain every FP32 addition, first-tile reset and carry operation. This is distinct from rejected native GEMM substitutions.
- The exact recurrent contractions expand a 512 KiB product into a 2 MiB padded HC tensor twice per linear layer. A direct-layout contraction, followed if necessary by a state-slab recurrence kernel, targets this movement without using native FDR/TF32 arithmetic.

**Persistent CCL alone is not supported as the answer.** All 80 decoder reductions already share persistent scratch/semaphores. The source-matching reduced profile extrapolates their complete kernels to about 0.946 ms/all 40. Their padding, input movement and following adds provide additional smaller work categories; none establishes the missing 3.45 ms as collective overhead. The earlier compensated scalar norm fusion, specialized selector and tiled SiLU improvements are already spent.

## Direct observations, target, and completion boundaries

`M/doc/optimized_full_model/before.json` records all 40, TP4 Blackhole, B1/context1024, an actual 128-token activation fixture and 127 timed decode steps. These are saved observations, not reproductions performed here.

| Observation | Recorded value |
|---|---:|
| Queued token production | 25.081180984 ms/token |
| Caller-observed token delivery | 25.158932228 ms/token |
| Separate model-only diagnostic | 24.601165787 ms/token |
| Separate sampler-only diagnostic | 0.487795354 ms/token |
| Separate 99-step HF-input model timing | 24.599893303 ms/token |
| Historical isolated layer estimate | 30×0.719504474 + 10×0.340836807 = 24.993502290 ms |
| Fresh queued minus historical estimate | 0.087678694 ms |

The historical estimate is **not a measured decomposition or mathematical lower bound** on the fresh workload. The separate model/sampler controls use different token/state streams. Do not subtract them to assign exact component costs. Their agreement supports investigation inside the decoder rather than a large host wrapper gap.

A CPU-only execution of the actual `M/tests/multichip_functional.py::byte_model` reproduces the topology audit:

| Convention | Mean useful bytes/rank/token | 10% latency ceiling |
|---|---:|---:|
| Positions128..255, 128 steps | 1,107,527,616 | 21.631398750 ms |
| Actual timed positions 128..254, 127 steps | 1,107,522,496 | 21.631298750 ms |

The latter includes decoder972,427,200 B, local head135,086,080 B, useful embedding row1,024 B and terminal norm weight8,192 B. At 512 GB/s/rank, queued efficiency is **8.62451%**. The actual127-step gap is **3.449882234 ms**, requiring13.7549% lower queued latency. Assigning it only to the 30 linear layers would require **114.996074 µs saved per linear layer**. Caller-observed delivery is another 0.077751244 ms slower, and remains the user-facing headline metric in this harness.

`run_full_model_performance.py:481-555` synchronizes before timing, queues complete trace pairs with `blocking=False`, and synchronizes before stopping the timer. `generator.py:560,618-623` executes model then sampling on CQ0. Thus the queued number includes completed work, not merely enqueue time. There are 127 model and 127 sampler replays, zero resident-loop token/position/page/activity refreshes, zero readbacks/captures, stable 8452 program-cache entries, and post-loop position/token checks. The separate delivery measurement includes 127 readbacks. The default loop already places `device_only()` outside the clock; the old stage 5 audit-context timer discrepancy is not a remaining fix here.

**Reporting discrepancy:** `before.json` says `PASS`, but the runner sets this after its execution/ownership assertions (`run_full_model_performance.py:715-716`) without evaluating this useful-byte floor. That PASS establishes successful benchmark execution, not stage 7 FUNCTIONAL acceptance. A subsequent evidence checker should record the calculated floor independently and fail it until met; preserve the raw baseline. Likewise, the variable/comment calling24.9935 ms a decoder lower bound is only a historical reference estimate.

The stage 5 owner decision accepts only the recorded stage 5 linear speed limitation (`M/doc/optimized_multichip_decoder/owner_decision.md`). It does not change this target or qualify the rejected numerical candidates.

## Evidence identity and numerical contract

All nine source hashes listed in fresh `before.json` match the inspected files, including `tt/model.py`, `tt/generator.py`, `tt/multichip_decoder.py` and the performance runner. The current mesh decoder hash is `d08412bdbd7bfbb2bcdef2974b1c3f5b924c8a7ffca4590be80316997cf0bf1b`.

For attribution, the preferred artifact is `M/doc/full_model/performance/profile_reduced_v4_drains/`: its `performance.json` lists matching current model/generator/mesh sources. It contains real layers0/3 plus full embedding/terminal/sampler,16 measured replays, and226 model operations/rank/replay. It is an instrumented **reduced** profile, not all 40 clean timing. Native program hashes, ordinal sequence, dtype/layout and source establish the selected operators.

The retained stage 5 profile at `M/doc/optimized_multichip_decoder/tracy/layer_{0,3}/` is secondary. Its inventory is130/80 decode operations, superseding the older AutoDebug177/79 inventory. Its `multichip_decoder.py` hash differs from current, while its fused decoder, compensated norm, selector and SiLU helper hashes match. The recurrence methods' ASTs also match the older committed methods. Use it for the identified unchanged operator chains and corroboration, not as a new whole-model timing result. The uninstalled experimental exact-recurrence helper has changed identity since some saved experiments; its current source is explanatory evidence, not the exact measured binary.

Preserve replicated BF16 residuals, BFP8/HiFi2 ordinary projections, BFP8/LoFi active experts, BF16 KV/CCL, the accepted FP32-input linear-partial collective exception, FP32 recurrent state, ordered sensitive BF16 projection scans, compensated norms, exact SiLU and the selected router behavior. No broad datatype sweep or native-family restoration follows from this report.

## 1. Collective boundaries: remove materialization before changing transport

Selected call sites are `multichip_decoder.py:1519,1533-1540`; singleton `decode_batch` reaches them at `:1595-1596`. There are 30 FP32 linear-attention partial reductions,10 BF16 full-attention reductions,40 BF16 MoE reductions, one embedding AG and two sampler candidate AGs per token. Replicated norms introduce no inter-layer gather.

`CollectiveWorkspace` already owns serial-CQ0 mutable scratch (`multichip_decoder.py:786-800,1087-1109`; `model.py:70-78`):0.5 MiB BF16 and1 MiB FP32/rank, plus one direct-AR semaphore. The input/output width shards are32×64 on32 cores; scratch shards are32×256. Retaining a new Python output object cannot eliminate per-token allocation work during already captured replay.

The explicit-scratch overload at `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async.cpp:541-572` invokes the direct primitive. Its device validator requires width sharding, aligned pages and sufficient scratch, **not logical row32** (`device/all_reduce_async_device_operation.cpp:19-74`); output specs preserve input logical shape (`:77-84`). Its factory derives work from physical shard tiles (`device/all_reduce_async_program_factory.cpp:203-228`).

In contrast, the Python caller converts to DRAM, pads logical 1→32, converts to the AR L1 shard, runs AR, then slices32→1 (`multichip_decoder.py:1228-1251`). The narrow probe is to retain logical 1/physical 32 through the same shard, scratch, dtype, rank reduction and output layout. Validate poisoned padding and exact valid-row outputs through the next norm. Removing padding does not shrink the collective's physical payload; it removes the two materialization operations.

The matching v4 rank 0 sequence gives:

| Two boundaries per layer | Linear µs | Full µs | 30 linear +10 full, ms |
|---|---:|---:|---:|
| Direct AR kernels | 25.693313 | 17.556938 | 0.946369 |
| FillPad + Slice | 12.432500 | 12.116813 | 0.494143 |
| Immediately supplying copies/conversions | 4.679250 | 4.101438 | 0.181392 |
| Following residual adds | 4.279750 | 3.455875 | 0.162951 |

AR ordinals are89/136 for linear and180/215 for full in model-only replay. These are per-rank kernel-work measurements extrapolated by layer count. They are not wall savings, and deleting all categories is impossible. Slice firmware spans include waits associated with preceding AR completion; do not price those spans as independent slice overhead.

A second probe removes the forced DRAM bridge for a resident producer one boundary at a time, charging any necessary reshard. The entire supplying movement category is only 0.181 ms. Preserve projection geometry and arithmetic.

### The actual direct algorithm and fences

The direct factory binds the persistent scratch CB and the operation's output CB, then selects `worker_reader.cpp`, `worker_writer.cpp`, `reduction_receiver.cpp`, and `reduction.cpp` (`all_reduce_async_program_factory.cpp:401-532`). Output allocation occurs during eager dispatch/capture and is retained for trace replay. It **all-gathers full payloads into rank-major scratch, then reduces locally**, rather than using the separate composite RS+AG path:

- Writer rank offset and line multicast: `device/kernels/dataflow/worker_writer.cpp:117,125-145`.
- Receiver waits for all four contributions, exposes data, then resets semaphore: `device/kernels/dataflow/reduction_receiver.cpp:31-36`.
- Compute adds rank pairs0+1 and2+3 before BF16 pack: `device/kernels/compute/reduction.cpp:41-60`.

Each rank receives3S payload:384 KiB for a BF16 physical 32×2048 input or768 KiB for FP32. These are payload calculations, not measured wire counters. A conventional optimized-AR estimate of192 KiB in the stage 5 topology ledger does not describe this direct implementation. Correcting that description must not inflate the useful-byte floor.

Persistent buffers do not remove connection setup,16 shard-read barriers per sender, per-destination write flushes, connection drain or semaphore handshakes (`worker_reader.cpp:49-64`; `worker_writer.cpp:84-110,133-146,172-179`). One-versus-two links is a bounded single-variable probe; staging independent shard reads before fewer barriers is a later C++ probe requiring a concrete CB/packet lifetime proof. No present semaphore/count mismatch or deadlock was established.

Do not reapply the reverted FP32 SFPU arithmetic repair during a transport test: the selected C++ still uses its accepted `add_tiles` path. The previous repair's component oracle did not close the complete decoder contract (`M/doc/optimized_multichip_decoder/AUTOFIX_fp32_all_reduce.md`).

## 2. Exact projection scans: movement and unused prefixes

Production selection is `multichip_decoder.py:633-685` for packed input/output and inherited `fused_decoder.py:688-709` for the ordered router. The current-source v4 rank 0 kernel work is 97.222688/42.369625/24.812313 µs, totaling **164.404625 µs per linear layer**. This is a large relevant category, but even a69.9% deletion would be required if these scans alone supplied114.996 µs.

| Projection | Local weight shape | Runs/rank | K32 iterations/run | Max runs/core |
|---|---|---:|---:|---:|
| Packed input | [4,1,512,3136] | 392 | 16 | 4 |
| Ordered output | [2,1,512,2048] | 128 | 16 | 2 |
| Ordered router | [4,1,512,256] | 32 | 16 | 1 |

The existing balanced contiguous schedule is `fused_decoder.py:2466-2476,2798-2799`. Each reader iteration reserves weight/coefficient/raw pages, reads a BF16 weight tile and activation tile, waits, constructs the pair-flipped coefficient column, then publishes matched weight/coefficient pages (`DIRECT_COEFFICIENT_READER`,`:2779`). Compute consumes one matched pair, uses FPU BF16 products into FP32DST, performs Default cumsum with reset only on the first tile of a K512 run, and packs only at the final tile (`STREAMING_SCAN_COMPUTE`,`:2558`). Writer consumes one final tile/run and copies physical row31 into a zero-padded logical row (`:2560`). The caller retains the existing `fast_reduce_nc` over K512 partials. Products, pair ordering, block grouping and final rounding all matter.

### Probe A: narrow activation reads, with identical coefficient construction

The reader fetches 2048 bytes but consumes only BF16 elements 0..15 and 256..271: two 32-byte half rows at byte offsets 0/512. Input/output/router source-issued activation bytes are12,845,056 /4,194,304 /1,048,576 per rank/layer. Two 32-byte reads reduce these to401,408 /131,072 /32,768; two 64-byte reads are a separate alignment/performance control.

Land both halves at the same scratch offsets and retain the pair-flip loop. Blackhole read alignment is 64 bytes (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:374-380`); the sanitizer checks source/destination low-address congruence (`tt_metal/hw/inc/internal/debug/sanitize.h:705-755`). Do not compact the second half into a destination with different low bits. Input/router activation metadata is DRAM interleaved; the output scan's activation is L1 width-sharded. These counts are source-issued traffic, not measured DRAM transactions or proof of a bandwidth bottleneck. Doubling transaction count can lose despite fewer bytes.

### Probe B: cache coefficients per worker/K512 block

Neighboring output-column runs reuse the same 16 coefficient tiles. A 16-page BF16 cache costs 32 KiB/core versus the current 4 KiB coefficient CB. With the current partition, coefficient constructions/tile reads fall:

| Projection | Current | Per-worker block reuse |
|---|---:|---:|
| Input | 6272 | 1792 |
| Output | 2048 | 1760 |
| Router | 512 | 512 |

Input has 112 worker/block groups, including two assignments crossing a K512 boundary. Output has 110. Router has one run/core and gains no local reuse under this schedule. Cache group changes explicitly; initialize every reused page's padding. Preserve run-major compute order: changing to Ktile-major execution would overwrite the single carried scan state. Cache-fill barriers and full-depth L1 headroom must be charged. Narrow reads and reuse should first be measured separately.

### Probe C: terminal-only destination stores, retaining the exact scan

`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cumsum.h:15-97` loads four vectors, transposes, replays the same adds/NOPs, transposes back, and stores four vectors for each of eight passes. Carry lives in LREG 4-7. The eight pass address sets are disjoint:

```
{0,2,16,18}  {4,6,20,22}  {8,10,24,26}  {12,14,28,30}
{32,34,48,50} {36,38,52,54} {40,42,56,58} {44,46,60,62}
```

No subsequent pass loads a prior pass's stored prefix. The selected writer reads only final-row words 752..767 and 1008..1023. Retaining all four stores in pass 7 of the final tile is a conservative superset of what it needs. Thus 508 of 512 SFPSTORE invocations/K512 run have no selected data consumer:199,136/input,65,024/output and16,256/router per rank/layer.

This is a **data-dependence argument, not a completed hardware timing proof**. Keep both transposes, all adds/arithmetic NOPs, first-tile reset, wrapper waits and per-tile init. `ADDR_MOD_7` has zero increments (`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h:75-87`); the SFPU wrapper sets destination/waits and clears it afterward (`llk_math_eltwise_sfpu_common.h:17-26`). Nevertheless, removed stores also supplied issue spacing. First replace them with four SFPNOPs to isolate data dependence, then shorten only after instruction-hazard/assembly review and bit-equal stress. Restrict the variant to terminal scan; products-only mode requires its full products.

The removed stores are 22.05% of nominal scan-body instruction slots, **not 99% of scan execution**. FPU, loads, transposes, replay adds, initialization and queue waits remain. Cumsum init hoisting is not part of this first probe because intervening FPU initialization can own relevant configuration. No hardware gain is claimed. No prior terminal-store-elision experiment was found.

## 3. Exact recurrence: fix the contraction layout, not the rejected adapter

The selected MRO reaches `LinearTP4Mixin._recurrent_step/_recurrent_sum` (`multichip_decoder.py:366-414`); production does not install `mesh_exact_recurrence_probe`. The baseline `max_batch_size == 1` capture is unmasked (`generator.py:508-510`) and commits directly to persistent state (`multichip_decoder.py:497-508`). One active request inside a larger fixed-slot capture still uses its masked path. Local state is FP32 [1,8,128,128], 512 KiB/rank/layer; four local Q/K heads each serve two value heads.

The selected sequence is:

```
D      = FP32(BF16_round(S) * decay)
m      = reduce128(FP32(D * k))
delta  = FP32(FP32(v - m) * beta)
Snew   = FP32(D + FP32(k * delta))
qscale = FP32(q * float32(128**-0.5))
o      = reduce128(FP32(Snew * qscale))
```

Here reduce128 is eight contiguous groups of 16 accumulated in order, then an ordered sum of the eight partials. It starts from the first operand, rather than inserting an initial zero. `ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc.cpp:37-56` processes the requested dimensions in descending order; its `device/kernels/reduce_nc.cpp:43-55` copies the first tile and adds subsequent tiles using SFPU. The same operation's `device/fast_reduce_nc_program_factory.cpp:285-302` selects direct FP32 destination unpack. Preserve these numerical boundaries, decay-LUT input and BF16-on-consumption state semantics. The live newly computed FP32 state contributes to the current output before the next decode rounds its consumption.

Current v4 native program hashes identify24 recurrence operations, with rank 0 total 107.066875 µs (other ranks 107.009188/107.073250/106.945625):

| Work | Operations | Kernel µs/rank 0 |
|---|---:|---:|
| Operand permutations/casts/reshapes | 6 | 21.514625 |
| Both reduction/layout sequences | 10 | 42.173500 |
| Other pointwise work + final permutation | 8 | 43.378750 |

Products and reductions here are **L1**, not DRAM. Each `_recurrent_sum` HC-permutes [1,8,128,128] into logical [1,128,8,128], physical [1,128,32,128], quadrupling 512 KiB to 2 MiB. First/second reduction outputs are128 KiB/16 KiB; a slice and HC permutation return128 KiB physically. Two contractions move approximately9.9 MiB of L1 tensor data under simple read/write accounting, excluding product creation. The padding-aware HC writer issues sub-tile and padding writes (`ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp:125-154,174-202`). Do not price that movement against DRAM bandwidth.

The smallest probe is an output-layout option for `_recurrent_sum`: its final HC permutation at `:373` is immediately undone by `:414` for the output contraction at `:403`. Return its sliced[1,1,8,128] directly for that call. Keep the memory contraction at `:399` head-major. The next actual consumer is BF16 cast→reshape→norm→SiLU (`:553-564`), so compare that whole consumer chain and padding. The two current-profile inverse permutations account for4.737781 µs/linear layer,0.142133 ms/all 30.

Next replace **only the contraction** with a direct-layout kernel consuming native products. Preserve16-then-8 additions and first-stage parallelism; a naïve four-core reducer would serialize work now distributed over32 first-stage cores. The entire existing two-contraction chain is 42.1735 µs/layer,1.265205 ms/all 30, before replacement cost.

If that is insufficient, a distinct full recurrence design can use32 independent workers: eight heads×four 32-column slabs, each owning [128,32] state,16 KiB. Keep decayed/new state local, carry out both exact contractions locally, then write persistentFP32 state and final output directly. This removes product tensors, padded HC materializations, global partials and a reread of newly written state. Local lane rearrangement must reproduce the current FP32 additions and prevent FMA contraction/TF32 narrowing. Batched masked state commit remains a separate required behavior. This is plausible kernel work, not a qualified drop-in implementation.

### Why the old exact/native experiments do not settle this proposal

The experimental adapter `mesh_exact_recurrence_probe.py:294-297` retains both original contractions and their padding. Its current reader constructs two full-tile broadcasts in phase1 and five in phase2 by scalar BRISC loops (`:123-162`):917,504 scalar32-bit stores,3.5 MiB, per recurrence/rank. Source-issued input reads total 4.25 MiB, and four full state/product outputs add 2 MiB before the retained reductions. This provides a concrete reason to redesign distribution instead of repeating pointwise-only fusion.

The saved 48/48 equality result covers the recorded component cases, not universal equivalence. The 0.869184612 ms whole-layer experiment also changed FP32 CCL, versus retained default 0.719504474 ms; attributing its entire regression to recurrence alone would be unsupported. Its saved helper identity differs from the current experimental file. Native recurrence also changes arithmetic, including an exp(log(decay)) adapter and FPU contractions. Its layer34 diagnostic anomalies remain unqualified. None of these controls rejects a new explicit-order direct-layout contraction; none proves it fast.

## 4. Norm fusion is incremental, not the old unspent windfall

Current singleton selection calls `normalized_singleton` (`multichip_decoder.py:1267-1275`; `mesh_compensated_norm.py:213-242`). It retains the square/reduction, terminal scan, compensated scalar epilogue and separate FP32 normalization, followed by multiplication with FP32 gamma and a BF16 output pack. Gamma is uploaded as FP32 (`fused_decoder.py:354-356`); a fusion must not narrow it. The scalar epilogue already encodes the explicit high/low BF16 rounding and sqrt/reciprocal refinements (`mesh_compensated_norm.py:56-134`).

The retained stage 5 rank 0 profile gives 78.107625 µs for both norm cores,82.406063 µs with surrounding conversions. Its **two scalar epilogues total 10.680438 µs**, and two terminal scans 12.885438 µs. The earlier 109.1825 µs unfused scalar chain is gone. Repeating that old gain estimate would materially overstate the remaining opportunity.

Two narrow follow-ups are still plausible: append the unchanged epilogue to `NormTerminalScan`, preserving exact packed-bit scalar extraction; or combine the final wide normalization and gamma products while retaining explicit intermediate FP32 rounding and finalBF16 pack. The existing final multiply operators total 24.202938 µs across both norms, not a promised saving. Each needs complete conversion-inclusive timing. A generic RMSNorm/rsqrt substitution would reopen rejected numerical policy and is excluded.

## Ranked probes and a falsifiable path to the target

Run these serially under the parent's device ownership. The ranking balances small intervention, clean falsification and potential contribution; it is not permission to claim projected gains.

| Priority | Single-variable probe | Discriminator / required evidence |
|---|---|---|
| 1 | Direct AR logical 1/physical 32, remove pad/slice | Same valid-row results, padding isolation, shard/scratch/reduction unchanged; complete next-norm timing improves. About0.494 ms/all 40 kernel work is exposed. |
| 2 | Remove output-only inverse HC pair | Exact recurrence and following BF16 norm/SiLU outputs, padded geometry and repeated state agree. About0.142 ms/all 30 kernel work is exposed. |
| 3 | Two half-row activation reads, no compute change | Exact coefficient bytes, products, K512 partials and final projections; compare32/64-byte transfers and complete scan chain. |
| 4 | Terminal-only stores, first with retained NOP spacing | Prove data independence before shortening instruction spacing; retain every arithmetic instruction/carry/reset and examine Blackhole hazards. |
| 5 | Per-worker K512 coefficient cache | Same run order and partials; quantify input reuse, cache-fill barriers and full-depthL1 census. Router has no current-schedule reuse benefit. |
| 6 | Direct-layout exact recurrent contraction | Same16-then-8 reductions on captured products; eliminate padded HC materialization while retaining useful parallelism. |
| 7 | Full exact state-slab recurrence, if needed | Preserve explicit operation boundaries and commit semantics; beat complete current recurrence/next consumer, not just the old adapter. |
| 8 | Remaining exact norm fusion; resident producer→AR bridge; link count | Isolate each. Smaller measured categories; no combined attribution without a fresh profile. |
| 9 | Explicit embedding/sampler AG buffers/worker override | Ownership-safe component win, small terminal contribution; cannot replace decoder repair. |

The parent concurrently owns terminal geometry work. The now-existing `M/doc/optimized_full_model/head_r3_b8_c16.json` records reduced real-layer 0/3 terminal 0.451993750→0.367280890 ms with identical full-vocabulary logits and unchanged BFP8/HiFi2/FP32 accumulation policy. That **0.084712860 ms component result** is not all 40 validation and cannot close the floor alone.

For planning only, even assuming the two small movement deletions converted their kernel work 1:1 into wall savings, and the recorded head component gain carried over, the remaining requirement would still be **2.728893 ms**, or **90.963094 µs per linear layer**. This arithmetic is an optimistic allocation of the target, **not an additive prediction**. A new scan/recurrence combination must deliver most of that in measured complete-consumer timing. For example, a 90+ µs/linear improvement distributed across the 164.4 µs scan category, 107.1 µs recurrence category and smaller norm remainder is conceivable; no current artifact demonstrates it. Recurrence kernel work alone is already below the full 114.996 µs/layer gap allocation. The historical stage 5 firmware interval of 124.245 µs would leave only about 9.249 µs for a replacement if recurrence alone closed the full gap. Kernel work and firmware intervals are different accounting measures, not a current all 40 wall bound; neither supports a recurrence-only promise.

Stop a component when its discriminator fails. If small reader/store/reduction probes provide insufficient savings, the honest next intervention is an exact kernel redesign with preserved policy, not rejected native defaults, another broad dtype sweep, an increased useful-byte denominator or a stage 5 waiver applied to stage 7.

## Persistent AG ownership and other potential issues

Explicit persistent AG outputs do have a real lowering effect: `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.cpp:635` disables its initialization barrier when persistent buffers are supplied. The remaining connection/completion fences stay. The v4 token-out sampler AG pair is 22.5675 µs; the sampler-only pair 25.84775 µs. Its embedding AG entry reports 15.785188 µs but shape[64,512], while current singleton preparation implies physical[32,512]; do not silently assume correspondence for that entry.

A simple `SamplerCCL` dictionary cache is unsafe without a lifetime repair: common `models/common/sampling/tt_sampling.py:1497-1498` explicitly deallocates sampling/gathered values, and `ttnn.deallocate` defaults to force=True (`ttnn/cpp/ttnn-nanobind/operations/core.cpp:185-189`; `ttnn/core/tensor/tensor.cpp:125-139`). A retained Python reference does not protect that persistent allocation. Adjust ownership/deallocation before returning shared values buffers, and keep all live traces' storage separate as needed. This is a candidate-implementation hazard, not a diagnosed baseline failure.

No current source violation proving missing experts, invalid state ownership, a producer/consumer queue mismatch or a collective hang was found. The observed failure is a speed-floor miss. Expensive exact arithmetic plus materialization explains the source-level opportunity; actual compute/NoC/dispatch dominance remains a measurement question. Do not infer bandwidth saturation from the useful-byte percentage.

## Required validation and claim review

Start with identical real captured operands and current-code component contrasts. The existing `tests/v6_direct_coefficient_stream_probe.py:410-430` exposes coefficient/product/partial/final comparisons. For recurrence, compare both contractions, the FP32 state before/after its consumption-rounding boundary, final output and next norm/SiLU. For CCL, compare every rank's valid output with controlled padding. Include changing-input/changing-position trace replay, cancellation/tie-sensitive cases and the existing sensitive layer 34/long-stream checks; keep aggregate and diagnostic criteria as previously defined rather than inventing a per-step 0.995 gate.

After independently passing components, measure the complete selected decoder and all 40 resident token loop with fresh hashes, no profiler/watcher/allocation-tracking contamination, boundary synchronization, unchanged 127-step accounting, no added host refresh/readback, stable traces/program cache and preserved precision/distribution. Report queued and delivered performance separately. A same-regime clean all 40 number must actually meet 21.631298750 ms for this convention; component equality or profiler-work subtraction cannot establish acceptance.

The first draft was rechecked against current Python dispatch, exact reader coefficient addresses, the native Blackhole cumsum carry/address contracts, direct-AR validation/factory/kernel semantics, fast-reduction order, inverse-permutation consumers, saved negative experiments, and current versus historical source identities. Claims requiring hardware remain explicitly probes. In particular, unused prefix stores are not called hardware-safe merely from data dependence; recurrence products are correctly identified as L1; the pointwise adapter's whole-layer regression is not assigned solely to it; and no historical scalar-norm gain is counted twice. The stage 7 floor remains unresolved pending a measured policy-preserving repair.
