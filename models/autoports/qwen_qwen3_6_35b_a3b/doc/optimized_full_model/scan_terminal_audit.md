# Ordered scan terminal-store hypothesis

CPU-only independent AutoFix hypothesis check, 2026-11-16. No production edits, device imports, hardware execution or performance claim. Target is the selected `DirectCoefficientStreamingScan`, not the rejected native GEMM projection family.

## Finding

The hypothesis is **false for DRAM/L1 prefix tensors but valid for internal SFPU-to-DST stores**. The selected streaming helper already avoids materializing intermediate K-tile prefix outputs. `STREAMING_SCAN_COMPUTE` packs only `h+1==ht`; `STREAMING_SCAN_WRITER` takes physical row31 from that final tile and writes it as logical row0. The output logical shape is `[blocks,1,1,N]`, not `[blocks,1,K,N]`. Repeating the earlier terminal-output fusion cannot remove those already-eliminated tensors.

However, `calculate_cumsum` calls eight `tile_columns_pass` operations on **every** input tile. Each pass loads four DST row vectors, transposes, executes the exact ordered additions, transposes back, and stores all four prefix vectors to DST. None of the first seven passes' DST results is read by a later pass; none of the first `ht-1` tiles is packed. Their results survive only in the running LREG4–7 carry. Removing these dead **DST writes**, while retaining both transposes and the same arithmetic/carry, is a new narrowly scoped candidate.

Source anchors:

- `../../tt/fused_decoder.py`: `DirectCoefficientStreamingScan`, literal `STREAMING_SCAN_COMPUTE`, `STREAMING_SCAN_WRITER`, `DIRECT_COEFFICIENT_READER`.
- `../../tt/multichip_decoder.py::LinearTP4Mixin.linear`: helper output feeds `fast_reduce_nc(dims=[0])`, reshape then BF16 cast for packed input, or FP32 DRAM partial for output projection.
- `../../tt/fused_decoder.py::FusedDecoder.linear`: ordered linear router uses the same helper with K512/N256, then unchanged block reduction to FP32 router logits.
- Repo `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cumsum.h::{tile_columns_pass,scan_tile_columns_pass,scan_tile_columns,calculate_cumsum,cumsum_init}`.
- Repo `tt_metal/hw/inc/api/compute/cumsum.h`: native SFPU unary-call wrapper, RC_custom mode and the prohibition on intervening SFPU operations while carry is live.

## Exact dependency proof and CPU check

For pass `p=0..7`, define `r=(p%4)*4+(p//4)*32`. Its four DST vector addresses are `{r,r+2,r+16,r+18}`. A standard-library CPU script enumerated these sets: all32 addresses are unique and all eight pass read sets are disjoint. Thus writing a prefix to DST in pass p cannot supply another pass's input. The arithmetic dependency is in LREG registers, not those stores.

The original tile pass performs these actions in order:

1. Four `SFPLOAD`s from the above DST addresses into LREG0..3 (even pass) or LREG4..7 (odd pass).
2. `SFPTRANSP` over the register arrangement.
3. Replay the original8 instructions: four ordered `SFPADD`s with their `SFPNOP`s. Even passes start from LREG7; odd passes from LREG3.
4. A second `SFPTRANSP` restoring the carried register arrangement.
5. Four `SFPSTORE`s back to DST.

The proposed omission is only step5 where output is not consumed. Both transposes stay; dropping the second transpose would corrupt carry. `cumsum_init`, first-run reset, FPU product formation, pair-swapped inputs, K512 partition, final block `fast_reduce_nc`, and cast/CCL boundaries stay unchanged. Default is FP32 DST accumulation, so no BF16 prefix-store rounding is being removed; the actual accumulation remains the original SFPADD instruction stream.

A hardware check is still required: source dependency reasoning does not prove instruction hazards, SFPU wrapper state, pack synchronization or numerical behavior on this binary. Do not label this universal exactness before those controls pass.

## Minimal probe design

Add a **test-local** alternate compute source/class first, using the original reader, writer, allocation shape, CB sizes, work partition, runtime arguments and config. Avoid a shared `cumsum` edit or model dtype/config change. Under `TRISC_MATH`, introduce distinct local template names implementing the same pass sequence with `emit` control. Retain native init/replay setup and invoke through the same `SFPU_UNARY_CALL(..., idst, VectorMode::RC_custom, first, emit)` convention, using a local calculate function.

Conceptual patch, not compiled source:

```cpp
// Same four loads, transpose, original replay, second transpose.
// Keep all stores for the final pass of the final K tile to avoid
// introducing a new row/face scatter or changing the writer.
if constexpr (PASS == 7) {
    if (emit) { original_four_stores(); }
}
```

Call with `first=(h==0)` and `emit=(h+1==ht)`. Every run must reset carry at h0, including cores owning multiple runs. `products_only` and `coefficients_only` diagnostics retain their original paths. For final pass7 keep all four stores initially; the writer's existing row31 extraction then needs no modification. Earlier rows of the final DST tile may remain products or old values, but the writer must continue to copy **only** row31 and initialize every padded output row to zero.

First discriminating check: compare original and candidate **partial FP32 terminal output bits** for each rank/run before `fast_reduce_nc`. This directly detects lost carry/store hazards without waiting for all40 accuracy. Use current real linear weights and saved real inputs for:

| Role | Exact target shape and partition |
|---|---|
| Combined linear input | `[4,1,512,3136]`, activation `[1,1,2048]`,392 runs, up to4/core on110 cores |
| Linear output projection | `[2,1,512,2048]`, activation `[1,1,1024]`,128 runs, up to2/core |
| Ordered linear router | `[4,1,512,256]`, activation `[1,1,2048]`,32 runs |
| Prefill-compatible direct input | `[2,1,1024,3136]` (and existing split1600/1536 fixture shapes if still reachable), original activation shape |

The class supports K512 and K1024, so do not use K512 as a synonym for decode: public singleton prefill can also select it. An initial integration may explicitly choose the new helper only from a decode phase flag; otherwise all reachable prefill uses require equivalence checks.

Validation order:

1. CPU AST/source assertions: original reader/writer and every arithmetic instruction unchanged, only local DST-store guards differ; no host fallback.
2. Exact helper bits for all three real decode roles, positive/negative/zero/cancellation inputs, and saved sensitive layer34 inputs. Include nonzero carry across all16/32 Ktiles and multiple runs/core. Zero padded output rows must match too.
3. Complete projection adapter FP32/BF16 outputs after unchanged block reduction, plus router IDs/scores on real near-ties. A component PCC alone is insufficient for an exactness claim.
4.100 warmed replays with changing inputs, original/candidate ordering and allocation tracker enabled with program-cache allocations included. Positive control remains active; warm all signatures before capture. Check that first-run reset prevents cross-run/core leakage.
5. Clean untracked traced timing for original and candidate under identical shapes/layouts. Profile the minimal helper if timing interpretation needs reader/compute/writer attribution. No full-model profiler run.
6. Only after bit checks and speed pass: integrate the isolated helper, then layer0 and sensitive layer34 changing-state comparisons, preserving actual policy and cache. Reuse expected outputs from the accepted path, not rejected native candidates. Finally rerun affected full-model accuracy/token feedback and selected-path performance gates.

## Performance opportunity and limits

The validated stage6 reduced native profile `../full_model/performance/profile_reduced_v4_drains/token_out_op_groups.csv` records these selected model GenericOp rows (B1, one real linear layer0 plus full layer3; physical K512, BF16 inputs, HiFi4 scan helpers):

| Rank | Combined input scan µs | Output scan µs | Ordered router scan µs | Sum µs |
|---|---:|---:|---:|---:|
|0|96.8655625|42.2995625|24.8570000|164.0221250|
|1|95.4905625|42.2991250|24.8550625|162.6447500|
|2|97.1350625|42.3088750|24.8301875|164.2741250|
|3|97.1214375|42.3250625|24.8084375|164.2549375|

Program hashes on rank0 are1301098237097590364 (combined),453056488307873026 (output); router is identified by N256/K512 and matching native metadata. Never sum parallel ranks to estimate latency. The max-rank sum164.274125µs ×30 layers = **4.92822375ms of attributed scan kernel work**. Eliminating *all* three scan operations would be the absolute available-work envelope, not a feasible store optimization or a clean-host latency upper bound.

At K512, existing cumsum does16 tiles ×8 passes ×4 stores =512 store instructions per run. Retaining only final pass7 leaves4, removing508 (99.21875% of these DST stores). Native pass instruction count is4 loads +2 transposes +8 replay instructions +4 stores =18. This removes508 of2304 inner-pass instructions =**22.0486% of that instruction-count proxy**. Applied optimistically to the entire164.274µs scan wall span, it suggests about36.22µs/linear layer, **1.087ms over30 layers**, but that assumes uniform cost and no reader/FPU/writer bottleneck. It is a prioritization estimate, not a measured speedup or strict elapsed-time bound. Actual gain can be near zero when another stage dominates. Init/replay programming, multiplication, reader traffic, pack/writer work and final reduction are untouched.

The historical full-model target gap is about3.45ms (25.0819→21.6314). This exact-store candidate alone is unlikely to close it on the instruction proxy. Pairing it later with independently verified coefficient reread reduction may help; do not conflate those hypotheses in the first probe. The reader currently rereads a2KiB padded activation tile per weight tile, so a reader bottleneck can hide the entire store gain.

## Prior candidates and numerical hazards

Earlier **accepted** terminal-scan and streaming fusions already removed full prefix tensors: `../fused_decoder/endurance/v6_terminal_projection_scan_device.json`, `v6_streaming_projection_scan_device.json`, and work-log entries2026-11-13 09:53–10:18. Their complete adapters/long layer34 controls established the current terminal-only tensor contract. This proposal keeps that contract and removes only dead internal DST writes; it must not claim those earlier wins again.

Earlier **rejected** native GEMM/norm/recurrence alternatives remain rejected: `../optimized_multichip_decoder/AUTOFIX_functional.md`, `repair_isolate_projection_l34.json` (HF diagnostic minimum0.988056859), `repair_corrected_native_both_l0.json`, `repair_dram_{native,hifi2,lofi}_l0.json`, and `repair_fp32_ar_native_exact_l34_hf.json`. They alter accumulation/product/normalization semantics; this proposal must not import their kernels, precision or faster measurements. Exact recurrence plus FP32 CCL was slower and also remains outside this hypothesis.

Specific hazards to reject before promotion:

- Replacing ordered SFPADD with a tree, GEMM, different FMA, compensated mode or SFPI expression is a numerical-policy change.
- The second transpose operates across carry registers; omit stores only, not transpose/replay or SFPNOPs.
- A new helper using SFPI temporaries may allocate/clobber live LREG4–7. Prefer direct instructions and the existing LLK calling convention.
- SFPU synchronization/init placement, load/store hazard spacing and `tile_regs_acquire/commit/wait/release` must remain correct after instruction removal; source reasoning alone cannot prove timing hazards absent.
- Store omission is safe only for unconsumed prefixes. General `ttnn.cumsum`, norm consumers and product-diagnostic outputs must keep their own contracts.
- Packing whole final DST is safe only because writer copies row31 and zeros all other output rows. Never expose partially uninitialized final DST as a whole logical output.
- Do not fold block reductions into the helper: partition/reset at K512 and final `fast_reduce_nc` order are part of selected arithmetic.
- Preserve per-rank sharding, signed zeros/NaNs at exposed outputs, BF16 pair swapping and FP32 output partial before CCL.

Resolution: exact-store-elision is **plausible, unmeasured**; full-prefix DRAM elimination is **already implemented**. A focused component probe, followed by real-sensitive-layer and full-model gates if successful, is the next discriminating action.

## Probe review correction (before promotion)

The first read32/read64/NOP hardware artifacts used 32 width shards `[32,32]` for the output activation. Production `TP4LocalSiluMultiply` uses eight row-major width shards `[32,128]` (`tt/multichip_decoder.py:89-106`). Those artifacts remain unchanged and their output-path results are limited to the former synthetic layout; no production promotion follows from them. The revised runner matches the production layout and records this limitation in each new artifact.

Device runs now require `--timing` (the existing switch that enables trace checks), actual position227 for every selected role, both baseline/candidate trace coverage, and four changed-input replays per trace. `--mode acceptance` requires allocation tracking1 and program-cache-skip0. `--mode timing` requires tracking0, skip0, and Watcher/profiler disabled; only this mode marks timings eligible. Component pass is unreachable without the required real227 replay coverage. Hardware reruns of the corrected output layout remain required.

## Isolated coefficient-cache candidate

`tt/scan_coefficient_cache_probe.py` retains one complete coefficient K block in CB1 per worker. Reader fills its16 K512 pages (32 for K1024), zeros all padded coefficient columns once per invocation, and publishes the block only after every activation read completes. Compute waits for the complete block, reads coefficient tile index `h`, and keeps the original run-major multiplication/scan order. It pops all pages only after its final local run in that block; the next reader fill reserves the same full depth and cannot overwrite live coefficients. Each new invocation rebuilds the cache, so changed persistent activation inputs remain visible on trace replay. This is an invocation-local L1 cache, not a stale cross-token tensor cache.

The current11×10 schedule gives input6272→1792 coefficient fills, output2048→1760, and router512→512. CB1 grows from4KiB to32KiB per K512 worker (+28KiB); K1024 grows to64KiB (+60KiB). Local L1 capacity, cache initialization time, loss of initial weight/read overlap and complete-model activation pressure must be charged. No throughput improvement is assumed from traffic arithmetic.

`tests/run_stage7_coefficient_cache_probe.py` reuses the revised scan harness and all its current-production oracles/trace/mode gates through an invocation-local helper substitution. It writes the exact generated C++ sources and additional source hashes/cache provenance. Variants `cache`, `cache32`, `cache64`, `cache32_nop` and `cache32_elide` separate cache-only and combined hypotheses. The base helper and production source are unchanged. CPU schedule producer/pop matching, generated-source assertions, fixture setup and Python compilation pass; hardware correctness and performance remain unverified. `scan_cache_cpu_setup.json` is CPU-only evidence.

## Cache initialization diagnosis and no-clear hypothesis

Parent-reported hardware results for zero-cleared caches were bit-exact under Watcher/tracking but uncompetitive: cache32 input112.1µs versus read32-only99.9µs, output65.0µs versus read32-only53.7µs, router55.7µs versus baseline35.85µs. These are investigative component observations, not accepted complete-model gains. The original cache initialization serially writes `ht*512` uint32 words per worker/invocation:8192 stores at K512,16384 at K1024, compared with1024 for the original two-page coefficient buffer. Router has no coefficient reuse and therefore pays this cost without a corresponding reuse benefit. Store-elision's lack of gain is consistent with a reader bottleneck, but does not prove this specific initialization loop accounts for all of it.

The selected coefficient consumer uses only column0, not its padded columns:

- `tt_metal/hw/inc/api/compute/bcast.h:178` selects `BroadcastType::COL` for both multiplication and unpack; `mul_bcast_cols_init` documents operand B as "column 0 broadcast" at line435.
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_unpack_AB.h:93-101` unpacks B once for each row of A faces, advancing B by its number of face columns. For a2×2 face tile it selects B faces0 and2; faces1 and3 are not selected.
- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h:134-142` selects `p_elwise::SRCB_BCAST_COL`. Within each selected B face the first column supplies every result column.

The complete consumed BF16 address set is `{(r//16)*512+(r%16)*16 | r=0..31}`. All32 values are assigned from the current activation before the cache is published. The other992 BF16 values/tile are physical padding and do not influence this selected operation. The coefficient diagnostic's logical shape ends in N1, so its valid output is likewise these32 values; its physical padded columns have no promised content. Products-only diagnostics still compare every logical FP32 product, and partial/final/changed-input trace checks remain mandatory.

New isolated variants `cache32_noclear`, `cache_noclear`, and `cache64_noclear` omit only the full-cache zero loop and its pointer declaration. Reserve/fill/push, all32 coefficient writes/tile, compute and block-pop ordering are unchanged. `cache32_poison` instead initializes all physical lanes to BF16 qNaN0x7fc1 before overwriting the32 consumed lanes. Its full product/partial/final exact checks are a deliberate hardware falsification control for the unused-padding claim. Poison mode is a correctness control, not a speed candidate. No-clear physical padding is unspecified; do not repurpose its raw coefficient buffers as dense tensors or claim all padded bytes are initialized.

CPU source-generation checks and fixture setup passed (`scan_cache_noclear_cpu_setup.json`). Hardware poison/no-clear correctness, clean timings, full-depth L1 headroom, and full-model regression evidence remain required; no promotion is implied by this source proof.
