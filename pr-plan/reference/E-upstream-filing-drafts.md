# Front E-upstream-filing-drafts (design/recon swarm, 2026-08-17)

## Verdict

All six deliverables are drafted and every technical claim was verified against the branch (file:line cited). Highlights from verification: (1) the gather repro is committed at tests/ttnn/unit_tests/operations/data_movement/test_gather.py:265-298 (commit 809cf5bda41), and upstream's only "multi-core RM" gather case (test_universal_input_tm_gather.py:404-412) never reaches RmSingleRowMultiCore because its index width is 1024 while selection keys on index width > 1920 — so the buggy factory is still untested upstream; (2) the 079021d6c12 fix is an 8-line JIT-only kernel change, PR draft ready; (3) PerfConfig.run() confirmed to never write variant_stimuli (perf/core.py:748-800) while TestConfig.run() does (test_config.py:1661), with an in-tree root-cause docstring at test_cgtceq_perf.py:363-374; (4) the MATH_PACK leak is fully documented in-tree (cgtceq_perf.cpp:1104-1115, CGTCEQ_RUNBOOK.md:118-128) and the deadlock mechanism verified against llk_math_common.h:119-160; (5) the wrong SFPSWAP comment exists on upstream main at ckernel_sfpu_topk_xl.h:519 ("puts max into b" = VD, opposite of SFPSWAP.md's VD=min for mod1=1), plus a second wrong comment in ckernel_sfpu_exp.h:228; ROW_2/3_MAX=5/6 (should be 7/8, Quasar has 7/8), ENABLE_ACC_STATS 45(BH)/46(WH)/stale-62, and the packer zero-run/histogram findings all located with silicon evidence in SORTING.md §A/§C; (6) the replay-LOAD hang plan is 5 experiments, cheapest-first, log-archaeology before any device time, explicitly stopping at axis attribution — not root cause. Nothing was filed, committed, or run on hardware.

## Plan

# FRONT E — Upstream filing drafts + housekeeping triage

All drafts verified against branch `nkapre/sorting` (HEAD 636d1f0abb9). Nothing filed; these are ready-to-file texts. Note for draft 6: debugging.md loaded — state codes are: CWFW=CB Wait Front Wait (blocked on cb_wait_front), CRBW=CB Reserve Back Wait (blocked on cb_reserve_back), UPMD=Unpack Math Done, MWDD=Math Wait Dest Done.

---

## DRAFT 1 — Issue: `ttnn.gather` ROW_MAJOR multi-core variant returns wrong values (index width > 1920)

**Target repo:** tenstorrent/tt-metal
**Suggested labels:** `bug`, `op_cat: data movement`, `silent-data-corruption`, `blackhole` (observed on BH p150a; the suspected mechanism is arch-independent NoC alignment, so likely WH too — untested)

### Title
`ttnn.gather` (ROW_MAJOR, interleaved): RmSingleRowMultiCore returns real-but-wrong values when index width exceeds 1920

### Summary
`ttnn.gather` on ROW_MAJOR interleaved tensors selects `RmSingleRowMultiCore` whenever the **index** width exceeds `GATHER_WT_THRESHOLD(60) * TILE_WIDTH(32) = 1920` (`ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp:17,27-31`). On silicon, that variant returns values that are real elements of the input row but at the wrong positions. Widths at or below 1920 (single-core variant, strictly-greater selection) are correct.

### Reproduction
Pytest repro (self-contained, no topk involved) — available as `tests/ttnn/unit_tests/operations/data_movement/test_gather.py::test_gather_row_major_interleaved` on our branch (commit `809cf5bda41`, happy to PR the test):

```python
@pytest.mark.parametrize("index_width", [1024, 1920, 2048])
def test_gather_row_major_interleaved(index_width, device):
    torch.manual_seed(0)
    input_shape = [1, 1, 32, 8192]
    index_shape = [1, 1, 32, index_width]
    input = torch.randn(input_shape, dtype=torch.bfloat16)
    index = torch.randint(0, input_shape[-1], index_shape, dtype=torch.int64)
    torch_gather = torch.gather(input, -1, index)
    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.ROW_MAJOR, device=device)
    ttnn_index = ttnn.from_torch(index, ttnn.uint32, layout=ttnn.Layout.ROW_MAJOR, device=device)
    ttnn_gather = ttnn.gather(ttnn_input, -1, index=ttnn_index)
    assert_allclose(torch_gather, ttnn.to_torch(ttnn_gather))
```

**Expected:** all three widths match `torch.gather`.
**Actual (BH p150a silicon):** 1024 and 1920 pass (`RmSingleRowSingleCore`); 2048 fails (`RmSingleRowMultiCore`) with real-but-wrong gathered values.

### Suspected mechanism (from static reading; the failure itself is silicon-confirmed)
In `gather_reader_rm_single_row_multi_core.cpp`, each core fetches its per-core index slice from the interleaved row page at byte offset `w_start * input_index_elem_size` (`kernels/dataflow/gather_reader_rm_single_row_multi_core.cpp:43,56-61`) into an **aligned** CB base. For uint32 indices, `w_start * 4` is not NoC-read-aligned for most cores (per-core `w_start` comes from `split_work_to_cores(core_grid, W_index)`, `gather_program_factory.cpp` `RmSingleRowMultiCore::create_descriptor`), so the read data lands shifted in L1 and the core gathers real-but-wrong elements. The writer has the mirror-image hazard on the output side: it writes its output slice at byte offset `w_start * output_elem_size` (`gather_writer_rm_single_row_multi_core.cpp:42,66-71`) — 2-byte granularity for bf16.

### Test-coverage gap (why this shipped)
At the time we hit this (2026-08-16), upstream had **zero** ROW_MAJOR gather tests: `tests/ttnn/unit_tests/operations/data_movement/test_gather.py` and `tests/ttnn/nightly/.../test_gather.py` are TILE-layout only (verified: 0 occurrences of ROW_MAJOR on main in both). The newer `tests/ttnn/nightly/unit_tests/operations/data_movement/test_universal_input_tm_gather.py` (on main since 2026-08-06) adds RM cases, **but its one "Multi-core RM" case never reaches the buggy factory**: the case at lines 404-412 uses input W=2048 / index W=1024 with the comment "Multi-core RM (W > 60 * TILE_W = 1920)" — yet factory selection keys on the **index** width only (`gather_device_operation.cpp:27-29`), and 1024 ≤ 1920 routes to `RmSingleRowSingleCore`. So `RmSingleRowMultiCore` remains untested upstream today.

### Workaround
Chunk wide gathers into slices of ≤1024 index columns and concat (we shipped this in a consumer before deleting the gather dependency entirely; see `git show 809cf5bda41:ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp` lines 242-257 for the as-landed workaround with constants `large_k_route_gather_max_index_width = 60*32`, `large_k_route_gather_chunk_width = 1024`).

### Suggested fixes
1. Reader: read the containing aligned window (round `w_start` down to NoC alignment) and index into the CB at the intra-window offset — or use an alignment-aware read helper.
2. Writer: same treatment for the output slice write.
3. Fix the misleading comment + widen the `test_universal_input_tm_gather.py` multi-core RM case to index width > 1920 so the multi-core factory is actually exercised (it would currently fail).
4. Adopt the attached direct repro as a unit test (1024 / 1920 / 2048 boundary triplet).

---

## DRAFT 2 — PR: `ttnn.topk` multi-core: fix silent value corruption for >32 flattened rows

**Target repo:** tenstorrent/tt-metal
**Suggested labels:** `bug`, `op_cat: reduction`, `silent-data-corruption`
**Local commit to upstream:** `079021d6c12` (8 lines added, JIT-only kernel change, no host change)

### Title
ttnn.topk multi-core: fix silent value corruption for >32 flattened rows (missing per-ht unpacker reconfig in topk_final.cpp values-gather)

### PR description

**Problem.** Every multi-core `ttnn.topk` call with more than 32 flattened rows (Ht > 1) returns corrupted **values** — fabricated ~1e38 numbers with heavy duplication in every row past the first height tile. Indices are unaffected. Plain `dim=-1` calls hit it whenever the multi-core factory engages at Ht > 1.

**Root cause.** `topk_final.cpp`'s per-`ht` aggregation loop (`ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_final.cpp:84`) begins each iteration with a values-gather (`copy_tile` from `input_dfb`) but had no unpacker reconfig at the top of the loop. At `ht == 0` the datacopy state inherited from `init_sfpu` is correct. At `ht >= 1` the state left by the **previous** iteration's final call — `transpose_and_pack(index_transposed_dfb_index, ...)` at `topk_final.cpp:174` — is TRANSPOSE mode with the INDEX (UInt16/UInt32) SRCA format. The bare `copy_tile` then unpacks the gathered bf16 values as garbage.

This is the standard `*_with_dt` reconfig rule biting a shipping kernel: `copy_tile_init` / state-configure alone performs no unpacker data-format reconfig; when the previous op left a different SRCA format, `copy_tile_to_dst_init_short_with_dt` (+ `reconfig_data_format_srca`) is required. The sibling kernel already does exactly this (`topk_local.cpp:187-188`).

**Fix.** Mirror `topk_local.cpp:187-188` at the top of the ht loop, before the values gather:

```cpp
reconfig_data_format_srca(input_dfb_index);
copy_tile_to_dst_init_short_with_dt(index_transposed_dfb_index, input_dfb_index);
```

(now at `topk_final.cpp:92-99` with an explanatory comment). JIT-only; no host rebuild, no program-factory or hash change.

**Why no test caught it.** No in-tree test ever ran the multi-core topk factory at Ht > 1; the corruption is invisible to any single-height-tile test matrix.

**Validation (BH p150a silicon).**
- Standalone Ht=2 repro: `dim=-1`, shape (1,1,64,8192), k=32 — rows 32-63 corrupt before the fix, 64/64 rows clean after.
- Full topk contract matrix 72/72 (first all-green run; the failing cell was gates-dim1).
- `tests/ttnn/unit_tests/operations/reduction/test_topk.py` suite: 220 passed.
- Downstream topk_large_indices suite: 154/154.

**Suggested regression test to include:** a multi-core topk case with ≥64 flattened rows asserting values (not just indices) against torch.

---

## DRAFT 3 — Issue: `PerfConfig.run()` silently ignores `variant_stimuli` (never written to L1)

**Target repo:** tenstorrent/tt-llk
**Suggested labels:** `test-infrastructure`, `perf`, `bug` (or `footgun/documentation` if maintainers consider it by-design)

### Title
PerfConfig.run() accepts variant_stimuli but never writes it to L1 — data-dependent perf kernels silently measure stale device memory

### Summary
`PerfConfig.__init__` accepts `variant_stimuli` (`tests/python_tests/helpers/perf/core.py:614`) and forwards it to `TestConfig.__init__` (`perf/core.py:645`), so a test author reasonably expects stimuli to be loaded before the kernel runs. But `PerfConfig.run()` (`perf/core.py:748`) overrides `TestConfig.run()` and its per-run loop calls only `self.write_runtimes_to_L1()` + `self.run_elf_files()` (`perf/core.py:799-800`) — `variant_stimuli.write(...)` is never called. `TestConfig.run()` does write it (`tests/python_tests/helpers/test_config.py:1654-1661`). `write_runtimes_to_L1` writes only tile sizes / format enums (`test_config.py:814-829`), not stimuli.

### Impact
- Any perf kernel whose **timing is data-dependent** (early-exit, skip logic, compression, count-dependent loops) measures whatever the previous test left in L1 — numbers are wrong and nondeterministic across suite orderings, with no error.
- Any **self-checking** perf arm counts stale data. Observed concretely (BH silicon, 2026-08-16): self-checking threshold-count arms all read count = 2048 ("everything above a negative threshold", i.e. zeros/junk), reproducibly, until the driver wrote stimuli itself.
- The failure is invisible for pure-throughput kernels with data-independent timing, which is why it has survived.

### Expected / Actual
- **Expected:** passing `variant_stimuli=StimuliConfig(...)` to `PerfConfig` results in that data being in L1 when the kernel executes (as it does for `TestConfig`).
- **Actual:** parameter is accepted, threaded through hashing/init, and never written to the device.

### Repro (no hardware needed to see the gap)
Code inspection: diff `TestConfig.run()` (`test_config.py:1635-1670`) against `PerfConfig.run()` (`perf/core.py:748-807`). On hardware: run any perf test whose kernel reads its input buffer after a functional test that dirties the same L1 region; observe the perf kernel consuming the leftover data.

### In-tree precedent
Our branch carries the root-cause note and workaround as a docstring: `tests/python_tests/test_cgtceq_perf.py:363-374` — the driver calls `configuration.variant_stimuli.write(TestConfig.TENSIX_LOCATION)` itself before the run loop (guarded on `BuildMode.PRODUCE`).

### Suggested fix (either, plus docs)
1. In `PerfConfig.run()`, before the `run_count` loop: if `self.variant_stimuli` is not None, write it once (per variant) — mirrors `TestConfig.run()` including the `STIMULI_MODE` handling.
2. If timing-only-by-design is intended: raise / warn when `variant_stimuli` is passed to `PerfConfig`, and document that perf kernels must not depend on L1 input content.
Option 1 is strictly safer: it preserves current timing for data-independent kernels (write happens outside the timed region) and fixes data-dependent ones.

---

## DRAFT 4 — Issue: math-side `_llk_math_dest_section_done_` with idle packer leaks a MATH_PACK semaphore token that deadlocks all later tests on an un-reset device

**Target repo:** tenstorrent/tt-llk
**Suggested labels:** `test-infrastructure`, `documentation`, `deadlock`, `LLK`

### Title
Leaked MATH_PACK semaphore token from a math-only kernel persists across tests on an un-reset device; `_llk_math_pack_sync_init_` deadlocks before it can SEMINIT

### Summary
A kernel that calls `_llk_math_dest_section_done_` while the pack thread is idle posts a `semaphore::MATH_PACK` token that nothing consumes (`tt_llk_blackhole/llk_lib/llk_math_common.h:119-129` — `set_math_semaphores()`). The tt-llk test harness re-runs kernels `run_count` times and runs many tests back-to-back **without any device/semaphore reset**, so the token persists. The next `_llk_math_pack_sync_init_` then spins forever in its pre-init drain loop `while (semaphore_read(semaphore::MATH_PACK) > 0)` (`llk_math_common.h:140-145`) — the `TTI_SEMINIT` that would zero the count sits *after* the wait, so it is never reached. Observed cascade: Math wedges in INIT, Unpacker wedges at a `mailbox_read` ("waited 2 seconds for Math, Unpacker"; Pack completes) — and **every subsequent test on the un-reset device fails**, so the first failure report points far away from the culprit (pure cross-test contamination).

### Concrete case (BH silicon, 2026-08-16)
First consumer run of a new perf suite: 12 passed / 39 failed / 1 error — all but the first failure were contamination from one leaked token. Full write-up in-tree on our branch: `tests/sources/CGTCEQ_RUNBOOK.md:118-128`; the fixed kernel documents the invariant at `tests/sources/cgtceq_perf.cpp:1104-1115` (fill arms deliberately skip `_llk_math_dest_section_done_`; the dest-half flip it would perform is re-established by the next run's `_llk_math_pack_sync_init_` via `reset_dest_offset_id` + `StartZero`, so skipping is state-clean).

### Expected / Actual
- **Expected:** a hung or ill-formed test fails alone; the harness (or LLK) restores sync state between tests.
- **Actual:** one unbalanced section-done poisons the device session; failures surface in unrelated later tests, including pure-timing arms.

### Suggested fixes (any subset)
1. **Docs (cheapest, highest value):** doxygen on `_llk_math_dest_section_done_` in `llk_math_common.h` stating the pairing contract — every math-side section-done must be consumed by a pack-side `_llk_pack_dest_section_done_`; posting with an idle packer deadlocks the *next* `_llk_math_pack_sync_init_` on this or any later kernel until hard reset.
2. **Harness hardening:** between tests (or on test-start), detect a nonzero MATH_PACK count and either SEMINIT-clear it from a known-idle state or fail loudly naming the previous test as the leaker. The current failure mode ("waited 2 seconds for Math, Unpacker" in an innocent test) is maximally misleading.
3. Optionally a debug-build assert in `_llk_math_pack_sync_init_` after N spin iterations, printing the semaphore count.

---

## DRAFT 5 — Doc-reports bundle (5 items): SFPSWAP direction comments, ROW_2/3_MAX encodings, ENABLE_ACC_STATS index, packer zero-run counter, packer exponent histogram

These are five small, independent reports. Items 5a/5b → **tt-llk** (one item also touches tt-metal); 5c → **tt-metal**; 5d/5e → **tt-isa-documentation**. Silicon evidence measured on BH p150a; local provenance SORTING.md §A/§C (branch `nkapre/sorting`) with in-tree test kernels `tests/sources/pack_exp_histogram_test.cpp`, `pack_exp_histogram_perf.cpp`, `pack_zero_compress_test.cpp`, `pack_zero_compress_perf.cpp` (cherry-pickable).

### 5a — SFPSWAP mod1=1 direction: two in-tree comments state the inverse (code is correct; comments mislead)
**Target:** tt-llk (primary) + tt-metal (one comment). **Labels:** `documentation`, `LLK`.

Ground truth (ISA doc + silicon, both agree): `SFPSWAP` mod1=1 (`SFPSWAP_MOD1_VEC_MIN_MAX` / LLK `p_sfpswap::ALL_ROWS_MAX`) puts **min into VD (lreg_dest) and max into VC (lreg_src_c)** — `tt-isa-documentation/BlackholeA0/TensixTile/TensixCoprocessor/SFPSWAP.md` functional-model table ("In all lanes, VD = min and VC = max"); mode 9 is the VD=max variant ("no enum currently defined for this mode"). Confirmed on BH silicon during a bitonic-kernel bring-up (chunk-skip accumulator; comment with the measurement at `ttnn/.../topk_large_indices/device/kernels/topk_large_indices_chunk_skip.hpp:235-236` on our branch). A correctly-worded shipping comment for cross-reference: `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h:607` ("group A max in LREG0", LREG0 being src_c).

Two shipping comments say the opposite:
1. **tt-llk** `tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_topk_xl.h:519` (upstream main; 809 on our branch): "All swap directions are encoded by the choice of operand order to `SFPSWAP` — `(a, b, MAX)` puts max(a, b) into `b` and min(a, b) into `a`." With `TTI_SFPSWAP(imm, lreg_src_c, lreg_dest, mod1)` (`ckernel_ops.h:832`), `b` is lreg_dest=VD — the comment claims VD=max, inverting the ISA/silicon behavior. The kernels are correct (direction encoded self-consistently by operand order); anyone using this comment as the SFPSWAP reference will build an inverted network.
2. **tt-metal** `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h:228`: "SFPSWAP (mode VEC_MIN_MAX = \"max into lreg_dest\")" — the parenthetical is wrong; the very next line's concrete mapping ("LREG1 = max…", LREG1 being src_c) is right.

Also worth a line in the same report: the enum **name** `ALL_ROWS_MAX = 1` (`tt_llk_blackhole/common/inc/ckernel_instr_params.h:430`) reads as "dest gets max" and likely seeded both comments.

**Suggested fix:** flip the two comment sentences; optionally add a `VD_MAX = 9` / `ALL_ROWS_MAX_INTO_VD` enum for mode 9 (it exists in silicon and the ISA doc, and is what a macro-scheduled compare-exchange needs since a macro forces VD=macroVD — see `ckernel_sfpu_binary_max_min.h:152` already using bare literal 9 with the comment "mod1=9 means set VD=max and VC=min").

### 5b — `p_sfpswap::ROW_2_MAX` / `ROW_3_MAX` encodings are wrong on WH and BH (silently duplicate ROW_0/1)
**Target:** tt-llk. **Labels:** `bug` (latent), `LLK`.
`ROW_2_MAX = 5`, `ROW_3_MAX = 6` on BH (`tt_llk_blackhole/common/inc/ckernel_instr_params.h:436-437`) and WH (`tt_llk_wormhole_b0/common/inc/ckernel_instr_params.h:379-380`) — identical to `ROW_0_MAX`/`ROW_1_MAX` (5/6). Correct values are 7/8: ISA `SFPSWAP.md` mode table (`SFPSWAP_MOD1_SUBVEC_MIN2_MAX013 = 7`, `SUBVEC_MIN3_MAX012 = 8`), sfpi (`runtime/sfpi/include/blackhole/sfpi_hw.h:271-272`), and Quasar's own tree has it right (`tt_llk_quasar/common/inc/ckernel_instr_params.h:550-551`). Latent — no in-tree user today — but the first user gets a silent wrong-subvector swap. Fix: change 5/6 → 7/8 on both arch trees.

### 5c — `ENABLE_ACC_STATS_Enable_ADDR32` differs across arch (45 BH vs 46 WH) with a stale third value (62) in a comment block
**Target:** tt-metal. **Labels:** `documentation`, `hazard`.
BH: `tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h:1154` → 45. WH: `tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines/cfg_defines.h:1455` → 46. A third value `ENABLE_ACC_STATS 62` sits inside an out-of-date commented-out block in `blackhole/tensix.h:459` (same stale 62 in `wormhole/tensix.h:375` and `tt-2xx/quasar/tensix.h:461`). Hazard: code (or a human) porting a WH accumulator-stats poke to BH with the WH index pokes the wrong config register **silently** (verified on BH silicon: the feature only engages at index 45). Ask: a one-line comment at both `cfg_defines.h` sites flagging the cross-arch difference, and delete/annotate the stale 62 blocks.

### 5d — Packer zero-run counter direction on Blackhole: counts zeroes PRECEDING the datum; WH page says "after"; BH has no compression page
**Target:** tt-isa-documentation. **Labels:** `documentation`, `blackhole`.
`WormholeB0/.../Packers/Compression.md:3` says the 4-bit counter records "how many zeroes appear **after** that datum". Measured on BH silicon (by construction: raw group of a stride-16 pattern was datums `[v0,v1,v2,v3,0]` with nibbles `[0,15,15,15,14]`; only the "before" reading reconstructs the source): on BH the counter is the zeroes **before** its datum. The layout otherwise matches the WH page exactly (uint16 row-start-index array, groups of 32 augmented datums + 32 nibble counters). A decoder written to the documented semantics is bit-perfect on symmetric patterns (all-zero, dense, front-loaded) and garbage on asymmetric ones — worst-case failure mode. Framed precisely: BH silicon differs from the WH page **and BH has no packer-compression page at all**, so the ask is a BlackholeA0 Compression page (or an arch-difference note on the WH page). Evidence kernels: `pack_zero_compress_test.cpp` / `pack_zero_compress_perf.cpp` (our branch). Local provenance: SORTING.md A1 (:481-492).

### 5e — Packer exponent histogram on Blackhole: sampled 1-in-8 (fixed pattern), not per-datum; plus three adjacent doc gaps
**Target:** tt-isa-documentation. **Labels:** `documentation`, `blackhole`.
`Packers/ExponentHistogram.md` describes the histogram as incremented per datum. Measured on BH silicon (`pack_exp_histogram_test.cpp`, our branch): every 1024-datum tile yields exactly **128** increments, format-independent (fp32 also 128 → per-datum rate, not per-16B-beat), with the sampled positions being the fixed pattern `p mod 64 < 8` (proved by construction: marking block 0 gives {31:128}, blocks 1-7 give {31:0}; a mod-8 phase sweep is flat, ruling out stride-8). Adjacent findings worth one page edit each (all measured, SORTING.md A4-A6):
- `WhichPackers` is ignored on histogram read modes 6/7 — all four reads return the identical array; summing them 4x-counts.
- Max-exponent mode 9 is subsampled too (a single exp-132 outlier among 1023 exp-127 datums is missed), where WH documents an unconditional per-datum update.
- `CLREXPHIST` issued from the PACK thread does not fence in-flight PACRs (reproducible ~39-count leak: 168 instead of 128); issuing it from the math thread ordered by the dest semaphore gives exactly 128.

---

## DRAFT 6 — Triage plan: replay-LOAD monolithic-battery hang (NOT a root-cause effort)

**Where this lives:** internal (branch housekeeping), no upstream filing yet — upstream only if E2-E4 attribute it to stock machinery.

### Known facts (verified sources)
- Hang occurred **2/2** when the full measurement battery ran **monolithically under Tracy** with the replay-LOAD arm active; **0 hangs in ~3,040 isolated per-cell subprocess runs** and 0 in the 90-cell A/B (per-cell subprocess). Recorded in TOPK_LEDGER.html:337-338 ("Open hazard: replay-LOAD monolithic-battery hang (2/2 under full-battery Tracy, 0/~3,040 isolated; STORE ships on its own clean record)") and cross-referenced in `paper-topk/evidence/validate/contract-debug.md:127-129`.
- The replay STORE+LOAD windows are default-on since `d64993c3273` (`tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h:73-78`; opt-out `-DTOPK_DISABLE_REPLAY_STEP`); the canonical harness flips arms via a guarded header edit + JIT-cache clear (`tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py:24-42`), and its per-cell-subprocess design is the standing workaround.
- Hard constraints: Watcher/DPRINT/Device Profiler share on-device SRAM — **Watcher cannot run under Tracy**; single device, flock discipline, never pipe device pytest through `| tail`.

### The 5 experiments, cheapest first (stop rule: axis attribution + one captured state snapshot; do NOT root-cause)

**E1 — Log archaeology (zero device time).** Mine the two historic full-battery Tracy logs / report dirs for the exact cell index and (shape,k,arm) at which each hang stopped; check whether the two positions match; confirm from the logs that LOAD was genuinely the active arm both times and re-verify the STORE-arm record is clean. Output: "same-cell deterministic position" vs "different-cell cumulative-state" prior, and the candidate prefix for E5. If the logs no longer exist, E2 regenerates them.

**E2 — Reproduce on current HEAD (1 battery run).** Re-run the exact monolithic battery under Tracy (LOAD default-on, fresh `tt-smi -r`, flock held, per-cell start/end stamps flushed to an un-piped log). The tree has changed massively since the 2/2 record (hybrid factory, I2/I3/I4, the topk_final fix) — if it no longer reproduces in 2 attempts, downgrade to watch-item and stop (the per-cell workaround already ships). If it reproduces: record the cell index; this is the anchor for E3-E5.

**E3 — Tracy axis (1-2 battery runs).** Same monolithic battery, same order, **no Tracy wrapper** (plain python, device profiler off). Hang persists → Tracy exonerated AND Watcher becomes usable (E5 path A). Hang gone in 2 runs → the device-profiler interaction is load-bearing (profiler buffers/dispatch under long single-process sessions); E5 path B.

**E4 — Arm axis (1 battery run).** Monolithic battery under Tracy with `-DTOPK_DISABLE_REPLAY_STEP` via the harness's guarded header-edit mechanism (JIT cache cleared, header restored on exit). Hang gone → the replay windows (slots 16-31 overlap discipline, `ckernel_sfpu_topk.h:81-100`) are implicated and this becomes upstream-relevant; hang persists → "LOAD" in the name is a coincidence of battery composition and the suspect shifts to session-cumulative state (program-cache growth, context count — cf. the known `MAX_CONTEXT_COUNT (32)` long-battery failure class).

**E5 — One state capture (1 short run).** Using the shortest hanging prefix from E2's cell index (order-preserving; one optional halving if cheap): (path A, hang survives without Tracy) rerun under `TT_METAL_WATCHER=10` — profiler off, mutually exclusive — let it hang, read `generated/watcher/` per-core `BRISC,NCRISC,TRISC0,TRISC1,TRISC2` state codes (CWFW/CRBW/K pattern vs NoC stalls vs all-cores-same-state) **before** `tt-smi -r`; (path B, Tracy-only) capture the host side instead — py-spy/gdb stack of the hung process + `tt-smi -s` scrape — to split host-side wait from device wedge. Deliverable: one snapshot + the axis verdict, filed as the input to a future root-cause session.

Budget: ≤6 battery-scale runs worst case, 0 if E1 finds the logs contradict the premise, 2 if E2 fails to reproduce.


## Evidence

- tests/ttnn/unit_tests/operations/data_movement/test_gather.py:265-298 (branch repro test, commit 809cf5bda41; 1024/1920/2048 boundary triplet)
- ttnn/cpp/ttnn/operations/data_movement/gather/device/gather_device_operation.cpp:17,27-31 (GATHER_WT_THRESHOLD=60; RM multi-core selected iff W_index > 60*32)
- ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_reader_rm_single_row_multi_core.cpp:43,56-61 (index slice read at byte offset w_start*elem_size)
- ttnn/cpp/ttnn/operations/data_movement/gather/device/kernels/dataflow/gather_writer_rm_single_row_multi_core.cpp:42,66-71 (output slice write at w_start*elem_size)
- tests/ttnn/nightly/unit_tests/operations/data_movement/test_universal_input_tm_gather.py:404-412 (upstream 'multi-core RM' case uses index W=1024, never reaches RmSingleRowMultiCore)
- git show main:tests/ttnn/unit_tests/operations/data_movement/test_gather.py — 0 ROW_MAJOR occurrences; nightly test_gather.py — 0 ROW_MAJOR occurrences
- git show 809cf5bda41:ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp lines 242-257 (chunk<=1024 workaround as-landed)
- commit 079021d6c12 — 8-line diff to ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_final.cpp
- ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_final.cpp:84,92-99,171-175 (ht loop, fix, trailing transpose_and_pack(index_transposed...))
- ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_local.cpp:187-188 (mirrored reconfig pattern)
- tt_metal/tt-llk/tests/python_tests/helpers/perf/core.py:614,645,748,799-800 (PerfConfig accepts/forwards variant_stimuli; run() never writes it)
- tt_metal/tt-llk/tests/python_tests/helpers/test_config.py:814-829,1635-1670 (write_runtimes_to_L1 writes no stimuli; TestConfig.run() writes variant_stimuli at :1661)
- tt_metal/tt-llk/tests/python_tests/test_cgtceq_perf.py:363-374 (in-tree root-cause docstring + workaround)
- tt_metal/tt-llk/tests/sources/CGTCEQ_RUNBOOK.md:118-128 (MATH_PACK leak bring-up finding)
- tt_metal/tt-llk/tests/sources/cgtceq_perf.cpp:1104-1115 (fill-arm comment: deliberately no _llk_math_dest_section_done_; state-clean argument)
- tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_common.h:119-129,140-160 (_llk_math_dest_section_done_ posts MATH_PACK; _llk_math_pack_sync_init_ spins while(semaphore_read(MATH_PACK)>0) BEFORE TTI_SEMINIT)
- tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_topk_xl.h:809 (branch) / main:519 — wrong comment '(a, b, MAX) puts max(a, b) into b'
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h:228 (wrong parenthetical 'max into lreg_dest')
- tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_reduce.h:607 (correctly-worded comment: max in src_c)
- ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/topk_large_indices_chunk_skip.hpp:235-236 (silicon-verified direction comment on branch)
- tt-isa-documentation/BlackholeA0/TensixTile/TensixCoprocessor/SFPSWAP.md functional model: VEC_MIN_MAX -> 'In all lanes, VD = min and VC = max'; mode 9 -> 'VD = max and VC = min; no enum currently defined'; :105-110 auto-stall note
- tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_instr_params.h:427-437 (p_sfpswap; ROW_2_MAX=5, ROW_3_MAX=6 duplicating ROW_0/1)
- tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/ckernel_instr_params.h:379-380 (WH same 5/6); tt_llk_quasar/common/inc/ckernel_instr_params.h:550-551 (Quasar correct 7/8)
- build_Release/libexec/tt-metalium/runtime/sfpi/include/blackhole/sfpi_hw.h:264-272 (SFPSWAP_MOD1_VEC_MIN_MAX=1; SUBVEC_MIN2_MAX013=7, SUBVEC_MIN3_MAX012=8)
- tt_metal/hw/inc/internal/tt-1xx/blackhole/cfg_defines.h:1154 (=45); tt_metal/hw/inc/internal/tt-1xx/wormhole/wormhole_b0_defines/cfg_defines.h:1455 (=46); blackhole/tensix.h:459 + wormhole/tensix.h:375 (stale 62 in commented-out block)
- SORTING.md:481-517 (A1 zero-run PRECEDING, A2-A6 packer findings), :572-585 (C1 ENABLE_ACC_STATS, C2 ROW_2/3_MAX, C3 mod1=9)
- tt_metal/tt-llk/tests/sources/: pack_exp_histogram_test.cpp, pack_exp_histogram_perf.cpp, pack_zero_compress_test.cpp, pack_zero_compress_perf.cpp (evidence kernels exist on branch)
- TOPK_LEDGER.html:337-338 (hang record: 2/2 full-battery Tracy, 0/~3,040 isolated; STORE clean)
- paper-topk/evidence/validate/contract-debug.md:127-134 (precedent + prescribed first action TT_METAL_WATCHER=10)
- tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py:24-42 (per-cell subprocess watchdog design; guarded header-edit arm mechanism)
- tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h:66-100 (replay defaults, TOPK_DISABLE_REPLAY_STEP opt-out, slot windows 16-31)
- commit d64993c3273 (default-on replay, measured 1.154x-1.202x, opt-out flag)

## Risks

- Draft 1 mechanism (NoC-unaligned per-core offsets) is a static-reading hypothesis; only the failure (2048 fails, 1024/1920 pass) is silicon-confirmed. The draft marks it 'suspected' accordingly — upstream may find a different byte-level cause.
- Upstream gather code may have moved since this checkout's main; the universal-input test analysis (multi-core RM case never fires) is as of main@cd0617a5d5b (2026-08-06). Re-verify against upstream HEAD immediately before filing.
- Draft 3: maintainers may declare PerfConfig timing-only by design (the in-tree docstring itself says so); the draft hedges with a warn/reject fallback so it lands as a doc fix at minimum.
- Draft 5 silicon claims (zero-run direction, histogram 1-in-8, WhichPackers, CLREXPHIST) come from campaign-authored test kernels on one BH p150a box; they were established by construction but not independently replicated — upstream doc maintainers may want the test kernels attached (they are on the branch and cherry-pickable).
- SFPSWAP item: SORTING.md:583 paraphrases mod1=1 as 'VD = min' consistently with the ISA doc, but the ledger notes 'the presumed SFPSWAP doc bug dissolved' — the ISA doc is correct and only in-tree comments are wrong; the draft is framed that way. Any filing that instead accuses SFPSWAP.md would be wrong.
- Draft 6 hang: n=2 for the positive class; the tree has changed substantially since — E2 may simply fail to reproduce, which the plan treats as a valid terminal outcome, not a failure of the plan.
- The 079021d6c12 PR needs an upstream-visible regression test (multi-core, >32 flattened rows, values asserted); the branch's contract suite is local-only, so the PR should carry a distilled pytest.
- No experiments were run and nothing was filed/committed, per the hard rule; all numbers quoted in drafts are from committed evidence (commit messages, SORTING.md, TOPK_LEDGER.html), not fresh measurement.
