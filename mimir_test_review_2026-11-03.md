# Mimir directed tests: can each one fail when its feature is broken? (2026-11-03)

Branch sagarwal/mimir_dram_tests, HEAD a330d5dcaaa. Scope: every test in tests/tt_metal/tt_metal/dram/ (181 tests,
counting each TEST/TEST_P once and the MM-7 parameterized instances one by one). Review only: nothing was built or run,
and nothing has run on the emulator since 2026-10-29. The earlier review (mimir_test_review.md, HEAD 16585a9c68a) is
left as it was; 57 commits since then moved, merged, deleted and rewrote tests, so every file:line here is new.

Update 2026-11-03: fix 3 (staging poison and readback in five tests) is now committed as 6ea8f29c6e4 on top of HEAD.
`mimir_tests` builds in build_Debug; the tests have not been run. Those five tests have a "Working tree: OK" note
below, with their HEAD verdict kept. Every file:line in this review is still at HEAD.

How it was done: one reviewer read the shared helpers and CCE kernels first (findings I1-I13 below). Ten reviewers then
read one area each, given the plan, the README, the shared findings and the earlier verdicts to re-check. I re-read
in source every DEFECT and the verdict flips listed under Confidence. Your answers of 2026-11-03 are applied (see
Decisions).

File shorthand: paths are under tests/tt_metal/tt_metal/dram/ unless they start with tt_metal/ or test_kernels/
(= tests/tt_metal/tt_metal/test_kernels/).

## Verdict classes
- DEFECT: the test, or a leg of it, cannot fail by design, or passes a fault in the very behaviour it is named for
  although it already holds the data to see it. Also DEFECT: a leg that asserts behaviour the hardware does not have
  (fails on a correct implementation).
- WEAK: it can fail, but some named part is not exercised, or the fault that passes is one no test of that kind can
  see (same-path host oracle; contention or overlap never forced).
- OK: gates on an independent oracle for what it claims.
- CONDITIONAL: depends on an open hardware fact (H1-H9) or on an intent question you could not answer; both outcomes
  are given.

## Decisions you made (2026-11-03) and how they are applied
1. MM-5/MM-6 route evidence: data placement plus alias decoys is enough; the logged route record is not a defect.
   The 10 remote tests the route leg made DEFECT are now OK or WEAK on their data legs.
2. Comment claims you did not know about (MM7T2FanOut "waits for every receiver", T3GcbSwitch "one DRAIN covers both
   GCBs", the trid BarrierThenDependent tests as barrier tests, MimirDmIdma.Copy "ack only after the data lands"):
   those tests are CONDITIONAL, with both verdicts given.
3. These legs should fail when the behaviour does not happen: DirtyLineRace configs with disable_inline_writes=1,
   FmtOpMatrix (format, op) points that never ack, and CleanDirtyLine's "control line still cached across launches".
   They are correct asserts, not defects. Where the emulator fails them today, that is a finding about the emulator
   or hardware.
4. Whether CCE hart loads from GDDR / remote-SRAM SPA addresses are cached: unknown (H9). Verdicts that depend on it
   are CONDITIONAL: the two MM-4 tests that re-poll GDDR, and the MM-5/MM-6 tests whose read leg follows the same
   CCE's write to the same address (MimirRemoteTl1.WriteRead, MimirRemoteDram.Tl1RoundTrip).

## Totals
| Section | Tests | DEFECT | WEAK | OK | CONDITIONAL |
|---|---|---|---|---|---|
| MM-0 host DRAM | 6 | 0 | 4 | 2 | 0 |
| MM-1 host TL1 | 8 | 0 | 7 | 1 | 0 |
| MM-2 firmware load | 7 | 2 | 3 | 2 | 0 |
| MM-3 kernel load | 7 | 0 | 1 | 6 | 0 |
| MM-4 local DRAM | 10 | 0 | 0 | 8 | 2 |
| MM-5 local/remote TL1 | 8 | 1 | 3 | 2 | 2 |
| MM-6 remote DRAM | 10 | 4 | 2 | 3 | 1 |
| MM-7 prefetcher | 55 | 0 | 9 | 41 | 5 |
| MM-8 DM-suite ports | 11 | 0 | 3 | 5 | 3 |
| 4.1 cache | 22 | 0 | 7 | 10 | 5 |
| 4.2 atomics | 20 | 1 | 10 | 8 | 1 |
| 4.3 transaction IDs | 17 | 2 | 12 | 0 | 3 |
| Total | 181 | 10 | 61 | 88 | 22 |
| Working tree (fix 3 applied) | 181 | 5 | 61 | 93 | 22 |

In the working tree, MM-5 is 0/3/3/2 and MM-6 is 0/2/7/1 (DEFECT/WEAK/OK/CONDITIONAL); the other sections are unchanged.

The ten DEFECTs: MimirFwLaunch.NoOpLaunchCompletes, MimirEmu.RuntimeFirmwareInitializesThroughPublicDeviceApi,
PrefetchesLocalGddrToRemoteCceSram.RoundTrip, MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs,
CopiesGddrWithRuntimeArgs and CopiesGddrLargerThanAlignment, MimirEmu/CopiesEveryMimirBlock (1x1 only),
MimirAtomicLocal.Ordering, MimirTridReset.NoStaleIdAfterReinit, MimirTridConcurrent.WithPrefetcher. The four MM-6 ones
were added by the critique pass (unpoisoned staging, the same hole as PrefetchesLocalGddrToRemoteCceSram). The five
staging DEFECTs are fixed in the working tree (fix 3), which leaves five: NoOpLaunchCompletes,
RuntimeFirmwareInitializesThroughPublicDeviceApi, Ordering, NoStaleIdAfterReinit and WithPrefetcher.

Conditional verdicts that turn DEFECT on one outcome:
- Pass a fault: the two BarrierThenDependent tests and MimirDmIdma.Copy if their claims are targets; the three
  MimirCacheSharedL2 tests if H1 holds (their gated shared-L2 read would pass a no-op writer cflush);
  MimirLocalTl1.SameChipletControl's no-ATT leg if H3 is false (it cannot fail); MimirCacheOps.FullInvalidate's discard
  leg and MimirDmCache.Op L1InvalidateLine/Full if H2 is false (a no-op cdiscard passes).
- Fail a correct implementation: SizeAlignmentMatrix and RepeatedAndOverlappingPerHart if H9 is "cached";
  MimirTridIdma.DuplicateId if 512 B commands never stay pending past the next issue; MimirDmCache.Op L2Flush* and
  FlushFenceOrdering's FlushNoFence leg if H2 is false; MimirAtomicLocal.NeighbourWriteRace if H1 is false with
  full-line writeback; MimirDmAddrgen Im2Col_Dilation1 if the wrap rule is modulo.

Earlier review for comparison: 49 DEFECT, 50 WEAK, 95 OK of 194 rows. Not like-for-like: the earlier rows counted the
four Group B tests twice (once in MM-4, once in MM-6), and this review counts 181 tests once. Most of the drop comes from deleted record-only
tests, per-run poison added to host tests, the MM-4 iDMA rewrite, gating added in 4.1/4.2, and your route decision.

## Legs that fail today, or may fail on a correct implementation
These are the opposite problem: a test that cannot pass.
- Fail today, correctly by your rule that the zero-fill is corruption: MimirHostDram.HostSideRejections 6-byte leg
  (test_mimir_host_dram.cpp:486-494) and MimirFwPreflight.FirmwareLoadPreflight 1-byte leg
  (test_mimir_host_tl1.cpp:564-603). Cause: a write whose size is not a multiple of 4 goes through
  `transport->write(size, 4, ...)` (umd tt-umd/src/tt_device/mimir_chippy_memory.cpp:58-69) to chippy `_write`, which
  builds the last word from zero (chippy lib/transport/transport_interface/transport_interface.h:61-87, comment
  "tracked separately"). README rows 1 and 2a do not say these legs are expected to fail.
- Fail on the emulator today, correct by your rule: DirtyLineRace configs 1 and 3 (README:430: acked, never visible).
  Side effect: the RaceUnresolved path sets the process-wide fresh_session_required() (test_mimir_dm_cache.cpp:662-671,
  :1987), so in a batch run every later MimirCacheOps/Coherence/Stress/SharedL2/WriterProbe/Race case skips.
- May fail on a correct implementation (conditional):
  - SizeAlignmentMatrix, RepeatedAndOverlappingPerHart: if H9 is "cached", hart 0 re-polls GDDR bytes whose contents
    changed between launches (cce_idma_copy.cpp:52-55, :385-392) and times out.
  - MimirTridIdma.DuplicateId: asserts pending high water > 1 with 512 B commands (test_mimir_trid.cpp:907, :916).
    T:975 records high water 1 for 256 B commands to TL1; the only multi-pending data on record is with 8 KiB.
  - MimirTridIdma.Saturate: per-backend high water >= 2 (T:995); the own-TL1 leg measured exactly 2 on 2026-10-22.
  - MimirTridConcurrent.WithPrefetcher: its overlap assert fails if one stream round outlasts the whole prefetch.
  - FlushFenceOrdering FlushNoFence leg and MimirDmCache.Op L2Flush*/HartToHartCoherent: fail if H2 is false.
  - HartToHartCoherent and NeighbourWriteRace: fail if H1 is false (with full-line L1 writeback for the latter).
  - Addrgen Im2Col_Dilation1: fails if the hardware wrap rule is modulo, not restart (H8).
  - All dedicated MM-6 tests: snapshot_d2d_errors_after_clearing_axi_errors (test_mimir_remote_dram.cpp:389-408)
    writes back the AXI error bits and TT_FATALs if any stays set. The write-1-to-clear behaviour rests only on the
    comment at :387. If the bits cannot be cleared, one earlier AXI error makes every later MM-6 test fail (H5).
  - expect_cce_initialized asserts PF_CTRL raw_lo == 0x1FF exactly (test_mimir_fw_load.cpp:141); UMD documents only
    bits 0-8. poison_cce_reset_vectors compares a 64-bit register with a 32-bit poison (mimir_fw_loader.cpp:54).
    Either fails a correct implementation if the unused bits are ever nonzero.
  - MimirCacheProbe.SnoopConfig, CrossCce leg 2, NonSnoopingProducer: fail if an interrupted process left the snoop
    config or a remap cmd_snoop bit set (state persists until INIT; I13). That is correct detection of dirty state.

## How to fix the DEFECTs
Revised after the critique pass (see "Critique pass" under Confidence). Summary:
- Delete now: NoOpLaunchCompletes (fix 1) and RuntimeFirmwareInitializesThroughPublicDeviceApi (fix 2); other tests
  already cover what they claim.
- Done in 6ea8f29c6e4 (built, not run): the five staging tests (fix 3).
- Need an answer first: Ordering (fix 4: whether `.rl`/`.aq` order the `tt.cache` instructions, and whether cdiscard
  is ordered before later loads without a fence), NoStaleIdAfterReinit (fix 5: a safe never-ack target), and
  WithPrefetcher (fix 6: H9, or use fresh destination slots).

### 1. MimirFwLaunch.NoOpLaunchCompletes (test_mimir_fw_load.cpp:321): delete it
Problem: the null kernel leaves no trace, and the enables check (:334) reads back the host's own write.
Recommended: delete the test and point README row 2b at MimirKernelLoad.EachHartAlone and ArgsReplaceStale.
- EachHartAlone, on its hart-0 pass, already launches the echo kernel on hart 0 alone into poisoned slots on every
  CCE and asserts every other slot is still poison (test_mimir_kernel_load.cpp:227-245). ArgsReplaceStale already
  covers a relaunch with new arguments (:249-286). An echo-kernel rewrite of NoOpLaunchCompletes would repeat both.
- The earlier draft said such a rewrite would catch firmware that ignores the enable mask by writing slots 1-7. That
  was wrong: drisc.cc:162 gates only hart 0's own kernel; harts 1-7 are started by start_subordinate_kernel_run_early
  (drisc.cc:71-77). Firmware that ignored the mask would make harts 1-7 jump to an unconfigured kernel_text_offset
  (:127-130), which most likely hangs rather than writing a slot.
If you keep the test anyway: launch the echo kernel on hart 0 with every slot poisoned and drop the enables check.
Notes for that route: different arguments on the repeat need update_echo_arguments (test_mimir_kernel_load.cpp:163-171,
since common args can be set only once, :153), the untouched-slot check is check_echo_slot_untouched (:116), and
sharing the helpers means moving EchoArguments, the constants and create_echo_kernel_from out of the kernel_load
namespace.

### 2. MimirEmu.RuntimeFirmwareInitializesThroughPublicDeviceApi (test_mimir_fw_load.cpp:407): delete it
Problem: the previous process's firmware is still in TL1 and can report DONE.
Recommended: delete the test. LoadAndReachInitDone covers the same claim with real oracles (zeroed text, poisoned reset
vectors, ELF compare).
Why not zero the firmware window first, as the earlier draft proposed: it fails a correct implementation in the
invocation the README prescribes. All `*MimirEmu*` cases run in one invocation with no preflight or hardware-state
checks (README:85-88); row 2b (`MimirFwLoad.*:*MimirFwLaunch*`, README:54) does not include this test, so the draft's
"as row 2b does" was wrong. Other MimirEmu tests may open a device first, a later open does not reload firmware
(test_mimir_fw_load.cpp:174-175), and zero_firmware_windows would then zero the text of running harts. The helpers also
need skip_unless_mimir_local_package and the watcher skip (:178-181) because they use kFwLoadChip and SMC register
access (mimir_fw_loader.cpp:39-57; test_mimir_fw_load.cpp:161-166), which would narrow a contract test that runs on
every descriptor today.

### 3. Unpoisoned staging: PrefetchesLocalGddrToRemoteCceSram and four MM-6 contract copy tests
Applies to PrefetchesLocalGddrToRemoteCceSram.RoundTrip (test_mimir_tl1.cpp:755),
MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs, CopiesGddrWithRuntimeArgs and CopiesGddrLargerThanAlignment
(test_mimir_remote_dram.cpp:1389-1632), and MimirEmu/CopiesEveryMimirBlock (:1716) on 1x1.
Problem: each copies GDDR to a TL1 staging buffer at the CCE's UNRESERVED base and on to a destination, with hart
loads/stores. Staging is never poisoned and the input is the same every run, so on a filtered rerun a GDDR read that
never lands forwards last run's identical staging, and the zeroed destination receives the right words.
- Before each launch, write a poison pattern (different from the input) to staging, as the MM-4 versions already do
  (test_mimir_local_dram.cpp:1128-1130, :1143).
- After the launch, read staging back and compare it with the expected input, which separates a failed GDDR read from
  a failed write. (CompileTimeArgs already reads staging, :1449-1451, but without the poison that read proves nothing.)
- Do not add a per-run seed to the input. With a fixed input, a stale cached GDDR line holds the same words and does no
  harm; with a seeded input, a correct implementation fails if hart GDDR loads are cached and lines survive between
  launches (H9). The destinations are already zeroed each run, so the seed adds nothing once staging is poisoned.
What now fails: a GDDR read that never reaches staging leaves the poison, which reaches the destination and the
staging readback.
Applied 2026-11-03 in 6ea8f29c6e4 (built, not run). Each test writes 0xDEADBEEF, which no input uses,
to the whole staging buffer right before LaunchProgram, and after the launch reads staging back and EXPECTs:
- PrefetchesLocalGddrToRemoteCceSram: staging on the source CCE equals the input.
- CopiesGddrWithCompileTimeArgs: the existing staging check (inputs.back()) now follows the poison.
- CopiesGddrWithRuntimeArgs, CopiesGddrLargerThanAlignment: new staging check against inputs.back().
- CopiesEveryMimirBlock: new staging check against the last Mimir's input (the kernel copies each Mimir in turn
  through the same buffer).
No per-run seed was added. The poison comment is copied from the MM-4 version.
Remaining gap, not fixed: in the three CopiesGddr* tests the input has no slot number, and the kernels reuse one staging
buffer for every slot. On a view with more than one slot, a failed read on slot 2 or later would forward the previous
slot's identical words and pass. The current descriptors have 512 MiB views, one slot each, so it cannot happen today.
Putting the slot number in the input (a fixed difference, not a per-run seed) would close it.

### 4. MimirAtomicLocal.Ordering (test_mimir_atomics.cpp:1040; kernel cce_amo_ops.cpp:681-720)
Problem: `flush_l1_dcache` and `invalidate_l1_dcache` both end in a full `fence` (risc_common.h:212, :223), which alone
orders data before the flag, so `.rl`/`.aq` never matter.
- Producer: use `tt.cache.cflush.d.l1` without the trailing fence, so only `amoswap.w.rl` orders the flush before the
  flag.
- Consumer: start by keeping its fence after the cdiscard and test only `.rl`. Removing the consumer fence leaves
  nothing ordering the cdiscard before the data loads (cce_amo_ops.cpp:716-720); `.aq` orders the AMO before later
  operations, not a cdiscard against later loads to the same line, so a fence-free consumer could fail on correct
  hardware for a reason other than `.aq`. Test `.aq` only once that ordering is known.
- Add a control leg with plain `amoswap.w`/`amoadd.w` and the same cache ops, and record its mismatch count.
How to read the result:
- If the control leg sometimes sees stale data and the `.rl` leg never does, the test is real.
- If the control never sees stale data either (likely on an in-order Rocket core), no broken `.rl`/`.aq` can be
  detected this way. The test should then claim only "message passing with flush works", and drop `.rl`/`.aq` from its
  name and claim.
Open facts: whether `.rl` orders the custom `tt.cache` instructions is not documented in anything read for this review;
if it does not, the fence-free producer fails on correct hardware. The release-only leg can be gated once H1 is
settled.

### 5. MimirTridReset.NoStaleIdAfterReinit (test_mimir_trid.cpp:1397)
Problem: the counters are already 0 before the reinit, and a dead counter also reads 0.
Deeper limit: in-flight commands finish and drain their counters whether or not a reinit happens. A stale count can
only survive a reinit if a command never acks, so this claim needs a fault path (a command to a target that never
responds). That is MIM-601 territory, where resets with commands in flight are known to hang.
- Liveness check (do this regardless): the batch already run after the reopen (:1438-1451) reports a pending high
  water, because record_pending runs right after each issue (cce_idma_copy.cpp:659). Assert it is at least 1; that
  catches a dead or unmapped counter with no new launches. NominalDepth already asserts the same (:957), and high water
  1 was recorded at 256 B, so the false-failure risk is small (lower than DuplicateId's > 1). Add one launch per trid
  only if every trid must be checked.
- Stale-ID leg (only with a safe never-ack target): session 1 issues on trids 0-15 and exits without draining, with the
  kernel status showing pending > 0 at exit; then reopen with FORCE_REINIT and expect 0. The expected value depends on
  H6.
If no safe never-ack target exists, rename the test to "trids usable after reinit", keep only the liveness check, and
drop the stale-ID claim.

### 6. MimirTridConcurrent.WithPrefetcher (test_mimir_trid.cpp:1568; kernel cce_idma_copy.cpp:645-688)
Problem: every round copies the same source to the same destination, so every round after the first is already
satisfied.
Proposed: rotate the source. Give stream mode a source count K; round r copies from source slot `r % K` to the one
destination; the host seeds the K slots with different data. issue_one already takes the source address per command
(cce_idma_copy.cpp:245-260), so no new setup_batch is needed; the other modes must reject the new argument.
CONDITIONAL on H9: the kernel's per-round check (complete_batch, data_landed) reads the GDDR destination with hart loads
(cce_idma_copy.cpp:52-55, :370-390). Nothing invalidates those lines between rounds of one launch, and the snoop tag
does not cover writes to GDDR. If hart GDDR loads are cached, round r reads round r-1's line and times out on a correct
implementation, and here the lines only need to survive within one launch. Today's same-source loop hides this.
Choices:
- Wait for the H9 answer, then rotate as above.
- Invalidate the destination lines in the hart's L1 and L2 before each round's poll (this brings in H2).
- Send each round to a fresh destination slot that the host poisoned beforehand and checks at the end (slot `r` for
  round r, up to the slot count). This avoids re-reading a line and does not depend on H9.
The per-round check in the kernel is what catches a dropped write in the middle of the stream; the host's final compare
of slot `(rounds - 1) % K` alone cannot see one when K = 2.

## Shared-layer findings (cited as I1-I13)
A missing status line can no longer pass: every status/done word is poisoned before launch and published last. What
remains is that the kernels' own completion checks are weak, so a test is only as strong as its host readback.
- I1 Trid "acked" means pending count == 0 (test_kernels/cce/cce_idma_copy.cpp:294, :892; cce_idma_blocking_copy.h:51;
  cmdbuff_api.hpp:1660). An ignored TR_ID, aliased trid bits, or a counter that never increments all pass at once. In
  kernel-verified modes the data poll then waits for the data, which hides an early ack. Only kArgCheckAtAck and
  asserted pending high water expose it.
- I2 Host-verified success with a destination other than own TL1 means "acked" only (cce_idma_copy.cpp:368-370).
- I3 data_landed compares two of the kernel's own reads (cce_idma_copy.cpp:62-82), through the same SPA route as the
  copy for GDDR and sibling-SRAM operands. It passes with nothing moved if the destination already holds the source.
- I4 XOR-only kernel checks: destination_xor_matches, accumulate_result_matches (cce_idma_copy.cpp:85-94, :315-332).
- I5 cce_copy_ops' XOR covers the source, not the destination (test_kernels/cce/cce_copy_ops.cpp:47-51).
- I6 compare_with_recheck (mimir_dm_common.hpp:181-210) sleeps 500 ms on a mismatch and re-compares, so "done before
  the data landed" passes and is only recorded. Used by test_mimir_dm_idma.cpp:443, test_mimir_dm_dram_unary.cpp,
  test_mimir_trid.cpp:537.
- I7 Stream mode passes with zero rounds if STOP is already set (cce_idma_copy.cpp:647-688). The one user,
  WithPrefetcher, now zeroes STOP and asserts rounds >= 1.
- I8 The runner waits for the magic, not the code (mimir_dm_common.hpp:90-104). Every test checked asserts kStatusCode.
- I9 D2D DEBUG_MISC bits already set are only logged as inconclusive (mimir_hw_state.cpp:378-404, :714-724); bits 9-11
  are sticky until chip reset, so after one no_match the check is blind for the rest of the session.
- I10 WaitProgramDone gates only on GO==DONE (tt_metal/llrt/llrt.cpp:338-356); launch_program_and_wait skips the idle
  check (mimir_access.hpp:379-382), so a launch onto a busy CCE is lost and the wait returns at once. Tests that poison
  their result words (all MM-3, MM-7, MM-8, 4.x) catch this; MM-2's null-kernel launches cannot.
- I11 make_pattern truncates the address to 32 bits (mimir_patterns.hpp:55, :65), so both views of a Mimir and both
  Mimirs get the same words; make_seed_pattern and make_guard ignore the seed. Harmless as poison; a hole only where a
  fixed pattern is the expected value at an unpoisoned destination. make_located_pattern exists only in
  test_mimir_local_dram.cpp:123.
- I12 The route record (mimir_access.hpp:724-752, crosses_d2d hard-coded true at :749) is only logged (:754). By your
  decision this is not a defect.
- I13 The remap check compares only ATTRS bit 0 (mimir_hw_state.cpp:68, :220-222), so a cmd_snoop bit left set by an
  interrupted ScopedSramRemapSnoop persists and is only logged.
- Watcher skip: MIMIR_SKIP_IF_WATCHER (mimir_hw_state.hpp:27-33) skips with an "SMC register reads" reason in files
  that make no SMC reads (atomics, addrgen, dram_unary, dm_idma, mesh_buffer, scoped_lock, dm_cache Op/WritePath).
- full_package descriptors: skip_unless_mimir_local_package skips every remote test, MimirLocalDram, the trid, prefetcher
  and DM-port suites, and most fw_load/host tests.
- Earlier cross-cutting findings: seed mask fixed (test_mimir_dm_cache.cpp:1449); setup mismatch now fails
  (mimir_hw_state.hpp:101-108); MIM-628 no longer cited to justify recording (only cce_idma_copy.cpp:6-7); UMD gitlink
  still records 292fe20 while the working tree uses f0d497f3, which the build needs.

## Per section
Each entry: verdict, what it tests, how it decides, and what broken implementation still passes. "Earlier" gives the
change from the 2026-10-29 review.

### MM-0 host DRAM (test_mimir_host_dram.cpp)
- MimirHostDramView.PatternReadback: OK. Host write/read 4 B to 128 KiB at four anchors per view with 64 B guards;
  per-run seeded patterns; EXPECT on payload and guards (:159-218). Passes: a consistent mistranslation (not its claim;
  cross-checked by MimirLocalDram.CceReadsDramToTl1). Earlier: OK.
- MimirHostDramView.BoundaryAndAlias: WEAK. Cross-view uniqueness and an address-bit alias walk (:229-264). The GiB
  loop (:270) has zero iterations and the view-end branch (:277-297) never runs with 512 MiB views. Passes: swapped
  address bits (same bijection both ways; :246 overclaims); a dropped write in the cross-view or alias-walk legs on a
  filtered rerun (fixed make_guard values, no poison). Aliasing is still caught. Earlier: WEAK.
- MimirHostDram.ReconnectPersists: WEAK. Nothing resets or disconnects (emu_skip_init, tt_cluster.cpp:574; no socket
  close in ~EmuTTDevice), so persistence is the only outcome. Earlier: WEAK.
- MimirHostDram.HostSideRejections: WEAK. Rejections of out-of-window, overflow, misaligned (:415-427), sentinels, and
  the 6-byte leg now expecting the other 2 bytes kept (:486-494, fails today, see above). Passes: every rejection is
  host-side (UMD/chippy); any UmdBaseException satisfies it; an access outside the 512 MiB view but inside 8 GiB is
  treated as legal (:405-410). Earlier: DEFECT (pinned the zero-fill); changed.
- AllocatedDramBufferHostLoopback.RoundTrip: OK. Poison 0xDEADBEEF^i written and ASSERTed first (:527-534), then data
  and per-channel page checks. Earlier: DEFECT (stale data); fixed.
- DramChannelsDoNotAliasThroughPublicDeviceApi.RoundTrip: WEAK. One word per channel pair, fixed values, no poison; on
  2x only pairs p and p+n/2 (:579). A dropped write passes on a filtered rerun. Earlier: WEAK.

### MM-1 host TL1 (test_mimir_host_tl1.cpp)
Suite-wide: write and readback share Cluster/UMD/chippy, so a CCE0/CCE1 swap inside one Mimir passes every MM-1 test.
The identity preamble probably catches an M0/M1 swap for MimirHost* (mimir_hw_state.cpp:495-520).
- MimirHostTl1.Tl1MapLayout (new, replaces DiscoverCceTl1Map): WEAK. Layout EXPECTs (:293-328) check HAL constants
  against the header they come from; per region, poison then a tag on the first word (:330-351). The permission
  matrix was deleted; only host RW on one word per region is asserted.
- MimirHostTl1Cce.PerCceVisibility: WEAK. Poison, pattern, guards at four offsets, 4 B-32 KiB (:354-417). The pattern
  base has no CCE index (:378-380), so full CCE aliasing passes here (BoundaryAndIsolation catches it). Earlier: WEAK.
- MimirHostTl1.BoundaryAndIsolation: WEAK. Every window poisoned and ASSERTed (:449-461), then per-CCE tags. Passes:
  a consistent CCE permutation. Earlier: DEFECT (stale data); fixed.
- MimirHostTl1.HostSideRejections: WEAK. Six EXPECT_THROWs plus sentinels (:512-541); past-end is caught by the GDDR
  check, not a TL1 bound; reset and protected-range legs absent. Earlier: WEAK.
- MimirFwPreflight.FirmwareLoadPreflight: WEAK. Every TL1 range the loader writes, poisoned then seeded; reset vectors
  poisoned and read back per hart over SMC (:617-630); SPA-routes chunk added (:203-207); 1-byte leg compares the whole
  word (fails today). Weak: bank-to-NOC split into 4 equal parts, not the runtime's sizes; "every hart" is UMD's
  software loop (tt_device.cpp:106-111). Earlier: DEFECT (log-only reset vector); fixed.
- CceSramRoundTripThroughUnitMesh.RoundTrip: WEAK. One word at 0x800 in the live mailbox launch ring, no poison; only
  the device open's rewrite of the ring rules out a stale value (:640-657). Earlier: WEAK.
- CceSramRoundTripThroughMetalCluster.RoundTrip: OK. Poison ~written ASSERTed, then write/read (:693-700). Earlier:
  DEFECT; fixed.
- MimirEmu.CceSramChannelsDoNotAliasThroughMetalCluster: WEAK. Same word and values as RoundTrip, no poison
  (:703-731); a dropped write passes after RoundTrip or a rerun. Aliasing is caught. Earlier: WEAK.

### MM-2 firmware load (test_mimir_fw_load.cpp)
- MimirFwLoad.LoadAndReachInitDone: WEAK. Reset vectors now poisoned per CCE and hart before the open (:185,
  mimir_fw_loader.cpp:39-57) and read back; text window zeroed and compared to the ELF (:183, :206-214); mailbox
  fields. Passes: a loader that drops the data/.bss spans (only the span at fw_base is zeroed and compared,
  :161-172, :199-214); a dropped WAIT/INIT write (mailbox fields not seeded). Earlier: DEFECT; fixed.
- MimirFwLoad.ImageManifest: WEAK. ELF spans checked against HAL windows, but the ELF is linked from the same
  constants and relocated by the same HAL call (:228-278). Earlier: WEAK.
- MimirFwLoad.AllCcesIndependent: OK. core_info xy against UMD translated coordinates (:291-296); every other CCE's
  mailbox unchanged across a null launch (:298-316). Earlier: OK.
- MimirFwLaunch.NoOpLaunchCompletes: DEFECT. `EXPECT_NE(read_launch_enables(...), 0)` (:334) reads back the launch
  message the host just wrote; firmware clears enables only in DISPATCH_MODE_DEV (drisc.cc:174-177), and firmware
  that skips the kernel still writes DONE (drisc.cc:162-173). The null kernel leaves no signature, so "the no-op ran"
  cannot fail. Fix: delete (fix 1). Earlier: DEFECT; still true.
- MimirFwReinit.ReloadsOnEveryOpen: WEAK. Proves a restart (fresh INIT answered), not a reload: no zero/compare of
  the text and no reset-vector poison between cycles; :365 is the host's own write. Earlier: WEAK.
- MimirFwLoadNegative.SkipLoadNeverDone: OK. The open must throw with "failed to initialize DRAM FW" (only at
  risc_firmware_initializer.cpp:1722); text zeroed first (:382). Caveat H7. Earlier: OK.
- MimirEmu.RuntimeFirmwareInitializesThroughPublicDeviceApi: DEFECT. Only `ASSERT_NO_THROW(open_unit_mesh())` and the
  arch (:407-415); text not zeroed and vectors not poisoned, so if the image write fails the previous process's
  firmware, still in TL1, reports DONE. LoadAndReachInitDone covers the same ground properly. Earlier: DEFECT; still
  true.
- The runtime's 1-byte fw_shared_globals_ready write (risc_firmware_initializer.cpp:1445-1450): no MM-2/3 test sees the
  zero-fill; DRISC firmware reads and writes only byte 0 (drisc.cc:98, :103), so it has no effect today.

### MM-3 kernel load (test_mimir_kernel_load.cpp)
Shared: mapping checked against UMD (:73-89); 8x16 result words poisoned every run (:91-94); cce_args_echo writes
magic, mhartid, CT marker, 4 RTA, 4 CRTA into the slot chosen by mhartid through the uncached alias, and the whole slot
is compared (:109-114). A lost launch leaves poison.
- MimirKernelLoad.MinimalKernelExecutes: OK (:211).
- MimirKernelLoad.EachHartAlone: OK. Target slot echoes, every other slot on every CCE still poison (:232-245).
- MimirKernelLoad.ArgsReplaceStale: OK. Poison before both launches (:267, :280); arg count change not covered.
- MimirKernelLoad.FirmwareKeepsRunningAcrossReopen: OK for launch-after-reopen, which is all you said it needs.
- MimirKernelLoad.SameBinaryAllCcesUnicast: OK. Multicast is impossible on DRAM cores (llrt.cpp:291).
- MimirEmu/HartZeroRunsDramKernel.WritesMagic: WEAK. Result word now poisoned (:507-512), but the kernel never reads
  mhartid, so the same kernel on another hart passes. Earlier: DEFECT; fixed apart from hart identity.
- MimirEmu/AllHartsRunDramKernel.WritesMagic: OK. Now uses the echo kernel with a per-hart marker into poisoned slots
  (:541-583). Earlier: DEFECT; fixed.

### MM-4 local DRAM (test_mimir_local_dram.cpp)
All ten move data by iDMA and gate on host readback against a host model, with destinations poisoned or zeroed every
run, so I1-I4 cannot turn a broken copy into a pass. kArgCheckAtAck is never set (early acks hidden, no false pass).
Pending high water is only logged (:332-334).
- MimirLocalDram.CceWritesTl1ToDram: OK. Poison and guards (:414), host compare (:430-431). Nothing asserts that at
  least one anchor ran (the `continue` at :394-402 does not fire on current descriptors).
- MimirLocalDram.CceReadsDramToTl1: OK. make_located_pattern salted by Mimir and view (:125-131, :475-482); a read
  that drops bit 32 or goes to the other Mimir now fails. Earlier: DEFECT; fixed.
- MimirLocalDram.LoopbackChecksIntermediate: OK for its claim. Read and write are separate launches (stated at
  :513-515), intermediate TL1 checked twice. Earlier: DEFECT; fixed.
- MimirLocalDram.SizeAlignmentMatrix: CONDITIONAL(H9). WEAK if hart GDDR loads are uncached: write direction only,
  offsets {0, 4, 60, 124, 32 KiB-4}, first fitting anchor only (:652), the shared leg skipped by `continue` when fewer
  than 2 harts fit (:707, always at 32 KiB), and the shared leg accepts any hart's word. DEFECT (fails on a correct
  implementation) if loads are cached and stale lines survive between launches. Earlier: WEAK.
- MimirLocalDram.RepeatedAndOverlappingPerHart: CONDITIONAL(H9), same split. WEAK part: one launch per op
  (:741-742, :859), so overlapping copies are never in flight together; the T5 negative half is absent; the
  AddressDerived half repeats in repeats 0 and 2 (comment :781 half true). Earlier: WEAK.
- IdmaCopiesGddrWithCompileTimeArgs.RoundTrip: OK. GDDR to staging to GDDR by iDMA on hart 0; the only MM-4 test of
  in-CCE read-then-write ordering. Staging poisoned, destination zeroed (:924-969).
- IdmaCopiesGddrWithRuntimeArgs.RoundTrip: OK. Unique words per CCE/Mimir/location (:1133-1138), poisoned (:1142).
- IdmaCopiesGddrLargerThanAlignment.RoundTrip: OK. 1 KiB; the earlier OR-encoding overlap is fixed (:1137).
- MimirEmu.AllCcesCopyLocalMimirBlock: OK. Tagged per CCE, staging poisoned (:1203-1276).
- AllHartsIdmaWriteGddr.WritesMagic: OK. Distinct value per hart (:1327), destination zeroed. Trid separation not
  gated (not its claim).

### MM-5 local and remote TL1 (test_mimir_tl1.cpp)
Every data check reads the destination back on the host against a host model, with seeded destinations and guards in
the other CCEs, so I5 does not bite. Read legs where hart h+1 of the writer's CCE reads back could be served from a
shared cache without crossing D2D if H9 is "cached".
- MimirLocalTl1.SameChipletControl: CONDITIONAL(H3). Data legs OK (:308-318). The no-ATT leg compares D2D DEBUG_LAST
  before and after (:270-271, :294-298): if H3 is false it cannot fail; if true, on 1x1 only M0.D2D0 is read
  (mimir_hw_state.cpp:127-139), so Cce1To0 cannot fail it, and it compares last values with no counter. The own control
  copy destination is not seeded. Uses package SPA plus the DMRISC remap, not the plan's local Mimir address.
- MimirRemoteTl1.WriteRead (MM5Direct x2, MM5Routed x6): CONDITIONAL(H9). Seeded landing and read-back slots,
  trailing guard, guards in all other CCEs (:380-465). The write leg is OK either way. The read-back is done by hart h+1
  on the writer's CCE: OK if SPA loads are uncached; WEAK if they are cached, since the read could be served without
  crossing D2D. Direct and Routed run the same body; the one-hop/two-hop labels are names only. Earlier: DEFECT (route
  leg); the route leg is no longer a defect by your decision.
- MimirRemoteTl1.HostInitRemoteRead: OK. Host-written Prng, seeded copy slot, guards (:519-566). Return-path endpoint
  is only logged. Earlier: DEFECT (route leg).
- MimirRemoteTl1.SizeAlignment: WEAK. 7 sizes x 5 offsets, write only (`read_back=false`, :590); no leading guard;
  every copy is 32-bit hart stores. Earlier: DEFECT + WEAK.
- MimirRemoteTl1Bidir.Concurrent: WEAK. Nothing forces or observes overlap. Patterns differ by direction (:482).
- MimirRemoteMixed.FanOutFanIn: WEAK. Overlap not forced; guards in non-destination CCEs cover only the first 64 B.
- WritesCrossMimirCceSram.WritesMagic: OK (narrow). Destinations zeroed, one word each (:727-748).
- PrefetchesLocalGddrToRemoteCceSram.RoundTrip: DEFECT. The staging buffer at the source CCE's UNRESERVED base
  (:783-784) is never poisoned and the input `0xD2D10000|src<<8|i` has no seed (:790-798). UNRESERVED TL1 persists, so
  on a rerun of the same instance a kernel whose GDDR read never reaches staging forwards last run's identical words
  and passes. A full in-order run overwrites that region first, so the hole is in filtered reruns. Fix: poison staging
  and read it back, no seed (fix 3). Earlier: OK; changed by the settled persistence facts (code unchanged apart from the move).
  Working tree: OK. Fix 3 applied; staging is poisoned before the launch and checked against the input after it.

### MM-6 remote DRAM (test_mimir_remote_dram.cpp)
The dedicated MM-6 tests' data checks are sound: host readback of destinations poisoned on this run. At HEAD the four
MimirEmu contract copy tests are not: their TL1 staging is never poisoned (see below). Fix 3 in the working tree
poisons it, which makes all four OK there. Earlier holes fixed: read legs no longer
draw all-zeros/all-ones (kReadLegPatterns, mimir_patterns.hpp:37); decoy count asserted (:569); preflight mismatch
fails. See H5 above for the AXI error-bit TT_FATAL. Residual: source Mimir +4 GiB is not decoyed (low likelihood).
- MimirRemoteDram.WriteFromTl1: OK. Seed-prefilled destination view, decoys, host compare (:608-644).
- MimirRemoteDram.ReadToTl1: OK. Host-written pattern, guarded TL1, decoys (:671-700).
- MimirRemoteDram.Tl1RoundTrip: CONDITIONAL(H9). Write leg OK. The read follows the same CCE's write to the same
  address (:743-810): OK if loads are uncached, WEAK if cached (a broken D2D read path would pass).
- MimirRemoteDramTrafficMatrix.TrafficMatrix: WEAK. Size x alignment only at unres_g0 (:1353); slice cases at
  unres_g0 and view_end (:1371); no bank or interleave sweep; the channel is never observed; all 4 B hart stores.
- MimirRemoteMixed.Concurrent: WEAK. Local DRAM, remote DRAM and remote TL1 traffic in one program; overlap not forced.
- MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip: DEFECT. GDDR to TL1 staging to GDDR with hart
  loads/stores (test_kernels/misc/dram_gddr_round_trip.cpp; cce_gddr.h:30-37). The destination is cleared and the values
  are tagged (:1418-1428), but staging is never poisoned and the input is the same every run, so on a filtered rerun a
  GDDR read that never lands forwards last run's staging and passes. The staging check against inputs.back()
  (:1449-1451) does not help, since staging already holds those words. The MM-4 iDMA versions poison staging for this
  reason (test_mimir_local_dram.cpp:1128-1130). Fix 3. Earlier: OK; changed (critique pass). Working tree: OK (fix 3
  applied; the existing staging check now follows a poison). Multi-slot views would still have the per-slot gap in fix 3.
- MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip: DEFECT. Same hole (:1503-1546). Fix 3. Working tree: OK
  (staging poisoned and checked against inputs.back()); same per-slot caveat.
- MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip: DEFECT. Same hole (:1585-1628). Also, the value is
  still OR'd (:1589), which loses the location tag on half the words; that part is not a false pass. Fix 3. Working
  tree: OK (staging poisoned and checked against inputs.back()); same per-slot caveat.
- MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic: OK. Distinct value per hart (:1676).
- MimirEmu/CopiesEveryMimirBlock.RoundTrip: DEFECT on 1x1, OK on 2x. Copies src to dst on every Mimir through one
  unpoisoned staging buffer (kernel misc/cce_dram_copy_all_partitions.cpp). On 2x the Mimir index in each word breaks a
  stale match; on 1x1 there is one Mimir, so the same rerun hole applies. Offset-0 views only; no decoys (:1716-1762).
  Fix 3. Earlier: OK; changed (critique pass). Working tree: OK on both (staging poisoned and checked against the last
  Mimir's input).

### MM-7 Tensor Prefetcher (test_mimir_prefetcher.cpp)
No test passes no matter what. All gate on poisoned receiver status lines (:650-660) and a validator that recomputes
every word from a seeded formula (cce_gcb_validator_receiver.cpp:69-87); negative tests check substrings unique to
each TT_FATAL. Caveat on every validator test: the receiver FIFO is never poisoned (:626-629), so with a pinned
--gtest_random_seed, or a tensor id repeated in one process, a sender that publishes credits without data could pass.
Fix: poison the FIFO in reset_kernel_blocks.
- WEAK (9):
  - MM7GatePreconditionsHold: the sender-list leg compares the function with itself on Quasar
    (mesh_device.cpp:2006-2008); the two other EXPECTs evaluate one predicate. Earlier: OK; changed.
  - MM7GateStartStopWorks: only "does not throw" (:1337-1339). Earlier: WEAK.
  - MimirPrefetcherT3SingleGcb.Delivers chunk-ladder instances SubChunkLocal0To1, SubChunkD2d0To2,
    BelowStageThirdLocal0To1, BelowStageThirdD2d0To2, PartialStageChunkLocal0To1, AboveStageThirdLocal0To1,
    AboveStageThirdD2d0To2: if the stage-ring cap (tensor_prefetcher_manager.cpp:132-133) were ignored, host and kernel
    would both grow the stage third and still pass; the chunk size is never observed (:1910-1917 logs it).
- CONDITIONAL(intent) (5), OK if the comment claim is not a target, WEAK if it is:
  - MM7T2FanOut0To123, MM7T2FanOut3To012: "the second round waits for every receiver to ack" (:1758-1760); a free-space
    check that ignores a receiver (cce_remote_cb.h:255-269) passes unless that receiver lags, and nothing makes it lag.
  - MimirPrefetcherT3GcbSwitch.Delivers GcbSwitchLocal0To1, GcbSwitchD2d0To2, ReceiverSwitch0To1Then2: "Stop's one
    DRAIN page names both GCBs" (:2074-2075) is checked only through the request count. Resume mid-FIFO does gate.
- OK (41): MM7GateUninitializedSenderFails; MM7GcbLayoutLocalReceiver, RemoteReceiver, TwoSenders,
  RejectsConflictingReceivers; MM7GcbSmokeDiscardDrain, ValidatorLocal, ValidatorWrongSeed, Resize, ValidatorD2d,
  DrainUntilStopped, StallUntilReleased, DrainTimeout, ReceiverWaitTimeout, RelaunchResumes,
  PreconditionsWithReceiverRunning; MM7LifecycleStartStop, StartStopTwice, BusySenderFailsStart,
  QueueRejectsReceiverOnSender, WaitCq, DrainAtStop; MM7T1Local0To1, Local3To2, D2d0To2, D2d3To1;
  MM7T2TwoSenders01To2, TwoSenders02To3; T3SingleGcb PageSplitLocal0To1, AbaLocal0To1, AbaD2d0To2,
  WrapAndResizeLocal0To1, WrapAndResizeD2d0To2; MM7T3FenceLocal0To1; MM7T4MismatchD2d2To0, CreditTimeoutD2d1To3,
  DrainTimeoutD2d1To3; MM7RejectKRowMajorTensor, RejectStreamingTensor, RejectRangePastTensor, RejectRangePastView.
  Small gaps that change no verdict: the end-state padding model is an independent re-coding of the same rule as
  cce_remote_cb.h:43-81; the T4 and Reject tests do not check the FIFO guard; MM7T3FenceLocal0To1's slot is written by
  the host, not the CQ path.
- MM7T5TraceCaptureReplay was deleted (earlier DEFECT).

### MM-8 DM-suite ports
- MimirDmAddrgen.Pattern (11 instances): CONDITIONAL(H8). 10 instances OK: a host model of the loop nest
  (test_mimir_dm_addrgen.cpp:53-104), independent of the kernel, predicts every popped address into poisoned TL1
  (:118, :174). Im2Col_Dilation1 is the only instance whose loop ends are not multiples of the stride; the model
  assumes "restart at 0" (:49-52) while addrgen_api.hpp:43-44 says modulo, and the RTL notes README:226 cites were not
  found. Restart: OK; modulo: fails on a correct implementation. Only addresses are popped; no endpoint data moves.
- MimirDmDramUnary.ReaderWriterHandshake: WEAK. The reader's slot-reuse wait is caught only if the reader wins the
  race; I6 forgives writer stores that land after its done magic. Earlier: WEAK.
- MimirDmIdma.Copy (12 instances): CONDITIONAL(intent). Data, misroutes, overruns and gather errors are caught
  (poisoned destinations, guards, host gather model). The "ack" half is pending == 0 (I1), and on the host-verified
  RemoteTl1/RemoteGddr write legs compare_with_recheck (:443) forgives an ack that comes before the data. DEFECT if
  "ack only after landing" is part of its claim; WEAK otherwise. Earlier: OK.
- MimirDmMeshBufferDram.InterleavedPlacement: WEAK. Write and oracle share Cluster's channel translation
  (host_device_transfer.cpp:149; mimir_contract.hpp:408-430). Earlier: WEAK.
- MimirDmMeshBufferDram.WriteRead: OK. MimirDmMeshBufferDram.MultipleWriteReadRounds: OK.
- MimirDmScopedLockClm.CacheOps (3 instances): OK. Locked lines NEW, unlocked OLD, uncached NEW, guards
  (test_mimir_dm_scoped_lock.cpp:356-384). The release relies on H2; if H2 is false the runtime release itself is
  insufficient, which would be a real finding.
- MimirDmCache.Op (12 instances): CONDITIONAL(H1, H2). OK for NoFlushNotVisible, L1FlushLine, L1FlushFull,
  L2InvalidateLine, InvalidateFreshRead, L2InvalidateFreshRead. L2FlushLine/Range/Full fail on a correct implementation
  if H2 is false; L1InvalidateLine/Full pass a no-op cdiscard if H2 is false and are fragile (README:414, row 75);
  HartToHartCoherent needs H1 and H2. Earlier: OK.
- MimirDmCachePerf.WritePath: WEAK (cached instances also CONDITIONAL on H2). Functional only; the plan's baselines are
  not implemented; cycles are logged. Earlier: WEAK.
- MimirDmRuntimeArgs.IllegalTooManyRuntimeArgs: OK. Throw at ceiling+1, echo at the ceiling into a poisoned slot. The
  ceiling uses the same HAL value as the runtime check, so a wrong KERNEL_CONFIG size moves both.
- MimirDmRuntimeArgs.IllegallyModifyRuntimeArgs: OK. See open question on the common-args leg.

### 4.1 cache (test_mimir_dm_cache.cpp, test_mimir_dm_scoped_lock.cpp MimirCacheScopedLock)
The three earlier DEFECTs are fixed: DirtyLineRace's `continue` is gone; InvalidateOnly now gates WarmOld and Matches
(:1573-1578); the own-CCE ExternalWriters SnoopOn legs that failed by design were removed (375b0f577ba; SnoopOn kept
only for SiblingSpa, :1311). Nothing now asserts that an own-CCE iDMA write is not snooped.
- MimirCacheProbe.EightHartLaunch: OK. Per-hart poisoned magic and code (:437-449). Its code is MM-3 shaped; RTA
  permutation passes (null kernel reports no hart ID).
- MimirCacheProbe.SnoopConfig: OK. Hart read vs host SMC read, value 0x6, 64 B line, 128 KiB total (:487-491).
  Earlier: DEFECT (record only); fixed.
- MimirCacheOps.FullInvalidate: CONDITIONAL(H2). Wipe leg OK (:839-844). Discard leg (:804-809): if H2 is false a no-op
  cdiscard passes.
- MimirCacheOps.FlushFenceOrdering: CONDITIONAL(H2). L1ThenL2 and FullHierarchy gate (:929-931; a no-op cflush is
  caught). FlushNoFence gates data reaching TL1 (:925-927) with bare FLUSH64: fails on a correct implementation if H2
  is false. Ordering is recorded only; fence-without-flush was removed (plan :595). Earlier: DEFECT; mostly fixed.
- MimirCacheOps.DirtyEvictionPartialLine: OK (:944-1010). Comment says 2x L2, constant is 3x (:962-967).
- MimirCacheOps.L1DcacheFlushLine, L1DcacheFlushFull: OK (:1039-1051).
- MimirCacheSharedL2.SameLineSameOffset, SameLineDiffOffset, AdjacentLine: CONDITIONAL(H1). The fence-only read
  RegB is record only (:1153-1158). If H1 holds, a no-op writer cflush passes the gated shared-L2 read RegC (a coherent
  miss fetches from the writer's L1): DEFECT for that leg. If not, WEAK. EightWriters now gates (:1219-1238);
  contention not forced. Earlier: DEFECT; substance fixed.
- MimirCacheWriterProbe.ExternalWriters (13 instances per CCE, 9 on 1x1): OK (:1573-1588, cmd_snoop ASSERT :1524).
  InvalidateOnly assumes INVALIDATE64 also drops the issuing hart's L1 line (emulator only, README:429). Earlier:
  DEFECT x2; fixed.
- MimirCacheCoherence.CrossCce: WEAK. Leg 2 checks StaleBefore only `if WarmOld == 64` (:1689) and never asserts
  WarmOld; the value is fixed per CCE pair (:1653). A no-op INVALIDATE64 on a rerun passes if L2 still holds the
  previous run's NEW lines. Fix: gate WarmOld as ExternalWriters does. Earlier: OK; changed.
- MimirCacheCoherence.CrossCceSnoopOn: OK (:1771-1780).
- MimirCacheCoherence.CrossD2d: WEAK. Same WarmOld gap (:1689, :1797). Earlier: OK; changed.
- MimirCacheCoherenceRace.DirtyLineRace: WEAK. Every config now gates the host view (:1998) and, for inline-write
  configs, the hart's words (:2000-2003); configs 1/3/5/7 must land per your decision. Still not a dirty-line race: the
  Same target never re-dirties (:1925) and the first action cleans the line before the iDMA (:1943-1949); the hart's
  cached views are not asserted; Evict never confirms an eviction. Earlier: DEFECT; fixed.
- MimirCacheStress.ReuseThrash: WEAK. ConflictStride degenerates to sequential at 2/4/8 KiB (cce_cache_ops.cpp:882-888).
- MimirCacheStress.PreloadChurn: WEAK. The hot set is idle during the preload (cce_cache_ops.cpp:964-967). Its sibling
  snoop expectation is right; the emulator failure is unexplained.
- MimirCacheStress.PerfSweep: OK. Value fixed per leg, region not poisoned (minor).
- MimirMm4xCompile.Kernel: OK (compile only, by design).
- MimirCacheScopedLock.MultiConsumer: WEAK. A no-op release is caught; a no-op consumer acquire is masked if H1 holds or
  if the producer's acquire INVALIDATE64 back-invalidates the consumers (producer.cpp:171); "unlocked OLD" is
  trivially true (producer writes only locked lines).
- MimirCacheScopedLock.RingWrap: WEAK. Same masking (producer.cpp:221).
- MimirCacheScopedLock.NonSnoopingProducer: OK (test_mimir_dm_scoped_lock.cpp:712-725).

### 4.2 atomics (test_mimir_atomics.cpp)
Host oracle after kernel flushes; every operand seeded per run; status and result areas poisoned. In every
accumulate test the host compares every word, so I4 does not bite (except FmtOpMatrix's record-only points).
- MimirAtomicProbe.TwoHartAmo: WEAK. Overlap not forced; README:340 reports 0 SC failures.
- MimirAtomicLocal.OpMatrix: WEAK. One random operand per op (:860, :868): a signedness bug in one amomin/amomax op
  escapes about half the runs; aq/rl variants checked only as adds.
- MimirAtomicLocal.Hotspot: WEAK. Interleave recorded only (:798-809).
- MimirAtomicLocal.Distributed: OK.
- MimirAtomicLocal.LrScCas: WEAK. An SC that succeeds without a reservation passes unless windows overlap (README:432:
  0 SC failures).
- MimirAtomicLocal.Wrap: OK.
- MimirAtomicLocal.Ordering: DEFECT. The release-only leg is record only (:1071). The gated leg (:1073-1075) passes with
  plain amoswap/amoadd, because the producer's flush_l1_dcache (cce_amo_ops.cpp:704) and the consumer's
  invalidate_l1_dcache (:716) both end in a full `fence` (tt_metal/hw/inc/internal/tt-2xx/risc_common.h:212, :223), so
  .rl/.aq never carry weight. Earlier: DEFECT; still true.
- MimirAtomicLocal.CleanDirtyLine: OK. The no-invalidate leg now asserts (:1178-1181); the "control line still cached"
  precondition (:1168) stays a failure per your decision. Word 0 only on the no-invalidate leg. Earlier: DEFECT; fixed.
- MimirAtomicLocal.CleanDirtyLineSnoopedWrite: WEAK. The same control-line precondition is recorded only here (:1234),
  so "snoop ignored but the line evicted" passes; only word 0 of the 64 B write is checked. For consistency with your
  CleanDirtyLine decision this should gate too.
- MimirAtomicLocal.Storm: WEAK. Shared-leg overlap not forced; the replay model is sound.
- MimirAtomicLocal.NeighbourWriteRace: CONDITIONAL(H1). WEAK if same-CCE L1 D$ is coherent (overlap not forced); if not,
  with full-line writeback, each hart's flush (cce_amo_ops.cpp:532) writes back a stale line and the test fails on a
  correct implementation.
- MimirAtomicLocal.Soak: WEAK. About half the seeds pick the disjoint layout (:1527); 1024 ops per hart.
- MimirAtomicItnControl.Cce0ToCce1PlainWrite: OK. Its cached-after-flag precondition is recorded only (:1576-1584).
- MimirAccumProbe.FmtOpMatrix: WEAK. The 6 modelled INT32/INT8/UINT8 points now gate (:1722-1728); UINT32 and 19
  unmodelled points are record only. The saturation-disable bit is effectively untested for INT32 and UINT8. A point
  that never acks fails the test per your decision. The host model and the emulator share the cce-dv reference.
  Earlier: DEFECT; mostly fixed.
- MimirAtomicIdmaAccum.SingleHart: OK. AllHartsShared: WEAK (overlap). AllHartsDisjoint: OK. BackToBack: OK.
  MaxOutstanding: WEAK (depth recorded only, :1950-1951). MixedPathControl: OK (sequential; the plan's concurrent
  NoC-atomic control does not apply, Mimir has no NoC atomics).
- Removed since the earlier review: Mim560Detector and AccumProbe.DirtyLine (both DEFECT). README:303 and :329 still
  mention MM42Detector.

### 4.3 transaction IDs (test_mimir_trid.cpp)
Data checks are sound (destinations poisoned each run, host compare), but trid behaviour is gated only where
kArgCheckAtAck or a pending high water is asserted. kArgCheckAtAck is set in three places (DuplicateId's two legs,
:901 and :912, and WriteAfterRead's read leg, :1108); the kernel rejects it outside kernel-verified TridBatch
(cce_idma_copy.cpp:761), so it is impossible on table modes, the stream and remote targets.
- MimirTridIdma.UniqueIdsOwnTl1, UniqueIdsSiblingTl1, UniqueIdsLocalGddr: WEAK. I1: an ignored TR_ID, aliased bits or a
  dead counter pass; kArgCheckAtAck could be set here but is not.
- MimirTridIdma.MixedTargets: WEAK. I1; remote entries also I2 and I6.
- MimirTridIdma.ReuseWrap: WEAK. Reuse while commands are pending is never shown.
- MimirTridIdma.DuplicateId: CONDITIONAL(pending depth). Gates your definition: kArgCheckAtAck, kCodeLanded, pending high
  water > 1 per leg (:894-920). Catches a dead counter, an ignored TR_ID, a counter tracking only the latest command,
  and an early done. Still passes a counter saturating at 2+ and an ack slightly early (the check walks commands 0 to
  15, so the last command is checked last). OK if 512 B commands stay pending past the next issue; fails on a correct
  implementation if not.
- MimirTridIdma.NominalDepth: WEAK. Asserts high water >= 1 (:957); the kernel reports the total high water but the test
  does not assert it.
- MimirTridIdma.Saturate: WEAK. Shared slots (:1001) let 31 of 32 writes drop; the VC stall is never asserted; the own-TL1
  leg sits exactly at its >= 2 threshold.
- MimirTridIdma.BarrierThenDependentReadAfterWrite: CONDITIONAL(intent). The dependent read now must be Acked and land
  (:1086-1090), which fixes the earlier DEFECT. The two legs run in separate launches (stated at :1042-1043), so a
  barrier inside a launch is never exercised. DEFECT if it is meant to test the barrier; WEAK otherwise (the write leg
  is I1/I2 only).
- MimirTridIdma.BarrierThenDependentWriteAfterRead: CONDITIONAL(intent). The read is polled to completion in launch 1 and
  the write issued in launch 2, so the write-after-read hazard cannot occur. DEFECT if it is meant to test the barrier;
  WEAK otherwise (the read leg's kArgCheckAtAck does gate "ack implies data arrived").
- MimirTridIdma.MultiHart: WEAK. I1; per-hart high water not read; overlap not forced.
- MimirTridIdma.Fairness: WEAK. Starvation within the whole budget passes; stalls recorded only.
- MimirTridIdma.Soak: WEAK. Records only; depth and delay not randomised.
- MimirTridRemote.RemoteTl1, RemoteGddr: WEAK. Leg 1 is Acked only plus I6.
- MimirTridReset.NoStaleIdAfterReinit: DEFECT. Session 1 must drain every trid to 0 to pass, so the post-reopen reads
  (:1434-1435) see 0 whether or not the reinit touches the counters; a dead counter also reads 0; the final batch passes
  without a reinit. Earlier: DEFECT; still true.
- MimirTridConcurrent.WithPrefetcher: DEFECT (data leg of the overlapping rounds). The no-overlap skip is fixed (overlap
  asserted, :1684) and I7 is closed (:1627-1628, :1702). But every round copies the same source to the same
  destination, and the prefetch is queued only after round 1 finishes (:1653-1662), so for every round that overlaps the
  prefetcher the data check is already satisfied. A dropped or misrouted stream write during the overlap passes.
  Earlier: DEFECT; skip fixed, data hole still true.
- The four MimirTridProbe tests (earlier DEFECT) were deleted.

## Plan cases with no test (coverage, not per-test defects)
- MM-0: access matrix (removed); channel/bank/interleave never confirmed; real reset/reconnect; reserved-address and
  timeout negatives.
- MM-1: per-DM permission matrix and granularity; sub-word sizes; reset persistence and protected ranges; the plan's
  host-vs-CCE "same target" check (plan line 176).
- MM-2/MM-3: proof the no-op ran; reload verified per reinit cycle; reset-vector cross-CCE aliasing; MM-3-T5
  reset-recovery half; arg-count change; multicast (unsupported on DRAM cores).
- MM-4: end-of-region writes; read direction of the size matrix; other anchors; channel/bank/interleave; overlapping ops
  in flight; T5 negative controls.
- MM-5/MM-6: T5 recovery and negative controls (D2D reset, unmapped windows, invalid coordinates, route timeout);
  TL1 bank/region sweep; forced concurrency; bounded completion time.
- MM-7: port item 5 (tensor written after Start, released through WaitForCq, then prefetched); T5 fast-dispatch
  rejection (implemented at tensor_prefetcher_manager.cpp:1029-1031, not asserted); legal sender count; a DRAIN with
  several targets; a lagging receiver.
- MM-8: addrgen against DRAM and remote endpoints; im2col data movement; remote strided iDMA; cache-perf baselines; CI
  placement.
- 4.1: fence without flush; ordering by timestamps; single-line reuse and hit/miss metrics; hot set active during
  preload; cross-CCE/D2D with DRAM backing; an own-CCE "not snooped" negative.
- 4.2: Tiers B-E have no test and no "unsupported" record (Tier B only a README note), which the plan's acceptance
  criteria require; atomics concurrent with DMA/prefetcher traffic; reset with atomics in flight.
- 4.3: fault injection; reset in flight; ID at conversion boundaries; per-path ordering and max outstanding; ID width
  (probes removed); Quasar interop; per-transaction trace.

## Confidence
Nothing was run; only the fix 3 change was built. "Can fail" means "has a reachable assert on an independent oracle", not "was seen to fail".
- High (re-read in source by me after the reviewer): every DEFECT (NoOpLaunchCompletes, RuntimeFirmwareInitializes,
  PrefetchesLocalGddrToRemoteCceSram staging, Ordering fences, NoStaleIdAfterReinit, WithPrefetcher's host queueing
  and the kernel's same-source-same-destination stream loop, cce_idma_copy.cpp:645-688);
  WriteAfterRead's two launches; the route record being log-only; the CrossCce WarmOld gap; the chippy zero-fill
  behind the two failing MM-0/MM-1 legs (chippy `_write`, the UMD call site, and EmuAxiTransport overriding only the
  32/64-bit calls, emu_axi_transport.h:53-56, :108-111); and, after the critique pass, the unpoisoned staging in the
  four MM-6 contract copy tests (test_mimir_remote_dram.cpp:1418-1428, :1503-1512, :1585-1592, :1716-1762, against the
  poisoned MM-4 versions), EachHartAlone's hart-0-alone coverage (test_mimir_kernel_load.cpp:227-245), the subordinate
  enable gating (drisc.cc:71-77), and the README invocation rules behind fix 2 (README:54, :85-88).
- Medium-high: every other verdict rests on one reviewer's reading with file:line evidence and no second read. That
  includes the other flips: GatePreconditionsHold, FanOut and GcbSwitch (OK to WEAK/CONDITIONAL), CrossD2d, DmIdma.Copy,
  DmCache.Op, and the fixed-DEFECT calls in MM-0/1/2/3, MM-4, 4.1 and 4.2.
- Medium: verdicts marked CONDITIONAL on H1-H9; they cannot be settled from source.
- Low: whether the contention tests (4.2, 4.3 MultiHart/Fairness, MM-5 Bidir/FanOutFanIn, MM-6 Concurrent) ever see a
  race on the emulator; only run logs can show that.

### Critique pass (2026-11-03)
An independent critique agent checked the fixes, the DEFECT verdicts, a sample of OK and CONDITIONAL verdicts, and the
totals against the code. It found the totals consistent with the sections and all six original DEFECTs real. Changes
made from its findings (the first four re-checked in source by me):
1. Wrong verdict: the three MimirEmuCrossMimirNoc CopiesGddr* tests and MimirEmu/CopiesEveryMimirBlock (1x1) were OK
   but have the same unpoisoned-staging hole as PrefetchesLocalGddrToRemoteCceSram. Now DEFECT; fix 3 covers all of
   them. DEFECT total 6 to 10, OK 92 to 88.
2. Fix 1 would have repeated EachHartAlone and ArgsReplaceStale, and its claimed catch (an ignored enable mask writing
   slots 1-7) was wrong. Now: delete the test.
3. Fix 2 would fail a correct implementation in the README's single `*MimirEmu*` invocation, and "as row 2b does" was
   wrong. Now: delete the test.
4. Fix 6 can fail a correct implementation if H9 is "cached" (per-round re-reads of the same GDDR line within one
   launch). Now CONDITIONAL on H9, with a fresh-slot alternative that avoids it.
5. Fix 3 no longer adds a per-run seed (it would add H9 exposure for no gain once staging is poisoned).
6. Fix 4 keeps the consumer's fence and tests `.rl` first: without it the cdiscard is unordered against the data loads.
7. Fix 5's liveness check now uses the post-reopen batch's high water (no new launches), and its risk is stated as
   lower than DuplicateId's.
8. The conditional-to-DEFECT summary under Totals now lists every conditional that turns DEFECT on one outcome, not
   just eight.
9. The comparison with the earlier review notes that its 194 rows counted the Group B tests twice.
Findings 5-9 were taken from the critique's reading without a separate re-check (medium confidence).

## Open hardware facts
H1 Do same-CCE DM harts keep L1 D$ coherent through L2? (SharedL2, NeighbourWriteRace, DmCache.Op HartToHartCoherent,
ScopedLock MultiConsumer/RingWrap)
H2 Does FLUSH64 probe the issuing hart's L1 D$? (FullInvalidate, FlushFenceOrdering, DmCache.Op, CachePerf, ScopedLockClm)
H3 Do D2D DEBUG_LAST_* latch CCE-originated SPA traffic? (SameChipletControl)
H4 What a CCE load from an unmapped GDDR SPA returns. (No verdict depends on it any more.)
H5 Are D2D DEBUG_MISC bits sticky, and are the CCE AXI error bits write-1-to-clear? (MM-6 error-bit legs)
H6 Does TT_METAL_FORCE_REINIT reset the CCE RoCC command buffers? (NoStaleIdAfterReinit beyond its DEFECT)
H7 Is CCE DRAM init on the emulator reliably under the fixed 10 s wait? (MM-2)
H8 Is the addrgen loop wrap "restart at 0" or modulo, from a document independent of the emulator? (Im2Col_Dilation1)
H9 Are CCE hart loads from GDDR or remote-SRAM SPA addresses cached, with lines surviving between launches? (MM-4
SizeAlignmentMatrix and RepeatedAndOverlappingPerHart; MM-5/MM-6 read legs; kArgCheckAtAck reads)
Also open: does INVALIDATE64 drop other harts' L1 copies, and the issuing hart's own L1 line on hardware rather than
the emulator? Do L1/L2 contents survive process exit (only the harts are reset)? Does a 512 B iDMA command to the
sibling TL1 or GDDR stay pending past the next issue?

## Open questions for you
1. The four comment claims you did not know about (Decisions, item 2) decide 8 conditional verdicts (2 FanOut, 3 GcbSwitch, 2 BarrierThenDependent, DmIdma.Copy).
2. MimirDmRuntimeArgs.IllegallyModifyRuntimeArgs: its common-args leg tests "set twice with the same count throws";
   upstream changes the count (4 to 6). Is the current form the intended port?
3. Should README rows 1 and 2a say the 6-byte and 1-byte legs fail until chippy `_write` does read-modify-write?
4. Should a host DRAM access outside the 512 MiB view but inside the 8 GiB window be legal (HostSideRejections uses
   one as a sentinel)?
5. Is full_package in scope? MimirLocalDram, every remote test and the DM ports skip on it, though most bodies use
   Mimir-local addresses only in failure messages.
6. README staleness found along the way: README:429 still says the own-CCE snoop legs are asserted; README:303 and
   :329 still name MM42Detector; README:226 cites addrgen_api.hpp:38-39 (now :43-44); README:224 claims MimirLocalDram
   covers every size and alignment (writes only, one anchor); the dram_unary 1x1 Cce2/Cce3 skip note is stale.
7. Before fixing Ordering: do `.rl`/`.aq` on an AMO order the custom `tt.cache.cflush`/`cdiscard` instructions? (See
   "How to fix the DEFECTs", item 4.) Also: is a `tt.cache.cdiscard` ordered before later loads to the same line
   without a fence?
8. Before fixing NoStaleIdAfterReinit: is there a safe target an iDMA command can be sent to that never acks, without
   hitting the MIM-601 reset hang? If not, narrow the test to "trids usable after reinit". (Item 5.)
