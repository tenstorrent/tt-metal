# Mimir directed tests: can each one fail when its feature is broken?

Branch sagarwal/mimir_dram_tests, HEAD 16585a9c68a. Review only: nothing was built or run.
13 reviewers (one per plan section 4.1, 4.2, 4.3, MM-0 to MM-8, plus debug tools) judged 190 test declarations
plus 7 debug-tool rows. The test-to-section map and the full per-test reviewer reports, with file:line evidence, are appended
at the end (Appendix A and B). The state-persistence reports are Appendix C, and the iDMA snoop-tag reports are Appendix D.

Since the review (2026-10-29):
- The state-persistence question was settled from source (see State persistence). The 8 STALE rows are now 6 DEFECT
  and 2 WEAK, and the tables below include that.
- The iDMA snoop-tag scope was settled from the CCE RTL and the specs (see iDMA snoop-tag scope). Three ExternalWriters
  snoop-on legs expect a snoop the hardware never does, so they fail on a correct implementation. The PreloadChurn
  snoop leg's expectation is right, and its failure is unexplained. No counts change: ExternalWriters was already
  DEFECT and PreloadChurn stays WEAK.
- Working tree refresh (2026-10-29). The working tree has uncommitted changes since HEAD, and none of them has run on
  the emulator. Every file:line in the main body now points at the working tree; a citation into a file with no
  change, or whose cited lines did not move, is unchanged. Findings the changes address are marked "Addressed in the
  working tree (uncommitted, not yet run)" next to the original text. The changes:
  1. MimirPrefetcher.MM7T5TraceCaptureReplay is deleted, so MM-7 has no DEFECT left.
  2. MM-4 moves data by iDMA (your decision; no load/store control kept). The 5 MimirLocalDram tests run
     cce_idma_copy kModeTridBatch, with location-salted patterns for the reads (make_located_pattern,
     test_mimir_local_dram.cpp:119-132, finding 8) and one launch per copy in T3 and T5.
     MimirEmu.AllCcesCopyLocalMimirBlock is converted in place. The 4 MimirEmuSameMimirDirect instances now
     instantiate new iDMA classes (test_mimir_emu.cpp:268, :631, :821, :1119); the old classes keep only their
     MimirEmuCrossMimirNoc (MM-6) instances. cce_idma_copy gained a kModeTridBatch poll region (MIM-560). The 2 MM-4
     DEFECTs are addressed.
  3. A setup mismatch (identity, or the remote preflight of SS_BAR, ATT and interleave) now fails the test instead of
     skipping it: MIMIR_REQUIRE_IDENTITY and MIMIR_REQUIRE_REMOTE_PREFLIGHT (mimir_hw_state.hpp:115-128), at 32 direct
     call sites plus 27 MIMIR_ROUTE_GUARD sites in test_mimir_prefetcher.cpp (59 in 7 test files). Finding 3 is
     addressed.
  4. MimirTridIdma.DuplicateId asserts a pending high water above 1 on both legs and checks every command's data once
     when the trid reads 0 (kArgCheckAtAck, failure code kCodeNotLandedAtAck). Its WEAK is addressed.
  5. The README splits row 3 into 3, 3a (iDMA smoke) and 3b (MimirLocalDram, own row), removes row 64's test, says a
     setup mismatch fails, and lists the rows to rerun after the cce_idma_copy change (README:346).
  The reviewed totals stay as they were at HEAD; the second table under Totals gives what is still open in the working
  tree: 193 rows, 46 DEFECT, 49 WEAK, 95 OK and 3 addressed but not yet run.

## Verdict classes
- DEFECT: the test, or a leg of it, cannot fail by design (records only, logs only, an assert that cannot fail,
  a skip when the behaviour does not occur), or a fault in the very behaviour it is named for passes even though
  the test already holds the data to see it (e.g. its patterns repeat at the aliased location).
  Also DEFECT: a leg that asserts behaviour the hardware does not have, so it fails on a correct implementation
  (the own-CCE iDMA snoop legs; see iDMA snoop-tag scope).
- STALE (resolved 2026-10-29; no rows left): fixed values with no poison or seed beforehand, so a dropped write
  passes if an earlier run left the same value. The review left these conditional on whether TL1, GDDR and uncore
  state persist across processes in a session. They do (see State persistence). Each former STALE row is now DEFECT
  if a stale value can survive at its address and hide the fault the test is named for, and WEAK otherwise.
- WEAK: it can fail, but some named or planned part is not exercised, or the fault that passes is one no test of
  that kind can see. Two groups stay WEAK on purpose: same-path oracles (a host-only test cannot see a mapping that
  is wrong the same way for write and read; the plan's host-vs-CCE cross-check is missing instead, finding 11), and
  contention tests (the fault is caught whenever the race happens; nothing forces the race).
- OK: gates correctly on an independent oracle for what it claims.

## Totals (Group B tests count once in MM-4 and once in MM-6, so 194 rows)
| Section | Tests | DEFECT | WEAK | OK |
|---|---|---|---|---|
| MM-0 host DRAM | 7 | 3 | 3 | 1 |
| MM-1 host TL1 | 8 | 4 | 4 | 0 |
| MM-2 firmware load | 7 | 3 | 2 | 2 |
| MM-3 kernel load | 9 | 2 | 1 | 6 |
| MM-4 local DRAM | 10 | 2 | 2 | 6 |
| MM-5 local/remote TL1 | 11 | 8 | 0 | 3 |
| MM-6 remote DRAM | 10 | 5 | 0 | 5 |
| MM-7 prefetcher | 56 | 1 | 8 | 47 |
| MM-8 DM-suite ports | 12 | 0 | 3 | 9 |
| 4.1 cache | 21 | 8 | 4 | 9 |
| 4.2 atomics | 22 | 5 | 10 | 7 |
| 4.3 transaction IDs | 21 | 8 | 13 | 0 |
| Total | 194 | 49 | 50 | 95 |
| Debug tools (extra) | 7 rows | 0 | 3 | 4 |

Before the persistence answer the totals were 43 DEFECT, 8 STALE, 48 WEAK and 95 OK.

Still open in the working tree (uncommitted, not yet run). "Addressed" counts rows whose DEFECT or WEAK the working
tree addresses; they are not counted under DEFECT, WEAK or OK. OK rows that were only rewritten (MM-4: CceWritesTl1ToDram,
AllCcesCopyLocalMimirBlock and the 4 SameMimirDirect instances, now iDMA) stay under OK, but have not run either.
| Section | Tests | DEFECT | WEAK | OK | Addressed, not yet run |
|---|---|---|---|---|---|
| MM-0 host DRAM | 7 | 3 | 3 | 1 | 0 |
| MM-1 host TL1 | 8 | 4 | 4 | 0 | 0 |
| MM-2 firmware load | 7 | 3 | 2 | 2 | 0 |
| MM-3 kernel load | 9 | 2 | 1 | 6 | 0 |
| MM-4 local DRAM | 10 | 0 | 2 | 6 | 2 (CceReadsDramToTl1, LoopbackChecksIntermediate) |
| MM-5 local/remote TL1 | 11 | 8 | 0 | 3 | 0 |
| MM-6 remote DRAM | 10 | 5 | 0 | 5 | 0 |
| MM-7 prefetcher | 55 | 0 | 8 | 47 | 0 (MM7T5TraceCaptureReplay deleted) |
| MM-8 DM-suite ports | 12 | 0 | 3 | 9 | 0 |
| 4.1 cache | 21 | 8 | 4 | 9 | 0 |
| 4.2 atomics | 22 | 5 | 10 | 7 | 0 |
| 4.3 transaction IDs | 21 | 8 | 12 | 0 | 1 (DuplicateId) |
| Total | 193 | 46 | 49 | 95 | 3 |
| Debug tools (extra) | 7 rows | 0 | 3 | 4 | 0 |

Finding 3 (setup mismatch) was never counted as DEFECT, so addressing it changes no count.

Whole tests that always pass for the property they are named for: MM7T5TraceCaptureReplay (bare skip; since deleted),
MimirKnownIssue.Mim560Detector, MimirCacheProbe.SnoopConfig, MimirCacheProbe.L2Geometry, MimirAccumProbe.DirtyLine,
MimirTridProbe.Width / PendingVisible / DepthBeyond16 / Scope, MimirHostDram.DiscoverAccessMatrix.

## Cross-cutting findings
1. Committed UMD pointer is wrong (lead verified). HEAD records UMD 292fe20, which has no mimir_local_address_windows
   or get_mimir_channels. Only the uncommitted working-tree checkout f0d497f has them. The branch as committed does
   not build these tests. Unchanged in the working tree: the checkout is still f0d497f and HEAD still records 292fe20.
2. Route evidence is only logged (lead verified, mimir_access.hpp:759-788). Every remote TL1 and remote DRAM test
   builds a predicted SPA / ATT entry / post-SS_BAR record with crosses_d2d hard-coded true, and only logs it. The
   plan's "exactly one translation" check cannot fail. Data checks are sound (host readback is independent of the
   kernel's route; seeds and decoys catch misroutes).
3. Setup mismatch skips (lead verified, test_mimir_tl1.cpp:129-142 at HEAD). A mismatch in identity, SS_BAR, ATT or GDDR
   interleave/capacity skips every MM-0/MM-1 host test, every remote test in MM-5/6/7/8 and five 4.3 tests
   (local legs included). No test fails on it. You were unsure whether skip or fail is better. Recommendation: keep
   the skip as a precondition, and add one test that fails on the mismatch, so a broken bring-up cannot produce a
   run that is all skips. Not counted as DEFECT above.
   Addressed in the working tree (uncommitted, not yet run), by a different route than recommended: every one of those
   tests now fails on the mismatch. MIMIR_REQUIRE_IDENTITY and MIMIR_REQUIRE_REMOTE_PREFLIGHT
   (mimir_hw_state.hpp:115-128) skip only when the watcher is on and otherwise ASSERT that the result is empty. The
   skip helpers no longer return the preflight (test_mimir_tl1.cpp:129-141 now ends `return "";`, also
   test_mimir_dm_idma.cpp idma_skip_reason, test_mimir_prefetcher.cpp:844 and :1612, and the deleted
   remote_legs_skip_reason in test_mimir_trid.cpp). mimir_identity_result() and mimir_remote_preflight_result() are now
   called only inside the two macros, so no GTEST_SKIP path carries an identity or preflight mismatch. Call sites:
   test_mimir_host_dram.cpp (5), test_mimir_host_tl1.cpp (5), test_mimir_tl1.cpp:620 onward (8),
   test_mimir_remote_dram.cpp:1001 onward (5), test_mimir_trid.cpp:1287 onward (7, two in fixture SetUp, where an
   ASSERT stops the test body), test_mimir_dm_idma.cpp:556 (1), test_mimir_prefetcher.cpp:962 (1), plus 27
   MIMIR_ROUTE_GUARD uses (test_mimir_prefetcher.cpp:1635-1641), which run the preflight only for a route that spans
   Mimirs. The post-CreateDevice remap check was already EXPECT/ASSERT at HEAD and is unchanged. Still skips, outside
   this finding: the watcher (finding 10), and MIMIR_HOST_ONLY_GUARD (mimir_hw_state.cpp:463-481), which skips when a
   CCE has harts out of reset or, where the test asks for it, its uncore held in reset. A bring-up that leaves an uncore
   held still turns MimirHostDram.ReconnectPersists and HostSideRejections (test_mimir_host_dram.cpp:426, :504) and
   all five MM-1 host tests (test_mimir_host_tl1.cpp:273 onward) into skips. The ASSERT is a precondition, not the
   property any test is named for, so no per-test verdict changes.
4. Stale data. Several tests write constants with no poison first. State persists across processes and bring-up
   within an emulator session (see State persistence), so 6 of the 8 such rows are DEFECT and 2 are WEAK.
5. Contention, overlap and concurrency are never required. 4.2 (7 tests), 4.3 MultiHart/Fairness, MM-5
   Bidir/FanOutFanIn, MM-6 Concurrent, debug-tools MultiWriter: lost updates are caught when a race happens, but
   nothing checks that participants overlapped.
6. Mis-cited Jira (lead verified MIM-628). MIM-628 covers the iDMA sideband done_id only; DV says the command-buffer
   path "uses the real response" and Jason Dubrow agrees. test_mimir_trid.cpp:7-8 uses it to justify recording,
   not gating, stale data after a trid ack. GRENSV-2290 is about MMIO/SPA store snoop, but the 4.1/4.2 tests cite it
   for the iDMA snoop tag (test_mimir_dm_cache.cpp:594-600, test_mimir_atomics.cpp:63-65). Chippy files
   cce_xcce_snoop_dma under GRENSV-3774 (load_fw_verify_scratch/test_config_mimir_cce_xcce_snoop.yaml:4, lead
   verified), and that test covers only a sibling CCE's TL1-to-TL1 copy.
7. Seed mask (lead verified). gtest's random seed is at most 99999, so `mimir_test_seed() & 0x7FF00000u`
   (test_mimir_dm_cache.cpp:1608) is always 0 and the "seeded" external-writer value never changes.
8. Pattern aliasing (lead verified, mimir_patterns.hpp:53,63). make_pattern truncates the base address to 32 bits
   and the base is Mimir-local, so both views of a Mimir and both Mimirs get identical words. A local read that
   drops bit 32, or goes to the other Mimir, still passes (MM-4).
   Addressed in the working tree for MM-4 (uncommitted, not yet run): make_located_pattern
   (test_mimir_local_dram.cpp:119-132) XORs a salt of 0x9E3779B9 * (((mimir << 8) | view) + 1) into the Prng seed and
   into every AddressDerived word. The multiplier is odd, so each (Mimir, view) gets a different salt, and a read
   that drops bit 32 or goes to the other Mimir finds words written with another salt (or poison). The read sources
   of T2, T3 and T5 use it (:545, :632, :879, :886). make_pattern itself is unchanged (mimir_patterns.hpp:53,63), so
   other users keep identical words across views and Mimirs; T1 and T4 still use it for TL1 sources, which is
   harmless there because their DRAM destinations are poisoned.
9. Runtime 1-byte write (lead verified, risc_firmware_initializer.cpp:1445-1450). The runtime writes 1 byte to
   fw_shared_globals_ready. Per your answer, the chippy zero-fill of the rest of the word is corruption. No test can
   see it: FirmwareLoadPreflight writes and reads only that byte. Medium confidence that it matters: it depends on
   what shares that word.
10. TT_METAL_WATCHER turns most suites into skips (mimir_hw_state.cpp:418-423), with a reason ("SMC register reads")
    that most of those tests never do. Not addressed in the working tree: the check moved out of identity_result()
    and preflight_result() into the two MIMIR_REQUIRE_* macros (mimir_hw_state.hpp:115-128), which still skip on it,
    and host_only_guard_reason (mimir_hw_state.cpp:465) and the route and iDMA skip helpers still return it.
11. Plan cases with no test at all: MM-4-T5 negative controls, MM-5-T5 and MM-6-T5 recovery/negative controls,
    4.2 tiers B-E (no test and no "unsupported" record, which the plan's acceptance criteria require), MM-7 port
    item 5 (DRAM writes visible before a queued request), perf baselines for 4.1 cache-perf.

## State persistence within one emulator session (settled 2026-10-29)
Source reading only; nothing was run. Four agents read the emulator model and chippy bring-up, the Mimir and CCE
specs, Metal and UMD process start and exit, and recorded runs and discussions (Appendix C). The lead checked these
points in source:
- Startup and exit reset only the CCE harts: `ALL_NEO_DMS` and `ALL_DATA_MOVEMENT` (risc_firmware_initializer.cpp:51-62).
- UMD changes the uncore reset bit only for `RiscType::ALL` (umd tt-umd/src/tt_device/tt_device.cpp:782-812).
- Metal sends no SiVal `INIT` (`emu_skip_init = true`, tt_metal/llrt/tt_cluster.cpp:562-575).
- The SiVal server does nothing when a client connects or disconnects (grendelemulation
  py/syseng_common/server.py:308-339).
- Bring-up only releases cold resets (chippy sival_arch/utility/cold_reset.py:190-193), and its uncore release holds
  the harts (sival_arch/utility/cce.py:386-390).

Nothing in the emulator clears state between client processes. Only a SiVal `INIT` (a BP_RESETN pulse) resets the
design, and neither Metal nor the bring-up scripts send one. So the answer depends only on what software writes.
TL1 offsets below are CCE-local; the host adds MEM_CCE_L1_NOC_OFFSET (0x20_0000_0000).

| | New test process | Chippy bring-up |
|---|---|---|
| TL1 | Kept, except what the first device open in a process rewrites: mailbox fields (0x80-0x8080), firmware text (0x8080), firmware data and .bss (0xE080), the 8 hart TLS/stack slots (0x17080-0x27080), the bank tables (0x27480) and the SPA routes (0x27C80). Kernel config, kernel text, kernel global slots 1-8 and all of UNRESERVED (0x884C0-0x400000) are never written. A host-only process writes nothing. | Kept. Bring-up writes no TL1 and never puts a CCE into cold or uncore reset. |
| GDDR | Kept. Nothing writes GDDR on open or close (`TT_METAL_CLEAR_DRAM` unset; DRAM barriers are no-ops on the emulator). Allocated pages are never freed. | Kept. Bring-up only takes GDDR out of reset and sets control registers; no training or init. |
| Uncore registers | Kept, except the first device open writes 0x8080 to all 8 hart reset vectors. The hart reset bits in `pf_ctrl` are set at open, cleared once firmware is loaded, and set again at exit. The uncore bit, the DMRISC and MLA remaps, the TL1 base and the group hash are never touched. | Partly rewritten: the TL1 base, the group hash, remap entries 0-2 of both remaps, and `pf_ctrl` (uncore released, harts held). Kept: the reset vectors, remap entries 3 and up, remap attribute bits other than `valid` (for example `cmd_snoop`), and `group_hash_fn0_addr10`. |

Confidence:
- High that nothing clears state between processes or across bring-up.
- Medium on the exact byte ranges rewritten at open. They depend on the built firmware's section sizes, which no one
  inspected.
- Unknown whether any reset auto-zeroes SRAM in the RTL, which is not in the checkout. That only matters after an
  `INIT`.
- No recorded run checks this directly. Three invocations would confirm it on the emulator: (1) write markers to TL1
  above 0x884C0, to GDDR and to remap entry 3; (2) read them back in a new process; (3) run bring-up, then read them
  back again.

Related findings:
- Runs recorded before commit 3ecb73d1637 (2026-10-21) used different reset rules. Before it, each process's first
  open reset the uncore and rewrote the DMRISC remap. The README results recorded at 3ef0a46e42a, 61d03f6e674 and
  3a3d04f9bd2 ran that way; HEAD does not.
- README:78 (line 75 at HEAD) is out of date. It says CreateDevice releases the uncore and programs the remap; nothing in Metal
  calls that code now (8fa06c8b177 removed the Cluster wrapper). Fixed in the working tree: README:78 now says chippy
  bring-up programs the remap and nothing rewrites it at CreateDevice (mimir_hw_state.hpp's remap comment likewise).
- kCceSramTestOffset = 0x800 (context/mimir_contract.hpp:36) sits inside the mailbox's launch-message ring, not in
  free TL1. A host-only process inherits the previous host-only process's value there, and the next device open
  overwrites it. Medium confidence on which ring entry, high that it is in the ring.
- A killed process skips the exit hart reset, so the harts keep running until the next bring-up holds them.

Former STALE rows:
| Test | Section | What it writes, and where | Survives from an earlier process? | New verdict |
|---|---|---|---|---|
| AllocatedDramBufferHostLoopback.RoundTrip | MM-0 | Fixed 0xA11xxxxx pattern in the first DRAM buffer (test_mimir_emu.cpp:637) | Yes: GDDR | DEFECT: after one passing run, a dropped write passes |
| MimirHostTl1.BoundaryAndIsolation | MM-1 | Host-only; fixed `make_guard` tags at TL1 region edges and at 0x1008 and 0x1038 (test_mimir_host_tl1.cpp:422; mimir_patterns.hpp:83) | Yes: no process writes most of those windows | DEFECT: a rerun passes a dropped edge write. Aliasing between CCEs or windows is still caught |
| CceSramRoundTripThroughMetalCluster | MM-1 | Host-only; 0xC0FFEE01+cce at 0x800 (test_mimir_backend.cpp:32) | Yes, until a device open | DEFECT: a rerun, or a run right after ChannelsDoNotAlias (same word, same value), passes a dropped write |
| MimirEmu.CceSramChannelsDoNotAliasThroughMetalCluster | MM-1 | Same word and values (test_mimir_backend.cpp:61) | Yes, until a device open | WEAK: a dropped write passes on a rerun, but the aliasing it is named for is still caught |
| CceSramRoundTripThroughUnitMesh | MM-1 | Opens a device, then writes 0xC0FFEE03+cce at 0x800 (test_mimir_emu.cpp:62) | No: the open rewrites the launch ring first | WEAK: it writes into a firmware mailbox, not free TL1 |
| MimirFwLoad.LoadAndReachInitDone | MM-2 | Checks the reset vectors (0x8080), mailbox fields and firmware text (test_mimir_fw_load.cpp:134, 187) | Reset vectors: yes; every open writes the same 0x8080 and nothing else writes them | DEFECT (reset-vector leg): a dropped reset-vector write passes after any earlier process. The text window is zeroed first and the runtime seeds the mailbox fields at open, so those legs stand |
| MimirEmu.RuntimeFirmwareInitializesThroughPublicDeviceApi | MM-2 | Only gate is the runtime's own DONE wait (test_mimir_emu.cpp:85) | Yes: firmware image and reset vectors | DEFECT: if the image write fails, the previous process's firmware runs and reports DONE. test_mimir_fw_load.cpp:171-183 zeroes the window for exactly this reason; this test does not |
| HartZeroRunsDramKernel.WritesMagic | MM-3 | Fixed 0xC0FFEE04+cce at the UNRESERVED base (test_mimir_emu.cpp:97) | Yes: nothing writes UNRESERVED | DEFECT: a kernel that never runs passes on a rerun |

## iDMA snoop-tag scope (settled 2026-10-29)
Source reading only; nothing was run. Two agents read the CCE RTL (aus-gitlab soc/cce @ ce574b32, opensrc-idma
@ 6a9cc46a, tt/master), chippy, the Grendel CCE spec v1.0, the Mimir spec v1.14, Jira and Slack (Appendix D). The
RTL reading and the spec reading reached the same answer independently.

The snoop tag only acts on writes that arrive in a CCE's TL1 from outside that CCE:
- RTL: the snoop master sits only on the inbound NoC-to-SPM path ("observes the requests from NOC to SPM ... For any
  write request that has AXI_CMD_SNOOP_BIT set, it sends out a snoop request to DMRISC L2 front end",
  cce_axi_snoop_master.sv:20-21). An iDMA write to an address in the CCE's own LOCAL_TL1_BASE window takes CCE xbar
  port 1 and a direct TL1 write port with no snoop master (tt_cce_mimir_wrap.sv:965-969, 1435-1451). Other writes
  leave through the ITN with bit 8 kept (:1217) and pass the target CCE's snoop master.
- Spec: the snoop block handles "all requests from ITN to SRAM" (Mimir spec §7.2.2), while "requests from AXI-DMA are
  either routed to local SRAM or ITN via CCE xbar" (§7.2.3, repeated in the CCE spec §5.2.2). The CCE spec describes
  the snoop only for a remote CCE writing this CCE's SPM (§3.4.4, §7.8). No sentence says outright that own-CCE iDMA
  writes are not snooped; that comes from where the block sits.
- DV: every Jira snoop test for iDMA is sibling TL1-to-TL1 (GRENSV-1577, GRENSV-1571, chippy cce_xcce_snoop_dma).
  No own-CCE, GDDR-sourced or preload snoop test exists. The designers' scope ticket GRENSV-2011 ("Clarify when ROCC
  snoop applies") has been To Do since 2026-04-23 with no answer, and nobody on record discusses the emulator results.

| Copy type, as the tests run it | Intended to snoop? | Emulator | Tests and verdict effect |
|---|---|---|---|
| Own CCE, own TL1 to own TL1 | No, by design | Not snooped | ExternalWriters `*_LocalIdma_SnoopOn` (README row 99): wrong expectation; asserted unconditionally (test_mimir_dm_cache.cpp:1636-1675), so it fails on a correct implementation |
| Own CCE, local GDDR to own TL1 | No, by design | Not snooped | `*_LocalGddrIdma_SnoopOn` (row 100): wrong expectation, same |
| Own CCE, remote GDDR to own TL1 | No, by design; the writer is the reader's own CCE despite the name | Not snooped | `*_RemoteGddrIdma_SnoopOn` (row 101, 2x only): wrong expectation, same |
| Sibling CCE, TL1 to this TL1 | Yes | Snooped (VC 2) | CrossCceSnoopOn, DirtyLineRace Cfg6, CleanDirtyLineSnoopedWrite: expectation right; they pass |
| Sibling CCE, local GDDR to this TL1 (preload) | Yes: the RTL sets the bit for every source and the spec has no source condition | Not snooped: 1664-1696 of 2048 words stale (VC 5) | PreloadChurn snoop leg (row 109): expectation right; failure unexplained |

Confidence:
- High for the three own-CCE legs: RTL, spec and emulator agree. Caveat: nobody confirmed that soc/cce @ ce574b32
  matches the RTL the emulator was built from, but the emulator's behaviour matches it.
- Low on why the preload leg fails. Source (GDDR, not TL1), virtual channel (5, not 2) and size (8 KiB, not at most
  256 B) all differ from the passing sibling tests. The NoC/ITN RTL that carries the tag between CCEs was not
  reachable, and the step from PACKET_TAGS.snoop_bit to the iDMA's `cmd_user[0]` was traced by name only. A sibling
  run that changes one variable at a time (TL1 source on VC 5, GDDR source on VC 2, or a 64 B GDDR copy) would
  separate them. It may be a real hardware or emulator bug worth filing.

Related findings:
- README:411 (cited as :406 at HEAD, where the sentence was at :407) frames the row 99-101 failures as an emulator gap
  ("expected to fail on this emulator until the tag is honoured"). Per the RTL they fail by design. Fixed in the
  working tree: README:411 now says the own-CCE legs fail by design and PreloadChurn's failure is unexplained.
- These rows (99-101, 109) and the rest of 4.1/4.2/4.3 launch cce_idma_copy, which changed in the working tree (poll
  region, kArgCheckAtAck, kNumArgs 37 to 38, and every mode but kModeTridBatch and kModeAccumulate now rejects a
  nonzero poll region). README:346 lists them for a rerun. No verdict here changes; the snoop legs do not use the new
  arguments.
- The gate constants `kIdmaSnoopBitRefreshesCache` and `kIdmaSnoopRefreshesHartCache` call the expectation
  "required" by GRENSV-2290; see finding 6.
- A test that fails on a correct implementation is a defect of the opposite kind to the rest of this report: it
  cannot pass, rather than cannot fail. It is listed under DEFECT.

## Per section

### MM-0 (reviewer mm0)
- DEFECT MimirHostDram.DiscoverAccessMatrix: no traffic; GDDR registers logged; only asserts are "legal rows > 0" and
  "negative rows > 0", true by construction (lead verified test_mimir_host_dram.cpp:252-253).
- DEFECT MimirHostDram.HostSideRejections: the 6-byte leg asserts the zero-fill of bytes 6-7 as expected (:596-617);
  you said that is corruption. Every other rejection happens in UMD/chippy before the device.
- DEFECT AllocatedDramBufferHostLoopback.RoundTrip (was STALE): constant pattern at a deterministic GDDR address, so
  after one passing run a dropped write passes; the channel-page read repeats ReadFromBuffer's own call.
- WEAK MimirHostDramView.BoundaryAndAlias: swapped address bits pass (same mapping both ways); the GiB-straddle loop
  and view-end branch never run with 512 MiB views.
- WEAK MimirHostDram.ReconnectPersists: nothing resets or reconnects (no INIT, no socket close), so the data can
  only persist.
- WEAK DramChannelsDoNotAliasThroughPublicDeviceApi.RoundTrip: on 2x it pairs only across Mimirs (p with p+n/2);
  same-Mimir view pairs are never checked; one word.
- OK MimirHostDramView.PatternReadback.

### MM-1 (reviewer mm1)
- Suite-wide: write and readback share one Cluster/UMD/chippy path, so a consistent CCE permutation (CCE0/CCE1 or
  M0/M1 swapped) passes every MM-1 test. Plan line 175 (host and CCE-DM reach the same target) is not done in MM-1.
- DEFECT MimirHostTl1.DiscoverCceTl1Map: the permission matrix is the logged string "host RW (assumed)"; granularity
  is a log line.
- DEFECT MimirFwPreflight.FirmwareLoadPreflight: the reset-vector leg is logged only; the SPA-routes chunk is
  missing; the 1 B chunk cannot see zero-fill.
- DEFECT BoundaryAndIsolation, CceSramRoundTripThroughMetalCluster (were STALE): host-only, fixed values, no poison,
  and nothing clears them between host-only processes, so a rerun passes a dropped write.
- WEAK CceSramChannelsDoNotAliasThroughMetalCluster (was STALE): same word and values as the RoundTrip test, so a
  dropped write passes on a rerun, but aliasing is still caught. CceSramRoundTripThroughUnitMesh (was STALE): its
  device open rewrites the word first (0x800 is in the mailbox launch ring), so no earlier value survives.
- WEAK PerCceVisibility (same-path oracle), HostSideRejections (past-end rejected by the GDDR-offset check, any
  UmdBaseException passes; reset and protected-range legs absent).

### MM-2 (reviewer mm2)
- DEFECT MimirFwLaunch.NoOpLaunchCompletes: `EXPECT_NE(enables, 0)` reads back the host's own write, which firmware
  never clears in host mode, and firmware reports DONE even when it skips the kernel (lead verified
  test_mimir_fw_load.cpp:334-339, drisc.cc:161-177).
- DEFECT LoadAndReachInitDone (was STALE; reset-vector leg): every open writes the same 0x8080 and nothing else
  writes the vectors, so a dropped reset-vector write passes after any earlier process. The firmware-text leg is
  zeroed first and the runtime seeds the mailbox fields at open, so those legs stand.
- DEFECT MimirEmu.RuntimeFirmwareInitializesThroughPublicDeviceApi (was STALE): the only gate is the runtime's own
  DONE wait, and the previous process's firmware stays in TL1, so a failed image write still reports DONE.
- WEAK ImageManifest (expected values come from the same HAL constants as the ELF), ReloadsOnEveryOpen (proves
  restart, not reload).
- OK AllCcesIndependent, SkipLoadNeverDone (small gap: a fixed 10 s DRAM-init wait).

### MM-3 (reviewer mm3)
- DEFECT AllHartsRunDramKernel.WritesMagic (lead reclassified from STALE): the kernel's address and value are
  compile-time args with no mhartid, so a launch that swaps two harts' text offsets still fills all 8 words, even on
  a fresh session. That is MM-3-T2's target. Its result words are also never poisoned.
- DEFECT HartZeroRunsDramKernel.WritesMagic (was STALE): the result word at the UNRESERVED base is never poisoned and
  nothing else writes it, so a kernel that never runs passes on a rerun.
- WEAK MimirCacheProbe.EightHartLaunch: status address comes from a runtime arg, so RTA swaps across harts pass.
- OK MinimalKernelExecutes, EachHartAlone, ArgsReplaceStale, SameBinaryAllCcesUnicast, MimirMm4xCompile.Kernel,
  FirmwareKeepsRunningAcrossReopen (OK per your answer that it only needs launch-after-reopen; its name claims more).
- Conditional (you did not know): if MM-3-T1/T4 need binary-placement readback, the T1/T4 tests are WEAK.

### MM-4 (reviewer mm4)
- Working tree: every MM-4 test now moves data by iDMA. The 5 MimirLocalDram tests run cce_idma_copy kModeTridBatch
  through run_hart_copies (test_mimir_local_dram.cpp:303-323) and gate on copies_completed (:379-405: status code and
  commands issued per hart) plus host readbacks with guards. Nothing below has run.
- DEFECT CceReadsDramToTl1 and LoopbackChecksIntermediate (lead reclassified from WEAK): identical patterns across
  views and Mimirs (finding 8), so a read that drops bit 32 or goes to the other Mimir passes. Misrouting is what
  the read case targets, and unique patterns would expose it.
  Addressed in the working tree (uncommitted, not yet run): both read sources come from make_located_pattern
  (CceReadsDramToTl1 :545-551, LoopbackChecksIntermediate :632-638), salted by Mimir and view (finding 8), and every
  TL1 destination is poisoned with guards. Both would be OK. Note on LoopbackChecksIntermediate (:585-588): the read
  and the write are now two launches, ordered by the host with the intermediate TL1 checked in between, so the CCE's
  own read-then-write ordering is not exercised; the intermediate check, which T3 is named for, is stronger.
- WEAK SizeAlignmentMatrix (write direction only, 4 B offsets, one anchor); RepeatedAndOverlappingPerHart (T5
  negative half absent).
  Still WEAK in the working tree. SizeAlignmentMatrix (:717) still writes only, uses offsets {0, 4, 60, 124,
  32 KiB - 4} and the first anchor that fits (:739). RepeatedAndOverlappingPerHart (:834) now runs each op as one
  launch of all harts (:832-833) with located patterns per repeat (:879, :886); the T5 negative cases are still absent.
- OK CceWritesTl1ToDram, AllCcesCopyLocalMimirBlock, and the four SameMimirDirect instantiations.
  Still OK in the working tree, rewritten for iDMA and not yet run. CceWritesTl1ToDram (:437) is kernel-verified
  with poisoned guarded destinations. AllCcesCopyLocalMimirBlock (test_mimir_emu.cpp:827-900) now poisons the staging
  (:851, :864), which the reviewer said would matter for a DMA read, and tags inputs by CCE (:857). The SameMimirDirect
  instances now instantiate IdmaCopiesGddrWithCompileTimeArgs (test_mimir_emu.cpp:268, kernel
  cce_idma_gddr_round_trip.cpp), IdmaCopiesGddrWithRuntimeArgs (:631) and IdmaCopiesGddrLargerThanAlignment (:821),
  both through idma_round_trip_through_staging (:479-549), and AllHartsIdmaWriteGddr (:1119, kernel
  cce_idma_gddr_write_one_uint32.cpp, one trid per hart). Each poisons the staging and clears the destination. The
  LargerThanAlignment encoding note (value OR'd into the location bits) no longer applies: the helper XORs the
  location tag into the high half (:513).
- Note: ASSERT_TRUE(run_cce_copy_ops(...)) always succeeds (mimir_access.hpp:502-567); gating comes from the done/XOR
  checks and host readbacks. "Direct" vs "Noc" in the suite names matches no code difference.
  Changed in the working tree: no MM-4 test calls run_cce_copy_ops now. The note still holds for its MM-5 and MM-6
  callers (test_mimir_tl1.cpp:284, :400, :433, :543; test_mimir_remote_dram.cpp:397). The suite names now differ in
  code, but still do not describe it: MimirEmuSameMimirDirect instantiates the iDMA classes and MimirEmuCrossMimirNoc
  the SPA load/store classes (test_mimir_emu.cpp:1211-1253); neither path is "direct" or NoC.
- Conditional (you did not know): every MM-4 test uses 32-bit hart loads/stores, not iDMA. If MM-4 means iDMA, all
  10 are WEAK for coverage.
  Resolved: you decided MM-4 means iDMA, and in the working tree all 10 MM-4 rows run iDMA, so this condition no
  longer applies.

### MM-5 (reviewer mm5)
- DEFECT (route-evidence leg, finding 2): all 8 remote tests in test_mimir_tl1.cpp (3 Direct, 3 Routed, Bidir,
  FanOutFanIn). Also WEAK: both SizeAlignment tests never test the read direction and have no leading guard;
  Bidir and FanOutFanIn never show overlap.
- OK SameChipletControl (its no-ATT leg is weak: it relies on DEBUG_LAST latching CCE traffic, and on 1x1 it can't
  fail for Cce1To0), WritesCrossMimirCceSram (narrow), PrefetchesLocalGddrToRemoteCceSram.
- Working tree: the 8 remote tests now fail on a preflight mismatch instead of skipping (test_mimir_tl1.cpp:620
  onward; finding 3). No verdict changes.

### MM-6 (reviewer mm6; lead normalized the route-evidence leg to DEFECT, as in MM-5)
- DEFECT WriteFromTl1, ReadToTl1, Tl1RoundTrip, TrafficMatrix, MimirRemoteMixed.Concurrent: stage evidence only
  logged; error bits already set are logged as "inconclusive". Also: Tl1RoundTrip, TrafficMatrix and Concurrent read
  legs draw all-zeros/all-ones patterns, which a misrouted read can match (lead verified
  test_mimir_remote_dram.cpp:50-51 vs :455-457). TrafficMatrix uses one anchor and no channel/bank/interleave sweep.
- OK CopiesEveryMimirBlock and the four CrossMimirNoc instantiations.
  Unchanged in the working tree. The CrossMimirNoc instances are now the only instances of the original SPA
  load/store classes (test_mimir_emu.cpp:180, :553, :741, :1044); their MM-4 twins moved to iDMA classes.

### MM-7 (reviewer mm7)
- DEFECT MM7T5TraceCaptureReplay: bare GTEST_SKIP. Deleted on 2026-10-29.
  Addressed in the working tree (uncommitted): the test is gone from test_mimir_prefetcher.cpp and README row 64 now
  reads "Removed". It leaves the counts.
- WEAK MM7GateStartStopWorks (only "does not throw"); the 7 T3 chunk-ladder rows (SubChunk L/D, BelowStageThird L/D,
  PartialStageChunk L, AboveStageThird L/D): the chunk size used is never observed, so ignoring the stage-ring cap
  passes.
- OK the other 47. Caveats: about 20 D2D rows skip on a setup mismatch; the receiver FIFO is never poisoned, so with a
  pinned --gtest_random_seed a rerun could pass a sender that sends credits without data; the padding oracle
  restates cce_remote_cb.h line for line.
  Working tree: the setup-mismatch caveat is addressed (uncommitted, not yet run). MIMIR_ROUTE_GUARD
  (test_mimir_prefetcher.cpp:1635-1641) skips only for the watcher or a missing CCE, then fails on a preflight
  mismatch when the route spans Mimirs; MM7GcbSmokeValidatorD2d does the same (:962). The other two caveats stand.

### MM-8 (reviewer mm8)
- WEAK DramUnary.ReaderWriterHandshake (the reader's slot-reuse wait is caught only if it loses a race),
  CachePerf.WritePath (perf baselines not implemented), MeshBufferDram.InterleavedPlacement (oracle shares Cluster's
  view translation).
- OK Addrgen.Pattern, Im2col.Pattern (Dilation1: medium; its wrap rule's source document was not found),
  DmCache.Op, ScopedLockClm.CacheOps, DmIdma.Copy, MeshBufferDram.WriteRead, MultipleWriteReadRounds, both
  RuntimeArgs tests.
- Working tree: DmIdma.Copy's remote cases now fail on a preflight mismatch instead of skipping
  (test_mimir_dm_idma.cpp:555-557; finding 3). No verdict changes.

### 4.1 cache (reviewer s41-cache)
- DEFECT SnoopConfig, L2Geometry (record only); FlushFenceOrdering (legs 0-1 not gated; a no-op cflush passes the
  gated legs if FLUSH64 probes L1); SharedL2 x3 (the fence-only read is record only; the EightWriters compare sets
  expected = observed, lead verified :1385); ExternalWriters (InvalidateOnly instances check nothing about the
  reader); DirtyLineRace (only config 0x6 gates; 7 configs `continue` past every data check, lead verified
  :2091, :2178).
- WEAK ReuseThrash (conflict stride degenerates to sequential below 16 KiB), PreloadChurn (one 8 KiB preload into the
  hot set; churn not tested), ScopedLock MultiConsumer and RingWrap (a no-op acquire likely still reads NEW).
- OK FullInvalidate, DirtyEvictionPartialLine, L1DcacheFlushLine, L1DcacheFlushFull, CrossCce, CrossCceSnoopOn,
  CrossD2d, PerfSweep, NonSnoopingProducer.
- Snoop-tag legs (settled, see iDMA snoop-tag scope): ExternalWriters' SnoopOn instances for own TL1, local GDDR
  and remote GDDR are a second DEFECT reason for ExternalWriters. Each copy is the reader's own CCE writing its own
  TL1, which the hardware never snoops, so those legs fail on a correct implementation. The PreloadChurn snoop leg
  (a sibling copy) expects the right behaviour and gates correctly. Its emulator failure is unexplained.

### 4.2 atomics (reviewer s42-atomics)
- DEFECT Ordering (release-only leg record only; the gated leg passes with broken .rl/.aq because flush_l1_dcache ends
  in a full fence, lead verified), CleanDirtyLine (no-invalidate leg record only), AccumProbe.FmtOpMatrix (points 1-7
  never compared though the host has a model), AccumProbe.DirtyLine, Mim560Detector.
- WEAK TwoHartAmo, Hotspot, LrScCas, Storm, Soak, IdmaAccum.AllHartsShared, NeighbourWriteRace (overlap not
  required); OpMatrix (one random operand per op, so an amomin/amomax signedness swap is missed about half the
  time); CleanDirtyLineSnoopedWrite; IdmaAccum.MaxOutstanding (depth never asserted).
- OK Distributed, Wrap, ItnControl, IdmaAccum SingleHart / AllHartsDisjoint / BackToBack / MixedPathControl.
- Note: accumulate operands are masked to 0x000FFFFF, so format semantics (sign, saturation) are never gated;
  MixedPathControl is sequential, not the plan's concurrent mixed-path control.

### 4.3 transaction IDs (reviewer s43-trid)
- Suite-wide: data checks are sound, but trid behaviour is never gated. wait_for_trids treats pending == 0 as acked
  (lead verified cce_idma_copy.cpp:305-316), so ignoring TR_ID, aliasing its bits, or a counter that never
  increments all pass; kernel data polls hide early acks.
  Working tree: still true for every test but DuplicateId, which now gates both (see below). Every 4.3 test that
  skipped on a remote preflight mismatch (MixedTargets, NominalDepth, Saturate, Fairness, Soak, the RemoteTl1 and
  RemoteGddr fixtures, WithPrefetcher) now fails on it (test_mimir_trid.cpp:1287 onward; finding 3).
- DEFECT PendingVisible, DepthBeyond16, Width, Scope (property record only); BarrierThenDependentReadAfterWrite
  (dependent read accepts a data timeout, stale words recorded, lead verified test_mimir_trid.cpp:1546-1552); WriteAfterRead (the hazard
  cannot occur: a data poll and a launch boundary enforce the order); NoStaleIdAfterReinit (counters are already 0
  before the reinit); WithPrefetcher (skips when no overlap, lead verified :2191-2193; rounds 2+ rewrite the same data).
- WEAK the other 13: UniqueIds x3, MixedTargets, ReuseWrap, DuplicateId, NominalDepth (asserts >= 1 not 16),
  Saturate, MultiHart, Fairness, Soak, RemoteTl1, RemoteGddr. (The reviewer's summary said 7 DEFECT / 14 WEAK; its table has 8 / 13.)
  Working tree: DuplicateId is addressed (see below), so 12 are still WEAK.
- DuplicateId: WEAK. You set the meaning of "duplicate-ID protection" (2026-10-29): several commands may share one
  trid, and the trid reads done only when every one of them has completed. The test issues 16 commands on one trid
  per leg but never asserts the pending count went above 1 (it records it), and the kernel polls the data before it
  reports, so a counter that tracks only the latest command, or a done before all 16 land, still passes.
  Addressed in the working tree (uncommitted, not yet run): both legs assert a pending high water above 1
  (test_mimir_trid.cpp:1362, :1371), and both set kArgCheckAtAck (:1356, :1367). With it the kernel compares every
  command's destination with its source once, right after every trid reads 0 and before the data poll, and a
  mismatch sets kCodeNotLandedAtAck (cce_idma_copy.cpp:366-380). run_scenario_batch expects kCodeLanded
  (test_mimir_trid.cpp:965-972), so that code fails the test. Destinations are poisoned on every run
  (TridScenario::allocate, :814-827), so a stale copy cannot satisfy the check. Both asserts can fail: a counter that
  tracks only the latest command keeps the high water at 1, and a done before all 16 land leaves poison at the check.
  Both legs are kernel-verified (Gddr0 and SiblingTl1 are not remote targets, :351), as kArgCheckAtAck requires
  (cce_idma_copy.cpp:814). Would be OK. Two limits: a done that precedes the data by less than the check's own
  latency still passes; and the high-water assert could fail on a correct implementation if each command completed
  before the next issue's pending read (the MM-4 helper logs a high water of 0 for that reason,
  test_mimir_local_dram.cpp:377-378). Only a run shows which.

### Debug tools (reviewer debug-tools)
- WEAK TestWatcherRingBufferCceMultiWriter (presence only, no forced overlap), CceTestWatcherSanitizeL1Overflow
  (one far address; the dual-view bounds are untested; its wait loop has no timeout, so a broken sanitizer hangs
  instead of failing), TestWatcherWaypointsCce (hart 0 only).
- OK WatcherAssertTests Drisc0-7, WatcherRingBuffer Drisc0-7, DevicePrintDram cases, the Blackhole prefetcher gate test.

## Mapping findings (reported as asked)
- Fit more than one section: the 12 MM8 ports (assigned to MM-8), the 4 Group B tests (split by instantiation;
  in the working tree the MM-4 instances are separate iDMA classes, so only the MM-6 instances keep the old classes),
  CopiesEveryMimirBlock (MM-4 and MM-6), PrefetchesLocalGddrToRemoteCceSram (MM-4 and MM-5; uses no prefetcher API),
  MimirMm4xCompile.Kernel (MM-3, 4.1, 4.2, 4.3).
- Named for one section, code exercises another: MimirCacheProbe.EightHartLaunch (MM41 name, MM-3 code).
- Fit no section: the 7 debug-tool rows.

## Confidence
Nothing was built or run. "Can fail" means "has a reachable assert on an independent oracle", not "was seen to fail".
- High: claims I checked in source myself, beyond the reviewer's reading:
  - the route record is logged only (mimir_access.hpp:759-788);
  - the setup-mismatch skip (test_mimir_tl1.cpp:129-142 at HEAD; replaced in the working tree, see finding 3);
  - the NoOpLaunchCompletes enables leg (test_mimir_fw_load.cpp:334-339, drisc.cc:161-177);
  - the 32-bit pattern truncation (mimir_patterns.hpp:53,63);
  - the seed mask (test_mimir_dm_cache.cpp:1608);
  - SharedL2 expected = observed (:1385);
  - the DirtyLineRace config gate (:2091, :2178);
  - the Ordering fence (risc_common.h:206-212);
  - ReadAfterWrite accepting a data timeout (test_mimir_trid.cpp:1546-1552);
  - the WithPrefetcher skip (:2191-2193);
  - pending == 0 treated as acked (cce_idma_copy.cpp:305-316);
  - the MIM-628 scope (read in Jira);
  - the UMD gitlink;
  - the 1-byte globals write;
  - the DiscoverAccessMatrix asserts;
  - the constant patterns in the MM-6 read legs;
  - the MM-7 stage-ring cap (tensor_prefetcher_manager.cpp:219-221).
- Medium-high: the other DEFECT/WEAK calls rest on one reviewer's reading with file:line evidence, without a second
  check. MM-1, MM-7 (except the cap), MM-8 and the debug-tool verdicts had no lead spot-check at all. Reviewer
  counts were wrong in five summaries (mm1, mm2, s41, s42, s43); every count here was taken from their tables.
- Medium: verdicts that depend on hardware facts nobody could settle from source: same-CCE coherence,
  FLUSH64 probing L1, DEBUG_LAST latching, and the value an unmapped load returns. These are marked conditional.
  State persistence is settled from source with high confidence; the exact TL1 ranges rewritten at each open are
  medium (see State persistence). The iDMA snoop-tag scope is settled from RTL and spec with high confidence for
  own-CCE copies; why the sibling GDDR preload copy is not snooped is low (see iDMA snoop-tag scope).
- Low: whether the 4.2/4.3 contention tests ever see a race on the emulator; only run logs can show that, and the
  MM4x logs the README cites are not on this host.
- Working tree (2026-10-29 refresh): the "addressed" marks rest on reading the uncommitted code, not on a run. High
  that the new asserts are reachable and the located patterns differ per Mimir and view (both read in source).
  Unknown whether the iDMA MM-4 tests, the new kernels (cce_idma_gddr_round_trip.cpp,
  cce_idma_gddr_write_one_uint32.cpp) and DuplicateId's high-water assert pass on a correct implementation; none has
  been built or run.

## Open questions (verdicts above are conditional on these where noted)
Answered "don't know" by you: skip vs fail on setup mismatch (since decided: fail; done in the working tree, see
finding 3); state persistence across processes (since settled from source, see State persistence); MM-4 path (since
decided: iDMA, with no load/store control kept; done in the working tree, see MM-4);
iDMA snoop-tag scope (since settled from RTL and spec, see iDMA snoop-tag scope); duplicate-ID meaning (since answered: several commands per trid, done only when all complete); binary readback vs execution signature (since answered: for MM-3 the
execution signature is enough, so no MM-3 verdict changes).
Hardware facts the reviewers could not settle from source:
1. Do same-CCE DM harts keep their L1 D$ coherent through L2? (4.1 ScopedLock and SharedL2 verdicts)
2. Does FLUSH64 probe the issuing hart's L1 D$? Only source: a comment at risc_common.h:187. (4.1 FullInvalidate,
   FlushFenceOrdering)
3. Do the D2D ATT DEBUG_LAST_* registers latch CCE-originated SPA traffic? (MM-5 SameChipletControl; whether route
   evidence can be gated)
4. What does a CCE load from an unmapped or misrouted GDDR SPA return on MMK: zeros, an error, or a hang? (MM-6
   all-zeros/all-ones hole)
5. Are the D2D DEBUG_MISC and CCE AXI error bits sticky? (MM-6 error-bit leg)
6. Does TT_METAL_FORCE_REINIT reset the CCE RoCC command buffers? (4.3 NoStaleIdAfterReinit)
7. Is CCE DRAM init on the emulator reliably under the fixed 10 s wait? (MM-2 SkipLoadNeverDone)
8. Which document fixes the addrgen loop-wrap rule independently of emulator output? (MM-8 Im2col Dilation1)

## Appendix A: test-to-section map

> Note (2026-10-29 refresh): the content below is as written at HEAD 16585a9c68a, and so are its file:line citations.
> Changed in the working tree since then (uncommitted), so line numbers into these files may have moved:
> dram/test_mimir_local_dram.cpp (rewritten for iDMA), context/test_mimir_emu.cpp, dram/test_mimir_trid.cpp,
> dram/test_mimir_prefetcher.cpp, dram/test_mimir_tl1.cpp, dram/test_mimir_host_tl1.cpp, dram/test_mimir_host_dram.cpp,
> dram/test_mimir_remote_dram.cpp and dram/test_mimir_dm_idma.cpp. Also: MimirPrefetcher.MM7T5TraceCaptureReplay no
> longer exists, and the MimirEmuSameMimirDirect instances of the Group B tests now instantiate the iDMA classes
> IdmaCopiesGddrWithCompileTimeArgs, IdmaCopiesGddrWithRuntimeArgs, IdmaCopiesGddrLargerThanAlignment and
> AllHartsIdmaWriteGddr. The main body has the current lines.

### Mimir test-to-section map (lead's map, shared with every teammate)

Branch: sagarwal/mimir_dram_tests at HEAD 16585a9c68a. Base: 4a9e61f4770 (merge-base with origin/main).
Plan: the Glean plan (https://app.glean.com/library/9568f93acfc347d6b0f6c2a6e079507f).
Mapping rule: each test goes to the section whose test-case text describes what the test's code checks.
Test names were used only to break ties. Paths below are relative to tests/tt_metal/tt_metal/.
"Plan case" is the lead's best match; a teammate may say the test matches a different case of the same section.

#### MM-0 Host read/write to DRAM (7)
| Test | Location | Plan case |
|---|---|---|
| MimirHostDram.DiscoverAccessMatrix | dram/test_mimir_host_dram.cpp:185 | MM-0-T1 |
| MimirHostDramView.PatternReadback | dram/test_mimir_host_dram.cpp:260 | MM-0-T2 |
| MimirHostDramView.BoundaryAndAlias | dram/test_mimir_host_dram.cpp:332 | MM-0-T3 |
| MimirHostDram.ReconnectPersists | dram/test_mimir_host_dram.cpp:429 | MM-0-T4 |
| MimirHostDram.HostSideRejections | dram/test_mimir_host_dram.cpp:510 | MM-0-T5 |
| AllocatedDramBufferHostLoopback.RoundTrip | context/test_mimir_emu.cpp:332 | MM-0-T2 (host DRAM via MeshBuffer + DRAM channel API) |
| DramChannelsDoNotAliasThroughPublicDeviceApi.RoundTrip | context/test_mimir_emu.cpp:849 | MM-0-T3 (aliasing via public API) |

#### MM-1 Host read/write to CCE TL1 (8)
| Test | Location | Plan case |
|---|---|---|
| MimirHostTl1.DiscoverCceTl1Map | dram/test_mimir_host_tl1.cpp:261 | MM-1-T1 |
| MimirHostTl1Cce.PerCceVisibility | dram/test_mimir_host_tl1.cpp:350 | MM-1-T2 |
| MimirHostTl1.BoundaryAndIsolation | dram/test_mimir_host_tl1.cpp:426 | MM-1-T3 |
| MimirHostTl1.HostSideRejections | dram/test_mimir_host_tl1.cpp:477 | MM-1-T5 |
| MimirFwPreflight.FirmwareLoadPreflight | dram/test_mimir_host_tl1.cpp:552 | MM-1-T4 |
| CceSramRoundTripThroughMetalCluster.RoundTrip | context/test_mimir_backend.cpp:32 | MM-1-T2 |
| MimirEmu.CceSramChannelsDoNotAliasThroughMetalCluster | context/test_mimir_backend.cpp:61 | MM-1-T3 |
| CceSramRoundTripThroughUnitMesh.RoundTrip | context/test_mimir_emu.cpp:57 | MM-1-T2 (through an opened unit mesh) |

#### MM-2 Runtime firmware load and launch (7)
| Test | Location | Plan case |
|---|---|---|
| MimirFwLoad.LoadAndReachInitDone | dram/test_mimir_fw_load.cpp:187 | MM-2-T2 |
| MimirFwLoad.ImageManifest | dram/test_mimir_fw_load.cpp:224 | MM-2-T1 |
| MimirFwLoad.AllCcesIndependent | dram/test_mimir_fw_load.cpp:283 | MM-2-T4 |
| MimirFwLaunch.NoOpLaunchCompletes | dram/test_mimir_fw_load.cpp:323 | MM-2-T3 |
| MimirFwReinit.ReloadsOnEveryOpen | dram/test_mimir_fw_load.cpp:343 | MM-2-T5 |
| MimirFwLoadNegative.SkipLoadNeverDone | dram/test_mimir_fw_load.cpp:375 | MM-2-T6 |
| MimirEmu.RuntimeFirmwareInitializesThroughPublicDeviceApi | context/test_mimir_emu.cpp:80 | MM-2-T2 |

#### MM-3 DM kernel load and launch (8)
| Test | Location | Plan case |
|---|---|---|
| MimirKernelLoad.MinimalKernelExecutes | dram/test_mimir_kernel_load.cpp:188 | MM-3-T1 |
| MimirKernelLoad.EachHartAlone | dram/test_mimir_kernel_load.cpp:222 | MM-3-T2 |
| MimirKernelLoad.ArgsReplaceStale | dram/test_mimir_kernel_load.cpp:258 | MM-3-T3 |
| MimirKernelLoad.FirmwareKeepsRunningAcrossReopen | dram/test_mimir_kernel_load.cpp:306 | MM-3-T5 |
| MimirKernelLoad.SameBinaryAllCcesUnicast | dram/test_mimir_kernel_load.cpp:340 | MM-3-T4 |
| HartZeroRunsDramKernel.WritesMagic | context/test_mimir_emu.cpp:92 | MM-3-T1 |
| AllHartsRunDramKernel.WritesMagic | context/test_mimir_emu.cpp:129 | MM-3-T2 |
| MimirCacheProbe.EightHartLaunch | dram/test_mimir_dm_cache.cpp:460 | MM-3-T2 (named MM41, but its code only launches a null kernel on all 8 harts and checks status) |

#### MM-4 CCE DM to local DRAM (6)
| Test | Location | Plan case |
|---|---|---|
| MimirLocalDram.CceWritesTl1ToDram | dram/test_mimir_local_dram.cpp:237 | MM-4-T1 |
| MimirLocalDram.CceReadsDramToTl1 | dram/test_mimir_local_dram.cpp:303 | MM-4-T2 |
| MimirLocalDram.LoopbackChecksIntermediate | dram/test_mimir_local_dram.cpp:371 | MM-4-T3 |
| MimirLocalDram.SizeAlignmentMatrix | dram/test_mimir_local_dram.cpp:467 | MM-4-T4 |
| MimirLocalDram.RepeatedAndOverlappingPerHart | dram/test_mimir_local_dram.cpp:586 | MM-4-T5 |
| MimirEmu.AllCcesCopyLocalMimirBlock | context/test_mimir_emu.cpp:513 | MM-4-T4 |

#### MM-5 CCE DM to local and remote TL1 (10)
| Test | Location | Plan case |
|---|---|---|
| MimirLocalTl1.SameChipletControl | dram/test_mimir_tl1.cpp:598 | MM-5-T1 |
| MimirRemoteTl1Direct.WriteRead | dram/test_mimir_tl1.cpp:614 | MM-5-T2 |
| MimirRemoteTl1Direct.HostInitRemoteRead | dram/test_mimir_tl1.cpp:627 | MM-5-T3 |
| MimirRemoteTl1Direct.SizeAlignment | dram/test_mimir_tl1.cpp:640 | MM-5-T4 |
| MimirRemoteTl1Routed.WriteRead | dram/test_mimir_tl1.cpp:657 | MM-5-T2 |
| MimirRemoteTl1Routed.HostInitRemoteRead | dram/test_mimir_tl1.cpp:670 | MM-5-T3 |
| MimirRemoteTl1Routed.SizeAlignment | dram/test_mimir_tl1.cpp:683 | MM-5-T4 |
| MimirRemoteTl1Bidir.Concurrent | dram/test_mimir_tl1.cpp:701 | MM-5-T4 (concurrent bidirectional) |
| MimirRemoteMixed.FanOutFanIn | dram/test_mimir_tl1.cpp:720 | MM-5-T4 (one-to-many, many-to-one) |
| WritesCrossMimirCceSram.WritesMagic | context/test_mimir_emu.cpp:578 | MM-5-T2 |

#### MM-6 CCE DM to remote DRAM (5)
| Test | Location | Plan case |
|---|---|---|
| MimirRemoteDram.WriteFromTl1 | dram/test_mimir_remote_dram.cpp:1197 | MM-6-T1 |
| MimirRemoteDram.ReadToTl1 | dram/test_mimir_remote_dram.cpp:1226 | MM-6-T2 |
| MimirRemoteDram.Tl1RoundTrip | dram/test_mimir_remote_dram.cpp:1255 | MM-6-T3 |
| MimirRemoteDramTrafficMatrix.TrafficMatrix | dram/test_mimir_remote_dram.cpp:1288 | MM-6-T4 |
| MimirRemoteMixed.Concurrent | dram/test_mimir_remote_dram.cpp:994 | MM-6-T4 (concurrent local DRAM, remote DRAM, remote TL1) |

#### MM-7 Tensor Prefetcher (56)
Every TEST(MimirPrefetcher, MM7*) in dram/test_mimir_prefetcher.cpp (lines 112 to 2625), including
MM7Gate*, MM7GcbLayout*, MM7GcbSmoke*, MM7Lifecycle*, MM7T1*..MM7T5*, MM7Reject*. List them with
`grep -n 'TEST(MimirPrefetcher' tests/tt_metal/tt_metal/dram/test_mimir_prefetcher.cpp`.

#### MM-8 Remaining DM suite (0 so far; see "Ownership pending" group A)

#### 4.1 DM cache (21)
| Test | Location |
|---|---|
| MimirCacheProbe.SnoopConfig | dram/test_mimir_dm_cache.cpp:495 |
| MimirCacheProbe.L2Geometry | dram/test_mimir_dm_cache.cpp:532 |
| MimirCacheOps.FullInvalidate | dram/test_mimir_dm_cache.cpp:921 |
| MimirCacheOps.FlushFenceOrdering | dram/test_mimir_dm_cache.cpp:1012 |
| MimirCacheOps.DirtyEvictionPartialLine | dram/test_mimir_dm_cache.cpp:1113 |
| MimirCacheOps.L1DcacheFlushLine | dram/test_mimir_dm_cache.cpp:1199 |
| MimirCacheOps.L1DcacheFlushFull | dram/test_mimir_dm_cache.cpp:1203 |
| MimirCacheSharedL2.SameLineSameOffset | dram/test_mimir_dm_cache.cpp:1400 |
| MimirCacheSharedL2.SameLineDiffOffset | dram/test_mimir_dm_cache.cpp:1402 |
| MimirCacheSharedL2.AdjacentLine | dram/test_mimir_dm_cache.cpp:1404 |
| MimirCacheWriterProbe.ExternalWriters | dram/test_mimir_dm_cache.cpp:1593 |
| MimirCacheCoherence.CrossCce | dram/test_mimir_dm_cache.cpp:1864 |
| MimirCacheCoherence.CrossCceSnoopOn | dram/test_mimir_dm_cache.cpp:1882 |
| MimirCacheCoherence.CrossD2d | dram/test_mimir_dm_cache.cpp:1957 |
| MimirCacheCoherenceRace.DirtyLineRace | dram/test_mimir_dm_cache.cpp:2042 |
| MimirCacheStress.ReuseThrash | dram/test_mimir_dm_cache.cpp:2213 |
| MimirCacheStress.PreloadChurn | dram/test_mimir_dm_cache.cpp:2427 |
| MimirCacheStress.PerfSweep | dram/test_mimir_dm_cache.cpp:2443 |
| MimirCacheScopedLock.MultiConsumer | dram/test_mimir_dm_scoped_lock.cpp:506 |
| MimirCacheScopedLock.RingWrap | dram/test_mimir_dm_scoped_lock.cpp:605 |
| MimirCacheScopedLock.NonSnoopingProducer | dram/test_mimir_dm_scoped_lock.cpp:682 |

#### 4.2 Atomics (22)
| Test | Location |
|---|---|
| MimirAtomicProbe.TwoHartAmo | dram/test_mimir_atomics.cpp:699 |
| MimirAtomicLocal.OpMatrix | dram/test_mimir_atomics.cpp:951 |
| MimirAtomicLocal.Hotspot | dram/test_mimir_atomics.cpp:1061 |
| MimirAtomicLocal.Distributed | dram/test_mimir_atomics.cpp:1093 |
| MimirAtomicLocal.LrScCas | dram/test_mimir_atomics.cpp:1138 |
| MimirAtomicLocal.Wrap | dram/test_mimir_atomics.cpp:1167 |
| MimirAtomicLocal.Ordering | dram/test_mimir_atomics.cpp:1193 |
| MimirAtomicLocal.CleanDirtyLine | dram/test_mimir_atomics.cpp:1258 |
| MimirAtomicLocal.CleanDirtyLineSnoopedWrite | dram/test_mimir_atomics.cpp:1351 |
| MimirAtomicLocal.Storm | dram/test_mimir_atomics.cpp:1634 |
| MimirAtomicLocal.NeighbourWriteRace | dram/test_mimir_atomics.cpp:1650 |
| MimirAtomicLocal.Soak | dram/test_mimir_atomics.cpp:1703 |
| MimirAtomicItnControl.Cce0ToCce1PlainWrite | dram/test_mimir_atomics.cpp:1725 |
| MimirAccumProbe.FmtOpMatrix | dram/test_mimir_atomics.cpp:1846 |
| MimirAccumProbe.DirtyLine | dram/test_mimir_atomics.cpp:1938 |
| MimirAtomicIdmaAccum.SingleHart | dram/test_mimir_atomics.cpp:2007 |
| MimirAtomicIdmaAccum.AllHartsShared | dram/test_mimir_atomics.cpp:2038 |
| MimirAtomicIdmaAccum.AllHartsDisjoint | dram/test_mimir_atomics.cpp:2084 |
| MimirAtomicIdmaAccum.BackToBack | dram/test_mimir_atomics.cpp:2135 |
| MimirAtomicIdmaAccum.MaxOutstanding | dram/test_mimir_atomics.cpp:2173 |
| MimirAtomicIdmaAccum.MixedPathControl | dram/test_mimir_atomics.cpp:2213 |
| MimirKnownIssue.Mim560Detector | dram/test_mimir_atomics.cpp:2270 |

#### 4.3 Transaction IDs (21)
| Test | Location |
|---|---|
| MimirTridProbe.PendingVisible | dram/test_mimir_trid.cpp:1044 |
| MimirTridProbe.DepthBeyond16 | dram/test_mimir_trid.cpp:1064 |
| MimirTridProbe.Width | dram/test_mimir_trid.cpp:1086 |
| MimirTridProbe.Scope | dram/test_mimir_trid.cpp:1122 |
| MimirTridIdma.UniqueIdsOwnTl1 | dram/test_mimir_trid.cpp:1244 |
| MimirTridIdma.UniqueIdsSiblingTl1 | dram/test_mimir_trid.cpp:1256 |
| MimirTridIdma.UniqueIdsLocalGddr | dram/test_mimir_trid.cpp:1269 |
| MimirTridIdma.MixedTargets | dram/test_mimir_trid.cpp:1288 |
| MimirTridIdma.ReuseWrap | dram/test_mimir_trid.cpp:1322 |
| MimirTridIdma.DuplicateId | dram/test_mimir_trid.cpp:1351 |
| MimirTridIdma.NominalDepth | dram/test_mimir_trid.cpp:1381 |
| MimirTridIdma.Saturate | dram/test_mimir_trid.cpp:1436 |
| MimirTridIdma.BarrierThenDependentReadAfterWrite | dram/test_mimir_trid.cpp:1509 |
| MimirTridIdma.BarrierThenDependentWriteAfterRead | dram/test_mimir_trid.cpp:1558 |
| MimirTridIdma.MultiHart | dram/test_mimir_trid.cpp:1608 |
| MimirTridIdma.Fairness | dram/test_mimir_trid.cpp:1657 |
| MimirTridIdma.Soak | dram/test_mimir_trid.cpp:1752 |
| MimirTridRemote.RemoteTl1 | dram/test_mimir_trid.cpp:1850 |
| MimirTridRemote.RemoteGddr | dram/test_mimir_trid.cpp:1852 |
| MimirTridReset.NoStaleIdAfterReinit | dram/test_mimir_trid.cpp:1875 |
| MimirTridConcurrent.WithPrefetcher | dram/test_mimir_trid.cpp:2049 |

#### Ownership decided by the user (2026-10-28)
These tests fit more than one section or none. They are still reported as mapping findings.
- Group A: owned by the MM-8 teammate.
- Group B: MimirEmuSameMimirDirect/* instantiations owned by MM-4; MimirEmuCrossMimirNoc/* owned by MM-6.
- Group C: CopiesEveryMimirBlock to MM-6; PrefetchesLocalGddrToRemoteCceSram to MM-5;
  MimirMm4xCompile.Kernel to MM-3 (MimirCacheProbe.EightHartLaunch is already in MM-3 above).
- Group D: owned by a separate "debug-tools" teammate.
- Rule for every teammate: a test, or a leg of a test, that cannot fail by design (records only,
  non-gating, "inconclusive", a skip when the targeted behaviour does not occur, a bare GTEST_SKIP)
  is a DEFECT, even when it is labelled a probe.

##### Group A: MM8-instantiated ports of upstream DM tests (fit MM-8 and a target stage)
The plan's MM-8 section owns ports of the existing DM suite (Tier A/B), and its Tier B table or Section 4.1
also names a target stage for each source.
| Test | Location | Upstream source | Other section it fits |
|---|---|---|---|
| MimirDmAddrgen.Pattern | dram/test_mimir_dm_addrgen.cpp:178 | quasar_addrgen | MM-4/5/6 (Tier B target) |
| MimirDmIm2col.Pattern | dram/test_mimir_dm_addrgen.cpp:186 | quasar_im2col | MM-4/6 (Tier B target) |
| MimirDmCache.Op | dram/test_mimir_dm_cache.cpp:258 | quasar_cache | 4.1 |
| MimirDmCachePerf.WritePath | dram/test_mimir_dm_cache.cpp:349 | quasar_cache_perf | 4.1 |
| MimirDmScopedLockClm.CacheOps | dram/test_mimir_dm_scoped_lock.cpp:212 | dataflow_buffer_scoped_lock_cache | 4.1 |
| MimirDmDramUnary.ReaderWriterHandshake | dram/test_mimir_dm_dram_unary.cpp:249 | dram_unary (Tier A) | MM-4 |
| MimirDmIdma.Copy | dram/test_mimir_dm_idma.cpp:553 | quasar_idma | MM-4/5/6 (Tier B target) |
| MimirDmMeshBufferDram.InterleavedPlacement | dram/test_mimir_dm_mesh_buffer.cpp:112 | test_quasar_mesh_buffers | MM-0 (host DRAM) |
| MimirDmMeshBufferDram.WriteRead | dram/test_mimir_dm_mesh_buffer.cpp:161 | test_quasar_mesh_buffers | MM-0 |
| MimirDmMeshBufferDram.MultipleWriteReadRounds | dram/test_mimir_dm_mesh_buffer.cpp:184 | test_quasar_mesh_buffers | MM-0 |
| MimirDmRuntimeArgs.IllegalTooManyRuntimeArgs | dram/test_mimir_kernel_load.cpp:397 | api/test_runtime_args | MM-3 (Tier B target) |
| MimirDmRuntimeArgs.IllegallyModifyRuntimeArgs | dram/test_mimir_kernel_load.cpp:459 | api/test_runtime_args | MM-3 (Tier B target) |

##### Group B: contract-suite tests whose instantiations take different paths (context/test_mimir_emu.cpp)
Each body branches on is_cross_mimir_case() (suite name contains "CrossMimir").
Proposed split: MimirEmuSameMimirDirect/* to MM-4, MimirEmuCrossMimirNoc/* to MM-6.
| Test | Location |
|---|---|
| CopiesGddrWithCompileTimeArgs.RoundTrip | context/test_mimir_emu.cpp:175 |
| CopiesGddrWithRuntimeArgs.RoundTrip | context/test_mimir_emu.cpp:255 |
| CopiesGddrLargerThanAlignment.RoundTrip | context/test_mimir_emu.cpp:436 |
| AllHartsWriteGddr.WritesMagic | context/test_mimir_emu.cpp:718 |

##### Group C: single tests whose code spans sections
| Test | Location | What the code does | Sections it fits |
|---|---|---|---|
| CopiesEveryMimirBlock.RoundTrip | context/test_mimir_emu.cpp:382 | One CCE copies GDDR to GDDR on every Mimir, its own and the other | MM-4 and MM-6 |
| PrefetchesLocalGddrToRemoteCceSram.RoundTrip | context/test_mimir_emu.cpp:635 | Local GDDR read to staging TL1, then writes to two CCEs on the other Mimir | MM-4 and MM-5; name suggests MM-7 |
| MimirMm4xCompile.Kernel | dram/test_mimir_dm_cache.cpp:2509 | Compile-only of the cache, AMO, iDMA and scoped-lock kernels for 8 harts | MM-3 (build), 4.1, 4.2, 4.3 |

##### Group D: Mimir/CCE cases in upstream debug-tool and API tests (fit no section)
| Test | Location |
|---|---|
| WatcherAssertTests params Drisc0..Drisc7 | debug_tools/watcher/test_assert.cpp:484 |
| WatcherRingBuffer params Drisc0..Drisc7 | debug_tools/watcher/test_ringbuf.cpp |
| MeshWatcherFixture.TestWatcherRingBufferCceMultiWriter | debug_tools/watcher/test_ringbuf.cpp |
| MeshWatcherFixture.CceTestWatcherSanitizeL1Overflow | debug_tools/watcher/test_sanitize.cpp |
| MeshWatcherFixture.TestWatcherWaypointsCce | debug_tools/watcher/test_waypoint.cpp |
| DevicePrintDramFixture.DramPrint* (4 cases, now not Blackhole-only) and PrintConcurrentDriscCores | debug_tools/device_print/test_dram_print_output.cpp |
| DramSenderGCBFixture.TensorPrefetcherGateKeepsBlackholeResultAndRejectsQuasarConfig | api/test_dram_sender_global_cb.cpp:650 (runs on Blackhole; relates to MM-7's capability gate) |

## Appendix B: reviewer reports

> Note (2026-10-29 refresh): the reports below are as written at HEAD 16585a9c68a, and so are their file:line
> citations. Changed in the working tree since then (uncommitted), so line numbers into these files may have moved: dram/test_mimir_local_dram.cpp (rewritten for iDMA; the MM-4 report describes the old hart
> load/store tests), context/test_mimir_emu.cpp, dram/test_mimir_trid.cpp, dram/test_mimir_prefetcher.cpp,
> dram/test_mimir_tl1.cpp, dram/test_mimir_host_tl1.cpp, dram/test_mimir_host_dram.cpp, dram/test_mimir_remote_dram.cpp,
> dram/test_mimir_dm_idma.cpp, dram/mimir_hw_state.cpp, dram/README.md, test_kernels/cce/cce_idma_copy.cpp and
> test_kernels/cce/cce_idma_copy_abi.h. UMD citations are unaffected: the UMD checkout is still f0d497f.

Each report is as its reviewer wrote it. Where the two differ, the main report above holds the final verdict:
- The lead changed these verdicts: MM-6 route-evidence leg WEAK to DEFECT (to match MM-5); MM-2 NoOpLaunchCompletes
  enables leg to DEFECT; MM-0 HostSideRejections to DEFECT (zero-fill counts as corruption, per the user);
  MM-2 FirmwareKeepsRunningAcrossReopen to OK (launch-after-reopen only, per the user); the stale-data rows into the
  STALE class (since resolved to DEFECT or WEAK, see State persistence); CceReadsDramToTl1, LoopbackChecksIntermediate and AllHartsRunDramKernel to DEFECT under the stated rule.
- Some summary lines miscount their own tables (mm1, mm2, s43-trid: the table has 8 DEFECT and 13 WEAK). The tables
  are right.
- The UNSURE questions in each report were put to the user; the answers are applied in the main report.
- File:line citations are at HEAD 16585a9c68a. MM7T5TraceCaptureReplay (test_mimir_prefetcher.cpp:2355) has since
  been deleted.

### B.1 MM-0 (reviewer mm0)

#### MM-0 review (host read/write to DRAM): 1 DEFECT (DiscoverAccessMatrix), 1 OK (PatternReadback), 5 WEAK, 0 UNSURE rows

Provenance:
- tt-metal HEAD 16585a9c68a.
- UMD read at the checked-out working tree f0d497f3. HEAD's gitlink is 292fe208, which does not contain grendel_noc_address_resolver.cpp or the emu_tt_device.cpp code these tests go through (the diff from 292fe208 to f0d497f3 is pure insertion; the submodule shows "M"). The committed branch does not include the UMD the tests exercise.
- Chippy read at /proj_sw/user_dev/sagarwal/chippy c9a8a6085.
- Descriptors are umd/tests/soc_descs/mimir_{1x1,2x_package}.yaml: view_size is 512 MiB on both, view offsets are 0 and 4 GiB per Mimir, dram_bank_size (window) is 8 GiB. 1x1 has 2 views; 2x has 4.

All Cluster DRAM paths are write_dram_vec/read_dram_vec (tt_cluster.cpp:907-939): preferred core plus address_offset, into GrendelNocAddressResolver (umd grendel_noc_address_resolver.cpp:261-297), into MimirChippyMemoryMap (mimir_chippy_memory.cpp:32-69, 136-162), into the chippy transport. There is no host cache (chippy lib/address_translation/address_translation.h:354-429).

Common skip surface for the 5 host tests (both descriptors, cached per process):
- watcher enabled (mimir_hw_state.cpp:426-431)
- mimir_identity_result mismatch: SS_BAR, ATT, GDDR INTLV/CAP and CCE remap vs Chippy (mimir_hw_state.cpp:379-392, 771-774; dram/README.md:73)
- any CCE hart out of reset (mimir_hw_state.cpp:471-489)
- ReconnectPersists and HostSideRejections only: any CCE uncore still held in reset
All of these are GTEST_SKIP, not failures.

Paths: host tests are tests/tt_metal/tt_metal/dram/test_mimir_host_dram.cpp (cited as ":N"); context tests are tests/tt_metal/tt_metal/context/test_mimir_emu.cpp.

##### 1. Table

| Test | Plan case | Feature (a) | Oracle (b) | Pass/fail and skips (c) | Passes on a broken impl? (d) | Vacuous? (e) | Verdict | Conf | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| MimirHostDram.DiscoverAccessMatrix | MM-0-T1 | Discover channels, interleave, windows; classify the address matrix | None. Classification is arithmetic on the descriptor's dram_stride; GDDR registers are only logged | Only EXPECT_GT(legal_rows,0) and EXPECT_GT(negative_rows,0); plus the common skips | Yes. Wrong interleave, wrong capacity or wrong window: values are logged, never compared. "window_end" is always negative and "begin" always legal, by construction | Yes | DEFECT | high | :91-113, :185-258; mimir_hw_state.cpp:291-304, 495-501 |
| MimirHostDramView.PatternReadback (View0..N-1) | MM-0-T2 | Host DRAM byte-exact pattern readback, 4 B to 128 KiB, guards | make_pattern expected vs read_dram_vec; 64 B tagged guards on both sides; A then B (B always differs from A) | EXPECT_TRUE per leg; common skips. The view>=num_views skip is dead | Dropped writes, truncation, overrun, wrong data: caught. A consistent wrong translation (bad address_offset or channel) passes here because write and read share one resolver; BoundaryAndAlias catches some of these. Prng is unique per run (time seed) | No | OK | high | :260-330; gtest googletest/src/gtest-internal-inl.h:112-115, gtest.cc:5815 |
| MimirHostDramView.BoundaryAndAlias (View0..N-1) | MM-0-T3 | Cross-view uniqueness, address-bit alias walk, interleave/GiB/view-end straddles | Tagged words read back through the same view; next view's offset 0 for the view-end straddle | EXPECT_TRUE/ASSERT_TRUE; common skips | Stuck bit or view aliasing: caught. Swapped address bits: NOT caught (a bijection round-trips; the comment at :368 claims otherwise). Channel/interleave selection: never checked. GiB straddle loop: 0 iterations. View-end branch: never taken. 4 GiB tile edge: never straddled. Walk stops at 2^28 | No | WEAK | high | :348-424; mimir_gddr_map.hpp:21-38 |
| MimirHostDram.ReconnectPersists | MM-0-T4 | Data persists across a JTAG reset or reconnect | Prng written by a new Cluster "writer", read by another new Cluster "reader"; DRAM anchors plus CCE TL1 | EXPECT_TRUE on compares; ASSERT on setup; common skips plus uncore | No reset and no real reconnect. emu_skip_init=true, so no INIT; ~EmuTTDevice closes no socket; the MetalContext cluster stays connected. It catches an INIT or clear at Cluster construction and a per-instance translation difference. Clearing behaviour and "bridge usable after reset" are not testable. GiB anchors: 0 iterations | No | WEAK | high | :429-508; tt_cluster.cpp:562-575; emu_tt_device.cpp:172-195; mimir_access.hpp:706-727 |
| MimirHostDram.HostSideRejections | MM-0-T5 | Invalid offset, overflow, past-TL1 and misaligned accesses are rejected; legal memory unchanged | EXPECT_THROW(UmdBaseException or std::invalid_argument); sentinel readbacks; L1-tag leg; 6-byte leg | EXPECT_THROW/EXPECT_TRUE; common skips plus uncore | Host-side checks are gated (a missing check_offset or contains() check fails the test). Every rejection happens in UMD or chippy before any wire traffic, so no device-side fault is tested and the sentinels are near-trivial. The misaligned rejection is chippy emu-only (require_aligned). The L1-tag leg reads back through the same resolver it wrote through. The 6-byte leg pins zero-fill clobbering as "expected" | No | WEAK | high | :510-633; grendel_noc_address_resolver.cpp:41-51, 154-188; mimir_chippy_memory.cpp:32-69; chippy emu_axi_transport.cpp:16-27; transport_interface.h:61-87 |
| AllocatedDramBufferHostLoopback.RoundTrip (Partition0..N-1) | MM-0-T2 (MeshBuffer) | Interleaved MeshBuffer host write/read; page p lands in view p | WriteToBuffer vs ReadFromBuffer; then ReadFromDRAMChannel(p, addr) | EXPECT_EQ x2; the partition>=channels skip is dead | Dropped writes pass on any rerun (constant pattern 0xA1100000\|p<<8\|i, deterministic buffer address, emu GDDR persists), and on Partition1.. after Partition0 in the same run (identical full-buffer input). The channel-page read is the same read_dram_vec(view=bank, buffer address) call ReadFromBuffer made. View aliasing: caught | No | WEAK | high | test_mimir_emu.cpp:332-378, 812-816; host_device_transfer.cpp:108-120, 148-171, 192-229, 380-413; mimir_contract.hpp:39-41 |
| DramChannelsDoNotAliasThroughPublicDeviceApi.RoundTrip (Partition0..N-1) | MM-0-T3 | Two DRAM views do not alias | Distinct words to the other view and then this view; both read back | ASSERT on API returns; EXPECT_EQ x2; skips if fewer than 2 views (never on these descriptors) | Full aliasing of the pair: caught. On 2x the pairs are only 0<->2 and 1<->3 (cross-Mimir, same offset); same-Mimir pairs 0<->1 and 2<->3 are never tested. One word at one offset. Values are constant per partition across runs, so a filtered single-case rerun with dropped writes passes; a full run rotates the values | No | WEAK | med | test_mimir_emu.cpp:849-890 |

##### 2. DEFECT / WEAK details

###### DiscoverAccessMatrix (DEFECT)
- Scenario: GDDR INTLV_FLAGOUTSTATUS0 != TILE or CAP != 2. The test never compares them, only logs (:194-196). identity_result does compare them (mimir_hw_state.cpp:390-391), so the whole of row 1 SKIPS (:189-192) rather than fails. The interleave feature T1 exists to discover is turned into a skip.
- The asserts at :255-256 are true by construction: "begin" is legal, and "window_end" (chiplet offset == dram_stride) is negative (:95, :102, :231).
- No traffic, so the matrix feeds nothing. Points such as view_edge (outside the view) are classified LEGAL and never exercised. The ">= 1 GiB" points (:107-111) have 0 iterations (512 MiB view).

###### PatternReadback (OK)
One minor note. Because the pairs chain cyclically (:68-78), pair i's A equals pair i-1's B at the same address, so the A-leg write is unverified for pairs 1..6. Every pattern is still verified once as some pair's B.

###### BoundaryAndAlias (WEAK)
- Swap address bits 2 and 3 in the resolver. The alias walk at :369-386 still passes, because write and read use the same mapping.
- Channel/bank selection is "confirmed" only by labels; mimir_channel_location is used only for strings (mimir_access.hpp:665).
- :392-394 and :399-421 never run on 1x1 or 2x: 512 MiB views, offsets 0 and 4 GiB, so view_offset+size != next offset.
- The 128 KiB edge is skipped by the comment at :388 on the assumption that the GiB edges cover it. They never run, so no 128 KiB edge is straddled.

###### ReconnectPersists (WEAK)
- "Reconnect" is Cluster writer/reader constructed alongside the live MetalContext cluster (:462-490). No INIT is sent (tt_cluster.cpp:574). EmuTTDevice never closes its socket (emu_tt_device.cpp:187-195).
- Nothing resets, so persistence is the only possible outcome. The plan's "persistence or clearing" and "bridge returns to a usable state" are not exercised.
- It does catch a Cluster constructor that clears GDDR/TL1 or sends INIT. The Prng seed is unique per run, so stale data from earlier runs does not pass.

###### HostSideRejections (WEAK)
- All rejections are host-side:
  - check_offset for 8 GiB, 64 GiB and past-TL1. The past-TL1 one is rejected by the GDDR window check (grendel_noc_address_resolver.cpp:154-157), not by a TL1 bound.
  - contains() for the 8 B overflow (mimir_chippy_memory.cpp:32-38).
  - chippy require_aligned for offset 2 (emu_axi_transport.cpp:16-27). This is an emu-only guard; its own comment says the server accepts misaligned accesses.
- So the sentinels at :567-584 can only change if UMD wraps an offset.
- The exception match is UmdBaseException, the base of every UMD error, so any UMD error passes. It is low risk, since the legal writes before it share the path.
- L1-tag leg (:596-602): write_dram_vec(view0, tag+off) and read_cce_tl1 resolve to the same write_core/read_core on the same core and address (view 0's address_offset is 0). That makes it a round trip; only the GDDR-untouched check at :603-609 is independent.
- :611-632 expects bytes 6-7 to change from 0xFF to 0x00 after a 6-byte write. That is silent corruption, which plan line 137 forbids ("no silent corruption"); it pins chippy _write zero-fill (transport_interface.h:61-87) as the contract.

###### AllocatedDramBufferHostLoopback (WEAK)
- Make WriteToDeviceDRAMChannel a no-op. Rerun after one passing run: both EXPECT_EQs pass from stale GDDR (emu GDDR persists until reset, mimir_contract.hpp:39-41). In one run, Partition1+ pass after Partition0.
- The channel-page check (:373-375) repeats ReadFromBuffer's own call (host_device_transfer.cpp:108-120 and 208-212 use bank_index as the view and buffer.address() as the address), so it adds no independent mapping check. Banks equal views per l1_banking_allocator.cpp:279; that rests on medium-confidence reading.

###### DramChannelsDoNotAliasThroughPublicDeviceApi (WEAK)
- On 2x, view 1 aliasing view 0 (the same-Mimir 4 GiB half) is never paired; that pair is covered only on 1x1. other = (p + n/2) % n at :864.
- One 4 B word at one offset, so a partial alias (wrong location that does not coincide with the partner) is not caught.

###### Firmware point (plan line 104)
The two context tests open a unit mesh, which runs CreateDevice and loads firmware. The firmware does not hide DRAM-path faults: the tests write above the allocator base through the same Cluster path. It does mean:
- They cannot serve as MM-0's no-firmware proof.
- A firmware or CreateDevice failure fails them for non-DRAM reasons.
- They live in unit_tests_context_smoke (context/sources.cmake:4-9), not mimir_tests, so they never share a process with row 1. But a killed run against the same server leaves harts running, and row 1 then skips (README:450).

##### 3. UNSURE / questions for the user
1. Is upper-byte zero-fill on a sub-word host DRAM write (HostSideRejections :611-632) the intended host contract? Or is it chippy client behaviour (transport_interface.h:61-87) that the test should reject or not pin, given plan T5's "no silent corruption"? If it is not the contract, that leg is a DEFECT rather than WEAK.
2. Should a host DRAM access outside the view (view_size 512 MiB) but inside the 8 GiB Mimir window count as legal? The tests treat it as legal: T1 classifies view_edge as LEGAL, and T5 writes a sentinel at view 0 offset 8 GiB-4. If it should be rejected, T5 is missing that negative case.
3. Should an identity, SS_BAR, ATT, GDDR interleave or capacity mismatch skip row 1 or fail it? Today it skips (README:73). The mismatch includes the GDDR interleave state MM-0-T1/T3 are meant to verify, and a swapped M0/M1 transport on 2x shows up only as this skip.
4. Is it acceptable that the branch's committed UMD gitlink (292fe208) lacks the resolver and EmuTTDevice code these tests depend on? I judged against the working tree f0d497f3.

##### 4. Belongs elsewhere
- ReconnectPersists TL1 leg (:479-507): host TL1 persistence, which is MM-1 territory.
- HostSideRejections L1-tag leg (:586-602): DRAM API reaching CCE TL1, an MM-1 address-semantics check.

##### Not checked
- The Grendel JTAG transport's alignment and partial-word behaviour (judged only on EmuAxi).
- The emulator-side GDDR model and the Bring-up Guide (no claim here depends on them).
- Whether MeshDevice close puts harts back in reset.

### B.2 MM-1 (reviewer mm1)

#### MM-1 report (host read/write to CCE TL1)

**Summary.** None of the 8 MM-1 tests is wholly vacuous, and none is OK as written: 2 are DEFECT (DiscoverCceTl1Map and FirmwareLoadPreflight, each because of one leg that only logs) and 6 are WEAK. Two problems run through the whole section:
- **Same-path oracle.** Every check writes and reads back through the same Cluster, UMD resolver and `MimirChippyMemoryMap` chain. If that chain maps CCEs consistently but wrongly (CCE0 and CCE1 swapped, or M0 and M1 swapped on the 2x package), every MM-1 test still passes. The plan's check that host and CCE-DM accesses reach the same physical target (plan line 175) is not done in MM-1.
- **Stale data.** Five tests write fixed values with no poison first. CCE SRAM survives between processes in a session (the bring-up routine does not clear it), so a host write path that silently drops writes passes these tests on any rerun.

**What I judged against.** HEAD 16585a9c68a records UMD at 292fe208. That commit has no `mimir_local_address_windows` or `GrendelAddressWindows` (`git grep` finds nothing). The working tree has UMD at f0d497f3 (`M tt_metal/third_party/umd`), which does have them. So I read UMD at f0d497f3. At the branch's recorded submodule pointer, `mimir_access.hpp` would not build. Chippy read at /proj_sw/user_dev/sagarwal/chippy c9a8a6085.

##### 1. Per-test table

| Test | Plan case | Feature (a) | Oracle (b) | Pass/fail and skips (c) | Passes on a broken impl? (d) | Vacuous? (e) | Verdict | Confidence | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| MimirHostTl1.DiscoverCceTl1Map | MM-1-T1 | TL1 region map and permissions; one-word R/W per region per CCE | Region table from HAL/dev_mem_map.h checked against itself (no overlap, fits in TL1, mailbox fields inside MAILBOX); write and read one guard word | EXPECTs on layout and readback. Skips: no descriptor; full_package; 0 CCEs; identity mismatch; any hart out of reset or uncore held. On both descriptors after bring-up it runs | Permission matrix is the string "host RW (assumed)", logged only. Granularity is a hard-coded log line. A dropped write passes on a rerun (fixed tag, no poison). Swapped CCEs pass. A HAL TL1 size that is too small passes; one too large is caught by UMD's 4 MiB window | No (layout and R/W legs gate) | DEFECT (permission and granularity legs only log) | High: lines read, UMD path traced | test_mimir_host_tl1.cpp:31-32, 280-306, 307, 340-345; mimir_hw_state.cpp:471-489, 771-774 |
| MimirHostTl1Cce.PerCceVisibility (Cce0..1 on 1x1, Cce0..3 on 2x) | MM-1-T2 | Per-CCE R/W at lowest, highest, interior and region-start offsets; sizes 4 B to 32 KiB | Poison 0xBADC0DE5, then write the pattern (Prng is seeded per run), read back and compare all words; 64 B guards above and below | EXPECT compare_words on payload and guards. Same skips | Stale data, dropped writes, overrun or underrun: all caught. Consistent CCE permutation passes. Sub-word sizes are not tested (host granularity is 4 B) | No | WEAK (same-path oracle) | High | test_mimir_host_tl1.cpp:369-423; mimir_patterns.hpp:71-78; gtest.cc:5815 with gtest-internal-inl.h:112-124 (seed comes from the clock) |
| MimirHostTl1.BoundaryAndIsolation | MM-1-T3 | Transfers across region boundaries and 16 B / 64 B boundaries; no aliasing between CCEs | 16 B tagged windows on every region edge plus 0x1008 and 0x1038; all CCEs written before any is read | EXPECT per window. Same skips | Full aliasing between CCEs is caught. Consistent permutation passes. Dropped writes pass when the test is run alone on a session that already ran it (fixed tags, no poison). In a full row-1 run, earlier tests overwrite most windows first | No | WEAK (stale data, same-path oracle) | Medium-high | test_mimir_host_tl1.cpp:442-474 |
| MimirHostTl1.HostSideRejections | MM-1-T5 (part) | Host rejects a word just past TL1, an 8 B access straddling the end, and an unaligned access; no side effects | EXPECT_THROW on the exception type, then sentinels at the last word, at 0x1000, and at word 0 of the other CCEs | 6 EXPECT_THROW and sentinel compares per CCE. Same skips | Past-end is rejected by the 8 GiB GDDR-offset check, not a TL1 bound: the resolver treats an address outside the L1 tag as GDDR. Any UmdBaseException passes. Straddle is rejected by `find_memory`. Unaligned is rejected by chippy `require_aligned` (`invalid_argument`) on both emu and JTAG (both use MimirChippyMemoryMap). A TL1 window that is too large is caught by the throw and the other-CCE sentinel, except on the last CCE. Reset persistence or clearing and protected/reserved ranges are not tested at all | No | WEAK (covers only part of T5; past-end passes for an unrelated reason) | Medium-high: throw origins traced in source, not run | test_mimir_host_tl1.cpp:493-548; grendel_noc_address_resolver.cpp:41-51, 175-189, 269-285; mimir_chippy_memory.cpp:31-37, 147-170; grendel_jtag_tt_device.cpp:64; chippy emu_axi_transport.cpp:20-28, 170, 179 |
| MimirFwPreflight.FirmwareLoadPreflight | MM-1-T4 | Every TL1 address and size the MM-2 loader writes, before firmware loads | Exact relocated ELF spans plus HAL mailbox fields. Poison 0xE5, then a Prng pattern per CCE, readback compared byte-for-byte. Firmware is built through minimal `initialize`, not loaded | ASSERT size > 0 and offset aligned (alignment 4 is hard-coded by the test); EXPECT readback; go-signal EXPECT. Same skips | Reset-vector leg only logs. SPA-routes chunk (written by the runtime at :1670-1675) is missing. The 1 B chunk cannot see that chippy zero-fills the other 3 bytes of the word (the runtime does the same write at :1445-1450). Bank-to-NOC is split into 4 equal parts, not the runtime's real writes. The go-signal EXPECT decodes the same bytes already compared, so it adds nothing | No | DEFECT (reset-vector leg only logs); otherwise WEAK | High | test_mimir_host_tl1.cpp:159-205, 240-258, 597-615; mimir_fw_loader.cpp:54-56; metal_context.cpp ("if (minimal) return"); risc_firmware_initializer.cpp:1430-1456, 1670-1675; llrt.cpp:261-266; chippy transport_interface.h:61-85 |
| CceSramRoundTripThroughMetalCluster.RoundTrip (per CCE) | MM-1-T2 | One-word host round trip per CCE through a directly built `Cluster` | Write 0xC0FFEE01+i at MEM_CCE_L1_NOC_OFFSET+0x800 and read it back | EXPECT_EQ. Skips unless an emu or JTAG server is set (always skips under TT_METAL_SOC_DESC alone); skips if the CCE index is past the descriptor's count | Dropped write passes on a rerun, and the DoNotAlias test writes the same value to the same word. One address only. Permutation passes. Not guarded against running harts (0x800 is inside the mailbox), so firmware could cause a false fail | No | WEAK | High | test_mimir_backend.cpp:23-28, 33-58; mimir_contract.hpp:36; dev_mem_map.h:484-493 |
| MimirEmu.CceSramChannelsDoNotAliasThroughMetalCluster | MM-1-T3 | No aliasing between CCEs at one address | Unique value per CCE; all written before any is read | EXPECT_EQ per CCE; ASSERT num_cces >= 2 (2 on 1x1, 4 on 2x). Same server skip | Full aliasing is caught. Aliasing at other addresses or on some address bits is not covered. Dropped writes pass on a rerun (fixed values). Permutation passes | No | WEAK | High | test_mimir_backend.cpp:61-89 |
| CceSramRoundTripThroughUnitMesh.RoundTrip (per CCE) | MM-1-T2, run with firmware loaded | One-word host round trip through the runtime's logical->virtual mapping and the HAL NOC address | Write 0xC0FFEE03+i at HAL BASE noc addr + 0x800 and read it back, after the unit mesh opens | EXPECT_EQ. ASSERTs on device and arch. Skips: no programmable DRAM, no transport, no slow dispatch on a package without Tensix cores, CCE index past the count | Same-path oracle, one word. Stale result: see the note below | No | WEAK | Medium: mailbox field at 0x800 not resolved | test_mimir_emu.cpp:57-78; mimir_contract.hpp:448-462 |

**Unit-mesh test and plan line 104.** Opening the unit mesh loads runtime firmware and the harts run, but the host TL1 path itself (Cluster->UMD) is the same, so it can detect the same translation faults as the backend test.
- **What it adds:** coverage of `IDevice::virtual_core_from_logical_core` plus the HAL `get_dev_noc_addr`, instead of `translated_dram_core` plus a hard-coded MEM_CCE_L1_NOC_OFFSET.
- **What it loses is isolation.** If firmware load fails, it fails for an MM-2 reason, and a pass presupposes MM-2 works. 0x800 also sits in the live mailbox (0x80 to 0x8080).
- **Stale result, under each alternative.** If CreateDevice rewrites that word on every open, stale data cannot hide a dropped write, but a firmware write could cause a false fail. If nothing rewrites it, a dropped write passes on a rerun.

##### 2. DEFECT and WEAK details

- **DiscoverCceTl1Map (DEFECT leg).**
  - The T1 permission matrix is a constant label, "host RW (assumed)", logged at :284-290 and :299-305. The granularity is a literal log line at :307. Neither has an assert.
  - The layout EXPECTs (:281-294, :309-332) compare a table built from dev_mem_map.h with the HAL, which comes from the same header. They catch drift, not hardware behaviour.
  - The guard written at :341 is `make_guard(cce*256+idx)`, the same value every run, with no poison first. So the read at :343-345 passes on a rerun even if the write is dropped.
- **PerCceVisibility (WEAK).** Write and readback share the whole chain. The TT_FATAL at mimir_access.hpp:190-198 compares two functions of the same UMD library, so it is not an independent check. If `mimir_chippy_memory.cpp:107-112` mapped location 0 to the SRAM base of location 1 and the reverse, every assert would pass.
- **BoundaryAndIsolation (WEAK).** Window contents are fixed tags (:458-459) with no poison. Running `--gtest_filter=*BoundaryAndIsolation` twice on one session, with a write path that drops writes, passes the second time. A consistent permutation also passes.
- **HostSideRejections (WEAK).**
  - The past-end call (:519-522) throws from `check_offset(0x2000400000, 8 GiB)` in the GDDR branch (grendel_noc_address_resolver.cpp:155, 285). That is not a TL1-bounds rejection, and any UmdBaseException satisfies the EXPECT.
  - The reset persistence and clearing leg of T5 and the protected/reserved ranges of T5 are absent.
- **FirmwareLoadPreflight (DEFECT leg plus WEAK).**
  - :611-615 reads the reset vectors and logs them "(logged only)", although the runtime writes one at risc_firmware_initializer.cpp:1453-1456.
  - `CceSpaRoutes` at BANK_TO_NOC_SCRATCH+size (:1670-1675) has no chunk.
  - The 1 B `fw_shared_globals_ready` chunk (:203, written and read 1 byte at :236-237, :256-257) cannot see the zero-fill of the other 3 bytes in chippy `_write`.
  - A better approach: read and compare the whole containing word, with known neighbouring bytes seeded first.
- **CceSramRoundTripThroughMetalCluster and DoNotAlias (WEAK).** Both write the constant 0xC0FFEE01+i to the same address (:52, :81), with no poison and no run-unique value. One address does not cover partial aliasing on some address bits.
- **CceSramRoundTripThroughUnitMesh (WEAK).** Fixed value 0xC0FFEE03+i at :73, one word, same-path oracle. It also runs with firmware live, against plan line 104.
- **Skip paths (a note, not a defect).**
  - `identity_result()` skips every host MM-1 test on any D2D, ATT, SS_BAR or CCE DMRISC remap mismatch (mimir_hw_state.cpp:379-392). The host TL1 path does not cross D2D or ATT, so this skip is too broad. It does not create a false pass.
  - The bring-up routine (chippy `validation/metal_bringup/mimir_bringup_common.py:29-54`) releases the uncore and programs the remap and ATT. So these skips, and MIMIR_HOST_ONLY_GUARD, should not fire after a correct bring-up on either descriptor.

##### 3. UNSURE / questions for the user

1. Which `mailboxes_t` field sits at CCE TL1 offset 0x800, used by all three backend and contract round-trip tests? Does CreateDevice or idle CCE firmware write that word? This decides the unit-mesh stale verdict above. I had no build artifacts to compute the offset.
2. Does a fresh chippy bring-up clear CCE SRAM, or does it persist across processes and bring-ups in one emulator session? If bring-up clears it, the stale-data weakness only applies to reruns within one process or between bring-ups.
3. Should MM-1-T4 include the SPA-routes write and the reset-vector register write as gated legs? Both are MM-2 writes the plan lists ("configuration", "reset-vector").
4. Is the 4 B host access granularity (no byte writes; chippy zero-fills a partial word) the intended MM-1 contract for TL1? The runtime issues a 1-byte write at risc_firmware_initializer.cpp:1445-1450.

##### 4. Tests that belong elsewhere

- CceSramRoundTripThroughUnitMesh.RoundTrip runs after runtime firmware is loaded. Its preconditions are MM-2, so it is an MM-1 check only at the access level.

**Not checked:**
- Exception plumbing in UMD `Cluster::write_to_device` above `GrendelTTDevice` (whether anything there wraps `std::invalid_argument`).
- Whether minimal `MetalContext::initialize` or `WatcherServer` construction touches the device.
- The real `mailboxes_t` offsets.
- Emulator model behaviour for TL1 sub-banks. The guide says 256 x 16 B sub-banks; the 0x1008 and 0x1038 windows do straddle 16 B and 64 B boundaries.

Nothing was run.

Files: /localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/dram/test_mimir_host_tl1.cpp, /localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/context/test_mimir_backend.cpp, /localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/context/test_mimir_emu.cpp, /localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/dram/mimir_access.hpp, /localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/dram/mimir_hw_state.cpp, /localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/dram/mimir_fw_loader.cpp, /localdev/sagarwal/tt-metal/tt_metal/third_party/umd/tt-umd/src/coordinates/grendel_noc_address_resolver.cpp, /localdev/sagarwal/tt-metal/tt_metal/third_party/umd/tt-umd/src/tt_device/mimir_chippy_memory.cpp, /localdev/sagarwal/tt-metal/tt_metal/impl/device/firmware/risc_firmware_initializer.cpp

### B.3 MM-2 (reviewer mm2)

#### MM-2 report (mm2)

MM-2 review result: 3 OK, 4 WEAK, 0 DEFECT, 0 UNSURE rows (4 questions). No test is fully vacuous, but one leg of NoOpLaunchCompletes cannot fail in the documented run order.

Read at HEAD 16585a9c68a. UMD read at the checked-out f0d497f33, not the commit HEAD records (292fe20); see Q1. Every case runs on mimir_1x1 (2 CCEs) and mimir_2x_package (4 CCEs). Both are mimir_local and have no Tensix, so each case needs TT_METAL_SLOW_DISPATCH_MODE=1 and the watcher off. The chippy server targets are EmuAxi/GrendelJtag and TT_METAL_SOC_DESC alone gives Silicon, so none of them is mock or emulated and run_launch_phase does launch firmware (risc_firmware_initializer.cpp:299, tt_cluster.hpp:359, rtoptions.cpp:536,543).

##### 1. Rows (map order)

| Test | Plan | Feature (a) | Oracle (b) | Pass/fail and skips (c) | Passes on a broken impl? (d) | Vacuous? (e) | Verdict | Conf | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| MimirFwLoad.LoadAndReachInitDone | MM-2-T2 | Runtime loads CCE firmware into TL1, programs the reset vector, releases the harts, and the firmware reaches init DONE | The CreateDevice wait for DONE; globals_ready==GO (host writes WAIT=0); subordinate_sync==0 (host writes INIT); reset vector of each of 8 harts equals HAL fw_launch_addr_value (read over the SMC register path); core_reset 0xFF and raw 0x1FF; text span readback equals the ELF, after the test zeroes the 24 KiB text window | EXPECT/ASSERT. Skips only on config (package, transport, watcher). Asserts that the uncore is released. Relies on being the first device open in the process, which row 2b provides | Loader skips DRAM, or writes the wrong offset: caught (zeroed text plus DONE wait). Only hart 0 released: caught (subordinate wait hangs, core_reset). Reset-vector write dropped, or UMD hart broadcast broken: NOT caught when an earlier process left the same value, because the register is never seeded. Data/init-scratch spans not compared | No | WEAK (reset-vector leg) | Med: retention of the register across processes is inferred from code comments | test_mimir_fw_load.cpp:134-153,172-183,194-221; risc_firmware_initializer.cpp:1392-1457,1699-1723; drisc.cc:98,112,146; umd tt_device.cpp:93-114 |
| MimirFwLoad.ImageManifest | MM-2-T1 | Image and target manifest from the HAL and the built ELF | Relocated ELF spans and HAL config compared with dev_mem_map constants | EXPECT. No mimir_local skip | Text not at fw_base, text over 24 KiB, or a span outside the regions or straddling the window: caught. A map wrong relative to the hardware: not caught, because the ELF is linked from the same MEM_CCE_* constants and relocated by the same hal.relocate_dev_addr (failure mode 4). Tautological today: hart configs equal, num_fw_binaries==1, launch_value==fw_base. Not checked: reset-vector register address, launch/mailbox layout, alignment | No | WEAK | High | test_mimir_fw_load.cpp:224-281; mimir_fw_loader.cpp:27-42; qa_hal_dram.cpp:102-115; dev_mem_map.h:481,494-508 |
| MimirFwLoad.AllCcesIndependent | MM-2-T4 | No CCE's core_info or mailbox aliases another's, and a launch on one CCE leaves the others alone | core_info xy (written by the runtime per core) against UMD logical-to-translated; whole mailbox of every other CCE before and after a null launch | EXPECT. The cce_count<2 skip never fires on either descriptor | SRAM aliasing, or a wrong logical-to-virtual DRAM map: caught by core_info. A launch broadcast to, or misdirected at, another CCE: caught by the mailbox compare. A dropped launch: caught by WaitProgramDone, because RESET_READ_PTR is written before GO. No distinct per-CCE markers (one image) | No | OK | Med-high | test_mimir_fw_load.cpp:283-319; risc_firmware_initializer.cpp:1651-1666; llrt.cpp:175-219,329-357 |
| MimirFwLaunch.NoOpLaunchCompletes (MM2/Cce0..N-1) | MM-2-T3 | A launch completes: launch data accepted, GO to DONE | WaitProgramDone (GO to DONE) and launch enables != 0 | Gating is the wait only. The cce_index skip is unreachable. Every instantiation behaves the same | The enables!=0 check reads back the host's own write. Firmware never clears enables in HOST mode, and in row 2b AllCcesIndependent has already launched on every CCE with no reinit between opens, so enables is already nonzero before repeat 0. A firmware that skips the kernel (wrong enable-bit test) still reports DONE and passes, because the kernel is null | Leg :339 yes; the wait leg no | WEAK | High | test_mimir_fw_load.cpp:323-341; drisc.cc:162-177; metal_context.cpp:196-204,341,537; cce_null.cpp |
| MimirFwReinit.ReloadsOnEveryOpen | MM-2-T5 | Reset, reinit and relaunch on each open, 3 cycles | Per cycle: expect_cce_initialized; enables==0 before the launch (stale 1 from the previous cycle unless the host re-wrote the launch msg); enables!=0 after | EXPECT. Skips unless TT_METAL_FORCE_REINIT=1 (README row 2d sets it; an unfiltered run reports SKIPPED) | No reinit at all: caught by enables==0 on cycles 1-2. Reinit that reuses the retained image without reloading it: NOT caught. There is no text zeroing or readback between cycles, yet the name and README row 2d claim "reload". GO, subordinate_sync, reset vectors and core_reset alone would also be satisfied by stale state | No | WEAK (name vs coverage) | High | test_mimir_fw_load.cpp:343-373; metal_context.cpp:159-204,341-372; risc_firmware_initializer.cpp:346-379,1413-1450 |
| MimirFwLoadNegative.SkipLoadNeverDone | MM-2-T6 | With TT_METAL_SKIP_LOADING_FW, DRAM init never reports done | The open must throw "failed to initialize DRAM FW"; afterwards go==INIT and harts released (0xFF); the text window was zeroed first | ADD_FAILURE if the open succeeds. Skips unless the env var is set (README row 2c) | Skip flag ignored, DRAM wait removed, or a stale DONE: all caught (the open succeeds, so ADD_FAILURE). Harts not released: caught. An unrelated error gives a different message: test fails. The substring exists only at :1722 (the worker wait says "failed to initialize FW!"). Small gap: the DRAM wait is a fixed 10 s on non-simulator targets. If the skip flag were ignored but init took over 10 s, the test would pass with go still INIT. It does not re-read the text window to show it stayed zero. Nothing tears down on the throw path before the checks | No | OK | Med-high | test_mimir_fw_load.cpp:375-404; risc_firmware_initializer.cpp:140-158,1395,1716-1723; llrt.cpp:329-357 |
| MimirEmu.RuntimeFirmwareInitializesThroughPublicDeviceApi | MM-2-T2 | Firmware init through the public MeshDevice API | ASSERT_NO_THROW(create_unit_mesh) and arch==QUASAR only | Gating is only the runtime's own DRAM DONE wait. Package/transport skip | Passes if the runtime's DRAM wait is weakened or removed. Passes with a broken loader when TL1 still holds an earlier process's firmware (the test does not zero TL1). Passes trivially if it is not the first open in its process (no reinit). Never checks readback, reset vector or alive state | No | WEAK | Med | test_mimir_emu.cpp:80-88; metal_context.cpp:196-204; test_mimir_fw_load.cpp:171 (author's comment on TL1 retention) |

##### 2. WEAK details

**LoadAndReachInitDone.** The plan's "expected reset vector is used" leg reads the 8 reset-vector registers but never seeds them first (test_mimir_fw_load.cpp:145-149). The uncore, which holds those registers, is deliberately not reset at teardown (risc_firmware_initializer.cpp:57-62). So if the runtime's write_reg at :1453-1456 were dropped, or UMD's hart-0 broadcast (tt_device.cpp:106-111) broke, the previous process's identical value would still pass. The same is true of globals_ready and subordinate_sync if the host init writes (:1430-1450) missed: the test zeroes only the text window, not the mailbox. Fix: write a poison value to the vectors and the mailbox fields before the open.

**ImageManifest.** Every expected value comes from the same HAL constants and relocation function the ELF and the runtime use (mimir_fw_loader.cpp:37; qa_hal_dram.cpp:105-110). The plan's T1 items for reset-vector location (fw_launch_addr against UMD CCE_RESET_VECTOR_BASE), launch/config layout and alignment are not checked. build_cce_fw_manifest collects that layout but this test never calls it. The asserts at :275-280 cannot fail with the current qa_hal_dram.cpp:102-115.

**NoOpLaunchCompletes.** `EXPECT_NE(read_launch_enables(...), 0u)` at :339 reads back a host-written field that firmware never clears in DISPATCH_MODE_HOST (drisc.cc:174-177). In the row 2b order, AllCcesIndependent (:308) has already left enables nonzero on every CCE, and mesh close does not tear down MetalContext (metal_context.cpp:196-204; teardown only at :194/:202/:537). So this leg cannot fail. The comment at :338 says it guards against the kernel being skipped, but it does not: a skipped kernel still yields DONE (drisc.cc:162-173). Only the WaitProgramDone transition gates. Fix: a kernel that writes a per-launch marker, with the host seeding the marker first.

**ReloadsOnEveryOpen.** The relaunch is proven: enables==0 at :364 on cycles 1-2, then firmware rewrites GO and subordinate_sync. The reload is not: no zeroing or readback of the text between cycles. A reinit that skipped test_load_write_read_risc_binary but restarted the retained image would pass, against the test name and README row 2d's "reload". Fix: zero the text window after each close and compare it after each open, as LoadAndReachInitDone does.

**RuntimeFirmwareInitializesThroughPublicDeviceApi.** The only gate is the runtime's own DRAM DONE wait (risc_firmware_initializer.cpp:1716-1723), so the test adds no independent evidence. If the image is stale from an earlier process (test_mimir_fw_load.cpp:171), or the case is not the process's first open, it passes with a broken loader. LoadAndReachInitDone already covers the same ground more strongly.

##### 3. UNSURE / questions for the user

1. HEAD records UMD 292fe20, whose grendel_implementation.hpp has no CCE_* constants. mimir_hw_state.cpp:27,47-53 needs CCE_RESET_VECTOR_BASE, CCE_PF_CTRL_RESET_BASE and the others, which exist only in the uncommitted checkout f0d497f33 (42 commits ahead). Is the submodule bump meant to be committed? I judged against f0d497f33.
2. Does the Mimir emulator keep the CCE reset-vector registers and the TL1 mailbox across processes and device closes without a fresh bring-up? If it does, the stale-state gaps in LoadAndReachInitDone and the MimirEmu test are real. If bring-up clears them before every invocation, they shrink.
3. Which binary and run order execute MimirEmu.RuntimeFirmwareInitializesThroughPublicDeviceApi, and is it the first device open of its process? The dram README covers only mimir_tests. I did not verify gtest's registration order from its source.
4. The CCE DRAM-init wait is a fixed 10 s on non-simulator targets (risc_firmware_initializer.cpp:140-158) and ignores TT_METAL_OPERATION_TIMEOUT_SECONDS. Is CCE init on the emulator reliably under 10 s? If it is not, the positive tests fail, and SkipLoadNeverDone could pass for the wrong reason.

##### 4. Section placement

None. MimirEmu.RuntimeFirmwareInitializesThroughPublicDeviceApi is MM-2-T2 as mapped. It lives in the contract suite and duplicates LoadAndReachInitDone with less evidence.

Not checked: README recorded runs for rows 2b-2d (I found none), the chippy/emulator models, the bring-up guide HTML, and data/init-scratch span loading.

### B.4 MM-3 (reviewer mm3)

MM-3 report for the 9 tests (HEAD 16585a9c68a, branch sagarwal/mimir_dram_tests, /localdev/sagarwal/tt-metal).

All five MimirKernelLoad tests gate correctly. HartZeroRunsDramKernel and AllHartsRunDramKernel never clear or seed their result words before launching, so a rerun in the same emulator session can pass on a broken load. EightHartLaunch and MimirMm4xCompile gate, but each covers less than its plan case.

Why the MimirKernelLoad oracle is independent (failure mode 4): the host poisons and reads back through `virtual_core_from_logical_core` -> `Cluster::write_core/read_core` at the HAL BASE NoC address plus offset (`context/mimir_contract.hpp:448-462`). The kernel writes `offset + MEM_L1_UNCACHED_BASE + mhartid*64` from its own hart (`test_kernels/cce/cce_args_echo.cpp:19-36`). Before any launch, `check_logical_core_is_cce` checks Metal's logical->virtual map against UMD's LOGICAL->TRANSLATED translation (`dram/test_mimir_kernel_load.cpp:76-92`). Each slot is compared over all 16 words, and the 5 words the kernel does not write must still read poison, which proves the poison write landed (`:111-117`).

Firmware facts the rows rely on: hart h runs `kernel_text_offset[h]` only when its enable bit is set, and hart 0 posts DONE after all harts finish (`tt_metal/hw/firmware/src/tt-2xx/drisc.cc:146-173`). `is_mock_or_emulated()` covers only Mock and Emule (`tt_metal/llrt/tt_cluster.hpp:358-360`).

##### 1. Rows

| Test | Plan case | Feature (a) | Oracle (b) | Pass/fail and skips (c) | Passes on a broken impl? (d) | Vacuous? (e) | Verdict | Confidence | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| MimirKernelLoad.MinimalKernelExecutes (MM3/.../CceN) | MM-3-T1 | Build, load and launch one CCE DM kernel on hart 0 of each CCE | Echo slot: magic, mhartid, CT marker, 4 RTAs, 4 CRTAs, then poison tail; host readback after poison | `ASSERT_TRUE` on the core map, `EXPECT_TRUE(check_echo_slot)`. Skips: package/transport/slow-dispatch macro only; `cce_index >= num_cces` never fires (same descriptor builds `cce_indices`). Runs on both descriptors | Kernel not run, enables bit wrong, RTA/CRTA not written, wrong hart, wrong CCE: all caught. The null-kernel leg has no assert of its own (it must only not hang). No binary readback, so a skipped binary write over an identical stale binary is not caught (low) | No | OK | High | `test_mimir_kernel_load.cpp:188-220`, `:94-117` |
| MimirKernelLoad.EachHartAlone | MM-3-T2 | Each of the 8 harts launched alone on each CCE, with no cross-target writes | As above, plus every other hart's slot on every CCE must still read poison | `EXPECT_TRUE` per slot per CCE; all CCEs poisoned before each launch. Skips as above | Text offset or RTA indexed by the wrong hart (mhartid echo catches it), launch reaching other harts or CCEs, enables mask ignored: all caught | No | OK | High | `:222-256`, `cce_args_echo.cpp:19-28` |
| MimirKernelLoad.ArgsReplaceStale | MM-3-T3 | RTAs and CRTAs re-delivered on relaunch of the same Program, all 8 harts | Echo of the first args, then of updated args (`seed+0x100`, all distinct), poison before each launch | `EXPECT_TRUE` per hart on both launches | Args written only on first launch, or CRTA updated in place but not re-sent: caught. Arg count is fixed, so a count change is not covered (README:213 says so) | No | OK | High | `:258-300`, `:60-62`, `:166-174` |
| MimirKernelLoad.FirmwareKeepsRunningAcrossReopen | MM-3-T5 (back-to-back half only) | 4 back-to-back launches with changing `runtime[0]`, then mesh close/reopen and one more launch | Echo per launch, poison before each; `runtime[0]` is unique per launch | `EXPECT_TRUE` per launch | Stale args or stale done: caught. Reset recovery (the other half of T5) is not exercised; the comment delegates it to MimirFwReinit. Nothing checks that firmware actually kept running, so the property in the name is not gated (passes the same if reopen reloaded firmware) | No | WEAK | High | `:302-336`; `tt_metal/impl/context/metal_context.cpp:196-205` confirms the early return |
| MimirKernelLoad.SameBinaryAllCcesUnicast | MM-3-T4 | Same compile-time binary on all harts of all CCEs in one Program (unicast) | Per-CCE, per-hart echo with args unique per (CCE, hart) | `EXPECT_TRUE` per slot; `num_cces==0` skip is unreachable on both descriptors | Host-read aliasing between CCEs and per-target arg mixups: caught. The multicast leg is not possible on this branch: `test_load_multicast_write_risc_binary` asserts `is_worker_core` (`tt_metal/llrt/llrt.cpp:283-292`). No binary readback (low) | No | OK | High | `:340-380` |
| HartZeroRunsDramKernel.WritesMagic (MimirEmu/.../CceN) | MM-3-T1 | Hart 0 of each CCE runs a kernel that writes a constant | Host reads TL1 UNRESERVED+0 and compares it with `0xC0FFEE04+cce` | `EXPECT_EQ`; skips are the macro and `cce >= num_cces` (unreachable) | Yes. No poison before launch, a deterministic value, and TL1 is cleared only under `TT_METAL_CLEAR_L1` (unset). Example: enables bit for hart 0 dropped -> firmware posts DONE without running the kernel -> a magic left by an earlier run in the same session passes | No on a fresh session; yes on a rerun | DEFECT (stale data) | High | `context/test_mimir_emu.cpp:92-125`; `tt_metal/impl/device/firmware/risc_firmware_initializer.cpp:238-239` |
| AllHartsRunDramKernel.WritesMagic | MM-3-T2 | 8 kernels, one per hart, in one Program | Host reads 8 words and compares them with `0xC0FFEE10+(cce<<8)+hart` | `EXPECT_EQ(vector)`, `ASSERT_EQ(num_harts,8)`; same skips | Yes. (1) Same stale-data hole: all `kernel_text_offset[h]` pointing at hart 0's binary leaves words 1-7 stale-correct on a rerun. (2) The write address and value are compile-time args with no mhartid, so a swapped text offset for harts 3 and 5 passes even on a fresh session. (3) Harts are not launched independently, and other CCEs are not checked for corruption | No on a fresh session | DEFECT (stale data) + WEAK (no hart identity) | High | `test_mimir_emu.cpp:129-176`; `test_kernels/misc/dram_write_one_uint32.cpp` |
| MimirCacheProbe.EightHartLaunch (MM41/.../CceN) | MM-3-T2 (named MM41) | All 8 harts of a CCE launch and post status | `cce_cache_ops` kModeNull writes code and done magic to a status line poisoned beforehand; idle-CCE check before launch | Deadline -> `FAIL()`; `EXPECT_EQ(code, kCodeDone)` per hart. Skips: macro, watcher on (README says unset), `cce >= num_cces` (unreachable) | A hart that never runs: caught. RTA permutation across harts (status offset is a runtime arg, no mhartid; same binary on every hart): passes. EachHartAlone and ArgsReplaceStale cover that | No | WEAK | Medium (the gap is real, but the test works as an "all harts run" probe) | `dram/test_mimir_dm_cache.cpp:404-490`, `test_kernels/cce/cce_cache_ops.cpp:1189-1300` |
| MimirMm4xCompile.Kernel (MM41/.../<kernel>) | MM-3-T1 (build step only) | Five MM4x kernels JIT-compile for all 8 harts of CCE 0 | `program.impl().compile` does not throw | `EXPECT_NO_THROW`. A build failure throws (`tt_metal/jit_build/build.cpp:149`), and dephash tracking rebuilds after a source edit (`build.cpp:753-758`). Watcher skip only | Broken source, include or link: caught. Nothing is loaded or launched (by design). With a Mock cluster (`TT_METAL_MOCK_CLUSTER_DESC_PATH`, not in the README environment), compile returns without building (`tt_metal/impl/program/program.cpp:3226-3230`) and the test passes vacuously | No under the README environment | OK | High | `test_mimir_dm_cache.cpp:2492-2522` |

##### 2. DEFECT and WEAK details

- **HartZeroRunsDramKernel.WritesMagic (DEFECT):** `test_mimir_emu.cpp:107-124` goes from CreateKernel to LaunchProgram to `read_cce_sram` with no write to the result word first.
  - The value `0xC0FFEE04+cce` is the same on every run. `clear_l1_state` runs only with `TT_METAL_CLEAR_L1` (`risc_firmware_initializer.cpp:238`), and README:21 requires it unset.
  - A second run in the same emulator session therefore passes even if firmware posts DONE without jumping to the kernel (`drisc.cc:162-173`).
  - Fix: poison the word before launch, as MimirKernelLoad does (`test_mimir_kernel_load.cpp:94-97`).
  - If the lead counts "needs a prior run in the session" as WEAK rather than DEFECT, the verdict is WEAK; the scenario is the same.
- **AllHartsRunDramKernel.WritesMagic (DEFECT + WEAK):**
  - Same stale hole at `test_mimir_emu.cpp:150-175`.
  - Separately, `dram_write_one_uint32.cpp` takes its address and value from compile-time args and never reads mhartid. A launch message that swaps the text offsets of two harts (`drisc.cc:127,163`) still fills all 8 words correctly on a fresh session.
  - Only EachHartAlone catches that swap.
- **FirmwareKeepsRunningAcrossReopen (WEAK):**
  - The four launches before the reopen and the one after it are gated (`:324-335`).
  - The name's claim, that firmware is not reset or reloaded by the reopen, is asserted nowhere. The test passes the same either way.
  - The reset-recovery half of MM-3-T5 is not in this test; the comment at `:302-305` delegates it to MimirFwReinit.
- **MimirCacheProbe.EightHartLaunch (WEAK):**
  - Each hart's status line address comes from a runtime arg (`test_mimir_dm_cache.cpp:446-452`), not from mhartid. All 8 kernels share one binary.
  - So a runtime that delivers hart A's runtime args to hart B and B's to A still posts 8 good status lines.
  - It does detect a hart that never launched, which is its stated purpose.

##### 3. UNSURE / questions for the user

1. HartZeroRunsDramKernel and AllHartsRunDramKernel are built into `unit_tests_context` (`context/sources.cmake:8`), not `mimir_tests`, and the README run order does not list them. Which sessions and descriptors run them, and are they rerun in one emulator session without `TT_METAL_CLEAR_L1`? That decides how much the stale-data hole matters in practice.
2. Is "firmware keeps running across reopen" meant to be checked (for example by asserting that a firmware-state word written before close is unchanged after reopen)? Or is the test only meant to show that a launch works after reopen? The verdict for FirmwareKeepsRunningAcrossReopen depends on this.
3. Plan MM-3-T1 and T4 ask for binary-placement readback "where supported". `TT_METAL_KERNEL_READBACK_ENABLE` exists (`llrt.cpp:270-275`), but the README requires it unset, and no MM-3 test reads back a kernel binary. Is the execution signature an accepted substitute for placement readback?

Not checked: whether `Device::close` or `DeviceManager` touch DRAM cores on Quasar at reopen. I relied on the `metal_context.cpp:196-205` early return. I also did not check whether slow dispatch writes the kernel binary to the same TL1 offset on every launch, which decides whether the stale-binary scenario is realistic.

##### 4. Tests that belong to another section

- MimirCacheProbe.EightHartLaunch is instantiated as MM41 (plan 4.1), but it is an MM-3-T2 launch probe.
- MimirMm4xCompile.Kernel is instantiated as MM41, but only the MM-3-T1 build step is exercised; it touches nothing in 4.1-4.3 at runtime.

### B.5 MM-4 (reviewer mm4)

#### MM-4 report (CCE DM read/write to local DRAM)

##### Sources and common facts

- Superproject HEAD 16585a9c68a.
- UMD read at the working-tree checkout f0d497f3, not the recorded pointer 292fe20. The recorded commit has no mimir_1x1.yaml or mimir_2x_package.yaml (`git diff --stat 292fe20 f0d497f3` adds both files).
- Both descriptors: dram_view_size 512 MiB, two views per Mimir at address_offset 0 and 4 GiB, address_space mimir_local (mimir_1x1.yaml:31-50,81; mimir_2x_package.yaml:35-68,98).

Facts every row relies on:

- **No skips fire.**
  - skip_unless_mimir_local_package (mimir_access.hpp:599) passes on both descriptors.
  - topology_skip_reason (test_mimir_local_dram.cpp:41) never fires, because the params come from cce_indices().
  - Every MimirLocalDram/Cce* case runs: 2 on 1x1, 4 on 2x.
- **Only two anchors exist.**
  - With a 512 MiB view, gib_count is 0 in mimir_dram_anchors, so only unres_g0 and view_end exist (mimir_access.hpp:708-725).
  - The matrix test and T5 use only unres_g0 (first_anchor_holding).
  - view_end slots pack upward from the window start (test_mimir_local_dram.cpp:87-92), so they stop about 95 KiB short of the view end. The end-of-region case is never exercised.
- **Oracle independence (mode 4): checked, OK.**
  - Host DRAM: write_dram_vec -> write_core(addr + address_offset) (tt_cluster.cpp:907-939) -> UMD mimir_local windows with base kMimirGddrDramLocalBase, then chippy (grendel_noc_address_resolver.cpp:55-79,285).
  - Kernel: the package SPA built from GrendelAddressWindows{} in the CCE route table (risc_firmware_initializer.cpp:85-108; cce_gddr.h:21-28), then the CCE remapper.
  - The two paths share no code. The only shared assumption is channel == SPA instance, which risc_firmware_initializer.cpp:91-96 enforces.
- **The data path is hart load/store.** The kernels use 32-bit volatile loads and stores (cce_copy_ops.cpp:12-53; cce_test_common.h:19-26), not DMA or NoC.
- **ASSERT_TRUE(run_cce_copy_ops(...)) can never fail.**
  - run_cce_copy_ops always returns AssertionSuccess (mimir_access.hpp:502-567).
  - Gating comes from the done/XOR checks plus the host readbacks.
  - The done word is poisoned before launch (:547-549).
- **Seed behaviour.**
  - mimir_test_seed comes from gtest random_seed, which is time-based when the flag is 0 (.cpmcache/gtest/96129d89.../googletest/src/gtest.cc:5815; gtest-internal-inl.h:112-124).
  - Prng data therefore changes between runs but is fixed within one process.
  - AddressDerived data is identical every run.

##### 1. Table

| Test | Plan case | Feature (a) | Oracle (b) | Pass/fail and skips (c) | Passes on a broken impl? (d) | Vacuous? (e) | Verdict | Conf | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| MimirLocalDram.CceWritesTl1ToDram | MM-4-T1 | CCE harts write TL1 to both local DRAM views, all 8 harts, 2 anchors | Host-generated TL1 source vs host DRAM readback of [guard\|payload\|guard]. Destination poisoned 0xDEADBEEF with unique guards. The XOR only shows the TL1 read happened | EXPECT done, EXPECT XOR, EXPECT compare_words. No skips on 1x1 or 2x | Write misrouted to the other half or Mimir: caught (poison stays). Dropped write: caught. Write overrun: caught by guards. Wrong hart or binary: caught (per-hart sizes and data) | No | OK (minor: end of region not reached) | high | test_mimir_local_dram.cpp:237-300; mimir_access.hpp:708-725 |
| MimirLocalDram.CceReadsDramToTl1 | MM-4-T2 | CCE reads local DRAM into its own TL1 | Host-written DRAM pattern vs host TL1 readback with guards. TL1 poisoned | As T1 | Yes, for read aliasing. (i) The read path drops bit 32 or the view's 4 GiB address_offset: the view-1 read hits view 0, which holds identical data written moments earlier. (ii) On 2x, a Mimir-1 CCE read misrouted to Mimir 0: identical data is already there from the Cce0/Cce1 cases | No | WEAK | high | :317,341-343; mimir_patterns.hpp:53,63 |
| MimirLocalDram.LoopbackChecksIntermediate | MM-4-T3 | DRAM -> TL1 -> other DRAM slot per hart | Host checks of source, intermediate TL1 (guarded) and destination (guarded). Destination and TL1 poisoned | EXPECT done (XOR deliberately unchecked), 3 compare_words. No skips | A fault in the shared cce_gddr_address: caught by the write leg. A read-only alias as in T2: passes the intermediate and final checks, because the source pattern repeats across views and Mimirs. Ordering is checked only through the done word | No | WEAK | high | :407-413,441-459 |
| MimirLocalDram.SizeAlignmentMatrix | MM-4-T4 | TL1 -> DRAM, sizes 4 B to 32 KiB, offsets {0,4,60,124,32K-4}, disjoint and shared destinations | Disjoint: host readback with guards per hart. Shared: each word must equal some hart's word at that index, plus guards | EXPECT done/XOR, compare. ASSERT_NE anchor, ASSERT_GE harts (both satisfied). No skips | Disjoint leg catches dropped or misplaced writes. Shared leg passes if all but one hart's write is lost (inherent). Not covered: DRAM -> TL1 reads, sub-word offsets, other anchors, bursts | No | WEAK (coverage) | high | :467-580,167-186 |
| MimirLocalDram.RepeatedAndOverlappingPerHart | MM-4-T5 | 4-op table per hart: two overlapping DRAM writes, two read-backs, 3 repeats | Host word model in program order vs host readback of the DRAM span and TL1 C/D (all poisoned per repeat) | EXPECT done/XOR, 3 compares. No skips | Reordered overlapping writes: caught. Read-before-write or stale read: caught. Read alias: caught on repeats 0 and 1 by the Prng half. Negative controls (invalid address, region crossing, unordered ops, fault, timeout) are absent from the suite | No | WEAK (plan's negative half missing) | high | :586-692 |
| MimirEmu.AllCcesCopyLocalMimirBlock | MM-4-T4 (map) | All CCEs at once, each copies 64 B within its own location's view through TL1 staging | Constant inputs vs host DRAM readback. Destination zeroed first | EXPECT_EQ per CCE. No skips on either descriptor | Misrouted read or write: caught (unique cce-tagged values, destination zeroed). Staging is not poisoned; harmless for a load/store loop, would matter for an async/DMA read | No | OK | high | test_mimir_emu.cpp:513-574 |
| CopiesGddrWithCompileTimeArgs.RoundTrip (MimirEmuSameMimirDirect) | MM-4-T3 (+MM-3 compile args) | Local GDDR -> TL1 -> GDDR with compile-time args, per CCE x local view | Inputs tagged cce/mimir/location vs host DRAM readback (destination zeroed), plus staged TL1 == inputs.back() | 2 EXPECT_EQ. Remote partitions skip as a param filter: 1x1 runs 4 of 4, 2x runs 8 of 16 | Wrong half or Mimir: caught. Dropped write: caught. num_slots is 1 (mimir_contract.hpp:225-230), so the slot loop runs once | No | OK | high | test_mimir_emu.cpp:175-251,800-804 |
| CopiesGddrWithRuntimeArgs.RoundTrip (SameMimirDirect) | MM-4-T3 (+MM-3 runtime args) | As above with runtime args | Host DRAM readback only, no staging check | EXPECT_EQ, same skip filter | Same as above. Args not delivered: zero words copied, caught. Low-confidence note: args are identical across runs of the same case, so stale args left in TL1 would match | No | OK | high | :255-328,818-822 |
| CopiesGddrLargerThanAlignment.RoundTrip (SameMimirDirect) | MM-4-T4 (size) | 1 KiB (64x16) local GDDR round trip | As above | EXPECT_EQ, same filter | Caught. The value encoding ORs i into the location<<4 bits, so location 0 and 1 data share half their words; the other half still differ | No | OK | high | :436-511,829-833 |
| AllHartsWriteGddr.WritesMagic (SameMimirDirect) | MM-4-T1 (all harts) | Each of 8 harts writes one tagged word to local GDDR | Expected per-hart words vs host readback of 32 B. Destination zeroed | ASSERT_EQ harts == 8, EXPECT_EQ, same filter | Missing hart, wrong binary or misroute: caught | No | OK | high | :718-787,841-842 |

The four Group B bodies do not branch on locality beyond the skip filter. is_cross_mimir_case() is used only in cce_partition_case_applies (mimir_contract.hpp:276-279,326-344). "Direct" vs "Noc" in the suite names matches no code difference: both use the same kernel and SPA route table, and the local route is port 0 (risc_firmware_initializer.cpp:67-69). No test checks which path the transaction took.

##### 2. WEAK details

- **CceReadsDramToTl1**
  - Cause: make_pattern seeds the Prng and builds AddressDerived words from static_cast<uint32_t>(base_address) (mimir_patterns.hpp:53,63). The base is chiplet_offset (test_mimir_local_dram.cpp:341-342), and views 1 and 3 differ from views 0 and 2 only in bit 32. Neither the Mimir nor the CCE index is part of the base, and the seed is fixed within one process. So all four views get identical words at the same view offset.
  - Effect: a read path that loses bit 32 passes on view 1, because it reads view 0's copy written in the same test. On 2x, a Mimir-1 CCE read misrouted to Mimir 0 passes.
  - Fix: fold view, Mimir and CCE into the pattern base or seed, or poison the alias location before the read.
- **LoopbackChecksIntermediate**
  - Same mechanism (:412-413). A read-only alias fills TL1 with identical words, then a correct write produces the expected destination.
  - Only a fault in the shared cce_gddr_address is caught, and only by the write leg.
  - The plan's "completion ordering" is checked only through the done word.
- **SizeAlignmentMatrix**
  - The matrix covers only the write direction, offsets that are multiples of 4 bytes (the kernel is a word loop), and the unres_g0 anchor.
  - Packet and burst boundaries are nominal because the kernel issues no bursts.
  - In the shared-destination leg (:546-576), a hart's dropped write is masked by another hart's word at the same index. That is inherent to concurrent writes, but the leg cannot show that every hart's write landed. The disjoint leg does catch dropped writes.
- **RepeatedAndOverlappingPerHart**
  - The repeat and overlap legs gate correctly.
  - The plan's T5 negative cases (invalid address, legal-region crossing, unordered ops, fault or timeout response) appear nowhere in dram/ or context/; a grep of TEST names finds no MM-4 negative case.

##### 3. UNSURE / questions for the user

1. test_mimir_local_dram.cpp:27-30 says these tests use the cce_gddr.h hart load/store path through the SPA window instead of the NoC dataflow or DMA API the plan names, while the plan's MM-4 purpose says "CCE DMA path". Is MM-4's target the hart load/store path or a NoC/DMA-engine path?
   - If load/store: the rows above stand.
   - If DMA or NoC: all 10 tests exercise a different path (mode 6), T4's packet and burst coverage is nominal, and every row becomes WEAK for coverage.
2. The plan assumes Mimir A -> A traffic uses the local Mimir address, not a D2D translation. The kernel issues the package SPA (dram_base + ch*stride + offset) and relies on the CCE remapper claiming the port-0 window (cce_spa_routes.h:14-17). Is that the accepted "local address" path? No test observes whether the access left the die.
3. Is a full_package descriptor in scope for MM-4? skip_unless_mimir_local_package would skip all five MimirLocalDram tests there. The only Mimir-local use in their bodies is describe_location, which builds failure messages.

##### 4. Tests that belong elsewhere

- CopiesGddrWithCompileTimeArgs and CopiesGddrWithRuntimeArgs also test kernel-arg delivery (MM-3). Judged here as MM-4.

##### Not checked

- Emulator behaviour: no evidence of model gaps was found, so no row is marked "fails on emulator".
- CCE cache state across a device reopen.
- Whether TL1 or GDDR contents persist across emulator processes. TL1 is cleared only with TT_METAL_CLEAR_L1 (risc_firmware_initializer.cpp:238).

### B.6 MM-5 (reviewer mm5)

#### MM-5 report (CCE DM to local and remote TL1), teammate mm5

What I read: tt-metal HEAD 16585a9c68a. UMD is the working-tree checkout f0d497f. HEAD records UMD 292fe20, but `git grep` shows that commit has no get_mimir_channels or mimir_local_address_windows, and the tests and risc_firmware_initializer.cpp:85-107 call both. So HEAD as committed cannot build against its recorded UMD.

Correction to the message I sent earlier: there are 8 remote tests in test_mimir_tl1.cpp, not 9 (3 Direct, 3 Routed, Bidir, FanOutFanIn).

##### Shared findings (cited once, used in every row below)
- S1 Oracle independence: the kernel reaches remote SRAM with cce_sram_address = sram_spa_base + cce*4MiB (tt_metal/hw/inc/experimental/cce_gddr.h:53-55). That goes through the package SPA, then the DMRISC remap or D2D. The host reads back through UMD's MimirLocal flat window and a per-chiplet chippy transport (umd device/src/tt_device/emu_tt_device.cpp:101-107,143-158; mimir_chippy_memory.cpp:94-110). That path does not use the kernel's SPA routes or D2D. So the host readback is an independent oracle.
- S2 Data gating is sound in run_remote_transfers and run_remote_host_init_read (test_mimir_tl1.cpp:375-461, 527-563). Before each op the host seeds the landing slot with make_seed_pattern, which can never equal any test pattern. It writes a guard just past the window, plus a guard at the same offset in every other CCE. After the op it reads the destination back on the host, checks the guards, and gates on new D2D/AXI error bits. A misroute, a missing translation, a double translation or a swapped CCE all leave the seed in place or break a guard, so the test fails. The done-block XOR proves nothing on write legs: the kernel computes it from its own source words (test_kernels/cce/cce_copy_ops.cpp:44-49). The host readback is what gates.
- S3 DEFECT (route-evidence leg, all 8 remote tests): make_sram_record builds a predicted "issued SPA / ATT entry 64+dest / after SS_BAR1" record, hard-codes crosses_d2d=true, and only passes it to log_transfer. Nothing compares it with anything observed (dram/mimir_access.hpp:759-788; test_mimir_tl1.cpp:348-373,397,430,540). So the plan's "translation exactly once / D2D-boundary / return-path" pass criterion (MM-5-T2/T3/T4) cannot fail. read_debug_last exists and SameChipletControl uses it, but no remote test checks that DEBUG_LAST moved to the expected translation.
- S4 Preflight becomes a skip: remote_skip_reason returns mimir_remote_preflight_result() (test_mimir_tl1.cpp:141). That result is the identity check plus SS_BAR/ATT/interleave readback against the chippy values (mimir_hw_state.cpp:395-408). On mismatch, every remote test calls GTEST_SKIP. I found no test in the suite that fails on that mismatch: every caller of it, and of mimir_identity_result, skips (test_mimir_host_dram.cpp:189-191; test_mimir_host_tl1.cpp:268-269, ...). By contrast, mimir_remote_remap_result() is gated with EXPECT_TRUE (test_mimir_tl1.cpp:623 etc.). But its expected values come from the same UMD cce_dmrisc_remap_entries() that programmed the remap (mimir_hw_state.cpp:268-289), so it catches a failed write, not a wrong formula (failure mode 4, minor).
- S5 Watcher: when TT_METAL_WATCHER is set, mimir_watcher_skip_reason turns every test in test_mimir_tl1.cpp into a skip (test_mimir_tl1.cpp:119-121; mimir_hw_state.cpp:426-431). That is failure mode 9. The README says to leave it unset.
- S6 Error-bit leg: a bit already set before the test is only logged as "inconclusive". The code assumes DEBUG_MISC bits are sticky, and nothing in the source shows that (mimir_hw_state.cpp:652,705-715). This leg is secondary.
- S7 Descriptors: both mimir_1x1 and mimir_2x_package are address_space mimir_local, with 2 CCEs per Mimir and 8 harts (dev_mem_map.h:500). Remote tests and the two emu tests run on 2x only and skip on 1x1 for topology reasons. No skip fires on both descriptors.

##### 1. Per-test table

| Test | Plan case | Feature (a) | Oracle (b) | Pass/fail and skips (c) | Passes on a broken impl? (d) | Vacuous? (e) | Verdict | Conf. | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| MimirLocalTl1.SameChipletControl | MM-5-T1 | Sibling-CCE TL1 write and read over SPA, plus an own-TL1 control copy, without using the ATT | Host readback of the seeded sibling slot, the own control copy and the seeded sibling-read slot; guards in the other CCEs; DEBUG_LAST of the source Mimir unchanged; no new error bits | EXPECT on all of these. Skips on watcher (S5). 1x1: Cce0To1, Cce1To0. 2x: 4 instances | Wrong sibling, misrouted SPA, broken read: caught. A local access that goes through the ATT: caught only if DEBUG_LAST latches that traffic. On 1x1 only M0.D2D0 is read, so for Cce1To0 (CCE1 exits through D2D1 per the guide) the no-ATT leg cannot fail | No | OK (no-ATT leg WEAK) | Med | test_mimir_tl1.cpp:234-320; mimir_hw_state.cpp:635-647,717-763 |
| MimirRemoteTl1Direct.WriteRead | MM-5-T2 | M0.CCE1<->M1.CCE0 remote write, then a read back by the next hart | S1/S2 host readback of the destination and of the read-back slot | EXPECT; ASSERT on run_cce_copy_ops (always returns success; a throw becomes ADD_FAILURE). Skips: S4, S5, topology | Misroute or wrong CCE: caught. No D2D crossing or double translation that still lands the data: not caught (S3). The one AllZeros hart's read leg would pass if a misrouted read returned 0, but the other 7 harts still fail the test | No | DEFECT (S3 leg); data OK | High | test_mimir_tl1.cpp:470-484,614-625 |
| MimirRemoteTl1Direct.HostInitRemoteRead | MM-5-T3 | Host fills the remote window; the CCE reads it into its own TL1 | Host readback of the seeded own slot, trailing guard, guards in the other CCEs, XOR | As above | Wrong remote CCE: the 64 B guard at the slot start fails the first 16 words, even when data from earlier tests is still in that CCE. Return-path and endpoint evidence: not checked (S3) | No | DEFECT (S3 leg); data OK | High | test_mimir_tl1.cpp:488-567,627-638 |
| MimirRemoteTl1Direct.SizeAlignment | MM-5-T4 | Sizes 4 B to 4 KiB x offsets {0,4,12,60,124}, 35 combinations in 5 rounds | Write leg only (read_back=false), Prng only | As above; returns on a fatal failure | Write misroute: caught. A broken read at any size or alignment: not exercised. A write before the window start: no front guard in the destination | No | DEFECT (S3) + WEAK (no read leg, no leading guard) | High | test_mimir_tl1.cpp:570-594,640-651 |
| MimirRemoteTl1Routed.WriteRead | MM-5-T2 | Same as Direct for the 6 non-adjacent pairs (README calls them 1-hop/2-hop) | Same as Direct | Same as Direct | Same as Direct. Nothing checks the hop or route taken; only the pair names say "routed" | No | DEFECT (S3); data OK | High | test_mimir_tl1.cpp:114-116,657-668 |
| MimirRemoteTl1Routed.HostInitRemoteRead | MM-5-T3 | Same as Direct | Same as Direct | Same as Direct | Same as Direct | No | DEFECT (S3); data OK | High | test_mimir_tl1.cpp:670-681 |
| MimirRemoteTl1Routed.SizeAlignment | MM-5-T4 | Same as Direct | Same as Direct | Same as Direct | Same as Direct | No | DEFECT (S3) + WEAK | High | test_mimir_tl1.cpp:683-694 |
| MimirRemoteTl1Bidir.Concurrent | MM-5-T4 (concurrent bidir) | Both direct directions, 16 kernels in one Program | Same as WriteRead | Same as WriteRead | Each direction's data: caught. A bug that only appears under concurrency: not shown to be exercised, because nothing records timing or overlap | No | DEFECT (S3) + WEAK (concurrency claimed by name only) | Med | test_mimir_tl1.cpp:701-712 |
| MimirRemoteMixed.FanOutFanIn | MM-5-T4 (1-to-many, many-to-1) | 8 transfers with disjoint landing slots, read back | Same as WriteRead | Same as WriteRead; skips unless location 1 exists | Misroute or aliasing between windows: caught by seeds and guards. Concurrency: as in Bidir | No | DEFECT (S3) + WEAK (concurrency) | High | test_mimir_tl1.cpp:720-744 |
| WritesCrossMimirCceSram.WritesMagic | MM-5-T2 | 4 B SPA write to both CCEs of the other Mimir | Host zeroes both destinations, then reads back magic\|src<<8\|dest | EXPECT_EQ. 2x only (4 instances); no preflight or watcher skip | Swapped, missing, or local-landing write: caught. Read direction, sizes, guards, route evidence: none | No | OK (narrow: write leg only) | High | context/test_mimir_emu.cpp:578-631; test_kernels/misc/cce_write_cross_mimir_sram.cpp |
| PrefetchesLocalGddrToRemoteCceSram.RoundTrip | MM-4 + MM-5-T4 (1-to-2) | Local GDDR read into staging TL1, then a write to both remote CCEs | Host writes a src-specific pattern to GDDR, zeroes the destinations, reads them back | EXPECT_EQ. 2x only | Wrong GDDR slot or base (it would read another src's pattern), remote misroute: caught. The staging TL1 is never seeded, but every executed GDDR read overwrites it, so leftover data matters only if the read never runs | No | OK | High | context/test_mimir_emu.cpp:635-714; test_kernels/misc/cce_prefetch_local_gddr_to_remote_sram.cpp |

##### 2. DEFECT and WEAK details
- S3, all 8 remote tests in test_mimir_tl1.cpp (mimir_access.hpp:759-788; test_mimir_tl1.cpp:348-373,397,430,540): the stage-by-stage address record is a host-side prediction that is only logged. Example: an implementation that reaches M1 SRAM without the expected ATT entry, or through a different route, but still lands the data, passes. The plan's "expected translation exactly once" cannot fail.
- S4 skip path (test_mimir_tl1.cpp:129-142,618-620 and the same pattern in each remote test): wrong SS_BAR/ATT/interleave or identity on mimir_2x_package makes every remote case skip, and no other test fails on it. If chippy/emu_boot D2D programming is part of what MM-5 should catch, this is a DEFECT. If it is an environment precondition, it is acceptable, but the run then reports "skipped", not "failed" (see Q1).
- SizeAlignment, both suites (test_mimir_tl1.cpp:585-589): read_back=false, so the size and alignment matrix never tests the read direction. The destination also has no guard before landing_offset+alignment (test_mimir_tl1.cpp:381-384).
- Bidir.Concurrent and FanOutFanIn (test_mimir_tl1.cpp:701-744): "concurrent" rests on putting all kernels in one slow-dispatch Program. Nothing asserts overlap, unlike the README's row 124, which asserts it.
- SameChipletControl no-ATT leg (test_mimir_tl1.cpp:266-295): DEBUG_LAST offsets 0x8-0x2C match chippy's Mimir D2D tile register map (chippy lib/register_map/grendel/mimir/tt_mimir_d2d_tile.cpp:15248-15270). Whether these registers latch DMRISC-sourced SPA traffic is unverified. On 1x1 only M0.D2D0 is read (mimir_hw_state.cpp:635-647), so Cce1To0 has nothing observable for this leg.
- Plan divergence, not a defect: T1 asks for "the local Mimir address". The test issues the package SPA and relies on DMRISC remap entry 1 to keep it on-die. cce_gddr.h:46-52 records the emulator finding that hart stores to the local 0x4040_0000 never reached the sibling. The plan also asks for direction-specific SRAM addresses; the implementation uses one formula for both directions (risc_firmware_initializer.cpp:102-103).

##### 3. UNSURE / questions for the user
1. Is chippy/emu_boot programming of SS_BAR, ATT and interleave inside what MM-5 must catch? If yes, the preflight skip in every remote TL1 test (test_mimir_tl1.cpp:141) is a DEFECT and should become a failure. If it is only a precondition, it can stay a skip, but some test should then gate on mimir_remote_preflight_result().
2. Do the Mimir D2D ATT DEBUG_LAST_* registers (base 0x04290000 + 0x200000 per tile) latch CCE/DMRISC-originated SPA traffic on MMK? If not, SameChipletControl's no-ATT check cannot fail, and DEBUG_LAST cannot serve as the missing remote route evidence (S3).
3. Should the branch record UMD f0d497f? HEAD records 292fe20, which lacks the APIs these tests call.

##### 4. Tests that belong to another section
- PrefetchesLocalGddrToRemoteCceSram.RoundTrip: MM-4 (local GDDR read) plus MM-5-T4 (one-to-two remote TL1 write). It uses no tensor_prefetcher API (cce_prefetch_local_gddr_to_remote_sram.cpp), so it is not MM-7 despite the name.

Not checked: run logs, and whether the routed pairs (for example M0.CCE0, whose D2D0 has no Mimir peer per mimir_2x_package.yaml) take the route the names imply. I did not run anything.

Counts by primary verdict: OK 3, WEAK 0, DEFECT 8, UNSURE 0. Four of the DEFECT rows also carry a WEAK note, and one OK row has a WEAK leg.

### B.7 MM-6 (reviewer mm6)

#### MM-6 review: CCE DM to remote DRAM over D2D (10 tests)

Read at HEAD 16585a9c68a in /localdev/sagarwal/tt-metal, UMD submodule f0d497f3386 (tt_metal/third_party/umd). The descriptors are under tt_metal/third_party/umd/tests/soc_descs/, not under the repo-root tests/soc_descs path the brief gives.

**Conclusion.** The host oracle is independent of the code under test in all 10 tests:
- Host DRAM reads go through `Cluster::read_dram_vec`, which adds the view's address_offset (llrt/tt_cluster.cpp:937-938).
- The UMD resolver then turns the view and offset into a channel address with `dram_address(channel, offset)` (umd device/src/coordinates/grendel_noc_address_resolver.cpp:154-156, 271-298).
- The read finishes on that Mimir's own chippy transport (umd device/src/tt_device/mimir_chippy_memory.cpp:96-103).
- None of this touches the CCE SPA routes, the DMRISC remap, the ATT or SS_BAR.

So a CCE access that lands on the wrong endpoint shows up as a data mismatch at the intended destination. Verdicts:
- **The 5 dedicated tests are WEAK.** Their data gates are real. But some read legs use constant patterns that a misrouted read would match, the D2D error-bit leg turns into logging once a bit is set, a wrong D2D setup produces a skip instead of a failure, and the stage-by-stage address evidence the plan requires is only logged.
- **The 5 emu tests are OK** for endpoint selection. They have no aliasing decoys, no error-bit check and no stage evidence.

**Which descriptors run what.**
- mimir_1x1 skips every MM-6 case except CopiesEveryMimirBlock, which runs there as a local-only copy (test_mimir_remote_dram.cpp:113, 998; mimir_contract.hpp:340).
- mimir_2x_package (address_space mimir_local, 4 views of 512 MiB, routes M0 to M1 over D2D1 and M1 to M0 over D2D0) runs:
  - 8 of the 16 instances of each MimirRemoteDram test;
  - all 4 TrafficMatrix instances;
  - the single Concurrent test;
  - 8 of the 16 instances of each CrossMimirNoc suite;
  - all 4 CopiesEveryMimirBlock instances.
- The other instances always skip on both descriptors because their view is local (:134; mimir_contract.hpp:326-344). That is how the parameters are enumerated, not a defect.
- Hardware wiring backs the routes: grendelemulation models/mmk/emu_hdl/tb_emu_d2d_link.sv:5-7 connects m0.d2d_tile_1 to m1.d2d_tile_0. The bring-up guide states that M0 D2D0 is not connected to Mimir 1. So a wrong-port route cannot deliver the data.

##### 1. Per-test table

| Test | Plan case | Feature (a) | Oracle (b) | Pass/fail and skips (c) | Passes on a broken impl? (d) | Vacuous? (e) | Verdict | Confidence | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| MimirRemoteDram.WriteFromTl1 | MM-6-T1 | CCE TL1 to remote GDDR write through `cce_gddr_address`, both directions, 8 harts, 2 anchors per view | Host readback of the destination view, which is pre-filled with a seed pattern. Seed-pattern decoys sit at the same offset on the source Mimir and at the other view (±4 GiB) of the destination Mimir. | Gated by EXPECT on done magic and XOR, ADD_FAILURE on the data compare, and ADD_FAILURE on new D2D or AXI error bits. Skips: 1 Mimir, local view, preflight mismatch (SS_BAR/ATT/interleave), watcher enabled. | Caught: wrong instance stride, wrong D2D port, remote window claimed by the local remap (data lands on the source decoy), dropped write (pre-fill stays). Not caught: wrong SS_BAR2 or ATT setup, which skips instead of failing. Not gated: predicted SPA, ATT entry and post-SS_BAR address. | No | WEAK | High | :543-603, :466-524, :1197-1224; mimir_hw_state.cpp:395-408, 705-715 |
| MimirRemoteDram.ReadToTl1 | MM-6-T2 | Remote GDDR to own TL1 read, both directions | Host writes an AddressDerived or Prng pattern to remote DRAM and pre-fills TL1 with a guard. Host TL1 readback (host path, not the CCE's) plus the same decoys. | Same gates and skips as T1 | Caught: a misrouted read returns decoy or guard data. Constant patterns are excluded (:50-51, :455-457). Skip leg and log-only stage evidence as in T1. | No | WEAK | High | :606-660, :1226-1253 |
| MimirRemoteDram.Tl1RoundTrip | MM-6-T3 | TL1 A to remote DRAM to TL1 B on the next hart, as two programs | The write leg is checked by host DRAM readback against a seed pre-fill. The read leg is checked by host readback of TL1 B against a guard pre-fill. Decoys as in T1. | Same, plus the read is skipped (with a failure recorded) if the write failed | The write leg is gated. Read leg hole: the pattern comes from all 7 patterns, so AllZeros or AllOnes cases pass a misrouted read that returns zeros or ones. The kernel XOR matches as well. "Exactly one translation" is inferred from data placement only. | No | WEAK | High | :664-771 (patterns :455-457, check :760-762) |
| MimirRemoteDramTrafficMatrix.TrafficMatrix | MM-6-T4 | Sizes {4, 64, 124, 512, 4K, 32K} by alignments {0, 4, 60, 124, 32K-4} plus an end-of-anchor case, for each remote view, write then read | 64 B window guards, alias decoys, destination pre-fill, TL1 B guard and tail guard | Same gates and skips. The read leg runs only after the write leg passes. | AllZeros/AllOnes read-leg hole as in T3 (:787-788, :875-880). Claimed coverage that is not exercised: only the `unres_g0` anchor is used (:1315), so there are no translation-window edges; no channel, bank or interleave sweep; "packets" are 4 B CPU stores (cce_test_common.h:19-26). | No | WEAK | High | :777-891, :1288-1334 |
| MimirRemoteMixed.Concurrent | MM-6-T4 (concurrent) | One program: per CCE, hart 1 local DRAM, hart 2 remote DRAM, hart 3 remote TL1, both directions. A second program reads everything back. | Host readback of every destination, remote DRAM decoys (a TT_FATAL check confirms no overlap), TL1 B readback | Same gates and skips | AllZeros/AllOnes read-back hole (:959-963, :1181-1188). The test cannot show the traffic actually overlapped; a serialized implementation passes. Stage evidence is log-only. | No | WEAK | Medium (concurrency not observable) | :914-990, :994-1193 |
| CopiesEveryMimirBlock.RoundTrip | MM-6 (Group C) | One CCE copies src to dst on every Mimir (local and remote) through `cce_gddr_read`/`cce_gddr_write` | Host writes a pattern containing the cce and mimir bytes, clears dst, then reads back the base view of each Mimir | EXPECT_EQ per Mimir. Skip only if the CCE does not exist. | Caught: remote access routed to the local Mimir (remote dst stays 0), and a stale staging buffer (the previous iteration's data differs by the mimir byte). No decoys or error bits. | No | OK | High | test_mimir_emu.cpp:382-432; cce_dram_copy_all_partitions.cpp:17-20 |
| MimirEmuCrossMimirNoc/CopiesGddrWithCompileTimeArgs.RoundTrip | MM-6-T3-like | Remote read into TL1 staging, then remote write, compile-time args | Host clears dst. The pattern encodes cce, mimir and location. Host dst readback, plus a host check of the staging area. | 2 EXPECT_EQ. Skips for local partitions and on 1x1. | Caught: wrong instance, port or view (pattern bytes differ, dst stays 0). Minor: staging is not seeded before the run. | No | OK | High | :175-251, mimir_contract.hpp:276-279, 326-344 |
| MimirEmuCrossMimirNoc/CopiesGddrWithRuntimeArgs.RoundTrip | MM-6-T3-like | Same with runtime args | Host dst readback | EXPECT_EQ, same skips | Same as above. No staging check. | No | OK | High | :255-328; cce_dram_buffer_round_trip.cpp:20-24 |
| MimirEmuCrossMimirNoc/CopiesGddrLargerThanAlignment.RoundTrip | MM-6-T4-like (size) | Same at min(16 x alignment, 32 KiB) | Host dst readback | EXPECT_EQ, same skips | Same as above | No | OK | High | :436-511 |
| MimirEmuCrossMimirNoc/AllHartsWriteGddr.WritesMagic | MM-6-T1-like (8 harts) | All 8 harts write one word each to remote GDDR | Host clears dst, unique value per hart, host readback | ASSERT 8 harts, EXPECT_EQ, same skips | Caught: a wrong route on any hart | No | OK | High | :718-787; cce_gddr_write_one_uint32.cpp:18-25 |

##### 2. Details for WEAK rows and defect-class legs

**A. A wrong D2D setup produces a skip, not a failure.** This is a defect-class leg in all 5 dedicated tests.
- Every test calls `GTEST_SKIP() << mimir_remote_preflight_result()` (:1001-1004, :1206-1209, :1235-1238, :1264-1267, :1297-1300).
- `preflight_result` compares the SS_BAR0-3, ATT ENABLE/ENDPOINT and GDDR interleave readback against expected values copied from chippy (mimir_hw_state.cpp:221-266, 395-408).
- If SS_BAR2 BASE_21 or OFFSET_21 is wrong, which is exactly the "exactly one translation" property, the case skips.
- With watcher enabled, all cases skip silently (:426-431). That is an environment variable turning the gate into a skip.
- By contrast, the DMRISC remap check is correctly gated with EXPECT_TRUE (:1007, :1212, :1241, :1270, :1303).
- Neither Metal nor UMD programs SS_BAR or ATT; my grep found no writer.
- If you count that setup as MM-6's feature, this leg is a DEFECT. If it is a chippy precondition, the leg is an environment guard and the tests stay WEAK. See question 1.

**B. "Inconclusive" D2D error bits are logged, not failed.** Defect-class leg; the tests stay WEAK because the data gate remains.
- `expect_no_new_d2d_error_bits` only logs a bit that was already set before the snapshot (mimir_hw_state.cpp:708-710).
- The code itself says stickiness is assumed (:652).
- If the bits are sticky, one earlier fault in the process makes this leg record-only for every later case.

**C. Stage-by-stage address evidence is never compared.**
- `make_gddr_record` builds the issued SPA, the ATT endpoint and the address after SS_BAR2 (:206-219). These only go to `log_transfer` and the failure message.
- `dump_debug_last` reads DEBUG_LAST_* only after a failure, and only logs them (mimir_hw_state.cpp:765-769).
- The plan's requirements on translation, return mapping and the "exactly one translation" check are covered only indirectly, by pre-fill plus decoys.
- Decoys that cannot be placed are dropped with a log_info (:286-304), and nothing asserts how many were placed.

**D. Read legs that use AllZeros or AllOnes** (Tl1RoundTrip, TrafficMatrix, Concurrent).
- Scenario: the remote read is misrouted to an unwritten page, or one that holds 0x0 or 0xFFFFFFFF, and returns that value.
- The TL1 B compare passes, because the guard pre-fill is overwritten with the matching constant.
- `xor_of(AllZeros)` is 0, so the done check passes too.
- This affects about 2 in 7 cases.
- ReadToTl1 avoids this deliberately (:50-51), but the other read legs use `kAllMimirPatterns` (:457, :788, :960).

**E. TrafficMatrix coverage is narrower than claimed.**
- Only anchor `unres_g0` is used (:1315).
- With 512 MiB views, `mimir_dram_anchors` produces no GiB-edge anchors at all (mimir_access.hpp:708-726), so translation-window edges are not covered.
- There is no channel, bank or interleave sweep.
- Every size is a loop of 4 B stores, so packet size is not exercised.

##### 3. UNSURE / questions for the user
1. Is checking the chippy-programmed D2D setup (SS_BAR/ATT readback, mimir_hw_state.cpp:221-266) part of what MM-6 validates? If yes, the preflight skip in all 5 dedicated tests is a DEFECT and should fail instead. If no, it is an acceptable environment guard.
2. Are the ATT DEBUG_MISC bits 9-11 and the CCE AXI_READ/WRITE_ERROR bits sticky on the MMK emulator? If they are, an earlier fault makes the error-bit leg log-only for the rest of the run.
3. On the MMK emulator, what does a CCE load from an unmapped or misrouted GDDR SPA return: zeros, an error response, or a hang? This sets how serious the AllZeros/AllOnes read-leg hole (2D) is.
4. Should the predicted stage chain (issued SPA, ATT endpoint entry, address after SS_BAR2) be asserted against ATT DEBUG_LAST_*, as the plan's "stage-by-stage address evidence" asks? Today it is only logged.
5. MM-6-T5 (D2D recovery and the negative controls) has no test on this branch; my grep found none. Is that intended?
6. Are CCE hart loads from the GDDR SPA window cacheable, and do cache lines survive between programs? If so, ReadToTl1 cases that reuse AddressDerived data written earlier by the same hart at the same address could be served from cache without crossing D2D. I did not check this.

**Not checked:** slow-dispatch launch and completion plumbing; the register semantics behind the expected SS_BAR/ATT values (taken from comments that cite chippy noc2axi_m2k.py); recorded emulator runs; whether the 12 Concurrent launches actually overlap in time.

##### 4. Tests that may belong to another section
- CopiesEveryMimirBlock.RoundTrip on mimir_1x1 is a local-only copy there (MM-4). It only crosses D2D on mimir_2x_package.
- MimirEmuCrossMimirNoc/*: the "Noc" in the name does not mean a different path. These run the same `cce_gddr` SPA load/store kernels as MimirEmuSameMimirDirect/*. The only difference is which partitions run, chosen by matching the suite name (mimir_contract.hpp:276-279). They are correctly MM-6.
- MimirRemoteMixed.Concurrent includes a remote-TL1 leg (an MM-5 feature) as part of the T4 concurrency mix. It is correctly MM-6.

Files: /localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/dram/test_mimir_remote_dram.cpp, /localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/context/test_mimir_emu.cpp, /localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/dram/mimir_hw_state.cpp, /localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/dram/mimir_access.hpp, /localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/context/mimir_contract.hpp, /localdev/sagarwal/tt-metal/tt_metal/impl/device/firmware/risc_firmware_initializer.cpp (routes :65-110), /localdev/sagarwal/tt-metal/tt_metal/third_party/umd/tests/soc_descs/mimir_2x_package.yaml.

### B.8 MM-7 (reviewer mm7)

#### MM-7 Tensor Prefetcher review (56 tests)

Totals: OK 47, WEAK 8, DEFECT 1 (MM7T5TraceCaptureReplay), UNSURE rows 0 (5 questions in section 3).

I read everything at HEAD 16585a9c68a in /localdev/sagarwal/tt-metal. I built nothing and ran nothing.

Short names:
- TF = tests/tt_metal/tt_metal/dram/test_mimir_prefetcher.cpp
- MGR = tt_metal/impl/buffers/tensor_prefetcher_manager.cpp
- KRN = tt_metal/impl/buffers/kernels/tensor_prefetcher_cce.cpp
- RCB = tt_metal/hw/inc/experimental/cce_remote_cb.h

Every test starts with skip_unless_programmable_dram_package (tests/tt_metal/tt_metal/context/mimir_contract.hpp:122-136). It skips when the package has no programmable DRAM, no transport is configured, or TT_METAL_SLOW_DISPATCH_MODE is unset. Every test also skips when the watcher is on (C6). Neither skip fires in the README environment on mimir_1x1 (2 CCEs, 2 views) or mimir_2x_package (4 CCEs, 4 views). In the table, "both" and "2x" name the descriptors a case runs on. C1 to C8 are shared caveats, explained after the table.

##### 1. One row per test

| Test | Plan case | Feature (a) | Oracle (b) | Pass/fail and skips (c) | Passes on a broken impl? (d) | Vacuous? (e) | Verdict | Conf | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| MM7GatePreconditionsHold | Port item 1 (capability gate) | Preconditions ok, IsTensorPrefetcherSupported, one sender (view,0) per view | The production predicate, plus a hard-coded (view,0) rule | EXPECT ok/supported, ASSERT sender list; both | Yes if the predicate always returns ok, but GateUninitializedSenderFails catches that | No | OK | high | TF:112-139; MGR:2399-2496 |
| MM7GateUninitializedSenderFails | Port item 1 | A sender at RUN_MSG_INIT fails the preconditions, which name that CCE and the signal | Reason string contains "CCE n " and "0x<INIT>" | ASSERT DONE first, then EXPECT !ok and both substrings; both | Not caught: a check that only inspects the first view (only bank 0 is set to INIT). Caught: a check that ignores GO_MSG | No | OK | high | TF:141-164; MGR:2470-2494 |
| MM7GcbLayoutLocalReceiver | Port item 4 (GCB contract) | DRAM-receiver GCB state block, counters and config page; a second GCB is disjoint | Fill pattern before creation, then host readback against arena/HAL addresses and test CCE indices | About 30 EXPECTs; both | Caught: a missed write (pattern stays), a wrong receiver CCE (round trip logical_cce_core -> package_dram_core_index), an overlapping second GCB | No | OK | high | TF:188-320, 323-345, 358-377; global_circular_buffer.cpp:263-362 |
| MM7GcbLayoutRemoteReceiver | Port item 4 | Same, remote receiver listed first, table order kept | As above | Skips with one Mimir; 2x | Caught: a sorted receiver table | No | OK (layout only, no D2D traffic) | high | TF:379-396 |
| MM7GcbLayoutTwoSenders | Port item 4 | Per-sender blocks for 2 senders; a second GCB is disjoint | As above | Skips with one Mimir, ASSERTs a second bank; 2x | Caught: a shared receiver count, overlap | No | OK | high | TF:398-441 |
| MM7GcbLayoutRejectsConflictingReceivers | Port item 4 (negative) | Rejects aliases, the sender (or its alias) as receiver, duplicates | Message substrings | 4 EXPECT_NE on find; ASSERTs the alias pairs exist; both | Caught: a missing check (empty error string fails find). The substrings are specific to reject_conflicting_dram_receivers | No | OK | high | TF:443-501; global_circular_buffer.cpp:117-156, 741 |
| MM7GcbSmokeDiscardDrain | Port items 3/4 | RCB protocol: test sender into a discard receiver, 20 pages that wrap | Both statuses (poisoned before) plus a host model of credits, pointers and guard | EXPECTs; both | Caught: bad credit or wrap math (model), unpublished acks (DRAIN_TIMEOUT), overrun past the FIFO (guard) | No | OK (C5) | high | TF:538-562, 639-649, 751-779, 855-879 |
| MM7GcbSmokeValidatorLocal | Port item 3 | Validator receiver, local data path | Validator recomputes gcb_page_word, plus end state | EXPECTs; both | Credits with no stored data pass only under C2 | No | OK (C2, C3) | high | TF:883-921; cce_gcb_validator_receiver.cpp:56-91 |
| MM7GcbSmokeValidatorWrongSeed | Port item 3 / T4 | The validator reports a mismatch and keeps acking | code == Mismatch, mismatched words > 0 | EXPECTs; both | A sender that delivers nothing still passes, but that is not this row's claim; ValidatorLocal covers delivery | No | OK | high | TF:899-904, 924-936 |
| MM7GcbSmokeResize | Port item 4 | Three page sizes; resize and wrap padding | Validator plus host padding model | EXPECTs; both | A misreading of the padding rule that the host model and the header share passes (C5) | No | OK (C5) | med | TF:940-954 |
| MM7GcbSmokeValidatorD2d | Port items 3/4, D2D | Validator on the other Mimir | As ValidatorLocal, plus EXPECT remap empty | Skips with one Mimir and on preflight (C1); 2x | Caught: wrong endpoint (validator stalls, then reports Stopped) | No | OK (C1) | high | TF:844-852, 956-972 |
| MM7GcbSmokeDrainUntilStopped | Port item 3 | Discard receiver runs until the host stops it | Status still poison after the sender ends, then Stopped, then end state | EXPECTs; both | Caught: a receiver that exits on its own | No | OK | high | TF:975-1006 |
| MM7GcbSmokeStallUntilReleased | T4 basis | A full FIFO gives the sender CREDIT_TIMEOUT, then release | Sender code, pages, wait receiver, sent/acked; holding line; Released; rd_ptr stays 0 | EXPECT/ASSERT; both | Caught: no flow control, wrong slowest receiver, rd_ptr written back | No | OK (C7) | high | TF:1011-1054; cce_gcb_discard_receiver.cpp:106-124, 159-162 |
| MM7GcbSmokeDrainTimeout | T4 basis | Held pages make the sender's final barrier give DRAIN_TIMEOUT | Code, pages, holding, Released, end state with acked 0 | EXPECT/ASSERT; both | Caught: a barrier that does not wait | No | OK (C7) | high | TF:1058-1088 |
| MM7GcbSmokeReceiverWaitTimeout | T4 basis | Receiver's page wait times out after host credits | CreditTimeout, 4 pages, rd_ptr written back | EXPECTs; both | Caught: no expiry, missing rd_ptr write-back | No | OK (C7) | high | TF:1092-1122 |
| MM7GcbSmokeRelaunchResumes | Port item 4 / T3 | A second sender/validator pair resumes from the TL1 pointers | Two validators plus combined end state | EXPECTs; both | Caught: a reset to the FIFO start (wr_ptr model and data mismatch) | No | OK | high | TF:1126-1162 |
| MM7GcbSmokePreconditionsWithReceiverRunning | Port item 1 | GO on a sender CCE keeps the preconditions ok | go == GO, ok, receiver Stopped | EXPECTs, ASSERT a second sender; both | Caught: a check that rejects GO | No | OK | high | TF:1166-1198; MGR:2476 |
| MM7GateStartStopWorks | Port item 1 | Start then Stop with no requests | Start and Stop do not throw | ASSERT supported; both | Yes: a sender kernel that exits without reading the ring or writing status (failed_cce_senders only throws on Failed) | No, but redundant with LifecycleStartStop | WEAK | high | TF:1339-1349; MGR:2236-2290, 2319-2356 |
| MM7LifecycleStartStop | Port item 1 | Sender runs, takes STOP, reports Stopped/Ok | go == GO after Start; state, code, request index 0; bytes_acked = 1 page | EXPECTs; both | Caught: kernel not launched, STOP not consumed | No | OK | high | TF:1263-1270, 1352-1371; KRN:307-363 |
| MM7LifecycleStartStopTwice | Port item 1 | Second Start zeroes the control block; a stale ring slot is never read | Pattern over the zeroed region plus a bad command in slot 0; then status, bytes_sent, CQ slots 0 | EXPECTs; both | Caught: skipped zeroing (BadCmd, CQ slots non-zero) | No | OK | high | TF:1378-1410; MGR:1100-1107, 1275-1276 |
| MM7LifecycleBusySenderFailsStart | Port item 1 (negative) | Start fails on a busy sender before launching or zeroing | "sender CCE n " substring; inactive; idle CCE stays DONE; both control blocks keep the pattern | EXPECTs; bare skip unreachable (C8); both | Caught: zeroing or launching before the check | No | OK | high | TF:1414-1458; MGR:1080-1098, 1255 |
| MM7LifecycleQueueRejectsReceiverOnSender | Port item 1 (negative) | Queue rejects a receiver that is a provisioned sender | 2 substrings; each sender takes only STOP (1 request) | EXPECTs; bare skip unreachable (C8); both | Caught: a missing check (the empty tensor list then gives a different error), a drain recorded before the check | No | OK | high | TF:1462-1487; MGR:1697-1701, 1755-1778 |
| MM7LifecycleWaitCq | Port items 1/5 | WAIT_CQ holds until the command queue writes its slot | WaitingCq seen and held 5 s; 3 requests; slot == value + 1 | ASSERT/EXPECTs; both | Caught: a WAIT_CQ that does not wait (held poll sees Running), host and kernel disagreeing on the slot (CqTimeout, Stop throws) | No | OK | high | TF:1493-1538; MGR:1888-1972; KRN:99-124 |
| MM7LifecycleDrainAtStop | Port item 1 | Stop's DRAIN holds until the page is acked | Draining seen and held 5 s; Stop does not throw; 2 requests | EXPECTs; both | Caught: a DRAIN that does not wait, wrong counters (DrainTimeout) | No | OK | high | TF:1543-1588; KRN:62-96 |
| MM7T1Local0To1 | MM-7-T1 | One-page prefetch from DRAM to a local CCE, two sessions | Validator per page (tensor 0, then 1); 3 sender requests per session; end state | EXPECT/ASSERT; both | Caught: wrong bank or view, wrong receiver, no data (the alternating tensors defeat stale bytes within the test) | No | OK (C2, C3) | high | TF:1704-1749 |
| MM7T1Local3To2 | MM-7-T1 | Same on Mimir 1 (view offset 4 GiB) | As above | Skips with fewer than 4 CCEs; 2x | As above | No | OK | high | TF:1751-1757 |
| MM7T1D2d0To2 | MM-7-T1 | D2D, M0.CCE0 to M1.CCE0 | As above, plus remap EXPECT | C1; 2x | Caught: wrong endpoint, wrong translation (validator reads its own TL1) | No | OK (C1) | high | TF:1759-1766 |
| MM7T1D2d3To1 | MM-7-T1 | D2D, M1.CCE1 to M0.CCE1 | As above | C1; 2x | As above | No | OK (C1) | high | TF:1768-1775 |
| MM7T2FanOut0To123 | MM-7-T2 | One sender to 3 receivers, each its own slab; second round waits for all acks | 3 validators (slab = slot); 3 end states | EXPECTs; C1; 2x | Caught: wrong slab mapping, a free-space check that ignores a receiver (overrun, mismatch) | No | OK (C1) | high | TF:1802-1834, 1879-1886 |
| MM7T2FanOut3To012 | MM-7-T2 | Same from M1.CCE1 | As above | C1; 2x | As above | No | OK (C1) | high | TF:1888-1895 |
| MM7T2TwoSenders01To2 | MM-7-T2 | 2 senders, 2 GCBs, one receiver CCE on 2 harts | Disjoint ranges, 2 validators, 2 end states | EXPECTs; C1; 2x | Caught: cross-stream corruption (tensor ids differ), overlapping windows | No | OK (C1) | high | TF:1838-1877, 1897-1904 |
| MM7T2TwoSenders02To3 | MM-7-T2 | Same, senders on both Mimirs | As above | C1; 2x | As above | No | OK (C1) | high | TF:1906-1913 |
| MM7T3SubChunkLocal0To1 | MM-7-T3 | 16 KiB block against a 12 KiB stage third, 8 KiB sub-chunks | Validator, end state, 3 requests | EXPECTs; both | Yes for the named path: if the stage ring cap is ignored, one chunk is used and the case still passes | No | WEAK | med | TF:1949-2001, 2073-2079; MGR:219-221; KRN:185-229 |
| MM7T3SubChunkD2d0To2 | MM-7-T3 | Same across D2D | As above | C1; 2x | As above | No | WEAK | med | TF:2081-2088 |
| MM7T3BelowStageThirdLocal0To1 | MM-7-T3 | 4 x 4 KiB blocks against an 8 KiB third | As above | both | Named path not asserted (same scenario) | No | WEAK | med | TF:2005-2012, 2090-2096 |
| MM7T3BelowStageThirdD2d0To2 | MM-7-T3 | Same across D2D | As above | C1; 2x | As above | No | WEAK | med | TF:2098-2105 |
| MM7T3PartialStageChunkLocal0To1 | MM-7-T3 | 12 KiB round as an 8 KiB then a 4 KiB chunk | As above | both | As above | No | WEAK | med | TF:2016-2023, 2107-2113 |
| MM7T3AboveStageThirdLocal0To1 | MM-7-T3 | 8 KiB block against the default 4 KiB third | As above | both | As above | No | WEAK | med | TF:2026-2032, 2115-2121 |
| MM7T3AboveStageThirdD2d0To2 | MM-7-T3 | Same across D2D | As above | C1; 2x | As above | No | WEAK | med | TF:2123-2130 |
| MM7T3PageSplitLocal0To1 | MM-7-T3 | One Queue call split into 2 PREFETCH pages | 4 sender requests, plus validator for both tensors | EXPECTs; both | Caught: no split (3 requests), a bad second page (mismatch) | No | OK | high | TF:2036-2041, 2133-2139; tensor_prefetcher_request.hpp:75; MGR:1551-1576 |
| MM7T3AbaLocal0To1 | MM-7-T3 | Tensors A, B, A' with resize and wrap | Validator, padding model, 5 requests | EXPECTs; both | Caught: bad resize, wrong resume | No | OK (C5) | high | TF:2046-2055, 2141-2147 |
| MM7T3AbaD2d0To2 | MM-7-T3 | Same across D2D | As above | C1; 2x | As above | No | OK (C1, C5) | high | TF:2149-2156 |
| MM7T3WrapAndResizeLocal0To1 | MM-7-T3 | 4K, 8K, 8K, 4K pages in a 12 KiB FIFO, gap credited at the wrap | As above, 6 requests | both | As above | No | OK (C5) | high | TF:2061-2071, 2158-2164 |
| MM7T3WrapAndResizeD2d0To2 | MM-7-T3 | Same across D2D | As above | C1; 2x | As above | No | OK (C1, C5) | high | TF:2166-2173 |
| MM7T3GcbSwitchLocal0To1 | MM-7-T3 | GCB1, GCB2, GCB1 resuming mid-FIFO; one DRAIN names both | 2 validators on 2 harts, disjoint ranges, 5 requests | EXPECTs; both | Caught: no resume from the state block, a DRAIN that misses a GCB | No | OK | high | TF:2186-2228 |
| MM7T3GcbSwitchD2d0To2 | MM-7-T3 | Same across D2D | As above | C1; 2x | As above | No | OK (C1) | high | TF:2230-2237 |
| MM7T3ReceiverSwitch0To1Then2 | MM-7-T3 | Switch receiver CCE 1, then 2 across D2D, then back | As above | C1; 2x | As above | No | OK (C1) | high | TF:2240-2248 |
| MM7T3FenceLocal0To1 | Port item 5 / T3 | A WAIT_CQ fence holds a queued PREFETCH | WaitingCq held; bytes_acked 0; receiver status poison; counters 0; then 4 requests and validator | ASSERT/EXPECTs; both | Caught: a fence that does not wait. Not covered: DRAM writes visible before release (tensor written before Start; slot written by the host, not the CQ path) | No | OK (gap, Q4) | high | TF:2254-2298 |
| MM7T5TraceCaptureReplay | MM-7-T5 | Trace capture and replay | None | Unconditional bare GTEST_SKIP(); both | n/a | Yes | DEFECT | high | TF:2355 |
| MM7T4MismatchD2d2To0 | MM-7-T4 | Validator MISMATCH, then recovery | Mismatch, 1024 words, first word 0, observed == true word 0; recovery validator Ok | EXPECTs; C1; 2x | Words 1 to 1023 are not checked against the true data in the failure leg; the recovery leg checks a full page | No | OK (C1, C4) | high | TF:2360-2404, 2330-2352 |
| MM7T4CreditTimeoutD2d1To3 | MM-7-T4 | Stalled receiver: CREDIT_TIMEOUT, ring fill, dropped pages, Stop throws, recovery | Status fields; holding 1; exact Stop message parts (3 dropped, 128 unread); bytes_sent - bytes_acked == ring size | EXPECT/ASSERT; C1; 2x | Caught: no flow control, wrong drop accounting | No | OK (C1, C4, C7) | high | TF:2410-2467; MGR:2108-2131, 2282-2290, 2319-2344 |
| MM7T4DrainTimeoutD2d1To3 | MM-7-T4 | Stop's DRAIN gives DRAIN_TIMEOUT, Stop throws, recovery | "ended Failed"; status code, request index 1, target, sent, acked | EXPECT/ASSERT; C1; 2x | Caught: a DRAIN that does not wait (Stop does not throw), wrong target | No | OK (C1, C4, C7) | high | TF:2471-2509; KRN:62-96 |
| MM7RejectKRowMajorTensor | Port item 1 / negative dims | Queue rejects an interleaved (K-row-major) tensor | Substring; sender takes only the valid page (3 requests); validator | EXPECTs; both | Caught: no check (no throw, or 4 requests) | No | OK (C4) | high | TF:2521-2578; MGR:1453-1465 |
| MM7RejectStreamingTensor | Same | Rejects a streaming rotation | "has a streaming rotation" | both | As above | No | OK (C4) | high | TF:2580-2598; MGR:1369-1377 |
| MM7RejectRangePastTensor | Same | Rejects slabs that are not on the sender's bank | "so the request runs past the tensor" | Bare skip with fewer than 2 views, unreachable (C8); both | As above | No | OK (C4) | high | TF:2601-2621; MGR:753-788 |
| MM7RejectRangePastView | Same | Rejects a read past the DRAM view | "B DRAM view" (only MGR:791 has it) | both; an error while creating the buffer propagates and fails the test | As above | No | OK (C4) | high | TF:2625-2652; MGR:789-796 |

###### Shared caveats

- C1. D2D tests skip instead of failing when the D2D setup is wrong. route_skip_reason and gcb_smoke_d2d_skip_reason return mimir_remote_preflight_result(). That skips on an identity mismatch or on SS_BAR/ATT/interleave values that differ from the expected ones (TF:844-852, 1612-1627; mimir_hw_state.cpp:395-408). About 20 tests on 2x are affected. The CCE remap check after CreateDevice does gate (EXPECT_TRUE(mimir_remote_remap_result().empty())). See Q1.
- C2. The receiver FIFO is never cleared or seeded.
  - GcbSmoke poisons only the kernel block and writes a guard after the FIFO (TF:615-618, 639-649). On a fresh mesh the FIFO sits at the same TL1 offset every run (TF:221).
  - Protection against stale bytes therefore rests on mimir_test_seed() (mimir_patterns.hpp:71-78). It uses gtest's random seed, which is taken from the clock per process when the flag is 0 (googletest gtest.cc:5815; gtest-internal-inl.h:112-124).
  - With --gtest_random_seed=N or GTEST_RANDOM_SEED, a rerun on CCEs that were not reset finds the previous run's bytes. A sender that publishes credits without storing data would then pass every validator test.
  - Running many tests in one process has the same risk for tensor ids that repeat: T1 (0/1), T4 (20/21), Reject (30). The README runs one test per invocation.
- C3. Data correctness is checked only by the validator kernel on the device; no host reads the FIFO back. I accept this because ValidatorWrongSeed (both descriptors) and T4Mismatch (2x only) show the comparison works. On 1x1 only ValidatorWrongSeed backs it.
- C4. The guard after the FIFO is compared only in expect_fifo_state. These would not catch a write past the FIFO end, though their main asserts still gate:
  - the T4 failure legs and expect_t4_recovery (TF:2330-2352);
  - run_rejected_queue_case (TF:2521-2555);
  - PreconditionsWithReceiverRunning and BusySenderFailsStart.
- C5. The padding and wrap oracle is a host copy of the device code. expected_gcb_fifo_end_state (TF:538-562) restates RCB:71-109 line for line, and sender and receiver share RCB. A misreading of the Blackhole remote-CB padding rule shared by all three passes Resize, Aba*, WrapAndResize* and the end-state checks.
- C6. Setting TT_METAL_WATCHER turns every test into a skip (mimir_hw_state.cpp:426-431).
- C7. The rdcycle budgets are tuned for the emulator (2^20 and 2^24 cycles, TF:515-520). On silicon they are short, so the stall and timeout tests may fail or be flaky there. This does not make them vacuous.
- C8. There are bare GTEST_SKIP() calls with no message at TF:1424, 1472 and 2607. None is reachable on either descriptor.

##### 2. DEFECT and WEAK details

- MM7T5TraceCaptureReplay (TF:2355): the body is only GTEST_SKIP(), with no message and no assert, so it cannot fail. The plan accepts "explicitly documented as unavailable", but this documents nothing. A version that can fail could assert that tensor_prefetcher_preconditions rejects fast dispatch (MGR:2454-2456).
- MM7GateStartStopWorks (TF:1339-1349):
  - It passes as long as Start and Stop do not throw.
  - Stop throws only on a WaitProgramDone timeout, a worker give-up, or a Failed status (MGR:2273-2290).
  - A sender kernel that returns at once leaves the status NotStarted, and the test passes.
  - MM7LifecycleStartStop (TF:1352-1371) checks the status and go signal, so this test is redundant rather than harmful.
- The 7 T3 fit-ladder tests (SubChunk local/D2D, BelowStageThird local/D2D, PartialStageChunk local, AboveStageThird local/D2D; TF:1992-2032, 2073-2130):
  - Each name claims a chunking path, chosen by stage_ring_bytes acting as a cap in carve_kernel_region (MGR:219-221). No test asserts the path was taken (no host-visible stage_ring_size or layout is checked).
  - If the cap were ignored, the stage ring would grow to most of the ~92 KB kernel region and every case would copy in one chunk. All 7 tests would still pass.
  - The data is checked byte for byte, so a broken chunk loop is caught whenever its path actually runs.

##### 3. UNSURE / questions for the user

1. Who sets up the D2D SS_BAR/ATT/interleave registers that mimir_remote_preflight_result checks: the chippy bring-up script or Metal/UMD? If Metal/UMD owns it, about 20 D2D tests on 2x skip when that setup is broken (failure mode 2) and should fail instead.
2. Should the validator tests poison the receiver FIFO before each run (C2)? Without it, a rerun with a fixed seed on CCEs that were not reset can pass a sender that publishes credits without data.
3. For MM-7-T5, is a bare skip acceptable as "documented as unavailable", or should the test assert the rejection (for example, the fast-dispatch precondition)?
4. Port item 5 says "Validate that DRAM writes are visible before a queued request is consumed". No test writes a tensor after Start, fences with WaitForCqOnTensorPrefetcher, then prefetches it. MM7LifecycleWaitCq moves no data, and MM7T3FenceLocal0To1 writes the slot from the host. Is that in scope for MM-7?
5. Are these tests meant to run on silicon? If so, the fixed rdcycle budgets (C7) need rescaling.

##### 4. Tests that belong to another section

None. The Gate, GcbLayout, GcbSmoke, Lifecycle and Reject tests map to MM-7 porting items 1, 3, 4 and 5 rather than to a T-case.

Not checked: the internals of launch_cce_program, read_cce_tl1 and write_cce_tl1, the UMD write_core path, and how the tests behave on the emulator. I did not judge any row "fails on emulator, gates correctly".

### B.9 MM-8 (reviewer mm8)

#### MM-8 report (Group A, 12 tests)

Read at HEAD 16585a9c68a (branch sagarwal/mimir_dram_tests, /localdev/sagarwal/tt-metal); each file exists in one copy only.
Conclusion: none of the 12 tests is vacuous. Every one gates on a host-checked oracle that is independent of the code under test, or on a boundary pair. No DEFECTs. Three WEAK, each a gap rather than a pass-anything path: DramUnary (the reader's slot-reuse wait is not gated directly), CachePerf (the plan's perf baselines are not implemented), MeshBuffer InterleavedPlacement (the oracle shares Cluster's view translation; within its stated scope).

##### 1. Per-test table

| Test | Plan case | Feature (a) | Oracle (b) | Pass/fail and skips (c) | Passes on a broken impl? (d) | Vacuous? (e) | Verdict | Confidence | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| MimirDmAddrgen.Pattern (9 inst.) | MM-8-T3, Tier B quasar_addrgen | Addrgen 0 on CCE0 hart 0: 1D, 2D and Face patterns, src-only, dst-only and both | A host loop model (SideModel) predicts every src/dest address. The kernel pops the addresses into TL1, which is poisoned first, and the host reads them back | ASSERT status present, ASSERT code==Done, EXPECT mismatches==0. Skips: package/transport and watcher only. Runs on both descriptors | Checked: strides ignored; dest loops not cleared by reset (a "DstOnly" side would stride); pops of more than one address; wrong face advance. All caught. Every loop end here is a multiple of its stride, so "restart at 0" and "mod end" predict the same addresses and the model is independent of the hardware | No | OK | High | test_mimir_dm_addrgen.cpp:53-104,118-119,137-172; cce_addrgen.cpp:21-66; cce_addrgen_abi.h:63-72 |
| MimirDmIm2col.Pattern (2 inst.) | MM-8-T3, Tier B quasar_im2col | Addrgen with the im2col patterns, including the software patch loop | Same host model, plus patch base rewrites | Same as above | FullImage: same as above, High. Dilation1: its loop ends are stride+1 (cce_addrgen_abi.h:89-90), so the two wrap readings diverge. The model uses "restart", which the test comment says is what the CCE does. If the model was fitted to emulator output, a wrong RTL wrap would match it (Q1) | No | OK (Dilation1: medium) | FullImage High; Dilation1 Medium | test_mimir_dm_addrgen.cpp:49-52,93-96; cce_addrgen_abi.h:73-92; README:204 |
| MimirDmCache.Op (12 inst.) | 4.1 functional (quasar_cache port) | L1 line/full flush, L1 line/full invalidate, L2 line/range/full flush, L2 invalidate, fresh read, hart-to-hart coherence | Per-run host prefill of old values plus a guard. The kernel's uncached-alias check counts new words; host readback and guard check are a second observer. The kernel's cached-before (==64) and uncached-before (==0) checks prove the stores ran and stayed in cache | ASSERT code==Done, EXPECT cached/uncached-before, EXPECT matches==64 or 0, EXPECT host data and guard. Fresh-read cases: EXPECT stale_before==64. Skips: watcher only. Runs on both | Checked: no-op cflush (the line is then discarded, host sees old: caught); no-op cdiscard (the L2 range flush writes new: caught by the expect-old cases); no-op or flush-like INVALIDATE64; no-op FLUSH64; caching disabled (uncached_before==64 or stale_before==0); stores lost. All caught. Not caught: a broken L1 full flush combined with a broken L2 invalidate in L2InvalidateLine (needs two bugs; L1FlushFull covers the first). In HartToHartCoherent, stale_before==0 is asserted before the invalidate, so the invalidate leg cannot detect a broken invalidate (the case targets coherence) | No | OK | High | test_mimir_dm_cache.cpp:140-159,206-302; cce_cache_ops.cpp:87-121,1236-1275; quasar/cache.h:17-41; risc_common.h:186-187,206-224 |
| MimirDmCachePerf.WritePath (4 inst.) | 4.1 perf / Tier B quasar_cache_perf | The 4 write paths (uncached 1 B and 8 B, cached plus range flush, cached plus bare FLUSH64), sizes 8 B to 32 KiB | Pass 0 writes the complement, pass 1 the values. Kernel uncached check, host readback and guard | ASSERT Done, EXPECT matches==num_words, EXPECT host data and guard. No timing gate. Skips: watcher only. Runs on both | Checked: no-op flush (TL1 keeps the prefill: caught); lost pass-1 store (the complement shows: caught); widened byte stores (caught); overrun (guard: caught). Perf regressions are not detected: cycles appear only in log_info (:382) | Functional part no | WEAK (plan coverage gap) | High | test_mimir_dm_cache.cpp:344-387; cce_cache_ops.cpp:123-160; plan 4.1 perf |
| MimirDmScopedLockClm.CacheOps (3 inst.) | 4.1 / Tier B scoped_lock_cache, CLM rows | scoped_lock_acquire_cache_ops (L2 range invalidate) and scoped_lock_release_cache_ops (L2 range flush) on the locked lines only | The host seeds OLD lines and seeded result lines. The consumer writes its warm, cached-after and uncached-after reads, which the host reads. Unlocked lines must stay OLD (negative control). For FlushOnRelease, the TL1 value at the producer's first read is checked | ASSERT both codes Done (a timeout gives ADD_FAILURE). Invalidate cases: EXPECT warm==OLD, uncached==NEW, locked cached==NEW, unlocked cached==OLD. FlushOnRelease: EXPECT LandedAtFirstRead, locked==NEW, unlocked==OLD. Guards. Skips: watcher only. Runs on both | Checked: no-op acquire (locked reads OLD: caught); over-wide acquire (unlocked reads NEW: caught); no-op release (data timeout, ASSERT fails: caught); over-wide release (unlocked lands: caught); caching disabled (unlocked reads NEW: caught). The kernels call the real runtime header | No | OK | High | test_mimir_dm_scoped_lock.cpp:212-385; cce_scoped_lock_consumer.cpp:143-200; cce_scoped_lock_producer.cpp:106-161; scoped_lock_cache_ops.h:14-30 |
| MimirDmDramUnary.ReaderWriterHandshake (per CCE: 1x1 Cce0-1, 2x Cce0-3) | MM-8-T2 Tier A dram_unary (fits MM-4) | A reader hart (GDDR to TL1 ring by SPA loads) and a writer hart (ring to GDDR by SPA stores) synchronised by two DRAM-core semaphores | Per-run PRNG source in GDDR. The destination and ring are pre-seeded with different patterns, with guards on both sides. Host readback through the GDDR view; expected ring contents. The host prefills the semaphores with 0x5E3A5E3A and checks they read 0 at start, sit at the runtime's address, and end at 4 | EXPECT codes, chunks==4, AtStart==0, semaphore addresses match, final semaphore==4, destination, ring, 4 guards. A host deadline gives ADD_FAILURE. Skips: mimir-local package, watcher, CCE not present (no longer reached on either descriptor). Runs on both | Checked: no semaphore init at launch (AtStart is the prefill: caught); wrong get_semaphore (address mismatch: caught); writer not waiting on filled (copies the seed pattern: caught deterministically); wrong SPA route or chiplet offset (destination keeps its prefill: caught). Not deterministically caught: a reader that never waits on drained before reusing a slot. It fails only if it wins the race; kStatusMaxWaitCycles is recorded but never compared | No | WEAK | Medium | test_mimir_dm_dram_unary.cpp:263-408; cce_dram_unary_reader.cpp:57-87; cce_dram_unary_writer.cpp:57-84 |
| MimirDmIdma.Copy (12 inst.) | MM-8-T3 Tier B quasar_idma (fits MM-4/5/6) | iDMA command-buffer copies: own TL1, max packet, sibling TL1, local GDDR0 and GDDR1, remote TL1 and GDDR over D2D, strided gather of 64 B and 8 B elements from TL1 and GDDR | Per-run PRNG source, destination poisoned per leg, guards before and after. Host readback through the GDDR view or host TL1 path. Strided cases compare against the host's gathered elements | EXPECT code==Landed or Acked, elements issued, data (with recheck), both guards. ASSERT remap after CreateDevice. A host deadline gives ADD_FAILURE. Skips: watcher; SiblingTl1 needs 2 CCEs per Mimir and LocalGddr1 needs 2 views (both descriptors have them, so both run); Remote* need 2 Mimirs (skip on 1x1, legitimate) and skip on a remote preflight mismatch (2x) | Checked: copy not issued or no ack (code: caught); wrong address or VC (poison or guard: caught); contiguous instead of gather (caught); overrun (caught). On 2x, an SS_BAR/ATT/interleave readback mismatch becomes a skip. Chippy programs those registers (mimir_hw_state.cpp:221), not Metal or UMD, so this is an external precondition (Q2) | No | OK | High | test_mimir_dm_idma.cpp:120-139,261-327,455-577; cce_idma_copy.cpp:39-56,838-948 |
| MimirDmMeshBufferDram.InterleavedPlacement | Tier B mesh_buffers (fits MM-0) | Host interleaved DRAM MeshBuffer puts page i in bank i at buffer.address() | Per-bank distinct PRNG pages (base address folded into the seed). Raw view read at the view the allocator lists for each bank, then a round trip | ASSERT banks>0, pages==banks, page size. EXPECT per-bank placement and round trip. Skips: watcher only. Runs on both (2 banks on 1x1, 4 on 2x) | Checked: page placed in the wrong bank or at the wrong offset (caught, pages distinct); missing view (caught). Not caught: a wrong view-to-GDDR translation in Cluster::write_dram_vec/read_dram_vec, because the write (channel = bank_id) and the raw read (channel = view) go through the same Cluster path (MM-0's job). The oracle ignores get_bank_offset; the interleaved write path also uses no offset (:156), so they agree today | No | WEAK (scope: shared translation) | Medium | test_mimir_dm_mesh_buffer.cpp:58-159; host_device_transfer.cpp:156,380-394; mimir_contract.hpp:368-418 |
| MimirDmMeshBufferDram.WriteRead | Tier B mesh_buffers (MM-0) | MeshBuffer write then read | Per-run iota base (gtest seed is time-derived) | EXPECT round trip. Skips: watcher only. Runs on both | Checked: write dropped (stale data differs per run: caught). A shared-path translation error is not caught, but upstream intent is only a round trip. With --gtest_random_seed=N pinned, stale data from a prior run with the same N would pass | No | OK | High | test_mimir_dm_mesh_buffer.cpp:161-182; mimir_patterns.hpp:71-78; gtest.cc:5815 |
| MimirDmMeshBufferDram.MultipleWriteReadRounds | Tier B mesh_buffers (MM-0) | 10 write/read rounds | iota(round*1000), different each round (not per run) | ASSERT each round. Skips: watcher only | Checked: dropped write (round 0 would read WriteRead's or the last run's data: caught); dropped write in a later round (reads the previous round: caught) | No | OK | High | test_mimir_dm_mesh_buffer.cpp:184-207 |
| MimirDmRuntimeArgs.IllegalTooManyRuntimeArgs (CCE0 only) | Tier B test_runtime_args (fits MM-3) | Combined unique and common RTA ceiling on a DRAM core | Boundary pair: NO_THROW at the ceiling, ANY_THROW at ceiling+1, both orders. Then a launch at the ceiling whose echo kernel returns the last 4 unique and last 4 common args into a poisoned slot | EXPECT_NO_THROW / EXPECT_ANY_THROW x2, EXPECT echo slot. No skip besides the package. Runs on both | Checked: no ceiling check (caught); off-by-one either way (caught by the pair); ceiling args not delivered (echo: caught). The "any error passes" risk is closed by the NO_THROW twin. Not caught: a wrong HAL KERNEL_CONFIG size, because the test's ceiling uses the same HAL value as kernel.cpp:886-890 | No | OK | High | test_mimir_kernel_load.cpp:125-154,384-451; cce_args_echo.cpp:9-37; kernel.cpp:866-895 |
| MimirDmRuntimeArgs.IllegallyModifyRuntimeArgs (CCE0 only) | Tier B test_runtime_args (MM-3) | Changing the unique arg count is rejected; a second SetCommonRuntimeArgs is rejected; stored args are unchanged | Echo before; ANY_THROW x2; GetRuntimeArgs and GetCommonRuntimeArgs equal the originals | EXPECT echo, ANY_THROW x2, EXPECT stored args | Checked: rejection missing (caught); rejected call corrupts args (caught). The only other throw path is ValidateLegacyRuntimeArgsAPI, which would also make the later Get calls throw. Not caught here: an implementation that rejects every post-launch SetRuntimeArgs, including same-count ones (no positive control; MM3 ArgsReplaceStale covers it). The common case is "set twice", not upstream's count change (6 vs 4) (Q3) | No | OK | High | test_mimir_kernel_load.cpp:459-489; tt_metal.cpp:169-183,994-1043; kernel.cpp:956-976; upstream api/test_runtime_args.cpp:749-797 |

##### 2. WEAK details

MimirDmDramUnary.ReaderWriterHandshake
- The reader waits on drained before reusing a slot (cce_dram_unary_reader.cpp:58-75). If that wait were removed, chunk 2 could overwrite slot 0 before the writer copies chunk 0.
- The test only notices if the reader wins that race on the emulator. Nothing deterministic gates it: the final counts (:382-383) and the ring contents (:404) are the same either way.
- kStatusMaxWaitCycles is recorded (:195-207) but never compared.
- The writer-side wait is caught deterministically, because the ring is pre-seeded (:273-280).

MimirDmCachePerf.WritePath
- The functional checks gate properly (:383-385).
- The plan's perf deliverable (Mimir cycle, bandwidth and flush-cost baselines, plan 4.1 perf) is not implemented. Cycles go only to log_info (:382), with no property, baseline or threshold.
- So this is a coverage gap against the plan, not a vacuous leg.

MimirDmMeshBufferDram.InterleavedPlacement
- The write (WriteToDeviceDRAMChannel(bank_index, ...), host_device_transfer.cpp:156) and the oracle (read_gddr_view, then ReadFromDRAMChannel(view), mimir_contract.hpp:394-418) both go through Cluster's channel translation.
- The test independently proves which page lands in which bank and at which address. It does not prove that channel N reaches its own GDDR tile.
- The oracle reads at buffer.address() with no get_bank_offset. It would fail on correct hardware if a descriptor ever had nonzero DRAM bank offsets (zero on both descriptors today).

Shared low-severity note (rule 9)
- 11 of the 12 tests (all but the two RuntimeArgs cases) call GTEST_SKIP when TT_METAL_WATCHER is set, via mimir_watcher_skip_reason() (mimir_hw_state.cpp:426-431).
- The skip reason is "SMC register reads", which none of these tests perform.
- The README says to leave the variable unset (README:19), so this is reached only by misconfiguration. Still, an environment variable silently turns every gate into a skip.

Other notes (not WEAK)
- HartToHartCoherent: its invalidate leg can't detect a broken invalidate, so don't count it as invalidate coverage.
- Addrgen/Im2col scope: these ports only pop addresses. The plan's "cover local DRAM, remote DRAM, remote CCE TL1 endpoints" is met only by MimirDmIdma Strided1D* (own TL1 and local GDDR). Remote addrgen endpoints are not ported (README:202).
- README is stale on DramUnary: README:285 says DramUnary Cce2/Cce3 skip on 1x1. At HEAD cce_indices() builds the list from the descriptor (mimir_contract.hpp:167-182), so 1x1 instantiates only Cce0 and Cce1 and nothing skips. The README run summary does not match HEAD.

##### 3. UNSURE / questions for the user

1. For MimirDmIm2col.Pattern/Im2Col_Dilation1: which document fixes the addrgen loop wrap ("restart at 0" vs "(index + stride) mod end") independently of emulator output? README:204 cites neosim overlay-addrgen-rtl-semantics.md and tensix-ip_docs overlay/rocc/address_generator.md. I could not find either under /proj_sw/user_dev/sagarwal. If the host model was fitted to what the emulator returned, a wrong RTL wrap would match it. The other 10 addrgen/im2col instantiations do not depend on this.
2. For the remote MimirDmIdma cases on mimir_2x_package: should a D2D setup readback mismatch (SS_BAR, ATT, interleave, which chippy programs per mimir_hw_state.cpp:221) skip or fail? Today it skips (test_mimir_dm_idma.cpp:132-137, :557-558), so a broken bring-up setup reads as "skipped", not "failed". If Metal or UMD is ever meant to own that setup, it becomes a rule-2 DEFECT for RemoteTl1, RemoteGddr and RemoteTl1KernelVerified.
3. For MimirDmRuntimeArgs.IllegallyModifyRuntimeArgs: is "SetCommonRuntimeArgs twice with the same count throws" the intended port? Upstream tests a common-arg count change (4 to 6) through the Metal 2.0 API (api/test_runtime_args.cpp:785-796). The legacy API rejects any second set (kernel.cpp:974-976), so the count rule itself is not exercised for common args.

##### 4. Tests that belong to another section
- MimirDmDramUnary.ReaderWriterHandshake: is also MM-4 work (CCE to local GDDR through TL1).
- MimirDmIdma.Copy: its Remote* instantiations are MM-5/MM-6 (D2D) work; the local ones are MM-4.
- MimirDmMeshBufferDram.*: are host-DRAM checks, which fit MM-0.
- MimirDmRuntimeArgs.*: fit MM-3 (they live in test_mimir_kernel_load.cpp next to the MM3 suite).
- MimirDmCache.Op, MimirDmCachePerf.WritePath, MimirDmScopedLockClm.CacheOps: fit Section 4.1.

Not checked: I did not compare emulator RTL against the cache semantics (whether L1 D$ is write-back, whether an L2 invalidate also invalidates L1). I inferred them from risc_common.h:186-187, cache.h:17-18 and the README's recorded passes. I did not read the rest of cce_cache_ops.cpp (the MM41 modes) or cce_idma_copy.cpp:145-837 (the MM4x modes), since my tests don't use them.

### B.10 4.1 cache (reviewer s41-cache)

#### Section 4.1 (DM cache) review: 21 tests

Counts: 9 OK, 4 WEAK, 8 DEFECT, 0 UNSURE. (Correction: my earlier chat message said "5 WEAK and 8 OK"; the table below has 9 OK and 4 WEAK.)

Read at HEAD 16585a9c68a. The descriptors exist only in the working-tree UMD (f0d497f33), not in the UMD commit HEAD records (292fe2086). mimir_1x1 has 1 Mimir and 2 CCEs; mimir_2x_package has 2 Mimirs and 4 CCEs. Both are `address_space: mimir_local` with programmable CCEs, so `skip_unless_programmable_dram_package` and `skip_unless_mimir_local_package` never fire on either. Suite prefix is `MM41`. Paths below: test: = tests/tt_metal/tt_metal/dram/test_mimir_dm_cache.cpp, test_scoped_lock: = dram/test_mimir_dm_scoped_lock.cpp, kernel = test_kernels/cce/cce_cache_ops.cpp, idma = test_kernels/cce/cce_idma_copy.cpp, producer/consumer = test_kernels/cce/cce_scoped_lock_{producer,consumer}.cpp.

**Claim in the source contradicted by Jira.** Every iDMA snoop note cites GRENSV-2290 (test:594-600). GRENSV-2290 is "Snooping - CCE MMIO access": remapper `cmd_snoop=1` on MMIO/SPA stores. It supports the `SiblingSpa` SnoopOn leg, not the iDMA snoop packet tag. The only evidence for the iDMA tag is chippy `cce_xcce_snoop_dma` / `cce_xcce_snoop_reader`, which cover a cross-CCE TL1-to-TL1 copy (CCE1 to CCE0) only.

##### 1. Per-test table

| Test | Plan case | Feature (a) | Oracle (b) | Pass/fail and skips (c) | Passes on a broken impl? (d) | Vacuous? (e) | Verdict | Conf. | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| MimirCacheProbe.SnoopConfig | MM41 probe (MM4x plan 339) feeding the 4.1 race sweep | Hart reads NOC_SNOOP_TL_MASTER_CFG and L2 CONFIGURATION | None; values go to RecordProperty | Gates only "kernel done". Every CCE of both descriptors. Watcher skip only | Yes: a wrong register address or garbage value passes | Yes for the property | DEFECT | high | test:494-528; kernel 1231-1233 |
| MimirCacheProbe.L2Geometry | MM41 probe (MM4x plan 340) | L2 eviction vs stride/count | None; mask and first count recorded | Gates only "done" per stride | Yes. Derived constants at test:585-589 (16 KiB, 24 reads) are hard-coded and never checked against it | Yes | DEFECT | high | test:530-575, 585-589 |
| MimirCacheOps.FullInvalidate | 4.1 L1 full invalidate, full hierarchy invalidate | Discard: `cdiscard.d.l1 x0` drops stores. Wipe: FULLINVALIDATE + discard makes a stale line fresh | Discard: uncached TL1 Matches==0 plus host old. Wipe: WarmOld, StaleBefore==64 (precondition), cached Matches==64, host new | FAIL on wipe timeout (977-983), else EXPECTs. Fresh-session skip (see section 2) | Discard acting as flush: caught. Discard no-op: caught only because the bare FLUSH64 probes L1 (risc_common.h:187 comment); without that, passes, and the only discriminator `wipe_stale_words_before_drop` (976) is record-only. FULLINVALIDATE no-op that reads 0: caught (L2 still stale) | No | OK (conditional on Q4) | med | test:916-992; kernel 300-347, 358-447, 455-488 |
| MimirCacheOps.FlushFenceOrdering | 4.1 flush/fence ordering | 4 legs: flush no fence, fence no flush, L1 then L2, full hierarchy | Consumer's uncached read at the flag; host readback after the final flush | Legs 0-1 gate only passes and the final host read. Legs 2-3 gate complete-at-flag. RegB ("never complete", data movement) gated for no leg | Legs 0-1: any fence/flush bug passes. L1ThenL2 and FullHierarchy: a no-op `cflush` passes since FLUSH64 probes L1. A no-op range flush is caught | Legs 0-1 yes | DEFECT | high | test:1006-1086 (1074-1077); kernel 490-567 |
| MimirCacheOps.DirtyEvictionPartialLine | 4.1 dirty eviction / partial-line writes | Byte/half/word/dword cached stores survive eviction and flush; untouched bytes kept | Host computes the image independently (1089-1104) and reads back; plus kernel uncached checks | Gates RegA>0 (some eviction), RegB==0, RegC==0, Matches==64, host | Whole-line write-back with stale bytes: caught. Dirty line dropped on eviction: caught (RegA=0, Matches). Byte-merge bug: caught | No | OK | high | test:1106-1159; kernel 569-656 |
| MimirCacheOps.L1DcacheFlushLine | 4.1 L1 line flush | `cflush.d.l1` per line, then discard and L2 range flush | Uncached Matches==64, UncachedBefore==0, host new | EXPECTs. Whether cflush alone reaches TL1 is record-only (1186-1191) | No-op cflush: discard drops the dirty line, TL1 stays old: caught, if L1 is write-back. If L1 were write-through it passes; write-back is shown only by FullInvalidate's discard leg | No | OK | med | test:1161-1201; kernel 658-674 |
| MimirCacheOps.L1DcacheFlushFull | 4.1 L1 full flush | Same with `cflush.d.l1 x0` | Same | Same | Same | No | OK | med | test:1203-1205 |
| MimirCacheSharedL2.SameLineSameOffset | 4.1 shared-L2 stress | WriterReader: cached write, fence, other hart reads. EightWriters: 8 harts store the same words | WriterReader: reader cached read after writer flush and reader L1+L2 invalidate, plus host. EightWriters: each word is "some hart's last value" | WriterReader fence-only read (RegB) is RecordProperty only (1311-1316). EightWriters host compare tautological for contested words (1385) | WriterReader: broken writer flush caught by host read. No-op reader invalidate passes if same-CCE L1s are coherent. Fence-only shared-L2 read can never fail | Fence-only leg yes | DEFECT | high | test:1249-1398; kernel 676-775 |
| MimirCacheSharedL2.SameLineDiffOffset | as above | Writer words 8-11 vs reader 0-3 on one line; 8 harts x 8 B in one line | As above; EightWriters host expects every hart's own words | As above | EightWriters: false-sharing write-back clobbering a neighbour's bytes: caught. WriterReader as above | Fence-only leg yes | DEFECT | high | test:1232, 1388-1396 |
| MimirCacheSharedL2.AdjacentLine | as above | Writer line 1, reader line 0; one line per hart | As above | As above | As above | Fence-only leg yes | DEFECT | high | test:1233 |
| MimirCacheWriterProbe.ExternalWriters | 4.1 external-writer coherence (3 modes) | Host, local iDMA, local-GDDR iDMA, sibling SPA, remote CCE, remote-GDDR iDMA writers vs a warm cached reader | Reader cached read (kernel). Host TL1 readback only shows the writer landed | 16 instances per CCE; on 1x1 RemoteCce/RemoteGddrIdma skip (5 per CCE). SnoopOffInvalidate gates Matches. SnoopOn gates WarmOld, StaleBefore==0, Matches. InvalidateOnly gates nothing about the reader | InvalidateOnly: no-op INVALIDATE64 passes. SnoopOffInvalidate: same-Program iDMA writers have no warm wait (1615-1618), so an early landing passes without exercising the invalidate. SnoopOn iDMA legs fail on emulator (README 424, 441): gates correctly if Q1 holds | InvalidateOnly instances yes | DEFECT | high | test:1585-1751 (1735-1749); kernel 777-814 |
| MimirCacheCoherence.CrossCce | 4.1 cross-CCE visibility | Leg 1: flush, sibling reads by SPA. Leg 2: sibling SPA write, then invalidate | Leg 1: producer uncached Matches, remote SPA Matches, host. Leg 2: StaleBefore==64 (non-snooped pinned, only when WarmOld==64), Matches after invalidate, host | ASSERT done; EXPECTs. Both descriptors (sibling exists on 1x1) | No-op flush: caught. No-op INVALIDATE64: caught. Write-through cache passes leg 1 (UncachedBefore not asserted). A coherent SPA path would fail leg 2 (pinned probe result) | No | OK | med-high | test:1757-1877; kernel 816-843, 1225-1249 |
| MimirCacheCoherence.CrossCceSnoopOn | 4.1 snoop-enabled cross-CCE | Sibling iDMA with snoop tag refreshes a warm cached copy | WarmOld, StaleBefore==0 (no invalidate), Matches; host. Writer acks without data poll (idma 364-366); reader polls data itself (kernel 788) | ASSERTs + EXPECTs; constexpr gate always true. Both descriptors | Snoop tag ignored: stale: caught. Mirrors chippy cce_xcce_snoop_dma | No | OK | high | test:1879-1953; idma 225-227 |
| MimirCacheCoherence.CrossD2d | 4.1 cross-D2D visibility | CrossCce to the same location on the other Mimir | As CrossCce | GTEST_SKIP on 1x1 (one Mimir; legitimate). 2x only | As CrossCce | No | OK | med | test:1955-1973 |
| MimirCacheCoherenceRace.DirtyLineRace | 4.1 dirty-line race, config sweep | External snooped write to a dirty line under evict/flush/invalidate, 8 snoop-master configs | Host reads the line after final FLUSH64: external words kept; hart words new (Neighbour, non-invalidate) | Only Cfg6 gates (2091; `continue` at 2178). Other 7 fail only on RaceUnresolved or writer not acked | Non-default configs: stale write-back overwriting external words after they landed passes. Same-target Flush/Invalidate: no re-dirty, line clean or gone before the host-launched iDMA lands. No check that Evict evicted | Cfg!=6 instances yes | DEFECT | high | test:1998-2191; kernel 845-885 |
| MimirCacheStress.ReuseThrash | 4.1 reuse/thrash | Seeded sequential, conflict-stride, random passes over 2-192 KiB with write-back | Kernel checks each pass vs thrash_word; uncached Matches and XOR after flush; host first/last line and guard | EXPECTs; cycles recorded (plan accepts proxies) | Lost dirty eviction or stale hit: caught. Seed varies per run | No | WEAK | med | test:2206-2299; kernel 887-960 |
| MimirCacheStress.PreloadChurn | 4.1 preload/churn | Sibling iDMA preload into the hot set; snoop-off + INVALIDATE64 vs snoop-on | Consumer cached reads per round (StaleBefore==0, Matches==4); host final round | Both legs gate. Snoop leg fails on emulator (README 427, 443): gates correctly if Q1 holds | No-op INVALIDATE64 in invalidate leg: caught. Hot set never accessed while the preload runs; preload is one 8 KiB packet into the hot set itself | No | WEAK | med | test:2301-2438; kernel 962-1005 |
| MimirCacheStress.PerfSweep | 4.1 perf port, functional sweep | 4 write paths at 64, 128, 192 KiB | Kernel uncached Matches, host first/last line, guard; pass 0 writes the complement | EXPECTs; cycles recorded | Lost stores or flush: caught. Stale TL1 differs per leg (fixed order, distinct values); only an interrupted earlier run could match | No | OK | med-high | test:2440-2490; kernel 123-160 |
| MimirCacheScopedLock.MultiConsumer | 4.1 multi-consumer scoped lock | Invalidate-on-acquire for 7 consumers; flush-on-release by producer | Per-consumer result lines (warm old, locked NEW cached, unlocked OLD cached, uncached), host probe words | ASSERT producer done; EXPECT per consumer. Both descriptors; no fresh-session skip | No-op release: producer data poll times out: caught. No-op acquire: likely passes (section 2) | Acquire: effectively | WEAK | med | test_scoped_lock:503-600; producer 162-208; tt_metal/hw/inc/internal/scoped_lock_cache_ops.h |
| MimirCacheScopedLock.RingWrap | 4.1 ring-wrap | Slot reuse after wrap reads the new round | Consumer kernel mismatch count (cached), round counts, host final slots | ASSERTs + EXPECTs | No-op release: caught. No-op acquire: masked by same-CCE coherence as above | Acquire: effectively | WEAK | med | test_scoped_lock:602-676; producer 209-257; consumer 110-141 |
| MimirCacheScopedLock.NonSnoopingProducer | 4.1 non-snooping producer | Sibling-CCE SPA producer (not snooped); consumer acquire makes locked lines fresh | Consumer waits for TL1 data uncached, acquires, reads cached; locked NEW, unlocked OLD (pinned non-snooped) | ASSERTs + EXPECTs; both descriptors | No-op acquire: OLD: caught | No | OK | med-high | test_scoped_lock:678-749; consumer 172-201; producer 258-279 |

##### 2. Notes on each DEFECT and WEAK

**SnoopConfig (DEFECT).** Only "done" is asserted (test:505-508); both registers go to RecordProperty (526-527). A wrong register address or a non-0x6 default passes. DirtyLineRace's host-vs-hart compare (2060) is the only consistency check.

**L2Geometry (DEFECT).** Only "done" is asserted (560-563); eviction results recorded (571, 574). The 16 KiB stride and 24-read constants (585-589) are frozen from one run (README 317) and nothing re-checks them. DirtyLineRace's Evict action never confirms an eviction happened.

**FlushFenceOrdering (DEFECT).** Legs 0-1 are not gated on ordering or data (1074-1077 cover legs 2-3 only), and RegB is never asserted, so the plan's "data movement and ordering checked independently" is not met. Kernel 509-518: legs 2-3 each end in FLUSH64; per risc_common.h:187 FLUSH64 probes L1, so a no-op `cflush.d.l1` (per line or x0) passes both gated legs. The plan's rdcycle ordering check was dropped (comment at 1008-1010).

**MimirCacheSharedL2, all 3 (DEFECT).** In WriterReader, the read the plan asks for ("another reads it after the required fence") is RegB, record-only (1311-1316). The gated read (1321) follows the writer's range flush and drain and the reader's L1+L2 invalidate (kernel 690-693, 741), so it is a TL1 round trip, not a shared-L2 check. In EightWriters SameLineSameOffset, `expected[word] = observed[word]` (1385) makes the host compare of words 0-3 always true; only the "some hart's last value" check (1383) can fail.

**ExternalWriters (DEFECT).** InvalidateOnly: 1735-1749 gate only SnoopOffInvalidate and SnoopOn; `expect_host_on_cce` (1750) reads TL1 and proves only that the writer landed. That is 6 of 16 instances per CCE (4 on 1x1). README 407 already says INVALIDATE64 alone gives a fresh read, so the expected outcome is known but not asserted. SnoopOffInvalidate has no WarmOld gate; LocalIdma, LocalGddrIdma, RemoteGddrIdma start in the same Program as the reader (1673), so an early landing passes trivially. `mimir_test_seed() & 0x7FF00000u` (1608) is always 0: gtest normalizes its seed to [1, 99999] (.cpmcache/gtest/96129d89.../googletest/src/gtest-internal-inl.h:83, 112-124), so the value is run-invariant despite record_leg saying it comes from the seed; host prefill and kernel line drop limit the effect.

**DirtyLineRace (DEFECT).** `gates = config == kSnoopMasterDefault && ...` (2091) and `if (!gates) continue;` (2178) skip every data-preservation EXPECT for configs 0-5 and 7. Chippy `cce_xcce_snoop_dma_cache_invalidate` programs 0x2 as a working config, so at least Cfg2 has a known expectation. Same-target legs never re-dirty (2107; kernel 870-873): the first FLUSH64/INVALIDATE64 cleans or drops the line long before the iDMA, which the host launches only after seeing the flag (2124-2130), so those legs are not a dirty-line race.

**ReuseThrash (WEAK, minor).** ConflictStride at 2, 4, 8 KiB (below the 16 KiB stride) has stride_lines (256) > num_lines, so the order is plain sequential (kernel 895-901); those 3 legs duplicate Sequential under another name. The plan's "repeatedly access one cache line" is not generated.

**PreloadChurn (WEAK).** The plan asks for a hot set kept active while a large preload runs, with hot-set eviction measured. Here the preload is one 8 KiB packet (2302, 2370) into the hot set itself, and while it runs the consumer polls only through the uncached alias (kernel 977-984). Data correctness is gated; churn is not tested.

**MultiConsumer and RingWrap (WEAK).** Acquire is INVALIDATE64 of the range only (scoped_lock_cache_ops.h). The producer stores through the cache on the same CCE (producer 186-190, 235-239). MM-8 HartToHartCoherent asserts that a consumer's cached read sees a flushed store before its own invalidate (test:248) and that case passes (README 402, 420), so a no-op acquire most likely still reads NEW. "Unlocked lines OLD" in MultiConsumer is trivially true: the producer never writes unlocked lines (producer 187-189). Only NonSnoopingProducer makes the acquire observable.

**Skip path common to MM41 fixtures.** `fresh_session_skip_reason` (765-769) makes every later MM41 cache case in the process GTEST_SKIP on either descriptor after one FullInvalidate wipe timeout (978) or DirtyLineRace RaceUnresolved (2169). ScopedLock and probe tests are not affected. Harmless under the README's one-case-per-invocation order; a batch run hides those cases as skips.

##### 3. UNSURE / questions for the user

1. Should the iDMA snoop packet tag refresh a CCE's cached copy for its own TL1-to-TL1 copy, for GDDR-sourced copies on VC 5 (local and remote), and for the PreloadChurn preload, as `kIdmaSnoopBitRefreshesCache` assumes? GRENSV-2290 covers MMIO/remap snoop; chippy covers only the cross-CCE TL1-to-TL1 iDMA. If yes, the ExternalWriters iDMA SnoopOn legs and PreloadChurn snoop leg fail on the emulator but gate correctly; if no, they are false failures.
2. Should DirtyLineRace assert data preservation for the 7 non-0x6 configs, at least 0x2 (which chippy uses)? If configs with `disable_inline_writes=1` are unsupported, should they be removed rather than recorded?
3. Does the Mimir spec say same-CCE DM harts are hardware-coherent (L1s kept coherent through L2)? If yes, MultiConsumer and RingWrap need an uncached or other-CCE producer to show invalidate-on-acquire, and SharedL2 WriterReader's fence-only read has a known expected value that could be gated.
4. Does FLUSH64 probe the issuing hart's L1 D$ (and other harts' L1s) for dirty data? The only source is the comment at tt_metal/hw/inc/internal/tt-2xx/risc_common.h:187. FullInvalidate's discard leg and FlushFenceOrdering's L1 legs depend on it.
5. Should CrossCce and CrossD2d also run with local and remote DRAM backing, as the plan says ("repeat with local and remote DRAM backing")? That would require CCE harts to cache GDDR addresses.

##### 4. Tests that may belong to another section

- MimirCacheCoherence.CrossD2d and the ExternalWriters RemoteCce / RemoteGddrIdma instances cross D2D (MM-6 territory); judged here as 4.1 cache cases.

##### Not checked

`experimental::cce_sram_address` (whether SPA loads/stores bypass the caches), `run_cce_kernels_with_deadline`, `launch_on_idle_cces`, `wait_for_status_magics`, `cce_test::poll_bounded`, cce_idma_copy.cpp past line 440, the UMD register constants behind mimir_snoop_ctl.cpp, the Mimir spec and bring-up guide HTML. README used only for recorded run outcomes, not as hardware evidence.

##### Files read

- /localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/dram/test_mimir_dm_cache.cpp
- /localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/dram/test_mimir_dm_scoped_lock.cpp
- /localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/test_kernels/cce/cce_cache_ops.cpp and cce_cache_ops_abi.h
- /localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/test_kernels/cce/cce_scoped_lock_producer.cpp, cce_scoped_lock_consumer.cpp, cce_scoped_lock_abi.h
- /localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/test_kernels/cce/cce_idma_copy.cpp (1-440)
- /localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/dram/mimir_snoop_ctl.hpp and .cpp
- /localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/dram/README.md (305-460)
- /localdev/sagarwal/tt-metal/tt_metal/hw/inc/internal/tt-2xx/risc_common.h, internal/scoped_lock_cache_ops.h, internal/tt-2xx/quasar/cache.h
- /localdev/sagarwal/tt-metal/tt_metal/third_party/umd/tests/soc_descs/mimir_1x1.yaml and mimir_2x_package.yaml
- /proj_sw/user_dev/sagarwal/chippy/validation/firmware/apps/ip/cce/cce_xcce_snoop_{dma,dma_cache_invalidate,reader}/src/main.c
- ~/.claude/plans/the-changes-in-this-cozy-lollipop.md (360-400, 495-510)
- GRENSV-2290 (Jira)

### B.11 4.2 atomics (reviewer s42-atomics)

#### Section 4.2 (atomics) review: 22 tests in test_mimir_atomics.cpp

Result: 5 DEFECT, 10 WEAK, 7 OK, 0 UNSURE rows (4 questions for the user in section 3).
Source read at HEAD 16585a9c68a in /localdev/sagarwal/tt-metal; UMD submodule as checked out in the working tree (f0d497f3, marked modified in git status).

Path abbreviations: A = /localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/dram/test_mimir_atomics.cpp, K = .../test_kernels/cce/cce_amo_ops.cpp, KH = .../cce_amo_ops_abi.h, I = .../test_kernels/cce/cce_idma_copy.cpp, IH = .../cce_idma_copy_abi.h.

Facts that apply to every row:
- Descriptors: every test is a TEST_P over cce_indices(): Cce0-1 on mimir_1x1, Cce0-3 on mimir_2x_package (DRAM grid y = 2 per channel, UMD device/src/soc_descriptor.cpp:129). sibling_cce is the other CCE on the same Mimir on both descriptors.
- Skips: none reachable on either descriptor except environment skips in SetUp (A:687-692): no programmable DRAM, no transport, no slow dispatch, or watcher enabled. The probe-gated skips (A:165, 939-943, 1094, 1194, 1635, 1651, 1704) are dead code because kTwoHartAmoPassed and kEightHartLaunchPassed are constexpr true (A:59-61). If either were set false, run_gated_sweep_leg (A:938-945) would silently record "skipped" and pass (failure mode 9, latent).
- Seed: gtest random_seed() is time-based unless the flag is set (.cpmcache gtest-internal-inl.h:112-115), so patterns differ per run.
- Independent oracle: host readback uses read_cce_tl1, a different path from the kernel's cached or iDMA path. Kernels flush L1 D$ and L2 before the host reads.
- Run status (README, author's account): all 4.2 rows passed on 1x1 (2026-10-23). On 2x only rows 110-112 ran; rows 126-130 (FmtOpMatrix, DirtyLine, IdmaAccum, MaxOutstanding, detector) have not run on 2x.
- Emulator gaps: nothing in 4.2 falls under "fails on emulator, gates correctly".

##### 1. Table

| Test | Plan case | Feature (a) | Oracle (b) | Pass/fail and skips (c) | Passes on a broken impl? (d) | Vacuous? (e) | Verdict | Conf. | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| MimirAtomicProbe.TwoHartAmo | 4.2 Tier A, cross-hart AMO probe | 2-hart amoadd.w and LR/SC on shared cached words | Host model of total and return sum; host readback after flush | EXPECT code done, kernel totals, return sums, host view; no test skip | Lost update with overlap: caught. Non-atomic RMW when harts do not overlap: passes. README:318 says 0 SC failures, "may barely have overlapped" | No | WEAK | High | A:762-789; K:446-498 |
| MimirAtomicLocal.OpMatrix | Tier A op/width matrix | Every amo*.w/.d, LR/SC CAS (match and fail), LR/SC and CAS adds, aq/rl adds | Host op_result model; returned vs old; host readback of every line | EXPECT returned, final value, whole region | Wide write clobbering upper word: caught. CAS fail storing: caught. amomin/max implemented unsigned (or reverse): missed when the one random sample has equal signs, about 50% per op per run. aq/rl only checked as plain adds | No | WEAK | High | A:997-1055; K:500-545 |
| MimirAtomicLocal.Hotspot | Local RISC-V hotspot (atomic_semaphore_bandwidth sweep) | amoadd.w on one word, 1/2/8 senders, 14 points | Host totals, return sums, untouched words | EXPECT per point; interleave count recorded only | Lost update: caught. Non-atomic update when senders do not overlap: passes ("contention not exercised" is a string) | No | WEAK | High | A:919-934, 1061-1090 |
| MimirAtomicLocal.Distributed | Distributed contention | Own word per hart: adjacent words, doublewords, separate lines | Host per-word totals, per-participant return sums, untouched words | EXPECT; no skip | Aliasing or address-map errors: caught. False-sharing clobber needs overlap, not checked | No | OK | Med | A:1093-1134, 896-902 |
| MimirAtomicLocal.LrScCas | Tier A LR/SC and CAS increments | LR/SC and CAS adds, .w/.d, 1 and 8 harts | Host totals and return sums | EXPECT; SC failures recorded | SC that always fails: caught (retries exhausted). SC succeeding without reservation: caught only if harts overlap, unverified | No | WEAK | High | A:1138-1164; K:186-257 |
| MimirAtomicLocal.Wrap | Wrap behaviour | Counters start below 2^32 / 2^64 and wrap | Host masked totals and return sums; neighbour word untouched | EXPECT | Carry into next word: caught. Wrong width: caught | No | OK | High | A:842-845, 1167-1188 |
| MimirAtomicLocal.Ordering | Ordering and visibility (rl/aq) | Message passing with amoswap.w.rl and amoadd.w.aq | Kernel consumer mismatch count (no host oracle possible) | flush_release: EXPECT mismatches == 0. release_only: recorded only | release_only cannot fail on its property. flush_release passes with broken .rl/.aq, since flush_l1_dcache and invalidate_l1_dcache each end in a full fence | No (one leg is) | DEFECT | High | A:1230-1234; K:719-753; tt_metal/hw/inc/internal/tt-2xx/risc_common.h:206-224 |
| MimirAtomicLocal.CleanDirtyLine | Clean/dirty, flush-before, invalidate modes | AMO on clean, dirty, flushed lines; after host write with and without invalidate | Host model; host readback | Legs 1-3 EXPECT returned and line. Invalidate leg EXPECT. no_invalidate leg recorded only | Legs 1-3 OK. Invalidate leg: broken INVALIDATE64/cdiscard passes if the line was not still cached (control word recorded only). no_invalidate leg cannot fail | No (one leg is) | DEFECT | High | A:1288-1292, 1315-1344 |
| MimirAtomicLocal.CleanDirtyLineSnoopedWrite | Snoop-enabled external write before AMO | Sibling iDMA with snoop tag refreshes the hart's cached line | Returned == iDMA value; uncached TL1 check | ASSERT writer acked; EXPECT TL1 and returned (constant kIdmaSnoopRefreshesHartCache = true) | Snoop tag ignored but line evicted or not held: passes. control_held recorded only | No | WEAK | High | A:65, 1398-1418 |
| MimirAtomicLocal.Storm | Mixed-operation storm | 11 AMO kinds x 2 widths, shared and disjoint | Host replay model (classes commutative), swap-token set, XOR/sum balance, per-participant monotonic returns, untouched bytes | EXPECT progress, violations, conservation, all words | Lost/duplicate update with overlap: caught. Shared leg overlap not gated | No | WEAK | High | A:1573-1629; KH:154-206 |
| MimirAtomicLocal.NeighbourWriteRace | Atomic plus ordinary-write race (partial) | Hart 0 amoadd.w on word 0 while harts 1-7 store neighbour words | Host model; host readback word by word | EXPECT return sum, final value, every word | Whole-line writeback clobber: caught only if the race happened, not evidenced. Plan's NoC/DMA/snoop sweeps absent | No | WEAK | Med | A:1684-1698; K:758-788 |
| MimirAtomicLocal.Soak | Long randomized soak | Seeded storm, random harts (2-8), layout, bursts, barriers | As Storm plus equal barrier counts | EXPECT | As Storm. Only 1024 ops per hart. Barrier-count check redundant (same schedule by construction, KH:209-211) | No | WEAK | Med | A:1703-1717, 1591-1597 |
| MimirAtomicItnControl.Cce0ToCce1PlainWrite | Tier A CCE0->CCE1 local-path control | Sibling plain SPA (ITN) write, then receiver invalidate and AMO | Returned == written; host readback written+7 | EXPECT pre-read, returned, line | Write lost or misrouted: caught (independent host readback). Invalidate exercised only if line still cached (recorded only); the feature itself is gated | No | OK | Med | A:1766-1780; K:931-986 |
| MimirAccumProbe.FmtOpMatrix | Accumulate format/op matrix | iDMA L1 accumulate per format and op | Host cce-dv model, but no host compare; index 0 relies on kernel XOR+first-word poll | Index 0: EXPECT landed. Every point must ack (ADD_FAILURE). Other results recorded | INT8/UINT8/UINT32 broken (e.g. treated as INT32): passes. Index 0 trusts kernel self-check while observed (A:1903) only goes into a string | Mostly | DEFECT | High | A:1903-1931 |
| MimirAccumProbe.DirtyLine | Clean/dirty line for accumulate | Accumulate onto a dirty cached line | Kernel poll (result landed in TL1 before flush) | EXPECT landed only | Dirty writeback overwriting the result: passes (recorded only) | Yes for its property | DEFECT | High | A:1966-1978; I:420-496 |
| MimirAtomicIdmaAccum.SingleHart | Tier M accumulate | 4x256 B INT32 accumulates onto one block | Host model; host readback with guards | EXPECT landed, destination, both guards | Copy instead of add, dropped command, guard overrun: caught. Format semantics never exercised (see 2.7) | No | OK | High | A:2007-2032, 642-660 |
| MimirAtomicIdmaAccum.AllHartsShared | Tier M all-sender contention | 8 harts accumulate onto a shared block | Host total; host readback | EXPECT every hart landed, destination, guards | Lost contribution under overlap: caught. Overlap not evidenced or gated | No | WEAK | Med | A:2038-2079 |
| MimirAtomicIdmaAccum.AllHartsDisjoint | Tier M distributed | 8 harts, own blocks | Host model per block; region readback | EXPECT | Misaddressed block or neighbour clobber: caught | No | OK | High | A:2084-2131 |
| MimirAtomicIdmaAccum.BackToBack | Back-to-back burst pressure | Bursts of 1/8/64 on cmdbuf 0 and 1 | Host model; readback per leg | EXPECT per leg | Dropped/repeated command, cmdbuf 1 broken: caught | No | OK | High | A:2135-2169 |
| MimirAtomicIdmaAccum.MaxOutstanding | Exhaustion: max outstanding | 64 posted accumulates over trids 0-15 | Host model; readback | EXPECT landed and destination; pending_high_water recorded | Correctness caught. Depth never asserted; one-at-a-time implementation passes | No | WEAK | High | A:2173-2208; I:442 |
| MimirAtomicIdmaAccum.MixedPathControl | Plan's mixed-path control, redesigned (see section 4) | Sequential AMO -> accumulate -> invalidate + AMO on one line | Host model; returned values; readback | EXPECT each step and combined line | Accumulate not landing, stale second AMO: caught | No | OK | Med | A:2213-2256 |
| MimirKnownIssue.Mim560Detector | MIM-560 mixed concurrent detector | AMO, accumulate, uncached partial stores vs ITN stream on one line | Corruption classifier (recorded) | ASSERT only detector started and status magics; corruption, kernel codes, kCodeTrafficNotSeen all recorded | Any corruption passes | Yes | DEFECT | High | A:2357-2485 |

##### 2. Detail on each DEFECT or WEAK

1. Ordering (DEFECT). release_only result is record_leg only; EXPECT exists only for flush_release (A:1232-1234). The gating leg cannot detect broken .rl/.aq: producer's flush_l1_dcache (risc_common.h:206-212) and consumer's invalidate_l1_dcache (:217-224) each issue a full fence, so plain amoswap/amoadd would order the same way (K:724-737).
2. CleanDirtyLine (DEFECT) and CleanDirtyLineSnoopedWrite (WEAK). The no_invalidate leg only records the host view (A:1338-1344). In the invalidate leg (A:1325-1337) and SnoopedWrite (A:1401-1418), whether the line was still cached is computed and printed ("inconclusive") but never gated. If INVALIDATE64/cdiscard is a no-op, or the snoop tag is ignored, and the line was evicted between the warm launch and the AMO launch, the AMO fetches fresh data and passes. drisc.cc does no data-cache maintenance between launches (only L2 invalidate of the kernel binary and I$, tt_metal/hw/firmware/src/tt-2xx/drisc.cc:52-58, 128-129), so the precondition usually holds, consistent with the README showing green. Fix: fail when control_held is false.
3. FmtOpMatrix (DEFECT). Points 1-7 (UINT32, INT8, UINT8 at both saturation settings, all modeled) are never compared with EXPECT; a kCodeDataTimeout there continues silently since A:1917 checks only the ack flag. Point 0 is gated only by the kernel's XOR-plus-first-word poll (I:329-346), though the host already reads observed at A:1903.
4. DirtyLine (DEFECT). The named property (whether the dirty writeback replaced the result, kStatusProbeFirst) is recorded only (A:1966-1976). The only gate is kCodeLanded (A:1978).
5. Mim560Detector (DEFECT). Non-gating by design (A:9, A:2263-2269; MM4x plan says non-gating). No EXPECT on corruption counts or the three kernel codes (A:2371-2379, 2479-2485). Under the MM42Detector prefix, so MM42/* does not run it.
6. Contention not verified (WEAK): TwoHartAmo, Hotspot, LrScCas, Storm, Soak, AllHartsShared, NeighbourWriteRace. Correctness gates are sound: any lost or duplicate update that happens fails. None requires that participants overlapped: interleave counts (A:919-930, 1581-1590) are recorded strings, and AllHartsShared and NeighbourWriteRace have no overlap measure. A RMW that is non-atomic only under contention passes when the race does not happen; README:318 already reports 0 SC failures for TwoHartAmo. Soak also runs only 1024 ops per hart.
7. Accumulate format semantics never gated. Every functional Tier M row masks operands to 0x000FFFFF (A:54), so sign-magnitude, saturation and lane width are never exercised; a plain two's-complement 32-bit add passes SingleHart, AllHarts*, BackToBack, MaxOutstanding and MixedPathControl. Only FmtOpMatrix covers them, and it records only. The host model is ported from the same cce-dv reference the emulator checks against (A:500-505), so on the emulator model and device share a source.
8. OpMatrix (WEAK). One random operand per op (A:993-994): a signed/unsigned swap in amomin/amomax is missed whenever old and operand have equal signs, about 50% per op per run. aq/rl variants checked only functionally (A:981-984).
9. MaxOutstanding (WEAK). pending_high_water and vc_space_cycles are RecordProperty only (A:2203-2204). README row 118 notes trid counts clear on accept (MIM-628), so depth reached may be about 1 and the test still passes.

##### 3. UNSURE / questions for the user

1. The brief's rule makes the records-only legs DEFECT: Ordering release_only, CleanDirtyLine no_invalidate, FmtOpMatrix points 1-27, DirtyLine, Mim560Detector. The author labelled these probes or non-gating on purpose (MM4x plan agrees). Accept them as intentional probes, or add a gate to each? For FmtOpMatrix points 1-7 the host already has a model, so the gate is easy.
2. SnoopedWrite (A:1416-1417) cites GRENSV-2290. That Jira is about MMIO-write snooping through the SMC address remapper with cmd_snoop=1, not the iDMA snoop packet tag; the iDMA expectation rests on chippy cce_xcce_snoop_dma (/proj_sw/user_dev/sagarwal/chippy/validation/firmware/apps/ip/cce/cce_xcce_snoop_dma). Is that chippy test the authority for kIdmaSnoopRefreshesHartCache = true? README rows 99, 100, 109 show own-CCE and GDDR iDMA snoop staying stale, while row 112 reads fresh.
3. In the 1x1 run where row 112 "read fresh", did the logged control field say "still cached"? If "not held", that pass did not exercise the snoop. The run logs (/localdev/sagarwal/mm4x_runs_*) are not present, so I could not check.
4. The whole suite skips when watcher is enabled (A:153-156) with the reason "SMC register reads"; I found none in this file. Is that skip intended for 4.2?

##### 4. Placement and plan divergence

- All 22 tests belong to 4.2. MimirAtomicIdmaAccum.* and MimirAccumProbe.* are "Tier M" (iDMA accumulate), a reclassification from the MM4x plan (~/.claude/plans/the-changes-in-this-cozy-lollipop.md around line 57), not a tier in the Glean plan.
- MixedPathControl is not the Glean plan's mixed-path control. The plan wants a concurrent local AMO and remote NoC atomic on one word to document non-atomicity; this test is a sequential handoff. Closest match to the plan's intent is the non-gating Mim560Detector.
- Tiers B to E have no tests. B (remote NoC atomics) is n/a per the MM4x plan (Mimir has no NoC atomics). C (self-targeted NoC) was reclassified. D (cross-D2D atomics) and E (interop) have no test, and nothing records them as unsupported, which the plan's acceptance criteria ask for.

Not checked: the emulator model (emu_l1_atomic.cpp) against the host accumulate model; whether L1 D$ is coherent between harts; the bring-up guide beyond one line noting a snoop master watches ITN writes into SRAM.

### B.12 4.3 transaction IDs (reviewer s43-trid)

#### Section 4.3 (transaction IDs) review: 21 tests, OK 0 / WEAK 14 / DEFECT 7 / UNSURE 0

The data gates are real everywhere; nothing in the file gates the transaction-ID behaviour on its own terms.

Read: HEAD 16585a9c68a, tests/tt_metal/tt_metal/dram/test_mimir_trid.cpp (all 2207 lines), test_kernels/cce/cce_idma_copy.cpp and cce_idma_copy_abi.h (all), the helpers cited below, and Jira MIM-539 and MIM-628. Paths below are relative to tests/tt_metal/tt_metal/ unless they start with "umd" or a top-level directory.

##### Premise contradicted by source

- The file header (trid.cpp:7-8) and the MM4x plan (lines 430-432, 591) cite MIM-628 ("an ack does not prove the data landed") as the reason the barrier cases record staleness instead of gating it.
  - MIM-628 is about the iDMA *sideband* done_id path. Its DV note says: "The command-buffer (RoCC) path is not measured; by the RTL it uses the real response."
  - Jason Dubrow's comments (2026-10-14 and 2026-10-26) say "the cmdbuf path uses the trid and counts and would be waiting for responses". The ticket was closed as Future Fix, sideband only.
  - The kernel uses only the cmdbuf path (pending_count -> __builtin_riscv_ttrocc_cmdbuf_idma_tr_ack_trid, cce_idma_copy.cpp:165-168).
  - So "trid reads 0 means the data landed" is an expected contract the suite could gate, and does not.
- The Width comment (trid.cpp:84-85) says MIM-539 cannot be told apart from field truncation. The README's own probe result contradicts that premise (README:319): TR_ID read back 17 (bit 4 kept) while the count landed on trid 1.

##### Suite-wide findings (rows cite them as S1-S4)

- **S1. The trace check is not an independent gate.**
  - The rank-uniqueness assert (trid.cpp:898-900) cannot fail, because the kernel hands out rank++ once per trid (cce_idma_copy.cpp:308-311).
  - kTraceNotAcked (trid.cpp:896) is set in the same condition that yields kCodeAckTimeout (cce_idma_copy.cpp:319-323), so it repeats the code check.
  - kTraceNotUsed for unused trids is the kernel's own initial value (:297-300).
- **S2. Trid tagging and aliasing cannot be detected.**
  - Broken implementations checked: (a) TR_ID ignored, so every command counts on trid 0; (b) trid bits aliased, e.g. bit 3 dropped; (c) a counter that never increments.
  - In each case wait_for_trids reads 0 for the untracked trids at once (cce_idma_copy.cpp:305-316). The data still lands, the status is success and the trace is consistent.
  - Nothing asserts pending(trid_i) > 0 for each distinct trid. kStatusPendingHighWater is a max over all trids (:283-285, :442, :621).
- **S3. Early acks are masked.**
  - Non-host-verified legs wait for 0 pending, then poll the data until it matches (cce_idma_copy.cpp:362-391, :637-650).
  - Host-verified legs are read by the host only after the status magic arrives and WaitProgramDone returns (mimir_dm_common.hpp:107-116).
  - A per-trid ack that fires before the data lands therefore passes every gated leg. The one leg that could expose it (ReadAfterWrite's dependent read) only records.
- **S4. Constants turn gates into records (failure mode 9).**
  - kPendingVisibleHighWater = 12 (trid.cpp:77) is the only thing keeping the NominalDepth (:1412) and Saturate (:1452, :1494) depth asserts live; setting it to 0 turns them into records.
  - kEightHartLaunchPasses = true (:79) guards a bare skip (:1609-1611).
  - kTridCountersPerHart = false (:74) is hard-coded from a recorded probe result.
  - All are currently gating, but none is derived at run time.

##### Descriptor coverage

- Probe and Idma cases run on mimir_1x1 (2 CCEs) and mimir_2x_package (4 CCEs).
- Both descriptors are mimir_local (umd tests/soc_descs/*.yaml, address_space), so skip_unless_mimir_local_package is never taken.
- On 2x, MixedTargets, NominalDepth, Saturate, Fairness and Soak skip entirely, local legs included, when the D2D preflight mismatches (remote_legs_skip_reason, trid.cpp:997-1002).
- Remote and Concurrent cases run on 2x only.
- Reset always skips unless TT_METAL_FORCE_REINIT=1 is set (:1862-1864).
- The seed is gtest's random_seed, which is time-based unless --gtest_random_seed is fixed (.cpmcache gtest gtest.cc:5815, gtest-internal-inl.h:112-124), so patterns differ per run.

##### 1. Table

| Test | Plan case | Feature (a) | Oracle (b) | Pass/fail and skips (c) | Passes on a broken impl? (d) | Vacuous? (e) | Verdict | Conf | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| MimirTridProbe.PendingVisible | 4.3 width/range (pending visibility) | Pending count visible on one trid during 16 x 8 KiB | The copy only: kernel data poll plus host samples of 2 lines | EXPECT code, count and sample; high water only in RecordProperty | Yes: a dead counter (always 0) passes | Yes, for the property | DEFECT | high | trid.cpp:1052-1060, 279-300 |
| MimirTridProbe.DepthBeyond16 | 4.3 max outstanding | More than 16 outstanding over 16 trids | Same as PendingVisible | Landing gated; VC stall and high water recorded ("Non-gating") | Yes: any depth or stall behaviour passes; README:322 says at most 1 pending per trid, so depth > 16 never happens | Yes, for the property | DEFECT | high | trid.cpp:1063-1081; README:322 |
| MimirTridProbe.Width | 4.3 width (MIM-539) | 4-bit vs 5-bit TR_ID | None for the property | RecordProperty only ("Records only") | Yes: any width or alias result passes | Yes | DEFECT | high | trid.cpp:1084-1116 |
| MimirTridProbe.Scope | 4.3 counter scope | Per-hart vs shared counters | Issuer and observer pending reads | Codes and sample gated; scope recorded, with an "inconclusive" path | Yes: either answer passes | Yes | DEFECT | high | trid.cpp:1172-1179 |
| MimirTridIdma.UniqueIdsOwnTl1 | 4.3 unique IDs, local TL1 | Trids 0-15 per command into own TL1 (VC 5) | Host compare of the full model plus guards; host path (UMD read_core) independent of the kernel path (iDMA SPA) | EXPECT code, count, trace and data; no skips on either descriptor | Yes for S2 (a)-(c) and S3; data bugs are caught | No | WEAK | high | trid.cpp:1218-1253, 762-789 |
| MimirTridIdma.UniqueIdsSiblingTl1 | 4.3 unique IDs, Mimir0.CCE0 -> CCE1 | To the sibling (VC 2) and back (VC 5), 2 independent legs | As above; each leg re-seeded | As above | S2 and S3: yes | No | WEAK | high | trid.cpp:1256-1265 |
| MimirTridIdma.UniqueIdsLocalGddr | 4.3 unique IDs, local DRAM | 4 legs: write GDDR0 and GDDR1, read each back | Host reads via WriteToDRAMChannel / ReadFromDRAMChannel | As above | S2 and S3: yes | No | WEAK | high | trid.cpp:1269-1281 |
| MimirTridIdma.MixedTargets | 4.3 unique IDs, all targets | 7 local table entries (plus 2 remote on 2x), one trid each | Host model for every written region | As above; on 2x a preflight skip drops the local entries too | S2 and S3: yes | No | WEAK | high | trid.cpp:1288-1315 |
| MimirTridIdma.ReuseWrap | 4.3 reuse and wrap | 64 commands over 16 trids: a write, then a read with stride 5 | Host model | As above | S2 and S3: yes. "Reuse while pending" (comment :1320) is never shown: 256 B commands finish first (README:322) | No | WEAK | med | trid.cpp:1320-1348 |
| MimirTridIdma.DuplicateId | 4.3 duplicate ID | 16 commands on one trid (trid 9 to GDDR0, trid 15 to sibling) | Host model | Landing gated; high water only in RecordProperty | Yes: a counter that tracks 1 command instead of 16 passes (S3) | No | WEAK | high | trid.cpp:1351-1375 |
| MimirTridIdma.NominalDepth | 4.3 nominal depth | 16 trids outstanding back to back | Host model plus high_water >= 1 | EXPECT_GE(hw, 1) plus the above; 2x preflight skip | A dead counter is caught. Aliasing (S2a) and serialised issue (1 outstanding) pass. The kernel's total pending (probe_second, cce_idma_copy.cpp:622-628) is never read | No | WEAK | high | trid.cpp:1381-1415 |
| MimirTridIdma.Saturate | 4.3 saturate pools | 32 x 8 KiB per VC on one trid, then all VCs combined | Host model plus hw >= 2 per leg and total >= 2 combined | EXPECT_GE gated (see S4); 2x preflight skip | Catches a dead counter, serialised issue, and TR_ID ignored for trids 1, 2, 3 and 5. Saturation itself is not asserted (no VC stall required). With shared slots, 31 of the 32 writes could drop and pass | No | WEAK | med | trid.cpp:1436-1500 |
| MimirTridIdma.BarrierThenDependentReadAfterWrite | 4.3 ordering (upstream test_transaction_id) | Write GDDR0 (ack only), then a dependent read | GDDR0 region gated; the dependent read's stale words only recorded | Read leg passes on kCodeAcked or kCodeDataTimeout; OwnDestination never gated | Yes: an early ack, or a stale dependent read, passes | The dependent leg is | DEFECT | high | trid.cpp:1543-1554 |
| MimirTridIdma.BarrierThenDependentWriteAfterRead | 4.3 ordering | Read GDDR0, then overwrite it | Host model on both regions | Gated | Yes for the barrier: the read leg's XOR poll (cce_idma_copy.cpp:374-375) and the launch boundary make the WAR hazard impossible, so a broken trid barrier passes | For the barrier, yes | DEFECT | med | trid.cpp:1558-1601, 1502-1505 |
| MimirTridIdma.MultiHart | 4.3 per-hart concurrency | 8 harts, hart h on trid 2h | Host model per hart | Gated per hart; the kEightHartLaunchPasses skip is not taken | S2 and S3: yes (cross-hart aliasing is masked). Nothing shows the harts overlapped | No | WEAK | med | trid.cpp:1608-1649 |
| MimirTridIdma.Fairness | 4.3 fairness and starvation | Heavy CCE at depth 16 while the other CCEs run table traffic | Host model per source; each source must finish inside the 2^22-cycle budget | Completion gated; stalls only in RecordProperty; 2x preflight skip | A source starved for up to ~50 s passes; the heavy CCE's depth is never shown | No | WEAK | med | trid.cpp:1657-1744 |
| MimirTridIdma.Soak | 4.3 randomised soak | 4 rounds x 64 random entries, random trids and sizes | Host model; seed recorded | Gated per round; 2x preflight skip | S2 and S3: yes; no variation in depth or response delay | No | WEAK | high | trid.cpp:1752-1805 |
| MimirTridRemote.RemoteTl1 | 4.3 remote TL1, Mimir0.CCE0 -> Mimir1.CCE0 and reverse | D2D write (VC 3), read back (VC 5) | Host reads the remote CCE via UMD | Gated; the 1x skip is correct (needs 2 Mimirs); 2x preflight skip | S2 and S3: yes; boundary IDs are not observed | No | WEAK | high | trid.cpp:1819-1850 |
| MimirTridRemote.RemoteGddr | 4.3 remote DRAM | As RemoteTl1, to the remote base view | Host ReadFromDRAMChannel | As RemoteTl1 | S2 and S3: yes | No | WEAK | high | trid.cpp:1852 |
| MimirTridReset.NoStaleIdAfterReinit | 4.3 reset, no stale ID | Counters read 0 after FORCE_REINIT; trids reusable | Observer pending == 0 per trid, then a post-reinit batch | EXPECT pending == 0; skips unless the env var is set | Yes: run 1 is gated to drain (:1887), so the counters are already 0 before the reinit, and "reinit never clears counters" passes. Only the post-reinit batch is a real gate | The stale-ID leg is | DEFECT | high | trid.cpp:1875-1931 |
| MimirTridConcurrent.WithPrefetcher | 4.3 mixed traffic with the prefetcher | Trid stream running concurrently with the prefetcher | Stream status plus host data; receiver page count | Stream and receiver gated; no overlap -> GTEST_SKIP; 1x skip (needs 3 CCEs) | Yes: no concurrency ends in a skip; the data polls of rounds 2 and later are trivially true | The overlap leg is | DEFECT | high | trid.cpp:2142-2198 |

##### 2. DEFECT and WEAK details

- **PendingVisible, DepthBeyond16, Width and Scope.**
  - Each targeted property reaches only RecordProperty (trid.cpp:1059, 1079-1080, 1115, 1179). Scope has an explicit "inconclusive" branch (:1172-1175).
  - The gated copy is host-checked only at the first line of command 0 and the last line of the last command (:294-299). prepare_batch poisons only the first line of each command (:249-255), so the rest of the up-to-128 KiB destination rests on the kernel's own data poll (failure mode 4).
  - Width could gate the documented MIM-539 behaviour (pending on trid 17 % 16 after the issue), so that a fix or a regression shows up.
- **UniqueIds\*, MixedTargets, ReuseWrap, MultiHart, Soak and Remote\*.**
  - The data oracles are sound and independent: poison is unique per tag and transfer (:817-820), guards are compared (:784-788), and the host and the kernel use different address paths.
  - The trid dimension, though, passes S2 (a)-(c) and S3. One assert after each issue would close it: pending(trid) >= 1, and the pending counts of the other used trids unchanged.
- **DuplicateId.** A pending counter that saturates or wraps below 16 lets the wait return early, and the data poll hides it. The high water is recorded (:1362, 1373) but never asserted >= 2, as Saturate does.
- **NominalDepth.** It claims 16 outstanding but asserts only >= 1 (:1412-1414). The kernel already reports the total pending high water (kStatusProbeSecond, cce_idma_copy.cpp:622-628); NominalDepth never reads it.
- **ReadAfterWrite.**
  - Line 1547 accepts kCodeDataTimeout as success, and the dependent destination only feeds mismatched_words -> RecordProperty (:1551-1553).
  - Scenario: the write's trid acks before the GDDR write commits, the dependent read copies stale data, the XOR poll times out, and the test passes.
  - Given the MIM-628 comments (the cmdbuf path waits for the real response), this leg should gate. If the emulator then fails it, that is "fails on emulator, gates correctly".
- **WriteAfterRead.**
  - The read leg's destination is own TL1 and host-verified, so the kernel polls the XOR until the data arrives (cce_idma_copy.cpp:364-375). The write runs in a later Program.
  - A read whose trid acks before its source reads finish therefore cannot be overtaken by the write. The comment at trid.cpp:1502-1503 admits there is no in-launch barrier.
  - The data gates do catch copy bugs.
- **Fairness.** Stalls are only recorded (:1736-1742). "Bounded progress" means only "finished inside 2^22 cycles (~50 s)". The heavy CCE's depth of 16 is not asserted.
- **NoStaleIdAfterReinit.**
  - Session 1 must end with every trid at 0 (run_scenario_batch -> wait_for_trids). The 16 reads after the reopen (:1914) then see 0 whether or not the reinit touches the RoCC counters.
  - Reset with commands in flight is never exercised; the plan (line 593) marks it blocked by MIM-601.
  - Not traced: whether FORCE_REINIT resets the CCE harts or the cmdbufs at all. metal_context.cpp:159 only sets force_reinit_.
- **WithPrefetcher.**
  1. Lines 2196-2198 skip when either counter did not move in the window, which is a skip when the targeted concurrency does not occur. pages_sent_before is sampled after QueueTensorPrefetcherRequest (:2118-2143), so a prefetch that finishes first ends in a skip.
  2. Every round writes the same source to the same destination (:2098-2104; kModeTridStream repeats kModeTridBatch rounds, cce_idma_copy.cpp:673-712). From round 2 on, the kernel's data_landed compares against a destination that already holds round 1's data, and the host check (:2179) sees the same words. A dropped or misdirected write in any later round passes, including in the rounds that actually overlap the prefetcher.
  3. expected_commands is derived from the kernel's own round count (:2174), so that check is only self-consistent.

##### 3. UNSURE / questions for the user

1. Does "duplicate-ID protection" in plan 4.3 mean the hardware rejects a reused in-flight ID (impossible with count semantics), or only that N commands on one trid all complete? DuplicateId tests only the latter. It is WEAK under either reading.
2. Should the barrier cases fail when the dependent op sees stale data, given MIM-628's comments that the cmdbuf path waits for the real response? If yes, ReadAfterWrite is a DEFECT that is easy to fix. If the user has evidence that cmdbuf acks also come early on the target, the leg should still be a gated expected failure, not a record.
3. When the 2x D2D preflight mismatches, should MixedTargets, NominalDepth, Saturate, Fairness and Soak skip their local legs too? Today they do (trid.cpp:997-1002).
4. Does TT_METAL_FORCE_REINIT reset the RoCC command buffers of the CCE harts? Not traced. The answer decides whether NoStaleIdAfterReinit's post-reinit batch tests anything beyond the firmware reload.

##### 4. Tests that may belong to another section

- MimirTridConcurrent.WithPrefetcher also fits MM-7 (prefetcher concurrency); the MM4x plan's MM-7 rows (line 64) cite it.
- MimirTridReset.NoStaleIdAfterReinit depends on the MM-2 MimirFwReinit mechanism.

##### Not checked

- CceReceiverProgram::stop_and_wait in mimir_prefetcher_common.hpp, and the cce_gcb_discard_receiver kModeDrainUntilStopped semantics behind trid.cpp:2193-2194. The WithPrefetcher verdict does not depend on them.
- The internals of UMD's WriteToDRAMChannel.
- Spec chapters on cmdbuf TR_ACK semantics, beyond the Jira tickets.
- The grendelemulation iDMA model source. I read only models/qsr1_a1/tests/dx/RUNQUEUE.md rows 38-61. Those notes say the emulator's cmdbuf ack counter did not guarantee source-read visibility, so a gated ReadAfterWrite may fail on the emulator.

### B.13 Debug tools (reviewer debug-tools)

#### Debug-tools teammate report (Group D, 7 rows): 4 OK, 3 WEAK, 0 DEFECT, 0 UNSURE rows

None is vacuous. I read the code at HEAD 16585a9c68a (UMD submodule f0d497f3386). I ran nothing. Paths are relative to tests/tt_metal/tt_metal/ unless they start with tt_metal/ or tt_stl/.

Plan case for every row: none (the plan has no watcher or DEVICE_PRINT entry). Descriptors: all CCE rows run on mimir_1x1 and mimir_2x_package under slow dispatch. The README records no runs of these cases.

##### 1. Table

**1. WatcherAssertTests Drisc0..Drisc7** (debug_tools/watcher/test_assert.cpp:484-491)
- Feature: a watcher ASSERT on CCE hart N is reported with the right core, label and kernel.
- Oracle: the full watcher exception string is matched by regex: core, virtual coords, "drisc{N}", message, kernel path (:338-358). The hart label comes from the device's `which` field, which is set to read_hw_thread_idx (tt_metal/hw/inc/api/debug/assert.h:42). The reader names it through get_riscv_name (watcher_device_reader.cpp:83-98, :999).
- Gates and skips: ASSERT_FALSE(exception.empty()) with a 5 s timeout (:276), then EXPECT regex_match (:358). Skips: core type absent; processor_type >= count; fast dispatch (:406-442). The HAL gives 8 DRAM harts on Quasar (qa_hal_dram.cpp:102, MEM_CCE_LOCAL_HARTS=8) and 1 on Blackhole (bh_hal_dram.cpp:98-108), so Drisc1..7 skip on Blackhole.
- Broken implementations checked:
  - DramConfig.processor ignored: caught by Drisc1..7, not by Drisc0.
  - Reader labels every hart drisc0, or uses an off-by-one index: caught.
  - Reader does not poll CCE, or uses the wrong mailbox address: caught (empty exception).
  - Subordinate hart never launched: caught.
- Vacuous: no.
- Verdict: OK. Confidence: high (both sides traced).
- Evidence: test_assert.cpp:178-190, :276-358; assert.h:25-44.

**2. WatcherRingBuffer Drisc0..Drisc7** (test_ringbuf.cpp:423-430)
- Feature: the CCE MPSC ring buffer, single writer: head kept in the mailbox, wrap, hart tag.
- Oracle: 140 pushes (capacity 128 + 12). The expected 128 newest-first entries are built on the host from DramConfig.processor and checked in order (:40-59, :212). The device takes the head from the mailbox (ring_buffer.h:37-43). The reader takes rb->head for DRAM cores (watcher_device_reader.cpp:1121-1123).
- Gates and skips: EXPECT FileContainsAllStringsInOrder (:212), same skips as row 1. The log is cleared and devices are reopened on every test (debug_tools_fixture.hpp:55, :170). The watcher mailbox is zeroed at init (watcher_server.cpp:466).
- Broken implementations checked:
  - Processor ignored: caught for N>=1.
  - Head never stored, so every push lands in slot 0: caught.
  - Reader uses sem_head for DRAM: caught (order wrong after wrap).
  - Broken lock: not caught. A single writer never contends; row 3 covers that.
- Shared code: the expected and actual labels both come from FormatRingBuffer and get_processor_class_name ("CCE{N}"). Only the name is shared; the data and index are independent.
- Vacuous: no.
- Verdict: OK. Confidence: high.
- Evidence: test_ringbuf.cpp:172-213; ring_buffer.h:32-56; watcher_device_reader.cpp:1111-1141.

**3. MeshWatcherFixture.TestWatcherRingBufferCceMultiWriter** (test_ringbuf.cpp:216-270, :451-461)
- Feature: the cached atomic lock serializes the 8 CCE harts' claims on head.
- Oracle: 8 harts each push 5 entries. Each of the 40 (hart, seq) strings must appear somewhere (:258-269).
- Gates and skips: EXPECT FileContainsAllStrings. Skips on non-Quasar, fast dispatch, or fewer than 2 harts; the hart-count skip never fires on Quasar (8 harts).
- Broken implementations checked:
  - No lock, or a lock that does not serialize across harts: caught only if two harts actually race in the head++ window. Nothing forces or checks that they overlap.
  - Duplicate entries or a short head: not checked (presence only, no head==40 read).
- Vacuous: no.
- Verdict: WEAK. Confidence: medium (actual overlap on the emulator is unknown).
- Evidence: test_ringbuf.cpp:239-269; ring_buffer.h:38-42.

**4. MeshWatcherFixture.CceTestWatcherSanitizeL1Overflow** (test_sanitize.cpp:1188-1249)
- Feature: the L1 access sanitizer on a CCE hart. CCE is built with COMPILE_FOR_DM and COMPILE_FOR_DRISC (hal_2xx_common.cpp:83-87), so the DM dual-view check applies (sanitize.h:781-788).
- Oracle: a CoreLocalMem write to 0xDDDDDDDD on hart 0. EXPECT_EQ on the exact message, "drisc0 core overflowed L1 with access to 0xdddddddd of length 4 (...)", matching reader lines :301 and :914.
- Gates and skips: if the run throws, the error must contain "Aborting wait due to watcher error" or "overflowed L1" (:1224-1227). Then EXPECT_EQ (:1246). Skips on TT_METAL_WATCHER_DISABLE_NOC_SANITIZE, fast dispatch, or non-Quasar.
- Broken implementations checked:
  - Sanitizer not compiled for CCE, or reader does not poll the DRAM sanitize slot: test hangs (see 2b), it does not pass.
  - Bound that rejects the legal uncached alias (4-8 MiB), or misses an access that straddles the seam: not caught. There is no in-range control and no straddle case; upstream Tensix has SanitizeL1OverflowStraddle (:62, :402).
  - Sanitizer that flags every access: not caught, since this kernel makes only one sanitized access.
- Vacuous: no.
- Verdict: WEAK. Confidence: high.
- Evidence: test_sanitize.cpp:1209-1246; test_kernels/misc/cce_watcher_sanitize_l1.cpp:13-19; sanitize.h:62-66.

**5. MeshWatcherFixture.TestWatcherWaypointsCce** (test_waypoint.cpp:350-408)
- Feature: the watcher shows a CCE kernel's WAYPOINT while the kernel runs.
- Oracle: the kernel posts AAAA and blocks on a host flag at the uncached alias. The host polls the log for `dram core(..)*: AAAA`, releases the flag, then checks the line in order. Hart 0 is the first comma-joined field after ": " (watcher_device_reader.cpp:674, :1312-1333). After the kernel finishes, firmware overwrites the waypoint, so a stale AAAA is impossible.
- Gates and skips: ASSERT_LT with a 30 s timeout (:388), then EXPECT (:404). Skips on fast dispatch or non-Quasar.
- Broken implementations checked:
  - Reader reads the wrong CCE window or mailbox: caught.
  - Kernel not launched: caught.
  - Subordinate harts write the wrong slot (waypoint.h:36 indexes by get_hw_thread_idx): not caught. Only hart 0 runs and only hart 0's slot is checked; upstream checks every processor on the core.
- Vacuous: no.
- Verdict: WEAK (narrow). Confidence: high.
- Evidence: test_waypoint.cpp:362-408; test_kernels/misc/watcher_waypoints.cpp.

**6. DevicePrintDramFixture: DramPrintSimpleString, SingleUintArg, BasicTypes, Factorial and PrintConcurrentDriscCores** (debug_tools/device_print/test_dram_print_output.cpp)
- Feature: DEVICE_PRINT from CCE through the new shared DevicePrintBuffer<1632,8> (device_print_mem.h).
- Oracle, the four single-hart cases: distinct expected strings, including values such as 42 and 120, in a fresh per-test memfd (debug_tools_fixture.hpp:305-324). They run on logical {0,1} hart 0.
- Oracle, PrintConcurrentDriscCores: 8 harts x 5 iterations, with an exact count of 8 for each "Test iteration: N" (:166-178).
- Gates and skips: EXPECT_TRUE / EXPECT_EQ. Skips: fast dispatch, no DRAM core type; the concurrent case also skips on non-Quasar.
- Where {0,1} lands: on Mimir it is view 0, endpoint index 1, the second location's CCE. dram_bank_endpoint_coords[0] has 2 entries for `dram: [[0-0, 1-0]]` (metal_soc_descriptor.cpp:387-410), so the core is valid. The concurrent case uses {0,0}.
- Broken implementations checked:
  - Host reads the wrong buffer offset, or the server does not drain DRAM cores: caught.
  - Lost or duplicated messages under concurrency: caught. Here contention is built in: 40 messages fill the shared 1632-byte buffer, so harts wait on the drain together. That is why this case is OK while row 3 is WEAK.
  - Wrong hart attribution: not checked.
- Vacuous: no.
- Verdict: OK. Confidence: high.
- Evidence: test_dram_print_output.cpp:39-56, :127-185.

**7. DramSenderGCBFixture.TensorPrefetcherGateKeepsBlackholeResultAndRejectsQuasarConfig** (api/test_dram_sender_global_cb.cpp:650-665)
- Feature: on Blackhole, IsTensorPrefetcherSupported still returns true, and a Quasar config is rejected on non-Quasar.
- Oracle, leg 1: EXPECT_EQ(supported, !streaming_profiler).
- Oracle, leg 2: ThrowsMessage<std::runtime_error> with a substring that only tensor_prefetcher_manager.cpp:1203 produces. TT_FATAL throws std::runtime_error (tt_stl/tt_stl/assert.hpp:112). The preconditions return ok on Blackhole (:2416-2418), so start() reaches the check.
- Gates and skips: leg 1 always gates. Leg 2 is skipped when TT_METAL_STREAMING_PROFILER is set (an environment-driven skip). The fixture is BlackholeSingleCardFixture, so the test never runs on Mimir.
- Broken implementations checked:
  - Blackhole gate regressed to false: caught.
  - Quasar config accepted on Blackhole: caught.
  - Any Quasar or Mimir gate behaviour: not exercised.
- Vacuous: no.
- Verdict: OK for its stated claim. Confidence: high.
- Evidence: test_dram_sender_global_cb.cpp:650-665; tensor_prefetcher_manager.cpp:1194-1205, :2398-2418, :2505-2518; api/dram_sender_fixture.hpp:29-41.

##### 2. The WEAK rows

**2a. CceMultiWriter.** Lost updates show up only if harts actually overlap inside the 3-instruction head++ critical section (ring_buffer.h:38-42). The check is presence only (test_ringbuf.cpp:269). It never reads mailbox head (it should be 40) and never checks for duplicates. A lock that does not work across harts passes whenever the harts happen to run one after another. Fix: read head back and EXPECT 40, and push more per hart (16 x 8 = 128) to widen the race window.

**2b. CceTestWatcherSanitizeL1Overflow.** The single address, 0xDDDDDDDD, is outside every view, so the branch's dual-view rule (sanitize.h:781-788, :62-66) is never tested at its boundaries. A bound that rejects the legal uncached alias at 4-8 MiB, or misses an access straddling the seam, passes this test. Only hart 0 runs. The `do { } while (exception.empty())` loop at test_sanitize.cpp:1241-1244 has no timeout, so a broken sanitizer hangs the test instead of failing it.

**2c. TestWatcherWaypointsCce.** One kernel on hart 0, one slot checked (test_waypoint.cpp:370-375, :395-404). The 8-hart slot indexing is not exercised.

##### 3. UNSURE / questions for the user

1. Has unit_tests_debug_tools been run on mimir_1x1 or mimir_2x_package? The README records no runs of the Group D cases, so I cannot say which of them fail on the emulator. This does not change any verdict.
2. On Mimir, DramPrint* run on logical {0,1}, which is the second location's CCE. The comment there still says the Blackhole "subchannel 0 is syseng-owned". The watcher tests use {0,0} with the comment "Mimir exposes one CCE per logical DRAM channel, with no subchannel dimension", but Mimir does have a y=2 logical dimension. Is targeting CCE1 in the print tests intentional?

##### 4. Placement

- TensorPrefetcherGate... is a Blackhole-only regression guard next to MM-7. It does not test the Mimir capability gate; MimirPrefetcher.MM7Gate* do.

##### 5. Not checked or not covered

- The branch reworked the Quasar NOC-side check in debug_valid_drisc_addr (view-relative mailbox offset, sanitize.h:292-315). No Group D test issues a CCE NOC write to local SRAM, so that change has no test.
- I did not verify that hardware coherence of the cached lock against the uncached mailbox head is sound. I only checked the code ordering.
- Smaller edits outside my list: WatcherNonDefaultAssertTests renames "Drisc" to "Drisc0" (same case as row 1, Drisc0). TestWatcherRingBufferMpscMultiWriter gained an empty-grid skip, which is correct on Mimir since Mimir has no Tensix.

## Appendix C: state-persistence reports (2026-10-29)

> Note (2026-10-29 refresh): the reports below are as written at HEAD 16585a9c68a, and so are their file:line
> citations. Changed in the working tree since then (uncommitted), so line numbers into these files may have moved: context/test_mimir_emu.cpp, dram/test_mimir_host_dram.cpp, dram/test_mimir_host_tl1.cpp,
> dram/test_mimir_trid.cpp and dram/README.md. Metal runtime, firmware, UMD (checkout still f0d497f), chippy and
> grendelemulation citations are unaffected.

Each report is as its agent wrote it; the State persistence section above holds the conclusions. Paths in
grendelemulation and chippy refer to /proj_sw/user_dev/sagarwal/grendelemulation (2b916f42) and
/proj_sw/user_dev/sagarwal/chippy (c9a8a608); UMD paths refer to the working tree at f0d497f3.

### C.1 Emulator model and chippy bring-up

#### Emulator model and chippy bring-up: does state survive?

**Conclusion.** Within one emulator session nothing resets the design or clears memories on its own. That holds on connect, on disconnect, and when a new client process starts. Only an explicit SiVal `INIT` pulses BP_RESETN, and a `QUIT` ends the session. The chippy bring-up scripts send neither of these, and they never assert a CCE cold reset or an uncore warm reset. They write no CCE TL1 and no GDDR data. They overwrite only a fixed set of CCE uncore registers. So TL1 and GDDR contents, the CCE reset vectors, and every uncore field bring-up does not program carry over from an earlier process. A test that writes fixed values without poisoning first can pass on data left by an earlier run.

**Sources.**
- GE = `/proj_sw/user_dev/sagarwal/grendelemulation`, commit 2b916f42b5fc (2026-10-27).
- CH = `/proj_sw/user_dev/sagarwal/chippy`, commit c9a8a6085433 (2026-10-27).
- Both are the checkouts the lead named, and `git status` is clean in both.
- Both models are covered: `models/mimir` (1x1) and `models/mmk`. The bring-up scripts target both.

**Which server chippy talks to.** It is `SiValServer` in `GE/py/syseng_common/server.py`. `GE/models/{mimir,mmk}/tests/test_sival_server.py` hosts it on port 8080.
- `GE/py/emu_server/emu_server.py` is not it. That is the `ec` RUN-yaml harness.
- `CH/lib/transport/emu_axi_transport/emu_server/syseng_test_server.py` is not it either. It is a mock protocol mirror (see `README`/`mock_server.py` there).

##### 1. Findings

###### A1. Resets reachable after the emulator starts

- **SiVal `INIT`**: `INIT` calls `emu_hooks.initialize()` (GE/py/syseng_common/server.py:380-382).
  - Mimir: `initialize` = `make_reinitialize` → `tb.reset(straps)` (GE/models/mimir/tests/test_sival_server.py:47,50-51; GE/py/syseng_common/sival.py:102-108).
  - MMK: `reinitialize` → `reset_with_boot_straps` → `tb.reset` (GE/models/mmk/tests/test_sival_server.py:229-237,217,245-246).
  - Preload (fuse/ROM `memory_load`) is not repeated (sival.py:97-98; mmk test_sival_server.py:232).
- **What `tb.reset` drives**: it forces `tb_bp_resetn=0` and `tb_xtor_reset_n=0` for 500 ns, then 1. powergood stays 1.
  - Mimir: GE/models/mimir/py/common/tb.py:166-179.
  - MMK: GE/models/mmk/py/common/tb.py:429-440, which asserts `tb_bp_vddcore_powergood=1` and then pulses BP_RESETN.
  - BP_RESETN is the chip reset pin (mimir tb_emu.sv:103). Which flops and memories it reaches is decided by RTL I could not read: `deps/mimir_soc` is absent from GE. Expected to reset registers to POR, but **unknown** for memories.
- **SiVal `QUIT`**: sets `shutdown_event`, so the server and the emulator session exit (server.py:464-466, 300-306).
- **Who sends INIT/QUIT**:
  - The chippy test runner sends `INIT` at setup (CH/src/tenstorrent/chippy/test_orchestrator/runner.py:63-65).
  - Its teardown calls `transport.teardown()`, which sends `QUIT` (runner.py:115-134; CH/lib/transport/emu_axi_transport/emu_axi_transport.cpp:163-167).
  - The gdb stub also calls `initialize()` (CH/src/tenstorrent/chippy/test_orchestrator/gdb_stub/__main__.py:180) and `teardown()` (line 228).
  - The bring-up scripts build the transport directly and call neither (CH/validation/metal_bringup/bringup_mimir_1x1.py:9-13, bringup_two_mimir.py:11-12, bringup_mmk.py:15-16).
  - `GrendelAxiEmu` construction does not call either (CH/lib/arch/grendel/grendel_emu.h:45-52; CH/src/tenstorrent/chippy/test_orchestrator/asic_factory.py:32-36).
- **JTAG**:
  - The MMK JTAG sideband wires `INIT` to `reinitialize` (GE/models/mmk/tests/test_sival_jtag_server.py:233-243,460-461), with the same BP_RESETN effect.
  - The Mimir JTAG sideband passes no `initialize`, so `INIT` there is the default no-op (GE/models/mimir/tests/test_sival_jtag_server.py:71-74; server.py:103-104,235).
  - The JTAG xtor's own `rstn` is `tb_xtor_reset_n` (GE/models/mimir/emu_hdl/jtagm_bind.sv:20). It reaches only the xtor.
- **SMC / pf_ctrl registers**: any client can WR32 `reset_unit.ss_cold_reset_n` bits to 0 (cold reset) or `pf_ctrl.reset.uncore_reset` to 0 (uncore warm reset).
  - The server forwards writes blindly (server.py:365-378).
  - Nothing in the emulator side does this by itself. The grendelemulation TB's `Cce.manage_reset` can (GE/py/tb/grendel/cce.py:189-222), but only pytest TB tests use it, not the SiVal server.
  - What these resets reach in RTL is for the spec agent. One comment says the DMRISC remap block does not answer while the uncore is held (CH/.../gdb_stub/fabric_init.py:219-222; comment only).
- **Testbench reset inputs**: `tb_bp_resetn`, `tb_bp_powergood`, `tb_bp_gpio` and the other `tb_emu.tb_*` nets are dynamically forceable (GE/models/mimir/build/zforce.utf:2). They are driven only from `tb.reset` above.
  - In MMK, `M{0,1}_SRAM_AUTOZERO_STRAP` are top pins that `tb.reset` never forces (GE/models/mmk/emu_hdl/tb_emu.sv:54,69,316,331). Their RTL effect is **unknown**.
- **ZeBu run control**: the TB API exposes `run`, `force`/`release`/`deposit`, `memory_load` and `memory_store` (GE/py/tb/discover/zebu_api.py:32,55-60). It has no memory-clear call.
  - For Mimir, `memory_load` targets only the SMC efuse/ROM at preload (GE/py/tb/grendel/smc.py:134; GE/py/tb/grendel/mimir.py:188-200).
  - Preload runs once at server start (mimir test_sival_server.py:33; mmk test_sival_server.py:112-122).
  - `LOAD_SMC_FW` writes SMC SPM only (server.py:49-61,407-422).

###### A2. How CCE TL1 SRAM is modeled

- It is RTL built from mimir_soc with `-D USE_VERILOG_MEM_ARRAY -D RAM_MODELS -D TT_EMULATION` (GE/models/mimir/build/gen_flist.sh:73-75).
  - The only `sparse_memory` directive targets `ddr_model_ram` (GE/models/mimir/build/top.utf:16). MMK adds `axi_ram` (GE/models/mmk/build/top.utf:14,18).
  - So TL1 is presumably a plain Verilog array compiled to ZeBu memory. I did not read the array source, so this is **inferred, medium**.
- No init on reset or on command was found:
  - No emulator-side `memory_load` targets CCE SRAM (A1 list).
  - The emulation patches do not touch CCE SRAM (GE/models/mimir/mimir_patch/cce/cce.patch: PROMISE sensor only; mimir_soc.patch: GDDR stub/ATB only).
  - The only zeroing found is `Cce.preload` when `using_init_flow` is set (GE/py/tb/grendel/cce.py:107-115). That is pytest-only and not on the SiVal/chippy path.
- Whether RTL reset or cold-reset release auto-zeros TL1 is **unknown** (RTL not available).
  - A chippy comment claims cold reset leaves *D2D* SRAM zeroed (CH/validation/python/tenstorrent/sival_arch/utility/d2d.py:430-431). This is a comment, not traced code, and it is not about TL1.

###### A3. Can the GDDR sparse page table or its contents be cleared within a session?

- No code path to clear it was found. `ddr_model` sits behind `axi_mem_gddr7` with `rst = ~async_axi_resetn` (GE/models/mimir/emu_hdl/axi_mem_gddr7.sv:13-22,73-77).
  - The ddr_model source is not readable: `/proj_soc` is not mounted here.
  - TB knobs on it are timing forces only (GE/py/tb/grendel/mimir.py:142-186).
  - A comment says the "ZeBu sparse page table is not cleared between tests". Each of those tests calls `tb.init_and_reset()`, so that BP_RESETN pulse does not clear it (GE/models/mimir/tests/test_gddr.py:162-163,170-171; comment only, re-verified as present).
- So within a session: **not cleared** by anything I found. Medium confidence, based on a comment plus unreadable model source.

###### A4. Per-client server state, and connect/disconnect behaviour

- `handle_client` only reads newline commands and closes the writer on EOF. No hook runs on connect or disconnect (server.py:308-339).
- Server-global state that persists across clients:
  - The MMK `SELECT_CHIPLET` context lives in a closure dict shared by every connection and is never reset (GE/py/syseng_common/sival.py:154-160; mmk test_sival_server.py:242,253-254).
  - Chippy's `MultiChipletEmuAxiTransport` re-selects before every R/W, so it is immune (CH/lib/transport/emu_axi_transport/emu_axi_transport.cpp:298-317). A client that does not re-select inherits the previous client's chiplet.
  - Waves state (`WAVES_ON`/`OFF`) also persists (server.py:391-399).
- The transport constructor only does a TCP connect (emu_axi_transport.cpp:124-157). `initialize()`/`teardown()` are explicit (emu_axi_transport.cpp:159-167).
- The Metal/UMD path in GE (`GE/py/umd/remote.py`) is Quasar-only (`tb.qsr`, remote.py:367,382; reset.py:16-27). It calls `tb.init_and_reset()` once at start (remote.py:515). It does not apply to the Mimir/MMK models. Metal's own client behaviour is for the Metal/UMD agent.

###### B1. Resets asserted or released by bring-up, in order

**Common** `mimir_cce_and_gddr(mimir, inst)` (CH/validation/metal_bringup/mimir_bringup_common.py:29-54):
1. `cce_config_clock(pll_clk_sel=0)`: CGM muxes only, no reset write (CH/validation/python/tenstorrent/sival_arch/utility/cce.py:86-116; the `if pll_sel` branch at :112 is skipped).
2. `cold_reset_lift(SMN|ITN|CCE0|CCE1)`: **release only**, an RMW OR into `ss_cold_reset_n` (CH/.../sival_arch/libs/cce_bringup.py:60-66; CH/.../utility/cold_reset.py:190-193).
3. `bring_up_d2d(D2D0)` (CH/.../utility/d2d.py:318-345):
   - Releases cold reset for d2d0, smn and itn (OR) (d2d.py:333-336).
   - Releases the NOC2AXI strap resets one field at a time, setting each to 1 (d2d.py:211-238).
   - Releases D2D `async_cpu_ctrl.uncore_reset=0` (d2d.py:248-269).
4. `cce_config_clock(pll_clk_sel=1)`: full write of `reset_unit.ss_force_to_ref_clk=(1<<16)|(1<<20)`. This releases D2D0/D2D1 force-to-refclk, which is active-low (cce.py:112-114).
5. `gddr_app_bringup_emu(0)`, then `(1)`: cold-reset lift of GDDR0/GDDR1 (OR), then `gddr7_config.reset_ctrl=0x1F`, which releases all GDDR block resets (CH/.../libs/gddr_app.py:180-183).
6. For CCE0, then CCE1: `cce_uncore_release(hold_cores=True)` is an RMW of `pf_ctrl.reset` to `uncore_reset=1, core_reset=0` (cce.py:386-390).
   - This **releases the uncore** (or leaves it released) and **asserts or holds hart warm reset on all 8 harts**. If an earlier run left harts running, this write puts them back into reset.

**No CCE cold reset and no uncore warm reset is asserted anywhere before `cce_uncore_release`.** Every cold-reset write is OR-only, and the uncore field only goes to 1 (high confidence).

**Script-specific:**
- `bringup_mimir_1x1.py`: the common sequence on m0 only (bringup_mimir_1x1.py:12-13).
- `bringup_two_mimir.py`:
  - `bring_up_d2d` on m0 D2D0, m0 D2D1, m1 D2D0, m1 D2D1 (bringup_two_mimir.py:17-20), each as step 3 above.
  - `m2k_host_prepare_mimir_d2d` on m0 and m1, which walks the strap resets again, release only (CH/.../libs/noc2axi_m2k.py:344-347; d2d.py:241-245) (bringup_two_mimir.py:21-22).
  - The common sequence on m0, then m1 (:24-25).
  - `stage_d2d` on m1 D2D0 and m0 D2D1: `load_d2d_fw` writes `cpu_ctrl` with `core_reset=1, uncore_reset=0, debug_reset=1`. This **asserts the D2D Rocket core reset** (d2d.py:439-444) (:30-31).
  - `release_d2d` on m1 D2D0, then m0 D2D1: writes `cpu_ctrl` to 0x00010000 (core and uncore out, debug held) unless it already holds that value (d2d.py:285-295) (:32-33).
- `bringup_mmk.py`:
  - Same as two_mimir, plus `bring_up_d2d(k0, D2D2, hsio_tile=2)`, which releases cold reset for d2d2, smn and hsio0-2 (d2d.py:195-208) (bringup_mmk.py:25), and `m2k_host_prepare_keraunos_d2d(k0, 2)`, which walks the strap resets (noc2axi_m2k.py:350-353) (:28).
  - After the common sequence on m0 and m1 (:30-31) comes `initialize_fabrics(k0, "hsio.2.cce.{0,1}", release_cores=False)` (:32-33). This releases, by OR, the Keraunos cold reset bits SMN|HSIO2 and the HSIO2 internal resets CCE/TL1/FABRIC/CCE_DBG. It then writes the Keraunos HSIO2 CCE `pf_ctrl.reset` to uncore=1, core=0, with cores held (CH/src/tenstorrent/chippy/test_orchestrator/gdb_stub/fabric_init.py:333-334,351-359,376-377,259-262).
  - It stages m1 D2D0, m1 D2D1, k0 D2D2 and m0 D2D1, each asserting D2D core reset (:37-40). It then releases them in the order M1 D2D0, M1 D2D1, K0 D2D2, M0 D2D1 (:41-44).

###### B2. Memory writes, with address ranges

- **CCE TL1: none.** No called function writes `cce.sram`. It is used only for `global_address`, `local_address` and `size` (cce.py:183-187; mimir_bringup_common.py:38; fabric_init.py:186-190,210). High confidence.
- **GDDR: none.** No training patterns and no memory init. Emu GDDR bring-up is reset release plus `hidestrb_enable=0` (gddr_app.py:9-19,180-185). `gddr_noc_config` writes 4 ITN sideband flag registers at 0x04004054/0x04004050/0x04004354/0x04004350, which are not DRAM (gddr_app.py:96-102,131-134). High confidence.
- **D2D SRAM** (`load_d2d_fw`, d2d.py:446-456,407-417). D2D tile bases are Mimir `0x04200000 + idx*0x200000` and Keraunos `0x10200000 + idx*0x200000` (d2d.py:33-37,112-120).
  - **Firmware image**: every *nonzero* 32-bit word of `validation/firmware/libs/chip_init/d2d_fw/d2d_fw.bin` (27624 B = 0x6BE8, a path relative to cwd), written to `[tile_base+0x2000, tile_base+0x2000+0x6BE8)`. Zero words are skipped, so old contents survive at those offsets (d2d.py:452-455).
  - **Config table**: `tile_base+0x2000+0xFCC0` gets 0x1234BEAF, then 25 words at `+0xFCC4..+0xFD24` (d2d.py:375-404,412-417).
  - **Tiles written** (chiplet-local addresses):
    - two_mimir: M1 D2D0 at 0x04202000 and M0 D2D1 at 0x04402000.
    - mmk: those two plus M1 D2D1 at 0x04402000 and K0 D2D2 at 0x10602000.
    - 1x1: none.
- **Anything else**: registers only (B3, plus SMC/ITN/SMN/HSIO firewall filters, D2D strap/BAR/ATT/ET tables, and SMC CGM muxes). The earlier note "bring-up writes only registers and D2D SRAM" still holds at CH c9a8a608.

###### B3. CCE uncore registers programmed by bring-up

Mimir CCE0 and CCE1 on each Mimir brought up, in this order:
1. `mimir_cce_ctrl.local_tl1_base.addr = cce.sram.global_address`, an RMW of the `.addr` field (mimir_bringup_common.py:36-38; cce.py:152-154).
2. `tl1_csr.group_hash_fn_ctrl` is set to `hash_fn0_en=1, hash_fn1_en=0`. `group_hash_fn0_mask.bit_sel` and `group_hash_fn0_match.bit_sel` are set to 0. Raw values go to `group_hash_fn0_addr4..addr9` (cce.py:52-59,357-374).
   - `addr10` is **not** written by chippy. GE's TB table has a 7th tap, 0x00048480, for it (GE/py/tb/grendel/cce.py:35). So addr10 keeps its POR value or whatever an earlier process wrote.
3. `pf_ctrl.reset` gets `uncore_reset=1, core_reset=0` (cce.py:386-390).
4. `mimir_cce_ctrl.mla_remap_entries[0..2]`: region_start, region_end and region_remap_start are written. `region_attrs` is an RMW that sets only `valid=1` (cce.py:226-242,277-286).
5. `dmrisc_addr_remap.dmrisc_remap_entries[0..2]`: same as MLA. `cmd_snoop` is not set because the default is False, but its RMW does not clear a `cmd_snoop` left by an earlier process either (cce.py:238-242,323-336).

**Not programmed:**
- `tt_cluster_ctrl.reset_vector[0..7]`. `cce_reset_vectors_set` exists (cce.py:131-141) but is not imported by mimir_bringup_common.py:6-13, and `initialize_fabrics` passes `reset_vector=None` (fabric_init.py:264-265,311-313).
- Remap entries 3 and up.
- Any attribute bit other than `valid`.
- CCE scratch.
- The POR value of reset_vector is said to be 0 in a comment only (fabric_init.py:234-237; not traced).

**Keraunos (mmk only):** HSIO2 CCE0/1 `pf_ctrl.reset` gets uncore=1, core=0. No TL1, MLA or DMRISC programming (fabric_init.py:372-377).

**Side finding (low confidence).** On a second bring-up in the same session the uncore is already released, so steps 1-2 land on a live fabric. The library docstrings say these must be written while the uncore is held: cce.py:149-150, :352-353, and fabric_init.py:194-198, which warns "moving the address decode on a live fabric stalls the access path". The values are identical on a rerun, so this is probably harmless, but it is not verified.

##### 2. Table

| State | New process, emulator-model side only | Chippy bring-up |
|---|---|---|
| **CCE TL1 SRAM** | **Untouched.** No connect or disconnect hook exists, and the transport ctor only connects (server.py:308-339; emu_axi_transport.cpp:124-157). No emulator-side memory_load or clear targets it (zebu_api.py:55-56; smc.py:134). **High.** If some client sends `INIT` (BP_RESETN): **unknown**, because the RTL array is unread (gen_flist.sh:74-75). | **Untouched.** Nothing writes `cce.sram` (B2). No CCE cold or uncore reset is asserted (cold_reset.py:190-193; cce.py:386-390). **High.** Caveat: whether a cold-reset *release* after a fresh BP_RESETN auto-zeros SRAM is **unknown**. |
| **Mimir GDDR** | **Untouched.** No hook and no clear API (server.py:308-339; top.utf:16; mimir.py:142-186). **High.** Under `INIT`: **unknown, leaning untouched**, because the page table persists across `init_and_reset` per a comment (test_gddr.py:162-163) and the ddr_model source is unreadable. **Medium.** | **Untouched.** Only cold-reset lift, `reset_ctrl=0x1F`, `hidestrb_enable=0` and ITN flag registers are written. No training and no init (gddr_app.py:9-19,131-134,180-185). **High** for "no writes". ddr_model `rst` behaviour is unread, but bring-up only releases it. |
| **CCE uncore registers** (reset vectors, DMRISC remap, MLA remap, local TL1 base, group hash, pf_ctrl) | **Untouched.** No hook (server.py:308-339). **High.** Under `INIT`: **reset to defaults by BP_RESETN**, presumably (tb.py:171-179). The per-register RTL reset scope is unread, so **medium**. | **Partly overwritten** by `mimir_cce_and_gddr`: `local_tl1_base`; `group_hash_fn_ctrl`/`fn0_mask`/`fn0_match`/`fn0_addr4-9`; `mla_remap_entries[0..2]` and `dmrisc_remap_entries[0..2]` (start/end/remap_start, attrs valid=1 by RMW); `pf_ctrl.reset` to uncore=1, cores held (mimir_bringup_common.py:36-39,50-54; cce.py:152-154,357-374,386-390,226-242). **Left as found**: `reset_vector[0..7]`, `group_hash_fn0_addr10`, remap entries ≥3, non-`valid` attr bits such as `cmd_snoop`. **High.** |

##### Not checked

- mimir_soc RTL: the TL1 array, BP_RESETN and cold-reset scopes, SRAM auto-zero logic, and what `M*_SRAM_AUTOZERO_STRAP` does. `deps/mimir_soc` is absent from GE.
  - Mimir straps bit 13 (`DEFAULT_STRAPS`, mimir tb.py:12) is unmapped here. On Keraunos, bit 13 is SEP_BYPASS_MEM_REPAIR and bit 26 is DISABLE_SMC_AUTO_ZERO (GE/py/tb/grendel/keraunos.py:35-44).
  - MMK `reset()` leaves the gpio strap force commented out (mmk tb.py:419-422).
- ddr_model source (`/proj_soc` not mounted): its reset and force behaviour, and sparse-memory clear semantics.
- What the SMC production ROM does after BP_RESETN. The servers load it by default (mimir test_sival_server.py:30-31). Only binaries exist under GE/firmware/mimir/smc/prod_rom-*.
- Metal/UMD client behaviour: whether it sends INIT/QUIT or writes reset vectors or poison. Another agent covers that.
- The `bring_up_d2d` / D2D strap register internals beyond the reset fields.
- I did not run anything or connect to an emulator.

### C.2 Metal and UMD at process start and exit

#### What Metal and UMD software write or reset at process start and exit (Mimir CCE, emulator)

Sources read: tt-metal at HEAD 16585a9c68a (branch sagarwal/mimir_dram_tests). No uncommitted changes under tt_metal/impl, tt_metal/llrt or tt_metal/hw. UMD: the working tree at f0d497f3 (the code that gets built). The gitlink records 292fe2086594, which is an ancestor of f0d497f3. Paths below are relative to /localdev/sagarwal/tt-metal; UMD paths are relative to tt_metal/third_party/umd.

Scope: the emulator transport (TT_METAL_EMU_SERVER, so TargetDevice::EmuAxi, which builds a UMD SimulationChip around an EmuTTDevice), with the README environment: slow dispatch, and watcher, DPRINT, profiler, CLEAR_L1, CLEAR_DRAM, SKIP_LOADING_FW and FORCE_REINIT all unset.

- **TT_METAL_GRENDEL_JTAG_SERVER behaves the same.** It builds a SimulationChip around a GrendelJtagTTDevice (`tt-umd/src/cluster.cpp:339-347, 724-731`), which also derives from GrendelTTDevice (`tt-umd/api/tt-umd/tt_device/grendel_jtag_tt_device.hpp:26`), and its `create` asserts no reset (`tt-umd/src/tt_device/grendel_jtag_tt_device.cpp:21-67`). I did not read the OpenOCD transport constructor.
- **Two of the files you pointed me to are not on this path.** `rtl_simulation_tt_device.cpp` and `rtl_sim_communicator.cpp` belong to the `TT_METAL_SIMULATOR` (ChipType::SIMULATION) path. EmuAxi goes through `EmuTTDevice::create` and `GrendelTTDevice` instead (`tt-umd/src/cluster.cpp:721-723`), so I did not use them.

##### Two premises the source contradicts

1. **Startup does not pass RiscType::ALL to the CCEs.** At HEAD it passes `ALL_NEO_DMS`, which resets the harts only (`tt_metal/impl/device/firmware/risc_firmware_initializer.cpp:51-55`). Teardown passes `ALL_DATA_MOVEMENT` (`:57-62`), which is also harts only. Commit 3ecb73d1637, "Only reset drisc core for Quasar" (2026-10-21), made the startup change. The comment at `:52-53` says why: "RiscType::ALL also holds the CCE uncore and, on deassert, reprograms the DMRISC remap ... chippy owns both on the Mimir-local path."
   - **Before that commit, startup passed RiscType::ALL.** `git show 3ecb73d1637` shows `dram_core_reset_riscs` returning `RiscType::ALL`, and `initialize_firmware` calling `cluster_.release_cce_uncore_for_firmware_load` after the assert. So each first device open asserted the CCE uncore, released it and reprogrammed **only** the DMRISC remap; the deassert of `ALL` reprogrammed it again (`tt_device.cpp:805-811`).
   - **Some recorded runs predate the change.** The commits the README names for its runs 3ef0a46e42a (probes, 2026-10-20), 61d03f6e674 (MM4x, 2026-10-22) and 3a3d04f9bd2 (2026-10-23 and 2026-10-26 sessions 1-5) do **not** contain 3ecb73d1637 (`git merge-base --is-ancestor`). 86924f535c9 (2026-10-27, session 6) does.
   - So the earlier runs worked under different persistence rules than HEAD. Each device-opening process reset the uncore, and the MLA remap, local TL1 base and group hash were not restored by software; whether they survived that reset is an emulator or spec question.
2. **README line 75 is stale.** It says "CreateDevice is where UMD releases each CCE uncore and programs the remap before it loads firmware". Nothing in Metal calls `release_cce_uncore_for_firmware_load` or `program_cce_dmrisc_remap`: a grep of tt_metal/ (excluding third_party) and the tests finds no caller, and commit 8fa06c8b177 removed the Cluster wrapper. The UMD header comment at `tt-umd/api/tt-umd/tt_device/tt_device.hpp:436-442` still describes that call sequence, which no caller uses any more.

##### 1. Resets a mimir_tests process asserts on the CCEs

**How UMD treats a CCE reset.**
- `TTDevice::assert_risc_reset` and `deassert_risc_reset` (`tt-umd/src/tt_device/tt_device.cpp:782-819`) handle any DRAM core on QUASAR/GRENDEL through `PF_CTRL_RESET` (`CCE_PF_CTRL_RESET_BASE` = 0x02200000, `tt-umd/api/tt-umd/arch/grendel_implementation.hpp:328`).
- In that register, bit 0 is the uncore and bits 1-8 are harts 0-7, with 1 = released (`grendel_implementation.hpp:324-325, 337`).
- The uncore bit changes only when `selected_riscs == RiscType::ALL` exactly (`tt_device.cpp:787, 804`).
- `ALL_NEO_DMS` and `ALL_DATA_MOVEMENT` both map to hart bits 1-8 (`tt-umd/src/arch/grendel_implementation.cpp:28-35`).
- The write is a read-modify-write, so bit 0 keeps whatever value it had (`tt_device.cpp:169-190`).
- Confidence: high.

**When a tt::Cluster is constructed (every process that touches MetalContext::get_cluster(), with or without a device):**
- No INIT is sent. Metal sets `emu_skip_init = true` (`tt_metal/llrt/tt_cluster.cpp:562-575`), and `EmuTTDevice::create` sends INIT only when `send_init` is set (`tt-umd/src/tt_device/emu_tt_device.cpp:172-177`).
- `start_driver` calls `umd::Cluster::start_device` (`tt_cluster.cpp:619-626`).
  - That calls `SimulationChip::start_device`, which is empty (`tt-umd/src/simulation/simulation_chip.cpp:71`).
  - Then `deassert_resets_and_set_clock_state` runs (`tt-umd/src/cluster.cpp:1535-1556`). Inside it, `assert_risc_reset()` returns early on a package with no Tensix cores (`cluster.cpp:1100-1111`), `SimulationChip::deassert_risc_resets` is empty (`simulation_chip.cpp:162`), and `set_clock_state` does nothing on GrendelTTDevice's SimulationDeviceFirmware (`tt-umd/src/tt_device/firmware/simulation_device_firmware.hpp:44-46`).
- `Cluster::Cluster` then calls `assert_risc_reset()`, which takes the same early return (`tt_cluster.cpp:280-282, 439`).
- On a full_package descriptor that has Tensix cores, `Chip::assert_risc_reset(ALL)` and `deassert_risc_reset` loop over Tensix cores only (`tt-umd/src/chip/chip.cpp:102-114`), so the CCEs are still untouched.
- Result: no CCE reset and no CCE write at Cluster construction. Confidence: high.

**When a device is opened (CreateDevice or MeshDevice::create_unit_mesh, which is what ScopedDevice uses at `tests/tt_metal/tt_metal/dram/mimir_access.hpp:405`):**
- The open path is `MetalContext::initialize_device_manager` → `initialize` (`tt_metal/impl/context/metal_context.cpp:135-148, 150-323`).
- `initialize` returns early if the context is already initialized with the same parameters and no force-reinit (`:178-206`). So the firmware sequence below runs on the **first** open in a process, and again on each `TT_METAL_FORCE_REINIT` cycle, which first tears down (`:197-202`).
- `run_launch_phase` is gated on `!is_mock_or_emulated()` (`risc_firmware_initializer.cpp:299`). That check is true only for Mock and Emule (`tt_metal/llrt/tt_cluster.hpp:359-361`), so the gated block **does** run for EmuAxi and GrendelJtag.
- The resets, in order:
  1. `reset_cores` → `assert_dram_cores(ALL_NEO_DMS)` on every CCE (`risc_firmware_initializer.cpp:565-616`, `:610`, `:510-518`).
  2. Per CCE, `initialize_firmware` asserts `ALL_NEO_DMS` again (`:1392-1394`).
  3. After the TL1 writes, `deassert_risc_reset(ALL_NEO_DMS)` (`:1699-1701`).
- Net effect: harts held, then released. The uncore bit and every uncore register are untouched. Confidence: high.

**When a device is closed (CloseDevice or MeshDevice close):**
- No reset and no CCE write. `DeviceManager::close_devices` tears down only the Dispatch, Fabric, Profiler and CommandQueue initializers (`tt_metal/impl/device/device_manager.cpp:708-743`).
  - With slow dispatch those return early (`dispatch_kernel_initializer.cpp:95-99`).
  - Fabric returns early when TERMINATE_FABRIC is not set (`fabric_firmware_initializer.cpp:360-372`).
  - Profiler and CommandQueue write only when the profiler is on (`profiler_initializer.cpp:81-93`; `command_queue_initializer.cpp:62-76`).
- `Device::close` writes nothing to the device (`tt_metal/impl/device/device.cpp:674-709`); `invalidate_smc_dispatch_telemetry_control` is covered in section 3.
- `RiscFirmwareInitializer::teardown` is not called on this path. Confidence: high.

**At process exit (atexit):**
- The call chain is `register_handlers_locked` → `std::atexit(destroy_all_instances)` (`metal_context.cpp:509-518`) → `~MetalContext` → `teardown()` (`:536-537`) → `risc_firmware_initializer_->teardown` (`:367-369`) → `assert_cores` (`risc_firmware_initializer.cpp:346-363`) → `assert_dram_cores(ALL_DATA_MOVEMENT)` (`:618-626`).
- So the CCE harts are reset (bits 1-8 cleared) and the uncore is left as found. `l1_barrier` after it is a no-op (`simulation_chip.cpp:156`).
- `teardown()` returns at once if the context was never initialized (`metal_context.cpp:344-346`).
- The Metal `~Cluster` calls `close_device()`, which reaches `SimulationChip::close_device`, also empty (`tt_cluster.cpp:628-630`; `simulation_chip.cpp:73`).
- `~EmuTTDevice` sends no QUIT (`emu_tt_device.cpp:187-195`).
- If the process is killed, none of this runs and the harts stay released (this agrees with README line 449).
- **The exit hart reset depends on two earlier calls succeeding.**
  - `assert_cores` calls `assert_active_ethernet_cores_to_reset` and `assert_inactive_ethernet_cores` (`risc_firmware_initializer.cpp:620-623`) before `assert_dram_cores` (`:624`). Both go through `get_control_plane_()`.
  - If either throws a `std::runtime_error`, the catch at `:359-362` logs it and skips the CCE hart reset.
  - In practice they do not throw. `~MetalContext` runs `teardown()` (`metal_context.cpp:537`) before it deletes the env (`:539-541`). The control plane was already built during `initialize` (`:259`). And on Mimir there are no Ethernet cores, so the lists are empty.
  - This is the condition that README line 449 relies on.
- Confidence: high.

**A host-only process (one that uses tt::Cluster or MetalContext::instance() and never calls initialize):**
- The infrastructure asserts no reset and writes nothing. The only device activity is Cluster construction, covered above, and the atexit teardown is skipped because `initialized_` is false (`metal_context.cpp:344-346`).
- The test bodies still do their own TL1 and GDDR writes. For example, `tests/tt_metal/tt_metal/dram/test_mimir_host_tl1.cpp:236, 520-527`; and `mimir_snoop_ctl.cpp:64`, which writes CCE uncore registers through the SMC for rows 98 and 106 and restores them (README line 392).
- Confidence: high for the infrastructure; I did not audit every test body.

##### 2. Bring-up's uncore registers: answer (c)

- **The premise does not hold.** Startup's CCE reset is harts only (`ALL_NEO_DMS`), so the uncore is never reset. That is why the bring-up values survive. Neither (a) nor (b) applies.
- **Metal never reprograms these registers.** No Metal or test code calls `release_cce_uncore_for_firmware_load` or `program_cce_dmrisc_remap` (grep, zero hits outside UMD).
- **UMD has no code for three of them.** A grep of `tt-umd/` for MLA remap, group hash or local TL1 base finds nothing. Only the DMRISC remap exists (`tt_device.cpp:122-156`), and it is reachable only through `release_cce_uncore_for_firmware_load` (`:158-167`) and `deassert_risc_reset(RiscType::ALL)` (`:805-811`).
- **If some caller did pass RiscType::ALL:**
  - Assert would clear bit 0 (`tt_device.cpp:791-793`).
  - Deassert would set bit 0 and rewrite the three DMRISC remap entries from `cce_dmrisc_remap_entries` (`tt_device.cpp:805-811`; `tt-umd/api/tt-umd/coordinates/grendel_noc_address_resolver.hpp:143-154`) before releasing the harts.
  - Software would not restore the MLA remap, local TL1 base or group hash.
  - That uncore reset clears the remap rests only on the code comment at `tt_device.cpp:806-808` ("Asserting the uncore drops the remap block") and the header at `tt_device.hpp:437-441` ("The block does not answer while the uncore is in reset, so a full reset ... drops the entries"). I found no spec reference in UMD.
- Confidence: high that HEAD software never touches these registers; low on what an uncore reset would do (comment only).

**Hart reset vectors: who writes them and when.**
- Metal writes `fw_launch_addr_value` = `MEM_CCE_SRAM_LOCAL_BASE + MEM_CCE_FIRMWARE_BASE` = 0x8080 to `CCE_BOOT_HART_RESET_VECTOR` = 0x02000000 (`risc_firmware_initializer.cpp:1452-1456`; `tt_metal/llrt/hal/tt-2xx/quasar/qa_hal_dram.cpp:105-108`; `tt_metal/hw/inc/internal/tt-2xx/quasar/dev_mem_map.h:481, 494, 524-525`).
- It does this inside `initialize_firmware` for each CCE: after the hart assert at `:1393` and before the deassert at `:1700`. That is the first device open per process, plus each FORCE_REINIT cycle. It happens even with SKIP_LOADING_FW, because the write is outside that `if`.
- The write path is `Cluster::write_reg` (`tt_cluster.cpp:1066-1087`) → SimulationChip (`simulation_chip.cpp:121-129`) → `GrendelTTDevice::write_to_device_reg` → `write_to_device` (`tt-umd/src/tt_device/grendel_tt_device.cpp:79-106`).
- There, `apply_cce_reset_vector_write` intercepts it and broadcasts hart 0's value to all 8 per-hart reset-vector registers in that CCE's tt_cluster_ctrl plane (`tt_device.cpp:90-120`).
- Nothing writes them at close or exit. Confidence: high.

##### 3. CCE TL1 ranges written on device open

**Address model.**
- Offsets below are the CCE-local (hart-view) offsets, computed from `dev_mem_map.h:480-525`.
- The host NOC address is the offset plus `MEM_CCE_L1_NOC_OFFSET` = 0x20_0000_0000 (`dev_mem_map.h:480`; `qa_hal_dram.cpp:110`).
- TL1 is 4 MiB (`dev_mem_map.h:483`).
- Unless marked otherwise, each write happens once per process on the first open, and again per FORCE_REINIT cycle.

**Layout derived from the header:**

| Offset | Region |
|---|---|
| 0x0 | reserved (128 B) |
| 0x40 | watcher ring-buffer lock |
| 0x80 | mailbox (32 KiB) |
| 0x8080 | firmware text (24 KiB) |
| 0xE080 | global slots, 9 × 4 KiB (slot 0 is the firmware .data/.bss, slots 1-8 are kernel data) |
| 0x17080 | TLS/stack, 8 × 8 KiB |
| 0x27080 | init-local scratch (1 KiB) |
| 0x27480 | bank-to-NOC scratch (2 KiB) |
| 0x27C80 | SPA routes (64 B) |
| 0x27CC0 | kernel config (2 KiB) |
| 0x284C0 | kernel text, 8 × 48 KiB |
| 0x884C0 - 0x400000 | UNRESERVED |

**Host writes at open:**
- **Watcher block** (`mailboxes_t::watcher`, the full `watcher_msg_t`). `WatcherServer::init_device` writes it with `enable = WatcherDisabled` even when the watcher is off (`metal_context.cpp:266-270, 312-314`; `tt_metal/impl/debug/watcher_server.cpp:427-460, 556-566, 594-599`). This happens before the hart reset in `run_launch_phase`.
- **core_info** (full `core_info_msg_t`) (`risc_firmware_initializer.cpp:1648-1660`).
- **Bank tables** in the bank-to-NOC scratch at 0x27480: `dram_bank_to_noc_xy`, `l1_bank_to_noc_xy`, then the DRAM and L1 bank offsets. Their sizes depend on the bank counts, so this is a partial write of the 2 KiB range (`:756-847`, `:1186`).
- **Firmware image.** `get_risc_binary` uses DISCRETE loading by default (`tt_metal/llrt/llrt.hpp:35-37`), so the host writes one span per PT_LOAD segment's file bytes (`tt_metal/llrt/tt_memory.cpp:62-104`; `llrt.cpp:236-266`):
  - `.text` at 0x8080, within 0x8080-0xE080 (`qa_hal.cpp:106-113`, linker script `tt_metal/hw/toolchain/script_tng.ld:125-137`).
  - `.data` and `.rodata` at 0xE080, within slot 0 (`script_tng.ld:171-191`).
  - The `.tdata` image at 0x17080, in hart 0's TLS slot (`script_tng.ld:140-150`).
  - `.bss` and `.tbss` have no file bytes and are not written by the host.
  - No relocation applies: 0xE080 and 0x17080 do not match `MEM_LOCAL_BASE` 0x802000 (`qa_hal.cpp:591-605`; `dev_mem_map.h:43`), so the init-local scratch 0x27080-0x27480 is not used.
  - Skipped when SKIP_LOADING_FW is set (`:1395`).
- **Mailbox fields** (`:1416-1450`):
  - All 8 `launch[]` entries (`launch_msg_buffer_num_entries` = 8, `tt_metal/hw/inc/hostdev/dev_msgs.h:444`).
  - The first `go_msg_t` of `go_messages`, with signal INIT. Only `go_msg.size()` bytes are written; the other 8 of the 9 entries are not.
  - `launch_msg_rd_ptr` = 0 and `go_message_index` = 0.
  - `subordinate_sync` = 0x40404040404040 (8 B).
  - `fw_shared_globals_ready[0]` = `SHARED_GLOBALS_READY_WAIT` (0) (1 B; `dev_msgs.h:125`).
- **SPA routes** (`sizeof(CceSpaRoutes)` ≤ 64 B) at 0x27C80, the end of the bank-to-NOC scratch (`risc_firmware_initializer.cpp:1667-1676`).

**Host writes that do not happen** in this environment, with the gate that stops each:
- CLEAR_L1 would zero the full 4 MiB (`:236-240, 414-426`).
- The DPRINT server is gated off (`metal_context.cpp:261-265`).
- The profiler control buffer: `ClearProfilerControlBuffer` → `setControlBuffer` returns unless profiler state is on (`tt_metal/impl/profiler/tt_metal_profiler.cpp:80-90, 739-744`).
- The fast-dispatch launch-pointer reset (`device.cpp:462-467`) is reached only from DispatchKernelInitializer, which returns under slow dispatch.
- L1 barriers are no-ops on SimulationChip (`simulation_chip.cpp:156`), so `MEM_L1_BARRIER` (offset 12) is not written.
- SMC dispatch telemetry is not written, because SimulationDeviceFirmware reports no buffer (`tt_metal/impl/dispatch/dispatch_telemetry.cpp:35-55`; `simulation_device_firmware.hpp:99-105`).
- Confidence: high for each gate.

**What the firmware writes when it starts** (`tt_metal/hw/firmware/src/tt-2xx/drisc.cc:87-182`):
- **Hart 0:**
  - `do_crt1` zeroes `.bss` in global slot 0. The `.data` copy is skipped because the image is already in place (`drisc.cc:92-94`; `tt_metal/hw/inc/internal/firmware_common.h:53-65`).
  - It sets `fw_shared_globals_ready[0]` = GO (`:98`).
  - `noc_bank_table_init` copies the scratch tables into `.bss` (`:138`).
  - It writes `launch_msg_rd_ptr` = 0 (`:144`) and `go_messages[0].slot[0].signal` = DONE (`:146`).
- **Every hart:**
  - `do_thread_crt1` zeroes its own `.tbss` and copies `.tdata` from hart 0's image into its 8 KiB TLS slot at 0x17080 + hart × 0x2000 (`drisc.cc:100-101`; `firmware_common.h:67-77`).
  - Its stack grows down from the top of that slot (`tt_metal/hw/toolchain/crt0.S`: tp = `__local_base` + mhartid × stride, sp = tp + stride; `qa_hal.cpp:122-127`).
  - Harts 1-7 write their `subordinate_sync` byte as DONE (`drisc.cc:111-112`).
- **Not active in this environment:** `WATCHER_RING_BUFFER_INIT` (writes the lock at 0x40 and the ring head) compiles to nothing without WATCHER_ENABLED (`tt_metal/hw/inc/api/debug/ring_buffer.h:92-96`). `DEVICE_PRINT_INITIALIZE_LOCK` compiles to nothing without DEBUG_PRINT_ENABLED (`tt_metal/hw/inc/api/debug/device_print.h:173, 1854-1858`).
- **Not traced:** `noc_init` and `overlay_cmd_buff_init` are passed `MEM_NOC_ATOMIC_RET_VAL_ADDR` = 4 (`dev_mem_map.h:78`). I did not check whether they write TL1 at that address or only registers.
- Confidence: high on the code path; medium on exact byte extents, which depend on the built ELF that I did not inspect.

**Where the tests' fixed probe address falls.**
- `kCceSramTestOffset` = 0x800 (`tests/tt_metal/tt_metal/context/mimir_contract.hpp:36`) is used as a bare offset from TL1 base:
  - `test_mimir_backend.cpp:51, 79` is host-only and writes through a raw Cluster.
  - `test_mimir_emu.cpp:72` writes through `cce_sram_noc_addr` after opening a device.
- 0x800 is **inside the mailbox, in the `launch[8]` ring**, not in UNRESERVED. My estimate:
  - `launch[]` starts at or below 0xB0, after `ncrisc_halt` (8 B), `subordinate_sync` and `launch_msg_rd_ptr`, aligned to 16 (`dev_msgs.h:450-455, 380`).
  - Each Quasar `launch_msg_t` is at least about 430 B, because `MaxProcessorsPerCoreType` = 24 (`tt_metal/hw/inc/internal/tt-2xx/quasar/core_config.h:112`; `dev_msgs.h:168-215`).
  - The ring therefore ends past 0xE00, and 0x800 falls in roughly entry 4.
  - Confidence: medium on the exact entry, high that it is in `launch[]`.
- **Consequences:**
  - A value written there survives from one host-only process to the next.
  - It is wiped by the next device open's full `launch[0..7]` rewrite (`risc_firmware_initializer.cpp:1430-1434`).
  - `test_mimir_emu.cpp:72` writes into a slow-dispatch launch entry that the running firmware does not read. It reads only `launch[launch_msg_rd_ptr]`, and `rd_ptr` stays 0 except in DISPATCH_MODE_DEV (`drisc.cc:154-177`).
  - A host-only test that writes a fixed value there can read back the previous host-only process's value if its own write is lost.

**What nothing writes at open or close:**
- 0x0-0x7F, except possibly offset 4 (not traced).
- Unwritten mailbox fields: `ncrisc_halt`, `go_messages[1..8]`, `link_status_check_timestamp`, `shared_globals_ready[]`, `fw_shared_globals_ready[1..]`, `dprint_buf`, `aerisc_run_flag`, `profiler`, and the tail up to 0x8080.
- Firmware-text and slot-0 bytes beyond the image and `.bss`.
- Kernel global slots 1-8 (0xF080-0x17080).
- The unused parts of each TLS slot, below the deepest stack use.
- The init-local scratch 0x27080-0x27480.
- The bank-scratch tail beyond the tables.
- Kernel config 0x27CC0-0x284C0.
- Kernel text 0x284C0-0x884C0.
- **All of UNRESERVED, 0x884C0-0x400000**, which is where the tests' fixed-value buffers sit.
- Nothing is written to TL1 at close or exit.
- Confidence: high for UNRESERVED, kernel config and kernel text; medium for the small gaps.

##### 4. Does any software path write Mimir GDDR on device open or close?

No, not under the README environment at HEAD. Confidence: high for the paths listed; medium overall, because I did not read the allocator or MeshDevice creation exhaustively.

- `clear_dram_state` runs only with `TT_METAL_CLEAR_DRAM` (`risc_firmware_initializer.cpp:241-243, 431-441`).
- Every `dram_barrier` reaches `SimulationChip::dram_membar`, which is empty (`tt_cluster.cpp:1214-1221`; `tt-umd/src/cluster.cpp:1423-1425`; `simulation_chip.cpp:158-160`).
- The DRAM-backed command-queue pointer writes run only under fast dispatch with DRAM-backed CQs (`device.cpp:346-358`; `command_queue_initializer.cpp:79-87`; `tt_metal/impl/dispatch/system_memory_manager.cpp:914-916`). The `SystemMemoryManager` constructor itself writes no DRAM.
- The profiler is gated off (`tt_metal_profiler.cpp:86-90`; `profiler_initializer.cpp:30-77`, which also needs TRACY_ENABLE).
- The DPRINT DRAM writes (`tt_metal/impl/debug/dprint_server.cpp:938, 1273`) need DPRINT on.
- NOC-transfer logging is gated by its runtime option (`tt_metal/impl/debug/noc_logging.cpp:112-116`).
- `Device::initialize` only builds the allocator and the SHM stats (`device.cpp:553-672`); `Device::close` frees host state (`:674-709`).
- The prefetcher writes GDDR only when a test calls it.
- Chippy owns GDDR init (comment at `tt_cluster.cpp:571-573`).
- Exit writes no GDDR (section 1).

##### Table (software side only)

| | Process with a device open (start + exit) | Host-only process |
|---|---|---|
| **CCE TL1** | **Partly overwritten** on the first open. The host writes the watcher block, core_info, launch[0..7], go_messages[0], launch_msg_rd_ptr, go_message_index, subordinate_sync and fw_shared_globals_ready[0] (mailbox 0x80-0x8080); the firmware text at 0x8080; .data at 0xE080; the .tdata image at 0x17080; the bank tables at 0x27480; and the SPA routes at 0x27C80. The firmware then zeroes .bss in slot 0 and each hart's .tbss, copies .tdata into the 8 TLS slots (0x17080-0x27080), uses the top of each slot as stack, and sets a few mailbox flags. **Nothing writes** kernel config, kernel text or UNRESERVED (0x27CC0-0x400000), kernel global slots 1-8, or the init-local scratch. Nothing is written at close or exit. Evidence: sections 1 and 3. Confidence: high (byte extents medium) | **Untouched by infrastructure.** Cluster construction writes nothing and teardown is skipped (`metal_context.cpp:344-346`; `tt-umd/src/cluster.cpp:1109-1111`; `simulation_chip.cpp:71-73`). Test bodies write their own data. Confidence: high |
| **Mimir GDDR** | **Untouched** (CLEAR_DRAM unset, barriers no-ops, slow dispatch, profiler and DPRINT off; section 4). Confidence: high (medium overall, see section 4) | **Untouched by infrastructure.** Confidence: high |
| **CCE uncore registers** | `PF_CTRL_RESET`: hart bits 1-8 are cleared at open (twice), set when the firmware is released, and cleared at exit (`ALL_DATA_MOVEMENT`). Bit 0 (uncore) is preserved by the read-modify-write. The **8 hart reset vectors are overwritten** with 0x8080 at open. **DMRISC remap, MLA remap, local TL1 base and group hash are untouched**: there is no caller in Metal, and UMD has no code for the last three. A killed process skips the exit hart reset. Evidence: sections 1 and 2. Confidence: high | **Untouched by infrastructure.** The test helpers in rows 98 and 106 write uncore registers through the SMC and restore them (README line 392). Confidence: high |

##### UMD 292fe2086594 (the gitlink) compared with f0d497f3 (built)

- The gitlink has no CCE or emulator path. `tt-umd/src/tt_device/emu_tt_device.cpp` and `grendel_tt_device.cpp` do not exist there.
- Its `tt_device.cpp` has no `cce_channel` or `CCE_PF_CTRL` code, and `grendel_implementation.{hpp,cpp}` has none of the CCE reset constants or `cce_hart_release_bits`.
- Its `tt-umd/api/tt-umd/cluster.hpp` has no `EMU_AXI` or `emu_skip_init`, and `cluster.cpp` lacks `has_tensix_cores` and the early return in `assert_risc_reset`. `git diff --stat 292fe2086594 f0d497f3` shows 717 lines added across these files.
- Metal at HEAD refers to `ChipType::EMU_AXI` and `emu_skip_init` (`tt_cluster.cpp:562-575`), so it would not compile against the gitlink. That is inferred from grep; I did not build.
- `SimulationChip`'s empty start, close and membar functions are the same in both.
- Every UMD finding above is therefore specific to f0d497f3. To reproduce the built behaviour, update the gitlink to f0d497f3.

##### What I did not check

- Whether `noc_init` or `overlay_cmd_buff_init` write TL1 at offset 4.
- The exact sizes of the built ELF segments and the bank tables.
- ControlPlane construction, which `MetalContext::initialize` reaches at `:259-260`.
- MeshDevice creation and the allocator internals, beyond the grep that found no device writes.
- The `TT_METAL_SOC_DESC`-with-no-server path, which is TargetDevice::Silicon and uses LocalChip rather than SimulationChip.
- The emulator model itself, meaning whether state survives between sessions; other agents cover that.
- Test bodies, beyond noting that they write data themselves.

### C.3 Spec reset domains

#### CCE / Mimir reset domains: what the spec says

Sources read:
- [M] Mimir.GDDR7 Architecture, "Version 1.14, September 7, 2026" (Drive 1nQrV25Qk4Fb-zGV1e7bMiSewPsm4bC4w, "mimir_latest.pdf"). Pages cited are the printed "| Page N" headers. The Drive text export stopped at printed p.75, so pp.76-181 (including Appendix 15.1) were read from Glean's full copy of the same file.
- [G] Grendel CCE spec, "Grendel_cce_latest.pdf" (Drive 1ikHjbY495ioD-2YJqrMTqodCsC_aQeth), revision 0.9.2 dated 08.18.2025, marked "Pre-release version". Used because [M] §7.2 sends readers there ("Additional information on CCE can be found in Grendel CCE"), and [M] Table 53 is a Mimir subset of [G] Table 31. Its §9.2 says the register details are in "CCE.html"/"CCE.rdl"; I did not read those.
- [C] Confluence SIVAL 2451964648, "Mimir Microarchitecture Specification". This page is compiled from other sources ("reconstructed from source snippets") and states no reset domains of its own. Cited only for the bring-up order.
- [I] Confluence SoC 1173160182, "Mimir SoC Initialization Sequence", which [M] §3.5 note 3 points to. It is a DV/testbench-level sequence.
- [X] Non-spec items found through Glean (generated RDL dump, testbench and bring-up code). Each one is labeled [X] where used.

##### Q1. Every reset that reaches a CCE, and how software asserts it

- 1a. External power-on reset pin. [M] Table 8 (§3.5, p.14): "reset_n | External pin | SMC, DAP/TAP | External cold reset pin (BP_RESETN)". [M] §3.5 p.14: it "is asynchronous and resetting SMC internal reset unit, which employs memory-mapped register (MMR) to assert corresponding reset signals". It reaches the CCE only indirectly, through the SMC reset unit (1b). Software cannot assert it.
- 1b. Per-subsystem cold reset `SS_COLD_RESET_N`. [M] Table 10 (§4.2.4, p.18): "0x40 SS_COLD_RESET_N[31:0] RW all0 Each subsystem Cold reset signal". With reset value all0 (active low), every subsystem is held in cold reset after power-on until SMC firmware writes this register. [M] Table 8: "Used to cold reset a subsystem - will reset the subsystem's reset and clock control block". The SMC Reset Unit base address is not given in the spec text I read; only the 0x40 offset is.
- 1c. CCE bits of `SS_COLD_RESET_N`. [M] Table 11 (p.19): "[8] CCE Tile 0 Cold Reset | [9] CCE Tile 0 Debug Module Cold Reset | [10] CCE Tile 0 SRAM (TL1) Cold Reset | [12] CCE Tile 1 Cold Reset | [13] CCE Tile 1 Debug Module Cold Reset | [14] CCE Tile 1 SRAM (TL1) Cold Reset". [M] Table 53 (§7.4.2, p.61) maps these: "cce cold reset | ss_cold_reset_n in SMC" and "tl1 cold reset | ss_cold_reset_n in SMC". Revision 0.9.25 added "CCE debug module reset via SMC cold reset register". The debug-module reset (bits 9/13) does not appear in Table 53.
- 1d. Rule against partial cold reset. [M] §3.5 p.14: "To avoid any unexpected side effect, it is required to cold reset the entire subsystem together despite the capability of resetting a subset of the subsystems by controlling the bit positions that way." Read literally, toggling bit 8 without bit 10 (or the reverse) goes against the spec's own guidance.
- 1e. SMC warm reset is not supported. [M] Table 10: "0x44 SS_WARM_RESET_N[31:0] RW all1 N/A Warm reset signal"; note under Table 10: "Mimir enables only SS_COLD_RESET_N". [M] §3.5 p.14: "the first generation of Grendel chiplets including Mimir does not support warm reset". [M] Table 86 (§15.1, p.131): ss_warm_reset_n, the hold signals, ss_reset_complete_n and ss_force_to_ref_clk are all "NOT supported".
- 1f. CCE uncore warm reset. [M] Table 53: "DMRISC uncore | cce uncore warm reset | Reset register in CCE | MCP Yes" and "AXI-DMA | cce uncore warm reset | Reset register in CCE | Yes". [G] §7.3.2 p.39: "CCE IP supports 1 uncore warm reset and 8 core warm resets, driven by a register in CCE reset register (CCE base + 0x20_0000)." [G] §7.3.3 step 3: "Write to uncore reset bit in CCE reset register to de-assert uncore warm reset." The spec does not give the bit position.
- 1g. CCE core (hart) warm resets x8. [M] Table 53: "DMRISC core | 8 cce core warm resets | Reset register in CCE | MCP No". [G] Table 31 (p.39) also lists "Accelerators | 8 core warm resets, 1 per core" and "Context-switch accelerator | OR of 8 core warm resets". [G] §7.3.3 step 4: "Write to core reset bits in reset control register to de-assert core warm resets ... CCE base + 0x20_0000." The spec does not give the bit positions.
- 1h. Register name and bits (not in the spec) [X]. The generated RDL dump `chippy validation/jtag/helper/scripts/raw_files/cce_1_pf_ctrl_raw.txt` ("Generated from ... cce/rdl/cce_ctrl.rdl") lists `{0x00000000, 64, 1, 0x0, 0x1FF, ...} // reset (reset)` with "fields: uncore_reset, core_reset". Testbench `cce_reset_common.py` documents "pf_ctrl.reset value: uncore_reset[0] + core_reset[8:1], 1 = released", and chippy `cce.py` documents "uncore_reset=1 — uncore out of reset". So the register is `pf_ctrl.reset` at offset 0, reset value 0x0 (uncore and all harts held in reset), active-high release. This agrees with [G]'s wording that writing the bits de-asserts reset, but it comes from code, not the spec.
- 1i. Inconsistency inside [G] about the reset register location. [G] Table 2 (§3.1, p.7) assigns "CCE control | CCE base + 0x20_0000 | Reserved for software-defined registers ... 128 64b registers". [G] §7.3.2/§7.3.3 put the reset register at the same address, "CCE base + 0x20_0000". Not reconciled in the spec.
- 1j. Inconsistency inside [G] about who drives the Address Remapper reset. [G] Table 31 has "Address Remapper | uncore warm reset | Reset from SMC | Yes", directly under "AXI-DMA | uncore warm reset | Reset register in CCE | Yes". The table names an uncore warm reset but says the SMC drives it. [M] Table 53 has no Address Remapper row. Not resolved.
- 1k. CCE IP view of its own reset input. [G] §7.1 p.36 "External interfaces: 1. Reset - cold reset". [G] §7.3.2: "CCE IP has a single reset input, uncore cold reset ... A reset synchronizer is added in each". In Mimir this input is the "cce cold reset" from SS_COLD_RESET_N bit 8/12 ([M] Table 53).
- 1l. Other resets. [M] §3.5 p.15, SMC watchdog: "If its second stage WDT triggers or SEP's WDT triggers, a reset will be sent to both CPU cores". That names the SMC CPU cores; whether it reaches the CCE is not stated. [M] Table 8: "SMC | SMC | Special, debug only warm reset flow to handle SMC processor hang scenarios"; effect on the CCE not stated. [M] Table 8 note: "JTAG interface also has a TAP reset (TRST) pin which is not listed above"; effect on the CCE not stated. [G] §4.2 p.22 (Athena CCE) "Receive Function-Level Reset (FLR) requests ... drive reset requests to SMC". This applies to Athena PCIe only, since Mimir has [M] Table 52 "EN_VF 0" and is not PCIe.
- 1m. Order of reset release. [M] §7.4.3 (p.61-62): "1. De-assert cce and tl1 cold resets: This allows access to all CCE registers and Banked SRAM via CCE's AXI Snoop block. 2. De-assert cce uncore warm reset: This brings CCE IP's DMRISC uncore, AXI-DMA, address remaps, etc. out of reset. 3. De-assert cce core warm resets". [G] §7.3.3 adds a step between cold and uncore: "2. Write DMRISC core reset vectors in cluster reset vector registers ... CCE base + i*0x8, i = 0,…,7." [C] §C gives this order: "release_cold_reset ... program_cce_mla_remap_entries(); program_local_tl1_base(...); ... program_cce_reset_vector(...); release_uncore_reset(); release_core_reset();".

##### Q2. Which registers each reset returns to defaults

The specs give no per-register reset values. [G] §9.2: "CCE global MMR can be found here in html format: CCE.html ... rdl format: CCE.rdl" (not read). Below, domain membership is taken from the tables; where a register is not listed by name, the line says "inferred".

- 2a. Rule for all CCE IP registers. [G] §7.3.2 p.39: "Note that all registers, and register network in CCE are in the uncore cold reset domain allowing accessing reset vectors and error banks, while other CCE IP components are in reset." [G] Table 31: "Register blocks | uncore cold reset | Reset from SMC". [M] Table 53 has no "Register blocks" row. It lists "Register xbar | cce cold reset" and §7.4.3 step 1 says cold de-assert "allows access to all CCE registers". By [G], CCE IP register contents are reset only by the cce cold reset (SS_COLD_RESET_N bit 8/12), not by the uncore or core warm resets.
- 2b. Hart reset-vector registers (`tt_cluster_ctrl.reset_vector[i]`, CCE base + i*0x8). These are register blocks, so they are in the cold domain per 2a. Supporting this, [G] §7.3.3 has software write them after the cold de-assert and before the uncore warm de-assert. Cold reset: returns them to default. Uncore warm and core warm: they are not in those domains (inferred from 2a, not stated per register). Reset value: not stated in the spec. [G] §8.3 p.50: "Each DMRISC core reads the reset vector from cluster control register in CCE global CSR ... CCE reset is TBD."
- 2c. DMRISC address remap (CCE base + 0x20_2000, [G] Table 2 "DMRISC Address remap | Address remapper configuration registers"). The configuration registers are register blocks and so in the cold domain (2a, inferred). The remap logic is in the uncore warm domain: [G] Table 31 "Address Remapper | uncore warm reset"; [M] §7.4.3 step 2 "address remaps, etc. out of reset". Neither spec says whether uncore warm reset clears the remap configuration or only the datapath. Reset value: not stated.
- 2d. MLA remap and local TL1 base, the "Mimir CCE control registers". [M] §7.2.5 p.60: "Mimir CCE control registers are used to drive control signals Group ID tagger, MLA Remap, and local SRAM base address used by CCE xbar". [M] §7.2.3 places the MLA remap block and CCE xbar in the "Mimir CCE network", and [M] Table 53 has "Mimir CCE network | cce cold reset | ss_cold_reset_n in SMC". So by Table 53 row these are cleared by the cce cold reset only, not by either warm reset (inferred; the control registers are not listed by name). Reset values: not stated. The Group ID tag works the same way: [M] §7.2.4 "SMC can program a Group ID tag register in Mimir CCE control register block ... CCE cores don't have access".
- 2e. Group hash (TL1 CSR). [M] §7.3.2 p.61: "TL1 CSRs hold control and status registers for the TL1 IP, including hashing function ... accessible via the wrapper register interface in CCE IP ... treated as CCE internal registers". [M] §7.3 makes TL1 CSR part of "Banked SRAM", and Table 53 has "Banked SRAM | tl1 cold reset | ss_cold_reset_n in SMC". So by Table 53 row the hash is cleared by the tl1 cold reset (bit 10/14), which is a different bit from the CCE tile cold reset (bit 8/12), and by neither warm reset (inferred). Reset value: not stated. The register name "TL1_CSR_GROUP_HASH_FN0_MASK" comes from code [X], not the spec.
- 2f. `pf_ctrl` (the reset register itself). By 2a it is a register block, so only the cold reset resets it (inferred). It is plainly not reset by the warm resets it drives, but the spec does not say so. Reset value: not stated in the spec. [X] RDL dump: 0x0, mask 0x1FF.
- 2g. iDMA / AXI-DMA state. [M] Table 53: "AXI-DMA | cce uncore warm reset | Reset register in CCE | Yes"; [G] Table 31 says the same. So uncore warm reset (and cold reset) resets the AXI-DMA. Hart warm reset does not. The AXI-DMA programming registers at "CCE base + 0x2000" ([G] Table 2, "Sideband interface to program AXI-DMA") are not separately assigned to a domain. Reset values: not stated.
- 2h. Hart warm reset. It covers only the DMRISC core, accelerators and context-switch accelerator ([G] Table 31). None of the seven items above is in that domain. Not stated per item.
- 2i. Possible contradiction from testbench code [X]. `mimir_soc_ai cce_reset_order.py` comment: "re-asserting the uncore discards configuration firmware programmed after stage 2". That would conflict with 2a if the configuration lives in register blocks. This is testbench commentary, not spec. I recommend the Metal/UMD or emulator agent check it against RTL.

##### Q3. Is TL1 or GDDR in a reset domain, and does anything clear or initialize them?

- 3a. TL1 (4 MB) is in a reset domain. [M] Table 53: "Banked SRAM | tl1 cold reset | ss_cold_reset_n in SMC | Yes"; Table 11 "[10]/[14] CCE Tile n SRAM (TL1) Cold Reset". [M] §7.3.1 p.60: "The TL1 IP is configured to be 4MB in size, with 256 sub-banks". TL1 is in neither the uncore warm nor the core warm domain (Table 53), so by domain membership, not by an explicit statement, neither warm reset touches TL1.
- 3b. Whether any reset clears, scrubs or ECC-initializes the TL1 array: not stated. Neither [M] nor [G] says what tl1 cold reset does to array contents. [M] §7.3.2 only says "TL1 EDC block is used to detect errors in TL1 internal SRAMs". No TL1 init or scrub FSM is described.
- 3c. The SRAM-init text in Table 86 is not Mimir behavior. [M] §15.1 p.131 does describe "reset FSMs (for example to initialize SRAMs)" and "SRAM MBIST and any other self checking initializations", and "hold signals ... are used to prevent certain registers and SRAMs from being cleared by the warm reset". That text belongs to the Grendel warm-reset/hold handshake, which Table 86 marks "NOT supported" in Mimir. It does not describe what Mimir does.
- 3d. L1 D$, L1 I$ and L2. [G] §3.2 p.9: "The L1 I-cache supports parity, while the L1 D-cache supports SECDED ECC. All DMRISC cores share a 128KB L2 cache ... supports parity." Which reset domain holds the cache arrays is not stated ([G] Table 31 has only "DMRISC core" for core warm and "DMRISC uncore" for uncore warm). Whether any reset or init sequence invalidates, scrubs or ECC-initializes them: not stated. The only invalidation text is the RAS response, [G] §10.1 p.55 "Response (Logic): 1. Invalidate entire array/structure 2. Replay/re-issue request" on a detected error.
- 3e. Mimir GDDR contents. They are not in any CCE reset domain. The GDDR7 tile has its own cold reset bits: [M] Table 11 "[24] GDDR7 Tile 0 Cold Reset ... [28] GDDR7 Tile 1 Cold Reset". Whether DRAM contents survive a GDDR7 tile cold reset: not stated. Whether any init clears or scrubs DRAM: not stated. ECC-related facts: [M] Table 55 p.64, "Error Check and Scrub (Auto ECS) ◦ Auto ECS will use On-die ECC and operate in the background to detect, correct and write back corrected data to the array"; [M] §8.2 p.65, "In-line ECC is not supported by the Rambus GDDR7 Controller Core ... the on-die ECC is considered strong enough". No in-line ECC means no controller-side ECC initialization is needed; the spec does not say this directly.
- 3f. GDDR controller and PHY. [M] §8.3 p.74: "PHY initialization: PHY initialization is required after every reset as the PLL calibration is required." [M] §3.5 note 4 p.15: "the PHY asserts the dfi_init_complete signal. This indicates that the PHY and memory devices are ready to accept commands." Controller CSR behavior across reset: not stated (it defers to [Rambus User Guide], not read). [C] §D.1 (derived) lists "Hold controller/PHY in safe reset; program Rambus controller over APB", then "Execute JEDEC figure 12 power-up and figure 13 reset sequences".
- 3g. The init-sequence page says nothing about scrubbing. [I] covers cold reset removal, strap programming, remap/BAR/NoC2AXI tables and D2D training. It has no TL1 or DRAM clear, scrub or ECC-init step. Its only memory line is DV-only: "SRAM/DRAM Preload: Testbench loads memory contents from hex files using $readmemh. ECC and data are loaded separately for IRAM and DRAM."

##### Q4. Asserting an uncore or cold reset without lowering the AXI clock

- 4a. The requirement. [M] §7.4.2 p.61: "The cce cold reset, tl1 cold reset and cce uncore warm reset are all multi-cycled i.e. they cannot be asserted in mission-mode and need the AXI clock to be lowered before asserting and de-asserting the reset." [G] §7.3.2 p.39 says the same with "uncore clock" in place of "AXI clock".
- 4b. The consequence of not lowering the clock: not stated in [M] or [G]. Neither spec gives the failure mode (metastability, partial reset, hang, corruption).
- 4c. Nearby text that suggests a reason but is not a statement about this case. [M] Table 86 p.131: ss_force_to_ref_clk "forces the GCDN for that subsystem's clock to run on the 100MHz reference clock. This is required to ensure reset deassertion can meet timing." It is marked "NOT supported" in Mimir. [M] §15.1 p.132: SMC can "force all the clocks that are generated from SMC PLLs, to refclk ... The selection between PLL clock output and refclk can be controlled via CGM_MUX_SELECT and CGM_AG_MUX_SELECT registers." Separately, [X] chippy `cce.py cce_config_clock` writes `smc.registers.reset_unit.ss_force_to_ref_clk`, a register Table 86 calls unsupported in Mimir. This is a code-vs-spec mismatch for the lead to note, not mine to resolve.
- 4d. Core warm resets have no clock requirement but do have an idle requirement. [M] §7.4.2: "the cce core warm resets are not multi-cycled and can be asserted in mission-mode. In order to put one or more DMRISC cores in reset, the cores first have to be brought to an idle condition (e.g. executed wfi instruction)". What happens if a non-idle core is reset: not stated.

##### Q5. Retention of memory or register contents across warm reset

- 5a. The only explicit retention mechanism is not supported in Mimir. [M] §15.1 p.131: "The hold signals which the SMC may assert to each subsystem, along with a warm_reset_n assertion to that subsystem, are used to prevent certain registers and SRAMs from being cleared by the warm reset." Table 86 lists ss_sram_hold "SRAM arrays", ss_configuration_state_hold, ss_critical_signal_hold and ss_debug_hold, all "NOT supported" for Mimir. So the spec gives Mimir no warm-reset retention mechanism at the SMC level.
- 5b. The only positive statement about CCE contents surviving a reset is [G] §7.3.2: "all registers, and register network in CCE are in the uncore cold reset domain allowing accessing reset vectors and error banks, while other CCE IP components are in reset". [G] Table 31 also puts the "RERI Error block" in uncore cold. So, by [G] and not stated in [M], register contents and error banks survive uncore and core warm resets. Survival of TL1 and DRAM contents across warm resets follows from Table 53 domain membership only (3a); no sentence says it.
- 5c. [M] §3.5 p.14: "debug and RAS information can still be obtained as long as the path to debug and RAS register through SMN is alive, despite the absence of warm reset capability."
- 5d. GDDR retention is described only for a low-power mode, not for any reset. [M] Table 64 p.75: "01 Retention Deep Sleep Mode (HSR) | RESET_N 1 | CKE_N 1 | Hi-Z"; §8.3 p.75: "During low power Retention Deep Sleep mode, the DRAM is put to Hibernate Self-Refresh (HSR) Sleep state, which requires CA0 and RESET_N to be stable high." DRAM retention across a GDDR7 tile cold reset or a CCE reset: not stated.

##### Coverage
- [M]: read §3.5, §4.2.4 Tables 10/11, all of Ch.7 (§7.1 to §7.4.3), §8.3 PHY init and retention (pp.74-75), Ch.14, §15.1 (pp.130-132) in full. Found the rest of the document by keyword search only (reset, scrub, ECC, init, retention, TL1, MBIST, remap, hash). The GDDR controller chapter (pp.63-70) was keyword-searched, not read line by line.
- [G]: read Table 2, §3.2 memory subsystem, §7.3 Clock/Reset (Table 31, §7.3.2, §7.3.3), §8.3 Boot and §9 in full; keyword-searched the rest. The Drive text export ended at printed p.75 of about 90; Appendix B and later were not checked.
- Not read: CCE.rdl / CCE.html (per-register reset values), the Grendel Reset Specification, the Rambus GDDR7 Controller User Guide, the Cadence PHY User Guide, mimir_boot_api.h.

### C.4 Recorded runs and discussions

#### Evidence: Mimir state across processes, runs and bring-up

Read-only. Nothing was run, edited or connected to an emulator. Quotes come from files I read and from search results. Slack, Glean and Atlassian hits were read as search snippets, except the two Jira tickets, which I read through JQL with comments. No Confluence page was opened in full.

Repo paths: README = `/localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/dram/README.md`. Test dir = `/localdev/sagarwal/tt-metal/tests/tt_metal/tt_metal/dram/`. Grendel = `/proj_sw/user_dev/sagarwal/grendelemulation`. Chippy = `/proj_sw/user_dev/sagarwal/chippy`.

##### Step 1: recorded results and rules about state left by an earlier process or bring-up

- Run order and invocation:
  - "Each row of the run order is a separate invocation, and the order matters." README:3
  - Env unset list: `TT_METAL_CLEAR_DRAM` README:20, `TT_METAL_CLEAR_L1` README:21, `TT_METAL_FORCE_REINIT` README:24.
  - "Rows 2c, 2d and 125 set the variables named in their rows. Every other row leaves them unset." README:28. The README does not say why CLEAR_L1 matters.
  - "Never set `TT_METAL_CLEAR_DRAM`. It writes every view in full, 1 GiB per Mimir, which is far too slow on the emulator." README:455
- Row 2c: "The test zeroes the firmware window in TL1 itself" README:55. Row 2d: "Firmware reset, reload and relaunch on every device open, three cycles" README:56.
- "The CCE DMRISC remap is compared with UMD's values once per process, right after the first CreateDevice." README:75.
- "Run them after a fresh session and bring-up (see Fresh-session rule), one test per invocation, in this order." README:82 (rows 17 onward).
- Rows that resume from state the earlier test left in TL1:
  - Row 34, `MimirPrefetcher.MM7GcbSmokeRelaunchResumes`: "A second sender and validator pair resumes from the pointers the first left in TL1" README:103.
  - Row 36, `MimirPrefetcher.MM7LifecycleStartStopTwice`: "A second Start zeroes the control block the first session left; an unknown command left in ring slot 0 is never read" README:105.
- Row ordering driven by state a case leaves behind:
  - Row 75: "Run `Op/L1InvalidateFull` last: the whole-L1 D$ discard can drop the kernel's own stack" README:297.
  - Row 96: "A wipe that never reads 0 needs a fresh session, so run this row last" README:351.
  - Row 96 result: "the discard leaves TL1 old (0 of 64 new words) and the wipe turns 64 stale cached words fresh." README:439.
  - Row 109 describes "how many stale words the hart's cache still held from earlier launches before each leg drops them" README:364. Its result notes say "the snoop leg first read lines the invalidate leg had left cached; the host's re-seed of TL1 is not snooped" README:408.
  - Rows 98 and 106: "write CCE uncore registers from the host through the SMC (RegionAttrs bit 30 on remap entry 1; the snoop-master config). Each guard restores the value it found." README:392.
- Host-deadline rule (MIM-601):
  - "New rule: after any row that ends on a host deadline, start a fresh emulator session and run bring-up before the next row (MIM-601)." README:290.
  - Rows 81 onward: "...after any host deadline or any kernel timeout code start a fresh session and run bring-up before the next row (MIM-601)." README:327.
- Row 125 (FORCE_REINIT): filter `MM43/MimirTridReset.NoStaleIdAfterReinit/*`, "All 16 pending counts read 0 after a hart reinit, then every trid completes a command" README:380.
- Session results that depend on bring-up or session state:
  - "It ran in six emulator sessions, each after bring-up ... Each ended when the session ended." README:415.
  - "Only row 106's failures needed a fresh session; each was followed by bring-up, and `SnoopConfig` and an iDMA case passed before the next case." README:415.
  - "After each case that stopped for a fresh session, bring-up was rerun and `SnoopConfig` and an iDMA case passed before the next case." README:435.
  - Row 106 failures: "...the case stopped for a fresh session" README:407, :424, :441. The failure itself is "a sibling iDMA write was acked but never became visible in TL1" README:424. The README does not tie it to earlier state.
  - A probe row: "hang (no status in 400 s; session wedged)" README:267.
  - Cause unknown: "...both passed on soc-zebu-04, so they are recorded as that session's failures, cause not known." README:151.
- Fresh-session rule, README:445-451:
  - README:447 "Before a full run, start a fresh emulator session and run chippy bring-up."
  - README:449 "Run the host suite (row 1) right after bring-up. The host suites skip when any CCE hart is out of reset. A process that opened a device resets the CCE harts on clean exit and leaves the uncore released, as bring-up does, so row 1 also runs after such a process. After a killed process the harts may still be running, and row 1 skips."
  - README:451 "If a run is killed mid-stall, the session is probably wedged. Use the same rule: a fresh session plus bring-up."
- Host suites stay on `Cluster`, not on contract helpers: "The host suites (`MimirHost*`), the firmware-load and reset-state checks, and the register reads ... stay on `Cluster`. They test the backend boundary itself, and no contract helper covers register reads." README:457.
- Code comments with the same rule:
  - `dram/mimir_dm_common.hpp:39-40` `kFreshSessionNote`: ". The kernel may still be running; start a fresh emulator session before the next run (MIM-601)."
  - `dram/test_mimir_atomics.cpp:145` and `dram/test_mimir_trid.cpp:150`: "A timeout code may leave a command in flight, so the next run needs a fresh session (MIM-601)."
  - `dram/test_mimir_dm_cache.cpp:759`: "Set when a case ends in a state only a fresh emulator session clears; later MM41 cases in this process skip." Same file :767: "An earlier MM41 case in this process left the CCE needing a fresh emulator session (MIM-601)." This is in-process state.
  - `dram/test_mimir_fw_load.cpp:171`: "An earlier process's firmware stays in TL1, and the released harts would run it." `zero_firmware_windows` (:172-183) first asserts `uncore_released` (:175-176, "uncore is in reset, so TL1 is unreachable") and then zeroes the firmware window in TL1 (:178-182).
  - `dram/test_mimir_fw_load.cpp:185-186`: "Defined first so that it is the first device open of its invocation: a later CreateDevice in the same process does not reload firmware."
- Reviewer claim, not a run record: `mimir_test_review.md:741` for `HartZeroRunsDramKernel.WritesMagic`: "TL1 is cleared only under `TT_METAL_CLEAR_L1` (unset). Example: ... a magic left by an earlier run in the same session passes | No on a fresh session; yes on a rerun | DEFECT (stale data)".

##### Step 2: do the two tests span processes or reopen in one process?

- `ReconnectPersists` reopens inside one process. It does not span two OS processes.
  - `dram/test_mimir_host_dram.cpp:429` `TEST(MimirHostDram, ReconnectPersists)`.
  - The writer is `Cluster writer(writer_options);` at :463, inside a scope that closes at :487.
  - The reader is `Cluster reader(reader_options);` at :490, same TEST body.
  - The MetalContext cluster is used at :436.
  - README:52 puts `*MimirHost*` (row 1) right after bring-up.
  - Teardown: `EmuTTDevice::~EmuTTDevice() = default;` at `tt_metal/third_party/umd/tt-umd/src/tt_device/emu_tt_device.cpp:195`. The comment at :187-191 says teardown deliberately does not send QUIT. This only applies if the test's `Cluster` takes the EmuAxi path. I did not trace which path the test's `Cluster` takes.
  - INIT: `tt-umd/src/cluster.cpp:722` passes `!options.emu_skip_init`. `tt_metal/llrt/tt_cluster.cpp:570-574` sets `.emu_skip_init = true` on the EmuAxi branch, with the comment "INIT on the SiVal server re-asserts reset and drops that bring-up." `emu_tt_device.cpp:172-175`: "On the SiVal servers INIT re-runs the reset phase, which drops any bring-up an earlier client performed, so it is suppressed when the caller attaches to an already-brought-up model."
  - The review at `mimir_test_review.md:541` says the same: "No reset and no real reconnect. emu_skip_init=true, so no INIT; ~EmuTTDevice closes no socket".
- `FirmwareKeepsRunningAcrossReopen` is not in `test_mimir_fw_load.cpp`. A grep of the dram dir and the repo finds it only at `dram/test_mimir_kernel_load.cpp:306`, as `TEST_P(MimirKernelLoad, ...)`. It reopens inside one process.
  - Launches at :324-326.
  - `device_guard.close();` at :329.
  - `ScopedDevice reopened_guard;` at :330.
  - Launch on the reopened device at :335.
  - Comment at :302-305: "CloseDevice then CreateDevice in one process reloads and resets nothing: MetalContext::initialize returns early on unchanged parameters (metal_context.cpp:203-205) and CCE harts are reset only at MetalContext teardown."
  - Early return: `tt_metal/impl/context/metal_context.cpp:196-204`. `if (force_reinit_) {...} else { return; }`, with the `return;` at :204.
  - The test checks an echo per launch. `mimir_test_review.md:739`: "Nothing checks that firmware actually kept running ... (passes the same if reopen reloaded firmware)".

##### Step 3: grendel and chippy scripts, logs and docs

- `models/mimir/tests/test_gddr.py:162-163` (inside the `test_gddr_all_pages` docstring, :143-166): "The ZeBu sparse page table is not cleared between tests, so this walk can overflow if earlier tests already allocated pages."
  - :151 skip reason: "opt-in sparse-page walk (session-global ZeBu page table); pass +gddr_all_pages to run".
  - :171 `tb.init_and_reset()` in that test.
  - :189 `@pytest.mark.skip(reason="Uncatchable page overflow error")`.
- `models/mimir/tests/run_axi_reg_dump_multipass.py`:
  - :89-95 loads rows from `zrun.11/mimir_axi_reg_dump_report.json` ("Loaded {len(rows)} rows from {prior}"), and :95 and :110 treat earlier `ok` AXI rows as already done. Values from an earlier emulator run therefore go into the new report with no session check. The script has no comment about this.
  - :118-121 each pass calls `start_probe()`, which starts a new `emu run ... test_axi --probe` (:49-58). :56-63 kill and restart on a hang, :81-82 `kill_zrun`. Each pass is a new emulator run. The script does not read back to test persistence across passes.
  - :18 `REPO` points at `/proj_soc/zebu/user/gpowley/grendelemulation`, not sagarwal's checkout.
- `py/tb/grendel/cce.py:107-111`: "If full init flow is enabled, zero the SRAM before loading the firmware." Zeroing 0x10000 bytes runs only if `self.using_init_flow`. I did not check where that flag is set.
- `models/mimir/tests/run_cce_via_sival.py:325`: "Phase 9b: Zero SRAM + load firmware". The script zeroes SRAM before loading firmware.
- `py/tb/discover/limitations.py:66-73`, entry `cce-soft-reload`, status "open", models include "mimir":
  - :70 "Reloading CCE firmware mid-session does not work with preload + release_reset alone."
  - :71 "The old program keeps running (warm uncore caches), or after an uncore pulse the remapper state is gone and CSR/SRAM windows read zero."
  - :73 "Start a new boot, or pulse uncore reset and redo boot-equivalent CCE bring-up."
- `docs/dx/WRITE-VISIBILITY-AND-ORDERING.md:186-187` (CCE, module not named): "a core-only reset leaves the CCE's caches holding the old program and old parameters -- verified by reading SRAM back byte-equal to the new image while a poisoned scratch word stayed poisoned."
- `docs/dx/WRITE-VISIBILITY-AND-ORDERING.md:190-193`: "Do not pulse the uncore to work around that. It clears the caches but the DMRISC remap that boot firmware programs is not restored by the testbench's `setup_remapper()`, after which the core reads its own die's window as all zeros -- which looks exactly like catastrophic data loss."
- `docs/att-smc-fw.md:628-629` and :637-638 (section 6.3, Keraunos, not Mimir): "the shared per-tile TL1 hash CSRs and the DMRISC remap registers do not respond until at least one CCE's uncore in that tile is out of reset." and "Core resets are left asserted so the testbench can still preload each CCE's image before releasing them."
- `validation/firmware/libs/chip_enable/README.md:19-25` (chippy): "`cfg.noc` is opt-in ... Calling `chip_init()` first leaves `cfg.noc` zeroed: no error, no log, and a GDDR address map still at its reset value. The first access routed through the ITN into that chiplet then never returns." This is the bring-up ordering rule for GDDR. It is not about retained state across runs.
- `validation/.claude/runbooks/emu-debugging.md:136` (precedent cross-ref): "Address decode misprogrammed ... Precedent: `chip_init()` called with a zeroed `cfg.noc`, so ITN2DDR ne...".
- `validation/.claude/rules/emulation.md:361`: "**Bad reset vector**: reset_vector pointed at the SoC-global SRAM view ... instead of the CORE-LOCAL view ... Symptom: `Firmware reported FAIL (postcode=0xDEADBEEF)` immediately, with `FwLog sequence=0 total_bytes=0` (firmware trapped before reaching `fw_log_init()`)." This is about which view the reset vector uses, not about retained contents.
- `docs/chippy/gdb-server.md:189-194`: "The load target's connect resets *and halts all* harts ... starting that session re-holds every hart at its reset vector and throws a mid-execution hart back to reset. That is required before `load`: attaching on its own halts only the selected hart, leaving the others running while `load` rewrites SRAM under them."
- `docs/GLOBAL_ADDRESSING_COMPONENTS.md:356-357`: "GDDR is a sparse behavioral memory in emulation; no controller/PHY init is needed." No retention statement.
- Not persistence (tooling state): `validation/.claude/runbooks/emu-debugging.md:15-16`: "Stale entries in `silval_server_info.json` from previous runs are normal -- `run_validation_test.py` overwrites the entry on each run."
- `validation/.claude/skills/run-emu/SKILL.md:444-448` (the `_backdoor` variant, test not named): "The test sets reset vectors to empty SPM, starts the cores, then polls scratch for a PASS postcode that nothing will ever write." Shows the test sets reset vectors itself, not that they persist.
- Coverage notes for this step:
  - The first README grep was truncated at 25 of 69 hits. I re-grepped with a wider term list and read the rest (lines above). The rtk tee log did not return the rest.
  - The grendel grep returned 663 matches and I read the first 150.
  - The chippy grep over `validation/` and `docs/` returned about 23,700 matches, mostly in `.venv` and vendored packages. I read the first 150 lines of output. My first command against `validation/.claude` did not run because of a wrong cwd. I then grepped `validation/.claude` directly, and the results are above.
  - About 40 candidate files from the chippy `grep -l` were not opened. I opened chip_enable/README.md, emulation.md, emu-debugging.md, run-emu SKILL.md and gdb-server.md.
  - Other hits, such as the Quasar `docs/dx/BRINGUP.md:287-293` "no later op inherits a stale count", are not Mimir.

##### Step 4: Slack, Glean, Atlassian

Relevant hits:

1. Samarth Agarwal DM with Kevin Stevens, 2026-10-09 (the user's own message). https://tenstorrent.enterprise.slack.com/archives/D09CCEBH2JX/p1791576178468939
   - "Pages are never freed. The emulator's own test, models/mimir/tests/test_gddr.py:162, says 'The ZeBu sparse page table is not cleared between tests'. That holds even though each test starts with a DUT reset (tb.init_and_reset())."
   - The same message cites the skip at test_gddr.py:186. In the copy I read, it is at :189.
   - Context message, ts 1791575653.888369, no permalink captured: "the host-side tests had to be run before any other tests seems like it was a Metal issue during teardown."
2. Kevin Stevens DM, 2026-10-15. https://tenstorrent.enterprise.slack.com/archives/D09CCEBH2JX/p1792081047354419
   - "if we restrict the sizes to the actual backing memory size in each lane, pages won't be reused."
   - "I'll follow up with the team & HW people to decide how we actually want to handle the Mimir uncore reset."
3. #mimir-emulation, 2025-10-19, Paul Zehr; reply by George Powley. https://tenstorrent.enterprise.slack.com/archives/C08P9FRP5BR/p1760924625837069?thread_ts=1760924625.837069&cid=C08P9FRP5BR
   - Log: "Error due to Page OverFlow for memory tb_emu.axi_itn_to_mem.gen_mem_channel_0__gen_mem_slice_0__axi_sparse_memory.uRAM0.mem"
   - George Powley: "Yes, there is a limited amount of memory available in the sparse memory model."
4. Jira MIM-601, "A CCE tile cold reset under in-flight ITN traffic permanently wedges the ITN path to that tile: no timeout, no tracker flush, no error, iDMA reports done". Status "In Review". https://tenstorrent.atlassian.net/browse/MIM-601
   - Description: "Only a full reset of the application network clears it."
   - Same description: "the reset clears those FIFOs. So while in reset it kept accepting ... and then forgot them. After release it remembers nothing." This is about the fabric FIFOs, not TL1 or GDDR contents.
   - Nothing in MIM-601 says whether TL1, GDDR or CCE register contents survive a new emulator session. It is the ticket behind the host-deadline rule.
5. Jira MIM-628, "CCE iDMA sideband: done_id advances when the descriptor is accepted, not when the transfer completes; software reading the destination after done_id sees old data". Status "Future Fix". https://tenstorrent.atlassian.net/browse/MIM-628
   - Comment, Stephen Co, 2026-10-27: "the sideband done-before-data window stretches to microseconds when the GDDR target is waking up ... An SMC read in between got the previous job's data from TL1."
   - This is stale TL1 data from an earlier job in the same session. It is not about a new process.
   - Row 118 of the README cites MIM-628 at README:443.
6. #mm-bringup-itn-cce-frequency-debug, thread root 2026-10-06, Subha Vaidyanathan. https://tenstorrent.enterprise.slack.com/archives/C0C4UAZKETS/p1791343841665629?thread_ts=1791343841.665629&cid=C0C4UAZKETS
   - Eric Richards, 2026-10-07: "Yes, there is a requirement that CCE uncore is brought out of reset on refclk." https://tenstorrent.enterprise.slack.com/archives/C0C4UAZKETS/p1791384179539709?thread_ts=1791343841.665629&cid=C0C4UAZKETS
   - Eric Richards reply, ts 1791385239.156819, same thread, no permalink captured: "I don't think sram instances will be directly affected by the too-fast frequency while coming out of reset, but flop initial states will fail to reset to expected values."
7. #mimir-soc, 2025-12-03, Paul Zehr (relaying Paul Scheidt). https://tenstorrent.enterprise.slack.com/archives/C07MVFKPK54/p1764796642605929?thread_ts=1764796642.605929&cid=C07MVFKPK54
   - "GPIO[26] ... It also has a strap function for SRAM auto-zero disable (on power-up)." This is a silicon board strap. It is not about emulator contents.
8. Atlassian SP, Q1 Compute Module Board Specification, snippet only. https://tenstorrent.atlassian.net/wiki/spaces/SP/pages/2192965653/Q1+Compute+Module+Board+Specification
   - "chiplets: disable SRAM auto-zero (Mimir); sampled at reset, driven by PCAL6524 U17, internal pull up". Board-level strap. It does not say what TL1 holds after reset.
9. Glean, chippy `validation/jtag/python_tests/soc/mimir/memory_sweep_jtag/README.md`, recorded mimir_emu runs, owner James McCallum, updated 2026-08-21. https://yyz-gitlab.local.tenstorrent.com/syseng-platform/chippy/-/blob/main/validation/jtag/python_tests/soc/mimir/memory_sweep_jtag/README.md
   - "Each CCE's mimir_cce_ctrl.local_tl1_base must point at its SRAM aperture before any SoC-bus access, or reads of 0x4000_0000 never reach the CCE SRAM at all. That register lives in the uncore-domain wrapper, so it is written after the uncore reset lift."
   - Recorded run: "PASSED on mimir_emu (emu_axi) 2026-07-30 (run mimir_memsweep_d2d ...) ... zero mismatches". It sweeps memory and does not test retention across runs.
10. Atlassian SIVAL, Quasar Microarchitecture Specification, snippet only. Quasar, not Mimir. https://tenstorrent.atlassian.net/wiki/spaces/SIVAL/pages/2451964588/Quasar+Microarchitecture+Specification
    - "warm reset can run at PLL and preserves config registers"
11. Glean, Mimir MM SW test plan working copy (artifact). Glean says artifacts may be AI-generated. https://app.glean.com/library/9568f93acfc347d6b0f6c2a6e079507f
    - "Actions: Load the selected base/runtime firmware image ... program the startup/reset vector; release the CCE DM from reset."
    - Same artifact: "Reset or reconnect JTAG between operations and repeat representative writes, reads, and readbacks."
12. Glean, test_gddr.py. https://yyz-gitlab.local.tenstorrent.com/tensix/soc/grendelemulation/-/blob/main/models/mimir/tests/test_gddr.py
    - Same quote as Step 3, "The ZeBu sparse page table is not cleared between tests".
13. Glean, tt_ocah_smu emu README, not Mimir. https://yyz-gitlab.local.tenstorrent.com/tensix/soc/tt_ocah_smu/-/blob/main/emu/README.md
    - "after a full init_and_reset() the counter equals the total elapsed time (the 1 us pulse never reset it)". Here the reset pulse did not reset a counter, in the OCAH SMU model.
14. Glean, grendel `py/tb/discover/examples.py`. https://yyz-gitlab.local.tenstorrent.com/tensix/soc/grendelemulation/-/blob/main/py/tb/discover/examples.py
    - "TB.init_and_reset() runs PRELOAD (fuses, ROM, SPI), RESET (straps, BP_RESETN), and CONFIGURE (emu_boot SMC firmware on every chiplet in parallel), then trains D2D." No statement on memory retention.
15. Slack #quasar-sw-bringup, 2026-04-13, Arik Yaacob, Quasar not Mimir. https://tenstorrent.enterprise.slack.com/archives/C0AG3E2UP97/p1776099192826659
    - "Between tests we issue deassert (soft reset) to each DM core and the global reset to triscs ... DM0 doesn't see a jump instruction at that address"
    - "What is the state of the i$ when cores are going on and out of soft reset?"
16. Slack #aether-environment, thread root 2026-04-15, Arik Yaacob; reply by Marko Lausin (ts 1776258447.877989). Aether sim and emu, not Mimir. https://tenstorrent.enterprise.slack.com/archives/C083G8K70SD/p1776257795728709?thread_ts=1776257795.728709&cid=C083G8K70SD
    - Marko Lausin: "...it is kept asserted through whole sim/emu time." The first verb in the snippet is garbled.
17. Slack #aether-environment, Kian Moniri, 2026-04-16, Quasar DM. https://tenstorrent.enterprise.slack.com/archives/C083G8K70SD/p1776354698396319
    - "When uncore reset is asserted, it is possible to write to the L1 by deasserting NOC reset and AI reset and programming a NoC write transaction."
    - "Data movement L2 cache (128kB) persists"
18. Slack #chippy-dev, thread 2026-08-24, Kenny Lee and Nick Chin. Mimir D2D Rocket reset vector, not a retention statement. https://tenstorrent.enterprise.slack.com/archives/C09FM5YB6RE/p1787612856090919?thread_ts=1787612856.090919&cid=C09FM5YB6RE
    - Code comment: "Only now release the Rocket core -- the vector is latched at reset exit." The vector is written before release each time.
19. Slack #aether-environment, 2026-06-17, Leonard Giordano, Aether not Mimir. https://tenstorrent.enterprise.slack.com/archives/C083G8K70SD/p1781710785370659?thread_ts=1781021552.087779&cid=C083G8K70SD
    - "init_and_reset() : (top-level pins only)"
20. Glean, tt-llk PR 1440, LLK not Mimir. https://github.com/tenstorrent/tt-llk/pull/1440
    - "Added a way to automatically restart the emulator between two test runs." (`--reset-simulator-per-test`).
21. Atlassian SIVAL, Mimir Microarchitecture Specification, snippet only. https://tenstorrent.atlassian.net/wiki/spaces/SIVAL/pages/2451964648/Mimir+Microarchitecture+Specification
    - Topic list only: "CCE SRAM download/reset vector, Group ID tagging, MLA remap, local SRAM routing". No retention statement in the snippet.

Auto-zero, reset vector and uncore hits that are not CCE TL1 (for completeness):
- `#quas-4171` 2026-03-05, Aaron Nicholas: "It's a good point, but I don't think that the SRAM auto-zeroization applies here - that affects the SMC scratch memory, not the memory we are dumping in QUAS-4171". https://tenstorrent.enterprise.slack.com/archives/C0AK0CJCN4R/p1772734051168799?thread_ts=1772734051.168799&cid=C0AK0CJCN4R
- `#sep-smc-devel` 2025-03-20, Roy Franz: "The SMC SRAM is automatically cleared (? SMC ROM or MBIST??) unless the SRAM_AUTO_ZERO_DISABLE strap is set." https://tenstorrent.enterprise.slack.com/archives/C07G64RU5K5/p1742520771432459
- `#tt-umd-simulators` 2026-07-01, Marko Lausin (Quasar): "DM uncore resets - 1 wire per tile that has DM cores". https://tenstorrent.enterprise.slack.com/archives/C0B3E777B27/p1782921739093739?thread_ts=1782401838.221149&cid=C0B3E777B27

Queries with no relevant Mimir hit (none found):

- Slack "mimir TL1" (does TL1 persist between runs or clients): none found. Hits were status and freeze notes.
- Slack "init_and_reset": only Aether, Saturn and Keraunos hits (item 19 and related). None for Mimir memory retention.
- Slack "reset TL1" (is TL1 or uncore state cleared after reset): none found for Mimir TL1 contents.
- Slack "uncore reset": items 2, 6, 16 and 17 above. No Mimir statement on what uncore reset keeps.
- Slack "reset vector": only item 18 (Mimir D2D Rocket) and Ceres/Callandor. No retention statement.
- Slack "auto-zero SRAM": items 7 and 8, plus the SMC/SEP and Quasar hits above. None for CCE TL1.
- Jira JQL `key in (MIM-601, MIM-628)`: items 4 and 5. Neither ticket has a statement on retained TL1, GDDR or register contents across a session.
- Glean "mimir emulator TL1 persist": none found for Mimir TL1 persistence.
- Glean "zebu sparse memory cleared": items 1 to 3 and 12 (same test_gddr.py text).
- Glean "mimir emulator restart between tests": only item 20 (LLK).
- Glean "init_and_reset zebu": items 13 and 14. No Mimir retention statement.
- Atlassian "Mimir emulator TL1 SRAM persist between runs reset": only item 10 (Quasar).
- Atlassian "Mimir emulator restart between tests stale state CCE reset vector": none found.
- Atlassian CQL `text ~ "auto-zero" AND type = page`: no Mimir CCE page.

## Appendix D: iDMA snoop-tag reports (2026-10-29)

> Note (2026-10-29 refresh): the reports below are as written at HEAD 16585a9c68a, and so are their file:line
> citations. Changed in the working tree since then (uncommitted), so line numbers into these files may have moved: dram/README.md, test_kernels/cce/cce_idma_copy.cpp and test_kernels/cce/cce_idma_copy_abi.h.
> test_mimir_dm_cache.cpp, test_mimir_atomics.cpp, the RTL, specs, chippy and Jira citations are unaffected.

Each report is as its agent wrote it; the iDMA snoop-tag scope section above holds the conclusions. Chippy and
grendelemulation paths refer to the checkouts named in Appendix C.

### D.1 Specs, Jira and Slack

#### Mimir CCE iDMA snoop tag: what the design intends (docs, tickets, messages)

Sources and versions I read. Grendel CCE spec, Drive 1ikHjbY495ioD-2YJqrMTqodCsC_aQeth, title "Grendel_cce_latest.pdf". Its revision table ends at **1.0 (08.20.2025, "Promoted to v1.0, no other updates from 0.9.5")**, not 0.9.2 as the brief says. 0.9.5 is the revision that "Added Keraunos CCE and Mimir CCE". Mimir spec, Drive 1nQrV25Qk4Fb-zGV1e7bMiSewPsm4bC4w, "mimir_latest.pdf", latest rev 1.14 (09.07.26); the Drive text was complete. I also read the Quasar Overlay Specification (SharePoint, modified 2026-10-07) and the overlay repo's noc_partial_snoop.md. Neither spec shows figures as text: Fig 8/32 (Mimir CCE subsystem) and Fig 17 (snoop master uArch) were not visible, so every block-placement statement below comes from prose.

##### Q1. What the iDMA snoop packet tag does
- PACKET_TAGS is per command buffer: "`[0]` snoop_bit ... Set on outgoing packet header; tells destination NIU to fire a snoop on the write. `[1]` flush_bit ... Optional flush companion to the snoop." (overlay repo docs/noc/noc_partial_snoop.md §4.1, design doc by Nate Mathew). This is written for the Quasar NIU, not the Mimir AXI path.
- Quasar NIU design intent for the same bit covers reads too: "The feature works symmetrically for **writes** (snoop fires when remote data lands in local L1) and **read-response writes** (snoop fires when read data is written back into the requester's L1)" (noc_partial_snoop.md §1). This is Quasar NIU behaviour. Neither the CCE spec nor the Mimir spec says the CCE AXI-DMA does the same, i.e. that a snooped copy refreshes the issuing CCE's own caches. That is the shape the failing own-CCE / GDDR-sourced legs assume.
- Self-targeted snoop works on Quasar because the write comes back in through the NIU. On the Quasar RTL simulator: "DM core 0 issuing NoC writes and atomics to its own tile, so the traffic arrives at its own NIU like any incoming write ... the core's cached poll saw the new data" (tt-metal issue #62349, Matt Craighead, relayed in #tt-metal-github-ping 2026-10-23, https://tenstorrent.enterprise.slack.com/archives/C0AJF4YF5QF/p1792792544441609). On Mimir, own-CCE AXI-DMA writes to own SRAM go xbar -> local SRAM (Mimir §7.2.3), not ITN, so the Quasar result does not carry over.
- On Mimir the tag travels as AXI user bit 8: "AxUSER[8]: CCE Command Snoop" (Mimir spec §5.3.2 "AXI User Field Definition", p.31). Designer: "CCE also needs 1 AxUSER bit (which is direct map from CMD_SNOOP_BIT in QNP)" (Neeraj Kulkarni, #mimir-soc, 2024-10-08, https://tenstorrent.enterprise.slack.com/archives/C07MVFKPK54/p1728436264665539). Designer: "I have pushed the fix for AXI_CMD_SNOOP_BIT = 8 in cce ip" (Neeraj Kulkarni, #cce_ip, 2025-03-18, https://tenstorrent.enterprise.slack.com/archives/C07UG9TGJLV/p1742281052640409).
- The spec names this bit two ways. §5.3.2 calls it "CCE Command Snoop". Table 35 "ITN User Field Definition" (§6.4.6, p.~45) calls it "[8] No Snoop | Hint information to Mimir CCE". The NOC2AXI RDL quoted on Slack has "cce_cmd_snoop[8:8] ... CCE cmd snoop for this flit" (Djordje Maksimovic, #quasar_noc, 2024-11-06, https://tenstorrent.enterprise.slack.com/archives/C064G6BUD55/p1730913578964299). The TB check `sec_exp.cce_cmd_snoop = noc_hdr.command.cmd_snoop_bit` (Bo Qin, #keraunos_pcie, 2025-06-27, https://tenstorrent.enterprise.slack.com/archives/C08949QUHRN/p1751040572712329) also reads it as active-high snoop. The "No Snoop" label in Table 35 contradicts both.
- Whose caches: the snoop block sits at the destination SRAM and updates that SRAM's own DMRISC caches: "The AXI snoop coherence master between NOC and SPM I/F snoops on all requests and invalidates/updates the appropriate cached entries in DMRISC complex" (CCE spec §3.2.7 "Dedicated SPM", p.10). Also: "When the local CCE writes to write pointer in remote CCE's SPM, the AXI Snoop Coherence block snoops on the write request and updates the cached write pointer in the DMRISC complex" (CCE spec §3.4.4, p.13-14). Nothing says the issuing CCE's caches are snooped.
- Which transfers: only writes that arrive at the SRAM from the fabric. Mimir: "It's primarily used to update cached data in DMRISC complex, when an external agent modifies the data (specifically read/write pointers) in the associated SRAM. Thus, all requests from ITN to SRAM are routed through this block which monitors writes to the read/write pointers" (Mimir spec §7.2.2 "AXI Snoop Coherence Master", p.59). CCE spec: "All requests from remote CCEs to the local SPM, should be routed via local CCE's NOC and SPM I/F" (§3.2.7, p.10).
- Only tagged writes are tracked: "The AW requests to write pointers are specially marked in the AWUSER bits. This table holds ... only for such outstanding write pointer requests" (CCE spec §7.8.1 "Pending write pointer tracker", p.45). The update is sent when the write's response comes back: "On AXI write response ... If match, send cache update to DMRISC" (§7.8.1). The block "intercepts the AXI interface (specifically, AW, W, and B channels) between the NOC and SPM" (§7.8.2).
- Designer: "AXI snoop master to send two 64B requests for each incoming AXI transaction with cmd_snoop bit set" (MIM-298, Neeraj Kulkarni, CRB 2025-06-26, Done).
- Source type (TL1 vs GDDR): not stated. Neither spec conditions the snoop on where the copy reads from. The snoop block only sees the write (AW/W/B) into SRAM (CCE spec §7.8.2).
- Destination (own vs sibling TL1): stated only indirectly, through where the block sits. Mimir spec §7.2.3 (p.60), repeated in CCE spec §5.2.2 (p.31): "All requests from AXI-DMA are either routed to local SRAM or ITN via CCE xbar based on the address ... Requests from DMRISC Mem port are directly routed to local SRAM while requests from DMRISC MMIO port are routed to ITN." So an own-CCE iDMA write to its own SRAM goes CCE xbar -> local SRAM and does not take the "ITN to SRAM" path that §7.2.2 routes through the snoop block. No sentence says outright "own-CCE iDMA writes are not snooped".
- VC / backend: not stated for snoop. The only VC statement is "Each DMA backend can be considered conceptually equivalent to a QNP VC" (CCE spec §3.2.4, p.10). Table 51 (Mimir spec p.59, MemTile0) lists example BE IDs per path, e.g. "2 SRAM0 SRAM1 2", "5 GDDR0 SRAM1 5", "4 GDDR0 SRAM0 4", and says "the programmer is free to use a different path to BE ID mapping". The emulator's "VC 2 honoured / VC 5 not" matches Table 51's example BEs for SRAM0->SRAM1 (2) and GDDR0->SRAM1 (5), so the VC numbers look like per-path BE choices. The table attaches no snoop semantics to any BE or VC.
- Mimir parameters: "EN_EXT_DMRISC_SNOOP 0 Disables support for external snoop requests to DMRISC complex"; "EN_SNOOP_ADDR_REMAP 0 Disables address remap logic in AXI Snoop Coherence Master" (Mimir spec Table 52, p.60; the SIVAL page 2451964648 §6 repeats both).

##### Q2. Coherence of hart caches against writes into this CCE's TL1, and the snoop-master config
- Harts among themselves: "The L2 cache is an inclusive last-level cache to the L1 I$ and L1 D$. It is shared amongst the 8 DM cores" (Quasar Overlay Spec, "L2 Cache"). Mimir CCE has "8 in-order Rocket cores, 4 KB L1 I-cache, 4KB L1 D-cache, 128KB shared L2" (Mimir spec §7.2.2, p.59). DV on how a snoop reaches the L1s: "the tilelink which when it receives a request for snoop in the L2, sends a query to all L1's" (CCE-191, Jason Dubrow comment 2026-03-25; status BUG CONFIRMED, about cores held in reset).
- (a) Own iDMA into own TL1: not stated as snooped. The path is "AXI-DMA ... routed to local SRAM ... via CCE xbar" (Mimir §7.2.3), which is not the ITN->SRAM path the snoop block sits on (Mimir §7.2.2). Emulation team doc: when LOCAL_TL1_BASE matched, "Every access CCE1 made to its own die's CCE0 SRAM was then serviced locally out of CCE1's own L1 at the same offset and never reached the NoC" (grendelemulation docs/dx/mimir-cce1-local-tl1-base-claims-peer-sram.md, George Powley, 2026-09-23). That shows own-window DMA goes the local route. The overlay spec's "Local iDMA" section ("shuffle data around in local tensix L1") says nothing about snooping. TL1 client names quoted in MIM-194 (Jason Dubrow, 2025-05-05) list separate `itn_wr_client`, `cce_dma_wr_client` and `cce_dmrisc_client` ports on mimir_tl1_wrap. The diff touches itn_rd_client_csr_cfg, cce_dmrisc_client_csr_cfg, cce_dma_rd_client_csr_cfg and cce_dma_wr_client_csr_cfg, and Jason's comment adds "It was only set for itn_wr_client_csr_cfg's". That is consistent with own-DMA writes using a separate TL1 port; these are RTL names, not spec statements.
- Keraunos design-side port naming (same CCE IP, different chiplet) is consistent with the same split: separate SRAM request ports `snoop_to_sram_req_i`, `cce_memport_to_sram_req_i`, `cce_to_sram_req_i`, `axi_dma_to_sram_req_i` (Kian Moniri, #keraunos_soc, 2025-02-04, https://tenstorrent.enterprise.slack.com/archives/C07U3205069/p1738691676257409). Designer: "Paths via axi_dma_to_sram, snoop_to_sram, cce_to_sram path: Pipeline stage added" (Neeraj Kulkarni, #keraunos-timing, 2026-04-21, https://tenstorrent.enterprise.slack.com/archives/C0ADY2VAD4G/p1776790273063699). A separate `axi_dma_to_sram` port next to `snoop_to_sram` suggests the CCE's own DMA writes do not pass through the snoop block. These are port names, not a statement about snoop scope.
- (b) Sibling CCE iDMA: intended to snoop. CCE spec §3.4.4 (p.13-14) and §7.8 (p.44): "1. Remote CCE writes data in local SPM 2. Remote CCE writes write pointer in local SPM 3. DMRISC on local CCE waits on write pointer update". Mimir §7.2.2: "when an external agent modifies the data ... in the associated SRAM". The designer's own use pattern: "CCE copies payload data to remote CCE ... CCE updates write pointer on remote CCE: DMRISC cores initiate inline write command" (MIM-538, Neeraj Kulkarni CRB, 2026-01-28).
- (c) GDDR-sourced iDMA: not stated. No spec text mentions source type. Following the placement text, the issuer decides the path. Own CCE issuing GDDR->own SRAM writes through xbar->local SRAM (§7.2.3), so no snoop path. Sibling issuing GDDR->this SRAM writes through ITN->this SRAM, which the snoop block covers (§7.2.2) with no source condition. So for a sibling-issued GDDR copy the spec text implies the snoop should fire, and the emulator result does not match that.
- (d) Host / SMC writes: host traffic enters D2D NOC2AXI -> ITN -> CCE SRAM. NOC2AXI carries "AxUSER[8]: CCE Command Snoop" and "Mimir NoC is responsible for carrying these AxUser bits to destinations" (Mimir §5.3.2). Host writes are therefore snooped only when the QNP cmd_snoop bit is set. The DV test plan says the same: "Traffic generator D2D sets AxUSER ... 'CCE command snoop bit set' ... Update this circular buffer through NoC interface. If writes to addresses that are being cached, cache is invalidated" (GRENSV-549, Ryan Faxigue, To Do). SMC reaches SRAM over SMN->ITN, and the reset text has "access to all CCE registers and Banked SRAM via CCE's AXI Snoop block" (Mimir §7.4.3, p.61-62). Whether SMC writes can carry AxUSER[8]=1 is not stated; Table 36 only assigns Source IDs. A Keraunos commit says SMC-written params needed manual maintenance: "Fix stale mode read: flush seed + invalidate before read (CCE D-cache not AXI-snooped)" (GRENSV-3544, commit ed28af76, 2026-07-16, Keraunos).
- Snoop-master config register: neither the CCE spec nor the Mimir spec defines one. The only snoop-related MMR the CCE spec names is "Snoop Address remap CCE base + 0x20_3000" (CCE spec Table 2), and it is disabled on Mimir (EN_SNOOP_ADDR_REMAP=0; MIM-623 says the Mimir RDL marks it reserved).
- The register DV programs is `noc_snoop_tl_master_cfg`, defined in the Quasar Overlay Spec ("Configuration Registers", cluster_ctrl.rdl:139-168): "[0] disable_inline_writes 0x0 ... [1] invalidate_flush_dirty_en 0x1 ... [2] clean_putdata_en 0x1". Reset value is 0x6.
  - bit0: "Value 1 (Safe Mode): Forces invalidation-only mode - All snoop operations trigger cache line invalidations".
  - bit1: "Value 0: Invalidate only ... Dirty data WILL BE LOST ... Value 1 (Default): Flush if dirty, invalidate if clean"; "This is a chicken bit added as a late-stage bug fix".
  - bit2: "Value 1 (Default): Mark inline writes as clean ... Only applies when disable_inline_writes=0".
- The overlay spec's coherence scope is incoming fabric writes: "writes from the NoC into TL1 will update Overlay cache hierarchies. However, writes from the Overlay cores to the TL1 will use a software flush mechanism" (Quasar Overlay Spec, "NoC Overlay Cache Coherency").
- Config values the lead saw break visibility (0x1/0x3/0x7): all three set bit0 (invalidation-only). The overlay spec says that mode should still invalidate ("CPU cores fetch fresh data from T6 L1 memory on cache miss"). The docs do not explain the loss of visibility in that mode.
- Related config tickets: GRENSV-1571 test plan calls 0x2 "Cache invalidation" mode. By the overlay bit table, 0x2 keeps inline writes on and only clears clean_putdata_en; the chippy README for that variant says the same ("keeps inline writes and invalidate-flush-dirty enabled but drops clean-putdata"). GRENSV-3790 (Mo Odusanya, Under Review, [SW WAR]): "CCE snoop logic can lose data when clean_putdata_en is set". WAR: "Use the non-cacheable port to write to any snoop variables ... 64B align shared snoop variables".

##### Q3. Designer / DV statements on tag scope, and on the emulator results
- Open designer ticket on exactly this scope: "The snoop discussion exposed ambiguity between inline writes, write-update vs invalidate mode, and MMIO remapper snoop behavior. Clarify when ROCC snoop applies and how inline writes relate to snooping" (GRENSV-2011, reporter Ryan Faxigue, assignee Neeraj Kulkarni, created 2026-04-23, **To Do**; only comment is Mo Odusanya 2026-05-04 "just a reminder to please look at this task"). No answer is recorded.
- The DV test definitions for iDMA snoop are all sibling TL1->TL1. GRENSV-1577 "Snooping - Write update" (Done): "CCE0 Mhart0 does cached access to SRAM address x ... CCE1 Mhart0 does dma to TL1 SRAM address x from its TL1 SRAM address y ... use rocc command with snoop_bit = 1 ... ROCC comand snoop_bit =1 always uses inline writes". GRENSV-1571 "Snooping - Cache invalidate" (Done) uses the same CCE1->CCE0 TL1->TL1 setup with cfg 0x2.
- GRENSV-1557 MR !468 (Petar Arnaut, merged) has the same scope: "cross-CCE snooped-DMA cache coherency (mimir_emu, two CCEs) ... CCE1: DMAs NEW into CCE0's X with snoop_bit=1". The chippy cce_xcce_snoop_reader README: "PASS iff it now observes NEW — i.e. the snooped remote DMA invalidated CCE0's cached line". These state DV scope, not design intent. (The Jira key GRENSV-1557 is itself "SMC memory sweep on CCE"; the MR reused the key.)
- GRENSV-2339 / GRENSV-2340 "Inline DMA with cmd_snoop enabled" (Done) cover cmd_snoop=1/0 inline DMA to "Remote SRAM/ GDDR" and check only "data in destination for integrity". No cache-coherence check. Comment: "it will be used to update rd/wr pointers which can be implemented with a) software variables (need cmd_snoop set too), or b) llk registers (without cmd_snoop)".
- I found no own-CCE iDMA snoop test, no GDDR-sourced snoop test and no preload snoop test in Jira (searched `text ~ "snoop"` across MIM, CCE, GRENSV) or in the chippy READMEs Glean returned.
- The test-plan source for the failing legs does not claim the snoop works on every path. "External-writer coherence: Cache data in a CCE, then update it through host/JTAG, local DMA, a remote CCE, local DRAM, and remote D2D/DRAM paths. Compare snoop-enabled, snoop-disabled plus explicit L2 invalidation, and invalidate-only modes" and "Preload/cache-churn stress: ... Compare snoop-enabled preload against snoop-disabled preload". It also says: "The exact Mimir cache geometry, line size, snoop behavior, and D2D visibility rules must remain parameterized until the authoritative cache contract is finalized" (mimir_only_package_sw_test_plan_working_copy.md, Kevin Stevens, Glean artifact, lines 592-598). This is an SW test plan, not a design statement.
- Emulator results of 2026-10-22/23 (stale cached reader after own-CCE or GDDR-sourced snooped iDMA): I found no Jira, Slack or Confluence message discussing them, so nobody on record calls them a bug, a known limitation or intended. Searches: Slack `snoop`+`iDMA`, `snoop`+`CCE`+`own`, `cmd_snoop`, `snoop_bit`, `snoop` in #cce_ip, `snoop`+`GDDR`; Glean "iDMA snoop packet tag CCE" and "snoop local SRAM own CCE DMA not snooped"; Jira as above.
- Related bug/limitation tickets on the snoop master (none about source or destination scope):
  - CCE-191 BUG CONFIRMED: snoop fails when some cores are in reset.
  - GRENSV-1522 [SW WAR]: same bug, Under Review.
  - CCE-217 NEEDS ACTION: >32 same-ID writes "the snoop request is silently never generated"; "should not impact KER or MIM".
  - GREN-1942 Future Fix: snoop master cold-reset vs warm-reset DMA/L2; "applies to Mimir as well".
  - MIM-298 Done: >64B snoop support added.
  - CCE-228 PLANNING (future CCE): "The AXI snoop module currently monitors AXI requests and sends invalidate/flush requests to the CPU".
- Confluence SIVAL 2451964648 (titled "Mimir Microarchitecture Specification", Ryan Faxigue, last modified 2026-05-24) has nothing on snoop scope. It repeats EN_EXT_DMRISC_SNOOP=0 and the routing "CCE DMA | Local SRAM address | CCE xbar -> local SRAM" (Appendix A.3).

##### Q4. GRENSV-2290
- Summary "Snooping - CCE MMIO access"; "Feature ID: Axi snoop master"; "Test objective: Cache write update with snoop bit for MMIO access" (GRENSV-2290, reporter Ryan Faxigue, assignee Subha Vaidyanathan, created 2026-04-23, status Done, no comments, no links).
- Procedure: "SMC FW: Configures CCE address remapper ... Configure remap active , cmd_snoop=1. CCE FW: Issue CCE0 to CCE1 MMIO write/read/verify with SPA address to local mimir resource (CCE/GDDR); Issue CCE1 to CCE0 ...". Expected: "Snooping is enabled on the MMIO access issued to the address region covered by the remapper. Cached accesses will be updated when the CCE overwrites the value in SRAM using MMIO access".
- It says nothing about iDMA. The chippy README for its test states the split: "the snoop is triggered by a per-transfer DMA flag (idma_1d_axi_transfer(..., snoop_bit=1)). This test instead exercises the DMRISC address remapper's cmd_snoop region attribute ... The firmware here contains no idma_1d_axi_transfer() call at all" (cce_xcce_snoop_mmio/README.md, Subha Vaidyanathan).
- GRENSV-2288 / GRENSV-2289 ("Mimir-Snooping - CCE MMIO access") are To Do duplicates with only the objective line.

##### Verdict per copy type
- Own-CCE iDMA into own TL1: **not intended (implied by block placement, Mimir §7.2.2 / §7.2.3 and CCE §3.2.7 / §5.2.2; no explicit sentence)**. The snoop block covers only "requests from ITN to SRAM", and own AXI-DMA writes go "CCE xbar" -> "local SRAM". The Quasar NIU "read-response writes" snoop (noc_partial_snoop.md §1) is not documented for the CCE.
- Sibling TL1->TL1: **intended to snoop**. CCE spec §3.4.4 and §7.8, Mimir §7.2.2, DV plan GRENSV-1577.
- Local GDDR: **not stated**. If the copy's own CCE issues it into its own TL1, it takes the same xbar path as own-CCE, so not intended (implied by block placement). If the sibling issues it, the write enters via ITN and the spec implies it should snoop: no source-type condition anywhere. The emulator result contradicts that, and no ticket explains it (GRENSV-2011 is still open).
- Remote GDDR: **not stated**. Same split as local GDDR: own-CCE-issued is not intended (implied by block placement); sibling-issued is implied by Mimir §7.2.2 but never stated.
- Preload: **not stated**. If the preload is an own-CCE iDMA into own TL1, it falls under own-CCE: not intended (implied by block placement, Mimir §7.2.2 / §7.2.3; no explicit sentence). The test plan only asks to "Compare snoop-enabled preload against snoop-disabled preload" (MM SW test plan l.594).

##### Not checked
- Figures 8/32/33 (Mimir CCE subsystem, CCE IP) and Fig 17 (snoop master uArch): the Drive text has captions only, so block placement is from prose.
- RTL (cce_axi_snoop_master.sv, CCE xbar, tt_cce_mimir_wrap.sv), so I have not confirmed whether the xbar->local-SRAM path bypasses the snoop block. That is the one check that would settle the own-CCE and preload legs.
- Whether the iDMA drives AWUSER[8] on writes whose source is GDDR (read from ITN). The only DV check I saw (overlay idma_xbar.cpp, "The packet-tag snoop bit rides AWUSER[AXI_CMD_SNOOP_USER_BIT]") uses an L1 source.
- The Mimir/Grendel CCE xlsx sheets linked from the SIVAL page, and the Grendel Security/System specs.

### D.2 RTL, chippy and the tests

#### iDMA snoop tag: code, RTL and chippy evidence

Conclusion: the CCE RTL makes the tag a property of the inbound NoC-to-TL1 port only. An iDMA write from a CCE into its own TL1 window never passes the snoop master, whatever the source (own TL1, local GDDR, remote GDDR). So expecting the tag to refresh the cache for ExternalWriters LocalIdma, LocalGddrIdma and RemoteGddrIdma goes against the RTL. Those legs fail by design, not because of an emulator defect. A sibling CCE's iDMA into this CCE's TL1 leaves through the ITN and comes back through the snoop master with AWUSER[8] set from the tag. So the RTL supports the expectation for sibling copies, whether TL1- or GDDR-sourced. The CCE RTL does not gate on VC/backend or on source. PreloadChurn (sibling, GDDR source, VC 5) fails on the emulator, and nothing I read in the CCE IP explains it. The cause is outside the CCE IP (NoC/ITN carriage of AWUSER[8], which I did not read) or a confound that the tests cannot separate. Chippy uses and tests the tag only cross-CCE TL1-to-TL1. The tests and README cite `cce_xcce_snoop_dma` as GRENSV-2290, but chippy files that test under GRENSV-3774.

Refs read: chippy `/proj_sw/user_dev/sagarwal/chippy` @ c9a8a6085433 (2026-10-27). grendelemulation `/proj_sw/user_dev/sagarwal/grendelemulation` @ 2b916f42b5fc (2026-10-27). CCE RTL: aus-gitlab `soc/cce` (project 445) @ ce574b3216f2 (the ref GitLab search returned for default branch `main`; I did not check it is the RTL the emulator was built from). iDMA: aus-gitlab `opensrc/opensrc-idma` (project 403) @ 6a9cc46a27f6 (`tt/master`). tt-metal `/localdev/sagarwal/tt-metal` @ 16585a9c68a9 plus the working tree (README.md modified, uncommitted).

##### 1. Chippy: what `cce_xcce_snoop_dma` does, and its scope

- **Pair.** `cce_xcce_snoop_reader` runs on CCE0 and `cce_xcce_snoop_dma` on CCE1, one Mimir, boot hart only (`validation/firmware/apps/ip/cce/cce_xcce_snoop_reader/README.md:8-24`). The CI config is `validation/jtag/python_tests/ip/cce/load_fw_verify_scratch/test_config_mimir_cce_xcce_snoop.yaml:4,46-47`, whose header reads "Config for GRENSV-3774". It is enabled in `validation/ci/regression_test_table.yml:431-446` on mimir_emu and mimir_jtag_emu.
- **Steps.**
  - The reader primes X = OLD at CCE0 SRAM offset 0x4000 uncached, then reads it cached, which gives a clean line (`cce_xcce_snoop_reader/src/main.c:89-105`).
  - The reader DMAs a READY sentinel with snoop_bit=0 (`:115-119`).
  - The DUT step, in `cce_xcce_snoop_dma/src/main.c:77-81`: `idma_1d_axi_transfer(CCE_DMA_BE_ID=0, SRAM_CCE_1_MEM_BASE_ADDR + 0x3000, SRAM_CCE_0_MEM_BASE_ADDR + 0x4000, 0x40 B, cmdbuf 0, tr_id 1, num_iter 1, snoop_bit=1, flush_bit=0, wait_for_ack)`. Source is CCE1's own TL1, destination the sibling CCE0's TL1, absolute MLA addresses, one 64 B line, backend 0.
  - The DONE handshake DMA uses snoop_bit=0 (`:96-100`).
  - The reader passes only if a cached re-read returns NEW. scratch[6] (uncached) separates "write landed" from "snoop missed" (`cce_xcce_snoop_reader/src/main.c:138-154`). The constants are in `validation/common/include/cce/cce_dma_common.h:121-151`.
  - The run needs `--test-config-enable-remap false`, because the firmware uses absolute MLA addresses (`cce_xcce_snoop_reader/README.md:61-66`).
- **Tag bits.** `idma_1d_axi_transfer` rejects snoop/flush values above 1 (`validation/firmware/libs/idma/src/idma.c:58-65,95-98`), then calls `setup_packet_tags_cmdbuf(cmdbuf, snoop_bit, flush_bit)` before issue (`idma.c:110-111`). In the register description, PACKET_TAGS bit 0 is `snoop_bit`, "Whether the transaction should be sent with a snoop bit for notifying a **remote** datamovement core", and bit 1 is `flush_bit` (`validation/common/include/registers/mimir/mimir_soc.json`, `mimir_soc/cce_0/tt_rocc_accel/tt_rocc_cpu*_cmd_buf_{r,w}/PACKET_TAGS`; source `deps/overlay/meta/registers/rdl/tt_rocc_cmd_buf_regsdef.rdl`).
- **Snoop-master config.** `cce_xcce_snoop_dma` does not program it, so the default 0x6 applies. The variant `cce_xcce_snoop_dma_cache_invalidate` writes `NOC_SNOOP_TL_MASTER_CFG = 0x2` (invalidate_flush_dirty_en=1, clean_putdata_en=0) and is otherwise the same test (`cce_xcce_snoop_dma_cache_invalidate/src/main.c:72-77`, README `:13-37`). The writer is `idma_configure_noc_snoop_tl_master_cfg` (`libs/idma/src/idma.c:583-592`). Field descriptions, from `mimir_soc.json`, `cce_0/tt_cluster_ctrl/noc_snoop_tl_master_cfg` (`deps/overlay/meta/registers/rdl/cluster_ctrl.rdl`):
  - `disable_inline_writes`: "all snoops will cause an invalidation".
  - `invalidate_flush_dirty_en`: "only affects ... invalidate-snoop mode (i.e NoC atomic port or all ports if disable_inline_writes=1)", reset 1.
  - `clean_putdata_en`: reset 1.
- **Other chippy uses of the tag.**
  - `cce_dma_inline_spa` sends `idma_inline_write(..., snoop)` across CCEs over the peer's SPA window, CCE0 to CCE1 and CCE1 to CCE0. Snoop-on phases must read fresh and snoop-off phases stale (`validation/firmware/apps/ip/cce/cce_dma_inline_spa/README.md:14-64`; snoop encoding at `libs/idma/src/idma.c:557-568`).
  - Every GDDR-touching or own-CCE chippy iDMA passes `snoop_bit=0`: `apps/soc/mimir_soc/cce_to_gddr_2d/src/main.c:365,423`, `apps/ip/cce/cce_dma_rocc_remote_rx_mla/src/main.c:236,253,275,296,409` (this includes its local CCE0-to-CCE0 T3 transfer), `cce_dma_2d_banks_spa/src/main.c:429,457`, `cce_dma_2d_addrgen_spa/src/main.c:466`, every `chiplet_config_data_*.c` (`.snoop_enable = false`), and `jtag/python_tests/ip/cce/cce_fw_operation_queue/test_cce_fw_op_queue.py:447,628,652,775,796,838`.
  - **No chippy test uses or tests the tag on an own-CCE copy or a GDDR-sourced copy.**
- **Remap `cmd_snoop` (RegionAttrs bit 30).**
  - The RDL describes it as "this value will be mapped to AxUSER[9]" (`mimir_soc.json` `cce_0/dmrisc_addr_remap/dmrisc_remap_entries_*/region_attrs/cmd_snoop`, from `deps/cce/meta/registers/cce/rdl/cce_dmrisc_addr_remap.rdl`). The RTL instantiates the DMRISC remap with `AXI_CMD_SNOOP_USER_BIT(tt_cce_pkg::AXI_CMD_SNOOP_USER_BIT)`, which is 8 on Mimir (soc/cce `rtl/cce_addr_remap.sv:227`, `packages/tt_cce_mimir_pkg.sv:68-70`). The "[9]" in the RDL text is out of date: the bit is the same AWUSER bit the iDMA sets.
  - `sival_arch/utility/cce.py:205-241,296-345` sets it only on DMRISC entry 1 (peer CCE SRAM), for CPU stores: "any CPU-issued write that hits this region gets tagged ... so the receiving CCE's `cce_axi_snoop_master` invalidates/updates its own cached copy". `cce_xcce_snoop_mmio` (`README.md:8-23`) is the GRENSV-2290 test, cross-CCE CPU stores.
  - grendelemulation `models/mimir/tests/run_cce_via_sival.py:292` sets `cmd_snoop` on the DMRISC GDDR entry, a CPU path. emu_boot sets it on every non-self DMRISC entry and leaves the CCE's own window at attrs=0x1 (`firmware/mimir/smc/emu_boot/main.c:664-690`).
- **Scope statements found.**
  - PACKET_TAGS.snoop_bit: "for notifying a **remote** datamovement core" (register description above).
  - `pf_ctrl/snoop_disable/disable_snoop`: "AWUSER[tt_cce_pkg::AXI_CMD_SNOOP_USER_BIT] will be set to 0 on requests CCE and no snoop requests will be sent to DMRISC cores" (`cce_ctrl.rdl`, via `mimir_soc.json`).
  - grendelemulation `firmware/keraunos/cce/keraunos_flood_snoop/src/keraunos_flood_snoop.cpp:15-32,567-570` says "GLOBAL snoop writes trigger cache invalidation on the dirty CCE" and, when skipping its own slot, "dirty CCE can't snoop itself". That is Keraunos and the CPU-store path, and the own slot is skipped because nobody writes it. Weak support only.
  - No chippy comment states the tag's scope for own-CCE or GDDR-sourced copies. The grendelemulation docs, including `docs/dx/WRITE-VISIBILITY-AND-ORDERING.md` (no "snoop" in it), say nothing on the iDMA tag.

##### 2. RTL: where the tag is consumed and on which paths

**Tag onto AWUSER.**
- Every Mimir iDMA backend is the same `tt_idma_backend_r_init_rw_axi_ext` with `SnoopBitPos(tt_cce_pkg::AXI_CMD_SNOOP_USER_BIT)` (soc/cce `rtl/cce_axi_dma.sv:256-300`). On Mimir `AXI_CMD_SNOOP_USER_BIT = 8` (`packages/tt_cce_mimir_pkg.sv:68-70`).
- `IDMA_DFC_EN` is all zero across the 12 backends (`tt_cce_mimir_pkg.sv:40`), so no backend has a different write path.
- In the legalizer, the write AW is `user: ... | (opt_tf_q.cmd_user[0] << SnoopBitPos)` on every write burst, whatever the source protocol or VC (opensrc-idma `target/rtl/tt_custom/tt_idma_legalizer_r_init_rw_axi_ext.sv`, the "Write meta channel" block, around line 555).
- The step from PACKET_TAGS.snoop_bit to `cmd_user[0]` happens in the cmd-buf frontend, which I did not trace. That link rests on names.

**Where AWUSER[8] is acted on.**
- `rtl/cce_axi_snoop_master.sv`. The header says: "observes the requests from NOC to SPM AXI5L interface: ... For any write request that has AXI_CMD_SNOOP_BIT set, it sends out a snoop request to DMRISC L2 front end, on receiving a B response from SPM" (`:20-21`).
- `:82` masks the bit with `i_snoop_disable`. `:240-246` records `pending_aw_req.axi_snoop = aw.user[AXI_CMD_SNOOP_USER_BIT]`. `:340` allocates a pending write-data entry only when `axi_snoop == 1`.
- On the SPM B response, the entry becomes an L2 front-end write. That write is inline data, or the INVALIDATE64 / INVALIDATEFLUSHDIRTY register when `disable_inline_writes` is set (`:99-104`, and the dmrisc_l2front AW/W assigns near the end of the file).
- Instantiation: `noc_req_i(i_noc_req) -> spm_req_o(o_spm_req)` only (`rtl/cce_dma_snoop_remap_partition.sv:206-228`). The ext-snoop mux exists only `if (EN_EXT_DMRISC_SNOOP)`, and the Mimir wrap ties `s_axi_dmrisc_snoop_*` to 0 (`rtl/tt_cce_mimir_wrap.sv:790-835`). In `rtl/cce.sv`, `s_axi5l_noc_*` feeds `noc_req` (`:573`), `spm_req` drives `m_axi5l_spm_*` (`:572`) and the iDMA masters drive `m_axi_dma_*` (`:566-568`).

**Mimir wrap paths.**
- **iDMA writes.** All `NUM_DMA_BE` masters go into `i_cce_xbar` (`rtl/tt_cce_mimir_wrap.sv:971-1001`). Its only rule sends addresses in `[local_tl1_base_addr, local_tl1_base_addr + MIMIR_L1_SIZE)` to master port 1 (`:965-969`); everything else takes default port 0.
  - Port 1 goes to `i_dma_spm_axi_to_l1`, which drives `o_tl1_dma_wr_req_*`, a direct TL1 write port (`:1435-1451`). **This path has no snoop master.**
  - Port 0 goes to `dma_mla_remap` (`:1165`), then `i_cce_itn_mux` (`:1199`), then `m_axi_itn_*`. `m_axi_itn_awuser = {user[W-1:8], cce_group_id, user[3:0]}` keeps bit 8 (`:1217`).
- **Inbound writes.** `s_axi5l_noc_*` goes into `cce_inst`, through the snoop master, out on `m_axi5l_spm_*`, to `axi_spm_req` and then `i_spm_axi_to_l1`, which drives `o_tl1_itn_wr_req_*` (`:1307-1316`). **Only this path passes the snoop master.**
- **LOCAL_TL1_BASE.** emu_boot sets it to the CCE's own package SPA window (`emu_boot/main.c:666-667`). A live readback showed `cce0=0x1280000000, cce1=0x1280400000` (grendelemulation `models/qsr1_a1/tests/dx/RUNQUEUE.md:133`, row 89). The tt-metal kernel's `kKindOwnTl1` command address is `cce_sram_address(own_cce, offset)`, i.e. that same SPA (`tests/tt_metal/tt_metal/test_kernels/cce/cce_idma_copy.cpp:40-44`, `tt_metal/hw/inc/experimental/cce_gddr.h:52-55`). So an own-CCE iDMA write to its own TL1 takes port 1 and is never snooped, whatever the source. RUNQUEUE row 35 (`:79`) measured the same mechanism on the emulator: accesses inside the declared local window "never reached the NoC".
- **Sibling writes.** A sibling iDMA write into this CCE's TL1 falls outside the sibling's local window. It goes out port 0, through the ITN, and arrives at this CCE's `s_axi5l_noc`, so it passes the snoop master. The RTL I read makes this the same for TL1 and GDDR sources (the source only changes the AR side) and for every backend/VC.
- **Unread.** The NoC/ITN/noc2axi fabric between the sibling's `m_axi_itn` and the target's `s_axi5l_noc` is in the Mimir SoC/NoC RTL, not in soc/cce. I could not find it on aus-gitlab. A project search for "mimir" returned only soc/mimir-die-to-die-tile, riscv_external/chips-mimir and soc/cce, and yyz-gitlab tools were not available to me. The D2D noc2axi has its own per-TLB `AXI2NOC_HEADER_TLB_SEC.cce_cmd_snoop` ("CCE cmd snoop for this flit", `mimir_soc.json` `d2d_*/noc2axi/noc2axi_tlbs_a/AXI2NOC_HEADER_TLB_SEC`). That suggests D2D-arriving traffic may get the bit from a TLB rather than from the originator. I have not verified this.
- **Nothing found on VC 2 vs VC 5.** Nothing in soc/cce treats the request VC as a snoop gate. The VC/backend split is above the identical backends.

**What I tried for RTL.** aus-gitlab blob search for `AXI_CMD_SNOOP_USER_BIT`, `cce_axi_snoop_master`, `SnoopBitPos`, `s_axi5l_noc_awuser`, `cce_cmd_snoop` and `cmd_user snoop`. I read soc/cce `cce_axi_snoop_master.sv` (all 921 lines), `cce_dma_snoop_remap_partition.sv` (all), `tt_cce_mimir_wrap.sv` (lines 1-260 and 330-1453), `cce.sv` (lines 20-815) and `cce_axi_dma.sv` (lines 200-460), plus opensrc-idma's legalizer (all). There is no `/proj_soc` on this machine, and grendelemulation has no CCE RTL (only `emu_hdl` monitors). I did not use Glean `code_search` past one query whose output overflowed and was not read.

##### 3. tt-metal legs that expect the tag to refresh the cache

Gates: `kIdmaSnoopBitRefreshesCache = true` (`tests/tt_metal/tt_metal/dram/test_mimir_dm_cache.cpp:594-600`) and `kIdmaSnoopRefreshesHartCache = true` (`test_mimir_atomics.cpp:63-65`). Both comments say "Expected behaviour, not a probe result ... as chippy's GRENSV-2290 test cce_xcce_snoop_dma requires". The basis is miscited: chippy files `cce_xcce_snoop_dma` under **GRENSV-3774** (`test_config_mimir_cce_xcce_snoop.yaml:4`), and GRENSV-2290 is the MMIO/remap test `cce_xcce_snoop_mmio` (`README.md:9`). Either way, the cited chippy test covers only a cross-CCE TL1-to-TL1 copy. The kernel sets the tag with `overlay::setup_packet_tags_cmdbuf<CMDBUF>({.snoop = true})` (`test_kernels/cce/cce_idma_copy.cpp:225-227`). VCs: `kBackendLocalSram = 5, kBackendRemoteSram = 2, kBackendFromRemote = 5` (`cce_idma_copy_abi.h:107-110`).

**ExternalWriters `*_LocalIdma_SnoopOn`, README row 99** (`test_mimir_dm_cache.cpp:1636-1675,1739-1749`)
- Copy: the reader CCE's own hart 1, own TL1 to own TL1 (`own_tl1_operand` for both ends), VC 5, 7 x 8 KiB.
- Asserts warm-old == 64, stale-before == 0 and matches == 64, with no invalidate. Always asserted: the leg ignores `kIdmaSnoopBitRefreshesCache`.
- README `:423` (2x, 2026-10-26) and `:440` (1x1): fails on all CCEs, stale on all 64 words, "as expected".

**ExternalWriters `*_LocalGddrIdma_SnoopOn`, row 100** (same code)
- Copy: the reader CCE's own iDMA, local GDDR view to own TL1, VC 5.
- Same asserts and gating as row 99.
- README `:423`, `:440`: fails, stale on 64 words.

**ExternalWriters `*_RemoteGddrIdma_SnoopOn`, row 101** (same code, `:1653-1657`)
- Copy: the reader CCE's own iDMA, other-Mimir GDDR to own TL1, VC 5. Despite the name, this is an own-CCE writer.
- Same asserts and gating as row 99.
- README `:423`: fails, stale on 64 words, 2x only.

**CrossCceSnoopOn, row 104** (`:1879-1953`)
- Copy: the sibling CCE's iDMA, sibling's own TL1 to this CCE's TL1 SPA, VC 2, 64 words.
- Asserts warm-old == 64 and stale-before == 0 if `kIdmaSnoopBitRefreshesCache`, then matches after an invalidate.
- README `:422`, `:439`: pass on all CCEs.

**DirtyLineRace `Cfg6`, row 106** (`:2033-2191`, gate `:2091`)
- Copy: the sibling's iDMA, sibling TL1 to this CCE's TL1 line, VC 2, 8 words, against a line the hart holds dirty.
- Asserts the external words survive and the hart's neighbour words survive, under config 0x6 only, and only if `kIdmaSnoopBitRefreshesCache`.
- README `:424`, `:441`: Cfg6 passes; configs with bit 0 set fail intermittently and are not gated.

**PreloadChurn snoop leg, row 109** (`:2305-2423,2427-2438`)
- Copy: the sibling's iDMA, local GDDR to this CCE's TL1 SPA, VC 5, 8 KiB per round, 4 rounds.
- Asserts stale-before == 0 and matches == 4, if `!snoop || kIdmaSnoopBitRefreshesCache`.
- README `:426`, `:442`: snoop leg fails, 1664 or 1696 of 2048 words stale, 0 of 4 rounds; the invalidate leg passes.

**CleanDirtyLineSnoopedWrite, row 112** (`test_mimir_atomics.cpp:1348-1419`)
- Copy: the sibling's iDMA, sibling TL1 (`source_kind` default 0) to this CCE's TL1 SPA, VC 2, one line.
- Asserts the AMO returns the iDMA value, if `kIdmaSnoopRefreshesHartCache`.
- README `:427`, `:439`: pass.

**ExternalWriters `*_SiblingSpa_SnoopOn`, row 98, not iDMA**
- Copy: the sibling's CPU SPA stores, with remap entry 1 `cmd_snoop` set.
- Asserts the same three checks as rows 99-101; the failure note names R2.
- README `:422`: pass.

**Not gated.** `MimirAtomicItnControl` only records a snoop verdict (`test_mimir_atomics.cpp:1766-1774`). No other atomics leg sets `snoop`; `IdmaBatch.snoop` defaults to 0 (`:363`).

README `:406` records the 2026-10-22 finding. The snoop-on legs of rows 99-101 and PreloadChurn's snoop leg "are expected to fail on this emulator until the tag is honoured". The README frames this as an emulator defect. For rows 99-101 the RTL above contradicts that framing.

##### Table

| Copy type | What the code/RTL says | Emulator (2026-10-22/23/26) | Tests that assert the tag refreshes the cache |
|---|---|---|---|
| Own-CCE TL1 to own TL1 | The destination is in LOCAL_TL1_BASE, so the write takes xbar port 1 and `o_tl1_dma_wr_req`, bypassing the snoop master (`tt_cce_mimir_wrap.sv:965-969,1435-1451`). **Not snooped by design.** No chippy test. | Not honoured (stale 64/64) | ExternalWriters LocalIdma SnoopOn (row 99); always asserts, unconditionally. Expectation contradicts the RTL |
| Sibling TL1 to TL1 | Out port 0, through the ITN with bit 8 kept (`:1217`), into the target's `s_axi5l_noc`, through the snoop master (`cce_axi_snoop_master.sv:240-246,340`). **Snooped.** Chippy's `cce_xcce_snoop_dma` covers exactly this case | Honoured (VC 2) | CrossCceSnoopOn (104), DirtyLineRace Cfg6 (106), CleanDirtyLineSnoopedWrite (112) |
| Local GDDR to TL1 | As run by the tests, the writer is the reader's own CCE, the destination is its own TL1 and the write takes port 1, so **not snooped by design**. If a sibling wrote it, the CCE RTL path would be the next row. No chippy test | Not honoured (own-CCE, VC 5) | ExternalWriters LocalGddrIdma SnoopOn (row 100). Expectation contradicts the RTL |
| Remote GDDR to TL1 | As run, the writer is again the reader's own CCE and the destination its own TL1: port 1, **not snooped by design**. Only the AR crosses D2D. No chippy test | Not honoured (own-CCE, VC 5) | ExternalWriters RemoteGddrIdma SnoopOn (row 101, 2x only). Expectation contradicts the RTL |
| Preload (sibling, local GDDR to TL1, VC 5) | Write path same as sibling TL1 to TL1 (port 0, ITN, snoop master); the legalizer sets the bit regardless of source and backends are identical, so the **CCE RTL predicts snooped**. Not explained by anything read. Source (GDDR vs TL1), VC (5 vs 2) and size (8 KiB vs at most 256 B) all change together, and the NoC/ITN carriage of AWUSER[8] was not read. No chippy test | Not honoured (1664/1696 of 2048 words stale) | PreloadChurn snoop leg (row 109) |

Suggestion: one single-variable run would separate VC from source on the sibling path. Options: a sibling TL1-to-TL1 copy on VC 5, a sibling GDDR-to-TL1 copy on VC 2, or a 64 B sibling GDDR copy.

##### Not checked

- The cmd-buf frontend mapping from PACKET_TAGS.snoop_bit to `cmd_user[0]`. This rests on names.
- Mimir SoC, NoC, ITN and noc2axi RTL: per-VC routing into `s_axi5l_noc`, and whether AWUSER[8] survives each VC. I could not reach it.
- Whether soc/cce @ ce574b32 matches the emulator build.
- That `local_tl1_base` in a tt-metal emulator session equals the SPA. Taken from emu_boot and the RUNQUEUE readback, not from a tt-metal session.
- Jira and Slack. Those belong to the other agent.
