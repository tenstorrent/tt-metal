# AutoFix dispatch hypothesis audit — corrected after halt-state evidence

## Conclusion

**The preserved post-triage stall is explained by an intentionally halted prefetch BRISC, with a consistent full FetchQ. It does not establish a dispatch implementation defect or the cause of the original suspected failure.** Earlier interpretations in this audit that prioritized stale device loads, wrong cursors, or lost publication are superseded by the halt-state evidence below. No decoder-local or generic dispatch implementation repair is justified.

The safe recovery conclusion is **operator reset after preserving evidence and coordinating the run's processes/device ownership**. Do not bypass triage's `cont` suppression, invoke an unpatched resume path, or assume the halted kernel can safely continue. No recovery or device/process interaction was performed by this audit agent.

## Decisive evidence and chronology

`triage/prefetch_private.json`, timestamp 2026-11-12 16:51:51 UTC, records `initially_halted: true`, `finally_halted: true`, PC `0x11368`, static `prefetch_q_rd_ptr = 0x1d340`, GPR15/GPR17 both `0x1d340`, `inflight_count = 0`, `stall_state = 0`, and `issue_fence = 0x55a80`. The cursor agrees with the last consumed-slot credit. A halted PC in the empty-slot loop does not mean the core is currently polling, nor does it imply that the slot remains empty after the host continues submitting work.

`tools/triage/triage.py:959–1000`, `_patch_risc_debug`, explicitly documents a Blackhole/Wormhole hardware problem where HALT → READ/WRITE → CONTINUE breaks cores. It patches both `BabyRiscDebugHardware.cont` and `continue_without_debug` into no-ops on affected devices, and records triage-halted cores. `tools/triage/triage_session.py:13–15` identifies Blackhole and Wormhole as affected. Leaving the prefetch BRISC halted is intentional in this workflow.

Filesystem timestamps, independently checked with `stat`, put every cited stable host stop after the first triage artifact:

| Preserved artifact | UTC mtime on 2026-11-12 |
|---|---:|
| `triage/full_long_snapshot.txt` | 16:35:01.913818915 |
| `triage/host_stack.txt` | 16:36:06.031371397 |
| `triage/host_locals_01.txt` | 16:36:19.369902527 |
| `triage/host_locals_02.txt` | 16:37:50.766542954 |

These mtimes are artifact chronology, not exact halt timestamps. Together with the intentional halt behavior and later `initially_halted` observation, they invalidate using the repeated host samples as proof that the same stall existed before triage. The original long full-attention run may have had a fault, unusual runtime, or another cause for concern; this evidence cannot distinguish those possibilities. A fresh bounded run after authorized recovery is needed to establish any original failure.

## Ring and MMIO audit

`triage/fetchqueue_words.json`, timestamp 16:50:04 UTC, reads credit `0x1d33c` before and after a 1536-word ring dump. A local parse confirms exactly 1535 nonzero entries; the only zero is slot 1407 at `0x1d33c`. The next slot, `0x1d340`, contains `0xa88`. This is the expected physical pattern for a full queue whose consumer is stopped: the last consumed slot is the unavailable spare slot; later publications fill every other slot until host W equals its credit.

Using compiled geometry preserved in AUTOTRIAGE (`B=0x1bd40`, `E=0x1d540`, `e=4`, `N=1536`), host `W = cached L = 0x1d33c` is valid interior slot 1407, and device `R = next(L) = 0x1d340` is slot 1408. Host binary reserve's saved return address `...11e021c1` follows its first, pre-wrap lambda call. Both pre-wrap and post-wrap checks exist, matching `system_memory_manager.cpp:738–750`. Neither immediate wrapping nor incorrect geometry is shown. The host can fill this ring after triage stops the consumer; the frozen earlier empty-wait PC and later full ring are consistent across that intervention.

The host MMIO interpretation also remains useful. `triage/host_gdb_registers.txt:84–111` records actual frame-zero RIP `0x78fe112c55df`, **after** `mov (%rbx),%r12d` at `...55dc`, with `r12d = 0x1d33c`. This sampled load completed. Current local `build_Release/lib/libtt-umd.so.0` has matching instructions at ELF offsets `0x2c55dc` and `0x2c55df`; the sequence agrees with `tt_metal/third_party/umd/tt-umd/src/pcie/device_memcpy.cpp:515–523`.

The `out` and odd `ja` near the beginning of the saved short disassembly are artifacts of starting decoding in the middle of instructions. Disassembling from valid boundary `0x2c55a0` gives `vzeroupper` at `...55d1`, a clock call at `...55d4`, and the expected move/load/compare. They are not evidence of port I/O or corruption.

Selected frame 4's `=> ...11e02467` is a saved return address after `read_core`, not the executing PC. The current load has reached frame-zero `r12d` but has not yet copied into caller-local `fence`. Frame-4 disassembly maps `r15` to SystemMemoryManager, `r14` to CQ id, `r13` to the credit-address reference, and `rbx` to local fence; manager offsets `0xe8`/`0x100` hold write-pointer/cached-fence vectors. `host_queue_values.txt:86–94` reports CQ0, credit address `0x1b7c0`, and three `0x1d33c` values consistent with the supplied local-fence/W/cached-fence interpretation, although the GDB expressions themselves are absent. The returned value agrees with the later independent device credit. An indefinitely blocked instance of the captured MMIO load is refuted; universal transport health or a measured polling rate is not claimed.

## AutoFix disposition and recovery

There is no violated queue arithmetic, publication, or decoder contract demonstrated by the corrected evidence. The halted consumer explains the full ring, host reservation wait, lack of further model progress, and downstream starvation. Adding a queue fence, enlarging buffers, changing wrap arithmetic, replacing RoPE, or adding periodic synchronization would not repair this state.

The repository's explicit Blackhole CONTINUE workaround must remain intact. Operator reset is the recovery boundary for this captured halted run. The installed `tt-device-usage` skill states: “A reset is an operator action,” and directs agents without reset authorization to preserve evidence and report reset as the operator's next step. After authorized recovery, device visibility and bounded mesh-open/close health checks must succeed before restarting the model experiment. Any rerun must establish progress/failure before a triage halt and clearly mark when triage alters execution state. No original-failure root cause, long-context pass, or stage completion is claimed.

## Validation and scope

Read the supplied AUTOTRIAGE, GDB artifacts, later ring/private-state JSON, relevant dispatch/UMD source, triage halt patch and device selector, and hardware recovery skill. Checked artifact timestamps; parsed ring counts; inspected the local UMD ELF with `nm -D` and `objdump -d --start-address=0x2c55a0 --stop-address=0x2c56b0`. The local ELF sequence matches preserved GDB instructions, but full runtime build-ID verification was not performed.

This audit agent performed no device access, process attachment, target-code execution, reset, termination, or implementation edits. Device-state artifacts were supplied by the parent task. This report is the only file changed by this agent.
