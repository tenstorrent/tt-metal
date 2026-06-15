---
name: firmware-host-mailbox-mismatch
description: profiler sync timeout=N/complete=0 + 30-min embedding hang = host-lib vs JIT-firmware mailbox mismatch (e.g. unreverted dev_msgs.h)
metadata: 
  node_type: memory
  type: project
  originSessionId: 73e2e6db-60ce-49ca-9a6e-bf0b6caab1ac
---

On the Kimi BH 8×4 mesh, symptoms **realtime-profiler `sync ... timed out after 2000ms` (complete=0, timeout=416) at device init + 30+ min hang during embedding distribution (0 layer files opened)** are caused by a **host-lib vs device-firmware mailbox-layout MISMATCH**, NOT degraded device state. tt-smi `-glx_reset_auto` does NOT fix it (it resets chips, not the host/fw mismatch).

Concrete cause seen 2026-06-14: `tt_metal/hw/inc/hostdev/dev_msgs.h` `launch_msg_buffer_num_entries` left at 16 (an experiment edit) while `build_Release/lib/libtt_metal.so` was built with 8. Runtime JIT-compiles device firmware/kernels from current headers (launch=16) but the host lib reads mailboxes at the launch=8 offset → profiler sync marker at wrong offset (timeout) + launch-msg ring corruption (dispatch hang).

**Trigger of the trigger:** `git checkout fileA fileB` where fileB pathspec doesn't exist makes git abort and revert NEITHER file — silently. So `git checkout dev_msgs.h dev_msgs.hpp` (no such .hpp) left dev_msgs.h unreverted.

**Fix:** `git checkout tt_metal/hw/inc/hostdev/dev_msgs.h` alone; re-grep the constant to confirm. Firmware cache is hash-keyed by header content, so reverting is sufficient — no rebuild needed. Verify with a 1-layer smoke: profiler should log `sync complete: ... 32` and reach a chunk in ~16s.

**Lesson:** always verify a single-file `git checkout` SUCCEEDED (check exit code + re-grep); never batch a real path with a bogus one. See [[kimi-chunked-prefill-work-state]].
