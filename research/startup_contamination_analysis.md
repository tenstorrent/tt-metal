<!--
SUMMARY: Analysis of T3K topology check contamination — whether tt_umd.Cluster() leaves ETH channels dirty
KEYWORDS: topology-check, contamination, 0x49706550, close_device, destructor, base-UMD, ETH, t3k, FIX-M
SOURCE: Local code analysis of nsexton/0-racecondition-hunt-fix-ae branch worktree
SCOPE: Cluster constructor/destructor behavior, topology check shell script, fabric init recovery path
USE WHEN: Debugging T3K topology check failures, understanding ETH channel state transitions, reviewing FIX M
-->

# T3K Topology Check Contamination Analysis

## Executive Summary

**The original contamination hypothesis is WRONG.** `tt_umd.Cluster()` (the default
constructor used by METHOD 1) does NOT call `deassert_risc_resets()` and therefore
does NOT load base-UMD firmware on ETH channels. The topology check cannot cause
the ETH contamination described in the shell script comments.

The `~Cluster()` destructor DOES call `close_device()` (as of commit 065867e0 on this
branch), but the comment in `py_api_cluster.cpp` line 24 claiming "the default
Cluster::~Cluster() does NOT call close_device()" is **factually wrong for the
branch code** (though correct for the Docker image's older tt_umd).

## Detailed Step-by-Step Analysis

### What `tt_umd.Cluster()` actually does

The Python `tt_umd.Cluster()` call maps to `Cluster::Cluster(ClusterOptions{})` in
`tt_metal/third_party/umd/device/cluster.cpp:339`.

Constructor call chain:
1. `TopologyDiscovery::discover()` — reads PCIe config, ETH link status, remote ASIC
   IDs via the existing relay. **Reads only, no firmware writes.**
2. `construct_chip_from_cluster()` — creates `LocalChip` and `RemoteChip` objects.
   Each calls `init_tt_device()` which does: `probe_arc()`, `wait_arc_core_start()`,
   creates `ArcMessenger` and `TelemetryReader`. **No RISC deassert. No firmware load.**
3. `construct_cluster()` — sets arch_name, checks ETH broadcast capability.
   **No RISC deassert.**

**Critical: `start_device()` is never called.** That function (cluster.cpp:1053) is
what triggers `deassert_resets_and_set_power_state()` → `chip->deassert_risc_resets()`
→ firmware load on all ETH channels. The default `Cluster()` constructor does not
call it.

Therefore: **`tt_umd.Cluster()` does not contaminate ETH channels.**

### What happens when `close_device()` is unavailable (old Docker image)

On the runner `tt-metal-ci-vm-t3k-04`, the Docker image predates commit 065867e0.
The old `~Cluster()` (pre-065867e0) was:

```cpp
Cluster::~Cluster() {
    log_debug(LogUMD, "Cluster::~Cluster");
    cluster_desc.reset();
}
```

No `close_device()` call. But since the constructor never called `start_device()`,
there is nothing to clean up — no RISC deasserts were done, no firmware was loaded,
no ETH channels were modified. The destructor's omission of `close_device()` is
**harmless in this code path** because there is nothing dirty to clean.

### The REAL failure: `cluster.close_device()` AttributeError → exit 1

The topology check Python code:

```python
cluster = tt_umd.Cluster()
n = len(cluster.get_target_device_ids())
if hasattr(cluster, 'close_device'):
    cluster.close_device()
else:
    sys.exit(2)  # fallback to METHOD 2
```

On the old Docker image:
1. `tt_umd.Cluster()` succeeds — cluster opened, 8 chips found.
2. `get_target_device_ids()` succeeds — returns 8 chip IDs.
3. `hasattr(cluster, 'close_device')` → `False` (old nanobind without the binding).
4. `sys.exit(2)` — signals METHOD 1 unavailable.
5. Python process exits, `~Cluster()` runs (old destructor: just `cluster_desc.reset()`).

Shell captures exit code 2, falls back to METHOD 2 (PCIe sysfs). **This is correct
behavior.** No contamination occurs.

### Wait — then what causes the "METHOD 1 failed unexpectedly" error?

Re-reading the shell script flow:

```bash
raw_output=$(t3k_count_via_umd 2>/dev/null) && umd_rc=0 || umd_rc=$?
```

The `2>/dev/null` redirects stderr. If `sys.exit(2)` fires, `umd_rc=2`, shell goes
to the `elif [[ $umd_rc -eq 2 ]]` branch → METHOD 2 fallback. **This should work.**

The "METHOD 1 failed unexpectedly" error (line 219) fires when `umd_rc` is
something other than 0 or 2 — specifically exit code 1 (from the `except` block).

**The ONLY way to hit exit code 1 is if `tt_umd.Cluster()` itself throws an
exception** (e.g., `Cluster()` fails due to hardware state, ARC timeout, PCIe hang,
NOC hang detection).

### Root cause hypothesis: Cluster() constructor failure after tt-smi -r

After `tt-smi -r`, the devices are in LONG_IDLE state. The `init_tt_device()` call
inside the constructor does `is_noc_hung()` check and `wait_arc_core_start()`. If the
ARC hasn't fully recovered from the warm reset when the topology check runs
(milliseconds after `tt-smi -r` completes), the constructor could throw:
- `NocHangError` if NOC appears hung post-reset
- Timeout in `wait_arc_core_start()` if ARC is still initializing

This would cause exit code 1 → "METHOD 1 failed unexpectedly" → hard fail.

### Does the contamination premise apply at all?

**No, not from the topology check.** The shell script comments (lines 101-108) describe
the behavior of the ORIGINAL check (`import ttnn; ttnn.GetNumAvailableDevices()`),
which DOES open a full tt-metal device stack → calls `deassert_risc_resets()`. That
old check was replaced by `tt_umd.Cluster()` which does NOT do this.

The contamination scenario described in the comments is **historically accurate for the
old ttnn-based check** but **does not apply to the current tt_umd-based METHOD 1**.

## Answers to Specific Questions

### Q1: Does `~Cluster()` call `close_device()`?

**On the branch (post-065867e0): YES.** Lines 568-578 show `~Cluster()` calls
`close_device()` in a try-catch if `device_closed_` is false.

**On the old Docker image (pre-065867e0): NO.** The old destructor only called
`cluster_desc.reset()`.

**But it doesn't matter** — since the constructor never calls `start_device()`, there
are no RISC deasserts to undo and no firmware to clean up.

### Q2: Does contamination survive into the test phase?

**There is no contamination from the topology check.** `tt_umd.Cluster()` does not
modify ETH channel firmware state. The `0x49706550` sentinel is the pre-existing state
from boot/reset — it's NORMAL, not contamination.

### Q3: What does the test framework do with 0x49706550?

`terminate_stale_erisc_routers()` (fabric_firmware_initializer.cpp:1194) recognizes
`0x49706550` as `BASE_UMD_FIRMWARE_SENTINEL`. It adds the channel to
`base_umd_channels` and skips all writes (no TERMINATE, no zero). Then
`configure_fabric_cores()` receives these as `skip_soft_reset_channels` and skips
`assert_risc_reset_at_core()` for them. Instead, the firmware transition happens via
`write_launch_msg_to_core()`.

This is the **expected clean-boot path** for all fresh/reset devices. Seeing 0x49706550
on all ETH channels is **normal**, not a sign of contamination.

### Q4: Is the hasattr fix better or worse?

The current fix (hasattr check + exit 2 fallback) is **correct and harmless**:
- If `close_device()` is available: call it (clean close), print chip count, exit 0.
- If `close_device()` is unavailable: exit 2 (before printing, so no stdout for shell
  to parse), fall back to METHOD 2.
- The cluster was never "started" so there's nothing to contaminate.

The fix does NOT make things worse. It's the correct approach for graceful fallback.

### Q5: What is the correct fix?

**The current code is already correct.** The topology check cannot cause contamination
because `tt_umd.Cluster()` does not call `start_device()`.

However, there are two improvements worth making:

1. **Fix the misleading comment in `py_api_cluster.cpp` lines 22-25.** The comment
   claims `~Cluster()` does not call `close_device()`, but since commit 065867e0 it
   does. This comment will confuse future readers.

   File: `tt_metal/third_party/umd/nanobind/py_api_cluster.cpp:22-25`
   ```
   OLD:
   // Calling this before the Cluster object is destroyed ensures ETH
   // channels are torn down cleanly — the default Cluster::~Cluster()
   // does NOT call close_device(), leaving ETH in base-UMD firmware
   // state (0x49706550 sentinel) which contaminates subsequent callers.

   NEW:
   // Explicit close_device() call for Python callers.  As of commit
   // 065867e0, ~Cluster() does call close_device() in its destructor,
   // but an explicit call is still preferred for deterministic cleanup
   // (destructor timing in Python/nanobind is non-deterministic).
   ```

2. **Check hasattr BEFORE instantiating Cluster** to avoid unnecessary device opens:

   File: `tests/scripts/t3000/run_t3000_unit_tests.sh` inside `t3k_count_via_umd()`:
   ```python
   # Before creating the Cluster, verify close_device() is available.
   # This avoids opening PCIe devices when we can't clean up properly.
   if not hasattr(tt_umd.Cluster, 'close_device'):
       sys.exit(2)
   ```

   This is a **minor optimization** — since Cluster() doesn't contaminate, opening
   without close_device is harmless. But it's good practice to not open devices
   unnecessarily.

## The Real Bug (if there is one)

If the "T3K topology check (METHOD 1) failed unexpectedly" error is actually happening,
the cause is NOT contamination. Possible causes:

1. **`tt_umd.Cluster()` constructor throws** — ARC not ready after `tt-smi -r`, NOC
   hang detected, PCIe link issue. The error goes to stderr (redirected to /dev/null
   by `2>/dev/null`) so it's invisible in CI logs.

2. **Race with tt-smi -r completion** — the `timeout 30 tt-smi -r || true` may
   return before ARC firmware is fully ready on all chips. A short delay or ARC-ready
   poll between reset and topology check might be needed.

**Recommendation**: Remove the `2>/dev/null` from the `t3k_count_via_umd` call in the
shell script (line 207: `raw_output=$(t3k_count_via_umd 2>/dev/null)`) so that any
`Cluster()` constructor errors are visible in CI logs. This will immediately reveal
the true failure mode.

## Summary

```
Contamination from topology check?     NO — Cluster() never calls start_device()
~Cluster() calls close_device()?       YES (on branch), NO (old Docker image)
Does it matter?                         NO — nothing to clean up without start_device()
Is the hasattr fix correct?             YES — graceful fallback, no harm
What causes "METHOD 1 failed"?          Cluster() constructor exception (unknown cause)
Action needed?                          Remove 2>/dev/null to see the real error
```
