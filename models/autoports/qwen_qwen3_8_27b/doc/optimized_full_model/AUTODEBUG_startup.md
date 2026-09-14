# Startup recovery diagnosis

Source-only AutoFix investigation, 2026-09-12. This investigator did not import
TTNN, open devices, issue resets, or change implementation. The coordinator owns
all hardware commands. Scope: stage 7 Qwen3.8-27B startup on four Blackhole p300c
devices, KMD 2.8.0 and firmware 19.8.0.

## Evidence and current verdict

1. Initial ring mesh smoke failed before model execution: device 0 Ethernet
   virtual core 29-25 did not advance its heartbeat while returning to base
   firmware. `work_log.md` records exit 134.
2. First bounded reset returned 0 (`recovery_reset.*`); its following listing
   failed in `TENSTORRENT_IOCTL_QUERY_MAPPINGS` on device 0 (`recovery_list.log`).
   The recorded `recovery_list.exit_status` is **1**, despite an earlier oral
   observation of 0; use the artifact as the result.
3. Second bounded reset returned 0; the same listing tool then discovered all
   four devices (`recovery_reset2.*`, `recovery_list2.*`, listing exit 0).
4. The following smoke (`recovery_mesh.log`, exit 1) successfully completed
   topology discovery and constructed current-checkout UMD, then failed in
   `SiliconSysmemManager::pin_or_map_iommu`: expected sysmem NOC address
   `0x1000000000000000`, received `0x1000000040000000` (a 1 GiB offset).

Discovery has recovered. The present blocker is sysmem address ownership or
retained driver mapping state; the original Ethernet heartbeat has not yet been
retested because this smoke stops earlier. No model correctness or performance
conclusion follows from these failures.

## Hypotheses and source checks

| Hypothesis | Evidence and prediction | Verdict |
| --- | --- | --- |
| Transient post-reset PCI/firmware state | One reset leaves discovery broken; a second restores discovery without environment or device-node edits. Current UMD `device/api/umd/device/warm_reset_with_recovery.hpp:15` explicitly documents post-reset states requiring another reset. | Supported for the first post-reset discovery failure. Exact ARC/ETH cause is not proven. |
| Persistent stale Docker device nodes | `/dev/tenstorrent/0..3` have major/minor `237:0..3`, matching all four `/sys/class/tenstorrent/tenstorrent!*/dev` entries. `/dev` is container tmpfs, but the same nodes and tool successfully rediscover all devices. | No current supporting evidence; stale timestamp alone does not indicate an invalid character device. |
| Deterministic old UMD/KMD ABI mismatch | `tt-smi` uses `/home/mvasiljevic/tt-metal/python_env/.../tt_umd`, metadata version 0.9.5. Current environment metadata is 0.9.9, and smoke traceback resolves current-checkout `_ttnn.so` and UMD source. Old `pci_device.cpp:477–478` directly issues QUERY_MAPPINGS and discards errno; current `pci_device.cpp:438–442` uses `tt_device_query_bar_mappings` and includes the error text. However the same old tool succeeds after reset 2. | Different tools are confirmed; a persistent discovery ABI mismatch is weakened by the successful control. Do not attribute the current sysmem error to this difference without an isolated control. |
| Another process or retained KMD sysmem mapping occupies the base address | Current `device/chip_helpers/silicon_sysmem_manager.cpp:385–421` maps sysmem and rejects any NOC base different from `pcie_base_`. Its comment and error identify another holder as the usual cause. The observed 1 GiB offset is consistent with the first mapping interval being occupied. | Leading current hypothesis; owner or leaked mapping has not been identified. |
| Failed ARC recovery remains the immediate blocker | Latest UMD completes discovery, reports firmware and KMD versions, and reaches local-chip start before sysmem validation fails. | Not the immediate failure boundary. A later Ethernet failure remains possible after sysmem recovery. |

The initial heartbeat check is implemented at `tt_metal/llrt/llrt.cpp:557`:
read heartbeat, repeatedly send the stop flag, and wait for the value to change.
The timeout recommends board reset. Firmware 19.8.0 is above this error's stated
minimum 18.10.0; that minimum check alone does not establish full firmware/API
compatibility.

Read-only process inspection found no device fd visible to this user in this
container. `/proc/1/root` is permission-denied; the PID namespace and permissions
make this insufficient to rule out another host/container owner. The visible
Python zombies do not themselves retain open descriptors. The visible
`tools/tracy/serve_wasm.py --port 8080` is a static UI server, not evidence of a
hardware owner. Docker socket and kernel build tree are unavailable here.

## Focused next checks for the coordinator

1. Preserve `recovery_mesh.log` and inspect owners in the **host PID namespace**,
   with sufficient permissions: `fuser -v /dev/tenstorrent/*` or
   `lsof /dev/tenstorrent/*`, plus `/proc/*/fd` and `/proc/*/maps` if descriptors
   have been closed while mappings survive. Stop only identified stale processes
   belonging to this run. Container-local empty output is not proof of no owner.
2. After removing a confirmed stale owner, retry the exact bounded ring mesh
   open/close command once. Success verifies ownership as the cause; the same
   offset with no host owner supports retained driver mapping state instead.
   Clearing UMD named locks alone does not release a live KMD/sysmem mapping.
3. If no removable owner exists and the offset persists, follow
   `.agents/skills/tt-device-usage/SKILL.md` infrastructure recovery: operator
   host reboot/reservation re-acquire, then bounded list/reset/list and exact
   mesh smoke. Two reset attempts are already recorded; do not run unbounded
   additional resets or bypass the sysmem-address guard.
4. If discovery fails again, capture ioctl errno with a bounded, serialized
   `strace -f -e trace=openat,ioctl,mmap` around the existing listing command,
   if `strace` is available. Its old UMD error string omits errno, so the current
   log cannot distinguish `ENODEV`, `EINVAL`, or other driver errors. Avoid
   package changes until this experiment justifies a library hypothesis.

No implementation fix is justified by the present evidence. Resume the same
stage once the exact ring mesh open/close smoke passes, and record that as
infrastructure recovery rather than model validation.

## Follow-up: driver owner records

The coordinator's container-local `fuser -v /dev/tenstorrent/*` found no visible
owners. A subsequent read-only check of `/proc/driver/tenstorrent/{0,1,2,3}/pids`
returned four lines containing `0` for **each** device; this investigator
independently reproduced that result without opening devices.

Upstream KMD tag `ttkmd-2.8.0` implements `pids_proc_show` by iterating
`open_fds_list` and printing `pid_vnr(priv->pid)`; it does not print padding or
empty-device placeholders. These are four retained open-file records per device,
not necessarily four distinct processes. See
[KMD enumerate.c](https://github.com/tenstorrent/tt-kmd/blob/ttkmd-2.8.0/enumerate.c#L230).
Linux v6.17 `pid_vnr` translates to the calling task's active PID namespace;
`pid_nr_ns` returns zero when the recorded PID does not map to that namespace.
See [Linux pid.c](https://github.com/torvalds/linux/blob/v6.17/kernel/pid.c#L489).
This supports retained owners outside the container's PID namespace. Zero is
not a usable process ID and must never be fed into a kill command.

The exact installed KMD binary was not compared against upstream source; version
2.8.0 was read from sysfs. The owner PIDs, their current liveness, and which record
retains the base sysmem mapping still need host-level inspection. The operator
must identify/release the relevant owners or reboot/re-acquire the physical host
if safe owner cleanup is unavailable. Container-local lock deletion, another
model-code edit, or disabling the UMD address check cannot establish recovery.
