# AutoFix startup report

2026-09-12, stage 7 Qwen3.8-27B. Diagnosis:
[AUTODEBUG_startup.md](AUTODEBUG_startup.md). All hardware experiments were run
serially by the coordinator. The isolated investigator only inspected source,
logs, device-node metadata and read-only process/driver records.

## Starting evidence

The exact startup path is current-checkout `python_env/bin/python`, with
`PYTHONPATH=.:$PYTHONPATH`, bounded by `timeout 60`, configuring
`ttnn.FabricConfig.FABRIC_1D_RING` then opening/closing
`ttnn.MeshShape(1, 4)` with `trace_region_size=0`.
The initial attempt exited 134 while returning device 0 Ethernet core 29-25 to
base firmware. No model code ran.

## Hypothesis experiments

| Hypothesis | Experiment / result | Verdict and retained fix |
| --- | --- | --- |
| Recoverable post-reset discovery fault | Coordinator ran `timeout 180 /home/mvasiljevic/tt-metal/python_env/bin/tt-smi -r` twice, each exit 0. First `timeout 60 /home/mvasiljevic/tt-metal/python_env/bin/tt-smi -ls --local` failed QUERY_MAPPINGS; second returned all four devices, exit 0. `recovery_reset*` and `recovery_list*` preserve evidence. | Verified that second reset restores discovery. This is infrastructure recovery; the exact ARC/ETH defect remains unproven. |
| Persistently stale Docker device nodes prevent access | Read `/sys/class/tenstorrent/tenstorrent!*/dev` and inspected `/dev/tenstorrent/0..3`: matching `237:0..3`; second listing succeeds through the same nodes. | Refuted as the current discovery blocker. No node edits. |
| Old listing library is deterministically incompatible | Same old `tt-smi` / tt-umd 0.9.5 changes from failed to successful listing after reset 2; current smoke resolves this checkout's `_ttnn.so` and UMD source. | Persistent listing ABI mismatch is not supported; broader compatibility remains untested. No package changes. |
| Mesh is recovered after discovery | Exact ring smoke exits 1 at sysmem NOC address validation: expected `0x1000000000000000`, actual `0x1000000040000000`. See `recovery_mesh.log` and `.exit_status`. | Refuted. Mesh remains unavailable; original heartbeat path is not yet retested. |
| A visible stale model process owns base sysmem | Coordinator `fuser -v /dev/tenstorrent/*` and investigator `/proc/*/fd` inspection find no visible device owners. Visible Python jobs are orchestration/UI serving; no live model job. | No container-visible owner found. Limited permissions/PID namespace prevent a host-wide refutation. No process killed. |
| Retained device owners lie outside this PID namespace | Read `/proc/driver/tenstorrent/{0,1,2,3}/pids`: four zero lines/device. Upstream KMD 2.8.0 emits one translated opener PID per open-file record; Linux emits zero for a PID absent from reader namespace. Source links and limits are in AUTODEBUG_startup.md. | Strongly supported: retained open-file records exist but opener PIDs cannot be resolved here. Not proof of distinct/live owner count or which owner pins sysmem. |
| Deleting stale UMD locks would release occupied sysmem | Current UMD fails after sysmem allocation returns a different NOC base, not on a userspace named-lock wait. KMD/file mapping lifetime is a separate boundary. | Not a justified repair. No locks deleted and no address-check bypass. |

## Final status

**External environment limitation; stage goal remains unfulfilled.** Discovery
is healthy, while mesh smoke still fails on retained sysmem address state. No
model implementation was changed, and no correctness/performance result was
produced by this investigation. Only documentation was authored; no build is
required for that change.

The required external action is a physical Docker-host operator inspecting
`/proc/driver/tenstorrent/*/pids`, device fds/maps and associated processes in the
host PID namespace, then releasing confirmed stale owners. If owner cleanup
cannot safely recover the mapping, the
[device skill](../../../../../.agents/skills/tt-device-usage/SKILL.md) recovery
sequence calls for host reboot and reservation re-acquire. The coordinator has
requested operator help; no reboot or ownership cleanup has been verified.

After operator recovery, run the bounded list/reset/list sequence and the exact
ring TP4 open/close smoke. Only a successful close plus `MESH_SMOKE_OK` establishes
readiness to resume preserved stage-7 work. Do not interpret `0` owner records as
killable PID 0 or bypass the sysmem-address safety guard.

## 2026-09-13 resumed recurrence

The operator's recovery commit b70863460380bdf57cc0a9c0d5769fe654ac161d
records that terminating the host telemetry collector emptied driver owners and
restored the exact mesh smoke. This validates the ownership intervention for
the prior failure. A documented collector respawn required supervisor suspension
and another cleanup. Current `ownership_revalidation_resumed.json` again shows
two unresolved external records per chip; container-root fuser has no owner.
The source-only investigator found no host-control route available here. Current
owner identity needs host inspection; do not reuse historical PIDs blindly.
No new mesh/reset experiment was run. Host help was requested for the recurrence.
