# Data sources

Everything here is read-only. `/data` is shared NFS: `triage` searches it with `nice` and parses only the
files of one host that can hold runs from its window.

## Machine HC (`/data/ci-provisioner`)

Per-host health check driven by Slurm (`hc_dispatch`), one run per host per job.
Source of truth for the checks: `tools/scaleout/exabox/health_check_test_suite/`
(`HEALTH_CHECK.md`, `diag_runner.py`), triage tools in `tools/scaleout/kmd_triage/`.

Run directory: `workflows/hc_logs/<host>/<YYYY-MM-DD>/<slurmjob>[_post-reboot-N]/`

| File | What |
|---|---|
| `diag_report.json` | Structured verdict. Top level: `tool_version`, `tier`, `host`, `started_utc`, `ended_utc`, `tt_smi_version`, `detected_board_rev`, `overall_status`, `phases` (dict name -> `{name,status,duration_s,checks[]}`) |
| `console.log`, `harness.log`, `<host>-<job>.log` | Console output of the run |
| `logs/gddr_fast.log`, `eth_link_up.log`, `eth_bandwidth.log` | gtest output (tt-metal log lines `YYYY-mm-dd HH:MM:SS.mmm | level | module | msg`) |
| `logs/reset_N_r.log`, `reset_N_glx_reset.log` | tt-smi reset output (UMD errors, `Error when re-initializing chips!`) |
| `logs/triage_host_side.{txt,json}`, `triage_device_side.{txt,json}` | kmd_triage output. JSON = `{tool, version, checks[]}`; txt has a device table `ordinal BDF u<ubb>c<chip> bridge cur max ...` and a kernel-log section with PCIe/IOMMU/MCE lines |
| `logs/qsfp_dump_<host>.jsonl`, `qsfp.txt`, `qsfp.log` | QSFP/ETH dump + ingest summary |
| `logs/telemetry_<host>.prom` | Prometheus text telemetry |

Phases: `snapshot`, `reset_loop`, `snapshot_after_r1_r`, `snapshot_after_r2_glx_reset`,
`tests`, `post_test_reset`, `triage`, `qsfp_tests`, `sys_triage`. Each check:
`name`, `status` (PASS/WARN/FAIL/SKIP), `details` (human string), `data` (dict), `ip`
(asic/board/eth/fw/gddr/other/pcie/thermal). Triage check names get a `triage_` prefix.

Chip identity in checks: `details` strings carry `0000:81:00.0 (UBB4/U1)`; `data`
carries BDF lists (`missing_bdfs`, `failed_chips`, `failures[].bdf`, `off_speed`).
**The bus -> UBB mapping differs per host family** (on bh-glx-110-c10u08 bus 0x81 is
UBB4, on bh-glx-120-b10u08 bus 0x41 is UBB2), so learn it per host from details strings
and triage tables; never hardcode. Device ordinals (`device 17`, `/dev/tenstorrent/3`,
`PCIe ID 6`) are driver probe order, roughly sorted-BDF order of the chips present.

Volume (2026-10-14): 202 hosts (104 bh-glx-110, 98 bh-glx-120), 3,477 reports
(overall WARN 2,555 / FAIL 921). Almost every run WARNs (`cpld_fw_old`, `post_reset_state_stable`).

Failure examples worth knowing:
- `gddr_fast` / `eth_link_up` FAIL can be `TT_FATAL: Graph specified in MGD could not fit
  in the discovered physical topology` (an ETH/topology problem, not GDDR): classify test
  failures by log content, not by test name.
- `dram_status` 0/32 + `gddr_training_per_channel` 224/256 with `train_done=0 err=0` on
  channel == chip index: systemic (all chips), likely FW/telemetry, not 1 bad GDDR.
- `gddr_speed` 0/32 at 14G (all 16G on RevA/B): board-rev detection/FW config.
- `reset_1_r` FAIL: `Sending reset ioctl to device 17 failed ... KMD returned: -19`,
  `Failed to open device /dev/tenstorrent/3`, `post_pcie=31` (chip fell off the bus).
- `cpld_auto_recover` / `*_glx_reset` FAIL: `Error in resetting galaxy 6u trays! [Errno 6]`.
- `snapshot_capture` FAIL: `Read 0xffffffff over PCIe ID 6: the board should be reset.`
- `pcie_enum_count` 24/32 + `asic_location_per_ubb` `missing UBBs: [4]`: whole tray gone.
- `eth_links_up`: `failures[] {bdf, physical_asic, down_ports, expected_mask, actual_mask}`.

Machine HC also files Jira tickets in **DCAMP** (label `fabric-health-check`, summary
`[Fabric Health Check] Test failed on <host>`, 997 tickets) with run artifacts attached;
it comments on recurrence and auto-closes on recovery (`utils/jira_client.py`).

Other `workflows/` dirs:
- `hc_dispatch/<host>-slurm-<job>.log` (2,270): docker pull noise + run console + Jira lookup lines.
- `fabric_system_health_check_scheduled/` (884), `..._historical/<host>-slurm-<job>.log` (6,303): older fabric HC console logs.
- `fhc-reboot-test/`.

Funtest: `funtest_test_logs/*-ft-ddr-stress.log`, `*-ft-pt-{1,2}.log` (per host or container id);
`ccl_tests_output/test_results/*.log`.

## Cluster HC (`/data/dcamp`)

Multi-host checks run by the exabox tooling (`tools/scaleout/exabox/`), recorded as
`exabox.cluster_health.v1` JSON.

`cluster-health-check/reports/<type>/<date>/<record_id>.json` (11,313) + `reports-archive/` (220).
Keys: `schema, ts, test_type, status (passed|failed|degraded), hosts[] (FQDNs), artifact_uri,
source, trigger_kind, labels{superpod,ring,quad}, record_id, record_uri, cluster, triggered_by,
orchestrator_id, duration_s, analyzer_code, pass_pct, topology`. No failure reason field.

| type | passed | failed | degraded |
|---|---|---|---|
| recover | 7,053 | 3,405 | 62 |
| physical | 167 | 122 | 22 |
| fabric | 202 | 21 | 182 |
| host | 0 | 2 | 5 |
| dispatch | 27 | 7 | 2 |

`analyzer_code` (physical/fabric only) = exit code of `analyze_validation_results.py`:
0 healthy, 1 unhealthy links repeated, 2 unhealthy scattered, 3 DRAM failure,
4 missing connections, 5 extra connections, 6 missing global connection, 7 FSD error,
8 MGD error, 9 workload timeout, 10 ARC timeout, 11 AICLK timeout, 12 network error,
13 missing devices (`Read 0xffffffff over PCIe|PcieHangError|ArcStartupError`),
50 inconclusive, 66 input error (124/126 are shell timeout/exec failures of the wrapper).

`artifact_uri` often points at deleted paths (`/tmp/ttop-recovery-*`, `/home/user/quad-health/...`);
in a 1,393-record sample of recover, 430 exist. Physical 38/42 exist, fabric 34/58.

Logs:
- `cluster-health-check/logs/recover/<user>-<ts>.<rand>/recover_<ts>.log` (2,922): every line
  prefixed `[<host>][HH:MM:SS]`. Some are 0600 (unreadable): skip quietly.
  Useful lines: `Error: POST_RESET failed for device 25.`, `Reset failed | Exit code: 1`,
  `cross-host port down FAILED (exit code 241)`, `finding: pci:16 would not open: Query mappings failed on device 16.`,
  `Recovery attempt 1 of 1 failed (exit code 241).`
- `logs/{fabric,physical,host,dispatch}` are empty.
- `/data/stackstorm/` (890): `<execution-id>/logs/recover-<ts>.log`, `<id>/fabric_tests`,
  `nightly-health-check/<quad|SC>/physical_validation/<quad>/<id>_<ts>/cluster_validation_iteration_N.log`
  (+ `iteration_N/ethernet_metrics_report.csv`).

BMC reboot collection: `/data/dcamp/bmc-reboot-data-collection/<host>/<date>/<HH:MM:SS>/`
`reason.txt` (free text like `verifying preboot tier`), `console.log`, `diag_snapshot.json`,
`diag_report.json`, `logs/` (qsfp, triage). Same diag format as machine HC.
`qsfp.txt` format: `tt-bh-glx-cluster-debug ingest v1` then lines like
`qsfp_link_training FAIL 246/256 ... down: ubb=1/asic=1/eth=0 (LINK_TRAIN_TIMEOUT_LINK_UP)`.

Unreadable: `/data/dcamp/cluster-debug/logs/<host>/` (permission denied), some
`cluster_*.jsonl`, `/data/dcamp/bmc-update-logs` (0770 ubuntu).

## Jira

Site `tenstorrent.atlassian.net`, cloudId `b9d94484-5dbd-4ae2-b670-6f414aefb4cd`.

- **DIIM**: human-filed incidents, 589 issues, all Bug. Hosts named `bh-glx-120-d06u08`,
  short `a24u20`, YYZ2 `bh-glx-803-a30u08`, other sites `g11blx03`, `qb2-120-p06t06`;
  chips as `device 22`, `chip 19`, `ASIC7`, `UBB4`, `tray 2`, `bank 7`, `u2c5`.
  Only 158 have labels or components, so categorize by text.
- **DCAMP**: 1,686 issues. 995 are `[Fabric Health Check] Test failed on <host>` from
  `exabox-ci-jira-bot` (2026-04-02 on): one per machine HC failure that survives the reboot recovery.
  Description: node, date, `Slurm Job ID` (= the hc_logs run directory name), versions, last lines of
  output. The bot comments "failed again (recurring failure)" and closes the ticket when HC passes;
  humans mostly close duplicates. No root causes are recorded there. 691 other DCAMP issues are
  infra/ops tasks.
- `bmc bot` / `bmc_bots` = the tool people use to reach a host's BMC and power-cycle it;
  BMC tickets are "BMC ip not reachable", "bmc unavailable", ipmitool failures.

### DCAMP bot attachments

hc_logs on disk start 2026-09-11, but the bot tickets carry the failing runs' files back to April. `triage`
doesn't read them (they'd need downloading); they are the place to look for older history:

| era | attached |
| --- | --- |
| Apr-Jun | `<host>-<job>.log`: console of the older fabric health check (gtest `Cluster.*` + UMD lines) |
| Jul-Oct | `diag_report[-<job>].json`, `<check>[-<job>].log`, `triage_host_side-<job>.txt`, console `<host>-<job>.log` |

Every bot-uploaded `diag_report` is **base64 inside quotes and cut off** (~75% of the file): the snapshot
phase survives, the tests phase (why the run failed) is usually lost. Early tickets use pre-superpod names
(`bh-glx-b05u02`); some reports name a container (`6ddeaba6e2a2`) instead of the host.

Access: REST with an API token (`~/.config/device_triage/jira.env`, 0600), GET only. `text ~` searches
summary, description and comments, but doesn't split `bh-glx-120-b03u02` into words: search the full
name and the short one (`b03u02`).
