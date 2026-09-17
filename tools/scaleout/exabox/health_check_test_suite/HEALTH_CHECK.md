# Health check Blackhole Galaxy diagnostic tool

Pre-cluster hardware sanity check for Blackhole Galaxy 6U systems. Captures a
`tt-smi` snapshot, decodes per-chip telemetry, runs a reset stability loop,
invokes the `unit_tests_deployment` gtest binary, and on the longer tiers folds
in the first-step triage tools and the QSFP tests. Emits a single JSON
report with per-check PASS/WARN/FAIL/SKIP status grouped by IP.

## Quick start

```bash
export TT_METAL_HOME=/path/to/tt-metal
cd $TT_METAL_HOME

# Full light tier (snapshot + tt-smi -r + eth_link_up gtest, ~75s)
./tools/scaleout/exabox/health_check_test_suite/run_diag.sh light

# Snapshot-only smoke (fastest iteration)
./tools/scaleout/exabox/health_check_test_suite/run_diag.sh light --skip-reset --skip-tests

# Offline / dev iteration against a stored snapshot
./tools/scaleout/exabox/health_check_test_suite/run_diag.sh light --dry-run \
    --input-snapshot snap.json --output /tmp/diag_report.json
```

Output goes to `./diag_report.json` by default; gtest logs to `./logs/<test>.log`.

## Tiers

| Tier | Resets | Tests | Triage | QSFP tests | Duration | Use when |
|---|---|---|---|---|---|---|
| `light`  | `tt-smi -r` × 1                            | eth_link_up                                                                | — | — | ~75 s   | Smoke check on every new unit |
| `medium` | `tt-smi -r`, `tt-smi -glx_reset`, then `-glx_reset` after the tests | eth_link_up + eth_bandwidth + gddr_fast (DRAM_TEST_FAST=1)                | host_side + device_side | yes, if installed | ~5 min + triage + ~7 min | Pre-deployment validation |
| `deploy` | `tt-smi -r`, `tt-smi -glx_reset` × 2, then `-glx_reset` after the tests | eth_link_up + eth_bandwidth + full gddr matrix (3 DramDeployment tests) + didt_matmul_galaxy (pytest, ~9 min) | host_side + device_side | yes, if installed | ~18 min + triage + ~7 min | Final deploy gate |

The eth deployment tests are registered as `TensixDeploymentEthernet<NN><Name>`
(e.g. `TensixDeploymentEthernet00LinkUp`, `TensixDeploymentEthernet01Bandwidth`,
`TensixDeploymentEthernet02BandwidthBidir`), so the filters wildcard the
two-digit index: `eth_link_up` is `*TensixDeploymentEthernet*LinkUp` and
`eth_bandwidth` is `*TensixDeploymentEthernet*Bandwidth*`. The trailing wildcard
on `eth_bandwidth` means it runs both `Bandwidth` and `BandwidthBidir`, both of
which `tests/tt_metal/tt_metal/deployment/sources.cmake` now compiles.

(Filters without the index wildcard match zero tests, so gtest exits 0 and the
check is silently recorded as PASS without running — avoid that form.)

The deploy tier also runs `didt_matmul_galaxy`, a pytest-based dI/dt stress
test: 1000 matmul iterations on the full (8, 4) mesh with a determinism
re-check every 50, via the repo venv's pytest. Measured ~9 min on a BH galaxy
(mostly cold bring-up + the 20 output readbacks); bump
`--didt-workload-iterations` when triaging a suspect unit — a marginal chip
that passed 1000 iterations failed under a 5000-iteration soak. `--timeout
1200` overrides the repo-wide 300 s pytest-timeout (too short for this run)
while still bounding a wedged run in standalone invocations, and
zero-collected runs are recorded as FAIL, same trap as the zero-match gtest
filters above. Needs the venv (`./create_venv.sh`) and standard mesh cabling —
on torus-cabled units set `TT_MESH_GRAPH_DESC_PATH` to the matching descriptor
or fabric mesh mapping fails.

The reset cadence and test set are defined in `RESET_PLAN` / `TIER_TESTS` /
`PYTESTS` in `diag_runner.py`.

## Triage phase

`medium` and `deploy` end with a post-test `tt-smi -glx_reset` followed by the
first-step triage tools — `host_side.sh` (host, PCIe and driver state, read from
sysfs) and `device_side.sh` (per-chip liveness, ARC scratch, telemetry and a NOC0
node sweep). They live in `tools/scaleout/kmd_triage/`. Tables are
`POST_TEST_RESET_PLAN`, `TRIAGE_TOOLS` and `TIER_TRIAGE` in `diag_runner.py`.

**What the two tools actually check, how to run them by hand, their exit codes
and their known gaps are documented in
[`tools/scaleout/kmd_triage/README.md`](../../kmd_triage/README.md).** The rest
of this section is about how the phase drives them and how their findings get
into the report.

Neither tool depends on tt_metal or UMD, which is what makes them useful when
the runtime will not load. `device_side.sh` drives `kmd_triage`, a standalone
binary built as a normal CMake target into `build/tools/scaleout/kmd_triage`. It
used to be compiled at run time; it is a build artefact now, so a host running
the health check needs no compiler, and a missing binary is reported as lost
coverage rather than silently skipped.

The reset sits between the tests and the triage for two reasons: the triage probes
open every chip read-write, so they must not overlap the gtests, and the tools have
no SIGBUS handler — tt-kmd zaps every mapping on reset, so a reset concurrent with a
probe kills it outright instead of reporting cleanly. A discrete reset that completes
first rules both out. It is a *bare* reset: no revalidation snapshot, so it stays
clear of the post-reset snapshot dedupe (which assumes one batch of
`snapshot_after_*` phases, judged by `normalize_health_report()` on `post[-1]`).

Because `reset_loop()` names its checks `reset_*` and `report.py` classifies those
as reset ops and drops them from the verdict, the phase adds a
`post_test_reset_ok` check — nothing runs after this reset, so a reset that broke
enumeration would otherwise leave no mark on the verdict.

### The interface

The phase drives both scripts the same way — `<script> --json <path> -o <path>` —
and reads the JSON file back. Each check carries a `status` of
`PASS`/`WARN`/`FAIL`/`SKIP` and an `ip` from the `IP_ORDER` groups. The shape and
the emitter live in `kmd_triage/triage_json.sh`, which both scripts
source — one copy, because two would let the shape drift and the consumer would
have no way to tell which it was looking at. The `Phase` and its rollup are built
on ingest, not in the scripts, so FAIL > WARN > PASS stays computed in one place.

Each script emits **one check per class of finding, not per device**, so the
check names are the same on a 1-chip host and a 32-chip Galaxy; the offending
devices go in `details` and `data`. That matters because the dashboard keys its
routing on the check name, so a name that varied with chip count would fragment
the history.

`normalize_triage_check()` is defensive on ingest: names get a `triage_` prefix
(`CHECK_CATEGORY`, `EXCLUDED_CHECKS` and `_find_check()` in the analyzer are
keyed on the bare name across *all* phases, so an unprefixed `pcie_gen` would
inherit that check's routing), an unrecognised status becomes WARN rather than an
UNKNOWN severity in the CSV, and an unrecognised `ip` folds to `other` so it
can't vanish from the console summary.

### Failure modes are SKIP or WARN, never a silent PASS

- Script absent, tier doesn't ask for it, nothing to probe → **SKIP with the
  reason in `details`**. A check that silently disappears reads as coverage we had —
  the same trap as the zero-match gtest filters above.
- Timed out, wrote no JSON, or reported its own failure (`rc=3`) → **WARN**. That is
  lost coverage, not a statement about the hardware. A missing `kmd_triage` binary
  lands here; the tool's first line of stderr is appended to `details`, so the
  console says *why* rather than only that nothing came back.
- Findings from the tools' JSON → recorded as-is, except that **FAIL is held at WARN
  unless `--triage-gating` is passed**. Deliberate while the tools bed in: on the one
  32-chip Galaxy they were verified against, `host_side.sh` returned DEGRADED on five
  correctable-AER findings on a unit `device_side.sh` and the rest of the suite called
  healthy. Gating on that from day one would ticket the fleet.

The text reports land in `<output_dir>/logs/triage_<tool>.txt`, so
`collect_run_artifacts()` attaches them to the JIRA ticket with no extra wiring.

### In-container caveats

The health check runs unprivileged in the tt-metal image, which is fine for most of
what these tools read — `/sys` is a real view of host sysfs, so PCIe link state, AER
counters, BAR assignment and the driver's `tt_*` telemetry all work. The
`kmd_triage` binary has to be in the image: `build/tools` is part of
`ARTIFACT_PATHS` in `build-artifact.yaml` and the image copies `build/`
wholesale, so a normal `build_metal.sh` puts it there. Three things degrade, all
for want of a capability rather than a mount:

- **Kernel-log scan** needs `CAP_SYSLOG` (or `kernel.dmesg_restrict=0` on the host).
  Without it the tool reports `NOT CHECKED` rather than a clean log.
- **`lspci -vvv` capability blocks** need `CAP_SYS_ADMIN` in the initial user
  namespace; without it config-space reads are clamped to 64 bytes. AER counters are
  unaffected — they are sysfs attributes, not config-space reads.
- **debugfs driver mappings** are absent: the container gets a fresh `sysfs` mount,
  which does not carry the `/sys/kernel/debug` submount.

Also note `/proc/driver/tenstorrent/<N>/pids` lists *host* PIDs, which don't resolve
in the container's PID namespace — the holder count is right, the names are not.

## QSFP tests phase

`medium` and `deploy` finish by collecting an ETH dump with
`tt-bh-glx-cluster-debug collect --parallelize`, when that binary is on PATH.

**It answers a question nothing else in this suite reaches.** The snapshot phase
reads `ETH_LIVE_STATUS` out of tt-smi telemetry — a bitmask per chip — and the
eth gtests push traffic over links that already came up. Neither can say *which
far end* a port is actually talking to. The dump carries the expected partner
and the firmware's own `remote_info` on each of the 448 port records, so a
miscabled or half-trained link becomes a comparison instead of an inference. Its
cage and module records are also the only inventory of what is physically
plugged in, which is what makes "this module has thrown errors in three
different cages" answerable later from stored runs.

### Where it comes from, and what happens without it

The tool ships in the syseng cluster-debug **.deb**, built with PyInstaller — one
binary on PATH, no repo, no venv, no import path to arrange. So the phase probes
for a *command* rather than for a checkout the way the triage phase probes for
its scripts, and a host that has not had the package installed simply **SKIPs
with that reason on the report**. Nothing needs configuring to turn it on:
install the package and the next `medium` run collects a dump.

It runs last, after the post-test `-glx_reset`, for the two reasons the triage
phase sits there — it opens every chip, so it must not overlap the gtests, and
it has no SIGBUS handler, so a concurrent reset would kill it outright rather
than being reported. Reading the links *after* that reset is also the reading
that matters: it is the state the machine is being left in.

### The conversion layer

The dump is an inventory, not a verdict. By design it states what the topology
expects and what the firmware found, side by side, and leaves the comparison to
the reader — nothing in it says PASS or FAIL, and its JSONL record shape looks
nothing like this suite's report.

`qsfp_ingest.py` is that reader. Each fault class it derives is one of
the access patterns the dump was shaped to answer (`cluster_debug_spec.md`
§10.3, declared as SQL in the tool's own `visualizer/queries.py`), re-derived in
Python over a single dump so the health check needs neither the merge step nor
the SQLite database. It emits **the same shape the triage scripts emit** —
`{"checks": [{"name", "status", "details", "ip", "data"}]}` — so
`normalize_external_check()` folds it in on the same path, with the same
defences, and the checks reach the JSON report, the console summary and the
Superset CSVs with nothing downstream needing to know the phase exists.

That shared normalizer is why names carry a `qsfp_` prefix (the analyzer
keys `CHECK_CATEGORY`, `EXCLUDED_CHECKS` and `_find_check()` on the bare name
across *all* phases) and why an unrecognised status or `ip` is folded rather than
passed through. It also honours an optional `console_visible: false`, which is
how the store-only checks below stay out of the console without leaving the JSON
— the same treatment the snapshot phase gives its `gddr_info_*` counters.

The ingest module is separately runnable, with the triage scripts' interface, so
a dump collected on a sick machine is reviewable at a desk:

```bash
python3 qsfp_ingest.py qsfp_dump_bh-glx-110-a07u02.jsonl --json out.json
```

### Coverage depends on the descriptor, and the checks say so

Internal links — the 104 soldered ones — always have an expected partner, from
the topology table built into the collector. Cage-attached links only get one
from a `factory_system_descriptor.textproto`, passed with
`--qsfp-descriptor`. **Both halves of "a descriptor applied" matter**: a
file that was supplied but does not name this host leaves the cabled links
exactly as undescribed as no file at all, so the ingest checks `PRESENT` *and*
`MATCHED_HOSTNAME`.

Without one, `qsfp_cage_gaps` SKIPs (an empty cage cannot be told from an
unused one) and an undescribed trained link is recorded without alerting rather
than reported as a surprise — otherwise every one of the ~36 cabled links inside
the chassis would be a finding on every run.

### Statuses: PASS or SKIP by default — never WARN, and not yet FAIL

The tool is still being validated against the fleet, so by default the phase
reports only two things:

- **SKIP** — no reading was taken: the tier doesn't ask for one, the package
  isn't installed, the ingest module is missing, the collect timed out, or the
  dump would not parse. Always with the reason attached, on the same reasoning
  as the triage phase: a check that silently disappears reads as coverage we had.
- **PASS** — a reading was taken. Whatever the dump found is recorded in the
  check's `details` and `data`, annotated `[advisory: FAIL recorded as PASS,
  --qsfp-gating off]` or `[advisory: WARN recorded as PASS]` so a PASS above a
  detail line that sounds like a problem doesn't read as a contradiction.

Nothing is lost by this — the finding, its offending port paths and its full
`data` all reach the JSON report and the CSVs exactly as the ingest produced
them. Only the status is held, and only because a tool nobody has yet confirmed
is right should not be the thing an operator's eye is drawn to when triaging a
rack. A held FAIL goes straight to PASS rather than sliding down through WARN,
since the phase has undertaken not to raise one.

**The ingest module is not subject to this policy.** Run standalone against a
stored dump it reports WARN and FAIL normally, so reviewing a dump by hand shows
the real assessment. The policy is applied on ingest by `normalize_qsfp_check()`,
which is the only place that knows it is feeding a fleet verdict.

### Turning the findings on

`--qsfp-gating` changes this completely and in one step: findings report at their
real severity and the phase gates the run like any other — `overall_status`, the
exit code, `has_actionable_failure()`, the JIRA ticket and the Slurm
reboot-and-requeue all respond to a QSFP FAIL. `Phase.gates` in the JSON records
which mode a run was in, and `report.py` plus the CSV analyzer both read that one
flag rather than each keeping a list of which phases count. A phase with no
`gates` key gates, so every other phase and every report written before the flag
existed is unaffected.

Turning it on is the decision to make once there is enough fleet history to say
the findings are right — which is why the phase records them from the first run
either way. In `checks.csv` a non-gating phase's rows carry `acknowledged=1`,
keeping them out of `top_fail_category` and `checks_warn_actionable` while
leaving the raw counts intact.

The check most likely to produce a false FAIL under gating is
`qsfp_link_training`, and it is already filtered for it: a port behind an empty
QSFP cage cannot train, so it is set aside rather than counted. See the note
under the check table below for what that filter does and does not cover.

The dump lands in `<output_dir>/logs/qsfp_dump_<host>.jsonl` (~3.4 MB with
cages) alongside `qsfp.txt` and `qsfp.log`, so
`collect_run_artifacts()` attaches all three to the JIRA ticket with no extra
wiring.

**Budget.** The cage sweep is the dominant cost and the tool self-bounds it at
its own 600 s `--qsfp-budget`; a full run measures ~7 min. The phase's own
timeout is 1200 s as a backstop. That has to stay well inside
`run_health_check.py`'s whole-run `--timeout-minutes` (30 by default), which
kills the process group and takes the report with it — so raise it when running
`medium` or `deploy` on a host that has the package.

## Flags

| Flag | Default | Purpose |
|---|---|---|
| `--tier {light,medium,deploy}` | required | Selects reset cadence + gtest matrix |
| `--dry-run` | off | Print intended subprocess calls; skip destructive steps |
| `--skip-reset` | off | Skip the reset loop phase entirely |
| `--skip-tests` | off | Skip the gtest phase entirely |
| `--skip-triage` | off | Skip the post-test reset and the triage phase entirely. `--skip-reset` also suppresses the post-test reset. |
| `--triage-dir PATH` | `$HC_TRIAGE_DIR`, else `<repo>/tools/scaleout/kmd_triage` | Directory holding the triage scripts. Override only to run a working copy against a deployed checkout. |
| `--triage-gating` | off | Let triage FAILs gate the run. Off holds them at WARN (noted in `details`); findings are recorded either way. |
| `--skip-qsfp-tests` | off | Skip the QSFP tests entirely. Named for the phase, not the collector's own `--skip-qsfp`, which drops the cage sweep but still collects ETH. |
| `--qsfp-tool-path PATH` | `tt-bh-glx-cluster-debug` on PATH | Override the collector binary. A path that doesn't resolve is reported as its own SKIP rather than silently ignored. |
| `--qsfp-descriptor PATH` | — | `factory_system_descriptor.textproto`, which gives the cage-attached links an expected partner. Without it only the soldered internal links are compared against a topology. |
| `--qsfp-gating` | off | Report QSFP findings at their real severity and let them gate the run. Off records every finding in `details`/`data` but holds the status at PASS. |
| `--input-snapshot PATH` | — | Use a stored snapshot instead of calling tt-smi |
| `--tt-smi-path PATH` | `/opt/tt_metal_infra/.../tt-smi` else `tt-smi` on PATH | Override tt-smi binary or repo path |
| `--tt-metal-path PATH` | `$TT_METAL_HOME` | tt-metal repo root (must contain the deployment-test binary under `build_Release/`) |
| `--output PATH` | `./diag_report.json` | Where to write the JSON report. Gtest logs go to `<output_dir>/logs/`. |
| `--snapshot-out PATH` | `/tmp/diag_snapshot.json` | Where `tt-smi -f` writes the raw snapshot |

## Checks (grouped by IP)

### Board
| Check | Rule | On fail |
|---|---|---|
| `board_rev` | All chips' `board_info.board_id` share one known prefix → `RevA/B` (`00000471…`) or `RevC` (`00000473…`). Sets the per-rev expected values for `gddr_speed` and `pcie_gen` below. | **FAIL** if mixed-rev across chips, or any chip has an unrecognised prefix. Downstream rev-dependent checks SKIP. |

Per-rev hardware expectations (Confluence SYS-4055):

| Rev | board_id prefix | GDDR speed | PCIe gen (U6 only) |
|---|---|---|---|
| RevA/B | `00000471…` | `14G` | Gen4 |
| RevC   | `00000473…` | `16G` | Gen5 |

The detected rev is also recorded at the top of the JSON report as `detected_board_rev`.

### Host
| Check | Rule | On fail |
|---|---|---|
| `host_fru_info` | Store-only: BMC FRU inventory via `sudo -n ipmitool fru print`. UBB tray serials in the details; full field set in `data.devices`. | never alerts — **SKIP** when ipmitool/sudo/BMC is unavailable |

### PCIe
| Check | Rule | On fail |
|---|---|---|
| `pcie_enum_count` | All 32 chips enumerated | **FAIL** |
| `pcie_lane_width` | x8 on host-PCIe chips (`ASIC_LOCATION==0x6`); x1 elsewhere | **FAIL** |
| `pcie_gen` | Host-PCIe (U6) chips trained to at least the rev's expected gen (Gen4 on RevA/B, Gen5 on RevC). Above-spec is rendered with `(^)` but doesn't degrade status; under-spec is rendered with `(!)` and WARNs. | **WARN** (known: occasional Gen1 fallback). **SKIP** when `board_rev` is indeterminate. |

### GDDR
| Check | Rule | On fail |
|---|---|---|
| `dram_status` | `board_info.dram_status == True` per chip | **FAIL** |
| `enabled_gddr_full` | `ENABLED_GDDR == 0xff` per chip | **FAIL** |
| `gddr_training_per_channel` | All 256 channels (32 × 8) trained per `DDR_STATUS` | **FAIL** |
| `gddr_bist_per_channel` | All 256 channels BIST-passed per `DDR_STATUS` | **FAIL** |
| `gddr_speed` | `board_info.dram_speed` matches the rev's expected speed (`14G` on RevA/B, `16G` on RevC) | **FAIL**. **SKIP** when `board_rev` is indeterminate. |
| `gddr_info_*` (5.1+) | Store-only: per-pair GDDR temps, corr-err counts, `GDDR_UNCORR_ERRS`, `MAX_GDDR_TEMP` | never alerts — forensics only |

### ETH
| Check | Rule | On fail |
|---|---|---|
| `eth_links_up` | Per chip, every enabled internal (non-QSFP) port reports live. Expected mask = `(ETH_INTERNAL_BY_ASIC \| ETH_EXAMAX_BY_ASIC) & ENABLED_ETH`, with the topology tables indexed by *physical* `ASIC_LOCATION` (derived from the BDF, not FW telemetry). Compared against `ETH_LIVE_STATUS` masked to the same non-QSFP set. | **FAIL** if any expected port is down (reports OK chip count + first failing BDF and `down_ports`). **WARN** if `ETH_LIVE_STATUS=0x0` on all chips despite FW ≥ 19.9. **SKIP** if the field is absent from the snapshot, or is all-zero on FW < 19.9 (the field is only populated by FW bundle ≥ 19.9, so a capability gap isn't misattributed to a real link fault). |
| `eth_speed` | (no field in current tt-smi schemas) | SKIP — pending future schema |

The `eth_link_up` gtest (test phase) is distinct from the snapshot-phase
`eth_links_up` check above: the gtest actively pushes traffic over each link,
while `eth_links_up` reads the `ETH_LIVE_STATUS` telemetry from the snapshot.

### ASIC
| Check | Rule | On fail |
|---|---|---|
| `asic_location_per_ubb` | Each UBB tray reports ASIC_LOCATION 1..8 | **FAIL** (reports `missing UBBs` / `missing ASICs` / expected BDFs) |
| `physical_vs_fw_location` | BDF low nibble matches firmware `ASIC_LOCATION` | **FAIL** |
| `harvesting_state` (JSON-only) | `HARVESTING_STATE` ∈ {0, 1} per chip | **FAIL** |

### FW
| Check | Rule | On fail |
|---|---|---|
| `fw_bundle_version_consistent` | Same `fw_bundle_version` on all 32 chips | **WARN** |
| `cm_fw_consistent` | Same `cm_fw` on all chips | **WARN** |
| `eth_fw_consistent` | Same `eth_fw` on all chips | **WARN** |
| `gddr_fw_consistent` (5.1+) | Same `gddr_fw` (M-RISC fw) on all chips | **WARN** |
| `dm_app_fw_consistent`, `dm_bl_fw_consistent` | — | always **SKIP** (DMC not applicable to Galaxy) |

### Thermal (JSON-only)
`asic_thermal_precheck` records the hottest chip / temp vs `thm_limit` for forensics.

### QSFP tests (medium / deploy, when the package is installed)

Derived from the ETH dump; see [the phase section](#qsfp-tests-phase) above
for how they get here. **The status column below is what the ingest assesses.**
Without `--qsfp-gating` the phase records every one of them as PASS, keeping the
finding in `details` and `data`. Checks marked
JSON-only are store-only forensics kept out of the console summary.

| Check | Rule | On fail |
|---|---|---|
| `qsfp_collect` | The collect run itself | **SKIP** on timeout, no dump, an unreadable one, or when the tier doesn't ask, the package isn't installed or the ingest module is missing — all of them mean no reading was taken. |
| `qsfp_inventory` | 4 UBBs, 32 ASICs, 14 ETH ports per ASIC. Reaches the chips over the collector's own PCI enumeration and BMC reads rather than tt-smi, so it deliberately overlaps `pcie_enum_count` — two paths agreeing is worth more than either alone, and the collector records a *reason* per absent slot. | **FAIL** on a short count or an absent slot. **WARN** on unparseable dump lines. |
| `qsfp_board_rev` | All UBBs report one `BOARD_REV`, and all 8 ASICs of each agree on `board_id`. The revision selects the internal topology table, so a bad read invalidates the partner checks below too, not just this one. | **FAIL** on mixed revisions or intra-UBB disagreement |
| `qsfp_board_rev_agrees` | The dump's revision against the snapshot phase's `detected_board_rev` — two reads of one register down independent paths | **FAIL** on disagreement. **SKIP** when either side didn't determine one. |
| `qsfp_findings` | The collector's own `FINDINGS` list (unreadable descriptor, unusable ipmitool, a cage sweep that fell over) | **WARN** — lost coverage. This is what stops `collect`'s exit code 0 reading as a clean run. |
| `qsfp_collection_failures` | No record is `READ_FAILED` / `UNREACHABLE` / `ABSENT` | **FAIL** — a part that did not answer |
| `qsfp_collection_partial` | No record is `PARTIAL` / `SKIPPED` | **WARN** — lost coverage, kept separate from the above so it doesn't read as a fault. JSON-only when clean. |
| `qsfp_link_training` | Every port that *should* have trained reports `TRAIN_STATUS == LINK_TRAIN_PASS`. Excluded: harvested tiles and `PCIE`/`UNCONNECTED`/`INVALID_LOCATION` types; ports that were not read (a failed read is not a failed link); and cage-attached ports whose cage holds no module or was not read (see below). | **FAIL**, naming the ports |
| `qsfp_link_asymmetry` | Both ends of a resolved link agree on `LINK_UP`. Reported once per link, not once per record. | **FAIL** — the asymmetry names the end at fault. JSON-only when clean. |
| `qsfp_missing_channel` | Every port with an expected partner saw one. A pair blind at both ends collapses to one row. | **FAIL**. **SKIP** when no port carries an expectation. |
| `qsfp_miscabled` | The partner the firmware found is the one expected | **FAIL** on a wrong end. **WARN** on a link nobody described, but only when a descriptor matched this host — otherwise recorded without alerting. |
| `qsfp_partner_disagreement` | A names B and B names A. Needs no expectation at all, so it holds without a descriptor. | **FAIL** — the hardware contradicting itself. JSON-only when clean. |
| `qsfp_outside_channel` | Trained links leading to hardware this dump didn't read | never alerts — one galaxy is collected, so every inter-galaxy cable lands here. JSON-only. |
| `qsfp_cage_gaps` | Cages match the expected cabling | **WARN**. **SKIP** without expected cabling, or when the sweep didn't run. |
| `qsfp_eth_counters` | Store-only: `RETRAIN_COUNT`, `CORR_CW`, `UNCORR_CW` totals plus the worst ports | never alerts — JSON-only, like `gddr_info_*` |
| `qsfp_modules` | Store-only: the transceiver inventory (vendor PN/SN, length, cage) | never alerts — JSON-only |

**An empty QSFP cage is not a dead link.** A partly populated Galaxy is a
supported configuration, and a port behind a cage with nothing plugged into it
can never train. Counting those made a healthy measured 6U — 24 of 56 cages
filled — report 256/320 ports trained, a deterministic 64-port FAIL that scales
with how little of the chassis is cabled and would have taken healthy nodes out
of service under `--qsfp-gating`. So `qsfp_link_training` judges a cage-attached
port only once its cage is known to hold a module, and the same machine now
reports `256/256 cabled and internal ports trained (64 behind an empty cage not
counted)`.

Three things keep that from becoming a blind spot. Cage attachment is decided by
`PORT_TYPE` (`CHIP_TO_QSFPDD`, `CHIP_TO_WARP400`), not by the port's cage
pointer, which is null on a dump collected with `--skip-qsfp` where the port is
just as cage-attached. The soldered links — `CHIP_TO_CHIP` and both EXAMAX
types, which train with nothing plugged in — always count, so the check keeps
its real subject. And a port behind a cage that was *not read* is set aside
separately from one behind a cage known to be empty, because "no cable here" and
"nobody looked" are different facts; both counts appear in `details` and `data`,
so what the check stopped looking at is never silent.

An all-zero `remote_info` is treated as **no answer, not an answer**: an
untrained port still carries a zero-filled one, and reading it as a partner
makes every untrained link look trained to the same imaginary far end. Measured
on `bh-glx6u-37`, 320 of 384 ports were in that state, and before the
collector's own query grew this guard it called all 208 internal links
miscabled.

## Known issues

- **PCIe Gen1 fallback**: host-PCIe chips occasionally train down to Gen1 instead
  of the rev's expected gen. Surfaced as WARN by `pcie_gen` with `(!)` next to
  the offending chip. Real bug — investigate upstream when seen.
- **Above-spec PCIe gen on RevA/B**: bh-glx-110-d04u02 has some U6 chips training
  at Gen5 despite being RevA/B silicon (spec is Gen4). Surfaced as `(^)` in the
  `pcie_gen` listing with no status impact; flagged for visibility because the
  root cause (host BIOS, KMD, or firmware override) hasn't been confirmed.
- **DMC firmware fields N/A**: `dm_app_fw` and `dm_bl_fw` report `0.0.0.0` on Galaxy
  (DMC not used on this platform). Treated as SKIP, not a real signal.
- **`mrisc_fw` / `gddr_fw` only in tt-smi 5.1+**: 4.1.2 snapshots SKIP this check.
- **GDDR thermal + error counters only in tt-smi 5.1+**: 4.1.2 snapshots SKIP these.
- **Eth port muxing**: `ENABLED_ETH` reflects per-chip muxed enable bits; the wiring
  table in `diag_runner.py` (`ETH_*_BY_ASIC`) covers physical intent. Effective
  per-chip masks are the intersection.

## Sample healthy console output (5.1.1 unit, light tier, skip-reset)

```
[diag] using input snapshot: tt-smi-snapshot-5_1.json
[diag] wrote report: ./diag_report.json (overall=WARN)
  snapshot       WARN  (0.8s)
    board:  board_rev                        PASS   Rev: RevA/B (32/32 chips, board_id prefix 00000471)
    pcie:   pcie_enum_count                  PASS   32/32 chips enumerated
            pcie_lane_width                  PASS   32/32 chips at expected lane width (x8 on host ASIC_LOCATION=0x6, x1 elsewhere)
            pcie_gen                         WARN   U6 chips (expected Gen4 on RevA/B): 0000:06:00.0=Gen5(^), 0000:46:00.0=Gen1(!), 0000:86:00.0=Gen5(^), 0000:c6:00.0=Gen5(^); below: 0000:46:00.0; above-spec: 0000:06:00.0, 0000:86:00.0, 0000:c6:00.0
    gddr:   dram_status                      PASS   32/32 dram_status=True
            enabled_gddr_full                PASS   32/32 chips have ENABLED_GDDR=0xff
            gddr_training_per_channel        PASS   256/256 channels trained
            gddr_bist_per_channel            PASS   256/256 channels BIST passed
            gddr_speed                       PASS   32/32 chips at 14G (RevA/B)
            gddr_info_max_gddr_temp          PASS   MAX_GDDR_TEMP: 32 chips reporting
            ...
    eth:    eth_links_up                     SKIP   ETH_LIVE_STATUS requires FW bundle >= 19.9 (detected: 19.7.1.0) — firmware does not populate this field; cannot validate links
            eth_speed                        SKIP   no eth speed field in tt-smi snapshot schema
    asic:   asic_location_per_ubb            PASS   all UBBs complete
            physical_vs_fw_location          PASS   32/32 chips match
    fw:     fw_bundle_version_consistent     PASS   all chips: 19.7.1.0
            cm_fw_consistent                 PASS   all chips: 0.29.1.0
            eth_fw_consistent                PASS   all chips: 1.9.0
            gddr_fw_consistent               PASS   all chips: 2.13
            dm_app_fw_consistent             SKIP   not applicable to Galaxy
            dm_bl_fw_consistent              SKIP   not applicable to Galaxy
  tests          PASS  (25.2s)
    other:  eth_link_up                      PASS   filter=*TensixDeploymentEthernet*LinkUp rc=0 dur=25.2s passed=1 failed=0 log=./logs/eth_link_up.log
  OVERALL        WARN
```

Exit code: `1` on FAIL, `0` on PASS/WARN.

## Build prereqs

The deployment-test binary is not in `build_metal.sh`'s default target list.
Build with either:

```bash
./build_metal.sh --build-tests
# or, after configure:
ninja -C build_Release unit_tests_deployment
```

## Repo layout

```
tools/scaleout/exabox/health_check_test_suite/
├── run_diag.sh         # bash dispatcher (sets TT_METAL_HOME / PYTHONPATH / LD_LIBRARY_PATH, execs runner)
├── diag_runner.py      # Python orchestrator (all check logic lives here). Also exposes run_diag() as a
│                       #   programmatic entry point returning (exit_code, report_dict).
├── qsfp_ingest.py      # QSFP tests: cluster-debug JSONL dump -> checks. No hardware in it: a stored
│                       #   dump is all it needs, so it is tested at a desk. Runnable standalone.
├── HEALTH_CHECK.md     # this file
└── test_infrastructure/  # scheduled/CI harness around the diag suite
    ├── run_health_check.py              # entrypoint: run diag as a subprocess, then JIRA + CSV + SFTP
    ├── analyze_health_check_results.py  # diag_report.json -> runs/checks CSVs (Superset)
    ├── requirements.txt                 # runtime deps (requests, paramiko, prometheus_client)
    └── utils/                           # supporting modules
        ├── diag_execution.py            # invoke diag_runner.py as a subprocess (timeout/kill aware)
        ├── system_info.py               # tt-smi / kmd / fw version discovery
        ├── telemetry.py                 # tt-telemetry (Prometheus) collection + formatting
        ├── report.py                    # post-reset normalization + actionable-failure verdict
        ├── jira_client.py               # JIRA ticket create / update / close / attach
        ├── sftp_upload.py               # CSV upload to the Data-team SFTP endpoint
        ├── secrets_loader.py            # JIRA / SFTP credential-file parsing
        └── health_check_models.py       # Pydantic schema for the CSV output
```

The `test_infrastructure/` harness previously lived in the `exabox-infra` repo and
spun up a nested Docker container to run the diag suite. Now that the harness ships
in the same image as the diag suite, `run_health_check.py` invokes `diag_runner.py`
directly as a subprocess (see `diag_execution.py`) instead of via docker-in-docker.

The triage tools are a sibling directory,
[`tools/scaleout/kmd_triage/`](../../kmd_triage/README.md), next to the other
compiled scaleout tools rather than under the health check, because the binary
is a CMake target and the tools are useful on their own when triaging a unit by
hand.
