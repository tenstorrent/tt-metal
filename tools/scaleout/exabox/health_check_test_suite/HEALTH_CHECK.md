# Health check Blackhole Galaxy diagnostic tool

Pre-cluster hardware sanity check for Blackhole Galaxy 6U systems. Captures a
`tt-smi` snapshot, decodes per-chip telemetry, runs a reset stability loop,
invokes the `unit_tests_deployment` gtest binary, and on the longer tiers folds
in the first-step triage tools, the QSFP tests and the sys-triage bundle. The
`flash` tier reads every on-host ETH link twice with tt-triage instead, with no
resets and no tests, then collects the sys-triage bundle. Emits a single JSON
report with per-check PASS/WARN/FAIL/SKIP status grouped by IP.

## Quick start

```bash
export TT_METAL_HOME=/path/to/tt-metal
cd $TT_METAL_HOME

# Quick ETH link health, the unit read as found (snapshot + two triage reads 30 s apart + sys-triage)
./tools/scaleout/exabox/health_check_test_suite/run_diag.sh flash

# Daily tier (2 resets + GDDR fast + eth link_up/bandwidth + triage + QSFP tests)
./tools/scaleout/exabox/health_check_test_suite/run_diag.sh daily

# Snapshot-only smoke (fastest iteration)
./tools/scaleout/exabox/health_check_test_suite/run_diag.sh daily --skip-reset --skip-tests --skip-triage --skip-qsfp-tests

# Offline / dev iteration against a stored snapshot
./tools/scaleout/exabox/health_check_test_suite/run_diag.sh daily --dry-run \
    --input-snapshot snap.json --output /tmp/diag_report.json
```

Output goes to `./diag_report.json` by default; gtest logs to `./logs/<test>.log`.

### Running in Docker

`run_diag_docker.sh` runs the same suite inside the `exabox-tools` image (the one
the scheduled fleet run uses), so a host needs no build, and keeps the report,
the per-step logs and the console transcript on the host:

```bash
# Daily tier into ./test_output_<date>-<time>/
./tools/scaleout/exabox/health_check_test_suite/run_diag_docker.sh

# Quick ETH link health
./tools/scaleout/exabox/health_check_test_suite/run_diag_docker.sh flash

# Weekly tier with the QSFP phase, results in a named directory
./tools/scaleout/exabox/health_check_test_suite/run_diag_docker.sh weekly \
    --output-dir /data/hc-run-1 --gh-token "$GH_TOKEN"

# Data collection before a BMC reboot, with the sys-triage bundle
BMC_IP=... BMC_USER=... BMC_PASSWORD=... \
    ./tools/scaleout/exabox/health_check_test_suite/run_diag_docker.sh pre_bmc_reboot
```

- The tier defaults to `daily`; `--image` defaults to `exabox-tools:latest`,
  which is re-pulled on every run.
- The container runs the suite baked into the image, not the host checkout's, so
  a tier exists in docker only once the image is built from a commit that has
  it. `flash` also needs the image's venv to carry `tools/triage/requirements.txt`,
  which `dockerfile/exabox_tools/Dockerfile` installs and smoke-tests.
- `--gh-token` (default `$GH_TOKEN`) installs `tt-syseng-diag` from its private
  release (`tenstorrent/tt-syseng-diag-packages`, `--diag-pkg-version`, default
  `0.0.4`). Without it the QSFP and sys-triage phases report SKIP; every other
  check still runs. `flash` needs it for sys-triage only, as it runs no QSFP
  tests.
- `--bmc-ip` / `--bmc-user` / `--bmc-password` (default `$BMC_*`) feed the
  sys-triage phase, `flash`, `full`, `pre_bmc_reboot` and `post_bmc_reboot` only.
- `--bmc-api-url` / `--bmc-api-token` (default `$TT_BMC_API_URL` /
  `$TT_BMC_API_TOKEN`) feed the QSFP collector on every tier; without them it
  reads the FRU with ipmitool and collects no QSFP cages.
- Tokens and credentials reach the container through a 0600 `--env-file`, never
  a command line.
- `--output` is not forwardable — the wrapper owns the bind mount. Use
  `--output-dir`. Most flags in [Flags](#flags) are forwarded directly; pass
  `--tt-metal-path` after a `--` separator, using an in-container repo path.
  Host paths (`--input-snapshot`, `--qsfp-descriptor`, `--triage-dir`) supplied
  before the separator are mounted read-only for you.
- The exit code is the suite's own: `1` on FAIL, `0` on PASS/WARN.

## Tiers

| Tier | Resets | Tests | Triage | ETH status | QSFP tests | sys-triage | Duration | Use when |
|---|---|---|---|---|---|---|---|---|
| `flash` | none | — | — | 2 reads, 30 s apart | — | yes, if installed and BMC credentials set | snapshot + 2 triage reads + 30 s + sys-triage | Quick ETH link health, the unit read as found |
| `daily` | `tt-smi -r`, `tt-smi -glx_reset`, then `-glx_reset` after the tests | eth_link_up + eth_bandwidth + gddr_fast (DRAM_TEST_FAST=1)                | host_side + device_side | — | yes, if installed | — | ~5 min + triage + ~7 min | The scheduled fleet run |
| `weekly` | `tt-smi -r`, `tt-smi -glx_reset` × 2, then `-glx_reset` after the tests | eth_link_up + eth_bandwidth + full gddr matrix (3 DramDeployment tests) + didt_matmul_galaxy (pytest, ~9 min) | host_side + device_side | — | yes, if installed | — | ~18 min + triage + ~7 min | Every test, less often |
| `full` | as `weekly` | as `weekly` | host_side + device_side | — | yes, if installed | yes, if installed and BMC credentials set | `weekly` + sys-triage | A unit with an issue: everything, for the logs on its DIIM ticket |
| `pre_bmc_reboot` | none | — | host_side + device_side | — | yes, if installed | yes, if installed and BMC credentials set | snapshot + triage + ~7 min + sys-triage | Collect data before a BMC reboot |
| `post_bmc_reboot` | none | — | host_side + device_side | — | yes, if installed | yes, if installed and BMC credentials set | snapshot + triage + ~7 min + sys-triage | Collect data after a BMC reboot; for now the same as `pre_bmc_reboot` |

The [sys-triage phase](#sys-triage-phase) runs last on `flash`, `full`,
`pre_bmc_reboot` and `post_bmc_reboot` only; it collects a bundle for the ticket
and never gates the run.

The old names still run, with a deprecation warning: `light` and
`medium` as `daily`, `deploy` as `weekly`, `pre_reboot` as `pre_bmc_reboot`. The
report records the new name.

`flash` is the quick look at the ETH links. It runs the snapshot, the
[ETH status phase](#eth-status-phase) and the sys-triage phase and nothing else:
no reset, so a fault is still there for whoever picks the ticket up, and no
tests, so nothing has retrained or reset a link before it is read. The other
phases are recorded as SKIP with the reason. Unlike the triage and QSFP phases,
the ETH status phase gates the run, since it is the whole of what `flash` checks.
`flash` is also the one tier that ends with a
[summary for the person who ran it](#the-summary).

What each tier runs is defined in one place, `TIERS` in
[`tiers.py`](tiers.py): per tier, the pre-test resets, the tests, the post-test
reset, the triage tools, and whether the ETH status read, the QSFP tests and
sys-triage run. To add a
step to a tier, or drop one, edit its entry there. The steps it can name are
defined beside the phase that runs them: gtests in `TESTS` and pytests in
`PYTESTS` (`diag/device_tests.py`), triage tools in `TRIAGE_TOOLS`
(`diag/triage.py`). An unknown name fails at import. The phase order is fixed in
`diag_runner.run_diag()` and is not a per-tier choice. A new tier also has to be
added to the tier lists in `run_diag.sh`, `run_diag_docker.sh` and
`test_infrastructure/run_health_check.py`, and to the table above;
`tools/scaleout/exabox/tests/test_diag_tiers.py` fails until they agree with
`TIERS`.

`pre_bmc_reboot` collects data; it does not check the unit. It runs on a unit
that is about to be power cycled through the BMC, so it records the state first
and runs nothing that would change it. It never resets the unit: the snapshot, the
triage tools and the QSFP tests all read the unit as it was found. There are no
gtests or pytests either: a stress run on a unit that is already going to be
rebooted would change the state the later phases are meant to record. The reset
loop and test phases are recorded as SKIP with the reason, not left out; there
is no `post_test_reset` phase.

`post_bmc_reboot` runs after the reboot. For now it runs the same steps as
`pre_bmc_reboot`; it has its own entry in `tiers.py` so the two can change
independently.

The eth deployment tests are registered as `TensixDeploymentEthernet<NN><Name>`
(e.g. `TensixDeploymentEthernet00LinkUp`, `TensixDeploymentEthernet01Bandwidth`,
`TensixDeploymentEthernet02BandwidthBidir`), so the filters wildcard the
two-digit index: `eth_link_up` is `*TensixDeploymentEthernet*LinkUp` and
`eth_bandwidth` is `*TensixDeploymentEthernet*Bandwidth*`. The trailing wildcard
on `eth_bandwidth` means it runs both `Bandwidth` and `BandwidthBidir`, both of
which `tests/tt_metal/tt_metal/deployment/sources.cmake` now compiles. Both run
with `TT_METAL_SLOW_DISPATCH_MODE=1`: fast dispatch to active ethernet cores was
removed, and under fast dispatch each eth test skips itself and exits 0.

A gtest check passes only if at least one of its test cases passed, as a pytest
check does. A filter that matches nothing (one without the index wildcard, say)
and a run whose tests all skip both exit 0 having tested nothing, and are
recorded as FAIL with the `passed` / `failed` / `skipped` counts.

`weekly` and `full` also run `didt_matmul_galaxy`, a pytest-based dI/dt stress
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

The reset cadence and test set per tier are in `tiers.py`; the gtest filters and
pytest invocations themselves are `TESTS` / `PYTESTS` in `diag/device_tests.py`.

## Triage phase

`daily`, `weekly` and `full` end with a post-test `tt-smi -glx_reset` followed by
the first-step triage tools — `host_side.sh` (host, PCIe and driver state, read from
sysfs) and `device_side.sh` (per-chip liveness, ARC scratch, telemetry and a NOC0
node sweep). `pre_bmc_reboot` and `post_bmc_reboot` run the same tools with no
reset before them. They live in `tools/scaleout/kmd_triage/`. Which tiers run
them, and the reset before them, is `post_test_resets` / `triage` in `tiers.py`;
the tools themselves are `TRIAGE_TOOLS` in `diag/triage.py`.

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
- Two are held the other way: **`hostside_pcie_aer` and `hostside_kernel_log` record
  their WARN as PASS** (`TRIAGE_ADVISORY_WARN` in `diag/triage.py`). Both count host
  state accumulated over a boot, so they WARN on units everything else calls healthy,
  and one phase WARN is a run WARN. FAIL continues to follow `--triage-gating`; the
  finding still reaches `details` and `data`, annotated `[advisory: WARN recorded as PASS]`.

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

## ETH status phase

`flash` reads every on-host ETH link twice with tt-triage's `check_eth_status`,
30 s apart, and judges the two reads against the BH Galaxy UBB port map. It is
`diag/eth_status.py`; whether a tier runs it is `eth_status` in `tiers.py`.

### What it runs

```bash
<python> tools/triage/triage.py --run=device_info --run=check_eth_status --dev=all --llm-output \
    --inspector-rpc-host=127.0.0.1 --inspector-rpc-port=0 --inspector-disable-rank \
    --inspector-log-path=<output_dir>/logs/eth_status_no_inspector
sleep 30
<python> tools/triage/triage.py --run=check_eth_status --dev=all --llm-output \
    --inspector-rpc-host=127.0.0.1 --inspector-rpc-port=0 --inspector-disable-rank \
    --inspector-log-path=<output_dir>/logs/eth_status_no_inspector
```

Triage runs from the tt-metal tree the suite runs from, the host checkout
(`run_diag.sh`) or the image's (`run_diag_docker.sh`); nothing is started in a
container of its own. `<python>` is `python_env/bin/python` where it exists
(`create_venv.sh` installs `tools/triage/requirements.txt` there), else the
python running the suite, which in the `exabox-tools` image is its venv. Before
the first read the phase checks that this python can import triage's
requirements, and SKIPs naming the missing module if it cannot.

- **Inspector is kept out.** Triage asks a running Inspector RPC server
  (`localhost:50051`, reachable from the container through `--network host`)
  first and its saved logs (`/tmp/tt-metal/inspector`) second, and either would
  have it check the eth cores a metal run found trained rather than the links as
  they are now. Triage has no switch to turn Inspector off, so the RPC is aimed
  at TCP port 0, which nothing can listen on, and the log path at one that does
  not exist; `--inspector-disable-rank` stops an MPI rank from moving either.
- **`TT_METAL_VISIBLE_DEVICES` is dropped** from triage's environment, so every
  chip is read whatever the caller left visible to metal.
- **`device_info`** on the first read gives each triage chip id its PCI address,
  and from the bus byte its UBB and ASIC slot. If its table is missing or does
  not cover every chip, the phase falls back to `/dev/tenstorrent` order, which
  is how UMD numbers the chips; `eth_status_read` says which it used.

### PCIe first

Before any chip is opened, the phase reads each Tenstorrent PCI function's
config space from sysfs. A chip whose vendor ID reads `0xFFFF`, or whose
Command register has memory decoding off, has dropped off PCIe:
`eth_status_pcie_gate` FAILs naming it, and the links are **not read**. Opening
that chip could hang the read, and a reset to recover it wipes the evidence, so
the unit is left as it was found. A chip whose config space cannot be read at
all is not proof of anything, but it cannot be cleared either, so the gate
WARNs naming it and the links are not read. A chip that is simply missing is
`pcie_enum_count`'s finding; here it only means which links are there cannot be
judged, since the links to it cannot come up.

### What is judged and what is described

The link checks read `Port Status`, the live `Link` state and `Heartbeat` of
every eth core triage lists. The counter checks compare the two reads: the FEC,
CRC and retrain counters only go up until the chips are reset, so what they did
in 30 s is the reading, not their totals. A counter that went down means that
chip was reset between the reads, and its link is not rated.

The topology checks use `UBB_ETH_PORTS` in `diag/galaxy.py`, the UBB_BLACKHOLE
port map from tt-metal's `tools/scaleout/board/board.cpp`. Its **built-in**
links (20 board traces and 6 linking-board ports per UBB, 208 eth cores on a
Galaxy) exist whatever the cabling, so one that is not linked is a FAIL. A QSFP
port carries one cable, so a port with only one of its two lanes linked is a
FAIL too.

**The cabling itself is described, never judged.** It differs from host to host
and over time, and a QSFP port linked on no UBB has no cable, a cable to another
host or dead cables, which this phase cannot tell apart. `eth_status_cabling`
names what it finds where it has a name for it (tt-metal's mesh, Y, X and XY
torus, XY torus plus the Z ports, a Galaxy inside a sub-torus cluster) and
always PASSes. The one cabling finding is `eth_status_cable_trays_alike`: a port
linked on some UBBs and not on the others is a WARN, since UBBs are normally
cabled alike.

### Statuses

- **SKIP** — no reading: triage is not in the tree, its requirements do not
  import, no chip is on PCIe, one dropped off it, or one's PCIe state could not
  be read.
- **WARN on `eth_status_read`** — triage ran but listed no eth core (its last
  output lines are in `details`; a chip that stopped answering shows up as a
  register timeout there), or only the first read did, and the counters were
  judged on one read. Lost coverage, not a statement about the links.
- Every other check is a finding about the links, at the thresholds below
  (module constants in `diag/eth_status.py`):

| Reading | WARN | FAIL |
|---|---|---|
| Uncorrected FEC codewords, per link | ≥ 10/s | ≥ 100/s: the link corrupts data |
| Same, when only one read came back | ≥ 1,000,000 since the last reset | — |
| CRC errors (bad frames), per link | ≥ 1/s | — |
| Retrains | ≥ 3 since the last reset | any during the check: the link is flapping |
| ERR_STAT | any latched bit | — |

### Not checked here

- Whether the cabling is the one the host is meant to have; only the people who
  cabled it know.
- Links to other hosts: `check_eth_status` lists on-host links only.
- Where each cable goes: the far end of a link is not read, so cables in the
  wrong ports can look healthy. The [QSFP tests](#qsfp-tests-phase) compare
  partners.
- Traffic: a fabric router can wedge on a link that still reads healthy. The
  `eth_link_up` and `eth_bandwidth` gtests push traffic.

The raw reads land in `<output_dir>/logs/eth_status_read_{1,2}.csv` (triage's
own CSV tables) beside `eth_status_read_{1,2}.log` (command, exit code and
stderr), so `collect_run_artifacts()` attaches them to the JIRA ticket with no
extra wiring.

### The summary

A tier with `eth_summary` (only `flash`) ends, after the `OVERALL` block, with
one screen for whoever ran it by hand, and keeps the same text in
`<output_dir>/logs/eth_status_summary.txt`. It is `diag/eth_summary.py`,
rendered from the report alone, so it says nothing the JSON does not:

- **VERDICT** — the run's overall status in words (`HEALTHY`, `WARNING`,
  `PROBLEM`, or `NOT CHECKED` when the links were not read), then every finding
  one per line, problems before warnings, eight at most with a count of the
  rest. ETH checks list each link at fault from their `data["findings"]`; any
  other failing check, the snapshot's among them, gets one line of its own.
- **LINKS** — the link counts and the cabling found.
- **CHECK table** — one row per reading with its result, what is expected and
  `OK` / `** WARNING **` / `** PROBLEM **` / `NOT CHECKED`, plus a row for the
  firmware across chips and one for the rest of the tt-smi snapshot.
- **CABLING FOUND** — each QSFP port group (Y torus, X torus, Z ports) and on
  which UBBs it is linked; described, never judged.
- **WORST LINKS** — the five links with the most uncorrected FEC errors.
- **NOT CHECKED HERE**, then where the report, the raw reads and the summary are.

It is skipped on `--dry-run`, which reads nothing. Rendering runs after the
report is written, so a fault in it costs the summary and nothing else.

## QSFP tests phase

Every tier finishes by collecting an ETH dump with
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
install the package and the next run collects a dump.

The cage sweep reads the QSFP modules through the BMC API, so the collector also
needs `TT_BMC_API_URL` and `TT_BMC_API_TOKEN` in its environment
(`run_diag_docker.sh` forwards them from `--bmc-api-url` / `--bmc-api-token`).
Without them it reads the FRU with ipmitool and collects no cages: the ETH port
records are still there, but the cage and module records — and with them
`qsfp_cage_gaps` and `qsfp_modules` — are not.

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

`diag/qsfp_ingest.py` is that reader. Each fault class it derives is one of
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
python3 diag/qsfp_ingest.py qsfp_dump_bh-glx-110-a07u02.jsonl --json out.json
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
since while the holds are on the phase has undertaken not to raise one.

**The ingest module is not subject to this policy.** Run standalone against a
stored dump it reports WARN and FAIL normally, so reviewing a dump by hand shows
the real assessment. The policy is applied on ingest by `normalize_qsfp_check()`,
which is the only place that knows it is feeding a fleet verdict.

### Turning the findings on

`--qsfp-gating` changes this completely and in one step: findings report at their
real severity and the phase gates the run like any other — `overall_status`, the
exit code, `has_actionable_failure()`, the JIRA ticket and the Slurm
reboot-and-requeue all respond to a QSFP FAIL.

Both holds lift together, FAIL and WARN. Restoring only FAIL would leave the five
WARN-capable checks mute — `qsfp_findings` among them, whose whole job is to stop
`collect`'s exit code 0 reading as a clean run.

`Phase.gates` in the JSON records which mode a run was in, and `report.py` plus
the CSV analyzer both read that one flag rather than each keeping a list of which
phases count. A phase with no `gates` key gates, so every other phase and every
report written before the flag existed is unaffected.

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
any tier on a host that has the package.

## sys-triage phase

The last phase runs `sys-triage`, the syseng tool that collects a Blackhole
Galaxy triage bundle: a directory of text files covering the Quanta FW stack
(BMC, CPLD, Redfish device versions, KMD, ASIC FW bundle, UBB PHY), the full FRU,
MB/PDB and per-tray CPLD DIAG, `tt-smi -s`, `ipmitool sdr elist`, `lspci -vvv`,
`dmesg` and `journalctl -k`, each chip's reset-unit scratch registers and its
`tt-fw-terminal` vuart. It is forensics for whoever picks up the ticket, not
findings for this report to judge, so the phase **never gates** the run.

It runs on `flash`, `full`, `pre_bmc_reboot` and `post_bmc_reboot` only, as
their last step. On `pre_bmc_reboot` the bundle is the record of the unit's state
before the BMC power cycles it, and the vuart and scratch reads are worth most on
a unit nothing has reset since it misbehaved, as on `flash`; on `post_bmc_reboot`
it records the state the reboot left; on `full` it rounds out the logs for a DIIM
ticket. It follows the ETH status read and the QSFP tests because it opens every
chip, so nothing may run beside it. The other tiers record it as **SKIP** with
the reason.

**Where it comes from.** `sys-triage` ships in the same syseng `.deb` as
`tt-bh-glx-cluster-debug` and is probed for on PATH the same way
(`--sys-triage-path` overrides). No package, or a package version that does not
ship it yet, means **SKIP** with the reason; nothing else in the run changes.

**BMC credentials are required.** `sys-triage` refuses to start without the BMC
address, user and password, so the phase reports **SKIP** naming whichever is
missing. Each comes from `--bmc-ip` / `--bmc-user` / `--bmc-password`, else from
`$BMC_IP` / `$BMC_USER` / `$BMC_PASSWORD` — the variables `sys-triage` itself
reads, so a launcher that already exports them needs no flags. They reach the
tool through its environment only: never its argv, the logged command or the
report, and the password is scrubbed from the captured output. Prefer the
variable over `--bmc-password`, which is visible in the process list.

| Status | Meaning |
|---|---|
| **PASS** | `triage.txt` written, nothing reported missing |
| **WARN** | The bundle was written, but `sys-triage` printed `warning:` lines for sections it could not collect. They are listed in `details` and kept in `data.warnings`. |
| **SKIP** | No bundle: the tier doesn't ask, the package or the credentials are missing, the run timed out (900 s backstop), or it exited without writing `triage.txt` |

The bundle lands in `<output_dir>/logs/sys_triage/` beside `sys_triage.log`, so
`collect_run_artifacts()` attaches it to the JIRA ticket with no extra wiring.

**In a container** (`run_diag_docker.sh`), on the sys-triage tiers only, a few host tools
and paths the image lacks are provided, each best-effort and after the packages
every tier needs, since a missing one costs a section of the bundle and nothing
else. Other tiers get the same container as before the phase existed. An image
whose baked-in suite predates the phase has the `--skip-sys-triage` and
`--sys-triage-path` flags dropped rather than rejected, and simply runs without it.

- `kmod` (`modinfo`, `lsmod`) for the KMD row. `sys-triage` builds the whole
  FW-stack table in one pass, so without `modinfo` it loses the entire table,
  not just that row (`warning: fw-version failed: ... 'modinfo'`).
- `curl` for the Redfish device versions.
- `systemd` for `journalctl`, with `/var/log/journal`, `/run/log/journal` and
  `/etc/machine-id` mounted read-only so it reads the host journal. Installed
  separately and best-effort; `dmesg` still covers the recent kernel log.
- The KMD **package** version still reads `none` in a container:
  `dpkg-query` sees the image's package database, not the host's. The
  `module=` version from `modinfo` is correct.

## Flags

| Flag | Default | Purpose |
|---|---|---|
| `--tier {flash,daily,weekly,full,pre_bmc_reboot,post_bmc_reboot}` | required | Selects reset cadence + gtest matrix |
| `--dry-run` | off | Print intended subprocess calls; skip destructive steps |
| `--skip-reset` | off | Skip the reset loop phase entirely |
| `--skip-tests` | off | Skip the gtest phase entirely |
| `--skip-triage` | off | Skip the post-test reset and the triage phase entirely. `--skip-reset` also suppresses the post-test reset. |
| `--triage-dir PATH` | `$HC_TRIAGE_DIR`, else `<repo>/tools/scaleout/kmd_triage` | Directory holding the triage scripts. Override only to run a working copy against a deployed checkout. |
| `--triage-gating` | off | Let triage FAILs gate the run. Off holds them at WARN (noted in `details`); findings are recorded either way. |
| `--skip-eth-status` | off | Skip the ETH status read entirely |
| `--skip-qsfp-tests` | off | Skip the QSFP tests entirely. Named for the phase, not the collector's own `--skip-qsfp`, which drops the cage sweep but still collects ETH. |
| `--qsfp-tool-path PATH` | `tt-bh-glx-cluster-debug` on PATH | Override the collector binary. A path that doesn't resolve is reported as its own SKIP rather than silently ignored. |
| `--qsfp-descriptor PATH` | — | `factory_system_descriptor.textproto`, which gives the cage-attached links an expected partner. Without it only the soldered internal links are compared against a topology. |
| `--qsfp-gating` | off | Report QSFP findings at their real severity and let them gate the run. Off records every finding in `details`/`data` but holds the status at PASS. |
| `--skip-sys-triage` | off | Skip the sys-triage bundle entirely |
| `--sys-triage-path PATH` | `sys-triage` on PATH | Override the sys-triage binary. A path that doesn't resolve is reported as its own SKIP. |
| `--bmc-ip IP` | `$BMC_IP` | BMC address for sys-triage |
| `--bmc-user USER` | `$BMC_USER` | BMC user for sys-triage |
| `--bmc-password PASS` | `$BMC_PASSWORD` | BMC password for sys-triage. Prefer the variable: a flag value is visible in the process list. |
| `--input-snapshot PATH` | — | Use a stored snapshot instead of calling tt-smi |
| `--tt-smi-path PATH` | `/opt/tt_metal_infra/.../tt-smi` else `tt-smi` on PATH | Override tt-smi binary or repo path |
| `--tt-metal-path PATH` | `$TT_METAL_HOME` | tt-metal repo root. The deployment-test binary is looked up under `build_Release/`, `build/`, `build_Debug/` and `build_RelWithDebInfo/`, in that order (`DEPLOYMENT_BIN_CANDIDATES`). |
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

### ETH status (flash)

From the two `check_eth_status` reads; see [the phase section](#eth-status-phase)
for how they are taken and the thresholds. Links are named
`UBB<tray>/U<asic> ch<channel> (<port>, dev <triage chip id>)`, e.g.
`UBB2/U6 ch4 (TRACE P9, dev 13)`.

| Check | Rule | On fail |
|---|---|---|
| `eth_status_pcie_gate` | Every Tenstorrent PCI function answers in config space with memory decoding on. Read from sysfs before any chip is opened. | **FAIL**, naming the chips, and the links are not read. **WARN** when a chip's config space cannot be read, and the links are not read either. **SKIP** with no chip on PCIe. |
| `eth_status_read` | Both triage reads listed eth cores | **WARN** when triage listed none (with its last output lines) or only the first read did. **SKIP** when it could not start: triage absent, its requirements missing, or the PCIe gate stopped it. |
| `eth_status_links_down` | Every listed link not on an unused port is up and readable, and none went away between the reads | **FAIL**, naming the links |
| `eth_status_heartbeat` | No eth core's heartbeat stood still between triage's two samples (a core released to metal is fine) | **FAIL** — the core has stopped |
| `eth_status_port_state` | No port reports `Unknown`, `Invalid` or `Unsupported FW` | **WARN** |
| `eth_status_fec_uncorrected` | Uncorrected FEC codewords per link between the reads | **FAIL** ≥ 100/s, **WARN** ≥ 10/s; on one read, **WARN** ≥ 1,000,000 since the last reset |
| `eth_status_crc_errors` | CRC errors per link between the reads | **WARN** ≥ 1/s. **SKIP** on one read. |
| `eth_status_retrains` | No link retrained during the check | **FAIL** when one did; **WARN** at ≥ 3 since the last reset |
| `eth_status_err_stat` | No latched ETH_CTRL error | **WARN** |
| `eth_status_builtin_links` | All 208 built-in eth cores (board traces, linking boards) linked | **FAIL**, naming port and lane, and whether it is a trace or a linking board |
| `eth_status_cable_lanes` | No QSFP port with only part of its lanes linked | **FAIL** — the cable is there and part of it is dead |
| `eth_status_cable_trays_alike` | No QSFP port linked on some UBBs and not on the others | **WARN** |
| `eth_status_cabling` | Describes the QSFP ports linked on all 4 UBBs and the link counts | never alerts. The four topology checks **SKIP** when a chip is missing or the chips cannot be placed on 4 UBBs × 8 ASICs. |

### QSFP tests (every tier but flash, when the package is installed)

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
  table in `diag/galaxy.py` (`ETH_*_BY_ASIC`) covers physical intent. Effective
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
    other:  eth_link_up                      PASS   filter=*TensixDeploymentEthernet*LinkUp rc=0 dur=25.2s passed=1 failed=0 skipped=0 log=./logs/eth_link_up.log
  OVERALL        WARN
```

Exit code: `1` on FAIL, `0` on PASS/WARN.

## Build prereqs

The `exabox-tools` image already carries the deployment-test binary (under
`build/`), so `run_diag_docker.sh` needs no build. On a host checkout, note it is
not in `build_metal.sh`'s default target list. Build with either:

```bash
./build_metal.sh --build-tests
# or, after configure (build_Release/ or build/, whichever your configure used):
ninja -C build_Release unit_tests_deployment
```

The `flash` tier needs no build, only triage's Python requirements, which
`./create_venv.sh` installs into `python_env/` (or, into another environment,
`uv pip install -r tools/triage/requirements.txt`).

## Repo layout

```
tools/scaleout/exabox/health_check_test_suite/
├── run_diag.sh         # bash dispatcher (sets TT_METAL_HOME / PYTHONPATH / LD_LIBRARY_PATH, execs runner)
├── run_diag_docker.sh  # runs run_diag.sh inside the exabox-tools image; results land on the host
├── diag_runner.py      # CLI and the phase order. Also exposes run_diag() as a
│                       #   programmatic entry point returning (exit_code, report_dict).
├── tiers.py            # what each tier runs: resets, tests, triage, ETH status (+ summary), QSFP, sys-triage
├── diag/               # the phases and what they share
│   ├── model.py            # Check / Phase result model
│   ├── common.py           # subprocess, console and path helpers
│   ├── galaxy.py           # BH Galaxy topology: BDF <-> UBB/ASIC, ETH wiring and UBB port map, board revs
│   ├── snapshot.py         # phase 1: tt-smi snapshot capture, BMC FRU inventory
│   ├── snapshot_checks.py  # phase 1: the checks over a snapshot
│   ├── resets.py           # phase 2: pre-test reset loop; phase 4: post-test reset
│   ├── device_tests.py     # phase 3: deployment gtests (TESTS) and pytests (PYTESTS)
│   ├── external.py         # ingesting checks written by external tools
│   ├── triage.py           # phase 5: kmd_triage tools (TRIAGE_TOOLS)
│   ├── eth_status.py       # phase 6: ETH status read: PCIe gate, two tt-triage reads, and the
│   │                       #   judgement of them (pure functions, tested at a desk)
│   ├── eth_summary.py      # the flash tier's end-of-run ETH link summary, rendered from the report
│   ├── qsfp.py             # phase 7: QSFP tests, collection
│   ├── qsfp_ingest.py      # phase 7: cluster-debug JSONL dump -> checks. No hardware in it: a stored
│   │                       #   dump is all it needs, so it is tested at a desk. Runnable standalone.
│   └── sys_triage.py       # phase 8: sys-triage bundle
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
        ├── grafana.py                   # per-node tt-telemetry dashboard link for tickets
        ├── node_recovery.py             # Slurm reboot-and-requeue self-heal
        ├── sftp_upload.py               # CSV upload to the Data-team SFTP endpoint
        ├── secrets_loader.py            # JIRA / SFTP credential-file parsing
        └── health_check_models.py       # Pydantic schema for the CSV output
```

The unit tests for the parsers and the harness live outside this directory, in
`tools/scaleout/exabox/tests/` (e.g. `test_diag_runner_parsers.py`, `test_diag_tiers.py`,
`test_diag_eth_status.py`).

The `test_infrastructure/` harness previously lived in the `exabox-infra` repo and
spun up a nested Docker container to run the diag suite. Now that the harness ships
in the same image as the diag suite, `run_health_check.py` invokes `diag_runner.py`
directly as a subprocess (see `diag_execution.py`) instead of via docker-in-docker.

The triage tools are a sibling directory,
[`tools/scaleout/kmd_triage/`](../../kmd_triage/README.md), next to the other
compiled scaleout tools rather than under the health check, because the binary
is a CMake target and the tools are useful on their own when triaging a unit by
hand.
