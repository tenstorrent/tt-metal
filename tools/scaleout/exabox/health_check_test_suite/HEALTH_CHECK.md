# Health check Blackhole Galaxy diagnostic tool

Pre-cluster hardware sanity check for Blackhole Galaxy 6U systems. Captures a
`tt-smi` snapshot, decodes per-chip telemetry, runs a reset stability loop,
and invokes the `unit_tests_deployment` gtest binary. Emits a single JSON
report with per-check PASS/WARN/FAIL/SKIP status grouped by IP.

## Quick start

```bash
export TT_METAL_HOME=/path/to/tt-metal
cd $TT_METAL_HOME

# Full light tier (snapshot + tt-smi -r + eth_link_up gtest, ~75s)
./tools/scaleout/exabox/healt_check_test_suite/run_diag.sh light

# Snapshot-only smoke (fastest iteration)
./tools/scaleout/exabox/healt_check_test_suite/run_diag.sh light --skip-reset --skip-tests

# Offline / dev iteration against a stored snapshot
./tools/scaleout/exabox/healt_check_test_suite/run_diag.sh light --dry-run \
    --input-snapshot snap.json --output /tmp/diag_report.json
```

Output goes to `./diag_report.json` by default; gtest logs to `./logs/<test>.log`.

## Tiers

| Tier | Resets | Gtests | Duration | Use when |
|---|---|---|---|---|
| `light`  | `tt-smi -r` × 1                            | eth_link_up                                                                | ~75 s   | Smoke check on every new unit |
| `medium` | `tt-smi -r`, `tt-smi -glx_reset`          | eth_link_up + eth_bandwidth + gddr_fast (DRAM_TEST_FAST=1)                | ~5 min  | Pre-deployment validation |
| `deploy` | `tt-smi -r`, `tt-smi -glx_reset` × 2      | eth_link_up + eth_bandwidth + full gddr matrix (3 DramDeployment tests)   | ~15 min | Final deploy gate |

The `eth_bandwidth` filter is `*TensixDeploymentEthernetBandwidth*` so it
auto-picks up `BandwidthBidir` once that test gets added to
`tests/tt_metal/tt_metal/deployment/sources.cmake`. Currently outlogix's
sources.cmake only compiles `test_eth_bandwidth.cpp` (not the Bidir or
DataIntegrityDram source files), so the deploy tier only covers what the
build actually registers.

The reset cadence and test set are defined in `RESET_PLAN` / `TIER_TESTS` in
`diag_runner.py`.

## Flags

| Flag | Default | Purpose |
|---|---|---|
| `--tier {light,medium,deploy}` | required | Selects reset cadence + gtest matrix |
| `--dry-run` | off | Print intended subprocess calls; skip destructive steps |
| `--skip-reset` | off | Skip the reset loop phase entirely |
| `--skip-tests` | off | Skip the gtest phase entirely |
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
    other:  eth_link_up                      PASS   filter=*TensixDeploymentEthernetLinkUp rc=0 dur=25.2s passed=1 failed=0 log=./logs/eth_link_up.log
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
tools/scaleout/exabox/healt_check_test_suite/
├── run_diag.sh        # bash dispatcher (sets TT_METAL_HOME / PYTHONPATH / LD_LIBRARY_PATH, execs runner)
├── diag_runner.py     # Python orchestrator (this is where all check logic lives)
└── HEALTH_CHECK.md     # this file
```
