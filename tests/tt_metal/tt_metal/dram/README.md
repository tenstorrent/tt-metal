# mimir_tests

`mimir_tests` covers Mimir bring-up: host access to DRAM and TL1, firmware and kernel load, and CCE copies to local and remote DRAM and TL1. The gtest prefixes `MM0` to `MM6` number these bring-up stages in order. It is built only when `TT_UMD_BUILD_GRENDEL_JTAG=ON`. The binary is `build/test/tt_metal/mimir_tests`. Each row of the run order is a separate invocation, and the order matters.

## Environment

Set:
- `TT_METAL_EMU_SERVER`
- `TT_METAL_EMU_SOC_DESC`, set to the descriptor for the run (see Descriptors)
- `TT_METAL_SLOW_DISPATCH_MODE=1`

Unset:
- `TT_METAL_WATCHER`
- `TT_METAL_CLEAR_DRAM`
- `TT_METAL_CLEAR_L1`
- `TT_METAL_KERNEL_READBACK_ENABLE`
- `TT_METAL_SKIP_LOADING_FW`
- `TT_METAL_FORCE_REINIT`
- `TT_METAL_SIMULATOR`

Rows 2c and 2d set the variables named in their rows. Every other row leaves them unset.

From row 2b onward, prefix each command with `TT_METAL_OPERATION_TIMEOUT_SECONDS=600 timeout 3600` until a passing 32 KiB local run gives calibrated values. Then set both from that run.

Example:

```
TT_METAL_OPERATION_TIMEOUT_SECONDS=600 timeout 3600 ./build/test/tt_metal/mimir_tests --gtest_filter='<filter>'
```

## Descriptors

Descriptors are in `tt_metal/third_party/umd/tests/soc_descs/`:
- `mimir_1x1.yaml`
- `mimir_2x_package.yaml`

Run the tests once per descriptor:
- `mimir_1x1` with `models/mimir` and `bringup_mimir_1x1.py`.
- `mimir_2x_package` with `models/mmk` and `bringup_two_mimir.py` or `bringup_mmk.py`.

## Run order

| # | Filter | Covers | Descriptors |
|---|---|---|---|
| 1 | `*MimirHost*` | Host DRAM and host TL1 access | all |
| 2a | `*MimirFwPreflight*` | Firmware load preflight | all |
| 2b | `MimirFwLoad.*:*MimirFwLaunch*` | Firmware load and init, image manifest, CCE isolation, no-op launch per CCE | all |
| 2c | `TT_METAL_SKIP_LOADING_FW=1` prefix, filter `*MimirFwLoadNegative*` | Firmware never reports done when loading is skipped. The test zeroes the firmware window in TL1 itself | all |
| 2d | `TT_METAL_FORCE_REINIT=1` prefix, filter `*MimirFwReinit*` | Firmware reset, reload and relaunch on every device open, three cycles | all |
| 3 | `*MimirKernelLoad*:*MimirLocalDram*:*MimirLocalTl1*` | Kernel load, local DRAM copies, same-chiplet TL1 | all |
| 4 | `*MimirRemoteTl1Direct*` | Remote TL1 write/read, host-initiated remote read, size and alignment; direct pairs | 2x only |
| 5 | `*MimirRemoteTl1Bidir*` | Concurrent remote TL1 in both directions, direct pairs | 2x only |
| 6 | `*MimirRemoteTl1Routed*/M0Cce0ToM1Cce0` | Remote TL1 as in row 4, one-hop pair | 2x only |
| 7 | `*MimirRemoteTl1Routed*/M0Cce1ToM1Cce1` | Remote TL1 as in row 4, one-hop pair | 2x only |
| 8 | `*MimirRemoteTl1Routed*/M1Cce0ToM0Cce0` | Remote TL1 as in row 4, one-hop pair | 2x only |
| 9 | `*MimirRemoteTl1Routed*/M1Cce1ToM0Cce1` | Remote TL1 as in row 4, one-hop pair | 2x only |
| 10 | `*MimirRemoteTl1Routed*/M0Cce0ToM1Cce1` | Remote TL1 as in row 4, two-hop pair | 2x only |
| 11 | `*MimirRemoteTl1Routed*/M1Cce1ToM0Cce0` | Remote TL1 as in row 4, two-hop pair | 2x only |
| 12 | `*MimirRemoteDram*/SrcM0Cce1*` | Remote DRAM write, read, round trip and traffic matrix; one-hop source | 2x only |
| 13 | `*MimirRemoteDram*/SrcM1Cce0*` | Remote DRAM write, read, round trip and traffic matrix; one-hop source | 2x only |
| 14 | `*MimirRemoteDram*/SrcM0Cce0*` | Remote DRAM write, read, round trip and traffic matrix; two-hop source | 2x only |
| 15 | `*MimirRemoteDram*/SrcM1Cce1*` | Remote DRAM write, read, round trip and traffic matrix; two-hop source | 2x only |
| 16 | `*MimirRemoteMixed*` | Remote TL1 one-to-many and many-to-one; concurrent local DRAM, remote DRAM and remote TL1 | 2x only |

Notes on the rows:
- Row 1 starts with the identity check. If it fails, every test in row 1 skips.
- Rows 4 to 16 each start with a preflight readback of SS_BAR, ATT and interleave.
- The CCE DMRISC remap is compared with UMD's values once per process, right after the first CreateDevice. CreateDevice is where UMD releases each CCE uncore and programs the remap before it loads firmware, so the comparison runs after it.
- Rows 4 to 16 are 2x only. Each `TEST_P` also skips at runtime from the descriptor's topology.
- Rows 6 to 15 run one route each, so a hang in one route does not lose the others.
- Row 16 runs last because it includes routed destinations.

## Fresh-session rule

Before a full run, start a fresh emulator session and run chippy bring-up.

Run the host suite (row 1) right after bring-up. The host suites skip when any CCE hart is out of reset. A process that opened a device resets the CCE harts on clean exit and leaves the uncore released, as bring-up does, so row 1 also runs after such a process. After a killed process the harts may still be running, and row 1 skips.

If a run is killed mid-stall, the session is probably wedged. Use the same rule: a fresh session plus bring-up.

## Rules

- Never set `TT_METAL_CLEAR_DRAM`. It writes every view in full, 1 GiB per Mimir, which is far too slow on the emulator.

## Manual D2D AXI monitor step

The emulator's D2D AXI monitor is a manual debug step. It is a DPI bind (`models/mimir/emu_hdl/axi_mon_d2d_bind.sv`, `top.utf:22`). To use it, relaunch the SiVal server without `--disable-dpi`, which the server command in the repo-root `mimir_setup.md` passes. With DPI on, the SMC AXI monitor also logs every host beat.
