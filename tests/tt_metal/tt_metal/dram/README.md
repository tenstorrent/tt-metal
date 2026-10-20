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

## MM-7 Tensor Prefetcher

Rows 17 onward cover the Tensor Prefetcher on CCEs. They use the same environment and timeout prefix as rows 2b onward. Run them after a fresh session and bring-up (see Fresh-session rule), one test per invocation, in this order.

| # | Filter | Covers | Descriptors |
|---|---|---|---|
| 17 | `MimirPrefetcher.MM7GatePreconditionsHold` | Prefetcher preconditions hold and the port reports supported; one sender per DRAM view | all |
| 18 | `MimirPrefetcher.MM7GateUninitializedSenderFails` | A sender CCE whose GO_MSG reads INIT fails the preconditions, naming that CCE | all |
| 19 | `MimirPrefetcher.MM7GateStartStopWorks` | Start and Stop with no requests | all |
| 20 | `MimirPrefetcher.MM7GcbSmokePreconditionsWithReceiverRunning` | Preconditions stay ok while a receiver Program runs on a sender CCE | all |
| 21 | `MimirPrefetcher.MM7GcbLayoutLocalReceiver` | DRAM-receiver GCB state block, config page and credit lines; same-Mimir receiver | all |
| 22 | `MimirPrefetcher.MM7GcbLayoutRemoteReceiver` | The same with a receiver on the other Mimir, receiver order kept | 2x only |
| 23 | `MimirPrefetcher.MM7GcbLayoutTwoSenders` | Two senders, one receiver each, disjoint arena ranges | 2x only |
| 24 | `MimirPrefetcher.MM7GcbLayoutRejectsConflictingReceivers` | Rejects aliased, sender-as-receiver and duplicate receivers | all |
| 25 | `MimirPrefetcher.MM7GcbSmokeDiscardDrain` | Test sender into a discarding receiver, without the manager; both ends' counters agree | all |
| 26 | `MimirPrefetcher.MM7GcbSmokeValidatorLocal` | Test sender into a validating receiver on the same Mimir | all |
| 27 | `MimirPrefetcher.MM7GcbSmokeValidatorWrongSeed` | The validator reports a mismatch for a wrong seed | all |
| 28 | `MimirPrefetcher.MM7GcbSmokeResize` | Three page sizes in one GCB, with resize and wrap padding | all |
| 29 | `MimirPrefetcher.MM7GcbSmokeValidatorD2d` | Validating receiver on the other Mimir | 2x only |
| 30 | `MimirPrefetcher.MM7GcbSmokeDrainUntilStopped` | Receiver drains until the host stops it | all |
| 31 | `MimirPrefetcher.MM7GcbSmokeStallUntilReleased` | The receiver holds a full FIFO, so the sender gives CREDIT_TIMEOUT; then the host releases it | all |
| 32 | `MimirPrefetcher.MM7GcbSmokeDrainTimeout` | The receiver holds every page, so the sender's final barrier gives DRAIN_TIMEOUT | all |
| 33 | `MimirPrefetcher.MM7GcbSmokeReceiverWaitTimeout` | No sender: the receiver drains host-credited pages, then its wait gives CREDIT_TIMEOUT | all |
| 34 | `MimirPrefetcher.MM7GcbSmokeRelaunchResumes` | A second sender and validator pair resumes from the pointers the first left in TL1 | all |
| 35 | `MimirPrefetcher.MM7LifecycleStartStop` | Prefetcher Start and Stop; the sender reports Stopped | all |
| 36 | `MimirPrefetcher.MM7LifecycleStartStopTwice` | A second Start zeroes the control block the first session left | all |
| 37 | `MimirPrefetcher.MM7LifecycleBusySenderFailsStart` | Start fails on a sender CCE that runs a receiver Program, before it launches or zeroes anything | all |
| 38 | `MimirPrefetcher.MM7LifecycleQueueRejectsReceiverOnSender` | Queue rejects a GCB whose receiver is a sender CCE | all |
| 39 | `MimirPrefetcher.MM7LifecycleWaitCq` | A WAIT_CQ for a value written through the command queue | all |
| 40 | `MimirPrefetcher.MM7LifecycleDrainAtStop` | Stop sends DRAIN to a live receiver, then STOP | all |
| 41 | `MimirPrefetcher.MM7T1Local0To1` | One prefetch, CCE 0 to CCE 1, then Start, Queue and Stop again | all |
| 42 | `MimirPrefetcher.MM7T1Local3To2` | One prefetch, CCE 3 to CCE 2, on the second Mimir | 2x only |
| 43 | `MimirPrefetcher.MM7T1D2d0To2` | One prefetch, CCE 0 to CCE 2, one D2D hop | 2x only |
| 44 | `MimirPrefetcher.MM7T1D2d3To1` | One prefetch, CCE 3 to CCE 1, one D2D hop | 2x only |
| 45 | `MimirPrefetcher.MM7T2FanOut0To123` | Fan-out from CCE 0 to CCEs 1, 2 and 3 | 2x only |
| 46 | `MimirPrefetcher.MM7T2FanOut3To012` | Fan-out from CCE 3 to CCEs 0, 1 and 2 | 2x only |
| 47 | `MimirPrefetcher.MM7T2TwoSenders01To2` | Two senders, CCEs 0 and 1, into two GCBs on CCE 2 | 2x only |
| 48 | `MimirPrefetcher.MM7T2TwoSenders02To3` | Two senders, CCEs 0 and 2, into two GCBs on CCE 3 | 2x only |
| 49 | `MimirPrefetcher.MM7T3SubChunkLocal0To1` | A block larger than a stage third, in sub-chunks through two stage slots | all |
| 50 | `MimirPrefetcher.MM7T3SubChunkD2d0To2` | The same across D2D | 2x only |
| 51 | `MimirPrefetcher.MM7T3BelowStageThirdLocal0To1` | Blocks smaller than a stage third, two per stage slot | all |
| 52 | `MimirPrefetcher.MM7T3BelowStageThirdD2d0To2` | The same across D2D | 2x only |
| 53 | `MimirPrefetcher.MM7T3PartialStageChunkLocal0To1` | A round that ends in a partial stage chunk | all |
| 54 | `MimirPrefetcher.MM7T3AboveStageThirdLocal0To1` | A block above the default stage third, in one-tile sub-chunks | all |
| 55 | `MimirPrefetcher.MM7T3AboveStageThirdD2d0To2` | The same across D2D | 2x only |
| 56 | `MimirPrefetcher.MM7T3PageSplitLocal0To1` | One Queue call split into two request pages; local only, since the split is on the host | all |
| 57 | `MimirPrefetcher.MM7T3AbaLocal0To1` | Tensors A, B, A with different page sizes; resize and wrap | all |
| 58 | `MimirPrefetcher.MM7T3AbaD2d0To2` | The same across D2D | 2x only |
| 59 | `MimirPrefetcher.MM7T3WrapAndResizeLocal0To1` | Resize padding and wraps in a FIFO the page size does not divide | all |
| 60 | `MimirPrefetcher.MM7T3WrapAndResizeD2d0To2` | The same across D2D | 2x only |
| 61 | `MimirPrefetcher.MM7T3GcbSwitchLocal0To1` | GCB 1, then 2, then 1, resuming mid-FIFO | all |
| 62 | `MimirPrefetcher.MM7T3GcbSwitchD2d0To2` | The same across D2D | 2x only |
| 63 | `MimirPrefetcher.MM7T3FenceLocal0To1` | A WAIT_CQ fence holds delivery until the slot is written | all |
| 64 | `MimirPrefetcher.MM7T5TraceCaptureReplay` | Skipped: trace capture needs fast dispatch | all |
| 65 | `MimirPrefetcher.MM7RejectKRowMajorTensor` | Queue rejects a K-row-major tensor | all |
| 66 | `MimirPrefetcher.MM7RejectStreamingTensor` | Queue rejects a streaming tensor | all |
| 67 | `MimirPrefetcher.MM7RejectRangePastTensor` | Queue rejects a request past the tensor | all |
| 68 | `MimirPrefetcher.MM7RejectRangePastView` | Queue rejects a request past the DRAM view | all |
| 69 | `MimirPrefetcher.MM7T4MismatchD2d2To0` | Wrong-seed validator reports MISMATCH; the sender stops OK | 2x only |
| 70 | `MimirPrefetcher.MM7T4CreditTimeoutD2d1To3` | A request larger than the FIFO gives CREDIT_TIMEOUT; the host drops the requests after it | 2x only |
| 71 | `MimirPrefetcher.MM7T4DrainTimeoutD2d1To3` | A receiver holds a request, so Stop's DRAIN gives DRAIN_TIMEOUT | 2x only |

Notes on the rows:
- Rows that need a second Mimir skip at runtime on `mimir_1x1`.
- Row 64 always skips: trace capture needs fast dispatch.
- The T4 rows run last: each ends with a sender or receiver timeout.
- For a JUnit report, add `--gtest_output=xml:<file>` to each command.

## Fresh-session rule

Before a full run, start a fresh emulator session and run chippy bring-up.

Run the host suite (row 1) right after bring-up. The host suites skip when any CCE hart is out of reset. A process that opened a device resets the CCE harts on clean exit and leaves the uncore released, as bring-up does, so row 1 also runs after such a process. After a killed process the harts may still be running, and row 1 skips.

If a run is killed mid-stall, the session is probably wedged. Use the same rule: a fresh session plus bring-up.

## Rules

- Never set `TT_METAL_CLEAR_DRAM`. It writes every view in full, 1 GiB per Mimir, which is far too slow on the emulator.

## Manual D2D AXI monitor step

The emulator's D2D AXI monitor is a manual debug step. It is a DPI bind (`models/mimir/emu_hdl/axi_mon_d2d_bind.sv`, `top.utf:22`). To use it, relaunch the SiVal server without `--disable-dpi`, which the server command in the repo-root `mimir_setup.md` passes. With DPI on, the SMC AXI monitor also logs every host beat.
