<!-- SUMMARY: Synthesized swarm findings on non-MMIO silent failure root causes, main comparison, UMD diff, and log cross-reference
KEYWORDS: non-mmio, silent-failure, GetNumAvailableDevices, topology-check, base-umd, relay-broken, FIX AY, FIX AC, FIX QW, UMD, teardown
SOURCE: Researcher A + B findings + log analysis of run 25246631881 (t3k-13) + UMD code diff
SCOPE: Full causal chain from topology check contamination to cascading resets to GAP-21 failure
USE WHEN: Investigating non-MMIO relay dead state, topology check contamination, branch vs main comparison -->

# Non-MMIO Silent Failure Synthesis

## Root Cause Chain

1. `GetNumAvailableDevices()` topology check in `run_t3000_unit_tests.sh` opens UMD cluster
2. `deassert_risc_resets()` installs base-UMD firmware on all 44 inter-chip ETH channels including non-MMIO
3. Python exits → `Cluster::~Cluster()` does NOT call `close_device()` → 44 channels left in `0x49706550` base-UMD state
4. `terminate_stale_erisc_routers()` (FIX M) detects sentinel → uses launch_msg for non-MMIO (relay-dependent)
5. launch_msg via stale relay fails silently → FIX AY deferred reset triggered but fails (UMD relay state not re-synced after FIX AC PCIe reset)
6. Non-MMIO ERISCs remain in base-UMD/partial-fabric state → FIX QW skips all hardware tests
7. Repeated FIX AC resets accumulate → hardware instability → GAP-21 `Device 0 init: failed to initialize FW!`

## Main Branch Comparison

- Main does NOT have the topology check → no contamination → never hits this scenario
- Main has zero teardown protection (no try/catch, no relay-broken skips) but doesn't need it
- Branch adds ~30 FIX annotations as defensive teardown

## UMD Branch Commits Over Main

- FIX PZ: Cache unreachable ASIC IDs, avoid repeated 5s probes
- FIX AF: Timeout on read_non_mmio() CMD queue full spin
- FIX NU: Capture MMIO EthCoord before relay-safety guards
- FIX NT: Preserve EthCoord for FIX-AQ-skipped devices
- FIX AE: relay_broken_ flag → wait_for_non_mmio_flush() returns immediately
- FIX W/AQ/AQ-2/AU/X: Topology discovery crash guards
- 5a2e723c3: SiliconSysmemManager destructor try-catch (prevents SIGABRT, prerequisite)

## Log Cross-Reference (run 25246631881, t3k-13)

```
07:28:56  GetNumAvailableDevices() contaminated 44 channels (MMIO: 6×4, non-MMIO: 4×4)
07:29:02  terminate_stale_erisc_routers: all 8 devices show base_umd channels
07:29:08  MultiCQFabricMeshDevice2x4Fixture.AsyncExecutionWorksCQ0 SKIPPED (FIX QW, 11.5s)
07:29:43  FIX AQ: UMD topology probing dead non-MMIO chips, 5s timeouts per device
07:30:08  relay dead on devices 4-7, skipping erisc_send_exit_signal
07:30:28  MeshDevice1x4Fixture.AllGatherPersistentOutput SKIPPED (65s — repeated probe timeouts)
07:31:20  Device 4 not active (FIX BC), all hardware tests skip
07:31:31-07:33:01  5 resets in 90s (script tt-smi -r + conftest ensure_cluster_healthy)
07:33:57  Device 0 fails FW init → RuntimeError: Try resetting the board
```

## Remaining Gaps

1. FIX AY fails when UMD relay state not re-synced after FIX AC PCIe reset
2. Topology check is the contamination source — needs non-cluster-opening replacement
3. Repeated FIX AC resets accumulate hardware instability faster than recovery
