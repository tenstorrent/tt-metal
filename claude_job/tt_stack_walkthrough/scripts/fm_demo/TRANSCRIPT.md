TTFM run_fabric_manager split-lifecycle demo — single T3K (2x4), t3k-node-b, 2026-07-15
Binary: build_Release/tools/scaleout/run_fabric_manager   (FabricManagerMode split lifecycle)
============================================================================================

### PHASE 1 — INIT  (run_fabric_manager --initialize-fabric --fabric-config FABRIC_2D
###                  --reliability-mode RELAXED_INIT --mesh-shape 2x4)   → SUCCESS
# Metal takes the INIT_FABRIC path, compiles + launches the EDM routers, LEAVES FABRIC UP.
2026-07-15 12:13:40.720 | info     |     Distributed | Setting fabric config to: FABRIC_2D (fabric_manager_utils.cpp:33)
2026-07-15 12:13:40.720 | info     |     Distributed | Using mesh shape: 2x4 (fabric_manager_utils.cpp:33)
2026-07-15 12:13:40.728 | info     |          Device | Slow dispatch mode: Using full logical grid (8, 8) (core_descriptor.cpp:220)
2026-07-15 12:13:40.730 | info     |          Device | Slow dispatch mode: Using full logical grid (8, 8) (core_descriptor.cpp:220)
2026-07-15 12:13:41.468 | info     |           Metal | Initializing Fabric (fabric_firmware_initializer.cpp:296)
2026-07-15 12:13:47.275 | info     |           Metal | Fabric initialized on Device 0 (device.cpp:548)
2026-07-15 12:13:49.572 | info     |           Metal | Fabric initialized on Device 1 (device.cpp:548)
2026-07-15 12:13:49.572 | info     |           Metal | Fabric initialized on Device 2 (device.cpp:548)
2026-07-15 12:13:49.573 | info     |           Metal | Fabric initialized on Device 3 (device.cpp:548)
2026-07-15 12:13:49.576 | info     |           Metal | Fabric initialized on Device 4 (device.cpp:548)
2026-07-15 12:13:49.584 | info     |           Metal | Fabric initialized on Device 5 (device.cpp:548)
2026-07-15 12:13:49.587 | info     |           Metal | Fabric initialized on Device 6 (device.cpp:548)
2026-07-15 12:13:49.595 | info     |           Metal | Fabric initialized on Device 7 (device.cpp:548)
2026-07-15 12:13:49.595 | info     |           Metal | Fabric initialized on 8 devices (fabric_firmware_initializer.cpp:444)
2026-07-15 12:13:49.595 | info     |           Metal | Fabric Initialized with config FabricConfig::FABRIC_2D (fabric_firmware_initializer.cpp:313)
2026-07-15 12:13:49.597 | info     |     Distributed | Fabric Node IDs: (fabric_manager_utils.cpp:36)
2026-07-15 12:13:49.597 | info     |     Distributed |   Chip ID: 0, Fabric Node ID: (M0, D2) (fabric_manager_utils.cpp:36)
2026-07-15 12:13:49.597 | info     |     Distributed |   Chip ID: 1, Fabric Node ID: (M0, D6) (fabric_manager_utils.cpp:36)
2026-07-15 12:13:49.597 | info     |     Distributed |   Chip ID: 2, Fabric Node ID: (M0, D5) (fabric_manager_utils.cpp:36)
2026-07-15 12:13:49.597 | info     |     Distributed |   Chip ID: 3, Fabric Node ID: (M0, D1) (fabric_manager_utils.cpp:36)
2026-07-15 12:13:49.597 | info     |     Distributed |   Chip ID: 4, Fabric Node ID: (M0, D0) (fabric_manager_utils.cpp:36)
2026-07-15 12:13:49.597 | info     |     Distributed |   Chip ID: 5, Fabric Node ID: (M0, D7) (fabric_manager_utils.cpp:36)
2026-07-15 12:13:49.597 | info     |     Distributed |   Chip ID: 6, Fabric Node ID: (M0, D4) (fabric_manager_utils.cpp:36)
2026-07-15 12:13:49.597 | info     |     Distributed |   Chip ID: 7, Fabric Node ID: (M0, D3) (fabric_manager_utils.cpp:36)
2026-07-15 12:13:49.597 | info     |     Distributed | ✓ Fabric status written to: /home/namvu/dual-t3k/tt-metal/claude_job/tt_stack_walkthrough/scripts/fm_demo/fabric_status.txt (fabric_manager_utils.cpp:36)

### fabric_status.txt written by the manager:
    Fabric Status Report
    Generated: 2026-07-15 12:13:49
    ==========================================
    Fabric Configured: Yes
    Fabric Initialized: No

    SetFabricConfig Parameters:
      Fabric Config: FABRIC_2D
      Reliability Mode: STRICT_SYSTEM_HEALTH_SETUP_MODE
      Num Routing Planes: 0
      Fabric Tensix Config: DISABLED
      Fabric UDM Mode: DISABLED
      Fabric Manager: INIT_FABRIC

    Additional Status:
      Number of Active Ethernet Channels: 0
      Active Chips: 0
      Active Hosts: 0

### PHASE 2 — ATTACH  (fm_workload.py: set_fabric_config(FABRIC_2D, RELAXED_INIT,
###                    fabric_manager_mode=ENABLED) + open_mesh_device(2x4))   → FAILED on T3K
# The attach never reaches the ENABLED 'Fabric initialized through Fabric Manager' path:
# UMD topology discovery must probe the T3K's remote chips (4-7) OVER ETHERNET, but the EDM
# routers the manager left running block the eth-heartbeat, so the device won't even open.
RuntimeError: Timed out waiting for ETH heartbeat on device ASIC ID: 10225917990, ETH core e1-6 (NOC0) to advance. Stuck at 0xabcd7841
 1. tt::umd::TopologyDiscovery::eth_heartbeat_running(tt::umd::TTDevice*, unsigned long, tt::umd::CoreCoord)
 2. tt::umd::TopologyDiscovery::discover_remote_devices()
 (raised inside SetFabricConfig -> MetalContext::instance -> Cluster ctor -> topology discovery)

### PHASE 3 — TERMINATE  (run_fabric_manager --terminate-fabric ... --mesh-shape 2x4)  → FAILED same way
# The teardown tool ALSO opens the device first, so it hits the identical eth-heartbeat wall
# (unhandled UmdException -> std::terminate). The FM fabric can neither be attached nor torn
# down via the tool on a T3K; recovery is a chip reset.
terminate called after throwing an instance of 'tt::umd::error::UmdException<tt::umd::error::EthFirmwareHeartbeatError>'
  what():  Timed out waiting for ETH heartbeat on device ASIC ID: 10225917957, ETH core e4-0 (NOC0) to advance. Stuck at 0xabcd5201

### RECOVERY — tt-smi -r on BOTH hosts concurrently, wait ~90s, re-check test_system_health.
============================================================================================
TAKEAWAY (separate-process CLI): run_fabric_manager proves the INIT half of the split lifecycle
+ the FabricManagerMode mechanism on real HW, but its ENABLED/TERMINATE halves are Galaxy-oriented
— they assume every chip is directly (PCIe) accessible. On a T3K, remote chips are reached over
ethernet, so a fabric that's already up blocks the very discovery the tool needs to re-open the
device. This is why the in-repo fabric-manager CCL tests (tests/scale_out/test_ccl_fabric_manager.py)
are 8x4 Galaxy-only and still gated behind 'TODO: Enable once Fabric Manager is ready'.

>>> FOLLOW-UP FIX — see 04_TRANSCRIPT_inprocess.md (+ 04_lifecycle.log, fm_lifecycle.py).
The eth-heartbeat wall above only bites a FRESH process (new UMD Cluster => re-discovery). Driving
the SAME FabricManagerMode split lifecycle (INIT_FABRIC -> ENABLED -> TERMINATE_FABRIC) in ONE
process builds the UMD Cluster once, before any fabric is up, and reuses it — so the ENABLED ATTACH
now SUCCEEDS on the T3K (captured: "Fabric initialized through Fabric Manager", fabric_firmware_
initializer.cpp:320). Remaining boundary: a workload DISPATCHED under the in-process ENABLED reattach
hangs on its first device write on a T3K (remote-chip dispatch tunnels over the ethernet the
ENABLED-mode control-plane reconfigure disturbs), so a fully-green ENABLED workload is still not
achievable on this T3K/commit. The fully-green fabric proof stays the DEFAULT-mode 16-chip workload
(../PASS_output.txt).
