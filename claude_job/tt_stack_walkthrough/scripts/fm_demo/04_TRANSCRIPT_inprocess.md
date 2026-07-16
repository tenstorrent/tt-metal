PHASE 4 — In-process split lifecycle (fm_lifecycle.py), single T3K 2x4, 2026-07-15
Goal: clear the eth-heartbeat wall that broke the CLI's ENABLED/TERMINATE on a T3K by
driving the SAME FabricManagerMode enum values in ONE process (one UMD Cluster, built
once before any fabric is up -> no re-discovery).
-------------------------------------------------------------------------------------
## Topology discovery happens EXACTLY ONCE (start of INIT, before fabric up):
2026-07-15 18:17:14.058 | info     |             UMD | Cluster constructor started. (cluster.cpp:348)
2026-07-15 18:17:14.060 | info     |             UMD | Starting topology discovery. (topology_discovery.cpp:114)
2026-07-15 18:17:14.103 | info     |             UMD | Completed topology discovery. (topology_discovery.cpp:118)

## PHASE INIT (INIT_FABRIC) -> fabric up, left up:
=== PHASE INIT (fabric_manager_mode=FabricManagerMode.INIT_FABRIC) ===
2026-07-15 18:17:14.351 | info     |           Metal | Initializing Fabric (fabric_firmware_initializer.cpp:296)
2026-07-15 18:17:20.136 | info     |           Metal | Fabric Initialized with config FabricConfig::FABRIC_2D (fabric_firmware_initializer.cpp:313)
[INIT] mesh open (2, 4) = 8 chips
[INIT] mesh closed

## PHASE ENABLED -> ATTACH SUCCEEDS (this is the line the CLI could never reach on a T3K):
=== PHASE ENABLED (fabric_manager_mode=FabricManagerMode.ENABLED) ===
2026-07-15 18:17:22.864 | info     |           Metal | Fabric config changed from FabricConfig::FABRIC_2D to FabricConfig::FABRIC_2D, reinitializing control plane (metal_env.cpp:319)
2026-07-15 18:21:58.280 | info     |           Metal | Fabric initialized through Fabric Manager (fabric_firmware_initializer.cpp:320)
[ENABLED] mesh open (2, 4) = 8 chips
   NOTE: no second 'topology discovery' above -> the reattach reused the existing UMD
   Cluster, so the eth-heartbeat probe (which aborted the separate-process CLI) never ran.

## THEN: the first workload dispatch under the ENABLED reattach HUNG.
   No '[ENABLED] PASS ADD' line was ever printed. Diagnosis (live): no compiler procs,
   0 kernels built in 3 min, main thread parked in futex_wait, one thread spinning 99%
   => a hung device op, not a slow compile. It stalled on the first from_torch/replicate
   write, which on a T3K tunnels dispatch to the remote chips (4-7) over the same ethernet
   the ENABLED-mode control-plane reconfigure (configure_ethernet_cores_for_fabric_routers)
   disturbs while the routers are live. Process was SIGTERM'd; fabric left up -> tt-smi -r
   both hosts to recover (health green after).
-------------------------------------------------------------------------------------
RESULT: the in-process method FIXES the original hard failure — the ENABLED attach and the
split lifecycle no longer hit the eth-heartbeat abort (attach proven above). But a workload
RUN under the in-process ENABLED reattach hangs on first dispatch on a T3K, so a fully-green
ENABLED workload is not achievable on this T3K in commit 3632044cfd6. The fully-green fabric
proof remains the DEFAULT-mode 16-chip workload (scripts/PASS_output.txt); DEFAULT mode is
INIT|TERMINATE, i.e. the same lifecycle with Metal owning both ends and no mid-life reattach.
