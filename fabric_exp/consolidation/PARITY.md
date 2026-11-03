# Parity: configurations main's router runs that the new router does not

Check before the mainline port's router commit (2026-11-03). Router: `fabric_exp/variants/v4txa11` (its
supported-configuration `static_assert`s, `router/router_common.hpp`, plus the per-file asserts). Main: origin/main
`d2bd12fcf1f` (the port's base). Evidence is marked: **built** (the cov_m coverage builds of v4txa11 at -O3 and -Os,
3,124 configs; main's router was built on the same cases as `main_O3`) or **read** (source only, no build).

## 1. Rejected by the new router's asserts, produced by main's host

| # | Configuration | How main gets it | New router | CI / users | Evidence |
| --- | --- | --- | --- | --- | --- |
| 1 | Wormhole | every WH fabric | `static_assert(is_blackhole_arch)` | all WH CI | known, accepted (WH_BRINGUP.md) |
| 2 | VC1 + VC2 on one router: a 2D multi-mesh fabric with `TT_METAL_ENABLE_FABRIC_VC2=1` | `compute_intermesh_vc_config`: "VC2 is independent of VC1" (Blackhole, 2D, no UDM / mux) | `vc2_config_supported` requires `!has_vc1` (`rc:303`) | none found: the only multi-mesh VC2 test (`MultiHost.BHDualGalaxyNonUniformFabricEnvAcrossRanksFatal`) expects a throw before any router is built; VC2 is enabled only by test yamls (`use_vc2`, single mesh) | built: `bh_vc1_vc2*` fail at `rc:303` (4 cases, 131 configs). Whether main's router builds it is not checked yet (no main build on these cases); I check it with a mock capture at the base |
| 3 | UDM on a torus (`FABRIC_2D_TORUS_*` + `FabricTensixConfig::UDM`) | host requires 2D routing only (`fabric_context.cpp:163`) | `udm_config_supported` requires `!enable_deadlock_avoidance` | none: every UDM fixture is `FABRIC_2D` | read |
| 4 | UDM on a multi-mesh fabric (VC1) | no host check found | `udm_config_supported` requires `!has_vc1` | none found | read |
| 5 | Tensix mux mode on a multi-mesh fabric (VC1) | no host check found (only express + mux is fatal) | `!(tensix_mux_mode && has_vc1)` (`rc:301`) | none found | read |
| 6 | Stream-register credits on one TX queue (every one-ERISC Blackhole router: `TT_METAL_DISABLE_FABRIC_TWO_ERISC`, mux, UDM) | credit plan: counters only with multi-TXQ (`fabric_builder_context.cpp:52`) | `static_assert(sender_channel_uses_counter_credits)` (`rs:94`, `rr:173`) | all one-ERISC CI | built: `*_sc` cases fail. **Not a gap:** commit 4 gives the router kind counters on every VC (plan §3.6) |

Configs the host never produces (constants in the builder), rejected by both routers or by the new one only, so no gap:
`ENABLE_RISC_CPU_DATA_CACHE` (constexpr false), `ETH_TXQ_SPIN_WAIT_SEND_NEXT_DATA` (const false),
`FUSE_RECEIVER_FLUSH_AND_COMPLETION_PTR=0` (main asserts 1 too), dynamic NoC mode (both assert), first-level acks on
VC1 (host constant 0), VC2 forwarding (host constant `DISABLE_RX_CH2_FORWARDING=1`), perf recorder modes other than
low-resolution bandwidth (the enum has no others).

## 2. Accepted by the asserts, but a main feature the new router does not implement

| # | Feature | Main | New router | CI | Evidence |
| --- | --- | --- | --- | --- | --- |
| 7 | Streaming profiler's link clock sync (`TT_METAL_STREAMING_PROFILER=1`, `-DPROFILE_STREAMING`) | `link_sync::RouterHook::step(heartbeat_counter)` in the main loop (`fabric_erisc_router.cpp:2247`, since 2026-10-21); the host gives ERISC0 of each synced link a transmitter or receiver role (`streaming_profiler/sync/link_sync.hpp`, `add_router_compile_args`) | no hook: the links' clock-sync ports are never served, so cross-chip sync has no data on fabric links | `tests/ttnn/tracy/test_streaming_profiler_sync.py` and the streaming profiler tests | read. Small to add (one call per round on the sender / single-ERISC loop, under `PROFILE_STREAMING`) |

## 3. Accepted by the asserts, never built by the new router

No static_assert rejects these (by reading), but no coverage case built them, so a compile failure or a wrong path is
possible. I add mock captures for the ones a mock cluster can produce and build them after commit 5, before any device
job.

| # | Configuration | Where it runs | Why it is different |
| --- | --- | --- | --- |
| 8 | `TT_METAL_FABRIC_DISABLE_INTERMESH_VC=1` (new on origin/main): inter-mesh traffic lands on VC0, the Z boundary router forwards it to N and S only, N/S routers on boundary chips get a fifth VC0 sender | default in the GLM-5.3 and Kimi K3 SC4 prefill runners (multi-Galaxy); no CI entry found (the PR's CI test `bh-galaxy-d2d-socket-sync-shared-links` is one 8x4 mesh, so no intermesh) | VC0 intermesh ingress on a Z-facing router; 5 VC0 senders without express |
| 9 | Multi-mesh over Z ports (Z-facing intermesh boundary routers: 5 VC0 / 4 VC1 senders, VC0 crossover from Z) | multi-Galaxy and split-Galaxy CI (`galaxy_e2e_tests.yaml`, exabox) | v4's VC1 coverage is the P150x8 2x2 split, intermesh on cardinal ports only (`bh_vc1`: no `MY_DIRECTION=4` with `IS_INTERMESH_ROUTER=1`) |
| 10 | Mesh pass-through (`TT_METAL_ENABLE_FABRIC_MESH_PASS_THROUGH=1`, experimental): inter-mesh routers service VC1 (`FABRIC_2D_VC1_SERVICED` with `IS_INTERMESH_ROUTER`) and forward VC1 onto Z | `galaxy-fabric-multi-mesh-*` CI (BH Galaxy split into 3 meshes) | VC1 serviced on an intermesh router |
| 11 | Multi-mesh with per-mesh tori (VC1 plus bubble flow control on VC0) | `galaxy-fabric-per-mesh-torus-yx` CI (`test_fabric_per_mesh_heterogeneous.yaml`, 4 MGDs), `test_bh_glx_2d_torus_multi_mesh.yaml` | VC1 coverage is mesh topology only |
| 12 | Express on the E/W axis (`FABRIC_EXPRESS_AXIS=1`, new on origin/main) | express MGDs with E/W chords | the codec takes the axis (port: `Fabric2DRoutingConfig::decode_action`); v4 keeps N/S live on a Z-facing router (main now trims them): a superset, so correct by reading, not built |
| 13 | 1D routes past 16 hops (`FABRIC_1D_PKT_HDR_EXTENSION_WORDS > 0`) | Galaxy-wide 1D lines and rings | v4 implements the refill (`rrt:155-170`); every coverage config has 0 extension words |

## 4. Where the new router supports more than main (for the record)

- VC2 on one ERISC: main's router fails to build it (`bh_2d_vc2_1erisc`, `main_O3`: "VC2 receiver channel not
  serviced"); v4 builds it (validated: `vc2_1e` in variant_val_runs.sh).

## 5. Covered (built and run in the study), for completeness

1D line / ring, NeighborExchange (speedy path), 2D mesh / torus X, Y, XY, BH Galaxy mesh / ring / torus and N/S express
(5 VC0 senders, Z routers), VC1 on the P150x8 2x2 split (both ERISC modes), VC2 neighbour exchange (single mesh, both
ERISC modes), Tensix mux (1D, 2D, ring), UDM (`Fabric2DUDMModeFixture`), channel-trimming capture / replay /
overrides including the speedy-trim paths, telemetry (dynamic stats, BW telemetry, code profiling), router commands
(PAUSE / RUN / RETRAIN), context switching (idle and interval, dispatch's interval 16), DPRINT / watcher (presets off;
code size under the watcher not measured).
