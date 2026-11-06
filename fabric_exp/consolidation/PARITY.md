# Parity: configurations main's router runs that the new router does not

Check before the mainline port's router commit (2026-11-03). Router: `fabric_exp/variants/v4txa11` (its
supported-configuration `static_assert`s, `router/router_common.hpp`, plus the per-file asserts). Main: origin/main
`d2bd12fcf1f` (the port's base). Evidence is marked: **built** (the cov_m coverage builds of v4txa11 at -O3 and -Os,
3,124 configs; main's router was built on the same cases as `main_O3`) or **read** (source only, no build).

## 1. Rejected by the new router's asserts, produced by main's host

| # | Configuration | How main gets it | New router | CI / users | Evidence |
| --- | --- | --- | --- | --- | --- |
| 1 | Wormhole | every WH fabric | `static_assert(is_blackhole_arch)` | all WH CI | known, accepted (WH_BRINGUP.md) |
| 2 | VC1 + VC2 on one router: a 2D multi-mesh fabric with `TT_METAL_ENABLE_FABRIC_VC2=1` | `compute_intermesh_vc_config`: "VC2 is independent of VC1" (Blackhole, 2D, no UDM / mux) | `vc2_config_supported` requires `!has_vc1` (`rc:303`) | none found: the only multi-mesh VC2 test (`MultiHost.BHDualGalaxyNonUniformFabricEnvAcrossRanksFatal`) expects a throw before any router is built; VC2 is enabled only by test yamls (`use_vc2`, single mesh) | built: `bh_vc1_vc2*` fail at `rc:303` (4 cases, 131 configs). Main's router builds it: a mock capture at the base (P150x8 split into two 2x2 meshes, VC2 on) JIT-built 96 router ELFs, all with `FABRIC_2D_VC1_ACTIVE`, `FABRIC_2D_VC2_SERVICED` and one VC2 sender channel |
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

## Decisions (user, 2026-11-03) and the port's handling

- **VC1 + VC2 (row 2):** supported. The port drops `!has_vc1` from `vc2_config_supported` (as the v4txa11v probe did,
  which passed VC1 + VC2 correctness on two Blackhole ERISCs). Host opt level: Os when VC1 and VC2 are both active.
  2026-11-04: VC1 + VC2 with deadlock avoidance (per-mesh torus) does not build (transaction IDs; see the coverage
  section); the -Os rule was carried over from the study builder without a measurement.
- **Known gaps, rejected by the host with a TT_FATAL naming the combination** (for the PR description):
  - UDM on a torus or ring axis (UDM with deadlock avoidance): `erisc_datamover_builder.cpp`, per router;
  - UDM on a multi-mesh fabric with an inter-mesh VC (VC1): `fabric_builder_context.cpp`, compute_intermesh_vc_config;
  - Tensix mux on a multi-mesh fabric with an inter-mesh VC (VC1): same check.
  - Wormhole (accepted; WH_BRINGUP.md).
- **Streaming-profiler link clock sync (row 7):** added as its own commit after the router commit.
- **Rows 8-13:** offline coverage builds right after the router commit; 8-11 (intermesh VC disabled, Z-port multi-mesh,
  pass-through, per-mesh tori) are must-pass device runs (Galaxy, quad) before landing; 12-13 on device if easy.

## Coverage builds after the router commit (mainline worktree, offline, 2026-11-03)

Mock captures of the worktree's own binaries (tt-run discovery mode for multi-mesh, as CI's CPU-only fabric tests), the
JIT building the new router for every link; then an offline rebuild of every logged build. No device runs yet.

| Row | Case (mock) | Router ERISC builds | Result |
| --- | --- | ---: | --- |
| 8 | dual BH Galaxy, `TT_METAL_FABRIC_DISABLE_INTERMESH_VC=1`, two and one ERISCs | 232 + 116 | build |
| 9 | dual BH Galaxy (Z-port boundary routers, VC1 over Z); BH Galaxy split into 3 meshes | 230; 424 | build, after a fix in the router commit: sender stream registers are initialized for the router's own channels only (the host sizes the channel arrays for the fabric's widest router, and a mesh router next to a Z boundary router had positions past its channels: compile error before the fix) |
| 10 | BH Galaxy split into 3 meshes, `TT_METAL_ENABLE_FABRIC_MESH_PASS_THROUGH=1` (CI's yaml) | 428 | build |
| 11 | BH Galaxy, CI's per-mesh torus Y/X/Y MGD (VC1 + bubble flow control) | 208 | build |
| 12 | BH Galaxy 4x8 with express on E/W (`FABRIC_EXPRESS_AXIS=1`) | 260 | build |
| 13 | BH Galaxy 1x32 line and ring (one extension word) | 72 + 76 | build |
| 2 | P150x8 split into two 2x2 meshes with VC2 (VC1 + VC2) | 96 | build |

Code size (loadable bytes, E0 + E1 upper bound per case against the 26,624 B kernel config buffer): largest 25,848 B
(3-mesh pass-through), then 24,152 B (dual-Galaxy Z-port), 23,860 B (3-mesh); every case's router programs passed the
host's program-size check in the mock runs. **WRONG (2026-11-04):** a mock run only compiles the fabric
(`compile_fabric_only`); it never lays out the program, so the host's size check never ran there. Program sizes with
runtime args, from a scratch host that does lay them out, are in the next section.

### VC1 + VC2 program sizes and the per-mesh torus with VC2 (offline, 2026-11-04)

Program size = runtime args + E0 + E1 text, as the host's `finalize_offsets` computes it (scratch host, never
committed: worktree `tt-metal-mainline-slots` at 58a9a1e, the mock compile path also lays out and logs each fabric
program). Kernel config buffer 26,624 B. Main's tier slots unless marked.

| Config (VC1 + VC2) | -Os (as pushed) | -O3 | -O3 margin |
| --- | ---: | ---: | ---: |
| P150x8 2x2 split, two ERISCs | 23,504 | 25,168 | 1,456 left |
| same, slots 17,8,16,5,4 + 3,2 (dynamic allocator's row) | 23,536 | 25,248 | 1,376 left |
| P150x8 2x2 split, one ERISC (allocator row) | 19,472 | 20,304 (20,448) | 6,320 (6,176) left |
| BH Galaxy 2x2 Z-ring, two / one ERISC | 24,368 / - | 26,016 / 21,312 | 608 / 5,312 left |
| dual BH Galaxy mesh | 25,312 | 27,024 | **400 B over** (1 of 64 programs) |
| BH Galaxy split into 3 meshes | 25,312 | 27,024 | **400 B over** (4 of 32) |
| same, mesh pass-through | 25,728 | 28,048 | **1,424 B over** (22 of 32) |

- The host builds VC1 + VC2 at -Os (the study builder's rule, carried over; not measured in the port). At -O3 three
  Galaxy multi-mesh configs do not fit.
- **The per-mesh torus with VC2 (row 11 + row 2) does not build, at -Os or -O3:** routers with deadlock avoidance get
  `NUM_TRANSACTION_IDS` 8 on 3 receiver channels (24 IDs for 16), and the transaction-ID tracker's `static_assert`
  ("Invalid transaction ID") fails. Not in the coverage set before; a gap of the VC1 + VC2 lift (row 2).

### CI multi-mesh program sizes after the code-size commits (offline, 2026-11-04)

Final code = cleanup branch tip cb42eb72501 (on 4db4f45790b): the cold attribute on the control paths and landing option 1
(multicast landing out of line, cold); the cold wait_for_notification commit was dropped (it changed single-ERISC loop
code in mux / UDM builds). Program size = runtime args + E0 + E1 text, worst program per config, from a scratch host with
the program-size log (never committed; patch ~/.tt-metal-mainline-cov-ci-final/sizelog_impl_patch_at_6a99933cdc4.patch,
applied to tt-metal-mainline-glx at the tip for this run, then reverted); CI's configs as CI runs them (VC2 off, so the
host builds -O3). Kernel config buffer 26,624 B. Logs: ~/.tt-metal-mainline-cov-ci-final2 (ci_table_final2.txt; the
with-coldwait run: cov-ci-final); main: ~/.tt-metal-mainline-cov-ci-main (scratch host `tt-metal-main-sz` at d2bd12f,
removed; its patch: sizelog_impl_patch_at_d2bd12f.patch there).

| CI config (-O3) | main | port at 58a9a1e | final | final margin |
| --- | ---: | ---: | ---: | ---: |
| 2x2 multi-mesh over Z | 25,184 | 23,536 | 22,800 | 3,824 |
| 2x2 multi-mesh N/S/E/W ring | 25,184 | 23,536 | 22,800 | 3,824 |
| channel trimming over Z | 26,096 | 24,368 | 23,744 | 2,880 |
| channel trimming N/S/E/W | 26,096 | 24,368 | 23,648 | 2,976 |
| heterogeneous big mesh | 25,744 | 24,400 | 23,872 | 2,752 |
| heterogeneous combo | 24,608 | 23,136 | 22,400 | 4,224 |
| heterogeneous per-mesh torus | 24,864 | 23,392 | 22,640 | 3,984 |
| heterogeneous 2x4 | 24,480 | 22,800 | 22,288 | 4,336 |
| 3 meshes 4x4 + 2x4s | 25,744 | 24,400 | 23,664 | 2,960 |
| same, pass-through | 26,128 | 26,416 | 25,056 | 1,568 |
| 3 meshes 2x8 + 2x4s | 25,744 | 24,768 | 23,664 | 2,960 |
| same, pass-through (G0-mm-2x8) | **26,720 (96 B over)** | **27,344 (720 B over)** | 25,888 | **736** |
| dual Galaxy | 25,744 | 24,592 | 23,664 | 2,960 |
| P150x8 VC1 split | 24,480 | 22,560 | 21,808 | 4,816 |

Every CI multi-mesh config fits at -O3 on the final code; the 2x8 pass-through line, which main fails by 96 B, has
736 B left (896 B with the dropped cold wait_for_notification, which saved 160-224 B per config).

Final commits (cleanup branch, unpushed): 3b4ccab4dd7 (cold control paths) and cb42eb72501 (landing option 1), on
4db4f45790b. Offline gate (full builds of the 46-case coverage set, per function with branch / call targets normalized):
non-router kernels 191/191 identical; 3b4ccab4dd7 changes only the three cold functions, `_start` immediates and layout;
cb42eb72501 changes 227 E1 images (inter-mesh ingress routers) in the landing functions and `_start`, and in 44 mesh
pass-through E1 images (glx3_passthrough) also receiver::run's register allocation: **a codegen side effect, cause
unknown** (not IPA-RA: an -fno-ipa-ra build shows the same change), same calls, stack frame 240 -> 224 B, stack loads /
stores -2 to -6, length -4 (4 images), -3 (30) or +15 (10) instructions. Not measured on device; not a perf claim.

**Multicast intermesh landings are unmeasured on device.** Landing option 1 moves the multicast landing rebuild out of
line (a call, cold placement): the path it can slow. test_tt_fabric generates no inter-mesh multicast ("multicast
cannot cross mesh boundaries"), and the only inter-mesh multicast tests are the multihost gtests for T3K / dual 2x4,
so that path has no runnable traffic on the P150. The P150 A/B (study variant v4txa11land against v4txa11, census_vc1
rows) covers the unicast landings, which run the changed set_landing_route on every ingress packet.

**Known gap (non-CI): VC1 + VC2 on the 2x8 pass-through does not fit.** Final code, the host's rule for VC1 + VC2
(-Os): 27,312 B, **688 B over** (4 of 32 programs); at -O3 28,352 B, 1,728 B over (12 of 32). Main does not build
VC1 + VC2 at all (the port lifted that, row 2). Left for later code-size merges (an init/open merge prototype measured
27,312 B at -O3; not on the branch).

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

## NoC API against main (port + cleanup batch, 2026-11-04)

The port adds no NoC function names; it only extends main's functions (after 288afd237a7 on the cleanup branch, which
removes the last added name, `noc_write_to_node_with_state`). Each extension is kept by an A/B that removed it (P150
broker 717-720, Galaxy 172435; cycles/packet, removal arm against v4txa11; TODO.md "Mainline port follow-ups"):

- `noc_inline_dw_write_with_state` (dataflow) / `noc_fast_write_dw_inline_with_state` (Blackhole, Wormhole; Quasar v2/v3
  take the arguments): `update_addr_lo` and `update_addr_hi` together, the node from a trailing `dst_noc_addr`
  argument (as `noc_wwrite_with_state` takes it); `skip_cmdbuf_chk` template argument. The node-addressed inline credit
  write is required: A5 (the stateless credit write instead) hung on the Galaxy N/S rows with over-credited and lost
  credits (job 172435), so it is functionally required, not only faster.
- `ncrisc_noc_set_transaction_id` (Blackhole, Wormhole): `skip_cmdbuf_chk`. A1 (both skip flags off): lin3 +10.6
  (+8.5%), multi-downstream proxy 3-hop +19.7 (+13.6%), Galaxy N/S +16..+23 (pass a); kept.
- `noc_wwrite_with_state` (Blackhole; Wormhole and Quasar v2/v3 accept it unused): `update_addr_mid` (default true;
  false skips the NOC_RET_ADDR_MID store for on-chip destinations). A2 (main's form, with the MID store): +0.1..+0.4 on
  the P150 multi-downstream proxy, +0.7..+1.0 on the Galaxy N/S rows (pass a); user decision: extend main's function
  (option b) rather than keep a separate name. Gate: 5,007 / 5,007 images byte-identical to cb42eb72501 (same register
  writes). Pushed as 288afd237a7.
- Removed: `noc_inline_dw_write(..., cmd_buf)` / `emulated_write_on_cmd_buf` (the router uses the stock call and restores
  its local-write preset), both `*_inline_*_to_node_with_state` forms, and `noc_write_to_node_with_state`.
- Router-internal A/Bs on the same jobs (no NoC API involved): A3 (header load/unpack) E/W scatter +6.8, ring1 +5.9,
  Y-torus N/S +6.4, ring multicast -1.8; A4 (one credit frame per event) ring1 +12.6, Y-torus N/S +14.3, 2D multicast
  +10.3; both kept.

## Device validation notes (port, 2026-11-04)

- **VC1 + VC2 split "hang" (broker 660, batch tip 8b615f7, MultiMeshAllToAll 4096 B unicast_write): a test-harness
  race in our own `.scratch/vc1/vc1_sanity.yaml`, not a router fault.** Observed (worker decode of the capture): rank
  0's 16 inter-mesh receivers each sat at packet 1 of 100 while their rank-1 senders had sent all 100 and the routers
  were idle and empty; every one of their 64-slot rings held packets 65-100 in slots 1-36 and 37-64 in slots 37-64,
  every word correct (the overrun is observed). The yaml allows it: 100 x 4096 B into a 256 KiB ring, no receiver
  credits, and a start sync per mesh only. "Rank 0 started late" is a hypothesis (the sleep-on-one-rank A/B was
  skipped). Re-runs: 669 (batch tip) and 670 (pushed 58a9a1e) 3/3 each. Fix: num_packets capped at the ring's slots
  (100 -> 64, `slotalloc/cap_ring_packets.py`); main's `test_fabric_multi_mesh_sanity_common.yaml` MultiMeshAllToAll
  has the same pattern (num_packets 100 at 4096 B, sync on, no flow control).
- **Pushed and validated (2026-11-04): origin/aagarwal/fabric-router-mainline at 4db4f45790b** (the cleanup batch on
  b2473363016, without the -Os-for-VC1 opt rule). P150 runs: 659 inline_udm (pass); 650 two, non-trim runs (pass; its
  trim2d used the earlier random-pairing yaml, which hangs by design); 674 vc1 on the 4db4f45790b tree with the capped
  yaml (vc1 / vc1_1e at -O3, vc1_vc2 at -Os: 12/12 on both ranks each).
- **Code-size tip on the Galaxy (Slurm 172825, bh-glx-110-a04u14, staged tree mainline_cb42eb7):** G-PORT P10-mm-4x4
  and P10-mm-2x8 PASS: multi-mesh stability (104 tests) and the mesh pass-through yaml (48 tests) on 3 ranks each, 0
  failed, no program-size error; the 2x8 pass-through line, which main fails 96 B over, runs. Router images of
  cb42eb72501 = 288afd237a7 = 2fc06852f48 (offline gate). P150: 723 / 724 / 725 clean on cb42eb72501.

## Bubble-flow-control injection rule (branch aagarwal/fabric-router-mainline-bfc, 2026-11-04)

Commit `ac86bf0f116` on 4db4f45790b (not pushed; RING_JAM_FIX.md §10). Deadlock-avoiding routers inject with
T = max(2, R - 16) free far VC0 slots (main and the port before it: 2 always). Parity notes:
- **Changed, against main and 4db4f45790b:** ring routers with 32-slot receivers (1D ring on the P150, the Galaxy ring and
  the 1x32 ring) inject with 16 free far slots. It is one immediate in the injection compare, code size +0.
- **Unchanged (images byte-identical to 4db4f45790b, 49 configs):**
  - every torus (16-slot receivers: T stays 2): XY / X / Y, express, the per-mesh torus with VC1;
  - every config without deadlock avoidance;
  - ring + Tensix mux: in mux mode the router flags no injection channel, so main's mux-side rule (the mux's own
    2-slot space check) is all there is, as before.
- **New host arg:** `BUBBLE_FLOW_CONTROL_INJECTION_SENDER_CHANNEL_MIN_FREE_SLOTS`, only with deadlock avoidance; its
  value is from `builder_config::bubble_flow_control_injection_min_free_slots(R)`, cap
  `bubble_flow_control_injection_max_far_receiver_occupancy` (16).
- **Asserts:** the router static_asserts 2 <= T <= R for each injection channel. The host's existing check (far VC0
  receiver >= 2 slots with bubble flow control) is unchanged, so no config main or the port accepts is newly rejected.
- **Not covered by the gate set:** Wormhole (the port's WH bring-up is separate); 64-slot ring receivers (no default
  allocation gives one; T would be 48).
- **Device (2026-11-04):**
  - P150 broker 758: mesh, 1D, ring, Y and XY torus validation all pass on the branch. Broker 759: ring rows -0.6% to
    +0.1% against 4db4f45790b; torus rows unchanged.
  - Galaxy 172983: full-ring flow-control validation 10 / 10 on both trees; jam rows -23% to -28%.
  - The base in both was an own build of 4db4f45790b (the shared 4db4f45 trees had moved on or been removed).
