# Regression test plan: landing the new fabric router

Companion to MAINLINE_PLAN.md (2026-11-03). Design only: nothing here was run, submitted or committed.

**Sources** (all read from origin/main `91b4da52da0`, the base of the port):
- the CI test lists `tests/pipeline_reorg/*.yaml`, each entry with `name`, `cmd` and `skus: {sku: {timeout}}`;
- the workflows in `.github/workflows/` that consume them;
- the SKU-to-runner map `.github/sku_config.yaml`;
- `tests/scripts/{t3000,tg,multihost}/*.sh`;
- the perf-target code: `models/model_targets.yaml`, `models/demos/utils/model_targets.py`,
  `tests/tt_metal/tt_fabric/test_infra/golden/`;
- the Slurm partitions as `sinfo -s` lists them today.

**Runtimes** are the CI job timeouts from the test lists. They are upper bounds, not measured durations. Anything
I could not verify is marked *(unverified)*.

---

## 0. What can regress, and the two run modes

MAINLINE_PLAN §7 lands the work in two kinds of change:

| Change | MAINLINE_PLAN commits | What it can break | Router under test |
| --- | --- | --- | --- |
| **Shared code** | 1 NoC API, 2 Ethernet API, 3 fabric library + packet header + ct args, 4 host builder | Every kernel that includes `dataflow_api.h` / `noc_nonblocking_api.h` (all arches, Quasar included); main's router on Wormhole and Blackhole; host args, slot and credit choices | **main's router** (the default) |
| **New router** | 5 router, 6 slot policy, 7 Wormhole bring-up | Fabric traffic, CCLs and models when the new router is selected | **new router** (opt-in) |

So every test group below runs in one of two modes:
- **`main`**: default router. The selection variable is unset.
- **`new`**: the new router is selected.
  - The variable name is not final; MAINLINE_PLAN §1.6 proposes `TT_METAL_FABRIC_ROLE_ROUTER=1`.
  - In the study tree today it is `TT_FABRIC_KERNEL_VERSION=2` plus `swap.py apply`.
  - The runner keeps this mapping in one place (§4.2).

**CI cannot pass the selection variable.** Workflows take no free-form env. To get CI coverage of the new router, push
a throwaway branch whose only extra commit flips the default to the new router, and dispatch the workflows on that
branch. Never merge that commit.

**Perf A/B rule** (the study's, from memory notes): main and new run in one job on one host, with passes in reversed
order (main→new, then new→main). Multi-hop rows are compared within one host only.

### 0.1 Systems

| System | Arch | Chips | CI SKU(s) (`sku_config.yaml`) | Exabox Slurm partition (`sinfo`) | Notes |
| --- | --- | ---: | --- | --- | --- |
| P150 single card | BH | 1 | `bh_p150`, `bh_p150b_civ2`, `bh_p150_perf`, `bh_p100` (no Ethernet) | none. A LoudBox with `TT_VISIBLE_DEVICES=0` is a stand-in *(unverified equivalence)* | no fabric. Covers the NoC API change only |
| P300 | BH | 2 | `bh_p300` exists, but every fabric entry has it commented out | none. A `bh_qb2` node limited to one card *(unverified)* | no active CI coverage |
| QuietBox 2 (2× P300) | BH | 4 | `bh_quietbox_2` | `bh_qb2` (9 nodes `qb2-120-*`) | the fabric golden CSVs are `blackhole_p150_x4`. Whether QB2 reports cluster type `P150_X4` is *(unverified)* |
| LoudBox (P150×8) | BH | 8 | `bh_loudbox` | `bh_lb_C45`, `bh_lb_C67`, `bh_lb_120_A45`, `bh_lb_120_A456`, `bh_lb_single` | the study host a09u32 is one. Allowed nodes per memory: a07u28, a08u28, a04u24 |
| BH Galaxy | BH | 32 | `bh_galaxy`, `bh_galaxy_perf`; exabox `bh_sc1`, `bh_sc1_high_power` | `bh_sc36_6`, `bh_sc36_6_110a_pod*`, `bh_sc36_6_110a_lp` (user rule: 110-a nodes only, ≤ 4 concurrent study allocations) | a02u14 cannot map ring/torus |
| BH multi-Galaxy | BH | 64-128 | `bh_sc4`, `bh_sc4_high_power` (quad); `bh_sc16` (no test uses it) | pods `bh_sc36_6_110a_pod3/6/7/8/9` (4 Galaxies each); `bh_110a_sc16` (16, 6 h limit) | **multi-Galaxy harness: the separate agent** |
| N300 | WH | 2 | `wh_n300_civ2`, `wh_n300` | none | |
| T3K (LLMBox) | WH | 8 | `wh_llmbox`, `wh_llmbox_perf`, `wh_llmbox_civ2` | none | |
| WH Galaxy (6U / TG) | WH | 32 | `wh_galaxy`, `wh_galaxy_perf` | none | |
| WH multi-Galaxy | WH | 64-128 | runner labels `multi-host-galaxy`, `multi-host-glx-4x` (`multi-host-physical.yaml`, `multi-host-deepseekv3.yaml`) | none | **multi-Galaxy** |

The exabox Slurm cluster has **Blackhole only**. Every Wormhole row in this plan is CI-only.

---

## 1. The relevant tests

Each table lists:
- **ID**: used by the manifest (§4);
- **Kind**: C = coverage (functional), P = perf;
- **CI**: the workflow, then the entry `name` in its test list (the GitHub job is that name plus ` [sku]`);
- **≤ min**: the CI timeout.

Commands are under each table, run from `TT_METAL_HOME` with a Release build that has tests and python bindings.

### 1.0 Before any device

| ID | Test | Kind | Where | ≤ min | Why |
| --- | --- | --- | --- | ---: | --- |
| X0-build | Build every arch's kernels that include the NoC API; Quasar sim tests | C | CI merge gate (`merge-gate.yaml`); `sanity-tests.yaml` with `run-sim` (SKUs `sim_quasar*`) | | MAINLINE_PLAN §6.4: the API commit can break Quasar compiles |
| X0-bytes | Offline byte-compare of main's router images, before vs after commits 1-4 | C | study tooling: `fabric_exp/consolidation/offline_build.py` + bytecmp | | main's router must be unchanged by the shared commits |
| X0-cpu | CPU-only fabric tests (control plane, mesh graph, mock clusters, pipeline placement) | C | `tm-fabric-tests.yaml` → `fabric_cpu_only_unit_tests.yaml` (14 entries, `cpu_medium`) | 10-30 each | host builder / arg changes. Slurm: `cpu_only` partition (1 h limit) |

```bash
# X0-cpu (all groups; each CI entry runs a subset with --group)
./tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh --group all
```

### 1.1 L0: fabric directly

**Single card (BH P150, WH N150/N300): NoC API consumers only.**

| ID | Test | Kind | Arch / SKU | CI | ≤ min |
| --- | --- | --- | --- | --- | ---: |
| S1-dm | `unit_tests_data_movement`, slow-dispatch data movement, NoC debugging | C | BH `bh_p150b_civ2`; WH `wh_n150_civ2`, `wh_n300_civ2` | `runtime-unit-tests.yaml` → `runtime_data_movement`, `runtime_sd_data_movement`, `runtime_noc_debugging` | 15 |
| S1-ttnn-dm | ttnn data-movement post-commit | C | same | `tt-metal-l2-nightly.yaml` → `ttnn nightly data_movement tests (post-commit suite)` | 10 |
| N300-fab | `run_cpp_fabric_tests.sh` | C | WH `wh_n300_civ2` | `merge-gate.yaml` → `scaleout unit tests`; also `tm-fabric-tests.yaml` → `WH N300 scaleout unit tests` | 15 |

```bash
./build/test/tt_metal/unit_tests_data_movement --gtest_filter='-*NIGHTLY_*:*NocEstimator*'
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_data_movement
./build/test/tt_metal/unit_tests_noc_debugging
pytest tests/ttnn/unit_tests/operations/data_movement -xv -m "not disable_fast_runtime_mode" --timeout=300
./tests/scripts/run_cpp_fabric_tests.sh            # N300
```

**BH LoudBox (P150×8) and QuietBox 2.** Same commands on both; CI fans out per SKU.

| ID | Test | Kind | SKUs | CI | ≤ min |
| --- | --- | --- | --- | --- | ---: |
| B0-1d | `Fabric1D*Fixture.*` | C | LB, QB2 | `tm-fabric-tests.yaml` → `BH fabric 1D unit tests` | 15 |
| B0-2d | `Fabric2D*Fixture.*` + `FabricTrafficGeneratorKernelIntegrationTest.*` | C | LB, QB2 | → `BH fabric 2D fixture unit tests` | 15 |
| B0-1e | the fast set, also one ERISC (`TT_METAL_DISABLE_MULTI_AERISC=1`) | C | LB, QB2 | `sanity-tests.yaml` → `fabric fast unit tests (1D, 2D, single erisc)`; `blackhole-e2e-tests.yaml` (same name) | 15 |
| B0-mux | `*FabricMuxV2*Fixture.*` | C | LB, QB2 | → `BH fabric mux v2 unit tests` | 5 |
| B0-udm | `Fabric2DUDMModeFixture.*`, `NightlyFabric2DUDMModeFixture.*`, `unit_tests_ttnn_udm` | C | LB, QB2 | → `BH fabric UDM mode unit tests`, `BH ttnn UDM unit tests` | 15 + 10 |
| B0-trim | `*ChannelTrimming*`, two ERISCs and one | C | LB, QB2 | → `BH fabric channel trimming tests` | 10 |
| B0-health | `test_system_health` | C | LB, QB2 | → `BH fabric system health tests` | 15 |
| B0-infra | sanity_common, sanity_at_least_2x2, vc2_at_least_2x2, code profiling yamls | C | LB, QB2 | → `BH fabric infra unit tests` | 15 |
| B0-stab | `test_fabric_stability.yaml --filter 'name.*_short'` | C | LB, QB2 | → `BH fabric stability nightly tests` | 30 |
| B0-night | `NightlyFabric*Fixture.*` (includes the inline-write and fused-atomic fixtures the study validates) | C | LB, QB2 | not in CI for BH boxes (T3K and Galaxy run it) | ~30 |
| B0-eth | `unit_tests_eth` (fast, slow dispatch, one ERISC) | C | LB | `runtime-integration-tests.yaml` → `bh_multicard_eth` | 15 |
| B0-bench | `test_fabric_ubench_at_least_2x2_mesh.yaml` (BW + latency) | **P** | LB, QB2 | `blackhole-e2e-tests.yaml` → `fabric sanity benchmark` | 15 |
| B0-muxbw | mux v2 throughput (golden CSV per arch) | **P** | QB2 | `tm-fabric-tests.yaml` (perf) → `Fabric ubench - Mux V2 Throughput` | 45 |
| B0-study | the study's validation yamls (`fabric_exp/v4/val_ring_p150.yaml`, `val_torus_y_p150.yaml`, `val_mux.yaml`, `val_nbr.yaml`) | C | LB | not on main (MAINLINE_PLAN commit 8 moves them) | ~25 |

```bash
F=./build/test/tt_metal/tt_fabric/fabric_unit_tests; T=./build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric
Y=./tests/tt_metal/tt_fabric/test_infra/test_yamls
$F --gtest_filter="Fabric1D*Fixture.*"
$F --gtest_filter="Fabric2D*Fixture.*"; $F --gtest_filter="FabricTrafficGeneratorKernelIntegrationTest.*"
TT_METAL_DISABLE_MULTI_AERISC=1 $F --gtest_filter="Fabric1D*Fixture.*"
$F --gtest_filter="*FabricMuxV2*Fixture.*"
$F --gtest_filter="Fabric2DUDMModeFixture.*"; $F --gtest_filter="NightlyFabric2DUDMModeFixture.*"; ./build/test/ttnn/unit_tests_ttnn_udm
$F --gtest_filter="*ChannelTrimming*"; TT_METAL_DISABLE_MULTI_AERISC=1 $F --gtest_filter="*ChannelTrimming*"
./build/test/tt_metal/tt_fabric/test_system_health
$T --test_config $Y/test_fabric_sanity_common.yaml
$T --test_config $Y/test_fabric_sanity_at_least_2x2_mesh.yaml
$T --test_config $Y/test_fabric_vc2_at_least_2x2_mesh.yaml
TT_FABRIC_PROFILE_RX_CH_FWD=1 $T --test_config $Y/test_fabric_code_profiling.yaml
TT_METAL_OPERATION_TIMEOUT_SECONDS=0 $T --test_config $Y/test_fabric_stability.yaml --filter 'name.*_short'
$F --gtest_filter="NightlyFabric*Fixture.*"
./build/test/tt_metal/unit_tests_eth; TT_METAL_SLOW_DISPATCH_MODE=true ./build/test/tt_metal/unit_tests_eth; TT_METAL_DISABLE_MULTI_AERISC=1 ./build/test/tt_metal/unit_tests_eth
$T --test_config $Y/test_fabric_ubench_at_least_2x2_mesh.yaml                  # perf
TT_METAL_DEVICE_PROFILER=0 pytest -svv tests/tt_metal/microbenchmarks/ethernet/test_fabric_mux_v2_throughput.py   # perf
```

**BH Galaxy.**

| ID | Test | Kind | CI | ≤ min |
| --- | --- | --- | --- | ---: |
| G0-night | `NightlyFabric*Fixture.*` | C | `galaxy-tests.yaml` (e2e) → `Galaxy Fabric unit tests` | 30 |
| G0-torus-n | `test_fabric_2d_torus_nightly.yaml` | C | → `Galaxy Fabric 2D Torus Nightly Tests` | 37 |
| G0-torus-s | `test_fabric_2d_torus_stability.yaml` (full) | C | → `BH Galaxy Fabric Torus Stability Tests` | 20 |
| G0-health | `run_cluster_validation` + torus stability `_short` | C | `galaxy-sanity.yaml` (health, `blackhole: true`) → `Galaxy Blackhole health` | 10 |
| G0-mm | 2×2 multi-mesh sanity and trimming, inter-mesh over Z and over N/S/E/W (4 entries); per-mesh heterogeneous (4 MGDs); split 4x4+2x4s and 2x8+2x4s stability, with and without mesh pass-through | C | → `Galaxy Fabric 2x2 multi-mesh ...`, `... per-mesh heterogeneous ...`, `Galaxy Fabric Multi-Mesh ...` | 8-15 each |
| G0-sock | socket pipeline cross-tray; D2D socket sync | C | → `BH Galaxy Socket Pipeline ...`, `BH Galaxy D2D socket sync ...` | 10 + 3 |
| G0-eth | `unit_tests_eth` | C | `runtime-integration-tests.yaml` → `galaxy_eth` | 15 |
| G0-fix | `Fabric1D*` / `Fabric2D*` / `*FabricMuxV2*` fixtures | C | not in CI for BH Galaxy (WH Galaxy runs them) | ~30 |
| G0-bench | `test_fabric_ubench_2d_torus_6U_galaxy.yaml` | **P** | not in CI. No BH Galaxy golden CSV exists | ~30 *(unverified)* |

```bash
$F --gtest_filter="NightlyFabric*Fixture.*"
$T --test_config tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_2d_torus_nightly.yaml
TT_METAL_OPERATION_TIMEOUT_SECONDS=0 $T --test_config tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_2d_torus_stability.yaml
./build/tools/scaleout/run_cluster_validation --cabling-descriptor-path tools/tests/scaleout/cabling_descriptors/bh_galaxy_xy_torus.textproto --hard-fail --send-traffic --num-iterations 1
# G0-mm, one example (the others swap the MGD and yaml as in tests/pipeline_reorg/galaxy_e2e_tests.yaml):
MGD=tests/tt_metal/tt_fabric/custom_mesh_descriptors/single_galaxy_2x2_ring_4stage_ring_mesh_graph_descriptor.textproto
TT_METAL_OPERATION_TIMEOUT_SECONDS=0 tt-run --mesh-graph-descriptor $MGD --hosts "$(hostname)" --mpi-args "--allow-run-as-root" \
  $T --test_config tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_multi_mesh_sanity_common.yaml
pytest models/demos/deepseek_v3_d_p/tests/test_d2d_socket_sync.py -k "fabric2d-8x4" -vvv --tb=short
$T --test_config tests/tt_metal/tt_fabric/test_infra/test_yamls/test_fabric_ubench_2d_torus_6U_galaxy.yaml       # perf
```

**BH multi-Galaxy (SC4 quad). Marked for the multi-Galaxy harness.**

| ID | Test | Kind | CI | ≤ min |
| --- | --- | --- | --- | ---: |
| MG0-phys | physical discovery, health, torus stability `_short`, DC neighbour exchange (4×32) | C | `fabric-multihost-exabox.yaml` → `SC4 physical discovery, health, stability, and DC validation` | 40 |
| MG0-express | express links 32×4, 4×32, quad 8×4, dual 16×4, 8×4+24×4: sanity + multicast source-inject (10 entries) | C | → `Fabric express links ...` | 10-15 each |
| MG0-hetero | per-mesh heterogeneous config | C | → `Fabric per-mesh heterogeneous-config sanity (SC4)` | 20 |

The commands use `ttrun.py` with `--hosts`, `/etc/ttop/hostfile` and `ens5f0np0` (see `tests/pipeline_reorg/fabric_multihost_exabox_tests.yaml`).

**Wormhole (CI only).** Main's router in the shared-commit phase; the new router after MAINLINE_PLAN commit 7.

| ID | Test | Kind | System | CI | ≤ min |
| --- | --- | --- | --- | --- | ---: |
| W0-t3k | `run_t3000_ttfabric_tests`: control plane, 1D/2D/mux fixtures, BW-telemetry runs, sanity yamls, ubench, code profiling | C+P | T3K | `sanity-tests.yaml` → `t3k fabric tests`; `tm-fabric-tests.yaml` (same, uploads BW/latency) | 30 |
| W0-t3k-udm | UDM fixtures + `unit_tests_ttnn_udm` | C | T3K | `tm-fabric-tests.yaml` → `t3k UDM tests` | 30 |
| W0-t3k-night | `NightlyFabric*Fixture.*`; `test_addrgen_write`; full `test_fabric_stability.yaml` | C | T3K | `t3000-tests.yaml` (e2e) → `t3k_fabric_unit_tests`, `t3k_addrgen_write_tests`, `t3k_fabric_stability_tests` | 30 / 10 / 120 |
| W0-t3k-bench | ubench at_least_2x2 (BW + latency vs `golden_*_wormhole_b0_t3k.csv`); mux BW; mux v2; P2P unicast `ci_run_unicast.sh` | **P** | T3K perf | `tm-fabric-tests.yaml` → `T3K ubench - ...`, `Fabric Collectives ubench - Point to Point` | 45 each |
| W0-6u | `6U fabric quick`: cluster validation, 1D/2D/mux fixtures (8×4 and 1×32), sanity, torus deadlock stability, ubench quick | C+P | WH Galaxy | `tm-fabric-tests.yaml` → `6U fabric quick` | 25 |
| W0-6u-mp | multi-process: big-mesh / multi-mesh sanity, sockets, quad 2×4 cyclic / acyclic | C | WH Galaxy | → `6U multi-process` | 25 |
| W0-tg | control plane TG, 2D fixtures slow dispatch, mux | C | WH Galaxy | `galaxy-tests.yaml` (unit) → `Galaxy Fabric tests`, `Galaxy Multi-Process tests` | 5 + 15 |
| W0-tg-e2e | `NightlyFabric*`; 2D torus nightly; multi-mesh stability | C | WH Galaxy | `galaxy-tests.yaml` (e2e) | 25-39 |
| W0-tg-stress | `test_fabric_stability.yaml` `_short` | C | WH Galaxy perf | `galaxy-stress-tests.yaml` (Sat 03:00) | 100 |
| W0-mg | dual / quad WH Galaxy, dual T3K unit tests | C | WH multi-Galaxy | `multi-host-physical.yaml` | **multi-Galaxy** |

### 1.2 L1: CCL ops on fabric

| ID | Tests | Kind | System | CI | ≤ min |
| --- | --- | --- | --- | --- | ---: |
| B1-apc | `tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/all_post_commit` (all_gather, reduce_scatter, all_reduce 2D + turning, all_broadcast, all_to_all, dispatch / combine, deepseek CCL ops) | C | LB, QB2 | `sanity-tests.yaml` → `ttnn ops fast unit tests (ccl, DRAM prefetcher)`; `blackhole-e2e-tests.yaml` (same) | 15 |
| B1-night | `.../blackhole_CI/box/nightly` (minimal RS async, matmul+RS, slot / batch invariance, MoE compute, reduce_to_root, 2D variants); then `.../box/watcher` with the watcher on | C | QB2 (`ops_unit_tests`), LB + QB2 (`fabric_tests`) | `tt-metal-l2-nightly.yaml` → `ccl nightly tests`; `tm-fabric-tests.yaml` → `BH ccl nightly tests` | 60 / 30 |
| B1-qb2 | high-BW all-gather QB2 perf; ring indexer; SDPA ring (fabric-based) | C+**P** | QB2 | `sanity-tests.yaml` → `TTNN multi-device CCL, SDPA PERF, and indexer tests (QB2 only)` | 40 |
| B1-p2p | `point_to_point`; host-device send/recv | C | P150, N150, N300 | `tt-metal-l2-nightly.yaml` → `ttnn nightly point_to_point tests (post-commit suite)`, `blackhole host to device smoke tests` | 5-10 |
| B1-dp | disaggregated-prefill op unit tests (dispatch / combine over fabric, ring joint MLA, D2D streams) | C | LB, QB2, BH Galaxy, P150 | `tt-metal-l2-nightly.yaml` → `Disaggregated prefill op unit tests (LB / QB2 / BH Galaxy / P150)` | 30 / 10 / 40 / 4 |
| B1-sock | `distributed_unit_tests --gtest_filter='*HDSocketFixture*'` | C | LB | `runtime-integration-tests.yaml` → `bh_multicard_sockets` | 10 |
| G1-night | `.../blackhole_CI/galaxy/galaxy_nightly` (strided all-gather + matmul, all-to-all generic, dispatch / combine, RS, broadcast) | C | BH Galaxy | `tt-metal-l2-nightly.yaml` → `BH Galaxy CCL tests` | 40 |
| G1-iso | multi-user isolation (single 32×1, tp2 16×2, tray 4×8): all-gather nightly subsets | C | BH Galaxy | `galaxy-tests.yaml` → `Multi-user isolation ... blackhole` | 45 each |
| G1-mh | `run_multi_host_ccl_tests.sh` with `MESH_DEVICE=SINGLE_BH` (8×4, no MPI) | C | BH Galaxy | not in CI *(unverified that every test has an `_8x4` case)* | ~30 |
| MG1-mh | the same with `MESH_DEVICE=DUAL_BH` (16×4) / `QUAD_BH` (32×4) and `HOSTS=...` | C | BH multi-Galaxy | not in CI; **multi-Galaxy** | ~60 |
| MG1-hbw | `high_bw_all_gather CCL perf + accuracy` | C+**P** | exabox single Galaxy (`bh_sc1_high_power`) | `blaze-models-prefill-tests.yaml` (test-type `ccl` / `sdpa_perf`) | 20 |
| W1-n300 | `test_ccl_l1_small_semaphores`, `test_minimals::test_rms_fuse_n300`, `test_ccl_mux_l1` | C | N300 | `tt-metal-l2-nightly.yaml` → `ttnn n300 ccl tests` | 10 |
| W1-t3k-s | `run_t3000_ccl_tests` (16 selected all_gather / RS / all_reduce / P2P / broadcast / dispatch / combine / neighbor_pad / AG+matmul cases) | C | T3K | `sanity-tests.yaml` → `t3k_ttnn_ccl_tests` | 15 |
| W1-t3k | `pytest tests/nightly/t3000/ccl`; `pytest tests/nightly/t3000/2d_ccl` | C | T3K | `t3000-tests.yaml` (e2e) → `t3k_ccl_tests`, `t3k_ccl_2d_tests` | 90 + 10 |
| W1-tg-s | Galaxy CCL smoke (one test per op) | C | WH Galaxy | `galaxy-sanity.yaml` → `Fabric CCL tests` | 20 |
| W1-tg | `pytest tests/nightly/tg/ccl --ignore tests/nightly/tg/ccl/moe`; MoE with 1×16 and 1×8 torus MGDs | C | WH Galaxy | `tt-metal-l2-nightly.yaml` → `Galaxy CCL tests`, `Galaxy MoE tests` | 60 + 60 |
| W1-tg-perf | `models/demos/llama3_70b_galaxy/tests/tg_perf_unit_tests/test_ccl_async_perf_TG_{llama,qwen}.py` | **P** | WH Galaxy | not found in any CI list *(unverified whether something else runs it)* | |

```bash
pytest tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/all_post_commit
pytest tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly
TT_METAL_WATCHER=30 TT_METAL_WATCHER_APPEND=1 TT_METAL_WATCHER_NOINLINE=1 TT_METAL_WATCHER_DISABLE_ETH=1 \
  pytest tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/watcher -xv --timeout=600
pytest tests/ttnn/unit_tests/operations/experimental/test_high_bw_all_gather.py::test_high_bw_all_gather_quietbox_ci_perf -q   # QB2 perf
pytest tests/ttnn/unit_tests/operations/point_to_point -xv -m "not disable_fast_runtime_mode" --timeout=300
pytest tests/ttnn/unit_tests/operations/ccl/blackhole_CI/galaxy/galaxy_nightly
MESH_DEVICE=SINGLE_BH bash tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/multi_host/scripts/run_multi_host_ccl_tests.sh
MESH_DEVICE=QUAD_BH HOSTS=h1,h2,h3,h4 bash tests/ttnn/unit_tests/operations/ccl/blackhole_CI/exabox/multi_host/scripts/run_multi_host_ccl_tests.sh
source tests/scripts/t3000/run_t3000_unit_tests.sh && run_t3000_ccl_tests          # T3K
pytest tests/nightly/t3000/ccl; pytest tests/nightly/t3000/2d_ccl                  # T3K
pytest tests/nightly/tg/ccl --ignore tests/nightly/tg/ccl/moe                     # WH Galaxy
```

The BH box CCL "perf" test (`blackhole_CI/box/nightly/test_ccl_perf.py`) skips itself on main ("Issue #29658"). It is no
perf gate today.

### 1.3 L2: production and key models

**Selection rule.**
- Tier 1 models (`models/model_ci_tiers.md`) on a multi-chip system: release blockers, which use fabric CCLs.
- DeepSeek-V3 (decode on WH Galaxy, B1 on BH Galaxy).
- The exabox prefill pipeline (Blaze) and the disaggregated-prefill legs.
- One single-card model (Llama-3.1-8B on P150 / N150) for the NoC API change.

**Weights and runners.**
- Most need `/mnt/MLPerf/...` or `/mnt/weka/...`. a09u32 has `/mnt/weka` and `/mnt/models`, no `/mnt/MLPerf`.
- Galaxy and QB2 Slurm nodes *(unverified)*.
- vLLM rows need the vLLM install that CI's images carry: run them in CI.

| ID | Model / test | Kind | System | CI (workflow → entry) | ≤ min |
| --- | --- | --- | --- | --- | ---: |
| S2-l8b | Llama-3.1-8B e2e: token matching, eval-32, prefetcher | C+P | P150 / N150 | `models-t1-e2e-tests.yaml` (model `llama3.1-8b`) → `Llama 3.1-8B e2e tests` | 11 |
| B2-q36 | Qwen3.6-27B e2e (traced 128, determinism, accuracy, vision) | C+P | QB2 | t1 e2e → `Qwen3.6-27B e2e tests` | 45 |
| B2-q36moe | Qwen3.6-35B-A3B e2e | C+P | QB2 | t1 e2e → `Qwen3.6-35B-A3B e2e tests` | 45 |
| B2-oss | GPT-OSS 120B e2e (`gpt_oss/demo/text_demo.py`) | C+P | QB2, BH Galaxy, WH Galaxy | t1 e2e → `GPT-OSS 120B e2e tests` | 35 / 30 / 30 |
| B2-gemma | Gemma-4-12B / 26B-A4B / 31B e2e | C+P | QB2 | t1 e2e → `Gemma-4-* e2e tests` | 13-26 |
| B2-l70perf | Llama-3.3-70B / Qwen3-32B / Llama-3.1-8B device perf (`ttt_compat/tests/test_device_perf.py`) | **P** | QB2 | `models-t1-device-perf-tests.yaml` → `... device perf tests` | 20-30 |
| B2-flux | Flux.1-schnell / dev e2e (DiT CCLs) | C+P | QB2 | t1 e2e → `TT-DiT Flux.1 ... e2e tests` | 20 |
| B2-vllm-lb | vLLM Gemma4-31B tp=8; Llama-3.1-8B sampling (P150×8) | C+P | LB | `vllm-model-tests.yaml` → `Gemma4-31B tp=8`, `Llama-3.1-8B-Instruct with sampling-tests (P150x8)` | 50 / 75 |
| B2-dp | disaggregated prefill accuracy and perf, LB and QB2 (Kimi-K3 MLA / KDA perf) | C+P | LB, QB2 | `blackhole-e2e-tests.yaml` (model `disaggregated_prefill`) | 10-117 |
| G2-qwen | Qwen3-32B Galaxy e2e (`llama3_70b_galaxy/demo/text_qwen_demo.py`) | C+P | BH Galaxy (WH Galaxy too) | t1 e2e → `Qwen3-32B e2e tests (Galaxy)` | 15 |
| G2-mm3 | MiniMax-M3 unit tests (8×4) | C | BH Galaxy | `models-t1-unit-tests.yaml` → `MiniMax-M3 unit tests (BH Galaxy 8x4)` | 38 |
| G2-dsb1 | DeepSeek V3 B1 decoder sweep vs reference; B1 unit tests incl. `Fabric2DHostIO`, `MoreFabric` | C | BH Galaxy | `demo-sp-release.yaml` → `Single Galaxy - Decoder Sweep vs reference (DeepSeek V3 B1)` and the `bh_galaxy_DeepSeek_B1_Unit_Tests_*` entries | 15-75 |
| G2-vllm | vLLM Qwen3-32B device | C+P | BH Galaxy | `vllm-model-tests.yaml` → `Qwen3-32B device (BH Galaxy)` | 65 |
| SC1-blaze | Blaze prefill: perf gates (Kimi-K2.7 chunked prefill perf, GLM-5.3 perf, Gemma4 D_P perf gate, MiniMax-M3 perf, DSV4-Flash perf) and accuracy legs | C+**P** | exabox single Galaxy (`bh_sc1`, `bh_sc1_high_power`) | `blaze-models-prefill-tests.yaml`, dispatch `test-type` = `gate` / `ccl` / stage names | 10-289 per entry |
| SC1-dit | Wan2.2 single-Galaxy perf; LTX-2.3 e2e | C+P | exabox single Galaxy | t1 e2e → `TT-DiT Wan2.2-* single galaxy perf tests`, `TT-DiT LTX-2.3 * e2e tests` | 35-75 |
| MG2-wan | Wan2.2 T2V / I2V quad e2e and perf | C+**P** | SC4 | t1 e2e → `TT-DiT Wan2.2-* multihost quad e2e / perf tests` | 35-75; **multi-Galaxy** |
| MG2-blaze | GLM-5.3 / Kimi-K3 disaggregated prefill SC4 legs | C+**P** | SC4 | `blaze-models-prefill-tests.yaml` (`disagg_prefill`) | 20-200; **multi-Galaxy** |
| W2-l70 | Llama-3.3-70B Galaxy e2e (`llama3_70b_galaxy/demo/text_demo.py` token matching + eval-1) | C+P | WH Galaxy perf | t1 e2e → `Llama 3.3-70B e2e tests (Galaxy)` | 30 |
| W2-l8dp | Llama-3.1-8B DP=32 | C+P | WH Galaxy perf | t1 e2e → `Llama 3.1-8B data-parallel e2e tests (Galaxy)` | 20 |
| W2-ds | DeepSeek-V3 decode stress; decode device perf; fused-op device perf | C+**P** | WH Galaxy perf | t1 e2e → `DeepSeek-V3 decode stress e2e tests (Galaxy)`; `models-t1-device-perf-tests.yaml` → `DeepSeek-V3 * device perf tests (Galaxy)` | 15 / 20 / 45 |
| W2-l70perf | Llama-70B TG decoder / prefill device perf (`run_tg_model_perf_tests.sh`) | **P** | WH Galaxy | not found in `pipeline_reorg` *(unverified)* | |
| W2-t3k | GPT-OSS 120B unit tests; Llama-3.1-8B, Qwen3-32B e2e; Llama-3.3-70B seqlen sweep | C+P | T3K | t1 / t2 unit, e2e, sweep | 13-25 |
| W2-mg | DeepSeek-V3 dual / quad unit, module, demo, teacher-forced | C+P | WH multi-Galaxy | `multi-host-deepseekv3.yaml` | **multi-Galaxy** |

```bash
# B2-q36 (QB2)
export HF_HOME=/mnt/MLPerf/huggingface HF_MODEL=Qwen/Qwen3.6-27B TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/Qwen/Qwen3.6-27B MESH_DEVICE=P150x4 QWEN36_SPEC=0
pytest --timeout 1200 models/demos/blackhole/qwen36/demo/text_demo.py -k "traced_128 and not traced_128k"
pytest --timeout 1200 models/demos/blackhole/qwen36/demo/text_demo.py -k "determinism_128"
pytest --timeout 1200 models/demos/blackhole/qwen36/demo/text_demo.py -k "accuracy_512"
# B2-oss (QB2 / Galaxy)
export HF_MODEL=openai/gpt-oss-120b TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/openai--gpt-oss-120b/ TT_MM_THROTTLE_PERF=2
pytest models/demos/gpt_oss/demo/text_demo.py --timeout 1500 --skip-model-load
# B2-l70perf (QB2)
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000 HF_MODEL=meta-llama/Llama-3.3-70B-Instruct TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/meta-llama/Llama-3.3-70B-Instruct
pytest --timeout 900 models/ttt_compat/tests/test_device_perf.py -k "prefill-llama3_70b-2-131072-2-2-1-1-False"
pytest --timeout 900 models/ttt_compat/tests/test_device_perf.py -k "decode-llama3_70b-2-131072-2-10-1-1-False"
# G2-qwen (BH Galaxy)
export HF_MODEL=Qwen/Qwen3-32B TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/Qwen/Qwen3-32B QWEN_BH_PREFETCHER=1
pytest --timeout 900 models/demos/llama3_70b_galaxy/demo/text_qwen_demo.py -k "ci-token-matching"
# W2-l70 (WH Galaxy, CI)
export HF_MODEL=meta-llama/Llama-3.3-70B-Instruct TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/meta-llama/Llama-3.3-70B-Instruct
pytest --timeout 1000 models/demos/llama3_70b_galaxy/demo/text_demo.py -k "ci-token-matching"
pytest --timeout 1000 models/demos/llama3_70b_galaxy/demo/text_demo.py -k "ci-eval-1"
# W2-ds (WH Galaxy, CI)
export DEEPSEEK_V3_HF_MODEL=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized-stacked DEEPSEEK_V3_CACHE=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI MESH_DEVICE=TG
pytest models/demos/deepseek_v3/demo/test_demo.py -k "tg_stress or tg_upr8" --timeout 900
```

The rows whose commands are long scripts (G2-dsb1, SC1-*, MG2-*, the vLLM rows) are taken verbatim from their entries
in `tests/pipeline_reorg/{demo_sp_release,blaze_models_prefill,models_e2e,vllm_model}_tests.yaml`. The runner should
read the `cmd` from there rather than copy it (§4.1, `from_ci`).

### 1.4 Which commit needs which groups

| MAINLINE_PLAN commit | Mode | Must pass | Nice to have |
| --- | --- | --- | --- |
| 1 NoC API | main | X0-build (all arches incl. Quasar sim), X0-bytes, S1-dm, S1-ttnn-dm, N300-fab, B0-1d/2d on one BH box, W0-t3k | S2-l8b |
| 2 Ethernet API (main's router call sites) | main | X0-bytes (main's router may change only where the call sites did), B0-* on LB, W0-t3k, W0-6u, N300-fab | B1-apc, W1-t3k-s |
| 3 fabric library + ct args | main | X0-bytes (main's router byte-identical), B0-*, W0-t3k, W0-6u | B1-apc |
| 4 host | main | X0-cpu, B0-*, G0-night, W0-t3k, W0-6u, W0-tg. Named-arg diff of generated kernel caches: only additions | B1, W1 smoke |
| 5 router (opt-in) | new **and** main | all BH L0 (B0, G0), L1 (B1, G1), L2 smoke (B2-q36, B2-oss, G2-qwen); main mode unchanged | MG0, MG1 |
| 6 slot policy | new | B0-bench, G0-bench, B1-qb2, B2-l70perf, SC1-blaze perf gates | MG2 perf |
| 7 Wormhole bring-up | new | W0-*, W1-*, then W2 smoke | W0-mg |
| before default-on (later) | new | everything above, both arches, CI on a default-on branch | stress (W0-tg-stress, B0-stab full) |

---

## 2. CI quick reference: what to dispatch on a branch

| Workflow (`.github/workflows/`) | Schedule on main | Runs | Dispatch knobs |
| --- | --- | --- | --- |
| `tm-fabric-tests.yaml` | daily 07:00 | `fabric_tests.yaml` (N300, T3K, 6U, LB, QB2) + `fabric_perf_tests.yaml` + CPU-only | per-SKU booleans (`run-bh_loudbox`, ...) |
| `sanity-tests.yaml` | every 2 h (cron runs Blackhole only) | fabric sanity, BH multi-card sanity (CCL apc, QB2 perf), T3K sanity (fabric, CCL) | `run-wormhole`, `run-blackhole`, `run-sim`, ... |
| `blackhole-e2e-tests.yaml` | every 8 h | `blackhole_e2e_tests.yaml` (fabric infra / bench / UDM / fast / stability, CCL apc, disaggregated prefill) | `system-type`, `model` |
| `galaxy-tests.yaml` | daily 00:00 | Galaxy unit / integration / e2e (fabric nightly, torus, multi-mesh, sockets) / multi-user isolation | `run-e2e-tests`, `e2e-test-selection` |
| `galaxy-sanity.yaml` | every 4 h | Galaxy CCL smoke + health (WH; BH via `blackhole: true`) | `wormhole`, `blackhole` |
| `galaxy-stress-tests.yaml` | Sat 03:00 | WH Galaxy stability | |
| `t3000-tests.yaml` | daily 06:00 | T3K unit / integration / e2e (CCL nightly, fabric nightly, stability) | `run-e2e-tests`, model filters |
| `tt-metal-l2-nightly.yaml` | daily 06:00 | `ops_unit_tests.yaml` (CCL nightly QB2, BH and WH Galaxy CCL, N300 CCL, P2P, data movement, disaggregated-prefill ops) | |
| `runtime-integration-tests.yaml` | daily 05:00 | `unit_tests_eth` (LB, Galaxy), HD sockets | |
| `fabric-multihost-exabox.yaml` | daily 05:00 | SC4 (quad BH Galaxy) | none |
| `models-t1-e2e-tests.yaml` / `-unit-` / `-device-perf-` / `-sweep-` | daily 02:00 / 04:00, 20:00 / 07:00 / Wed+Sat | Tier-1 model rows | `model`, `sku` |
| `blaze-models-prefill-tests.yaml` | daily 05:00 | exabox SC1 / SC4 prefill | `test-type` (stages / tags) |
| `demo-sp-release.yaml` | daily 01:00 | DeepSeek V3 B1 on BH Galaxy | |
| `vllm-model-tests.yaml` | daily 00:00 | vLLM rows | |
| `multi-host-physical.yaml`, `multi-host-deepseekv3.yaml` | 04:00 and every 3 h / 04:00 | WH dual / quad Galaxy, dual T3K | |

To compare against main without spending runners, use the scheduled main runs of the same workflows: their artifacts
and the perf dashboards (§3).

---

## 3. Perf baselines on main

### 3.1 What produces perf numbers

| Source | Groups | Output | Pass / regression rule in the repo |
| --- | --- | --- | --- |
| `test_tt_fabric` benchmark and latency yamls (`test_fabric_ubench_*`) | B0-bench, G0-bench, W0-t3k(-bench), W0-6u | `generated/fabric/bandwidth_{results,summary_results,summary_results_<arch>_upload,diff,comparison_statistics}_<arch>.csv`, the same for `latency_` | Golden per cluster type in `tests/tt_metal/tt_fabric/test_infra/golden/golden_{bandwidth,latency}_summary_<arch>_<cluster>.csv`. Per-row `tolerance_percent` (BH P150×4 bandwidth: 10 / 12 / 22 %; latency 8 %). **abs(diff) > tolerance fails the test, in either direction.** Rows without golden are `NO_GOLDEN`. Goldens exist for `blackhole_p150_x4`, `wormhole_b0_t3k`, `wormhole_b0_galaxy` only: no LoudBox (P150×8), no BH Galaxy |
| Fabric mux microbenches | B0-muxbw, W0-t3k-bench | pytest log; golden `tests/tt_metal/microbenchmarks/ethernet/fabric_mux_v2_throughput_golden_<arch>.csv`, `fabric_mux_bandwidth_golden.csv` | `BENCHMARK_REL_TOLERANCE = 0.05` (mux v2); `FABRIC_MUX_V2_THROUGHPUT_GOLDEN` overrides the golden path |
| P2P unicast collective bench | W0-t3k-bench | `generated/test_reports/unicast/*.csv` | p50 targets in `ci_run_unicast.sh`, `--tolerance-pct 10` |
| High-BW all-gather QB2 perf | B1-qb2 | pytest log | in-test target |
| Model e2e (tt_transformers, demos) | S2, B2, G2, W2 | `generated/benchmark_data/*.json`; validated by `.github/scripts/utils/validate_perf_targets.py --path-profile repo-root --sku <sku>` | `models/model_targets.yaml`: per-metric `<metric>_tolerance`, else `tolerance`, else `DEFAULT_PERF_TOLERANCE = 0.15`. Drift **in either direction** fails ("too fast flags an outdated target"). Some demos also check in-test `ci_target_*` dicts |
| Model device perf | B2-l70perf, W2-ds | pytest asserts; tracy CSVs under `generated/profiler/` | in-test expected values; markers `models_device_performance_bare_metal`, `model_perf_t3000`, `model_perf_tg` |
| Blaze prefill perf gates | SC1-blaze, MG2-blaze | per-stage logs | in-test gates |

### 3.2 How to collect the baseline now

1. **Freeze one main commit** (the origin/main the port starts from) and build it once per system class: the
   `main@<sha>` tree of §4.3.
2. **Fabric bandwidth / latency.**
   - Run B0-bench on a LoudBox and on a QB2.
   - Run G0-bench on a BH Galaxy. Not in CI: this is the only BH Galaxy fabric perf baseline there will be.
   - Repeat each 3 times in one allocation and keep every `generated/fabric/*.csv`.
   - Wormhole: take the nightly artifacts of `tm-fabric-tests.yaml` (T3K perf, 6U quick) for the same sha, or
     dispatch it once.
3. **Microbenches:** B0-muxbw on QB2, 3 passes.
4. **Models:**
   - Run each L2 row with perf once on main, keeping `generated/benchmark_data/` and the validator output.
   - For CI-only rows, record the run id of the scheduled main run at that sha. The model dashboards hold history too.
5. **Store** under `/data/aagarwal/regress/baselines/<sha>/<system>/<group>/pass<k>/` with a `meta.json` (node, sha,
   router mode, env, start / end).

### 3.3 What counts as a regression (main vs new)

- **Coverage:** a test that passes on main and fails on new is a regression. It is re-run once to separate flakes.
  Only the comparison against main's same-run result counts, not CI history.
- **Fabric BW / latency (A/B in one job):**
  - per row, diff = new / main − 1;
  - noise = the max pass-to-pass spread of main in that job;
  - a regression is diff < −max(noise, 1 %) on bandwidth (latency: > +max(noise, 1 %));
  - the golden `tolerance_percent` (10-22 %) is far too coarse for router A/Bs. It only decides whether CI's golden
    check will still pass.
  - Note for landing: a faster router also **fails** the golden check (absolute diff). Any row more than its tolerance
    faster needs re-goldening in the same PR that flips the default.
- **Models:**
  - CI's rule is the `model_targets.yaml` tolerance (default 15 %).
  - For the A/B, compare new to main directly. A regression is beyond the main pass-to-pass spread, or more than 2 %
    on decode t/s/u or TTFT, whichever is larger *(threshold proposed, not from the repo)*.
- **Telemetry.**
  - On main, `test_tt_fabric` turns bandwidth telemetry on in benchmark / latency mode (`test_tt_fabric.cpp:206`), so
    CI numbers are telemetry-on.
  - The study's `TT_FABRIC_NO_BW_TELEMETRY` does not exist on main. Both trees run the same way, so the A/B is fair.
    The numbers are not the study's telemetry-off figures.

---

## 4. Batching: Slurm design (no scripts yet)

### 4.1 Manifest format

One yaml, version-controlled next to this plan (`fabric_exp/consolidation/regress/manifest.yaml`). Groups are
Slurm-sized: one allocation, one system, ≤ ~2.5 h.

```yaml
version: 1
router_env:                     # the one place that maps a router mode to env (MAINLINE_PLAN §1.6)
  main: {}
  new: {TT_METAL_FABRIC_ROLE_ROUTER: "1"}     # study tree instead: {TT_FABRIC_KERNEL_VERSION: "2"} + swap
systems:
  bh_loudbox: {partitions: [bh_lb_C45, bh_lb_C67, bh_lb_120_A45], nodes: [bh-lb-120-a07u28, bh-lb-120-a08u28, bh-lb-120-a04u24]}
  bh_qb2:     {partitions: [bh_qb2]}                                   # ask the user before first use
  bh_galaxy:  {partitions: [bh_sc36_6, bh_sc36_6_110a_lp], node_regex: "bh-glx-110-a.*", exclude: [bh-glx-110-a02u14], max_concurrent: 4}
  bh_quad:    {partitions: [bh_sc36_6_110a_pod7, bh_sc36_6_110a_pod9], hosts: 4, harness: multi_galaxy}   # other agent
  cpu:        {partitions: [cpu_only], time_limit: "01:00:00"}
groups:
  - id: LB-L0-a
    level: L0
    system: bh_loudbox
    kind: coverage
    routers: [main, new]        # runs the steps once per router, same node
    timeout_min: 150
    weights: none               # none | mlperf | weka: preflight checks the mount
    steps:
      - {id: B0-1d,   cmd: './build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric1D*Fixture.*"', timeout_min: 15, result: gtest}
      - {id: B0-trim, cmd: '...', timeout_min: 10, result: gtest}
      - {id: B1-apc,  from_ci: {list: blackhole_e2e_tests.yaml, name: "ttnn ops fast unit tests (ccl, DRAM prefetcher)"}, timeout_min: 15, result: pytest}
  - id: LB-P-fabric
    level: L0
    system: bh_loudbox
    kind: perf
    routers: [main, new]
    passes: [[main, new], [new, main]]   # A/B order, both on one node
    timeout_min: 90
    steps:
      - {id: B0-bench, cmd: './build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric --test_config ...ubench_at_least_2x2_mesh.yaml', timeout_min: 15, result: fabric_csv}
```

Field rules:
- `from_ci` pulls `cmd` from a `pipeline_reorg` entry by name, so commands never drift from CI.
- `result` picks the collector:
  - `gtest`: `--gtest_output=xml:`;
  - `pytest`: `--junitxml`;
  - `fabric_csv`: `generated/fabric/*.csv`;
  - `benchmark_json`: `generated/benchmark_data` + the target validator;
  - `log_marker`: a regex, e.g. `All tests completed successfully`.
- `multi_galaxy: true` groups are only listed here. The multi-Galaxy agent's harness runs them.

### 4.2 Runner interface

```text
regress plan    --manifest M [--levels L0,L1] [--systems bh_loudbox,bh_galaxy] [--routers main,new]
                -> prints the groups, the sbatch each would get, total node-hours; no submission
regress run     --manifest M --group G --tree T --router {main|new} --label L [--steps a,b] [--dry-run]
                -> on the allocated node: preflight, then each step under its timeout; writes results/L/G/...
regress submit  --manifest M --groups G1,G2 --trees main=T1,new=T2 --label L [--max-concurrent N]
                -> one sbatch per group (both routers inside it for A/B groups); honours per-system caps
regress collect L            -> copies results/L back to a09u32 (fabric_exp/results/regress/L)
regress compare --base L_main --test L_new [--group G] -> report (markdown + csv)
```

**Preflight** (on the node, before any step):
- the hostname is allowed for the system (110-a rule; LB node list);
- no other user's test / python process holds the devices (the Galaxy sbatch's `ps` check; stop on KMD-lock wait);
- the weights mount is present when `weights` is set;
- the tree's binaries exist;
- the router env resolves;
- for `new` in the study tree, the swap manifest is clean.

**On a hang** (step timeout), as the study's `run_cap` does:
- capture (`tt-triage` and the NIU / PC dump scripts), kill, reset;
- run the host check (`lb2_hostcheck.sh` on LoudBoxes);
- mark the step `HANG`, and stop the group if the device is gone (LB rule: "stop a job at the first run with no
  result line").

**Results layout:**

```text
results/<label>/<group>/<router>/pass<k>/<step>/
    cmd.txt  env.txt  log.txt  status.json   # rc, PASS/FAIL/HANG/SKIP, start, end, node, tree sha, router
    junit.xml | fabric/*.csv | benchmark_data/*.json
results/<label>/<group>/summary.json
```

**`compare` does:**
- coverage diff per test case: main PASS → new FAIL, then flaky re-run results;
- the fabric CSV join on `(test_name, ftype, ntype, topology, num_devices, num_links, packet_size)` for bandwidth,
  and on the latency key columns for latency, with the §3.3 noise rule;
- the model JSON diff per metric, plus the target validator verdict per tree;
- one table per group, worst rows first.

An optional shortcut for systems with no golden: write main's summary CSV as a temporary golden into the **run copy**
of the tree. `test_tt_fabric` then reports per-row PASS / FAIL and a geomean speedup itself. That reuses the repo's
comparator, but the golden tolerance is coarse.

### 4.3 Trees

- **Two trees per comparison,** built once each, Release, with tests and python bindings:
  - `main@<sha>`: origin/main;
  - `new@<sha>`: the port branch.

  Build with `./build_metal.sh --build-tests` and `./create_venv.sh`; L1 and L2 need the venv.
- **Placement.**
  - Galaxy nodes see `/data/aagarwal` but not `/home/aagarwal`, so trees live on `/data/aagarwal/regress/trees/`.
    The job sets `LD_LIBRARY_PATH` into the build, as `frs_glx.sbatch` does, and hard-link-copies (`cp -al`) per
    job: 4-10 min on NFS.
  - LoudBox scripts expect `/home/aagarwal/tt-metal`, so LB jobs stage a tarball as the lb2 harness does.
- **Study tree.** For the study tree (before the port exists), `new` = the study tree with `swap.py apply` of the
  variant inside the job, and `main` = the same tree unswapped. The existing harnesses already do this.
- **The venv is the open part.** The study's `/data` trees are lean (no `python_env`). Whether a venv built on a09u32
  runs unchanged on Galaxy nodes is *(unverified)*: build it on a Galaxy node once and check.

### 4.4 The batches

Wall time is the sum of the CI timeouts (upper bound) plus ~10 min setup. "×2" means both routers in one allocation.

| Batch | System / partition | Groups (§1 IDs) | Kind | Wall (upper bound) | Notes |
| --- | --- | --- | --- | ---: | --- |
| X-cpu | `cpu_only` | X0-cpu (split by `--group` into ≤ 1 h jobs) | C | 4-6 × 1 h | host-arg changes |
| LB-L0-a | LoudBox | B0-1d, B0-2d, B0-1e, B0-mux, B0-trim, B0-udm | C ×2 | 2 × 75 | |
| LB-L0-b | LoudBox | B0-health, B0-infra, B0-stab, B0-night, B0-eth, B0-study | C ×2 | 2 × 115 | B0-study only where the yamls exist |
| LB-P | LoudBox | B0-bench, 2 passes reversed | P | 4 × 15 + setup ≈ 75 | one node for the whole A/B |
| LB-L1 | LoudBox | B1-apc, B1-night (incl. watcher), B1-dp (LB), B1-sock | C ×2 | 2 × 90 | |
| LB-L2 | LoudBox | B2-dp perf (LB), B2-vllm-lb (prefer CI) | C+P | per row | weights *(unverified on Slurm LBs)* |
| QB-L0 | QB2 | B0-* fast set + B0-infra + B0-stab | C ×2 | 2 × 75 | `bh_qb2`: ask first |
| QB-P | QB2 | B0-bench (P150×4 golden), B0-muxbw, B1-qb2 | P | ~150 | |
| QB-L1 | QB2 | B1-apc, B1-night, B1-dp (QB2) | C ×2 | 2 × 85 | |
| QB-L2 | QB2 | B2-q36, B2-q36moe, B2-oss, B2-gemma, B2-l70perf, B2-flux (split into 2-3 jobs) | C+P | ~4 h per router | weights |
| G-L0-a | BH Galaxy (110-a) | G0-night, G0-torus-n, G0-health, G0-fix | C ×2 | 2 × 110 | |
| G-L0-b | BH Galaxy | G0-torus-s, G0-mm (all), G0-sock, G0-eth | C ×2 | 2 × 120 | `tt-run` single host |
| G-P | BH Galaxy | G0-bench, 2 passes reversed | P | ~150 | exclude a02u14 |
| G-L1 | BH Galaxy | G1-night, G1-mh (SINGLE_BH), B1-dp (Galaxy), G1-iso (one shape) | C ×2 | 2 × 150 | |
| G-L2 | BH Galaxy | G2-qwen, B2-oss (Galaxy), G2-mm3, G2-dsb1 (split) | C+P | ~3 h per router | weights |
| SC1-L2 | exabox single Galaxy | SC1-blaze gates, SC1-dit, MG1-hbw | C+P | per stage | CI (`blaze-models-prefill-tests.yaml`) is simpler; Slurm only for A/B perf |
| **MG-L0/L1/L2** | quad pods | MG0-*, MG1-mh (DUAL / QUAD), MG2-wan, MG2-blaze | C+P | — | **multi-Galaxy harness (other agent)** |
| WH-* | CI only | N300-fab, W0-*, W1-*, W2-* | C+P | — | dispatch on the branch (§2); baseline = main's scheduled runs |

**Order:** X-cpu and LB-L0 first (cheap; they catch most breakage), then QB / G L0, then L1, then perf, then L2.
Stop the line on any L0 regression.

**Concurrency limits:**
- ≤ 4 Galaxy allocations (all agents);
- LB A/B jobs keep both routers on one node;
- never cancel running jobs; reprioritize only the not-started ones.

---

## 5. Unverified, and open questions for the user

- QB2's reported cluster type: whether the `blackhole_p150_x4` goldens apply to it.
- P300 single card: no active CI SKU, no Slurm partition. Is a QB2 limited to one card acceptable?
- Weight mounts (`/mnt/MLPerf`, `/mnt/weka`) on the Galaxy and QB2 Slurm nodes, and the venv working there.
- Whether `run_multi_host_ccl_tests.sh` has `_8x4` cases for every op (SINGLE_BH).
- What, if anything, runs `run_tg_model_perf_tests.sh` and the TG CCL perf unit tests (not in any `pipeline_reorg`
  list).
- **Partitions:**
  - `bh_qb2` and `bh_lb_single` are not on the user's allowed list.
  - The 110-a rule covers Galaxies only.
- The regression thresholds in §3.3 (1 % fabric, 2 % models) are proposals. The study's own rule ("accept 1-2
  cycles/packet") is in cycles, not percent.
