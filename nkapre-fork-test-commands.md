# nkapreTT Fork — Hardware Parity Test Commands

Tests for parity between nkapreTT fork and upstream tt-metal.
No fabric CPU-only tests. Requires real TT hardware for all suites below.

## Setup

```bash
export TT_METAL_HOME=/path/to/nkapre-metal-fork
export ARCH_NAME=wormhole_b0   # or blackhole
export PYTHONPATH=$TT_METAL_HOME
cd $TT_METAL_HOME
```

---

## 1. TTNN Tests (single card, requires 1 TT device)

### C++ unit tests
```bash
./build/test/ttnn/unit_tests_ttnn
./build/test/ttnn/unit_tests_ttnn_tensor
./build/test/ttnn/unit_tests_ttnn_ccl
./build/test/ttnn/unit_tests_ttnn_ccl_multi_tensor
./build/test/ttnn/unit_tests_ttnn_ccl_ops
./build/test/ttnn/unit_tests_ttnn_accessor
./build/test/ttnn/test_ccl_multi_cq_multi_device
```

### Python unit tests
```bash
pytest tests/ttnn/unit_tests/ -xvvv
```

---

## 2. T3000 Tests (requires T3K = 8-chip Wormhole, single host)

### TT-Metal distributed tests
```bash
./build/test/tt_metal/distributed/distributed_unit_tests

# TT_VISIBLE_DEVICES multiprocess
./tests/tt_metal/distributed/multiprocess/run_visible_devices_mp_tests.sh

# Eth kernel tests (slow dispatch)
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth \
  --gtest_filter="MeshDeviceFixture.ActiveEthKernelsDirectSendAllConnectedChips"
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth \
  --gtest_filter="MeshDeviceFixture.ActiveEthKernelsSendInterleavedBufferAllConnectedChips"
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth \
  --gtest_filter="MeshDeviceFixture.ActiveEthKernelsDirectRingGatherAllChips"
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth \
  --gtest_filter="MeshDeviceFixture.ActiveEthKernelsInterleavedRingGatherAllChips"

# Dispatch tests
TT_METAL_ENABLE_REMOTE_CHIP=1 ./build/test/tt_metal/unit_tests_dispatch \
  --gtest_filter="CommandQueueSingleCard*Fixture.*"
./build/test/tt_metal/unit_tests_dispatch \
  --gtest_filter="CommandQueueMultiDevice*Fixture.*"
TT_METAL_ENABLE_REMOTE_CHIP=1 ./build/test/tt_metal/unit_tests_dispatch \
  --gtest_filter="UnitMeshCQSingleDevice*Fixture.*"
./build/test/tt_metal/unit_tests_dispatch \
  --gtest_filter="UnitMeshCQMultiDevice*Fixture.*"

# Debug tools
./build/test/tt_metal/unit_tests_debug_tools \
  --gtest_filter="DPrintMeshFixture.*:MeshWatcherFixture.*"

# Programming examples
./build/programming_examples/distributed/distributed_program_dispatch
./build/programming_examples/distributed/distributed_buffer_rw
./build/programming_examples/distributed/distributed_eltwise_add
./build/programming_examples/distributed/distributed_trace_and_events
```

### TT-Fabric tests on T3K (simulator / ttsim)

Mock cluster descriptors and WH multichip ttsim — no real T3K hardware required.

```bash
# One-time WH multichip sim setup
mkdir -p ~/sim_wh_multichip
cp /path/to/libttsim_wh.so ~/sim_wh_multichip/libttsim.so
cp $TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml ~/sim_wh_multichip/soc_descriptor.yaml
export TT_METAL_SIMULATOR=~/sim_wh_multichip/libttsim.so
export TT_METAL_SIMULATOR_HOME=~/sim_wh_multichip
export TT_METAL_DRAM_BACKED_CQ=1
export TT_METAL_SIMULATOR_CQ_WAIT_CLOCKS=10000

# Control plane (mock cluster only — simulator unset)
env -u TT_METAL_SIMULATOR -u TT_METAL_SIMULATOR_HOME ARCH_NAME=wormhole_b0 \
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
  --gtest_filter="ControlPlaneFixture.*T3k*"
env -u TT_METAL_SIMULATOR -u TT_METAL_SIMULATOR_HOME ARCH_NAME=wormhole_b0 \
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
  --gtest_filter="T3kCustomMeshGraphControlPlaneTests*"
env -u TT_METAL_SIMULATOR -u TT_METAL_SIMULATOR_HOME ARCH_NAME=wormhole_b0 \
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
  --gtest_filter="T3k*MeshGraphFabric2DDynamicTests*"

# Worker/EDM datapath (WH multichip ttsim)
ARCH_NAME=wormhole_b0 \
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
  ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
  --gtest_filter="*WorkerFabricEdmDatapath*:*EdmFabric*"

# 1x8 mesh on T3K with 2D fabric (mock cluster only)
env -u TT_METAL_SIMULATOR -u TT_METAL_SIMULATOR_HOME ARCH_NAME=wormhole_b0 \
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
  TT_METAL_SLOW_DISPATCH_MODE=1 \
  TT_MESH_GRAPH_DESC_PATH=tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_1x8_mesh_graph_descriptor.textproto \
  ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
  --gtest_filter="*Fabric2DFixture.TestUnicast*"

# Fabric 2D/1D fixture tests (WH multichip ttsim; with BW telemetry)
ARCH_NAME=wormhole_b0 \
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
  TT_METAL_FABRIC_BW_TELEMETRY=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
  --gtest_filter="Fabric2D*Fixture.*"
ARCH_NAME=wormhole_b0 \
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
  TT_METAL_FABRIC_BW_TELEMETRY=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
  --gtest_filter="Fabric1D*Fixture.*"
ARCH_NAME=wormhole_b0 \
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
  ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"
ARCH_NAME=wormhole_b0 \
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
  ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric1D*Fixture.*"
env -u TT_METAL_SIMULATOR -u TT_METAL_SIMULATOR_HOME ARCH_NAME=wormhole_b0 \
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests \
  --gtest_filter="T3k*MeshGraphFabric2DDynamicTests*"

# Fabric sanity microbenchmarks (WH multichip ttsim)
ARCH_NAME=wormhole_b0 \
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
  ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric \
  --test_config $TT_METAL_HOME/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_sanity_common.yaml
ARCH_NAME=wormhole_b0 \
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
  ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric \
  --test_config $TT_METAL_HOME/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_sanity_at_least_2x2_mesh.yaml
ARCH_NAME=wormhole_b0 \
  TT_METAL_MOCK_CLUSTER_DESC_PATH=tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/t3k_cluster_desc.yaml \
  ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric \
  --test_config $TT_METAL_HOME/tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_ubench_at_least_2x2_mesh.yaml
```

### TTNN distributed tests (T3K)
```bash
./build/test/ttnn/unit_tests_ttnn
./build/test/ttnn/unit_tests_ttnn_udm

pytest tests/ttnn/unit_tests/operations/transformers/test_prefetcher.py::test_run_prefetcher_post_commit_multi_device
pytest tests/ttnn/distributed/test_tensor_parallel_example_T3000.py
pytest tests/ttnn/distributed/test_data_parallel_example.py
pytest tests/ttnn/distributed/test_hybrid_data_tensor_parallel_example_T3000.py
```

### Multiprocess tests (T3K, requires MPI / tt-run)
```bash
# 2x2 mesh
tt-run --mpi-args "--allow-run-as-root --oversubscribe" \
  --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml \
  ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric \
  --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_t3k_2x2.yaml

tt-run --mpi-args "--allow-run-as-root --oversubscribe" \
  --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml \
  ./build/test/tt_metal/multi_host_fabric_tests

tt-run --mpi-args "--allow-run-as-root --oversubscribe" \
  --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml \
  ./build/test/tt_metal/test_mesh_socket_main \
  --test_config tests/tt_metal/multihost/fabric_tests/mesh_socket_t3k_2x2.yaml

# 2x4 Big-Mesh
tt-run --mpi-args "--allow-run-as-root --oversubscribe" \
  --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml \
  build/test/tt_metal/distributed/multiprocess/distributed_multiprocess_tests \
  --gtest_filter="*BigMeshDualRankTest2x4*"

tt-run --mpi-args "--allow-run-as-root --oversubscribe" \
  --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml \
  build/test/tt_metal/distributed/multiprocess/distributed_multiprocess_tests \
  --gtest_filter="*BigMeshDualRankMeshShapeSweep*"

# TTNN 2x2 multiprocess
tt-run --mpi-args "--allow-run-as-root --oversubscribe" \
  --rank-binding tests/tt_metal/distributed/config/2x2_multiprocess_rank_bindings.yaml \
  ./build/test/ttnn/multiprocess/unit_tests_dual_rank_2x2

# TTNN 2x4 multiprocess
tt-run --mpi-args "--allow-run-as-root --oversubscribe" \
  --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml \
  build/test/ttnn/multiprocess/unit_tests_dual_rank_2x4

tt-run --mpi-args "--allow-run-as-root --oversubscribe" \
  --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml \
  build/test/ttnn/unit_tests_ttnn --gtest_filter="*LaunchOperation*"

tt-run --mpi-args "--allow-run-as-root --oversubscribe" \
  --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml \
  pytest -svv tests/ttnn/distributed/test_data_parallel_example.py

tt-run --mpi-args "--allow-run-as-root --oversubscribe" \
  --rank-binding tests/tt_metal/distributed/config/2x4_multiprocess_rank_bindings.yaml \
  pytest -svv tests/ttnn/distributed/test_submesh_not_spanning_all_ranks_T3000.py
```

---

## 3. Single Galaxy Single-Process Tests (requires 32-chip WH Galaxy, single host)

### TTNN unit tests on Galaxy
```bash
pytest tests/ttnn/distributed/test_data_parallel_example_TG.py --timeout=900
pytest tests/ttnn/distributed/test_multidevice_TG.py --timeout=900
pytest tests/ttnn/unit_tests/base_functionality/test_multi_device_trace_TG.py --timeout=900
```

---

## 4. LLK Tests (simulator, no hardware needed)

Paths match **craq-sim2** official weekly driver (`scripts/llk-weekly.sh`), not multichip fabric staging.

**Pins (green on yyzeon):**

| Component | Ref |
|-----------|-----|
| craq-sim | **`main` @ b8c07375** (`/data/rsong/craq-sim`) — avoid multichip sim for LLK weekly |
| tt-metal | **1ce6c3e8144** @ `/data/rsong/tt-metal-llk-pin` (`LLK_TT_METAL_HOME`) |

### One-time setup

```bash
export CRAQ_SIM=/data/rsong/craq-sim
export LLK_TT_METAL_HOME=/data/rsong/tt-metal-llk-pin   # craq-sim CI pin
export TT_METAL_HOME=$LLK_TT_METAL_HOME

# SFPI (once): symlink from tt-metal3 if pin worktree lacks it
ln -sfn /data/rsong/tt-metal3/tt_metal/tt-llk/tests/sfpi \
  $TT_METAL_HOME/tt_metal/tt-llk/tests/sfpi

# Python venv (once; can reuse tt-metal3 venv)
python3 -m venv /data/rsong/cache/tt-llk-venv
/data/rsong/cache/tt-llk-venv/bin/pip install -r $TT_METAL_HOME/tt_metal/tt-llk/tests/requirements.txt -q

# Build sim (once per craq-sim pull)
cd $CRAQ_SIM/src && python3 ../make.py -j16 _out/release_bh/libttsim.so _out/release_wh/libttsim.so
```

### Sample gate → full weekly (Slurm, recommended)

```bash
# ~25 failure-bucket cases/arch first; full weekly only if sample exits 0
bash /data/rsong/cache/slurm-llk/submit-llk-weekly-gated.sh
# Logs: /data/rsong/cache/slurm-llk/llk-sample-<jid>.out, llk-weekly-<jid>.out
```

Env (set automatically by `llk-pytest-sweep.sh` / sbatch):

- `TT_METAL_SIMULATOR` = staged `release_{bh,wh}/libttsim.so` + soc_descriptor from `$TT_METAL_HOME`
- `TT_METAL_DISABLE_SFPLOADMACRO=0`, `DISABLE_SFPLOADMACRO=0`
- `PYTHONPATH=$CRAQ_SIM/scripts/llk_pytest_shim:$TT_METAL_HOME`
- `TT_LLK_ARTEFACTS_DIR`, `TT_LLK_LOCK_DIR` — isolated per run
- **No** `TT_METAL_DRAM_BACKED_CQ`, **no** multichip mock yaml for LLK weekly

### Local full weekly (both arches, ~2–3h @ -j16)

```bash
export TT_METAL_HOME=/data/rsong/tt-metal-llk-pin
export CRAQ_SIM=/data/rsong/craq-sim2
cd $CRAQ_SIM
scripts/llk-weekly.sh both --workers 16 --timeout 300
```

Target: BH **29,227 pass / 0 fail**, WH **32,515 pass / 0 fail**.

### Legacy multichip path (fabric parity only — NOT for LLK weekly gate)

The old `llk-bh.sbatch` / `llk-wh.sbatch` multichip + DRAM CQ path had ~299 BH failures (job 12485). Do not use for LLK weekly sign-off.

### Results log

See `$TT_METAL_HOME/llk-sim-test-run.log` for job IDs, pass/fail counts, log paths, and known blockers.

---

## Summary

| Suite | Hardware |
|-------|---------|
| TTNN single-card | 1 TT card |
| T3000 metal/ttnn (dist) | T3K (8-chip WH, single host) |
| T3000 fabric | ttsim (WH multichip + T3K mock cluster desc) |
| T3000 multiprocess | T3K + MPI (tt-run) |
| Single Galaxy unit | 32-chip WH Galaxy, single host |
