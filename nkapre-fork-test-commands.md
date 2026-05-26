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

```bash
# One-time setup
mkdir -p ~/sim
cp /path/to/libttsim_bh.so ~/sim/
cp $TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml ~/sim/soc_descriptor.yaml

export TT_METAL_SIMULATOR=~/sim/libttsim.so
export TT_METAL_DISABLE_SFPLOADMACRO=1

# If venv not set up yet:
cd $TT_METAL_HOME/tt_metal/third_party/tt_llk/tests
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -q
CHIP_ARCH=blackhole bash setup_testing_env.sh
```

Weekly (all non-nightly, non-quasar, non-perf):
```bash
cd $TT_METAL_HOME/tt_metal/third_party/tt_llk/tests/python_tests
../.venv/bin/python -m pytest \
  -m "not quasar and not nightly and not perf" \
  --run-simulator --forked --timeout=300 -n 4 \
  --junitxml=/tmp/llk-weekly.xml .
```

Nightly (specific files only):
```bash
../.venv/bin/python -m pytest \
  -m "nightly" \
  --run-simulator --forked --timeout=600 -n 4 \
  --junitxml=/tmp/llk-nightly.xml \
  test_unpack_matmul.py \
  test_math_matmul.py \
  test_zzz_eltwise_unary_sfpu.py
```

For Wormhole: swap `libttsim_bh.so` → `libttsim_wh.so` and `blackhole_140_arch.yaml` → `wormhole_b0_80_arch.yaml`.

---

## Summary

| Suite | Hardware |
|-------|---------|
| TTNN single-card | 1 TT card |
| T3000 metal/ttnn (dist) | T3K (8-chip WH, single host) |
| T3000 fabric | ttsim (WH multichip + T3K mock cluster desc) |
| T3000 multiprocess | T3K + MPI (tt-run) |
| Single Galaxy unit | 32-chip WH Galaxy, single host |
