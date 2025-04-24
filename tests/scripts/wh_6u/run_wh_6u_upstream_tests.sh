#!/usr/bin/env bash
set -euo pipefail

echo "[upstream-tests] Running minimal model unit tests"
pytest tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py
pytest tests/ttnn/unit_tests/operations/test_prefetcher_TG.py
# pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_matmul_1d_gather_in0.py::test_matmul_1d_ring_llama_perf Failing with a no program cache found error
pytest tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py
# pytest tests/ttnn/unit_tests/operations/ccl/test_minimals.py hang???

echo "[upstream-tests] running metalium section. Note that skips should be treated as failures"
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestQuantaGalaxyControlPlaneInit"
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCardFixture.*"
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCardProgramFixture.*"
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCardBufferFixture.ShardedBufferLarge*ReadWrites"
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"
