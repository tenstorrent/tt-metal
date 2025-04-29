#!/usr/bin/env bash
set -euo pipefail

echo "[upstream-tests] running metalium section. Note that skips should be treated as failures"
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="ControlPlaneFixture.TestQuantaGalaxyControlPlaneInit"
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCardFixture.*"
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCardProgramFixture.*"
TT_METAL_SKIP_ETH_CORES_WITH_RETRAIN=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCardBufferFixture.ShardedBufferLarge*ReadWrites"
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="Fabric2D*Fixture.*"

echo "[upstream-tests] Running minimal model unit tests"
pytest tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py
pytest tests/ttnn/unit_tests/operations/test_prefetcher_TG.py
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_matmul_1d_gather_in0.py::test_matmul_1d_ring_llama_perf
pytest tests/ttnn/unit_tests/operations/ccl/test_ccl_async_TG_llama.py
# pytest tests/ttnn/unit_tests/operations/ccl/test_minimals.py hang???

if [ -z "${LLAMA_DIR}" ]; then
  echo "Error: LLAMA_DIR environment variable not detected. Please set this environment variable to tell the tests where to find the downloaded Llama weights." >&2
  exit 1
fi

if [ -d "$LLAMA_DIR" ] && [ "$(ls -A $LLAMA_DIR)" ]; then
  echo "[upstream-tests] Llama weights exist, continuing"
else
  echo "[upstream-tests] Error: Llama weights do not seem to exist in $LLAMA_DIR, exiting" >&2
  exit 1
fi

echo "[upstream-tests] Running validation model tests with weights"
pytest models/demos/llama3_subdevices/tests/test_llama_model.py -k "quick"
pytest models/demos/llama3_subdevices/tests/unit_tests/test_llama_model_prefill.py

echo "[upstream-tests] Unsetting LLAMA_DIR to ensure later tests can't use it"
unset LLAMA_DIR
