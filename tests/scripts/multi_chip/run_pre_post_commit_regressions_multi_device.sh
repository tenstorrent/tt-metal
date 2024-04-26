#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

if [[ -z "$ARCH_NAME" ]]; then
  echo "Must provide ARCH_NAME in environment" 1>&2
  exit 1
fi

TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests --gtest_filter="DeviceFixture.EthKernelsDirectSendAllConnectedChips"
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests --gtest_filter="DeviceFixture.EthKernelsSendInterleavedBufferAllConnectedChips"
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests --gtest_filter="DeviceFixture.EthKernelsDirectRingGatherAllChips"
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests --gtest_filter="DeviceFixture.EthKernelsInterleavedRingGatherAllChips"

TT_METAL_ENABLE_REMOTE_CHIP=1 ./build/test/tt_metal/unit_tests_fast_dispatch --gtest_filter="CommandQueueSingleCardFixture.*"
./build/test/tt_metal/unit_tests_fast_dispatch --gtest_filter="CommandQueueMultiDeviceFixture.*"
./build/test/tt_metal/unit_tests_fast_dispatch --gtest_filter="DPrintFixture.*:WatcherFixture.*"
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_all_gather.py -k post_commit

# ttnn multi-chip apis unit tests
pytest tests/ttnn/unit_tests/test_multi_device.py

# Falcon40B 4 chip decode tests
pytest models/demos/t3000/falcon40b/tests/test_falcon_decoder.py::test_FalconDecoder_inference[BFLOAT8_B-SHARDED-falcon_40b-layer_0-decode_batch32-4chips-enable_program_cache]
pytest models/demos/t3000/falcon40b/tests/test_falcon_end_to_end.py::test_FalconCausalLM_end_to_end_with_program_cache[BFLOAT8_B-SHARDED-falcon_40b-layers_1-decode_batch32-4chips-enable_program_cache]

# Falcon40B 8 chip decode tests
pytest models/demos/t3000/falcon40b/tests/test_falcon_decoder.py::test_FalconDecoder_inference[BFLOAT8_B-SHARDED-falcon_40b-layer_0-decode_batch32-8chips-enable_program_cache]
pytest models/demos/t3000/falcon40b/tests/test_falcon_end_to_end.py::test_FalconCausalLM_end_to_end_with_program_cache[BFLOAT8_B-SHARDED-falcon_40b-layers_1-decode_batch32-8chips-enable_program_cache]

# Falcon40B 8 chip prefill tests; we need 8x8 grid size
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/t3000/falcon40b/tests/ci/test_falcon_end_to_end_t3000_prefill.py

pytest tests/ttnn/unit_tests/test_multi_device_async.py

pytest models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_mlp.py
pytest models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_attention.py
pytest models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_decoder.py
pytest models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_causallm.py
