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

cd $TT_METAL_HOME
export PYTHONPATH=$TT_METAL_HOME

TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests --gtest_filter="DeviceFixture.EthKernelsDirectSendAllConnectedChips"
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests --gtest_filter="DeviceFixture.EthKernelsSendInterleavedBufferAllConnectedChips"
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests --gtest_filter="DeviceFixture.EthKernelsDirectRingGatherAllChips"
TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests --gtest_filter="DeviceFixture.EthKernelsInterleavedRingGatherAllChips"

TT_METAL_ENABLE_REMOTE_CHIP=1 ./build/test/tt_metal/unit_tests_fast_dispatch --gtest_filter="CommandQueueSingleCardFixture.*"
./build/test/tt_metal/unit_tests_fast_dispatch --gtest_filter="CommandQueueMultiDeviceFixture.*"
pytest tests/tt_eager/python_api_testing/unit_testing/misc/test_all_gather.py -k post_commit
pytest models/demos/falcon40b/tests/test_falcon_decoder.py::test_FalconDecoder_inference[BFLOAT8_B-SHARDED-0.99-0.99-0.99-tiiuae/falcon-40b-instruct-0-prefill_seq32-4]
pytest models/demos/falcon40b/tests/test_falcon_decoder.py::test_FalconDecoder_inference[BFLOAT8_B-SHARDED-0.99-0.99-0.99-tiiuae/falcon-40b-instruct-0-decode_batch32-4]
pytest models/demos/falcon40b/tests/test_falcon_end_to_end.py::test_FalconCausalLM_end_to_end_with_program_cache[BFLOAT8_B-SHARDED-falcon_40b-layers_1-decode_batch32-4]

#pytest models/demos/falcon40b/tests/test_falcon_decoder.py::test_FalconDecoder_inference[BFLOAT8_B-SHARDED-0.99-0.99-0.99-tiiuae/falcon-40b-instruct-0-prefill_seq32-8]
pytest models/demos/falcon40b/tests/test_falcon_decoder.py::test_FalconDecoder_inference[BFLOAT8_B-SHARDED-0.99-0.99-0.99-tiiuae/falcon-40b-instruct-0-decode_batch32-8]
pytest models/demos/falcon40b/tests/test_falcon_end_to_end.py::test_FalconCausalLM_end_to_end_with_program_cache[BFLOAT8_B-SHARDED-falcon_40b-layers_1-decode_batch32-8]
pytest tests/ttnn/unit_tests/test_multi_device.py
