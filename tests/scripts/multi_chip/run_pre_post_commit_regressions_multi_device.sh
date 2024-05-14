
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

# Falcon40b unit tests; prefill required 8x8 grids
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/t3000/falcon40b/tests/test_falcon_mlp.py
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/t3000/falcon40b/tests/test_falcon_attention.py
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/t3000/falcon40b/tests/test_falcon_decoder.py
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/t3000/falcon40b/tests/test_falcon_causallm.py
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/t3000/falcon40b/tests/ci/test_falcon_end_to_end_1_layer_t3000.py

# Mistral8x7b 8 chip decode tests (env flags set inside the tests)
pytest models/demos/t3000/mixtral8x7b/tests/test_mixtral_attention.py
pytest models/demos/t3000/mixtral8x7b/tests/test_mixtral_mlp.py
pytest models/demos/t3000/mixtral8x7b/tests/test_mixtral_rms_norm.py
pytest models/demos/t3000/mixtral8x7b/tests/test_mixtral_embedding.py
pytest models/demos/t3000/mixtral8x7b/tests/test_mixtral_moe.py
pytest models/demos/t3000/mixtral8x7b/tests/test_mixtral_decoder.py
pytest models/demos/t3000/mixtral8x7b/tests/test_mixtral_model.py::test_mixtral_model_inference[1-1-pcc]

# Falcon7B data parallel tests
pytest models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_mlp.py
pytest models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_attention.py
pytest models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_decoder.py
pytest models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_causallm.py
