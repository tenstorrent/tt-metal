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

# Falcon40B 4 chip decode tests
pytest models/demos/falcon40b/tests/test_falcon_decoder.py::test_FalconDecoder_inference[BFLOAT8_B-SHARDED-falcon_40b-layer_0-decode_batch32-4chips-enable_program_cache]
pytest models/demos/falcon40b/tests/test_falcon_end_to_end.py::test_FalconCausalLM_end_to_end_with_program_cache[BFLOAT8_B-SHARDED-falcon_40b-layers_1-decode_batch32-4chips-enable_program_cache]

# Falcon40B 8 chip decode tests
pytest models/demos/falcon40b/tests/test_falcon_decoder.py::test_FalconDecoder_inference[BFLOAT8_B-SHARDED-falcon_40b-layer_0-decode_batch32-8chips-enable_program_cache]
pytest models/demos/falcon40b/tests/test_falcon_end_to_end.py::test_FalconCausalLM_end_to_end_with_program_cache[BFLOAT8_B-SHARDED-falcon_40b-layers_1-decode_batch32-8chips-enable_program_cache]

# Falcon40B 8 chip prefill tests; we need 8x8 grid size
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest models/demos/falcon40b/tests/ci/test_falcon_end_to_end_t3000_prefill.py
