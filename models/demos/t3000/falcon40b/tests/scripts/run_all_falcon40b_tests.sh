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

# prefill required 8x8 core grids
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml

# Run all component tests
pytest models/demos/t3000/falcon40b/tests/test_falcon_mlp.py -k test_FalconMLP_inference
pytest models/demos/t3000/falcon40b/tests/test_falcon_attention.py -k test_FalconAttention_inference
pytest models/demos/t3000/falcon40b/tests/test_falcon_decoder.py -k run_test_FalconDecoder_inference
pytest models/demos/t3000/falcon40b/tests/test_falcon_model.py -k test_FalconModel_inference
pytest models/demos/t3000/falcon40b/tests/test_falcon_causallm.py -k test_FalconCausalLM_inference
pytest models/demos/t3000/falcon40b/tests/test_falcon_end_to_end.py -k test_FalconCausalLM_end_to_end_with_program_cache
