#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

# working on both
# Falcon7B
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings -q -s --input-method=cli --cli-input="YOUR PROMPT GOES HERE!"  models/demos/wormhole/falcon7b/demo_wormhole.py::test_demo[wormhole_b0-True-user_input0-1-default_mode_1024_stochastic]
# llama3.1-8B
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/wormhole/llama31_8b/demo/demo.py --timeout 600
# Mistral7B
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/wormhole/mistral7b/demo/demo.py --timeout 420
