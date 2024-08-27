#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

# llama3.1-8B
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/wormhole/llama31_8b/demo/demo_with_prefill.py --timeout 600

# Mistral7B
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/wormhole/mistral7b/demo/demo_with_prefill.py --timeout 420

# Mamba
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings -q -s --input-method=json --input-path='models/demos/wormhole/mamba/demo/prompts.json' models/demos/wormhole/mamba/demo/demo.py --timeout 420

# Falcon7b (perf verification for 128/1024/2048 seq lens and output token verification)
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto --disable-warnings -q -s --input-method=json --input-path='models/demos/falcon7b_common/demo/input_data.json' models/demos/wormhole/falcon7b/demo_wormhole.py
