#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

echo "Running unstable nightly tests for WH B0 only"

SLOW_MATMULS=1 WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml env pytest tests/ttnn/integration_tests/stable_diffusion
