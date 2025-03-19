#!/bin/bash

set -eo pipefail

if [[ "$ARCH_NAME" == "wormhole_b0" ]]; then
  export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
fi

env pytest -n auto tests/ttnn/integration_tests -m "not models_performance_bare_metal and not models_device_performance_bare_metal" -k "not stable_diffusion"
