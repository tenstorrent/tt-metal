#/bin/bash

# set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi
fail=0

echo "Running nightly tests for WH B0 only"
env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto tests/nightly/wh_b0_only_eth ; fail+=$?
env pytest -n auto tests/nightly/wh_b0_only ; fail+=$?

if [[ $fail -ne 0 ]]; then
  exit 1
fi
