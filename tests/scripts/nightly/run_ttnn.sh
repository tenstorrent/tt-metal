#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

echo "Running ttnn nightly tests for GS only"

env pytest tests/ttnn/integration_tests -m "not models_performance_bare_metal and not models_device_performance_bare_metal"
