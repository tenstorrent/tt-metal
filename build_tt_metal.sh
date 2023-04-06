#!/bin/bash

set -eo pipefail

if [[ "$TT_METAL_ENV" ]]; then
  echo "build_tt_metal.sh: TT_METAL_ENV set to $TT_METAL_ENV"
fi

make clean
make build
