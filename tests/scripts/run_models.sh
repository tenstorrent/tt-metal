#!/bin/bash

##############################################################################
##############################################################################
##############################################################################
##############################################################################

# Do not add to this script anymore. Instead, take a look at tests/scripts/nightly

##############################################################################
##############################################################################
##############################################################################
##############################################################################
set -eo pipefail
if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi
cd $TT_METAL_HOME
export PYTHONPATH=$TT_METAL_HOME

# THIS DOES NOT RUN LOL leaving this here for record keeping
if [[ $ARCH_NAME == "wormhole" ]]; then
  env pytest tests/ttnn/integration_tests/unet
  env pytest tests/ttnn/integration_tests/stable_diffusion
fi
