#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

source tests/scripts/single_card/run_demos_single_card_wh_common.sh

# Not working on N150, working on N300
unset WH_ARCH_YAML
rm -rf built
pytest -n auto --disable-warnings models/demos/metal_BERT_large_11/demo/demo.py -k batch_7
