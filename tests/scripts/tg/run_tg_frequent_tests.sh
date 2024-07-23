
#/bin/bash
set -eo pipefail

run_tg_tests() {
  # Add tests here
  echo "LOG_METAL: running run_tg_frequent_tests"

  pytest -n auto tests/ttnn/multichip_unit_tests/test_multidevice_TG.py --timeout=900 ; fail+=$?
  pytest -n auto models/demos/tg/llama3_70b/tests/test_llama_mlp_galaxy.py --timeout=300 ; fail+=$?
  pytest -n auto models/demos/tg/llama3_70b/tests/test_llama_attention_galaxy.py --timeout=480 ; fail+=$?
  pytest -n auto models/demos/tg/llama3_70b/tests/test_llama_decoder_galaxy.py --timeout=600 ; fail+=$?
}

main() {
  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  # Run all tests
  cd $TT_METAL_HOME
  export PYTHONPATH=$TT_METAL_HOME

  run_tg_tests
}

main "$@"
