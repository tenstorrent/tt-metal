#/bin/bash

set -eo pipefail

if [[ -z "$TT_METAL_HOME" ]]; then
  echo "Must provide TT_METAL_HOME in environment" 1>&2
  exit 1
fi

if [ "$TT_METAL_ENV" != "dev" ]; then
  echo "Must set TT_METAL_ENV as dev" 1>&2
  exit 1
fi

cd $TT_METAL_HOME

./build_tt_metal.sh
make tests

source build/python_env/bin/activate
export PYTHONPATH=$TT_METAL_HOME
python -m pip install -r tests/python_api_testing/requirements.txt

./tests/scripts/run_pre_post_commit_regressions.sh

env python tests/scripts/run_tt_metal.py

# Tests tensor and tt_dnn op APIs
./tests/scripts/run_tt_lib_regressions.sh

# Tests profiler module
./tests/scripts/run_profiler_regressions.sh

# Please put model runs in here from now on - thank you
./tests/scripts/run_models.sh
