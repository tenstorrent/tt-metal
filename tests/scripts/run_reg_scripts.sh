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

make clean
make build
make tests

source build/python_env/bin/activate
export PYTHONPATH=$TT_METAL_HOME
python -m pip install -r tests/python_api_testing/requirements.txt

env python tests/scripts/run_build_kernels_for_riscv.py -j 16
env python tests/scripts/run_llrt.py --skip-driver-tests
env ./build/test/llrt/test_silicon_driver
env python tests/scripts/run_tt_metal.py

env pytest $TT_METAL_HOME/tests/python_api_testing/unit_testing/ -s
env python tests/python_api_testing/models/bert/bert.py

env pytest tests/python_api_testing/models/t5 -k t5_dense_act_dense
env python tests/python_api_testing/models/t5/t5_layer_norm.py
env python tests/python_api_testing/models/t5/t5_layer_ff.py
env python tests/python_api_testing/models/t5/t5_layer_self_attention.py
env python tests/python_api_testing/models/t5/t5_layer_cross_attention.py

env python tests/python_api_testing/models/synthetic_gradients/batchnorm1d_test.py
env python tests/python_api_testing/models/synthetic_gradients/linear_test.py
env python tests/python_api_testing/models/synthetic_gradients/block_test.py
env python tests/python_api_testing/models/synthetic_gradients/full_inference.py
