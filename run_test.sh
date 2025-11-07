export TT_METAL_RUNTIME_ROOT=/root/tt-metal
export PROFILER_BIN_DIR=$TT_METAL_RUNTIME_ROOT/build/tools/profiler/bin
export PYTHONPATH=/root/tt-metal:/root/tt-metal/ttnn:/root/tt-metal/tools
# export TT_VISIBLE_DEVICES=3

TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_DIR=`pwd` TT_METAL_PROFILER_SYNC=1 \
python -m tracy -r -m  --tracy-tools-folder $PROFILER_BIN_DIR \
pytest tests/ttnn/unit_tests/test_unary_hash_demo.py
exit
