# Sourced inside the tt-xla-ird container to run the moe_compute repro against the
# model's real runtime tt-metal (the tt-mlir-nested checkout, 68e82deb155 build).
RT=/home/mvasiljev/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal
source /home/mvasiljev/tt-xla/venv/bin/activate
export TT_METAL_HOME="$RT"
export TT_METAL_RUNTIME_ROOT="$RT"
export PYTHONPATH="$RT/ttnn:$RT/tools/"
export ARCH_NAME=wormhole_b0
export USE_TORCH_XLA=0
export ACCELERATE_USE_XLA=false
cd /home/mvasiljev/tt-metal/moe_compute_combine_deadlock_repro
