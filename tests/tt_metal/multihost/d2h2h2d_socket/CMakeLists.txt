# The kernels here are JIT-compiled at runtime from source: every leg benchmark under
# tests/tt_metal/distributed compiles this directory in as TT_DIRECT_KERNEL_DIR and opens
# kernels/*.cpp through CreateKernel. Nothing in here is built by CMake.
