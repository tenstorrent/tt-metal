PROGRAMMING_EXAMPLES_TESTDIR = $(OUT)/programming_examples
PROGRAMMING_EXAMPLES_OBJDIR = $(OBJDIR)/programming_examples

PROGRAMMING_EXAMPLES_INCLUDES = $(COMMON_INCLUDES)
PROGRAMMING_EXAMPLES_LDFLAGS = -ltt_metal -ldl -lstdc++fs -pthread -lyaml-cpp -lm

include $(TT_METAL_HOME)/tt_metal/programming_examples/hello_world_compute_kernel/module.mk
include $(TT_METAL_HOME)/tt_metal/programming_examples/hello_world_datamovement_kernel/module.mk
include $(TT_METAL_HOME)/tt_metal/programming_examples/hello_world_datatypes_kernel/module.mk
include $(TT_METAL_HOME)/tt_metal/programming_examples/add_2_integers_in_riscv/module.mk
include $(TT_METAL_HOME)/tt_metal/programming_examples/add_2_integers_in_compute/module.mk
include $(TT_METAL_HOME)/tt_metal/programming_examples/loopback/module.mk
include $(TT_METAL_HOME)/tt_metal/programming_examples/eltwise_binary/module.mk
include $(TT_METAL_HOME)/tt_metal/programming_examples/eltwise_sfpu/module.mk
include $(TT_METAL_HOME)/tt_metal/programming_examples/profiler/module.mk
include $(TT_METAL_HOME)/tt_metal/programming_examples/matmul_single_core/module.mk
include $(TT_METAL_HOME)/tt_metal/programming_examples/matmul_multi_core/module.mk
include $(TT_METAL_HOME)/tt_metal/programming_examples/matmul_multicore_reuse/module.mk
include $(TT_METAL_HOME)/tt_metal/programming_examples/matmul_multicore_reuse_mcast/module.mk
include $(TT_METAL_HOME)/tt_metal/programming_examples/contributed/module.mk

# Need to depend on set_up_kernels.

PROFILER_TESTS += \
		  programming_examples/profiler/test_custom_cycle_count\
		  programming_examples/profiler/test_full_buffer\
		  programming_examples/profiler/test_multi_op

CONTRIBUTED_EXAMPLES += \
		  programming_examples/contributed/vecadd

programming_examples: programming_examples/hello_world_compute_kernel \
                      programming_examples/hello_world_datamovement_kernel \
                      programming_examples/hello_world_datatypes_kernel \
                      programming_examples/add_2_integers_in_riscv \
                      programming_examples/add_2_integers_in_compute \
                      programming_examples/loopback \
                      programming_examples/eltwise_binary \
                      programming_examples/eltwise_sfpu \
                      programming_examples/matmul_single_core \
                      programming_examples/matmul_multi_core \
                      programming_examples/matmul_multicore_reuse \
                      programming_examples/matmul_multicore_reuse_mcast \
                      $(PROFILER_TESTS) \
                      $(CONTRIBUTED_EXAMPLES)

programming_examples/hello_world_compute_kernel:$(PROGRAMMING_EXAMPLES_TESTDIR)/hello_world_compute_kernel
programming_examples/hello_world_datamovement_kernel:$(PROGRAMMING_EXAMPLES_TESTDIR)/hello_world_datamovement_kernel
programming_examples/hello_world_datatypes_kernel:$(PROGRAMMING_EXAMPLES_TESTDIR)/hello_world_datatypes_kernel
programming_examples/add_2_integers_in_riscv:$(PROGRAMMING_EXAMPLES_TESTDIR)/add_2_integers_in_riscv
programming_examples/add_2_integers_in_compute:$(PROGRAMMING_EXAMPLES_TESTDIR)/add_2_integers_in_compute
programming_examples/loopback: $(PROGRAMMING_EXAMPLES_TESTDIR)/loopback;
programming_examples/eltwise_binary: $(PROGRAMMING_EXAMPLES_TESTDIR)/eltwise_binary;
programming_examples/eltwise_sfpu: $(PROGRAMMING_EXAMPLES_TESTDIR)/eltwise_sfpu;
programming_examples/matmul_single_core: $(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_single_core;
programming_examples/matmul_multi_core: $(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multi_core;
programming_examples/matmul_multicore_reuse: $(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multicore_reuse;
programming_examples/matmul_multicore_reuse_mcast: $(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multicore_reuse_mcast;
programming_examples/profiler/%: $(PROGRAMMING_EXAMPLES_TESTDIR)/profiler/%;
programming_examples/contributed/%: $(PROGRAMMING_EXAMPLES_TESTDIR)/contributed/%;
