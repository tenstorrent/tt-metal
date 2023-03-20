PROGRAMMING_EXAMPLES_TESTDIR = $(OUT)/programming_examples
PROGRAMMING_EXAMPLES_OBJDIR = $(OBJDIR)/programming_examples

PROGRAMMING_EXAMPLES_INCLUDES = $(COMMON_INCLUDES)
PROGRAMMING_EXAMPLES_LDFLAGS = -ltt_metal_impl -ltt_metal -lllrt -ltt_gdb -ldevice -lbuild_kernels_for_riscv -lhlkc_api -ldl -lcommon -lprofiler -lstdc++fs -pthread -lyaml-cpp

include $(TT_METAL_HOME)/tt_metal/programming_examples/loopback/module.mk
include $(TT_METAL_HOME)/tt_metal/programming_examples/eltwise_binary/module.mk

programming_examples: programming_examples/loopback \
                      programming_examples/eltwise_binary

programming_examples/loopback: $(PROGRAMMING_EXAMPLES_TESTDIR)/loopback;
programming_examples/eltwise_binary: $(PROGRAMMING_EXAMPLES_TESTDIR)/eltwise_binary;
