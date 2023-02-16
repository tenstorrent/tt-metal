PROGRAMMING_EXAMPLES_TESTDIR = $(TESTDIR)/programming_examples
PROGRAMMING_EXAMPLES_OBJDIR = $(OBJDIR)/programming_examples

PROGRAMMING_EXAMPLES_INCLUDES = $(COMMON_INCLUDES)
PROGRAMMING_EXAMPLES_LDFLAGS = -lll_buda_impl -lll_buda -lllrt -ltt_gdb -ldevice -lbuild_kernels_for_riscv -lhlkc_api -ldl -lcommon -lprofiler -lstdc++fs -pthread -lyaml-cpp

include programming_examples/basic_empty_program/module.mk
include programming_examples/loopback/module.mk

programming_examples/loopback: $(PROGRAMMING_EXAMPLES_TESTDIR)/loopback;
programming_examples/basic_empty_program: $(PROGRAMMING_EXAMPLES_TESTDIR)/basic_empty_program;

