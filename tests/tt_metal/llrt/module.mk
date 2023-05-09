# Every variable in subdir must be prefixed with subdir (emulating a namespace)
#LLRT_TESTS += $(basename $(wildcard llrt/*.cpp))
LLRT_TESTS += tests/llrt/test_silicon_driver \
				tests/llrt/test_run_brisc_binary \
				tests/llrt/test_run_add_two_ints \
				tests/llrt/test_silicon_driver_dram_sweep \
				tests/llrt/test_silicon_driver_l1_sweep \
				tests/llrt/test_run_blank_brisc_triscs \
				tests/llrt/test_run_datacopy \
				tests/llrt/test_run_datacopy_switched_riscs \
				tests/llrt/test_run_eltwise_sync \
				tests/llrt/test_run_sync \
				tests/llrt/test_run_sync_db \
				tests/llrt/test_run_host_write_read_stream_register \
				tests/llrt/test_run_risc_read_speed \
				tests/llrt/test_run_risc_write_speed \
				tests/llrt/test_run_risc_rw_speed_banked_dram \
				tests/llrt/test_run_test_debug_print \
				tests/llrt/test_run_transpose_hc \
				tests/llrt/test_run_matmul_small_block \
				tests/llrt/test_run_dataflow_cb_test \
				# tests/llrt/test_dispatch_v1

LLRT_TESTS_SRCS = $(addprefix tests/tt_metal/, $(addsuffix .cpp, $(LLRT_TESTS:tests%=%)))

LLRT_TEST_INCLUDES = $(TEST_INCLUDES) $(LLRT_INCLUDES) -Itt_gdb -I$(TT_METAL_HOME)/tt_metal/device/${ARCH_NAME}
LLRT_TESTS_LDFLAGS = -lllrt -ltt_gdb -lprofiler -ldevice -lcommon -lstdc++fs -pthread -lyaml-cpp

LLRT_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(LLRT_TESTS_SRCS:.cpp=.o))
LLRT_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(LLRT_TESTS_SRCS:.cpp=.d))

-include $(LLRT_TESTS_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tests/llrt: $(LLRT_TESTS)
tests/llrt/all: $(LLRT_TESTS)
tests/llrt/test_%: $(TESTDIR)/llrt/test_% ;

.PRECIOUS: $(TESTDIR)/llrt/test_%
$(TESTDIR)/llrt/test_%: $(OBJDIR)/llrt/tests/test_%.o $(LLRT_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(LLRT_TEST_INCLUDES) -o $@ $^ $(LDFLAGS) $(LLRT_TESTS_LDFLAGS)

.PRECIOUS: $(OBJDIR)/llrt/tests/test_%.o
$(OBJDIR)/llrt/tests/test_%.o: tests/tt_metal/llrt/test_%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(LLRT_TEST_INCLUDES) -c -o $@ $<

llrt/tests: tests/llrt
llrt/tests/all: tests/llrt/all
