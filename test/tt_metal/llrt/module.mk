# Every variable in subdir must be prefixed with subdir (emulating a namespace)
#LLRT_TESTS += $(basename $(wildcard llrt/*.cpp))
LLRT_TESTS += test/llrt/test_silicon_driver \
				test/llrt/test_load_and_write_brisc_binary \
				test/llrt/test_run_brisc_binary \
				test/llrt/test_run_add_two_ints \
				test/llrt/test_silicon_driver_dram_sweep \
				test/llrt/test_silicon_driver_l1_sweep \
				test/llrt/test_run_dram_copy \
				test/llrt/test_run_dram_copy_ncrisc \
				test/llrt/test_run_dram_copy_brisc_ncrisc \
				test/llrt/test_run_dram_copy_sweep \
				test/llrt/test_run_dram_copy_looped_sweep \
				test/llrt/test_run_blank_brisc_triscs \
				test/llrt/test_run_datacopy \
				test/llrt/test_run_datacopy_switched_riscs \
				test/llrt/test_run_dram_to_l1_to_dram_copy_sweep \
				test/llrt/test_run_reader \
				test/llrt/test_run_reader_small \
				test/llrt/test_run_dram_to_l1_copy_pattern \
				test/llrt/test_run_dram_to_l1_copy_pattern_tilized \
				test/llrt/test_run_eltwise_sync \
				test/llrt/test_run_sync \
				test/llrt/test_run_sync_db \
				test/llrt/test_run_host_write_read_stream_register \
				test/llrt/test_run_write_read_stream_register \
				test/llrt/test_run_l1_to_dram_copy_pattern \
				test/llrt/test_run_dram_to_l1_to_dram_copy_pattern \
				test/llrt/test_run_l1_to_l1_copy_pattern \
				test/llrt/test_run_risc_read_speed \
				test/llrt/test_run_risc_write_speed \
				test/llrt/test_run_risc_rw_speed_banked_dram \
				test/llrt/test_run_test_debug_print \
				test/llrt/test_run_transpose_hc \
				test/llrt/test_run_matmul_small_block \
				test/llrt/test_run_dataflow_cb_test

LLRT_TESTS_SRCS = $(addprefix test/tt_metal/, $(addsuffix .cpp, $(LLRT_TESTS:test/%=%)))

LLRT_TEST_INCLUDES = $(TEST_INCLUDES) $(LLRT_INCLUDES) -Itt_gdb -Illrt/tests
LLRT_TESTS_LDFLAGS = -lllrt -ltt_gdb -ldevice -lcommon -lstdc++fs -pthread -lyaml-cpp

LLRT_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(LLRT_TESTS_SRCS:.cpp=.o))
LLRT_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(LLRT_TESTS_SRCS:.cpp=.d))

-include $(LLRT_TESTS_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
test/llrt: $(LLRT_TESTS)
test/llrt/all: $(LLRT_TESTS)
test/llrt/test_%: $(TESTDIR)/llrt/test_% ;

.PRECIOUS: $(TESTDIR)/llrt/test_%
$(TESTDIR)/llrt/test_%: $(OBJDIR)/llrt/test/test_%.o $(BACKEND_LIB) $(LLRT_LIB) $(VERIF_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(LLRT_TEST_INCLUDES) -o $@ $^ $(LDFLAGS) $(LLRT_TESTS_LDFLAGS)

.PRECIOUS: $(OBJDIR)/llrt/test/test_%.o
$(OBJDIR)/llrt/test/test_%.o: test/tt_metal/llrt/test_%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(LLRT_TEST_INCLUDES) -c -o $@ $<

llrt/tests: test/llrt
llrt/tests/all: test/llrt/all
