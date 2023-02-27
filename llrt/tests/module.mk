# Every variable in subdir must be prefixed with subdir (emulating a namespace)
#LLRT_TESTS += $(basename $(wildcard llrt/tests/*.cpp))
LLRT_TESTS += llrt/tests/test_silicon_driver \
				llrt/tests/test_load_and_write_brisc_binary \
				llrt/tests/test_run_brisc_binary \
				llrt/tests/test_run_add_two_ints \
				llrt/tests/test_silicon_driver_dram_sweep \
				llrt/tests/test_silicon_driver_l1_sweep \
				llrt/tests/test_run_dram_copy \
				llrt/tests/test_run_dram_copy_ncrisc \
				llrt/tests/test_run_dram_copy_brisc_ncrisc \
				llrt/tests/test_run_dram_copy_sweep \
				llrt/tests/test_run_dram_copy_looped_sweep \
				llrt/tests/test_run_blank_brisc_triscs \
				llrt/tests/test_run_datacopy \
				llrt/tests/test_run_datacopy_switched_riscs \
				llrt/tests/test_run_dram_to_l1_to_dram_copy_sweep \
				llrt/tests/test_run_reader \
				llrt/tests/test_run_reader_small \
				llrt/tests/test_run_dram_to_l1_copy_pattern \
				llrt/tests/test_run_dram_to_l1_copy_pattern_tilized \
				llrt/tests/test_run_eltwise_sync \
				llrt/tests/test_run_sync \
				llrt/tests/test_run_sync_db \
				llrt/tests/test_run_host_write_read_stream_register \
				llrt/tests/test_run_write_read_stream_register \
				llrt/tests/test_run_l1_to_dram_copy_pattern \
				llrt/tests/test_run_dram_to_l1_to_dram_copy_pattern \
				llrt/tests/test_run_l1_to_l1_copy_pattern \
				llrt/tests/test_run_risc_read_speed \
				llrt/tests/test_run_risc_write_speed \
				llrt/tests/test_run_risc_rw_speed_banked_dram \
				llrt/tests/test_run_test_debug_print \
				llrt/tests/test_run_transpose_hc \
				llrt/tests/test_run_matmul_small_block \
				llrt/tests/test_run_dataflow_cb_test

LLRT_TESTS_SRCS = $(addsuffix .cpp, $(LLRT_TESTS))

LLRT_TEST_INCLUDES = $(LLRT_INCLUDES) -Itt_gdb -Illrt/tests -Icompile_trisc -Iverif
LLRT_TESTS_LDFLAGS = -lllrt -ltt_gdb -ldevice -lcommon -lstdc++fs -pthread -lyaml-cpp -lprofiler

LLRT_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(LLRT_TESTS_SRCS:.cpp=.o))
LLRT_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(LLRT_TESTS_SRCS:.cpp=.d))

-include $(LLRT_TESTS_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
llrt/tests: $(LLRT_TESTS)
llrt/tests/all: $(LLRT_TESTS)
llrt/tests/%: $(TESTDIR)/llrt/tests/% ;

.PRECIOUS: $(TESTDIR)/llrt/tests/%
$(TESTDIR)/llrt/tests/%: $(OBJDIR)/llrt/tests/%.o $(BACKEND_LIB) $(LLRT_LIB) $(VERIF_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(LLRT_TEST_INCLUDES) -o $@ $^ $(LDFLAGS) $(LLRT_TESTS_LDFLAGS)

.PRECIOUS: $(OBJDIR)/llrt/tests/%.o
$(OBJDIR)/llrt/tests/%.o: llrt/tests/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(LLRT_TEST_INCLUDES) -c -o $@ $<
