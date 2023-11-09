MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLE_SRC = $(TT_METAL_HOME)/tt_metal/programming_examples/matmul_multicore_reuse_mcast/matmul_multicore_reuse_mcast.cpp

MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLES_DEPS = $(PROGRAMMING_EXAMPLES_OBJDIR)/matmul_multicore_reuse_mcast.d

-include $(MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLES_DEPS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multicore_reuse_mcast
$(PROGRAMMING_EXAMPLES_TESTDIR)/matmul_multicore_reuse_mcast: $(PROGRAMMING_EXAMPLES_OBJDIR)/matmul_multicore_reuse_mcast.o
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -o $@ $^ $(LDFLAGS) $(PROGRAMMING_EXAMPLES_LDFLAGS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_OBJDIR)/matmul_multicore_reuse_mcast.o
$(PROGRAMMING_EXAMPLES_OBJDIR)/matmul_multicore_reuse_mcast.o: $(MATMUL_MULTI_CORE_REUSE_MCAST_EXAMPLE_SRC)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -c -o $@ $<
