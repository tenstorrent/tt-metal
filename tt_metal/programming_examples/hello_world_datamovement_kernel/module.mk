ADD_IN_COMPUTE_EXAMPLE_SRC = $(TT_METAL_HOME)/tt_metal/programming_examples/hello_world_datamovement_kernel/hello_world_datamovement_kernel.cpp

ADD_IN_COMPUTE_EXAMPLES_DEPS = $(PROGRAMMING_EXAMPLES_OBJDIR)/hello_world_datamovement_kernel.d

-include $(ADD_IN_COMPUTE_EXAMPLES_DEPS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_TESTDIR)/hello_world_datamovement_kernel
$(PROGRAMMING_EXAMPLES_TESTDIR)/hello_world_datamovement_kernel: $(PROGRAMMING_EXAMPLES_OBJDIR)/hello_world_datamovement_kernel.o $(TT_METAL_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -o $@ $^ $(LDFLAGS) $(PROGRAMMING_EXAMPLES_LDFLAGS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_OBJDIR)/hello_world_datamovement_kernel.o
$(PROGRAMMING_EXAMPLES_OBJDIR)/hello_world_datamovement_kernel.o: $(ADD_IN_COMPUTE_EXAMPLE_SRC)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -c -o $@ $<
