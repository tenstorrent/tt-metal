.PRECIOUS: $(PROGRAMMING_EXAMPLES_TESTDIR)/profiler/device/grayskull/%
$(PROGRAMMING_EXAMPLES_TESTDIR)/profiler/device/grayskull/%: $(PROGRAMMING_EXAMPLES_OBJDIR)/profiler/device/grayskull/%.o $(TT_METAL_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -o $@ $^ $(LDFLAGS) $(PROGRAMMING_EXAMPLES_LDFLAGS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_OBJDIR)/profiler/device/grayskull/%.o
$(PROGRAMMING_EXAMPLES_OBJDIR)/profiler/device/grayskull/%.o: $(TT_METAL_HOME)/tt_metal/programming_examples/profiler/device/grayskull/%/*.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -c -o $@ $<
