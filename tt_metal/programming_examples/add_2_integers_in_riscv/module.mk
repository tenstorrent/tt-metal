ADD_IN_RISCV_EXAMPLE_SRC = $(TT_METAL_HOME)/tt_metal/programming_examples/add_2_integers_in_riscv/add_2_integers_in_riscv.cpp

ADD_IN_RISCV_EXAMPLES_DEPS = $(PROGRAMMING_EXAMPLES_OBJDIR)/add_2_integers_in_riscv.d

-include $(ADD_IN_RISCV_EXAMPLES_DEPS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_TESTDIR)/add_2_integers_in_riscv
$(PROGRAMMING_EXAMPLES_TESTDIR)/add_2_integers_in_riscv: $(PROGRAMMING_EXAMPLES_OBJDIR)/add_2_integers_in_riscv.o $(TT_METAL_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -o $@ $^ $(LDFLAGS) $(PROGRAMMING_EXAMPLES_LDFLAGS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_OBJDIR)/add_2_integers_in_riscv.o
$(PROGRAMMING_EXAMPLES_OBJDIR)/add_2_integers_in_riscv.o: $(ADD_IN_RISCV_EXAMPLE_SRC)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -c -o $@ $<
