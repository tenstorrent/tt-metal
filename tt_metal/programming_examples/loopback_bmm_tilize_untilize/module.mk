LOOPBACK_BMM_TILIZE_UNTILIZE_EXAMPLE_SRC = $(TT_METAL_HOME)/tt_metal/programming_examples/loopback_bmm_tilize_untilize/loopback_bmm_tilize_untilize.cpp

LOOPBACK_BMM_TILIZE_UNTILIZE_EXAMPLES_DEPS = $(PROGRAMMING_EXAMPLES_OBJDIR)/loopback_bmm_tilize_untilize.d

-include $(LOOPBACK_BMM_TILIZE_UNTILIZE_EXAMPLES_DEPS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_TESTDIR)/loopback_bmm_tilize_untilize
$(PROGRAMMING_EXAMPLES_TESTDIR)/loopback_bmm_tilize_untilize: $(PROGRAMMING_EXAMPLES_OBJDIR)/loopback_bmm_tilize_untilize.o $(BACKEND_LIB) $(TT_METAL_LIB) $(VERIF_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -o $@ $^ $(LDFLAGS) $(PROGRAMMING_EXAMPLES_LDFLAGS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_OBJDIR)/loopback_bmm_tilize_untilize.o
$(PROGRAMMING_EXAMPLES_OBJDIR)/loopback_bmm_tilize_untilize.o: $(LOOPBACK_BMM_TILIZE_UNTILIZE_EXAMPLE_SRC)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -c -o $@ $<
