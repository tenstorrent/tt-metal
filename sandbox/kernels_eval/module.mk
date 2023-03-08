KERNELS_EVAL += kernels_eval/test_run_conv

KERNELS_EVAL_SRCS = $(addsuffix .cpp, $(KERNELS_EVAL))

KERNELS_EVAL_INCLUDES = -Ikernels_eval -Illrt/tests/test_libs
KERNELS_EVAL_LDFLAGS = -lstdc++fs -pthread -lyaml-cpp

KERNELS_EVAL_OBJS = $(addprefix $(OBJDIR)/, $(KERNELS_EVAL_SRCS:.cpp=.o))
KERNELS_EVAL_DEPS = $(addprefix $(OBJDIR)/, $(KERNELS_EVAL_SRCS:.cpp=.d))

-include $(KERNELS_EVAL_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
kernels_eval: $(KERNELS_EVAL)
kernels_eval/all: $(KERNELS_EVAL)
kernels_eval/%: $(TESTDIR)/kernels_eval/% ;

.PRECIOUS: $(TESTDIR)/kernels_eval/%
$(TESTDIR)/kernels_eval/%: $(OBJDIR)/kernels_eval/%.o
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(KERNELS_EVAL_INCLUDES) -o $@ $^ $(LDFLAGS) $(KERNELS_EVAL_LDFLAGS)

.PRECIOUS: $(OBJDIR)/kernels_eval/%.o
$(OBJDIR)/kernels_eval/%.o: kernels_eval/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(KERNELS_EVAL_INCLUDES) -c -o $@ $<
