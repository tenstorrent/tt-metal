# Every variable in subdir must be prefixed with subdir (emulating a namespace)
JIT_BUILD_DEFINES =
JIT_BUILD_INCLUDES += -I$(TT_METAL_HOME)/tt_metal/jit_build $(BASE_INCLUDES) $(COMMON_INCLUDES)
JIT_BUILD_CFLAGS = $(CFLAGS) -Werror

JIT_BUILD_SRCS_RELATIVE = \
	jit_build/build.cpp \
	jit_build/genfiles.cpp \
	jit_build/data_format.cpp \
	jit_build/settings.cpp \
	jit_build/kernel_args.cpp

JIT_BUILD_SRCS = $(addprefix tt_metal/, $(JIT_BUILD_SRCS_RELATIVE))

JIT_BUILD_OBJS = $(addprefix $(OBJDIR)/, $(JIT_BUILD_SRCS:.cpp=.o))
JIT_BUILD_DEPS = $(addprefix $(OBJDIR)/, $(JIT_BUILD_SRCS:.cpp=.d))

-include $(JIT_BUILD_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
jit_build: $(COMMON_OBJS) $(JIT_BUILD_LIB)

$(OBJDIR)/tt_metal/jit_build/%.o: tt_metal/jit_build/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(JIT_BUILD_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(JIT_BUILD_INCLUDES) $(JIT_BUILD_DEFINES) -c -o $@ $<
