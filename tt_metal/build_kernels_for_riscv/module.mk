# Every variable in subdir must be prefixed with subdir (emulating a namespace)
BUILD_KERNELS_FOR_RISCV_DEFINES =
BUILD_KERNELS_FOR_RISCV_INCLUDES += -I$(TT_METAL_HOME)/tt_metal/build_kernels_for_riscv $(BASE_INCLUDES) $(COMMON_INCLUDES)
BUILD_KERNELS_FOR_RISCV_CFLAGS = $(CFLAGS) -Werror

BUILD_KERNELS_FOR_RISCV_SRCS_RELATIVE = \
	build_kernels_for_riscv/build_kernels_for_riscv.cpp \
	build_kernels_for_riscv/data_format.cpp \
	build_kernels_for_riscv/build_kernel_options.cpp

BUILD_KERNELS_FOR_RISCV_SRCS = $(addprefix tt_metal/, $(BUILD_KERNELS_FOR_RISCV_SRCS_RELATIVE))

BUILD_KERNELS_FOR_RISCV_OBJS = $(addprefix $(OBJDIR)/, $(BUILD_KERNELS_FOR_RISCV_SRCS:.cpp=.o))
BUILD_KERNELS_FOR_RISCV_DEPS = $(addprefix $(OBJDIR)/, $(BUILD_KERNELS_FOR_RISCV_SRCS:.cpp=.d))

-include $(BUILD_KERNELS_FOR_RISCV_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
build_kernels_for_riscv: $(COMMON_OBJS) $(BUILD_KERNELS_FOR_RISCV_OBJS)

$(OBJDIR)/tt_metal/build_kernels_for_riscv/%.o: tt_metal/build_kernels_for_riscv/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(BUILD_KERNELS_FOR_RISCV_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(BUILD_KERNELS_FOR_RISCV_INCLUDES) $(BUILD_KERNELS_FOR_RISCV_DEFINES) -c -o $@ $<
