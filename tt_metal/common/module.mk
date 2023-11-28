# Every variable in subdir must be prefixed with subdir (emulating a namespace)

include $(TT_METAL_HOME)/tt_metal/tracy.mk

COMMON_INCLUDES = $(BASE_INCLUDES)

ifeq ("$(ARCH_NAME)", "wormhole_b0")
  COMMON_INCLUDES+= -Isrc/firmware/riscv/wormhole
  COMMON_INCLUDES+= -Isrc/firmware/riscv/wormhole/wormhole_b0_defines
  COMMON_INCLUDES+= -I$(UMD_HOME)/device/wormhole/.
  COMMON_INCLUDES+= -I$(UMD_HOME)/src/firmware/riscv/wormhole
else
  COMMON_INCLUDES+= -Isrc/firmware/riscv/$(ARCH_NAME)
  COMMON_INCLUDES+= -I$(UMD_HOME)/device/$(ARCH_NAME)/.
  COMMON_INCLUDES+= -I$(UMD_HOME)/src/firmware/riscv/$(ARCH_NAME)
endif

ifeq ("$(ARCH_NAME)", "wormhole")
  COMMON_INCLUDES+= -Isrc/firmware/riscv/wormhole
  COMMON_INCLUDES+= -Isrc/firmware/riscv/wormhole/wormhole_a0_defines
  COMMON_INCLUDES+= -I$(UMD_HOME)/device/wormhole/.
  COMMON_INCLUDES+= -I$(UMD_HOME)/src/firmware/riscv/wormhole
endif

COMMON_DEFINES =
COMMON_INCLUDES += -I$(TT_METAL_HOME)/tt_metal/common/.
COMMON_CFLAGS = $(CFLAGS) -Werror

COMMON_SRCS += \
	$(wildcard tt_metal/common/*.cpp)

COMMON_OBJS = $(addprefix $(OBJDIR)/, $(COMMON_SRCS:.cpp=.o))
COMMON_DEPS = $(addprefix $(OBJDIR)/, $(COMMON_SRCS:.cpp=.d))

-include $(COMMON_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
common: $(COMMON_OBJS) umd_device $(TRACY_OBJS)

$(OBJDIR)/tt_metal/common/%.o: tt_metal/common/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(COMMON_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(COMMON_INCLUDES) $(COMMON_DEFINES) -c -o $@ $<
