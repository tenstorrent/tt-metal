# Every variable in subdir must be prefixed with subdir (emulating a namespace)

COMMON_INCLUDES = $(BASE_INCLUDES)

ifeq ("$(ARCH_NAME)", "wormhole_b0")
  COMMON_INCLUDES+= -Isrc/firmware/riscv/wormhole
  COMMON_INCLUDES+= -Isrc/firmware/riscv/wormhole/wormhole_b0_defines
else
  COMMON_INCLUDES+= -Isrc/firmware/riscv/$(ARCH_NAME)
endif

ifeq ("$(ARCH_NAME)", "wormhole")
  COMMON_INCLUDES+= -Isrc/firmware/riscv/wormhole/wormhole_a0_defines
endif

COMMON_LIB = $(LIBDIR)/libcommon.a
COMMON_DEFINES = -DGIT_HASH=$(shell git rev-parse HEAD)
COMMON_INCLUDES += -I$(TT_METAL_HOME)/tt_metal/common/.
COMMON_LDFLAGS = -lyaml-cpp
COMMON_CFLAGS = $(CFLAGS) -Werror

COMMON_SRCS += \
	$(wildcard tt_metal/common/*.cpp)

COMMON_OBJS = $(addprefix $(OBJDIR)/, $(COMMON_SRCS:.cpp=.o))
COMMON_DEPS = $(addprefix $(OBJDIR)/, $(COMMON_SRCS:.cpp=.d))

-include $(COMMON_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
common: $(COMMON_LIB)

$(COMMON_LIB): $(COMMON_OBJS)
	@mkdir -p $(@D)
	ar rcs -o $@ $(COMMON_OBJS)

$(OBJDIR)/tt_metal/common/%.o: tt_metal/common/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(COMMON_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(COMMON_INCLUDES) $(COMMON_DEFINES) -c -o $@ $(COMMON_LDFLAGS) $<
