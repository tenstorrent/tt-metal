# Every variable in subdir must be prefixed with subdir (emulating a namespace)

COMMON_INCLUDES = $(BASE_INCLUDES)

ifeq ("$(ARCH_NAME)", "wormhole_b0")
  COMMON_INCLUDES+=-Isrc/firmware/riscv/wormhole
  COMMON_INCLUDES+=-Isrc/firmware/riscv/wormhole/wormhole_b0_defines
else
  COMMON_INCLUDES+=-Isrc/firmware/riscv/$(ARCH_NAME)
endif

ifeq ("$(ARCH_NAME)", "wormhole")
  COMMON_INCLUDES+=-Isrc/firmware/riscv/wormhole/wormhole_a0_defines
endif

COMMON_LIB = $(LIBDIR)/libcommon.a
COMMON_DEFINES = -DGIT_HASH=$(shell git rev-parse HEAD)
COMMON_INCLUDES += -Icommon -Imodel -Inetlist
COMMON_LDFLAGS = -lyaml-cpp
COMMON_CFLAGS = $(CFLAGS) -Werror

COMMON_SRCS += \
	$(wildcard common/model/*.cpp) \
	$(wildcard common/*.cpp)

COMMON_OBJS = $(addprefix $(OBJDIR)/, $(COMMON_SRCS:.cpp=.o))
COMMON_DEPS = $(addprefix $(OBJDIR)/, $(COMMON_SRCS:.cpp=.d))

-include $(COMMON_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
common: $(COMMON_LIB)

$(COMMON_LIB): $(COMMON_OBJS)
	@mkdir -p $(@D)
	ar rcs -o $@ $(COMMON_OBJS)

$(OBJDIR)/common/model/%.o: common/model/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(COMMON_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(COMMON_INCLUDES) $(COMMON_DEFINES) -c -o $@ $(COMMON_LDFLAGS) $<

$(OBJDIR)/common/%.o: common/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(COMMON_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(COMMON_INCLUDES) $(COMMON_DEFINES) -c -o $@ $(COMMON_LDFLAGS) $<
