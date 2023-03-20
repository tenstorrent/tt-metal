# Every variable in subdir must be prefixed with subdir (emulating a namespace)

TT_GDB_LIB = $(LIBDIR)/libtt_gdb.a
TT_GDB_DEFINES = -DGIT_HASH=$(shell git rev-parse HEAD)
TT_GDB_INCLUDES = $(COMMON_INCLUDES) -I$(TT_METAL_HOME)/tt_metal/tools/tt_gdb -I$(TT_METAL_HOME)/tt_metal/third_party/json
TT_GDB_LDFLAGS = -L$(TT_METAL_HOME) -lcommon -lllrt -ltt_metal
TT_GDB_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

TT_GDB_SRCS = tt_metal/tools/tt_gdb/tt_gdb.cpp

TT_GDB_OBJS = $(addprefix $(OBJDIR)/, $(TT_GDB_SRCS:.cpp=.o))
TT_GDB_DEPS = $(addprefix $(OBJDIR)/, $(TT_GDB_SRCS:.cpp=.d))

-include $(TT_GDB_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tools/tt_gdb: $(TT_GDB_LIB)

$(TT_GDB_LIB): $(COMMON_LIB) $(TT_GDB_OBJS)
	@mkdir -p $(@D)
	ar rcs -o $@ $(TT_GDB_OBJS)

$(OBJDIR)/tt_metal/tools/tt_gdb/%.o: tt_metal/tools/tt_gdb/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TT_GDB_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TT_GDB_INCLUDES) $(TT_GDB_DEFINES) -c -o $@ $<
