# Every variable in subdir must be prefixed with subdir (emulating a namespace)

TT_GDB_LIB = $(LIBDIR)/libtt_gdb.a
TT_GDB_DEFINES = -DGIT_HASH=$(shell git rev-parse HEAD)
TT_GDB_INCLUDES = $(COMMON_INCLUDES) $(MODEL_INCLUDES) $(NETLIST_INCLUDES) -I$(BUDA_HOME)/tt_gdb -I$(BUDA_HOME)/. -Ithird_party/json
TT_GDB_LDFLAGS = -L$(BUDA_HOME) -lcommon -lllrt -ltt_metal
TT_GDB_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

TT_GDB_SRCS = tt_gdb/tt_gdb.cpp

TT_GDB_OBJS = $(addprefix $(OBJDIR)/, $(TT_GDB_SRCS:.cpp=.o))
TT_GDB_DEPS = $(addprefix $(OBJDIR)/, $(TT_GDB_SRCS:.cpp=.d))

-include $(TT_GDB_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tt_gdb: $(TT_GDB_LIB)

$(TT_GDB_LIB): $(COMMON_LIB) $(NETLIST_LIB) $(TT_GDB_OBJS)
	@mkdir -p $(@D)
	ar rcs -o $@ $(TT_GDB_OBJS)

$(OBJDIR)/tt_gdb/%.o: tt_gdb/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TT_GDB_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TT_GDB_INCLUDES) $(TT_GDB_DEFINES) -c -o $@ $<
