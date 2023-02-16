# Every variable in subdir must be prefixed with subdir (emulating a namespace)

TOOLS = \
	tools/memset

TOOLS_SRCS = $(addsuffix .cpp, $(TOOLS))

TOOLS_INCLUDES = $(COMMON_INCLUDES) -I$(BUDA_HOME)/tools -I$(BUDA_HOME)/. -Illrt
TOOLS_LDFLAGS = $(LDFLAGS) -ldevice -lllrt  -lcommon -lyaml-cpp -lstdc++fs

TOOLS_DEPS = $(addprefix $(OBJDIR)/, $(TOOLS_SRCS:.cpp=.d))

-include $(TOOLS_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tools: $(OBJDIR)/tools/memset

.PRECIOUS: $(OBJDIR)/tools/%
$(OBJDIR)/tools/memset: $(OBJDIR)/tools/memset.o
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TOOLS_INCLUDES) -o $@ $^ $(TOOLS_LDFLAGS) 

$(OBJDIR)/tools/memset.o: tools/memset.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TOOLS_INCLUDES) -c -o $@ $<
