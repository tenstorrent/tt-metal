# Every variable in subdir must be prefixed with subdir (emulating a namespace)
HLKC_ROSE_CXXFLAGS = $(shell rose-config cppflags) #$(shell rose-config cxxflags)
HLKC_ROSE_LDFLAGS  = $(shell rose-config libdirs) $(shell rose-config ldflags)

HLKC_MEOW_HASH_FLAGS = -mavx -maes

HLKC_SRCS = hlkc/hlkc.cpp
HLKC = $(BINDIR)/hlkc
HLKC_INCLUDES = -I/usr/rose/include/rose
HLKC_OBJS = $(addprefix $(OBJDIR)/, $(HLKC_SRCS:.cpp=.o))
HLKC_DEPS = $(addprefix $(OBJDIR)/, $(HLKC_SRCS:.cpp=.d))
HLKC_LDFLAGS = -ldl -lstdc++ -lstdc++fs

HLKC_API_SRCS = hlkc/hlkc_api.cpp
HLKC_API_LIB = $(LIBDIR)/libhlkc_api.a
HLKC_API_OBJS = $(addprefix $(OBJDIR)/, $(HLKC_API_SRCS:.cpp=.o))
HLKC_API_DEPS = $(addprefix $(OBJDIR)/, $(HLKC_API_SRCS:.cpp=.d))
HLKC_API_LDFLAGS = $(HLKC_LDFLAGS)

-include $(HLKC_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
# TODO: make src/ckernels a real dependency of $(HLKC)
hlkc: $(HLKC) $(TT_METAL_HOME)/src/ckernels
hlkc/api: $(HLKC_API_LIB)

$(HLKC): $(HLKC_OBJS)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(HLKC_ROSE_CXXFLAGS) $(HLKC_MEOW_HASH_FLAGS) -o $@ $^ $(LDFLAGS) $(HLKC_ROSE_LDFLAGS) $(HLKC_LDFLAGS)

$(HLKC_API_LIB): $(HLKC_API_OBJS)
	@mkdir -p $(@D)
	ar rcs -o $@ $^

$(OBJDIR)/hlkc/hlkc_api.o: hlkc/hlkc_api.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(HLKC_ROSE_CXXFLAGS) $(HLKC_MEOW_HASH_FLAGS) -c -o $@ $< $(HLKC_ROSE_LDFLAGS) $(HLKC_LDFLAGS)

$(OBJDIR)/hlkc/%.o: hlkc/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(HLKC_ROSE_CXXFLAGS) $(HLKC_MEOW_HASH_FLAGS) -c -o $@ $< $(HLKC_ROSE_LDFLAGS) $(HLKC_LDFLAGS)
