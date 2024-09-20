# Every variable in subdir must be prefixed with subdir (emulating a namespace)
TT_METAL_IMPL_DEFINES =
TT_METAL_IMPL_INCLUDES = $(COMMON_INCLUDES) -I$(TT_METAL_HOME)/tt_metal/impl -I$(TT_METAL_HOME)/.
TT_METAL_IMPL_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

TT_METAL_IMPL_SRCS = \
	tt_metal/impl/device/device.cpp \
	tt_metal/impl/device/device_pool.cpp \
	tt_metal/impl/device/mesh_device.cpp \
	tt_metal/impl/device/mesh_device_view.cpp \
	tt_metal/impl/buffers/buffer.cpp \
	tt_metal/impl/buffers/circular_buffer.cpp \
	tt_metal/impl/buffers/semaphore.cpp \
	tt_metal/impl/kernels/kernel.cpp \
	tt_metal/impl/allocator/algorithms/free_list.cpp \
	tt_metal/impl/allocator/allocator.cpp \
	tt_metal/impl/allocator/basic_allocator.cpp \
	tt_metal/impl/allocator/l1_banking_allocator.cpp \
	tt_metal/impl/program/program.cpp \
	tt_metal/impl/dispatch/debug_tools.cpp \
	tt_metal/impl/dispatch/command_queue.cpp \
	tt_metal/impl/debug/dprint_server.cpp \
	tt_metal/impl/debug/watcher_server.cpp \
	tt_metal/impl/trace/trace.cpp

TT_METAL_IMPL_OBJS = $(addprefix $(OBJDIR)/, $(TT_METAL_IMPL_SRCS:.cpp=.o))
TT_METAL_IMPL_DEPS = $(addprefix $(OBJDIR)/, $(TT_METAL_IMPL_SRCS:.cpp=.d))

-include $(TT_METAL_IMPL_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tt_metal/impl: $(COMMON_OBJS) $(TT_METAL_IMPL_OBJS)

$(OBJDIR)/tt_metal/impl/%.o: tt_metal/impl/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TT_METAL_IMPL_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TT_METAL_IMPL_INCLUDES) $(TT_METAL_IMPL_DEFINES) -c -o $@ $<
