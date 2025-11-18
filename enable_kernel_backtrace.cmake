# Add TT_METAL_KERNEL_BACKTRACE define to graph_tracking.cpp
set_source_files_properties(
    tt_metal/graph/graph_tracking.cpp
    PROPERTIES
        COMPILE_FLAGS
            "-DTT_METAL_KERNEL_BACKTRACE"
)
