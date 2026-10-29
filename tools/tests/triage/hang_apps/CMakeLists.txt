link_libraries(TT::STL)

# stop_simulation_on_termination.hpp is shared by the apps below.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_compile_definitions(OVERRIDE_KERNEL_PREFIX="tools/tests/triage/hang_apps/")

add_subdirectory(add_2_integers_hang)
add_subdirectory(ttnn_add_integers_hang)
