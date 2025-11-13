link_libraries(TT::STL)

add_compile_definitions(OVERRIDE_KERNEL_PREFIX="tools/tests/triage/hang_apps/")

add_subdirectory(add_2_integers_hang)
