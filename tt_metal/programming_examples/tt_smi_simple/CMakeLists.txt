cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_tt_smi_simple)

add_executable(metal_example_tt_smi_simple)
target_sources(metal_example_tt_smi_simple PRIVATE tt_smi_simple.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(
    metal_example_tt_smi_simple
    PUBLIC
        TT::Metalium
    PRIVATE
        pthread
        stdc++fs
)

# Try to find ncurses for better TUI support (optional)
find_package(Curses QUIET)
if(CURSES_FOUND)
    target_link_libraries(metal_example_tt_smi_simple PRIVATE ${CURSES_LIBRARIES})
    target_include_directories(metal_example_tt_smi_simple PRIVATE ${CURSES_INCLUDE_DIRS})
endif()

# Disable specific warnings that come from TT-Metal headers
target_compile_options(
    metal_example_tt_smi_simple
    PRIVATE
        -Wno-error=unused-parameter
        -Wno-unused-parameter
)
