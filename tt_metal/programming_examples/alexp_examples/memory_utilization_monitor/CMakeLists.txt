cmake_minimum_required(VERSION 3.22...3.30)
project(memory_utilization_monitor)

# Legacy memory utilization monitoring tools
add_executable(memory_monitor memory_monitor.cpp)
add_executable(memory_monitor_simple memory_monitor_simple.cpp)
add_executable(memory_monitor_minimal memory_monitor_minimal.cpp)
add_executable(memory_monitor_test memory_monitor_with_test.cpp)
add_executable(test_tracking_cpp test_tracking_cpp.cpp)
add_executable(test_mesh_allocation_cpp test_mesh_allocation_cpp.cpp)
add_executable(test_graphtracker_check test_graphtracker_check.cpp)

# Tracy-based memory monitor (NEW - recommended)
add_executable(tracy_memory_monitor_client tracy_memory_monitor_client.cpp)
add_executable(tracy_memory_monitor_examples example_integration.cpp)
add_executable(tracy_memory_server tracy_memory_server.cpp)

# Link with TT-Metal libraries
target_link_libraries(memory_monitor PUBLIC TT::Metalium)
target_link_libraries(memory_monitor_simple PUBLIC TT::Metalium)
target_link_libraries(memory_monitor_minimal PUBLIC TT::Metalium)
target_link_libraries(memory_monitor_test PUBLIC TT::Metalium)
target_link_libraries(test_tracking_cpp PUBLIC TT::Metalium)
target_link_libraries(test_mesh_allocation_cpp PUBLIC TT::Metalium)
target_link_libraries(test_graphtracker_check PUBLIC TT::Metalium)

# Tracy memory monitor needs access to impl headers
# CMAKE_SOURCE_DIR = /home/tt-metal-apv (top-level, unaffected by project())
# Header is at: /home/tt-metal-apv/tt_metal/impl/profiler/tracy_memory_monitor.hpp
# Include as: "tt_metal/impl/profiler/tracy_memory_monitor.hpp"
# So we need: /home/tt-metal-apv (CMAKE_SOURCE_DIR)
target_link_libraries(tracy_memory_monitor_client PUBLIC TT::Metalium)
target_include_directories(tracy_memory_monitor_client PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(tracy_memory_monitor_examples PUBLIC TT::Metalium)
target_include_directories(tracy_memory_monitor_examples PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(tracy_memory_server PUBLIC TT::Metalium)
target_include_directories(tracy_memory_server PRIVATE ${CMAKE_SOURCE_DIR})
