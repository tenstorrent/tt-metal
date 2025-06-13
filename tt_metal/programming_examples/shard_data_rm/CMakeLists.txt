cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_shard_data_rm)

add_executable(metal_example_shard_data_rm)
target_sources(metal_example_shard_data_rm PRIVATE shard_data_rm.cpp)

find_package(TT-Metalium REQUIRED)
target_link_libraries(metal_example_shard_data_rm PUBLIC TT::Metalium)
