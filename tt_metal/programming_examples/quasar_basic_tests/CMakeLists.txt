cmake_minimum_required(VERSION 3.22...3.30)
project(quasar_l1_test)

add_executable(quasar_l1_test)
target_sources(quasar_l1_test PRIVATE single_dm_l1_write.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(quasar_l1_test PUBLIC TT::Metalium)
