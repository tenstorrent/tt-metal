cmake_minimum_required(VERSION 3.22...4.2)
project(wh_alias_dest_reuse_repro)

add_executable(wh_alias_dest_reuse_repro)
target_sources(wh_alias_dest_reuse_repro PRIVATE wh_alias_dest_reuse_repro.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(wh_alias_dest_reuse_repro PUBLIC TT::Metalium)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(wh_alias_dest_reuse_repro TT::CommonPCH)
endif()
