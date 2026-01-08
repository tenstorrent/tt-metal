cmake_minimum_required(VERSION 3.22...3.30)
project(metal_fabric_toy_ccls)

set(SOURCES
    all_gather_1d_ring.cpp
    # add more source files here
)

foreach(source_file ${SOURCES})
    get_filename_component(exec_name ${source_file} NAME_WE)
    add_executable(${exec_name})
    target_sources(${exec_name} PRIVATE ${source_file})

    target_link_libraries(
        ${exec_name}
        PUBLIC
            tt_metal
            TT::Metalium
    )

    target_include_directories(
        ${exec_name}
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}/tt_metal
            ${PROJECT_SOURCE_DIR}/tests
            ${PROJECT_SOURCE_DIR}/tests/tt_metal/test_utils
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
endforeach()
