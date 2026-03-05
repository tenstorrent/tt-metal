add_executable(precompile_fw ${CMAKE_CURRENT_SOURCE_DIR}/precompile_fw.cpp)
target_link_libraries(
    precompile_fw
    PRIVATE
        Metalium::Metal
        TT::STL
        yaml-cpp::yaml-cpp
)
target_include_directories(
    precompile_fw
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(
    precompile_fw
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)

target_precompile_headers(precompile_fw REUSE_FROM TT::CommonPCH)

add_custom_command(
    OUTPUT
        ${PROJECT_SOURCE_DIR}/pre-compiled/.stamp
    COMMAND
        $<TARGET_FILE:precompile_fw>
    COMMAND
        ${CMAKE_COMMAND} -E touch ${PROJECT_SOURCE_DIR}/pre-compiled/.stamp
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS
        precompile_fw
    COMMENT "Pre-compiling firmware binaries"
)

add_custom_target(precompile-fw ALL DEPENDS ${PROJECT_SOURCE_DIR}/pre-compiled/.stamp)

if(TT_INSTALL)
    install(
        DIRECTORY
            ${PROJECT_SOURCE_DIR}/pre-compiled/
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/pre-compiled
        COMPONENT metalium-runtime
    )
endif()
