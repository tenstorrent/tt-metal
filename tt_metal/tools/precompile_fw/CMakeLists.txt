include(sources.cmake)

add_executable(precompile_fw ${PRECOMPILE_FW_SOURCES})
target_link_libraries(
    precompile_fw
    PRIVATE
        tt_metal
        yaml-cpp::yaml-cpp
)

target_include_directories(precompile_fw PRIVATE ${PROJECT_SOURCE_DIR}/tt_metal)

set_target_properties(
    precompile_fw
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)

target_precompile_headers(precompile_fw REUSE_FROM TT::CommonPCH)

add_custom_command(
    OUTPUT
        ${PROJECT_BINARY_DIR}/tt_metal/pre-compiled/.stamp
    COMMAND
        ${CMAKE_COMMAND} -E env ASAN_OPTIONS=detect_odr_violation=0 $<TARGET_FILE:precompile_fw>
    COMMAND
        ${CMAKE_COMMAND} -E touch ${PROJECT_BINARY_DIR}/tt_metal/pre-compiled/.stamp
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS
        precompile_fw
    COMMENT "Pre-compiling firmware binaries"
)

add_custom_target(precompile-fw ALL DEPENDS ${PROJECT_BINARY_DIR}/tt_metal/pre-compiled/.stamp)

if(TT_INSTALL)
    install(
        DIRECTORY
            ${PROJECT_SOURCE_DIR}/tt_metal/pre-compiled/
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/pre-compiled
        COMPONENT metalium-runtime
    )
endif()
