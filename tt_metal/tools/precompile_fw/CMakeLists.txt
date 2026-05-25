include(sources.cmake)

add_executable(precompile_fw ${PRECOMPILE_FW_SOURCES})
target_link_libraries(
    precompile_fw
    PRIVATE
        tt_metal
        Metalium::Metal::Hardware
        yaml-cpp::yaml-cpp
)

target_include_directories(precompile_fw PRIVATE ${PROJECT_SOURCE_DIR}/tt_metal)

set_target_properties(
    precompile_fw
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)

tt_reuse_precompile_headers(precompile_fw TT::CommonPCH)

set(_skip_precompile_fw_default OFF)
if(DEFINED ENV{TT_SKIP_PRECOMPILE_FW})
    if(NOT "$ENV{TT_SKIP_PRECOMPILE_FW}" STREQUAL "" AND NOT "$ENV{TT_SKIP_PRECOMPILE_FW}" STREQUAL "0")
        set(_skip_precompile_fw_default ON)
    endif()
endif()
option(TT_SKIP_PRECOMPILE_FW "Skip runtime firmware precompile step" ${_skip_precompile_fw_default})

if(TT_SKIP_PRECOMPILE_FW)
    message(STATUS "Skipping firmware precompile step (TT_SKIP_PRECOMPILE_FW)")
else()
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
endif()

if(TT_INSTALL)
    if(EXISTS ${PROJECT_SOURCE_DIR}/tt_metal/pre-compiled/)
        install(
            DIRECTORY
                ${PROJECT_SOURCE_DIR}/tt_metal/pre-compiled/
            DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/pre-compiled
            COMPONENT metalium-runtime
        )
    else()
        message(STATUS "Skipping install of precompiled firmware directory; not present")
    endif()
endif()
