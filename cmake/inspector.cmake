# Helper function to set up Cap'n Proto RPC for a target
function(TT_ADD_INSPECTOR_RPC_SUPPORT TARGET_NAME)
    set(options "")
    set(oneValueArgs OUTPUT_DIR)
    set(multiValueArgs
        CAPNP_SCHEMAS
        RPC_CHANNEL_SCHEMAS
    )
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_CAPNP_SCHEMAS)
        message(FATAL_ERROR "TT_ADD_INSPECTOR_RPC_SUPPORT requires at least one CAPNP_SCHEMA")
    endif()

    if(NOT ARG_OUTPUT_DIR)
        set(ARG_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    endif()
    file(MAKE_DIRECTORY ${ARG_OUTPUT_DIR})

    set(CAPNPC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    file(MAKE_DIRECTORY ${CAPNPC_OUTPUT_DIR})

    set(GENERATED_SOURCES)
    set(GENERATED_HEADERS)
    set(COPIED_CAPNP_SCHEMAS)

    # Generate C++ sources from Cap'n Proto schemas
    foreach(SCHEMA ${ARG_CAPNP_SCHEMAS})
        get_filename_component(SCHEMA_NAME ${SCHEMA} NAME_WE)
        set(COPIED_SCHEMA ${ARG_OUTPUT_DIR}/${SCHEMA_NAME}.capnp)
        add_custom_command(
            OUTPUT
                ${COPIED_SCHEMA}
            COMMAND
                ${CMAKE_COMMAND} -E copy ${SCHEMA} ${COPIED_SCHEMA}
            DEPENDS
                ${SCHEMA}
            COMMENT "Copying Cap'n Proto schema ${SCHEMA} to ${COPIED_SCHEMA}"
        )
        if(NOT EXISTS "${COPIED_SCHEMA}")
            file(COPY_FILE ${SCHEMA} ${COPIED_SCHEMA})
        endif()
        list(APPEND COPIED_CAPNP_SCHEMAS ${COPIED_SCHEMA})
    endforeach()
    set(ORIGINAL_CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    set(CMAKE_CURRENT_SOURCE_DIR ${CAPNPC_OUTPUT_DIR})
    foreach(SCHEMA ${COPIED_CAPNP_SCHEMAS})
        get_filename_component(SCHEMA_NAME ${SCHEMA} NAME_WE)
        capnp_generate_cpp(
            GENERATED_${SCHEMA_NAME}_SOURCES
            GENERATED_${SCHEMA_NAME}_HEADER
            ${SCHEMA}
        )
        list(APPEND GENERATED_SOURCES ${GENERATED_${SCHEMA_NAME}_SOURCES})
        list(APPEND GENERATED_HEADERS ${GENERATED_${SCHEMA_NAME}_HEADER})
    endforeach()
    set(CMAKE_CURRENT_SOURCE_DIR ${ORIGINAL_CMAKE_CURRENT_SOURCE_DIR})

    # Generate Inspector RPC callbacks server implementation
    foreach(SCHEMA ${ARG_RPC_CHANNEL_SCHEMAS})
        get_filename_component(SCHEMA_NAME ${SCHEMA} NAME_WE)
        set(RPC_CHANNEL_GENERATED_HEADER ${ARG_OUTPUT_DIR}/${SCHEMA_NAME}_channel_generated.hpp)
        set(RPC_CHANNEL_GENERATED_SOURCE ${ARG_OUTPUT_DIR}/${SCHEMA_NAME}_channel_generated.cpp)
        add_custom_command(
            OUTPUT
                ${RPC_CHANNEL_GENERATED_HEADER}
                ${RPC_CHANNEL_GENERATED_SOURCE}
            COMMAND
                python3 ${PROJECT_SOURCE_DIR}/scripts/generate_rpc_channel.py ${SCHEMA} ${RPC_CHANNEL_GENERATED_HEADER}
                ${RPC_CHANNEL_GENERATED_SOURCE}
            DEPENDS
                ${SCHEMA}
                ${PROJECT_SOURCE_DIR}/scripts/generate_rpc_channel.py
            COMMENT "Generating RPC server implementation for ${SCHEMA}"
        )
        list(APPEND GENERATED_SOURCES ${RPC_CHANNEL_GENERATED_SOURCE})
        list(APPEND GENERATED_HEADERS ${RPC_CHANNEL_GENERATED_HEADER})
    endforeach()

    # Add generated sources to target
    target_sources(${TARGET_NAME} PRIVATE ${GENERATED_SOURCES})

    # Disable linting for generated files
    set_source_files_properties(
        ${GENERATED_SOURCES}
        ${GENERATED_HEADERS}
        PROPERTIES
            SKIP_LINTING
                TRUE
    )

    # Link Cap'n Proto libraries privately
    target_link_libraries(
        ${TARGET_NAME}
        PRIVATE
            capnp
            capnp-rpc
    )

    # Add include directory for generated headers
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

    # Register with CI dependency tracking if available
    if(TARGET all_generated_files)
        set(DEPENDENCY_TARGET ${TARGET_NAME}_capnp_generated)
        add_custom_target(${DEPENDENCY_TARGET} DEPENDS ${GENERATED_HEADERS})
        add_dependencies(all_generated_files ${DEPENDENCY_TARGET})
    endif()
endfunction()
