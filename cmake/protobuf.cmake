function(GENERATE_PROTO_FILES PROTO_FILE)
    # Optional Arguments:
    # - TARGET: Target to associate the generated files with
    # - OUTPUT_DIR: Directory to place the generated files (CMAKE_CURRENT_BINARY_DIR/protobuf by default)
    set(oneValueArgs
        TARGET
        OUTPUT_DIR
    )
    cmake_parse_arguments(PROTO_ARGS "" "${oneValueArgs}" "" "${ARGN}")

    get_filename_component(PROTO_FILE_NAME ${PROTO_FILE} NAME_WE)

    # Compute output directory; always generate into a 'protobuf' subfolder
    if(PROTO_ARGS_OUTPUT_DIR)
        set(PROTO_GENERATED_BASE "${PROTO_ARGS_OUTPUT_DIR}")
    else()
        set(PROTO_GENERATED_BASE "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    set(PROTO_GENERATED_DIR "${PROTO_GENERATED_BASE}/protobuf")

    # Generate protobuf files by invoking protoc directly. This avoids depending on
    # the CMake-provided protobuf_generate() macro, which may be unavailable when
    # using protobuf via CPM.
    get_filename_component(PROTO_DIR ${PROTO_FILE} DIRECTORY)

    file(MAKE_DIRECTORY ${PROTO_GENERATED_DIR})

    set(GENERATED_CC ${PROTO_GENERATED_DIR}/${PROTO_FILE_NAME}.pb.cc)
    set(GENERATED_H ${PROTO_GENERATED_DIR}/${PROTO_FILE_NAME}.pb.h)

    add_custom_command(
        OUTPUT
            ${GENERATED_CC}
            ${GENERATED_H}
        COMMAND
            $<TARGET_FILE:protobuf::protoc> --experimental_allow_proto3_optional --cpp_out=${PROTO_GENERATED_DIR} -I
            ${PROTO_DIR} ${PROTO_FILE}
        DEPENDS
            protobuf::protoc
            ${PROTO_FILE}
        VERBATIM
    )

    # Set the generated files in parent scope
    set(PROTO_SRCS
        ${GENERATED_CC}
        ${GENERATED_H}
        PARENT_SCOPE
    )

    # Disable clang-tidy for generated protobuf files
    set_source_files_properties(
        ${GENERATED_CC}
        PROPERTIES
            SKIP_LINTING
                TRUE
    )

    # Add to all_generated_files target if it exists
    if(TARGET all_generated_files)
        # Use local variables for outputs; parent-scoped PROTO_SRCS is not visible here
        add_custom_target(
            ${PROTO_FILE_NAME}
            DEPENDS
                ${GENERATED_CC}
                ${GENERATED_H}
        )
        add_dependencies(all_generated_files ${PROTO_FILE_NAME})
    endif()
endfunction()
