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

    # Set default output directory if not specified
    if(PROTO_ARGS_OUTPUT_DIR)
        set(PROTO_GENERATED_DIR "${PROTO_ARGS_OUTPUT_DIR}")
    else()
        set(PROTO_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/protobuf")
    endif()

    # Generate protobuf files using the standard protobuf_generate function
    protobuf_generate(
        LANGUAGE cpp
        OUT_VAR PROTO_SRCS
        PROTOC_OPTIONS --experimental_allow_proto3_optional
        PROTOS ${PROTO_FILE}
        GENERATE_EXTENSIONS .pb.h .pb.cc
        PROTOC_OUT_DIR ${PROTO_GENERATED_DIR}
    )

    # Set the generated files in parent scope
    set(PROTO_SRCS ${PROTO_SRCS} PARENT_SCOPE)

    # Add to all_generated_files target if it exists
    if(TARGET all_generated_files)
        add_custom_target(${PROTO_FILE_NAME} DEPENDS ${PROTO_SRCS})
        add_dependencies(all_generated_files ${PROTO_FILE_NAME})
    endif()
endfunction()
