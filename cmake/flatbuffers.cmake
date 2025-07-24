function(GENERATE_FBS_HEADER FBS_FILE)
    # Optional Arguments:
    # - TARGET: Target to associate the generated header with (used for include directories)
    # - OUTPUT_DIR: Directory to place the generated header file (CMAKE_CURRENT_BINARY_DIR/flatbuffers by default)
    set(oneValueArgs
        TARGET
        OUTPUT_DIR
    )
    cmake_parse_arguments(FBS_ARGS "" "${oneValueArgs}" "" "${ARGN}")

    get_filename_component(FBS_FILE_NAME ${FBS_FILE} NAME_WE)

    # Set default output directory if not specified
    if(FBS_ARGS_OUTPUT_DIR)
        set(FBS_GENERATED_HEADER_DIR "${FBS_ARGS_OUTPUT_DIR}")
    else()
        set(FBS_GENERATED_HEADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers")
    endif()

    set(FBS_GENERATED_HEADER_FILE "${FBS_GENERATED_HEADER_DIR}/${FBS_FILE_NAME}_generated.h")

    set(incdirs "${CMAKE_CURRENT_SOURCE_DIR}")
    if(FBS_ARGS_TARGET)
        set(incdirs "$<TARGET_PROPERTY:${FBS_ARGS_TARGET},INCLUDE_DIRECTORIES>")
    endif()

    add_custom_command(
        OUTPUT
            ${FBS_GENERATED_HEADER_FILE}
        COMMAND
            flatc --keep-prefix --cpp --scoped-enums "$<$<BOOL:${incdirs}>:-I;$<JOIN:${incdirs},;-I;>>" -o
            "${FBS_GENERATED_HEADER_DIR}" ${FBS_FILE}
        DEPENDS
            flatc
            ${FBS_FILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Building C++ header for ${FBS_FILE}"
        COMMAND_EXPAND_LISTS
    )
    set(FBS_GENERATED_HEADER_FILE ${FBS_GENERATED_HEADER_FILE} PARENT_SCOPE)

    if(TARGET all_generated_files)
        add_custom_target(${FBS_FILE_NAME} DEPENDS ${FBS_GENERATED_HEADER_FILE})
        add_dependencies(all_generated_files ${FBS_FILE_NAME})
    endif()
endfunction()
