# Function to generate FlatBuffers headers
function(GENERATE_FBS_HEADER FBS_FILE)
    get_filename_component(FBS_FILE_NAME ${FBS_FILE} NAME)
    set(FBS_GENERATED_HEADER "${CMAKE_CURRENT_BINARY_DIR}/${FBS_FILE_NAME}_generated.h")
    add_custom_command(
        OUTPUT
            ${FBS_GENERATED_HEADER}
        COMMAND
            flatc --cpp --scoped-enums -I ${CMAKE_CURRENT_SOURCE_DIR} -o "${CMAKE_CURRENT_BINARY_DIR}" ${FBS_FILE}
        DEPENDS
            flatc
            ${FBS_FILE}
        COMMENT "Building C++ header for ${FBS_FILE}"
    )
    set(FBS_GENERATED_HEADER ${FBS_GENERATED_HEADER} PARENT_SCOPE)
endfunction()
