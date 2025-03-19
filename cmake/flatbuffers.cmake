# Function to generate FlatBuffers C++ headers from schema files
function(GENERATE_FBS_HEADER FBS_FILE)
    get_filename_component(FBS_FILE_NAME ${FBS_FILE} NAME_WE)
    set(FBS_GENERATED_HEADER_DIR "${CMAKE_CURRENT_BINARY_DIR}/flatbuffers")
    set(FBS_GENERATED_HEADER_FILE "${FBS_GENERATED_HEADER_DIR}/${FBS_FILE_NAME}_generated.h")
    add_custom_command(
        OUTPUT
            ${FBS_GENERATED_HEADER_FILE}
        COMMAND
            flatc --cpp --scoped-enums -I ${CMAKE_CURRENT_SOURCE_DIR} -o "${FBS_GENERATED_HEADER_DIR}" ${FBS_FILE}
        DEPENDS
            flatc
            ${FBS_FILE}
        COMMENT "Building C++ header for ${FBS_FILE}"
    )
    set(FBS_GENERATED_HEADER_FILE ${FBS_GENERATED_HEADER_FILE} PARENT_SCOPE)
endfunction()
