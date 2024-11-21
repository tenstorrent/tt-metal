set(JIT_BUILD_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/build.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data_format.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/genfiles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_args.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/settings.cpp
)

add_library(jit_build OBJECT ${JIT_BUILD_SRCS})
target_link_libraries(jit_build PUBLIC common)

find_program(GIT_EXECUTABLE git)
if(GIT_EXECUTABLE)
    execute_process(
        COMMAND
            git rev-parse --short=10 HEAD
        RESULT_VARIABLE git_parse_result
        OUTPUT_VARIABLE git_commit_hash_value
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(git_parse_result EQUAL "0")
        target_compile_definitions(jit_build PUBLIC "-DGIT_COMMIT_HASH=\"${git_commit_hash_value}\"")
    endif()
endif()
