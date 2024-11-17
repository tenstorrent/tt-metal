include(CTest)
enable_testing()

file(
    GLOB_RECURSE SOURCES
    LIST_DIRECTORIES true
    *.hpp
    *.cpp
) # probably should not do that...

#set(SOURCES)

add_executable(ttml_tests ${SOURCES})
target_link_libraries(
    ttml_tests
    GTest::gtest_main
    ttml
)
add_definitions(-DTEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data")

# Define the target file location
set(TOKENIZER_URL "https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1/resolve/main/tokenizer.json")
set(TOKENIZER_FILE "${CMAKE_BINARY_DIR}/data/tokenizer.json")

# Check if the file already exists before downloading
if(NOT EXISTS "${TOKENIZER_FILE}")
    message(STATUS "Downloading Tokenizer text file to ${TOKENIZER_FILE}")
    file(
        DOWNLOAD
            ${TOKENIZER_URL}
            ${TOKENIZER_FILE}
        SHOW_PROGRESS
    )
else()
    message(STATUS "Tokenizer text file already exists at ${TOKENIZER_FILE}, skipping download.")
endif()

include(GoogleTest)
gtest_discover_tests(ttml_tests)
