include(CTest)
enable_testing()

file(
    GLOB_RECURSE SOURCES
    LIST_DIRECTORIES true
    *.hpp
    *.cpp
) # probably should not do that...

add_executable(ttml_tests ${SOURCES})
target_link_libraries(
    ttml_tests
    GTest::gmock_main
    ttml
)
add_definitions(-DTEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data")

# Define the target file location
set(TOKENIZER_URL "https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1/resolve/main/tokenizer.json")
set(TOKENIZER_FILE "${CMAKE_SOURCE_DIR}/data/tokenizer.json")

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

add_definitions(-DTEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data")

# variable to enable CI only tests
# 0 = off, 1 = on
add_definitions(-DENABLE_NIGHTLY_TT_TRAIN_TESTS=0)

# Define the target file location
set(SHAKESPEARE_URL "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
set(SHAKESPEARE_FILE "${CMAKE_CURRENT_BINARY_DIR}/shakespeare.txt")

# Check if the file already exists before downloading
if(NOT EXISTS "${SHAKESPEARE_FILE}")
    message(STATUS "Downloading Shakespeare text file to ${SHAKESPEARE_FILE}")
    file(
        DOWNLOAD
            ${SHAKESPEARE_URL}
            ${SHAKESPEARE_FILE}
        SHOW_PROGRESS
    )
else()
    message(STATUS "Shakespeare text file already exists at ${SHAKESPEARE_FILE}, skipping download.")
endif()

include(GoogleTest)
gtest_discover_tests(ttml_tests)
