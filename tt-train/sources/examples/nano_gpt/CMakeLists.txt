project(nano_gpt)

set(SOURCES
    main.cpp
    utils.cpp
)

add_executable(nano_gpt ${SOURCES})
target_link_libraries(nano_gpt PRIVATE ttml)
add_definitions(-DDATA_FOLDER="${CMAKE_SOURCE_DIR}/data")
add_definitions(-DCONFIGS_FOLDER="${CMAKE_SOURCE_DIR}/configs")

# Define the target file location
set(SHAKESPEARE_URL "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
set(SHAKESPEARE_FILE "${CMAKE_SOURCE_DIR}/data/shakespeare.txt")

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
