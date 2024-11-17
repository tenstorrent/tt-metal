project(nano_gpt)

set(SOURCES
    main.cpp
    utils.cpp
    models.cpp
)

add_executable(nano_gpt ${SOURCES})
target_link_libraries(nano_gpt PRIVATE ttml)

add_definitions(-DDATA_FOLDER="${CMAKE_BINARY_DIR}/data")

# Define the target file location
set(SHAKESPEARE_URL "https://www.cs.princeton.edu/courses/archive/spring20/cos302/files/shakespeare.txt")
set(SHAKESPEARE_FILE "${CMAKE_BINARY_DIR}/data/shakespeare.txt")

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
