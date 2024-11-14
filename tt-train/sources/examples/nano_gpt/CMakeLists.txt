project(nano_gpt)

set(SOURCES
    main.cpp
    utils.cpp
    models.cpp
)

add_executable(nano_gpt ${SOURCES})
target_link_libraries(nano_gpt PRIVATE ttml)

add_definitions(-DDATA_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}/data")
