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
add_definitions(-DTEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test_data")

include(GoogleTest)
gtest_discover_tests(ttml_tests)
