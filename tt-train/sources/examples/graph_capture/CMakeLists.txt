project(graph_capture)

set(SOURCES main.cpp)

add_executable(graph_capture ${SOURCES})
target_link_libraries(graph_capture PRIVATE ttml)

check_function_exists(
    backtrace
    HAVE_BACKTRACE
)

if(NOT HAVE_BACKTRACE)
    find_library(EXECINFO_LIB execinfo)
    if(EXECINFO_LIB)
        target_link_libraries(graph_capture PRIVATE ${EXECINFO_LIB})
    endif()
endif()
