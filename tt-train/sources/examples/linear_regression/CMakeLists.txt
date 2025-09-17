project(linear_regression)

set(SOURCES main.cpp)
add_executable(linear_regression ${SOURCES})
target_link_libraries(linear_regression PRIVATE ttml)

check_function_exists(
    backtrace
    HAVE_BACKTRACE
)

if(NOT HAVE_BACKTRACE)
    find_library(EXECINFO_LIB execinfo)
    if(EXECINFO_LIB)
        target_link_libraries(linear_regression PRIVATE ${EXECINFO_LIB})
    endif()
endif()
