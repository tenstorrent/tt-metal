project(linear_regression_ddp)

set(SOURCES main.cpp)
add_executable(linear_regression_ddp ${SOURCES})
target_link_libraries(linear_regression_ddp PRIVATE ttml)

check_function_exists(
    backtrace
    HAVE_BACKTRACE
)

if(NOT HAVE_BACKTRACE)
    find_library(EXECINFO_LIB execinfo)
    if(EXECINFO_LIB)
        target_link_libraries(linear_regression_ddp PRIVATE ${EXECINFO_LIB})
    endif()
endif()
