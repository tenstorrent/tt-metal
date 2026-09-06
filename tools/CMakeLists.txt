# Tools subdirectory
# This directory contains various development and deployment tools

add_subdirectory(scaleout)
add_subdirectory(tracy_inspect)

if(TT_METAL_BUILD_TESTS)
    add_subdirectory(tests)
endif()
