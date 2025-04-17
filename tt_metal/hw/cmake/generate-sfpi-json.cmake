# Script to generate SFPI version JSON file
include(${CMAKE_CURRENT_LIST_DIR}/sfpi-version.cmake)

file(WRITE "${OUTPUT_FILE}" "{\n")

get_cmake_property(_vars VARIABLES)
foreach(_var ${_vars})
    if(_var MATCHES "^SFPI_.*_RELEASE$")
        string(REPLACE "SFPI_" "" _arch_os "${_var}")
        string(REPLACE "_RELEASE" "" _arch_os "${_arch_os}")
        list(GET ${_var} 0 sfpi_file)
        list(GET ${_var} 1 sfpi_md5)
        file(APPEND "${OUTPUT_FILE}" "  \"${_arch_os}\": [\"${sfpi_file}\", \"${sfpi_md5}\"],\n")
    endif()
endforeach()

file(APPEND "${OUTPUT_FILE}" "  \"_end\": null\n}\n")
