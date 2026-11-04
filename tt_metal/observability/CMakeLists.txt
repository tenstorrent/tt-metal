# The public observability headers the host and the device share, api/. Host code includes them as
# <observability/api/...> through this target. Device code never reaches this folder: the JIT build stages a copy of
# it, merged with the device's own versions in tt_metal/hw/observability/api, into its cache
# (tt_metal/jit_build/staged_includes.hpp). An install ships the folder twice: the public headers for host code, and
# the JIT's copy source in libexec.
include(sources.cmake)

# The list is what an install ships, the JIT's copy source among it, so a header in the folder must be on it.
file(
    GLOB OBSERVABILITY_API_FOLDER
    CONFIGURE_DEPENDS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/api/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/api/*.hpp"
)
foreach(header IN LISTS OBSERVABILITY_API_FOLDER)
    if(NOT header IN_LIST OBSERVABILITY_API_HEADERS)
        message(
            FATAL_ERROR
            "tt_metal/observability/${header} is not in OBSERVABILITY_API_HEADERS (tt_metal/observability/sources.cmake), "
            "so an install would not ship it"
        )
    endif()
endforeach()

# The host include root holds observability/api alone, as a link to the source folder, so a target that links this one
# can include <observability/api/...> and nothing else of tt_metal.
set(OBSERVABILITY_API_ROOT "${CMAKE_CURRENT_BINARY_DIR}/include")
file(MAKE_DIRECTORY "${OBSERVABILITY_API_ROOT}/observability")
file(
    CREATE_LINK
        "${CMAKE_CURRENT_SOURCE_DIR}/api"
        "${OBSERVABILITY_API_ROOT}/observability/api"
    SYMBOLIC
)
list(
    TRANSFORM OBSERVABILITY_API_HEADERS
    PREPEND
        "${OBSERVABILITY_API_ROOT}/observability/"
    OUTPUT_VARIABLE OBSERVABILITY_API_FILES
)

add_library(ttmetalium_observability_api INTERFACE)
add_library(TT::Metalium::ObservabilityApi ALIAS ttmetalium_observability_api)
target_sources(
    ttmetalium_observability_api
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${OBSERVABILITY_API_ROOT}
        FILES ${OBSERVABILITY_API_FILES}
)

# Every header must compile on its own for host code as old as C++17, which code outside tt-metal may use. The device
# compile of the same headers is checked in tt_metal/hw/CMakeLists.txt.
set(OBSERVABILITY_API_HOST_CHECKS "")
foreach(header IN LISTS OBSERVABILITY_API_HEADERS)
    get_filename_component(name "${header}" NAME)
    set(stamp "${CMAKE_CURRENT_BINARY_DIR}/checked/${name}")
    add_custom_command(
        OUTPUT
            ${stamp}
        COMMAND
            ${CMAKE_CXX_COMPILER} -std=c++17 -Wall -Wextra -Werror -fsyntax-only -x c++ -I${OBSERVABILITY_API_ROOT}
            -include observability/api/${name} /dev/null
        COMMAND
            ${CMAKE_COMMAND} -E touch ${stamp}
        DEPENDS
            ${OBSERVABILITY_API_HEADERS}
        COMMENT "Checking observability/api/${name} compiles for the host"
        VERBATIM
    )
    list(APPEND OBSERVABILITY_API_HOST_CHECKS ${stamp})
endforeach()
add_custom_target(observability_api_host_check ALL DEPENDS ${OBSERVABILITY_API_HOST_CHECKS})

install(TARGETS ttmetalium_observability_api EXPORT Metalium FILE_SET api COMPONENT metalium-dev)
# The JIT's copy source. Plain files, not a file set, which would add the folder itself to its consumers' include path.
install(
    FILES
        ${OBSERVABILITY_API_HEADERS}
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/observability/api
    COMPONENT metalium-runtime
)
