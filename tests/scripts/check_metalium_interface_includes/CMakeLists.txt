# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# Fails if the installed TT::Metalium exports libexec/tt-metalium as an include dir (#58511).
# Exported header file set base dirs are checked too: they do not show up in INTERFACE_INCLUDE_DIRECTORIES.
cmake_minimum_required(VERSION 3.24)
project(check_metalium_interface_includes LANGUAGES CXX)

find_package(TT-Metalium REQUIRED)

set(seen "")
set(stack TT::Metalium)
set(dirs "")
while(stack)
    list(POP_FRONT stack tgt)
    if(tgt IN_LIST seen)
        continue()
    endif()
    list(APPEND seen ${tgt})

    get_target_property(inc ${tgt} INTERFACE_INCLUDE_DIRECTORIES)
    if(inc)
        list(APPEND dirs ${inc})
    endif()

    get_target_property(sets ${tgt} INTERFACE_HEADER_SETS)
    foreach(set IN LISTS sets)
        get_target_property(set_dirs ${tgt} HEADER_DIRS_${set})
        if(set_dirs)
            list(APPEND dirs ${set_dirs})
        endif()
    endforeach()

    get_target_property(libs ${tgt} INTERFACE_LINK_LIBRARIES)
    foreach(lib IN LISTS libs)
        string(REGEX REPLACE "^\\$<LINK_ONLY:(.*)>$" "\\1" lib "${lib}")
        if(TARGET ${lib})
            list(APPEND stack ${lib})
        endif()
    endforeach()
endwhile()

list(LENGTH seen seen_count)
if(seen_count LESS 2)
    message(
        FATAL_ERROR
        "Only visited ${seen_count} target(s) from TT::Metalium; the walk is not seeing its dependencies."
    )
endif()

message(STATUS "TT::Metalium interface include dirs (from ${seen_count} targets):")
set(leaks "")
foreach(dir IN LISTS dirs)
    message(STATUS "  ${dir}")
    if(dir MATCHES "/libexec/tt-metalium/?>?$")
        list(APPEND leaks ${dir})
    endif()
endforeach()

if(leaks)
    message(FATAL_ERROR "TT::Metalium exports the libexec/tt-metalium header tree to consumers: ${leaks}")
endif()
