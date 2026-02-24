# SPDX-FileCopyrightText: (c) 2026 Olof Johansson <olof@lixom.net>
#
# SPDX-License-Identifier: Apache-2.0

set(EZ_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/device_context.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/program_builder.cpp
)

add_library(tt_metal_ez STATIC ${EZ_SRCS})

target_include_directories(tt_metal_ez PRIVATE ${PROJECT_SOURCE_DIR})

target_link_libraries(
    tt_metal_ez
    PUBLIC
        metal_common_libs
        tt_metal
)
