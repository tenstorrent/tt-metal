# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2019-2023 Lars Melchior and contributors

set(CPM_DOWNLOAD_VERSION 0.40.2)
set(CPM_HASH_SUM "c8cdc32c03816538ce22781ed72964dc864b2a34a310d3b7104812a5ca2d835d")

# Always Require the CMake option, but provide default, respecting env var
if(DEFINED ENV{CPM_SOURCE_CACHE} AND NOT DEFINED CPM_SOURCE_CACHE)
    set(CPM_SOURCE_CACHE $ENV{CPM_SOURCE_CACHE} CACHE STRING "Path to CPM source cache")
else()
    set(CPM_SOURCE_CACHE "${CMAKE_SOURCE_DIR}/.cpmcache" CACHE STRING "Path to CPM source cache")
endif()

set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

# Expand relative path. This is important if the provided path contains a tilde (~)
get_filename_component(CPM_DOWNLOAD_LOCATION ${CPM_DOWNLOAD_LOCATION} ABSOLUTE)

file(
    DOWNLOAD
        https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
        ${CPM_DOWNLOAD_LOCATION}
    EXPECTED_HASH SHA256=${CPM_HASH_SUM}
)

include(${CPM_DOWNLOAD_LOCATION})
