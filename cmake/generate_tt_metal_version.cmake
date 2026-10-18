# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

if(NOT VERSION_TEMPLATE OR NOT VERSION_OUTPUT)
    message(FATAL_ERROR "VERSION_TEMPLATE and VERSION_OUTPUT are required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/version.cmake")
ParseGitDescribe()
GenerateVersionHeader()
