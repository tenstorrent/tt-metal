# SPDX-FileCopyrightText: 2024 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# May be called prior to project()
function(ParseGitDescribe)
    set(version "")
    # These will be filled in by `git archive`.
    # Building the source outside of git from something that was not exported via `git archive`
    # is left as an exercise to whoever is wanting to do that.
    set(fallbackVersion "$Format:%(describe)$")
    set(fallbackHash "$Format:%h$")

    find_package(Git)
    if(Git_FOUND)
        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} describe --abbrev=10 --first-parent --dirty=-dirty
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE version
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} rev-parse --short=10 HEAD
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE VERSION_HASH
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} rev-parse HEAD
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE VERSION_SHA
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()
    if(NOT VERSION_HASH)
        set(VERSION_HASH ${fallbackHash})
    endif()
    if(NOT version)
        set(version ${fallbackVersion})
        # A shallow Git clone will fail a git describe, but also will not have substituted the fallbackVersion
        if(version MATCHES "Format")
            set(version "0.0-alpha0-1-g${VERSION_HASH}-dirty")
        endif()
    endif()

    # Local modifications (dirty), or not
    set(dirtyFlagRegex "\\-dirty")
    set(VERSION_DIRTY FALSE)
    if("${version}" MATCHES "${dirtyFlagRegex}$")
        set(VERSION_DIRTY TRUE)
        string(REGEX REPLACE "^(.*)${dirtyFlagRegex}$" "\\1" version "${version}")
    endif()

    # On a Tagged commit, or not
    set(untaggedRegex "^(.*)\\-([0-9]+)\\-g([0-9a-f]+)$") # tag-count-ghash
    if("${version}" MATCHES "${untaggedRegex}")
        set(VERSION_TAGGED FALSE)
        string(REGEX REPLACE "${untaggedRegex}" "\\1" tagname "${version}")
        string(REGEX REPLACE "${untaggedRegex}" "\\2" VERSION_COMMIT_COUNT "${version}")
    else()
        set(VERSION_TAGGED TRUE)
        set(tagname "${version}")
    endif()

    set(major "([0-9]+)")
    set(segment "\\.[0-9]+")
    set(status "\\-([a-zA-Z]+[0-9]+)") # eg: alpha, beta, RC
    set(tagRegex "^[^0-9]*(${major}(${segment}(${segment}(${segment})?)?)?)(${status})?$")
    if(NOT "${tagname}" MATCHES "${tagRegex}")
        message(WARNING "Cannot parse tag ${tagname}")
        return()
    endif()

    # Major[.Minor[.Patch[.Tweak]]] suitable for CMake
    string(REGEX REPLACE "${tagRegex}" "\\1" VERSION_NUMERIC "${tagname}")

    # Build a new regex because we cannot access a capture group that was not matched.
    # And also only the first 9 capture groups are referenceable.
    set(statusRegex ".*(${status})$")
    if("${tagname}" MATCHES "${statusRegex}")
        string(REGEX REPLACE "${statusRegex}" "\\2" VERSION_STATUS "${tagname}")
    endif()

    set(VERSION_FULL "${VERSION_NUMERIC}")
    set(VERSION_DEB "${VERSION_NUMERIC}")
    if(VERSION_STATUS)
        string(APPEND VERSION_FULL "-${VERSION_STATUS}")
        string(APPEND VERSION_DEB "~${VERSION_STATUS}") # Debian versioning uses a ~ for "less than blank"
    endif()
    if(VERSION_COMMIT_COUNT)
        string(APPEND VERSION_FULL "+${VERSION_COMMIT_COUNT}.${VERSION_HASH}")
        string(APPEND VERSION_DEB "+${VERSION_COMMIT_COUNT}.${VERSION_HASH}")
    endif()
    if(VERSION_DIRTY)
        string(APPEND VERSION_FULL "+m")
        string(APPEND VERSION_DEB "+m")
    endif()

    # Include Ubuntu's version to disambiguate packages
    execute_process(
        COMMAND
            lsb_release -sr
        OUTPUT_VARIABLE UBUNTU_RELEASE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(APPEND VERSION_DEB "~ubuntu${UBUNTU_RELEASE}")

    message(STATUS "Version: ${VERSION_FULL}")

    # Output variables
    set(VERSION_FULL "${VERSION_FULL}" PARENT_SCOPE)
    set(VERSION_DEB "${VERSION_DEB}" PARENT_SCOPE)
    set(VERSION_NUMERIC "${VERSION_NUMERIC}" PARENT_SCOPE)
    set(VERSION_HASH "${VERSION_HASH}" PARENT_SCOPE)
    set(VERSION_SHA "${VERSION_SHA}" PARENT_SCOPE)
    set(VERSION_DIRTY "${VERSION_DIRTY}" PARENT_SCOPE)
endfunction()

function(GenerateVersionHeader)
    if(NOT VERSION_SHA)
        set(VERSION_SHA "${VERSION_HASH}")
    endif()
    if(CMAKE_BUILD_TYPE)
        set(VERSION_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
    else()
        set(VERSION_BUILD_TYPE "Unknown")
    endif()
    if(VERSION_DIRTY)
        set(VERSION_DIRTY_CPP "true")
    else()
        set(VERSION_DIRTY_CPP "false")
    endif()
    configure_file(
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/tt_metal_version.hpp.in"
        "${PROJECT_BINARY_DIR}/generated/tt_metal/impl/version.hpp"
        @ONLY
    )
endfunction()
