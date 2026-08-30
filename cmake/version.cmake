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
    set(fallbackSha "$Format:%H$")

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
    if(NOT VERSION_SHA)
        set(VERSION_SHA "${fallbackSha}")
    endif()
    if(NOT VERSION_HASH)
        set(VERSION_HASH "${fallbackHash}")
    endif()
    # An unsubstituted $Format token is not a real hash.
    if(VERSION_SHA MATCHES "Format")
        set(VERSION_SHA "")
    endif()
    if(VERSION_HASH MATCHES "Format")
        set(VERSION_HASH "")
    endif()
    if(NOT version)
        set(version ${fallbackVersion})
        # A shallow Git clone will fail a git describe, but also will not have substituted the fallbackVersion
        if(version MATCHES "Format")
            set(version "0.0-alpha0-1-g${VERSION_HASH}-dirty")
        endif()
    endif()

    # Package +m follows describe -dirty; report dirty follows the worktree.
    set(dirtyFlagRegex "\\-dirty")
    set(VERSION_DESCRIBE_DIRTY FALSE)
    if("${version}" MATCHES "${dirtyFlagRegex}$")
        set(VERSION_DESCRIBE_DIRTY TRUE)
        string(REGEX REPLACE "^(.*)${dirtyFlagRegex}$" "\\1" version "${version}")
    endif()
    set(VERSION_DIRTY FALSE)
    if(Git_FOUND)
        execute_process(
            COMMAND
                ${GIT_EXECUTABLE} diff-index --quiet HEAD --
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            RESULT_VARIABLE _git_worktree_dirty
            ERROR_QUIET
        )
        # diff-index returns 1 only when the worktree has local modifications.
        if(_git_worktree_dirty EQUAL 1)
            set(VERSION_DIRTY TRUE)
        endif()
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
    if(VERSION_DESCRIBE_DIRTY)
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

    if(NOT VERSION_PARSE_QUIET)
        message(STATUS "Version: ${VERSION_FULL}")
    endif()

    # Output variables
    set(VERSION_FULL "${VERSION_FULL}" PARENT_SCOPE)
    set(VERSION_DEB "${VERSION_DEB}" PARENT_SCOPE)
    set(VERSION_NUMERIC "${VERSION_NUMERIC}" PARENT_SCOPE)
    set(VERSION_HASH "${VERSION_HASH}" PARENT_SCOPE)
    set(VERSION_SHA "${VERSION_SHA}" PARENT_SCOPE)
    set(VERSION_DIRTY "${VERSION_DIRTY}" PARENT_SCOPE)
endfunction()

function(GenerateVersionHeader)
    if(VERSION_DIRTY)
        set(VERSION_DIRTY_CPP "true")
    else()
        set(VERSION_DIRTY_CPP "false")
    endif()
    if(NOT VERSION_TEMPLATE)
        set(VERSION_TEMPLATE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/tt_metal_version.hpp.in")
    endif()
    if(NOT VERSION_OUTPUT)
        set(VERSION_OUTPUT "${PROJECT_BINARY_DIR}/generated/tt_metal/impl/version.hpp")
    endif()
    if(NOT EXISTS "${VERSION_TEMPLATE}")
        message(FATAL_ERROR "Missing version header template: ${VERSION_TEMPLATE}")
    endif()
    get_filename_component(_version_outdir "${VERSION_OUTPUT}" DIRECTORY)
    file(MAKE_DIRECTORY "${_version_outdir}")
    set(_version_tmp "${VERSION_OUTPUT}.tmp")
    configure_file("${VERSION_TEMPLATE}" "${_version_tmp}" @ONLY)
    execute_process(
        COMMAND
            "${CMAKE_COMMAND}" -E copy_if_different "${_version_tmp}" "${VERSION_OUTPUT}"
    )
    file(REMOVE "${_version_tmp}")
endfunction()

function(AddVersionHeaderTarget)
    set(_template "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/tt_metal_version.hpp.in")
    set(_script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/generate_tt_metal_version.cmake")
    set(_output "${PROJECT_BINARY_DIR}/generated/tt_metal/impl/version.hpp")
    add_custom_target(
        tt_metal_version_header
        COMMAND
            "${CMAKE_COMMAND}" "-DVERSION_TEMPLATE=${_template}" "-DVERSION_OUTPUT=${_output}"
            "-DVERSION_PARSE_QUIET=TRUE" -P "${_script}"
        BYPRODUCTS
            "${_output}"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        VERBATIM
        COMMENT "Refreshing tt-metal version header"
    )
endfunction()
