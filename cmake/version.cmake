# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
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
    set(VERSION_RPM "${VERSION_NUMERIC}")
    if(VERSION_STATUS)
        string(APPEND VERSION_FULL "-${VERSION_STATUS}")
        string(APPEND VERSION_DEB "~${VERSION_STATUS}") # Debian versioning uses a ~ for "less than blank"
        string(APPEND VERSION_RPM "~${VERSION_STATUS}") # RPM also uses ~ for pre-release versions
    endif()
    if(VERSION_COMMIT_COUNT)
        string(APPEND VERSION_FULL "+${VERSION_COMMIT_COUNT}.${VERSION_HASH}")
        string(APPEND VERSION_DEB "+${VERSION_COMMIT_COUNT}.${VERSION_HASH}")
        # RPM uses . instead of + for additional version segments
        string(APPEND VERSION_RPM ".${VERSION_COMMIT_COUNT}.${VERSION_HASH}")
    endif()
    if(VERSION_DIRTY)
        string(APPEND VERSION_FULL "+m")
        string(APPEND VERSION_DEB "+m")
        string(APPEND VERSION_RPM ".m")
    endif()

    # Include distro version to disambiguate packages
    # Detect distro type from /etc/os-release
    set(DISTRO_SUFFIX "")
    if(EXISTS "/etc/os-release")
        file(STRINGS "/etc/os-release" OS_RELEASE_ID REGEX "^ID=")
        file(STRINGS "/etc/os-release" OS_RELEASE_VERSION REGEX "^VERSION_ID=")
        if(OS_RELEASE_ID MATCHES "ID=\"?([^\"]+)\"?")
            string(TOLOWER "${CMAKE_MATCH_1}" DISTRO_ID)
        endif()
        if(OS_RELEASE_VERSION MATCHES "VERSION_ID=\"?([^\"]+)\"?")
            set(DISTRO_VERSION "${CMAKE_MATCH_1}")
        endif()
    endif()

    # For backwards compatibility, also check lsb_release
    if(NOT DISTRO_VERSION)
        execute_process(
            COMMAND
                lsb_release -sr
            OUTPUT_VARIABLE DISTRO_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()

    # Add distro suffix based on detected OS
    if(DISTRO_ID STREQUAL "fedora")
        string(APPEND VERSION_RPM ".fc${DISTRO_VERSION}")
    elseif(
        DISTRO_ID
            STREQUAL
            "rhel"
        OR DISTRO_ID
            STREQUAL
            "centos"
        OR DISTRO_ID
            STREQUAL
            "rocky"
        OR DISTRO_ID
            STREQUAL
            "almalinux"
    )
        string(APPEND VERSION_RPM ".el${DISTRO_VERSION}")
    endif()
    # Ubuntu/Debian suffix for DEB packages
    if(DISTRO_VERSION)
        string(APPEND VERSION_DEB "~ubuntu${DISTRO_VERSION}")
    endif()

    message(STATUS "Version: ${VERSION_FULL}")

    # Output variables
    set(VERSION_FULL "${VERSION_FULL}" PARENT_SCOPE)
    set(VERSION_DEB "${VERSION_DEB}" PARENT_SCOPE)
    set(VERSION_RPM "${VERSION_RPM}" PARENT_SCOPE)
    set(VERSION_NUMERIC "${VERSION_NUMERIC}" PARENT_SCOPE)
    set(VERSION_HASH "${VERSION_HASH}" PARENT_SCOPE)
endfunction()
