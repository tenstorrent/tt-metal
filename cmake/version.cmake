# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Helper function to get RPM version suffix based on distribution
#
# This function generates the appropriate RPM version suffix (e.g., .fc43, .el9, .lp15.5)
# based on the detected Linux distribution. The suffix follows standard RPM conventions.
#
# Parameters:
#   DISTRO_ID_VAL - Lowercase distribution ID (e.g., "fedora", "rhel", "opensuse")
#   DISTRO_ID_LIKE_VAL - Lowercase ID_LIKE value (e.g., "fedora", "rhel fedora")
#   DISTRO_VERSION_VAL - Distribution version (e.g., "43", "9", "15.5")
#   OUTPUT_VAR - Variable name to set with the suffix result
#
# Supported distributions:
#   - Fedora: .fc{VERSION}
#   - RHEL/CentOS/Rocky/AlmaLinux: .el{VERSION}
#   - openSUSE Leap: .lp{VERSION}
#   - openSUSE Tumbleweed: .tw
#   - SLES: .sles{VERSION}
#
# For unknown distributions, a warning is issued and an empty suffix is returned.
function(get_rpm_version_suffix DISTRO_ID_VAL DISTRO_ID_LIKE_VAL DISTRO_VERSION_VAL OUTPUT_VAR)
    set(SUFFIX "")

    if(DISTRO_ID_VAL STREQUAL "fedora")
        string(APPEND SUFFIX ".fc${DISTRO_VERSION_VAL}")
    elseif(
        DISTRO_ID_VAL
            STREQUAL
            "rhel"
        OR DISTRO_ID_VAL
            STREQUAL
            "centos"
        OR DISTRO_ID_VAL
            STREQUAL
            "rocky"
        OR DISTRO_ID_VAL
            STREQUAL
            "almalinux"
    )
        string(APPEND SUFFIX ".el${DISTRO_VERSION_VAL}")
    elseif(DISTRO_ID_LIKE_VAL MATCHES "rhel|fedora")
        # Handle RHEL/Fedora derivatives that might not be explicitly listed
        # (e.g., Oracle Linux, Amazon Linux, Scientific Linux, etc.)
        # Use .el suffix for RHEL derivatives, .fc for Fedora derivatives
        if(DISTRO_ID_LIKE_VAL MATCHES "rhel")
            string(APPEND SUFFIX ".el${DISTRO_VERSION_VAL}")
        else()
            string(APPEND SUFFIX ".fc${DISTRO_VERSION_VAL}")
        endif()
    elseif(DISTRO_ID_VAL STREQUAL "opensuse-leap" OR DISTRO_ID_VAL STREQUAL "opensuse")
        # openSUSE Leap uses version numbers like 15.5
        string(APPEND SUFFIX ".lp${DISTRO_VERSION_VAL}")
    elseif(DISTRO_ID_VAL STREQUAL "opensuse-tumbleweed")
        # Tumbleweed is rolling release, use "tw" suffix
        string(APPEND SUFFIX ".tw")
    elseif(DISTRO_ID_VAL STREQUAL "sles")
        # SLES uses version numbers like 15.5, 12.5
        string(APPEND SUFFIX ".sles${DISTRO_VERSION_VAL}")
    else()
        # Unknown RPM distro - warn but don't fail
        # This allows the build to continue even if the distro isn't explicitly supported
        if(DISTRO_ID_VAL)
            message(
                WARNING
                "Unknown RPM distribution '${DISTRO_ID_VAL}' - no version suffix added. "
                "Supported distros: fedora, rhel, centos, rocky, almalinux, opensuse, sles. "
                "If packaging for this distro, set VERSION_RPM explicitly via -DVERSION_RPM=..."
            )
        endif()
    endif()

    set(${OUTPUT_VAR} "${SUFFIX}" PARENT_SCOPE)
endfunction()

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
        message(WARNING "Cannot parse tag '${tagname}', using fallback version 0.0.0")
        # Provide a fallback version so builds can continue
        set(VERSION_NUMERIC "0.0.0" PARENT_SCOPE)
        set(VERSION_FULL "0.0.0+unknown" PARENT_SCOPE)
        set(VERSION_DEB "0.0.0+unknown" PARENT_SCOPE)
        set(VERSION_RPM "0.0.0.unknown" PARENT_SCOPE)
        set(VERSION_HASH "${VERSION_HASH}" PARENT_SCOPE)
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
    include(detect-distro)
    detect_distro()

    # Add distro suffix based on detected OS for RPM packages
    get_rpm_version_suffix("${DISTRO_ID}" "${DISTRO_ID_LIKE}" "${DISTRO_VERSION}" RPM_SUFFIX)
    string(APPEND VERSION_RPM "${RPM_SUFFIX}")

    # Ubuntu/Debian suffix for DEB packages
    # Distinguish between Debian and Ubuntu (including derivatives)
    if(DISTRO_VERSION)
        if(DISTRO_ID STREQUAL "debian")
            string(APPEND VERSION_DEB "~debian${DISTRO_VERSION}")
        elseif(DISTRO_ID STREQUAL "ubuntu" OR DISTRO_ID_LIKE MATCHES "ubuntu")
            # Ubuntu or Ubuntu-based distros (e.g., Pop!_OS, Linux Mint)
            string(APPEND VERSION_DEB "~ubuntu${DISTRO_VERSION}")
        else()
            # Unknown DEB distro - warn and don't add suffix
            # This prevents incorrect version suffixes for unsupported distros
            if(DISTRO_ID)
                message(
                    WARNING
                    "Unknown DEB distribution '${DISTRO_ID}' - no version suffix added. "
                    "Supported distros: debian, ubuntu. "
                    "If packaging for this distro, set VERSION_DEB explicitly via -DVERSION_DEB=..."
                )
            endif()
        endif()
    endif()

    message(STATUS "Version: ${VERSION_FULL}")

    # Output variables
    set(VERSION_FULL "${VERSION_FULL}" PARENT_SCOPE)
    set(VERSION_DEB "${VERSION_DEB}" PARENT_SCOPE)
    set(VERSION_RPM "${VERSION_RPM}" PARENT_SCOPE)
    set(VERSION_NUMERIC "${VERSION_NUMERIC}" PARENT_SCOPE)
    set(VERSION_HASH "${VERSION_HASH}" PARENT_SCOPE)
endfunction()
