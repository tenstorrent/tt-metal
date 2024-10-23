function(CPMAddPackage)
    list(LENGTH ARGN argnLength)
    if(argnLength EQUAL 1)
        cpm_parse_add_package_single_arg("${ARGN}" ARGN)

        # The shorthand syntax implies EXCLUDE_FROM_ALL and SYSTEM
        set(ARGN "${ARGN};EXCLUDE_FROM_ALL;YES;SYSTEM;YES;")
    endif()

    set(oneValueArgs
        NAME
        FORCE
        VERSION
        GIT_TAG
        DOWNLOAD_ONLY
        GITHUB_REPOSITORY
        GITLAB_REPOSITORY
        BITBUCKET_REPOSITORY
        GIT_REPOSITORY
        SOURCE_DIR
        FIND_PACKAGE_ARGUMENTS
        NO_CACHE
        SYSTEM
        GIT_SHALLOW
        EXCLUDE_FROM_ALL
        SOURCE_SUBDIR
        CUSTOM_CACHE_KEY
    )

    set(multiValueArgs
        URL
        OPTIONS
        DOWNLOAD_COMMAND
        PATCHES
    )

    cmake_parse_arguments(CPM_ARGS "" "${oneValueArgs}" "${multiValueArgs}" "${ARGN}")
endfunction()
