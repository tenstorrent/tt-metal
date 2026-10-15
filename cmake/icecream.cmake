# Opt-in distributed compilation via Icecream, layered on top of ccache (see ccache.cmake,
# which calls this and wires the result into its launcher).
#
# tt_icecream_configure(<out_env> <out_disable_pch>)
#   <out_env>          list of NAME=value entries for the ccache launcher's environment;
#                      empty when Icecream is not enabled
#   <out_disable_pch>  set to ON when Icecream requires precompiled headers to be off;
#                      empty otherwise. The caller must apply it (in its own scope and
#                      its parent's) before any PCH targets are created.
function(tt_icecream_configure out_env out_disable_pch)
    set(${out_env} "" PARENT_SCOPE)
    set(${out_disable_pch} "" PARENT_SCOPE)

    # Opt-in distributed compilation via Icecream, for dev environments with spare
    # compute nodes (e.g. exabox). Off everywhere else by default -- including CI --
    # since it only activates when TT_ICECC_SCHEDULER_HOST is set, which we expect
    # dev environments to export globally (e.g. from the exabox image), never
    # something this repo defaults on its own.
    set(TT_ICECC_SCHEDULER_HOST
        $ENV{TT_ICECC_SCHEDULER_HOST}
        CACHE STRING "Icecream scheduler host; set to enable distributed ccache builds"
    )
    if(TT_ICECC_SCHEDULER_HOST)
        # TT_ICECC_SCHEDULER_HOST being set is a deliberate ask for distributed builds.
        # If we can't actually deliver that, fail the configure loudly rather than quietly
        # falling back to plain ccache -- a silent downgrade here is easy to miss and leaves
        # a dev wondering for a long time why their build never distributes. Note this is
        # independent of the scheduler/compute nodes themselves being flaky at *build* time:
        # icecc already falls back to local compilation per-job when the scheduler or a
        # daemon is unreachable, which is the flakiness this doesn't need to handle.
        find_program(ICECC_EXECUTABLE icecc)
        if(NOT ICECC_EXECUTABLE)
            message(FATAL_ERROR "TT_ICECC_SCHEDULER_HOST is set but icecc is not installed -- install it or unset TT_ICECC_SCHEDULER_HOST to build with plain ccache")
        endif()

        # Icecream ships the compiler itself to remote nodes, but the bundle is built
        # per-toolchain -- and per-ABI for clang's libc++/libstdc++ split, since compiling
        # (not linking) against libc++ vs libstdc++ resolves a different header tree --
        # via icecc-create-env. Map the selected toolchain to its env name.
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND ENABLE_LIBCXX)
            set(_tt_icecc_env_name "clang-20-libcxx")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            set(_tt_icecc_env_name "clang-20-libstdcxx")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            set(_tt_icecc_env_name "gcc-${CMAKE_CXX_COMPILER_VERSION}")
        else()
            message(FATAL_ERROR "TT_ICECC_SCHEDULER_HOST is set but ${CMAKE_CXX_COMPILER_ID} has no known Icecream env mapping")
        endif()

        # Cache generated envs per-user (survives build-dir wipes, shared across checkouts).
        # Generation only runs once per toolchain; delete the tarball to force a rebuild
        # after a compiler upgrade.
        set(TT_ICECC_ENV_CACHE_DIR
            "$ENV{HOME}/.cache/tt-icecc-envs"
            CACHE PATH "Where generated Icecream toolchain env tarballs are cached"
        )
        set(_tt_icecc_tarball "${TT_ICECC_ENV_CACHE_DIR}/${_tt_icecc_env_name}.tar.gz")

        if(NOT EXISTS "${_tt_icecc_tarball}")
            find_program(ICECC_CREATE_ENV_EXECUTABLE icecc-create-env)
            if(NOT ICECC_CREATE_ENV_EXECUTABLE)
                message(FATAL_ERROR "TT_ICECC_SCHEDULER_HOST is set but icecc-create-env is not installed -- can't generate the ${_tt_icecc_env_name} toolchain env")
            endif()

            message(STATUS "Generating Icecream env for ${_tt_icecc_env_name} (first use only, cached at ${_tt_icecc_tarball})")
            set(_tt_icecc_scratch "${TT_ICECC_ENV_CACHE_DIR}/.scratch-${_tt_icecc_env_name}")
            file(REMOVE_RECURSE "${_tt_icecc_scratch}")
            file(MAKE_DIRECTORY "${_tt_icecc_scratch}")

            if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
                set(_tt_icecc_create_env_args --clang "${CMAKE_C_COMPILER}" "${CMAKE_CXX_COMPILER}")
            else()
                set(_tt_icecc_create_env_args --gcc "${CMAKE_C_COMPILER}" "${CMAKE_CXX_COMPILER}")
            endif()

            execute_process(
                COMMAND ${ICECC_CREATE_ENV_EXECUTABLE} ${_tt_icecc_create_env_args}
                WORKING_DIRECTORY "${_tt_icecc_scratch}"
                RESULT_VARIABLE _tt_icecc_create_env_result
                OUTPUT_VARIABLE _tt_icecc_create_env_output
                ERROR_VARIABLE _tt_icecc_create_env_error
            )

            file(GLOB _tt_icecc_generated "${_tt_icecc_scratch}/*.tar.gz")
            list(LENGTH _tt_icecc_generated _tt_icecc_generated_count)

            if(NOT _tt_icecc_create_env_result EQUAL 0 OR NOT _tt_icecc_generated_count EQUAL 1)
                message(FATAL_ERROR
                    "TT_ICECC_SCHEDULER_HOST is set but generating the Icecream env for "
                    "${_tt_icecc_env_name} failed (icecc-create-env exit ${_tt_icecc_create_env_result}, "
                    "produced ${_tt_icecc_generated_count} tarball(s)):\n"
                    "${_tt_icecc_create_env_output}\n${_tt_icecc_create_env_error}"
                )
            endif()

            file(RENAME "${_tt_icecc_generated}" "${_tt_icecc_tarball}")
            file(REMOVE_RECURSE "${_tt_icecc_scratch}")
            message(STATUS "Icecream env for ${_tt_icecc_env_name} generated: ${_tt_icecc_tarball}")
        endif()

        set(${out_env}
            "CCACHE_PREFIX=${ICECC_EXECUTABLE}"
            "ICECC_VERSION=${_tt_icecc_tarball}"
            "ICECC_SCHEDULER_HOST=${TT_ICECC_SCHEDULER_HOST}"
            PARENT_SCOPE
        )
        message(STATUS "Icecream distributed compilation enabled (scheduler: ${TT_ICECC_SCHEDULER_HOST}, env: ${_tt_icecc_env_name})")

        # icecc and PCH don't mix: the PCH is built and consumed against the local header
        # tree, but icecc preprocesses locally and compiles remotely, so the remote node
        # can't load the (host-specific) PCH and jobs fall back to local or fail outright.
        # Icecream wins -- tell the caller to turn PCH off rather than making the dev choose.
        if(NOT CMAKE_DISABLE_PRECOMPILE_HEADERS)
            message(STATUS "Icecream is enabled -- disabling precompiled headers (incompatible with distributed compilation)")
        endif()
        set(${out_disable_pch} ON PARENT_SCOPE)
    endif()
endfunction()
