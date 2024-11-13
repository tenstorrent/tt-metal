include(FetchContent)

# Declare should be defined in the global scope
FetchContent_Declare(
    msgpack
    GIT_REPOSITORY https://github.com/msgpack/msgpack-c.git
    GIT_TAG
        cpp-6.1.0 # You can specify a version tag or branch name
)

FetchContent_GetProperties(msgpack)
FetchContent_Populate(msgpack)

set(MSGPACK_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(MSGPACK_BUILD_TESTS OFF CACHE INTERNAL "")
set(MSGPACK_BUILD_DOCS OFF CACHE INTERNAL "")
set(MSGPACK_ENABLE_CXX ON CACHE INTERNAL "")
set(MSGPACK_USE_BOOST OFF CACHE INTERNAL "")
set(MSGPACK_BUILD_HEADER_ONLY ON CACHE INTERNAL "")
set(MSGPACK_ENABLE_SHARED OFF CACHE INTERNAL "")
set(MSGPACK_ENABLE_STATIC OFF CACHE INTERNAL "")
set(MSGPACK_CXX20 ON CACHE INTERNAL "")
set(MSGPACK_NO_BOOST ON CACHE BOOL "Disable Boost in msgpack" FORCE)

FetchContent_MakeAvailable(msgpack)
