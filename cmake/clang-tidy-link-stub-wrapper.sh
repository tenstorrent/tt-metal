#!/bin/bash
set -euo pipefail
# Linker wrapper for the clang-tidy presets.
#
# clang-tidy only analyses COMPILE edges; it never inspects link edges. Linking
# the ~277 first-party test/executable binaries is therefore pure overhead during
# analysis. To skip it we stub the link by `cmake -E touch`-ing the output binary
# instead of invoking the real linker.
#
# BUT a handful of executables built in this same build graph are *host code
# generators* (protobuf's protoc, flatbuffers' flatc, capnproto's capnp/capnpc-*)
# that ninja EXECUTES later in the build to generate .pb.cc/.pb.h/*_generated.h
# sources. Those must be linked for real, otherwise they end up as empty 0-byte
# files and the codegen edges fail with "Permission denied", stopping the build.
#
# These generator tools are all third-party packages pulled in via CPM, so their
# binaries live under the dependency build trees (_deps/ or .cpmcache/). We use
# that as the discriminator: real-link anything under a dependency tree, stub
# everything else (the first-party leaf binaries we never run during the build).
#
# Usage (from CMAKE_CXX_LINK_EXECUTABLE / CMAKE_C_LINK_EXECUTABLE):
#   clang-tidy-link-stub-wrapper.sh <TARGET> <real link command...>

TARGET="$1"
shift

# CMake passes <TARGET> relative to the build dir, so the dependency-tree
# marker can appear at the very start of the path (e.g. "_deps/protobuf-build/
# .../protoc") or in the middle of an absolute path. Match both.
case "/$TARGET" in
    */_deps/* | */.cpmcache/*)
        # Third-party host tool (protoc/flatc/capnp/...) executed during the
        # build: perform the real link.
        exec "$@"
        ;;
    *)
        # First-party leaf binary that is never executed during analysis: stub it.
        exec cmake -E touch "$TARGET"
        ;;
esac
