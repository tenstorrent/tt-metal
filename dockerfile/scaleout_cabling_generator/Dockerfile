# Minimal image containing ONLY the scaleout "merge cluster configs" tool.
#
# It ships three things:
#   - tools/scaleout/cabling_generator/merge_cluster_configs.py  (pure-Python driver)
#   - build/tools/scaleout/run_cabling_generator                 (compiled binary it invokes)
#   - the two tt-metal shared libraries that binary needs        (libtt_stl, libtt-umd)
#
# It intentionally does NOT contain ttnn, the Python venv, or the rest of the
# tt-metal build tree, so it is dramatically smaller than the upstream-tests image.
#
# The build context is expected to contain:
#   - _tt-metal/  : the EXTRACTED tt-metal build artifact (ttm_any.tar.zst), providing
#                   _tt-metal/build/tools/scaleout/run_cabling_generator and _tt-metal/build/lib/*
#   - tools/      : the tt-metal source checkout (for the Python driver script)
# This mirrors how dockerfile/upstream_test_images/Dockerfile.template consumes _tt-metal/build.

# Must match the Ubuntu version the binary was built against (build-artifact platform),
# otherwise the binary can require a newer glibc than the runtime base provides. The
# workflow passes this as a build-arg derived from its `platform` input.
ARG UBUNTU_VERSION=22.04
FROM mirror.gcr.io/ubuntu:${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

LABEL org.opencontainers.image.source="https://github.com/tenstorrent/tt-metal"
LABEL org.opencontainers.image.description="Scaleout cabling generator (merge cluster configs) tool"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Runtime shared libraries required by run_cabling_generator (as reported by ldd).
# libhwloc/libnsl/libtirpc/libatomic/libgssapi are pulled in transitively by
# libtt-umd and are not present in the base ubuntu image. python3 runs the driver.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        libhwloc15 \
        libnsl2 \
        libtirpc3 \
        libatomic1 \
        libgssapi-krb5-2 \
        zlib1g \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV TT_METAL_HOME=/opt/tt-metal
WORKDIR /opt/tt-metal

# Compiled binary + only the tt-metal shared libraries it links against.
COPY _tt-metal/build/tools/scaleout/run_cabling_generator ./build/tools/scaleout/run_cabling_generator
COPY _tt-metal/build/lib/libtt_stl.so* ./build/lib/
COPY _tt-metal/build/lib/libtt-umd.so* ./build/lib/

# run_cabling_generator records a build-tree DT_RUNPATH. DT_RUNPATH has lower
# precedence than LD_LIBRARY_PATH, so pointing the loader at the libs we shipped
# in build/lib is enough to resolve them without patching the binary.
ENV LD_LIBRARY_PATH=/opt/tt-metal/build/lib

# Pure-Python driver. It resolves the repo root three directories up from its own
# location and looks for the binary under <root>/build/tools/scaleout, which is why
# the layout above must be preserved.
COPY tools/scaleout/cabling_generator/merge_cluster_configs.py ./tools/scaleout/cabling_generator/merge_cluster_configs.py

# run_cabling_generator writes transient output to <repo>/out/scaleout, relative to
# its working directory (/opt/tt-metal). Pre-create it world-writable so the tool
# works when the container is run with an arbitrary --user (e.g. the host uid/gid)
# without needing a `--tmpfs /opt/tt-metal/out` mount at runtime.
RUN mkdir -p /opt/tt-metal/out/scaleout && chmod -R 777 /opt/tt-metal/out

# Fail the build early if any required shared library is unresolved or the binary
# cannot start.
RUN ldd ./build/tools/scaleout/run_cabling_generator | tee /tmp/ldd.txt \
    && ! grep -q "not found" /tmp/ldd.txt \
    && ./build/tools/scaleout/run_cabling_generator --help >/dev/null \
    && rm -f /tmp/ldd.txt

ENTRYPOINT ["python3", "/opt/tt-metal/tools/scaleout/cabling_generator/merge_cluster_configs.py"]
CMD ["--help"]
