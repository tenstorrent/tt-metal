# ghcr.io/tenstorrent/tt-metal/exabox-tools: image for recover.sh, run_cluster_validation
# and the fabric system health check. Built by .github/workflows/exabox-tools-image.yaml.
#
# Same content as dockerfile/upstream_test_images/Dockerfile.template (build tree, runtime/,
# ttnn wheel, sources at the build SHA), minus the test-script entrypoint, plus the
# health-check requirements, the tt-syseng-diag package, OCI labels and a smoke test at the end.
#
# Build context: repo checkout, _tt-metal/ (extracted ttm_any.tar.zst), ttnn-*.whl,
# _tt-syseng-diag/tt-syseng-diag_*.deb (checksum-verified by the workflow).

# The workflow passes the dev image the build artifact was produced with.
ARG TT_METAL_DEV_IMAGE=ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest
FROM ${TT_METAL_DEV_IMAGE} AS base

ARG TT_METAL_COMMIT_SHA
ARG IMAGE_VERSION=unknown
ARG TT_SYSENG_DIAG_VERSION=unknown

ENV DEBIAN_FRONTEND=noninteractive

LABEL org.opencontainers.image.source="https://github.com/tenstorrent/tt-metal"
LABEL org.opencontainers.image.description="Exabox operations tools: recover.sh, cluster validation, fabric system health check"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.revision="${TT_METAL_COMMIT_SHA}"
LABEL org.opencontainers.image.version="${IMAGE_VERSION}"
LABEL com.tenstorrent.tt-syseng-diag.version="${TT_SYSENG_DIAG_VERSION}"

USER user

# No apt: the CI build containers cannot reach the Ubuntu mirrors. pciutils for the health
# check is installed at container start by launch-health-check.sh.

RUN git clone https://github.com/tenstorrent/tt-metal --recurse-submodules /home/user/tt-metal

WORKDIR /home/user/tt-metal

RUN test -n "${TT_METAL_COMMIT_SHA}" \
    && git fetch origin "${TT_METAL_COMMIT_SHA}" \
    && git reset --hard "${TT_METAL_COMMIT_SHA}" \
    && test "$(git rev-parse HEAD)" = "${TT_METAL_COMMIT_SHA}" \
    && git submodule update --init --recursive

ENV PYTHONPATH=/home/user/tt-metal
ENV LD_LIBRARY_PATH=/home/user/tt-metal/build/lib
ENV TT_METAL_HOME=/home/user/tt-metal

COPY _tt-metal/build/ ./build/
COPY _tt-metal/runtime/ ./runtime/
COPY ttnn-*.whl ./

# UMD api_tests embed UMD_TESTS_ROOT_PATH=/work/tt_metal/third_party/umd/tests at compile time.
# The metal build runs at /work, but this image keeps sources under /home/user/tt-metal.
RUN sudo mkdir -p /work/tt_metal/third_party/umd && \
    sudo ln -sfn /home/user/tt-metal/tt_metal/third_party/umd/tests /work/tt_metal/third_party/umd/tests

RUN uv pip install $(ls -1 *.whl)
RUN uv pip install -r tools/scaleout/exabox/health_check_test_suite/test_infrastructure/requirements.txt

# tt-syseng-diag installs tt-bh-glx-cluster-debug (the health check's QSFP phase) into
# /usr/local/bin. It declares no dependencies, so dpkg works without apt. Bind-mounted rather
# than COPY'd so the .deb itself does not stay behind in a layer.
RUN --mount=type=bind,source=_tt-syseng-diag,target=/tmp/tt-syseng-diag \
    sudo dpkg -i /tmp/tt-syseng-diag/tt-syseng-diag_*.deb

# Smoke test, no hardware needed. Fails the build instead of pushing a broken image.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN ./build/tools/scaleout/run_cluster_validation --help > /dev/null \
    && ldd ./build/tools/scaleout/run_cluster_validation | tee /tmp/ldd.txt \
    && ! grep -q "not found" /tmp/ldd.txt \
    && rm -f /tmp/ldd.txt \
    && TT_METAL_LOGGER_LEVEL=Fatal ./build/test/tools/scaleout/test_descriptor_merger \
    && TT_METAL_LOGGER_LEVEL=Fatal ./build/test/tools/scaleout/test_instance_filter \
    && bash -n tools/scaleout/exabox/recover.sh \
    && bash -n tools/scaleout/exabox/run_validation.sh \
    && bash -n tools/scaleout/exabox/mpi-docker \
    && python3 tools/scaleout/exabox/health_check_test_suite/test_infrastructure/run_health_check.py --help > /dev/null \
    && tt-bh-glx-cluster-debug --help > /dev/null \
    && TRACY_NO_INVARIANT_CHECK=1 python3 -c 'import ttnn'

# Give others full access to the user directory (for cross-user MPI runs)
RUN sudo chmod -R 777 /home/user

# Allow files with insecure permissions to be used as authorized keys by disabling SSH StrictModes
# (for MPI operator key mounts)
RUN sudo mkdir -p /run/sshd
RUN echo "StrictModes no" | sudo tee -a /etc/ssh/sshd_config

# The base image's entrypoint starts sshd and fails as a non-root user. Callers supply the command.
ENTRYPOINT []
CMD ["/bin/bash"]
