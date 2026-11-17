# ghcr.io/tenstorrent/tt-metal/exabox-tools: image for recover.sh, run_cluster_validation,
# the validation/fabric/dispatch runners and the fabric system health check. Built by
# .github/workflows/exabox-tools-image.yaml.
#
# A runtime image on plain Ubuntu: the prebuilt binaries, the source tree the JIT kernel
# build and the Python suites read, and a venv with only what those suites import. No
# compilers, no .git, no CI Python stack. The dev image the artifact was built in is only a
# donor for the two things that must match it exactly: the ULFM Open MPI the binaries link
# against and tt-smi.
#
# Build context: repo checkout with submodules, _tt-metal/ (build/ and runtime/ from
# ttm_any.tar.zst), ttnn-*.whl. Nothing from the context is COPYed into a layer as is;
# it is bind-mounted, so whatever is pruned below never reaches the image.

ARG TT_METAL_DEV_IMAGE=ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64:latest
# Keep in step with UV_IMAGE in dockerfile/docker-bake.hcl.
ARG UV_IMAGE=ghcr.io/astral-sh/uv@sha256:9a23023be68b2ed09750ae636228e903a54a05ea56ed03a934d00fe9fbeded4b

FROM ${TT_METAL_DEV_IMAGE} AS dev
FROM ${UV_IMAGE} AS uv

#############################################################
# The tt-metal tree, assembled once so it lands in a single layer with its final
# permissions. A chmod in a later layer would store every file a second time.
#############################################################

FROM mirror.gcr.io/ubuntu:22.04 AS tree

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Sources: what the JIT build (tt_metal/, ttnn/ kernels), the suites (tools/, tests/) and
# the deploy-tier pytest (conftest.py, models/, tt_py_test_utils_common/) read. Docs,
# reports, CI config and media files are never read on a host.
# build/: everything the scripts run (tools/, test/, lib/*.so), minus the tracy web UI,
# the programming examples, tt-train, static libraries and CMake bookkeeping.
RUN --mount=type=bind,target=/ctx \
    mkdir -p /out/tt-metal \
    && tar -C /ctx -cf - \
        --exclude=./_tt-metal \
        --exclude='./ttnn-*.whl' \
        --exclude='./*.tar.zst' \
        --exclude=./.github \
        --exclude=./docs \
        --exclude=./tech_reports \
        --exclude=./tt-train \
        --exclude=./infra \
        --exclude=./models/experimental \
        --exclude=./models/sample_data \
        --exclude='*.png' --exclude='*.jpg' --exclude='*.jpeg' --exclude='*.gif' \
        --exclude='*.mp4' --exclude='*.wav' --exclude='*.pth' \
        . \
      | tar -C /out/tt-metal -xf - \
    && tar -C /ctx/_tt-metal -cf - \
        --exclude=build/profiler \
        --exclude=build/programming_examples \
        --exclude=build/tt-train \
        --exclude=build/lib/cmake \
        --exclude=build/lib/pkgconfig \
        --exclude='build/lib/*.a' \
        --exclude=CMakeFiles \
        build runtime \
      | tar -C /out/tt-metal -xf - \
    && test -x /out/tt-metal/build/tools/scaleout/run_cluster_validation \
    && test -d /out/tt-metal/runtime/sfpi \
    && test -d /out/tt-metal/tt_metal/third_party/umd/tests \
    && chmod -R a+rwX /out/tt-metal

#############################################################
# Runtime image
#############################################################

FROM mirror.gcr.io/ubuntu:22.04

ARG TT_METAL_COMMIT_SHA
ARG IMAGE_VERSION=unknown

ENV DEBIAN_FRONTEND=noninteractive

LABEL org.opencontainers.image.source="https://github.com/tenstorrent/tt-metal"
LABEL org.opencontainers.image.description="Exabox operations tools: recover.sh, cluster validation, fabric system health check"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.revision="${TT_METAL_COMMIT_SHA}"
LABEL org.opencontainers.image.version="${IMAGE_VERSION}"

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Shared libraries the binaries and Open MPI link against (hwloc, numa, nsl, atomic,
# libevent) and the SFPI cross compiler the JIT kernel build runs (gmp, mpfr, mpc), and
# the tools the suites shell out to:
#   pciutils, kmod, ipmitool   chip count after reset, KMD version, FRU and BMC reset
#   sudo                       the suites' privileged steps, as the image user
#   gh, curl, ca-certificates  the tt-syseng-diag download
#   openssh-server             MPI operator workers; it brings the client that
#                              generate_rank_bindings.py's multi-host mpirun needs
#   python3-venv               ensurepip, for callers that make their own venv on top
#                              (tt-orchestration's health-check job)
#   socat                      port and socket relays when the image runs as an
#                              operations container on exabox hosts
# journalctl (systemd) is left to the pre_reboot launchers, the only tier that reads it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gh \
        ipmitool \
        kmod \
        libatomic1 \
        libevent-core-2.1-7 \
        libevent-pthreads-2.1-7 \
        libgmp10 \
        libhwloc15 \
        libmpc3 \
        libmpfr6 \
        libnsl2 \
        libnuma1 \
        openssh-server \
        pciutils \
        python3 \
        python3-venv \
        socat \
        sudo \
    && rm -rf /var/lib/apt/lists/*

# uid 1001, passwordless sudo: what the dev image gives the launchers to rely on.
# StrictModes off so MPI operator key mounts with loose permissions are accepted.
RUN useradd --uid 1001 --create-home --shell /bin/bash user \
    && usermod -aG sudo user \
    && echo 'user ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/user \
    && chmod 0440 /etc/sudoers.d/user \
    && mkdir -p /run/sshd \
    && echo "StrictModes no" >> /etc/ssh/sshd_config

# The Open MPI the artifact was linked against (libtt_metal.so's RUNPATH names this path).
COPY --from=dev --link /opt/openmpi-v5.0.7-ulfm /opt/openmpi-v5.0.7-ulfm
COPY --from=dev --link /usr/local/bin/tt-smi /usr/local/bin/tt-smi

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:/opt/openmpi-v5.0.7-ulfm/bin:${PATH}"

# torch, pytest and loguru are the deploy tier's didt pytest; versions follow
# tt_metal/python_env/requirements-dev.txt. The wheel brings ttnn's own dependencies.
# unsafe-best-match because the CPU torch index also serves numpy and friends.
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=type=bind,target=/ctx \
    uv venv --python /usr/bin/python3 "${VIRTUAL_ENV}" \
    && uv pip install --no-cache --python "${VIRTUAL_ENV}/bin/python" \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match \
        torch==2.11.0 \
        pytest==9.0.3 \
        pytest-timeout==2.4.0 \
        loguru==0.6.0 \
        -r /ctx/tools/scaleout/exabox/health_check_test_suite/test_infrastructure/requirements.txt \
        /ctx/ttnn-*.whl

COPY --from=tree --link --chown=1001:1001 /out/tt-metal /home/user/tt-metal

# UMD api_tests embed UMD_TESTS_ROOT_PATH=/work/tt_metal/third_party/umd/tests at compile time.
# The metal build runs at /work, but this image keeps sources under /home/user/tt-metal.
# COPY creates the destination directory itself 0755, whatever the source's mode, so it and
# /home/user are opened up here, for cross-user MPI runs that write generated/ at the root.
RUN mkdir -p /work/tt_metal/third_party/umd \
    && ln -sfn /home/user/tt-metal/tt_metal/third_party/umd/tests /work/tt_metal/third_party/umd/tests \
    && chmod 0777 /home/user /home/user/tt-metal

ENV PYTHONPATH=/home/user/tt-metal
ENV LD_LIBRARY_PATH=/home/user/tt-metal/build/lib
ENV TT_METAL_HOME=/home/user/tt-metal
# The installed ttnn wheel otherwise roots the runtime at its own package directory, which
# has no runtime/sfpi, so the JIT kernel build would not find the compiler.
ENV TT_METAL_RUNTIME_ROOT=/home/user/tt-metal

USER user
WORKDIR /home/user/tt-metal

# Smoke test, no hardware needed. Fails the build instead of pushing a broken image:
# every binary a script runs resolves its libraries, the SFPI compiler the JIT build runs
# compiles, the tools the suites shell out to
# are present, and the Python entry points import, including the deploy-tier pytest's
# conftest and test base. What it writes is removed again: created here as the image user,
# generated/ would not be writable by the host uids mpi-docker runs as.
RUN for bin in \
        build/tools/scaleout/run_cluster_validation \
        build/tools/scaleout/run_regen_descriptors \
        build/tools/scaleout/run_mpi_stress_test \
        build/tools/scaleout/kmd_triage \
        build/test/tt_metal/unit_tests_deployment \
        build/test/tt_metal/unit_tests_dispatch \
        build/test/tt_metal/tt_fabric/test_physical_discovery \
        build/test/tt_metal/tt_fabric/test_infra/test_tt_fabric; do \
        ldd "$bin" > /tmp/ldd.txt || { echo "ERROR: ldd failed on $bin" >&2; exit 1; }; \
        if grep "not found" /tmp/ldd.txt; then echo "ERROR: $bin has unresolved libraries" >&2; exit 1; fi; \
    done \
    && for elf in $(find runtime/sfpi -type f -perm -u+x); do \
        head -c4 "$elf" | grep -q ELF || continue; \
        if ldd "$elf" 2>/dev/null | grep "not found"; then echo "ERROR: $elf has unresolved libraries" >&2; exit 1; fi; \
    done \
    && rm -f /tmp/ldd.txt \
    && echo 'int main() { return 0; }' \
        | "$TT_METAL_RUNTIME_ROOT/runtime/sfpi/compiler/bin/riscv-tt-elf-g++" -mcpu=tt-bh -O2 -x c++ -c - -o /tmp/sfpi-smoke.o \
    && rm -f /tmp/sfpi-smoke.o \
    && for tool in lspci modinfo ipmitool gh curl sudo ssh sshd mpirun setpriv; do \
        command -v "$tool" > /dev/null || { echo "ERROR: $tool not found" >&2; exit 1; }; \
    done \
    && test "$(stat -c %a /home/user/tt-metal)" = 777 \
    && tt-smi --version \
    && ./build/tools/scaleout/run_cluster_validation --help > /dev/null \
    && TT_METAL_LOGGER_LEVEL=Fatal ./build/test/tools/scaleout/test_descriptor_merger \
    && TT_METAL_LOGGER_LEVEL=Fatal ./build/test/tools/scaleout/test_instance_filter \
    && bash -n tools/scaleout/exabox/recover.sh \
    && bash -n tools/scaleout/exabox/run_validation.sh \
    && bash -n tools/scaleout/exabox/mpi-docker \
    && bash -n tools/scaleout/kmd_triage/host_side.sh \
    && bash -n tools/scaleout/kmd_triage/device_side.sh \
    && python3 tools/scaleout/exabox/health_check_test_suite/test_infrastructure/run_health_check.py --help > /dev/null \
    && python3 tools/scaleout/exabox/health_check_test_suite/diag_runner.py --help > /dev/null \
    && python3 tests/tt_metal/tt_fabric/utils/generate_rank_bindings.py --help > /dev/null \
    && TRACY_NO_INVARIANT_CHECK=1 PYTHONDONTWRITEBYTECODE=1 \
        python3 -c 'import ttnn, pytest_timeout, conftest, tests.didt.op_test_base' \
    && python3 -m venv --system-site-packages /tmp/venv-smoke \
    && /tmp/venv-smoke/bin/python3 -m pip --version \
    && rm -rf /tmp/venv-smoke generated build/profiler /home/user/.cache

# Callers supply the command.
ENTRYPOINT []
CMD ["/bin/bash"]
