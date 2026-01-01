# Cursor Cloud Agent base for tt-metal.
#
# The published dev image already has the toolchain used to build TT-Metalium
# (clang 20, CMake, Ninja, ccache, SFPI, OpenMPI, and the Python venv).
# Cursor checks the repository out itself; do not COPY the source tree in.
FROM ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-24.04-dev-amd64:latest

# The dev image entrypoint starts sshd before exec'ing the command. sshd needs
# root, and Cloud Agents run commands directly, so drop it.
ENTRYPOINT []

# safe.directory is set only in root's global gitconfig. Agents run as the
# non-root `user` account below, including against checkouts they do not own.
# WORKDIR creates /workspace as root, which that account cannot write.
RUN git config --system --add safe.directory '*' \
    && mkdir -p /workspace \
    && chown user:user /workspace

# `user` (uid 1001) is the account the dev image creates with passwordless
# sudo. The image's `ubuntu` account cannot sudo without a password.
USER user
WORKDIR /workspace
