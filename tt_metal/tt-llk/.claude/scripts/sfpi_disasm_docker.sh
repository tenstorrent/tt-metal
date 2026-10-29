#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Run sfpi_disasm.sh inside Ubuntu 22.04 (see sfpi_disasm.Dockerfile).
#
# sfpi 7.9x links against glibc 2.34, which Ubuntu 20.04 does not have. The
# container supplies it. Everything else stays on the host: --work holds the
# trees, the toolchains and the dumps, so a second run costs only the compile.
#
# The work tree is mounted at its own path, not under a prefix. Every path the
# inner script is given then means the same thing on both sides, and the work
# survives the container.
#
# The host must already have a Python environment with the test requirements.
# Its packages are used through PYTHONPATH, not through its bin/python, because
# that interpreter belongs to the host.
#
# Usage:
#   sfpi_disasm_docker.sh --work <dir> --site-packages <dir> [--stub <dir>]
#                         --script <sfpi_disasm.sh>
#                         -- <sfpi_disasm.sh arguments>

set -euo pipefail

IMAGE="llk-disasm:22.04"
WORK="" SITE="" STUB="" SCRIPT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --image) IMAGE=$2; shift 2 ;;
        --work) WORK=$2; shift 2 ;;
        --site-packages) SITE=$2; shift 2 ;;
        --stub) STUB=$2; shift 2 ;;
        --script) SCRIPT=$2; shift 2 ;;
        --) shift; break ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[ -n "$WORK" ] && [ -n "$SITE" ] && [ -n "$SCRIPT" ] || {
    echo "usage: see the header" >&2; exit 2; }

mkdir -p "$WORK"
ROOT=$(cd "$(dirname "$WORK")" && pwd)

PYPATH="$SITE"
[ -n "$STUB" ] && PYPATH="$STUB:$PYPATH"

# Write as the caller, so nothing the container leaves behind needs root to
# clean up.
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$ROOT:$ROOT" \
    -e "PYTHONPATH=$PYPATH" \
    -e "PYTHONDONTWRITEBYTECODE=1" \
    -e "HOME=$ROOT" \
    -w "$ROOT" \
    "$IMAGE" bash -c '
set -e
command -v git >/dev/null || {
    echo "[ERROR] the image has no git. Use one that does, or pre-build it." >&2
    exit 1
}
exec "$@"
' bash "$SCRIPT" --python /usr/bin/python3 "$@"
