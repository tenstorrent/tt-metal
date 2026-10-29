#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Run sfpi_disasm.sh inside Ubuntu 22.04.
#
# sfpi 7.9x links against glibc 2.34, which Ubuntu 20.04 does not have. The
# container supplies it. Everything else stays on the host: --work holds the
# trees, the toolchains and the dumps, so a second run costs only the compile.
#
# The host must already have a Python environment with the test requirements.
# Its packages are used through PYTHONPATH, not through its bin/python, because
# that interpreter belongs to the host.
#
# Usage:
#   sfpi_disasm_docker.sh --work <dir> --site-packages <dir> [--stub <dir>]
#                         -- <sfpi_disasm.sh arguments>

set -euo pipefail

IMAGE="ubuntu:22.04"
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

# One writable place for everything, so the container needs no host paths of
# its own and leaves nothing behind.
ROOT=$(cd "$(dirname "$WORK")" && pwd)

PYPATH="/host$SITE"
[ -n "$STUB" ] && PYPATH="/host$STUB:$PYPATH"

docker run --rm \
    -v "$ROOT:/host$ROOT" \
    -e "PYTHONPATH=$PYPATH" \
    -e "PYTHONDONTWRITEBYTECODE=1" \
    -w "/host$ROOT" \
    "$IMAGE" bash -c '
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get -qq update
apt-get -qq install -y --no-install-recommends \
    git curl ca-certificates xz-utils python3 python3-distutils >/dev/null
exec "$@"
' bash "/host$SCRIPT" --python /usr/bin/python3 "$@"
