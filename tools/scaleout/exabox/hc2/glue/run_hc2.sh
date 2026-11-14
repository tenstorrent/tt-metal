#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# run_hc2.sh: entry point of the hc2 galaxy health check. Runs the hc2 core (core/run.sh, a copy of tray-test's
# traytest/hc, see ../SYNC) against this tt-metal checkout. Arguments go to core/run.sh unchanged, for example:
#   tools/scaleout/exabox/hc2/glue/run_hc2.sh                 plain run (about 30 min)
#   tools/scaleout/exabox/hc2/glue/run_hc2.sh --mode=full     the same stages, longer
#   tools/scaleout/exabox/hc2/glue/run_hc2.sh --stub          fake 32-chip box, no device: proves the code paths only
#
# What it sets before it starts the core:
#   TT_METAL_HOME     this checkout (four levels above hc2/), whatever the caller had
#   UB0_PYTHON        the checkout's own python_env (create_venv.sh or build_metal_exabox.sh); PYTHON_ENV_DIR moves it
#   HC_SETUP          glue/check_env.sh: checks that `import ttnn` resolves inside TT_METAL_HOME; installs nothing
#   H3_TT_METAL_BUILD a view of this checkout for synthetic H3 (no pinned tt-metal sha, no pip, no wheel)
#   TRAY_RUN_PARENT   where run folders go: HC2_OUT_DIR, default $TT_METAL_HOME/generated/hc2
#   max-util          the checkout's build/test/didt/unit_tests_didt_max_util when built and the core takes it
#                     (TRAY_MAXUTIL_BIN), else FLEET_MAXUTIL_SKIP, so the core skips it with its WARNING
# HC_HOLDFILE, HC_HOLD and HC_SITE_FILE are not set here (hold scripts and site data are off unless the caller sets them).
# HC2_CORE names another core (a core/ copy, or a tray-test checkout with traytest/run.sh).
# Resets in the core are tt-smi -glx_reset with the root-port rescan; nothing here runs tt-smi -r.
set -euo pipefail

die() {
    echo "run_hc2.sh: $*" >&2
    exit 2
}

GLUE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
HC2=$(dirname "$GLUE")
TT_METAL_HOME=$(cd "$HC2/../../../.." && pwd)
[ -f "$TT_METAL_HOME/create_venv.sh" ] || die "$TT_METAL_HOME is not a tt-metal checkout (no create_venv.sh)"
export TT_METAL_HOME

CORE=${HC2_CORE:-$HC2/core}
if [ -f "$CORE/run.sh" ]; then
    RUN_DIR=$CORE
elif [ -f "$CORE/traytest/run.sh" ]; then
    RUN_DIR=$CORE/traytest
else
    die "no hc2 core at $CORE (no run.sh): import it with tray-test's traytest/tools/hc2_sync.sh, or set HC2_CORE"
fi
RUN_DIR=$(cd "$RUN_DIR" && pwd)

stub=0
for a in "$@"; do [ "$a" = --stub ] && stub=1; done
# A core without the HC_SETUP hook runs its own setup.sh, which pip-installs ttnn: hc2 never does that on a real box.
if [ "$stub" = 0 ] && ! grep -qE '\$\{?HC_SETUP' "$RUN_DIR/run.sh"; then
    die "the core at $RUN_DIR has no HC_SETUP hook and would pip-install ttnn; import a core with the HC_* hooks"
fi

PYTHON_ENV_DIR=${PYTHON_ENV_DIR:-$TT_METAL_HOME/python_env}
export PYTHON_ENV_DIR UB0_PYTHON=$PYTHON_ENV_DIR/bin/python
export HC_SETUP=${HC_SETUP:-$GLUE/check_env.sh}

OUT_PARENT=${HC2_OUT_DIR:-$TT_METAL_HOME/generated/hc2}
mkdir -p "$OUT_PARENT"
export TRAY_RUN_PARENT=${TRAY_RUN_PARENT:-$(cd "$OUT_PARENT" && pwd)}

# Synthetic H3 takes a build laid out as <dir>/{READY,python_env,tt-metal}: a view of links to this checkout, so it
# never falls back to the core's pinned tt-metal sha or its wheel route
if [ -z "${H3_TT_METAL_BUILD:-}" ]; then
    H3_TT_METAL_BUILD=$TRAY_RUN_PARENT/.h3-build
    if [ "$stub" = 0 ]; then
        mkdir -p "$H3_TT_METAL_BUILD"
        ln -sfn "$TT_METAL_HOME" "$H3_TT_METAL_BUILD/tt-metal"
        ln -sfn "$PYTHON_ENV_DIR" "$H3_TT_METAL_BUILD/python_env"
        : > "$H3_TT_METAL_BUILD/READY"
    fi
    export H3_TT_METAL_BUILD
fi

# max-util: the checkout's own binary, never the core's pinned prebuilt archive
MU_BIN=$TT_METAL_HOME/build/test/didt/unit_tests_didt_max_util
if [ -z "${FLEET_MAXUTIL_SKIP:-}" ] && [ -z "${TRAY_MAXUTIL_BIN:-}" ]; then
    if [ ! -x "$MU_BIN" ]; then
        export FLEET_MAXUTIL_SKIP="unit_tests_didt_max_util is not built in $TT_METAL_HOME (build tt-metal with its tests)"
    elif grep -qs TRAY_MAXUTIL_BIN "$RUN_DIR/fleet/maxutil.py"; then
        export TRAY_MAXUTIL_BIN=$MU_BIN
    else
        export FLEET_MAXUTIL_SKIP="this hc2 core runs only its pinned prebuilt max-util archive, not $MU_BIN"
    fi
fi

cd "$RUN_DIR"
exec bash ./run.sh "$@"
