#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# check_env.sh: the HC_SETUP command run_hc2.sh gives the hc2 core. It checks tt-metal's own python_env and installs
# nothing (no pip, no wheel): `import ttnn` must work in $PYTHON_ENV_DIR (default $TT_METAL_HOME/python_env) and
# resolve to a file inside $TT_METAL_HOME. Exit 0 when it does, 3 with the reason otherwise.
set -euo pipefail

: "${TT_METAL_HOME:?TT_METAL_HOME is not set (start hc2 through glue/run_hc2.sh)}"
env_dir=${PYTHON_ENV_DIR:-$TT_METAL_HOME/python_env}
py=$env_dir/bin/python
build="./build_metal.sh and ./create_venv.sh, or tools/scaleout/exabox/build_metal_exabox.sh"

fail() {
    echo "hc2 env check: $*" >&2
    exit 3
}

[ -x "$py" ] || fail "no python at $py: build tt-metal and its python_env first ($build)"
# PYTHONPATH and the working dir are cleared so the import shows where the env itself takes ttnn from
f=$(cd / && env -u PYTHONPATH "$py" -c 'import os, ttnn; print(os.path.realpath(ttnn.__file__))' 2>&1) \
    || fail "import ttnn fails in $py: $(tail -n 1 <<< "$f")"
f=$(tail -n 1 <<< "$f")
home=$(cd "$TT_METAL_HOME" && pwd -P)
case $f in
    "$home"/*) echo "hc2 env check: ttnn from $f (python $py)" ;;
    *) fail "ttnn imports from $f, not from TT_METAL_HOME $home ($build)" ;;
esac
