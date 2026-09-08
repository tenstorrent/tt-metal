#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Verify the tt-metal runtime env before a bring-up run. A stale TT_METAL_RUNTIME_ROOT is silent:
# ttnn/__init__.py only auto-detects the root for an editable install when the var is UNSET, and the
# device-less UMD reader (umd_dram_reader.cpp) resolves TT_METAL_RUNTIME_ROOT *before* TT_METAL_HOME,
# so a dangling root wins over a correct HOME. Machine provisioning (/etc/profile.d/*.sh) is a common
# source. Run this first; it exits non-zero and prints the exports to fix it.
set -u

repo="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
repo="$(cd "$repo" && pwd -P)"
fail=0
say() { printf '%s\n' "$*" >&2; }

for var in TT_METAL_HOME TT_METAL_RUNTIME_ROOT; do
    val="${!var:-}"
    if [ -z "$val" ]; then
        say "FAIL  $var is unset"
        fail=1
    elif [ ! -d "$val" ]; then
        say "FAIL  $var=$val does not exist"
        fail=1
    elif [ "$(cd "$val" && pwd -P)" != "$repo" ]; then
        say "WARN  $var=$val is not the repo you are running from ($repo)"
    fi
done

# the exact lookup umd_dram_reader.cpp does, in its order
root="${TT_METAL_RUNTIME_ROOT:-${TT_METAL_HOME:-$PWD}}"
if [ ! -d "$root/tt_metal/soc_descriptors" ]; then
    say "FAIL  $root/tt_metal/soc_descriptors missing (UMD SOC-descriptor lookup)"
    fail=1
fi
[ -d "$repo/build/lib" ] || say "WARN  $repo/build/lib missing - repo not built?"
case ":${LD_LIBRARY_PATH:-}:" in
    *":$repo/build/lib:"*) ;;
    *) say "WARN  LD_LIBRARY_PATH does not include $repo/build/lib" ;;
esac

# --- interpreter ---------------------------------------------------------------
# A bare `import ttnn` is NOT evidence of a working env: the repo has a ttnn/ directory, so from the
# repo root ANY interpreter imports it as an empty namespace package (no bfloat16, no real module)
# while torch/transformers fail. That reads as a half-broken install rather than the wrong python.
py="$(command -v python3 || true)"
if [ -z "$py" ]; then
    say "FAIL  no python3 on PATH"
    fail=1
else
    if ! "$py" -c "import ttnn; ttnn.bfloat16" >/dev/null 2>&1; then
        say "FAIL  $py: ttnn is not the built module (namespace-package shell, or not installed)"
        fail=1
        py_bad=1
    fi
    for mod in torch transformers; do
        "$py" -c "import $mod" >/dev/null 2>&1 || { say "FAIL  $py: cannot import $mod"; fail=1; py_bad=1; }
    done
    case "$py" in /usr/bin/python3|/bin/python3)
        say "WARN  $py is the system python - a venv is normally required here" ;;
    esac
fi

if [ "${py_bad:-0}" -ne 0 ]; then
    say ""
    say "Build the env before going further - setup is owned elsewhere, not by this script:"
    say "  - load the 'build-metal' skill if this box has one (build + venv + per-machine gotchas),"
    say "    otherwise see INSTALLING.md section 'Virtual Environment Setup' / create_venv.sh."
    say "  - the usual remedy is an editable install into the active venv: uv pip install -e ."
    say "    Do NOT 'export PYTHONPATH=\$TT_METAL_HOME' - that CAUSES the empty-namespace import"
    say "    rather than fixing it (the repo root holds a ttnn/ dir shadowing ttnn/ttnn/)."
fi

if [ "$fail" -ne 0 ]; then
    say ""
    say "Fix (in your shell, after /etc/profile.d has run):"
    say "  export TT_METAL_HOME=$repo"
    say '  export TT_METAL_RUNTIME_ROOT="$TT_METAL_HOME"'
    say '  export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:$LD_LIBRARY_PATH"'
    exit 1
fi
echo "runtime env ok: root=$root"
