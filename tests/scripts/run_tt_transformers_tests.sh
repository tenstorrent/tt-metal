#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Run tt-transformers tests against the ttnn built from this tree.
#
# usage: tests/scripts/run_tt_transformers_tests.sh <pytest args>
#
# Paths in <pytest args> are relative to the tt-transformers checkout, e.g.
#   tests/scripts/run_tt_transformers_tests.sh tests/modules/mlp/test_mlp_1d.py -m "not slow"
#
# The checkout is pinned by tests/scripts/tt_transformers_ref.txt. Override with
# TT_TRANSFORMERS_REF=<sha> to try another revision, and TT_TRANSFORMERS_DIR=<dir>
# to reuse an existing checkout (it must be clean).
set -euo pipefail

tt_metal_home="${TT_METAL_HOME:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
cd "$tt_metal_home"
ref="${TT_TRANSFORMERS_REF:-$(grep -v '^#' tests/scripts/tt_transformers_ref.txt | tr -d '[:space:]')}"
checkout="${TT_TRANSFORMERS_DIR:-$tt_metal_home/generated/tt-transformers}"
reports="$tt_metal_home/generated/test_reports"
mkdir -p "$reports"

# Clone once per job. Later calls in the same job reuse the checkout.
if [ "$(git -C "$checkout" rev-parse HEAD 2>/dev/null)" != "$ref" ]; then
    rm -rf "$checkout"
    git init -q "$checkout"
    git -C "$checkout" fetch -q --depth 1 https://github.com/tenstorrent/tt-transformers.git "$ref"
    git -C "$checkout" checkout -q --detach FETCH_HEAD
fi
if [ -n "$(git -C "$checkout" status --porcelain)" ]; then
    echo "tt-transformers checkout $checkout has local changes" >&2
    exit 1
fi
git -C "$checkout" rev-parse HEAD > "$reports/tt-transformers-revision.txt"

# tt-transformers pins a released ttnn. --no-deps keeps the ttnn built from this
# tree; tt_metal/python_env/requirements-dev.txt already provides the test tools.
if ! python -c 'import tt_transformers' 2>/dev/null ||
    [ "$(python -c 'import os, tt_transformers; print(os.path.dirname(os.path.dirname(os.path.dirname(tt_transformers.__file__))))')" != "$checkout" ]; then
    uv pip install --no-deps -e "$checkout"
fi
python -c 'import jsonschema, pytz' 2>/dev/null || uv pip install 'jsonschema>=4.23,<5' 'pytz>=2024.1'

# Model demos write benchmark results to generated/benchmark_data relative to the
# working directory. Copy them to this tree's, which the e2e workflow uploads.
collect_benchmark_data() {
    if [ -d "$checkout/generated/benchmark_data" ]; then
        mkdir -p "$tt_metal_home/generated/benchmark_data"
        cp -r "$checkout/generated/benchmark_data/." "$tt_metal_home/generated/benchmark_data/"
        rm -rf "$checkout/generated/benchmark_data"
    fi
}
trap collect_benchmark_data EXIT

# Run from the checkout with its own pytest config. Both repos have a top-level
# `tests` package and a root conftest.py with the same options, so put the
# checkout first on sys.path and keep conftest discovery inside it.
cd "$checkout"
export PYTHONPATH="$checkout${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest \
    --rootdir "$checkout" -c "$checkout/pyproject.toml" --confcutdir "$checkout" \
    -o timeout=300 --junitxml "$reports/tt_transformers_$(date +%Y%m%d_%H%M%S_%N).xml" \
    "$@"
