#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Include What You Use over the host build's compilation database.
#
#   run_host_iwyu.sh <build-dir> [source-path ...]
#
# <build-dir> must have been configured with CMAKE_EXPORT_COMPILE_COMMANDS
# (cmake --preset iwyu) and had its generated sources built
# (--target all_generated_files). Optional source paths restrict the scan to
# those files/directories; the default is the whole host tree.
#
# Report-only: recommendations land in <build-dir>/iwyu/iwyu.txt and the
# analyzer's exit status is recorded, not propagated. Run from anywhere.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <build-dir> [source-path ...]" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
cd "$repo_root"
build_dir=$(realpath "$1")
shift
out="$build_dir/iwyu"
mkdir -p "$out"

# Everything CMake compiles is host code: device/kernel sources are JIT-compiled
# at runtime by the Tensix toolchain and never enter the compilation database.
# What is left to drop is code our build compiles that is not ours: vendored
# submodules under tt_metal/third_party, CPM checkouts (default cache
# .cpmcache; CPM_SOURCE_CACHE may relocate it, in which case it already falls
# outside the host roots) and CMake's header-set verification stubs and other
# generated sources in the build tree.
python3 - "$build_dir/compile_commands.json" "$out/compile_commands.json" "$repo_root" <<'PY'
import json, os, sys

src, dst, root = sys.argv[1:4]
host_roots = [os.path.join(root, d) for d in ("tt_metal", "ttnn", "tt_stl", "tools", "tt-train", "tests")]
excluded = [os.path.join(root, d) for d in ("tt_metal/third_party", ".cpmcache", ".build")]

def under(path, parent):
    return path == parent or path.startswith(parent + os.sep)

kept = []
for entry in json.load(open(src)):
    path = os.path.realpath(os.path.join(entry["directory"], entry["file"]))
    if any(under(path, r) for r in host_roots) and not any(under(path, e) for e in excluded):
        kept.append(entry)
json.dump(kept, open(dst, "w"), indent=1)
print(f"{len(kept)} host translation units selected for IWYU")
PY

include-what-you-use --version | tee "$out/iwyu-version.txt"

# --cxx17ns: forward-declaration advice in the nested-namespace form the
#            codebase writes (namespace tt::tt_metal {), not namespace tt { namespace tt_metal {.
# --max_line_length: .clang-format ColumnLimit.
# Exit status is zero for recommendations and nonzero for analysis failures.
status=0
iwyu_tool.py -p "$out" -j "$(nproc)" "$@" -- \
  -Xiwyu --mapping_file="$repo_root/.github/iwyu-host.imp" \
  -Xiwyu --cxx17ns \
  -Xiwyu --max_line_length=120 \
  > "$out/iwyu.txt" 2>&1 || status=$?
echo "$status" > "$out/iwyu-exit-code.txt"

python3 "$script_dir/summarize_host_iwyu.py" "$out/iwyu.txt" "$status" | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"

if [ "$status" -ne 0 ]; then
  echo "::warning::IWYU reported analysis failures; see iwyu.txt in the iwyu-host-report artifact."
fi
