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
# (--target all_generated_files). Optional source paths (relative to the repo
# root, or absolute) restrict the scan to the translation units under them; the
# default is the whole host tree. IWYU analyzes translation units, so a
# header-only directory such as tt_metal/api selects nothing by itself: its
# headers are reported on through the .cpp files that include them (for
# tt_metal/api that is mostly tt_metal/impl).
#
# Report-only: recommendations land in <build-dir>/report/iwyu.txt and the
# analyzer's exit status is recorded, not propagated. The script itself fails
# only when there is nothing to analyze (a source path that does not exist or
# selects no translation units), since a report on zero files would otherwise
# pass for a clean run. Run from anywhere.
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
# A subdirectory of the build tree: the scoped compilation database written
# below must not replace CMake's compile_commands.json next to it.
out="$build_dir/report"
mkdir -p "$out"

# Everything CMake compiles is host code: device/kernel sources are JIT-compiled
# at runtime by the Tensix toolchain and never enter the compilation database.
# What is left to drop is code our build compiles that is not ours: vendored
# submodules under tt_metal/third_party, CPM checkouts (default cache
# .cpmcache; CPM_SOURCE_CACHE may relocate it, in which case it already falls
# outside the host roots) and CMake's header-set verification stubs and other
# generated sources in the build tree.
#
# The optional source paths are applied here as well, so that IWYU's driver
# gets a database holding exactly the selected units and nothing can select
# zero of them unnoticed. (iwyu_tool.py's own positional filter warns on
# stderr and exits 0 when a path matches no entry, which reads as a clean run.)
python3 - "$build_dir/compile_commands.json" "$out/compile_commands.json" "$repo_root" "$@" <<'PY'
import json, os, sys

src, dst, root = sys.argv[1:4]
selection = sys.argv[4:]
host_roots = [os.path.join(root, d) for d in ("tt_metal", "ttnn", "tt_stl", "tools", "tt-train", "tests")]
excluded = [os.path.join(root, d) for d in ("tt_metal/third_party", ".cpmcache", ".build")]

def under(path, parent):
    return path == parent or path.startswith(parent + os.sep)

def fail(message):
    sys.stdout.flush()  # keep the counts above the error in the job log
    print(f"::error::{message}", file=sys.stderr)
    sys.exit(1)

host = []
for entry in json.load(open(src)):
    path = os.path.realpath(os.path.join(entry["directory"], entry["file"]))
    if any(under(path, r) for r in host_roots) and not any(under(path, e) for e in excluded):
        host.append((path, entry))
print(f"{len(host)} host translation units in {src}")

if selection:
    kept = {}  # index -> entry, so overlapping paths do not analyze a unit twice
    for source in selection:
        prefix = os.path.realpath(os.path.join(root, source))
        if not os.path.exists(prefix):
            fail(f"source path '{source}' does not exist under {root}")
        matched = [i for i, (path, _) in enumerate(host) if under(path, prefix)]
        if not matched:
            fail(f"source path '{source}' contains no host translation units; "
                 "IWYU analyzes .cpp files, so select the directories whose sources include the headers of interest")
        print(f"{len(matched)} under {source}")
        kept.update((i, host[i][1]) for i in matched)
    kept = [kept[i] for i in sorted(kept)]
else:
    kept = [entry for _, entry in host]

if not kept:
    fail(f"no host translation units found in {src}")
json.dump(kept, open(dst, "w"), indent=1)
print(f"{len(kept)} translation units selected for IWYU")
PY

include-what-you-use --version | tee "$out/iwyu-version.txt"

# --cxx17ns: forward-declaration advice in the nested-namespace form the
#            codebase writes (namespace tt::tt_metal {), not namespace tt { namespace tt_metal {.
# --max_line_length: .clang-format ColumnLimit.
# Exit status is zero for recommendations and nonzero for analysis failures.
status=0
iwyu_tool.py -p "$out" -j "$(nproc)" -- \
  -Xiwyu --mapping_file="$repo_root/.github/iwyu-host.imp" \
  -Xiwyu --cxx17ns \
  -Xiwyu --max_line_length=120 \
  > "$out/iwyu.txt" 2>&1 || status=$?
echo "$status" > "$out/iwyu-exit-code.txt"
echo "IWYU report written to $out/iwyu.txt"

python3 "$script_dir/summarize_host_iwyu.py" --rewrite-c-headers "$out/iwyu.txt"

python3 "$script_dir/summarize_host_iwyu.py" "$out/iwyu.txt" "$status" | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"

if [ "$status" -ne 0 ]; then
  echo "::warning::IWYU reported analysis failures; see iwyu.txt in the iwyu-host-report artifact."
fi
