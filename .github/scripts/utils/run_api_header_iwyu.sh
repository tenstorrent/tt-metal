#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Include What You Use on tt_metal/api headers, one header at a time.
#
#   run_api_header_iwyu.sh <build-dir> --all
#   run_api_header_iwyu.sh <build-dir> <header> [<header> ...]
#
# <build-dir> must have been configured with the `iwyu` preset (which turns on
# CMAKE_VERIFY_INTERFACE_HEADER_SETS and CMAKE_EXPORT_COMPILE_COMMANDS) and have
# had the `metalium_GeneratedHeaders` target built.
#
# IWYU analyzes translation units, and a header-only directory has none. CMake's
# header-set verification already generates one per public header, containing
#   #include <tt-metalium/foo.hpp> // IWYU pragma: associated
# and puts it in the compilation database with the real compile flags. The
# pragma makes IWYU report on foo.hpp itself, so this script only selects those
# entries; nothing is compiled and no flags are assembled by hand.
#
# Headers are given relative to the repo root (or absolute). A header with no
# verification TU (not in the tt_metal public header set) is skipped with a
# warning. Report-only, like run_host_iwyu.sh: recommendations land in
# <build-dir>/report-api/iwyu.txt and the analyzer's exit status is recorded,
# not propagated.
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <build-dir> --all | <header> [<header> ...]" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(git -C "$script_dir" rev-parse --show-toplevel)
cd "$repo_root"
build_dir=$(realpath "$1")
shift
out="$build_dir/report-api"
mkdir -p "$out"

python3 - "$build_dir" "$out/compile_commands.json" "$repo_root" "$@" <<'PY'
import json, os, sys

build, dst, root = sys.argv[1:4]
args = sys.argv[4:]
stub_dir = os.path.join(build, "tt_metal", "tt_metal_verify_interface_header_sets") + os.sep
api_dir = os.path.join(root, "tt_metal", "api") + os.sep

stubs = {}
for entry in json.load(open(os.path.join(build, "compile_commands.json"))):
    path = os.path.realpath(os.path.join(entry["directory"], entry["file"]))
    if path.startswith(stub_dir):
        stubs[path[len(stub_dir):-len(".cxx")]] = entry
if not stubs:
    print(f"::error::no header verification TUs under {stub_dir}; was the build configured with the iwyu preset?",
          file=sys.stderr)
    sys.exit(1)

if args == ["--all"]:
    kept = [stubs[name] for name in sorted(stubs)]
else:
    kept = []
    for header in args:
        path = os.path.realpath(os.path.join(root, header))
        name = path[len(api_dir):] if path.startswith(api_dir) else None
        if name in stubs:
            kept.append(stubs[name])
        else:
            print(f"::warning::{header} has no header verification TU; skipped", file=sys.stderr)

print(f"{len(kept)} header(s) selected for IWYU")
if not kept:
    print("::warning::no header selected; nothing analyzed", file=sys.stderr)
json.dump(kept, open(dst, "w"), indent=1)
PY

if [ "$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$out/compile_commands.json")" -eq 0 ]; then
  exit 0
fi

include-what-you-use --version | tee "$out/iwyu-version.txt"

# Same options as run_host_iwyu.sh.
status=0
iwyu_tool.py -p "$out" -j "$(nproc)" -- \
  -Xiwyu --mapping_file="$repo_root/.github/iwyu-host.imp" \
  -Xiwyu --cxx17ns \
  -Xiwyu --max_line_length=120 \
  > "$out/iwyu.txt" 2>&1 || status=$?
echo "$status" > "$out/iwyu-exit-code.txt"
echo "IWYU report written to $out/iwyu.txt"

python3 "$script_dir/summarize_host_iwyu.py" "$out/iwyu.txt" "$status" \
  "Include What You Use (tt_metal API headers, report-only)" iwyu-api-headers-report | tee -a "${GITHUB_STEP_SUMMARY:-/dev/null}"

if [ "$status" -ne 0 ]; then
  echo "::warning::IWYU reported analysis failures; see iwyu.txt in the iwyu-api-headers-report artifact."
fi
