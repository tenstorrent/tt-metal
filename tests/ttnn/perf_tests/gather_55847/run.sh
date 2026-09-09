#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
mode=${1:?accuracy or perf}
final=cf8684d95bd01f0f3cb3bed53163ff78690ed62c
base=89e1256c982a5b4739d173bcc446c8c748a44b40
evidence="$PWD/generated/test_reports/gather_55847"
mkdir -p "$evidence"
cp tests/ttnn/perf_tests/gather_55847/{bench.py,watchdog.py} "$evidence/"
git fetch origin "$final" "$base"
git checkout --detach "$final"
tt-smi -s > "$evidence/hardware.json"
git rev-parse HEAD > "$evidence/tested-head.txt"
export TT_METAL_OPERATION_TIMEOUT_SECONDS=30
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=20000
if [[ "$mode" == accuracy ]]; then
    python3 "$evidence/watchdog.py" "$evidence/invalid.log" 900 \
        python3 -m pytest -xv --timeout=90 \
        tests/ttnn/nightly/unit_tests/operations/data_movement/test_gather_invalid_indices.py
    python3 "$evidence/watchdog.py" "$evidence/existing.log" 1800 \
        python3 -m pytest --import-mode=importlib -xv --timeout=120 \
        tests/ttnn/unit_tests/operations/data_movement/test_gather.py \
        tests/ttnn/nightly/unit_tests/operations/data_movement/test_gather.py \
        tests/ttnn/nightly/unit_tests/operations/data_movement/test_gather_codegen_routing.py
    exit
fi
# Host-code sources are identical between these revisions. Use the same installed
# host library and replace only the two JIT readers for the base measurement.
readers=(ttnn/cpp/ttnn/operations/data_movement/gather/codegen/kernels/gather_reader.cpp
         ttnn/cpp/ttnn/operations/data_movement/gather/codegen/kernels/gather_reader_tiled.cpp)
trap 'git restore --source="$final" -- "${readers[@]}"' EXIT
for pass in fix-a base-a base-b fix-b; do
    revision=$final
    [[ "$pass" == base-* ]] && revision=$base
    git restore --source="$revision" -- "${readers[@]}"
    git diff --stat > "$evidence/$pass-source-diff.txt"
    sha256sum "${readers[@]}" > "$evidence/$pass-reader-sha256.txt"
    export TT_METAL_CACHE="$PWD/generated/gather_55847_cache-$pass"
    python3 "$evidence/watchdog.py" "$evidence/$pass-host.log" 1200 \
        python3 "$evidence/bench.py" --label "$pass" --output "$evidence/$pass-host.json"
    python3 "$evidence/watchdog.py" "$evidence/$pass-device.log" 1200 \
        python3 -m tracy -r -p -o "$evidence/$pass-profile" \
        "$evidence/bench.py" --label "$pass" --profile --output "$evidence/$pass-profile-host.json"
done
