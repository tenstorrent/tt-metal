#!/usr/bin/env bash
# Serial Blackhole silicon acceptance for the test-only Reduce-SDPA A/B.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$HERE/../../../.." && pwd)}"
LLK_TESTS="$TT_METAL_HOME/tt_metal/tt-llk/tests"
PYTEST="${PYTEST:-$LLK_TESTS/.venv/bin/pytest}"
OBJDUMP="${OBJDUMP:-$LLK_TESTS/sfpi/compiler/bin/riscv-tt-elf-objdump}"
PYTEST_WORKERID_PLUGIN_DIR="${PYTEST_WORKERID_PLUGIN_DIR:-/localdev/nkapre/sfpi-gcc-lreg-artifacts}"
OUT="${1:-/localdev/nkapre/reduce-sdpa-bh-silicon-$(date -u +%Y%m%dT%H%M%SZ)}"
PROFILE_KEY=test_sfpu_reduce_sdpa
PROFILE_DIR="$LLK_TESTS/../perf_data/$PROFILE_KEY"

mkdir -p "$OUT"

run_pytest() {
    local run_dir=$1
    shift
    mkdir -p "$run_dir/temp"
    (
        cd "$LLK_TESTS"
        PYTHONPATH="$PYTEST_WORKERID_PLUGIN_DIR${PYTHONPATH:+:$PYTHONPATH}" \
            CHIP_ARCH=blackhole TT_METAL_DISABLE_SFPLOADMACRO=1 \
            RUNNER_TEMP="$run_dir/temp" \
            "$PYTEST" -q -s -o addopts='' -p pytest_workerid_plugin --timeout=600 "$@"
    ) > "$run_dir/run.log" 2>&1
}

archive_pack_elf() {
    local run_dir=$1
    local archive_dir="$run_dir/elf"
    local elf hash
    mkdir -p "$archive_dir"
    elf="$(find "$run_dir/temp" -name pack.elf -type f | sort | tail -1)"
    test -n "$elf" || { echo "no pack.elf under $run_dir/temp" >&2; exit 1; }
    hash="$(sha256sum "$elf" | awk '{print $1}')"
    cp "$elf" "$archive_dir/$hash.pack.elf"
    "$OBJDUMP" -D -C "$elf" > "$archive_dir/$hash.pack.objdump"
    find "$run_dir/temp" -name build.h -type f -exec cp '{}' "$archive_dir/$hash.build.h" \;
    printf '%s  %s\n' "$hash" "$elf" > "$archive_dir/pack-elf.sha256"
}

copy_profile_rows() {
    local run_dir=$1
    test -f "$PROFILE_DIR/$PROFILE_KEY.csv"
    test -f "$PROFILE_DIR/$PROFILE_KEY.post.csv"
    cp "$PROFILE_DIR/$PROFILE_KEY.csv" "$run_dir/raw.csv"
    cp "$PROFILE_DIR/$PROFILE_KEY.post.csv" "$run_dir/post.csv"
    rg 'REDUCE_SDPA_DEVICE_PROFILE' "$run_dir/run.log" > "$run_dir/device-profile.txt"
}

{
    printf 'tt_metal_head\t'; git -C "$TT_METAL_HOME" rev-parse HEAD
    printf 'sfpi_target\t'; readlink -f "$LLK_TESTS/sfpi"
    printf 'compiler_sha256\t'; sha256sum "$LLK_TESTS/sfpi/compiler/bin/riscv-tt-elf-g++" | awk '{print $1}'
    printf 'objdump_sha256\t'; sha256sum "$OBJDUMP" | awk '{print $1}'
    printf 'host\t'; hostname
    printf 'utc_start\t'; date -u +%Y-%m-%dT%H:%M:%SZ
} > "$OUT/provenance.tsv"
"$LLK_TESTS/sfpi/compiler/bin/riscv-tt-elf-g++" --version > "$OUT/compiler-version.txt"
tt-smi -s > "$OUT/tt-smi.json"
git -C "$TT_METAL_HOME" status --short --branch > "$OUT/git-status.txt"

correctness="$OUT/correctness"
run_pytest "$correctness" \
    'python_tests/test_sfpu_reduce_sdpa.py::test_sfpu_reduce_sdpa[formats:Float16_b->Float16_b-dest_acc:No-mathop:ReduceColumn-reduce_pool:Max-input_dimensions:[512, 64]-reduce_impl:0]' \
    'python_tests/test_sfpu_reduce_sdpa.py::test_sfpu_reduce_sdpa[formats:Float16_b->Float16_b-dest_acc:No-mathop:ReduceColumn-reduce_pool:Max-input_dimensions:[512, 64]-reduce_impl:1]'
rg '2 passed' "$correctness/run.log"

for impl in handwritten_replay generated_sfpi; do
    selector=0
    test "$impl" = generated_sfpi && selector=1
    for repetition in 1 2 3; do
        run_dir="$OUT/profile-$impl-r$repetition"
        run_pytest "$run_dir" \
            "python_tests/test_sfpu_reduce_sdpa.py::test_sfpu_reduce_sdpa_device_profile[$selector-$impl]"
        copy_profile_rows "$run_dir"
        archive_pack_elf "$run_dir"
    done
done

find "$OUT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$OUT/SHA256SUMS"
printf 'evidence\t%s\n' "$OUT"
