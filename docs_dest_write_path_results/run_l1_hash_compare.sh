#!/usr/bin/env bash
# Runs writer-only path B (Float32, 4 tiles per bank) once per Unpacker 2 l1_client counter, with the L1 hash off and
# on, and keeps each run under logs/l1_client/hash_<0|1>/sel_<selection>/.
#
# Usage, from tt_metal/tt-llk/tests/python_tests with the emulator environment set:
#   ../../../../docs_dest_write_path_results/run_l1_hash_compare.sh
# Then compare:
#   python3 ../../../../docs_dest_write_path_results/collect_l1_client.py ../../../../docs_dest_write_path_results/logs/l1_client/hash_0
#   python3 ../../../../docs_dest_write_path_results/collect_l1_client.py ../../../../docs_dest_write_path_results/logs/l1_client/hash_1
set -u
here=$(cd "$(dirname "$0")" && pwd)
test_id='quasar/perf_dest_write_path_exhaustive_quasar.py::test_perf_dest_write_path_writer_only_quasar[path:B_DONE-tiles_per_bank:4-formats:(<DataFormat.Float32: Float32/4B>, <DataFormat.Float32: Float32/4B>)-run_types:[<PerfRunType.L1_TO_L1: 1>]]'

# Unpacker 2 is subports 21-24 (one interface, 4 lanes); selection = subport*8 + event.
# Events 2/3 (issue stall/work) are per SBank, indexed by the subport, so 21-24 cover all 4 SBanks.
# Events 4/5/6 (flex stall/work, pending requests) are per lane: lane 0 only.
sels=(170 171 178 179 186 187 194 195 172 173 174)

for hash in 0 1; do
    out=$here/logs/l1_client/hash_$hash
    for sel in "${sels[@]}"; do
        dest=$out/sel_$sel
        mkdir -p "$dest"
        rm -f /tmp/tt-llk-build/dest_write_path_bank_stamps.jsonl
        PERF_L1_HASH=$hash LLK_PERF_L1_CLIENT_SEL=$sel pytest --timeout=600 -rN --run-simulator --enable-perf-counters \
            "$test_id" >"$dest/pytest.log" 2>&1
        status=$?
        echo "hash $hash sel $sel exit=$status $(tail -1 "$dest/pytest.log")"
        if [ $status -ne 0 ]; then
            echo "stopping: see $dest/pytest.log"
            exit $status
        fi
        latest=$(readlink -f ../../perf_data/latest)
        cp "$latest"/perf_dest_*/perf_dest_*_quasar.csv "$dest"/ 2>/dev/null
        cp /tmp/tt-llk-build/dest_write_path_bank_stamps.jsonl "$dest"/ 2>/dev/null
    done
done
