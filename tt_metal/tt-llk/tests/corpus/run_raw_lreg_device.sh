#!/usr/bin/env bash
# Run the controlled raw-register experiments on Blackhole, sequentially.
# Usage: bash run_raw_lreg_device.sh /absolute/path/to/results
# Existing TT_LLK_EXTRA_COMPILER_OPTIONS may select a matching backend/headers;
# do not put scheduling experiments in that base string. Use a new results path.
# Requires the ordinary LLK Python/device environment; does not install/reset it.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
output=${1:?usage: run_raw_lreg_device.sh /absolute/path/to/results}
mkdir -p "$output"
output=$(cd "$output" && pwd)
cd "$here/../python_tests"
python=${PYTHON:-../.venv/bin/python}
base_flags=${TT_LLK_EXTRA_COMPILER_OPTIONS:-}
failed=0
for opt in O2 O3; do
    for scheduling in default scheduled; do
        for pass in enabled disabled; do
            name="$opt-$scheduling-$pass"
            flags="$base_flags -$opt"
            [[ $scheduling == scheduled ]] && flags+=" -fschedule-insns -fschedule-insns2"
            selection=()
            if [[ $pass == disabled ]]; then
                flags+=" -fdisable-rtl-rvtt_lreg_livein"
                # Effect markers require the pass; never call their disabled
                # results a correctness test of the implemented mechanism.
                selection=(-k 'not effects')
            else
                flags+=" -fenable-rtl-rvtt_lreg_livein"
            fi
            echo "RUN $name: $flags"
            # Separate roots retain each configuration's final build artifacts.
            if { printf 'configuration=%s\ncompiler_options=%s\n' "$name" "$flags";
                CHIP_ARCH=blackhole TT_LLK_EXTRA_COMPILER_OPTIONS="$flags" \
                RUNNER_TEMP="$output/$name-build" \
                timeout "${CASE_TIMEOUT_SECONDS:-900}" "$python" -m pytest \
                test_raw_lreg_device.py -s -q "${selection[@]}" \
                --junitxml="$output/$name.xml";
            } > "$output/$name.log" 2>&1; then
                echo "COMPLETE $name (inspect diagnostic XFAILs separately)"
            else
                rc=$?
                echo "FAILED $name rc=$rc; see $output/$name.log"
                failed=1
            fi
            tail -3 "$output/$name.log"
        done
    done
done
exit "$failed"
