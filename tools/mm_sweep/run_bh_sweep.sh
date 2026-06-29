#!/bin/bash
# Hardened single-card BH sweep runner: repeatedly invokes the resumable driver. The driver exits 2 on a
# hang (after cleanly checkpointing, via TT_METAL_OPERATION_TIMEOUT_SECONDS -> no kill -9 / no PCIe wedge);
# we tt-smi -r and resume. Exits 0 when all configs are attempted. Mirrors run_safe_pytest's reset-on-hang.
#
# Usage: tools/mm_sweep/run_bh_sweep.sh [ckpt.json] [max_rounds]
#   BH_SHAPES="MxKxN,..." to override shapes. Start fresh: rm the ckpt first.
set -u
REPO="/localdev/cglagovich/tt-metal"
cd "$REPO"
source /home/cglagovich/bh_env.sh >/dev/null 2>&1
source python_env/bin/activate >/dev/null 2>&1

CKPT="${1:-tools/mm_sweep/bh_sweep_ckpt.json}"
MAX_ROUNDS="${2:-400}"           # one group per round; ~33 groups + reset retries
OP_TIMEOUT="${OP_TIMEOUT:-15}"   # dispatch-layer hang timeout (s); ops here run <20ms so 15s is safe

# Driver exit codes: 0 = all groups done; 10 = group done, more remain (re-invoke, NO reset);
#                    2 = hang (reset + resume); other = unexpected (reset + retry).
for r in $(seq 1 "$MAX_ROUNDS"); do
    MM_CLOCK_HZ=1.35e9 TT_METAL_DEVICE_PROFILER=1 TT_METAL_OPERATION_TIMEOUT_SECONDS="$OP_TIMEOUT" \
        python tools/mm_sweep/bh_sweep_driver.py "$CKPT"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "===== SWEEP COMPLETE (round $r) ====="
        exit 0
    elif [ "$rc" -eq 10 ]; then
        :  # group done, more remain; loop without reset
    elif [ "$rc" -eq 2 ]; then
        echo "----- round $r: hang -> tt-smi -r + resume -----"
        tt-smi -r >/dev/null 2>&1
        sleep 3
    else
        echo "----- round $r: driver exit rc=$rc -> reset + retry -----"
        tt-smi -r >/dev/null 2>&1
        sleep 3
    fi
done
echo "===== gave up after $MAX_ROUNDS rounds (check $CKPT) ====="
exit 1
