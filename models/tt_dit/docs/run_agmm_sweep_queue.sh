#!/bin/bash
# Queue script for remaining AGMM sweeps on WH Galaxy
# Run from tt-metal root: bash models/tt_dit/docs/run_agmm_sweep_queue.sh

set -e

source python_env/bin/activate

echo "=========================================="
echo "AGMM Sweep Queue — WH Galaxy"
echo "Started: $(date)"
echo "=========================================="

# --- 1. (512, 768, 4608) QKV ---
echo ""
echo "[1/3] Starting: AGMM QKV (512, 768, 4608) 8x8"
echo "Time: $(date)"
pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
    -k "512_768_4608_8x8_agmm_qkv-wh_4x8" -x -s
echo "[1/3] Done: $(date)"

# --- 2. (1024, 768, 768) to_out ---
echo ""
echo "[2/3] Starting: AGMM to_out (1024, 768, 768) 8x8"
echo "Time: $(date)"
pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
    -k "1024_768_768_8x8_agmm_to_out-wh_4x8" -x -s
echo "[2/3] Done: $(date)"

# --- 3. (512, 768, 768) to_out ---
echo ""
echo "[3/3] Starting: AGMM to_out (512, 768, 768) 8x8"
echo "Time: $(date)"
pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
    -k "512_768_768_8x8_agmm_to_out-wh_4x8" -x -s
echo "[3/3] Done: $(date)"

echo ""
echo "=========================================="
echo "All AGMM sweeps complete!"
echo "Finished: $(date)"
echo "Results in: sweep_results_mm.csv"
echo "=========================================="
