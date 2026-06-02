#!/bin/bash
# Queue script for missing plain MM sweeps on WH Galaxy
# Run from tt-metal root: bash models/tt_dit/docs/run_missing_mm_sweep_queue.sh

set -e
source python_env/bin/activate

echo "=========================================="
echo "Missing MM Sweep Queue — WH Galaxy"
echo "Started: $(date)"
echo "=========================================="

# --- 8x8 grid plain MM shapes ---

echo ""
echo "[1/7] Starting: MM (1024, 6144, 4608) 8x8"
echo "Time: $(date)"
pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
    -k "1024_6144_4608_8x8_mm_plain-wh_4x8" -x -s
echo "[1/7] Done: $(date)"

echo ""
echo "[2/7] Starting: MM (512, 6144, 4608) 8x8"
echo "Time: $(date)"
pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
    -k "512_6144_4608_8x8_mm_plain-wh_4x8" -x -s
echo "[2/7] Done: $(date)"

echo ""
echo "[3/7] Starting: MM (1024, 6144, 768) 8x8"
echo "Time: $(date)"
pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
    -k "1024_6144_768_8x8_mm_plain-wh_4x8" -x -s
echo "[3/7] Done: $(date)"

echo ""
echo "[4/7] Starting: MM (512, 6144, 768) 8x8"
echo "Time: $(date)"
pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
    -k "512_6144_768_8x8_mm_plain-wh_4x8" -x -s
echo "[4/7] Done: $(date)"

# --- 8x9 grid plain MM shapes ---

echo ""
echo "[5/7] Starting: MM (1024, 6144, 4608) 8x9"
echo "Time: $(date)"
pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
    -k "1024_6144_4608_8x9_mm_plain-wh_4x8" -x -s
echo "[5/7] Done: $(date)"

echo ""
echo "[6/7] Starting: MM (512, 6144, 4608) 8x9"
echo "Time: $(date)"
pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
    -k "512_6144_4608_8x9_mm_plain-wh_4x8" -x -s
echo "[6/7] Done: $(date)"

echo ""
echo "[7/7] Starting: MM (1024, 128, 768) 8x9"
echo "Time: $(date)"
pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
    -k "1024_128_768_8x9_mm_plain-wh_4x8" -x -s
echo "[7/7] Done: $(date)"

echo ""
echo "=========================================="
echo "All missing MM sweeps complete!"
echo "Finished: $(date)"
echo "Results in: sweep_results_mm.csv"
echo "=========================================="
