#!/bin/bash
# Queue script for MM+RS (fused matmul + reduce scatter) sweeps on WH Galaxy
# Run from tt-metal root: bash models/tt_dit/docs/run_mmrs_sweep_queue.sh

set -e
source python_env/bin/activate
export SWEEP_CSV="sweep_results_mmrs.csv"

echo "=========================================="
echo "MM+RS Sweep Queue — WH Galaxy"
echo "Output CSV: $SWEEP_CSV"
echo "Started: $(date)"
echo "=========================================="

echo ""
echo "[1/4] Starting: MM+RS (1024, 2304, 6144) 8x7"
echo "Time: $(date)"
pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
    -k "1024_2304_6144_8x7_mmrs_mmrs-wh_4x8" -x -s
echo "[1/4] Done: $(date)"

echo ""
echo "[2/4] Starting: MM+RS (512, 2304, 6144) 8x7"
echo "Time: $(date)"
pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
    -k "512_2304_6144_8x7_mmrs_mmrs-wh_4x8" -x -s
echo "[2/4] Done: $(date)"

echo ""
echo "[3/4] Starting: MM+RS (1024, 3072, 6144) 8x7"
echo "Time: $(date)"
pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
    -k "1024_3072_6144_8x7_mmrs_mmrs-wh_4x8" -x -s
echo "[3/4] Done: $(date)"

echo ""
echo "[4/4] Starting: MM+RS (512, 3072, 6144) 8x7"
echo "Time: $(date)"
pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep \
    -k "512_3072_6144_8x7_mmrs_mmrs-wh_4x8" -x -s
echo "[4/4] Done: $(date)"

echo ""
echo "=========================================="
echo "All MM+RS sweeps complete!"
echo "Finished: $(date)"
echo "Results in: $SWEEP_CSV"
echo "=========================================="
