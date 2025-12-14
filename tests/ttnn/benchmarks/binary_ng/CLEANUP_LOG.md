# Cleanup Log - Binary NG Benchmarks

**Date**: November 13, 2025
**Action**: Removed all old, out-of-date files

---

## Files Removed

### Old Documentation (24 files)
- ANALYSIS.md
- BIDIRECTIONAL_BROADCAST_ANALYSIS.md
- COLUMN_BROADCAST_ANALYSIS.md
- CONTINUE_HERE.md
- CURRENT_STATUS.md
- EXPORT.md
- FINAL_ANALYSIS_QUICK_REFERENCE.txt
- FINAL_STATUS.md
- HOW_TO_WRITE_SINGLE_TEST.md
- KERNEL_DURATION_EXTRACTION.md
- MAX_ABC_ANALYSIS.md
- MAX_ABC_CSV_FILES.md
- NEXT_CHAT_CONTINUE.md
- NEXT_SESSION.md
- NO_BROADCAST_ANALYSIS.md
- QUICK_START_NEXT_CHAT.txt
- README.md (old version)
- ROW_BROADCAST_ANALYSIS.md
- ROW_BROADCAST_CONCLUSIVE_ANALYSIS.md
- ROW_BROADCAST_CORRECTED_ANALYSIS.md
- ROW_BROADCAST_FINAL_CONCLUSIVE_ANALYSIS.md
- SESSION_SUMMARY.md
- SHARDING_COMBINATIONS.md
- START_HERE.txt
- STATUS.md

### Old CSV Results (9 files)
- results/compute_grid_no_broadcast_height_height.csv
- results/compute_grid_row_broadcast_interleaved_block.csv
- results/compute_grid_row_broadcast_interleaved_height.csv
- results/compute_grid_row_broadcast_interleaved_interleaved.csv
- results/compute_grid_row_broadcast_interleaved_width.csv
- results/compute_grid_row_broadcast_width_block.csv
- results/compute_grid_row_broadcast_width_height.csv
- results/compute_grid_row_broadcast_width_interleaved.csv
- results/compute_grid_row_broadcast_width_width.csv

### Old Analysis Text Files (4 files)
- results/max_abc_analysis_summary.txt
- results/profiling_analysis_report.txt
- results/row_broadcast_height_interleaved_summary.txt
- results/strategy_comparison_analysis.txt

### Old Python Scripts (18 files)
- analyze_grid_results.py
- analyze_max_abc.py
- analyze_strategy_comparison.py
- benchmark_grid_selection.py
- compare_all_strategies.py
- merge_profiler_data.py
- parse_worker_grid.py
- run_all_row_broadcast.py
- run_argument_swap_profiling.py
- run_bidirectional_profiling.py
- run_broadcast_swap_profiling.py
- run_core_selection_profiling.py
- run_mixed_sharding_profiling.py
- run_profiling_analysis.py
- run_row_broadcast_compute_grid.py
- run_row_broadcast_profiling.py
- run_sharded_cores_profiling.py
- visualize_results.py

### Old Test Files (13 files)
- test_all_row_broadcast.py
- test_argument_swap.py
- test_bidirectional_broadcast.py
- test_broadcast_swap.py
- test_col_broadcast.py
- test_core_selection.py
- test_grid_impact.py
- test_mixed_sharding.py
- test_row_broadcast_focused.py
- test_row_broadcast_height_height.py
- test_row_broadcast.py
- test_sharded_cores.py
- test_small_validation.py

### Old Log Files (4 files)
- profiling_run.log
- row_broadcast_max_abc_run.log
- row_broadcast_with_timing.log
- strategy_comparison_report.txt

### Old Shell Scripts (1 file)
- run_all_benchmarks.sh

### Old C++ Examples (1 file)
- example_heuristic_implementation.cpp

**Total Removed**: 74 files

---

## Files Kept (Current Session)

### Documentation (7 files) ✅
- API_REFERENCE.md - Complete API reference for tensor sharding
- BLOCK_SHARDING_FIX.md - Block sharding grid validation fix
- QUICK_REFERENCE.md - Quick commands and common tasks
- README_SESSION.md - Navigation hub for all documentation
- SESSION_EXPORT.md - Full session export documentation
- STRATEGY_COMPARISON_ANALYSIS.md - Detailed max_ab vs max_abc analysis
- STRATEGY_COMPARISON_SUMMARY.txt - Quick visual comparison summary

### Code (5 files) ✅
- __init__.py - Package init file
- compare_strategies.py - Script to compare max_ab vs max_abc strategies
- conftest.py - Pytest configuration
- example_single_test.py - Main benchmark test file
- run_single_strategy.sh - Shell script to run benchmarks

### Results (2 files) ✅
- results/example_multiple_ops_max_ab_20251113_013450.csv - Latest max_ab results
- results/example_multiple_ops_max_abc_20251113_062946.csv - Latest max_abc results

**Total Kept**: 14 files
**Added Later**: 7 files (visualizations + script)

---

## Files Added After Cleanup

### Visualization Files (7 files) 🆕
- visualizations/scatter_kernel_time_comparison.png
- visualizations/bar_percentage_differences.png
- visualizations/histogram_time_differences.png
- visualizations/heatmap_sharding_comparison.png
- visualizations/compute_cores_analysis.png
- visualizations/visualization_summary.txt
- visualize_comparison.py

---

## Summary

- **Removed**: 74 old/outdated files
- **Kept**: 14 current files from this session
- **Cleanup Ratio**: 84% reduction in file count
- **Result**: Clean, organized directory with only current, relevant files

All old analysis results, test files, and documentation from previous sessions have been removed. Only the current session's work remains.

---

## Current Directory Structure

```
tests/ttnn/benchmarks/binary_ng/
├── Documentation (7 files)
│   ├── README_SESSION.md          ← Start here
│   ├── QUICK_REFERENCE.md
│   ├── SESSION_EXPORT.md
│   ├── API_REFERENCE.md
│   ├── BLOCK_SHARDING_FIX.md
│   ├── STRATEGY_COMPARISON_ANALYSIS.md
│   └── STRATEGY_COMPARISON_SUMMARY.txt
│
├── Code (5 files)
│   ├── example_single_test.py     ← Main test
│   ├── compare_strategies.py      ← Strategy comparison
│   ├── conftest.py                ← Pytest config
│   ├── run_single_strategy.sh     ← Run script
│   └── __init__.py
│
├── Results (2 files)
│   ├── example_multiple_ops_max_ab_20251113_013450.csv
│   └── example_multiple_ops_max_abc_20251113_062946.csv
│
└── This file: CLEANUP_LOG.md
```

---

**Status**: ✅ Cleanup Complete
**Next Step**: Ready for new chat continuation with clean workspace
