#!/usr/bin/env bash
set -euo pipefail
base=models/autoports/qwen_qwen3_8_27b/doc/optimized_full_model
python -m models.autoports.qwen_qwen3_8_27b.tests.probe_head_geometry --generic --block 4 --subblock 2 --output "$base/head_generic_k4_sub2.json" > "$base/head_generic_k4_sub2.log" 2>&1
python -m models.autoports.qwen_qwen3_8_27b.tests.probe_head_geometry --generic --block 4 --fidelity HiFi2 --output "$base/head_generic_k4_hifi2.json" > "$base/head_generic_k4_hifi2.log" 2>&1
python -m models.autoports.qwen_qwen3_8_27b.tests.probe_head_geometry --chunk 6144 --readers 2 --block 10 --fidelity HiFi2 --output "$base/head_c6144_r2_k10_hifi2.json" > "$base/head_c6144_r2_k10_hifi2.log" 2>&1
python -m models.autoports.qwen_qwen3_8_27b.tests.probe_head_geometry --generic --block 16 --output "$base/head_generic_k16.json" > "$base/head_generic_k16.log" 2>&1
