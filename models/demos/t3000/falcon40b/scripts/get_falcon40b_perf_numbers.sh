#!/bin/bash

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# This script runs the performance tests for the falcon40b model with different sequence lengths

seq_lens=(128 2048)

for seq_len in ${seq_lens[@]}; do
    echo "Running seq length: $seq_len"
    output_folder="generated/profiler/reports/"
    WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python3 -m tracy -r -m "pytest models/demos/t3000/falcon40b/tests/test_perf_falcon.py::test_perf_bare_metal[wormhole_b0-True-falcon_40b-prefill_seq${seq_len}_bfp8_layers1-8chips]"
    # get latest folder in output folder
    latest_created_folder=$(ls -td $output_folder/* | head -n 1)
    # find csv file that starts with "ops_perf_results"
    csv_file=$(find $latest_created_folder -name "ops_perf_results*.csv")
    # create output summary file from csv file
    output_perf_filename="${csv_file%.*}_seqlen_${seq_len}_summary.csv"
    echo "CSV file: $csv_file"
    echo "Output file: $output_perf_filename"
    python models/demos/t3000/falcon40b/scripts/perf_summary.py --all $csv_file -o $output_perf_filename --remove-warmup -g --num-chips 8 --layers 60 --seq ${seq_len}
done
