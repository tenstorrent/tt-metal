#!/bin/bash

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# This script runs the performance tests for the falcon40b model with different sequence lengths

seq_lens=(128 2048)

for seq_len in ${seq_lens[@]}; do
    echo "Running seq length: $seq_len"
    output_folder="generated/profiler/reports"
    WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python3 -m tracy -r -m "pytest models/demos/t3000/falcon40b/tests/test_perf_falcon.py::test_device_perf_bare_metal[wormhole_b0-True-falcon_40b-prefill_seq${seq_len}_bfp8_layers1-8chips]"
    # get latest folder in output folder
    latest_created_folder=$(ls -td $output_folder/* | head -n 1)
    # find csv file that starts with "ops_perf_results"
    csv_file=$(find $latest_created_folder -name "ops_perf_results*.csv")
    # create output summary file from csv file
    output_perf_filename="${csv_file%.*}_seqlen_${seq_len}_summary.csv"
    echo "CSV file: $csv_file"
    echo "Output file: $output_perf_filename"

    # Skip a specific number of ops based on sequence length
    if [ $seq_len -eq 128 ]; then
        skip_num_ops_end = 7
    elif [ $seq_len -eq 2048 ]; then
        skip_num_ops_end = 9
    else
        echo "No skip_num_ops_end known for given sequence length!"
        return -1
    fi
    python models/demos/t3000/mixtral8x7b/scripts/op_perf_results.py --signpost PERF_RUN --skip-last ${skip_num_ops_end} --skip-first 3 --prefill --seqlen ${seq_len} --estimate-full-model 60 --write-ops-to-csv $output_perf_filename $csv_file
done
