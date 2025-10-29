# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.perf.device_perf_utils import check_device_perf, prep_device_perf_report, run_device_perf


def run_perf_device(batch_size, test, command, expected_perf):
    print("\n")
    print("-------------------------------------------------------------")
    print("Perf Device Resnet50, Run Perf Device - BEGIN                ")
    print("-------------------------------------------------------------")
    print("\n")

    subdir = "resnet50"
    num_iterations = 1
    margin = 0.03
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]
    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    print("\n")
    print("-------------------------------------------------------------")
    print("Perf Device Resnet50, Run Perf Device - 1                    ")
    print("-------------------------------------------------------------")
    print("\n")

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size, has_signposts=True)

    print("\n")
    print("-------------------------------------------------------------")
    print("Perf Device Resnet50, Run Perf Device - 2                    ")
    print("-------------------------------------------------------------")
    print("\n")

    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)

    print("\n")
    print("-------------------------------------------------------------")
    print("Perf Device Resnet50, Run Perf Device - 3                    ")
    print("-------------------------------------------------------------")
    print("\n")

    prep_device_perf_report(
        model_name=f"ttnn_resnet50_batch_size{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments=test,
    )

    print("\n")
    print("-------------------------------------------------------------")
    print("Perf Device Resnet50, Run Perf Device - END                  ")
    print("-------------------------------------------------------------")
    print("\n")
