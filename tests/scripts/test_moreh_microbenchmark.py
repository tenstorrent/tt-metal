# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import copy
import re
import csv
import time
import random
import toolz
import subprocess as sp
from pathlib import Path
from itertools import chain
from functools import partial
from loguru import logger
import pytest
import numpy as np
import sys

import tt_lib as ttl

from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config


def run_moreh_single_test(test_name, test_entry):
    full_env = copy.deepcopy(os.environ)
    logger.info(f"========= RUNNING MOREH TEST - {test_name}")
    print(test_entry)
    result = sp.run(test_entry, shell=True, capture_output=True, env=full_env)
    print(result.stdout.decode("utf-8"))
    print(result.stderr.decode("utf-8"))
    return result


def capture_line_from_str_output(str_output, keyword):
    lines = str_output.strip().split("\n")
    for line in lines:
        if keyword in line:
            return line


def capture_terminal_line(log, keyword):
    str_output = log.stdout.decode("utf-8")

    return capture_line_from_str_output(str_output, keyword)


def capture_line_result(line, position):
    float_pattern = r"[-+]?\d*\.\d+|\d+"
    float_values = re.findall(float_pattern, line)
    float_values = [float(value) for value in float_values]
    return float_values[position]


def generate_csv(file_name, header, data):
    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)


def append_to_csv(file_path, header, data, write_header=True):
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists or write_header:
            writer.writerow(header)
        writer.writerows(data)


def profile_results():
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    devices_data = import_log_run_stats(setup)
    deviceID = list(devices_data["devices"].keys())[0]
    total_cycle = devices_data["devices"][deviceID]["cores"]["DEVICE"]["analysis"]["device_fw_duration"]["stats"][
        "Average"
    ]
    return total_cycle


def profile_results_kernel_duration():
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    devices_data = import_log_run_stats(setup)
    deviceID = list(devices_data["devices"].keys())[0]
    total_cycle = devices_data["devices"][deviceID]["cores"]["DEVICE"]["analysis"]["device_kernel_duration"]["stats"][
        "Average"
    ]
    return total_cycle


def get_device_freq():
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    freq = deviceData["deviceInfo"]["freq"]
    return freq


def profile_noc_results():
    setup = device_post_proc_config.test_noc()
    setup.deviceInputLog = profiler_log_path
    devices_data = import_log_run_stats(setup)
    deviceID = list(devices_data["devices"].keys())[0]
    min = devices_data["devices"][deviceID]["cores"]["DEVICE"]["analysis"]["NoC For Loop"]["stats"]["Min"]
    max = devices_data["devices"][deviceID]["cores"]["DEVICE"]["analysis"]["NoC For Loop"]["stats"]["Max"]
    return min, max


# pcie write
def test_write_device_l1(iter=1, buffer_type=0, size=2048):
    command = (
        "./build/test/tt_metal/perf_microbenchmark/old/pcie/test_rw_device_l1 "
        + "--iter "
        + str(iter)
        + " --buffer_type "
        + str(buffer_type)
        + " --size "
        + str(size)
    )
    result = run_moreh_single_test("pcie write to device l1", command)
    line = capture_terminal_line(result, "WriteToDeviceL1")
    bw = capture_line_result(line, -1)
    return bw


def test_write_device_dram_channel(iter=1, buffer_type=0, size=2048):
    command = (
        "./build/test/tt_metal/perf_microbenchmark/old/pcie/test_rw_device_dram "
        + "--iter "
        + str(iter)
        + " --buffer_type "
        + str(buffer_type)
        + " --size "
        + str(size)
    )
    result = run_moreh_single_test("pcie write to device dram", command)
    line = capture_terminal_line(result, "WriteToDeviceDRAMChannel")
    bw = capture_line_result(line, -1)
    return bw


def test_write_buffer(iter=1, buffer_type=0, size=2048):
    command = (
        "./build/test/tt_metal/perf_microbenchmark/old/pcie/test_rw_buffer_old "
        + "--iter "
        + str(iter)
        + " --buffer_type "
        + str(buffer_type)
        + " --size "
        + str(size)
    )
    result = run_moreh_single_test("pcie write to buffer", command)
    line = capture_terminal_line(result, "WriteToBuffer")
    bw = capture_line_result(line, -1)
    return bw


def test_enqueue_write_buffer(iter=1, buffer_type=0, size=2048, timeout=600):
    command = (
        "./build/test/tt_metal/perf_microbenchmark/old/pcie/test_enqueue_rw_buffer "
        + "--iter "
        + str(iter)
        + " --buffer_type "
        + str(buffer_type)
        + " --size "
        + str(size)
    )
    result = run_moreh_single_test("pcie enqueue write to buffer", command)
    line = capture_terminal_line(result, "EnqueueWriteBuffer")
    bw = capture_line_result(line, -1)
    return bw


# pcie read
def test_read_device_l1(iter=1, buffer_type=0, size=2048):
    command = (
        "./build/test/tt_metal/perf_microbenchmark/old/pcie/test_rw_device_l1 "
        + "--iter "
        + str(iter)
        + " --buffer_type "
        + str(buffer_type)
        + " --size "
        + str(size)
    )
    result = run_moreh_single_test("pcie read from device l1", command)
    line = capture_terminal_line(result, "ReadFromDeviceL1")
    bw = capture_line_result(line, -1)
    return bw


def test_read_device_dram_channel(iter=1, buffer_type=0, size=2048):
    command = (
        "./build/test/tt_metal/perf_microbenchmark/old/pcie/test_rw_device_dram "
        + "--iter "
        + str(iter)
        + " --buffer_type "
        + str(buffer_type)
        + " --size "
        + str(size)
    )
    result = run_moreh_single_test("pcie read from device dram", command)
    line = capture_terminal_line(result, "ReadFromDeviceDRAMChannel")
    bw = capture_line_result(line, -1)
    return bw


def test_read_buffer(iter=1, buffer_type=0, size=2048):
    command = (
        "./build/test/tt_metal/perf_microbenchmark/old/pcie/test_rw_buffer_old "
        + "--iter "
        + str(iter)
        + " --buffer_type "
        + str(buffer_type)
        + " --size "
        + str(size)
    )
    result = run_moreh_single_test("pcie read from buffer", command)
    line = capture_terminal_line(result, "ReadFromBuffer")
    bw = capture_line_result(line, -1)
    return bw


def test_enqueue_read_buffer(iter=1, buffer_type=0, size=2048, timeout=600):
    command = (
        "./build/test/tt_metal/perf_microbenchmark/old/pcie/test_enqueue_rw_buffer "
        + "--iter "
        + str(iter)
        + " --buffer_type "
        + str(buffer_type)
        + " --size "
        + str(size)
    )
    result = run_moreh_single_test("pcie enqueue read from buffer", command)
    line = capture_terminal_line(result, "EnqueueReadBuffer")
    bw = capture_line_result(line, -1)
    return bw


def run_dram_read_cmd(k, n, num_blocks, df, num_banks, bank_start_id):
    command = (
        "TT_METAL_DEVICE_PROFILER=1 ./build/test/tt_metal/perf_microbenchmark/8_dram_adjacent_core_read/test_dram_read "
        + " --k "
        + str(k)
        + " --n "
        + str(n)
        + " --num-blocks "
        + str(num_blocks)
        + " --num-tests "
        + str(1)
        + " --data-type "
        + str(df)
        + " --num-banks "
        + str(num_banks)
        + " --bank-start-id "
        + str(bank_start_id)
        + " --bypass-check "
    )
    run_moreh_single_test("DRAM BW test multi-core", command)


# noc
def test_noc_local(r=9, c=12, nt=256, cb=1):
    command = (
        "./build/test/tt_metal/perf_microbenchmark/old/noc/test_noc_read_local_l1 "
        + "--r "
        + str(r)
        + " --c "
        + str(c)
        + " --nt "
        + str(nt)
        + " --cb "
        + str(cb)
    )
    run_moreh_single_test("noc read local l1", command)


def test_noc_global_type_a(r=9, c=12, nt=256, cb=1):
    command = (
        "./build/test/tt_metal/perf_microbenchmark/old/noc/test_noc_read_global_l1 "
        + "--r "
        + str(r)
        + " --c "
        + str(c)
        + " --nt "
        + str(nt)
        + " --cb "
        + str(cb)
        + " --same_buffer_read 1 --one_buffer_share 1"
    )
    run_moreh_single_test("noc read global l1 (type a)", command)


def test_noc_global_type_b(r=9, c=12, nt=256, cb=1):
    command = (
        "./build/test/tt_metal/perf_microbenchmark/old/noc/test_noc_read_global_l1 "
        + "--r "
        + str(r)
        + " --c "
        + str(c)
        + " --nt "
        + str(nt)
        + " --cb "
        + str(cb)
        + " --same_buffer_read 1 --one_buffer_share 0"
    )
    run_moreh_single_test("noc read global l1 (type b)", command)


# matmul
def test_matmul_global(
    r=9, c=12, mt=72, nt=96, kt=24, per_core_mt=8, per_core_nt=8, l1_in0=0, l1_in1=0, l1_out=0, in0_block_w=4
):
    command = (
        "./build/test/tt_metal/perf_microbenchmark/old/matmul/matmul_global_l1 "
        + "--r "
        + str(r)
        + " --c "
        + str(c)
        + " --mt "
        + str(mt)
        + " --nt "
        + str(nt)
        + " --kt "
        + str(kt)
        + " --l1_in0 "
        + str(l1_in0)
        + " --l1_in1 "
        + str(l1_in1)
        + " --l1_out "
        + str(l1_out)
        + " --in0_block_w "
        + str(in0_block_w)
        + " --per_core_mt "
        + str(per_core_mt)
        + " --per_core_nt "
        + str(per_core_nt)
    )
    run_moreh_single_test("matmul global l1", command)


def test_matmul_local(r=9, c=12, mt=72, nt=96, kt=24):
    command = (
        "./build/test/tt_metal/perf_microbenchmark/old/matmul/matmul_local_l1 "
        + "--r "
        + str(r)
        + " --c "
        + str(c)
        + " --mt "
        + str(mt)
        + " --nt "
        + str(nt)
        + " --kt "
        + str(kt)
    )
    run_moreh_single_test("matmul local l1", command)


def test_matmul_single_core(
    m, n, k, dtype, fidel, matmul_block, num_blocks, packer_l1_acc, fp32_dest_acc, interm_cb_dtype, subblock_index
):
    command = (
        "TT_METAL_DEVICE_PROFILER=1 ./build/test/tt_metal/perf_microbenchmark/1_compute_mm/test_compute_mm "
        + "--m "
        + str(m)
        + " --n "
        + str(n)
        + " --k "
        + str(k)
        + " --dtype "
        + str(dtype)
        + " --fidel "
        + str(fidel)
        + " --block "
        + str(matmul_block)
        + " --num-tests "
        + str(1)
        + " --fast-dispatch --bypass-check --one-core 1 "
        + " --num-blocks "
        + str(num_blocks)
        + " --packer "
        + str(packer_l1_acc)
        + " --fp32 "
        + str(fp32_dest_acc)
        + " --interm-cb "
        + str(interm_cb_dtype)
        + " --subblock-index "
        + str(subblock_index)
    )
    run_moreh_single_test("matmul single core sharded", command)


@pytest.mark.parametrize(
    "iteration, test_vector_small, test_vector_large",
    [(2, np.array([8192, 32768, 131072, 524288, 2097152, 8388608]), np.array([33554432, 134217728, 536870912]))],
)
def test_pcie_h2d_dram(iteration, test_vector_small, test_vector_large):
    file_name = PROFILER_LOGS_DIR / "moreh_old_H2D_DRAM_Bandwidth.csv"
    header = ["Transfer Size", "WriteToDeviceDRAMChannel", "WriteToBuffer", "EnqueueWriteBuffer"]
    data = []
    for test_point in test_vector_small:
        bw_wdd = test_write_device_dram_channel(iteration, 0, test_point)
        bw_wb = test_write_buffer(iteration, 0, test_point)
        bw_ewb = test_enqueue_write_buffer(iteration, 0, test_point)
        data_entry = [test_point, bw_wdd, bw_wb, bw_ewb]
        data.append(data_entry)
    for test_point in test_vector_large:
        bw_wdd = test_write_device_dram_channel(1, 0, test_point)
        bw_wb = test_write_buffer(1, 0, test_point)
        bw_ewb = test_enqueue_write_buffer(1, 0, test_point)
        data_entry = [test_point, bw_wdd, bw_wb, bw_ewb]
        data.append(data_entry)
    generate_csv(file_name, header, data)
    return


@pytest.mark.parametrize(
    "iteration, test_vector_small, test_vector_large",
    [(2, np.array([8192, 32768, 131072, 524288, 2097152, 8388608]), np.array([33554432, 134217728, 536870912]))],
)
def test_pcie_d2h_dram(iteration, test_vector_small, test_vector_large):
    file_name = PROFILER_LOGS_DIR / "moreh_old_D2H_DRAM_Bandwidth.csv"
    header = ["Transfer Size", "ReadFromDeviceDRAMChannel", "ReadFromBuffer", "EnqueueReadBuffer"]
    data = []
    for test_point in test_vector_small:
        bw_wdd = test_read_device_dram_channel(iteration, 0, test_point)
        bw_wb = test_read_buffer(iteration, 0, test_point)
        bw_ewb = test_enqueue_read_buffer(iteration, 0, test_point)
        data_entry = [test_point, bw_wdd, bw_wb, bw_ewb]
        data.append(data_entry)
    for test_point in test_vector_large:
        bw_wdd = test_read_device_dram_channel(1, 0, test_point)
        bw_wb = test_read_buffer(1, 0, test_point)
        bw_ewb = test_enqueue_read_buffer(1, 0, test_point)
        data_entry = [test_point, bw_wdd, bw_wb, bw_ewb]
        data.append(data_entry)
    generate_csv(file_name, header, data)
    return


@pytest.mark.parametrize(
    "arch, iteration, L1_size, test_vector",
    [
        ("grayskull", 2, 1048576, np.array([4096, 16384, 65536, 262144, 1048576, 4194304, 16777216])),
        ("wormhole_b0", 2, 1499136, np.array([4096, 16384, 65536, 262144, 1048576, 4194304, 16777216])),
        ("blackhole", 2, 1499136, np.array([4096, 16384, 65536, 262144, 1048576, 4194304, 16777216])),
    ],
)
def test_pcie_h2d_l1(arch, iteration, L1_size, test_vector):
    file_name = PROFILER_LOGS_DIR / "moreh_old_H2D_L1_Bandwidth.csv"
    header = ["Transfer Size", "WriteToDeviceL1", "WriteToBuffer", "EnqueueWriteBuffer"]
    data = []
    for test_point in test_vector:
        if test_point < L1_size:
            bw_wdd = test_write_device_l1(iteration, 1, test_point)
        else:
            bw_wdd = 0
        bw_wb = test_write_buffer(iteration, 1, test_point)
        bw_ewb = test_enqueue_write_buffer(iteration, 1, test_point)
        data_entry = [test_point, bw_wdd, bw_wb, bw_ewb]
        data.append(data_entry)
    generate_csv(file_name, header, data)
    return


@pytest.mark.parametrize(
    "arch, iteration, L1_size, test_vector",
    [
        ("grayskull", 2, 1048576, np.array([4096, 16384, 65536])),
        ("wormhole_b0", 2, 1499136, np.array([4096, 16384, 65536])),
        ("blackhole", 2, 1499136, np.array([4096, 16384, 65536])),
    ],
)
def test_pcie_d2h_l1(arch, iteration, L1_size, test_vector):
    file_name = PROFILER_LOGS_DIR / "moreh_old_D2H_L1_Bandwidth.csv"
    header = ["Transfer Size", "ReadFromDeviceL1", "ReadFromBuffer", "EnqueueReadBuffer"]
    data = []
    for test_point in test_vector:
        if test_point < L1_size:
            bw_wdd = test_read_device_l1(iteration, 1, test_point)
        else:
            bw_wdd = 0
        bw_wb = test_read_buffer(iteration, 1, test_point)
        bw_ewb = test_enqueue_read_buffer(iteration, 1, test_point)
        data_entry = [test_point, bw_wdd, bw_wb, bw_ewb]
        data.append(data_entry)
    generate_csv(file_name, header, data)
    return


@pytest.mark.parametrize(
    "arch, r, c, nt, test_vector",
    [
        ("grayskull", 9, 12, 256, np.array([1, 8, 16, 32])),
        ("wormhole_b0", 6, 6, 256, np.array([1, 8, 16, 32])),
    ],
)
def test_noc(arch, r, c, nt, test_vector):
    file_name = PROFILER_LOGS_DIR / "moreh_old_NoC_Read_Performance.csv"
    header = [
        "Requests",
        "Local L1 (min)",
        "Local L1 (max)",
        "Global L1 (type A) (min)",
        "Global L1 (type A) (max)",
        "Global L1 (type B) (min)",
        "Global L1 (type B) (max)",
    ]
    data = []
    for test_point in test_vector:
        test_noc_local(r, c, nt, test_point)
        min_1, max_1 = profile_noc_results()
        test_noc_global_type_a(r, c, nt, test_point)
        min_2, max_2 = profile_noc_results()
        test_noc_global_type_b(r, c, nt, test_point)
        min_3, max_3 = profile_noc_results()
        data.append([test_point, min_1, max_1, min_2, max_2, min_3, max_3])
    generate_csv(file_name, header, data)
    return


@pytest.mark.parametrize(
    "arch, freq, r, c, test_vector",
    [
        (
            "grayskull",
            1020,
            9,
            12,
            np.array([[4608, 6144, 6144], [3456, 3840, 1024], [3456, 3072, 1024], [2304, 3072, 768]]),
        ),
        ("wormhole_b0", 1000, 6, 6, np.array([[2304, 1920, 1024], [2304, 1536, 1024], [1536, 1536, 768]])),
    ],
)
def test_matmul_dram(arch, freq, r, c, test_vector):
    file_name = PROFILER_LOGS_DIR / "moreh_old_Matmul_DRAM.csv"
    header = ["M", "N", "K", "Cycles", "Time (ms)", "TFLOPS"]
    data = []
    for vec in test_vector:
        mt = int(vec[0] / 32)
        nt = int(vec[1] / 32)
        kt = int(vec[2] / 32)
        per_core_mt = int((mt - 1) / r) + 1
        per_core_nt = int((nt - 1) / c) + 1
        test_matmul_global(r, c, mt, nt, kt, per_core_mt, per_core_nt, 0, 0, 0, 2)
        cycle = profile_results()
        num_op = vec[0] * vec[1] * vec[2] * 2
        time = cycle / freq / 1000.0
        throughput = num_op / time / 1000.0 / 1000.0 / 1000.0
        data.append(vec + [cycle, time, throughput])
    generate_csv(file_name, header, data)
    return


# TODO (abhullar): Uplift frequency and baseline when faster BH chips are available
@pytest.mark.parametrize(
    "arch, freq, test_vector, dtype, fidel, matmul_block, num_blocks, packer_l1_acc, fp32_dest_acc, interm_cb_dtype, subblock_index, baseline",
    [
        # ########################### 512 512 512 x 8 subblock 4 2 ################################
        ("wormhole_b0", 1000, np.array([[512, 512, 512]]), 0, 0, 1, 8, 0, 0, 0, 0, 717089.0),
        ("wormhole_b0", 1000, np.array([[512, 512, 512]]), 0, 1, 1, 8, 0, 0, 0, 0, 1233930.0),
        ("wormhole_b0", 1000, np.array([[512, 512, 512]]), 0, 0, 1, 8, 1, 0, 0, 0, 664492.0),
        ("wormhole_b0", 1000, np.array([[512, 512, 512]]), 0, 1, 1, 8, 1, 0, 0, 0, 1173029.0),
        ("blackhole", 1100, np.array([[512, 512, 512]]), 0, 0, 1, 8, 0, 0, 0, 0, 635270.0),
        ("blackhole", 1100, np.array([[512, 512, 512]]), 0, 1, 1, 8, 0, 0, 0, 0, 1192194.0),
        ("blackhole", 1100, np.array([[512, 512, 512]]), 0, 0, 1, 8, 1, 0, 0, 0, 582810.0),
        ("blackhole", 1100, np.array([[512, 512, 512]]), 0, 1, 1, 8, 1, 0, 0, 0, 1139919.0),
        # ########################### 512 512 256x8 subblock 4 2 ################################
        ("wormhole_b0", 1000, np.array([[512, 512, 256]]), 0, 0, 1, 8, 0, 0, 0, 0, 399068.0),
        ("wormhole_b0", 1000, np.array([[512, 512, 256]]), 0, 1, 1, 8, 0, 0, 0, 0, 658522.0),
        ("wormhole_b0", 1000, np.array([[512, 512, 256]]), 0, 0, 1, 8, 1, 0, 0, 0, 346350.0),
        ("wormhole_b0", 1000, np.array([[512, 512, 256]]), 0, 1, 1, 8, 1, 0, 0, 0, 597457.0),
        ("blackhole", 1100, np.array([[512, 512, 256]]), 0, 0, 0, 8, 0, 0, 0, 0, 969369.0),
        ("blackhole", 1100, np.array([[512, 512, 256]]), 0, 1, 0, 8, 0, 0, 0, 0, 969349.0),
        ("blackhole", 1100, np.array([[512, 512, 256]]), 0, 0, 1, 8, 0, 0, 0, 0, 352615.0),
        ("blackhole", 1100, np.array([[512, 512, 256]]), 0, 1, 1, 8, 0, 0, 0, 0, 631106.0),
        ("blackhole", 1100, np.array([[512, 512, 256]]), 0, 0, 1, 8, 1, 0, 0, 0, 300156.0),
        ("blackhole", 1100, np.array([[512, 512, 256]]), 0, 1, 1, 8, 1, 0, 0, 0, 578722.0),
    ],
)
def test_matmul_single_core_sharded(
    arch,
    freq,
    test_vector,
    dtype,
    fidel,
    matmul_block,
    num_blocks,
    packer_l1_acc,
    fp32_dest_acc,
    interm_cb_dtype,
    subblock_index,
    baseline,
):
    file_name = PROFILER_LOGS_DIR / "moreh_single_core_Matmul_Sharded.csv"
    header = ["Kernel Duration (Cycles)"]
    kernel_durations_cycle = []
    for vec in test_vector:
        m = int(vec[0])
        n = int(vec[1])
        k = int(vec[2])
        test_matmul_single_core(
            m,
            n,
            k,
            dtype,
            fidel,
            matmul_block,
            num_blocks,
            packer_l1_acc,
            fp32_dest_acc,
            interm_cb_dtype,
            subblock_index,
        )
        cycle = profile_results_kernel_duration()
        dev_freq = get_device_freq()
        logger.info("cycle: " + str(cycle))
        num_op = vec[0] * vec[1] * vec[2] * 2
        time = cycle / dev_freq / 1000.0
        throughput = num_op / time / 1000.0 / 1000.0 / 1000.0
        kernel_durations_cycle.append([cycle])

    max_diff = 0.01

    percentage_diff = np.abs(kernel_durations_cycle[0][0] - baseline) / baseline
    is_within_range = percentage_diff < max_diff
    if is_within_range == False:
        logger.info(
            "Diff is too large! kernel duration: {}, baseline: {}, diff: {}",
            kernel_durations_cycle[0][0],
            baseline,
            percentage_diff,
        )

    write_header = not os.path.exists(file_name)
    append_to_csv(file_name, header, kernel_durations_cycle, write_header)
    assert is_within_range


@pytest.mark.parametrize(
    "arch, freq, test_vector, num_tests, nblock, data_format, num_banks, bank_start_id",
    [
        ("wormhole_b0", 1000, np.array([32768, 12 * 128]), 1, 8, 0, 12, 0),
        ("wormhole_b0", 1000, np.array([32768, 12 * 128]), 1, 8, 1, 12, 0),
        ("blackhole", 1100, np.array([32768, 8 * 128]), 1, 8, 0, 8, 0),
        ("blackhole", 1100, np.array([32768, 8 * 128]), 1, 8, 1, 8, 0),
    ],
)
def test_dram_read_multi_core(arch, freq, test_vector, num_tests, nblock, data_format, num_banks, bank_start_id):
    data = []
    cycle_list = []
    time_list = []
    throughput_list = []
    for _ in range(num_tests):
        k = int(test_vector[0])
        n = int(test_vector[1])
        if data_format == 0:
            input_size = k * n * 1088 // 1024
        elif data_format == 1:
            input_size = k * n * 2048 // 1024
        run_dram_read_cmd(k, n, nblock, data_format, num_banks, bank_start_id)
        cycle = profile_results_kernel_duration()
        time = cycle / freq / 1000.0 / 1000.0
        throughput = input_size / cycle
        logger.info("DRAM read cycle: " + str(cycle))
        logger.info("DRAM read time: " + str(time))
        logger.info("DRAM read throughput: " + str(throughput))
        cycle_list.append(cycle)
        time_list.append(time)
        throughput_list.append(throughput)
    cycle = sum(cycle_list) / len(cycle_list)
    time = sum(time_list) / len(time_list)
    throughput = sum(throughput_list) / len(throughput_list)
    logger.info("DRAM read cycle: " + str(cycle))
    logger.info("DRAM read time: " + str(time))
    logger.info("DRAM read throughput: " + str(throughput))
    data.append([throughput])
    # check within range
    dev_freq = get_device_freq()
    bw_bound = 260.0 * dev_freq / 1000.0
    assert bw_bound <= throughput


@pytest.mark.parametrize(
    "arch, freq, r, c, test_vector_global, test_vector_local",
    [
        ("grayskull", 1020, 9, 12, np.array([[3456, 3072, 1024], [2304, 3072, 768]]), np.array([[2304, 3072, 768]])),
        ("wormhole_b0", 1000, 6, 6, np.array([[2304, 1536, 1024], [1536, 1536, 768]]), np.array([[1536, 1536, 768]])),
    ],
)
def test_matmul_l1(arch, freq, r, c, test_vector_global, test_vector_local):
    file_name = PROFILER_LOGS_DIR / "moreh_old_Matmul_SRAM.csv"
    header = ["M", "N", "K", "Cycles", "Time (ms)", "TFLOPS"]
    data = []
    for vec in test_vector_global:
        mt = int(vec[0] / 32)
        nt = int(vec[1] / 32)
        kt = int(vec[2] / 32)
        per_core_mt = int((mt - 1) / r) + 1
        per_core_nt = int((nt - 1) / c) + 1
        test_matmul_global(r, c, mt, nt, kt, per_core_mt, per_core_nt, 1, 1, 1, 4)
        cycle = profile_results()
        num_op = vec[0] * vec[1] * vec[2] * 2
        time = cycle / freq / 1000.0
        throughput = num_op / time / 1000.0 / 1000.0 / 1000.0
        data.append(vec + [cycle, time, throughput])
    for vec in test_vector_local:
        mt = int(vec[0] / 32)
        nt = int(vec[1] / 32)
        kt = int(vec[2] / 32)
        test_matmul_local(r, c, mt, nt, kt)
        cycle = profile_results()
        num_op = vec[0] * vec[1] * vec[2] * 2
        time = cycle / freq / 1000.0
        throughput = num_op / time / 1000.0 / 1000.0 / 1000.0
        data.append(vec + [cycle, time, throughput])
    generate_csv(file_name, header, data)
    return


@pytest.fixture(scope="function")
def create_moreh_microbenchmark_csv(request):
    microbenchmark_name = request.node.name.split("[")[0]

    file_name = PROFILER_LOGS_DIR / f"moreh_{microbenchmark_name}.csv"

    yield file_name


@pytest.fixture(scope="function")
def record_moreh_microbenchmark_csv(capsys, create_moreh_microbenchmark_csv):
    yield

    captured = capsys.readouterr()

    result_output = captured.out

    def get_entries(result_output, marker):
        line = capture_line_from_str_output(result_output, marker)
        return line.split(":")[1:]

    csv_microbenchmark_name = get_entries(result_output, "CSV_MICROBENCHMARK")[0]
    csv_inputs_and_values = get_entries(result_output, "CSV_INPUT")
    csv_outputs_and_values = get_entries(result_output, "CSV_OUTPUT")
    csv_result_and_value = get_entries(result_output, "CSV_RESULT")

    assert len(csv_result_and_value) == 2, f"CSV_RESULT needs to be a single name and value"
    assert len(csv_inputs_and_values) >= 2
    assert len(csv_outputs_and_values) >= 2

    def get_names(inputs_and_values):
        return list(toolz.itertoolz.take_nth(2, inputs_and_values))

    def get_values(inputs_and_values):
        return list(toolz.itertoolz.take_nth(2, inputs_and_values[1:]))

    csv_inputs_names = get_names(csv_inputs_and_values)
    csv_inputs_values = get_values(csv_inputs_and_values)

    csv_outputs_names = get_names(csv_outputs_and_values)
    csv_outputs_values = get_values(csv_outputs_and_values)

    csv_result_name = get_names(csv_result_and_value)[0]
    csv_result_value = get_values(csv_result_and_value)[0]

    headers = csv_inputs_names + csv_outputs_names + [csv_result_name]
    data = csv_inputs_values + csv_outputs_values + [csv_result_value]

    file_name = create_moreh_microbenchmark_csv

    file_path = Path(file_name)

    if not file_path.exists():
        with open(file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    assert file_path.is_file()
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)


@pytest.mark.parametrize(
    "r, c, num_tiles, tiles_per_transfer, noc_index, noc_direction, access_type, use_device_profiler",
    [(9, 12, 204800, 1, 0, 0, 0, 0), (9, 12, 204800, 1, 0, 0, 0, 0)],
)
def test_noc_adjacent(
    r,
    c,
    num_tiles,
    tiles_per_transfer,
    noc_index,
    noc_direction,
    access_type,
    use_device_profiler,
    record_moreh_microbenchmark_csv,
):
    command = (
        "./build/test/tt_metal/perf_microbenchmark/2_noc_adjacent/test_noc_adjacent "
        + "--cores-r "
        + str(r)
        + " --cores-c "
        + str(c)
        + " --num-tiles "
        + str(num_tiles)
        + " --tiles-per-transfer"
        + str(tiles_per_transfer)
        + " --noc-index"
        + str(noc_index)
        + " --noc-direction "
        + str(noc_direction)
        + " --access-type "
        + str(access_type)
    )
    if use_device_profiler:
        command += " --use-device-profiler"
    result = run_moreh_single_test("test_noc_adjacent", command)
