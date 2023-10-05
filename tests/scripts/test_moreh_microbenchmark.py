# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import copy
import argparse
import re
import csv
import json
import time
import random
import click
import subprocess as sp
from pathlib import Path
from itertools import chain
from functools import partial
from loguru import logger

from tests.scripts.common import (
    run_single_test,
)

def run_moreh_single_test(test_name, test_entry):
    full_env = copy.deepcopy(os.environ)
    logger.info(f"========= RUNNING MOREH TEST - {test_name}")
    result = sp.run(test_entry, shell=True, capture_output=True, env=full_env)
    # print(result.stdout.decode('utf-8'))
    return result

def capture_terminal_line(log, keyword):
    lines = log.stdout.decode('utf-8').strip().split('\n')
    for line in lines:
        if keyword in line:
            return line

def capture_line_result(line, position):
    float_pattern = r"[-+]?\d*\.\d+|\d+"
    float_values = re.findall(float_pattern, line)
    float_values = [float(value) for value in float_values]
    return float_values[position]

def run_process_device_log(home_path, log_path, params=""):
    full_env = copy.deepcopy(os.environ)
    args = " --no-print-stats --no-webapp --no-plots --no-artifacts " + params
    cmd = ["./process_device_log.py" + args]
    sp.run(cmd, shell=True, text=True, cwd=log_path)
    return

def generate_csv(file_name, header, data):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

def get_device_data(in_file_name):
    with open(in_file_name, mode='r') as in_file:
        in_data = json.load(in_file)
        total_cycle = in_data['data']['devices']['0']['cores']['DEVICE']['analysis']['T0 -> ANY CORE ANY RISC FW end']['stats']['Average']
    return total_cycle

def get_device_noc_data(in_file_name):
    with open(in_file_name, mode='r') as in_file:
        in_data = json.load(in_file)
        min = in_data['data']['devices']['0']['cores']['DEVICE']['analysis']['NoC For Loop']['stats']['Min']
        max = in_data['data']['devices']['0']['cores']['DEVICE']['analysis']['NoC For Loop']['stats']['Max']
    return min, max

# pcie write
def test_write_device_l1(iter=1, buffer_type=0, size=2048):
    command = "./build/test/tt_metal/perf_microbenchmark/pcie/test_rw_device_l1 " + \
                "--iter " + str(iter) + " --buffer_type " + str(buffer_type) + " --size " + str(size)
    result = run_moreh_single_test('pcie write to device l1', command)
    line = capture_terminal_line(result, 'WriteToDeviceL1')
    bw = capture_line_result(line, -1)
    return bw

def test_write_device_dram_channel(iter=1, buffer_type=0, size=2048):
    command = "./build/test/tt_metal/perf_microbenchmark/pcie/test_rw_device_dram " + \
                "--iter " + str(iter) + " --buffer_type " + str(buffer_type) + " --size " + str(size)
    result = run_moreh_single_test('pcie write to device dram', command)
    line = capture_terminal_line(result, 'WriteToDeviceDRAMChannel')
    bw = capture_line_result(line, -1)
    return bw

def test_write_buffer(iter=1, buffer_type=0, size=2048):
    command = "./build/test/tt_metal/perf_microbenchmark/pcie/test_rw_buffer " + \
                "--iter " + str(iter) + " --buffer_type " + str(buffer_type) + " --size " + str(size)
    result = run_moreh_single_test('pcie write to buffer', command)
    line = capture_terminal_line(result, 'WriteToBuffer')
    bw = capture_line_result(line, -1)
    return bw

def test_enqueue_write_buffer(iter=1, buffer_type=0, size=2048, timeout=600):
    command = "./build/test/tt_metal/perf_microbenchmark/pcie/test_enqueue_rw_buffer " + \
                "--iter " + str(iter) + " --buffer_type " + str(buffer_type) + " --size " + str(size)
    result = run_moreh_single_test('pcie enqueue write to buffer', command)
    line = capture_terminal_line(result, 'EnqueueWriteBuffer')
    bw = capture_line_result(line, -1)
    return bw

# pcie read
def test_read_device_l1(iter=1, buffer_type=0, size=2048):
    command = "./build/test/tt_metal/perf_microbenchmark/pcie/test_rw_device_l1 " + \
                "--iter " + str(iter) + " --buffer_type " + str(buffer_type) + " --size " + str(size)
    result = run_moreh_single_test('pcie read from device l1', command)
    line = capture_terminal_line(result, 'ReadFromDeviceL1')
    bw = capture_line_result(line, -1)
    return bw

def test_read_device_dram_channel(iter=1, buffer_type=0, size=2048):
    command = "./build/test/tt_metal/perf_microbenchmark/pcie/test_rw_device_dram " + \
                "--iter " + str(iter) + " --buffer_type " + str(buffer_type) + " --size " + str(size)
    result = run_moreh_single_test('pcie read from device dram', command)
    line = capture_terminal_line(result, 'ReadFromDeviceDRAMChannel')
    bw = capture_line_result(line, -1)
    return bw

def test_read_buffer(iter=1, buffer_type=0, size=2048):
    command = "./build/test/tt_metal/perf_microbenchmark/pcie/test_rw_buffer " + \
                "--iter " + str(iter) + " --buffer_type " + str(buffer_type) + " --size " + str(size)
    result = run_moreh_single_test('pcie read from buffer', command)
    line = capture_terminal_line(result, 'ReadFromBuffer')
    bw = capture_line_result(line, -1)
    return bw

def test_enqueue_read_buffer(iter=1, buffer_type=0, size=2048, timeout=600):
    command = "./build/test/tt_metal/perf_microbenchmark/pcie/test_enqueue_rw_buffer " + \
                "--iter " + str(iter) + " --buffer_type " + str(buffer_type) + " --size " + str(size)
    result = run_moreh_single_test('pcie enqueue read from buffer', command)
    line = capture_terminal_line(result, 'EnqueueReadBuffer')
    bw = capture_line_result(line, -1)
    return bw

# noc
def test_noc_local(n=9, c=12, nt=256, cb=1):
    command = "./build/test/tt_metal/perf_microbenchmark/noc/test_noc_read_local_l1 " + \
                "--n " + str(n) + " --c " + str(c) + " --nt " + str(nt) + " --cb " + str(cb)
    run_moreh_single_test('noc read local l1', command)

def test_noc_global_type_a(n=9, c=12, nt=256, cb=1):
    command = "./build/test/tt_metal/perf_microbenchmark/noc/test_noc_read_global_l1 " + \
                "--n " + str(n) + " --c " + str(c) + " --nt " + str(nt) + " --cb " + str(cb) + \
                " --same_buffer_read 1 --one_buffer_share 1"
    run_moreh_single_test('noc read global l1 (type a)', command)

def test_noc_global_type_b(n=9, c=12, nt=256, cb=1):
    command = "./build/test/tt_metal/perf_microbenchmark/noc/test_noc_read_global_l1 " + \
                "--n " + str(n) + " --c " + str(c) + " --nt " + str(nt) + " --cb " + str(cb) + \
                " --same_buffer_read 1 --one_buffer_share 0"
    run_moreh_single_test('noc read global l1 (type b)', command)

# matmul
def test_matmul_global(n=9, c=12, mt=72, nt=96, kt=24, per_core_mt=8, per_core_nt=8, l1_in0=0, l1_in1=0, l1_out=0, in0_block_w=4):
    command = "./build/test/tt_metal/perf_microbenchmark/matmul/matmul_global_l1 " + \
                "--n " + str(n) + " --c " + str(c) + " --mt " + str(mt) + " --nt " + str(nt) + " --kt " + str(kt) + \
                " --l1_in0 " + str(l1_in0) + " --l1_in1 " + str(l1_in1) + " --l1_out " + str(l1_out) + " --in0_block_w " + str(in0_block_w) + \
                " --per_core_mt " + str(per_core_mt) + " --per_core_nt " + str(per_core_nt)
    run_moreh_single_test('matmul global l1', command)

def test_matmul_local(n=9, c=12, mt=72, nt=96, kt=24):
    command = "./build/test/tt_metal/perf_microbenchmark/matmul/matmul_local_l1 " + \
                "--n " + str(n) + " --c " + str(c) + " --mt " + str(mt) + " --nt " + str(nt) + " --kt " + str(kt)
    run_moreh_single_test('matmul local l1', command)

def test_pcie_h2d_dram(path):
    test_vector_small = [
                8192,
                32768,
                131072,
                524288,
                2097152,
                8388608]
    test_vector_large = [
                33554432,
                134217728,
                536870912]
    file_name = path + '/' + 'logs/H2D_DRAM_Bandwidth.csv'
    header = ['Transfer Size', 'WriteToDeviceDRAMChannel', 'WriteToBuffer', 'EnqueueWriteBuffer']
    data = []
    iter=10
    for test_point in test_vector_small:
        bw_wdd = test_write_device_dram_channel(iter, 0, test_point)
        bw_wb = test_write_buffer(iter, 0, test_point)
        bw_ewb = test_enqueue_write_buffer(iter, 0, test_point)
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

def test_pcie_d2h_dram(path):
    test_vector_small = [
                8192,
                32768,
                131072,
                524288,
                2097152,
                8388608]
    test_vector_large = [
                33554432,
                134217728,
                536870912]
    file_name = path + '/' + 'logs/D2H_DRAM_Bandwidth.csv'
    header = ['Transfer Size', 'ReadFromDeviceDRAMChannel', 'ReadFromBuffer', 'EnqueueReadBuffer']
    data = []
    iter=10
    for test_point in test_vector_small:
        bw_wdd = test_read_device_dram_channel(iter, 0, test_point)
        bw_wb = test_read_buffer(iter, 0, test_point)
        bw_ewb = test_enqueue_read_buffer(iter, 0, test_point)
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

def test_pcie_h2d_l1(path):
    test_vector = [
                4096,
                16384,
                65536,
                262144,
                1048576,
                4194304,
                16777216]
    file_name = path + '/' + 'logs/H2D_L1_Bandwidth.csv'
    header = ['Transfer Size', 'WriteToDeviceDRAMChannel', 'WriteToBuffer', 'EnqueueWriteBuffer']
    data = []
    iter=10
    for test_point in test_vector:
        bw_wdd = test_write_device_l1(iter, 1, test_point)
        bw_wb = test_write_buffer(iter, 1, test_point)
        bw_ewb = test_enqueue_write_buffer(iter, 1, test_point)
        data_entry = [test_point, bw_wdd, bw_wb, bw_ewb]
        data.append(data_entry)
    generate_csv(file_name, header, data)
    return

def test_pcie_d2h_l1(path):
    test_vector = [
                4096,
                16384,
                65536,
                262144,
                1048576,
                4194304,
                16777216]
    file_name = path + '/' + 'logs/D2H_L1_Bandwidth.csv'
    header = ['Transfer Size', 'ReadFromDeviceDRAMChannel', 'ReadFromBuffer', 'EnqueueReadBuffer']
    data = []
    iter=10
    for test_point in test_vector:
        bw_wdd = test_read_device_l1(iter, 1, test_point)
        bw_wb = test_read_buffer(iter, 1, test_point)
        bw_ewb = test_enqueue_read_buffer(iter, 1, test_point)
        data_entry = [test_point, bw_wdd, bw_wb, bw_ewb]
        data.append(data_entry)
    generate_csv(file_name, header, data)
    return

def test_noc(home_path, log_path):
    test_vector = [
                1,
                8,
                16,
                32]
    process_device_log_param = "-s test_noc"
    in_file_name = log_path + '/' + '/output/device/device_analysis_data.json'
    file_name = log_path + '/' + 'logs/NoC_Read_Performance.csv'
    header = ['Requests', 'Local L1 (min)', 'Local L1 (max)', \
            'Global L1 (type A) (min)', 'Global L1 (type A) (max)', \
            'Global L1 (type B) (min)', 'Global L1 (type B) (max)']
    data = []
    for test_point in test_vector:
        test_noc_local(9, 12, 256, test_point)
        run_process_device_log(home_path, log_path, process_device_log_param)
        min_1, max_1 = get_device_noc_data(in_file_name)
        test_noc_global_type_a(9, 12, 256, test_point)
        run_process_device_log(home_path, log_path, process_device_log_param)
        min_2, max_2 = get_device_noc_data(in_file_name)
        test_noc_global_type_b(9, 12, 256, test_point)
        run_process_device_log(home_path, log_path, process_device_log_param)
        min_3, max_3 = get_device_noc_data(in_file_name)
        data.append([test_point, min_1, max_1, min_2, max_2, min_3, max_3])
    generate_csv(file_name, header, data)
    return

def test_matmul_dram(home_path, log_path, frequency):
    test_vector = [
                [4608, 6144, 6144],
                [3456, 4096, 1024],
                [3456, 3072, 1024],
                [2304, 3072, 768]]
    in_file_name = log_path + '/' + '/output/device/device_analysis_data.json'
    file_name = log_path + '/' + 'logs/Matmul_DRAM.csv'
    header = ['M', 'N', 'K', 'Cycles', 'Time (ms)', 'TFLOPS']
    data = []
    for vec in test_vector:
        mt = int(vec[0] / 32)
        nt = int(vec[1] / 32)
        kt = int(vec[2] / 32)
        per_core_mt = int((mt-1) / 9) + 1
        per_core_nt = int((nt-1) / 12) + 1
        test_matmul_global(9, 12, mt, nt, kt, per_core_mt, per_core_nt, 0, 0, 0, 2)
        run_process_device_log(home_path, log_path)
        cycle = get_device_data(in_file_name)
        num_op = vec[0] * vec[1] * vec[2] * 2
        time = cycle / frequency / 1000.0
        throughput = num_op / time / 1000.0 / 1000.0 / 1000.0
        data.append(vec + [cycle, time, throughput])
    generate_csv(file_name, header, data)
    return

def test_matmul_l1(home_path, log_path, frequency):
    test_vector_global = [
                [3456, 3072, 1024],
                [2304, 3072, 768]]
    test_vector_local = [
                [2304, 3072, 768]]
    in_file_name = log_path + '/' + '/output/device/device_analysis_data.json'
    file_name = log_path + '/' + 'logs/Matmul_SRAM.csv'
    header = ['M', 'N', 'K', 'Cycles', 'Time (ms)', 'TFLOPS']
    data = []
    for vec in test_vector_global:
        mt = int(vec[0] / 32)
        nt = int(vec[1] / 32)
        kt = int(vec[2] / 32)
        per_core_mt = int((mt-1) / 9) + 1
        per_core_nt = int((nt-1) / 12) + 1
        test_matmul_global(9, 12, mt, nt, kt, per_core_mt, per_core_nt, 1, 1, 1, 4)
        run_process_device_log(home_path, log_path)
        cycle = get_device_data(in_file_name)
        num_op = vec[0] * vec[1] * vec[2] * 2
        time = cycle / frequency / 1000.0
        throughput = num_op / time / 1000.0 / 1000.0 / 1000.0
        data.append(vec + [cycle, time, throughput])
    for vec in test_vector_local:
        mt = int(vec[0] / 32)
        nt = int(vec[1] / 32)
        kt = int(vec[2] / 32)
        test_matmul_local(9, 12, mt, nt, kt)
        run_process_device_log(home_path, log_path)
        cycle = get_device_data(in_file_name)
        num_op = vec[0] * vec[1] * vec[2] * 2
        time = cycle / frequency / 1000.0
        throughput = num_op / time / 1000.0 / 1000.0 / 1000.0
        data.append(vec + [cycle, time, throughput])
    generate_csv(file_name, header, data)
    return

def get_cmdline_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", default='tt_metal/tools/profiler', help="Timeout in seconds for each test")
    parser.add_argument("--frequency", default=1202, help="Timeout in seconds for each test")
    return parser.parse_args()

if __name__ == "__main__":

    home_path = os.environ.get('TT_METAL_HOME')
    cmdline_args = get_cmdline_args()
    test_pcie_h2d_dram(home_path + '/' + cmdline_args.log_path)
    test_pcie_d2h_dram(home_path + '/' + cmdline_args.log_path)
    test_pcie_h2d_l1(home_path + '/' + cmdline_args.log_path)
    test_pcie_d2h_l1(home_path + '/' + cmdline_args.log_path)
    test_noc(home_path, home_path + '/' + cmdline_args.log_path)
    test_matmul_dram(home_path, home_path + '/' + cmdline_args.log_path, cmdline_args.frequency)
    test_matmul_l1(home_path, home_path + '/' + cmdline_args.log_path, cmdline_args.frequency)
