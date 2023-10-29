#!/usr/bin/env python3

import os
import copy
from process_device_log import import_log_run_stats
import plot_setup

tt_home = os.environ.get('TT_METAL_HOME')
profiler_path = os.path.join(tt_home, 'tt_metal/tools/profiler')
profiler_log_path = os.path.join(profiler_path, 'logs/ops_device/tt::operations::primary::Matmul')

def profile_results(profiler_log_path_ops):
    setup = plot_setup.test_noc()
    setup.deviceInputLog = profiler_log_path_ops
    devices_data = import_log_run_stats(setup)
    deviceID = list(devices_data["devices"].keys())[0]
    total_cycle = devices_data['devices'][deviceID]['cores']['DEVICE']['analysis']['NoC For Loop']['stats']['Average']
    return total_cycle


total_cycle_list = []
for i in range(8):
    # matmul
    profiler_log_path_ops = os.path.join(profiler_log_path, str(i+1))
    profiler_log_path_ops = os.path.join(profiler_log_path_ops, 'profile_log_device.csv')
    total_cycle = profile_results(profiler_log_path_ops)
    total_cycle_list.append(total_cycle)


print('total cycles')
for cyc in total_cycle_list:
    print(str(int(cyc)))
