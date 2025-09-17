# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from tracy.merge_meta_class import MergeMetaclass
from tracy.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG


class default_setup(metaclass=MergeMetaclass):
    riscs = [
        "BRISC",
        "NCRISC",
        "TRISC_0",
        "TRISC_1",
        "TRISC_2",
        "ERISC",
    ]

    riscTypes = [
        "BRISC",
        "NCRISC",
        "TRISC",
        "ERISC",
        "CORE_AGG",
    ]

    timerAnalysis = {
        "trace_fw_duration": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "CORE_AGG", "zone_name": "TRACE-FW"},
            "end": {"core": "ANY", "risc": "CORE_AGG", "zone_name": "TRACE-FW"},
        },
        "trace_kernel_duration": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "CORE_AGG", "zone_name": "TRACE-KERNEL"},
            "end": {"core": "ANY", "risc": "CORE_AGG", "zone_name": "TRACE-KERNEL"},
        },
        "trace2trace - FW": {
            "across": "device",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "ANY", "zone_phase": "ZONE_END", "zone_name": "TRACE-FW"},
            "end": {"core": "ANY", "risc": "ANY", "zone_phase": "ZONE_START", "zone_name": "TRACE-FW"},
        },
        "trace2trace - kernel": {
            "across": "device",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "ANY", "zone_phase": "ZONE_END", "zone_name": "TRACE-KERNEL"},
            "end": {"core": "ANY", "risc": "ANY", "zone_phase": "ZONE_START", "zone_name": "TRACE-KERNEL"},
        },
        "op2op": {
            "across": "core",
            "type": "adjacent",
            "start": {
                "core": "ANY",
                "risc": "ANY",
                "zone_phase": "ZONE_END",
                "zone_name": [f"{risc}-KERNEL" for risc in riscTypes],
            },
            "end": {
                "core": "ANY",
                "risc": "ANY",
                "zone_phase": "ZONE_START",
                "zone_name": [f"{risc}-KERNEL" for risc in ["BRISC", "NCRISC"]],
            },
        },
        "device_kernel_first_to_last_start": {
            "across": "ops",
            "type": "op_first_last",
            "start": {
                "core": "ANY",
                "risc": "ANY",
                "zone_phase": "ZONE_START",
                "zone_name": [f"{risc}-KERNEL" for risc in riscTypes],
            },
            "end": {
                "core": "ANY",
                "risc": "ANY",
                "zone_phase": "ZONE_START",
                "zone_name": [f"{risc}-KERNEL" for risc in riscTypes],
            },
        },
        "device_kernel_duration_per_core": {
            "across": "ops",
            "type": "op_core_first_last",
            "start": {"core": "ANY", "risc": "ANY", "zone_name": [f"{risc}-KERNEL" for risc in riscTypes]},
            "end": {"core": "ANY", "risc": "ANY", "zone_name": [f"{risc}-KERNEL" for risc in riscTypes]},
        },
        "device_fw_duration": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "ANY", "zone_name": [f"{risc}-FW" for risc in riscTypes]},
            "end": {"core": "ANY", "risc": "ANY", "zone_name": [f"{risc}-FW" for risc in riscTypes]},
        },
        "device_kernel_duration": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "ANY", "zone_name": [f"{risc}-KERNEL" for risc in riscTypes]},
            "end": {"core": "ANY", "risc": "ANY", "zone_name": [f"{risc}-KERNEL" for risc in riscTypes]},
        },
        "device_kernel_duration_dm_start": {
            "across": "ops",
            "type": "op_first_last",
            "start": {
                "core": "ANY",
                "risc": "ANY",
                "zone_name": [f"{risc}-KERNEL" for risc in ["BRISC", "NCRISC", "ERISC"]],
            },
            "end": {"core": "ANY", "risc": "ANY", "zone_name": [f"{risc}-KERNEL" for risc in riscTypes]},
        },
        "device_brisc_kernel_duration": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "BRISC", "zone_name": "BRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "BRISC", "zone_name": "BRISC-KERNEL"},
        },
        "device_ncrisc_kernel_duration": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "NCRISC", "zone_name": "NCRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "NCRISC", "zone_name": "NCRISC-KERNEL"},
        },
        "device_trisc0_kernel_duration": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "TRISC_0", "zone_name": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_0", "zone_name": "TRISC-KERNEL"},
        },
        "device_trisc1_kernel_duration": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "TRISC_1", "zone_name": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_1", "zone_name": "TRISC-KERNEL"},
        },
        "device_trisc2_kernel_duration": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "TRISC_2", "zone_name": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_2", "zone_name": "TRISC-KERNEL"},
        },
        "device_erisc_kernel_duration": {
            "across": "ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "ERISC", "zone_name": "ERISC-KERNEL"},
            "end": {"core": "ANY", "risc": "ERISC", "zone_name": "ERISC-KERNEL"},
        },
        "device_compute_cb_wait_front": {
            "across": "ops",
            "type": "sum",
            "marker": {"risc": "TRISC_0", "zone_name": "CB-COMPUTE-WAIT-FRONT"},
        },
        "device_compute_cb_reserve_back": {
            "across": "ops",
            "type": "sum",
            "marker": {"risc": "TRISC_2", "zone_name": "CB-COMPUTE-RESERVE-BACK"},
        },
        "dispatch_total_cq_cmd_op_time": {
            "across": "dispatch_ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "BRISC", "zone_name": "CQ_DISPATCH_*"},
            "end": {"core": "ANY", "risc": "BRISC", "zone_name": "CQ_DISPATCH_*"},
        },
        "dispatch_go_send_wait_time": {
            "across": "dispatch_ops",
            "type": "op_first_last",
            "start": {"core": "ANY", "risc": "NCRISC", "zone_name": "CQ_DISPATCH_CMD_SEND_GO_SIGNAL"},
            "end": {"core": "ANY", "risc": "NCRISC", "zone_name": "CQ_DISPATCH_CMD_SEND_GO_SIGNAL"},
        },
    }

    displayStats = ["Count", "Average", "Max", "Median", "Min", "Sum", "Range"]

    detectOps = True

    outputFolder = f"output/device"
    deviceInputLog = f"{PROFILER_LOGS_DIR}/{PROFILER_DEVICE_SIDE_LOG}"
    deviceAnalysisData = "device_analysis_data.json"
    deviceStatsTXT = "device_stats.txt"
    deviceTarball = "device_perf_results.tgz"


class test_timestamped_events(default_setup):
    timerAnalysis = {
        "erisc_events": {
            "across": "device",
            "type": "event",
            "marker": {"risc": "ERISC"},
        },
        "all_events": {
            "across": "device",
            "type": "event",
            "marker": {"risc": "ANY"},
        },
    }
    detectOps = False


class test_multi_op(default_setup):
    timerAnalysis = {
        "BRISC KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "BRISC", "zone_name": "BRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "BRISC", "zone_name": "BRISC-KERNEL"},
        },
    }
    detectOps = False


class test_custom_cycle_count(default_setup):
    timerAnalysis = {
        "BRISC KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "BRISC", "zone_name": "BRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "BRISC", "zone_name": "BRISC-KERNEL"},
        },
        "NCRISC KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "NCRISC", "zone_name": "NCRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "NCRISC", "zone_name": "NCRISC-KERNEL"},
        },
        "TRISC_0 KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "TRISC_0", "zone_name": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_0", "zone_name": "TRISC-KERNEL"},
        },
        "TRISC_1 KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "TRISC_1", "zone_name": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_1", "zone_name": "TRISC-KERNEL"},
        },
        "TRISC_2 KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "TRISC_2", "zone_name": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_2", "zone_name": "TRISC-KERNEL"},
        },
    }
    detectOps = False


class test_custom_cycle_count_slow_dispatch(default_setup):
    timerAnalysis = {
        "BRISC KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "BRISC", "zone_name": "BRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "BRISC", "zone_name": "BRISC-KERNEL"},
        },
        "NCRISC KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "NCRISC", "zone_name": "NCRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "NCRISC", "zone_name": "NCRISC-KERNEL"},
        },
        "TRISC_0 KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "TRISC_0", "zone_name": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_0", "zone_name": "TRISC-KERNEL"},
        },
        "TRISC_1 KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "TRISC_1", "zone_name": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_1", "zone_name": "TRISC-KERNEL"},
        },
        "TRISC_2 KERNEL_START->KERNEL_END": {
            "across": "core",
            "type": "adjacent",
            "start": {"core": "ANY", "risc": "TRISC_2", "zone_name": "TRISC-KERNEL"},
            "end": {"core": "ANY", "risc": "TRISC_2", "zone_name": "TRISC-KERNEL"},
        },
    }
    detectOps = False


class test_full_buffer(default_setup):
    timerAnalysis = {
        "Marker Repeat": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "ANY", "zone_name": "TEST-FULL"},
            "end": {"risc": "ANY", "zone_name": "TEST-FULL"},
        },
        "Marker Repeat ETH": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "ERISC", "zone_name": "TEST-FULL"},
            "end": {"risc": "ERISC", "zone_name": "TEST-FULL"},
        },
    }
    detectOps = False


class test_dispatch_cores(default_setup):
    timerAnalysis = {
        "Tensix CQ Dispatch Write Offset": {
            "across": "core",
            "type": "adjacent",
            "start": {
                "risc": "BRISC",
                "zone_name": "CQ-DISPATCH:0:CQ_DISPATCH_CMD_SET_WRITE_OFFSET:0:CQ_DISPATCH_CMD_SET_WRITE_OFFSET",
            },
            "end": {
                "risc": "BRISC",
                "zone_name": "CQ-DISPATCH:0:CQ_DISPATCH_CMD_SET_WRITE_OFFSET:0:CQ_DISPATCH_CMD_SET_WRITE_OFFSET",
            },
        },
        "Tensix CQ Dispatch Terminate": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "BRISC", "zone_name": "CQ-DISPATCH:0:CQ_DISPATCH_CMD_TERMINATE"},
            "end": {"risc": "BRISC", "zone_name": "CQ-DISPATCH:0:CQ_DISPATCH_CMD_TERMINATE"},
        },
        "Tensix CQ Dispatch Wait": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "BRISC", "zone_name": "CQ-DISPATCH:0:CQ_DISPATCH_CMD_WAIT"},
            "end": {"risc": "BRISC", "zone_name": "CQ-DISPATCH:0:CQ_DISPATCH_CMD_WAIT"},
        },
        "Tensix CQ Dispatch Write Linear": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "BRISC", "zone_name": "CQ-DISPATCH:0:CQ_DISPATCH_CMD_WRITE_LINEAR"},
            "end": {"risc": "BRISC", "zone_name": "CQ-DISPATCH:0:CQ_DISPATCH_CMD_WRITE_LINEAR"},
        },
        "Tensix CQ Dispatch Write Linear H Host": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "BRISC", "zone_name": "CQ-DISPATCH:0:CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST"},
            "end": {"risc": "BRISC", "zone_name": "CQ-DISPATCH:0:CQ_DISPATCH_CMD_WRITE_LINEAR_H_HOST"},
        },
        "Tensix CQ Dispatch Write Packed Large Type Program Binaries": {
            "across": "core",
            "type": "adjacent",
            "start": {
                "risc": "BRISC",
                "zone_name": "CQ-DISPATCH:0:CQ_DISPATCH_CMD_WRITE_PACKED_LARGE:0:CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_TYPE_PROGRAM_BINARIES",
            },
            "end": {
                "risc": "BRISC",
                "zone_name": "CQ-DISPATCH:0:CQ_DISPATCH_CMD_WRITE_PACKED_LARGE:0:CQ_DISPATCH_CMD_PACKED_WRITE_LARGE_TYPE_PROGRAM_BINARIES",
            },
        },
        "Tensix CQ Dispatch Write Packed Flag Type Event": {
            "across": "core",
            "type": "adjacent",
            "start": {
                "risc": "BRISC",
                "zone_name": "CQ-DISPATCH:0:CQ_DISPATCH_CMD_WRITE_PACKED:0:CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_TYPE_EVENT",
            },
            "end": {
                "risc": "BRISC",
                "zone_name": "CQ-DISPATCH:0:CQ_DISPATCH_CMD_WRITE_PACKED:0:CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_TYPE_EVENT",
            },
        },
        "Tensix CQ Dispatch Write Packed Flag Type Launch": {
            "across": "core",
            "type": "adjacent",
            "start": {
                "risc": "BRISC",
                "zone_name": "CQ-DISPATCH:0:CQ_DISPATCH_CMD_WRITE_PACKED:0:MCAST;CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_TYPE_LAUNCH",
            },
            "end": {
                "risc": "BRISC",
                "zone_name": "CQ-DISPATCH:0:CQ_DISPATCH_CMD_WRITE_PACKED:0:MCAST;CQ_DISPATCH_CMD_PACKED_WRITE_FLAG_TYPE_LAUNCH",
            },
        },
        "Tensix CQ Dispatch Write Paged": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "BRISC", "zone_name": "CQ-DISPATCH:0:CQ_DISPATCH_CMD_WRITE_PAGED"},
            "end": {"risc": "BRISC", "zone_name": "CQ-DISPATCH:0:CQ_DISPATCH_CMD_WRITE_PAGED"},
        },
        "Tensix CQ Dispatch Notify Subordinate Go Signal": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "BRISC", "zone_name": "CQ-DISPATCH:0:CQ_DISPATCH_NOTIFY_SUBORDINATE_GO_SIGNAL"},
            "end": {"risc": "BRISC", "zone_name": "CQ-DISPATCH:0:CQ_DISPATCH_NOTIFY_SUBORDINATE_GO_SIGNAL"},
        },
        "Tensix CQ Prefetch": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "BRISC", "zone_name": "CQ-PREFETCH"},
            "end": {"risc": "BRISC", "zone_name": "CQ-PREFETCH"},
        },
    }
    detectOps = False


class test_ethernet_dispatch_cores(default_setup):
    timerAnalysis = {
        "Ethernet CQ Dispatch": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "ERISC", "zone_name": "CQ-DISPATCH"},
            "end": {"risc": "ERISC", "zone_name": "CQ-DISPATCH"},
        },
        "Ethernet CQ Prefetch": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "ERISC", "zone_name": "CQ-PREFETCH"},
            "end": {"risc": "ERISC", "zone_name": "CQ-PREFETCH"},
        },
    }
    detectOps = False


class test_noc(default_setup):
    timerAnalysis = {
        "NoC For Loop": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "zone_name": "NOC-FOR-LOOP"},
            "end": {"risc": "NCRISC", "zone_name": "NOC-FOR-LOOP"},
        }
    }
    detectOps = False
