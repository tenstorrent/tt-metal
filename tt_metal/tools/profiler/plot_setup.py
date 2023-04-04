from merge_meta_class import MergeMetaclass

class default_setup(metaclass=MergeMetaclass):
    timerAnalysis = {
        "FW start": {
            "across": "risc",
            "type": "adjacent",
            "start": {"risc": "BRISC", "timerID": 0},
            "end": {"risc": "BRISC", "timerID": 1},
        },
        "BRISC kernel start -> BRISC kernel end": {
            "across": "risc",
            "type": "adjacent",
            "start": {"risc": "BRISC", "timerID": 2},
            "end": {"risc": "BRISC", "timerID": 3},
        },
        "NCRISC kernel start -> NCRISC kernel end": {
            "across": "risc",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 2},
            "end": {"risc": "NCRISC", "timerID": 3},
        },
        "Launch delta": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "ANY", "timerID": 4},
            "end": {"risc": "BRISC", "timerID": 1},
        },
        "Core end": {
            "across": "core",
            "type": "first_last",
            "start": {"risc": "BRISC", "timerID": 0},
            "end": {"risc": "ANY", "timerID": 4},
        },
        "Core start -> Core end": {
            "across": "core",
            "type": "first_last",
            "start": {"risc": "ANY", "timerID": 1},
            "end": {"risc": "ANY", "timerID": 4},
        },
        "Device start -> Device end": {
            "across": "device",
            "type": "first_last",
            "start": {"core":"ANY", "risc": "ANY", "timerID": 1},
            "end": {"core":"ANY", "risc": "ANY", "timerID": 4},
        },
    }

    riscsData = {
        'BRISC': {
            "color":"light:g"
        },
        'NCRISC': {
            "color":"light:r"
        },
        'TENSIX': {
            "color":"light:gray"
        }

    }

    riscs = ["BRISC","NCRISC","TENSIX"]

    timerIDLabels = [
        (0, "Start"),
        (1, "Firmware Start"),
        (2, "Data Movement Kernel start"),
        (3, "Data Movement Kernel End"),
        (4, "Firmware End"),
    ]

    coreFreq = 1.2 #GHz

    displayStats = ["Count","Average","Max","Median","Min"]

    plotBaseHeight = 200
    plotPerCoreHeight = 90

    webappPort = 8050

    outputFolder = "output"
    deviceInputLog = "logs/profile_log_device.csv"
    deviceRearranged = "device_rearranged_timestamps.csv"
    devicePerfHTML = "timeline.html"
    deviceStatsTXT = "device_stats.txt"
    deviceTarball = "device_perf_results.tgz"


class test_matmul_multi_core_multi_dram(default_setup):
    timerAnalysis = {
        "Compute~": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 6},
            "end": {"risc": "BRISC", "timerID": 5},
        },
    }


class test_matmul_multi_core_multi_dram_in0_mcast(default_setup):
    timerAnalysis = {
        "NCRISC start sender -> BRISC kernel end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 10},
            "end": {"risc": "BRISC", "timerID": 3},
        },
        "NCRISC start reciever -> BRISC kernel end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 7},
            "end": {"risc": "BRISC", "timerID": 3},
        }
    }


class test_matmul_multi_core_multi_dram_in1_mcast(default_setup):
    timerAnalysis = {
        "NCRISC start sender -> BRISC kernel end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 20},
            "end": {"risc": "BRISC", "timerID": 3},
        },
        "NCRISC start reciever -> BRISC kernel end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 16},
            "end": {"risc": "BRISC", "timerID": 3},
        }
    }

class test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast(default_setup):
    timerAnalysis = {
        "NC_in0_s_in1_r -> B_end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 24},
            "end": {"risc": "BRISC", "timerID": 3},
        },
        "NC_in0_s_in1_s -> B_end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 29},
            "end": {"risc": "BRISC", "timerID": 3},
        },
        "NC_in0_r_in1_r -> B_end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 34},
            "end": {"risc": "BRISC", "timerID": 3},
        },
        "NC_in0_r_in1_s -> B_end": {
            "across": "core",
            "type": "adjacent",
            "start": {"risc": "NCRISC", "timerID": 39},
            "end": {"risc": "BRISC", "timerID": 3},
        },
    }
