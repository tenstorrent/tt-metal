class test_base:
    plotHeight = 9000

    colors = {
        "dark green": "rgba(78, 150, 78, 1.0)",
        "green": "rgba(78, 150, 78, 0.7)",
        "light green": "rgba(78, 150, 78, 0.3)",
        "dark red": "rgba(246, 78, 139, 1.0)",
        "red": "rgba(246, 78, 139, 0.7)",
        "light red": "rgba(246, 78, 139, 0.5)",
        "light light red": "rgba(246, 78, 139, 0.2)",
        "dark blue": "rgba(78, 78, 246, 1.0)",
        "blue": "rgba(78, 78, 246, 0.8)",
        "light blue": "rgba(78, 78, 246, 0.5)",
        "light light blue": "rgba(78, 78, 246, 0.3)",
        "blank": "rgba(255, 255, 255, 0.0)",
    }

    riscTimerCombo = {}
    timerAnalysis = {}


class test_add_two_ints(test_base):
    plotHeight = 500
    riscTimerCombo = {
        "BRISC": [
            ("START", "1", "blank", ""),
            ("1", "2", "green", "Add two nums"),
            ("2", "END", "blank", ""),
        ]
    }

    timerAnalysis = {
        "B_start -> B_end": {
            "type": "diff",
            "start": {"risc": "BRISC", "timerID": "1"},
            "end": {"risc": "BRISC", "timerID": "2"},
        }
    }


class test_matmul_multi_core_multi_dram(test_base):
    riscTimerCombo = {
        "BRISC": [
            ("START", "4", "blank", ""),
            ("4", "5", "green", "Main start and wait on firt tile write"),
            ("5", "6", "light green", "Write tiles"),
            ("6", "END", "blank", ""),
        ],
        "NCRISC": [
            ("START", "1", "blank", ""),
            ("1", "2", "blue", "Main start to first block read"),
            ("2", "3", "light blue", "Reading blocks"),
            ("3", "END", "blank", ""),
        ],
    }

    timerAnalysis = {
        "NC_start -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "1"},
            "end": {"risc": "BRISC", "timerID": "6"},
        },
        "compute~": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "2"},
            "end": {"risc": "BRISC", "timerID": "5"},
        },
        "B_end": {"type": "single", "risc": "BRISC", "timerID": "6"},
    }


class test_matmul_multi_core_multi_dram_in0_mcast(test_base):
    riscTimerCombo = {
        "BRISC": [
            ("START", "4", "blank", ""),
            ("4", "5", "green", "Kernel start and wait on firt tile write"),
            ("5", "6", "light green", "Write tiles"),
            ("6", "END", "blank", ""),
        ],
        "NCRISC": [
            ("START", "11", "blank", ""),
            ("11", "12", "red", "Kernel start to first multicast done"),
            ("12", "13", "light red", "first multi multicast to first block push"),
            ("13", "14", "light light red", "Pushing blocks"),
            ("14", "END", "blank", ""),
            ("START", "7", "blank", ""),
            ("7", "8", "blue", "Kernel start to first NOC signal"),
            ("8", "9", "light blue", "First NOC signal to first block push"),
            ("9", "10", "light light blue", "Pushing blocks"),
            ("10", "END", "blank", ""),
        ],
    }

    timerAnalysis = {
        "NC_start_sender -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "11"},
            "end": {"risc": "BRISC", "timerID": "6"},
        },
        "NC_start_reciever -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "7"},
            "end": {"risc": "BRISC", "timerID": "6"},
        },
    }


class test_matmul_multi_core_multi_dram_in0_mcasti_mainfunc(
    test_matmul_multi_core_multi_dram_in0_mcast
):
    riscTimerCombo = {
        "BRISC": [
            ("START", "15", "blank", ""),
            ("15", "4", "dark green", "Main to kernel start"),
            ("4", "5", "green", "Kernel start and wait on firt tile write"),
            ("5", "6", "light green", "Write tiles"),
            ("6", "END", "blank", ""),
        ],
        "NCRISC": [
            ("START", "16", "blank", ""),
            ("16", "11", "dark red", "Main to kernel start"),
            ("11", "12", "red", "Kernel start to first multicast done"),
            ("12", "13", "light red", "first multi multicast to first block push"),
            ("13", "14", "light light red", "Pushing blocks"),
            ("14", "END", "blank", ""),
            ("16", "7", "dark blue", "Main to kernel start"),
            ("7", "8", "blue", "Kernel start to first NOC signal"),
            ("8", "9", "light blue", "First NOC signal to first block push"),
            ("9", "10", "light light blue", "Pushing blocks"),
            ("10", "END", "blank", ""),
        ],
    }
