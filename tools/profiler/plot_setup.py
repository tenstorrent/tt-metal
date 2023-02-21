class test_base:
    plotHeight = 9000

    colors = {
        #Green
        "dark green": "rgba(78, 150, 78, 1.0)",
        "green": "rgba(78, 150, 78, 0.7)",
        "light green": "rgba(78, 150, 78, 0.3)",
        #Red
        "dark red": "rgba(246, 78, 139, 1.0)",
        "red": "rgba(246, 78, 139, 0.7)",
        "light red": "rgba(246, 78, 139, 0.5)",
        "light light red": "rgba(246, 78, 139, 0.2)",
        #Blue
        "dark blue": "rgba(78, 78, 246, 1.0)",
        "blue": "rgba(78, 78, 246, 0.8)",
        "light blue": "rgba(78, 78, 246, 0.5)",
        "light light blue": "rgba(78, 78, 246, 0.3)",
        #Orange
        "dark orange": "rgba(235, 147, 52, 1.0)",
        "orange": "rgba(235, 147, 52, 0.8)",
        "light orange": "rgba(235, 147, 52, 0.5)",
        "light light orange": "rgba(235, 147, 52, 0.3)",
        #Purple
        "dark purple": "rgba(177, 52, 235, 1.0)",
        "purple": "rgba(177, 52, 235, 0.8)",
        "light purple": "rgba(177, 52, 235, 0.5)",
        "light light purple": "rgba(177, 52, 235, 0.3)",
        #Transparent
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

class test_matmul_multi_core_multi_dram_in1_mcast(test_base):
    riscTimerCombo = {
        "BRISC": [
            ("START", "4", "blank", ""),
            ("4", "5", "green", "Kernel start and wait on firt tile write"),
            ("5", "6", "light green", "Write tiles"),
            ("6", "END", "blank", ""),
        ],
        "NCRISC": [
            ("START", "20", "blank", ""),
            ("20", "21", "red", "Kernel start to first multicast done"),
            ("21", "22", "light red", "First NOC signal to first block push"),
            ("22", "23", "light light red", "Pushing blocks"),
            ("23", "END", "blank", ""),
            ("START", "16", "blank", ""),
            ("16", "17", "blue", "Kernel start to first NOC signal"),
            ("17", "18", "light blue", "First NOC signal to first block push"),
            ("18", "19", "light light blue", "Pushing blocks"),
            ("19", "END", "blank", ""),
        ],
    }

    timerAnalysis = {
        "NC_start_sender -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "20"},
            "end": {"risc": "BRISC", "timerID": "6"},
        },
        "NC_start_reciever -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "16"},
            "end": {"risc": "BRISC", "timerID": "6"},
        },
    }


class test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast(test_base):
    riscTimerCombo = {
        "BRISC": [
            ("START", "4", "blank", ""),
            ("4", "5", "green", "Kernel start and wait on firt tile write"),
            ("5", "6", "light green", "Write tiles"),
            ("6", "END", "blank", ""),
        ],
        "NCRISC": [
            #NC_in0_s_in1_r
            ("START", "24", "blank", ""),
            ("24", "25", "dark orange", "Kernel start to NOC_0 done"),
            ("25", "26", "orange", "NOC_0 to NOC_1 signal"),
            ("26", "27", "light orange", "NOC_1 to first CB push"),
            ("27", "28", "light light orange", "Pushing blocks"),
            ("28", "END", "blank", ""),
            #NC_in0_s_in1_s
            ("START", "29", "blank", ""),
            ("29", "30", "dark red", "Kernel start to NOC_0 done"),
            ("30", "31", "red", "NOC_0 to NOC_1 signal"),
            ("31", "32", "light red", "NOC_1 to first CB push"),
            ("32", "33", "light light red", "Pushing blocks"),
            ("33", "END", "blank", ""),
            #NC_in0_r_in1_r
            ("START", "34", "blank", ""),
            ("34", "35", "dark purple", "Kernel start to NOC_0 done"),
            ("35", "36", "purple", "NOC_0 to NOC_1 signal"),
            ("36", "37", "light purple", "NOC_1 to first CB push"),
            ("37", "38", "light light purple", "Pushing blocks"),
            ("38", "END", "blank", ""),
            #NC_in0_r_in1_s
            ("START", "39", "blank", ""),
            ("39", "40", "dark blue", "Kernel start to NOC_0 done"),
            ("40", "41", "blue", "NOC_0 to NOC_1 signal"),
            ("41", "42", "light blue", "NOC_1 to first CB push"),
            ("42", "43", "light light blue", "Pushing blocks"),
            ("43", "END", "blank", ""),
        ],
    }

    timerAnalysis = {
        "NC_in0_s_in1_r -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "24"},
            "end": {"risc": "BRISC", "timerID": "6"},
        },
        "NC_in0_s_in1_s -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "29"},
            "end": {"risc": "BRISC", "timerID": "6"},
        },
        "NC_in0_r_in1_r -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "34"},
            "end": {"risc": "BRISC", "timerID": "6"},
        },
        "NC_in0_r_in1_s -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "39"},
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
