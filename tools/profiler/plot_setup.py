class test_base:
    plotHeight = 500

    colors = {
        #Green
        "dark green": "rgba(78, 150, 78, 1.0)",
        "green": "rgba(78, 150, 78, 0.7)",
        "light green": "rgba(78, 150, 78, 0.5)",
        "light light green": "rgba(78, 150, 78, 0.2)",
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
        #Gray
        "dark gray": "rgba(0, 0, 0, 0.8)",
        "gray": "rgba(0, 0, 0, 0.6)",
        "light gray": "rgba(0, 0, 0, 0.4)",
        "light light gray": "rgba(0, 0, 0, 0.2)",
        #Transparent
        "blank": "rgba(255, 255, 255, 0.0)",
    }

    riscTimerCombo = {
        "BRISC": [
            ("START", "1", "blank", ""),
            ("1", "2", "dark green", "Main start -> Kernel start"),
            ("2", "3", "green", "Kernel start -> kernel end"),
            ("3", "4", "light green", "kernel end -> Main end"),
            ("4", "END", "blank", ""),
        ],
        "NCRISC": [
            ("START", "1", "blank", ""),
            ("1", "2", "dark red", "Main start -> Kernel start"),
            ("2", "3", "red", "Kernel start -> kernel end"),
            ("3", "4", "light red", "kernel end -> Main end"),
            ("4", "END", "blank", ""),
        ]
    }

    timerAnalysis = {
        "B_start -> B_end": {
            "type": "diff",
            "start": {"risc": "BRISC", "timerID": "2"},
            "end": {"risc": "BRISC", "timerID": "3"},
        },
        "NC_start -> NC_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "2"},
            "end": {"risc": "NCRISC", "timerID": "3"},
        },
    }


class test_pipeline_across_rows(test_base):
    plotHeight = 900

    riscTimerCombo = {
        "NCRISC": [
            ("START", "1", "blank", ""),
            ("1", "2", "green", "NCRISC start to end"),
            ("2", "END", "blank", ""),
            ("START", "5", "blank", ""),
            ("5", "6", "green", "NCRISC start to end"),
            ("6", "END", "blank", ""),
        ],
        "BRISC": [
            ("START", "42", "blank", ""),
            ("42", "420", "orange", ""),

            ("420", "3", "purple", ""),
            ("3", "4", "red", "BRISC start to end"),
            ("4", "END", "blank", ""),

            ("420", "7", "purple", ""),
            ("7", "8", "red", "BRISC start to end"),
            ("8", "END", "blank", ""),
        ],
    }

class test_matmul_multi_core_multi_dram(test_base):
    plotHeight = 9000
    riscTimerCombo = {
        "BRISC": [
            ("START", "1", "blank", ""),
            ("1", "2", "dark green", "Main start -> Kernel start"),
            ("2", "5", "green", "Kernel start -> First write"),
            ("5", "3", "light green", "First write -> kernel end"),
            ("3", "4", "light light green", "kernel end -> Main end"),
            ("4", "END", "blank", ""),
        ],
        "NCRISC": [
            ("START", "1", "blank", ""),
            ("1", "2", "dark red", "Main start -> Kernel start"),
            ("2", "6", "red", "Kernel start -> First read"),
            ("6", "3", "light red", "First read -> kernel end"),
            ("3", "4", "light light red", "kernel end -> Main end"),
            ("4", "END", "blank", ""),
        ],
    }

    timerAnalysis = {
        "NC_start -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "2"},
            "end": {"risc": "BRISC", "timerID": "3"},
        },
        "compute~": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "6"},
            "end": {"risc": "BRISC", "timerID": "5"},
        },
        "B_end": {"type": "single", "risc": "BRISC", "timerID": "4"},
    }

class test_matmul_multi_core_multi_dram_in0_mcast(test_base):
    plotHeight = 9000
    riscTimerCombo = {
        "BRISC": [
            ("START", "1", "blank", ""),
            ("1", "2", "dark green", "Main start -> Kernel start"),
            ("2", "5", "green", "Kernel start -> First write"),
            ("5", "3", "light green", "First write -> kernel end"),
            ("3", "4", "light light green", "kernel end -> Main end"),
            ("4", "END", "blank", ""),
        ],
        "NCRISC": [
            ("START", "1", "blank", ""),
            ("1", "2", "dark red", "Main start -> Kernel start"),

            ("2", "8", "purple", "Kernel start -> First read"),
            ("8", "9", "light purple", "First read -> First CB push"),
            ("9", "3", "light red", "First CB Push -> kernel end"),

            ("2", "11", "orange", "Kernel start -> First read"),
            ("11", "12", "light orange", "First read -> First CB push"),
            ("12", "3", "light red", "First CB Push -> kernel end"),

            ("3", "4", "light light red", "kernel end -> Main end"),
            ("4", "END", "blank", ""),
        ],
    }

    timerAnalysis = {
        "NC_start_sender -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "10"},
            "end": {"risc": "BRISC", "timerID": "3"},
        },
        "NC_start_reciever -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "7"},
            "end": {"risc": "BRISC", "timerID": "3"},
        },
    }

class test_matmul_multi_core_multi_dram_in1_mcast(test_base):
    plotHeight = 9000
    riscTimerCombo = {
        "BRISC": [
            ("START", "1", "blank", ""),
            ("1", "2", "dark green", "Main start -> Kernel start"),
            ("2", "5", "green", "Kernel start -> First write"),
            ("5", "3", "light green", "First write -> kernel end"),
            ("3", "4", "light light green", "kernel end -> Main end"),
            ("4", "END", "blank", ""),
        ],
        "NCRISC": [
            ("START", "1", "blank", ""),
            ("1", "2", "dark red", "Main start -> Kernel start"),

            ("2", "17", "purple", "Kernel start -> First read"),
            ("17", "18", "light purple", "First read -> First CB push"),
            ("18", "3", "light red", "First CB Push -> kernel end"),

            ("2", "21", "orange", "Kernel start -> First read"),
            ("21", "22", "light orange", "First read -> First CB push"),
            ("22", "3", "light red", "First CB Push -> kernel end"),

            ("3", "4", "light light red", "kernel end -> Main end"),
            ("4", "END", "blank", ""),
        ],
    }

    timerAnalysis = {
        "NC_start_sender -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "20"},
            "end": {"risc": "BRISC", "timerID": "3"},
        },
        "NC_start_reciever -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "16"},
            "end": {"risc": "BRISC", "timerID": "3"},
        },
    }

class test_matmul_multi_core_multi_dram_in0_mcast_in1_mcast(test_base):
    plotHeight = 9000
    riscTimerCombo = {
        "BRISC": [
            ("START", "1", "blank", ""),
            ("1", "2", "dark green", "Main start -> Kernel start"),
            ("2", "5", "green", "Kernel start -> First write"),
            ("5", "3", "light green", "First write -> kernel end"),
            ("3", "4", "light light green", "kernel end -> Main end"),
            ("4", "END", "blank", ""),
        ],
        "NCRISC": [
            ("START", "1", "blank", ""),
            ("1", "2", "dark red", "Main start -> Kernel start"),

            #NC_in0_s_in1_r
            ("2", "25", "dark orange", "Kernel start to NOC_0 done"),
            ("25", "26", "orange", "NOC_0 to NOC_1 signal"),
            ("26", "27", "light orange", "NOC_1 to first CB push"),
            ("27", "3", "light light orange", "Pushing blocks"),
            #NC_in0_s_in1_s
            ("2", "30", "dark red", "Kernel start to NOC_0 done"),
            ("30", "31", "red", "NOC_0 to NOC_1 signal"),
            ("31", "32", "light red", "NOC_1 to first CB push"),
            ("32", "3", "light light red", "Pushing blocks"),
            #NC_in0_r_in1_r
            ("2", "35", "dark purple", "Kernel start to NOC_0 done"),
            ("35", "36", "purple", "NOC_0 to NOC_1 signal"),
            ("36", "37", "light purple", "NOC_1 to first CB push"),
            ("37", "3", "light light purple", "Pushing blocks"),
            #NC_in0_r_in1_s
            ("2", "40", "dark blue", "Kernel start to NOC_0 done"),
            ("40", "41", "blue", "NOC_0 to NOC_1 signal"),
            ("41", "42", "light blue", "NOC_1 to first CB push"),
            ("42", "3", "light light blue", "Pushing blocks"),

            ("3", "4", "light light red", "kernel end -> Main end"),
            ("4", "END", "blank", ""),
        ],
    }

    timerAnalysis = {
        "NC_in0_s_in1_r -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "24"},
            "end": {"risc": "BRISC", "timerID": "3"},
        },
        "NC_in0_s_in1_s -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "29"},
            "end": {"risc": "BRISC", "timerID": "3"},
        },
        "NC_in0_r_in1_r -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "34"},
            "end": {"risc": "BRISC", "timerID": "3"},
        },
        "NC_in0_r_in1_s -> B_end": {
            "type": "diff",
            "start": {"risc": "NCRISC", "timerID": "39"},
            "end": {"risc": "BRISC", "timerID": "3"},
        },
    }
