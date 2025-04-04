import time
from ttexalens.tt_exalens_init import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device

# BASE_ADDRESS = 0x3380 #Wormohle
BASE_ADDRESS = 0x33C0  # Blackhole

SEM_PC_DICT = {
    1: "semaphore::MATH_PACK",
    2: "semaphore::UNPACK_TO_DEST",
    3: "semaphore::UNPACK_OPERAND_SYNC",
    4: "semaphore::PACK_DONE",
    5: "semaphore::UNPACK_SYNC",
    6: "semaphore::UNPACK_MATH_DONE",
    7: "semaphore::MATH_DONE",
    8: "pc::TRISC0",
    9: "pc::TRISC1",
    10: "pc::TRISC2",
}

TYPE_DICT = {
    1: SEM_PC_DICT,
}

CORES_TO_MONITOR = ["0,0", "0,1", "1,0", "1,1"]

RISC_FILTER = [0, 1, 2]

monitor_data = [[0] * len(RISC_FILTER) for _ in range(len(CORES_TO_MONITOR))]


def print_dump(core, risc, code, type):
    print(f"Core: {core}, Risc: {risc}, Code: {code}, Type: {type}")
    if type not in TYPE_DICT:
        return
    dump_desc = TYPE_DICT[type]
    for key, value in dump_desc.items():
        data = read_word_from_device(core, BASE_ADDRESS + (risc * 51 + 1 + key) * 4)
        print(f"{value}: {data}")


def monitor_cores(cores, riscs):
    for i in range(len(cores)):
        for j in range(len(riscs)):
            data = read_word_from_device(cores[i], BASE_ADDRESS + (riscs[j] * 51 * 4))
            if data != monitor_data[i][j]:
                last_code = monitor_data[i][j] & 0xFFFF
                code = data & 0xFFFF
                type = (data >> 16) & 0xFF
                monitor_data[i][j] = data
                if code != last_code + 1:
                    print(
                        f"Warning: Non contiguous code, core: {cores[i]}, risc: {riscs[j]}, last code: {last_code}, current code: {code}"
                    )
                print_dump(cores[i], riscs[j], code, type)


while 1:
    monitor_cores(CORES_TO_MONITOR, RISC_FILTER)
    time.sleep(0.1)
