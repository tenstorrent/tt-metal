import sys
import argparse

def print_relative_cycles(file_name):
    f = open(file_name, "r")
    lines = f.readlines()[2:]

    start = 0
    for line in lines:
        if "BRISC, 1" in line:
            start = int(line.split(",")[-1])
            break

    stop_flag = 0   # Keep the first iteration (Addition) in Eltwise_binary and filter out the second iteration (Multiplication)

    for line in lines:
        if "BRISC" in line:
            stop_flag = 1
        if "NCRISC" in line and stop_flag:
            break
        lst = line.split(",")
        # print(int(lst[-1])-start)
        print(lst[:-1], int(lst[-1])-start)

def test_kernel_profiler_overhead(file_name):
    f = open(file_name, "r")
    lines = f.readlines()[2:]
    cycle_list = []
    for line in lines:
        profiler_ID = int(line.split(",")[4])
        if profiler_ID in [i for i in range(5, 17)]:
            cycle_list.append(int(line.split(",")[-1]))
    for i in range(len(cycle_list)-1):
        print(cycle_list[i+1]-cycle_list[i])

def profile_test_kernel_profiler_overhead(file_name):
    f = open(file_name, "r")
    lines = f.readlines()
    cycle_list = []
    c_lst = []
    for i in range(len(lines)):
        if i%11 == 0:
            if len(c_lst) == 11:
                lst = c_lst.copy()
                cycle_list.append(lst)
            c_lst = []
        c_lst.append(int(lines[i]))
    flag = True
    if len(cycle_list) > 0:
        print(sum(cycle_list[0])/11)
    else:
        print(sum(c_lst)/11)

    for i in range(1, len(cycle_list)):
        if cycle_list[i] != cycle_list[0]:
            flag = False
            print(cycle_list[i], cycle_list[0])

def extract_in0_row_noc_addr(file_name):
    f = open(file_name, "r")
    lines = f.readlines()
    for line in lines:
        if line[:16] == "in0_row_noc_addr":
            print("noc_async_read({}, {}, dim_x);".format(line.split()[-3], line.split()[-1]))

def profile_noc_async_read_and_barrier_time_NCRISC(file_name):
    f = open(file_name, "r")
    lines = f.readlines()
    dic_cycle = {}
    for line in lines:
        if "0, 0, 0, NCRISC, 5, " in line:
            dic_cycle[5] = int(line.split(",")[-1])
        elif "0, 0, 0, NCRISC, 6, " in line:
            dic_cycle[6] = int(line.split(",")[-1])
        elif "0, 0, 0, NCRISC, 7, " in line:
            dic_cycle[7] = int(line.split(",")[-1])
    print("issue:", dic_cycle[6]-dic_cycle[5], "barrier:", dic_cycle[7]-dic_cycle[6])

def profile_Tensix2Tensix_issue_barrier(file_name, read_or_write):
    f = open(file_name, "r")
    lines = f.readlines()
    dic_cycle = {5:[], 6:[], 7:[]}
    if read_or_write == "read":
        prefix = "0, 1, 0, BRISC"
        overhead = 35
    elif read_or_write == "write":
        prefix = "0, 0, 0, NCRISC"
        overhead = 27
    for line in lines:
        if prefix + ", 5, " in line:
            dic_cycle[5].append(int(line.split(",")[-1]))
        elif prefix + ", 6, " in line:
            dic_cycle[6].append(int(line.split(",")[-1]))
        elif prefix + ", 7, " in line:
            dic_cycle[7].append(int(line.split(",")[-1]))

    for i in range(4):
        print("issue:", dic_cycle[6][i]-dic_cycle[5][i]-overhead, "barrier:", dic_cycle[7][i]-dic_cycle[6][i]-overhead)

def profile_noc_cmd_buf_write_reg(file_name, read_or_write):
    f = open(file_name, "r")
    lines = f.readlines()
    dic_cycle = {}
    if read_or_write == "read":
        prefix = "0, 1, 0, BRISC"
        overhead = 35
    elif read_or_write == "write":
        prefix = "0, 0, 0, NCRISC"
        overhead = 27
    for line in lines:
        lst = line.split()
        marker = lst[-2][:-1]
        if prefix in line:
            dic_cycle[int(marker)] = int(line.split(",")[-1])

    print("8:", dic_cycle[9]-dic_cycle[8]-overhead, end=" ")
    print()

def profile_Tensix2Tensix_fine_grain(file_name, read_or_write):
    f = open(file_name, "r")
    lines = f.readlines()
    dic_cycle = {}
    if read_or_write == "read":
        prefix = "0, 1, 0, BRISC"
        overhead = 35
    elif read_or_write == "write":
        prefix = "0, 0, 0, NCRISC"
        overhead = 27
    for line in lines:
        lst = line.split()
        marker = lst[-2][:-1]
        if prefix in line:
            dic_cycle[int(marker)] = int(line.split(",")[-1])

    print("5:", dic_cycle[11]-dic_cycle[5]-overhead, end=" ")
    for i in range(11, 18):
        print("{}:".format(i), dic_cycle[i+1]-dic_cycle[i]-overhead, end=" ")
    print("18:", dic_cycle[6]-dic_cycle[18]-overhead, end=" ")
    print()

def profile_overhead(file_name, read_or_write):
    f = open(file_name, "r")
    lines = f.readlines()
    dic_cycle = {}
    if read_or_write == "read":
        prefix = "0, 1, 0, BRISC"
    elif read_or_write == "write":
        prefix = "0, 0, 0, NCRISC"
    for line in lines:
        lst = line.split()
        marker = lst[-2][:-1]
        if prefix in line:
            dic_cycle[int(marker)] = int(line.split(",")[-1])

    for i in range(5, 16):
        print("{}:".format(i), dic_cycle[i+1]-dic_cycle[i], end=" ")
    print()

def profile_RISC_overhead(file_name, prefix):
    f = open(file_name, "r")
    lines = f.readlines()
    dic_cycle = {}
    for line in lines:
        lst = line.split()
        marker = lst[-2][:-1]
        if prefix in line:
            dic_cycle[int(marker)] = int(line.split(",")[-1])
    lst = []
    print(prefix.split()[-1], end=" ")
    for i in range(5, 16):
        lst.append(dic_cycle[i+1]-dic_cycle[i])
        print("{}:".format(i), dic_cycle[i+1]-dic_cycle[i], end=" ")
    print("avg:", sum(lst)/len(lst)*1.0)

def print_Elewise_binary_fine_grain(dic, name, function):
    print(name)
    overhead_dic = {"NCRISC":38, "BRISC":45, "TRISC_0":49, "TRISC_1":55, "TRISC_2": 43}
    overhead = overhead_dic[name]
    for key in dic.keys():
        if key > 4 and key + 1 in dic.keys() and "{}-{}".format(key, key+1) in function.keys():
            print(function["{}-{}".format(key, key+1)], dic[key+1] - dic[key] - overhead)
        elif not key + 1 in dic.keys() and "{}-{}".format(key, 4) in function.keys():
            print(function["{}-{}".format(key, 4)], dic[4] - dic[key] - overhead)

def profile_Elewise_binary_fine_grain(file_name):
    f = open(file_name, "r")
    lines = f.readlines()
    read_dic = {}
    write_dic = {}
    unpack_dic = {}
    math_dic = {}
    pack_dic = {}
    prefix_read = "0, 0, 0, NCRISC"
    prefix_write = "0, 0, 0, BRISC"
    prefix_unpack = "0, 0, 0, TRISC_0"
    prefix_math = "0, 0, 0, TRISC_1"
    prefix_pack = "0, 0, 0, TRISC_2"

    read_function = {"5-6": "cb_reserve_back", "6-7": "noc_async_read", "7-8": "noc_async_read_barrier", "8-9": "cb_push_back", "10-11": "cb_reserve_back", "11-12": "noc_async_read", "12-13": "noc_async_read_barrier", "13-14": "cb_push_back"}
    write_function = {"5-6": "cb_wait_front", "6-7": "noc_async_write", "7-8": "noc_async_write_barrier", "8-9": "cb_pop_front"}
    unpack_function = {"5-6": "cb_wait_front", "6-7": "cb_wait_front", "7-8": "cb_pop_front", "8-9": "cb_pop_front"}
    math_function = {"5-6": "acquire_dst", "6-7": "ELTWISE_OP", "7-8": "release_dst"}
    pack_function = {"5-6": "cb_reserve_back", "6-7": "acquire_dst", "7-8": "pack_tile", "8-9": "release_dst", "9-10": "cb_push_back"}


    for line in lines:
        lst = line.split()
        if prefix_read in line:
            read_dic[int(lst[-2][:-1])] = int(lst[-1])
        elif prefix_write in line:
            write_dic[int(lst[-2][:-1])] = int(lst[-1])
        elif prefix_unpack in line:
            unpack_dic[int(lst[-2][:-1])] = int(lst[-1])
        elif prefix_math in line:
            math_dic[int(lst[-2][:-1])] = int(lst[-1])
        elif prefix_pack in line:
            pack_dic[int(lst[-2][:-1])] = int(lst[-1])

    print_Elewise_binary_fine_grain(read_dic, "NCRISC", read_function)
    print_Elewise_binary_fine_grain(write_dic, "BRISC", write_function)
    print_Elewise_binary_fine_grain(unpack_dic, "TRISC_0", unpack_function)
    print_Elewise_binary_fine_grain(math_dic, "TRISC_1", math_function)
    print_Elewise_binary_fine_grain(pack_dic, "TRISC_2", pack_function)

def profile_Elewise_binary_read_write(file_name):
    f = open(file_name, "r")
    lines = f.readlines()
    read_dic = {}
    write_dic = {}
    prefix_read = "0, 0, 0, NCRISC"
    prefix_write = "0, 0, 0, BRISC"

    read_function = {"7-8": "noc_async_read_barrier"}
    write_function = {"7-8": "noc_async_write_barrier"}


    for line in lines:
        lst = line.split()
        if prefix_read in line:
            read_dic[int(lst[-2][:-1])] = int(lst[-1])
        elif prefix_write in line:
            write_dic[int(lst[-2][:-1])] = int(lst[-1])

    print_Elewise_binary_fine_grain(read_dic, "NCRISC", read_function)
    print_Elewise_binary_fine_grain(write_dic, "BRISC", write_function)


def get_args():
    parser = argparse.ArgumentParser('Profile raw results.')
    parser.add_argument("--file-name", help="file to profile")
    parser.add_argument("--profile-target", choices=["profile_Tensix2Tensix_issue_barrier", "profile_Tensix2Tensix_fine_grain", "profile_overhead", "profile_noc_cmd_buf_write_reg", "profile_elewise_binary", "profile_Elewise_binary_fine_grain", "profile_RISC_overhead", "profile_Elewise_binary_read_write"], help="profile target choice")
    parser.add_argument("--read-or-write", choices=["read", "write"], help="read or write choice")
    parser.add_argument("--log-prefix", default="0, 0, 0, NCRISC", type=str, help="core coordination and RISCV core type prefix in profile log")
    args = parser.parse_args()
    return args

args = get_args()
file_name = args.file_name
if args.profile_target == "profile_Tensix2Tensix_issue_barrier":
    profile_Tensix2Tensix_issue_barrier(file_name, args.read_or_write)
if args.profile_target == "profile_Tensix2Tensix_fine_grain":
    profile_Tensix2Tensix_fine_grain(file_name, args.read_or_write)
if args.profile_target == "profile_noc_cmd_buf_write_reg":
    profile_noc_cmd_buf_write_reg(file_name, args.read_or_write)
if args.profile_target == "profile_overhead":
    profile_overhead(file_name, args.read_or_write)
if args.profile_target == "profile_elewise_binary":
    print_relative_cycles(file_name)
if args.profile_target == "profile_Elewise_binary_fine_grain":
    profile_Elewise_binary_fine_grain(file_name)
if args.profile_target == "profile_RISC_overhead":
    profile_RISC_overhead(file_name, args.log_prefix)
if args.profile_target == "profile_Elewise_binary_read_write":
    profile_Elewise_binary_read_write(file_name)
