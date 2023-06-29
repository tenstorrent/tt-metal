import sys


def print_relative_cycles(file_name):
    f = open(file_name, "r")
    lines = f.readlines()[2:]

    start = 0
    for line in lines:
        if "NCRISC, 2" in line:
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

def profile_noc_async_read_and_barrier_time(file_name):
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
    print("issue:", dic_cycle[6]-dic_cycle[5], "barrier: ", dic_cycle[7]-dic_cycle[6])

file_name = sys.argv[1]
# print_relative_cycles(file_name)
# test_kernel_profiler_overhead(file_name)
# profile_test_kernel_profiler_overhead(file_name)
# extract_in0_row_noc_addr(file_name)
profile_noc_async_read_and_barrier_time(file_name)
