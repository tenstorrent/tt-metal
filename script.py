import sys

def profile_issue_barrier(file_name):
    f = open(file_name, "r")
    lines = f.readlines()
    issue = []
    barrier = []
    for line in lines:
        lst = line.split()
        issue.append(int(lst[1]))
        barrier.append(int(lst[-1]))
    issue_avg = sum(issue)/len(issue)
    barrier_avg = sum(barrier)/len(barrier)
    print("issue: {:.2f} barrier: {:.2f}".format(issue_avg, barrier_avg))

def print_read(dic, read_write_bar):
    if read_write_bar:
        print("Read Speed")
    else:
        print("Write Speed")
    for transaction_power in range(6, 14):
        print(2**transaction_power, end=" ")
    print()
    for buffer_power in range(13, 20):
        print(2**buffer_power, end=" ")
        for transaction_power in range(6, 14):
            buffer = 2**buffer_power
            transaction = 2**transaction_power
            for tup in dic.keys():
                if buffer == tup[0] and transaction == tup[1]:
                    if read_write_bar:
                        print("{:.2f}".format(dic[tup][0]), end=" ")
                    else:
                        print("{:.2f}".format(dic[tup][1]), end=" ")
        print()

def profile_riscv_rw_dram(file_name):
    dic = {}
    f = open(file_name, "r")
    lines = f.readlines()
    for line in lines:
        lst = line.split()
        if "Test arguments" in line:
            transaction = int(lst[-1])
            buffer = int(lst[-7][:-1])
        elif "Read speed GB/s" in line:
            read = float(lst[-1])
        elif "Write speed GB/s" in line:
            write = float(lst[-1])
        elif "Test " in line:
            dic[(buffer, transaction)] = (read, write)
    print_read(dic, 1)
    print_read(dic, 0)



file_name = sys.argv[1]
# profile_issue_barrier(file_name)
profile_riscv_rw_dram(file_name)
