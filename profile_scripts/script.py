import sys
import argparse
import copy

def profile_issue_barrier(file_name):
    f = open(file_name, "r")
    lines = f.readlines()
    issue = []
    barrier = []
    buffer = 0
    transaction = 0
    for line in lines:
        if "Buffer" in line:
            if len(issue) > 0:
                issue_avg = sum(issue)/len(issue)
                barrier_avg = sum(barrier)/len(barrier)
                print("Buffer: {} Transaction: {} issue: {:.2f} barrier: {:.2f}".format(buffer, transaction, issue_avg, barrier_avg))
                issue = []
                barrier = []
            buffer = int(line.split()[1])
            transaction = int(line.split()[-1])
        elif "issue" in line:
            lst = line.split()
            issue.append(int(lst[1]))
            barrier.append(int(lst[-1]))
    issue_avg = sum(issue)/len(issue)
    barrier_avg = sum(barrier)/len(barrier)
    print("Buffer: {} Transaction: {} issue: {:.2f} barrier: {:.2f}".format(buffer, transaction, issue_avg, barrier_avg))
    issue = []
    barrier = []


def print_dram_rw(dic, read_write_bar):
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
    print_dram_rw(dic, 1)
    print_dram_rw(dic, 0)

def print_tensix_bandwidth(dic):
    for transaction_power in range(6, 18):
        print(2**transaction_power, end=" ")
    print()
    for buffer_power in range(6, 18):
        print(2**buffer_power, end=" ")
        for transaction_power in range(6, buffer_power+1):
            buffer = 2**buffer_power
            transaction = 2**transaction_power
            for tup in dic.keys():
                if buffer == tup[0] and transaction == tup[1]:
                        print("{:.2f}".format(dic[tup]), end=" ")
        print()

def profile_riscv_tensix(file_name, read_write_bar):
    if read_write_bar:
        marker = "Read"
    else:
        marker = "Write"
    print(marker)
    dic = {}
    f = open(file_name, "r")
    lines = f.readlines()
    for line in lines:
        lst = line.split()
        if "Test arguments" in line:
            transaction = int(lst[-7][:-1])
            buffer = int(lst[-13][:-1])
        elif marker + " speed GB/s" in line:
            time = float(lst[-1])
            dic[(buffer, transaction)] = time
    print_tensix_bandwidth(dic)

def print_tensix_issue_barrier(file_name):
    dic_issue = {}
    dic_barrier = {}
    f = open(file_name, "r")
    lines = f.readlines()
    for line in lines:
        lst = line.split()
        if "Buffer" in line:
            transaction = int(lst[3])
            buffer = int(lst[1])
            issue = float(lst[5])
            barrier = float(lst[-1])
            dic_issue[(buffer, transaction)] = issue
            dic_barrier[(buffer, transaction)] = barrier
        elif "write" in line:
            print("read")
            print("issue")
            print_tensix_bandwidth(dic_issue)
            print("barrier")
            print_tensix_bandwidth(dic_barrier)
            dic = {}
    print("write")
    print("issue")
    print_tensix_bandwidth(dic_issue)
    print("barrier")
    print_tensix_bandwidth(dic_barrier)



def get_args():
    parser = argparse.ArgumentParser('Profile raw results.')
    parser.add_argument("--file-name", help="file to profile")
    parser.add_argument("--profile-target", choices=["Tensix2Tensix", "DRAM2Tensix", "Tensix2Tensix_Issue_Barrier", "Print_Tensix2Tensix_Issue_Barrier"], help="profile target choice")
    parser.add_argument("--read-or-write", choices=["read", "write"], help="read or write choice")
    args = parser.parse_args()
    return args

args = get_args()
file_name = args.file_name
if args.profile_target == "Tensix2Tensix":
    if args.read_or_write == "read":
        profile_riscv_tensix(file_name, 1)
    elif args.read_or_write == "write":
        profile_riscv_tensix(file_name, 0)
elif args.profile_target == "DRAM2Tensix":
    profile_riscv_rw_dram(file_name)
elif args.profile_target == "Tensix2Tensix_Issue_Barrier":
    profile_issue_barrier(file_name)
elif args.profile_target == "Print_Tensix2Tensix_Issue_Barrier":
    print_tensix_issue_barrier(file_name)
