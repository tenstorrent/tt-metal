import sys

file_name = sys.argv[1]
f = open(file_name, "r")
lines = f.readlines()[2:]

start = 0
for line in lines:
    if "BRISC" in line:
        start = int(line.split(",")[-1])
        print(line)
        break

print(start)

stop_flag = 0   # Keep the first iteration (Addition) in Eltwise_binary and filter out the second iteration (Multiplication)

for line in lines:
    if "BRISC" in line:
        stop_flag = 1
    if "NCRISC" in line and stop_flag:
        break
    lst = line.split(",")
    # print(int(lst[-1])-start)
    print(lst[:-1], int(lst[-1])-start)
