import sys

file_name = sys.argv[1]
f = open(file_name, "r")
lines = f.readlines()[2:]

start = int(lines[4].split(",")[-1])
if len(lines) > 12:
    for line in lines[:20]:
        lst = line.split(",")
        print(lst[:-1], int(lst[-1])-start)
    if len(lines) > 24:
        lines = lines[20:]
        start = int(lines[4].split(",")[-1])
        for line in lines[:20]:
            lst = line.split(",")
            print(lst[:-1], int(lst[-1])-start)
else:
    for line in lines[:8]:
        lst = line.split(",")
        print(lst[:-1], int(lst[-1])-start)
