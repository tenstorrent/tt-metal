import re
import sys

# vcdvars.py <vcd> <regex on full name>: print matching scope.var (width)
pat = re.compile(sys.argv[2])
stack = []
n = 0
with open(sys.argv[1], "rb") as f:
    for raw in f:
        l = raw.decode("latin1").strip()
        if l.startswith("$scope"):
            stack.append(l.split()[2])
        elif l.startswith("$upscope"):
            stack.pop()
        elif l.startswith("$var"):
            t = l.split()
            name = ".".join(stack) + "." + t[4]
            if pat.search(name):
                print(name, t[2])
                n += 1
                if n > 200:
                    break
        elif l.startswith("$enddefinitions"):
            break
