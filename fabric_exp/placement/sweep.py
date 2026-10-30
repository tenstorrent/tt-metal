import subprocess, re, sys, collections
CXX = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-g++"
F = open("flags.txt").read().split()
def cat(line):
    if "credit_addr" in line: return "credit"
    if re.search(r"\br\.(slot|sent|completed)|= r\.", line): return "record"
    if re.search(r"\bw\.|far\.|IN_FLIGHT|in_flight|semaphore", line): return "hot"
    if "while (rd" in line or line.rstrip().endswith("}") or "for (int n" in line: return "loop"
    return "send"
def one(opt, mode, n, extra=()):
    o = f"s_{opt}_{mode}_{n}.o"
    subprocess.run([CXX, f"-{opt}", *F, f"-DMODE={mode}", f"-DN={n}", "-DIFSEP=1", *extra, "-c", "placement_test.cpp", "-o", o], check=True, capture_output=True)
    out = subprocess.run(["python3", "analyze.py", o, "--slots"], capture_output=True, text=True).stdout.splitlines()
    head = out[0]
    m = re.search(r"insn\s+(\d+) sp ld\s+(\d+) sp st\s+(\d+)", head)
    c = collections.Counter()
    for l in out[1:]:
        mm = re.match(r"\s+sp\+\S+\s+ld\s+(\d+) st\s+(\d+)\s+(.*)", l)
        if mm:
            c[cat(mm.group(3).split(" x")[0])] += int(mm.group(1)) + int(mm.group(2))
    return m.groups(), c
opts = sys.argv[1].split(",")
for opt in opts:
    print(f"## -{opt}: pass loop insn / stack ld / stack st; stack accesses by what they hold")
    for n in (1, 2, 3, 4, 6, 8):
        row = []
        for mode in (0, 1, 2):
            (i, ld, st), c = one(opt, mode, n)
            row.append(f"M{mode}: {i:>4}/{ld:>3}/{st:>3} hot {c['hot']:2} send {c['send']:2} rec {c['record']:2} cold {c['credit']:2}")
        print(f"N={n}  " + "   ".join(row))
