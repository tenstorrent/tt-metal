import re, subprocess, sys
R = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-readelf"
t = subprocess.run([R, "--debug-dump=info", sys.argv[1]], capture_output=True, text=True).stdout.splitlines()
want = sys.argv[2:]
out = {}
for i, l in enumerate(t):
    m = re.search(r"DW_AT_name\s*:\s*(?:\(.*\):\s*)?(\w+)\s*$", l)
    if m and m.group(1) in want:
        tag = t[i - 1]
        if "DW_TAG_variable" not in tag:
            continue
        for l2 in t[i + 1:i + 8]:
            if "DW_AT_location" in l2:
                out[m.group(1)] = "frame" if "fbreg" in l2 else "split" if "location list" in l2 else "?"
                break
            if "Abbrev Number" in l2:
                break
        out.setdefault(m.group(1), "-")
print(" ".join(f"{k}:{out.get(k, 'n/a')}" for k in want))
