import sys, re, subprocess, collections
sys.path.insert(0, '/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3l')
from loopstat import fn_range
OD = "/home/aagarwal/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump"
elf, sub = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "line"
a, b = fn_range(elf, sub)
out = subprocess.run([OD, "-d", "-l", "-C", "--no-show-raw-insn", f"--start-address={a:#x}", f"--stop-address={b:#x}", elf], capture_output=True, text=True).stdout
cur_fn = "?"; cur_line = "?"
cnt = collections.Counter()
for l in out.splitlines():
    m = re.match(r"^(\S.*)\(\):$", l)
    if m: cur_fn = m.group(1)[:90]; continue
    m = re.match(r"^(/\S+):(\d+)", l)
    if m: cur_line = m.group(1).split('/')[-1] + ":" + m.group(2); continue
    if re.match(r"\s+[0-9a-f]+:\s", l):
        cnt[cur_line if mode == "line" else cur_fn] += 4
tot = sum(cnt.values())
print("total", tot)
for k, v in cnt.most_common(int(sys.argv[4]) if len(sys.argv) > 4 else 40):
    print(v, k)
