import sys
sys.path.insert(0, '/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/v3l')
from loopstat import fn_range, dis
import re
elf = sys.argv[1]; sub = sys.argv[2] if len(sys.argv) > 2 else "e0::run_main_loop"
a, b = fn_range(elf, sub)
for ad, op, ar in dis(elf, a, b):
    print(f"{ad:x}: {op} {ar}")
