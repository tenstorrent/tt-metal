"""Set or restore the per-chip TDP limit at run time through the documented CMFW message SET_TDP_LIMIT (0x22, in CMFW
since 19.8.0: arg0 = limit in W, arg1 bit 0 = restore the default). Prints the limit telemetry before and after.
Usage: set_tdp.py <watts>   |   set_tdp.py restore"""
import sys
import time
import pyluwen

arg = sys.argv[1]
chips = pyluwen.detect_chips()
for i, c in enumerate(chips):
    t = c.get_telemetry()
    print(
        f"chip {i} before: tdp_limit_max {t.tdp_limit_max} W, board_power_limit {t.board_power_limit} W, tdp now {t.tdp} W"
    )
for i, c in enumerate(chips):
    if arg == "restore":
        r = c.arc_msg(0x22, wait_for_done=True, arg0=0, arg1=1, timeout=2.0)
    else:
        r = c.arc_msg(0x22, wait_for_done=True, arg0=int(arg), arg1=0, timeout=2.0)
    print(f"chip {i} SET_TDP_LIMIT reply: {r}")
time.sleep(0.5)
for i, c in enumerate(chips):
    t = c.get_telemetry()
    print(f"chip {i} after:  tdp_limit_max {t.tdp_limit_max} W")
