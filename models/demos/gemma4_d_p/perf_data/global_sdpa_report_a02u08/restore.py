"""Restore the checkout's original branch (the MoE session's) and rebuild, after the report runs."""
import subprocess
import sys


def run(c):
    print("+", c, flush=True)
    r = subprocess.run(c, shell=True)
    if r.returncode:
        print("FAILED", c)
        sys.exit(r.returncode)


run("git checkout -q mmanzoor/moe-post-combine-reduce-batched-reads")
run(
    "ninja -C build > /data/mmanzoor/global_sdpa_report_a02u08/restore.ninja.log 2>&1 && ninja -C build install >> /data/mmanzoor/global_sdpa_report_a02u08/restore.ninja.log 2>&1"
)
print("MM restored", flush=True)
