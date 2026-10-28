#!/usr/bin/env python3
"""Crash forensics for the GovReport run: one fsync'd line every 2 s, so the last line survives a host reset.

Logs host memory/swap, the training process's RSS/swap, Blackhole hwmon telemetry (read from the
driver's sysfs, which does not touch the device firmware the way tt-smi does), host CPU temperature,
training-log progress and new dmesg lines. If the training log stops advancing after training has
started, runs tt-triage once against the live process (before anything is killed or reset).

    forensics.py <training_log> <out_dir>

FORENSICS_CARD_INTERVAL (seconds, default 2.0) sets the card-telemetry period; 0.1 gives 10 Hz.
"""

import glob
import os
import re
import subprocess
import sys
import time

TRAIN_LOG, OUT = sys.argv[1], sys.argv[2]
INTERVAL = float(os.environ.get("FORENSICS_CARD_INTERVAL", "2.0"))  # card telemetry period
SLOW_INTERVAL = 2.0  # memory / dmesg / training-log period
STALL_SECONDS = 240  # a training update is ~32 s, an eval ~100 s
TT_METAL = os.environ.get("TT_METAL_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))


def read(path, default=""):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def meminfo():
    vals = {}
    for line in read("/proc/meminfo").splitlines():
        k, v = line.split(":", 1)
        vals[k] = int(v.split()[0]) // 1024  # MiB
    return vals


def train_pid():
    for pid in os.listdir("/proc"):
        if (
            pid.isdigit()
            and "finetune_govreport.py" in read(f"/proc/{pid}/cmdline").replace("\0", " ")
            and read(f"/proc/{pid}/comm").startswith("python")
        ):
            return pid
    return None


def proc_mem(pid):
    status = read(f"/proc/{pid}/status")
    get = lambda key: int(m.group(1)) // 1024 if (m := re.search(rf"{key}:\s+(\d+)", status)) else -1  # noqa: E731
    return get("VmRSS"), get("VmSwap")


def cards():
    out = []
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        if read(f"{h}/name") != "blackhole":
            continue
        bdf = os.path.basename(os.path.realpath(f"{h}/device"))
        out.append(
            f"{bdf[5:7]}:{int(read(f'{h}/temp1_input', '0')) / 1000:.0f}C/"
            f"{int(read(f'{h}/power1_input', '0')) / 1e6:.0f}W/"
            f"{int(read(f'{h}/curr1_input', '0')) / 1000:.0f}A/"
            f"{read(f'{h}/in0_input', '0')}mV"
        )
    return " ".join(out)


def cpu_temp():
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        if read(f"{h}/name") == "k10temp":
            return f"{int(read(f'{h}/temp1_input', '0')) / 1000:.0f}C"
    return "?"


def last_update():
    try:
        with open(TRAIN_LOG, "rb") as f:
            f.seek(max(0, os.path.getsize(TRAIN_LOG) - 20000))
            tail = f.read().decode(errors="replace")
        n = re.findall(r"train/examples_seen (\d+)", tail)
        return int(n[-1]) if n else 0, "LoRA rank" in tail or bool(n)
    except OSError:
        return 0, False


def dmesg_lines():
    try:
        return subprocess.run(["dmesg", "-T"], capture_output=True, text=True, timeout=10).stdout.splitlines()
    except Exception as e:  # keep logging even if dmesg breaks
        return [f"<dmesg failed: {e}>"]


def triage(log):
    base = os.path.join(OUT, f"triage_{time.strftime('%H%M%S')}")
    cmd = [
        f"{TT_METAL}/python_env/bin/python",
        f"{TT_METAL}/tools/tt-triage.py",
        "--llm-output",
        f"--llm-output-path={base}.txt",
        f"--triage-summary-path={base}_summary.txt",
    ]
    env = dict(os.environ, TT_METAL_HOME=TT_METAL, ARCH_NAME="blackhole", PYTHONPATH=TT_METAL)
    emit(log, f"STALL: running {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env, cwd=TT_METAL)
        with open(f"{base}_console.txt", "w") as f:
            f.write(r.stdout + r.stderr)
            f.flush()
            os.fsync(f.fileno())
        emit(log, f"STALL: tt-triage exited {r.returncode}, output {base}*.txt")
    except subprocess.TimeoutExpired:
        emit(log, "STALL: tt-triage timed out after 300 s")


def emit(log, line):
    log.write(line + "\n")
    log.flush()
    os.fsync(log.fileno())


def main():
    os.makedirs(OUT, exist_ok=True)
    seen_dmesg = len(dmesg_lines())
    triaged = False
    seen_pid = False
    with open(os.path.join(OUT, "forensics.log"), "a") as log:
        emit(log, f"=== forensics start {time.strftime('%F %T')} (log {TRAIN_LOG}) ===")
        last_slow = 0.0
        while True:
            if time.time() - last_slow < SLOW_INTERVAL:
                now = time.time()
                emit(log, f"{time.strftime('%T', time.localtime(now))}.{int(now * 10) % 10} cards {cards()}")
                time.sleep(INTERVAL)
                continue
            last_slow = time.time()
            pid = train_pid()
            seen_pid = seen_pid or pid is not None
            m = meminfo()
            rss, swap = proc_mem(pid) if pid else (-1, -1)
            age = time.time() - os.path.getmtime(TRAIN_LOG) if os.path.exists(TRAIN_LOG) else -1
            examples, training = last_update()
            emit(
                log,
                f"{time.strftime('%T')} avail {m.get('MemAvailable', -1)}M swapfree {m.get('SwapFree', -1)}M "
                f"dirty {m.get('Dirty', -1)}M | pid {pid} rss {rss}M swap {swap}M | ex {examples} logage {age:.0f}s "
                f"| cpu {cpu_temp()} | {cards()}",
            )
            lines = dmesg_lines()
            for line in lines[seen_dmesg:]:
                emit(log, f"DMESG {line}")
            seen_dmesg = len(lines)
            if pid and training and age > STALL_SECONDS and not triaged:
                triaged = True
                triage(log)
            if pid is None and seen_pid:
                emit(log, f"=== training process gone at {time.strftime('%F %T')} ===")
                return
            time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
