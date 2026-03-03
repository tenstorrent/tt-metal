#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Usage: dump_l1_cores.py [--num-devices=N] <core> [<core> ...]
#   core: R,C for tensix or ER,C for eth  (e.g. 2,1 or E0,3)
#   --num-devices=N: number of devices to dump (default: 1)

import socket, subprocess, sys

HOSTNAME = socket.gethostname()
L1_WORDS = {"tensix": 1536 * 1024 // 4, "eth": 512 * 1024 // 4}  # blackhole


def parse_core(spec):
    eth = spec.upper().startswith("E")
    r, c = spec.lstrip("Ee").split(",")
    return ("eth" if eth else "tensix"), int(r), int(c)


def get_all_device_ids():
    from ttexalens.tt_exalens_init import init_ttexalens

    return list(init_ttexalens().device_ids)


args = sys.argv[1:]
devices_arg = next((a.split("=")[1] for a in args if a.startswith("--devices=")), "0")
if devices_arg == "all":
    device_ids = get_all_device_ids()
else:
    device_ids = [int(d) for d in devices_arg.split(",")]
cores = [a for a in args if not a.startswith("--")]

for dev in device_ids:
    for spec in cores:
        ctype, x, y = parse_core(spec)
        fname = f"bh-glx-{HOSTNAME}_d{dev}_{ctype}_x{x}_y{y}_l1.txt"
        cmd = f'tt-exalens --commands="brxy {spec} 0 {L1_WORDS[ctype]} -d {dev}; exit"'
        print(f"{spec} device {dev} -> {fname}")
        with open(fname, "w") as f:
            subprocess.run(cmd, shell=True, stdout=f)
