#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_rs_probe

Description:
    fabric_reduce_scatter hang probe. With RS_PROBE=<path> the op writes <path> (a JSON map: every chip's port and
    final cores, the addresses of its counters and progress words); on a hang this script reads those L1 words on
    every listed core and writes them to <path>.dump. Does nothing without RS_PROBE.

Owner:
    mstaletovic
"""

import json
import os

from metal_device_id_mapping import run as get_metal_device_id_mapping
from triage import ScriptConfig, run_script
from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.tt_exalens_lib import read_word_from_device

script_config = ScriptConfig(
    depends=["metal_device_id_mapping"],
)


def run(args, context: Context):
    path = os.environ.get("RS_PROBE")
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        probe = json.load(f)
    mapping = get_metal_device_id_mapping(args, context)
    names, addrs = probe["names"], probe["addrs"]
    lines = [" ".join(["chip", "dev", "role", "x", "y"] + names)]
    for ch in probe["chips"]:
        dev_id = mapping.get_device_id(ch["dev"])
        dev = context.devices[dev_id if dev_id is not None else ch["dev"]]
        for core in ch["cores"]:
            loc = OnChipCoordinate(core["x"], core["y"], "logical", dev, "tensix")
            vals = [read_word_from_device(loc, a, context=context) for a in addrs]
            row = [f"{ch['coord'][0]},{ch['coord'][1]}", str(ch["dev"]), core["role"].replace(" ", "_")]
            row += [str(core["x"]), str(core["y"])] + [str(v) for v in vals]
            lines.append(" ".join(row))
    with open(path + ".dump", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"dump_rs_probe: {len(lines) - 1} cores -> {path}.dump")
    return None


if __name__ == "__main__":
    run_script()
