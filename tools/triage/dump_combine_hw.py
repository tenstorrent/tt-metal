#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Usage:
    dump_combine_hw

Description:
    The combine's hang words (sparse_combine_rf CombineProbes.hw, SPC_HW=1): every combine core's MISC words 476..511
    (ports: loop counters, the blocked row, the layouts present at start; accumulators: rows done / wanted, per
    source taken / still wanted, the most-wanted source's ring head - decoded)
    and its ring layout words (LAYR / LAYO), into generated/moe_block/combine_hw_dump.txt. Runs only when the probed
    combine wrote generated/moe_block/combine_hw.json (the misc / layout addresses and the cores).

Owner:
    mstaletovic
"""

import json
import os

from ttexalens.context import Context
from ttexalens.tt_exalens_lib import read_words_from_device
from triage import ScriptConfig, run_script

script_config = ScriptConfig(depends=["run_checks"])


def run(args, context: Context):
    p = os.path.join(os.environ.get("TT_METAL_HOME", "."), "generated/moe_block/combine_hw.json")
    if not os.path.exists(p):
        return
    cfg = json.load(open(p))
    out = open(os.path.join(os.path.dirname(p), "combine_hw_dump.txt"), "w")
    for dev in context.devices.values():
        for x, y in cfg["cores"]:
            try:
                w = read_words_from_device(
                    f"{x}-{y}", cfg["misc"] + 476 * 4, device_id=dev.id, word_count=36, context=context
                )
            except Exception as e:  # noqa: BLE001
                print(f"COMBINE_HW dev {dev.id} {x}-{y}: read failed {e}", file=out)
                continue
            print(f"COMBINE_HW dev {dev.id} {x}-{y}: " + " ".join(f"{v:08x}" for v in w), file=out)
            if w[20] >> 16 == 0x4143:  # (an accumulator: MISC 496 = 'AC' | calls)
                m = lambda i: w[i - 476]  # noqa: E731
                srcs = " ".join(f"{k}:{m(503 + k) & 0xFFFF}/{m(503 + k) >> 16}" for k in range(8))
                print(
                    f"  acc call {m(496) & 0xFFFF}: rows {m(497)} of {m(498)}, local {m(499)} of {m(500)}, me "
                    f"{m(501) & 0xFF} home {m(501) >> 8 & 0xFF}, pass {m(502)} | per source taken / still wanted {srcs} "
                    f"| the most-wanted source's ring: source {m(476) & 0xFF} depth {m(476) >> 8 & 0xFF} base "
                    f"{m(476) >> 16}, head slot pieces landed (2 bits a piece) {m(477):04x}",
                    file=out,
                )
            for k in ("layr", "layo"):
                if k in cfg:
                    v = read_words_from_device(
                        f"{x}-{y}", cfg[k], device_id=dev.id, word_count=cfg["nap"], context=context
                    )
                    print(f"COMBINE_{k.upper()} dev {dev.id} {x}-{y}: " + " ".join(f"{u:08x}" for u in v), file=out)
    out.close()


if __name__ == "__main__":
    run_script()
