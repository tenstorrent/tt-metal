# bs16 fused heads op (v1 compute): where the compute time goes, phase by phase.
# A scratch copy of the compute kernel adds, on each TRISC (unpack / math / pack), the wall-clock time since the
# previous phase marker to that phase's total, and DPRINTs the totals once per core. Phases are serialized through
# CBs, so the pack thread's split is each phase's latency (unpack + math + pack). One untraced call, input in L1 (the
# fused design's case: the read is gone) or DRAM; cos / sin in L1 as in the model. PH_MODE=unit: per-unit split only
# (wait for input / output space / cos-sin, compute), which works for v1 and v3 (QWEN_FUSED_COMPUTE_V3=1).
# Usage: bench_heads_bs16_phases.py [batch] [L1|DRAM]
import os
import re
import sys
import tempfile

B = int(sys.argv[1]) if len(sys.argv) > 1 else 16
PLACE = sys.argv[2] if len(sys.argv) > 2 else "L1"
SCRATCH = os.environ.get("ABL_DIR") or tempfile.mkdtemp(prefix="heads_ph_")
DPRINT_FILE = os.path.join(SCRATCH, "dprint.log")
# DPRINT must be configured before the device opens
os.environ.setdefault("TT_METAL_DPRINT_CORES", "all")
os.environ.setdefault("TT_METAL_DPRINT_RISCVS", "TR0,TR1,TR2")
os.environ["TT_METAL_DPRINT_FILE"] = DPRINT_FILE

import torch  # noqa: E402

import ttnn  # noqa: E402
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads_norm import op as hop  # noqa: E402
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads_norm.constants import (  # noqa: E402
    make_norm_constants,
)
from models.tt_transformers.tt.common import get_rot_transformation_mat  # noqa: E402

NH, NKV, DH, EPS, S = 32, 8, 128, 1e-6, 512
MODE = os.getenv("PH_MODE", "phase")
V3 = os.getenv("QWEN_FUSED_COMPUTE_V3", "0") == "1"
PHASES = ["wait in", "wait out", "wait qout", "wait cos/sin", "compute+push"] if MODE == "unit" else ["x2", "reduce", "eps+rsqrt", "bcast inv", "gamma", "rot matmul", "sin mul", "cos mul", "add",
          "wait in", "V copy", "unit tail"]  # fmt: skip

PRELUDE = r"""
#include "risc_common.h"
#include "api/debug/dprint.h"
namespace {
uint32_t ph_acc[12];
uint32_t ph_last;
}
#define PH(i)                                         \
    do {                                              \
        const uint32_t ph_now = get_timestamp_32b();  \
        ph_acc[i] += ph_now - ph_last;                \
        ph_last = ph_now;                             \
    } while (0)
"""


def epilogue(n):
    args = ", ".join(f"ph_acc[{i}]" for i in range(n))
    fmt = " ".join(["{}"] * (n + 1))
    return (
        "".join(f'    DPRINT_{t}("PH{t[0]} {fmt}\\n", num_work_units, {args});\n' for t in ("UNPACK", "MATH", "PACK"))
        + "}\n"
    )


UNIT_MARKS = [
    ("        in.wait_front(unit_tiles);\n", 0),
    ("        out.reserve_back(out_tiles);\n", 1),
    ("            qout.reserve_back(sub_q_tiles);\n        }\n", 2),
    ("            csin.wait_front(Wt);\n        }\n", 3),
    ("        in.pop_front(unit_tiles);\n", 4),
]


def instrument(src):
    marks = (
        UNIT_MARKS
        if MODE == "unit"
        else [  # (anchor, marker index): PH(i) goes right after the anchor
            ("    x2.push_back(Wt);\n", 0),
            ("    reduce_uninit(cb_x2);\n", 1),
            ("    red.pop_front(1);\n", 2),
            ("    inv.pop_front(1);\n", 3),
            ("        nrm.push_back(Wt);\n    }\n", 4),
            ("    rot.push_back(Wt);\n", 5),
            ("    rot.pop_front(Wt);\n", 6),
            ("    nrm.pop_front(Wt);\n", 7),
            ("    si.pop_front(Wt);\n", 8),
            ("            csin.wait_front(Wt);\n        }\n", 9),
            ("                tile_regs_release();\n            }\n        }\n", 10),
            ("        in.pop_front(unit_tiles);\n", 11),
        ]
    )
    n = len(marks)
    for anchor, i in marks:
        assert src.count(anchor) == 1, (anchor, src.count(anchor))
        indent = re.match(r" *", anchor.splitlines()[-1]).group(0)
        src = src.replace(anchor, anchor + f"{indent}PH({i});\n")
    start = "    uint32_t sub = work_unit_start % q_split;"
    assert src.count(start) == 1
    src = src.replace(
        start, f"    for (uint32_t i = 0; i < {n}; ++i) ph_acc[i] = 0;\n    ph_last = get_timestamp_32b();\n" + start
    )
    assert src.endswith("    }\n}\n"), repr(src[-120:])
    src = src[: -len("}\n")] + epilogue(n)
    return src.replace(
        '#include "api/dataflow/circular_buffer.h"\n', '#include "api/dataflow/circular_buffer.h"\n' + PRELUDE, 1
    )


def main():
    assert MODE == "unit" or not V3, "per-phase anchors are v1's; use PH_MODE=unit for v3"
    cattr = "COMPUTE_KERNEL_BS1" if V3 else "COMPUTE_KERNEL"
    kern = os.path.join(SCRATCH, "compute_phases.cpp")
    open(kern, "w").write(instrument(open(getattr(hop, cattr)).read()))
    orig = getattr(hop, cattr)
    setattr(hop, cattr, kern)
    D = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        GQ, GK, SC, EP = make_norm_constants(torch.rand(DH) + 0.5, torch.rand(DH) + 0.5, EPS, D)
        ang = torch.rand(1, 1, S, DH) * 6.28
        DR = ttnn.DRAM_MEMORY_CONFIG
        L1 = ttnn.L1_MEMORY_CONFIG
        RM = DR if os.getenv("PH_ROPE", "L1") == "DRAM" else L1  # cos / sin placement (model: L1)
        mk = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=D, memory_config=RM)
        cos, sin, T = mk(torch.cos(ang)), mk(torch.sin(ang)), mk(get_rot_transformation_mat(32))
        x = ttnn.from_torch(
            torch.randn(B, 1, S, (NH + 2 * NKV) * DH),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=D,
            memory_config=L1 if PLACE == "L1" else DR,
        )
        outs = hop.nlp_create_qkv_heads_norm_headsplit(
            x, GQ, GK, SC, EP, num_heads=NH, num_kv_heads=NKV, memory_config=DR, rot_cos=cos, rot_sin=sin,
            trans_mat=T, q_dtype=ttnn.bfloat8_b, kv_dtype=ttnn.bfloat8_b, norm_eps=EPS,
        )  # fmt: skip
        ttnn.synchronize_device(D)
        [ttnn.deallocate(t) for t in outs]
    finally:
        setattr(hop, cattr, orig)
        ttnn.close_device(D)
    report()


def report():
    rows = {"PHU": [], "PHM": [], "PHP": []}
    for line in open(DPRINT_FILE, errors="ignore"):
        m = re.search(r"(PH[UMP]) ((?:\d+ ?){%d})" % (len(PHASES) + 1), line)
        if m:
            rows[m.group(1)].append([int(v) for v in m.group(2).split()])
    print(
        f"RES B{B} {'v3' if V3 else 'v1'} {MODE} input {PLACE}: cores reporting U/M/P = {[len(v) for v in rows.values()]}"
    )
    for tag, name in (("PHP", "pack"), ("PHM", "math"), ("PHU", "unpack")):
        r = rows[tag]
        if not r:
            continue
        units = sum(v[0] for v in r) / len(r)
        # the op ends with its slowest core: report the core with the largest total
        slow = max(r, key=lambda v: sum(v[1:]))
        tot_mean = sum(sum(v[1:]) for v in r) / len(r)
        print(f"RES {name}: {units:.1f} units/core, total ticks mean {tot_mean:.0f}, slowest core {sum(slow[1:])}")
        for i, p in enumerate(PHASES):
            mean = sum(v[1 + i] for v in r) / len(r)
            print(f"RES   {p:11s} {100 * mean / tot_mean:5.1f}%   {mean / max(units, 1):8.0f} ticks/unit")


if __name__ == "__main__":
    main()
