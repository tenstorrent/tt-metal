# Heads op at the model's batched placement (Q in DRAM unless HQ_L1=1, K / V in L1, preallocated outputs), one untraced
# call after warm-up, with the reader / writer instrumented: per-core wall-clock accumulators (DPRINT once per core per
# call) for waiting on compute (CB reserve / wait_front), issuing NoC commands and waiting on barriers; the slowest
# writers and a logical-core grid of writer issue + barrier time (NEGATIVE_RESULTS 73).
# Usage: bench_heads_dm_waits.py <B_in> <B_out>   (the op's QWEN_HEADS_* env applies)
import os
import re
import statistics
import sys
import tempfile

D_ = tempfile.mkdtemp(prefix="dmw_")
os.environ["TT_METAL_DPRINT_CORES"] = "all"
os.environ["TT_METAL_DPRINT_RISCVS"] = "BR,NC"
os.environ["TT_METAL_DPRINT_FILE"] = os.path.join(D_, "dprint.log")
import torch

import ttnn
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads_norm import op as hop
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_qkv_heads_norm.constants import make_norm_constants
from models.tt_transformers.tt.common import get_rot_transformation_mat

PRE = r"""
#include "risc_common.h"
#include "api/debug/dprint.h"
namespace { uint32_t dm_acc[4]; uint32_t dm_last; }
#define DM(i) do { const uint32_t n_ = get_timestamp_32b(); dm_acc[i] += n_ - dm_last; dm_last = n_; } while (0)
"""


def instr(path, tag, marks, start_anchor):
    s = open(path).read()
    s = s.replace('#include "api/dataflow/circular_buffer.h"\n', '#include "api/dataflow/circular_buffer.h"\n' + PRE, 1)
    for anchor, i in marks:
        assert s.count(anchor) == 1, (path, anchor)
        s = s.replace(anchor, anchor + f"DM({i});\n")
    assert s.count(start_anchor) == 1
    s = s.replace(
        start_anchor,
        "    for (int i_ = 0; i_ < 4; ++i_) {\n        dm_acc[i_] = 0;\n    }\n    dm_last = get_timestamp_32b();\n"
        + start_anchor,
    )
    assert s.endswith("\n}\n")
    s = (
        s[: -len("}\n")]
        + '    DPRINT("'
        + tag
        + ' {} {} {} {} {} S{}\\n", num_work_units, dm_acc[0], dm_acc[1], dm_acc[2], dm_acc[3], work_unit_start);\n}\n'
    )
    out = os.path.join(D_, os.path.basename(path))
    open(out, "w").write(s)
    return out


# reader: 0 = wait for CB space (compute), 1 = issue reads, 2 = barrier (+ rot), 3 = push / rest
R = instr(
    hop.READER_KERNEL,
    "RD",
    [
        ("        cb.reserve_back(unit_tiles);\n", 0),
        (
            "                l1_write_offset += tile_size_bytes;\n            }\n        }\n        bool load_rot = false;\n",
            1,
        ),
        ("        cb.push_back(unit_tiles);\n", 2),
    ],
    "    uint32_t last_s_tile = 0xFFFFFFFFu;\n",
)
# writer: 0 = wait for compute output, 1 = issue writes, 2 = barrier, 3 = pop / rest
W = instr(
    hop.WRITER_KERNEL,
    "WR",
    [
        ("            cb.wait_front(unit_tiles);\n        }\n", 0),
        (
            "        if (flush_only) {\n            noc.async_writes_flushed();\n        } else {\n            noc.async_write_barrier();\n        }\n",
            2,
        ),
    ],
    "    for (uint32_t w = 0; w < num_work_units; ++w) {\n",
)
hop.READER_KERNEL, hop.WRITER_KERNEL = R, W

B_IN, B_OUT = int(sys.argv[1]), int(sys.argv[2])
NH, NKV, DH, EPS, S = 32, 8, 128, 1e-6, 512
D = ttnn.open_device(device_id=0, l1_small_size=32768)
L1, DR, B8 = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat8_b
try:
    GQ, GK, SC, EP = make_norm_constants(torch.rand(DH) + 0.5, torch.rand(DH) + 0.5, EPS, D)
    ang = torch.rand(1, 1, S, DH) * 6.28
    tl = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1)
    cos, sin, T = tl(torch.cos(ang)), tl(torch.sin(ang)), tl(get_rot_transformation_mat(32))
    x = ttnn.from_torch(
        torch.randn(B_IN, 1, S, (NH + 2 * NKV) * DH), dtype=B8, layout=ttnn.TILE_LAYOUT, device=D, memory_config=L1
    )
    qmc = L1 if os.getenv("HQ_L1", "0") == "1" else DR
    alloc = lambda shp, mc: ttnn.allocate_tensor_on_device(ttnn.Shape(shp), B8, ttnn.TILE_LAYOUT, D, mc)
    outs = (alloc([B_OUT, NH, S, DH], qmc), alloc([B_OUT, NKV, S, DH], L1), alloc([B_OUT, NKV, S, DH], L1))
    for _ in range(3):
        hop.nlp_create_qkv_heads_norm_headsplit(
            x,
            GQ,
            GK,
            SC,
            EP,
            num_heads=NH,
            num_kv_heads=NKV,
            memory_config=qmc,
            kv_memory_config=L1,
            rot_cos=cos,
            rot_sin=sin,
            trans_mat=T,
            q_dtype=B8,
            kv_dtype=B8,
            norm_eps=EPS,
            use_v3=True,
            out_tensors=outs,
            batch_offset=0,
        )
        ttnn.synchronize_device(D)
finally:
    ttnn.close_device(D)
rows = {"RD": [], "WR": []}
for line in open(os.environ["TT_METAL_DPRINT_FILE"], errors="ignore"):
    m = re.search(r"(RD|WR) (\d+) (\d+) (\d+) (\d+) (\d+) S(\d+)", line)
    if m:
        rows[m.group(1)].append([int(v) for v in m.groups()[1:]] + [line[: m.start()].strip()[-36:]])
names = {
    "RD": ["wait CB space", "issue reads", "barrier+rot", "push/rest"],
    "WR": ["wait output", "(unused)", "issue+barrier", "pop/rest"],
}
for tag, r in rows.items():
    r = r[len(r) * 2 // 3 :]  # the last call's cores
    if not r:
        continue
    units = statistics.mean(v[0] for v in r)
    tot = [sum(v[1:5]) for v in r]
    print(
        f"RES {tag} B{B_IN}->{B_OUT} cores {len(r)} units {units:.1f} total cycles mean {statistics.mean(tot):.0f} max {max(tot)}"
    )
    for i, n in enumerate(names[tag]):
        print(f"RES   {n:14s} {statistics.mean(v[1 + i] for v in r) / units:8.0f} cycles/unit")
    if tag == "WR":
        for v in sorted(r, key=lambda v: -sum(v[1:5]))[:10] + sorted(r, key=lambda v: sum(v[1:5]))[:3]:
            print(f"RES   core total {sum(v[1:5])} wait {v[1]} issue+barrier {v[3]} start {v[5]} {v[6]}")
# grid of writer issue+barrier (k cycles) and its first unit, by core x-y from the DPRINT prefix ("<dev>:<x>-<y>:BR")
grid, start = {}, {}
for v in rows["WR"]:  # the last call wins per core
    m = re.search(r"(\d+)-(\d+):BR", v[6])
    if m:
        grid[(int(m.group(1)), int(m.group(2)))] = v[3] // 1000
        start[(int(m.group(1)), int(m.group(2)))] = v[5]
xs = sorted({k[0] for k in grid})
ys = sorted({k[1] for k in grid})
print("RES grid issue+barrier kcycles, rows y, cols x " + " ".join(f"{x:4d}" for x in xs))
for y in ys:
    print(f"RES grid y={y:2d} " + " ".join(f"{grid.get((x, y), -1):4d}" for x in xs))
print("RES grid first unit // 128 (batch), rows y")
for y in ys:
    print(f"RES grid y={y:2d} " + " ".join(f"{start.get((x, y), -1) // 128:4d}" for x in xs))
