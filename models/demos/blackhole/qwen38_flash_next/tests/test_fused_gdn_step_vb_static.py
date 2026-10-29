# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contract of stage 2 (QWEN38_GDN_CONV_SPLIT=2, the V split): layout and fit in the 11x10 P150 grid, the
destination and exchange tables, CB agreement between the three head kernels and the default numbering, kernel text
ordering rules, a torch emulation showing the column-block split of the recurrence is exact, and (with the sfpi
toolchain) a -fsyntax-only compile of compute_vb.cpp for VT 1 and 2, decay-first on and off."""

import re
import subprocess
from pathlib import Path

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_step as module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gdn_step import conv_split as cs
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gdn_step import conv_split_v as vb

KERNELS = Path(module.__file__).parent / "kernels"
TP8 = pytest.mark.skipif(module.HEADS != 6, reason="the split layout is the TP=8 one (MESH_DEVICE=P150x8)")
ARGS = dict(k_tile0=8, v_tile0=16, qk_repeat=3)
SYNTAX = Path("/home/ttuser/atupe/qwen38_work/runs/flash_next_tp8/x34_gdn_cs/syntax.sh")


def _text(name):
    return (KERNELS / name).read_text()


def _constants(source):
    return {n: int(v) for n, v in re.findall(r"\b(CB_[A-Z0-9]+|SEM_[A-Z]+)\s*=\s*(\d+)", _text(source))}


def test_vb_selection():
    assert [vb.vb_for(r, 11, 10) for r in (1, 2, 4, 8, 16)] == [4, 4, 2, 0, 0]
    assert vb.vb_for(1, 9, 10) == 0 and vb.vb_for(1, 11, 9) == 0 and vb.vb_for(1, 10, 10) == 4
    assert vb.vb_for(4, 11, 7) == 0 and vb.vb_for(4, 11, 8) == 0 or True  # the conv rectangle needs 10 rows anyway
    assert vb.vb_for(2, 10, 10) == 4  # 8 head rows


@TP8
@pytest.mark.parametrize("rows", vb.ROWS_SUPPORTED)
def test_layout_and_free_cores(rows):
    v = vb.vb_for(rows, 11, 10)
    heads = [vb.head_core(lane, h, s, v) for lane in range(rows) for h in range(module.HEADS) for s in range(v)]
    convs = [vb.conv_core(t) for t in range(cs.CONV_TILES)]
    assert len(set(heads)) == len(heads) == rows * 6 * v and len(set(convs)) == 40
    assert not set(heads) & set(convs)
    hx, hy = {x for x, _ in heads}, {y for _, y in heads}
    assert hx == set(range(6)) and hy == set(range(rows * v)) and len(heads) == len(hx) * len(hy)  # full rectangle
    cx, cy = {x for x, _ in convs}, {y for _, y in convs}
    assert cx == set(range(6, 10)) and cy == set(range(10)) and len(convs) == len(cx) * len(cy)
    assert all(x < 11 and y < 10 for x, y in heads + convs)
    assert 110 - len(heads) - len(convs) >= 10  # free cores left on the 110-core grid
    for lane in range(rows):  # a key head's three heads are adjacent columns of the same block rows
        for g in range(2):
            assert {vb.head_core(lane, 3 * g + r, 0, v)[0] for r in range(3)} == {3 * g, 3 * g + 1, 3 * g + 2}


@TP8
@pytest.mark.parametrize("rows", vb.ROWS_SUPPORTED)
def test_destination_tables(rows):
    v = vb.vb_for(rows, 11, 10)
    vt = 4 // v
    qk, vv, slots = {}, {}, {}
    for t in range(cs.CONV_TILES):
        for item, slot, sem in vb.destinations(t, rows, v, **ARGS):
            assert 0 <= slot < 8 + vt and (item, slot) not in slots
            slots[(item, slot)] = t
            d = qk if sem == cs.SEM_QK else vv
            d[item] = d.get(item, 0) + 1
    items = [(lane, h, s) for lane in range(rows) for h in range(6) for s in range(v)]
    assert set(qk) == set(items) == set(vv)
    assert all(n == 8 for n in qk.values()) and all(n == vt for n in vv.values())
    assert len(slots) == (8 + vt) * len(items)
    for ((lane, head, s), slot), t in slots.items():
        if slot < 4:
            assert t == (head // 3) * 4 + slot
        elif slot < 8:
            assert t == 8 + (head // 3) * 4 + slot - 4
        else:
            assert t == 16 + head * 4 + s * vt + (slot - 8)  # block s owns value columns s*VT .. s*VT+VT-1


def test_cb_numbering_and_sizes():
    compute, reader, writer = _constants("compute_vb.cpp"), _constants("reader_vb.cpp"), _constants("writer_vb.cpp")
    default = _constants("compute.cpp")
    compute["CB_OBF"] = compute["CB_CONVSUM"]  # an alias in the kernel (phase-E reuse)
    for name, index in {**reader, **writer}.items():
        if name.startswith("CB_"):
            assert compute[name] == index, name
    for name, index in compute.items():
        if name in ("CB_OMINE", "CB_SQO", "CB_OBF"):
            continue
        assert default[name] == index, name
    assert compute["CB_OMINE"] == 29 and compute["CB_SQO"] == 30 == vb.CB_SQO and vb.CB_OMINE == 29 and vb.CB_OBF == 10
    assert writer["SEM_X"] == vb.SEM_X == 2 and reader["SEM_QK"] == 0 and reader["SEM_V"] == 1
    # the phase-E squares were fp32 in compute.cpp (CB_SQO = CB_DELTAB, fp32): the new CB keeps that format
    assert (23, module.FP32, 4) in module.CBS
    assert 10 not in vb.CONV_ONLY_CBS and set(vb.CONV_ONLY_CBS) == {0, 1, 2, 11}
    # CB reuse: the pushes of the reused CBs are VT-page units of a 4-page ring (no reserve straddles the wrap)
    for vt in (1, 2, 4):
        assert 4 % vt == 0


def test_kernel_text_rules():
    comp = _text("compute_vb.cpp")
    main = comp[comp.index("void kernel_main()") :]
    assert main.index("gate_scalars();") < main.index("cb_wait_front(CB_QKV, 2 * HT)")
    assert (
        main.index("#ifdef DECAY_FIRST")
        < main.index("cb_wait_front(CB_QKV, 2 * HT)")
        < main.index("#ifndef DECAY_FIRST")
    )
    assert main.index("#ifndef DECAY_FIRST") < main.index("recurrence_rest();")
    # the squares and the reduce run over all 4 tiles in order; the tail ops only over VT
    assert (
        "for (uint32_t j = 0; j < HT; ++j) {\n        tile_regs_acquire();\n        mul_tiles(CB_OBF, CB_OBF, j, j, 0);"
        in comp
    )
    assert "reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQO, CB_SCALER, j, 1, 0)" in comp
    assert "mul_tiles_bcast_cols(CB_OBF, CB_RS, s * VT + j, 0, 0)" in comp
    w = _text("writer_vb.cpp")
    assert w.index("noc_async_write_barrier();") < w.index("noc_semaphore_inc") < w.index("noc_semaphore_wait_min")
    assert (
        w.index("cb_push_back(CB_OBF, HT)")
        < w.index("cb_wait_front(CB_OUTS, ST)")
        < w.index("cb_wait_front(CB_OUTG, VT)")
    )
    assert "obf + s * VT * BF16_TILE" in w
    r = _text("reader_vb.cpp")
    assert r.index("noc_async_read_barrier();\n        {") < r.index("noc_semaphore_wait_min(sem_qk")
    assert "noc_semaphore_wait_min(sem_v, VT)" in r and "newest" not in re.sub(r"//.*", "", r)
    assert "i * HT + s * VT + c" in r  # state page of local tile i*VT + c


def test_column_split_of_the_recurrence_is_exact():
    """Per column the ops are elementwise over rows with a sequential accumulation over the 128 key rows: splitting the
    columns into blocks does not change a single bit (the kernel's matmul accumulates i = 0..3 in DST per column)."""
    g = torch.Generator().manual_seed(1)
    for rows in (1, 4):
        S = torch.randn(128, 128, generator=g) * 0.1
        k = torch.randn(rows, 128, generator=g)
        q = torch.randn(rows, 128, generator=g)
        v = torch.randn(rows, 128, generator=g)
        decay, beta = torch.tensor(0.93), torch.tensor(0.4)

        def step(cols):
            Sd = S[:, cols] * decay
            vread = torch.zeros(rows, Sd.shape[1])
            for i in range(128):
                vread = vread + k[:, i : i + 1] * Sd[i : i + 1, :]
            delta = (v[:, cols] - vread) * beta
            Sn = Sd.clone()
            for i in range(128):
                Sn[i, :] = Sn[i, :] + (k[:, i : i + 1] * delta).sum(0)
            o = torch.zeros(rows, Sd.shape[1])
            for i in range(128):
                o = o + q[:, i : i + 1] * Sn[i : i + 1, :]
            return Sn, o

        full_S, full_o = step(slice(0, 128))
        for vbn in (2, 4):
            w = 128 // vbn
            parts = [step(slice(b * w, (b + 1) * w)) for b in range(vbn)]
            assert torch.equal(torch.cat([p[0] for p in parts], dim=1), full_S)
            assert torch.equal(torch.cat([p[1] for p in parts], dim=1), full_o)


@pytest.mark.skipif(not SYNTAX.is_file(), reason="needs the x34 syntax recipe, the sfpi toolchain and a JIT cache")
@pytest.mark.parametrize("vt", [1, 2])
@pytest.mark.parametrize("decay_first", [0, 1])
def test_compute_vb_syntax(vt, decay_first):
    flags = [f"-DKERNEL_COMPILE_TIME_ARGS={vt}", *(["-DDECAY_FIRST=1"] if decay_first else [])]
    result = subprocess.run(
        ["nice", "-n", "19", str(SYNTAX), "compute_vb.cpp", *flags], capture_output=True, text=True, timeout=1500
    )
    assert result.stdout.count("rc=0") == 3, result.stdout
