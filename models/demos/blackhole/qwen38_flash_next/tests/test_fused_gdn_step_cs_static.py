# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contract of the conv-split GDN step (QWEN38_GDN_CONV_SPLIT=1, stage 1): the layout and the destination
tables, the CB index agreement between the new kernels, the default kernels left untouched, the kernel text ordering
rules and (when the sfpi toolchain and a JIT cache are present) a -fsyntax-only compile of the two compute kernels
for the three TRISCs."""

import re
import subprocess
from pathlib import Path

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_step as module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gdn_step import conv_split as cs

KERNELS = Path(module.__file__).parent / "kernels"
REPO = Path(module.__file__).resolve().parents[7]
NEW = ("conv_reader.cpp", "conv_compute.cpp", "conv_writer.cpp", "reader_cs.cpp", "compute_cs.cpp")
TP8 = pytest.mark.skipif(module.HEADS != 6, reason="the split layout is the TP=8 one (MESH_DEVICE=P150x8)")
ARGS = dict(k_tile0=module.K_TILE0, v_tile0=module.V_TILE0, qk_repeat=module.QK_REPEAT)


def _text(name):
    return (KERNELS / name).read_text()


def _constants(source):
    return {n: int(v) for n, v in re.findall(r"\b(CB_[A-Z0-9]+|SEM_[A-Z]+)\s*=\s*(\d+)", _text(source))}


@TP8
def test_geometry_assumptions():
    assert (module.HEADS, module.QK_HEADS, module.QK_REPEAT) == (6, 2, 3)
    assert module.K_TILE0 == 8 and module.V_TILE0 == 16
    assert cs.CONV_TILES == 8 + 8 + 4 * module.HEADS


@TP8
@pytest.mark.parametrize("rows", cs.ROWS_SUPPORTED)
def test_layout_rectangles(rows):
    items = [(lane, head) for lane in range(rows) for head in range(module.HEADS)]
    heads = [cs.head_core(*i) for i in items]
    convs = [cs.conv_core(t) for t in range(cs.CONV_TILES)]
    assert len(set(heads)) == len(heads) and len(set(convs)) == cs.CONV_TILES
    assert not set(heads) & set(convs)
    hx, hy = {x for x, _ in heads}, {y for _, y in heads}
    assert hx == set(range(cs.HEAD_COLUMNS)) and hy == set(range(2 * rows))  # the rectangle (0,0)-(2,2*rows-1) is full
    assert len(heads) == len(hx) * len(hy)
    cx, cy = {x for x, _ in convs}, {y for _, y in convs}
    assert cx == set(range(cs.CONV_X0, cs.CONV_X0 + cs.CONV_COLUMNS)) and cy == set(range(cs.CONV_ROWS))
    assert len(convs) == len(cx) * len(cy)
    # a key head's three heads of one lane share a row of the head rectangle
    for lane in range(rows):
        for g in range(module.QK_HEADS):
            assert len({cs.head_core(lane, 3 * g + r)[1] for r in range(3)}) == 1
    assert max(x for x, _ in heads + convs) < 11 and max(y for _, y in heads + convs) < 10  # the P150 11x10 grid


@TP8
@pytest.mark.parametrize("rows", cs.ROWS_SUPPORTED)
def test_destination_tables(rows):
    qk, v, slots = {}, {}, {}
    for t in range(cs.CONV_TILES):
        for item, slot, sem in cs.destinations(t, rows, **ARGS):
            assert sem in (cs.SEM_QK, cs.SEM_V) and 0 <= slot < 12
            (qk if sem == cs.SEM_QK else v)[item] = (qk if sem == cs.SEM_QK else v).get(item, 0) + 1
            assert (item, slot) not in slots  # every slot of every item written once
            slots[(item, slot)] = t
    items = [(lane, head) for lane in range(rows) for head in range(module.HEADS)]
    assert set(qk) == set(items) and set(v) == set(items)
    assert all(n == cs.QK_INCREMENTS for n in qk.values()) and all(n == cs.V_INCREMENTS for n in v.values())
    assert len(slots) == 12 * len(items)
    # slot contents: q c, k 4+c of the item's key head, v 8+c of the item's own head
    for (lane_head, slot), t in slots.items():
        head = lane_head[1]
        if slot < 4:
            assert t == (head // 3) * 4 + slot
        elif slot < 8:
            assert t == module.K_TILE0 + (head // 3) * 4 + slot - 4
        else:
            assert t == module.V_TILE0 + head * 4 + slot - 8


def test_selection_thresholds():
    assert [cs.grid_fits(11, 10, r) for r in (1, 2, 4, 8, 16, 32)] == [True, True, True, False, False, False]
    assert not cs.grid_fits(7, 10, 1) and not cs.grid_fits(11, 9, 1) and cs.grid_fits(8, 10, 1)
    assert not cs.grid_fits(8, 10, 8)
    assert not cs.grid_fits(11, 7, 4) and cs.grid_fits(11, 10, 4)


def test_cb_indices_agree():
    compute, reader = _constants("compute_cs.cpp"), _constants("reader_cs.cpp")
    for name, index in reader.items():
        if name.startswith("CB_"):
            assert compute[name] == index, name
    default = _constants("compute.cpp")
    for name, index in compute.items():
        assert default[name] == index, name  # the head kernel keeps the default numbering (FP32_COPY_CBS stay valid)
    conv_c, conv_r, conv_w = (
        _constants("conv_compute.cpp"),
        _constants("conv_reader.cpp"),
        _constants("conv_writer.cpp"),
    )
    for table in (conv_r, conv_w):
        for name, index in table.items():
            assert conv_c[name] == index, name
    assert conv_c["CB_QKV"] == reader["CB_QKV"] == conv_w["CB_QKV"] == cs.CB_QKV == 11
    assert reader["SEM_QK"] == cs.SEM_QK and reader["SEM_V"] == cs.SEM_V
    assert cs.HEAD_ONLY_CBS_DROPPED == (conv_c["CB_P"], conv_c["CB_S"], conv_c["CB_T"])
    # the head compute uses neither CB_P / CB_S / CB_T nor CB_CONVSUM except as CB_OBF (4 pages on the head)
    body = re.sub(r"//.*", "", _text("compute_cs.cpp"))
    for name in ("CB_P", "CB_S", "CB_T"):
        assert not re.search(rf"\b{name}\b[,)]", body.split("kernel_main")[1]), name
    assert (10, module.BF16, 4) in module.CBS and (11, module.BF16, 12) in module.CBS


def test_default_kernels_untouched():
    for name in ("reader.cpp", "compute.cpp", "writer.cpp"):
        path = (KERNELS / name).relative_to(REPO)
        assert subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(path)], cwd=REPO).returncode == 0, name
    init = Path(module.__file__).read_text()
    assert init.count("QWEN38_GDN_CONV_SPLIT") == 1 and "conv_split.admits(rows, mesh)" in init
    # selection only on =="1" (flag unset/0 = today's program) and never with debug taps
    assert re.search(r'QWEN38_GDN_CONV_SPLIT.*in \("1", "2"\) and debug is None', init) and cs.enabled.__doc__ is None


def test_kernel_sources_exist_and_have_arg_counts():
    for name in NEW:
        assert (KERNELS / name).is_file()
    conv_reader = _text("conv_reader.cpp")
    assert conv_reader.count("TensorAccessorArgs<") == 9 and conv_reader.count("get_arg_val<uint32_t>(arg++)") == 10
    head_reader = _text("reader_cs.cpp")
    assert head_reader.count("TensorAccessorArgs<") == 4 and head_reader.count("get_arg_val<uint32_t>(arg++)") == 5 + 2
    assert "newest" not in re.sub(r"//.*", "", head_reader) and "noc_async_write" not in head_reader


def test_kernel_text_ordering_rules():
    reader = _text("reader_cs.cpp")
    # the semaphore waits come after the state reads' barrier and after the mask push
    assert reader.index("noc_async_read_barrier();\n        {") < reader.index("noc_semaphore_wait_min(sem_qk")
    assert reader.index("cb_push_back(CB_MASK, 1)") < reader.index("noc_semaphore_wait_min(sem_qk")
    assert reader.index("noc_semaphore_wait_min(sem_qk") < reader.index("noc_semaphore_wait_min(sem_v")
    assert reader.count("noc_semaphore_set") == 2 and reader.index("cb_reserve_back(CB_QKV") < reader.index(
        "read_tiles(CB_Z"
    )
    writer = _text("conv_writer.cpp")
    assert (
        writer.index("noc_async_write_barrier()")
        < writer.index("noc_semaphore_inc")
        < writer.index("noc_async_atomic_barrier")
    )
    assert writer.index("noc_async_write(") < writer.index("noc_async_write_barrier()")
    compute = _text("compute_cs.cpp")
    loop = compute[compute.index("for (uint32_t item") :]
    assert (
        loop.index("gate_scalars();") < loop.index("cb_wait_front(CB_QKV, 2 * HT)") < loop.index("l2_norm(0, CB_QROW)")
    )
    assert "conv_silu" not in compute and "CB_P, " not in loop


def test_conv_compute_is_the_default_conv_body():
    """conv_silu of compute.cpp with t = 0: the op sequence (not the loop and wait scaffolding) must be identical."""

    def ops(text):
        body = text[text.index("tile_regs_acquire();") :]
        body = body[: body.index("cb_pop_front(CB_CONVSUM, 1);")]
        return [
            re.sub(r"\s+", " ", line.strip())
            for line in body.splitlines()
            if line.strip() and not line.strip().startswith("//")
        ]

    default = ops(_text("compute.cpp")[_text("compute.cpp").index("ALWI void conv_silu()") :])
    default = [l for l in default if not l.startswith(("#ifdef", "#endif", "if (t == 0)", "tap(", "}"))]
    conv = [l for l in ops(_text("conv_compute.cpp")) if l != "}"]
    assert default == conv


@TP8
def test_per_tile_conv_equals_whole_row():
    torch.manual_seed(0)
    rows = 4
    window = [torch.randn(rows, module.QKV_WIDTH).bfloat16() for _ in range(4)]
    taps = [torch.randn(module.QKV_WIDTH).bfloat16() for _ in range(4)]
    whole = torch.nn.functional.silu(sum(w.float() * t.float() for w, t in zip(window, taps)).bfloat16())
    pieces = []
    for tile in range(cs.CONV_TILES):
        sl = slice(32 * tile, 32 * tile + 32)
        pieces.append(
            torch.nn.functional.silu(sum(w[:, sl].float() * t[sl].float() for w, t in zip(window, taps)).bfloat16())
        )
    assert torch.equal(torch.cat(pieces, dim=1), whole)


SYNTAX = Path("/home/ttuser/atupe/qwen38_work/runs/flash_next_tp8/x34_gdn_cs/syntax.sh")


@pytest.mark.skipif(not SYNTAX.is_file(), reason="needs the x34 syntax recipe, the sfpi toolchain and a JIT cache")
@pytest.mark.parametrize("kernel", ["compute_cs.cpp", "conv_compute.cpp"])
def test_compute_kernels_syntax(kernel):
    result = subprocess.run(["nice", "-n", "19", str(SYNTAX), kernel], capture_output=True, text=True, timeout=1500)
    assert result.stdout.count("rc=0") == 3, result.stdout
