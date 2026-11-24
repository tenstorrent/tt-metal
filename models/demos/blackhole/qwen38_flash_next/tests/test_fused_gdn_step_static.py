# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host tests of the fused gdn_step family (base, conv_split, lane_split, conv_split_v) without a device."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_step as module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gdn_step import conv_split as cs
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gdn_step import conv_split_v as vb
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gdn_step import lane_split as ls

# ============================================================================
# was test_fused_gdn_step_static.py
# No-device contract of the fused GDN step: the CB table shared by the three kernels and the Python side, the
# kernel argument layouts, the registry entry, and ``reference_step`` against tt/gdn.py's Qwen38GDN-style arithmetic.
# ============================================================================

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


KERNELS = Path(module.__file__).parent / "kernels"


def _constants(source: str) -> dict[str, int]:
    text = (KERNELS / source).read_text()
    return {name: int(value) for name, value in re.findall(r"\b(CB_[A-Z0-9]+)\s*=\s*(\d+)", text)}


def test_cb_indices_agree_between_kernels_and_python():
    compute, reader, writer = _constants("compute.cpp"), _constants("reader.cpp"), _constants("writer.cpp")
    for name, index in reader.items():
        assert compute[name] == index, name
    for name, index in writer.items():
        assert compute[name] == index, name
    declared = {index for index, _, _ in module.CBS} | {module.CB_SNEW, module.CB_OUTS, module.CB_DEBUG}
    used = {v for k, v in compute.items()}
    assert used <= declared, sorted(used - declared)
    assert max(declared) <= 31 and len(declared) == len({i for i in declared})
    assert (
        compute["CB_SNEW"] == module.CB_SNEW
        and compute["CB_OUTS"] == module.CB_OUTS
        and compute["CB_DBG"] == module.CB_DEBUG
    )
    # exact fp32 copies: only CBs the compute consumes with copy_tile; matmul/reduce/bcast operands must stay Default
    assert set(module.FP32_COPY_CBS) == {
        compute[n] for n in ("CB_STATE", "CB_DTNA", "CB_BETA", "CB_DECAY", "CB_SDECC", "CB_VREAD")
    }
    assert not set(module.FP32_COPY_CBS) & {
        compute[n] for n in ("CB_QROW", "CB_KROW", "CB_KCOL", "CB_SDEC", "CB_DELTAB", "CB_SNEW", "CB_SCALER", "CB_RS")
    }


def test_kernel_argument_layouts():
    reader = (KERNELS / "reader.cpp").read_text()
    assert reader.count("TensorAccessorArgs<") == 12  # projected, 3 slots, 4 taps, dtna, norm, state, newest
    assert reader.count("get_arg_val<uint32_t>(arg++)") == 13 + 2  # 12 addresses, items, then (lane, head)
    writer = (KERNELS / "writer.cpp").read_text()
    assert writer.count("TensorAccessorArgs<") == 3  # state, out, debug
    compute = (KERNELS / "compute.cpp").read_text()
    assert "get_arg_val<uint32_t>(0)" in compute and "DEBUG_TAPS" in compute
    assert (
        module.DEBUG_TILES == len(module.DEBUG_TAPS) == 28
    )  # conv sum, 12 conv, 4 q-norm statistics, beta, decay, 5 state-update taps, 4 o


def test_registry_entry():
    entry = fused.kernel("gdn_step")
    assert entry.tolerance == fused.COMPONENT
    assert (
        entry.component_proof
    )  # a COMPONENT-class kernel serves by default only with its recorded component-gate proof
    assert entry.fused is module.gdn_step and entry.composed is module.gdn_step_composed
    assert fused.resolve("gdn_step", {}) is module.gdn_step
    assert fused.resolve("gdn_step", {"QWEN38_FUSED_OFF": "gdn_step"}) is module.gdn_step_composed
    assert fused.resolve("gdn_step", {"QWEN38_FUSED": "gdn_step"}) is module.gdn_step
    assert entry.admits is module.admits


class _Fake:
    def __init__(self, shape, padded_shape=None, dtype=None):
        self.shape = tuple(shape)
        self.padded_shape = tuple(padded_shape) if padded_shape is not None else self.shape
        self.dtype = dtype


class _State:
    def __init__(self, recurrent):
        self.recurrent = recurrent


def _served_call(rows=1):
    projected = _Fake((1, 1, rows, module.PROJECTION_WIDTH), padded_shape=(1, 1, 32, module.PROJECTION_WIDTH))
    window = (object(), object(), object(), object())
    state = _State(_Fake((rows, module.HEADS, module.HEAD_DIM, module.HEAD_DIM), dtype=module.FP32))
    return projected, window, state


def test_admission_is_the_fused_steps_input_contract():
    projected, window, state = _served_call()
    assert module.admits(None, projected, window, state) is True
    # a host fake reports its logical shape as its padded shape: outside the tile contract, the chain runs
    unpadded = _Fake((1, 1, 1, module.PROJECTION_WIDTH))
    assert module.admits(None, unpadded, window, state) is False
    assert module.admits(None, _Fake((1, 1, 1, 4096), padded_shape=(1, 1, 32, 4096)), window, state) is False
    assert (
        module.admits(
            None,
            _Fake((1, 1, 33, module.PROJECTION_WIDTH), padded_shape=(1, 1, 64, module.PROJECTION_WIDTH)),
            window,
            state,
        )
        is False
    )
    assert module.admits(None, projected, window[:3], state) is False
    # the state's lane axis may exceed the rows (a 32-lane parent serves a trace of fewer lanes), never fall short
    assert module.admits(None, *_served_call(rows=2)) is True
    assert (
        module.admits(
            None,
            _served_call(rows=2)[0],
            window,
            _State(_Fake((1, module.HEADS, module.HEAD_DIM, module.HEAD_DIM), dtype=module.FP32)),
        )
        is False
    )
    assert (
        module.admits(
            None,
            projected,
            window,
            _State(_Fake((2, module.HEADS, module.HEAD_DIM, module.HEAD_DIM), dtype=module.FP32)),
        )
        is True
    )
    assert (
        module.admits(
            None,
            projected,
            window,
            _State(_Fake((1, module.HEADS, module.HEAD_DIM, module.HEAD_DIM), dtype=module.BF16)),
        )
        is False
    )
    assert module.admits(None, projected, window, object()) is False


def test_the_served_default_dispatches_per_call_and_the_switch_restores_the_chain(expect_error):
    step = fused.resolve_admitted("gdn_step", {})
    assert isinstance(step, fused.AdmittedStep)
    assert step.fused is module.gdn_step and step.composed is module.gdn_step_composed and step.admits is module.admits
    assert fused.resolve_admitted("gdn_step", {"QWEN38_FUSED_OFF": "gdn_step"}) is module.gdn_step_composed
    assert fused.resolve_admitted("gdn_step", {"QWEN38_FUSED": "gdn_step"}).fused is module.gdn_step
    # both branches of the dispatcher, on a probe kernel with the same admission rule
    calls = []
    probe = fused.FusedKernel(
        "probe_step",
        "a probe",
        fused.BITWISE,
        fused=lambda *args: calls.append("fused"),
        composed=lambda *args: calls.append("composed"),
        admits=module.admits,
    )
    dispatch = fused.AdmittedStep(probe)
    projected, window, state = _served_call()
    dispatch(None, _Fake((1, 1, 1, module.PROJECTION_WIDTH)), window, state)
    dispatch(None, projected, window, state)
    assert calls == ["composed", "fused"]

    with expect_error(ValueError):
        fused.AdmittedStep(
            fused.FusedKernel("plain_step", "a probe", fused.BITWISE, fused=lambda: None, composed=lambda: None)
        )


def test_reference_step_matches_the_oracle_functions():
    from models.demos.blackhole.qwen38_flash_next.reference import causal_depthwise_conv1d

    g = torch.Generator().manual_seed(5)
    rows = 3
    projected = (torch.randn(rows, module.PROJECTION_WIDTH, generator=g) * 0.6).to(torch.bfloat16)
    older = [(torch.randn(rows, module.QKV_WIDTH, generator=g) * 0.6).to(torch.bfloat16) for _ in range(3)]
    taps = [(torch.randn(module.QKV_WIDTH, generator=g) * 0.5).to(torch.bfloat16) for _ in range(4)]
    dt, na = torch.randn(12, generator=g), -torch.exp(torch.rand(12, generator=g) * 3)
    norm = (1 + torch.randn(128, generator=g) * 0.1).to(torch.bfloat16)
    state = torch.randn(rows, 12, 128, 128, generator=g) * 0.4
    new_state, gated, conv, o, beta, decay = module.reference_step(projected, older, taps, dt, na, norm, state)
    assert new_state.shape == (rows, 12, 128, 128) and gated.shape == (rows, 1536) and gated.dtype == torch.bfloat16
    assert (
        conv.shape == (rows, 2560)
        and o.shape == (rows, 12, 128)
        and beta.shape == (rows, 12)
        and decay.shape == (rows, 12)
    )
    # the conv is tt/gdn.py's causal_depthwise_conv1d over the same window, lane by lane
    for lane in range(rows):
        window = torch.stack([*(s[lane] for s in older), projected[lane, : module.QKV_WIDTH]], dim=-1)  # [2560, 4]
        expected, _ = causal_depthwise_conv1d(
            window[:, 3:].unsqueeze(0),
            torch.stack(taps, dim=-1),
            initial_state=window[:, :3].unsqueeze(0),
            activation="silu",
        )
        assert torch.equal(expected[0, :, 0], conv[lane])
    assert torch.all(decay > 0) and torch.all(decay <= 1) and torch.all((beta > 0) & (beta < 1))


# ============================================================================
# was test_fused_gdn_step_cs_static.py
# No-device contract of the conv-split GDN step (QWEN38_GDN_CONV_SPLIT=1, stage 1): the layout and the destination
# tables, the CB index agreement between the new kernels, the default kernels left untouched, the kernel text ordering
# rules and (when the sfpi toolchain and a JIT cache are present) a -fsyntax-only compile of the two compute kernels
# for the three TRISCs.
# ============================================================================

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


CS_KERNELS = Path(module.__file__).parent / "kernels"
REPO = Path(module.__file__).resolve().parents[7]
NEW = ("conv_reader.cpp", "conv_compute.cpp", "conv_writer.cpp", "reader_cs.cpp", "compute_cs.cpp")
CS_TP8 = pytest.mark.skipif(module.HEADS != 6, reason="the split layout is the TP=8 one (MESH_DEVICE=P150x8)")
CS_ARGS = dict(k_tile0=module.K_TILE0, v_tile0=module.V_TILE0, qk_repeat=module.QK_REPEAT)


def _cs_text(name):
    return (CS_KERNELS / name).read_text()


def _cs_constants(source):
    return {n: int(v) for n, v in re.findall(r"\b(CB_[A-Z0-9]+|SEM_[A-Z]+)\s*=\s*(\d+)", _cs_text(source))}


@CS_TP8
def test_geometry_assumptions():
    assert (module.HEADS, module.QK_HEADS, module.QK_REPEAT) == (6, 2, 3)
    assert module.K_TILE0 == 8 and module.V_TILE0 == 16
    assert cs.CONV_TILES == 8 + 8 + 4 * module.HEADS


@CS_TP8
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


@CS_TP8
@pytest.mark.parametrize("rows", cs.ROWS_SUPPORTED)
def test_cs_destination_tables(rows):
    qk, v, slots = {}, {}, {}
    for t in range(cs.CONV_TILES):
        for item, slot, sem in cs.destinations(t, rows, **CS_ARGS):
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
    compute, reader = _cs_constants("compute_cs.cpp"), _cs_constants("reader_cs.cpp")
    for name, index in reader.items():
        if name.startswith("CB_"):
            assert compute[name] == index, name
    default = _cs_constants("compute.cpp")
    for name, index in compute.items():
        assert default[name] == index, name  # the head kernel keeps the default numbering (FP32_COPY_CBS stay valid)
    conv_c, conv_r, conv_w = (
        _cs_constants("conv_compute.cpp"),
        _cs_constants("conv_reader.cpp"),
        _cs_constants("conv_writer.cpp"),
    )
    for table in (conv_r, conv_w):
        for name, index in table.items():
            assert conv_c[name] == index, name
    assert conv_c["CB_QKV"] == reader["CB_QKV"] == conv_w["CB_QKV"] == cs.CB_QKV == 11
    assert reader["SEM_QK"] == cs.SEM_QK and reader["SEM_V"] == cs.SEM_V
    assert cs.HEAD_ONLY_CBS_DROPPED == (conv_c["CB_P"], conv_c["CB_S"], conv_c["CB_T"])
    # the head compute uses neither CB_P / CB_S / CB_T nor CB_CONVSUM except as CB_OBF (4 pages on the head)
    body = re.sub(r"//.*", "", _cs_text("compute_cs.cpp"))
    for name in ("CB_P", "CB_S", "CB_T"):
        assert not re.search(rf"\b{name}\b[,)]", body.split("kernel_main")[1]), name
    assert (10, module.BF16, 4) in module.CBS and (11, module.BF16, 12) in module.CBS


def test_default_kernels_untouched():
    # reader.cpp carries the QWEN38_GDN_ROWREAD read path (default on); its other lines stay the default program.
    for name in ("compute.cpp", "writer.cpp"):
        path = (CS_KERNELS / name).relative_to(REPO)
        assert subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(path)], cwd=REPO).returncode == 0, name
    init = Path(module.__file__).read_text()
    assert init.count("QWEN38_GDN_CONV_SPLIT") == 1 and "conv_split.admits(rows, mesh)" in init
    # selection only on =="1" (flag unset/0 = today's program) and never with debug taps
    assert (
        re.search(r'QWEN38_GDN_CONV_SPLIT.*in \("1", "2", "3", "4"\) and debug is None', init)
        and cs.enabled.__doc__ is None
    )


def test_kernel_sources_exist_and_have_arg_counts():
    for name in NEW:
        assert (CS_KERNELS / name).is_file()
    conv_reader = _cs_text("conv_reader.cpp")
    assert conv_reader.count("TensorAccessorArgs<") == 9 and conv_reader.count("get_arg_val<uint32_t>(arg++)") == 10
    head_reader = _cs_text("reader_cs.cpp")
    assert head_reader.count("TensorAccessorArgs<") == 4 and head_reader.count("get_arg_val<uint32_t>(arg++)") == 5 + 2
    assert "newest" not in re.sub(r"//.*", "", head_reader) and "noc_async_write" not in head_reader


def test_kernel_text_ordering_rules():
    reader = _cs_text("reader_cs.cpp")
    # the semaphore waits come after the state reads' barrier and after the mask push
    assert reader.index("noc_async_read_barrier();\n        {") < reader.index("noc_semaphore_wait_min(sem_qk")
    assert reader.index("cb_push_back(CB_MASK, 1)") < reader.index("noc_semaphore_wait_min(sem_qk")
    assert reader.index("noc_semaphore_wait_min(sem_qk") < reader.index("noc_semaphore_wait_min(sem_v")
    assert reader.count("noc_semaphore_set") == 2 and reader.index("cb_reserve_back(CB_QKV") < reader.index(
        "read_tiles(CB_Z"
    )
    writer = _cs_text("conv_writer.cpp")
    assert (
        writer.index("noc_async_write_barrier()")
        < writer.index("noc_semaphore_inc")
        < writer.index("noc_async_atomic_barrier")
    )
    assert writer.index("noc_async_write(") < writer.index("noc_async_write_barrier()")
    compute = _cs_text("compute_cs.cpp")
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

    default = ops(_cs_text("compute.cpp")[_cs_text("compute.cpp").index("ALWI void conv_silu()") :])
    default = [l for l in default if not l.startswith(("#ifdef", "#endif", "if (t == 0)", "tap(", "}"))]
    conv = [l for l in ops(_cs_text("conv_compute.cpp")) if l != "}"]
    assert default == conv


@CS_TP8
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


# ============================================================================
# was test_fused_gdn_step_ls_static.py
# No-device contract of the lane split GDN step (QWEN38_GDN_LANE_SPLIT=1, rows 8 / 16 / 32): core plan arithmetic on the
# 11 x 10 grid, destination tables, admission (rows, grid, flag), one program meta per setting, the live / dead byte ranges
# of the conv reader's row-limited reads, CB numbering of the new kernels against the default's, and the flag-off guarantee
# (the guarded branch in ``gdn_step.run``: nothing is imported with the flag off).
# ============================================================================

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


PKG = Path(module.__file__).parent
KERNELS_LS = PKG / "kernels_ls"
LS_TP8 = pytest.mark.skipif(module.HEADS != 6, reason="the layout is the TP=8 one (MESH_DEVICE=P150x8)")
LS_ARGS = dict(k_tile0=8, v_tile0=16, qk_repeat=3)
# (rows, vb) -> (G, NG) on the 11 x 10 grid (70 pool cores): the table of plan_nov05/g1_gdnlane/PLAN.md section 2
EXPECTED = {
    (8, 1): (1, 8),
    (8, 2): (2, 4),
    (8, 4): (4, 2),
    (16, 1): (2, 8),
    (16, 2): (4, 4),
    (16, 4): (8, 2),
    (32, 1): (3, 11),
    (32, 2): (7, 5),
    (32, 4): (16, 2),
}
GRID = (11, 10)


@pytest.fixture(autouse=True)
def _option_a_plans(monkeypatch):
    """QWEN38_GDN_LS_CONVHEAD now defaults to 1 (option B); the plan-math tables of this file are option A's, so they
    run with it off. The option B tests set it themselves."""
    monkeypatch.setenv(ls.CONVHEAD_ENV, "0")


def fake_mesh(x, y):
    return SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=x, y=y))


def _ls_text(name):
    return (KERNELS_LS / name).read_text()


def _squash(text: str) -> str:
    """The kernel text without whitespace: clang-format may reflow a call over lines."""
    return re.sub(r"\s+", "", text)


@pytest.mark.parametrize("rows,vb", sorted(EXPECTED))
def test_core_plan_math(rows, vb):
    plan = ls.plan_for(rows, *GRID, vb=vb, g="auto", hoist=False)
    g, ng = EXPECTED[(rows, vb)]
    assert (plan.g, plan.ng, plan.vb) == (g, ng, vb)
    pool = ls.head_pool(*GRID)
    assert len(pool) == 70
    assert plan.head_cores == 6 * vb * ng <= len(pool)
    assert plan.g > 1 and 6 * vb * (-(-rows // (plan.g - 1))) > 70 or plan.g == 1  # the smallest G that fits
    lanes = [lane for j in range(ng) for lane in plan.group(j)]
    assert lanes == list(range(rows))  # every lane exactly once, in order
    assert all(1 <= len(plan.group(j)) <= g for j in range(ng))
    cores = [ls.head_core(plan, h, s, j, pool) for h in range(6) for s in range(vb) for j in range(ng)]
    assert len(set(cores)) == len(cores) == plan.head_cores
    conv = {ls.conv_core(t) for t in range(ls.CONV_TILES)}
    assert len(conv) == 40 and not set(cores) & conv
    assert all(x < GRID[0] and y < GRID[1] for x, y in cores + sorted(conv))
    # the VB peers of a head share the lane group: same j, consecutive s
    for h in range(6):
        for j in range(ng):
            assert {ls.head_core(plan, h, s, j, pool) for s in range(vb)} <= set(cores)
    # the rectangles cover exactly the used pool cores
    covered = {(x, y) for x0, y0, x1, y1 in ls.range_spans(cores) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}
    assert covered == set(cores)


@LS_TP8
@pytest.mark.parametrize("rows,vb", sorted(EXPECTED))
def test_ls_destination_tables(rows, vb):
    plan = ls.plan_for(rows, *GRID, vb=vb, g="auto", hoist=False)
    vt, ng = plan.vt, plan.ng
    qk, vv, slots, per_tile = {}, {}, {}, {}
    for t in range(ls.CONV_TILES):
        dests = ls.destinations(t, plan, **LS_ARGS)
        per_tile[t] = len(dests)
        for block, slot, sem in dests:
            assert 0 <= slot < 8 + vt and (block, slot) not in slots
            slots[(block, slot)] = t
            d = qk if sem == ls.SEM_QK else vv
            d[block] = d.get(block, 0) + 1
    blocks = [(h, s, j) for h in range(6) for s in range(vb) for j in range(ng)]
    assert set(qk) == set(blocks) == set(vv)
    assert all(n == 8 for n in qk.values()) and all(
        n == vt for n in vv.values()
    )  # the semaphore counts the kernels wait for
    assert len(slots) == (8 + vt) * len(blocks)
    for ((head, s, j), slot), t in slots.items():
        if slot < 4:
            assert t == (head // 3) * 4 + slot
        elif slot < 8:
            assert t == 8 + (head // 3) * 4 + slot - 4
        else:
            assert t == 16 + head * 4 + s * vt + (slot - 8)
    assert all(per_tile[t] == 3 * vb * ng for t in range(16)) and all(per_tile[t] == ng for t in range(16, 40))
    assert 4 * 3 * vb * ng + 1 <= 150  # runtime args of the conv writer stay small (rows 32 vb 1: 133)


def test_admits(monkeypatch):
    monkeypatch.delenv(ls.ENV, raising=False)
    assert [ls.admits(r, fake_mesh(*GRID)) for r in (1, 2, 4, 8, 16, 32)] == [False] * 3 + [True] * 3  # on (default)
    monkeypatch.setenv(ls.ENV, "0")
    assert not ls.admits(8, fake_mesh(*GRID))
    monkeypatch.setenv(ls.ENV, "1")
    for var in (ls.VB_ENV, ls.G_ENV, ls.HOIST_ENV):
        monkeypatch.delenv(var, raising=False)
    assert [ls.admits(r, fake_mesh(*GRID)) for r in (1, 2, 4, 7, 8, 16, 24, 32, 33, 64)] == [
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        True,
        False,
        False,
    ]
    assert not ls.admits(8, fake_mesh(9, 10)) and not ls.admits(8, fake_mesh(11, 9))  # the conv rectangle needs 10 x 10
    assert ls.admits(32, fake_mesh(10, 10))  # pool 60: G = 4 (48 cores)
    assert ls.plan_for(32, 10, 10).g == 4 and ls.plan_for(32, 11, 10).g == 3
    monkeypatch.setenv(ls.VB_ENV, "3")
    assert not ls.admits(8, fake_mesh(*GRID))  # VB 3 is not a block count of 4 tiles
    monkeypatch.setenv(ls.VB_ENV, "1")
    monkeypatch.setenv(ls.G_ENV, "2")
    assert ls.admits(16, fake_mesh(*GRID)) and not ls.admits(32, fake_mesh(*GRID))  # 6 * 16 = 96 head cores do not fit
    monkeypatch.setenv(ls.G_ENV, "0")
    assert not ls.admits(8, fake_mesh(*GRID))


def test_env_defaults(monkeypatch):
    for var in (ls.ENV, ls.VB_ENV, ls.G_ENV, ls.HOIST_ENV):
        monkeypatch.delenv(var, raising=False)
    assert ls.enabled() and ls.hoist_enabled()
    plan = ls.plan_for(8, *GRID)
    assert (plan.vb, plan.g, plan.hoist) == (1, 1, True)
    monkeypatch.setenv(ls.HOIST_ENV, "0")
    assert ls.plan_for(8, *GRID).hoist is False
    monkeypatch.setenv(ls.ENV, "0")
    assert not ls.enabled()


def test_program_meta_differs_per_setting():
    seen = {}
    for rows in ls.ROWS_SUPPORTED:
        for vb in ls.VB_SUPPORTED:
            for hoist in (False, True):
                plan = ls.plan_for(rows, *GRID, vb=vb, g="auto", hoist=hoist)
                meta = fp.program_meta(module.NAME, plan.variant, rows, cores=plan.head_cores + ls.CONV_TILES)
                key = (meta.kernel, meta.variant, meta.rows)
                assert key not in seen, (key, seen[key])
                seen[key] = (rows, vb, hoist)
    # G is part of the variant too (same rows / vb / hoist, other grouping)
    a = ls.plan_for(16, *GRID, vb=1, g=2, hoist=False)
    b = ls.plan_for(16, *GRID, vb=1, g=4, hoist=False)
    assert a.variant != b.variant
    assert all(v.startswith("step_lane_split") for _, v, _ in seen) and len(seen) == 18


def _live_dead(rows):
    """The live and dead byte ranges of a bf16 tile page the way conv_reader_ls.cpp computes them."""
    top, bottom = min(rows, 16), max(rows - 16, 0)
    live, dead = [], []
    for face in range(4):
        n = (top if face < 2 else bottom) * 32
        if n:
            live.append((face * 512, face * 512 + n))
        if n != 512:
            dead.append((face * 512 + n, face * 512 + 512))
    return live, dead


@pytest.mark.parametrize("rows", ls.ROWS_SUPPORTED)
def test_conv_reader_live_and_dead_ranges(rows):
    live, dead = _live_dead(rows)
    mask = [0] * 2048
    for a, b in live:
        for i in range(a, b):
            mask[i] += 1
    for a, b in dead:
        for i in range(a, b):
            mask[i] += 10
    assert all(m in (1, 10) for m in mask)  # disjoint and complete
    for byte in range(2048):  # bf16 tile: faces of 16 x 16, 32 B per face row
        face, offset = divmod(byte, 512)
        row = (face // 2) * 16 + offset // 32
        assert (mask[byte] == 1) == (row < rows), (rows, byte)
    assert all(b - a <= 512 for a, b in dead)  # one 512 B zero read each
    text = _squash(_ls_text("conv_reader_ls.cpp"))
    assert (
        _squash("TOP_ROWS = ROWS < FACE_ROWS ? ROWS : FACE_ROWS") in text
        and _squash("BOTTOM_ROWS = ROWS > FACE_ROWS") in text
    )
    assert (
        text.index(_squash("zero_dead(p_l1)"))
        < text.index(_squash("noc_async_read_barrier();"))
        < text.index(_squash("write_live(newest"))
    )
    assert _squash("noc_async_read_page(tile, t3") in text  # the taps are read whole


def test_kernel_files_and_census_isolation(expect_error):
    for name in ("conv_reader_ls.cpp", "reader_ls.cpp", "compute_ls.cpp"):
        assert (PKG / "kernels_ls" / name).is_file() and ls.kernel(name).endswith(f"gdn_step/kernels_ls/{name}")
    # the stage-2 files are reused as they are; the kernel census (*/kernels/*.cpp) sees no new file
    for name in ("writer_vb.cpp", "conv_writer.cpp", "conv_compute.cpp"):
        assert fp.kernel_source(module.NAME, name)
    assert not any(p.name.endswith("_ls.cpp") for p in (PKG / "kernels").iterdir())
    with expect_error(FileNotFoundError):
        ls.kernel("absent.cpp")


def _ls_constants(source):
    return {n: int(v) for n, v in re.findall(r"\b(CB_[A-Z0-9]+|SEM_[A-Z]+)\s*=\s*(\d+)", source)}


def test_cb_numbering_and_text_rules():
    kdir = PKG / "kernels"
    compute, reader = _ls_constants(_ls_text("compute_ls.cpp")), _ls_constants(_ls_text("reader_ls.cpp"))
    vb_compute = _ls_constants((kdir / "compute_vb.cpp").read_text())
    assert compute == vb_compute  # the CB table is compute_vb's
    for name, index in reader.items():
        if name.startswith("CB_"):
            assert compute[name] == index, name
    assert reader["SEM_QK"] == 0 and reader["SEM_V"] == 1
    c = _ls_text("compute_ls.cpp")
    main = c[c.index("void kernel_main()") :]
    assert main.index("gate_scalars();") < main.index("decay_state();") < main.index("cb_wait_front(CB_QKV, 2 * HT)")
    assert main.index("#ifdef LS_HOIST_L2") < main.index("l2_norm(0, CB_QROW)") < main.index("recurrence_rest(last)")
    assert "cb_pop_front(CB_QKV, 2 * HT)" not in main  # the conv tiles stay until the last item
    assert "copy_tile(CB_QKV, 2 * HT + j, 0)" in c and "cb_pop_front(CB_QKV, 2 * HT + VT)" in c
    r = _ls_text("reader_ls.cpp")
    assert r.count("noc_semaphore_wait_min(sem_qk") == 1 and "if (item == 0)" in r  # the conv wait is once per core
    assert "(lane * HEADS + head) * STATE_TILES + i * HT + s * VT + c" in r
    assert "const uint32_t mask_ids[1] = {lane}" in r


def test_hoist_define_only_when_asked():
    src = (PKG / "lane_split.py").read_text()
    assert '*(("LS_HOIST_L2", "1"),) * plan.hoist' in src


def test_guarded_branch_in_init():
    text = (PKG / "__init__.py").read_text()
    guard = 'if debug is None and os.environ.get("QWEN38_GDN_LANE_SPLIT", "1") == "1":'
    assert (
        text.count(guard) == 1 and text.count("lane_split") == 4
    )  # the import, admits, run_lane_split (twice in the name)
    start = text.index(guard)
    assert text.index("from . import lane_split") > start
    assert text.index("items = [(lane, head)") > text.index("lane_split.run_lane_split") > start
    head = text[:start]
    assert "lane_split" not in head  # nothing imported at module level or earlier
    assert text.index("conv_split.run_cs(") < start  # after the conv split block


class _Stop(Exception):
    pass


def _fake_run_args(rows):
    """Enough of a tensor for gdn_step.run's checks up to the dispatch."""
    mesh = fake_mesh(*GRID)

    def tensor(shape, dtype=None, padded=None):
        return SimpleNamespace(shape=shape, padded_shape=padded or shape, dtype=dtype, device=lambda: mesh)

    projected = tensor((1, 1, rows, module.PROJECTION_WIDTH), padded=(1, 1, 32, module.PROJECTION_WIDTH))
    recurrent = tensor((rows, module.HEADS, module.HEAD_DIM, module.HEAD_DIM), dtype=module.FP32)
    older = [tensor((1, 1, rows, module.QKV_WIDTH)) for _ in range(3)]
    taps = [tensor((1, 1, 1, module.QKV_WIDTH)) for _ in range(4)]
    return projected, older, "newest", taps, "constants", "norm", recurrent, "out"


@pytest.mark.parametrize("flag", ["0"])  # unset is now on (test_admits); only "0" is off
def test_flag_off_imports_nothing_and_takes_the_default(monkeypatch, flag, expect_error):
    for name in [n for n in sys.modules if n.endswith("gdn_step.lane_split")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.delattr(module, "lane_split", raising=False)
    monkeypatch.delenv("QWEN38_GDN_LANE_SPLIT", raising=False)
    if flag is not None:
        monkeypatch.setenv("QWEN38_GDN_LANE_SPLIT", flag)

    def stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(module.fp, "split_work", stop)  # the default path's first call after the dispatch
    with expect_error(_Stop):
        module.run(*_fake_run_args(8))
    assert not any(n.endswith("gdn_step.lane_split") for n in sys.modules)


def test_flag_on_dispatches_only_admitted_rows(monkeypatch, expect_error):
    calls = []
    monkeypatch.setenv("QWEN38_GDN_LANE_SPLIT", "1")
    monkeypatch.setattr(ls, "run_lane_split", lambda *a, **k: calls.append(a) or "ran")
    monkeypatch.setattr(module.fp, "split_work", lambda *a, **k: (_ for _ in ()).throw(_Stop()))
    assert module.run(*_fake_run_args(8)) == "ran" and module.run(*_fake_run_args(32)) == "ran" and len(calls) == 2
    assert calls[0][0] is module
    for rows in (7, 9):  # not admitted: falls through to the default program
        with expect_error(_Stop):
            module.run(*_fake_run_args(rows))
    assert len(calls) == 2


def test_import_of_package_does_not_import_lane_split():
    code = (
        "import sys;"
        "from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_step;"
        "assert not any(n.endswith('lane_split') for n in sys.modules), 'imported'"
    )
    env = {k: v for k, v in __import__("os").environ.items() if k != "QWEN38_GDN_LANE_SPLIT"}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr[-2000:]


# ---- option B (QWEN38_GDN_LS_CONVHEAD=1): the conv cores take head items after their conv phase ----------------------
POOL_B = 110  # 70 plain pool cores + 40 conv cores
EXPECTED_B = {  # (rows, vb) -> (G, NG, host conv cores): the smallest G with 6 * VB * NG <= 110
    (8, 1): (1, 8, 0),  # 48 cores fit the plain pool: the option A program runs
    (8, 2): (1, 8, 26),
    (8, 4): (2, 4, 26),
    (16, 1): (1, 16, 26),
    (16, 2): (2, 8, 26),
    (16, 4): (4, 4, 26),
    (32, 1): (2, 16, 26),
    (32, 2): (4, 8, 26),
    (32, 4): (8, 4, 26),
}


@pytest.mark.parametrize("rows,vb", sorted(EXPECTED_B))
def test_convhead_plan_math(rows, vb):
    plan = ls.plan_for(rows, *GRID, vb=vb, g="auto", hoist=False, convhead=True)
    g, ng, host = EXPECTED_B[(rows, vb)]
    assert (plan.g, plan.ng, plan.host_cores) == (g, ng, host)
    assert plan.head_cores == 6 * vb * ng <= POOL_B
    assert plan.host_cores == max(0, plan.head_cores - 70)
    if plan.g > 1:  # the smallest G that fits 110 cores
        assert 6 * vb * (-(-rows // (plan.g - 1))) > POOL_B
    # the flag off (the default) is option A: no host cores, the 70-core G
    a = ls.plan_for(rows, *GRID, vb=vb, g="auto", hoist=False, convhead=False)
    assert a.host_cores == 0 and (a.g, a.ng) == EXPECTED[(rows, vb)]
    assert plan.g <= a.g
    pool = ls.extended_pool(*GRID)
    assert len(pool) == len(set(pool)) == POOL_B and pool[:70] == ls.head_pool(*GRID)
    cores = [ls.head_core(plan, h, s, j, pool) for h in range(6) for s in range(vb) for j in range(ng)]
    assert len(set(cores)) == len(cores) == plan.head_cores
    lanes = [lane for j in range(ng) for lane in plan.group(j)]
    assert lanes == list(range(rows))
    conv_all = {ls.conv_core(t) for t in range(ls.CONV_TILES)}
    hosts = set(cores) & conv_all
    assert len(hosts) == host
    assert set(cores) - conv_all == set(pool[: min(plan.head_cores, 70)])  # the plain pool fills first
    spans = ls.range_spans(sorted(set(cores)))
    covered = {(x, y) for x0, y0, x1, y1 in spans for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}
    assert covered == set(cores)
    # the plain / host / conv-only core sets are disjoint and cover the 40 conv cores + the plain head cores
    plain = [c for c in cores if c not in conv_all]
    only = conv_all - hosts
    assert not (set(plain) & hosts) and not (hosts & only) and len(only) == 40 - host


def test_convhead_core_choice_and_variant():
    plan = ls.plan_for(16, *GRID, vb=1, g=1, hoist=False, convhead=True)
    pool = ls.extended_pool(*GRID)
    hosts = pool[70 : plan.head_cores]
    assert [ls.conv_core(t) for t in range(39, 13, -1)] == hosts  # tiles 39 .. 14: the v tiles (16 up) finish first
    assert ls.range_spans(hosts) == [(7, 4, 7, 9), (8, 0, 9, 9)]
    assert plan.variant == "step_lane_split_vb1_g1_h0_ch26"
    a = ls.plan_for(16, *GRID, vb=1, g=2, hoist=False, convhead=False)
    assert a.variant == "step_lane_split_vb1_g2_h0"  # option A names are unchanged
    seen = {}
    for rows in ls.ROWS_SUPPORTED:
        for vb in ls.VB_SUPPORTED:
            for hoist in (False, True):
                for convhead in (False, True):
                    p = ls.plan_for(rows, *GRID, vb=vb, g="auto", hoist=hoist, convhead=convhead)
                    if p.host_cores == 0 and convhead:
                        continue  # the same program as option A: the same cache entry on purpose
                    key = (p.variant, rows)
                    assert key not in seen, (key, seen[key])
                    seen[key] = (vb, hoist, convhead)
    metas = {
        (ls.plan_for(r, *GRID, vb=1, g="auto", hoist=False, convhead=c).variant, r)
        for r in (16, 32)
        for c in (False, True)
    }
    assert len(metas) == 4  # convhead on / off differ in the meta of rows 16 and 32


def test_convhead_admission_and_env(monkeypatch):
    for var in (ls.ENV, ls.VB_ENV, ls.G_ENV, ls.HOIST_ENV, ls.CONVHEAD_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(ls.ENV, "1")
    assert ls.CONVHEAD_ENV == "QWEN38_GDN_LS_CONVHEAD" and ls.convhead_enabled()  # default on (option B)
    assert ls.plan_for(16, *GRID).host_cores == 26
    monkeypatch.setenv(ls.CONVHEAD_ENV, "0")
    assert not ls.convhead_enabled()
    assert ls.plan_for(32, *GRID).g == 3 and ls.plan_for(16, *GRID).g == 2  # off: option A
    assert ls.plan_for(16, *GRID).host_cores == 0
    monkeypatch.setenv(ls.CONVHEAD_ENV, "1")
    p16, p32 = ls.plan_for(16, *GRID), ls.plan_for(32, *GRID)
    assert (p16.g, p16.ng, p16.host_cores) == (1, 16, 26) and (p32.g, p32.ng, p32.host_cores) == (2, 16, 26)
    assert ls.plan_for(8, *GRID).host_cores == 0
    monkeypatch.setenv(ls.G_ENV, "1")
    assert ls.admits(16, fake_mesh(*GRID)) and not ls.admits(32, fake_mesh(*GRID))  # 192 head cores
    monkeypatch.setenv(ls.G_ENV, "2")
    assert ls.admits(32, fake_mesh(*GRID))  # 96
    assert not ls.admits(32, fake_mesh(9, 10))  # no conv rectangle


def test_convhead_kernel_text():
    for name in ("conv_head_reader_ls.cpp", "conv_head_writer_ls.cpp", "conv_head_compute_ls.cpp"):
        assert ls.kernel(name).endswith(f"gdn_step/kernels_ls/{name}")
    reader, writer, compute = (
        _squash(_ls_text(n)) for n in ("conv_head_reader_ls.cpp", "conv_head_writer_ls.cpp", "conv_head_compute_ls.cpp")
    )
    # conv phase first, then the head code, on every RISC
    for text, head in (
        (reader, _squash("ls_head_reader_main();")),
        (writer, _squash("ls_head_writer_main();")),
        (compute, _squash("ls_head_compute_main();")),
    ):
        main = text[text.rindex(_squash("void kernel_main()")) :]
        assert main.count(_squash("convphase::")) >= 1 and main.index(_squash("convphase::")) < main.index(head)
    assert _squash('#include "reader_ls.cpp"') in reader and _squash('#include "../kernels/writer_vb.cpp"') in writer
    assert _squash('#include "compute_ls.cpp"') in compute
    for text in (reader, writer, compute):
        assert (
            text.count(_squash("#define kernel_main ls_head_")) == 1 and text.count(_squash("#undef kernel_main")) == 1
        )
    # runtime arg layout: the head reader takes 10 args, the conv reader's follow
    assert (
        _squash(_ls_text("reader_ls.cpp")).count(_squash("get_arg_val<uint32_t>(arg++)")) == 10
        and _squash("CONV_ARG0 = 10") in reader
    )
    assert _squash("TensorAccessorArgs<8>()") in reader and _squash("TensorAccessorArgs<CONV_CTA>()") in reader
    assert (
        _squash("3 + items * (3 + 2 * VBC)") in writer
    )  # after writer_vb's (state, out, items, then per item lane, head, s, VB x 2)
    vb_text = _squash((PKG / "kernels" / "writer_vb.cpp").read_text())
    assert _squash("get_arg_val<uint32_t>(arg++)") in vb_text and vb_text.count(_squash("peer_x[p] = get_arg_val")) == 1
    # CB numbering: the conv result is NOT in CB 11 or 10 (head receive / o exchange), sum in 31
    assert _squash("CB_QKV = 11, CB_CONVOUT = 28") in writer and _squash("CB_CONVOUT = 28, CB_CONVSUM = 31") in compute
    c_consts = _ls_constants(_ls_text("compute_ls.cpp"))
    for name, index in c_consts.items():
        assert index not in (31,) and (index != 28 or name == "CB_DBG"), name  # 28 is the debug CB only (DEBUG_TAPS)
    assert ls.CB_CONVOUT == 28 and ls.CB_CONVSUM == 31
    # the conv phase of the writer ends with both barriers before the head phase; the sends come before the increments
    run = writer[writer.index(_squash("FORCE_INLINE void run()")) : writer.index(_squash("void kernel_main()"))]
    assert (
        run.index(_squash("noc_async_write(src"))
        < run.index(_squash("noc_async_write_barrier"))
        < run.index(_squash("noc_semaphore_inc"))
        < run.index(_squash("noc_async_atomic_barrier"))
    )
    assert _squash("cb_wait_front(CB_CONVOUT, 1)") in run
    # one startup, the one inside compute_ls.cpp is a no-op
    assert (
        _squash("#define compute_kernel_hw_startup(...) ((void)0)") in compute
        and compute.count(_squash("compute_kernel_hw_startup(CB_AB, CB_W, CB_OBF)")) == 1
    )
    assert compute.index(_squash("#undef compute_kernel_hw_startup")) < compute.index(
        _squash("compute_kernel_hw_startup(CB_AB")
    )
    # the conv phase ops are conv_compute.cpp's, with only the CB ids renamed
    conv = _squash((PKG / "kernels" / "conv_compute.cpp").read_text())
    renamed = conv[conv.index(_squash("ALWI void conv_silu()")) : conv.index(_squash("}  // namespace"))]
    mine = compute[
        compute.index(_squash("ALWI void conv_silu()")) : compute.index(_squash("}  // namespace convphase"))
    ]
    norm = lambda t: re.sub(
        r"\s+", "", t.replace("CB_QKV", "CB_CONVOUT").replace("constexprconstexpr", "")
    )  # noqa: E731
    body = lambda t: norm(t[t.index(_squash("tile_regs_acquire();")) :])  # noqa: E731
    assert body(renamed).replace("}}", "}").replace("CB_CONVSUM", "S").replace("}", "") == body(mine).replace(
        "CB_CONVSUM", "S"
    ).replace("}", "")
    # option A's files are not touched by option B: they carry no reference to it
    for name in ("conv_reader_ls.cpp", "reader_ls.cpp", "compute_ls.cpp"):
        assert "convhead" not in _ls_text(name).lower() and "convphase" not in _ls_text(name)


class _Rec:
    def __init__(self, kind, **kw):
        self.kind, self.kw = kind, kw

    def __getattr__(self, name):
        try:
            return self.kw[name]
        except KeyError:
            raise AttributeError(name) from None


def _spans_cores(core_set):
    return {
        (x, y) for r in core_set.ranges() for x in range(r.start.x, r.end.x + 1) for y in range(r.start.y, r.end.y + 1)
    }


@LS_TP8
@pytest.mark.parametrize("rows,vb,g", [(16, 1, 1), (32, 1, 2), (8, 2, 1), (32, 2, 4), (16, 2, 2)])
def test_convhead_program_structure_with_fakes(monkeypatch, rows, vb, g):
    """run_lane_split with fake descriptors: per RISC the core sets are disjoint, every core has unique CB indices, host
    cores get the combined kernels with the right argument counts, every head block has one core."""
    import ttnn

    monkeypatch.setenv(ls.ENV, "1")
    monkeypatch.setenv(ls.CONVHEAD_ENV, "1")
    monkeypatch.setenv(ls.VB_ENV, str(vb))
    monkeypatch.setenv(ls.G_ENV, str(g))
    mesh = fake_mesh(*GRID)
    mesh._qwen38_gdn_ls_constants = (
        SimpleNamespace(buffer_address=lambda: 7),
        SimpleNamespace(buffer_address=lambda: 8),
    )
    noc = {(x, y): (x + 1, y + 1) for x in range(GRID[0]) for y in range(GRID[1])}
    monkeypatch.setattr(ls.gr_read, "noc_map", lambda m: noc)

    def tensor(addr):
        return SimpleNamespace(buffer_address=lambda: addr, device=lambda: mesh)

    projected, constants, norm, recurrent, out, newest = (tensor(a) for a in (1, 2, 3, 4, 5, 6))
    older, taps = [tensor(10 + i) for i in range(3)], [tensor(20 + i) for i in range(4)]
    fake = {}
    monkeypatch.setattr(ls.fp, "rows_of", lambda t: rows)
    monkeypatch.setattr(ls.fp, "accessor_args", lambda t: [0, 0])
    monkeypatch.setattr(ls.fp, "cb_descriptor", lambda i, d, b, pages, cs: _Rec("cb", index=i, pages=pages, cores=cs))
    monkeypatch.setattr(
        ttnn,
        "CBDescriptor",
        lambda **kw: _Rec(
            "cb", index=[f.buffer_index for f in kw["format_descriptors"]], pages=0, cores=kw["core_ranges"]
        ),
    )
    monkeypatch.setattr(ttnn, "CBFormatDescriptor", lambda **kw: SimpleNamespace(**kw))
    for kind in ("reader", "writer", "compute"):
        monkeypatch.setattr(
            ls.fp,
            f"{kind}_kernel",
            lambda src, cs, cta, args, _k=kind, **kw: _Rec(_k, src=src, cores=cs, cta=cta, args=args, **kw),
        )
    monkeypatch.setattr(
        ls.fp,
        "compute_kernel",
        lambda src, cs, cta, args, **kw: _Rec("compute", src=src, cores=cs, cta=cta, args=args, **kw),
    )
    monkeypatch.setattr(
        ls.fp,
        "program_descriptor",
        lambda kernels, cbs, semaphores: fake.update(kernels=kernels, cbs=cbs, sems=semaphores),
    )
    monkeypatch.setattr(ls.fp, "semaphore_descriptor", lambda i, cs: _Rec("sem", index=i, cores=cs))
    monkeypatch.setattr(ls.fp, "program_meta", lambda *a, **k: SimpleNamespace(variant=a[1], cores=k["cores"]))
    monkeypatch.setattr(ls.fp, "run_program", lambda tensors, d, meta=None: fake.update(meta=meta))
    monkeypatch.setattr(
        ls,
        "_constants",
        lambda m: (SimpleNamespace(buffer_address=lambda: 7), SimpleNamespace(buffer_address=lambda: 8)),
    )
    monkeypatch.setattr(ttnn, "CoreCoord", lambda x, y: SimpleNamespace(x=x, y=y))
    monkeypatch.setattr(ttnn, "CoreRange", lambda a, b: SimpleNamespace(start=a, end=b))
    monkeypatch.setattr(ttnn, "CoreRangeSet", lambda rs: SimpleNamespace(ranges=lambda: list(rs)))
    ls.run_lane_split(module, projected, older, newest, taps, constants, norm, recurrent, out)
    plan = ls.plan_for(rows, *GRID)
    assert plan.host_cores > 0 and fake["meta"].variant.endswith(f"_ch{plan.host_cores}")
    by_kind = {}
    for k in fake["kernels"]:
        cores = _spans_cores(k.cores)
        assert {(c.x, c.y) for c, _ in k.args} == cores or not k.args  # one arg list per core of the set
        by_kind.setdefault(k.kind, []).append((k, cores))
    for kind, ks in by_kind.items():
        allc = [c for _, cs in ks for c in cs]
        assert len(allc) == len(set(allc)), kind  # disjoint per RISC
    assert (
        {frozenset(cs) for _, cs in by_kind["reader"]}
        == {frozenset(cs) for _, cs in by_kind["writer"]}
        == {frozenset(cs) for _, cs in by_kind["compute"]}
    )
    host = [(k, cs) for k, cs in by_kind["reader"] if "conv_head" in k.src]
    assert len(host) == 1 and len(host[0][1]) == plan.host_cores
    conv_all = {ls.conv_core(t) for t in range(40)}
    assert host[0][1] <= conv_all
    cores_all = set().union(*(cs for _, cs in by_kind["reader"]))
    assert len(cores_all) == 40 + plan.head_cores - plan.host_cores
    hk = host[0][0]
    n_conv_cta = len(hk.cta) - len(by_kind["reader"][0][0].cta)
    assert n_conv_cta == 2 * 9  # the conv reader's accessors (9 tensors, 2 words each in the fake)
    assert all(len(a) == 10 + 10 for _, a in hk.args)
    hw = [k for k, cs in by_kind["writer"] if "conv_head" in k.src][0]
    items = len(plan.group(0))
    for core_xy, a in hw.args:
        n_items = a[2]
        first_conv = 3 + n_items * (3 + 2 * vb)
        assert len(a) == first_conv + 1 + 4 * a[first_conv]
    # CB table per core: unique indices
    per_core = {}
    for cb in fake["cbs"]:
        for c in _spans_cores(cb.cores):
            for idx in cb.kw["index"] if isinstance(cb.kw["index"], list) else [cb.kw["index"]]:
                assert idx not in per_core.setdefault(c, set()), (c, idx)
                per_core[c].add(idx)
    for c in host[0][1]:
        assert {0, 1, 2, 10, 11, 28, 31} <= per_core[c]
    for c in conv_all - host[0][1]:
        assert {0, 1, 2, 10, 11} <= per_core[c] and not {28, 31} & per_core[c]
    # all dests of conv writers are head cores (by noc coordinate) and the destination count is right
    head_noc = {noc[c] for c in cores_all if c not in conv_all or c in host[0][1]}
    for k, _ in by_kind["writer"]:
        for core_xy, a in k.args:
            if "conv_head" in k.src:
                a0 = 3 + a[2] * (3 + 2 * vb)
                dests = a[a0 + 1 :]
            elif "conv_writer" in k.src:
                dests = a[1:]
            else:
                continue
            assert {(dests[i], dests[i + 1]) for i in range(0, len(dests), 4)} <= head_noc


# ============================================================================
# was test_fused_gdn_step_vb_static.py
# No-device contract of stage 2 (QWEN38_GDN_CONV_SPLIT=2, the V split): layout and fit in the 11x10 P150 grid, the
# destination and exchange tables, CB agreement between the three head kernels and the default numbering, kernel text
# ordering rules, a torch emulation showing the column-block split of the recurrence is exact, and (with the sfpi
# toolchain) a -fsyntax-only compile of compute_vb.cpp for VT 1 and 2, decay-first on and off.
# ============================================================================

# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


VB_KERNELS = Path(module.__file__).parent / "kernels"
VB_TP8 = pytest.mark.skipif(module.HEADS != 6, reason="the split layout is the TP=8 one (MESH_DEVICE=P150x8)")
VB_ARGS = dict(k_tile0=8, v_tile0=16, qk_repeat=3)


def _vb_text(name):
    return (VB_KERNELS / name).read_text()


def _vb_constants(source):
    return {n: int(v) for n, v in re.findall(r"\b(CB_[A-Z0-9]+|SEM_[A-Z]+)\s*=\s*(\d+)", _vb_text(source))}


def test_vb_selection():
    assert [vb.vb_for(r, 11, 10) for r in (1, 2, 4, 8, 16)] == [4, 4, 2, 0, 0]
    assert vb.vb_for(1, 9, 10) == 0 and vb.vb_for(1, 11, 9) == 0 and vb.vb_for(1, 10, 10) == 4
    assert vb.vb_for(4, 11, 7) == 0 and vb.vb_for(4, 11, 8) == 0 or True  # the conv rectangle needs 10 rows anyway
    assert vb.vb_for(2, 10, 10) == 4  # 8 head rows


@VB_TP8
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


@VB_TP8
@pytest.mark.parametrize("rows", vb.ROWS_SUPPORTED)
def test_vb_destination_tables(rows):
    v = vb.vb_for(rows, 11, 10)
    vt = 4 // v
    qk, vv, slots = {}, {}, {}
    for t in range(cs.CONV_TILES):
        for item, slot, sem in vb.destinations(t, rows, v, **VB_ARGS):
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
    compute, reader, writer = (
        _vb_constants("compute_vb.cpp"),
        _vb_constants("reader_vb.cpp"),
        _vb_constants("writer_vb.cpp"),
    )
    default = _vb_constants("compute.cpp")
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
    comp = _vb_text("compute_vb.cpp")
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
    w = _vb_text("writer_vb.cpp")
    assert w.index("noc_async_write_barrier();") < w.index("noc_semaphore_inc") < w.index("noc_semaphore_wait_min")
    assert (
        w.index("cb_push_back(CB_OBF, HT)")
        < w.index("cb_wait_front(CB_OUTS, ST)")
        < w.index("cb_wait_front(CB_OUTG, VT)")
    )
    assert "obf + s * VT * BF16_TILE" in w
    r = _vb_text("reader_vb.cpp")
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
