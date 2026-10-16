# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contract of ``gdn_rows_scan`` (the verify rows' fold): the CB table shared by the kernels and the Python
side, the kernel argument layouts, the registry entry (opt-in, COMPONENT, never a default without a proof), the
shape-only admission, what ``attach`` gives a rows state and when, the layer's dispatch and commit wiring in
ttnn/gdn.py, the pick's accept-count read, and ``reference_rows`` as rows sequential ``gdn_step.reference_step`` calls.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_rows_scan as module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_rows_wrap as wrap
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_step
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

NAME = module.NAME
KERNELS = Path(module.__file__).parent / "kernels"
TILE, HEADS, HEAD_DIM = module.TILE, module.HEADS, module.HEAD_DIM
GDN_SOURCE = Path(gdn_module.__file__).read_text(encoding="utf-8")


def _constants(source: str) -> dict[str, int]:
    text = (KERNELS / source).read_text()
    return {name: int(value) for name, value in re.findall(r"\b(CB_[A-Z0-9]+)\s*=\s*(\d+)", text)}


# ------------------------------------------------------------------------------------------- the CB table


def test_cb_indices_agree_between_kernels_and_python():
    compute, reader, writer = _constants("compute.cpp"), _constants("reader.cpp"), _constants("writer.cpp")
    for name, index in reader.items():
        assert compute[name] == index, name
    for name, index in writer.items():
        assert compute[name] == index, name
    for rows, vbt in ((5, 4), (2, 1), (8, 4)):
        table = module.cb_table(rows, vbt)
        declared = {index for index, _, _ in table} | {module.CB_SNEWC, module.CB_OUTS, module.CB_DEBUG}
        assert len(declared) == len(table) + 3 and max(declared) <= 31
        used = set(compute.values())
        assert used <= declared, sorted(used - declared)
        # the depths that follow the item: the conv tiles, the state tiles, one a/b pair and one mask per row
        pages = {index: pages for index, _, pages in table}
        assert pages[0] == 8 + vbt and pages[1] == 3 * (8 + vbt) and pages[2] == 4 * (8 + vbt) and pages[11] == 8 + vbt
        assert pages[7] == pages[20] == pages[21] == pages[24] == 4 * vbt
        assert pages[4] == 2 and pages[8] == rows and pages[31] == 2 * vbt  # the a / b tiles once, one mask per row
        assert pages[18] == rows and pages[19] == rows  # one beta / decay tile per row, replicated by the reader
        dtypes = {index: dtype for index, dtype, _ in table}
        assert pages[13] == 2 and dtypes[13] == ttnn.float32  # beta_all / decay_all (compute -> reader)
    assert (
        compute["CB_SNEWC"] == module.CB_SNEWC == 30
        and compute["CB_OUTS"] == module.CB_OUTS == 25
        and compute["CB_OROWS"] == module.CB_OROWS == 29
        and compute["CB_OBF"] == module.CB_OBF == 31
        and compute["CB_STATE"] == module.CB_STATE
    )
    # the o hand-off to the writer is its own buffer (gdn_step aliases it with the conv sum, which only the compute pops)
    assert compute["CB_OBF"] != compute["CB_CONVSUM"]
    # exact fp32 copies: the CBs the compute consumes with copy_tile; matmul / reduce / bcast operands stay Default
    assert set(module.FP32_COPY_CBS) == {
        compute[n] for n in ("CB_STATE", "CB_DTNA", "CB_BETA", "CB_DECAY", "CB_SDECC", "CB_VREAD", "CB_SNEWC")
    }
    assert not set(module.FP32_COPY_CBS) & {
        compute[n] for n in ("CB_QROW", "CB_KROW", "CB_KCOL", "CB_SDEC", "CB_DELTAB", "CB_SNEW", "CB_SCALER", "CB_RS")
    }
    # the state carry is read back by copy_tile (CB_SNEWC) and multiplied by matmul (CB_SNEW): two packs, two modes
    text = (KERNELS / "compute.cpp").read_text()
    assert "pack_tile(j, CB_SNEW);" in text and "pack_tile(j, CB_SNEWC);" in text
    assert "matmul_tiles(CB_QROW, CB_SNEW" in text and "copy_tile(src, t, 0)" in text
    # the gates: one whole-tile SFPU pass per gate (no per-row scalar chain), the results to the reader as fp32 tiles,
    # which fans each row's element out into the full tiles the recurrence multiplies (exact 32-bit copies)
    assert "gates_all();" in text and "gate_scalars" not in text and "unary_bcast" not in text
    assert (
        "copy_tile(CB_AB, 1, 0);" in text and "copy_tile(CB_AB, 0, 0);" in text and "cb_push_back(CB_GALL, 2);" in text
    )
    assert "cb_wait_front(CB_BETA, row + 1);" in text and "cb_wait_front(CB_DECAY, row + 1);" in text
    reader_text = (KERNELS / "reader.cpp").read_text()
    assert "cb_wait_front(CB_GALL, 2);" in reader_text and "fill_tile_words(" in reader_text
    assert "4 * fused_rows::face_element(r, head)" in reader_text and "cb_push_back(CB_AB, 2);" in reader_text
    assert reader["CB_GALL"] == compute["CB_GALL"] == 13 and reader["CB_BETA"] == 18 and reader["CB_DECAY"] == 19


def test_per_core_l1_of_the_largest_form_fits():
    for rows, vbt in ((module.MAX_ROWS, 4), (5, 4), (5, 1)):
        table = module.cb_table(rows, vbt)
        total = (
            sum(pages * fp.TILE_BYTES[dtype] for _, dtype, pages in table) + 2 * 4 * vbt * fp.TILE_BYTES[ttnn.float32]
        )
        assert total <= 1_100_000, (rows, vbt, total)  # a Blackhole Tensix holds 1.5 MB of L1


def test_kernel_argument_layouts():
    reader = (KERNELS / "reader.cpp").read_text()
    assert reader.count("TensorAccessorArgs<") == 9  # projected, history, 4 taps, dtna, norm, state
    assert reader.count("get_arg_val<uint32_t>(arg++)") == 10 + 2  # 9 addresses, items, then (head, vb)
    assert "TensorAccessorArgs<2>" in reader and "get_compile_time_arg_val(0)" in reader
    writer = (KERNELS / "writer.cpp").read_text()
    assert writer.count("TensorAccessorArgs<") == 3 and "TensorAccessorArgs<2>" in writer  # prefix, out, debug
    assert writer.count("get_arg_val<uint32_t>(arg++)") == 3 + 1 + 2 + 2  # + the debug address, the peers' (x, y)
    assert "noc_semaphore_wait_min(sem, PEERS)" in writer and "get_semaphore(0)" in writer
    pick = (KERNELS / "pick.cpp").read_text()
    assert pick.count("TensorAccessorArgs<") == 3 and "TensorAccessorArgs<1>" in pick  # accepted, prefix, recurrent
    assert pick.count("get_arg_val<uint32_t>(arg++)") == 4 + 2
    compute = (KERNELS / "compute.cpp").read_text()
    assert "get_arg_val<uint32_t>(0)" in compute
    assert "get_compile_time_arg_val(0)" in compute and "get_compile_time_arg_val(1)" in compute
    # the pick converts the fp32 accept count to a slot index on the RISC and clamps it to the rows
    assert "reinterpret_cast<volatile tt_l1_ptr float*>" in pick and "accepted >= ROWS" in pick
    # the row helpers are the shared header's
    for source in ("reader.cpp", "writer.cpp"):
        assert '#include "../../kernels/row_mask.h"' in (KERNELS / source).read_text()
    header = (KERNELS.parent.parent / "kernels" / "row_mask.h").read_text()
    assert "one_hot_row_bf16" in header and "copy_row_bf16" in header and "face_element" in header


def test_the_builder_passes_the_rows_and_the_split_as_compile_time_args():
    source = inspect.getsource(module.run)
    assert "reader_cta = [rows, vbt]" in source and "writer_cta = [rows, vbt," in source and "[rows, vbt]," in source
    assert "gr_read.noc_map(mesh)" in source and "fp.semaphore_descriptor(0, cores)" in source
    assert "unpack_to_dest_fp32=FP32_COPY_CBS" in source and "fidelity=ttnn.MathFidelity.HiFi4" in source
    assert "fp32_dest=True" in source and 'fp.program_meta(\n        NAME,\n        "verify_rows",' in source
    assert module.SPLIT in (1, 4) and module.HT % module.SPLIT == 0
    assert module.MAX_ROWS == 8 and module.PROGRAMS_PER_LAYER == 7 and module.COMMIT_PROGRAMS_PER_LAYER == 3
    # the wrap's six become one (+ the landing slice)
    pick = inspect.getsource(module.run_pick)
    assert 'fp.program_meta(\n        NAME,\n        "commit_pick",' in pick and "fp.reader_kernel(PICK" in pick


# --------------------------------------------------------------------------------------- the registry entry


def test_registry_entry_is_a_default_component_kernel_with_its_gate_record():
    entry = fused.kernel(NAME)
    assert entry.tolerance == fused.COMPONENT and entry.default_on is True and NAME in fused.DEFAULT_ON
    assert entry.fused is module.rows_body_scan and entry.composed is module.rows_body_fallback
    assert entry.admits is module.admits and entry.gate is None
    assert "probe-rows5-8edcaf9279c-162454" in entry.component_proof and "0.0048" in entry.component_proof
    assert "gdn_pre_rows" in entry.replaces and "prefix" in entry.replaces
    # the default: the admitted dispatcher; QWEN38_FUSED_OFF restores the wrap outright
    assert fused.resolve_admitted(NAME, {fused.OFF_ENV: NAME}) is module.rows_body_fallback
    assert fused.resolve(NAME, {fused.OFF_ENV: NAME}) is module.rows_body_fallback
    on = fused.resolve_admitted(NAME, {})
    assert isinstance(on, fused.AdmittedStep) and on.fused is module.rows_body_scan
    assert on.composed is module.rows_body_fallback and on.admits is module.admits
    assert fused.resolve_admitted(NAME, {fused.ENV: NAME, fused.OFF_ENV: NAME}) is module.rows_body_fallback
    assert fused.enabled(NAME, {}) is True and fused.enabled(NAME, {fused.OFF_ENV: NAME}) is False
    # the wrap stays the default beside it
    assert fused.kernel(wrap.NAME).default_on is True


# ------------------------------------------------------------------------------------- admission and attach


class _Fake:
    def __init__(self, shape=(1,), dtype=None):
        self.shape, self.dtype = tuple(shape), dtype

    def is_allocated(self) -> bool:
        return True

    def buffer_address(self) -> int:
        return 0


class _HostFake(_Fake):
    buffer_address = None


class _Constants:
    def __init__(self, *, rows=5, tile_rows=TILE):
        self.rows, self.tile_rows = rows, tile_rows


class _RowsState:
    def __init__(self, *, rows=5, tile_rows=TILE, flat_qk=False, owns_body=True, host=False):
        self.constants = _Constants(rows=rows, tile_rows=tile_rows)
        self.flat_qk, self.owns_body = flat_qk, owns_body
        fake = _HostFake if host else _Fake
        self.v = fake((1, 1, tile_rows, module.VALUE_WIDTH), ttnn.bfloat16)
        self.history = fake((1, 1, TILE, module.QKV_WIDTH), ttnn.bfloat16)
        self.qkv = fake((1, 1, tile_rows, module.QKV_WIDTH), ttnn.bfloat16)


def _state(shape=(1, HEADS, HEAD_DIM, HEAD_DIM), dtype=ttnn.float32):
    return SimpleNamespace(recurrent=_Fake(shape, dtype))


def test_qualifies_is_the_verify_form_with_up_to_max_rows():
    assert (
        module.qualifies(_RowsState()) and module.qualifies(_RowsState(rows=1)) and module.qualifies(_RowsState(rows=8))
    )
    assert not module.qualifies(_RowsState(rows=9))  # the prefix states are rows x 786 KB per layer
    assert not module.qualifies(_RowsState(rows=32)) and not module.qualifies(_RowsState(rows=128, tile_rows=128))
    assert not module.qualifies(_RowsState(flat_qk=True)) and not module.qualifies(_RowsState(owns_body=False))
    assert not module.qualifies(_RowsState(host=True))  # a host fake keeps today's stream
    assert not module.qualifies(SimpleNamespace())


def test_attach_needs_the_form_and_honours_the_off_switch(monkeypatch):
    allocated = []
    monkeypatch.setattr(
        fp, "allocate", lambda shape, dtype, layout, mesh, *a: allocated.append((shape, dtype)) or _Fake(shape, dtype)
    )
    monkeypatch.setattr(fp, "stamp_topology", lambda tensor, reference, shard_dim=None: tensor)
    monkeypatch.setattr(gdn_step, "_constants", lambda gdn: "dt-na-tiles")
    gdn = SimpleNamespace(mesh_device="mesh", out_proj_act_memory_config="out-proj-shard")
    # off (QWEN38_FUSED_OFF names the fold): nothing allocated; the default with a form past MAX_ROWS: refused
    assert module.attach(gdn, _RowsState(), {fused.OFF_ENV: NAME}) is None and allocated == []
    assert module.attach(gdn, _RowsState(rows=9), {}) is None and allocated == []
    state = _RowsState(rows=5)
    buffers = module.attach(gdn, state, {})  # the default: no switch needed
    assert buffers is module.buffers_of(state) is state.scan_buffers
    assert allocated == [((5, HEADS, HEAD_DIM, HEAD_DIM), ttnn.float32)]
    assert buffers.rows == 5 and buffers.constants == "dt-na-tiles" and buffers.gated_memory_config == "out-proj-shard"
    assert module.buffers_of(_RowsState()) is None


def test_admits_is_the_fused_bodys_input_contract():
    state = _RowsState()
    assert module.admits(None, None, state, _state()) is False  # no buffers attached
    state.scan_buffers = "buffers"
    assert module.admits(None, None, state, _state()) is True
    assert module.admits(None, None, state, _state((2, HEADS, HEAD_DIM, HEAD_DIM))) is False
    assert module.admits(None, None, state, _state(dtype=ttnn.bfloat16)) is False
    assert module.admits(None, None, state, SimpleNamespace()) is False
    other = _RowsState(rows=9)
    other.scan_buffers = "buffers"
    assert module.admits(None, None, other, _state()) is False


def test_commit_needs_the_selectors_accept_count(expect_error):
    with expect_error(RuntimeError):
        module.commit(None, None, SimpleNamespace(prefix=None), _state(), SimpleNamespace(accepted=None))
    selectors = gdn_module.Qwen38TTNNRowsSelectors(5, None, (), None)
    assert selectors.accepted is None  # defaulted: older constructions and the wrap's selectors carry none


# ------------------------------------------------------------------------------------------ the gdn.py wiring


def _methods():
    tree = ast.parse(GDN_SOURCE)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen38TTNNGDN")
    return {n.name: ast.get_source_segment(GDN_SOURCE, n) for n in cls.body if isinstance(n, ast.FunctionDef)}


def test_the_layer_resolves_the_fold_once_and_dispatches_through_it():
    methods = _methods()
    assert f'self._rows_scan_call = fused.resolve_admitted("{NAME}")' in methods["__init__"]
    assert 'self.__dict__.get("_rows_scan_call")' in methods["_rows_body"]
    assert f'fused.resolve_admitted("{NAME}")' in methods["_rows_body"]
    assert 'self.__dict__.get("_rows_body_call")' in methods["_rows_body_wrap"]
    assert "body = self._rows_body()" in methods["forward_rows"]
    # the fallback is the layer's wrap-or-chain getter
    assert "gdn._rows_body_wrap()(gdn, full_hidden, rows_state, state, full_tile=full_tile)" in inspect.getsource(
        module.rows_body_fallback
    )
    # the fold is attached first; the wrap only where it did not attach
    allocate = methods["allocate_rows_state"]
    assert "if fused.gdn_rows_scan.attach(self, rows_state) is None:" in allocate
    assert "fused.gdn_rows_wrap.attach(self, rows_state)" in allocate
    assert allocate.index("gdn_rows_scan.attach") < allocate.index("gdn_rows_wrap.attach")


def test_the_commit_under_the_fold_is_the_pick_and_the_history_advance():
    methods = _methods()
    commit = methods["commit_rows"]
    fold = commit[commit.index("scan = fused.gdn_rows_scan.buffers_of(rows_state)") :]
    fold = fold[: fold.index("if step_committed_rows:")]
    assert "fused.gdn_rows_scan.commit(self, rows_state, scan, state, selectors)" in fold
    assert "self._advance_history_rows(rows_state, selectors)" in fold and "state.validate()" in fold
    assert "raise ValueError" in fold and "step_on_full_rejection" in fold  # the 1-row anchors are refused
    assert "_chunk_rows" not in fold  # no masked re-run
    full = methods["commit_rows_full"]
    assert "if final_state is None and scan is not None:" in full
    assert "fused.gdn_rows_scan.commit_all_rows(scan, state)" in full
    # the fused body returns no final state: the rows result says so
    assert ", None\n" in inspect.getsource(module.rows_body_scan)
    assert "under the verify-rows fold" in GDN_SOURCE
    # the selectors carry the accept count the pick reads
    assert "accepted=accepted" in inspect.getsource(gdn_module.build_rows_selectors)
    assert "accepted: Any = None" in GDN_SOURCE and "scan_buffers: Any = None" in GDN_SOURCE


def test_the_fused_body_is_the_wraps_frame_around_one_program():
    calls = [
        ast.unparse(node.func)
        for node in sorted(
            (n for n in ast.walk(ast.parse(inspect.getsource(module.rows_body_scan))) if isinstance(n, ast.Call)),
            key=lambda n: (n.lineno, n.col_offset),
        )
    ]
    assert [c for c in calls if c.startswith("gdn._")] == [
        "gdn._project_rows_linear",
        "gdn._land_rows_qkv",
        "gdn._out_proj_tile",
        "gdn._rows_output_tile",
    ]
    # the placements come from the launch itself (the program meta's outputs), not from a hand stamp after it
    assert calls.count("run") == 1 and "fp.stamp_topology" not in calls
    source = inspect.getsource(module)
    assert "outputs=((out, 3), (prefix, 1), *([(debug, None)] if debug is not None else []))," in source
    assert "outputs=((recurrent, 1),)," in source  # commit_pick: the state slot written in place


# ------------------------------------------------------------------------------------------ the reference


def test_reference_rows_is_rows_sequential_reference_steps():
    g = torch.Generator().manual_seed(9)
    rows = 3
    projected = (torch.randn(TILE, module.PROJECTION_WIDTH, generator=g) * 0.6).to(torch.bfloat16)
    history = (torch.randn(TILE, module.QKV_WIDTH, generator=g) * 0.6).to(torch.bfloat16)
    taps = [(torch.randn(module.QKV_WIDTH, generator=g) * 0.5).to(torch.bfloat16) for _ in range(4)]
    dt, na = torch.randn(HEADS, generator=g), -torch.exp(torch.rand(HEADS, generator=g) * 3)
    norm = (1 + torch.randn(HEAD_DIM, generator=g) * 0.1).to(torch.bfloat16)
    state = torch.randn(HEADS, HEAD_DIM, HEAD_DIM, generator=g) * 0.4
    prefix, gated = module.reference_rows(projected, history, taps, dt, na, norm, state, rows)
    assert prefix.shape == (rows, HEADS, HEAD_DIM, HEAD_DIM) and prefix.dtype == torch.float32
    assert gated.shape == (rows, module.VALUE_WIDTH) and gated.dtype == torch.bfloat16
    # by hand: the ring of row r is window rows r .. r + 2 of [history rows 0..2 | the rows' q|k|v]
    window = torch.cat([history[:3], projected[:, : module.QKV_WIDTH]], dim=0)
    current = state.unsqueeze(0)
    for row in range(rows):
        older = [window[row + i : row + i + 1] for i in range(3)]
        current, expected, *_ = gdn_step.reference_step(projected[row : row + 1], older, taps, dt, na, norm, current)
        assert torch.equal(prefix[row], current[0]) and torch.equal(gated[row], expected[0])
    # rows past the real ones do not enter: the same prefix from a tile whose tail differs
    other = projected.clone()
    other[rows:] = 1.0
    prefix2, gated2 = module.reference_rows(other, history, taps, dt, na, norm, state, rows)
    assert torch.equal(prefix, prefix2) and torch.equal(gated, gated2)


def test_the_served_admission_charges_the_prefix_states_when_the_fold_is_on():
    """The MTP DRAM admission (tools/qwen38_chat_session.py) carries the fold's persistent prefix states, derived from
    k + 1 and the GDN layer count, when QWEN38_FUSED names the kernel: the 4x p150 line measured 87,394,112 bytes per
    bank at k = 4 with the fold (states 31,093,824) where the estimate without the term was 77,095,515."""

    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session

    assert session.fold_prefix_states_bytes_per_bank(4) == 36 * -(-(5 * HEADS * (HEAD_DIM // TILE) ** 2) // 8) * 4096
    assert session.fold_prefix_states_bytes_per_bank(module.MAX_ROWS) == 0
    record = session.mtp_capacity_admission(32768, drafts=4, verify_forms=2, gdn_rows_scan=True)
    assert record["mtp_growth_estimate_bytes_per_bank"]["states"] >= 31_093_824
    assert record["required_free_bytes_per_bank"] >= 87_394_112 and record["fits"]
    assert session.mtp_capacity_admission(32768, drafts=4, verify_forms=2)["required_free_bytes_per_bank"] < 87_394_112


# ------------------------------------------------------------------------------------------ the lanes form


def test_lanes_form_shares_the_compute_kernel_and_leaves_the_single_form_untouched():
    """The lanes form (B lanes x R rows lane-major, one (value head, lane) item per core) is the single form's
    compute kernel with ROWS = R plus its own reader / writer / pick; the single form's three kernels and ``run`` do
    not know the lanes exist (byte-identical served fold)."""

    for source in ("reader.cpp", "writer.cpp", "pick.cpp", "compute.cpp"):
        text = (KERNELS / source).read_text()
        assert "LANES" not in text and "lane *" not in text and "row0" not in text, source
    assert "lanes" not in inspect.getsource(module.run) and "lanes" not in inspect.getsource(module.run_pick)
    for source in ("reader_lanes.cpp", "writer_lanes.cpp", "pick_lanes.cpp"):
        assert (KERNELS / source).is_file(), source
    assert module.READER_LANES.endswith("reader_lanes.cpp") and module.WRITER_LANES.endswith("writer_lanes.cpp")
    assert module.PICK_LANES.endswith("pick_lanes.cpp")
    lanes_run = inspect.getsource(module.run_lanes)
    assert "COMPUTE," in lanes_run and "READER_LANES," in lanes_run and "WRITER_LANES," in lanes_run
    assert "reader_cta = [rows, vbt, lanes]" in lanes_run and "writer_cta = [rows, vbt, lanes," in lanes_run
    assert "[rows, vbt]," in lanes_run and "vbt = HT" in lanes_run  # the compute's args: R rows, the whole head
    assert "unpack_to_dest_fp32=FP32_COPY_CBS" in lanes_run and "fp32_dest=True" in lanes_run
    assert "fp.semaphore_descriptor" not in lanes_run  # no cross-core exchange: one core owns one lane of one head
    assert 'fp.program_meta(\n        NAME,\n        "verify_rows_lanes",' in lanes_run
    assert "outputs=((out, 3), (prefix, 1))," in lanes_run
    pick = inspect.getsource(module.run_pick_lanes)
    assert (
        'fp.program_meta(\n        NAME,\n        "commit_pick_lanes",' in pick
        and "fp.reader_kernel(PICK_LANES" in pick
    )
    assert "outputs=((recurrent, 1),)," in pick


def test_lanes_kernel_argument_layouts_and_the_finiteness_rule():
    reader = (KERNELS / "reader_lanes.cpp").read_text()
    assert reader.count("TensorAccessorArgs<") == 9 and "TensorAccessorArgs<3>" in reader  # after ROWS, VBT, LANES
    assert reader.count("get_arg_val<uint32_t>(arg++)") == 10 + 2  # 9 addresses, items, then (head, lane)
    assert "get_compile_time_arg_val(2)" in reader and "static_assert(VBT == HT" in reader
    assert "LANES * ROWS <= 32" in reader
    # lane u's state pages, lane u's history tile row, the masks and gates of tile rows row0 + r
    assert "(lane * HEADS + head) * HT * HT + kt * HT + c" in reader and "history_page0 + ids[t]" in reader
    assert "one_hot_row_bf16(l1 + r * BF16_TILE, row0 + r)" in reader
    assert "4 * fused_rows::face_element(row0 + r, head)" in reader
    # the per-lane patch after the assembly's barrier, RISC word copies, no-op for lane 0
    assert "patch_lane_rows(s_l1 + (3 * t) * BF16_TILE, row0)" in reader and "if (lane > 0)" in reader
    assert reader.index("noc_async_read_barrier();  // group t + 1") < reader.index("patch_lane_rows(s_l1")
    assert "j < s && j < ROWS" in reader and "copy_row_bf16_between(dst_tile, row0 + j, s0_tile, i + j)" in reader
    assert "FINITENESS RULE" in reader and "NaN * 0 = NaN" in reader
    writer = (KERNELS / "writer_lanes.cpp").read_text()
    assert writer.count("TensorAccessorArgs<") == 2 and "TensorAccessorArgs<3>" in writer  # prefix, out
    assert writer.count("get_arg_val<uint32_t>(arg++)") == 3 + 2  # addresses, items, (head, lane)
    assert "((row0 + r) * HEADS + head) * STATE_TILES + kt * HT + c" in writer  # prefix slot row0 + r
    assert "fused_rows::copy_row_bf16(orows + j * BF16_TILE, src + j * BF16_TILE, row0 + r)" in writer
    assert "out.get_noc_addr(page, off)" in writer and "if (lane + 1 == LANES)" in writer  # row pieces; pad rows
    assert "noc_semaphore" not in writer and "PEERS" not in writer
    pick = (KERNELS / "pick_lanes.cpp").read_text()
    assert pick.count("TensorAccessorArgs<") == 3 and "TensorAccessorArgs<2>" in pick  # counts, prefix, recurrent
    assert pick.count("get_arg_val<uint32_t>(arg++)") == 4 + 3  # addresses, items, (lane, head, col)
    assert "if (c == 0) {" in pick and "continue;" in pick  # c_u = 0 keeps the lane's state
    assert "const uint32_t slot = lane * ROWS + c - 1;" in pick and "c > ROWS ? ROWS : c" in pick
    for source in ("reader_lanes.cpp", "writer_lanes.cpp"):
        assert '#include "../../kernels/row_mask.h"' in (KERNELS / source).read_text()


def test_lanes_admission_and_prefix_bytes():
    assert module.MAX_LANES == 8
    assert module.lanes_admitted(4, 5) and module.lanes_admitted(6, 5) and module.lanes_admitted(8, 4)
    assert module.lanes_admitted(2, 8) and module.lanes_admitted(1, 1)
    assert not module.lanes_admitted(8, 5) and not module.lanes_admitted(9, 3)  # the tile; the lane count
    assert not module.lanes_admitted(2, 9) and not module.lanes_admitted(0, 5)  # MAX_ROWS per lane
    assert module.lanes_prefix_shape(4, 5) == (20, HEADS, HEAD_DIM, HEAD_DIM)
    # one slot = 12 x 16 tile pages of 4 KiB; 36 GDN layers at rows 20 = 70,778,880 B per bank over 8 banks
    assert module.prefix_states_bytes(1) == 786_432 and module.prefix_states_bytes(20, 36) // 8 == 70_778_880
    assert (
        module.prefix_states_bytes(30, 36) // 8 == 106_168_320
        and module.prefix_states_bytes(32, 36) // 8 == 113_246_208
    )


class _LaneConstants:
    def __init__(self, lanes, rows):
        self.lanes, self.rows = lanes, rows


class _LaneRowsState:
    def __init__(self, *, lanes=4, rows=5, tile_rows=TILE, host=False, mismatch=False):
        self.lanes = lanes
        self.constants = _Constants(rows=rows, tile_rows=tile_rows)
        self.lane_constants = _LaneConstants(lanes + (1 if mismatch else 0), rows)
        fake = _HostFake if host else _Fake
        self.v = fake((1, lanes, TILE, module.VALUE_WIDTH), ttnn.bfloat16)
        self.history = fake((1, lanes, TILE, module.QKV_WIDTH), ttnn.bfloat16)
        self.qkv = fake((1, 1, TILE, module.QKV_WIDTH), ttnn.bfloat16)
        self.qkv_lanes = fake((1, 1, lanes * TILE, module.QKV_WIDTH), ttnn.bfloat16)
        self.scan_buffers = None


def _lane_state(lanes=4, dtype=ttnn.float32):
    return SimpleNamespace(recurrent=_Fake((lanes, HEADS, HEAD_DIM, HEAD_DIM), dtype))


def test_qualifies_lanes_is_the_lane_major_verify_form_within_the_admission():
    assert module.qualifies_lanes(_LaneRowsState()) and module.qualifies_lanes(_LaneRowsState(lanes=8, rows=4))
    assert module.qualifies_lanes(_LaneRowsState(lanes=2, rows=8))
    assert not module.qualifies_lanes(_LaneRowsState(lanes=8, rows=5))  # 40 rows: not one tile
    assert not module.qualifies_lanes(_LaneRowsState(lanes=4, rows=9))  # MAX_ROWS per lane
    assert not module.qualifies_lanes(_LaneRowsState(tile_rows=128)) and not module.qualifies_lanes(
        _LaneRowsState(host=True)
    )
    assert not module.qualifies_lanes(_LaneRowsState(mismatch=True))  # the lane constants disagree with the state
    assert not module.qualifies_lanes(_RowsState())  # the single form's rows state has no lanes
    assert not module.qualifies_lanes(SimpleNamespace())


def test_attach_lanes_needs_the_form_and_honours_the_shared_off_switch(monkeypatch):
    allocated = []
    monkeypatch.setattr(
        fp, "allocate", lambda shape, dtype, layout, mesh, *a: allocated.append((shape, dtype)) or _Fake(shape, dtype)
    )
    monkeypatch.setattr(fp, "stamp_topology", lambda tensor, reference, shard_dim=None: tensor)
    monkeypatch.setattr(gdn_step, "_constants", lambda gdn: "dt-na-tiles")
    gdn = SimpleNamespace(mesh_device="mesh", out_proj_act_memory_config="out-proj-shard")
    # the same switch as the single form: QWEN38_FUSED_OFF=gdn_rows_scan turns the lanes form off too
    assert module.attach_lanes(gdn, _LaneRowsState(), {fused.OFF_ENV: NAME}) is None and allocated == []
    assert module.attach_lanes(gdn, _LaneRowsState(lanes=8, rows=5), {}) is None and allocated == []
    state = _LaneRowsState(lanes=4, rows=5)
    buffers = module.attach_lanes(gdn, state, {})  # the default: no switch needed
    assert buffers is module.lane_buffers_of(state) is state.scan_buffers
    assert allocated == [((20, HEADS, HEAD_DIM, HEAD_DIM), ttnn.float32)]
    assert (buffers.lanes, buffers.rows) == (4, 5) and buffers.constants == "dt-na-tiles"
    assert buffers.gated_memory_config == "out-proj-shard"
    assert module.lane_buffers_of(_LaneRowsState()) is None
    # the single form's buffers are not the lanes form's and vice versa
    single = _RowsState()
    module.attach(gdn, single, {})
    assert module.lane_buffers_of(single) is None and module.buffers_of(state) is state.scan_buffers


def test_admits_lanes_is_the_lane_bodys_input_contract(expect_error):
    state = _LaneRowsState()
    assert module.admits_lanes(state, _lane_state()) is False  # no buffers attached
    state.scan_buffers = module.LaneBuffers(prefix=None, constants=None, gated_memory_config=None, lanes=4, rows=5)
    assert module.admits_lanes(state, _lane_state()) is True
    assert module.admits_lanes(state, _lane_state(lanes=2)) is False  # another lane count
    assert module.admits_lanes(state, _lane_state(dtype=ttnn.bfloat16)) is False
    assert module.admits_lanes(state, SimpleNamespace()) is False
    with expect_error(RuntimeError):
        module.rows_body_scan_lanes(None, None, state, _lane_state(lanes=2))  # a wiring error, not a fallback


def test_commit_lanes_needs_the_selectors_committed_counts(expect_error):
    buffers = module.LaneBuffers(prefix=None, constants=None, gated_memory_config=None, lanes=4, rows=5)
    with expect_error(RuntimeError):
        module.commit_lanes(None, None, buffers, _lane_state(), SimpleNamespace(committed_counts=None))
    selectors = gdn_module.Qwen38TTNNRowsSelectorsLanes(4, 5, None, None, None, None, ())
    assert selectors.committed_counts is None  # defaulted: older constructions carry none
    assert "committed_counts=committed" in inspect.getsource(gdn_module.build_rows_selectors_lanes)
    assert "self.committed_counts," in inspect.getsource(gdn_module.Qwen38TTNNRowsSelectorsLanes.deallocate)


def test_the_lane_layer_attaches_the_fold_and_dispatches_the_body_and_the_commit():
    methods = _methods()
    assert "fused.gdn_rows_scan.attach_lanes(self, rows_state)" in methods["allocate_lane_rows_state"]
    # the lane body follows the registry exactly as forward_rows does: the one resolved step, its fused callable the
    # fold's lanes form on a lane rows state, its composed callable the lanes chain
    forward = methods["forward_rows_lanes"]
    assert "body = self._rows_body()" in forward and "body(self, full_hidden, rows_state, state)" in forward
    assert "_project_rows(" not in forward and "_chunk_rows_lanes(" not in forward
    chain = methods["_rows_body_lanes_chain"]
    assert "self._chunk_rows_lanes(rows_state, initial_state=state.recurrent)" in chain
    assert "return output, final_state" in chain
    assert "return rows_body_scan_lanes(gdn, full_hidden, rows_state, state)" in inspect.getsource(
        module.rows_body_scan
    )
    assert "gdn._rows_body_lanes_chain(full_hidden, rows_state, state)" in inspect.getsource(module.rows_body_fallback)
    assert "return admits_lanes(rows_state, state)" in inspect.getsource(module.admits)
    lane_state = _LaneRowsState()
    assert module.is_lane_rows_state(lane_state) and not module.is_lane_rows_state(_RowsState())
    assert module.admits(None, None, lane_state, _lane_state()) is False  # no lanes buffers: the lanes chain
    lane_state.scan_buffers = module.LaneBuffers(prefix=None, constants=None, gated_memory_config=None, lanes=4, rows=5)
    assert module.admits(None, None, lane_state, _lane_state()) is True
    commit = methods["commit_rows_lanes"]
    fold = commit[commit.index("scan = fused.gdn_rows_scan.lane_buffers_of(rows_state)") :]
    assert "fused.gdn_rows_scan.commit_lanes(self, rows_state, scan, state, selectors)" in fold
    assert fold.index("commit_lanes(") < fold.index("_chunk_rows_lanes(")  # the pick, else the masked re-run
    assert "self._lane_conv_window(rows_state)" in fold and "state.validate()" in fold  # the history advance stays
    assert "scan_buffers: Any = None" in ast.get_source_segment(
        GDN_SOURCE,
        next(
            n
            for n in ast.parse(GDN_SOURCE).body
            if isinstance(n, ast.ClassDef) and n.name == "Qwen38TTNNGDNLaneRowsState"
        ),
    )
    # the lane body: the projection frame, the landing and its expand (the commit's window), one program, the out-proj
    calls = [
        ast.unparse(node.func)
        for node in sorted(
            (n for n in ast.walk(ast.parse(inspect.getsource(module.rows_body_scan_lanes))) if isinstance(n, ast.Call)),
            key=lambda n: (n.lineno, n.col_offset),
        )
    ]
    assert [c for c in calls if c.startswith("gdn._")] == [
        "gdn._project_rows_linear",
        "gdn._land_rows_qkv",
        "gdn._select_rows",
        "gdn._out_proj_tile",
    ]
    assert calls.count("run_lanes") == 1 and "fp.stamp_topology" not in calls


def test_reference_rows_lanes_is_reference_rows_per_lane():
    g = torch.Generator().manual_seed(11)
    lanes, rows = 3, 2
    projected = (torch.randn(TILE, module.PROJECTION_WIDTH, generator=g) * 0.6).to(torch.bfloat16)
    history = (torch.randn(lanes, TILE, module.QKV_WIDTH, generator=g) * 0.6).to(torch.bfloat16)
    taps = [(torch.randn(module.QKV_WIDTH, generator=g) * 0.5).to(torch.bfloat16) for _ in range(4)]
    dt, na = torch.randn(HEADS, generator=g), -torch.exp(torch.rand(HEADS, generator=g) * 3)
    norm = (1 + torch.randn(HEAD_DIM, generator=g) * 0.1).to(torch.bfloat16)
    state = torch.randn(lanes, HEADS, HEAD_DIM, HEAD_DIM, generator=g) * 0.4
    prefix, gated = module.reference_rows_lanes(projected, history, taps, dt, na, norm, state, lanes, rows)
    assert prefix.shape == (lanes * rows, HEADS, HEAD_DIM, HEAD_DIM) and gated.shape == (
        lanes * rows,
        module.VALUE_WIDTH,
    )
    for lane in range(lanes):
        tile = torch.zeros_like(projected)
        tile[:rows] = projected[lane * rows : (lane + 1) * rows]
        p_lane, g_lane = module.reference_rows(tile, history[lane], taps, dt, na, norm, state[lane], rows)
        assert torch.equal(prefix[lane * rows : (lane + 1) * rows], p_lane)
        assert torch.equal(gated[lane * rows : (lane + 1) * rows], g_lane)
