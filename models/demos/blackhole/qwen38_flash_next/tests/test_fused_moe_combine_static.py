# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The one-pass MoE combine program (``ttnn/fused/moe_combine``) without a device: the registry entry (BITWISE,
default, admitted), the CB / argument contracts between the Python side and the three kernels, the exactness pins in
the compute kernel (the chain's tilize helper and the fused reduce's MAC init and slot order, read against the chain's
own kernel sources), the reader's score-tile addressing, the work plan and the L1 budget, the settings knobs, the
admission predicate on host fakes, and the release manifest."""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import contracts, fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import moe_combine as mc
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import moe_post as mp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

KERNELS = {name: (fp.REPO_ROOT / path).read_text() for name, path in mc.KERNELS.items()}
CHAIN_COMPUTE = (
    fp.REPO_ROOT
    / "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/kernels/deepseek_moe_fast_reduce_nc_fused_compute.cpp"
).read_text()
CHAIN_READER = (
    fp.REPO_ROOT
    / "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/device/kernels/deepseek_moe_fast_reduce_nc_fused_reader.cpp"
).read_text()
CHAIN_TILIZE = (fp.REPO_ROOT / "ttnn/cpp/ttnn/kernel/compute/tilize.cpp").read_text()


def _named(source: str) -> set[str]:
    return set(re.findall(r'get_named_compile_time_arg_val\("([a-z_0-9]+)"\)', source))


def _runtime_indices(source: str) -> list[int]:
    return sorted(int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d+)\)", source))


def test_registered_bitwise_opt_in_and_admitted():
    entry = fused.kernel(mc.NAME)
    assert entry.tolerance == fused.BITWISE
    assert entry.default_on, "the one-pass combine is the slab's default form (BITWISE, line gate 2026-09-26)"
    assert entry.fused is mc.moe_combine and entry.composed is mc.moe_combine_composed
    assert entry.admits is mc.admits and entry.gate is None
    assert fused.resolve(mc.NAME, {}) is mc.moe_combine
    assert fused.resolve(mc.NAME, {fused.OFF_ENV: mc.NAME}) is mc.moe_combine_composed
    assert fused.resolve_admitted(mc.NAME, {fused.OFF_ENV: mc.NAME}) is mc.moe_combine_composed
    assert isinstance(fused.resolve_admitted(mc.NAME, {}), fused.AdmittedStep)
    assert (
        inspect.signature(mc.moe_combine).parameters.keys()
        == inspect.signature(mc.moe_combine_composed).parameters.keys()
    )
    assert inspect.signature(mc.admits).parameters.keys() == inspect.signature(mc.moe_combine).parameters.keys()


def test_named_compile_time_args_match_the_python_side():
    named = {name for name, *_ in mc.CBS} | {
        "cols",
        "top_k",
        "groups",
        "hidden_tiles",
        "experts",
        "owner_bytes",
        "read_all",
    }
    source = inspect.getsource(mc.moe_combine_program)
    for name in named - {name for name, *_ in mc.CBS}:
        assert f'("{name}",' in source, name
    assert _named(KERNELS["reader"]) == {
        "cb_rm",
        "cb_scores",
        "cb_owner",
        "cb_route",
        "cols",
        "top_k",
        "groups",
        "experts",
        "owner_bytes",
        "read_all",
    }
    assert _named(KERNELS["compute"]) == {"cb_rm", "cb_tiled", "cb_scores", "cb_out", "cols", "top_k", "groups"}
    assert _named(KERNELS["writer"]) == {"cb_out", "cols", "groups", "hidden_tiles"}
    for kernel in KERNELS.values():
        assert _named(kernel) <= named


def test_runtime_args_are_the_declared_lists():
    assert _runtime_indices(KERNELS["reader"]) == list(range(len(mc.READER_ARGS)))
    assert _runtime_indices(KERNELS["compute"]) == list(range(len(mc.COMPUTE_ARGS)))
    assert _runtime_indices(KERNELS["writer"]) == list(range(len(mc.WRITER_ARGS)))
    source = inspect.getsource(mc.moe_combine_program)
    # the reader's four accessors in the order of its runtime addresses, then the units
    assert (
        source.index("fp.accessor_args(combine)")
        < source.index("fp.accessor_args(scores)")
        < source.index("fp.accessor_args(indices)")
        < source.index("fp.accessor_args(owner)")
    )
    reader_args = re.search(r"owner\.buffer_address\(\),\s*w\.start,\s*w\.count,\s*rows,", source)
    assert reader_args is not None  # unit_start, unit_count, rows after the four addresses
    assert re.search(r"const uint32_t rows =\s*get_arg_val<uint32_t>\(6\);", KERNELS["reader"])  # not compile-time
    for kernel in KERNELS.values():
        assert "if (unit_count == 0) {\n        return;" in kernel
    assert "[(w.core, [out.buffer_address(), w.start, w.count]) for w in work]" in source
    assert "[(w.core, [w.start, w.count]) for w in work]" in source
    assert "fidelity=ttnn.MathFidelity.HiFi4" in source and "fp32_dest=True" in source


def test_compute_kernel_mirrors_the_chain():
    """The tilize is the to_layout op's helper with its bf16 mode; the MAC is the fused reduce's init (acc_to_dest on
    the COL-broadcast ELWMUL at MATH_FIDELITY) and slot order into DEST tile 0 with one pack."""

    compute = KERNELS["compute"]
    assert "compute_kernel_lib::tilize<" in compute and "tilize_config::Fp32Mode::Fast" in compute
    assert "tilize_config::Fp32Mode::Fast" in CHAIN_TILIZE  # the op's bf16 branch
    for line in (
        "bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(",
        "llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(",
        "1 /*acc_to_dest*/",
        "mul_tiles_bcast_cols(",
        "tile_regs_acquire();",
        "tile_regs_commit();",
        "tile_regs_wait();",
        "tile_regs_release();",
    ):
        assert line in compute and line in CHAIN_COMPUTE, line
    assert compute.count("pack_tile(0, cb_out);") == 1
    # slot order e = 0 .. top_k - 1 inside one acquire, the score tile of slot e, DEST tile 0
    mac = re.search(
        r"for \(uint32_t e = 0; e < top_k; \+\+e\) \{.*?mul_tiles_bcast_cols\(cb_tiled, cb_scores, e \* cols \+ c, e, 0\);",
        compute,
        re.S,
    )
    assert mac is not None
    assert compute.index("tile_regs_acquire();") < mac.start() < compute.index("tile_regs_commit();")
    assert "compute_kernel_hw_startup(cb_tiled, cb_scores, cb_out);" in compute
    assert "reconfig_data_format(cb_tiled, cb_scores);" in compute and "pack_reconfig_data_format(cb_out);" in compute


def test_reader_builds_the_chain_score_tiles():
    """Column 0 of row j (face 0 for j < 16, face 2 for j >= 16), the owned score else +0.0; the owner row indexed by
    the expert with the out-of-range guard; the block rows at the tilize reader's pitch; the zero fill barriered."""

    reader = KERNELS["reader"]
    assert "(j < 16) ? j * 16 : 512 + (j - 16) * 16" in reader
    assert "(j < 16) ? j * 16 : face2_offset + (j - 16) * 16" in CHAIN_READER and "face2_offset = 512" in CHAIN_READER
    assert "stile[k * tile_u16 + col0] = own ? sc_u16[j * route_words + k] : static_cast<uint16_t>(0);" in reader
    assert "expert < experts && owner_u16[expert] != 0" in reader
    assert "const uint32_t dst = base + j * frag_bytes;" in reader
    assert ".page_id = e * rows + r * 32 + j, .offset_bytes = g * frag_bytes" in reader
    assert "noc.async_write_zeros(CoreLocalMem<uint32_t>(dst), frag_bytes, {});" in reader
    assert reader.index(
        "noc.async_read_barrier();\n                if (zeroed) {\n                    noc.write_zeros_l1_barrier();"
    ) < reader.index("rm.push_back(cols);")
    assert "route_pitch = 64" in reader  # a 20-byte DRAM page lands at its own 64-byte phase (W1)


def test_writer_addresses_the_tiled_output():
    writer = KERNELS["writer"]
    assert "const uint32_t first = r * hidden_tiles + g * cols;" in writer
    assert "out.get_noc_addr(first + c)" in writer
    assert writer.index("noc_async_write_barrier();") < writer.index("out_tiles.pop_front(cols);")


@pytest.mark.parametrize("cols", mc.COLS_ADMITTED)
def test_cb_plan_fits_l1_and_is_dram_aligned(cols):
    for _name, _index, _dtype, page_bytes, pages in mc.CBS:
        assert (page_bytes * pages(cols)) % 64 == 0
    assert mc.l1_bytes(cols) <= mc.L1_BUDGET
    assert mc.HIDDEN_TILES % cols == 0
    # the writer's ring holds whole units: 2 x cols pages
    assert dict((n, p(cols)) for n, _i, _d, _b, p in mc.CBS)["cb_out"] == 2 * cols
    # the tilize consumes cols pages per slot block and produces cols tiles; the MAC needs all top_k x cols
    plan = dict((n, p(cols)) for n, _i, _d, _b, p in mc.CBS)
    assert plan["cb_rm"] % cols == 0 and plan["cb_tiled"] == mc.TOP_K * cols and plan["cb_scores"] == 2 * mc.TOP_K


def test_cb_indices_are_distinct_and_out_is_16():
    indices = [index for _n, index, *_ in mc.CBS]
    assert len(set(indices)) == len(indices) and mc.CB_INDEX["cb_out"] == 16


def test_work_plan_covers_the_output_evenly(expect_error):
    """Units = row tiles x column groups, split contiguously as evenly as the grid allows (fp.split_work)."""

    assert mc.unit_count(2048, 8) == 640 and mc.unit_count(2048, 16) == 320 and mc.unit_count(32, 8) == 10
    with expect_error(ValueError):
        mc.unit_count(2048, 3)

    @dataclass
    class Grid:
        x: int
        y: int

    class Mesh:
        def compute_with_storage_grid_size(self):
            return Grid(11, 10)

    work = fp.split_work(mc.unit_count(2048, 8), Mesh())
    assert sum(w.count for w in work) == 640 and len(work) == 110
    assert {w.count for w in work} == {5, 6}
    assert [w.start for w in work] == [sum(v.count for v in work[:i]) for i in range(len(work))]  # contiguous


def test_settings_from_env_and_configured(expect_error, monkeypatch):
    assert mc.settings_from_env({}) == mc.Settings(cols=16, read_all=False)
    assert mc.settings_from_env({mc.COLS_ENV: "16", mc.READ_ALL_ENV: "1"}) == mc.Settings(16, True)
    with expect_error(ValueError, match="must be one of"):
        mc.settings_from_env({mc.COLS_ENV: "3"})
    with expect_error(ValueError, match="must be one of"):
        mc.settings_from_env({mc.COLS_ENV: "40"})  # 40 x 10 x 2 KB twice: over the L1 budget, so not admitted
    monkeypatch.setattr(mc, "_settings", None)
    monkeypatch.delenv(mc.COLS_ENV, raising=False)
    base = mc.settings()
    with mc.configured(cols=16, read_all=True) as cfg:
        assert cfg.cols == 16 and cfg.read_all and mc.settings() is cfg
    assert mc.settings() == base
    assert not hasattr(
        mc, "ZONES_ENV"
    )  # the phase zones come from the shared QWEN38_FUSED_ZONES (program.zone_defines)


def test_max_rows_is_the_slab_contract():
    assert mc.MAX_ROWS == contracts.MAX_SLAB_ROWS and mc.MIN_ROWS == 32
    assert mc.CHAIN_BLOCK_ROWS == 512


def test_owner_rows_are_the_ep4_shards():
    rows = mc.owner_rows()
    assert rows.shape == (4, mc.EXPERTS) and rows.dtype == torch.int16
    for d in range(4):
        assert torch.equal(rows[d].bool(), (torch.arange(mc.EXPERTS) // mc.EXPERTS_PER_DEVICE) == d)
    assert mc.owner_rows is mp.owner_rows  # one owner row definition for moe_post and moe_combine
    assert mc.chain_mapping is mp.chain_mapping and mc.chain_compute_config is mp.chain_compute_config


class _Fake:
    def __init__(self, shape, dtype, layout):
        self.shape = shape
        self.padded_shape = shape
        self.dtype = dtype
        self.layout = layout


def _fakes(rows=2048, **overrides):
    fakes = {
        "combine": _Fake((mc.TOP_K, rows, mc.HIDDEN), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        "scores": _Fake((1, 1, rows, mc.TOP_K), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        "indices": _Fake((1, 1, rows, mc.TOP_K), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT),
        "owner": _Fake((1, mc.EXPERTS), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT),
    }
    fakes.update(overrides)
    return fakes


def test_admits_the_slab_contract_and_refuses_the_rest(expect_error):
    assert mc.admits(**_fakes())
    assert mc.admits(**_fakes(rows=32)) and mc.admits(**_fakes(rows=4096)) and mc.admits(**_fakes(rows=96))
    assert not mc.admits(**_fakes(rows=4128))  # over MAX_SLAB_ROWS
    assert not mc.admits(**_fakes(rows=100))  # not whole tiles
    assert not mc.admits(**_fakes(combine=_Fake((mc.TOP_K, 2048, mc.HIDDEN), ttnn.bfloat16, ttnn.TILE_LAYOUT)))
    assert not mc.admits(**_fakes(combine=_Fake((mc.TOP_K, 1024, mc.HIDDEN), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)))
    assert not mc.admits(**_fakes(scores=_Fake((1, 1, 2048, mc.TOP_K), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)))
    assert not mc.admits(**_fakes(indices=_Fake((1, 1, 2048, mc.TOP_K), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)))
    assert not mc.admits(**_fakes(owner=_Fake((4, mc.EXPERTS), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT)))
    assert not mc.admits(None, None, None, None)
    with expect_error(ValueError, match="combine must be ROW_MAJOR bf16"):
        mc._check_inputs(**_fakes(combine=_Fake((mc.TOP_K, 2048, 64), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)))


class _Placed(_Fake):
    def __init__(self, shape, dtype, layout, device):
        super().__init__(shape, dtype, layout)
        self._device = device

    def device(self):
        return self._device


class _Dev:
    def __init__(self, ident):
        self._id = ident

    def id(self):
        return self._id


def test_admits_refuses_inputs_on_different_devices(expect_error):
    a, b = _Dev(0), _Dev(1)
    fakes = {k: _Placed(v.shape, v.dtype, v.layout, a) for k, v in _fakes().items()}
    assert mc.admits(**fakes)
    fakes["owner"] = _Placed(fakes["owner"].shape, ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, b)
    assert not mc.admits(**fakes)
    with expect_error(ValueError, match="share one device"):
        mc._check_inputs(**fakes)


def test_module_is_imported_by_the_package():
    assert "moe_combine" in fused.__all__ and fused.moe_combine is mc
