# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The fused-kernel foundation without a device: the registry and its switch, the rows contract, the work split, the
kernel-source convention and the writer kernel's argument contract."""

import inspect
import re
import sys
import types
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import registry
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.untilize_rows import untilize_rows, untilize_rows_composed

NAME = "untilize_rows"
DEFAULT_ON = registry.DEFAULT_ON  # the one list that decides what serves by default


def test_registry_default_is_the_composed_chain_for_an_opt_in_kernel():
    assert NAME in fused.kernels()
    assert fused.kernel(NAME).tolerance == fused.BITWISE and fused.kernel(NAME).default_on is False
    assert NAME not in fused.enabled_names({})
    assert fused.resolve(NAME, {}) is untilize_rows_composed
    assert fused.enabled(NAME, {}) is False


def test_registry_serves_the_proven_kernels_by_default():
    assert "router_tail" in DEFAULT_ON and DEFAULT_ON <= set(fused.kernels())
    assert fused.default_names() == DEFAULT_ON
    assert fused.enabled_names({}) == DEFAULT_ON
    for name in DEFAULT_ON:
        entry = fused.kernel(name)
        assert entry.default_on
        assert entry.tolerance == fused.BITWISE or (entry.tolerance == fused.COMPONENT and entry.component_proof)
        assert fused.resolve(name, {}) is fused.kernel(name).fused
        assert fused.resolve(name, {fused.OFF_ENV: name}) is fused.kernel(name).composed
    for name in set(fused.kernels()) - DEFAULT_ON:
        assert not fused.kernel(name).default_on and fused.resolve(name, {}) is fused.kernel(name).composed
    assert fused.enabled_names({fused.OFF_ENV: "router_tail"}) == DEFAULT_ON - {"router_tail"}
    assert fused.enabled_names({fused.OFF_ENV: "all"}) == frozenset()
    assert fused.enabled_names({fused.ENV: "all", fused.OFF_ENV: "all"}) == frozenset()
    both = {fused.ENV: "gr_read", fused.OFF_ENV: "router_tail"}
    assert fused.enabled_names(both) == (DEFAULT_ON | {"gr_read"}) - {"router_tail"}
    assert fused.enabled_names({fused.ENV: NAME}) == DEFAULT_ON | {NAME}


def test_default_on_list_must_name_registered_kernels(expect_error, monkeypatch):
    monkeypatch.setattr(registry, "DEFAULT_ON", registry.DEFAULT_ON | {"nope_kernel"})
    with expect_error(ValueError, match="DEFAULT_ON names unregistered fused kernels \\['nope_kernel'\\]"):
        fused.enabled_names({})


@pytest.mark.parametrize("value", [NAME, f" {NAME} ,", f"{NAME},{NAME}", "all"])
def test_registry_switch_on(value):
    assert fused.enabled(NAME, {fused.ENV: value}) is True
    assert fused.resolve(NAME, {fused.ENV: value}) is untilize_rows


def test_registry_rejects_unknown_names(expect_error):
    with expect_error(ValueError, match="QWEN38_FUSED names unregistered fused kernels \\['nope_kernel'\\]"):
        fused.enabled_names({fused.ENV: f"{NAME},nope_kernel"})
    with expect_error(ValueError, match="QWEN38_FUSED_OFF names unregistered fused kernels \\['nope_kernel'\\]"):
        fused.enabled_names({fused.OFF_ENV: "nope_kernel"})
    with expect_error(KeyError, match="no fused kernel 'nope'"):
        fused.kernel("nope")


def test_registry_validates_entries(expect_error):
    with expect_error(ValueError, match="registered twice"):
        registry.register(fused.kernel(NAME))
    with expect_error(ValueError, match="tolerance must be one of"):
        registry.FusedKernel("x_kernel", "y", "loose", untilize_rows, untilize_rows_composed)
    with expect_error(ValueError, match="name must match"):
        registry.FusedKernel("Router-Tail", "y", registry.BITWISE, untilize_rows, untilize_rows_composed)


def _tensor(shape, padded):
    return SimpleNamespace(shape=shape, padded_shape=padded)


@pytest.mark.parametrize("rows", [1, 5, 32])
def test_rows_contract_accepts_one_row_tile(rows):
    assert fp.rows_of(_tensor((1, 1, rows, 512), (1, 1, 32, 512))) == rows
    assert fp.tile_width_of(_tensor((1, 1, rows, 512), (1, 1, 32, 512))) == 512


@pytest.mark.parametrize(
    "shape, padded",
    [
        ((1, 1, 33, 512), (1, 1, 64, 512)),
        ((1, 1, 0, 512), (1, 1, 32, 512)),
        ((1, 4, 1, 640), (1, 4, 32, 640)),
        ((1, 32, 512), (1, 32, 512)),
    ],
)
def test_rows_contract_rejects_other_shapes(expect_error, shape, padded):
    with expect_error(ValueError, match="one row tile"):
        fp.rows_of(_tensor(shape, padded))


def test_tile_width_rejects_partial_tiles(expect_error):
    with expect_error(ValueError, match="whole tiles"):
        fp.tile_width_of(_tensor((1, 1, 1, 10), (1, 1, 32, 32)))


def test_split_work_covers_every_unit_once_in_linear_core_order(expect_error):
    mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
    work = fp.split_work(16, mesh)
    assert [w.count for w in work] == [1] * 16
    assert [(w.core.x, w.core.y) for w in work[:11]] == [(0, y) for y in range(10)] + [(1, 0)]
    work = fp.split_work(250, mesh)
    assert len(work) == 110 and sum(w.count for w in work) == 250
    assert [w.start for w in work] == [sum(v.count for v in work[:i]) for i in range(len(work))]
    assert {w.count for w in work} == {2, 3}
    with expect_error(ValueError, match="nothing to split"):
        fp.split_work(0, mesh)


def test_kernel_source_convention(expect_error):
    assert fp.kernel_source(NAME, "writer_rows.cpp") == f"{fp.KERNEL_ROOT}/{NAME}/kernels/writer_rows.cpp"
    assert (fp.REPO_ROOT / fp.KERNEL_ROOT / "program.py").is_file()
    with expect_error(FileNotFoundError):
        fp.kernel_source(NAME, "missing.cpp")


def test_writer_kernel_argument_contract_matches_the_python_side():
    source = (fp.REPO_ROOT / fp.kernel_source(NAME, "writer_rows.cpp")).read_text()
    assert "get_compile_time_arg_val(0)" in source and "elem_bytes" in source
    assert "get_compile_time_arg_val(1)" in source and "rows" in source
    assert "TensorAccessorArgs<2>()" in source
    assert re.search(
        r"get_arg_val<uint32_t>\(0\).*\n.*get_arg_val<uint32_t>\(1\).*\n.*get_arg_val<uint32_t>\(2\)", source
    )


def test_byte_tables_are_consistent():
    for dtype, elem in fp.ELEMENT_BYTES.items():
        assert fp.TILE_BYTES[dtype] == elem * fp.TILE * fp.TILE
    assert fp.TILE == 32 and fp.FACE == 16 and fp.ROWS_MAX == 32
    assert ttnn.TILE_SIZE == fp.TILE


def _reachable_sources(module: types.ModuleType, entry) -> dict[str, str]:
    """Module-level functions reachable from ``entry`` by direct reference (``name(`` in the same module, or
    ``alias.name(`` through a sibling fused module), with their sources."""

    def functions(mod):
        return {n: o for n, o in vars(mod).items() if inspect.isfunction(o) and o.__module__ == mod.__name__}

    def siblings(mod):
        return {
            n: o
            for n, o in vars(mod).items()
            if isinstance(o, types.ModuleType) and o.__name__.startswith(fused.__name__)
        }

    out: dict[str, str] = {}
    todo = [(module, entry.__name__)]
    while todo:
        mod, name = todo.pop()
        key = f"{mod.__name__}.{name}"
        funcs = functions(mod)
        if key in out or name not in funcs:
            continue
        out[key] = src = inspect.getsource(funcs[name])
        todo += [(mod, m) for m in funcs if re.search(rf"(?<![.\w]){m}\(", src)]
        for alias, sib in siblings(mod).items():
            todo += [(sib, m) for m in functions(sib) if f"{alias}.{m}(" in src]
    return out


def test_model_level_paths_read_mesh_tensors_per_device():
    """Every ``ttnn.to_torch`` on a path a model-level entry can reach reads one device's shard (``for x in
    ttnn.get_device_tensors(...)``) or passes a mesh composer: the 4-chip acceptance ON run of final_mixer at 10cdff7d8a
    died in gr_read.noc_map on ``ttnn.to_torch(mesh tensor)`` -- a path the single-chip tests never execute."""

    entries = []
    for kernel in registry.kernels().values():
        module = sys.modules[kernel.fused.__module__]
        entries.append((module, kernel.fused))
        entries += [
            (module, o)
            for n, o in vars(module).items()
            if inspect.isfunction(o) and n.endswith("_fused") and o is not kernel.fused
        ]
    assert entries
    checked = 0
    for module, entry in entries:
        for key, src in _reachable_sources(module, entry).items():
            for match in re.finditer(r"ttnn\.to_torch\((\w+)", src):
                checked += 1
                arg = match.group(1)
                call = src[match.start() : src.find("\n", match.start())]
                per_device = re.search(rf"for {arg} in ttnn\.get_device_tensors\(", src)
                assert per_device or "mesh_composer=" in call, (key, call.strip())
    assert checked >= 2  # final_mixer.norm_scale_rows and gr_read.noc_map at least
