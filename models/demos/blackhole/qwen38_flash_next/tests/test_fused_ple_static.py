# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The fused PLE without a device: registry entry, the hook pinned in ttnn/ple.py, the kernels' argument contracts and
the LLK pins that carry the chain's rounding points (the accurate fp32 fold, clamp/sqrt/sigmoid forms, the FUSE_GAMMA
FPU multiply, the ternary MAC sequence, the FPU delta add), and the manifest."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import ple
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp

HERE = Path(__file__).resolve().parents[1]
PLE_SOURCE = (HERE / "ttnn" / "ple.py").read_text()
K = {
    name: (fp.REPO_ROOT / getattr(ple, name)).read_text()
    for name in (
        "GATE_COMPUTE",
        "GATE_READER",
        "GATE_WRITER",
        "NORM_COMPUTE",
        "CONV_COMPUTE",
        "CONV_READER",
        "CONV_WRITER",
    )
}


def _ct_args(source):
    return sorted({int(i) for i in re.findall(r"get_compile_time_arg_val\((\d+)\)", source)})


def test_registered_bitwise_default_by_the_list():
    entry = fused.kernel("ple")
    assert entry.tolerance == fused.BITWISE and entry.gate is None
    assert entry.default_on is ("ple" in fused.DEFAULT_ON)  # the one list decides; ple joined it 2026-09-15
    assert entry.fused is ple.ple_fused and entry.composed is ple.ple_composed
    default = ple.ple_fused if "ple" in fused.DEFAULT_ON else ple.ple_composed  # the list decides
    assert fused.resolve("ple", {}) is default and fused.resolve("ple", {fused.ENV: "ple"}) is ple.ple_fused
    assert fused.resolve("ple", {fused.OFF_ENV: "ple"}) is ple.ple_composed


def test_hook_is_pinned():
    assert "    _fused_forward_prepared = None  # QWEN38_FUSED=ple binds" in PLE_SOURCE
    assert 'if fused_kernels.enabled("ple"):' in PLE_SOURCE
    assert 'self._fused_forward_prepared = functools.partial(fused_kernels.kernel("ple").fused, self)' in PLE_SOURCE
    body = PLE_SOURCE[PLE_SOURCE.index("    def forward_prepared(") :]
    branch = "        if self._fused_forward_prepared is not None:\n            return self._fused_forward_prepared(residual, prepared, state)\n"
    assert branch in body and body.index(branch) < body.index(
        'self._validate_residual(residual, label="PLE residual input")'
    )


def test_kernel_contracts():
    assert _ct_args(K["GATE_COMPUTE"]) == [0, 1, 2] and K["GATE_READER"].count("TensorAccessorArgs<") == 5
    assert _ct_args(K["GATE_READER"]) == [0, 1] and _ct_args(K["GATE_WRITER"]) == [0, 1]
    assert K["GATE_WRITER"].count("TensorAccessorArgs<") == 3
    assert _ct_args(K["NORM_COMPUTE"]) == [0, 1, 2, 3]
    assert _ct_args(K["CONV_COMPUTE"]) == [0, 1, 2, 3] and K["CONV_READER"].count("TensorAccessorArgs<") == 16
    assert _ct_args(K["CONV_READER"]) == [0, 1, 2] and _ct_args(K["CONV_WRITER"]) == [0, 1]
    assert (
        "get_arg_val<uint32_t>(16)" in K["CONV_READER"] and "get_arg_val<uint32_t>(1)" in K["CONV_WRITER"]
    )  # column block
    assert K["CONV_WRITER"].count("TensorAccessorArgs<") == 2
    assert sorted({int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d+)\)", K["CONV_READER"])}) == list(range(18))
    assert sorted({int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d+)\)", K["GATE_READER"])}) == list(range(7))
    assert sorted({int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d+)\)", K["GATE_WRITER"])}) == list(range(4))
    assert sorted({int(i) for i in re.findall(r"get_arg_val<uint32_t>\((\d+)\)", K["CONV_WRITER"])}) == [0, 1, 2]
    assert (
        "add_tiles(c_resid, c_rows, b, b, 0);" in K["CONV_COMPUTE"]
    )  # the layer's add(residual, delta), FPU 16-bit dest


def test_gate_kernel_carries_the_chain_ops_in_order():
    c = K["GATE_COMPUTE"]
    order = [
        "mul_binary_tile(0, 1, 0);",
        "ReduceFp32Mode::Accurate>(",
        "mul_binary_tile(0, 1, 0);  // gate",
        "abs_tile(0);",
        "clamp_tile(0, MIN_BITS, MAX_BITS);",
        "sqrt_tile<false>(0);",
        "sign_tile(1);",
        "sigmoid_tile<VectorMode::RC, 0u>(0);",
        "unary_bcast<BroadcastType::COL>(c_coef, 0, 0);",
    ]
    positions = [c.index(o) for o in order]
    assert positions == sorted(positions), order
    assert "constexpr uint32_t MIN_BITS = 0x358637bdu;" in c and "constexpr uint32_t MAX_BITS = 0x7f7fffffu;" in c
    assert "ReduceInputBlockShape::of(1, Wt, 1)" in c and "reduce_tile<" not in c
    assert "REDUCE_ROW>(1.0f);" in K["GATE_READER"]
    assert "unpack_to_dest_fp32=(2, 4, 5, 6, 7, 8)" in inspect.getsource(ple.gate)


def test_norm_kernel_is_the_fuse_gamma_form():
    n = K["NORM_COMPUTE"]
    assert (
        "mul_bcast_rows_init(c_unit, c_gamma);" in n
        and "mul_tiles_bcast_rows(c_unit, c_gamma, wt + wtr, wt + wtr, wtr);" in n
    )
    assert "mul_binary_tile" not in n  # gr_read's separate SFPU gamma multiply is the GR chain's form, not the PLE's
    assert "mul_tiles_bcast_cols(c_res, c_recip, wtr, 0, wtr);" in n  # the unit as rmsnorm_post_allgather's


def test_conv_kernel_is_the_chain_sequence():
    c = K["CONV_COMPUTE"]
    assert "mul_binary_tile(0, 1, 0);" in c  # tap 0: binary_ng's SFPU multiply
    assert "mac_tile<DataFormat::Float16_b>(0, 1, 2, 0);" in c  # ttnn.mac on bf16: the SFPU ternary kernel, a*b + c
    assert "mac_tile_init<DataFormat::Float16_b>();" in c and "MAC_FORM == 0" in c
    assert "silu_tile<false>(0);" in c and "add_tiles(c_gated, c_silu, t, 0, 0);" in c
    assert "fp32_dest=False" in inspect.getsource(
        ple.conv
    )  # every op of the stage packs bf16: 16-bit dest as the chain
    assert ple.MAC_FORM in (0, 1)
    r = K["CONV_READER"]
    assert r.index("stream(noc, TensorAccessor(a8") < r.index("shift_row(noc, c1, c0, first);")  # inputs streamed first
    assert r.count("shift_row(noc, ") == 9  # conv[k] <- conv[k+1] for k = 0..7, then conv[8] <- normalized


def test_layer_hook_is_pinned():
    layer_source = (HERE / "ttnn" / "layer.py").read_text()
    assert "    _fused_apply_ple = None  # QWEN38_FUSED=ple binds" in layer_source
    assert 'if ple is not None and fused_kernels.enabled("ple"):' in layer_source
    assert "self._fused_apply_ple = functools.partial(fused_ple.ple_layer_fused, self)" in layer_source
    body = layer_source[layer_source.index("    def _apply_ple(") :]
    branch = "        if self._fused_apply_ple is not None:\n            return self._fused_apply_ple(\n"
    assert branch in body and body.index(branch) < body.index("if self.ple is None:")
    source = inspect.getsource(ple.ple_layer_fused)
    assert (
        "ttnn.permute(residual, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)" in source
    )  # the chain's input permute
    assert "inject_residual=residual" in source and "type(layer)._apply_ple(" in source


def _class_names(source: str, class_name: str) -> set[str]:
    """Methods, class attributes and ``self.<name> =`` assignments of one class in a chain source file."""

    start = re.search(rf"\nclass {class_name}\b[(:]", source).start()
    body = source[start:]
    body = body[: body.find("\nclass ", 1) if body.find("\nclass ", 1) > 0 else len(body)]
    names = set(re.findall(r"\n    def ([A-Za-z_][A-Za-z_0-9]*)\(", body))
    names |= set(re.findall(r"\n    ([A-Za-z_][A-Za-z_0-9]*)\s*[:=]", body))
    names |= set(re.findall(r"self\.([A-Za-z_][A-Za-z_0-9]*)\s*=", body))
    return names


def test_every_chain_attribute_the_fused_ple_uses_exists():
    """The model-level glue is only exercised on four chips (the 4-chip acceptance ON run at 399c58088e failed on
    ``module._validate_residual_rows``, a name the PLE class does not have): every ``module.<name>`` must be a member of
    Qwen38TTNNPLE and every ``layer.<name>`` of Qwen38TTNNDecoderLayer."""

    fused_source = inspect.getsource(ple)
    layer_source = (HERE / "ttnn" / "layer.py").read_text()
    for prefix, names in (
        ("module", _class_names(PLE_SOURCE, "Qwen38TTNNPLE")),
        ("layer", _class_names(layer_source, "Qwen38TTNNDecoderLayer")),
    ):
        used = set(re.findall(rf"\b{prefix}\.([A-Za-z_][A-Za-z_0-9]*)", fused_source))
        assert used, prefix
        assert used <= names, (prefix, sorted(used - names))
    assert 'module._validate_residual(gated, label="PLE gated value")' in fused_source


def test_manifest_lists_the_files():
    manifest_path = HERE / "tools" / "release" / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("tools/release/manifest.json is not in this tree (the public tree ships without tools/release/)")
    manifest = json.loads(manifest_path.read_text())["public"]
    for path in ["tests/test_fused_ple_static.py", "ttnn/fused/ple/__init__.py"] + [
        f"ttnn/fused/ple/kernels/{k}"
        for k in (
            "gate_compute.cpp",
            "gate_reader.cpp",
            "gate_writer.cpp",
            "norm_gamma_rows_compute.cpp",
            "conv_compute.cpp",
            "conv_reader.cpp",
            "conv_writer.cpp",
        )
    ]:
        assert path in manifest, path


def test_lane_hook_and_lane_forms_are_pinned():
    """forward_prepared_lanes binds the lane forms with the same switch; the lane programs are the 1-row kernels on
    the lane's pages (stats / normalize / gate one core per lane; the conv per column block inside one lane, its taps
    the block's columns); the layer's inject_lanes keeps its permutes and the residual add."""

    assert "    _fused_forward_prepared_lanes = None  # its lane form" in PLE_SOURCE
    assert (
        "self._fused_forward_prepared_lanes = functools.partial(fused_kernels.ple.ple_lanes_fused, self)" in PLE_SOURCE
    )
    body = PLE_SOURCE[PLE_SOURCE.index("    def forward_prepared_lanes(") :]
    branch = (
        "        if self._fused_forward_prepared_lanes is not None:\n"
        "            return self._fused_forward_prepared_lanes(residual_lanes, prepared, lanes_state)\n"
    )
    assert branch in body[: body.index("    def inject_lanes(")]
    inject = PLE_SOURCE[PLE_SOURCE.index("    def inject_lanes(") :]
    assert (
        "ttnn.permute(residual_lanes, (0, 2, 1, 3)" in inject
        and "_fused" not in inject[: inject.index("return injected")]
    )
    gate_reader = (fp.REPO_ROOT / ple.GATE_READER).read_text()
    assert "const uint32_t first_kq = get_arg_val<uint32_t>(5);" in gate_reader
    assert "const uint32_t first_v = get_arg_val<uint32_t>(6);" in gate_reader
    assert "const uint32_t first = get_arg_val<uint32_t>(3);" in (fp.REPO_ROOT / ple.GATE_WRITER).read_text()
    conv_reader = (fp.REPO_ROOT / ple.CONV_READER).read_text()
    assert "const uint32_t tap_first = get_arg_val<uint32_t>(17);" in conv_reader
    assert conv_reader.count(", tap_first);") == 4  # the four taps
    source = inspect.getsource(ple.conv_lanes)
    assert "per_core = next(T for T in (1, 2, 4, 5, 10, 20) if tiles // T <= grid.x * grid.y)" in source
    assert "(w.start * per_core) % LOCAL_TILES" in source
    assert "[0, 0])]" in inspect.getsource(ple.gate)  # the 1-row gate reads pages 0..
    lanes_body = inspect.getsource(ple.ple_lanes_fused)
    for call in (
        "module._project_rows(embedding_tile, lanes)",
        "_group_norm_lanes(module, key,",
        "gate_lanes(key_global, query_global, value, lanes)",
        "conv_lanes(lanes_state.conv, normalized, module.weights.conv_taps, gated, lanes, shift=True)",
        "lanes_state.token_contexts = prepared.next_contexts",
    ):
        assert call in lanes_body, call
