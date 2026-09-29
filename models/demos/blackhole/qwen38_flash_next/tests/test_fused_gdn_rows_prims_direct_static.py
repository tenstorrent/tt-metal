# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contract of ``gdn_rows_prims_direct``: the registry entry (opt-in, BITWISE, off by default), the GDN
layer's one resolved dispatch, the shape-only admission, the transcription pinned against the composite's runtime
source it was read from, and the call-argument pin: on a recording ttnn the prims form hands the two prims exactly what
the composite's own call receives (the composite's relayout ops in its order, its constants, chunk size, scale and
state view) and frees only its own intermediates."""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import fused
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_rows_prims_direct as module

MODEL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MODEL_DIR.parents[3]
COMPOSITE = REPO_ROOT / "ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/chunk_gated_delta_rule.cpp"
GDN_SOURCE = MODEL_DIR / "ttnn" / "gdn.py"
NAME = module.NAME
HEADS, HEAD_DIM, TILE = module.HEADS, module.HEAD_DIM, module.TILE
SCALE = HEAD_DIM**-0.5


# --------------------------------------------------------------------------- the registry entry and the layer's dispatch


def test_registry_entry_is_opt_in_and_bitwise():
    entry = fused.kernel(NAME)
    assert entry.tolerance == fused.BITWISE and entry.default_on is False and NAME not in fused.DEFAULT_ON
    assert entry.fused is module.chunk_rows_prims and entry.composed is module.chunk_rows_composite
    assert entry.admits is module.admits and entry.gate is None and entry.component_proof is None
    assert "chunk_gated_delta_rule" in entry.replaces and "chunk_gdn_prep" in entry.replaces
    # off by default: the composed chain; on: the admitted dispatcher; the off switch wins
    assert fused.resolve_admitted(NAME, {}) is module.chunk_rows_composite
    assert fused.resolve_admitted(NAME, {fused.OFF_ENV: NAME}) is module.chunk_rows_composite
    on = fused.resolve_admitted(NAME, {fused.ENV: NAME})
    assert isinstance(on, fused.AdmittedStep)
    assert on.fused is module.chunk_rows_prims and on.composed is module.chunk_rows_composite
    assert on.admits is module.admits
    assert fused.resolve_admitted(NAME, {fused.ENV: NAME, fused.OFF_ENV: NAME}) is module.chunk_rows_composite
    assert fused.resolve_admitted(NAME, {fused.ENV: "all"}).fused is module.chunk_rows_prims
    assert fused.enabled(NAME, {}) is False and fused.enabled(NAME, {fused.ENV: NAME}) is True
    assert module.SCALE == SCALE and module.TILE == 32 and module.HEADS == 12 and module.HEAD_DIM == 128


def _gdn_methods() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(GDN_SOURCE.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen38TTNNGDN")
    return {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}


def test_the_layer_resolves_the_recurrence_once_and_dispatches_through_it(monkeypatch):
    """``__init__`` resolves the admitted kernel once; ``_chunk_rows`` hands the kernel's views to it (one call, the
    eight arguments); the composite call lives in ``_chunk_rows_composite`` alone; the composed chain forwards there."""

    methods = _gdn_methods()
    init = ast.get_source_segment(GDN_SOURCE.read_text(encoding="utf-8"), methods["__init__"])
    assert f'self._rows_chunk = fused.resolve_admitted("{NAME}")' in init
    getter = ast.get_source_segment(GDN_SOURCE.read_text(encoding="utf-8"), methods["_chunk_rows_kernel"])
    assert 'self.__dict__.get("_rows_chunk")' in getter and f'fused.resolve_admitted("{NAME}")' in getter
    chunk_calls = [ast.unparse(node.func) for node in ast.walk(methods["_chunk_rows"]) if isinstance(node, ast.Call)]
    assert chunk_calls.count("self._chunk_rows_kernel") == 1 and chunk_calls.count("kernel") == 1
    assert "ttnn.transformer.chunk_gated_delta_rule" not in chunk_calls
    dispatch = next(
        n for n in ast.walk(methods["_chunk_rows"]) if isinstance(n, ast.Call) and ast.unparse(n.func) == "kernel"
    )
    assert [ast.unparse(a) for a in dispatch.args] == [
        "self",
        "q_rows",
        "k_rows",
        "v_rows",
        "g_rows",
        "beta_rows",
        "initial_state",
        "constants",
    ]
    assert [a.arg for a in methods["_chunk_rows_composite"].args.args] == [
        "self",
        "q_rows",
        "k_rows",
        "v_rows",
        "g_rows",
        "beta_rows",
        "initial_state",
        "constants",
    ]
    composite_calls = [
        ast.unparse(node.func) for node in ast.walk(methods["_chunk_rows_composite"]) if isinstance(node, ast.Call)
    ]
    assert composite_calls == ["ttnn.transformer.chunk_gated_delta_rule"]
    # the composed chain is the layer's method, so the composite keeps the layer's ttnn (the fakes of the rows tests)
    assert "gdn._chunk_rows_composite(q_rows, k_rows, v_rows, g_rows, beta_rows, initial_state, constants)" in (
        inspect.getsource(module.chunk_rows_composite)
    )
    assert inspect.signature(module.chunk_rows_prims) == inspect.signature(module.chunk_rows_composite)
    assert list(inspect.signature(module.admits).parameters) == list(
        inspect.signature(module.chunk_rows_prims).parameters
    )
    # a layer built without __init__ (the rows tests' fakes) resolves lazily to the same choice as the constructor
    monkeypatch.delenv(fused.ENV, raising=False)
    monkeypatch.delenv(fused.OFF_ENV, raising=False)
    gdn = object.__new__(gdn_module.Qwen38TTNNGDN)
    assert gdn._chunk_rows_kernel() is module.chunk_rows_composite and gdn._rows_chunk is module.chunk_rows_composite
    monkeypatch.setenv(fused.ENV, NAME)
    gdn = object.__new__(gdn_module.Qwen38TTNNGDN)
    kernel = gdn._chunk_rows_kernel()
    assert isinstance(kernel, fused.AdmittedStep) and kernel.fused is module.chunk_rows_prims
    assert gdn._chunk_rows_kernel() is kernel  # resolved once


# --------------------------------------------------------------------------- admission


class _Fake:
    def __init__(self, shape, dtype, layout=ttnn.TILE_LAYOUT):
        self.shape, self.dtype, self.layout = tuple(shape), dtype, layout


def _served() -> dict[str, Any]:
    return {
        "q_rows": _Fake((1, TILE, HEADS, HEAD_DIM), ttnn.bfloat16),
        "k_rows": _Fake((1, TILE, HEADS, HEAD_DIM), ttnn.bfloat16),
        "v_rows": _Fake((1, TILE, HEADS * HEAD_DIM), ttnn.bfloat16),
        "g_rows": _Fake((1, TILE, HEADS), ttnn.float32),
        "beta_rows": _Fake((1, TILE, HEADS), ttnn.float32),
        "initial_state": _Fake((1, HEADS, HEAD_DIM, HEAD_DIM), ttnn.float32),
        "constants": SimpleNamespace(eye=object(), tril=object(), ones=object(), masks=object()),
    }


def test_admission_is_the_head_major_rows_form_at_one_tile():
    served = _served()
    assert module.admits(None, **served) is True
    # the slab's flat q/k (the composite's in-kernel norm), the 128-row long chunk, a bf16 or lanes state, a row-major
    # tensor, a v in head layout, a missing constant tile: the composite
    for name, tensor in (
        ("q_rows", _Fake((1, 1, TILE, 4 * HEAD_DIM), ttnn.bfloat16)),
        ("k_rows", _Fake((1, 1, TILE, 4 * HEAD_DIM), ttnn.bfloat16)),
        ("q_rows", _Fake((1, 4 * TILE, HEADS, HEAD_DIM), ttnn.bfloat16)),
        ("v_rows", _Fake((1, 4 * TILE, HEADS * HEAD_DIM), ttnn.bfloat16)),
        ("v_rows", _Fake((1, TILE, HEADS, HEAD_DIM), ttnn.bfloat16)),
        ("g_rows", _Fake((1, TILE, HEADS), ttnn.bfloat16)),
        ("beta_rows", _Fake((1, 4 * TILE, HEADS), ttnn.float32)),
        ("initial_state", _Fake((1, HEADS, HEAD_DIM, HEAD_DIM), ttnn.bfloat16)),
        ("initial_state", _Fake((2, HEADS, HEAD_DIM, HEAD_DIM), ttnn.float32)),
        ("q_rows", _Fake((1, TILE, HEADS, HEAD_DIM), ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)),
        ("constants", SimpleNamespace(eye=object(), tril=object(), ones=object(), masks=None)),
        ("constants", object()),
        ("q_rows", object()),
    ):
        assert module.admits(None, **{**served, name: tensor}) is False, name
    # a host fake whose dtype is not ttnn's (the rows tests' torch-backed tensors) takes the composite
    fake_dtype = _Fake((1, TILE, HEADS, HEAD_DIM), "bfloat16")
    assert module.admits(None, **{**served, "q_rows": fake_dtype}) is False
    # the dispatcher takes both branches on the same rule
    calls = []
    probe = fused.FusedKernel(
        "probe_rows_chunk",
        "a probe",
        fused.BITWISE,
        fused=lambda *args: calls.append("fused"),
        composed=lambda *args: calls.append("composed"),
        admits=module.admits,
    )
    dispatch = fused.AdmittedStep(probe)
    dispatch(None, *{**served, "q_rows": fake_dtype}.values())
    dispatch(None, *served.values())
    assert calls == ["composed", "fused"]


# --------------------------------------------------------------------------- the transcription against its source


def test_the_transcription_is_the_composites_source():
    """The C++ lines the Python form transcribes (chunk_gated_delta_rule.cpp) are still there in that form; a change
    to the composite's relayout or its prim calls must be carried into ``chunk_rows_prims`` and re-proven."""

    source = COMPOSITE.read_text(encoding="utf-8")
    # head_split_tile / headvec_split_tile: a permute on TILE, then the head-major view
    assert "t = ttnn::permute(t, ttnn::SmallVector<int64_t>{0, 2, 1, 3});  // [B, Hh, T, D] TILE" in source
    assert "t = ttnn::reshape(t, ttnn::Shape({B * Hh, T, D}));             // [BH, T, D] TILE" in source
    assert "t = ttnn::permute(t, ttnn::SmallVector<int64_t>{0, 2, 1});  // [B, Hn, T] TILE" in source
    assert "t = ttnn::reshape(t, ttnn::Shape({B * Hn, T}));             // [BH, T] TILE" in source
    # the flat v passes through (bf16 already); no GQA expand at G == 1; the scale fold when q/k are not flat
    assert (
        "ttnn::Tensor v = flat_v ? (v_in.dtype() != DataType::BFLOAT16 ? ttnn::typecast(v_in, DataType::BFLOAT16) : v_in)"
        in source
    )
    assert "if (G > 1 && !flat_qk) {" in source
    assert "const bool qk_norm = flat_qk && (C == 32);" in source
    assert "if (!qk_norm) {\n        q = ttnn::multiply(q, scale);\n    }" in source
    # no padding at T == C; the per-chunk views; g / beta to one column tile per head; the state view
    assert "const uint32_t pad = (C - (T % C)) % C;" in source
    assert "return ttnn::reshape(t, ttnn::Shape({BH, NC, C, D}));" in source
    assert "ttnn::Tensor g_c = ttnn::reshape(g, ttnn::Shape({BH, NC, C, 1}));" in source
    assert "ttnn::Tensor beta_c = ttnn::reshape(beta, ttnn::Shape({BH, NC, C, 1}));" in source
    assert "s0 = ttnn::reshape(s, ttnn::Shape({BH, K, V}));" in source
    # the composite's defaults the bindings reproduce when memory_config / compute_kernel_config are left unset
    assert "const auto out_mem = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);" in source
    assert (
        "MathFidelity::HiFi4,\n        /*default_approx_mode=*/false,\n        /*default_fp32_acc=*/true,\n        /*default_l1_acc=*/false);"
        in source
    )
    # the prep's positional order (the binding names them: v_flat, HV, qk_norm, scale, qk_flat, Hk) and the scan's
    prep = re.search(r"ttnn::prim::chunk_gdn_prep\((.*?)\);", source, re.S).group(1)
    assert [a.strip() for a in prep.split(",")] == [
        "q_c", "k_c", "v_c", "g_c", "beta_c", "eye_c", "tril_c", "ones_c", "masks_c", "C", "out_mem", "kernel_cfg",
        "flat_v", "HV", "qk_norm", "scale", "flat_qk", "H",
    ]  # fmt: skip
    scan = re.search(r"ttnn::prim::chunk_gdn_scan\((.*?)\);", source, re.S).group(1)
    assert [a.strip() for a in scan.split(",")] == [
        *(f"prep[{i}]" for i in range(7)), "s0", "C", "output_final_state", "out_mem", "kernel_cfg", "use_mcast",
    ]  # fmt: skip
    # the head-major output at pad == 0 and the final state: metadata views
    assert "o = ttnn::reshape(o_c, ttnn::Shape({BH, L, V}));  // [BH,T,V] TILE, metadata-only" in source
    assert "final_opt = ttnn::reshape(final_state, ttnn::Shape({B, HV, K, V}));" in source
    # and the Python form's op order is that source's, one call per C++ op (the two prims by their bound names)
    tree = ast.parse(inspect.getsource(module.chunk_rows_prims))
    ops = [
        ast.unparse(node.func).removeprefix("ttnn.")
        for node in sorted(
            (n for n in ast.walk(tree) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset)
        )
        if ast.unparse(node.func).startswith("ttnn.")
    ]
    assert ops == [
        "permute", "reshape",  # q: head_split_tile
        "permute", "reshape",  # k
        "permute", "reshape",  # g: headvec_split_tile
        "permute", "reshape",  # beta
        "multiply", "deallocate",  # q = multiply(q, scale): the permuted q released by the reassignment
        "reshape", "reshape",  # q_c, k_c: to_chunks_tile
        "reshape", "reshape",  # g_c, beta_c: [BH, NC, C, 1]
        "reshape",  # s0
        "prim.chunk_gdn_prep",
        "prim.chunk_gdn_scan",
        "reshape", "reshape",  # o [BH, T, V], final state [B, HV, K, V]
    ]  # fmt: skip


# --------------------------------------------------------------------------- the call-argument pin on a recording ttnn


@dataclass(frozen=True)
class _Rec:
    """A recorded op result: the op and its arguments (leaves are the caller's own objects)."""

    op: str
    args: tuple

    def is_allocated(self) -> bool:
        return True


class _Prims:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def chunk_gdn_prep(self, *args, **kwargs):
        self.calls.append(("chunk_gdn_prep", args, kwargs))
        return [_Rec(f"prep{i}", (args, tuple(sorted(kwargs)))) for i in range(7)]

    def chunk_gdn_scan(self, *args, **kwargs):
        self.calls.append(("chunk_gdn_scan", args, kwargs))
        return [_Rec("o_c", (args, tuple(sorted(kwargs)))), _Rec("final_c", (args, tuple(sorted(kwargs))))]


def _recording_ttnn(prims: _Prims, freed: list) -> SimpleNamespace:
    return SimpleNamespace(
        TILE_SIZE=32,
        TILE_LAYOUT=ttnn.TILE_LAYOUT,
        bfloat16=ttnn.bfloat16,
        float32=ttnn.float32,
        permute=lambda t, dims: _Rec("permute", (t, tuple(dims))),
        reshape=lambda t, shape: _Rec("reshape", (t, tuple(shape))),
        multiply=lambda t, scalar: _Rec("multiply", (t, scalar)),
        deallocate=lambda t: freed.append(t),
        prim=prims,
    )


def test_prims_form_hands_the_prims_the_composites_arguments(monkeypatch):
    """On a recording ttnn the prims form's prep receives the composite's own relayout of q / k / g / beta (the same ops
    in the same order), the untouched flat v, the four constant tiles by identity, the composite's chunk size and
    scale, ``v_flat`` with ``HV`` and ``qk_norm`` off (head-major q/k), the bindings' defaults for memory_config and
    compute_kernel_config (absent); the scan receives the seven prep outputs in order, the ``[BH, K, V]`` view of the
    state, the same chunk size and ``output_final_state``; the results are the composite's views; and the form frees
    exactly its own intermediates, never a caller's tensor or a view of the state."""

    prims, freed = _Prims(), []
    monkeypatch.setattr(module, "ttnn", _recording_ttnn(prims, freed))
    q, k, v, g, beta, state = "q_rows", "k_rows", "v_rows", "g_rows", "beta_rows", "initial_state"
    constants = SimpleNamespace(eye="eye", tril="tril", ones="ones", masks="masks")
    output, final_state = module.chunk_rows_prims(None, q, k, v, g, beta, state, constants)

    q_perm = _Rec("permute", (q, (0, 2, 1, 3)))
    k_perm = _Rec("permute", (k, (0, 2, 1, 3)))
    g_perm = _Rec("permute", (g, (0, 2, 1)))
    beta_perm = _Rec("permute", (beta, (0, 2, 1)))
    q_scaled = _Rec("multiply", (_Rec("reshape", (q_perm, (HEADS, TILE, HEAD_DIM))), SCALE))
    q_c = _Rec("reshape", (q_scaled, (HEADS, 1, TILE, HEAD_DIM)))
    k_c = _Rec("reshape", (_Rec("reshape", (k_perm, (HEADS, TILE, HEAD_DIM))), (HEADS, 1, TILE, HEAD_DIM)))
    g_c = _Rec("reshape", (_Rec("reshape", (g_perm, (HEADS, TILE))), (HEADS, 1, TILE, 1)))
    beta_c = _Rec("reshape", (_Rec("reshape", (beta_perm, (HEADS, TILE))), (HEADS, 1, TILE, 1)))
    s0 = _Rec("reshape", (state, (HEADS, HEAD_DIM, HEAD_DIM)))

    assert [name for name, _, _ in prims.calls] == ["chunk_gdn_prep", "chunk_gdn_scan"]
    _, prep_args, prep_kwargs = prims.calls[0]
    assert prep_args == (q_c, k_c, v, g_c, beta_c)
    assert prep_args[2] is v  # the flat v as it is (rank 3, bf16: the composite passes it through)
    assert prep_kwargs == {
        "eye": "eye",
        "tril": "tril",
        "ones": "ones",
        "masks": "masks",
        "chunk_size": 32,
        "scale": SCALE,
        "v_flat": True,
        "HV": HEADS,
        "qk_flat": False,
        "Hk": HEADS,
        "qk_norm": False,
    }
    assert type(prep_kwargs["scale"]) is float and prep_kwargs["scale"] == 128**-0.5
    prep_out = prims.chunk_gdn_prep(*prep_args, **prep_kwargs)  # the same seven records
    prims.calls.pop()
    _, scan_args, scan_kwargs = prims.calls[1]
    # the seven hand-offs in the order the prep returned them, then the [BH, K, V] state view, all positional (the
    # spelling of gdn_prefill_rows.chunk_prims); the two keywords are the composite's
    assert scan_args == (*prep_out, s0) and len(scan_args) == 8
    assert scan_kwargs == {"chunk_size": 32, "output_final_state": True}
    assert output == _Rec("reshape", (_Rec("o_c", (scan_args, tuple(sorted(scan_kwargs)))), (HEADS, TILE, HEAD_DIM)))
    assert final_state == _Rec(
        "reshape", (_Rec("final_c", (scan_args, tuple(sorted(scan_kwargs)))), (1, HEADS, HEAD_DIM, HEAD_DIM))
    )
    # freed: the four permutes, the scaled q, the two column relayouts and the seven prep outputs; nothing else
    assert freed[0] == q_perm  # released right after the scale multiply, as the composite's reassignment does
    assert freed[1:] == [k_perm, q_scaled, g_perm, beta_perm, g_c, beta_c, *prep_out]
    assert not any(isinstance(t, str) for t in freed) and s0 not in freed and output not in freed
    assert final_state not in freed and prep_args[2] not in freed


def test_composite_call_takes_the_same_constants_chunk_size_scale_and_state(monkeypatch):
    """The composed chain's one call (``_chunk_rows_composite`` through the layer's ttnn): the same five tensors in the
    same order, the constants by identity, ``chunk_size`` 32 and ``scale`` the prims form's, the state the prims form
    views, ``output_final_state`` and the head-major output."""

    calls = []

    def composite(q, k, v, g, beta, **kwargs):
        calls.append(((q, k, v, g, beta), kwargs))
        return "o", "final"

    monkeypatch.setattr(
        gdn_module, "ttnn", SimpleNamespace(transformer=SimpleNamespace(chunk_gated_delta_rule=composite))
    )
    gdn = object.__new__(gdn_module.Qwen38TTNNGDN)
    constants = SimpleNamespace(eye="eye", tril="tril", ones="ones", masks="masks")
    result = module.chunk_rows_composite(gdn, "q_rows", "k_rows", "v_rows", "g_rows", "beta_rows", "state", constants)
    assert result == ("o", "final")
    assert calls == [
        (
            ("q_rows", "k_rows", "v_rows", "g_rows", "beta_rows"),
            {
                "scale": SCALE,
                "initial_state": "state",
                "output_final_state": True,
                "chunk_size": 32,
                "output_head_major": True,
                "eye": "eye",
                "tril": "tril",
                "ones": "ones",
                "masks": "masks",
            },
        )
    ]
    # the two forms agree on everything the prims read: the prims form's prep gets chunk_size 32 and this scale, its
    # scan gets output_final_state True and the [BH, K, V] view of this initial_state (the test above)
    assert calls[0][1]["scale"] == module.SCALE and calls[0][1]["chunk_size"] == module.TILE
    assert gdn_module.CHUNK_SIZE == module.TILE and gdn_module.HEAD_DIM == module.HEAD_DIM
    assert gdn_module.VALUE_HEADS_PER_DEVICE == module.HEADS and gdn_module.VALUE_WIDTH_PER_DEVICE == module.VALUE_WIDTH


def test_release_skips_a_view_whose_buffer_is_already_freed(expect_error, monkeypatch):
    freed = []
    monkeypatch.setattr(module, "ttnn", SimpleNamespace(deallocate=lambda t: freed.append(t)))
    live, dead = SimpleNamespace(is_allocated=lambda: True), SimpleNamespace(is_allocated=lambda: False)
    module._release(live, dead, live)
    assert freed == [live, live]
    with expect_error(AttributeError):
        module._release(object())  # a caller's plain object never reaches the release
