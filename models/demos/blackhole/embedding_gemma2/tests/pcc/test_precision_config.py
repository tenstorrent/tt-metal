# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The text runtime consumes the precision policy (tt/precision.py, doc/datatype_sweep/).

test_default_loads_selected_file   the default construction path (no `precision=`) loads
    doc/datatype_sweep/selected_precision_config.json; every weight dtype, matmul / SDPA / norm
    compute config, residual and mask dtype of the built model and of the ops one forward issues
    equals what the file says (read here with plain json, independent of the loader).
test_override_changes_runtime      an override (group dtype + fidelity, a layer exception, SDPA
    exp_approx, mask dtype) changes the constructed tensors and the configs the forward passes.
test_invalid_value_rejected        a wrong value fails loudly instead of being ignored.
"""

import json

import torch

import ttnn
from models.demos.blackhole.embedding_gemma2.tests.common import make_batch, upload_inputs
from models.demos.blackhole.embedding_gemma2.tt.precision import (
    DEFAULT_PRECISION_PATH,
    PRECISION_ENV,
    PrecisionConfig,
    deep_merge,
    load_precision,
)
from models.demos.blackhole.embedding_gemma2.tt.text_model import TextModel

_DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
_FID = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi3": ttnn.MathFidelity.HiFi3,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}


def _expected(raw_text: dict, group: str, layer: int | None) -> dict:
    """Group policy for `layer` from the raw JSON text section (exceptions applied in order)."""
    p = raw_text["weights"][group]
    if layer is not None:
        for exc in raw_text["layer_exceptions"]:
            if layer in exc["layers"] and group in exc.get("weights", {}):
                p = exc["weights"][group]
    return p


def _expected_sdpa(raw_text: dict, layer: int) -> dict:
    p = raw_text["sdpa"]
    for exc in raw_text["layer_exceptions"]:
        if layer in exc["layers"] and exc.get("sdpa"):
            p = exc["sdpa"]
    return p


class _Spy:
    """Records the compute / program configs of the matmul, SDPA and norm ops of one forward,
    keyed by the weight tensor's buffer address (matmuls) or call order (SDPA)."""

    def __init__(self, monkeypatch):
        self.mm = {}
        self.sdpa = []
        self.norms = []
        orig_linear, orig_minimal = ttnn.linear, ttnn.experimental.minimal_matmul
        orig_sdpa, orig_norm = ttnn.transformer.scaled_dot_product_attention, ttnn.rms_norm

        def linear(a, b, **kw):
            self.mm[b.buffer_address()] = ("linear", kw["compute_kernel_config"], kw.get("program_config"), kw["dtype"])
            return orig_linear(a, b, **kw)

        def minimal(a, b, **kw):
            self.mm[b.buffer_address()] = ("minimal", kw["compute_kernel_config"], None, kw["dtype"])
            return orig_minimal(a, b, **kw)

        def sdpa(*a, **kw):
            self.sdpa.append((kw["compute_kernel_config"], kw["program_config"]))
            return orig_sdpa(*a, **kw)

        def norm(*a, **kw):
            self.norms.append(kw["compute_kernel_config"])
            return orig_norm(*a, **kw)

        monkeypatch.setattr(ttnn, "linear", linear)
        monkeypatch.setattr(ttnn.experimental, "minimal_matmul", minimal)
        monkeypatch.setattr(ttnn.transformer, "scaled_dot_product_attention", sdpa)
        monkeypatch.setattr(ttnn, "rms_norm", norm)


def _forward(model, device, snapshot, lengths=(100,)):
    ids, mask = make_batch(snapshot, list(lengths))
    ids_d, keep_d, S = upload_inputs(device, ids, mask, model.config.max_seq_len)
    model.prepare(S)
    last, emb = model(ids_d, keep_d)
    out = ttnn.to_torch(emb).float()
    assert torch.isfinite(out).all()


def _check_model(model, raw: dict, spy: _Spy) -> int:
    """Assert model tensors + recorded op configs == raw JSON; returns the number of checks."""
    t = raw["text"]
    n = 0

    def check_mm(weight, group, layer):
        nonlocal n
        exp = _expected(t, group, layer)
        assert weight.dtype == _DT[exp["dtype"]], f"{group} layer {layer}: weight {weight.dtype} != {exp['dtype']}"
        kind, ckc, _pc, out_dtype = spy.mm[weight.buffer_address()]
        assert ckc.math_fidelity == _FID[exp["fidelity"]], f"{group} layer {layer}: {ckc.math_fidelity}"
        assert bool(ckc.fp32_dest_acc_en) == exp["fp32_acc"], f"{group} layer {layer}: fp32 acc"
        assert out_dtype == _DT[exp["out_dtype"]], f"{group} layer {layer}: output dtype {out_dtype}"
        n += 4

    for i, layer in enumerate(model.layers):
        check_mm(layer.self_attn.wqkv, "attn_qkv", i)
        check_mm(layer.self_attn.wo, "attn_o", i)
        check_mm(layer.mlp.w_gate, "mlp_gate_up", i)
        check_mm(layer.mlp.w_up, "mlp_gate_up", i)
        check_mm(layer.mlp.w_down, "mlp_down", i)
        check_mm(layer.ple_block.w_gate, "ple_block", i)
        check_mm(layer.ple_block.w_proj, "ple_block", i)
        ckc, pc = spy.sdpa[i]
        exp = _expected_sdpa(t, i)
        assert ckc.math_fidelity == _FID[exp["fidelity"]] and bool(ckc.fp32_dest_acc_en) == exp["fp32_acc"], i
        assert bool(pc.exp_approx_mode) == exp["exp_approx"], f"SDPA layer {i} exp_approx {pc.exp_approx_mode}"
        assert layer.residual_dtype == _DT[t["residual_dtype"]]
        n += 4
    check_mm(model.per_layer_input.proj_weight, "ple_proj", None)
    check_mm(model.w_out, "final_proj", None)
    assert spy.norms, "no rms_norm recorded"
    for ckc in spy.norms:
        assert ckc.math_fidelity == _FID[t["norms"]["fidelity"]]
        assert bool(ckc.fp32_dest_acc_en) == t["norms"]["fp32_acc"]
        n += 2
    assert model.masks.dtype == _DT[t["mask_dtype"]]
    return n + 1


def test_default_loads_selected_file(eg2_device, eg2_state_dict, eg2_config, eg2_cache_dir, eg2_snapshot, monkeypatch):
    monkeypatch.delenv(PRECISION_ENV, raising=False)
    raw = json.loads(DEFAULT_PRECISION_PATH.read_text())
    model = TextModel.from_state_dict(eg2_state_dict, eg2_config, eg2_device, cache_dir=eg2_cache_dir)
    assert model.precision.source == str(DEFAULT_PRECISION_PATH)
    assert model.precision.config_id == raw["config_id"]
    spy = _Spy(monkeypatch)
    _forward(model, eg2_device, eg2_snapshot)
    n = _check_model(model, raw, spy)
    print(f"default precision {raw['config_id']} from {DEFAULT_PRECISION_PATH}: {n} checks passed")
    for k in ("kv_cache_dtype", "ccl_dtype", "logits_dtype", "sampling"):
        assert raw[k].startswith("n/a"), k


def test_override_changes_runtime(eg2_device, eg2_state_dict, eg2_config, eg2_cache_dir, eg2_snapshot, monkeypatch):
    base = json.loads(DEFAULT_PRECISION_PATH.read_text())
    sdpa = dict(base["text"]["sdpa"], exp_approx=not base["text"]["sdpa"]["exp_approx"])
    down = _expected(base["text"], "mlp_down", 1)
    new_down = {"dtype": "bfp4" if down["dtype"] != "bfp4" else "bfp8", "fidelity": "LoFi", "fp32_acc": False}
    exc_down = {"dtype": "bf16", "fidelity": "HiFi4", "fp32_acc": True, "out_dtype": "bfp8"}
    mask = "bfp8" if base["text"]["mask_dtype"] != "bfp8" else "bf16"
    override = deep_merge(
        base,
        {
            "config_id": "test_override",
            "text": {
                "weights": {"mlp_down": new_down},
                "layer_exceptions": [{"layers": [0], "weights": {"mlp_down": exc_down}}],
                "sdpa": sdpa,
                "mask_dtype": mask,
            },
        },
    )
    model = TextModel.from_state_dict(
        eg2_state_dict, eg2_config, eg2_device, cache_dir=eg2_cache_dir, num_layers=2, precision=override
    )
    spy = _Spy(monkeypatch)
    _forward(model, eg2_device, eg2_snapshot)
    _check_model(model, override, spy)
    # explicitly: layer 1 follows the override, layer 0 the exception, both differ from the file
    assert model.layers[1].mlp.w_down.dtype == _DT[new_down["dtype"]] != _DT[down["dtype"]]
    assert model.layers[0].mlp.w_down.dtype == ttnn.bfloat16
    assert spy.mm[model.layers[1].mlp.w_down.buffer_address()][1].math_fidelity == ttnn.MathFidelity.LoFi
    assert spy.mm[model.layers[0].mlp.w_down.buffer_address()][1].math_fidelity == ttnn.MathFidelity.HiFi4
    assert spy.mm[model.layers[0].mlp.w_down.buffer_address()][3] == ttnn.bfloat8_b  # exception out_dtype
    assert bool(spy.sdpa[0][1].exp_approx_mode) == sdpa["exp_approx"]
    assert model.masks.dtype == _DT[mask]


def test_invalid_value_rejected(tmp_path, monkeypatch, expect_error):
    raw = json.loads(DEFAULT_PRECISION_PATH.read_text())
    bad = deep_merge(raw, {"text": {"weights": {"mlp_down": {"dtype": "bfp5"}}}})
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with expect_error(ValueError, "bfp5"):
        load_precision(p)
    missing = deep_merge(raw, {})
    del missing["text"]["weights"]["attn_o"]
    with expect_error(ValueError, "attn_o"):
        PrecisionConfig.from_dict(missing)
    monkeypatch.setenv(PRECISION_ENV, str(p))
    with expect_error(ValueError, "bfp5"):
        load_precision()
