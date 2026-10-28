# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The text, vision and audio runtimes consume the precision policy (tt/precision.py, doc/datatype_sweep/).

test_default_loads_selected_file   the default construction path (no `precision=`) loads
    doc/datatype_sweep/selected_precision_config.json; every weight dtype, matmul / SDPA / norm
    compute config, residual and mask dtype of the built model and of the ops one forward issues
    equals what the file says (read here with plain json, independent of the loader).
test_override_changes_runtime      an override (group dtype + fidelity, a layer exception, SDPA
    exp_approx, mask dtype) changes the constructed tensors and the configs the forward passes.
test_invalid_value_rejected        a wrong value fails loudly instead of being ignored.
test_default_vision_loads_selected_file / test_default_audio_loads_selected_file (E6b)
    the same for the `vision` / `audio` sections: VisionModel / AudioModel built with no `precision=`,
    one forward with spies on the matmul / SDPA / norm / conv ops, every weight dtype, compute config,
    output dtype, SDPA / mask / norm / attention / lconv-shift setting checked against the file.
test_override_changes_vision_runtime / test_override_changes_audio_runtime
    an override of the tower section (group dtype + fidelity + output dtype, a layer exception, SDPA,
    mask, attention, norms) changes the built tensors and the configs the forward passes.
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

_DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "fp32": ttnn.float32}
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
    for tower, group in (("vision", "mlp_down"), ("audio", "ffn_down")):
        bad = deep_merge(raw, {tower: {"weights": {group: {"fidelity": "HiFi5"}}}})
        with expect_error(ValueError, "HiFi5"):
            PrecisionConfig.from_dict(bad)
        missing = deep_merge(raw, {})
        del missing[tower]["weights"][group]
        with expect_error(ValueError, group):
            PrecisionConfig.from_dict(missing)
    bad = deep_merge(raw, {"audio": {"norms": {"sharded": "yes"}}})
    with expect_error(ValueError, "sharded"):
        PrecisionConfig.from_dict(bad)


# ---------------------------------------------------------------------- vision / audio (E6b)
def _expected_tower(raw_sec: dict, group: str, layer: int | None) -> dict:
    p = raw_sec["weights"][group]
    if layer is not None:
        for exc in raw_sec["layer_exceptions"]:
            if layer in exc["layers"] and group in exc.get("weights", {}):
                p = exc["weights"][group]
    return p


class _TowerSpy:
    """Configs of the matmul / SDPA / norm / conv ops of one forward. Linear-style matmuls are keyed by
    the weight's buffer address (`mm`) and by the first operand's (`mm_a`, for the lconv shift
    matmuls whose constant is the first operand)."""

    def __init__(self, monkeypatch):
        self.mm, self.mm_a, self.bmm, self.sdpa, self.norms, self.lnorms, self.convs = {}, {}, [], [], [], [], []
        orig = {
            "linear": ttnn.linear,
            "minimal": ttnn.experimental.minimal_matmul,
            "matmul": ttnn.matmul,
            "sdpa": ttnn.transformer.scaled_dot_product_attention,
            "rms": ttnn.rms_norm,
            "ln": ttnn.layer_norm,
            "conv": ttnn.conv2d,
        }

        def linear(a, b, **kw):
            rec = ("linear", kw["compute_kernel_config"], kw.get("program_config"), kw.get("dtype"))
            self.mm[b.buffer_address()] = rec
            self.mm_a[a.buffer_address()] = rec
            return orig["linear"](a, b, **kw)

        def minimal(a, b, **kw):
            self.mm[b.buffer_address()] = ("minimal", kw["compute_kernel_config"], None, kw.get("dtype"))
            return orig["minimal"](a, b, **kw)

        def matmul(a, b, **kw):
            self.bmm.append(kw.get("compute_kernel_config"))
            return orig["matmul"](a, b, **kw)

        def sdpa(*a, **kw):
            m = kw.get("attn_mask")
            self.sdpa.append((kw["compute_kernel_config"], kw["program_config"], None if m is None else m.dtype))
            return orig["sdpa"](*a, **kw)

        def rms(*a, **kw):
            self.norms.append((kw["compute_kernel_config"], kw.get("program_config")))
            return orig["rms"](*a, **kw)

        def ln(*a, **kw):
            self.lnorms.append(kw["compute_kernel_config"])
            return orig["ln"](*a, **kw)

        def conv(**kw):
            self.convs.append((kw["conv_config"].weights_dtype, kw["compute_config"]))
            return orig["conv"](**kw)

        monkeypatch.setattr(ttnn, "linear", linear)
        monkeypatch.setattr(ttnn.experimental, "minimal_matmul", minimal)
        monkeypatch.setattr(ttnn, "matmul", matmul)
        monkeypatch.setattr(ttnn.transformer, "scaled_dot_product_attention", sdpa)
        monkeypatch.setattr(ttnn, "rms_norm", rms)
        monkeypatch.setattr(ttnn, "layer_norm", ln)
        monkeypatch.setattr(ttnn, "conv2d", conv)


def _check_ckc(ckc, exp: dict, what: str) -> int:
    assert ckc.math_fidelity == _FID[exp["fidelity"]], f"{what}: fidelity {ckc.math_fidelity} != {exp['fidelity']}"
    assert bool(ckc.fp32_dest_acc_en) == exp["fp32_acc"], f"{what}: fp32 acc {ckc.fp32_dest_acc_en}"
    return 2


def _check_tower_mm(spy, weight, exp: dict, what: str, out_dtype=None) -> int:
    assert weight.dtype == _DT[exp["dtype"]], f"{what}: weight {weight.dtype} != {exp['dtype']}"
    _kind, ckc, _pc, out = spy.mm[weight.buffer_address()]
    n = _check_ckc(ckc, exp, what)
    want = out_dtype if out_dtype is not None else _DT[exp["out_dtype"]]
    assert out == want, f"{what}: output dtype {out} != {want}"
    return n + 2


def _vision_model(eg2_device, eg2_snapshot, eg2_cache_dir, **kw):
    from models.demos.blackhole.embedding_gemma2.tt.vision.config import VisionConfig
    from models.demos.blackhole.embedding_gemma2.tt.vision.vision_model import VisionModel
    from models.demos.blackhole.embedding_gemma2.tt.vision.weights import (
        load_vision_config_dicts,
        load_vision_state_dict,
    )

    cfg = VisionConfig.from_hf(*load_vision_config_dicts(eg2_snapshot))
    return VisionModel.from_state_dict(
        load_vision_state_dict(eg2_snapshot), cfg, eg2_device, cache_dir=eg2_cache_dir, **kw
    )


def _vision_forward(model, device, snapshot):
    from models.demos.blackhole.embedding_gemma2.tests.pcc.vision.vision_common import load_case
    from models.demos.blackhole.embedding_gemma2.tt.vision.vision_encoder import VisionEncoder

    out = VisionEncoder(model, device).encode(*load_case(str(snapshot), "tiny"))
    assert all(torch.isfinite(o).all() for o in out)


def _check_vision(model, raw: dict, spy: _TowerSpy) -> int:
    v = raw["vision"]
    n = 0
    for i, L in enumerate(model.layers):
        a, f = L.self_attn, L.mlp
        n += _check_tower_mm(spy, a.wqkv, _expected_tower(v, "attn_qkv", i), f"vision attn_qkv L{i}")
        n += _check_tower_mm(spy, a.wo, _expected_tower(v, "attn_o", i), f"vision attn_o L{i}")
        n += _check_tower_mm(spy, f.w_gate, _expected_tower(v, "mlp_gate_up", i), f"vision gate L{i}")
        n += _check_tower_mm(spy, f.w_up, _expected_tower(v, "mlp_gate_up", i), f"vision up L{i}")
        n += _check_tower_mm(spy, f.w_down, _expected_tower(v, "mlp_down", i), f"vision mlp_down L{i}")
        exp = _expected_sdpa(v, i)
        ckc, pc, mask_dtype = spy.sdpa[i]
        n += _check_ckc(ckc, exp, f"vision SDPA L{i}")
        assert bool(pc.exp_approx_mode) == exp["exp_approx"], f"vision SDPA L{i} exp_approx {pc.exp_approx_mode}"
        assert mask_dtype == _DT[v["mask_dtype"]], f"vision SDPA L{i} mask {mask_dtype}"
        n += 2
    n += _check_tower_mm(spy, model.patch_embedder.w_in, v["weights"]["patch_embed"], "vision patch_embed")
    n += _check_tower_mm(spy, model.embedder.w_proj, v["weights"]["projector"], "vision projector")
    assert len(spy.norms) == 7 * len(model.layers) + 1, f"{len(spy.norms)} rms_norm calls"
    for ckc, _ in spy.norms:
        n += _check_ckc(ckc, v["norms"], "vision norm")
    assert model.mask_dtype == _DT[v["mask_dtype"]]
    return n + 1


def test_default_vision_loads_selected_file(eg2_device, eg2_snapshot, eg2_cache_dir, monkeypatch):
    monkeypatch.delenv(PRECISION_ENV, raising=False)
    raw = json.loads(DEFAULT_PRECISION_PATH.read_text())
    model = _vision_model(eg2_device, eg2_snapshot, eg2_cache_dir)
    assert model.precision.source == str(DEFAULT_PRECISION_PATH)
    spy = _TowerSpy(monkeypatch)
    _vision_forward(model, eg2_device, eg2_snapshot)
    n = _check_vision(model, raw, spy)
    w = {g: model.layers[0].mlp.w_down.dtype for g in ("mlp_down",)}
    print(f"default vision precision of {raw['config_id']}: {n} checks passed; layer-0 w_down {w['mlp_down']}")


def test_override_changes_vision_runtime(eg2_device, eg2_snapshot, eg2_cache_dir, monkeypatch):
    base = json.loads(DEFAULT_PRECISION_PATH.read_text())
    v = base["vision"]
    o = _expected_tower(v, "attn_o", 1)
    new_o = {
        "dtype": "bfp4" if o["dtype"] != "bfp4" else "bfp8",
        "fidelity": "LoFi",
        "fp32_acc": False,
        "out_dtype": "bfp8",
    }
    exc_down = {
        "dtype": "bfp8" if v["weights"]["mlp_down"]["dtype"] != "bfp8" else "bf16",
        "fidelity": "HiFi4",
        "fp32_acc": True,
        "out_dtype": "bf16",
    }
    sdpa = dict(v["sdpa"], exp_approx=not v["sdpa"]["exp_approx"])
    mask = "bf16" if v["mask_dtype"] != "bf16" else "bfp8"
    override = deep_merge(
        base,
        {
            "config_id": "test_override_vision",
            "vision": {
                "weights": {"attn_o": new_o},
                "layer_exceptions": [{"layers": [0], "weights": {"mlp_down": exc_down}}],
                "sdpa": sdpa,
                "mask_dtype": mask,
            },
        },
    )
    model = _vision_model(eg2_device, eg2_snapshot, eg2_cache_dir, num_layers=2, precision=override)
    spy = _TowerSpy(monkeypatch)
    _vision_forward(model, eg2_device, eg2_snapshot)
    _check_vision(model, override, spy)
    assert model.layers[1].self_attn.wo.dtype == _DT[new_o["dtype"]] != _DT[o["dtype"]]
    assert spy.mm[model.layers[1].self_attn.wo.buffer_address()][1].math_fidelity == ttnn.MathFidelity.LoFi
    assert spy.mm[model.layers[1].self_attn.wo.buffer_address()][3] == ttnn.bfloat8_b
    assert model.layers[0].mlp.w_down.dtype == _DT[exc_down["dtype"]]
    assert model.layers[1].mlp.w_down.dtype == _DT[v["weights"]["mlp_down"]["dtype"]]
    assert bool(spy.sdpa[0][1].exp_approx_mode) == sdpa["exp_approx"]
    assert spy.sdpa[0][2] == _DT[mask]


def _audio_model(eg2_device, eg2_snapshot, eg2_cache_dir, **kw):
    from models.demos.blackhole.embedding_gemma2.tt.audio.audio_model import AudioModel
    from models.demos.blackhole.embedding_gemma2.tt.audio.config import AudioConfig
    from models.demos.blackhole.embedding_gemma2.tt.audio.weights import load_audio_config_dicts, load_audio_state_dict

    cfg = AudioConfig.from_hf(*load_audio_config_dicts(eg2_snapshot))
    return AudioModel.from_state_dict(
        load_audio_state_dict(eg2_snapshot), cfg, eg2_device, cache_dir=eg2_cache_dir, **kw
    )


def _audio_forward(model, device, snapshot):
    from models.demos.blackhole.embedding_gemma2.tests.pcc.audio.audio_common import load_case
    from models.demos.blackhole.embedding_gemma2.tt.audio.audio_encoder import AudioEncoder

    out = AudioEncoder(model, device, snapshot).encode(*load_case(str(snapshot), "s2"))
    assert all(torch.isfinite(o).all() for o in out)


def _check_audio(model, raw: dict, spy: _TowerSpy) -> int:
    a = raw["audio"]
    n = 0
    for i, L in enumerate(model.layers):
        n += _check_tower_mm(spy, L.attn.wqkv, _expected_tower(a, "attn_qkv", i), f"audio attn_qkv L{i}")
        for which, cl in (
            ("attn_o", L.attn.post),
            ("lconv_start", L.lconv.linear_start),
            ("lconv_end", L.lconv.linear_end),
        ):
            n += _check_tower_mm(spy, cl.weight, _expected_tower(a, which, i), f"audio {which} L{i}")
        for ffn in (L.ffn1, L.ffn2):
            n += _check_tower_mm(spy, ffn.ffw1.weight, _expected_tower(a, "ffn_up", i), f"audio ffn_up L{i}")
            n += _check_tower_mm(spy, ffn.ffw2.weight, _expected_tower(a, "ffn_down", i), f"audio ffn_down L{i}")
        for shift in L.lconv.shifts[64]:  # the s2 case runs in the 64-token bucket at batch 1
            n += _check_ckc(spy.mm_a[shift.buffer_address()][1], a["lconv_shift"], f"audio lconv shift L{i}")
        for norm in (L.norm_pre_attn, L.norm_post_attn, L.norm_out, L.ffn1.pre_norm, L.lconv.conv_norm):
            assert norm.sharded == a["norms"]["sharded"], f"audio norm L{i} sharded {norm.sharded}"
            n += 1
    w = a["weights"]
    n += _check_tower_mm(spy, model.subsample.w_in, w["subsample_proj"], "audio subsample_proj")
    n += _check_tower_mm(spy, model.output_proj.weight, w["output_proj"], "audio output_proj")
    n += _check_tower_mm(spy, model.embedder.w_proj, w["projector"], "audio projector")
    assert spy.convs, "no conv2d recorded"
    for wdt, ckc in spy.convs:
        assert wdt == _DT[w["subsample_conv"]["dtype"]], f"audio conv weights {wdt}"
        n += 1 + _check_ckc(ckc, w["subsample_conv"], "audio conv")
    assert spy.bmm, "no block matmul recorded"
    for ckc in spy.bmm:  # batch 1: the band-attention rel / QK^T / PV matmuls
        n += _check_ckc(ckc, a["attention"], "audio attention block matmul")
    for ckc, pc in spy.norms:
        n += _check_ckc(ckc, a["norms"], "audio rms_norm")
    n_sharded = sum(pc is not None for _, pc in spy.norms)
    assert (n_sharded > 0) == a["norms"]["sharded"], f"{n_sharded} sharded rms_norm calls"
    for ckc in spy.lnorms:
        n += _check_ckc(ckc, a["norms"], "audio subsample layer_norm")
    return n + 1


def test_default_audio_loads_selected_file(eg2_device, eg2_snapshot, eg2_cache_dir, monkeypatch):
    monkeypatch.delenv(PRECISION_ENV, raising=False)
    raw = json.loads(DEFAULT_PRECISION_PATH.read_text())
    model = _audio_model(eg2_device, eg2_snapshot, eg2_cache_dir)
    assert model.precision.source == str(DEFAULT_PRECISION_PATH)
    spy = _TowerSpy(monkeypatch)
    _audio_forward(model, eg2_device, eg2_snapshot)
    n = _check_audio(model, raw, spy)
    print(
        f"default audio precision of {raw['config_id']}: {n} checks passed; layer-0 ffn_down {model.layers[0].ffn1.ffw2.weight.dtype}"
    )


def test_override_changes_audio_runtime(eg2_device, eg2_snapshot, eg2_cache_dir, monkeypatch):
    base = json.loads(DEFAULT_PRECISION_PATH.read_text())
    a = base["audio"]
    d = _expected_tower(a, "ffn_down", 1)
    new_down = {
        "dtype": "bfp8" if d["dtype"] != "bfp8" else "bfp4",
        "fidelity": "HiFi2",
        "fp32_acc": True,
        "out_dtype": "bf16",
    }
    exc_down = {"dtype": "bf16", "fidelity": "LoFi", "fp32_acc": True, "out_dtype": "bf16"}
    att = {"fidelity": "HiFi2" if a["attention"]["fidelity"] != "HiFi2" else "HiFi4", "fp32_acc": True}
    shift = {"fidelity": "HiFi2" if a["lconv_shift"]["fidelity"] != "HiFi2" else "HiFi4", "fp32_acc": True}
    norms = dict(a["norms"], sharded=not a["norms"]["sharded"])
    conv = dict(
        a["weights"]["subsample_conv"],
        fidelity="HiFi2" if a["weights"]["subsample_conv"]["fidelity"] != "HiFi2" else "HiFi4",
    )
    override = deep_merge(
        base,
        {
            "config_id": "test_override_audio",
            "audio": {
                "weights": {"ffn_down": new_down, "subsample_conv": conv},
                "layer_exceptions": [{"layers": [0], "weights": {"ffn_down": exc_down}}],
                "attention": att,
                "lconv_shift": shift,
                "norms": norms,
            },
        },
    )
    model = _audio_model(eg2_device, eg2_snapshot, eg2_cache_dir, num_layers=2, precision=override)
    spy = _TowerSpy(monkeypatch)
    _audio_forward(model, eg2_device, eg2_snapshot)
    _check_audio(model, override, spy)
    assert model.layers[1].ffn1.ffw2.weight.dtype == _DT[new_down["dtype"]] != _DT[d["dtype"]]
    assert model.layers[0].ffn1.ffw2.weight.dtype == ttnn.bfloat16
    assert spy.mm[model.layers[0].ffn2.ffw2.weight.buffer_address()][1].math_fidelity == ttnn.MathFidelity.LoFi
    assert spy.mm[model.layers[1].ffn2.ffw2.weight.buffer_address()][1].math_fidelity == ttnn.MathFidelity.HiFi2
    assert spy.bmm[0].math_fidelity == _FID[att["fidelity"]]
    assert model.layers[0].norm_out.sharded == norms["sharded"]
