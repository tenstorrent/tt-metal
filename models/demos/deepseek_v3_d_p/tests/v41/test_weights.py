# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 checkpoint preparation (F0): dequant formulas, name/role mapping, real layer 2, device path.

* CPU: FP8 32x32 and MXFP4 dequantization on hand-computed values; layer mapping on a tiny synthetic
  checkpoint whose names are written out here, independently of the loader's schema.
* Real weights (skipped when the pinned checkpoint is absent): every tensor of layer 2 equals the vendored
  reference's own dequantization bit for bit, and the reference GEMM kernels reproduce it.
* Device: real layer-2 attention weights materialize on the 2x4 mesh.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.kernel_cpu import fp4_gemm, fp8_gemm
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41.prototype_oracle import _dequant as reference_dequant
from models.demos.deepseek_v3_d_p.tt.v41 import weights as W
from models.demos.deepseek_v3_d_p.tt.v41.attention import TtV41Attention

E8M0 = torch.float8_e8m0fnu
E4M3 = torch.float8_e4m3fn


def _e8m0(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).to(E8M0)


# --- dequant formulas --------------------------------------------------------------------------------


def test_dequant_fp8_block_hand_values():
    """Each 32x32 block takes its own scale; E4M3 extremes (448, min subnormal 2^-9) stay exact."""
    w = torch.empty(64, 64)
    w[:32, :32], w[:32, 32:], w[32:, :32], w[32:, 32:] = 1.5, -0.5, 448.0, 2.0**-9
    scale = _e8m0([[2.0, 2.0**-3], [2.0**10, 2.0**-4]])
    out = W.dequant_fp8_block(w.to(E4M3), scale)

    expected = torch.empty(64, 64)
    expected[:32, :32], expected[:32, 32:], expected[32:, :32], expected[32:, 32:] = 3.0, -0.0625, 458752.0, 2.0**-13
    assert out.dtype == torch.bfloat16
    assert torch.equal(out.float(), expected)


def test_dequant_mxfp4_nibble_order_and_scale():
    """Low nibble = even element; one scale per 32 elements along the input dimension."""
    packed = torch.empty(2, 32, dtype=torch.uint8)
    packed[0, :16] = 0x21  # even: code 1 = 0.5, odd: code 2 = 1.0
    packed[0, 16:] = 0xF9  # even: code 9 = -0.5, odd: code 15 = -6.0
    packed[1, :16] = 0x70  # even: code 0 = 0.0, odd: code 7 = 6.0
    packed[1, 16:] = 0x08  # even: code 8 = -0.0, odd: code 0 = 0.0
    scale = _e8m0([[4.0, 0.5], [1.0, 0.25]])
    out = W.dequant_mxfp4(packed.view(torch.int8), scale).float()

    assert out.shape == (2, 64)
    assert torch.equal(out[0, 0:32:2], torch.full((16,), 2.0))
    assert torch.equal(out[0, 1:32:2], torch.full((16,), 4.0))
    assert torch.equal(out[0, 32::2], torch.full((16,), -0.25))
    assert torch.equal(out[0, 33::2], torch.full((16,), -3.0))
    assert torch.equal(out[1, 0:32:2], torch.zeros(16))
    assert torch.equal(out[1, 1:32:2], torch.full((16,), 6.0))
    assert torch.equal(out[1, 32:], torch.zeros(32))
    assert torch.signbit(out[1, 32::2]).all() and not torch.signbit(out[1, 33::2]).any()


def test_dequant_rejects_malformed_inputs(expect_error):
    w8 = torch.ones(32, 32).to(E4M3)
    with expect_error(ValueError, "NaN"):
        W.dequant_fp8_block(w8, torch.full((1, 1), 0xFF, dtype=torch.uint8).view(E8M0))
    with expect_error(ValueError, "scale shape"):
        W.dequant_fp8_block(w8, _e8m0([[1.0, 1.0]]))
    with expect_error(ValueError, "whole number"):
        W.dequant_fp8_block(torch.ones(32, 48).to(E4M3), _e8m0([[1.0, 1.0]]))
    with expect_error(ValueError, "expected float8_e4m3fn"):
        W.dequant_fp8_block(torch.ones(32, 32, dtype=torch.bfloat16), _e8m0([[1.0]]))
    with expect_error(ValueError, "scale shape"):
        W.dequant_mxfp4(torch.zeros(2, 16, dtype=torch.int8), _e8m0([[1.0, 1.0], [1.0, 1.0]]))


# --- name and role mapping on a synthetic checkpoint -------------------------------------------------


class TinyConfig(DeepSeekV41FlashConfig):
    """Small dims, the real layer schedule (2: ratio-2 KV+index source, 20: ratio-1, 24: index only)."""

    EMB_SIZE = 64
    Q_LORA_RANK = 32
    NUM_ATTENTION_HEADS = 2
    HEAD_DIM = 64
    O_LORA_RANK = 32
    O_GROUPS = 2
    MOE_INTERMEDIATE_SIZE = 32
    NUM_ROUTED_EXPERTS = 3
    INDEX_N_HEADS = 2
    INDEX_HEAD_DIM = 32


# Checkpoint names and stored shapes for TinyConfig, written out by hand (fp8 entries get a .scale).
_COMMON = {
    "attn.wq_a": ("fp8", (32, 64)),
    "attn.q_norm.weight": ("bf16", (32,)),
    "attn.wq_b": ("fp8", (128, 32)),
    "attn.wkv": ("fp8", (64, 64)),
    "attn.kv_norm.weight": ("bf16", (64,)),
    "attn.wo_a": ("fp8", (64, 64)),
    "attn.wo_b": ("fp8", (64, 64)),
    "attn.attn_sink": ("f32", (2,)),
    "attn_norm.weight": ("bf16", (64,)),
    "ffn_norm.weight": ("bf16", (64,)),
    "ffn.gate.weight": ("bf16", (3, 64)),
    "ffn.gate.bias": ("f32", (3,)),
    "ffn.gate.bias_vl": ("f32", (3,)),
    "ffn.shared_experts.w1": ("fp8", (32, 64)),
    "ffn.shared_experts.w2": ("fp8", (64, 32)),
    "ffn.shared_experts.w3": ("fp8", (32, 64)),
    "hc_attn_fn": ("f32", (24, 256)),
    "hc_attn_base": ("f32", (24,)),
    "hc_attn_scale": ("f32", (3,)),
    "hc_ffn_fn": ("f32", (24, 256)),
    "hc_ffn_base": ("f32", (24,)),
    "hc_ffn_scale": ("f32", (3,)),
}
_LAYER2_EXTRA = {
    "attn.compressor.wkv.weight": ("bf16", (64, 64)),
    "attn.compressor.wgate.weight": ("bf16", (64, 64)),
    "attn.compressor.norm.weight": ("bf16", (64,)),
    "attn.indexer.wq_b": ("fp8", (64, 32)),
    "attn.indexer.weights_proj.weight": ("bf16", (2, 64)),
    "attn.indexer.wk.weight": ("bf16", (32, 64)),
    "attn.indexer.k_norm.weight": ("bf16", (32,)),
}
_LAYER24_EXTRA = {
    "attn.indexer.wq_b": ("fp8", (64, 32)),
    "attn.indexer.weights_proj.weight": ("bf16", (2, 64)),
}
_EXPERT = {"w1": (32, 64), "w2": (64, 32), "w3": (32, 64)}


def _synthetic_tensors(layer: int, extra: dict, gen: torch.Generator) -> dict[str, torch.Tensor]:
    out = {}
    for name, (fmt, shape) in {**_COMMON, **extra}.items():
        full = f"layers.{layer}.{name}"
        values = torch.randn(shape, generator=gen)
        if fmt == "fp8":
            out[f"{full}.weight"] = values.to(E4M3)
            blocks = (shape[0] // 32, shape[1] // 32)
            out[f"{full}.scale"] = (2.0 ** torch.randint(-4, 4, blocks, generator=gen).float()).to(E8M0)
        else:
            out[full] = values.to(torch.bfloat16 if fmt == "bf16" else torch.float32)
    for e in range(TinyConfig.NUM_ROUTED_EXPERTS):
        for w, (rows, cols) in _EXPERT.items():
            full = f"layers.{layer}.ffn.experts.{e}.{w}"
            out[f"{full}.weight"] = torch.randint(-128, 128, (rows, cols // 2), dtype=torch.int8, generator=gen)
            out[f"{full}.scale"] = (2.0 ** torch.randint(-4, 4, (rows, cols // 32), generator=gen).float()).to(E8M0)
    return out


@pytest.fixture
def synthetic_checkpoint(tmp_path):
    """Layers 2 and 24 in separate shards (one shard per layer, as the release is exported)."""
    gen = torch.Generator().manual_seed(0)
    weight_map, raw = {}, {}
    for layer, extra in ((2, _LAYER2_EXTRA), (24, _LAYER24_EXTRA)):
        tensors = _synthetic_tensors(layer, extra, gen)
        shard = f"model-{layer:05d}.safetensors"
        save_file(tensors, tmp_path / shard)
        weight_map.update({name: shard for name in tensors})
        raw.update(tensors)
    (tmp_path / W.INDEX_FILE).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return W.V41Checkpoint(tmp_path), raw


def _fp8(raw, layer, name):
    return W.dequant_fp8_block(raw[f"layers.{layer}.{name}.weight"], raw[f"layers.{layer}.{name}.scale"])


def test_layer_mapping_kv_index_source(synthetic_checkpoint):
    ckpt, raw = synthetic_checkpoint
    got = W.load_layer(ckpt, 2, TinyConfig)
    r = lambda name: raw[f"layers.2.{name}"]

    assert set(got) == {
        "attn",
        "attn_norm",
        "ffn_norm",
        "hc_attn",
        "hc_ffn",
        "gate_weights",
        "gate_bias_vl",
        "shared_expert_weights",
        "routed_expert_weights",
        "compressor",
        "indexer",
    }
    for key in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
        assert torch.equal(got["attn"][key], _fp8(raw, 2, f"attn.{key}")), key
    assert torch.equal(got["attn"]["q_norm"], r("attn.q_norm.weight"))
    assert torch.equal(got["attn"]["kv_norm"], r("attn.kv_norm.weight"))
    assert torch.equal(got["attn"]["attn_sink"], r("attn.attn_sink"))
    assert torch.equal(got["attn_norm"], r("attn_norm.weight"))
    assert torch.equal(got["ffn_norm"], r("ffn_norm.weight"))
    for site in ("attn", "ffn"):
        fn, base, scale = got[f"hc_{site}"]
        assert torch.equal(fn, r(f"hc_{site}_fn")) and torch.equal(base, r(f"hc_{site}_base"))
        assert torch.equal(scale, r(f"hc_{site}_scale"))
    assert torch.equal(got["gate_weights"]["weight"], r("ffn.gate.weight"))
    assert torch.equal(got["gate_weights"]["e_score_correction_bias"], r("ffn.gate.bias"))
    assert torch.equal(got["gate_bias_vl"], r("ffn.gate.bias_vl"))
    for key, w in (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")):
        assert torch.equal(got["shared_expert_weights"][key], _fp8(raw, 2, f"ffn.shared_experts.{w}")), key
        for e, expert in enumerate(got["routed_expert_weights"]):
            stem = f"layers.2.ffn.experts.{e}.{w}"
            assert torch.equal(expert[key], W.dequant_mxfp4(raw[f"{stem}.weight"], raw[f"{stem}.scale"])), stem
    assert len(got["routed_expert_weights"]) == TinyConfig.NUM_ROUTED_EXPERTS
    assert torch.equal(got["compressor"]["wkv"], r("attn.compressor.wkv.weight"))
    assert torch.equal(got["compressor"]["wgate"], r("attn.compressor.wgate.weight"))
    assert torch.equal(got["compressor"]["norm"], r("attn.compressor.norm.weight"))
    assert torch.equal(got["indexer"]["wq_b"], _fp8(raw, 2, "attn.indexer.wq_b"))
    assert torch.equal(got["indexer"]["weights_proj"], r("attn.indexer.weights_proj.weight"))
    assert torch.equal(got["indexer"]["wk"], r("attn.indexer.wk.weight"))
    assert torch.equal(got["indexer"]["k_norm"], r("attn.indexer.k_norm.weight"))


def test_layer_mapping_index_only_source(synthetic_checkpoint):
    ckpt, raw = synthetic_checkpoint
    got = W.load_layer_dense(ckpt, 24, TinyConfig)
    assert "compressor" not in got and "routed_expert_weights" not in got
    assert set(got["indexer"]) == {"wq_b", "weights_proj"}
    assert torch.equal(got["indexer"]["wq_b"], _fp8(raw, 24, "attn.indexer.wq_b"))


def test_layer_load_rejects_bad_requests(synthetic_checkpoint, expect_error):
    ckpt, _ = synthetic_checkpoint
    with expect_error(KeyError, "missing"):
        W.load_layer_dense(ckpt, 3, TinyConfig)  # not in the synthetic checkpoint
    with expect_error(ValueError, "not a backbone layer"):
        W.load_layer_dense(ckpt, TinyConfig.NUM_LAYERS, TinyConfig)
    with expect_error(ValueError, "shape"):
        W.load_layer_dense(ckpt, 2, type("Wider", (TinyConfig,), {"EMB_SIZE": 96}))


def test_resolve_checkpoint_pins_revision(tmp_path, monkeypatch, expect_error):
    monkeypatch.setenv(W.CHECKPOINT_ENV, str(tmp_path))
    assert W.resolve_checkpoint() is None
    (tmp_path / W.INDEX_FILE).write_text(json.dumps({"weight_map": {}}))
    with expect_error(ValueError, W.CHECKPOINT_INDEX_BLOB):
        W.resolve_checkpoint()


# --- real layer 2 ------------------------------------------------------------------------------------

REAL_LAYER = 2


@pytest.fixture(scope="module")
def real_checkpoint():
    ckpt = W.resolve_checkpoint()
    if ckpt is None or not (ckpt.root / ckpt.weight_map[f"layers.{REAL_LAYER}.attn.wkv.weight"]).is_file():
        pytest.skip(f"pinned V4.1 checkpoint with layer {REAL_LAYER} not present (set {W.CHECKPOINT_ENV})")
    return ckpt


def _walk(d, path=()):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _walk(v, path + (k,))
        else:
            yield path + (k,), v


# Loader key path -> checkpoint name under layers.2. (shared experts and hc tuples are checked separately).
_REAL_NAMES = {
    ("attn", "wq_a"): "attn.wq_a",
    ("attn", "q_norm"): "attn.q_norm.weight",
    ("attn", "wq_b"): "attn.wq_b",
    ("attn", "wkv"): "attn.wkv",
    ("attn", "kv_norm"): "attn.kv_norm.weight",
    ("attn", "wo_a"): "attn.wo_a",
    ("attn", "wo_b"): "attn.wo_b",
    ("attn", "attn_sink"): "attn.attn_sink",
    ("attn_norm",): "attn_norm.weight",
    ("ffn_norm",): "ffn_norm.weight",
    ("gate_weights", "weight"): "ffn.gate.weight",
    ("gate_weights", "e_score_correction_bias"): "ffn.gate.bias",
    ("gate_bias_vl",): "ffn.gate.bias_vl",
    ("shared_expert_weights", "gate_proj"): "ffn.shared_experts.w1",
    ("shared_expert_weights", "up_proj"): "ffn.shared_experts.w3",
    ("shared_expert_weights", "down_proj"): "ffn.shared_experts.w2",
    ("compressor", "wkv"): "attn.compressor.wkv.weight",
    ("compressor", "wgate"): "attn.compressor.wgate.weight",
    ("compressor", "norm"): "attn.compressor.norm.weight",
    ("indexer", "wq_b"): "attn.indexer.wq_b",
    ("indexer", "weights_proj"): "attn.indexer.weights_proj.weight",
    ("indexer", "wk"): "attn.indexer.wk.weight",
    ("indexer", "k_norm"): "attn.indexer.k_norm.weight",
}


def _reference_value(ckpt, name: str) -> torch.Tensor:
    """The vendored reference's dequantization of a stored tensor (the prototype oracle's ``_dequant``)."""
    full = f"layers.{REAL_LAYER}.{name}"
    if f"{full}.scale" in ckpt:
        raw = ckpt.read([f"{full}.weight", f"{full}.scale"])
        weight = raw[f"{full}.weight"]
        if weight.dtype == torch.int8:
            weight = weight.view(torch.float4_e2m1fn_x2)
        return reference_dequant(SimpleNamespace(weight=weight, scale=raw[f"{full}.scale"]))
    return ckpt.read([full])[full]


@pytest.mark.timeout(1800)
def test_real_layer2_matches_reference(real_checkpoint):
    ckpt = real_checkpoint
    with open(ckpt.root / "config.json", encoding="utf-8") as handle:
        hf = json.load(handle)["text_config"]
    got = W.load_layer(ckpt, REAL_LAYER)

    # Shapes against the released config.json (independently of the loader's schema).
    dim, heads, head_dim = hf["hidden_size"], hf["num_attention_heads"], hf["head_dim"]
    assert got["attn"]["wq_b"].shape == (heads * head_dim, hf["q_lora_rank"])
    assert got["attn"]["wo_a"].shape == (hf["o_groups"] * hf["o_lora_rank"], heads * head_dim // hf["o_groups"])
    assert got["attn"]["wo_b"].shape == (dim, hf["o_groups"] * hf["o_lora_rank"])
    assert got["gate_weights"]["weight"].shape == (hf["n_routed_experts"], dim)
    assert len(got["routed_expert_weights"]) == hf["n_routed_experts"]
    assert got["routed_expert_weights"][0]["gate_proj"].shape == (hf["moe_intermediate_size"], dim)
    assert got["routed_expert_weights"][0]["down_proj"].shape == (dim, hf["moe_intermediate_size"])

    # Every dense tensor equals the reference's dequantization (dtype included).
    dense = {path: t for path, t in _walk({k: v for k, v in got.items() if k != "routed_expert_weights"})}
    hc = {("hc_attn",): "hc_attn", ("hc_ffn",): "hc_ffn"}
    assert set(dense) == set(_REAL_NAMES) | set(hc)
    for path, name in _REAL_NAMES.items():
        expected = _reference_value(ckpt, name)
        assert dense[path].dtype == expected.dtype and torch.equal(dense[path], expected), path
    for path, site in hc.items():
        for part, t in zip(("fn", "base", "scale"), dense[path]):
            assert torch.equal(t, _reference_value(ckpt, f"{site}_{part}")), (site, part)

    # Every routed expert matrix equals the reference's unpack_fp4 * scale.
    for e, expert in enumerate(got["routed_expert_weights"]):
        for key, w in (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")):
            assert torch.equal(expert[key], _reference_value(ckpt, f"ffn.experts.{e}.{w}")), (e, key)
            assert torch.isfinite(expert[key]).all(), (e, key)

    # The reference GEMM kernels, run on an identity activation, return W^T of the stored weight: the loaded
    # tensor is what the reference actually multiplies by (FP8 attention wkv, FP4 expert 0 down_proj).
    for loaded, name, gemm in (
        (got["attn"]["wkv"], "attn.wkv", "fp8"),
        (got["routed_expert_weights"][0]["down_proj"], "ffn.experts.0.w2", "fp4"),
    ):
        full = f"layers.{REAL_LAYER}.{name}"
        raw = ckpt.read([f"{full}.weight", f"{full}.scale"])
        k = loaded.shape[1]
        eye = torch.eye(k).to(E4M3)
        eye_scale = torch.ones(k, k // 32)
        if gemm == "fp8":
            c = fp8_gemm(eye, eye_scale, raw[f"{full}.weight"], raw[f"{full}.scale"], block_size=32)
        else:
            weight = raw[f"{full}.weight"].view(torch.float4_e2m1fn_x2)
            c = fp4_gemm(eye, eye_scale, weight, raw[f"{full}.scale"], act_block_size=32)
        assert torch.equal(c.float().T, loaded.float()), name

    # Repeat loads are identical.
    again = dict(_walk(W.load_layer_dense(ckpt, REAL_LAYER)))
    assert set(again) == set(dense)
    for path, t in dense.items():
        same = all(map(torch.equal, t, again[path])) if isinstance(t, tuple) else torch.equal(t, again[path])
        assert same, path


# --- device materialization --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 4),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-mesh-2x4",
        )
    ],
    indirect=True,
)
def test_real_layer2_attention_materializes(real_checkpoint, mesh_device, device_params):
    """Real layer-2 attention weights go through TtV41Attention's placement; readback equals host bfp8/bf16."""
    cfg = DeepSeekV41FlashConfig
    weights = W.load_layer_dense(real_checkpoint, REAL_LAYER)
    attn = TtV41Attention(mesh_device, cfg, REAL_LAYER, weights["attn"], seq_len=256, host=None)
    shape = tuple(mesh_device.shape)

    # wkv: TP-sharded over its input (hidden) dim, bfloat8_b. Host-side bfp8 packing is what from_torch
    # uploads, so the gathered readback equals a host-only bfp8 round trip bit for bit.
    wkv_t = weights["attn"]["wkv"].transpose(0, 1).contiguous()[None, None]
    host_bfp8 = ttnn.to_torch(ttnn.from_torch(wkv_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT))
    readback = ttnn.to_torch(attn.wkv, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, shape, dims=(0, 2)))
    assert readback.shape == (shape[0], 1, cfg.EMB_SIZE, cfg.HEAD_DIM)
    for row in range(shape[0]):
        assert torch.equal(readback[row : row + 1].float(), host_bfp8.float()), row

    # attn_sink: bf16, replicated; stored pre-divided by the softmax scale.
    sink = ttnn.to_torch(attn.sink_full, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    expected = (weights["attn"]["attn_sink"] / cfg.HEAD_DIM**-0.5).to(torch.bfloat16).reshape(1, 1, 1, -1)
    for chip in range(sink.shape[0]):
        assert torch.equal(sink[chip : chip + 1], expected), chip
