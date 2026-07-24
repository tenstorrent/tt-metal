# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Consolidated transformer stack tests:
#   - Single decoder layer PCC (TT vs PyTorch ref, real layer-0 weights)
#   - Mesh parallel decoder layer (bf8 EP vs dense bf16)
#   - Resident sharded backbone (scale / memory)
#   - Transformer weight-cache path helpers
#   - ISL sweep CSV (decoder layer PCC)
#   - Mesh integration: SP/TP/EP primitives (merged from test_parallel_2x2.py)
#
# Lean ISL: S=1, 32, 4096, 4160 (+ S=22784 max-context @pytest.mark.slow).
# Mesh tests default HY_NUM_LAYERS=8 (resident) or 2 (max-seq); bump via env.
#
# Run (fast):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_transformer.py -m "not slow" -v
# ISL sweep CSV:
#   HY_PCC_CSV=/tmp/hunyuan_transformer_isl.csv python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_transformer.py -k isl_sweep -v -s

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[5]
PCC_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PCC_DIR) not in sys.path:
    sys.path.insert(0, str(PCC_DIR))

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive
from models.experimental.hunyuan_image_3_0.ref.attention.rope_2d import build_batch_2d_rope
from models.experimental.hunyuan_image_3_0.ref.transformer_layer import HunyuanImage3DecoderLayer as RefLayer
from models.experimental.hunyuan_image_3_0.ref.weights import (
    load_prefixed_state_dict,
    resolve_base_model_dir,
)
from models.experimental.hunyuan_image_3_0.tt.cache import cache_dir_is_set, transformer_cache_dir
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel
from models.experimental.hunyuan_image_3_0.tt.transformer_layer import HunyuanTtDecoderLayer
from models.tt_dit.parallel.manager import CCLManager
from mesh_helpers import (
    build_mesh_model,
    causal_mask,
    mesh_ccl,
    replicate_to_mesh,
    run_embeds_forward,
)
from pcc_common import (
    DECODER_LAYER_PCC_CASES,
    LEAN_ISL_CASES,
    PRODUCTION_MODULE_CASES,
    PCC_BLOCK,
    image_slices_from_infos,
    isl_csv_path,
    max_seq_tile_aligned,
    pcc_metrics,
    rope_image_infos,
    transformer_cfg,
    write_isl_csv,
)

LAYER_NUM = 0
PCC_PARALLEL = 0.97  # bf8 mesh parallel vs dense bf16
NUM_LAYERS_RESIDENT = int(os.environ.get("HY_NUM_LAYERS", "8"))
NUM_LAYERS_MAXSEQ = int(os.environ.get("HY_NUM_LAYERS_MAXSEQ", os.environ.get("HY_NUM_LAYERS", "2")))

DECODER_LAYER_FAST = [
    (mode, batch, seq_len, image_infos, label)
    for mode, batch, seq_len, image_infos, label in DECODER_LAYER_PCC_CASES
    if seq_len < 4096
]
DECODER_LAYER_SLOW = [
    (mode, batch, seq_len, image_infos, label)
    for mode, batch, seq_len, image_infos, label in DECODER_LAYER_PCC_CASES
    if seq_len >= 4096
]

_LAYER_CACHE: dict[int, tuple] = {}


@pytest.fixture(scope="function")
def device():
    """Function-scoped device so mesh tests can open a 2×2 mesh after single-device cases."""
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


def _layer_tt_sd(layer_num: int = LAYER_NUM) -> dict[str, torch.Tensor]:
    sd = load_prefixed_state_dict(resolve_base_model_dir(), f"model.layers.{layer_num}.")
    prefix = f"model.layers.{layer_num}."
    return {f"{prefix}{k}": v for k, v in sd.items()}


def _build_ref_layer(c: dict, sd: dict[str, torch.Tensor]) -> RefLayer:
    ref = RefLayer(
        hidden_size=c["H"],
        num_attention_heads=c["HEADS"],
        num_key_value_heads=c["KV"],
        attention_head_dim=c["HD"],
        num_experts=c["E"],
        moe_topk=c["K"],
        moe_intermediate_size=c["MOE_INTER"],
        num_shared_expert=c["NUM_SHARED"],
        use_mixed_mlp_moe=c["MIXED"],
        norm_topk_prob=c["NORM_TOPK"],
        use_qk_norm=c["QKN"],
        rms_norm_eps=c["EPS"],
        layer_idx=LAYER_NUM,
    )
    ref.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    ref.eval()
    return ref


def _build_tt_layer(device, tt_sd: dict[str, torch.Tensor], c: dict, **kwargs) -> HunyuanTtDecoderLayer:
    kwargs.setdefault("stream_experts", True)
    return HunyuanTtDecoderLayer(
        device,
        tt_sd,
        layer_num=LAYER_NUM,
        hidden_size=c["H"],
        num_heads=c["HEADS"],
        num_kv_heads=c["KV"],
        head_dim=c["HD"],
        num_experts=c["E"],
        moe_topk=c["K"],
        use_qk_norm=c["QKN"],
        use_mixed_mlp_moe=c["MIXED"],
        norm_topk_prob=c["NORM_TOPK"],
        rms_norm_eps=c["EPS"],
        **kwargs,
    )


def _get_decoder_layer(device):
    if id(device) not in _LAYER_CACHE:
        # `device` is function-scoped, so each test gets a fresh object (and id).
        # Without eviction, every parametrized case would leave its (ref, tt)
        # behind — and `ref` is a full fp32 PyTorch decoder layer (~10GB with 64
        # experts). Across the parametrized decoder-layer suite that piles up to
        # ~150GB of host RAM and gets the process OOM-killed once the resident
        # backbone test then loads its 32 layers. Only one device is ever live at
        # a time, so drop stale entries before caching the current one.
        _LAYER_CACHE.clear()
        c = transformer_cfg()
        sd = load_prefixed_state_dict(resolve_base_model_dir(), f"model.layers.{LAYER_NUM}.")
        ref = _build_ref_layer(c, sd)
        tt_sd = _layer_tt_sd(LAYER_NUM)
        tt = _build_tt_layer(device, tt_sd, c)
        _LAYER_CACHE[id(device)] = (ref, tt, c)
    return _LAYER_CACHE[id(device)]


def _decoder_layer_run(device, batch, seq_len, image_infos=None, *, seed=0):
    ref, tt, c = _get_decoder_layer(device)
    text_only = image_infos is None
    infos = rope_image_infos(image_infos, batch) if not text_only else None

    torch.manual_seed(seed)
    x = torch.randn(batch, seq_len, c["H"], dtype=torch.bfloat16)

    cos, sin = build_batch_2d_rope(seq_len, c["HD"], image_infos=infos)
    if text_only:
        mask_add = None
        is_causal = True
    else:
        spans = image_slices_from_infos(image_infos)
        mask_add = to_additive(build_attention_mask(seq_len, spans, bsz=batch), dtype=torch.float32)
        is_causal = False

    with torch.no_grad():
        ref_out = ref(
            x.float(),
            attention_mask=mask_add,
            custom_pos_emb=(cos, sin),
            is_causal=is_causal,
        )

    x_tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_tt = tt.forward(x_tt, seq_len=seq_len, image_infos=infos, attention_mask=None)
    tt_out = ttnn.to_torch(out_tt)[..., : c["H"]]
    ttnn.deallocate(x_tt)
    return pcc_metrics(ref_out, tt_out, PCC_BLOCK)


def _mesh_causal_mask(batch: int, seq_len: int) -> torch.Tensor:
    return torch.triu(torch.full((seq_len, seq_len), -1.0e30), diagonal=1).reshape(batch, 1, seq_len, seq_len)


def _upload_mesh_input(mesh_device, x: torch.Tensor):
    return ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _build_resident_model(mesh_device, num_layers: int) -> HunyuanTtModel:
    c = transformer_cfg()
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    model_dir = resolve_base_model_dir()

    def layer_loader(i: int):
        sd = load_prefixed_state_dict(model_dir, f"model.layers.{i}.")
        return {f"model.layers.{i}.{k}": v for k, v in sd.items()}

    return HunyuanTtModel(
        mesh_device,
        num_layers=num_layers,
        hidden_size=c["H"],
        num_heads=c["HEADS"],
        num_kv_heads=c["KV"],
        head_dim=c["HD"],
        num_experts=c["E"],
        moe_topk=c["K"],
        use_qk_norm=c["QKN"],
        use_mixed_mlp_moe=c["MIXED"],
        norm_topk_prob=c["NORM_TOPK"],
        rms_norm_eps=c["EPS"],
        stream_experts=False,
        layer_loader=layer_loader,
        apply_final_norm=False,
        weight_dtype=ttnn.bfloat8_b,
        ccl_manager=ccl,
        expert_mesh_axis=1,
        tp_axis=1,
        tp_factor=2,
        sp_axis=0,
        sp_factor=2,
    )


def _backbone_forward(mesh_device, model: HunyuanTtModel, batch: int, seq_len: int) -> torch.Tensor:
    c = transformer_cfg()
    x = torch.randn(batch, seq_len, c["H"]) * 0.02
    x_tt = _upload_mesh_input(mesh_device, x)
    mask_tt = _upload_mesh_input(mesh_device, _mesh_causal_mask(batch, seq_len))
    out = model.forward(inputs_embeds=x_tt, seq_len=seq_len, image_infos=None, attention_mask=mask_tt)
    return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:batch].float()


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


# ---------------------------------------------------------------------------
# Single decoder layer PCC (real layer-0 weights)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode,batch,seq_len,image_infos,label", DECODER_LAYER_FAST)
def test_decoder_layer_pcc(device, mode, batch, seq_len, image_infos, label):
    infos = None if mode == "text" else image_infos
    p, d = _decoder_layer_run(device, batch, seq_len, infos)
    phase = "decode" if seq_len == 1 else "prefill"
    print(f"Decoder layer {mode} {phase} [{batch}, {seq_len}] ({label}): PCC={p:.8f}  max|diff|={d:.6f}")
    assert p >= PCC_BLOCK


@pytest.mark.slow
@pytest.mark.parametrize("mode,batch,seq_len,image_infos,label", DECODER_LAYER_SLOW)
def test_decoder_layer_large_isl_pcc(device, mode, batch, seq_len, image_infos, label):
    infos = None if mode == "text" else image_infos
    p, d = _decoder_layer_run(device, batch, seq_len, infos)
    print(f"Decoder layer {mode} [{batch}, {seq_len}] ({label}): PCC={p:.8f}")
    assert p >= PCC_BLOCK


@pytest.mark.slow
def test_decoder_layer_max_context_pcc(device):
    max_seq = max_seq_tile_aligned()
    p, d = _decoder_layer_run(device, 1, max_seq)
    print(f"Decoder layer max context S={max_seq}: PCC={p:.8f}  max|diff|={d:.6f}")
    assert p >= PCC_BLOCK


@pytest.mark.slow
@pytest.mark.parametrize("mode,seq_len,image_infos,label", PRODUCTION_MODULE_CASES)
def test_decoder_layer_production_pcc(device, mode, seq_len, image_infos, label):
    """Production submodule gate: decode S=1 and prefill S=4160 (text + image layout)."""
    infos = None if mode == "text" else image_infos
    p, d = _decoder_layer_run(device, 1, seq_len, infos)
    phase = "decode" if seq_len == 1 else "prefill"
    print(f"Decoder layer production {mode} {phase} [{label}] S={seq_len}: " f"PCC={p:.8f}  max|diff|={d:.6f}")
    assert p >= PCC_BLOCK


# ---------------------------------------------------------------------------
# Transformer weight-cache helpers (no device)
# ---------------------------------------------------------------------------
def test_transformer_cache_dir_key(monkeypatch):
    monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")
    path = transformer_cache_dir(
        model_name="hunyuan-image-3.0",
        mesh_shape=(2, 2),
        tp_axis=1,
        tp_factor=2,
        sp_axis=0,
        sp_factor=2,
        weight_dtype=ttnn.bfloat8_b,
        num_layers=32,
        bf16_layers={0, 1, 2, 3, 28, 29, 30, 31},
    )
    assert path is not None
    assert path.parts[-1] == "SP2a0_TP2a1_mesh2x2_L32_BFLOAT8_B_bf16_0-3_28-31"
    assert path.parent.name == "transformer"
    assert path.parent.parent.name == "hunyuan-image-3.0"


def test_cache_disabled_without_env(monkeypatch):
    monkeypatch.delenv("TT_DIT_CACHE_DIR", raising=False)
    assert not cache_dir_is_set()
    assert (
        transformer_cache_dir(
            model_name="hunyuan-image-3.0",
            mesh_shape=(2, 2),
            tp_axis=1,
            tp_factor=2,
            sp_axis=0,
            sp_factor=2,
            weight_dtype=ttnn.bfloat8_b,
            num_layers=4,
            bf16_layers=None,
        )
        is None
    )


# ---------------------------------------------------------------------------
# Mesh: parallel decoder layer + resident backbone
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_decoder_layer_parallel_vs_dense(mesh_device):
    mesh_device.enable_program_cache()
    c = transformer_cfg()
    sd = _layer_tt_sd(LAYER_NUM)
    B, S = 1, 32
    torch.manual_seed(0)
    x = torch.randn(B, S, c["H"]) * 0.05

    def upload():
        return _upload_mesh_input(mesh_device, x)

    dense = _build_tt_layer(mesh_device, sd, c, weight_dtype=ttnn.bfloat16)
    y_dense = dense.forward(upload(), seq_len=S, image_infos=None, attention_mask=None)

    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    par = _build_tt_layer(
        mesh_device,
        sd,
        c,
        weight_dtype=ttnn.bfloat8_b,
        stream_experts=False,
        ccl_manager=ccl,
        expert_mesh_axis=1,
        tp_axis=1,
        tp_factor=2,
    )
    y_par = par.forward(upload(), seq_len=S, image_infos=None, attention_mask=None)

    d0 = ttnn.to_torch(y_dense, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B].float()
    p0 = ttnn.to_torch(y_par, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B].float()
    passing, pcc = comp_pcc(d0, p0, PCC_PARALLEL)
    logger.info(f"decoder-layer parallel(bf8) vs dense(bf16) PCC: {pcc:.6f}  shape={tuple(p0.shape)}")
    assert passing, f"PCC {pcc:.6f} < {PCC_PARALLEL}"


@pytest.mark.slow
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("batch,seq_len,label", [(b, s, lbl) for b, s, lbl in LEAN_ISL_CASES if s < 4096])
def test_backbone_resident(mesh_device, batch, seq_len, label):
    mesh_device.enable_program_cache()
    c = transformer_cfg()
    model = _build_resident_model(mesh_device, NUM_LAYERS_RESIDENT)
    y = _backbone_forward(mesh_device, model, batch, seq_len)
    logger.info(
        f"resident backbone {NUM_LAYERS_RESIDENT}L S={seq_len} ({label}): "
        f"out={tuple(y.shape)} finite={bool(torch.isfinite(y).all())}"
    )
    assert y.shape == (batch, seq_len, c["H"])
    assert torch.isfinite(y).all()


@pytest.mark.slow
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("batch,seq_len,label", [(b, s, lbl) for b, s, lbl in LEAN_ISL_CASES if s >= 4096])
def test_backbone_resident_large_isl(mesh_device, batch, seq_len, label):
    mesh_device.enable_program_cache()
    c = transformer_cfg()
    model = _build_resident_model(mesh_device, NUM_LAYERS_RESIDENT)
    y = _backbone_forward(mesh_device, model, batch, seq_len)
    assert y.shape == (batch, seq_len, c["H"])
    assert torch.isfinite(y).all()


@pytest.mark.slow
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_backbone_max_seq(mesh_device):
    mesh_device.enable_program_cache()
    c = transformer_cfg()
    S = max_seq_tile_aligned()
    logger.info(f"max_position_embeddings={c['MAX_SEQ']} -> testing S={S} ({NUM_LAYERS_MAXSEQ} layers, bf8 resident)")
    model = _build_resident_model(mesh_device, NUM_LAYERS_MAXSEQ)
    y = _backbone_forward(mesh_device, model, 1, S)
    logger.info(f"max-seq forward OK: S={S} out={tuple(y.shape)} finite={bool(torch.isfinite(y).all())}")
    assert y.shape == (1, S, c["H"])
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# ISL sweep CSV (decoder layer PCC)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_decoder_layer_isl_sweep_table(device, tmp_path):
    rows = []
    cases = DECODER_LAYER_FAST + DECODER_LAYER_SLOW
    for mode, batch, seq_len, image_infos, label in cases:
        infos = None if mode == "text" else image_infos
        p, d = _decoder_layer_run(device, batch, seq_len, infos)
        rows.append(
            {
                "module": "decoder_layer",
                "mode": mode,
                "batch": batch,
                "seq_len": seq_len,
                "label": label,
                "pcc": f"{p:.8f}",
                "max_abs_diff": f"{d:.6f}",
                "threshold": PCC_BLOCK,
                "pass": p >= PCC_BLOCK,
            }
        )
        print(f"  Decoder layer {mode:5s} {label:40s}  S={seq_len:5d}  PCC={p:.8f}")

    max_seq = max_seq_tile_aligned()
    p, d = _decoder_layer_run(device, 1, max_seq)
    rows.append(
        {
            "module": "decoder_layer",
            "mode": "text",
            "batch": 1,
            "seq_len": max_seq,
            "label": f"max context S={max_seq}",
            "pcc": f"{p:.8f}",
            "max_abs_diff": f"{d:.6f}",
            "threshold": PCC_BLOCK,
            "pass": p >= PCC_BLOCK,
        }
    )

    out = isl_csv_path("hunyuan_transformer_isl.csv") or tmp_path / "hunyuan_transformer_isl.csv"
    write_isl_csv(rows, out)
    print(f"\nISL sweep CSV: {out}")
    assert all(r["pass"] for r in rows)


# ---------------------------------------------------------------------------
# Mesh integration: SP/TP/EP (unique gates; MoE/decoder-layer parallel in test_moe / above)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_mesh_reshard_roundtrip(mesh_device):
    """sp_shard then sp_gather round-trip == identity."""
    from models.experimental.hunyuan_image_3_0.tt.parallel_utils import sp_gather, sp_shard

    mesh_device.enable_program_cache()
    ccl = mesh_ccl(mesh_device)
    B, S, H = 1, 64, 128
    x = torch.randn(B, S, H)
    x_tt = replicate_to_mesh(mesh_device, x)
    sharded = sp_shard(ccl, x_tt, dim=1, mesh_axis=0, n=2)
    gathered = sp_gather(ccl, sharded, dim=1, mesh_axis=0, n=2)
    g0 = ttnn.to_torch(gathered, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B].float()
    passing, pcc = comp_pcc(x, g0, 0.999)
    logger.info(f"mesh reshard round-trip PCC={pcc:.6f}")
    assert passing


@pytest.mark.slow
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_sp_model_vs_no_sp(mesh_device):
    """SP=2 small-stack backbone vs sp_factor=1 on the same mesh."""
    mesh_device.enable_program_cache()
    ccl = mesh_ccl(mesh_device)
    c = transformer_cfg()
    B, S = 1, 256
    torch.manual_seed(0)
    x = torch.randn(B, S, c["H"]) * 0.05
    m = causal_mask(S)
    y_ref = run_embeds_forward(mesh_device, build_mesh_model(mesh_device, ccl, sp_factor=1), x, S, m)
    y_sp = run_embeds_forward(mesh_device, build_mesh_model(mesh_device, ccl, sp_factor=2), x, S, m)
    passing, pcc = comp_pcc(y_ref, y_sp, PCC_BLOCK)
    logger.info(f"SP=2 vs SP=1 model PCC={pcc:.6f}  shape={tuple(y_sp.shape)}")
    assert passing


@pytest.mark.slow
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_sp_model_unaligned_seq(mesh_device):
    """SP=2 with non-tile-aligned S=200 (pads to 256) vs sp=1."""
    mesh_device.enable_program_cache()
    ccl = mesh_ccl(mesh_device)
    c = transformer_cfg()
    B, S = 1, 200
    torch.manual_seed(0)
    x = torch.randn(B, S, c["H"]) * 0.05
    m = causal_mask(S)
    y_ref = run_embeds_forward(mesh_device, build_mesh_model(mesh_device, ccl, sp_factor=1), x, S, m)
    y_sp = run_embeds_forward(mesh_device, build_mesh_model(mesh_device, ccl, sp_factor=2), x, S, m)
    assert tuple(y_sp.shape) == (B, S, c["H"])
    passing, pcc = comp_pcc(y_ref, y_sp, PCC_BLOCK)
    logger.info(f"SP unaligned S=200 PCC={pcc:.6f}")
    assert passing


@pytest.mark.slow
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_full_sp_tp_ep_2x2(mesh_device):
    """Combined SP=2 + TP=2 + EP vs EP-only reference on 2×2 mesh."""
    mesh_device.enable_program_cache()
    ccl = mesh_ccl(mesh_device)
    c = transformer_cfg()
    B, S = 1, 256
    torch.manual_seed(0)
    x = torch.randn(B, S, c["H"]) * 0.05
    m = causal_mask(S)
    y_ref = run_embeds_forward(mesh_device, build_mesh_model(mesh_device, ccl, sp_factor=1, tp_factor=1), x, S, m)
    y_full = run_embeds_forward(mesh_device, build_mesh_model(mesh_device, ccl, sp_factor=2, tp_factor=2), x, S, m)
    passing, pcc = comp_pcc(y_ref, y_full, PCC_BLOCK)
    logger.info(f"FULL sp2+tp2+EP vs EP-only PCC={pcc:.6f}")
    assert passing
