# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PCC: ttnn vision submodules in `dots_visionTT.py` vs the matching module from
`modeling_dots_vision.py` (dynamic load from the HF transformers_modules checkpoint tree).

Reference attention uses the checkpoint's ``VisionSdpaAttention`` (matches the torch SDPA
path used in ``DotsAttnQkvprojTt``). Weights: local ``AutoModelForCausalLM`` checkpoint
(``DOTS_OCR_MODEL_PATH``).

Environment:
  - DOTS_OCR_MODEL_PATH: Dots-OCR (config + safetensors)
  - DOTS_OCR_CHECKPOINT_VISION_DIR: dir with ``modeling_dots_vision.py`` and ``configuration_dots.py``
  - DOTS_VISION_PCC_REQUIRED / DOTS_VISION_BLOCK_PCC: min PCC (default 0.97)
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from loguru import logger

import ttnn
from models.common.rmsnorm import RMSNorm as TtRmsNorm
from models.common.utility_functions import comp_pcc
from models.demos.dots_ocr.tt.dots_visionTT import (
    DotsAttnQkvprojTt,
    DotsMlpTt,
    DotsPatchMergerTt,
    DotsVisionBlockTt,
    DotsVisionTtConfig,
    _w128,
)
from models.tt_transformers.tt.common import Mode

_CHECKPOINT_PKG = "tt_metal_dots_vision_checkpoint"


def _default_model_dir() -> Path:
    override = os.environ.get("DOTS_OCR_MODEL_PATH")
    if override:
        return Path(override).resolve()
    p = Path(__file__).resolve().parent
    for _ in range(8):
        candidate = p / "reference" / "dots_ocr"
        if candidate.is_dir():
            return candidate
        if p.parent == p:
            break
        p = p.parent
    return Path(__file__).resolve().parent.parent.parent / "reference" / "dots_ocr"


def _install_flash_attn_import_stub() -> None:
    if "flash_attn" in sys.modules:
        return
    m = types.ModuleType("flash_attn")
    m.flash_attn_varlen_func = None  # type: ignore[assignment]
    try:
        from flash_attn import flash_attn_varlen_func  # type: ignore[import]

        m.flash_attn_varlen_func = flash_attn_varlen_func
    except ImportError:

        def _flash_unavailable(*_a, **_k):  # noqa: ANN001, ANN002
            raise RuntimeError("flash-attn is not installed; use sdpa or eager attention in tests only.")

        m.flash_attn_varlen_func = _flash_unavailable
    sys.modules["flash_attn"] = m


def _checkpoint_vision_dir() -> Path:
    p = os.environ.get(
        "DOTS_OCR_CHECKPOINT_VISION_DIR",
        "/home/ubuntu/.cache/huggingface/modules/transformers_modules/checkpoints",
    )
    return Path(p).resolve()


def _is_checkpoint_vision_dir_ok(ck: Path) -> bool:
    return (ck / "modeling_dots_vision.py").exists() and (ck / "configuration_dots.py").exists()


def _load_modeling_dots_vision() -> Any:
    mname = f"{_CHECKPOINT_PKG}.modeling_dots_vision"
    if mname in sys.modules:
        return sys.modules[mname]

    _install_flash_attn_import_stub()
    ck = _checkpoint_vision_dir()
    if not (ck / "modeling_dots_vision.py").exists() or not (ck / "configuration_dots.py").exists():
        raise FileNotFoundError(
            f"Expected modeling_dots_vision.py and configuration_dots.py under {ck} "
            "(set DOTS_OCR_CHECKPOINT_VISION_DIR or install checkpoint files)"
        )

    if _CHECKPOINT_PKG in sys.modules:
        pkg = sys.modules[_CHECKPOINT_PKG]
    else:
        pkg = types.ModuleType(_CHECKPOINT_PKG)
        pkg.__path__ = [str(ck)]
        sys.modules[_CHECKPOINT_PKG] = pkg

    cfg_name = f"{_CHECKPOINT_PKG}.configuration_dots"
    if cfg_name not in sys.modules:
        spec_c = importlib.util.spec_from_file_location(
            cfg_name, ck / "configuration_dots.py", submodule_search_locations=[str(ck)]
        )
        mod_c = importlib.util.module_from_spec(spec_c)
        mod_c.__package__ = _CHECKPOINT_PKG
        mod_c.__name__ = f"{_CHECKPOINT_PKG}.configuration_dots"
        sys.modules[cfg_name] = mod_c
        spec_c.loader.exec_module(mod_c)

    spec = importlib.util.spec_from_file_location(
        mname, ck / "modeling_dots_vision.py", submodule_search_locations=[str(ck)]
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = _CHECKPOINT_PKG
    mod.__name__ = mname
    sys.modules[mname] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_vision_config_from_checkpoint_config_class(vision_dict: dict) -> Any:
    _load_modeling_dots_vision()
    m = importlib.import_module(f"{_CHECKPOINT_PKG}.configuration_dots")
    return m.DotsVisionConfig(**vision_dict)


LAYER = 0


def _pcc_required() -> float:
    return float(os.environ.get("DOTS_VISION_BLOCK_PCC", os.environ.get("DOTS_VISION_PCC_REQUIRED", "0.97")))


def _weights_available(model_dir: Path) -> bool:
    if not (model_dir / "config.json").exists():
        return False
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        return any(model_dir.glob("model-*-of-*.safetensors"))
    return (model_dir / "model.safetensors").exists()


def _require_checkpoint_modules() -> None:
    if not _is_checkpoint_vision_dir_ok(_checkpoint_vision_dir()):
        pytest.skip(
            f"Checkpoint modeling not found at {_checkpoint_vision_dir()}. "
            "Set DOTS_OCR_CHECKPOINT_VISION_DIR to a folder with modeling_dots_vision.py and configuration_dots.py."
        )
    _load_modeling_dots_vision()


def _ckpt_vision_config(hf_vision) -> Any:
    d = {k: v for k, v in hf_vision.to_dict().items() if not k.startswith("_")}
    d.pop("model_type", None)
    return _load_vision_config_from_checkpoint_config_class(d)


def _tt_cfg(vision) -> DotsVisionTtConfig:
    return DotsVisionTtConfig(
        embed_dim=vision.embed_dim,
        num_hidden_layers=vision.num_hidden_layers,
        num_attention_heads=vision.num_attention_heads,
        intermediate_size=vision.intermediate_size,
        spatial_merge_size=vision.spatial_merge_size,
        rms_norm_eps=vision.rms_norm_eps,
        use_bias=vision.use_bias,
        post_norm=bool(vision.post_norm),
        hidden_size=vision.hidden_size,
        patch_size=vision.patch_size,
        num_channels=vision.num_channels,
    )


def _prepare_ttnn_input(emb: torch.Tensor, mesh_device) -> tuple[ttnn.Tensor, int, int]:
    s, d = int(emb.shape[0]), int(emb.shape[1])
    s_pad = _w128(s)
    p = emb.new_zeros(s_pad, d) if s_pad > s else None
    if p is not None:
        p[:s] = emb
        tile = p
    else:
        tile = emb
    tt = ttnn.from_torch(
        tile.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
    )
    tt = ttnn.to_memory_config(tt, ttnn.DRAM_MEMORY_CONFIG)
    return tt, s, s_pad


def _tt_out_to_seq2d(
    ttn: ttnn.Tensor,
    seqlen: int,
    dim: int,
) -> torch.Tensor:
    t = ttnn.to_torch(ttn)
    if t.dim() == 4:
        o = t[0, 0, :seqlen, :dim]
    else:
        o = t[0, :seqlen, :dim]
    if o.is_nested:
        raise RuntimeError("Unexpected nested output tensor")
    return o.float()


def _assert_pcc(
    a: torch.Tensor,
    b: torch.Tensor,
    msg: str,
) -> None:
    r = _pcc_required()
    ok, message = comp_pcc(a, b, r)
    logger.info(f"{msg}: {message}")
    assert ok, f"{msg} (required {r}): {message}"


# --- HF model + state (shared pattern with test_dots_vision_tt_pcc) ---


def _load_hf_vision_state():
    model_dir = _default_model_dir()
    if not _weights_available(model_dir):
        pytest.skip(
            f"Dots-OCR weights not found under {model_dir} "
            "(config.json + model-*.safetensors). Set DOTS_OCR_MODEL_PATH."
        )
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    hf_model.eval()
    vision = hf_model.vision_tower.to(dtype=torch.bfloat16)
    state_dict = hf_model.state_dict()
    return hf_model, vision, state_dict


# mesh_device param: match test_dots_vision_tt_pcc
_MESH_PARAM = {
    "N150": (1, 1),
    "N300": (1, 2),
    "T3K": (1, 8),
    "TG": (8, 4),
}.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
_MESH_MARK = [_MESH_PARAM]


# ------------------------ tests: ttnn (mesh) ------------------------


@torch.inference_mode()
@pytest.mark.parametrize("mesh_device", [pytest.param(_MESH_MARK[0], id="mesh")], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_dots_vision_mlp_pcc_vs_checkpoint(mesh_device) -> None:
    _require_checkpoint_modules()
    m = _load_modeling_dots_vision()
    hf, vision, sd = _load_hf_vision_state()
    vc = _ckpt_vision_config(hf.config.vision_config)
    ffn = m.DotsSwiGLUFFN(vc).bfloat16().eval()
    ffn.load_state_dict(
        {
            k.replace(f"vision_tower.blocks.{LAYER}.mlp.", ""): v
            for k, v in sd.items()
            if k.startswith(f"vision_tower.blocks.{LAYER}.mlp.")
        }
    )
    pfx = f"vision_tower.blocks.{LAYER}.mlp."
    tt_mlp = DotsMlpTt(mesh_device, sd, pfx, _tt_cfg(hf.config.vision_config), None)
    torch.manual_seed(2023)
    s, d = 16, vc.embed_dim
    x = torch.randn(s, d, dtype=torch.bfloat16, device="cpu")
    y_ref = ffn(x)
    ttn, seqlen, _ = _prepare_ttnn_input(x, mesh_device)
    y_tt = tt_mlp(ttn)
    y_t = _tt_out_to_seq2d(y_tt, seqlen, d)
    _assert_pcc(y_ref.cpu().float(), y_t, "MLP (DotsSwiGLU)")


@torch.inference_mode()
@pytest.mark.parametrize("mesh_device", [pytest.param(_MESH_MARK[0], id="mesh")], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_dots_vision_attention_pcc_vs_checkpoint(mesh_device) -> None:
    _require_checkpoint_modules()
    m = _load_modeling_dots_vision()
    hf, vision, sd = _load_hf_vision_state()
    vcfg = _ckpt_vision_config(hf.config.vision_config)
    attn = (
        m.VisionSdpaAttention(
            vcfg, vcfg.embed_dim, num_heads=vision.config.num_attention_heads, bias=vision.config.use_bias
        )
        .bfloat16()
        .eval()
    )
    attn.load_state_dict(
        {
            k.replace(f"vision_tower.blocks.{LAYER}.attn.", ""): v
            for k, v in sd.items()
            if k.startswith(f"vision_tower.blocks.{LAYER}.attn.")
        }
    )
    pfx = f"vision_tower.blocks.{LAYER}.attn."
    ttatt = DotsAttnQkvprojTt(mesh_device, sd, pfx, _tt_cfg(hf.config.vision_config), None)
    grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int32)  # 4 spatial tokens
    s = 4
    d = vcfg.embed_dim
    vit = m.DotsVisionTransformer(vcfg)
    rpe = vit.rot_pos_emb(grid_thw)[:s].bfloat16()
    cu = torch.tensor([0, s], dtype=torch.int32, device="cpu")
    torch.manual_seed(2)
    x = torch.randn(s, d, dtype=torch.bfloat16, device="cpu")
    y_ref = attn(x, cu_seqlens=cu, rotary_pos_emb=rpe)
    ttn, seqlen, _ = _prepare_ttnn_input(x, mesh_device)
    y_tt = ttatt(ttn, rpe, cu, seqlen)
    y_t = _tt_out_to_seq2d(y_tt, seqlen, d)
    _assert_pcc(y_ref.cpu().float(), y_t, "VisionSdpaAttention (QKV+proj)")


@torch.inference_mode()
@pytest.mark.parametrize("mesh_device", [pytest.param(_MESH_MARK[0], id="mesh")], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_dots_vision_patch_merger_pcc_vs_checkpoint(mesh_device) -> None:
    _require_checkpoint_modules()
    m = _load_modeling_dots_vision()
    hf, vision, sd = _load_hf_vision_state()
    vcfg = _ckpt_vision_config(hf.config.vision_config)
    merger = (
        m.PatchMerger(
            dim=vcfg.hidden_size,
            context_dim=vcfg.embed_dim,
            spatial_merge_size=vcfg.spatial_merge_size,
            pre_norm="layernorm",
            init_merger_std=vcfg.init_merger_std,
        )
        .bfloat16()
        .eval()
    )
    merger.load_state_dict(
        {k.replace("vision_tower.merger.", ""): v for k, v in sd.items() if k.startswith("vision_tower.merger.")}
    )
    pmerg = DotsPatchMergerTt(mesh_device, sd, "vision_tower.merger.", _tt_cfg(hf.config.vision_config), None)
    merge2 = vcfg.spatial_merge_size**2
    s = 8
    d = vcfg.embed_dim
    assert s % merge2 == 0, "test grid uses token count multiple of merge^2"
    torch.manual_seed(11)
    x2d = torch.randn(s, d, dtype=torch.bfloat16, device="cpu")
    y_ref = merger(x2d)
    ttn, seqlen, _ = _prepare_ttnn_input(x2d, mesh_device)
    y_tt = pmerg(ttn, seqlen)
    t = ttnn.to_torch(y_tt)
    if t.dim() == 5:
        s_merge = seqlen // merge2
        y_t = t[0, 0, 0, :s_merge, : vcfg.hidden_size].float()
    elif t.dim() == 4:
        s_merge = seqlen // merge2
        y_t = t[0, 0, :s_merge, : vcfg.hidden_size].float()
    else:
        raise RuntimeError(f"Unexpected merger rank {t.dim()}")
    _assert_pcc(y_ref.cpu().float().reshape(s_merge, -1), y_t, "PatchMerger")


@torch.inference_mode()
@pytest.mark.parametrize("mesh_device", [pytest.param(_MESH_MARK[0], id="mesh")], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_dots_vision_rms_norm1_pcc_vs_checkpoint(mesh_device) -> None:
    _require_checkpoint_modules()
    m = _load_modeling_dots_vision()
    hf, _, sd = _load_hf_vision_state()
    vcfg = _ckpt_vision_config(hf.config.vision_config)
    d = vcfg.embed_dim
    nmt = m.RMSNorm(d, eps=vcfg.rms_norm_eps).bfloat16().eval()
    p = f"vision_tower.blocks.{LAYER}.norm1."
    nmt.load_state_dict({k.replace(p, ""): v for k, v in sd.items() if k.startswith(p)})
    sp = f"vision_tower.blocks.{LAYER}."
    tt_r = TtRmsNorm(
        device=mesh_device,
        dim=d,
        state_dict=sd,
        state_dict_prefix=sp,
        weight_key="norm1",
        weight_cache_path=None,
        weight_dtype=ttnn.bfloat16,
        eps=vcfg.rms_norm_eps,
    )
    torch.manual_seed(7)
    s = 4
    x2d = torch.randn(s, d, dtype=torch.bfloat16, device="cpu")
    y_ref = nmt(x2d)
    ttn, seqlen, _ = _prepare_ttnn_input(x2d, mesh_device)
    ttn0 = ttnn.to_memory_config(ttn, ttnn.DRAM_MEMORY_CONFIG)
    nout = tt_r(ttn0, mode=Mode.PREFILL)
    y_t = _tt_out_to_seq2d(nout, seqlen, d)
    ttnn.deallocate(nout)
    _assert_pcc(y_ref.cpu().float(), y_t, "RMSNorm (norm1)")


@torch.inference_mode()
@pytest.mark.parametrize("mesh_device", [pytest.param(_MESH_MARK[0], id="mesh")], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_dots_vision_rms_norm2_pcc_vs_checkpoint(mesh_device) -> None:
    _require_checkpoint_modules()
    m = _load_modeling_dots_vision()
    hf, _, sd = _load_hf_vision_state()
    vcfg = _ckpt_vision_config(hf.config.vision_config)
    d = vcfg.embed_dim
    nmt = m.RMSNorm(d, eps=vcfg.rms_norm_eps).bfloat16().eval()
    p = f"vision_tower.blocks.{LAYER}.norm2."
    nmt.load_state_dict({k.replace(p, ""): v for k, v in sd.items() if k.startswith(p)})
    sp = f"vision_tower.blocks.{LAYER}."
    tt_r = TtRmsNorm(
        device=mesh_device,
        dim=d,
        state_dict=sd,
        state_dict_prefix=sp,
        weight_key="norm2",
        weight_cache_path=None,
        weight_dtype=ttnn.bfloat16,
        eps=vcfg.rms_norm_eps,
    )
    torch.manual_seed(19)
    s = 4
    x2d = torch.randn(s, d, dtype=torch.bfloat16, device="cpu")
    y_ref = nmt(x2d)
    ttn, seqlen, _ = _prepare_ttnn_input(x2d, mesh_device)
    ttn0 = ttnn.to_memory_config(ttn, ttnn.DRAM_MEMORY_CONFIG)
    nout = tt_r(ttn0, mode=Mode.PREFILL)
    y_t = _tt_out_to_seq2d(nout, seqlen, d)
    ttnn.deallocate(nout)
    _assert_pcc(y_ref.cpu().float(), y_t, "RMSNorm (norm2)")


@torch.inference_mode()
@pytest.mark.parametrize("mesh_device", [pytest.param(_MESH_MARK[0], id="mesh")], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_dots_vision_block_pcc_vs_checkpoint(mesh_device) -> None:
    _require_checkpoint_modules()
    m = _load_modeling_dots_vision()
    hf, vision, sd = _load_hf_vision_state()
    vcfg = _ckpt_vision_config(hf.config.vision_config)
    blk = m.DotsVisionBlock(vcfg, attn_implementation="sdpa").bfloat16().eval()
    blk.load_state_dict(
        {
            k.replace(f"vision_tower.blocks.{LAYER}.", ""): v
            for k, v in sd.items()
            if k.startswith(f"vision_tower.blocks.{LAYER}.")
        }
    )
    ttb = DotsVisionBlockTt(
        LAYER,
        mesh_device,
        sd,
        "vision_tower.",
        _tt_cfg(hf.config.vision_config),
        None,
    )
    grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int32)
    s, d = 4, vcfg.embed_dim
    vit = m.DotsVisionTransformer(vcfg)
    rpe = vit.rot_pos_emb(grid_thw)[:s].bfloat16()
    cu = torch.tensor([0, s], dtype=torch.int32, device="cpu")
    torch.manual_seed(5)
    h = torch.randn(s, d, dtype=torch.bfloat16, device="cpu")
    y_ref = blk(h, cu_seqlens=cu, rotary_pos_emb=rpe)
    ttn, seqlen, _ = _prepare_ttnn_input(h, mesh_device)
    y_tt = ttb(ttn, rpe, cu, seqlen)
    y_t = _tt_out_to_seq2d(y_tt, seqlen, d)
    _assert_pcc(y_ref.cpu().float(), y_t, f"DotsVisionBlock (layer {LAYER}, sdpa)")


@torch.inference_mode()
@pytest.mark.parametrize("mesh_device", [pytest.param(_MESH_MARK[0], id="mesh")], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_dots_vision_post_trunk_rms_pcc_vs_checkpoint(mesh_device) -> None:
    _require_checkpoint_modules()
    m = _load_modeling_dots_vision()
    hf, _, sd = _load_hf_vision_state()
    vcfg = _ckpt_vision_config(hf.config.vision_config)
    if not vcfg.post_norm or f"vision_tower.post_trunk_norm.weight" not in sd:
        pytest.skip("post_trunk_norm not in this checkpoint")
    d = vcfg.embed_dim
    nmt = m.RMSNorm(d, eps=vcfg.rms_norm_eps).bfloat16().eval()
    nmt.load_state_dict(
        {
            k.replace("vision_tower.post_trunk_norm.", ""): v
            for k, v in sd.items()
            if k.startswith("vision_tower.post_trunk_norm.")
        }
    )
    tt_r = TtRmsNorm(
        device=mesh_device,
        dim=d,
        state_dict=sd,
        state_dict_prefix="vision_tower.",
        weight_key="post_trunk_norm",
        weight_cache_path=None,
        weight_dtype=ttnn.bfloat16,
        eps=vcfg.rms_norm_eps,
    )
    torch.manual_seed(31)
    s = 8
    x2d = torch.randn(s, d, dtype=torch.bfloat16, device="cpu")
    y_ref = nmt(x2d)
    ttn, seqlen, _ = _prepare_ttnn_input(x2d, mesh_device)
    nout = tt_r(ttnn.to_memory_config(ttn, ttnn.DRAM_MEMORY_CONFIG), mode=Mode.PREFILL)
    y_t = _tt_out_to_seq2d(nout, seqlen, d)
    ttnn.deallocate(nout)
    _assert_pcc(y_ref.cpu().float(), y_t, "RMSNorm (post_trunk)")
