# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for I2I device PCC tests and demo."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[5]
HUNYUAN_UPSTREAM = os.environ.get("HUNYUAN_UPSTREAM")
for p in (str(ROOT),):
    if p not in sys.path:
        sys.path.insert(0, p)
if HUNYUAN_UPSTREAM and HUNYUAN_UPSTREAM not in sys.path:
    sys.path.insert(0, HUNYUAN_UPSTREAM)

from models.experimental.hunyuan_image_3_0.ref.attention.rope_2d import build_batch_2d_rope
from models.experimental.hunyuan_image_3_0.ref.image_gen.patch_embed import UNetDown as RefDown, UNetUp as RefUp
from models.experimental.hunyuan_image_3_0.ref.image_gen.timestep_embedder import TimestepEmbedder as RefTimeEmbed
from models.experimental.hunyuan_image_3_0.ref.image_processor import HunyuanImage3ImageProcessor
from models.experimental.hunyuan_image_3_0.ref.model_config import IMAGE_BASE_SIZE, load_config, transformer_cfg
from models.experimental.hunyuan_image_3_0.ref.tokenizer import (
    HunyuanTokenizer,
    build_i2i_cfg_conds,
    prepare_i2i_denoise_bundle,
    scatter_distill_step_embeds,
)
from models.experimental.hunyuan_image_3_0.ref.transformer_layer import HunyuanImage3DecoderLayer as RefLayer
from models.experimental.hunyuan_image_3_0.ref.weights import INSTRUCT_MODEL_DIR, MODEL_DIR

PROMPT = "a cat on a mat"
I2I_IMAGE_SIZE = int(os.environ.get("HY_I2I_SIZE", str(IMAGE_BASE_SIZE)))
NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "2"))
CFG_FACTOR = int(os.environ.get("HY_CFG", "1"))
VIT_LAYERS = int(os.environ.get("HY_VIT_LAYERS", "1"))
I2I_WEIGHTS = (
    INSTRUCT_MODEL_DIR
    if os.environ.get("HY_I2I_INSTRUCT", "1") != "0" and (INSTRUCT_MODEL_DIR / "model.safetensors.index.json").is_file()
    else MODEL_DIR
)
USE_INSTRUCT_I2I = I2I_WEIGHTS == INSTRUCT_MODEL_DIR

_WMAP = json.load(open(I2I_WEIGHTS / "model.safetensors.index.json"))["weight_map"]
_OPEN: dict = {}
_REF_LAYERS: dict = {}


def has_weights():
    return (I2I_WEIGHTS / "model.safetensors.index.json").is_file()


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def load_tensor(key):
    shard = _WMAP[key]
    f = _OPEN.get(shard) or _OPEN.setdefault(shard, safe_open(I2I_WEIGHTS / shard, framework="pt"))
    return f.get_tensor(key)


def load_prefix(prefix):
    return {k[len(prefix) + 1 :]: load_tensor(k) for k in _WMAP if k.startswith(prefix + ".")}


def model_cfg():
    """I2I/recaption helper dims from checkpoint config (legacy key names preserved)."""
    d = transformer_cfg(load_config(I2I_WEIGHTS))
    return dict(
        H=d["H"],
        HEADS=d["HEADS"],
        KV_HEADS=d["KV"],
        HEAD_DIM=d["HD"],
        NUM_EXPERTS=d["E"],
        MOE_TOPK=d["K"],
        MOE_INTER=d["MOE_INTER"],
        NUM_SHARED=d["NUM_SHARED"],
        NORM_TOPK=d["NORM_TOPK"],
        EPS=d["EPS"],
        USE_QK_NORM=d["QKN"],
        USE_MIXED=d["MIXED"],
    )


def pe_dims(down_sd):
    hid, latent = down_sd["model.0.weight"].shape[:2]
    hsz = down_sd["model.1.in_layers.2.weight"].shape[0]
    return int(latent), int(hid), int(hsz)


def ref_time_embed(prefix, hidden, timesteps):
    sd = load_prefix(prefix)
    te = RefTimeEmbed(hidden).eval()
    te.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    with torch.no_grad():
        return te(timesteps)


def ref_timestep_emb(hidden):
    sd = load_prefix("timestep_emb")
    te = RefTimeEmbed(hidden).eval()
    te.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    return te


def ref_layer(c, i):
    if i not in _REF_LAYERS:
        sd = load_prefix(f"model.layers.{i}")
        layer = RefLayer(
            hidden_size=c["H"],
            num_attention_heads=c["HEADS"],
            num_key_value_heads=c["KV_HEADS"],
            attention_head_dim=c["HEAD_DIM"],
            num_experts=c["NUM_EXPERTS"],
            moe_topk=c["MOE_TOPK"],
            moe_intermediate_size=c["MOE_INTER"],
            num_shared_expert=c["NUM_SHARED"],
            use_mixed_mlp_moe=c["USE_MIXED"],
            norm_topk_prob=c["NORM_TOPK"],
            use_qk_norm=c["USE_QK_NORM"],
            rms_norm_eps=c["EPS"],
            layer_idx=i,
        )
        layer.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
        layer.eval()
        _REF_LAYERS[i] = layer
    return _REF_LAYERS[i]


def processor():
    return HunyuanImage3ImageProcessor(json.load(open(I2I_WEIGHTS / "config.json")))


def rgb_image(size=I2I_IMAGE_SIZE):
    return Image.new("RGB", (size, size), color=(128, 64, 32))


def build_host_bundle(*, cfg_factor=CFG_FACTOR, vit_layers=VIT_LAYERS, seed=42):
    from models.experimental.hunyuan_image_3_0.ref.image_gen.model_loaders import (
        load_aligner,
        load_patch_embed,
        load_siglip2_vision,
        load_timestep_embedder,
    )
    from models.experimental.hunyuan_image_3_0.ref.weights import load_tensors

    if USE_INSTRUCT_I2I:
        from models.experimental.hunyuan_image_3_0.ref.system_prompt import get_system_prompt

        tok = HunyuanTokenizer.from_model_dir(I2I_WEIGHTS, sequence_template="instruct")
        system_prompt = get_system_prompt("en_unified", "image")
        sequence_template = "instruct"
    else:
        tok = HunyuanTokenizer.from_pretrained()
        system_prompt = None
        sequence_template = "pretrain"
    proc = processor()
    cond, _ = proc.get_image_with_size(rgb_image(), return_type="vae_vit")
    wte = load_tensors(I2I_WEIGHTS, ["model.wte.weight"])["model.wte.weight"]
    gen = torch.Generator().manual_seed(seed)
    bundle = prepare_i2i_denoise_bundle(
        tok,
        PROMPT,
        cond,
        proc,
        wte,
        patch_embed=load_patch_embed(I2I_WEIGHTS),
        time_embed=load_timestep_embedder("time_embed", I2I_WEIGHTS),
        timestep_emb=load_timestep_embedder("timestep_emb", I2I_WEIGHTS),
        vision_model=load_siglip2_vision(I2I_WEIGHTS, num_layers=vit_layers),
        aligner=load_aligner(I2I_WEIGHTS),
        image_size=I2I_IMAGE_SIZE,
        cfg_factor=cfg_factor,
        sequence_template=sequence_template,
        system_prompt=system_prompt,
        generator=gen,
    )
    cond_dict, uncond_dict = build_i2i_cfg_conds(bundle, wte, proc)
    row = cond_dict
    return dict(
        bundle=bundle,
        cond=cond_dict,
        uncond=uncond_dict,
        wte=wte,
        seq_len=bundle.seq_len,
        img_slice=row["gen_slice"],
        grid_hw=row["gen_hw"],
        image_infos=cond_dict["image_infos"],
        mask_add=cond_dict["attention_mask"],
    )


def prepare_step_host_embeds(
    cond,
    t_scalar,
    timestep_emb,
    *,
    scheduler=None,
    guidance_emb=None,
    timestep_r_emb=None,
    guidance_scalar=None,
    use_meanflow=False,
):
    host = cond["base_embeds_host"]
    t_r = None
    if use_meanflow and scheduler is not None:
        t_r = float(scheduler.get_timestep_r(t_scalar))
    return scatter_distill_step_embeds(
        host,
        t_scalar=float(t_scalar),
        gen_timestep_scatter_index=cond.get("gen_timestep_scatter_index"),
        timestep_emb=timestep_emb,
        guidance_scalar=guidance_scalar,
        guidance_scatter_index=cond.get("guidance_scatter_index"),
        guidance_emb=guidance_emb,
        t_r_scalar=t_r,
        gen_timestep_r_scatter_index=cond.get("gen_timestep_r_scatter_index"),
        timestep_r_emb=timestep_r_emb,
    )


def upload_mask(device, mask_add, batch=1):
    import ttnn

    s = mask_add.shape[-1]
    return ttnn.from_torch(
        mask_add.reshape(batch, 1, s, s).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def upload_base_embeds(device, embeds):
    import ttnn

    return ttnn.from_torch(embeds.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def upload_loop_cond(device, cond):
    import ttnn

    s = cond["attention_mask"].shape[-1]
    b = cond["batch"]
    return dict(
        cond,
        attention_mask=ttnn.from_torch(
            cond["attention_mask"].reshape(b, 1, s, s).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        ),
    )


def ref_i2i_step(
    c,
    latent,
    t_scalar,
    cond,
    down_sd,
    up_sd,
    image_infos,
    mask_add,
    img_slice,
    timestep_emb,
    *,
    num_layers=NUM_LAYERS,
):
    h_dim = c["H"]
    latent_ch, hid, hsz = pe_dims(down_sd)
    ref_down = RefDown(1, latent_ch, hsz, hid, hsz).eval()
    ref_up = RefUp(1, hsz, hsz, hid, latent_ch, out_norm=True).eval()
    ref_down.load_state_dict({k: v.float() for k, v in down_sd.items()}, strict=True)
    ref_up.load_state_dict({k: v.float() for k, v in up_sd.items()}, strict=True)

    t_emb1 = ref_time_embed("time_embed", h_dim, torch.tensor([float(t_scalar)]))
    t_emb2 = ref_time_embed("time_embed_2", h_dim, torch.tensor([float(t_scalar)]))
    cos, sin = build_batch_2d_rope(cond["base_embeds_host"].shape[1], c["HEAD_DIM"], image_infos=image_infos)

    with torch.no_grad():
        img_tokens, th, tw = ref_down(latent, t_emb1)
        h = prepare_step_host_embeds(cond, t_scalar, timestep_emb)
        h[:, img_slice, :] = img_tokens
        for i in range(num_layers):
            h = ref_layer(c, i)(h, attention_mask=mask_add, custom_pos_emb=(cos, sin))
        pred = ref_up(h[:, img_slice, :], t_emb2, th, tw)
    return pred
