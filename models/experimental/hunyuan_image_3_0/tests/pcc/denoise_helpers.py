# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers for denoise PCC tests (host-routed step + multi-step loop).

from __future__ import annotations

import gc
import os
import time

import torch
import ttnn

from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive
from models.experimental.hunyuan_image_3_0.ref.attention.rope_2d import build_batch_2d_rope
from models.experimental.hunyuan_image_3_0.ref.image_gen.patch_embed import UNetDown as RefDown, UNetUp as RefUp
from models.experimental.hunyuan_image_3_0.ref.image_gen.timestep_embedder import TimestepEmbedder as RefTimeEmbed
from models.experimental.hunyuan_image_3_0.ref.transformer_layer import HunyuanImage3DecoderLayer as RefLayer
from models.experimental.hunyuan_image_3_0.ref.weights import load_prefixed_state_dict, resolve_base_model_dir
from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown, HunyuanTtUNetUp
from models.experimental.hunyuan_image_3_0.tt.image_gen.timestep_embedder import HunyuanTtTimestepEmbedder
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel
from models.experimental.hunyuan_image_3_0.tt.pipeline import HunyuanTtDenoiseStep, denoise_loop
from models.experimental.hunyuan_image_3_0.tt.scheduler import HunyuanTtScheduler
from models.tt_dit.parallel.manager import CCLManager
from pipeline_helpers import patch_embed_dims

_REF_LAYER_CACHE: dict[int, RefLayer] = {}


def num_layers_step() -> int:
    return int(os.environ.get("HY_NUM_LAYERS", "4"))


def num_layers_loop() -> int:
    return int(os.environ.get("HY_NUM_LAYERS", "2"))


def denoise_steps() -> int:
    return int(os.environ.get("HY_STEPS", "3"))


def host_step_pcc_threshold(num_layers: int) -> float:
    return 0.99 if num_layers <= 8 else 0.85


def loop_pcc_threshold() -> float:
    return 0.99


def resident_loop_pcc_threshold() -> float:
    return 0.98


def production_loop_pcc_threshold(num_layers: int, steps: int) -> float:
    """PCC gate for multi-step denoise at production GRID=64.

    Single-step 32L uses ~0.85; error compounds across steps, so the full
    50-step schedule uses a looser default (override with HY_DENOISE_LOOP_PCC).
    """
    if env := os.environ.get("HY_DENOISE_LOOP_PCC"):
        return float(env)
    if num_layers <= 8:
        return 0.95 if steps >= 10 else 0.99
    if steps >= 50:
        return 0.70
    if steps >= 10:
        return 0.75
    return 0.85


def _layer_loader(i: int) -> dict[str, torch.Tensor]:
    sd = load_prefixed_state_dict(resolve_base_model_dir(), f"model.layers.{i}.")
    return {f"model.layers.{i}.{k}": v for k, v in sd.items()}


def clear_ref_layer_cache() -> None:
    _REF_LAYER_CACHE.clear()
    gc.collect()


def _make_ref_layer(c: dict, i: int) -> RefLayer:
    sd = load_prefixed_state_dict(resolve_base_model_dir(), f"model.layers.{i}.")
    layer = RefLayer(
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
        layer_idx=i,
    )
    layer.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    layer.eval()
    return layer


def cached_ref_layer(c: dict, i: int) -> RefLayer:
    if i not in _REF_LAYER_CACHE:
        _REF_LAYER_CACHE[i] = _make_ref_layer(c, i)
    return _REF_LAYER_CACHE[i]


def _forward_ref_layers(
    c: dict,
    h: torch.Tensor,
    num_layers: int,
    mask_add: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    stream_layers: bool,
) -> torch.Tensor:
    """Run ref decoder layers, optionally loading one MoE layer at a time to bound host RAM."""
    s = int(h.shape[1])
    t_all = time.time()
    for li in range(num_layers):
        if stream_layers:
            # Progress is intentional: 32L @ S=4160 is ~5–10 min with no other stdout,
            # so a quiet loop looks hung after "[backbone ref] stream_layers=True".
            t0 = time.time()
            print(f"[ref layers] {li + 1}/{num_layers} S={s} load+fwd...", flush=True)
            layer = _make_ref_layer(c, li)
            h = layer(h, attention_mask=mask_add, custom_pos_emb=(cos, sin))
            del layer
            gc.collect()
            print(f"[ref layers] {li + 1}/{num_layers} done in {time.time() - t0:.1f}s", flush=True)
        else:
            h = cached_ref_layer(c, li)(h, attention_mask=mask_add, custom_pos_emb=(cos, sin))
    if stream_layers:
        print(f"[ref layers] all {num_layers} done in {time.time() - t_all:.1f}s", flush=True)
        gc.collect()
    return h


def reference_host_step(
    c: dict,
    layout: dict,
    num_layers: int,
    latent: torch.Tensor,
    t_emb1: torch.Tensor,
    t_emb2: torch.Tensor,
    text_embeds: torch.Tensor,
    down_sd: dict,
    up_sd: dict,
    batch: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid = layout["grid"]
    s = layout["seq_len"]
    img_slice = layout["img_slice"]
    latent_ch, hid, hsz = patch_embed_dims(down_sd)

    ref_down = RefDown(1, latent_ch, hsz, hid, hsz).eval()
    ref_up = RefUp(1, hsz, hsz, hid, latent_ch, out_norm=True).eval()
    ref_down.load_state_dict({k: v.float() for k, v in down_sd.items()}, strict=True)
    ref_up.load_state_dict({k: v.float() for k, v in up_sd.items()}, strict=True)

    cos, sin = build_batch_2d_rope(s, c["HD"], image_infos=[[(img_slice, (grid, grid))]])
    mask_add = to_additive(build_attention_mask(s, image_slices=[img_slice], bsz=batch), dtype=torch.float32)

    with torch.no_grad():
        img_tokens, th, tw = ref_down(latent, t_emb1)
        h = text_embeds.clone()
        h[:, img_slice, :] = img_tokens
        h = _forward_ref_layers(c, h, num_layers, mask_add, cos, sin, stream_layers=(num_layers > 8))
        pred = ref_up(h[:, img_slice, :], t_emb2, th, tw)
    return pred, img_tokens


def run_host_routed_step_tt(
    device,
    c: dict,
    layout: dict,
    num_layers: int,
    latent: torch.Tensor,
    t_emb1: torch.Tensor,
    t_emb2: torch.Tensor,
    text_embeds: torch.Tensor,
    down_sd: dict,
    up_sd: dict,
    batch: int = 1,
) -> torch.Tensor:
    grid = layout["grid"]
    s = layout["seq_len"]
    img_slice = layout["img_slice"]
    n_img = layout["n_img"]
    h = c["H"]
    latent_ch, hid, hsz = patch_embed_dims(down_sd)

    t1_tt = ttnn.from_torch(t_emb1.reshape(1, 1, batch, h), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    t2_tt = ttnn.from_torch(t_emb2.reshape(1, 1, batch, h), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_down = HunyuanTtUNetDown(
        device,
        {f"patch_embed.{k}": v for k, v in down_sd.items()},
        in_channels=latent_ch,
        hidden_channels=hid,
        out_channels=hsz,
    )
    img_tok_tt, h2, w2 = tt_down(latent, t1_tt)
    img_tok = ttnn.to_torch(img_tok_tt).reshape(batch, n_img, h)
    img_tok_tt.deallocate(True)

    embeds = text_embeds.clone()
    embeds[:, img_slice, :] = img_tok
    backbone = HunyuanTtModel(
        device,
        num_layers=num_layers,
        hidden_size=h,
        num_heads=c["HEADS"],
        num_kv_heads=c["KV"],
        head_dim=c["HD"],
        num_experts=c["E"],
        moe_topk=c["K"],
        use_qk_norm=c["QKN"],
        use_mixed_mlp_moe=c["MIXED"],
        norm_topk_prob=c["NORM_TOPK"],
        rms_norm_eps=c["EPS"],
        stream_experts=True,
        layer_loader=_layer_loader,
        apply_final_norm=False,
    )
    image_infos = [[(img_slice, (grid, grid))]]
    mask_add = to_additive(build_attention_mask(s, image_slices=[img_slice], bsz=batch), dtype=torch.float32)
    mask_tt = ttnn.from_torch(
        mask_add.reshape(batch, 1, s, s), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    embeds_tt = ttnn.from_torch(embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    hidden_tt = backbone.forward(inputs_embeds=embeds_tt, seq_len=s, image_infos=image_infos, attention_mask=mask_tt)
    hidden = ttnn.to_torch(hidden_tt)[..., :h]
    hidden_tt.deallocate(True)
    mask_tt.deallocate(True)
    embeds_tt.deallocate(True)

    img_out = hidden[:, img_slice, :].reshape(1, 1, n_img, h)
    img_out_tt = ttnn.from_torch(img_out, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_up = HunyuanTtUNetUp(
        device,
        {f"final_layer.{k}": v for k, v in up_sd.items()},
        in_channels=hsz,
        hidden_channels=hid,
        out_channels=latent_ch,
    )
    pred_tt, h3, w3 = tt_up(img_out_tt, t2_tt, h2, w2, B=batch)
    pred = ttnn.to_torch(pred_tt).reshape(batch, h3, w3, latent_ch).permute(0, 3, 1, 2)
    pred_tt.deallocate(True)
    img_out_tt.deallocate(True)
    t1_tt.deallocate(True)
    t2_tt.deallocate(True)
    return pred


def reference_loop(
    c: dict,
    layout: dict,
    num_layers: int,
    init_latent: torch.Tensor,
    text_embeds: torch.Tensor,
    down_sd: dict,
    up_sd: dict,
    steps: int,
    batch: int = 1,
    *,
    stream_layers: bool | None = None,
    progress: bool = False,
) -> torch.Tensor:
    """CPU reference denoise loop.

    For 32L MoE, caching all layers OOMs (~60GB+). Default stream_layers=True when
    num_layers > 8 so only one layer's weights are resident at a time.
    """
    if stream_layers is None:
        stream_layers = num_layers > 8
    if stream_layers:
        clear_ref_layer_cache()

    grid = layout["grid"]
    s = layout["seq_len"]
    img_slice = layout["img_slice"]
    latent_ch, hid, hsz = patch_embed_dims(down_sd)

    sched = HunyuanTtScheduler(None)
    sched.set_timesteps(steps)
    sigmas, timesteps = sched.sigmas, sched.timesteps

    ref_down = RefDown(1, latent_ch, hsz, hid, hsz).eval()
    ref_up = RefUp(1, hsz, hsz, hid, latent_ch, out_norm=True).eval()
    ref_down.load_state_dict({k: v.float() for k, v in down_sd.items()}, strict=True)
    ref_up.load_state_dict({k: v.float() for k, v in up_sd.items()}, strict=True)
    te1 = RefTimeEmbed(c["H"]).eval()
    te1.load_state_dict(
        {k: v.float() for k, v in load_prefixed_state_dict(resolve_base_model_dir(), "time_embed.").items()},
        strict=True,
    )
    te2 = RefTimeEmbed(c["H"]).eval()
    te2.load_state_dict(
        {k: v.float() for k, v in load_prefixed_state_dict(resolve_base_model_dir(), "time_embed_2.").items()},
        strict=True,
    )

    cos, sin = build_batch_2d_rope(s, c["HD"], image_infos=[[(img_slice, (grid, grid))]])
    mask_add = to_additive(build_attention_mask(s, image_slices=[img_slice], bsz=batch), dtype=torch.float32)

    lat = init_latent.clone()
    for i, t in enumerate(timesteps):
        if progress:
            print(f"[ref denoise] step {i + 1}/{steps} t={float(t):.4f} stream_layers={stream_layers}", flush=True)
        tvec = torch.tensor([float(t)] * batch)
        with torch.no_grad():
            e1, e2 = te1(tvec), te2(tvec)
            img_tokens, th, tw = ref_down(lat, e1)
            h = text_embeds.clone()
            h[:, img_slice, :] = img_tokens
            h = _forward_ref_layers(c, h, num_layers, mask_add, cos, sin, stream_layers=stream_layers)
            pred = ref_up(h[:, img_slice, :], e2, th, tw)
        lat = lat + float(sigmas[i + 1] - sigmas[i]) * pred
    if stream_layers:
        clear_ref_layer_cache()
    return lat


def build_loop_step(
    device,
    c: dict,
    layout: dict,
    num_layers: int,
    down_sd: dict,
    up_sd: dict,
    *,
    mesh: bool = False,
) -> tuple[HunyuanTtDenoiseStep, HunyuanTtTimestepEmbedder, HunyuanTtTimestepEmbedder]:
    grid = layout["grid"]
    s = layout["seq_len"]
    img_slice = layout["img_slice"]
    h = c["H"]
    latent_ch, hid, hsz = patch_embed_dims(down_sd)

    patch_embed = HunyuanTtUNetDown(
        device,
        {f"patch_embed.{k}": v for k, v in down_sd.items()},
        in_channels=latent_ch,
        hidden_channels=hid,
        out_channels=hsz,
    )
    final_layer = HunyuanTtUNetUp(
        device,
        {f"final_layer.{k}": v for k, v in up_sd.items()},
        in_channels=hsz,
        hidden_channels=hid,
        out_channels=latent_ch,
    )
    backbone_kwargs = dict(
        num_layers=num_layers,
        hidden_size=h,
        num_heads=c["HEADS"],
        num_kv_heads=c["KV"],
        head_dim=c["HD"],
        num_experts=c["E"],
        moe_topk=c["K"],
        use_qk_norm=c["QKN"],
        use_mixed_mlp_moe=c["MIXED"],
        norm_topk_prob=c["NORM_TOPK"],
        rms_norm_eps=c["EPS"],
        stream_experts=not mesh,
        layer_loader=_layer_loader,
        apply_final_norm=False,
    )
    if mesh:
        ccl = CCLManager(device, num_links=1, topology=ttnn.Topology.Linear)
        backbone_kwargs.update(
            weight_dtype=ttnn.bfloat8_b,
            ccl_manager=ccl,
            expert_mesh_axis=1,
            tp_axis=1,
            tp_factor=2,
            sp_axis=0,
            sp_factor=2,
        )
    backbone = HunyuanTtModel(device, **backbone_kwargs)
    step = HunyuanTtDenoiseStep(
        device,
        patch_embed=patch_embed,
        backbone=backbone,
        final_layer=final_layer,
        img_slice=img_slice,
        grid_hw=(grid, grid),
        seq_len=s,
    )
    te_sd = lambda p: {f"{p}.{k}": v for k, v in load_prefixed_state_dict(resolve_base_model_dir(), f"{p}.").items()}
    time_embed = HunyuanTtTimestepEmbedder(device, h, te_sd("time_embed"), "time_embed")
    time_embed_2 = HunyuanTtTimestepEmbedder(device, h, te_sd("time_embed_2"), "time_embed_2")
    return step, time_embed, time_embed_2


def loop_cond_tensors(
    device, layout: dict, text_embeds: torch.Tensor, mask_add: torch.Tensor, batch: int = 1, replicate: bool = False
):
    img_start = layout["img_start"]
    n_img = layout["n_img"]
    s = layout["seq_len"]
    img_slice = layout["img_slice"]
    grid = layout["grid"]

    def _to_tt(t):
        kwargs = dict(
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if replicate:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
        return ttnn.from_torch(t, **kwargs)

    return dict(
        text_pre=_to_tt(text_embeds[:, :img_start, :]),
        text_post=_to_tt(text_embeds[:, img_start + n_img :, :]),
        image_infos=[[(img_slice, (grid, grid))]],
        attention_mask=_to_tt(mask_add.reshape(batch, 1, s, s)),
        batch=batch,
    )


def run_denoise_loop_tt(
    device,
    layout: dict,
    num_layers: int,
    init_latent: torch.Tensor,
    text_embeds: torch.Tensor,
    steps: int,
    c: dict,
    down_sd: dict,
    up_sd: dict,
    mesh: bool = False,
) -> torch.Tensor:
    img_slice = layout["img_slice"]
    s = layout["seq_len"]
    batch = init_latent.shape[0]

    mask_add = to_additive(build_attention_mask(s, image_slices=[img_slice], bsz=batch), dtype=torch.float32)
    step, time_embed, time_embed_2 = build_loop_step(device, c, layout, num_layers, down_sd, up_sd, mesh=mesh)
    cond = loop_cond_tensors(device, layout, text_embeds, mask_add, batch, replicate=mesh)

    sched = HunyuanTtScheduler(device)
    sched.set_timesteps(steps)
    kwargs = dict(time_embed=time_embed, time_embed_2=time_embed_2, cond=cond)
    if mesh:
        kwargs["mesh_device"] = device
    return denoise_loop(step, sched, init_latent.clone(), **kwargs)
