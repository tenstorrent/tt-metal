# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared builders for Hunyuan denoise Tracy perf tests (2×2 resident mesh)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive
from models.experimental.hunyuan_image_3_0.ref.image_gen.timestep_embedder import TimestepEmbedder as RefTimeEmbed
from models.experimental.hunyuan_image_3_0.ref.weights import ensure_base_weights
from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown, HunyuanTtUNetUp
from models.experimental.hunyuan_image_3_0.tt.image_gen.timestep_embedder import HunyuanTtTimestepEmbedder
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel
from models.experimental.hunyuan_image_3_0.tt.pipeline import HunyuanTtDenoiseStep
from models.tt_dit.parallel.manager import CCLManager

REGION_SIGNPOSTS: dict[str, tuple[str, str]] = {
    "patch_embed": ("start_patch_embed", "stop_patch_embed"),
    "scatter": ("start_scatter", "stop_scatter"),
    "backbone": ("start_backbone", "stop_backbone"),
    "final_layer": ("start_final_layer", "stop_final_layer"),
    "full": ("start_denoise_step", "stop_denoise_step"),
}


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v is not None else default


def denoise_grid() -> int:
    return env_int("HY_DENOISE_GRID", 8)


def denoise_num_layers() -> int:
    return env_int("HY_NUM_LAYERS", 4)


def denoise_perf_iters() -> int:
    return env_int("HY_DENOISE_PERF_ITERS", 3)


def denoise_perf_warmup() -> int:
    return env_int("HY_DENOISE_PERF_WARMUP", 1)


def denoise_resident_temb() -> bool:
    """Use TimestepEmbedder WIDTH_SHARDED M=32 t_emb (HY_DENOISE_RESIDENT_TEMB=0 to disable)."""
    return os.environ.get("HY_DENOISE_RESIDENT_TEMB", "1") != "0"


class _Checkpoint:
    def __init__(self, weights_dir: Path):
        self.weights_dir = weights_dir
        index_path = weights_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            index_matches = list(weights_dir.glob("*.index.json"))
            if not index_matches:
                raise FileNotFoundError(f"No safetensors index under {weights_dir}")
            index_path = index_matches[0]
        self._wmap = json.load(open(index_path))["weight_map"]
        self._open: dict[str, object] = {}

    def load(self, key: str) -> torch.Tensor:
        shard = self._wmap[key]
        handle = self._open.get(shard)
        if handle is None:
            handle = safe_open(self.weights_dir / shard, framework="pt")
            self._open[shard] = handle
        return handle.get_tensor(key)

    def load_prefix(self, prefix: str) -> dict[str, torch.Tensor]:
        needle = prefix + "."
        return {k[len(needle) :]: self.load(k) for k in self._wmap if k.startswith(needle)}


def model_cfg(ckpt: _Checkpoint) -> dict:
    cfg = json.load(open(ckpt.weights_dir / "config.json"))
    first = lambda v: v if isinstance(v, int) else v[0]
    return dict(
        H=cfg["hidden_size"],
        HEADS=cfg["num_attention_heads"],
        KV=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        HD=cfg.get("attention_head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
        E=first(cfg["num_experts"]),
        K=first(cfg["moe_topk"]),
        MI=first(cfg["moe_intermediate_size"]),
        NSH=first(cfg["num_shared_expert"]),
        NORM=cfg.get("norm_topk_prob", True),
        MIXED=cfg.get("use_mixed_mlp_moe", True),
        QKN=cfg.get("use_qk_norm", True),
        EPS=cfg.get("rms_norm_eps", 1e-5),
    )


def patch_embed_dims(down_sd: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    hid, latent = down_sd["model.0.weight"].shape[:2]
    hsz = down_sd["model.1.in_layers.2.weight"].shape[0]
    return int(latent), int(hid), int(hsz)


def host_time_embed(ckpt: _Checkpoint, prefix: str, hidden: int, timesteps: torch.Tensor) -> torch.Tensor:
    te = RefTimeEmbed(hidden).eval()
    te.load_state_dict({k: v.float() for k, v in ckpt.load_prefix(prefix).items()}, strict=True)
    with torch.no_grad():
        return te(timesteps)


@dataclass
class DenoisePerfRuntime:
    mesh_device: object
    batch: int
    grid: int
    seq_len: int
    img_slice: slice
    latent_ch: int
    hidden_size: int
    latent: torch.Tensor
    patch_embed: HunyuanTtUNetDown
    final_layer: HunyuanTtUNetUp
    backbone: HunyuanTtModel
    step: HunyuanTtDenoiseStep
    t_emb1: ttnn.Tensor
    t_emb2: ttnn.Tensor
    text_pre: ttnn.Tensor
    text_post: ttnn.Tensor
    attention_mask: ttnn.Tensor
    image_infos: list

    @property
    def n_img(self) -> int:
        return self.grid * self.grid


def build_denoise_perf_runtime(mesh_device) -> DenoisePerfRuntime:
    """Resident 2×2 denoise stack matching ``test_pipeline_step_resident.py``."""
    weights_dir = ensure_base_weights()
    ckpt = _Checkpoint(weights_dir)
    c = model_cfg(ckpt)
    grid = denoise_grid()
    text_pre_len = env_int("HY_DENOISE_TEXT_PRE", 32)
    text_post_len = env_int("HY_DENOISE_TEXT_POST", 32)
    n_img = grid * grid
    img_start = text_pre_len
    img_slice = slice(img_start, img_start + n_img)
    seq_len = text_pre_len + n_img + text_post_len
    num_layers = denoise_num_layers()
    batch = 1

    down_sd = ckpt.load_prefix("patch_embed")
    up_sd = ckpt.load_prefix("final_layer")
    latent_ch, hid_ch, hsz = patch_embed_dims(down_sd)
    hidden = c["H"]

    torch.manual_seed(0)
    latent = torch.randn(batch, latent_ch, grid, grid)
    text_embeds = torch.randn(batch, seq_len, hidden) * 0.02
    timesteps = torch.rand(batch)

    def _rep(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    mesh_device.enable_program_cache()

    patch_embed = HunyuanTtUNetDown(
        mesh_device,
        {f"patch_embed.{k}": v for k, v in down_sd.items()},
        in_channels=latent_ch,
        hidden_channels=hid_ch,
        out_channels=hsz,
    )
    final_layer = HunyuanTtUNetUp(
        mesh_device,
        {f"final_layer.{k}": v for k, v in up_sd.items()},
        in_channels=hsz,
        hidden_channels=hid_ch,
        out_channels=latent_ch,
    )

    if denoise_resident_temb():
        # Match pipeline.py: WIDTH_SHARDED M=32 t_emb sharded for ResBlock emb_layers.
        te_sd = lambda p: {f"{p}.{k}": v for k, v in ckpt.load_prefix(p).items()}
        time_embed = HunyuanTtTimestepEmbedder(mesh_device, hidden, te_sd("time_embed"), "time_embed")
        time_embed_2 = HunyuanTtTimestepEmbedder(mesh_device, hidden, te_sd("time_embed_2"), "time_embed_2")
        tvec = ttnn.from_torch(
            timesteps.reshape(1, 1, batch, 1).float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        t_emb1 = time_embed.forward(tvec, keep_resident=True, resident_next_n=2 * patch_embed.resblock.out_channels)
        t_emb2 = time_embed_2.forward(tvec, keep_resident=True, resident_next_n=2 * final_layer.resblock.out_channels)
        ttnn.deallocate(tvec)
        logger.info(
            f"[denoise perf] resident t_emb: sharded={ttnn.is_sharded(t_emb1)} "
            f"shape={list(t_emb1.shape)} next_n=({2 * patch_embed.resblock.out_channels}, "
            f"{2 * final_layer.resblock.out_channels})"
        )
    else:
        t1 = host_time_embed(ckpt, "time_embed", hidden, timesteps)
        t2 = host_time_embed(ckpt, "time_embed_2", hidden, timesteps)
        t_emb1 = _rep(t1.reshape(1, 1, batch, hidden))
        t_emb2 = _rep(t2.reshape(1, 1, batch, hidden))
        logger.info("[denoise perf] interleaved host t_emb (HY_DENOISE_RESIDENT_TEMB=0)")
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    layer_loader = lambda i, _c=ckpt: {
        f"model.layers.{i}.{k}": v for k, v in _c.load_prefix(f"model.layers.{i}").items()
    }
    backbone = HunyuanTtModel(
        mesh_device,
        num_layers=num_layers,
        hidden_size=hidden,
        num_heads=c["HEADS"],
        num_kv_heads=c["KV"],
        head_dim=c["HD"],
        num_experts=c["E"],
        moe_topk=c["K"],
        use_qk_norm=c["QKN"],
        use_mixed_mlp_moe=c["MIXED"],
        norm_topk_prob=c["NORM"],
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
    step = HunyuanTtDenoiseStep(
        mesh_device,
        patch_embed=patch_embed,
        backbone=backbone,
        final_layer=final_layer,
        img_slice=img_slice,
        grid_hw=(grid, grid),
        seq_len=seq_len,
    )

    mask_add = to_additive(build_attention_mask(seq_len, image_slices=[img_slice], bsz=batch), dtype=torch.float32)
    image_infos = [[(img_slice, (grid, grid))]]

    logger.info(
        f"[denoise perf] grid={grid}x{grid} seq_len={seq_len} layers={num_layers} "
        f"text_pre={text_pre_len} text_post={text_post_len}"
    )

    return DenoisePerfRuntime(
        mesh_device=mesh_device,
        batch=batch,
        grid=grid,
        seq_len=seq_len,
        img_slice=img_slice,
        latent_ch=latent_ch,
        hidden_size=hidden,
        latent=latent,
        patch_embed=patch_embed,
        final_layer=final_layer,
        backbone=backbone,
        step=step,
        t_emb1=t_emb1,
        t_emb2=t_emb2,
        text_pre=_rep(text_embeds[:, :img_start, :]),
        text_post=_rep(text_embeds[:, img_start + n_img :, :]),
        attention_mask=_rep(mask_add.reshape(batch, 1, seq_len, seq_len)),
        image_infos=image_infos,
    )


def _run_patch_embed(rt: DenoisePerfRuntime) -> tuple[ttnn.Tensor, int, int]:
    img_tok, th, tw = rt.patch_embed(rt.latent, rt.t_emb1)
    return img_tok, th, tw


def _run_scatter(rt: DenoisePerfRuntime, img_tok: ttnn.Tensor) -> ttnn.Tensor:
    rt.step.backbone_batch = rt.batch
    return rt.step._scatter(img_tok, rt.text_pre, rt.text_post)


def _run_backbone(rt: DenoisePerfRuntime, seq: ttnn.Tensor) -> ttnn.Tensor:
    return rt.backbone.forward(
        inputs_embeds=seq,
        seq_len=rt.seq_len,
        image_infos=rt.image_infos,
        attention_mask=rt.attention_mask,
    )


def _run_final_layer(rt: DenoisePerfRuntime, hidden: ttnn.Tensor, th: int, tw: int) -> ttnn.Tensor:
    img_out = ttnn.slice(
        hidden,
        [0, rt.img_slice.start, 0],
        [rt.batch, rt.img_slice.stop, rt.hidden_size],
    )
    img_out = ttnn.reshape(img_out, [1, 1, rt.n_img, rt.hidden_size])
    pred, _, _ = rt.final_layer(img_out, rt.t_emb2, th, tw, B=rt.batch)
    ttnn.deallocate(img_out, force=False)
    return pred


def _run_full_step(rt: DenoisePerfRuntime) -> ttnn.Tensor:
    return rt.step(
        rt.latent,
        text_pre=rt.text_pre,
        text_post=rt.text_post,
        t_emb1=rt.t_emb1,
        t_emb2=rt.t_emb2,
        image_infos=rt.image_infos,
        attention_mask=rt.attention_mask,
        batch=rt.batch,
    )


def warmup_region(rt: DenoisePerfRuntime, region: str, *, iters: int) -> None:
    """Program-cache warmup for one region (outside Tracy signposts)."""
    if region == "patch_embed":
        for _ in range(iters):
            img_tok, _, _ = _run_patch_embed(rt)
            ttnn.deallocate(img_tok, force=False)
    elif region == "scatter":
        img_tok, _, _ = _run_patch_embed(rt)
        for _ in range(iters):
            seq = _run_scatter(rt, img_tok)
            ttnn.deallocate(seq, force=False)
        ttnn.deallocate(img_tok, force=False)
    elif region == "backbone":
        img_tok, _, _ = _run_patch_embed(rt)
        seq = _run_scatter(rt, img_tok)
        ttnn.deallocate(img_tok, force=False)
        for _ in range(iters):
            hidden = _run_backbone(rt, seq)
            ttnn.deallocate(hidden, force=False)
        ttnn.deallocate(seq, force=False)
    elif region == "final_layer":
        img_tok, th, tw = _run_patch_embed(rt)
        seq = _run_scatter(rt, img_tok)
        hidden = _run_backbone(rt, seq)
        ttnn.deallocate(img_tok, force=False)
        ttnn.deallocate(seq, force=False)
        for _ in range(iters):
            pred = _run_final_layer(rt, hidden, th, tw)
            ttnn.deallocate(pred, force=False)
        ttnn.deallocate(hidden, force=False)
    elif region == "full":
        for _ in range(iters):
            pred = _run_full_step(rt)
            ttnn.deallocate(pred, force=False)
    else:
        raise ValueError(f"unknown denoise perf region: {region}")
    ttnn.synchronize_device(rt.mesh_device)


def profile_denoise_region(
    rt: DenoisePerfRuntime,
    region: str,
    *,
    warmup: int | None = None,
    iters: int | None = None,
    start_signpost: str | None = None,
    stop_signpost: str | None = None,
) -> None:
    """Warm up then time ``iters`` executions of ``region`` inside Tracy signposts."""
    from tracy import signpost

    warmup = denoise_perf_warmup() if warmup is None else warmup
    iters = denoise_perf_iters() if iters is None else iters
    default_start, default_stop = REGION_SIGNPOSTS[region]
    start_signpost = start_signpost or default_start
    stop_signpost = stop_signpost or default_stop

    logger.info(
        f"[denoise perf] region={region} warmup={warmup} iters={iters} signposts={start_signpost}/{stop_signpost}"
    )
    # Drop any cached DRAM matmul programs from prior runs / code versions before warmup.
    rt.mesh_device.disable_and_clear_program_cache()
    rt.mesh_device.enable_program_cache()
    warmup_region(rt, region, iters=warmup)

    # Build setup tensors OUTSIDE the timed window so Tracy regions match
    # pipeline.py (especially backbone / final_layer MoE vs UNetUp isolation).
    if region == "patch_embed":
        signpost(start_signpost)
        for _ in range(iters):
            img_tok, _, _ = _run_patch_embed(rt)
            ttnn.deallocate(img_tok, force=False)
        signpost(stop_signpost)
    elif region == "scatter":
        img_tok, _, _ = _run_patch_embed(rt)
        signpost(start_signpost)
        for _ in range(iters):
            seq = _run_scatter(rt, img_tok)
            ttnn.deallocate(seq, force=False)
        signpost(stop_signpost)
        ttnn.deallocate(img_tok, force=False)
    elif region == "backbone":
        img_tok, _, _ = _run_patch_embed(rt)
        seq = _run_scatter(rt, img_tok)
        ttnn.deallocate(img_tok, force=False)
        signpost(start_signpost)
        for _ in range(iters):
            hidden = _run_backbone(rt, seq)
            ttnn.deallocate(hidden, force=False)
        signpost(stop_signpost)
        ttnn.deallocate(seq, force=False)
    elif region == "final_layer":
        img_tok, th, tw = _run_patch_embed(rt)
        seq = _run_scatter(rt, img_tok)
        hidden = _run_backbone(rt, seq)
        ttnn.deallocate(img_tok, force=False)
        ttnn.deallocate(seq, force=False)
        signpost(start_signpost)
        for _ in range(iters):
            pred = _run_final_layer(rt, hidden, th, tw)
            ttnn.deallocate(pred, force=False)
        signpost(stop_signpost)
        ttnn.deallocate(hidden, force=False)
    elif region == "full":
        signpost(start_signpost)
        for _ in range(iters):
            pred = _run_full_step(rt)
            ttnn.deallocate(pred, force=False)
        signpost(stop_signpost)
    else:
        raise ValueError(f"unknown denoise perf region: {region}")
    ttnn.synchronize_device(rt.mesh_device)
    logger.info(f"[denoise perf] done region={region}")
