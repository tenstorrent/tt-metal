# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Stage 3 -- LoRA fine-tune the Wan2.2 A14B MoE with a flow-matching loss."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import ttnn
from torch.utils.data import DataLoader

import ttml
from ttml.models.wan2_2 import (
    WanConfig,
    WanTransformer3D,
    build_rope_params,
    load_expert_from_safetensors,
    patchify,
    patchify_output_order,
)

from pipeline_config import SUBFOLDER, Config
from utils.dataset import LatentEmbedDataset, make_collate_fn
from utils.device_setup import setup_device
from utils.logger import Logger
from utils.lora_export import save_all
from utils.lora_targets import resolve as resolve_lora_targets


def _to_ttml(arr: torch.Tensor | np.ndarray, dtype=ttnn.bfloat16):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().float().cpu().numpy()
    return ttml.autograd.Tensor.from_numpy(np.ascontiguousarray(arr, dtype=np.float32), ttnn.Layout.TILE, dtype)


def _loss_value(loss) -> float:
    return float(np.asarray(loss.to_numpy()).reshape(-1)[0])


def latent_shape(cfg: Config) -> tuple:
    """(B, C, F, H, W) of the VAE latent the DiT consumes: 8x spatial, 4x temporal."""
    return (
        cfg.BATCH,
        WanConfig.in_channels,
        (cfg.TRAIN_FRAMES - 1) // 4 + 1,
        cfg.TRAIN_H // 8,
        cfg.TRAIN_W // 8,
    )


def model_config_for(cfg: Config) -> WanConfig:
    return WanConfig(
        runner_type=(
            ttml.models.RunnerType.MemoryEfficient if cfg.GRADIENT_CHECKPOINTING else ttml.models.RunnerType.Default
        ),
    )


def build_lora_expert(role: str, cfg: Config) -> ttml.modules.LoraModel:
    sub = SUBFOLDER[role]
    print(f"[lora] loading {role}-noise expert ({sub}) from {cfg.MODEL_ID} ...")

    model_config = model_config_for(cfg)
    model = WanTransformer3D(model_config)
    load_expert_from_safetensors(model, cfg.MODEL_ID, subfolder=sub)

    lora_config = ttml.modules.LoraConfig(
        rank=cfg.LORA_RANK,
        alpha=float(cfg.LORA_ALPHA),
        target_modules=resolve_lora_targets(cfg.LORA_TARGET_SET),
        lora_dropout=0.0,
        use_rslora=False,
        verbose=True,
    )
    lora_model = ttml.modules.LoraModel(model, lora_config)

    all_params = lora_model.parameters()
    trainable = {name: p for name, p in all_params.items() if "lora" in name}
    print(f"[lora] {role}: {len(trainable)} LoRA params trainable, {len(all_params) - len(trainable)} frozen")
    if not trainable:
        raise RuntimeError("LoRA injection produced no trainable parameters")
    return lora_model


def _range_for(cfg: Config) -> tuple[float, float]:
    if cfg.TRAIN_EXPERTS == "low":
        return 0.0, cfg.BOUNDARY_RATIO
    if cfg.TRAIN_EXPERTS == "high":
        return cfg.BOUNDARY_RATIO, 1.0
    return 0.0, 1.0


def _sample_timestep(cfg: Config, lo: float, hi: float, generator: torch.Generator | None = None) -> torch.Tensor:
    shift = cfg.TRAIN_FLOW_SHIFT
    while True:
        u = torch.sigmoid(torch.randn(1, generator=generator) * cfg.LOGNORM_STD + cfg.LOGNORM_MEAN)
        t = shift * u / (1.0 + (shift - 1.0) * u)
        if lo <= float(t) < hi:
            return t


def _route(t: float, experts: dict, cfg: Config):
    if len(experts) == 1:
        return next(iter(experts.values()))
    return experts["high"] if t >= cfg.BOUNDARY_RATIO else experts["low"]


def flow_matching_step(
    model,
    batch,
    cfg: Config,
    t: torch.Tensor,
    rope_params,
    patch_size: tuple,
    fixed_noise: torch.Tensor | None = None,
):
    x0 = batch["latent"].float()
    sigma = t.view(x0.shape[0], *([1] * (x0.dim() - 1)))
    noise = torch.randn_like(x0) if fixed_noise is None else fixed_noise.float()
    x_t = (1.0 - sigma) * x0 + sigma * noise
    target = noise - x0

    # The model works in patch space. Patchify the target in proj_out's order so the loss
    # needs no permutation on device; MSE is unchanged by a consistent reordering.
    tokens = patchify(x_t.numpy(), patch_size)
    target_tokens = patchify_output_order(target.numpy(), patch_size)

    text_embed = batch["text_embed"].float().numpy()
    text_embed = text_embed.reshape(text_embed.shape[0], 1, *text_embed.shape[-2:])

    timesteps = (t * 1000.0).flatten().tolist()

    pred = model(_to_ttml(tokens), timesteps, _to_ttml(text_embed), rope_params)
    return ttml.ops.loss.mse_loss(pred, _to_ttml(target_tokens), reduce=ttml.ops.ReduceType.MEAN)


def validation_loss(experts, val_loader: DataLoader, cfg: Config, ctx, rope_params, patch_size: tuple) -> float:
    for m in experts.values():
        m.eval()
    ctx.set_gradient_mode(ttml.autograd.GradMode.DISABLED)
    lo, hi = _range_for(cfg)
    losses = []
    try:
        for batch in val_loader:
            idx = int(batch["idx"][0])
            g = torch.Generator().manual_seed(cfg.SEED + idx)
            t = _sample_timestep(cfg, lo, hi, generator=g)
            noise = torch.randn(batch["latent"].shape, generator=g)
            model = _route(float(t), experts, cfg)
            losses.append(
                _loss_value(flow_matching_step(model, batch, cfg, t, rope_params, patch_size, fixed_noise=noise))
            )
            ctx.reset_graph()
    finally:
        ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
        for m in experts.values():
            m.train()
    return float(np.mean(losses)) if losses else float("nan")


def train(cfg: Config) -> None:
    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    cache = Path(cfg.CACHE_DIR)
    embeds_path = cache / "embeds.pt"
    if not embeds_path.exists() or not (cache / "samples").exists():
        raise FileNotFoundError(f"missing cache at {cache} — run `precompute` first.")

    metadata = json.loads((cache / "metadata.json").read_text())
    all_idx = sorted(m["idx"] for m in metadata)
    if len(all_idx) <= cfg.VAL_HOLDOUT:
        raise RuntimeError(f"need > {cfg.VAL_HOLDOUT} samples; got {len(all_idx)}")
    val_idx, train_idx = all_idx[-cfg.VAL_HOLDOUT :], all_idx[: -cfg.VAL_HOLDOUT]
    print(f"[train] {len(train_idx)} train / {len(val_idx)} val | experts={cfg.TRAIN_EXPERTS}")

    dp_size, tp_size = cfg.MESH_SHAPE
    if tp_size > 1:
        # Every projection in ttml.models.wan2_2 is a plain LinearLayer, so weights would be
        # replicated on all TP devices rather than sharded -- no memory benefit and the LoRA
        # export path would gather shards that do not exist. Needs Column/RowParallelLinear.
        raise NotImplementedError(
            f"the Wan port is single-device for now: TP={tp_size} is not supported because its "
            f"projections are non-parallel LinearLayer. Use device.mesh_shape: [{dp_size}, 1]."
        )
    if cfg.GRAD_CLIP > 0.0 and tp_size > 1:
        # clip_grad_norm is per-device, so at TP>1 it would clip shard-local norms.
        raise ValueError(
            f"GRAD_CLIP={cfg.GRAD_CLIP} is not supported at TP={tp_size}: "
            f"ttml.core.clip_grad_norm is per-device, so the clip would apply to "
            f"shard-local norms instead of the global one. Set optimizer.grad_clip: 0, "
            f"or use device.mesh_shape: [{dp_size}, 1]."
        )

    ctx, _device = setup_device(dp_size, tp_size, seed=cfg.SEED)

    embeds = torch.load(embeds_path, weights_only=False)
    train_loader = DataLoader(
        LatentEmbedDataset(cfg.CACHE_DIR, train_idx),
        batch_size=cfg.BATCH,
        shuffle=True,
        collate_fn=make_collate_fn(embeds, cfg.TEXT_DROP_PROB, cfg.SEED),
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        LatentEmbedDataset(cfg.CACHE_DIR, val_idx),
        batch_size=1,
        shuffle=False,
        collate_fn=make_collate_fn(embeds, 0.0, cfg.SEED + 1),
        num_workers=0,
    )

    experts = {role: build_lora_expert(role, cfg) for role in cfg.experts_to_load()}
    for m in experts.values():
        m.train()

    trainable = {}
    for role, model in experts.items():
        for name, param in model.parameters().items():
            if "lora" in name:
                trainable[f"{role}/{name}"] = param
    adamw_config = ttml.optimizers.AdamWConfig.make(cfg.LR, 0.9, 0.999, 1e-8, cfg.WEIGHT_DECAY)
    optimizer = ttml.optimizers.AdamW(trainable, adamw_config)
    print(f"[train] constant lr={cfg.LR}, {len(trainable)} trainable tensors")

    # RoPE tables are constants for a given latent shape, so build them once.
    patch_size = model_config_for(cfg).patch_size
    shape = latent_shape(cfg)
    rope_params = build_rope_params(
        head_dim=model_config_for(cfg).head_dim,
        patch_size=patch_size,
        latent_shape=shape,
        max_seq_len=model_config_for(cfg).rope_max_seq_len,
    )
    print(f"[train] latent {shape} patch {patch_size} -> {rope_params.sequence_length} tokens")

    lo, hi = _range_for(cfg)
    run_name = f"wan22_14b_{cfg.STYLE.lower()}_{cfg.TRAIN_EXPERTS}_r{cfg.LORA_RANK}"
    logger = Logger(cfg.WANDB_ENABLED, cfg.WANDB_PROJECT, run_name, cfg.asdict())

    global_step, micro = 0, 0
    accum_loss, accum_n = 0.0, 0
    ema = None
    step_start = time.time()
    data_iter = iter(train_loader)
    optimizer.zero_grad()
    print(f"[train] loop: max_steps={cfg.MAX_STEPS} accum={cfg.GRAD_ACCUM} t-range=[{lo:.3f},{hi:.3f})")

    while global_step < cfg.MAX_STEPS:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        t = _sample_timestep(cfg, lo, hi)
        model = _route(float(t), experts, cfg)
        loss = flow_matching_step(model, batch, cfg, t, rope_params, patch_size)
        accum_loss += _loss_value(loss)
        accum_n += 1

        if cfg.GRAD_ACCUM > 1:
            loss = loss * (1.0 / float(cfg.GRAD_ACCUM))
        loss.backward(False)
        ctx.reset_graph()
        micro += 1

        if micro % cfg.GRAD_ACCUM == 0:
            if dp_size > 1:
                ttml.core.distributed.synchronize_gradients(model.parameters())
            if cfg.GRAD_CLIP > 0.0:
                ttml.core.clip_grad_norm(model.parameters(), cfg.GRAD_CLIP, 2.0, False)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            avg = accum_loss / accum_n
            ema = avg if ema is None else 0.9 * ema + 0.1 * avg
            dt = time.time() - step_start
            step_start = time.time()
            logger.log(
                {"train/loss": avg, "train/loss_ema": ema, "train/step_time_s": dt},
                step=global_step,
            )
            accum_loss, accum_n = 0.0, 0

            if cfg.VAL_LOSS_EVERY and global_step % cfg.VAL_LOSS_EVERY == 0:
                vloss = validation_loss(experts, val_loader, cfg, ctx, rope_params, patch_size)
                logger.log({"val/loss": vloss}, step=global_step)
                print(f"[train] step {global_step}: val/loss={vloss:.4f}")

            if cfg.CKPT_EVERY and global_step % cfg.CKPT_EVERY == 0:
                save_all(experts, cfg, suffix=f"_step{global_step:05d}")

    save_all(experts, cfg)
    print(f"[train] done at step {global_step}. LoRA(s): {', '.join(cfg.expert_path(r) for r in experts)}")
    logger.finish()
