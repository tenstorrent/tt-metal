# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared test helpers: PCC and the host-side GLM indexer stems (used to re-derive the pooled-key
index from HF layer inputs and compare it with the device index cache)."""

from __future__ import annotations

import torch


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() != b.numel():
        raise ValueError(f"pcc shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    if torch.isnan(b).any():
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def hf_index_stems(raw: dict[str, torch.Tensor], cfg, x_normed: torch.Tensor):
    """Host k (LayerNorm'd), gate logits and the ape of the indexer for normalized attention inputs
    [S, 4096]; ``raw`` is the layer's raw checkpoint dict (prefix stripped)."""
    sd = {k[len("self_attn.indexer.") :]: v for k, v in raw.items() if k.startswith("self_attn.indexer.")}
    k = torch.nn.functional.layer_norm(
        x_normed.float() @ sd["wk.weight"].float().t(),
        (cfg.index_head_dim,),
        sd["k_norm.weight"].float(),
        sd["k_norm.bias"].float(),
        1e-6,
    )
    g = x_normed.float() @ sd["index_kpool_compress_gate"].float().t()
    return k, g, sd["index_kpool_compress_ape"].float()


def hf_pool(k: torch.Tensor, g: torch.Tensor, ape: torch.Tensor, P: int) -> torch.Tensor:
    """[n*P, D] -> [n, D] HF k-pool compression over complete pools."""
    n = k.shape[0] // P
    kp, gp = k[: n * P].reshape(n, P, -1), g[: n * P].reshape(n, P, -1)
    prob = torch.softmax(gp + ape[None], dim=1)
    return (prob * kp).sum(1)


def attention_input(raw: dict[str, torch.Tensor], cfg, streams: torch.Tensor) -> torch.Tensor:
    """HF attn_hc collapse + input_layernorm on mHC streams [1, S, 4, 4096] -> [S, 4096] (bf16 as the model)."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection, Glm5NextTextRMSNorm

    from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
    from models.demos.glm53_flash.reference.hf_layer_reference import load_config

    _, text = load_config(default_snapshot_dir())
    hc = Glm5NextTextHyperConnection(text)
    hc.fn.data, hc.base.data, hc.scale.data = (
        raw["hc_attn_fn"].float(),
        raw["hc_attn_base"].float(),
        raw["hc_attn_scale"].float(),
    )
    norm = Glm5NextTextRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
    norm.weight.data = raw["input_layernorm.weight"]
    with torch.no_grad():
        _, _, collapsed = hc(streams)
        return norm(collapsed)[0].to(torch.bfloat16)


def per_row_stats(ref: torch.Tensor, out: torch.Tensor) -> str:
    """Distribution of the per-token PCC (each token row = 4 streams x 4096 values): a single-token
    prefill is one sample of this distribution, which is why its PCC is noisier than an aggregate."""
    S = ref.shape[1]
    r = ref[0].reshape(S, -1).float()
    o = out[0].reshape(S, -1).float()
    rc = r - r.mean(1, keepdim=True)
    oc = o - o.mean(1, keepdim=True)
    per = (rc * oc).sum(1) / (rc.norm(dim=1) * oc.norm(dim=1) + 1e-12)
    q = torch.quantile(per, torch.tensor([0.0, 0.05, 0.5]))
    return f"per-row pcc min {q[0]:.5f} p5 {q[1]:.5f} median {q[2]:.5f} rows<0.997 {int((per < 0.997).sum())}/{S}"


def log_dram(device, when: str) -> tuple[float, float]:
    """Allocator view of the device DRAM (the capacity evidence line of the context contract);
    returns (allocated, free) in GiB (nan when the memory view is unavailable)."""
    from loguru import logger

    try:
        import ttnn

        mv = ttnn.device.get_memory_view(device, ttnn.BufferType.DRAM)
        n, gib = mv.num_banks, 2**30
        alloc, free = n * mv.total_bytes_allocated_per_bank / gib, n * mv.total_bytes_free_per_bank / gib
        logger.info(
            f"DRAM {when}: {n} banks x {mv.total_bytes_per_bank / gib:.3f} GiB = "
            f"{n * mv.total_bytes_per_bank / gib:.2f} GiB allocatable, {alloc:.2f} GiB allocated, {free:.2f} GiB free"
        )
        return alloc, free
    except Exception as e:  # pragma: no cover - informational only
        logger.info(f"DRAM memory view unavailable ({when}): {e}")
        return float("nan"), float("nan")
