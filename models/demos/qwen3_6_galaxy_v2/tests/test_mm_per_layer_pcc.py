# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V4-9: per-layer PCC for multimodal prefill — localize where vision-feature
precision divergence accumulates fastest.

Mirrors test_64layer_per_layer_pcc.py but uses multimodal inputs (vision
features spliced into fused_embeddings) so we can see if specific decoder
layers are responsible for the PCC drop from text-only's 0.98 to
multimodal's 0.83.

Output is a per-layer PCC sweep printed to stdout.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest
import torch
from loguru import logger
from PIL import Image
from safetensors.torch import load_file as load_st

import ttnn

_SNAPSHOT = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)
_T_PREFILL = 128
_N_LAYERS = 64


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _load_state_dict_text(snapshot_dir: pathlib.Path) -> dict:
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    needed_prefixes = [
        "model.language_model.embed_tokens.",
        "model.language_model.norm.",
        "lm_head.",
        "model.language_model.layers.",
    ]
    needed_keys = [k for k in weight_map if any(k.startswith(p) for p in needed_prefixes)]
    files = sorted({weight_map[k] for k in needed_keys})
    sd: dict[str, torch.Tensor] = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        for k in needed_keys:
            if k in shard:
                sd[k] = shard[k]
    return sd


def _cpu_reference_per_layer_mm(state_dict_hf: dict, fused_x: torch.Tensor, position_ids_3d: torch.Tensor):
    """Run 64L CPU reference at fp32; return list of per-layer hidden states."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import HybridDecoderLayer, Qwen36Config
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mrope import build_mrope_cos_sin as build_mrope_cos_sin_v2

    with open(_SNAPSHOT / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)
    partial_rotary_dim = int(config.head_dim * config.partial_rotary_factor)

    T = fused_x.shape[1]
    cos, sin = build_mrope_cos_sin_v2(
        position_ids_3d,
        rope_theta=config.rope_theta,
        partial_rotary_dim=partial_rotary_dim,
        mrope_section=config.mrope_section,
        attention_scaling=1.0,
        dtype=torch.float32,
    )
    causal_mask = torch.zeros(1, 1, T, T)
    causal_mask = causal_mask.masked_fill(torch.triu(torch.ones(T, T), diagonal=1).bool(), float("-inf"))

    hidden = fused_x.float()
    per_layer_hidden: list[torch.Tensor] = []
    for layer_idx in range(_N_LAYERS):
        layer = HybridDecoderLayer(config, layer_idx).eval()
        pfx = f"model.language_model.layers.{layer_idx}."
        layer_sd: dict[str, torch.Tensor] = {}
        for k, v in state_dict_hf.items():
            if k.startswith(pfx):
                short = k[len(pfx) :]
                if short.startswith("self_attn."):
                    layer_sd["attention." + short[len("self_attn.") :]] = v.float()
                elif short.startswith("linear_attn."):
                    layer_sd["attention." + short[len("linear_attn.") :]] = v.float()
                else:
                    layer_sd[short] = v.float()
        layer.load_state_dict(layer_sd, strict=False)
        with torch.no_grad():
            hidden, _, _, _ = layer(hidden, cos, sin, attention_mask=causal_mask)
        per_layer_hidden.append(hidden.clone())
        del layer
        if (layer_idx + 1) % 8 == 0:
            logger.info(f"[CPU ref] layer {layer_idx + 1}/{_N_LAYERS} done")
    return per_layer_hidden, list(config.layer_types)


def _send_col_sharded_hidden(t: torch.Tensor, mesh, cluster_shape):
    if t.dim() == 2:
        t = t.unsqueeze(0)
    B, T, H = t.shape
    return ttnn.from_torch(
        t.reshape(1, 1, T, H).to(torch.bfloat16),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=cluster_shape),
    )


def _gather_col_sharded_to_full(tt_tensor, mesh, cluster_shape, T):
    out = ttnn.to_torch(
        tt_tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 3), mesh_shape=cluster_shape),
    )
    out = out[0:1]
    while out.dim() > 3 and out.shape[0] == 1:
        out = out.squeeze(0)
    if out.dim() == 3:
        out = out[:, :T, :]
    return out


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "BH_GLX": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_mm_per_layer_pcc(mesh_device, reset_seeds, ensure_gc):
    """Per-layer PCC for multimodal prefill."""
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = "Qwen/Qwen3.6-27B"

    from models.demos.qwen3_6_galaxy.reference.qwen36 import Qwen36Config
    from models.demos.qwen3_6_galaxy_v2.tt.generator import get_padded_prefill_len
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mm_generator import Qwen36MMGenerator
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mrope import build_mrope_tt_tensors
    from models.demos.qwen3_6_galaxy_v2.tt.vision_model_args import Qwen36VisionModelArgs
    from models.tt_dit.parallel.manager import CCLManager

    vision_args = Qwen36VisionModelArgs(mesh_device, dummy_weights=False, max_batch_size=1, max_seq_len=256)
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    logger.info("Loading text state_dict...")
    text_sd = _load_state_dict_text(_SNAPSHOT)
    text_embed_weight = text_sd["model.language_model.embed_tokens.weight"].float()

    with open(_SNAPSHOT / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)
    pattern = list(config.layer_types)
    args = TtQwen36ModelArgs(mesh_device)
    args.n_layers = _N_LAYERS
    args.linear_attention_pattern = pattern
    weight_cache_path = args.weight_cache_path(ttnn.bfloat8_b)
    weight_cache_path.mkdir(parents=True, exist_ok=True)
    logger.info("Building TT 64-layer model...")
    model = TtTransformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh_device,
        state_dict=text_sd,
        weight_cache_path=weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    )

    gen = Qwen36MMGenerator(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        vision_model_args=vision_args,
        text_model=model,
        text_embed_weight=text_embed_weight,
        dtype=ttnn.bfloat16,
    )

    img = Image.open("models/demos/multimodal/gemma3/dog.jpg").convert("RGB").resize((224, 224))
    prompt = "<|vision_start|><|image_pad|><|vision_end|>What is in this image?"
    logger.info("Running vision pipeline...")
    inputs, fused_embeddings_unpadded = gen.prepare_inputs(prompt, images=[img])
    S_unpadded = fused_embeddings_unpadded.shape[1]
    S = get_padded_prefill_len(S_unpadded)
    T_prompt = int(inputs.attention_mask.sum().item())

    # Pad
    pad_len = S - S_unpadded
    pad_emb = torch.zeros(
        *fused_embeddings_unpadded.shape[:-2],
        pad_len,
        fused_embeddings_unpadded.shape[-1],
        dtype=fused_embeddings_unpadded.dtype,
    )
    fused_embeddings = torch.cat([fused_embeddings_unpadded, pad_emb], dim=-2)
    last_pos = inputs.position_ids_3d[:, :, -1:].max().item()
    pad_positions = torch.arange(last_pos + 1, last_pos + 1 + pad_len, dtype=inputs.position_ids_3d.dtype)
    pad_positions_3d = pad_positions.view(1, 1, pad_len).expand(3, inputs.position_ids_3d.shape[1], pad_len)
    position_ids_3d_padded = torch.cat([inputs.position_ids_3d, pad_positions_3d], dim=-1)
    logger.info(f"S_unpadded={S_unpadded} S={S} T_prompt={T_prompt}")

    logger.info("Running CPU reference per-layer (slow, ~5-10 min)...")
    per_layer_ref, _ = _cpu_reference_per_layer_mm(text_sd, fused_embeddings, position_ids_3d_padded)
    logger.info(f"CPU ref captured {len(per_layer_ref)} layers")

    # TT setup
    x_tt = _send_col_sharded_hidden(fused_embeddings, mesh_device, args.cluster_shape)
    cos_tt, sin_tt = build_mrope_tt_tensors(
        position_ids_3d_padded,
        rope_theta=config.rope_theta,
        partial_rotary_dim=int(config.head_dim * config.partial_rotary_factor),
        mrope_section=config.mrope_section,
        mesh_device=mesh_device,
    )
    chunk_start_idx_tt = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Manual layer loop with per-layer capture
    logger.info("Running TT 64L (instrumented) per-layer prefill...")
    rot_mats = (cos_tt, sin_tt)
    x = x_tt
    h = None
    per_layer_pcc_last: list[float] = []
    per_layer_pcc_prompt: list[float] = []
    for i, layer in enumerate(model.layers):
        x, h = layer(
            x,
            h,
            None,
            rot_mats,
            0,
            "prefill",
            None,
            chunk_page_table=None,
            chunk_start_idx=0,
            chunk_start_idx_tensor=chunk_start_idx_tt,
            kv_cache=None,
            batch_size=1,
        )
        tt_hidden_cpu = _gather_col_sharded_to_full(x, mesh_device, args.cluster_shape, T=S)
        tt_hidden_cpu = tt_hidden_cpu.reshape(1, S, -1).float()
        ref_hidden = per_layer_ref[i][:, :S, :].float()
        pcc_prompt = _pcc(tt_hidden_cpu[:, :T_prompt, :], ref_hidden[:, :T_prompt, :])
        pcc_last = _pcc(tt_hidden_cpu[:, T_prompt - 1 : T_prompt, :], ref_hidden[:, T_prompt - 1 : T_prompt, :])
        per_layer_pcc_last.append(pcc_last)
        per_layer_pcc_prompt.append(pcc_prompt)
        logger.info(f"L{i:02d} ({pattern[i][:3]}): PCC_prompt={pcc_prompt:.4f} PCC_last={pcc_last:.4f}")

    logger.info("\n=== SUMMARY ===")
    logger.info(
        f"First layer with PCC_last < 0.95: "
        + str(next((i for i, p in enumerate(per_layer_pcc_last) if p < 0.95), None))
    )
    logger.info(
        f"First layer with PCC_last < 0.90: "
        + str(next((i for i, p in enumerate(per_layer_pcc_last) if p < 0.90), None))
    )
    logger.info(f"Final layer PCC_last: {per_layer_pcc_last[-1]:.6f}")
    logger.info(f"Final layer PCC_prompt: {per_layer_pcc_prompt[-1]:.6f}")
