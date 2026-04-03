# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Isolate prefill accuracy bug by comparing baseline vs batched prefill
at individual component level.
"""

import os

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.demos.qwen35_27b.tt.model import create_qwen35_model
from models.tt_transformers.tt.model_config import Mode


def pcc(a, b):
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    a_mean = a_flat - a_flat.mean()
    b_mean = b_flat - b_flat.mean()
    return ((a_mean * b_mean).sum() / (a_mean.norm() * b_mean.norm()).clamp(min=1e-8)).item()


def _get_model_path():
    return os.path.expanduser(os.environ.get("HF_MODEL", "~/models/Qwen3.5-27B-FP8"))


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "P150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 8))
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_isolate_gdn_layer0(mesh_device, reset_seeds, ensure_gc):
    """Compare GDN layer 0 output: baseline (decode per-token B=32) vs prefill (B=1)."""
    model_path = _get_model_path()
    batch_size = 32

    if mesh_device.get_num_devices() < 4:
        pytest.skip("Need TP>=4")
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = create_qwen35_model(mesh_device, model_path=model_path, max_batch_size=batch_size, max_seq_len=2048)
    args = model.args

    tokens = tokenizer.encode("The capital of France is")
    seq_len = len(tokens)
    logger.info(f"Tokens: {seq_len}")

    layer0 = model.layers[0]
    gdn = layer0.attention

    # ---- Baseline: decode per-token through full decoder block 0 ----
    logger.info("Running baseline (decode per-token, full decoder block 0)...")
    gdn.reset_state()
    baseline_last = None

    for t in range(seq_len):
        tok_batch = torch.full((batch_size,), tokens[t], dtype=torch.long)
        current_pos = torch.full((batch_size,), t, dtype=torch.long)
        tt_tok, tt_pos, tt_rot, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        x_embed = model._transform_decode_inputs_device(tt_tok)

        # Run full decoder block (norm -> attn -> residual -> ffn_norm -> mlp -> residual)
        x_out = layer0(x_embed, tt_pos, rot_mats_global=model.rope_setup.get_rot_mats(tt_rot), mode=Mode.DECODE)

        if t == seq_len - 1:
            baseline_last = ttnn.to_torch(x_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
            baseline_last = baseline_last[0, 0, 0, : args.dim].clone()  # user 0

    logger.info(f"Baseline last token output norm: {baseline_last.norm():.4f}")

    # ---- Batched prefill: framework prefill through decoder block 0 ----
    logger.info("Running batched prefill (decoder block 0)...")
    gdn.reset_state()
    gdn._init_prefill_states()

    tokens_tensor = torch.tensor([tokens], dtype=torch.long)
    prefill_inputs = model.prepare_inputs_prefill(tokens_tensor)
    tt_embeds = prefill_inputs[0]
    tt_rot_global = prefill_inputs[1]

    x_pf_out = layer0(tt_embeds, current_pos=None, rot_mats_global=tt_rot_global, mode=Mode.PREFILL)

    pf_out_cpu = ttnn.to_torch(x_pf_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    # Prefill output: [1, 1, seq_len, dim] — take last token
    pf_last = pf_out_cpu[0, 0, seq_len - 1, : args.dim].clone()

    logger.info(f"Batched last token output norm: {pf_last.norm():.4f}")

    # ---- Compare ----
    last_pcc = pcc(baseline_last, pf_last)
    logger.info(f"GDN Layer 0 (full decoder block) - Last token PCC: {last_pcc:.6f}")

    if last_pcc < 0.99:
        logger.error(f"LOW PCC! Investigating components...")

        # Now test just the GDN attention (without MLP) to narrow down
        logger.info("\n--- Testing GDN attention only (no MLP) ---")
        gdn.reset_state()
        baseline_attn_outputs = []
        for t in range(seq_len):
            tok_batch = torch.full((batch_size,), tokens[t], dtype=torch.long)
            current_pos = torch.full((batch_size,), t, dtype=torch.long)
            tt_tok, tt_pos, tt_rot, _ = model.prepare_inputs_decode(tok_batch, current_pos)
            x_embed = model._transform_decode_inputs_device(tt_tok)
            x_normed = layer0.attention_norm(x_embed, mode=Mode.DECODE)
            attn_out = gdn.forward(x_normed, current_pos=tt_pos, mode=Mode.DECODE)
            out_cpu = ttnn.to_torch(attn_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
            baseline_attn_outputs.append(out_cpu[0, 0, 0, : args.dim].clone())

        gdn.reset_state()
        gdn._init_prefill_states()
        prefill_inputs = model.prepare_inputs_prefill(tokens_tensor)
        tt_embeds = prefill_inputs[0]
        x_normed_pf = layer0.attention_norm(tt_embeds, mode=Mode.PREFILL)
        gdn_pf_out = gdn.forward(x_normed_pf, current_pos=None, mode=Mode.PREFILL)
        gdn_pf_cpu = ttnn.to_torch(gdn_pf_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))

        for t in range(seq_len):
            batched_t = gdn_pf_cpu[0, 0, t, : args.dim]
            t_pcc = pcc(baseline_attn_outputs[t], batched_t)
            status = "OK" if t_pcc > 0.99 else "BAD"
            logger.info(f"  GDN-only Token {t} PCC: {t_pcc:.6f} [{status}]")

    assert last_pcc > 0.95, f"GDN Layer 0 PCC too low: {last_pcc}"


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "P150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 8))
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_isolate_4layers(mesh_device, reset_seeds, ensure_gc):
    """Run first 4 layers (3 GDN + 1 attention) and compare baseline vs prefill."""
    model_path = _get_model_path()
    batch_size = 32

    if mesh_device.get_num_devices() < 4:
        pytest.skip("Need TP>=4")
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = create_qwen35_model(mesh_device, model_path=model_path, max_batch_size=batch_size, max_seq_len=2048)
    args = model.args

    tokens = tokenizer.encode("The capital of France is")
    seq_len = len(tokens)
    logger.info(f"Tokens: {seq_len}")

    # ---- Baseline: decode per-token through layers 0-3 ----
    logger.info("Running baseline (layers 0-3, decode per-token)...")
    for i in range(4):
        layer = model.layers[i]
        if hasattr(layer.attention, "reset_state"):
            layer.attention.reset_state()

    baseline_last = None
    for t in range(seq_len):
        tok_batch = torch.full((batch_size,), tokens[t], dtype=torch.long)
        current_pos = torch.full((batch_size,), t, dtype=torch.long)
        tt_tok, tt_pos, tt_rot, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        x = model._transform_decode_inputs_device(tt_tok)
        rot_mats = model.rope_setup.get_rot_mats(tt_rot)

        # Run through layers 0-3
        for i in range(4):
            x = ttnn.to_memory_config(x, args.get_residual_mem_config(Mode.DECODE))
            x = model.layers[i](x, tt_pos, rot_mats_global=rot_mats, mode=Mode.DECODE)

        if t == seq_len - 1:
            baseline_last = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
            baseline_last = baseline_last[0, 0, 0, : args.dim].clone()

    logger.info(f"Baseline last token norm: {baseline_last.norm():.4f}")

    # ---- Batched: prefill through layers 0-3 ----
    logger.info("Running batched prefill (layers 0-3)...")
    for i in range(4):
        layer = model.layers[i]
        if hasattr(layer.attention, "reset_state"):
            layer.attention.reset_state()
        if hasattr(layer.attention, "_init_prefill_states"):
            layer.attention._init_prefill_states()

    tokens_tensor = torch.tensor([tokens], dtype=torch.long)
    prefill_inputs = model.prepare_inputs_prefill(tokens_tensor)
    x_pf = prefill_inputs[0]
    rot_global = prefill_inputs[1]

    for i in range(4):
        x_pf = ttnn.to_memory_config(x_pf, args.get_residual_mem_config(Mode.PREFILL))
        x_pf = model.layers[i](x_pf, current_pos=None, rot_mats_global=rot_global, mode=Mode.PREFILL)

    pf_cpu = ttnn.to_torch(x_pf, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    pf_last = pf_cpu[0, 0, seq_len - 1, : args.dim].clone()

    logger.info(f"Batched last token norm: {pf_last.norm():.4f}")

    last_pcc = pcc(baseline_last, pf_last)
    logger.info(f"Layers 0-3 last token PCC: {last_pcc:.6f}")

    # Also check per-layer by running layers individually
    if last_pcc < 0.99:
        logger.error(f"LOW PCC at layer 3! Testing layer 3 (attention) standalone...")

        # Test layer 3 alone with same input
        attn = model.layers[3].attention
        attn.reset_state()

        # Run baseline for just layer 3
        rand_in = torch.randn(1, 1, seq_len, args.dim, dtype=torch.bfloat16)
        tt_in = ttnn.from_torch(
            rand_in,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        baseline_attn_last = None
        for t in range(seq_len):
            x_t = ttnn.slice(tt_in, (0, 0, t, 0), (1, 1, t + 1, args.dim))
            x_t_decode = ttnn.reshape(x_t, (1, batch_size, args.dim))  # fake B=32
            pos_t = ttnn.from_torch(
                torch.tensor([t] * batch_size, dtype=torch.int32),
                dtype=ttnn.int32,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            rot_t = model.rope_setup.get_rot_mats(pos_t)
            out_t = attn.forward(x_t_decode, pos_t, rot_t, mode=Mode.DECODE)
            out_cpu = ttnn.to_torch(out_t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
            baseline_attn_last = out_cpu[0, 0, 0, : args.dim].clone()

        # Run prefill for layer 3
        attn.reset_state()
        out_pf = attn.forward(tt_in, current_pos=None, rot_mats=None, mode=Mode.PREFILL)
        out_pf_cpu = ttnn.to_torch(out_pf, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
        pf_attn_last = out_pf_cpu[0, 0, seq_len - 1, : args.dim].clone()

        attn_pcc = pcc(baseline_attn_last, pf_attn_last)
        logger.info(f"Attention layer 3 standalone PCC: {attn_pcc:.6f}")

    assert last_pcc > 0.95, f"Layers 0-3 PCC too low: {last_pcc}"


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "P150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 8))
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_isolate_all_layers(mesh_device, reset_seeds, ensure_gc):
    """Run all 64 layers, comparing baseline vs prefill at checkpoints."""
    model_path = _get_model_path()
    batch_size = 32

    if mesh_device.get_num_devices() < 4:
        pytest.skip("Need TP>=4")
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = create_qwen35_model(mesh_device, model_path=model_path, max_batch_size=batch_size, max_seq_len=2048)
    args = model.args
    n_layers = args.n_layers

    tokens = tokenizer.encode("The capital of France is")
    seq_len = len(tokens)
    logger.info(f"Tokens: {seq_len}, Layers: {n_layers}")

    # ---- Baseline: decode per-token through ALL layers ----
    logger.info("Running baseline (all layers, decode per-token)...")
    for layer in model.layers:
        if hasattr(layer.attention, "reset_state"):
            layer.attention.reset_state()

    for t in range(seq_len):
        tok_batch = torch.full((batch_size,), tokens[t], dtype=torch.long)
        current_pos = torch.full((batch_size,), t, dtype=torch.long)
        tt_tok, tt_pos, tt_rot, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        x = model._transform_decode_inputs_device(tt_tok)
        rot_mats = model.rope_setup.get_rot_mats(tt_rot)

        for i in range(n_layers):
            x = ttnn.to_memory_config(x, args.get_residual_mem_config(Mode.DECODE))
            x = model.layers[i](x, tt_pos, rot_mats_global=rot_mats, mode=Mode.DECODE)

    baseline_last = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    baseline_last = baseline_last[0, 0, 0, : args.dim].clone()
    logger.info(f"Baseline last token norm: {baseline_last.norm():.4f}")

    # ---- Batched: prefill through ALL layers ----
    logger.info("Running batched prefill (all layers)...")
    for layer in model.layers:
        if hasattr(layer.attention, "reset_state"):
            layer.attention.reset_state()
        if hasattr(layer.attention, "_init_prefill_states"):
            layer.attention._init_prefill_states()

    tokens_tensor = torch.tensor([tokens], dtype=torch.long)
    prefill_inputs = model.prepare_inputs_prefill(tokens_tensor)
    x_pf = prefill_inputs[0]
    rot_global = prefill_inputs[1]

    for i in range(n_layers):
        x_pf = ttnn.to_memory_config(x_pf, args.get_residual_mem_config(Mode.PREFILL))
        x_pf = model.layers[i](x_pf, current_pos=None, rot_mats_global=rot_global, mode=Mode.PREFILL)

    pf_cpu = ttnn.to_torch(x_pf, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    pf_last = pf_cpu[0, 0, seq_len - 1, : args.dim].clone()
    logger.info(f"Batched last token norm: {pf_last.norm():.4f}")

    last_pcc = pcc(baseline_last, pf_last)
    logger.info(f"All 64 layers - Last token PCC: {last_pcc:.6f}")

    if last_pcc > 0.99:
        logger.info("HIGH PCC — mismatch was likely in LM head / final norm, not in layers")
    elif last_pcc > 0.95:
        logger.info("MODERATE PCC — precision drift over 64 layers, acceptable")
    else:
        logger.error(f"LOW PCC — significant divergence in prefill path")

    assert last_pcc > 0.90, f"All layers PCC too low: {last_pcc}"
