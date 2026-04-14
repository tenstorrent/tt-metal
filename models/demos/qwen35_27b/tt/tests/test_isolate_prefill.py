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

    # Per-token PCC breakdown from full decoder block outputs
    logger.info(f"\n--- Per-token PCC (full decoder block 0) ---")

    # Collect per-token baseline by re-running decode
    gdn.reset_state()
    baseline_per_token = []
    for t in range(seq_len):
        tok_batch = torch.full((batch_size,), tokens[t], dtype=torch.long)
        current_pos = torch.full((batch_size,), t, dtype=torch.long)
        tt_tok, tt_pos, tt_rot, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        x = model._transform_decode_inputs_device(tt_tok)
        rot_mats = model.rope_setup.get_rot_mats(tt_rot)
        x = ttnn.to_memory_config(x, args.get_residual_mem_config(Mode.DECODE))
        x = layer0(x, tt_pos, rot_mats_global=rot_mats, mode=Mode.DECODE)
        out_cpu = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
        baseline_per_token.append(out_cpu[0, 0, 0, : args.dim].clone())

    # Re-run batched prefill to get per-token outputs
    gdn.reset_state()
    gdn._init_prefill_states()
    prefill_inputs = model.prepare_inputs_prefill(tokens_tensor)
    tt_embeds = prefill_inputs[0]
    tt_rot_global = prefill_inputs[1]
    x_pf = ttnn.to_memory_config(tt_embeds, args.get_residual_mem_config(Mode.PREFILL))
    x_pf = layer0(x_pf, current_pos=None, rot_mats_global=tt_rot_global, mode=Mode.PREFILL)
    pf_cpu = ttnn.to_torch(x_pf, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))

    for t in range(seq_len):
        t_pcc = pcc(baseline_per_token[t], pf_cpu[0, 0, t, : args.dim])
        status = "OK" if t_pcc > 0.99 else "LOW"
        logger.info(
            f"  Token {t} PCC: {t_pcc:.6f} [{status}]  baseline_norm={baseline_per_token[t].norm():.4f}  prefill_norm={pf_cpu[0, 0, t, :args.dim].norm():.4f}"
        )

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
def test_chunked_vs_single_gdn_layer0(mesh_device, reset_seeds, ensure_gc):
    """Compare GDN layer 0: single-chunk prefill vs two-chunk prefill.

    Uses the full decoder block (norm + GDN + residual + MLP + residual).
    If two-chunk PCC is low vs single-chunk, the chunk boundary state handoff is broken.
    """
    model_path = _get_model_path()
    batch_size = 32

    if mesh_device.get_num_devices() < 4:
        pytest.skip("Need TP>=4")
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = create_qwen35_model(mesh_device, model_path=model_path, max_batch_size=batch_size, max_seq_len=2048)
    args = model.args
    layer0 = model.layers[0]
    gdn = layer0.attention

    # Use a prompt long enough to split into 2 chunks at chunk_size=64
    # (Using 128 tokens: chunk 0 = tokens 0-63, chunk 1 = tokens 64-127)
    tokens = tokenizer.encode("The capital of France is" * 10)[:128]
    seq_len = len(tokens)
    logger.info(f"Tokens: {seq_len}")

    tokens_tensor = torch.tensor([tokens], dtype=torch.long)
    prefill_inputs = model.prepare_inputs_prefill(tokens_tensor)
    tt_embeds = prefill_inputs[0]

    # ---- Run A: Single-chunk prefill (all tokens at once) ----
    logger.info("Run A: Single-chunk prefill (all tokens at once)...")
    gdn.reset_state()
    gdn._init_prefill_states()

    out_single = layer0(tt_embeds, current_pos=None, rot_mats_global=None, mode=Mode.PREFILL)
    single_cpu = ttnn.to_torch(out_single, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    ttnn.deallocate(out_single)

    # ---- Run B: Two-chunk prefill (split at midpoint) ----
    logger.info("Run B: Two-chunk prefill (split at midpoint)...")
    gdn.reset_state()
    gdn._init_prefill_states()

    mid = seq_len // 2
    # Chunk 1: tokens 0..mid-1
    chunk1 = ttnn.slice(tt_embeds, (0, 0, 0, 0), (1, 1, mid, tt_embeds.shape[-1]))
    out1 = layer0(chunk1, current_pos=None, rot_mats_global=None, mode=Mode.PREFILL)
    chunk1_cpu = ttnn.to_torch(out1, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    ttnn.deallocate(out1)

    # Chunk 2: tokens mid..seq_len-1
    chunk2 = ttnn.slice(tt_embeds, (0, 0, mid, 0), (1, 1, seq_len, tt_embeds.shape[-1]))
    out2 = layer0(chunk2, current_pos=None, rot_mats_global=None, mode=Mode.PREFILL)
    chunk2_cpu = ttnn.to_torch(out2, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    ttnn.deallocate(out2)

    # ---- Compare per-token ----
    logger.info(f"\n--- Per-token PCC: single vs two-chunk ---")
    for t in range(seq_len):
        single_t = single_cpu[0, 0, t, : args.dim]
        if t < mid:
            chunked_t = chunk1_cpu[0, 0, t, : args.dim]
        else:
            chunked_t = chunk2_cpu[0, 0, t - mid, : args.dim]
        t_pcc = pcc(single_t, chunked_t)
        status = "OK" if t_pcc > 0.99 else "LOW"
        if t < 5 or t == mid - 1 or t == mid or t == mid + 1 or t == seq_len - 1:
            logger.info(
                f"  Token {t} PCC: {t_pcc:.6f} [{status}]  "
                f"single_norm={single_t.norm():.4f}  chunked_norm={chunked_t.norm():.4f}"
            )

    # Final token comparison
    single_last = single_cpu[0, 0, seq_len - 1, : args.dim]
    chunked_last = chunk2_cpu[0, 0, seq_len - 1 - mid, : args.dim]
    final_pcc = pcc(single_last, chunked_last)
    logger.info(f"\nFinal token PCC (single vs two-chunk): {final_pcc:.6f}")
    logger.info(f"  single norm: {single_last.norm():.4f}")
    logger.info(f"  chunked norm: {chunked_last.norm():.4f}")


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
def test_chunked_vs_single_4layers(mesh_device, reset_seeds, ensure_gc):
    """Compare 4 layers (3 GDN + 1 attention): single-chunk vs two-chunk.

    This tests whether chunk boundaries cause error accumulation across layers.
    """
    model_path = _get_model_path()
    batch_size = 32

    if mesh_device.get_num_devices() < 4:
        pytest.skip("Need TP>=4")
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = create_qwen35_model(mesh_device, model_path=model_path, max_batch_size=batch_size, max_seq_len=4096)
    args = model.args
    n_test_layers = 4  # 3 GDN + 1 attention

    # Generate enough tokens — repeat a longer phrase to avoid tokenizer compression
    raw = "The capital of France is Paris and the capital of Germany is Berlin and " * 20
    tokens = tokenizer.encode(raw)[:512]
    seq_len = len(tokens)
    mid = seq_len // 2
    logger.info(f"Tokens: {seq_len}, split at {mid}")

    tokens_tensor = torch.tensor([tokens], dtype=torch.long)
    prefill_inputs = model.prepare_inputs_prefill(tokens_tensor)
    tt_embeds = prefill_inputs[0]
    rot_mats_full = prefill_inputs[1]

    # ---- Run A: Single-chunk through 4 layers ----
    logger.info("Run A: Single-chunk through 4 layers...")
    for i in range(n_test_layers):
        attn = model.layers[i].attention
        if hasattr(attn, "reset_state"):
            attn.reset_state()
        if hasattr(attn, "_init_prefill_states"):
            attn._init_prefill_states()

    x_single = tt_embeds
    for i in range(n_test_layers):
        layer_type = args.layer_types[i]
        is_attention = layer_type == "full_attention"
        rot = rot_mats_full if is_attention else None
        x_single = model.layers[i](x_single, current_pos=None, rot_mats_global=rot, mode=Mode.PREFILL)

    single_cpu = ttnn.to_torch(x_single, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    ttnn.deallocate(x_single)

    # ---- Run B: Two-chunk through 4 layers (layer-at-a-time chunking) ----
    logger.info("Run B: Two-chunk through 4 layers (layer-at-a-time)...")
    for i in range(n_test_layers):
        attn = model.layers[i].attention
        if hasattr(attn, "reset_state"):
            attn.reset_state()
        if hasattr(attn, "_init_prefill_states"):
            attn._init_prefill_states()

    x_chunked = tt_embeds
    for i in range(n_test_layers):
        layer_type = args.layer_types[i]
        is_attention = layer_type == "full_attention"

        if is_attention:
            # Attention: process full sequence (needs full KV context)
            rot = rot_mats_full
            x_out = model.layers[i](x_chunked, current_pos=None, rot_mats_global=rot, mode=Mode.PREFILL)
        else:
            # GDN: two-chunk processing
            c1 = ttnn.slice(x_chunked, (0, 0, 0, 0), (1, 1, mid, x_chunked.shape[-1]))
            out1 = model.layers[i](c1, current_pos=None, rot_mats_global=None, mode=Mode.PREFILL)

            c2 = ttnn.slice(x_chunked, (0, 0, mid, 0), (1, 1, seq_len, x_chunked.shape[-1]))
            out2 = model.layers[i](c2, current_pos=None, rot_mats_global=None, mode=Mode.PREFILL)

            x_out = ttnn.concat([out1, out2], dim=2)
            ttnn.deallocate(out1)
            ttnn.deallocate(out2)

        ttnn.deallocate(x_chunked)
        x_chunked = x_out

        # Log per-layer comparison
        chunked_layer_cpu = ttnn.to_torch(x_chunked, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
        single_layer_t = single_cpu[0, 0, seq_len - 1, : args.dim]
        chunked_layer_t = chunked_layer_cpu[0, 0, seq_len - 1, : args.dim]
        # Can only compare at the end since single_cpu is the final output
        if i == n_test_layers - 1:
            layer_pcc = pcc(single_layer_t, chunked_layer_t)
            logger.info(f"  After layer {i} ({layer_type}): last-token PCC = {layer_pcc:.6f}")

    chunked_cpu = ttnn.to_torch(x_chunked, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    ttnn.deallocate(x_chunked)

    # ---- Per-token comparison ----
    logger.info(f"\n--- Per-token PCC after 4 layers: single vs two-chunk ---")
    for t in [0, 1, mid - 1, mid, mid + 1, seq_len - 1]:
        if t < seq_len:
            s_t = single_cpu[0, 0, t, : args.dim]
            c_t = chunked_cpu[0, 0, t, : args.dim]
            t_pcc = pcc(s_t, c_t)
            status = "OK" if t_pcc > 0.99 else "LOW"
            logger.info(f"  Token {t} PCC: {t_pcc:.6f} [{status}]  single={s_t.norm():.4f}  chunked={c_t.norm():.4f}")


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
def test_pcc_vs_depth_long(mesh_device, reset_seeds, ensure_gc):
    """Measure PCC degradation per layer for a ~1000-token sequence.

    Runs sequential decode (baseline) and batched prefill through N layers,
    comparing the last token output at each layer boundary.
    """
    model_path = _get_model_path()
    batch_size = 32

    if mesh_device.get_num_devices() < 4:
        pytest.skip("Need TP>=4")
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = create_qwen35_model(mesh_device, model_path=model_path, max_batch_size=batch_size, max_seq_len=4096)
    args = model.args

    # Use ~500 tokens to test longer-sequence PCC scaling
    raw = "The capital of France is Paris and the capital of Germany is Berlin and " * 80
    tokens = tokenizer.encode(raw)[:500]
    seq_len = len(tokens)
    logger.info(f"Tokens: {seq_len}")

    n_test_layers = 8  # First 8 layers (6 GDN + 2 attention)

    # ---- Baseline: sequential decode, one layer at a time ----
    # Process all tokens through layer 0, save last-token output.
    # Then process all tokens through layer 1, save last-token output. Etc.
    # This lets us compare per-layer outputs between sequential and batched.
    logger.info(f"Running sequential decode baseline, layer-by-layer...")
    for i in range(n_test_layers):
        attn = model.layers[i].attention
        if hasattr(attn, "reset_state"):
            attn.reset_state()

    # First pass all tokens through all layers sequentially
    baseline_after_each_layer = {}
    for t in range(seq_len):
        tok_batch = torch.full((batch_size,), tokens[t], dtype=torch.long)
        current_pos = torch.full((batch_size,), t, dtype=torch.long)
        tt_tok, tt_pos, tt_rot, _ = model.prepare_inputs_decode(tok_batch, current_pos)
        x = model._transform_decode_inputs_device(tt_tok)
        rot_mats = model.rope_setup.get_rot_mats(tt_rot)

        for i in range(n_test_layers):
            x = ttnn.to_memory_config(x, args.get_residual_mem_config(Mode.DECODE))
            x = model.layers[i](x, tt_pos, rot_mats_global=rot_mats, mode=Mode.DECODE)

            if t == seq_len - 1:
                out_cpu = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
                baseline_after_each_layer[i] = out_cpu[0, 0, 0, : args.dim].clone()

    logger.info(f"Baseline done.")

    # ---- Batched prefill: layer-by-layer, saving output after each layer ----
    logger.info(f"Running batched prefill through {n_test_layers} layers...")
    for i in range(n_test_layers):
        attn = model.layers[i].attention
        if hasattr(attn, "reset_state"):
            attn.reset_state()
        if hasattr(attn, "_init_prefill_states"):
            attn._init_prefill_states()

    tokens_tensor = torch.tensor([tokens], dtype=torch.long)
    prefill_inputs = model.prepare_inputs_prefill(tokens_tensor)
    x_pf = prefill_inputs[0]
    rot_mats_full = prefill_inputs[1]

    prefill_per_layer = {}
    for i in range(n_test_layers):
        layer_type = args.layer_types[i]
        is_attention = layer_type == "full_attention"
        rot = rot_mats_full if is_attention else None
        x_pf = model.layers[i](x_pf, current_pos=None, rot_mats_global=rot, mode=Mode.PREFILL)

        # Save last token output after this layer
        pf_cpu = ttnn.to_torch(x_pf, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
        prefill_per_layer[i] = pf_cpu[0, 0, seq_len - 1, : args.dim].clone()

        if i < 8 or i == n_test_layers - 1:
            logger.info(f"  Layer {i} ({layer_type}): last-token norm = {prefill_per_layer[i].norm():.4f}")

    # ---- Compare per-layer ----
    logger.info(f"\n--- PCC after each layer (baseline decode vs batched prefill) ---")
    for i in range(n_test_layers):
        if i in baseline_after_each_layer and i in prefill_per_layer:
            layer_pcc = pcc(baseline_after_each_layer[i], prefill_per_layer[i])
            layer_type = args.layer_types[i]
            b_norm = baseline_after_each_layer[i].norm()
            p_norm = prefill_per_layer[i].norm()
            status = "OK" if layer_pcc > 0.999 else ("WARN" if layer_pcc > 0.99 else "BAD")
            logger.info(
                f"  Layer {i:2d} ({layer_type:17s}): PCC={layer_pcc:.6f} [{status}] "
                f"baseline_norm={b_norm:.2f} prefill_norm={p_norm:.2f}"
            )


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
