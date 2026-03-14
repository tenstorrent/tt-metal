# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Debug script: find exact layer where PCC diverges between TTNN and PyTorch."""

import sys

sys.path.append("/workdir/tt-metal")

import torch
from transformers import BarkModel

import ttnn


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    valid = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[valid], b[valid]
    if len(a) == 0:
        return 0.0
    a_c, b_c = a - a.mean(), b - b.mean()
    denom = torch.sqrt((a_c**2).sum() * (b_c**2).sum())
    if denom == 0:
        return 1.0 if (a_c * b_c).sum() == 0 else 0.0
    return ((a_c * b_c).sum() / denom).item()


def main():
    print("Loading HuggingFace Bark reference...")
    hf_model = BarkModel.from_pretrained("suno/bark-small")
    hf_model.eval()
    ref = hf_model.semantic  # BarkCausalModel

    batch_size, seq_len = 1, 64
    input_ids = torch.randint(0, 10048, (batch_size, seq_len))

    # === Reference forward (layer by layer) ===
    with torch.no_grad():
        ref_input_embeds = ref.input_embeds_layer(input_ids)
        ref_pos_ids = torch.arange(seq_len).unsqueeze(0)
        ref_pos_embeds = ref.position_embeds_layer(ref_pos_ids)
        ref_hidden = ref_input_embeds + ref_pos_embeds
        print(f"Ref combined embed shape: {ref_hidden.shape}")

        # Per-layer reference
        ref_layer_outputs = []
        for i, layer in enumerate(ref.layers):
            out = layer(ref_hidden)
            ref_hidden = out[0] if isinstance(out, tuple) else out
            ref_layer_outputs.append(ref_hidden.clone())

        ref_hidden_final = ref.layernorm_final(ref_hidden)
        ref_logits = ref.lm_head(ref_hidden_final)

    # === TTNN forward ===
    device = ttnn.open_device(device_id=0)

    from models.demos.wormhole.bark.tt.bark_gpt import BarkConfig, TtBarkGPT, preprocess_model_parameters

    config = BarkConfig(
        hidden_size=ref.config.hidden_size,
        num_heads=ref.config.num_heads,
        num_layers=ref.config.num_layers,
        block_size=ref.config.block_size,
        input_vocab_size=ref.config.input_vocab_size,
        output_vocab_size=ref.config.output_vocab_size,
        bias=getattr(ref.config, "bias", False),
    )
    params = preprocess_model_parameters(ref, device)
    tt_model = TtBarkGPT(device, params, config, is_causal=True)

    # --- Manually step through TtBarkGPT.__call__ to get per-layer PCC ---
    # Embedding
    tt_input_ids = ttnn.from_torch(
        input_ids.to(torch.int32), dtype=ttnn.uint32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    inputs_embeds = ttnn.embedding(tt_input_ids, tt_model.input_embeds_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    print(f"TT input_embeds shape after embedding: {inputs_embeds.shape}")

    inputs_embeds = ttnn.to_layout(inputs_embeds, ttnn.TILE_LAYOUT)
    print(f"TT input_embeds shape after TILE: {inputs_embeds.shape}")

    position_ids = torch.arange(0, seq_len, dtype=torch.int32).unsqueeze(0)
    tt_pos_ids = ttnn.from_torch(position_ids, dtype=ttnn.uint32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    pos_embeds = ttnn.embedding(tt_pos_ids, tt_model.position_embeds_weight, memory_config=ttnn.L1_MEMORY_CONFIG)
    print(f"TT pos_embeds shape: {pos_embeds.shape}")
    pos_embeds = ttnn.to_layout(pos_embeds, ttnn.TILE_LAYOUT)

    tt_hidden = ttnn.add(inputs_embeds, pos_embeds, memory_config=ttnn.L1_MEMORY_CONFIG)
    print(f"TT combined hidden shape: {tt_hidden.shape}")

    tt_hidden_torch = ttnn.to_torch(tt_hidden)
    print(f"TT combined hidden torch shape: {tt_hidden_torch.shape}")

    # Match shapes for PCC comparison
    ref_combined = ref_input_embeds + ref_pos_embeds
    if tt_hidden_torch.dim() != ref_combined.dim():
        tt_hidden_torch = tt_hidden_torch.view_as(ref_combined)
    embed_pcc = pcc(tt_hidden_torch, ref_combined)
    print(f"=== EMBEDDING PCC: {embed_pcc:.6f} ===")

    # --- Per-layer PCC ---
    memory_config = ttnn.L1_MEMORY_CONFIG
    print(f"\nLayer-by-layer PCC:")
    for i, block in enumerate(tt_model.blocks):
        tt_hidden, _ = block(tt_hidden, layer_past=None, use_cache=False, memory_config=memory_config)
        tt_layer_torch = ttnn.to_torch(tt_hidden)
        ref_layer = ref_layer_outputs[i]

        # Match shapes
        if tt_layer_torch.dim() != ref_layer.dim():
            tt_layer_torch = tt_layer_torch.view_as(ref_layer)
        layer_pcc = pcc(tt_layer_torch, ref_layer)
        print(f"  Layer {i:2d} PCC: {layer_pcc:.6f}  TT shape: {tt_hidden.shape}")

    # Final LN + LM head
    tt_hidden_ln = ttnn.layer_norm(tt_hidden, epsilon=1e-5, weight=tt_model.ln_f_weight, bias=tt_model.ln_f_bias)
    tt_logits = ttnn.linear(
        tt_hidden_ln,
        tt_model.lm_head_weight,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=tt_model.compute_kernel_config,
    )
    tt_logits_torch = ttnn.to_torch(tt_logits)
    if tt_logits_torch.dim() != ref_logits.dim():
        tt_logits_torch = tt_logits_torch.view_as(ref_logits)
    logits_pcc = pcc(tt_logits_torch, ref_logits)
    print(f"\n=== FINAL LOGITS PCC: {logits_pcc:.6f} ===")

    ttnn.close_device(device)
    print("Done.")


if __name__ == "__main__":
    main()
