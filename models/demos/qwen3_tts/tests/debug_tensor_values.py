#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Tensor value debugger: traces actual min/max/mean/std and max-abs-error
through each layer sub-operation (attn vs MLP) for reference vs TTNN.

Usage:
    python -u models/demos/qwen3_tts/tests/debug_tensor_values.py
"""
from pathlib import Path

import torch
import torch.nn.functional as F

import ttnn


def stats(name, t, ref=None, width=38):
    t = t.flatten().float()
    s = f"  {name:{width}s}  min={t.min():8.2f}  max={t.max():8.2f}  mean={t.mean():7.3f}  std={t.std():6.3f}"
    if ref is not None:
        ref = ref.flatten().float()
        err = (t - ref).abs()
        s += f"  |  maxE={err.max():7.3f}  meanE={err.mean():.4f}"
    print(s)


def main():
    print("Loading weights...")
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    model_path = Path(snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", allow_patterns=["*.safetensors"]))
    wd = {}
    for f in model_path.glob("*.safetensors"):
        if "speech_tokenizer" not in str(f):
            wd.update(load_file(f))

    from transformers import AutoTokenizer

    from models.demos.qwen3_tts.demo.demo_pure_reference_tts import TTSConfig, create_icl_embedding

    config = TTSConfig()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", trust_remote_code=True)
    cache = torch.load("/tmp/jim_ref.refcache.pt", weights_only=True)
    ref_codes = cache["ref_codes"].long()
    inputs_embeds, _, _ = create_icl_embedding(
        "Hello, welcome to Tenstorrent!!",
        "Jason, can you put up the high level overview slides?",
        ref_codes,
        tokenizer,
        wd,
        config,
        torch.zeros(1, 1, 2048),
        "english",
    )
    seq_len = inputs_embeds.shape[1]
    pad_seq = ((seq_len + 31) // 32) * 32
    print(f"ICL embedding: {inputs_embeds.shape}  pad_seq={pad_seq}")

    # Reference setup
    from models.demos.qwen3_tts.reference.functional import (
        Qwen3TTSConfig,
        attention,
        compute_mrope_frequencies,
        extract_talker_weights,
        rms_norm,
        swiglu_mlp,
    )

    rcfg = Qwen3TTSConfig()
    cos_ref, sin_ref = compute_mrope_frequencies(rcfg.head_dim, seq_len, rcfg.rope_theta)
    attn_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).unsqueeze(0).unsqueeze(0)
    talker_weights = extract_talker_weights(wd)

    # TTNN setup
    device = ttnn.open_device(device_id=0)
    from models.demos.qwen3_tts.tt.decoder_layer import DecoderLayer
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig
    from models.demos.qwen3_tts.tt.rope import get_rope_tensors, get_transformation_mat

    tcfg = Qwen3TTSTalkerConfig()
    pos_ids = torch.arange(pad_seq)
    cos_tt, sin_tt = get_rope_tensors(device, tcfg.head_dim, pad_seq, pos_ids, tcfg.rope_theta)
    trans_mat = get_transformation_mat(tcfg.head_dim, device)

    NUM_LAYERS = 8
    layers_tt = []
    for i in range(NUM_LAYERS):
        layers_tt.append(
            DecoderLayer(
                device=device,
                hidden_size=tcfg.hidden_size,
                num_heads=tcfg.num_attention_heads,
                num_kv_heads=tcfg.num_key_value_heads,
                head_dim=tcfg.head_dim,
                intermediate_size=tcfg.intermediate_size,
                state_dict=wd,
                layer_idx=i,
                layer_prefix="talker.model",
                rms_norm_eps=tcfg.rms_norm_eps,
                weight_dtype=ttnn.bfloat16,
            )
        )

    print()
    hdr = "  {:3s}  {:{w}s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>8s}  {:>9s}".format(
        "Lyr", "Sub-operation", "Ref-min", "Ref-max", "TT-min", "TT-max", "MaxAbsErr", "MeanAbsErr", w=38
    )
    print(hdr)
    print("-" * 110)

    ref_x = inputs_embeds.float()
    inp_padded = F.pad(inputs_embeds, (0, 0, 0, pad_seq - seq_len)).unsqueeze(1).to(torch.bfloat16)
    tt_x = ttnn.from_torch(
        inp_padded, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    for i in range(NUM_LAYERS):
        pfx = f"layers.{i}."
        lw = {k.replace(pfx, ""): v.float() for k, v in talker_weights.items() if k.startswith(pfx)}

        # ---- Reference sub-ops ----
        ref_normed = rms_norm(ref_x, lw["input_layernorm.weight"], rcfg.rms_norm_eps)
        ref_attn = attention(
            ref_normed,
            q_proj_weight=lw["self_attn.q_proj.weight"],
            k_proj_weight=lw["self_attn.k_proj.weight"],
            v_proj_weight=lw["self_attn.v_proj.weight"],
            o_proj_weight=lw["self_attn.o_proj.weight"],
            q_norm_weight=lw["self_attn.q_norm.weight"],
            k_norm_weight=lw["self_attn.k_norm.weight"],
            cos=cos_ref,
            sin=sin_ref,
            num_heads=rcfg.num_attention_heads,
            num_kv_heads=rcfg.num_key_value_heads,
            head_dim=rcfg.head_dim,
            rms_norm_eps=rcfg.rms_norm_eps,
            attention_mask=attn_mask,
            use_mrope=True,
        )
        ref_x_mid = ref_x + ref_attn
        ref_normed2 = rms_norm(ref_x_mid, lw["post_attention_layernorm.weight"], rcfg.rms_norm_eps)
        ref_mlp = swiglu_mlp(
            ref_normed2, lw["mlp.gate_proj.weight"], lw["mlp.up_proj.weight"], lw["mlp.down_proj.weight"]
        )
        ref_x_new = ref_x_mid + ref_mlp

        # ---- TTNN full layer ----
        tt_x_new, _ = layers_tt[i](
            tt_x, cos_tt, sin_tt, trans_mat, attention_mask=None, kv_cache=None, start_pos=0, mode="prefill"
        )
        tt_out = ttnn.to_torch(tt_x_new).squeeze(1).float()[:, :seq_len, :]

        # ---- Print per-layer sub-op stats ----
        def row(label, ref_t, tt_t=None):
            rmin, rmax = ref_t.min().item(), ref_t.max().item()
            if tt_t is not None:
                tmin, tmax = tt_t.min().item(), tt_t.max().item()
                err = (tt_t - ref_t).abs()
                print(
                    f"  {i:3d}  {label:{38}s}  {rmin:8.2f}  {rmax:8.2f}  {tmin:8.2f}  {tmax:8.2f}  {err.max():8.3f}  {err.mean():9.4f}"
                )
            else:
                print(f"  {i:3d}  {label:{38}s}  {rmin:8.2f}  {rmax:8.2f}")

        row("input x", ref_x)
        row("  after input_layernorm", ref_normed)
        row("  after attention", ref_attn)
        row("  after attn+residual", ref_x_mid)
        row("  after post_attn_layernorm", ref_normed2)
        row("  mlp raw output", ref_mlp)
        row("  LAYER OUTPUT (ref vs TTNN)", ref_x_new, tt_out)
        print()

        ref_x = ref_x_new
        tt_x = tt_x_new

    ttnn.close_device(device)
    print("Done.")


if __name__ == "__main__":
    main()
