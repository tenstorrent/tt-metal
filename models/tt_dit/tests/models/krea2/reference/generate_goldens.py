# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Generate golden reference tensors for the KREA-2 (Krea2) diffusion transformer port.

Reference = diffusers `main`, shadowed via PYTHONPATH (installed diffusers 0.38.0 lacks Krea2):

    cd /localdev/vsuresh/tt-metal
    source python_env/bin/activate
    export PYTHONPATH=/localdev/vsuresh/diffusers_main/src:$PYTHONPATH
    python models/tt_dit/tests/models/krea2/reference/generate_goldens.py

Writes one `.pt` per module to `goldens/<name>.pt`, each a dict with keys:
    config, inputs, state_dict, output, meta
All tensors are fp32 on CPU. Modules run in eval() / no_grad().
"""

import os

import torch
from diffusers.models.transformers.transformer_krea2 import (
    Krea2Attention,
    Krea2FinalLayer,
    Krea2RMSNorm,
    Krea2RotaryPosEmbed,
    Krea2SwiGLU,
    Krea2TextFusion,
    Krea2TextFusionBlock,
    Krea2TextProjection,
    Krea2TimestepEmbedding,
    Krea2Transformer2DModel,
    Krea2TransformerBlock,
)

GOLDENS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")

# ----------------------------------------------------------------------------
# Reduced ("small") config -- exercises every code path cheaply on CPU.
# Real config values are in the trailing comments.
# ----------------------------------------------------------------------------
SMALL = dict(
    in_channels=64,
    num_layers=2,  # real 28
    attention_head_dim=128,
    num_attention_heads=8,  # real 48
    num_key_value_heads=2,  # real 12
    intermediate_size=2048,  # real 16384
    timestep_embed_dim=256,
    text_hidden_dim=256,  # real 2560
    num_text_layers=12,
    text_num_attention_heads=4,  # real 20
    text_num_key_value_heads=4,  # real 20
    text_intermediate_size=512,  # real 6912
    num_layerwise_text_blocks=2,
    num_refiner_text_blocks=2,
    axes_dims_rope=(32, 48, 48),  # sum == head_dim == 128
    rope_theta=1000.0,
    norm_eps=1e-5,
)

# Real-width single-layer variant to catch shape/tiling issues at true width.
REAL1 = dict(
    in_channels=64,
    num_layers=1,  # real 28
    attention_head_dim=128,
    num_attention_heads=48,
    num_key_value_heads=12,
    intermediate_size=16384,
    timestep_embed_dim=256,
    text_hidden_dim=2560,
    num_text_layers=12,
    text_num_attention_heads=20,
    text_num_key_value_heads=20,
    text_intermediate_size=6912,
    num_layerwise_text_blocks=2,
    num_refiner_text_blocks=2,
    axes_dims_rope=(32, 48, 48),
    rope_theta=1000.0,
    norm_eps=1e-5,
)

# Sequence dimensions (shared by small and real1; kept small so real1 stays cheap).
BATCH = 1
TEXT_SEQ = 16
GRID_H = 8
GRID_W = 8
IMAGE_SEQ = GRID_H * GRID_W  # 64

_written = []  # (name, output_shape, checksum)


def hidden_size(cfg):
    return cfg["attention_head_dim"] * cfg["num_attention_heads"]


def to_fp32_state_dict(module):
    return {k: v.detach().float().cpu().clone() for k, v in module.state_dict().items()}


def to_fp32_cpu(t):
    return t.detach().float().cpu().clone()


def save_golden(name, config, inputs, module, output, seed, dims):
    payload = {
        "config": config,
        "inputs": {k: to_fp32_cpu(v) for k, v in inputs.items()},
        "state_dict": to_fp32_state_dict(module),
        "output": to_fp32_cpu(output),
        "meta": {"seed": seed, "dims": dims},
    }
    path = os.path.join(GOLDENS_DIR, f"{name}.pt")
    torch.save(payload, path)
    chk = output.float().abs().mean().item()
    _written.append((name, tuple(output.shape), chk))
    print(f"  wrote {name}.pt  output={tuple(output.shape)}  checksum={chk:.6f}")


def build_rope(cfg, position_ids):
    """Mirror Krea2Transformer2DModel: rope from position_ids via Krea2RotaryPosEmbed."""
    rope_mod = Krea2RotaryPosEmbed(theta=cfg["rope_theta"], axes_dim=list(cfg["axes_dims_rope"]))
    cos, sin = rope_mod(position_ids)  # each (seq, head_dim)
    return cos, sin


def build_position_ids():
    """position_ids (text_seq + image_seq, 3): text rows all-zero; image rows (0, h, w)."""
    seq = TEXT_SEQ + IMAGE_SEQ
    pos = torch.zeros(seq, 3)
    idx = TEXT_SEQ
    for h in range(GRID_H):
        for w in range(GRID_W):
            pos[idx, 0] = 0.0
            pos[idx, 1] = float(h)
            pos[idx, 2] = float(w)
            idx += 1
    return pos


def concat_key_padding_mask(encoder_attention_mask):
    """(B, text_seq) -> full-seq key-padding mask (B, 1, 1, text_seq+image_seq), matching the reference forward."""
    b = encoder_attention_mask.shape[0]
    image_mask = encoder_attention_mask.new_ones((b, IMAGE_SEQ))
    full = torch.cat([encoder_attention_mask, image_mask], dim=1)
    return full[:, None, None, :]


# ----------------------------------------------------------------------------
# Per-module golden generators
# ----------------------------------------------------------------------------
def gen_rms_norm(cfg, dims):
    seed = 1001
    torch.manual_seed(seed)
    dim = hidden_size(cfg)
    m = Krea2RMSNorm(dim, eps=cfg["norm_eps"]).eval()
    # weight is zero-init; randomize so scale != 1 exercises the "1+weight" path.
    with torch.no_grad():
        m.weight.copy_(torch.randn(dim) * 0.1)
    x = torch.randn(BATCH, TEXT_SEQ + IMAGE_SEQ, dim)
    with torch.no_grad():
        out = m(x)
    save_golden(f"rms_norm_{dims}", {"dim": dim, "eps": cfg["norm_eps"]}, {"hidden_states": x}, m, out, seed, dims)


def gen_swiglu(cfg, dims):
    seed = 1002
    torch.manual_seed(seed)
    dim = hidden_size(cfg)
    inter = cfg["intermediate_size"]
    m = Krea2SwiGLU(dim, inter).eval()
    x = torch.randn(BATCH, TEXT_SEQ + IMAGE_SEQ, dim)
    with torch.no_grad():
        out = m(x)
    save_golden(f"swiglu_{dims}", {"dim": dim, "hidden_dim": inter}, {"hidden_states": x}, m, out, seed, dims)


def gen_rotary_pos_embed(cfg, dims):
    seed = 1003
    torch.manual_seed(seed)
    position_ids = build_position_ids()
    m = Krea2RotaryPosEmbed(theta=cfg["rope_theta"], axes_dim=list(cfg["axes_dims_rope"])).eval()
    with torch.no_grad():
        cos, sin = m(position_ids)
    # No trainable params; output is (cos, sin). Store cos as `output` and sin in inputs for round-trip.
    payload_out = torch.stack([cos, sin], dim=0)  # (2, seq, head_dim)
    save_golden(
        f"rotary_pos_embed_{dims}",
        {"theta": cfg["rope_theta"], "axes_dim": list(cfg["axes_dims_rope"])},
        {"position_ids": position_ids, "cos": cos, "sin": sin},
        m,
        payload_out,
        seed,
        dims,
    )


def gen_attention(cfg, dims, with_mask):
    seed = 1004 if with_mask else 1005
    torch.manual_seed(seed)
    dim = hidden_size(cfg)
    seq = TEXT_SEQ + IMAGE_SEQ
    m = Krea2Attention(
        hidden_size=dim,
        num_heads=cfg["num_attention_heads"],
        num_kv_heads=cfg["num_key_value_heads"],
        eps=cfg["norm_eps"],
    ).eval()
    # Randomize the zero-init q/k norm weights so QK-RMSNorm scaling is non-trivial.
    with torch.no_grad():
        m.norm_q.weight.copy_(torch.randn(cfg["attention_head_dim"]) * 0.1)
        m.norm_k.weight.copy_(torch.randn(cfg["attention_head_dim"]) * 0.1)

    x = torch.randn(BATCH, seq, dim)
    position_ids = build_position_ids()
    cos, sin = build_rope(cfg, position_ids)

    inputs = {
        "hidden_states": x,
        "image_rotary_emb_cos": cos,
        "image_rotary_emb_sin": sin,
        "position_ids": position_ids,
    }
    if with_mask:
        encoder_attention_mask = torch.ones(BATCH, TEXT_SEQ)
        encoder_attention_mask[:, TEXT_SEQ // 2 :] = 0.0  # mask out some text keys
        attention_mask = concat_key_padding_mask(encoder_attention_mask)  # (B,1,1,seq) bool-like float
        attn_mask_bool = attention_mask.bool()
        inputs["encoder_attention_mask"] = encoder_attention_mask
        inputs["attention_mask"] = attention_mask
    else:
        attn_mask_bool = None

    with torch.no_grad():
        out = m(x, attention_mask=attn_mask_bool, image_rotary_emb=(cos, sin))

    name = f"attention_{'mask' if with_mask else 'nomask'}_{dims}"
    save_golden(
        name,
        {
            "hidden_size": dim,
            "num_heads": cfg["num_attention_heads"],
            "num_kv_heads": cfg["num_key_value_heads"],
            "eps": cfg["norm_eps"],
        },
        inputs,
        m,
        out,
        seed,
        dims,
    )


def gen_text_fusion_block(cfg, dims):
    seed = 1006
    torch.manual_seed(seed)
    dim = cfg["text_hidden_dim"]
    m = Krea2TextFusionBlock(
        dim=dim,
        num_heads=cfg["text_num_attention_heads"],
        num_kv_heads=cfg["text_num_key_value_heads"],
        intermediate_size=cfg["text_intermediate_size"],
        eps=cfg["norm_eps"],
    ).eval()
    with torch.no_grad():
        m.attn.norm_q.weight.copy_(torch.randn_like(m.attn.norm_q.weight) * 0.1)
        m.attn.norm_k.weight.copy_(torch.randn_like(m.attn.norm_k.weight) * 0.1)
    # Refiner-style usage: attend across token axis with a padding mask.
    x = torch.randn(BATCH, TEXT_SEQ, dim)
    encoder_attention_mask = torch.ones(BATCH, TEXT_SEQ)
    encoder_attention_mask[:, TEXT_SEQ // 2 :] = 0.0
    attn_mask = encoder_attention_mask[:, None, None, :].bool()
    with torch.no_grad():
        out = m(x, attention_mask=attn_mask)
    save_golden(
        f"text_fusion_block_{dims}",
        {
            "dim": dim,
            "num_heads": cfg["text_num_attention_heads"],
            "num_kv_heads": cfg["text_num_key_value_heads"],
            "intermediate_size": cfg["text_intermediate_size"],
            "eps": cfg["norm_eps"],
        },
        {"hidden_states": x, "encoder_attention_mask": encoder_attention_mask, "attention_mask": attn_mask.float()},
        m,
        out,
        seed,
        dims,
    )


def gen_text_fusion(cfg, dims):
    seed = 1007
    torch.manual_seed(seed)
    dim = cfg["text_hidden_dim"]
    m = Krea2TextFusion(
        num_text_layers=cfg["num_text_layers"],
        dim=dim,
        num_heads=cfg["text_num_attention_heads"],
        num_kv_heads=cfg["text_num_key_value_heads"],
        intermediate_size=cfg["text_intermediate_size"],
        num_layerwise_blocks=cfg["num_layerwise_text_blocks"],
        num_refiner_blocks=cfg["num_refiner_text_blocks"],
        eps=cfg["norm_eps"],
    ).eval()
    with torch.no_grad():
        for blk in list(m.layerwise_blocks) + list(m.refiner_blocks):
            blk.attn.norm_q.weight.copy_(torch.randn_like(blk.attn.norm_q.weight) * 0.1)
            blk.attn.norm_k.weight.copy_(torch.randn_like(blk.attn.norm_k.weight) * 0.1)
    ehs = torch.randn(BATCH, TEXT_SEQ, cfg["num_text_layers"], dim)
    encoder_attention_mask = torch.ones(BATCH, TEXT_SEQ)
    encoder_attention_mask[:, TEXT_SEQ // 2 :] = 0.0
    text_attention_mask = encoder_attention_mask[:, None, None, :].bool()
    with torch.no_grad():
        out = m(ehs, attention_mask=text_attention_mask)
    save_golden(
        f"text_fusion_{dims}",
        {
            "num_text_layers": cfg["num_text_layers"],
            "dim": dim,
            "num_heads": cfg["text_num_attention_heads"],
            "num_kv_heads": cfg["text_num_key_value_heads"],
            "intermediate_size": cfg["text_intermediate_size"],
            "num_layerwise_blocks": cfg["num_layerwise_text_blocks"],
            "num_refiner_blocks": cfg["num_refiner_text_blocks"],
            "eps": cfg["norm_eps"],
        },
        {
            "encoder_hidden_states": ehs,
            "encoder_attention_mask": encoder_attention_mask,
            "attention_mask": text_attention_mask.float(),
        },
        m,
        out,
        seed,
        dims,
    )


def gen_timestep_embedding(cfg, dims):
    seed = 1008
    torch.manual_seed(seed)
    dim = hidden_size(cfg)
    m = Krea2TimestepEmbedding(cfg["timestep_embed_dim"], dim).eval()
    timestep = torch.rand(BATCH)  # in [0,1]
    with torch.no_grad():
        out = m(timestep, dtype=torch.float32)
    save_golden(
        f"timestep_embedding_{dims}",
        {"embed_dim": cfg["timestep_embed_dim"], "hidden_size": dim},
        {"timestep": timestep},
        m,
        out,
        seed,
        dims,
    )


def gen_text_projection(cfg, dims):
    seed = 1009
    torch.manual_seed(seed)
    text_dim = cfg["text_hidden_dim"]
    dim = hidden_size(cfg)
    m = Krea2TextProjection(text_dim, dim, eps=cfg["norm_eps"]).eval()
    x = torch.randn(BATCH, TEXT_SEQ, text_dim)
    with torch.no_grad():
        out = m(x)
    save_golden(
        f"text_projection_{dims}",
        {"text_dim": text_dim, "hidden_size": dim, "eps": cfg["norm_eps"]},
        {"hidden_states": x},
        m,
        out,
        seed,
        dims,
    )


def gen_transformer_block(cfg, dims):
    seed = 1010
    torch.manual_seed(seed)
    dim = hidden_size(cfg)
    seq = TEXT_SEQ + IMAGE_SEQ
    m = Krea2TransformerBlock(
        hidden_size=dim,
        intermediate_size=cfg["intermediate_size"],
        num_heads=cfg["num_attention_heads"],
        num_kv_heads=cfg["num_key_value_heads"],
        norm_eps=cfg["norm_eps"],
    ).eval()
    with torch.no_grad():
        m.scale_shift_table.copy_(torch.randn(6, dim) * 0.1)
        m.attn.norm_q.weight.copy_(torch.randn(cfg["attention_head_dim"]) * 0.1)
        m.attn.norm_k.weight.copy_(torch.randn(cfg["attention_head_dim"]) * 0.1)

    x = torch.randn(BATCH, seq, dim)
    # temb = the 6*hidden modulation vector (temb_mod), shape (B, 1, 6*hidden).
    temb = torch.randn(BATCH, 1, 6 * dim) * 0.1
    position_ids = build_position_ids()
    cos, sin = build_rope(cfg, position_ids)

    encoder_attention_mask = torch.ones(BATCH, TEXT_SEQ)
    encoder_attention_mask[:, TEXT_SEQ // 2 :] = 0.0
    attention_mask = concat_key_padding_mask(encoder_attention_mask)
    attn_mask_bool = attention_mask.bool()

    with torch.no_grad():
        out = m(x, temb, (cos, sin), attention_mask=attn_mask_bool)

    save_golden(
        f"transformer_block_{dims}",
        {
            "hidden_size": dim,
            "intermediate_size": cfg["intermediate_size"],
            "num_heads": cfg["num_attention_heads"],
            "num_kv_heads": cfg["num_key_value_heads"],
            "norm_eps": cfg["norm_eps"],
        },
        {
            "hidden_states": x,
            "temb": temb,
            "image_rotary_emb_cos": cos,
            "image_rotary_emb_sin": sin,
            "position_ids": position_ids,
            "encoder_attention_mask": encoder_attention_mask,
            "attention_mask": attention_mask,
        },
        m,
        out,
        seed,
        dims,
    )


def gen_final_layer(cfg, dims):
    seed = 1011
    torch.manual_seed(seed)
    dim = hidden_size(cfg)
    m = Krea2FinalLayer(dim, out_channels=cfg["in_channels"], eps=cfg["norm_eps"]).eval()
    with torch.no_grad():
        m.scale_shift_table.copy_(torch.randn(2, dim) * 0.1)
    x = torch.randn(BATCH, IMAGE_SEQ, dim)
    # temb = raw time embedding (not temb_mod), shape (B, 1, hidden) as produced by time_embed.
    # FinalLayer adds a (2, hidden) table: temb (B,1,hidden) + table (2,hidden) -> (B,2,hidden),
    # then chunk(2, dim=1) -> scale/shift each (B,1,hidden), broadcasting over the token axis.
    temb = torch.randn(BATCH, 1, dim) * 0.1
    with torch.no_grad():
        out = m(x, temb)
    save_golden(
        f"final_layer_{dims}",
        {"hidden_size": dim, "out_channels": cfg["in_channels"], "eps": cfg["norm_eps"]},
        {"hidden_states": x, "temb": temb},
        m,
        out,
        seed,
        dims,
    )


def gen_transformer_full(cfg, dims, with_mask):
    seed = 1012 if with_mask else 1013
    torch.manual_seed(seed)
    m = Krea2Transformer2DModel(**cfg).eval()
    # Randomize all zero-init norm/table params so paths are non-trivial.
    with torch.no_grad():
        for _, p in m.named_parameters():
            pass  # keep default random init from constructor; norms/tables are zero-init on purpose.
        # Give the RMSNorm scales and modulation tables non-zero values.
        for mod in m.modules():
            if isinstance(mod, Krea2RMSNorm):
                mod.weight.copy_(torch.randn_like(mod.weight) * 0.1)
        for blk in m.transformer_blocks:
            blk.scale_shift_table.copy_(torch.randn_like(blk.scale_shift_table) * 0.1)
        m.final_layer.scale_shift_table.copy_(torch.randn_like(m.final_layer.scale_shift_table) * 0.1)

    hidden_states = torch.randn(BATCH, IMAGE_SEQ, cfg["in_channels"])
    encoder_hidden_states = torch.randn(BATCH, TEXT_SEQ, cfg["num_text_layers"], cfg["text_hidden_dim"])
    timestep = torch.rand(BATCH)  # in [0,1]
    position_ids = build_position_ids()

    if with_mask:
        encoder_attention_mask = torch.ones(BATCH, TEXT_SEQ)
        encoder_attention_mask[:, TEXT_SEQ // 2 :] = 0.0
    else:
        encoder_attention_mask = torch.ones(BATCH, TEXT_SEQ)

    with torch.no_grad():
        out = m(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            position_ids=position_ids,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )[0]

    name = f"transformer_full_{'mask' if with_mask else 'nomask'}_{dims}"
    save_golden(
        name,
        dict(cfg),
        {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "position_ids": position_ids,
            "encoder_attention_mask": encoder_attention_mask,
        },
        m,
        out,
        seed,
        dims,
    )


def main():
    os.makedirs(GOLDENS_DIR, exist_ok=True)
    torch.set_grad_enabled(False)

    for cfg, dims in [(SMALL, "small"), (REAL1, "real1")]:
        print(f"\n=== dims={dims} ===")
        gen_rms_norm(cfg, dims)
        gen_swiglu(cfg, dims)
        gen_rotary_pos_embed(cfg, dims)
        gen_attention(cfg, dims, with_mask=False)
        gen_attention(cfg, dims, with_mask=True)
        gen_text_fusion_block(cfg, dims)
        gen_text_fusion(cfg, dims)
        gen_timestep_embedding(cfg, dims)
        gen_text_projection(cfg, dims)
        gen_transformer_block(cfg, dims)
        gen_final_layer(cfg, dims)
        gen_transformer_full(cfg, dims, with_mask=False)
        gen_transformer_full(cfg, dims, with_mask=True)

    # ------------------------------------------------------------------
    # Round-trip verification: every file must load back and match.
    # ------------------------------------------------------------------
    print("\n=== round-trip verification ===")
    all_ok = True
    for name, shape, chk in _written:
        path = os.path.join(GOLDENS_DIR, f"{name}.pt")
        loaded = torch.load(path, weights_only=False)
        assert set(loaded.keys()) >= {"config", "inputs", "state_dict", "output", "meta"}, name
        out = loaded["output"]
        assert tuple(out.shape) == shape, f"{name}: shape mismatch {tuple(out.shape)} != {shape}"
        rl_chk = out.float().abs().mean().item()
        ok = abs(rl_chk - chk) < 1e-6
        all_ok = all_ok and ok
        print(f"  {name:36s} shape={str(shape):24s} checksum={rl_chk:.6f} {'OK' if ok else 'MISMATCH'}")

    print("\n=== summary ===")
    print(f"total golden files: {len(_written)}")
    print(f"round-trip: {'ALL OK' if all_ok else 'FAILURES PRESENT'}")


if __name__ == "__main__":
    main()
