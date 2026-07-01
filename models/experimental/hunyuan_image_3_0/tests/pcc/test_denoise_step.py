# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Integration PCC test — ONE HunyuanImage-3.0 diffusion denoising step, end to
# end, composing the already-ported building blocks against the PyTorch
# reference on the REAL checkpoint weights:
#
#     noised latent  --patch_embed(UNetDown)+time_embed-->  image tokens
#     image tokens scattered into a [text | image | text] sequence
#     sequence  --transformer backbone (N MoE layers, NO ln_f)-->  hidden
#     hidden[image span]  --final_layer(UNetUp)+time_embed_2-->  velocity pred
#
# This mirrors the gen_image path of HunyuanImage3ForCausalMM.forward
# (modeling_hunyuan_image_3.py): instantiate_vae_image_tokens -> self.model
# (ln_f skipped for gen_image) -> ragged_final_layer. It is the first test that
# wires patch_embed + backbone + final_layer together, validating the GLUE
# (image-token scatter, 2D-RoPE image span, bidirectional image mask, timestep
# conditioning hand-off) — each individual block already has its own PCC test.
#
# Conditioning: time_embed / time_embed_2 are evaluated once in fp32 (real
# weights) and the SAME vectors feed both the reference and TT patch/final
# modules, isolating the composition from time-embedding numerics (which have
# their own test). Module hand-offs round-trip through host — that is fine for a
# composition check; on-device chaining is the pipeline's job (next milestone).
#
# Layer count is env-driven. The default (4 layers) keeps backbone bf16 drift
# negligible so the integration glue is gated cleanly at PCC>=0.99. Running with
# HY_NUM_LAYERS=32 exercises the full real step but inherits the known
# free-running 32-layer backbone drift (~0.88, see test_model_teacher_forced.py)
# — so the gate is relaxed accordingly for large stacks.
#
# Run (pytest):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_denoise_step.py -v -s --timeout=1800
# Run (script, full step):
#   HY_NUM_LAYERS=32 python_env/bin/python \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_denoise_step.py

import os, sys, json, glob, gc
import torch
from safetensors import safe_open

ROOT = "/home/iguser/ign-tt/tt-metal"
HUNYUAN = "/home/iguser/ign-tt/hunyan_instruct"
WEIGHTS = "/home/iguser/ign-tt/base"
for p in (ROOT, HUNYUAN):
    if p not in sys.path:
        sys.path.insert(0, p)

import ttnn
from models.experimental.hunyuan_image_3_0.ref.transformer_layer import HunyuanImage3DecoderLayer as RefLayer
from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive
from models.experimental.hunyuan_image_3_0.ref.image_gen.patch_embed import UNetDown as RefDown, UNetUp as RefUp
from models.experimental.hunyuan_image_3_0.ref.image_gen.timestep_embedder import TimestepEmbedder as RefTimeEmbed
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel
from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown, HunyuanTtUNetUp
from hunyuan_image_3.modeling_hunyuan_image_3 import build_batch_2d_rope

# Sequence layout: a contiguous block of image tokens between two text spans.
B = 1
TEXT_PRE, TEXT_POST = 32, 32
GRID = 8  # latent grid -> token_h == token_w == GRID, n_img = GRID*GRID image tokens
N_IMG = GRID * GRID
S = TEXT_PRE + N_IMG + TEXT_POST  # 128
IMG_START = TEXT_PRE
IMG_SLICE = slice(IMG_START, IMG_START + N_IMG)

NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "4"))
# Negligible drift for a few layers; full 32-layer free-running path carries the
# known ~0.88 backbone drift, so gate large stacks more loosely.
PCC_THR = 0.99 if NUM_LAYERS <= 8 else 0.85


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


# --- sharded weight loading -------------------------------------------------
_WMAP = json.load(open(glob.glob(f"{WEIGHTS}/*.index.json")[0]))["weight_map"]
_OPEN = {}


def _load(key):
    shard = _WMAP[key]
    f = _OPEN.get(shard) or _OPEN.setdefault(shard, safe_open(f"{WEIGHTS}/{shard}", framework="pt"))
    return f.get_tensor(key)


def _load_prefix(prefix):
    return {k[len(prefix) + 1 :]: _load(k) for k in _WMAP if k.startswith(prefix + ".")}


def _cfg():
    cfg = json.load(open(f"{WEIGHTS}/config.json"))
    first = lambda v: v if isinstance(v, int) else v[0]
    return dict(
        H=cfg["hidden_size"],
        HEADS=cfg["num_attention_heads"],
        KV_HEADS=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        HEAD_DIM=cfg.get("attention_head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
        NUM_EXPERTS=first(cfg["num_experts"]),
        MOE_TOPK=first(cfg["moe_topk"]),
        MOE_INTER=first(cfg["moe_intermediate_size"]),
        NUM_SHARED=first(cfg["num_shared_expert"]),
        NORM_TOPK=cfg.get("norm_topk_prob", True),
        EPS=cfg.get("rms_norm_eps", 1e-5),
        USE_QK_NORM=cfg.get("use_qk_norm", True),
        USE_MIXED=cfg.get("use_mixed_mlp_moe", True),
    )


def _pe_dims(down_sd):
    hid, latent = down_sd["model.0.weight"].shape[:2]
    hsz = down_sd["model.1.in_layers.2.weight"].shape[0]
    return int(latent), int(hid), int(hsz)


def _time_embed(prefix, H, timesteps):
    """fp32 reference TimestepEmbedder on real weights -> [B, H]."""
    sd = _load_prefix(prefix)
    te = RefTimeEmbed(H).eval()
    te.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    with torch.no_grad():
        return te(timesteps)  # [B, H]


def _reference(c, latent, t_emb1, t_emb2, text_embeds, down_sd, up_sd):
    """fp32 gen_image step -> velocity prediction [B, latent_ch, GRID, GRID]."""
    LATENT, HID, HSZ = _pe_dims(down_sd)

    ref_down = RefDown(1, LATENT, HSZ, HID, HSZ).eval()
    ref_up = RefUp(1, HSZ, HSZ, HID, LATENT, out_norm=True).eval()
    ref_down.load_state_dict({k: v.float() for k, v in down_sd.items()}, strict=True)
    ref_up.load_state_dict({k: v.float() for k, v in up_sd.items()}, strict=True)

    cos, sin = build_batch_2d_rope(S, c["HEAD_DIM"], image_infos=[[(IMG_SLICE, (GRID, GRID))]])
    mask_add = to_additive(build_attention_mask(S, image_slices=[IMG_SLICE], bsz=B), dtype=torch.float32)

    with torch.no_grad():
        img_tokens, th, tw = ref_down(latent, t_emb1)  # [B, N_IMG, H]
        h = text_embeds.clone()
        h[:, IMG_SLICE, :] = img_tokens
        for i in range(NUM_LAYERS):
            sd = _load_prefix(f"model.layers.{i}")
            layer = RefLayer(
                hidden_size=c["H"],
                num_attention_heads=c["HEADS"],
                num_key_value_heads=c["KV_HEADS"],
                attention_head_dim=c["HEAD_DIM"],
                num_experts=c["NUM_EXPERTS"],
                moe_topk=c["MOE_TOPK"],
                moe_intermediate_size=c["MOE_INTER"],
                num_shared_expert=c["NUM_SHARED"],
                use_mixed_mlp_moe=c["USE_MIXED"],
                norm_topk_prob=c["NORM_TOPK"],
                use_qk_norm=c["USE_QK_NORM"],
                rms_norm_eps=c["EPS"],
                layer_idx=i,
            )
            layer.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
            layer.eval()
            h = layer(h, attention_mask=mask_add, custom_pos_emb=(cos, sin))  # gen_image: NO ln_f
            del layer
            gc.collect()
        img_out = h[:, IMG_SLICE, :]  # masked_select for a contiguous image block
        pred = ref_up(img_out, t_emb2, th, tw)  # [B, LATENT, GRID, GRID]

    return pred, (th, tw), (LATENT, HID, HSZ), img_tokens


def _run(device):
    c = _cfg()
    down_sd = _load_prefix("patch_embed")
    up_sd = _load_prefix("final_layer")
    H = c["H"]
    print(
        f"config: H={H} heads={c['HEADS']}/{c['KV_HEADS']} head_dim={c['HEAD_DIM']} "
        f"experts={c['NUM_EXPERTS']} topk={c['MOE_TOPK']}  layers={NUM_LAYERS}  "
        f"seq={S} (text {TEXT_PRE}|img {N_IMG}|text {TEXT_POST}), grid={GRID}x{GRID}"
    )

    torch.manual_seed(0)
    latent = torch.randn(B, _pe_dims(down_sd)[0], GRID, GRID)
    text_embeds = torch.randn(B, S, H) * 0.02  # stand-in for wte(prompt) embeddings
    timesteps = torch.rand(B)  # one fractional timestep in [0, 1)
    t_emb1 = _time_embed("time_embed", H, timesteps)  # for patch_embed
    t_emb2 = _time_embed("time_embed_2", H, timesteps)  # for final_layer

    ref_pred, (th, tw), (LATENT, HID, HSZ), ref_img_tokens = _reference(
        c, latent, t_emb1, t_emb2, text_embeds, down_sd, up_sd
    )

    # ---- TT path ----
    t1_tt = ttnn.from_torch(t_emb1.reshape(1, 1, B, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    t2_tt = ttnn.from_torch(t_emb2.reshape(1, 1, B, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # 1) patch_embed (UNetDown): noised latent -> image tokens [1,1,N_IMG,H]
    tt_down = HunyuanTtUNetDown(
        device,
        {f"patch_embed.{k}": v for k, v in down_sd.items()},
        in_channels=LATENT,
        hidden_channels=HID,
        out_channels=HSZ,
    )
    img_tok_tt, h2, w2 = tt_down(latent, t1_tt)
    img_tok = ttnn.to_torch(img_tok_tt).reshape(B, N_IMG, H)
    ttnn.deallocate(img_tok_tt)
    print(f"  patch_embed image-token PCC = {_pcc(ref_img_tokens, img_tok):.6f}  (tokens {h2}x{w2})")

    # 2) backbone: scatter TT image tokens into the sequence, run N MoE layers (no ln_f)
    embeds = text_embeds.clone()
    embeds[:, IMG_SLICE, :] = img_tok
    layer_loader = lambda i: {f"model.layers.{i}.{k}": v for k, v in _load_prefix(f"model.layers.{i}").items()}
    model = HunyuanTtModel(
        device,
        num_layers=NUM_LAYERS,
        hidden_size=H,
        num_heads=c["HEADS"],
        num_kv_heads=c["KV_HEADS"],
        head_dim=c["HEAD_DIM"],
        num_experts=c["NUM_EXPERTS"],
        moe_topk=c["MOE_TOPK"],
        use_qk_norm=c["USE_QK_NORM"],
        use_mixed_mlp_moe=c["USE_MIXED"],
        norm_topk_prob=c["NORM_TOPK"],
        rms_norm_eps=c["EPS"],
        stream_experts=True,
        layer_loader=layer_loader,
        apply_final_norm=False,
    )
    image_infos = [[(IMG_SLICE, (GRID, GRID))]]
    mask_add = to_additive(build_attention_mask(S, image_slices=[IMG_SLICE], bsz=B), dtype=torch.float32)
    mask_tt = ttnn.from_torch(mask_add.reshape(B, 1, S, S), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    embeds_tt = ttnn.from_torch(embeds, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    hidden_tt = model.forward(inputs_embeds=embeds_tt, seq_len=S, image_infos=image_infos, attention_mask=mask_tt)
    hidden = ttnn.to_torch(hidden_tt)[..., :H]
    ttnn.deallocate(hidden_tt)
    ttnn.deallocate(mask_tt)

    # 3) final_layer (UNetUp): image-span hidden -> velocity prediction
    img_out = hidden[:, IMG_SLICE, :].reshape(1, 1, N_IMG, H)
    img_out_tt = ttnn.from_torch(img_out, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_up = HunyuanTtUNetUp(
        device,
        {f"final_layer.{k}": v for k, v in up_sd.items()},
        in_channels=HSZ,
        hidden_channels=HID,
        out_channels=LATENT,
    )
    pred_tt, h3, w3 = tt_up(img_out_tt, t2_tt, th, tw, B=B)
    pred = ttnn.to_torch(pred_tt).reshape(B, h3, w3, LATENT).permute(0, 3, 1, 2)

    final_pcc = _pcc(ref_pred, pred)
    d = (ref_pred.float() - pred.float()).abs().max().item()
    rel = d / (ref_pred.float().abs().max().item() + 1e-9)
    return final_pcc, d, rel


def test_denoise_step_pcc(device):
    final_pcc, d, rel = _run(device)
    print(
        f"\nDENOISE STEP (patch_embed + {NUM_LAYERS} layers + final_layer): "
        f"velocity PCC={final_pcc:.6f} (>= {PCC_THR})  max|diff|={d:.4f}  rel={rel:.4%}"
    )
    assert final_pcc >= PCC_THR, f"denoise-step velocity PCC {final_pcc:.6f} below {PCC_THR}"


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        final_pcc, d, rel = _run(dev)
    finally:
        ttnn.close_device(dev)
    ok = final_pcc >= PCC_THR
    print("\n" + "=" * 64)
    print(f"Denoise step — patch_embed + {NUM_LAYERS} layers + final_layer, seq={S}")
    print(
        f"  [{'PASS' if ok else 'FAIL'}] velocity PCC={final_pcc:.6f} (>= {PCC_THR})  max|diff|={d:.4f}  rel={rel:.4%}"
    )
    print("=" * 64)
    sys.exit(0 if ok else 1)
