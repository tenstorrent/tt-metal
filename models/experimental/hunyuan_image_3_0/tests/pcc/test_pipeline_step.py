# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PCC test for tt/pipeline.py HunyuanTtDenoiseStep — the ON-DEVICE single
# denoise step. Same reference and weights as tests/pcc/test_denoise_step.py,
# but instead of round-tripping each module hand-off through host, the step
# scatters image tokens into the sequence with a device-side concat and slices
# the image span back out on device. This gates the composition glue
# (device scatter + slice) at the same PCC bar as the host-routed version.
#
# Run:
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_pipeline_step.py -v -s --timeout=1800
#   HY_NUM_LAYERS=32 python_env/bin/python \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_pipeline_step.py

import os, sys, json, glob, gc
import torch
from safetensors import safe_open

ROOT = "/home/iguser/Christy/tt-metal"
HUNYUAN = "/home/iguser/Christy/tt-metal/HunyuanImage-3.0"
WEIGHTS = "/home/iguser/Christy/HunyuanImage-3"
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
from models.experimental.hunyuan_image_3_0.tt.pipeline import HunyuanTtDenoiseStep
from hunyuan_image_3.modeling_hunyuan_image_3 import build_batch_2d_rope

B = 1
TEXT_PRE, TEXT_POST = 32, 32
GRID = 8
N_IMG = GRID * GRID
S = TEXT_PRE + N_IMG + TEXT_POST  # 128
IMG_START = TEXT_PRE
IMG_SLICE = slice(IMG_START, IMG_START + N_IMG)

NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "4"))
PCC_THR = 0.99 if NUM_LAYERS <= 8 else 0.85

# Backbone weight dtype — set HY_WEIGHT_DTYPE=bf8 to de-risk bf8 residency
# (MEMORY_FIT_PLAN.md step 1). bf8 weights are the only way the 80B model fits
# DRAM, so we need to know its accuracy cost.
_WEIGHT_DTYPE = ttnn.bfloat8_b if os.environ.get("HY_WEIGHT_DTYPE", "bf16") == "bf8" else ttnn.bfloat16
if _WEIGHT_DTYPE == ttnn.bfloat8_b:
    PCC_THR = 0.90  # observe the bf8 cost; gate loosely

# Layer-wise mixed precision: HY_BF16_LAYERS="0,1,2,3,28,29,30,31" keeps those
# layers' experts in bf16 (the rest bf8), trading DRAM headroom for accuracy.
_BF16_LAYERS = {int(s) for s in os.environ.get("HY_BF16_LAYERS", "").split(",") if s.strip() != ""}


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


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
    sd = _load_prefix(prefix)
    te = RefTimeEmbed(H).eval()
    te.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    with torch.no_grad():
        return te(timesteps)


def _reference(c, latent, t_emb1, t_emb2, text_embeds, down_sd, up_sd):
    LATENT, HID, HSZ = _pe_dims(down_sd)
    ref_down = RefDown(1, LATENT, HSZ, HID, HSZ).eval()
    ref_up = RefUp(1, HSZ, HSZ, HID, LATENT, out_norm=True).eval()
    ref_down.load_state_dict({k: v.float() for k, v in down_sd.items()}, strict=True)
    ref_up.load_state_dict({k: v.float() for k, v in up_sd.items()}, strict=True)

    cos, sin = build_batch_2d_rope(S, c["HEAD_DIM"], image_infos=[[(IMG_SLICE, (GRID, GRID))]])
    mask_add = to_additive(build_attention_mask(S, image_slices=[IMG_SLICE], bsz=B), dtype=torch.float32)

    with torch.no_grad():
        img_tokens, th, tw = ref_down(latent, t_emb1)
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
            h = layer(h, attention_mask=mask_add, custom_pos_emb=(cos, sin))
            del layer
            gc.collect()
        pred = ref_up(h[:, IMG_SLICE, :], t_emb2, th, tw)
    return pred


def _run(device):
    c = _cfg()
    down_sd = _load_prefix("patch_embed")
    up_sd = _load_prefix("final_layer")
    H = c["H"]
    LATENT, HID, HSZ = _pe_dims(down_sd)

    torch.manual_seed(0)
    latent = torch.randn(B, LATENT, GRID, GRID)
    text_embeds = torch.randn(B, S, H) * 0.02
    timesteps = torch.rand(B)
    t_emb1 = _time_embed("time_embed", H, timesteps)
    t_emb2 = _time_embed("time_embed_2", H, timesteps)

    ref_pred = _reference(c, latent, t_emb1, t_emb2, text_embeds, down_sd, up_sd)

    # ---- TT on-device step ----
    t1_tt = ttnn.from_torch(t_emb1.reshape(1, 1, B, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    t2_tt = ttnn.from_torch(t_emb2.reshape(1, 1, B, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    patch_embed = HunyuanTtUNetDown(
        device,
        {f"patch_embed.{k}": v for k, v in down_sd.items()},
        in_channels=LATENT,
        hidden_channels=HID,
        out_channels=HSZ,
    )
    final_layer = HunyuanTtUNetUp(
        device,
        {f"final_layer.{k}": v for k, v in up_sd.items()},
        in_channels=HSZ,
        hidden_channels=HID,
        out_channels=LATENT,
    )
    layer_loader = lambda i: {f"model.layers.{i}.{k}": v for k, v in _load_prefix(f"model.layers.{i}").items()}
    backbone = HunyuanTtModel(
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
        weight_dtype=_WEIGHT_DTYPE,
        bf16_layers=_BF16_LAYERS,
    )

    step = HunyuanTtDenoiseStep(
        device,
        patch_embed=patch_embed,
        backbone=backbone,
        final_layer=final_layer,
        img_slice=IMG_SLICE,
        grid_hw=(GRID, GRID),
        seq_len=S,
    )

    # Text-side embeddings split around the image span (device, TILE).
    pre = text_embeds[:, :IMG_START, :]
    post = text_embeds[:, IMG_START + N_IMG :, :]
    pre_tt = ttnn.from_torch(pre, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    post_tt = ttnn.from_torch(post, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    image_infos = [[(IMG_SLICE, (GRID, GRID))]]
    mask_add = to_additive(build_attention_mask(S, image_slices=[IMG_SLICE], bsz=B), dtype=torch.float32)
    mask_tt = ttnn.from_torch(mask_add.reshape(B, 1, S, S), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    pred_tt = step(
        latent,
        text_pre=pre_tt,
        text_post=post_tt,
        t_emb1=t1_tt,
        t_emb2=t2_tt,
        image_infos=image_infos,
        attention_mask=mask_tt,
        batch=B,
    )
    pred = ttnn.to_torch(pred_tt).reshape(B, GRID, GRID, LATENT).permute(0, 3, 1, 2)

    final_pcc = _pcc(ref_pred, pred)
    d = (ref_pred.float() - pred.float()).abs().max().item()
    return final_pcc, d


def test_pipeline_step(device):
    pcc, max_abs = _run(device)
    print(f"\non-device denoise step: PCC={pcc:.6f}  max|Δ|={max_abs:.4e}  (layers={NUM_LAYERS}, thr={PCC_THR})")
    assert pcc >= PCC_THR, f"PCC {pcc:.6f} < {PCC_THR}"


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0)
    try:
        pcc, max_abs = _run(dev)
        print(f"on-device denoise step: PCC={pcc:.6f}  max|Δ|={max_abs:.4e}")
    finally:
        ttnn.close_device(dev)
