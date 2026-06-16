# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PCC test for tt/pipeline.py denoise_loop — the MULTI-STEP diffusion loop.
#
# The single denoise step is already gated end-to-end (test_pipeline_step.py /
# test_denoise_step.py). This test validates the LOOP ORCHESTRATION on top of
# that trusted step: per-step timestep-embedding recompute, scheduler timestep
# indexing + Euler update (dt = sigma_next - sigma), and the latent hand-off
# across iterations. The reference is a host loop that chains the SAME reference
# single step over the IDENTICAL scheduler schedule, so any divergence isolates
# to the loop glue rather than the per-step numerics.
#
# Kept small (few layers, few steps) so the per-step bf16 drift stays negligible
# and the loop gate is meaningful.
#
# Run:
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_denoise_loop.py -v -s --timeout=1800

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
from models.experimental.hunyuan_image_3_0.tt.image_gen.timestep_embedder import HunyuanTtTimestepEmbedder
from models.experimental.hunyuan_image_3_0.tt.pipeline import HunyuanTtDenoiseStep, denoise_loop
from models.experimental.hunyuan_image_3_0.tt.scheduler import HunyuanTtScheduler
from hunyuan_image_3.modeling_hunyuan_image_3 import build_batch_2d_rope

B = 1
TEXT_PRE, TEXT_POST = 32, 32
GRID = 8
N_IMG = GRID * GRID
S = TEXT_PRE + N_IMG + TEXT_POST  # 128
IMG_START = TEXT_PRE
IMG_SLICE = slice(IMG_START, IMG_START + N_IMG)

NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "2"))
STEPS = int(os.environ.get("HY_STEPS", "3"))
PCC_THR = 0.99


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


# Cache reference layers so the host loop does not reload them every step.
_REF_LAYERS = {}


def _ref_layer(c, i):
    if i not in _REF_LAYERS:
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
        _REF_LAYERS[i] = layer
    return _REF_LAYERS[i]


def _ref_step(c, latent, t_emb1, t_emb2, text_embeds, ref_down, ref_up, cos, sin, mask_add):
    """fp32 reference single step -> velocity prediction [B, latent_ch, GRID, GRID]."""
    with torch.no_grad():
        img_tokens, th, tw = ref_down(latent, t_emb1)
        h = text_embeds.clone()
        h[:, IMG_SLICE, :] = img_tokens
        for i in range(NUM_LAYERS):
            h = _ref_layer(c, i)(h, attention_mask=mask_add, custom_pos_emb=(cos, sin))
        pred = ref_up(h[:, IMG_SLICE, :], t_emb2, th, tw)
    return pred


def _run(device):
    c = _cfg()
    down_sd = _load_prefix("patch_embed")
    up_sd = _load_prefix("final_layer")
    H = c["H"]
    LATENT, HID, HSZ = _pe_dims(down_sd)

    torch.manual_seed(0)
    init_latent = torch.randn(B, LATENT, GRID, GRID)
    text_embeds = torch.randn(B, S, H) * 0.02

    # Shared schedule (host reference and TT loop read the SAME sigmas).
    sched_ref = HunyuanTtScheduler(device)
    sched_ref.set_timesteps(STEPS)
    sigmas = sched_ref.sigmas
    timesteps = sched_ref.timesteps

    # ----- host reference loop -----
    ref_down = RefDown(1, LATENT, HSZ, HID, HSZ).eval()
    ref_up = RefUp(1, HSZ, HSZ, HID, LATENT, out_norm=True).eval()
    ref_down.load_state_dict({k: v.float() for k, v in down_sd.items()}, strict=True)
    ref_up.load_state_dict({k: v.float() for k, v in up_sd.items()}, strict=True)
    te1_ref = RefTimeEmbed(H).eval()
    te1_ref.load_state_dict({k: v.float() for k, v in _load_prefix("time_embed").items()}, strict=True)
    te2_ref = RefTimeEmbed(H).eval()
    te2_ref.load_state_dict({k: v.float() for k, v in _load_prefix("time_embed_2").items()}, strict=True)
    cos, sin = build_batch_2d_rope(S, c["HEAD_DIM"], image_infos=[[(IMG_SLICE, (GRID, GRID))]])
    mask_add = to_additive(build_attention_mask(S, image_slices=[IMG_SLICE], bsz=B), dtype=torch.float32)

    lat = init_latent.clone()
    for i, t in enumerate(timesteps):
        tvec = torch.tensor([float(t)] * B)
        with torch.no_grad():
            e1 = te1_ref(tvec)
            e2 = te2_ref(tvec)
        pred = _ref_step(c, lat, e1, e2, text_embeds, ref_down, ref_up, cos, sin, mask_add)
        dt = float(sigmas[i + 1] - sigmas[i])
        lat = lat + dt * pred
    ref_final = lat
    gc.collect()

    # ----- TT loop -----
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
    )
    time_embed = HunyuanTtTimestepEmbedder(
        device, H, {f"time_embed.{k}": v for k, v in _load_prefix("time_embed").items()}, "time_embed"
    )
    time_embed_2 = HunyuanTtTimestepEmbedder(
        device, H, {f"time_embed_2.{k}": v for k, v in _load_prefix("time_embed_2").items()}, "time_embed_2"
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

    pre = text_embeds[:, :IMG_START, :]
    post = text_embeds[:, IMG_START + N_IMG :, :]
    pre_tt = ttnn.from_torch(pre, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    post_tt = ttnn.from_torch(post, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    mask_tt = ttnn.from_torch(mask_add.reshape(B, 1, S, S), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cond = dict(
        text_pre=pre_tt, text_post=post_tt, image_infos=[[(IMG_SLICE, (GRID, GRID))]], attention_mask=mask_tt, batch=B
    )

    sched_tt = HunyuanTtScheduler(device)
    sched_tt.set_timesteps(STEPS)
    tt_final = denoise_loop(
        step, sched_tt, init_latent.clone(), time_embed=time_embed, time_embed_2=time_embed_2, cond=cond
    )

    pcc = _pcc(ref_final, tt_final)
    d = (ref_final.float() - tt_final.float()).abs().max().item()
    return pcc, d


def test_denoise_loop(device):
    pcc, max_abs = _run(device)
    print(f"\ndenoise loop ({STEPS} steps, {NUM_LAYERS} layers): PCC={pcc:.6f}  max|Δ|={max_abs:.4e}  (thr={PCC_THR})")
    assert pcc >= PCC_THR, f"PCC {pcc:.6f} < {PCC_THR}"


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        pcc, max_abs = _run(dev)
        print(f"denoise loop: PCC={pcc:.6f}  max|Δ|={max_abs:.4e}")
    finally:
        ttnn.close_device(dev)
