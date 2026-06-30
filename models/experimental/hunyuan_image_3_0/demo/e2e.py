# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Standalone end-to-end HunyuanImage-3.0 pipeline on RANDOM inputs.
#
# Unlike demo.py (real prompt -> recaption -> tokenize -> gen) this drives the
# device gen path directly from random noise + random text embeddings, so it runs
# without a prompt/tokenizer and is the quickest way to exercise the WHOLE chain:
#
#     noise latent --patch_embed--> tokens --[text|img|text]--> resident bf8
#         backbone (NO ln_f) --final_layer--> velocity  x STEPS (Euler)  => latent
#     latent --/scaling--> TTNN VAE decode (2x2 H/W-spatial) --(x/2+0.5)--> RGB -> PNG
#
# The heavy compute (backbone, VAE) stays on the 2x2 mesh. The resident bf8
# backbone fills DRAM, so it is FREED before the VAE mesh opens (mirrors demo.py).
#
# Config (env) — seq_len is driven by the image grid, like the real model:
#     HY_GRID        image-token grid side G; seq_len = TEXT_PRE + G*G + TEXT_POST
#                    default 64 (=> 64x64 latent -> 1024x1024 image). Raise toward
#                    the config's max_position_embeddings (asserted) for longer seq.
#     HY_TEXT_PRE / HY_TEXT_POST   text tokens around the image span (default 32).
#     HY_NUM_LAYERS  backbone layers (default 2).
#     HY_STEPS       diffusion steps (default 4).
#     HY_SEED        RNG seed for the random inputs (default 0).
#     HY_OUT         output PNG path (default ./hy_e2e.png).
#
# PCC validation against an fp32 host reference lives in the separate test
# tests/pcc/test_e2e_pipeline.py, which imports run_denoise / run_vae_decode from
# this file so the device pipeline is defined exactly once.
#
# Run (full 1024x1024, heavy):
#   python_env/bin/python models/experimental/hunyuan_image_3_0/demo/e2e.py
# Fast (small grid):
#   HY_GRID=8 python_env/bin/python models/experimental/hunyuan_image_3_0/demo/e2e.py

import os, sys, json, glob, time
import torch
from safetensors import safe_open

ROOT = "/home/iguser/christy/tt-metal"
HUNYUAN = "/home/iguser/christy/HunyuanImage-3.0"
WEIGHTS = "/home/iguser/christy/HunyuanImage-3"
for p in (ROOT, HUNYUAN):
    if p not in sys.path:
        sys.path.insert(0, p)
# ref/tt VAE weight loaders resolve from HUNYUAN_MODEL_DIR at import time.
os.environ.setdefault("HUNYUAN_MODEL_DIR", WEIGHTS)

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive
from models.experimental.hunyuan_image_3_0.ref.vae.decoder import Z_CHANNELS
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel
from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown, HunyuanTtUNetUp
from models.experimental.hunyuan_image_3_0.tt.image_gen.timestep_embedder import HunyuanTtTimestepEmbedder
from models.experimental.hunyuan_image_3_0.tt.pipeline import HunyuanTtDenoiseStep, denoise_loop, decode_latent
from models.experimental.hunyuan_image_3_0.tt.scheduler import HunyuanTtScheduler

B = 1
GRID = int(os.environ.get("HY_GRID", "64"))
TEXT_PRE = int(os.environ.get("HY_TEXT_PRE", "32"))
TEXT_POST = int(os.environ.get("HY_TEXT_POST", "32"))
N_IMG = GRID * GRID
S = TEXT_PRE + N_IMG + TEXT_POST
IMG_START = TEXT_PRE
IMG_SLICE = slice(IMG_START, IMG_START + N_IMG)

NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "2"))
STEPS = int(os.environ.get("HY_STEPS", "4"))
SEED = int(os.environ.get("HY_SEED", "0"))
SCALING = 0.562679178327931  # config.json vae.scaling_factor
OUT_PNG = os.environ.get("HY_OUT", "hy_e2e.png")

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
        KV=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        HD=cfg.get("attention_head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
        E=first(cfg["num_experts"]),
        K=first(cfg["moe_topk"]),
        INTER=first(cfg["moe_intermediate_size"]),
        SHARED=first(cfg["num_shared_expert"]),
        NORM=cfg.get("norm_topk_prob", True),
        MIXED=cfg.get("use_mixed_mlp_moe", True),
        QKN=cfg.get("use_qk_norm", True),
        EPS=cfg.get("rms_norm_eps", 1e-5),
        MAX_SEQ=int(cfg["max_position_embeddings"]),
    )


def _pe_dims(down_sd):
    hid, latent = down_sd["model.0.weight"].shape[:2]
    hsz = down_sd["model.1.in_layers.2.weight"].shape[0]
    return int(latent), int(hid), int(hsz)


def run_denoise(c, down_sd, up_sd, init_latent, text_embeds):
    """Resident bf8 backbone denoise loop on a fresh 2x2 mesh -> latent (torch NCHW).

    The mesh is opened and CLOSED here so the resident backbone is freed before the
    VAE mesh opens (they cannot share DRAM)."""
    LATENT, HID, HSZ = _pe_dims(down_sd)
    H = c["H"]

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 2), l1_small_size=32768)
    try:
        mesh_device.enable_program_cache()
        ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

        def rep(t):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        patch_embed = HunyuanTtUNetDown(
            mesh_device,
            {f"patch_embed.{k}": v for k, v in down_sd.items()},
            in_channels=LATENT,
            hidden_channels=HID,
            out_channels=HSZ,
        )
        final_layer = HunyuanTtUNetUp(
            mesh_device,
            {f"final_layer.{k}": v for k, v in up_sd.items()},
            in_channels=HSZ,
            hidden_channels=HID,
            out_channels=LATENT,
        )
        bf16_layers = {0, 1, 2, 3, NUM_LAYERS - 4, NUM_LAYERS - 3, NUM_LAYERS - 2, NUM_LAYERS - 1}
        layer_loader = lambda i: {f"model.layers.{i}.{k}": v for k, v in _load_prefix(f"model.layers.{i}").items()}
        backbone = HunyuanTtModel(
            mesh_device,
            num_layers=NUM_LAYERS,
            hidden_size=H,
            num_heads=c["HEADS"],
            num_kv_heads=c["KV"],
            head_dim=c["HD"],
            num_experts=c["E"],
            moe_topk=c["K"],
            use_qk_norm=c["QKN"],
            use_mixed_mlp_moe=c["MIXED"],
            norm_topk_prob=c["NORM"],
            rms_norm_eps=c["EPS"],
            stream_experts=False,
            layer_loader=layer_loader,
            apply_final_norm=False,
            weight_dtype=ttnn.bfloat8_b,
            ccl_manager=ccl,
            expert_mesh_axis=1,
            tp_axis=1,
            tp_factor=2,
            sp_axis=0,
            sp_factor=2,
            bf16_layers=bf16_layers,
        )
        te1 = HunyuanTtTimestepEmbedder(
            mesh_device, H, {f"time_embed.{k}": v for k, v in _load_prefix("time_embed").items()}, "time_embed"
        )
        te2 = HunyuanTtTimestepEmbedder(
            mesh_device, H, {f"time_embed_2.{k}": v for k, v in _load_prefix("time_embed_2").items()}, "time_embed_2"
        )
        step = HunyuanTtDenoiseStep(
            mesh_device,
            patch_embed=patch_embed,
            backbone=backbone,
            final_layer=final_layer,
            img_slice=IMG_SLICE,
            grid_hw=(GRID, GRID),
            seq_len=S,
        )

        post = text_embeds[:, IMG_START + N_IMG :, :]
        cond = dict(
            text_pre=rep(text_embeds[:, :IMG_START, :]),
            text_post=rep(post) if post.shape[1] > 0 else None,
            image_infos=[[(IMG_SLICE, (GRID, GRID))]],
            attention_mask=rep(
                to_additive(build_attention_mask(S, image_slices=[IMG_SLICE], bsz=B), dtype=torch.float32).reshape(
                    B, 1, S, S
                )
            ),
            batch=B,
        )

        sched = HunyuanTtScheduler(mesh_device)
        sched.set_timesteps(STEPS)
        print(f"[e2e] denoising {STEPS} steps on resident bf8 backbone (seq_len={S}) ...", flush=True)
        latent = denoise_loop(
            step, sched, init_latent.clone(), time_embed=te1, time_embed_2=te2, cond=cond, mesh_device=mesh_device
        )
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    return latent


def run_vae_decode(latent_bchw):
    """TTNN VAE decode (H/W-spatial-parallel) on a fresh 2x2 mesh -> RGB [B,3,H,W]."""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    vae_mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))
    try:
        vae_mesh.enable_program_cache()
        vae_ccl = CCLManager(vae_mesh, num_links=1, topology=ttnn.Topology.Linear)
        print(f"[e2e] VAE decode (2x2 H/W-spatial) -> {GRID * 16}x{GRID * 16} ...", flush=True)
        img = decode_latent(
            vae_mesh,
            latent_bchw,
            scaling_factor=SCALING,
            grid_hw=(GRID, GRID),
            ccl_manager=vae_ccl,
            h_mesh_axis=0,
            w_mesh_axis=1,
        )
    finally:
        ttnn.close_mesh_device(vae_mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    return img


def run_pipeline(seed=SEED):
    """Full random-input pipeline -> (rgb [B,3,H,W] in [0,1], latent [B,C,GRID,GRID])."""
    c = _cfg()
    assert S <= c["MAX_SEQ"], f"seq_len {S} (grid {GRID}) exceeds max_position_embeddings {c['MAX_SEQ']}"
    down_sd = _load_prefix("patch_embed")
    up_sd = _load_prefix("final_layer")
    LATENT, _, _ = _pe_dims(down_sd)
    assert LATENT == Z_CHANNELS, f"diffusion latent ch {LATENT} != VAE z-channels {Z_CHANNELS}"

    print(
        f"[e2e] grid={GRID}x{GRID}  seq_len={S} (<= max {c['MAX_SEQ']})  layers={NUM_LAYERS}  "
        f"steps={STEPS}  image={GRID * 16}x{GRID * 16}  seed={seed}",
        flush=True,
    )
    torch.manual_seed(seed)
    init_latent = torch.randn(B, LATENT, GRID, GRID)
    text_embeds = torch.randn(B, S, c["H"]) * 0.02

    t0 = time.time()
    latent = run_denoise(c, down_sd, up_sd, init_latent, text_embeds)
    rgb = run_vae_decode(latent)
    print(f"[e2e] pipeline done in {time.time() - t0:.0f}s  rgb={tuple(rgb.shape)}", flush=True)
    return rgb, latent


def main():
    rgb, _ = run_pipeline()
    from PIL import Image

    arr = (rgb[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    Image.fromarray(arr).save(OUT_PNG)
    print(f"[e2e] saved image -> {OUT_PNG}", flush=True)


if __name__ == "__main__":
    main()
