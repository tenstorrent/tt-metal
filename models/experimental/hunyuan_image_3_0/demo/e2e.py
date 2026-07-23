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
#     HY_MODEL       model variant: base (default) | instruct | distil. Selects the
#                    checkpoint dir and whether the per-step gen-timestep / guidance /
#                    timestep_r continuous tokens are scattered (read from config.json).
#                    Per-dir overrides: HUNYUAN_MODEL_DIR / HUNYUAN_INSTRUCT_MODEL_DIR /
#                    HUNYUAN_INSTRUCT_DISTIL_MODEL_DIR (mirror ref/weights.py).
#     HY_GUIDANCE    distil guidance scale (instruct/distil only; default 1.0).
#     HY_GRID        image-token grid side G; seq_len = TEXT_PRE + G*G + TEXT_POST
#                    default 64 (=> 64x64 latent -> 1024x1024 image). Raise toward
#                    the config's max_position_embeddings (asserted) for longer seq.
#     HY_TEXT_PRE / HY_TEXT_POST   text tokens around the image span (default 32).
#     HY_NUM_LAYERS  backbone layers (default 2).
#     HY_STEPS       diffusion steps (default 4).
#     HY_WEIGHT_DTYPE  backbone weights: bf8 (default, production mirror) | bf16
#                    (accuracy / e2e PCC vs fp32 host — set for 32L gates).
#     HY_SEED        RNG seed for the random inputs (default 0).
#     HY_OUT         output PNG path (default ./hy_e2e.png).
#
# PCC validation against an fp32 host reference lives in the separate test
# tests/pcc/test_pipeline.py, which imports run_denoise / run_vae_decode from
# this file so the device pipeline is defined exactly once.
#
# Run (base, full 1024x1024, heavy):
#   python_env/bin/python models/experimental/hunyuan_image_3_0/demo/e2e.py
# Fast (small grid):
#   HY_GRID=8 python_env/bin/python models/experimental/hunyuan_image_3_0/demo/e2e.py
# Instruct-Distil (8-step meanflow + cfg_distill, 32 layers):
#   HY_MODEL=distil HY_NUM_LAYERS=32 HY_STEPS=8 HY_GUIDANCE=2.5 \
#     python_env/bin/python models/experimental/hunyuan_image_3_0/demo/e2e.py

import os, sys, json, glob, time
from pathlib import Path
import torch
from safetensors import safe_open

ROOT = "/home/iguser/christy/tt-metal"
HUNYUAN = "/home/iguser/christy/HunyuanImage-3.0"

# Model variant for the random pipeline, selected by HY_MODEL (default "base"):
#   base     -> HunyuanImage-3            (T2I pretrain; cfg_distilled=meanflow=False)
#   instruct -> HunyuanImage-3-Instruct   (may set use_meanflow)
#   distil   -> HunyuanImage-3-Instruct-Distil (cfg_distilled=True, 8-step meanflow)
# The chain is identical for all three; only the per-step gen-timestep / guidance /
# timestep_r continuous-token embeds differ, and those are read from config.json and
# wired into denoise_loop below. The per-dir env overrides mirror ref/weights.py.
HY_MODEL = os.environ.get("HY_MODEL", "base").lower()
_MODEL_DIRS = {
    "base": os.environ.get("HUNYUAN_MODEL_DIR", "/home/iguser/christy/HunyuanImage-3"),
    "instruct": os.environ.get("HUNYUAN_INSTRUCT_MODEL_DIR", "/home/iguser/christy/HunyuanImage-3-Instruct"),
    "distil": os.environ.get(
        "HUNYUAN_INSTRUCT_DISTIL_MODEL_DIR", "/home/iguser/christy/HunyuanImage-3-Instruct-Distil"
    ),
}
if HY_MODEL not in _MODEL_DIRS:
    raise SystemExit(f"HY_MODEL={HY_MODEL!r} not in {sorted(_MODEL_DIRS)}")
WEIGHTS = _MODEL_DIRS[HY_MODEL]
for p in (ROOT, HUNYUAN):
    if p not in sys.path:
        sys.path.insert(0, p)
# ref/tt VAE weight loaders resolve from HUNYUAN_MODEL_DIR at import time — point them
# at the selected variant (the PCC gate pre-sets this to base before importing).
os.environ.setdefault("HUNYUAN_MODEL_DIR", WEIGHTS)

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive
from models.experimental.hunyuan_image_3_0.ref.vae.decoder import Z_CHANNELS
from models.experimental.hunyuan_image_3_0.ref.model_config import (
    PRODUCTION_LATENT_GRID,
    PRODUCTION_TEXT_POST,
    PRODUCTION_TEXT_PRE,
    VAE_SCALING_FACTOR,
    load_config,
    transformer_cfg,
)
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel, default_bf16_layers
from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown, HunyuanTtUNetUp
from models.experimental.hunyuan_image_3_0.tt.image_gen.timestep_embedder import HunyuanTtTimestepEmbedder
from models.experimental.hunyuan_image_3_0.tt.pipeline import HunyuanTtDenoiseStep, denoise_loop, decode_latent
from models.experimental.hunyuan_image_3_0.tt.denoise_dual_cq import open_denoise_mesh
from models.experimental.hunyuan_image_3_0.tt.vae_dual_cq import open_vae_mesh
from models.experimental.hunyuan_image_3_0.tt.scheduler import HunyuanTtScheduler

# Host ref TimestepEmbedder for the gen-timestep / guidance / timestep_r continuous tokens
# scattered each denoise step (instruct/distil variants only).
from models.experimental.hunyuan_image_3_0.ref.image_gen.model_loaders import load_timestep_embedder

B = 1
GRID = int(os.environ.get("HY_GRID", str(PRODUCTION_LATENT_GRID)))
TEXT_PRE = int(os.environ.get("HY_TEXT_PRE", str(PRODUCTION_TEXT_PRE)))
TEXT_POST = int(os.environ.get("HY_TEXT_POST", str(PRODUCTION_TEXT_POST)))
N_IMG = GRID * GRID
S = TEXT_PRE + N_IMG + TEXT_POST
IMG_START = TEXT_PRE
IMG_SLICE = slice(IMG_START, IMG_START + N_IMG)

NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "2"))
STEPS = int(os.environ.get("HY_STEPS", "4"))
SEED = int(os.environ.get("HY_SEED", "0"))
# Distil guidance scale (instruct/distil only): the per-step guidance token is scattered
# with 1000 * GUIDANCE inside denoise_loop. Unused by the base T2I path.
GUIDANCE = float(os.environ.get("HY_GUIDANCE", "1.0"))
# Base T2I classifier-free guidance scale. The real model runs cfg_factor=2 (a cond +
# an uncond pass per step, combined as uncond + scale*(cond-uncond)); see
# hunyuan_image_3_pipeline.py and the base generation_config (diff_guidance_scale=5.0).
# >1 enables CFG on the base path; set 1.0 to run a single unguided pass. Unused by
# the distil path (which uses the guidance TOKEN at cfg_factor=1 instead).
BASE_GUIDANCE = float(os.environ.get("HY_BASE_GUIDANCE", "5.0"))
SCALING = VAE_SCALING_FACTOR  # config.json vae.scaling_factor
OUT_PNG = os.environ.get("HY_OUT", "hy_e2e.png")


def _weight_dtype():
    """Resident backbone dtype. Default bf8 mirrors demo.py; HY_WEIGHT_DTYPE=bf16 for PCC."""
    return ttnn.bfloat8_b if os.environ.get("HY_WEIGHT_DTYPE", "bf8") == "bf8" else ttnn.bfloat16


_WMAP = json.load(open(glob.glob(f"{WEIGHTS}/*.index.json")[0]))["weight_map"]
_OPEN = {}


def _load(key):
    shard = _WMAP[key]
    f = _OPEN.get(shard) or _OPEN.setdefault(shard, safe_open(f"{WEIGHTS}/{shard}", framework="pt"))
    return f.get_tensor(key)


def _load_prefix(prefix):
    return {k[len(prefix) + 1 :]: _load(k) for k in _WMAP if k.startswith(prefix + ".")}


def _cfg():
    return transformer_cfg(load_config(WEIGHTS))


def _pe_dims(down_sd):
    hid, latent = down_sd["model.0.weight"].shape[:2]
    hsz = down_sd["model.1.in_layers.2.weight"].shape[0]
    return int(latent), int(hid), int(hsz)


def _model_flags():
    """(cfg_distilled, use_meanflow) from the selected variant's config.json.

    Base T2I returns (False, False) — the existing text_pre/text_post denoise path.
    Instruct/Distil enable the per-step guidance / timestep_r continuous-token scatter."""
    cfg = load_config(WEIGHTS)
    return bool(cfg.get("cfg_distilled", False)), bool(cfg.get("use_meanflow", False))


def gen_special_token_indices(img_start, cfg_distilled, use_meanflow):
    """Per-step continuous-token scatter indices, in the tokenizer's CANONICAL order.

    ``encode_sequence`` (tokenization_hunyuan_image_3.py / ref tokenizer
    ``_add_image_meta_info_token``) appends, immediately before the gen-image span:
    gen_timestep, then guidance (cfg_distilled), then timestep_r (use_meanflow).
    So the gen-timestep token is FURTHEST from the image and timestep_r / guidance
    sits adjacent to it. Verified empirically against ``prepare_gen_image_inputs``:
    e.g. distil => gen_timestep at IMG_START-2, guidance at IMG_START-1.

    Returns (gen_ts_idx, guidance_idx, timestep_r_idx, n_special); the *_idx fields
    are None when that token is absent."""
    n_special = 1 + int(cfg_distilled) + int(use_meanflow)
    nxt = img_start - n_special
    gen_ts_idx = nxt
    nxt += 1
    guidance_idx = nxt if cfg_distilled else None
    nxt += int(cfg_distilled)
    timestep_r_idx = nxt if use_meanflow else None
    return gen_ts_idx, guidance_idx, timestep_r_idx, n_special


def run_denoise(c, down_sd, up_sd, init_latent, text_embeds, text_embeds_uncond=None, cfg_guidance=1.0):
    """Resident bf8 backbone denoise loop on a fresh 2x2 mesh -> latent (torch NCHW).

    The mesh is opened and CLOSED here so the resident backbone is freed before the
    VAE mesh opens (they cannot share DRAM).

    Base T2I classifier-free guidance: pass ``text_embeds_uncond`` (the unconditional /
    empty-prompt embeddings) and ``cfg_guidance`` > 1 to run a second (uncond) pass per
    step, combined on device as ``uncond + cfg_guidance*(cond-uncond)`` — mirrors the
    real cfg_factor=2 path (hunyuan_image_3_pipeline.py). Ignored on the distil/meanflow
    path (which uses the guidance TOKEN at cfg_factor=1 instead)."""
    LATENT, HID, HSZ = _pe_dims(down_sd)
    H = c["H"]

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = open_denoise_mesh(ttnn.MeshShape(2, 2), l1_small_size=32768)
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
        weight_dtype = _weight_dtype()
        dtype_tag = "bf8" if weight_dtype == ttnn.bfloat8_b else "bf16"
        # Full bf16 resident 32L MoE OOMs on 2x2; stream experts when not on the bf8 path.
        stream_experts = weight_dtype != ttnn.bfloat8_b
        bf16_layers = (
            {int(s) for s in os.environ["HY_BF16_LAYERS"].split(",") if s.strip() != ""}
            if os.environ.get("HY_BF16_LAYERS")
            else (set() if stream_experts else default_bf16_layers(NUM_LAYERS))
        )
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
            stream_experts=stream_experts,
            layer_loader=layer_loader,
            apply_final_norm=False,
            weight_dtype=weight_dtype,
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

        image_infos = [[(IMG_SLICE, (GRID, GRID))]]
        mask_tt = rep(
            to_additive(build_attention_mask(S, image_slices=[IMG_SLICE], bsz=B), dtype=torch.float32).reshape(
                B, 1, S, S
            )
        )

        cfg_distilled, use_meanflow = _model_flags()
        sched = HunyuanTtScheduler(mesh_device)
        sched.set_timesteps(STEPS)

        if not (cfg_distilled or use_meanflow):
            # Base T2I: [text_pre | gen | text_post], no per-step continuous tokens.
            def _text_cond(te):
                post = te[:, IMG_START + N_IMG :, :]
                return dict(
                    text_pre=rep(te[:, :IMG_START, :]),
                    text_post=rep(post) if post.shape[1] > 0 else None,
                    image_infos=image_infos,
                    attention_mask=mask_tt,
                    batch=B,
                )

            cond = _text_cond(text_embeds)
            do_cfg = text_embeds_uncond is not None and cfg_guidance != 1.0
            uncond = _text_cond(text_embeds_uncond) if do_cfg else None
            print(
                f"[e2e] denoising {STEPS} steps on resident {dtype_tag} backbone (seq_len={S}"
                f"{f', CFG guidance={cfg_guidance}' if do_cfg else ''}) ...",
                flush=True,
            )
            latent = denoise_loop(
                step,
                sched,
                init_latent.clone(),
                time_embed=te1,
                time_embed_2=te2,
                cond=cond,
                uncond=uncond,
                guidance_scale=cfg_guidance if do_cfg else 1.0,
                mesh_device=mesh_device,
            )
        else:
            # Instruct/Distil: drive the gen-timestep (+ distil guidance / meanflow
            # timestep_r) continuous tokens. They sit in TEXT_PRE just before the gen
            # span; denoise_loop re-scatters them each step into base_embeds_host and
            # the patch_embed image tokens overwrite the gen span via _scatter_from_base.
            gen_ts_idx, guidance_idx, timestep_r_idx, n_special = gen_special_token_indices(
                IMG_START, cfg_distilled, use_meanflow
            )
            assert IMG_START >= n_special, (
                f"HY_TEXT_PRE={IMG_START} too small for {n_special} step tokens "
                f"(gen_timestep{' +guidance' if cfg_distilled else ''}{' +timestep_r' if use_meanflow else ''})"
            )
            li = lambda i: torch.tensor([[i]] * B, dtype=torch.long)

            timestep_emb = load_timestep_embedder("timestep_emb", Path(WEIGHTS), hidden_size=H)
            guidance_emb = (
                load_timestep_embedder("guidance_emb", Path(WEIGHTS), hidden_size=H) if cfg_distilled else None
            )
            timestep_r_emb = (
                load_timestep_embedder("timestep_r_emb", Path(WEIGHTS), hidden_size=H) if use_meanflow else None
            )
            cond = dict(
                base_embeds_host=text_embeds.to(torch.float32),
                gen_timestep_scatter_index=li(gen_ts_idx),
                guidance_scatter_index=li(guidance_idx) if cfg_distilled else None,
                gen_timestep_r_scatter_index=li(timestep_r_idx) if use_meanflow else None,
                image_infos=image_infos,
                attention_mask=mask_tt,
                batch=B,
            )
            mode = "distil" if cfg_distilled else "meanflow"
            print(
                f"[e2e] denoising {STEPS} steps on resident {dtype_tag} backbone "
                f"({mode}, guidance={GUIDANCE}, seq_len={S}) ...",
                flush=True,
            )
            latent = denoise_loop(
                step,
                sched,
                init_latent.clone(),
                time_embed=te1,
                time_embed_2=te2,
                cond=cond,
                guidance_scale=GUIDANCE,
                timestep_emb=timestep_emb,
                guidance_emb=guidance_emb,
                timestep_r_emb=timestep_r_emb,
                cfg_distilled=cfg_distilled,
                use_meanflow=use_meanflow,
                mesh_device=mesh_device,
            )
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    return latent


def run_vae_decode(latent_bchw):
    """TTNN VAE decode (H/W-spatial-parallel) on a fresh 2x2 mesh -> RGB [B,3,H,W]."""
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    vae_mesh = open_vae_mesh(ttnn.MeshShape(2, 2))
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

    cfg_distilled, use_meanflow = _model_flags()
    base_cfg = (not (cfg_distilled or use_meanflow)) and BASE_GUIDANCE != 1.0
    print(
        f"[e2e] model={HY_MODEL} ({WEIGHTS})  cfg_distilled={cfg_distilled} use_meanflow={use_meanflow}"
        f"{f'  base_CFG guidance={BASE_GUIDANCE}' if base_cfg else ''}\n"
        f"[e2e] grid={GRID}x{GRID}  seq_len={S} (<= max {c['MAX_SEQ']})  layers={NUM_LAYERS}  "
        f"steps={STEPS}  image={GRID * 16}x{GRID * 16}  seed={seed}",
        flush=True,
    )
    torch.manual_seed(seed)
    init_latent = torch.randn(B, LATENT, GRID, GRID)
    text_embeds = torch.randn(B, S, c["H"]) * 0.02
    # Unconditional (empty-prompt) embeddings for base CFG — a distinct deterministic draw.
    text_embeds_uncond = torch.randn(B, S, c["H"]) * 0.02 if base_cfg else None

    t0 = time.time()
    latent = run_denoise(
        c,
        down_sd,
        up_sd,
        init_latent,
        text_embeds,
        text_embeds_uncond=text_embeds_uncond,
        cfg_guidance=BASE_GUIDANCE if base_cfg else 1.0,
    )
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
