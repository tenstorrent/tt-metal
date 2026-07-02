# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Standalone end-to-end HunyuanImage-3.0 pipeline on RANDOM inputs.
#
# Unlike demo.py (real prompt -> recaption -> tokenize -> gen) this drives the device
# path directly from random inputs, so it runs without the TOKENIZER — but it still
# exercises the WHOLE model, stage by stage, each on its own mesh (freed before the
# next opens; the resident bf8 backbone fills DRAM, mirroring demo.py):
#
#   0) LM / recaption (default ON):  random input_ids --device wte--> LM backbone
#        (WITH ln_f) --LM head--> logits --AR sampling loop--> tokens (discarded).
#        This is the text half the gen path skips: wte embed + ln_f + LM head + sampling.
#   [opt] vision (I2I, HY_VISION=1): random cond image --SigLIP2 tower + aligner-->
#        projected cond tokens, scattered into a <img> span before the gen span.
#   1) text embeddings: real wte lookup of random token ids (host, exact).
#   2) noise latent --patch_embed--> tokens --[text|(cond img)|img|text]--> resident bf8
#        backbone (NO ln_f) --final_layer--> velocity  x STEPS (Euler)  => latent
#   3) latent --/scaling--> TTNN VAE decode (2x2 H/W-spatial) --(x/2+0.5)--> RGB -> PNG
#
# So the only thing skipped vs demo.py/demo_i2i.py is the tokenizer (prompt/chat
# template + detokenize); every device compute stage — LM, vision, gen, VAE — runs.
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
#     HY_SEED        RNG seed for the random inputs (default 0).
#     HY_OUT         output PNG path (default ./hy_e2e.png).
#     HY_RECAPTION   run the LM / recaption stage (default 1; 0 to skip).
#     HY_RECAPTION_PREFIX / HY_RECAPTION_TOKENS   AR prefix len / tokens to decode
#                    (defaults 32 / 4 — a quick smoke of the LM path).
#     HY_RECAPTION_SAMPLE   1 = host stochastic sampler (parity golden); default 0 =
#                    greedy argmax on device (ttnn.argmax; only the token id leaves device).
#     HY_VISION      run the optional I2I vision stage (default 0; needs instruct/distil).
#     HY_VIT_LAYERS  SigLIP2 encoder layers for the vision stage (default 27 = full).
#     HY_COND_H / HY_COND_W   random conditioning image size (default 128x128).
#
# PCC validation against an fp32 host reference lives in the separate test
# tests/pcc/test_e2e_pipeline.py, which imports run_denoise / run_vae_decode from
# this file so the device pipeline is defined exactly once. The LM and vision stages
# are PCC-gated by tests/vision/* and the recaption/generate tests; here they run as a
# full-chain exerciser (not PCC-checked in-line).
#
# Run (base T2I, full 1024x1024, LM recaption + gen + VAE):
#   python_env/bin/python models/experimental/hunyuan_image_3_0/demo/e2e.py
# Fast (small grid):
#   HY_GRID=8 python_env/bin/python models/experimental/hunyuan_image_3_0/demo/e2e.py
# Instruct-Distil (8-step meanflow + cfg_distill, 32 layers):
#   HY_MODEL=distil HY_NUM_LAYERS=32 HY_STEPS=8 HY_GUIDANCE=2.5 \
#     python_env/bin/python models/experimental/hunyuan_image_3_0/demo/e2e.py
# Instruct-Distil I2I (with the device vision tower feeding denoise):
#   HY_MODEL=distil HY_VISION=1 HY_NUM_LAYERS=32 HY_STEPS=8 HY_GUIDANCE=2.5 \
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
from models.experimental.hunyuan_image_3_0.tt.attention.mask import build_attention_mask_tt
from models.experimental.hunyuan_image_3_0.ref.vae.decoder import Z_CHANNELS
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel
from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown, HunyuanTtUNetUp
from models.experimental.hunyuan_image_3_0.tt.image_gen.timestep_embedder import HunyuanTtTimestepEmbedder
from models.experimental.hunyuan_image_3_0.tt.pipeline import HunyuanTtDenoiseStep, denoise_loop, decode_latent
from models.experimental.hunyuan_image_3_0.tt.scheduler import HunyuanTtScheduler

# LM (text) path: vocabulary projection + the shared AR sampling loop. These drive the
# recaption stage (device wte embed + ln_f backbone -> LM head -> sample), which the
# diffusion-only path skips. Tokenizer-free here: random input_ids in, tokens discarded.
from models.experimental.hunyuan_image_3_0.tt.lm_head import HunyuanTtLMHead
from models.experimental.hunyuan_image_3_0.tt.generate import (
    SamplingConfig,
    generate_text,
    make_backbone_logits_fn,
)

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
# Distil guidance scale (instruct/distil only): the per-step guidance token is scattered
# with 1000 * GUIDANCE inside denoise_loop. Unused by the base T2I path.
GUIDANCE = float(os.environ.get("HY_GUIDANCE", "1.0"))
# Base T2I classifier-free guidance scale. The real model runs cfg_factor=2 (a cond +
# an uncond pass per step, combined as uncond + scale*(cond-uncond)); see
# hunyuan_image_3_pipeline.py and the base generation_config (diff_guidance_scale=5.0).
# >1 enables CFG on the base path; set 1.0 to run a single unguided pass. Unused by
# the distil path (which uses the guidance TOKEN at cfg_factor=1 instead).
BASE_GUIDANCE = float(os.environ.get("HY_BASE_GUIDANCE", "5.0"))
SCALING = 0.562679178327931  # config.json vae.scaling_factor
OUT_PNG = os.environ.get("HY_OUT", "hy_e2e.png")

# LM / recaption stage (ON by default) — exercises the text half of the model the
# diffusion path skips: device wte embed + ln_f backbone -> LM head -> AR sampling.
# Tokenizer-free smoke: decode HY_RECAPTION_TOKENS from a random HY_RECAPTION_PREFIX
# prefix and discard the text (no tokenizer to build a prompt or detokenize).
RECAPTION = os.environ.get("HY_RECAPTION", "1") != "0"
RECAPTION_PREFIX = int(os.environ.get("HY_RECAPTION_PREFIX", "32"))
RECAPTION_TOKENS = int(os.environ.get("HY_RECAPTION_TOKENS", "4"))
# Greedy decode (default) picks the next token with ttnn.argmax ON DEVICE — only the
# token id crosses to host, not the [B, vocab] logits. HY_RECAPTION_SAMPLE=1 switches to
# the host stochastic sampler (rep-penalty/top-k/top-p/multinomial), the parity-gated
# golden re-exported by tt/generate.py.
RECAPTION_SAMPLE = os.environ.get("HY_RECAPTION_SAMPLE", "0") == "1"

# Optional vision (I2I) stage: encode a RANDOM conditioning image through the device
# SigLIP2 tower + aligner and scatter the projected features into a cond <img> span
# before the gen span (multi-span mask + 2D RoPE), so vision actually feeds denoise.
# Needs an Instruct/Instruct-Distil checkpoint (base T2I has no vision weights).
VISION = os.environ.get("HY_VISION", "0") == "1"
VIT_LAYERS = int(os.environ.get("HY_VIT_LAYERS", "27"))
COND_H = int(os.environ.get("HY_COND_H", "128"))
COND_W = int(os.environ.get("HY_COND_W", "128"))

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


def _model_flags():
    """(cfg_distilled, use_meanflow) from the selected variant's config.json.

    Base T2I returns (False, False) — the existing text_pre/text_post denoise path.
    Instruct/Distil enable the per-step guidance / timestep_r continuous-token scatter."""
    cfg = json.load(open(f"{WEIGHTS}/config.json"))
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


def run_denoise(
    c,
    down_sd,
    up_sd,
    init_latent,
    text_embeds,
    text_embeds_uncond=None,
    cfg_guidance=1.0,
    *,
    text_ids=None,
    text_ids_uncond=None,
    wte_weight=None,
):
    """Resident bf8 backbone denoise loop on a fresh 2x2 mesh -> latent (torch NCHW).

    The mesh is opened and CLOSED here so the resident backbone is freed before the
    VAE mesh opens (they cannot share DRAM).

    Base T2I classifier-free guidance: pass ``text_embeds_uncond`` (the unconditional /
    empty-prompt embeddings) and ``cfg_guidance`` > 1 to run a second (uncond) pass per
    step, combined on device as ``uncond + cfg_guidance*(cond-uncond)`` — mirrors the
    real cfg_factor=2 path (hunyuan_image_3_pipeline.py). Ignored on the distil/meanflow
    path (which uses the guidance TOKEN at cfg_factor=1 instead).

    Device wte (#4): when ``text_ids`` (+ ``wte_weight``) are given, the base-T2I text
    embeddings are produced on device via ``backbone.embed`` (ttnn.embedding) instead of
    a host wte lookup — the backbone is built with the wte table resident. The PCC gate
    calls without these (host ``text_embeds`` path, unchanged). Instruct/Distil keep the
    host ``base_embeds`` (their per-step continuous-token scatter runs on the resident
    device base via ``device_step_embeds``), so device wte applies to base T2I only."""
    LATENT, HID, HSZ = _pe_dims(down_sd)
    H = c["H"]
    cfg_distilled, use_meanflow = _model_flags()
    device_wte = text_ids is not None and wte_weight is not None and not (cfg_distilled or use_meanflow)

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
            embed_state_dict=({"model.wte.weight": wte_weight} if device_wte else None),
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

        image_infos = [[(IMG_SLICE, (GRID, GRID))]]
        # #1: attention mask built entirely on device (TTNN ops), like demo.py.
        mask_tt = build_attention_mask_tt(mesh_device, S, image_slices=[IMG_SLICE], bsz=B, dtype=ttnn.bfloat16)

        sched = HunyuanTtScheduler(mesh_device)
        sched.set_timesteps(STEPS)

        if not (cfg_distilled or use_meanflow):
            # Base T2I: [text_pre | gen | text_post], no per-step continuous tokens.
            if device_wte:
                # #4: embed token ids on device (backbone.embed = ttnn.embedding) and
                # slice the text spans on device — no host wte lookup.
                def _text_cond(ids):
                    ids_tt = ttnn.from_torch(
                        ids,
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=mesh_device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    seq = backbone.embed(ids_tt)  # [B, S, H] TILE (device)
                    ttnn.deallocate(ids_tt)
                    pre = ttnn.slice(seq, [0, 0, 0], [B, IMG_START, H])
                    post = ttnn.slice(seq, [0, IMG_START + N_IMG, 0], [B, S, H]) if S > IMG_START + N_IMG else None
                    ttnn.deallocate(seq)
                    return dict(text_pre=pre, text_post=post, image_infos=image_infos, attention_mask=mask_tt, batch=B)

                cond = _text_cond(text_ids)
                do_cfg = text_ids_uncond is not None and cfg_guidance != 1.0
                uncond = _text_cond(text_ids_uncond) if do_cfg else None
            else:

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
                f"[e2e] denoising {STEPS} steps on resident bf8 backbone (seq_len={S}"
                f"{f', CFG guidance={cfg_guidance}' if do_cfg else ''}"
                f"{', device-wte' if device_wte else ''}) ...",
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

            # #2: continuous-token embedders on device (HunyuanTtTimestepEmbedder — same
            # ported math as te1/te2). denoise_loop(device_step_embeds=True) embeds +
            # scatters them into a resident device base_embeds each step, replacing the
            # host re-scatter + per-step [B,S,H] upload.
            timestep_emb = HunyuanTtTimestepEmbedder(
                mesh_device,
                H,
                {f"timestep_emb.{k}": v for k, v in _load_prefix("timestep_emb").items()},
                "timestep_emb",
            )
            guidance_emb = (
                HunyuanTtTimestepEmbedder(
                    mesh_device,
                    H,
                    {f"guidance_emb.{k}": v for k, v in _load_prefix("guidance_emb").items()},
                    "guidance_emb",
                )
                if cfg_distilled
                else None
            )
            timestep_r_emb = (
                HunyuanTtTimestepEmbedder(
                    mesh_device,
                    H,
                    {f"timestep_r_emb.{k}": v for k, v in _load_prefix("timestep_r_emb").items()},
                    "timestep_r_emb",
                )
                if use_meanflow
                else None
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
                f"[e2e] denoising {STEPS} steps on resident bf8 backbone "
                f"({mode}, guidance={GUIDANCE}, seq_len={S}, device continuous-token embed) ...",
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
                device_step_embeds=True,
                mesh_device=mesh_device,
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


def run_lm_recaption(c):
    """Text AR smoke on the LM backbone (device wte embed + ln_f) + LM head.

    Exercises the text half of the model the diffusion path skips, mirroring demo.py's
    recaption stage but tokenizer-free: the resident backbone built as an LM
    (``apply_final_norm=True`` + device wte embed), the vocabulary projection
    (``HunyuanTtLMHead``), and the shared AR sampling loop (``generate_text``). It
    decodes ``RECAPTION_TOKENS`` from a RANDOM ``input_ids`` prefix and discards the
    text — no tokenizer to build a prompt or detokenize.

    Runs on its own 2x2 mesh, freed before the denoise/VAE meshes open (the resident
    bf8 LM backbone fills DRAM, same discipline as demo.py's recaption stage)."""
    H = c["H"]
    wte = _load("model.wte.weight")
    vocab = int(wte.shape[0])
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, vocab, (B, RECAPTION_PREFIX), dtype=torch.long)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 2), l1_small_size=32768)
    try:
        mesh_device.enable_program_cache()
        ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
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
            embed_state_dict={"model.wte.weight": wte},
            norm_state_dict={"model.ln_f.weight": _load("model.ln_f.weight")},
            apply_final_norm=True,  # LM backbone: ln_f before the head (unlike the gen path)
            weight_dtype=ttnn.bfloat8_b,
            ccl_manager=ccl,
            expert_mesh_axis=1,
            tp_axis=1,
            tp_factor=2,
            sp_axis=0,
            sp_factor=2,
            bf16_layers=bf16_layers,
        )
        lm_head = HunyuanTtLMHead(mesh_device, {"lm_head.weight": _load("lm_head.weight")})
        sampler = "host-stochastic" if RECAPTION_SAMPLE else "device-argmax"
        print(
            f"[e2e] LM recaption smoke: prefix={RECAPTION_PREFIX} -> {RECAPTION_TOKENS} tokens "
            f"(wte + {NUM_LAYERS}-layer backbone + ln_f + LM head + AR {sampler}, vocab={vocab}) ...",
            flush=True,
        )
        if not RECAPTION_SAMPLE:
            # #3: greedy sampling ON DEVICE. Run backbone + LM head, pick argmax on device
            # (ttnn.argmax over the last-token logits); only the chosen id crosses to host.
            # attention_mask=None => causal SDPA inside the backbone.
            ids = input_ids
            new_tokens = []
            for _ in range(RECAPTION_TOKENS):
                ids_tt = ttnn.from_torch(
                    ids,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                hidden = backbone.forward(input_ids=ids_tt, seq_len=ids.shape[1], attention_mask=None)
                logits_tt = lm_head(hidden, last_token_only=True)  # [B, 1, V]
                tok_tt = ttnn.argmax(logits_tt, dim=-1)  # next-token id(s), on device
                if mesh_device.get_num_devices() > 1:
                    tok = ttnn.to_torch(tok_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
                    tok = tok[: ids.shape[0]]
                else:
                    tok = ttnn.to_torch(tok_tt)
                tok = tok.reshape(ids.shape[0], -1)[:, -1].long()  # [B]
                ttnn.deallocate(tok_tt)
                ttnn.deallocate(logits_tt)
                ttnn.deallocate(hidden)
                ttnn.deallocate(ids_tt)
                new_tokens.append(int(tok[0]))
                ids = torch.cat([ids, tok.reshape(ids.shape[0], 1)], dim=1)
            print(f"[e2e] LM recaption (device argmax) produced {len(new_tokens)} tokens: {new_tokens}", flush=True)
            return new_tokens
        # Host stochastic path (parity-gated golden): backbone+head on device, sample on host.
        forward_logits_fn = make_backbone_logits_fn(backbone, lm_head, mesh_device)
        cfg = SamplingConfig(do_sample=True, max_new_tokens=RECAPTION_TOKENS)
        out = generate_text(forward_logits_fn, input_ids, config=cfg, generator=torch.Generator().manual_seed(SEED))
        print(f"[e2e] LM recaption produced {len(out['new_tokens'][0])} tokens: {out['new_tokens'][0]}", flush=True)
        return out["new_tokens"][0]
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def run_vision_encode():
    """Encode a RANDOM conditioning image through the device SigLIP2 tower + aligner.

    Exercises the I2I vision front-end (tt/vision): SigLIP2 encoder + LightProjector,
    producing projected cond-image embeddings [1, n_img, H] to scatter into the backbone
    sequence. Host-side image preprocessing uses the reference SigLIP2 processor (no
    tokenizer). Runs on a single device — the proven config from the vision PCC tests
    (tests/vision/test_i2i_inputs_embeds.py) — then hands the embeddings to host so the
    denoise mesh can open cleanly afterwards.

    Returns (image_embeds_host [1, n_img, H], (token_h, token_w), n_img)."""
    from PIL import Image

    # Device (TTNN) compute: SigLIP2 tower + aligner + encode. These run on device.
    from models.experimental.hunyuan_image_3_0.tt.vision.i2i import encode_cond_vision
    from models.experimental.hunyuan_image_3_0.tt.vision.siglip2 import (
        HunyuanTtLightProjector,
        HunyuanTtSiglip2Vision,
        Siglip2VisionInputs,
    )

    # Host-side image preprocessing via the tt/vision bridge (PIL -> pixel patches is
    # inherently host/CPU — tt/vision/preprocess wraps the golden ref processor verbatim).
    from models.experimental.hunyuan_image_3_0.tt.vision.preprocess import (
        build_cond_image_processor,
        process_cond_image,
    )

    # ref SigLIP2/aligner modules are used ONLY as host weight containers: the TTNN
    # modules consume `state_dict()`, and their key layout is PCC-validated against
    # exactly these (tests/vision/conftest.py). No vision math runs on them.
    from models.experimental.hunyuan_image_3_0.ref.vision.siglip2 import load_aligner, load_siglip2_vision

    mdir = Path(WEIGHTS)
    processor = build_cond_image_processor(mdir)
    torch.manual_seed(SEED)
    arr = torch.randint(0, 256, (COND_H, COND_W, 3), dtype=torch.uint8).numpy()
    image = Image.fromarray(arr, mode="RGB")
    pixel_values, (th, tw), pixel_attention_mask = process_cond_image(processor, image)
    if pixel_values.ndim == 2:  # normalise to per-image batched form [1, S, patch_dim] / [1, S]
        pixel_values = pixel_values.unsqueeze(0)
    if pixel_attention_mask.ndim == 1:
        pixel_attention_mask = pixel_attention_mask.unsqueeze(0)
    n_img = int(pixel_values.shape[1])

    print(
        f"[e2e] vision encode: cond image {COND_H}x{COND_W} -> {n_img} patches "
        f"(grid {th}x{tw}), SigLIP2 layers={VIT_LAYERS} ...",
        flush=True,
    )
    vision_sd = load_siglip2_vision(mdir, num_layers=VIT_LAYERS).state_dict()
    aligner_sd = load_aligner(mdir).state_dict()

    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:

        def up(t):
            return ttnn.from_torch(
                t.to(torch.float32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        vision_tt = HunyuanTtSiglip2Vision(dev, vision_sd, num_layers=VIT_LAYERS)
        vision_tt.prewarm_pos_geometries([(th, tw, n_img)])
        aligner_tt = HunyuanTtLightProjector(dev, aligner_sd)
        vision_inputs = Siglip2VisionInputs.create(up(pixel_values), ((th, tw),), up(pixel_attention_mask))
        emb_tt = encode_cond_vision(vision_tt, aligner_tt, vision_inputs)  # [1, n_img, H]
        image_embeds = ttnn.to_torch(emb_tt).float()
        ttnn.deallocate(emb_tt)
    finally:
        ttnn.close_device(dev)
    n_img = int(image_embeds.shape[1])
    print(
        f"[e2e] vision embeds {tuple(image_embeds.shape)} " f"finite={bool(torch.isfinite(image_embeds).all())}",
        flush=True,
    )
    return image_embeds, (th, tw), n_img


def run_denoise_i2i(c, down_sd, up_sd, init_latent, image_embeds, cond_hw, n_img):
    """I2I denoise: cond SigLIP2 tokens scattered into the sequence -> latent (torch NCHW).

    Mirrors the Instruct/Distil gen path of ``run_denoise`` but adds a conditioning ViT
    ``<img>`` span before the gen span: a multi-span attention mask + 2D RoPE (cond +
    gen), with the cond-image embeddings (from ``run_vision_encode``) scattered into
    ``base_embeds`` on host. The gen span is overwritten each step by patch_embed.

    The sequence layout is synthesised without a tokenizer (which normally emits it):

        [ cond-ViT span | pad-text + per-step special tokens | gen span | text_post ]

    gen-span boundaries are TILE-aligned (32) for the on-device scatter; per-step
    continuous tokens (gen_timestep / guidance / timestep_r) sit immediately before the
    gen span, in the tokenizer's canonical order (see ``gen_special_token_indices``)."""
    LATENT, HID, HSZ = _pe_dims(down_sd)
    H = c["H"]
    cfg_distilled, use_meanflow = _model_flags()

    # --- synthetic I2I sequence layout ------------------------------------------------
    _, _, _, n_special = gen_special_token_indices(0, cfg_distilled, use_meanflow)
    TILE = 32
    gen_start = ((max(n_img + n_special, TEXT_PRE) + TILE - 1) // TILE) * TILE
    gen_slice = slice(gen_start, gen_start + N_IMG)
    S_v = gen_start + N_IMG + TEXT_POST
    cond_slice = slice(0, n_img)
    assert gen_start - n_special >= n_img, (
        f"cond span ({n_img}) overlaps the {n_special} per-step special tokens before gen "
        f"(gen_start={gen_start}); raise HY_TEXT_PRE or shrink the cond image"
    )
    assert S_v <= c["MAX_SEQ"], f"I2I seq_len {S_v} exceeds max_position_embeddings {c['MAX_SEQ']}"
    # The cond ViT <img> span is the PADDED patch count (max_num_patches), exactly as
    # the tokenizer reserves it: all aligner rows (real + pad) are scattered into the
    # span, and build_2d_rope lays the (th,tw) grid over the first th*tw positions with
    # sequential ids on the padded tail (it reads only slice.start, never asserts
    # span==h*w). Mirrors the real I2I denoise bundle (verified in gen_image_inputs.py /
    # rope_2d.py) and what demo_i2i.py feeds the device backbone.
    assert n_img >= cond_hw[0] * cond_hw[1], (
        f"cond span {n_img} < patch grid {cond_hw[0]}x{cond_hw[1]}={cond_hw[0] * cond_hw[1]} "
        f"(unexpected — the aligner output should be the padded max_num_patches)"
    )
    gen_ts_idx, guidance_idx, timestep_r_idx, _ = gen_special_token_indices(gen_start, cfg_distilled, use_meanflow)

    # base embeds: wte(random ids) with cond ViT tokens scattered into the <img> span.
    wte = _load("model.wte.weight")
    torch.manual_seed(SEED + 7)
    ids = torch.randint(0, int(wte.shape[0]), (B, S_v), dtype=torch.long)
    base = torch.nn.functional.embedding(ids, wte).to(torch.float32)
    base[:, cond_slice, :] = image_embeds.to(base.dtype)

    image_infos = [[(cond_slice, cond_hw), (gen_slice, (GRID, GRID))]]

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
            img_slice=gen_slice,
            grid_hw=(GRID, GRID),
            seq_len=S_v,
        )

        # #1: multi-span attention mask (cond + gen) built entirely on device.
        mask_tt = build_attention_mask_tt(
            mesh_device, S_v, image_slices=[cond_slice, gen_slice], bsz=B, dtype=ttnn.bfloat16
        )

        li = lambda i: torch.tensor([[i]] * B, dtype=torch.long)
        # #2: continuous-token embedders on device (see run_denoise).
        timestep_emb = HunyuanTtTimestepEmbedder(
            mesh_device, H, {f"timestep_emb.{k}": v for k, v in _load_prefix("timestep_emb").items()}, "timestep_emb"
        )
        guidance_emb = (
            HunyuanTtTimestepEmbedder(
                mesh_device,
                H,
                {f"guidance_emb.{k}": v for k, v in _load_prefix("guidance_emb").items()},
                "guidance_emb",
            )
            if cfg_distilled
            else None
        )
        timestep_r_emb = (
            HunyuanTtTimestepEmbedder(
                mesh_device,
                H,
                {f"timestep_r_emb.{k}": v for k, v in _load_prefix("timestep_r_emb").items()},
                "timestep_r_emb",
            )
            if use_meanflow
            else None
        )
        cond = dict(
            base_embeds_host=base,
            gen_timestep_scatter_index=li(gen_ts_idx),
            guidance_scatter_index=li(guidance_idx) if cfg_distilled else None,
            gen_timestep_r_scatter_index=li(timestep_r_idx) if use_meanflow else None,
            image_infos=image_infos,
            attention_mask=mask_tt,
            batch=B,
        )
        sched = HunyuanTtScheduler(mesh_device)
        sched.set_timesteps(STEPS)
        mode = "distil" if cfg_distilled else ("meanflow" if use_meanflow else "instruct")
        print(
            f"[e2e] I2I denoising {STEPS} steps on resident bf8 backbone "
            f"({mode}, guidance={GUIDANCE}, seq_len={S_v}, cond_span=[0:{n_img}], gen_span={gen_slice}) ...",
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
            device_step_embeds=True,
            mesh_device=mesh_device,
        )
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    return latent


def run_pipeline(seed=SEED):
    """Full random-input pipeline -> (rgb [B,3,H,W] in [0,1], latent [B,C,GRID,GRID])."""
    c = _cfg()
    assert S <= c["MAX_SEQ"], f"seq_len {S} (grid {GRID}) exceeds max_position_embeddings {c['MAX_SEQ']}"
    down_sd = _load_prefix("patch_embed")
    up_sd = _load_prefix("final_layer")
    LATENT, _, _ = _pe_dims(down_sd)
    assert LATENT == Z_CHANNELS, f"diffusion latent ch {LATENT} != VAE z-channels {Z_CHANNELS}"

    cfg_distilled, use_meanflow = _model_flags()
    base_cfg = (not (cfg_distilled or use_meanflow)) and BASE_GUIDANCE != 1.0 and not VISION
    if VISION and HY_MODEL == "base":
        raise SystemExit("HY_VISION=1 needs Instruct/Instruct-Distil weights (base T2I has no vision tower)")
    print(
        f"[e2e] model={HY_MODEL} ({WEIGHTS})  cfg_distilled={cfg_distilled} use_meanflow={use_meanflow}"
        f"  recaption={RECAPTION} vision={VISION}"
        f"{f'  base_CFG guidance={BASE_GUIDANCE}' if base_cfg else ''}\n"
        f"[e2e] grid={GRID}x{GRID}  seq_len={S} (<= max {c['MAX_SEQ']})  layers={NUM_LAYERS}  "
        f"steps={STEPS}  image={GRID * 16}x{GRID * 16}  seed={seed}",
        flush=True,
    )
    t0 = time.time()

    # 0) LM / recaption stage (device wte + ln_f backbone + LM head + AR sampling).
    #    Own mesh, freed before the denoise backbone opens.
    if RECAPTION:
        run_lm_recaption(c)

    # 1) noise latent + random token ids. The wte lookup that turns ids into text
    #    embeddings is the "text embedding" stage: on the base-T2I path it runs ON DEVICE
    #    inside run_denoise (backbone.embed, #4); the host embeds below are the exact
    #    fallback and the base_embeds source for the Instruct/Distil path.
    wte = _load("model.wte.weight")
    vocab = int(wte.shape[0])
    torch.manual_seed(seed)
    init_latent = torch.randn(B, LATENT, GRID, GRID)
    text_ids = torch.randint(0, vocab, (B, S), dtype=torch.long)
    text_embeds = torch.nn.functional.embedding(text_ids, wte).to(torch.float32)
    # Unconditional (empty-prompt) ids/embeds for base CFG — a distinct deterministic draw.
    text_ids_uncond = torch.randint(0, vocab, (B, S), dtype=torch.long) if base_cfg else None
    text_embeds_uncond = torch.nn.functional.embedding(text_ids_uncond, wte).to(torch.float32) if base_cfg else None

    # 2) denoise. Vision (I2I): encode a random cond image on device, then denoise with
    #    the cond ViT span scattered into the sequence. Otherwise the T2I gen path.
    if VISION:
        image_embeds, cond_hw, n_img = run_vision_encode()
        latent = run_denoise_i2i(c, down_sd, up_sd, init_latent, image_embeds, cond_hw, n_img)
    else:
        latent = run_denoise(
            c,
            down_sd,
            up_sd,
            init_latent,
            text_embeds,
            text_embeds_uncond=text_embeds_uncond,
            cfg_guidance=BASE_GUIDANCE if base_cfg else 1.0,
            text_ids=text_ids,
            text_ids_uncond=text_ids_uncond,
            wte_weight=wte,
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
