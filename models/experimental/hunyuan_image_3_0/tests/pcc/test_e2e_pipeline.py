# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# PCC gate for the FULL end-to-end pipeline in demo/e2e.py, on RANDOM inputs.
#
# This test does NOT re-implement the device pipeline — it imports run_denoise /
# run_vae_decode (and the config helpers) from demo/e2e.py so the TT pipeline is
# defined exactly once. Here we only add the fp32 HOST reference (denoise loop +
# shape-agnostic torch VAE) and PCC-gate both hand-offs against it:
#
#     latent  (resident bf8 backbone denoise loop)            -> PCC #1
#     RGB     (latent -> TTNN VAE decode -> (x/2+0.5))         -> PCC #2
#
# Config is inherited from demo/e2e.py (HY_GRID / HY_TEXT_PRE / HY_TEXT_POST /
# HY_NUM_LAYERS / HY_STEPS). PCC thresholds: HY_LATENT_PCC (0.98), HY_RGB_PCC (0.97).
#
# NOTE (physical limits): PCC is computed at EVERY grid with no safeguards. The
# fp32 host reference (S*S attention; full-res VAE) gets very slow / memory-heavy
# at large grids — use a small HY_GRID for a fast, meaningful gate.
#
# Run:
#   HY_GRID=8 python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_e2e_pipeline.py -v -s --timeout=1800

import os, sys, gc, importlib.util
import torch

ROOT = "/home/iguser/christy/tt-metal"
HUNYUAN = "/home/iguser/christy/HunyuanImage-3.0"
WEIGHTS = "/home/iguser/christy/HunyuanImage-3"
for p in (ROOT, HUNYUAN):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("HUNYUAN_MODEL_DIR", WEIGHTS)

# Load the pipeline (demo/e2e.py is not in an importable package) and reuse its
# device pipeline + config verbatim — single source of truth.
_E2E_PATH = os.path.join(ROOT, "models/experimental/hunyuan_image_3_0/demo/e2e.py")
_spec = importlib.util.spec_from_file_location("hy_e2e_pipeline", _E2E_PATH)
e2e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2e)

from models.experimental.hunyuan_image_3_0.ref.transformer_layer import HunyuanImage3DecoderLayer as RefLayer
from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive
from models.experimental.hunyuan_image_3_0.ref.attention.rope_2d import build_batch_2d_rope
from models.experimental.hunyuan_image_3_0.ref.image_gen.patch_embed import UNetDown as RefDown, UNetUp as RefUp
from models.experimental.hunyuan_image_3_0.ref.image_gen.timestep_embedder import TimestepEmbedder as RefTimeEmbed
from models.experimental.hunyuan_image_3_0.ref.vae.decoder import (
    Z_CHANNELS,
    load_decoder as load_ref_decoder,
    vae_decode_output_to_rgb,
)
from models.experimental.hunyuan_image_3_0.tt.scheduler import HunyuanTtScheduler

# Config / constants come straight from the pipeline module (env-driven there).
B, GRID, S = e2e.B, e2e.GRID, e2e.S
IMG_START, IMG_SLICE, N_IMG = e2e.IMG_START, e2e.IMG_SLICE, e2e.N_IMG
NUM_LAYERS, STEPS, SCALING = e2e.NUM_LAYERS, e2e.STEPS, e2e.SCALING

LATENT_PCC_THR = float(os.environ.get("HY_LATENT_PCC", "0.98"))
RGB_PCC_THR = float(os.environ.get("HY_RGB_PCC", "0.97"))


def _pcc(a, b):
    a = a.float().flatten() - a.float().mean()
    b = b.float().flatten() - b.float().mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def _host_reference(c, down_sd, up_sd, init_latent, text_embeds, text_embeds_uncond=None, cfg_guidance=1.0):
    """fp32 host reference: denoise loop -> latent, then torch VAE decode -> RGB.

    Reads the SAME scheduler schedule the pipeline uses, so divergence isolates to
    the TTNN port rather than the schedule. When ``text_embeds_uncond`` is given and
    ``cfg_guidance`` != 1, runs the base classifier-free-guidance combine each step
    (mirrors run_denoise's CFG path)."""
    LATENT, HID, HSZ = e2e._pe_dims(down_sd)
    H = c["H"]
    sched = HunyuanTtScheduler(None)
    sched.set_timesteps(STEPS)
    sigmas, timesteps = sched.sigmas, sched.timesteps

    rd = RefDown(1, LATENT, HSZ, HID, HSZ).eval()
    ru = RefUp(1, HSZ, HSZ, HID, LATENT, out_norm=True).eval()
    rd.load_state_dict({k: v.float() for k, v in down_sd.items()}, strict=True)
    ru.load_state_dict({k: v.float() for k, v in up_sd.items()}, strict=True)
    te1r = RefTimeEmbed(H).eval()
    te1r.load_state_dict({k: v.float() for k, v in e2e._load_prefix("time_embed").items()}, strict=True)
    te2r = RefTimeEmbed(H).eval()
    te2r.load_state_dict({k: v.float() for k, v in e2e._load_prefix("time_embed_2").items()}, strict=True)
    layers = []
    for i in range(NUM_LAYERS):
        sd = e2e._load_prefix(f"model.layers.{i}")
        L = RefLayer(
            hidden_size=H,
            num_attention_heads=c["HEADS"],
            num_key_value_heads=c["KV"],
            attention_head_dim=c["HD"],
            num_experts=c["E"],
            moe_topk=c["K"],
            moe_intermediate_size=c["INTER"],
            num_shared_expert=c["SHARED"],
            use_mixed_mlp_moe=c["MIXED"],
            norm_topk_prob=c["NORM"],
            use_qk_norm=c["QKN"],
            rms_norm_eps=c["EPS"],
            layer_idx=i,
        )
        L.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
        layers.append(L.eval())
    cos, sin = build_batch_2d_rope(S, c["HD"], image_infos=[[(IMG_SLICE, (GRID, GRID))]])
    mask = to_additive(build_attention_mask(S, image_slices=[IMG_SLICE], bsz=B), dtype=torch.float32)

    do_cfg = text_embeds_uncond is not None and cfg_guidance != 1.0
    print(f"[e2e-pcc] host reference denoise loop{' (+CFG)' if do_cfg else ''} ...", flush=True)

    def _pred(lat, te_embeds, e1, e2_):
        img, th, tw = rd(lat, e1)
        h = te_embeds.clone()
        h[:, IMG_SLICE, :] = img
        for L in layers:
            h = L(h, attention_mask=mask, custom_pos_emb=(cos, sin))
        return ru(h[:, IMG_SLICE, :], e2_, th, tw)

    lat = init_latent.clone()
    for i, t in enumerate(timesteps):
        tv = torch.tensor([float(t)] * B)
        with torch.no_grad():
            e1, e2_ = te1r(tv), te2r(tv)
            pred = _pred(lat, text_embeds, e1, e2_)
            if do_cfg:
                pred_uncond = _pred(lat, text_embeds_uncond, e1, e2_)
                # uncond + scale*(cond - uncond) == classifier_free_guidance_tt
                pred = pred_uncond + cfg_guidance * (pred - pred_uncond)
        lat = lat + float(sigmas[i + 1] - sigmas[i]) * pred

    print("[e2e-pcc] host reference VAE decode ...", flush=True)
    with torch.no_grad():
        ref_img = load_ref_decoder()((lat.float() / SCALING).unsqueeze(2))
    ref_rgb = vae_decode_output_to_rgb(ref_img)
    return lat, ref_rgb


def _run():
    c = e2e._cfg()
    assert S <= c["MAX_SEQ"], f"seq_len {S} (grid {GRID}) exceeds max_position_embeddings {c['MAX_SEQ']}"
    down_sd = e2e._load_prefix("patch_embed")
    up_sd = e2e._load_prefix("final_layer")
    LATENT, _, _ = e2e._pe_dims(down_sd)
    assert LATENT == Z_CHANNELS, f"diffusion latent ch {LATENT} != VAE z-channels {Z_CHANNELS}"

    # Base classifier-free guidance: mirror demo/e2e.py — CFG when BASE_GUIDANCE != 1
    # on the base path (no per-step continuous tokens).
    cfg_distilled, use_meanflow = e2e._model_flags()
    base_cfg = (not (cfg_distilled or use_meanflow)) and e2e.BASE_GUIDANCE != 1.0
    cfg_guidance = e2e.BASE_GUIDANCE if base_cfg else 1.0

    print(
        f"\n[e2e-pcc] grid={GRID}x{GRID}  seq_len={S} (<= max {c['MAX_SEQ']})  layers={NUM_LAYERS}  "
        f"steps={STEPS}  image={GRID * 16}x{GRID * 16}{f'  base_CFG={cfg_guidance}' if base_cfg else ''}",
        flush=True,
    )
    torch.manual_seed(0)
    init_latent = torch.randn(B, LATENT, GRID, GRID)
    text_embeds = torch.randn(B, S, c["H"]) * 0.02
    # Same deterministic uncond draw as run_pipeline (manual_seed(0) above, then this is
    # the 3rd randn — keep both paths drawing it identically so cond/uncond match).
    text_embeds_uncond = torch.randn(B, S, c["H"]) * 0.02 if base_cfg else None

    # Host fp32 reference (denoise -> latent -> VAE -> RGB).
    ref_latent, ref_rgb = _host_reference(
        c,
        down_sd,
        up_sd,
        init_latent,
        text_embeds,
        text_embeds_uncond=text_embeds_uncond,
        cfg_guidance=cfg_guidance,
    )
    gc.collect()

    # TT pipeline — the SAME functions demo/e2e.py runs.
    print("[e2e-pcc] TT pipeline: resident backbone denoise ...", flush=True)
    tt_latent = e2e.run_denoise(
        c,
        down_sd,
        up_sd,
        init_latent,
        text_embeds,
        text_embeds_uncond=text_embeds_uncond,
        cfg_guidance=cfg_guidance,
    )
    print("[e2e-pcc] TT pipeline: VAE decode ...", flush=True)
    tt_rgb = e2e.run_vae_decode(tt_latent)

    return dict(
        latent_pcc=_pcc(ref_latent, tt_latent),
        rgb_pcc=_pcc(ref_rgb, tt_rgb),
        latent_dmax=(ref_latent.float() - tt_latent.float()).abs().max().item(),
        ref_rgb_shape=tuple(ref_rgb.shape),
        tt_rgb_shape=tuple(tt_rgb.shape),
    )


def test_gen_special_token_order():
    """Guards the special-token ordering fix: e2e.gen_special_token_indices() must
    match the tokenizer's canonical layout — encode_sequence appends gen_timestep,
    then guidance (distil), then timestep_r (meanflow) immediately before the gen
    span, so gen_timestep is FURTHEST from the image. Ground-truthed against
    prepare_gen_image_inputs for the cases the base vocab can build (base + distil);
    meanflow's <timestep_r> is absent from the base vocab, so its relative order is
    checked structurally. This is the gate for the distil/meanflow path, whose
    weights are not present on disk for a full device PCC."""
    import dataclasses
    from models.experimental.hunyuan_image_3_0.ref.tokenizer.hunyuan_tokenizer import HunyuanTokenizer
    from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import (
        prepare_gen_image_inputs,
        get_gen_image_slice,
    )

    tok = HunyuanTokenizer.from_pretrained()

    def canonical(cfg_distilled, use_meanflow):
        cfgo = dataclasses.replace(tok.config, cfg_distilled=cfg_distilled, use_meanflow=use_meanflow)
        tk = HunyuanTokenizer(cfgo, tok.tokenizer, tok.special, sequence_template=tok.sequence_template)
        b = prepare_gen_image_inputs(tk, "a cat", image_size=256)
        gs = get_gen_image_slice(b, 0)
        idx = lambda v: None if v is None else int(v[0, 0])
        return (
            gs.start,
            idx(b.gen_timestep_scatter_index),
            idx(b.guidance_scatter_index),
            idx(b.gen_timestep_r_scatter_index),
        )

    # Tokenizer ground truth for base + distil (vocab-buildable).
    for cfg_distilled in (False, True):
        img_start, g_ts, g_guid, g_tr = canonical(cfg_distilled, False)
        ts, guid, tr, n = e2e.gen_special_token_indices(img_start, cfg_distilled, False)
        assert ts == g_ts, f"gen_timestep idx {ts} != tokenizer {g_ts} (distil={cfg_distilled})"
        assert guid == g_guid, f"guidance idx {guid} != tokenizer {g_guid} (distil={cfg_distilled})"
        assert tr == g_tr, f"timestep_r idx {tr} != tokenizer {g_tr} (distil={cfg_distilled})"

    # Structural check incl. meanflow: order is gen_timestep (furthest) < guidance <
    # timestep_r (adjacent to image), tokens contiguous right before IMG_START.
    img_start = 32
    ts, guid, tr, n = e2e.gen_special_token_indices(img_start, True, True)
    assert n == 3 and ts < guid < tr and tr == img_start - 1 and ts == img_start - 3
    ts, guid, tr, n = e2e.gen_special_token_indices(img_start, False, True)
    assert n == 2 and guid is None and ts < tr and tr == img_start - 1
    print("[e2e-pcc] gen_special_token_order: indices match tokenizer canonical order", flush=True)


def test_e2e_pipeline():
    r = _run()
    print(
        f"\n[e2e-pcc] latent PCC={r['latent_pcc']:.6f} (thr {LATENT_PCC_THR})  max|Δ|={r['latent_dmax']:.4e}\n"
        f"[e2e-pcc] RGB    PCC={r['rgb_pcc']:.6f} (thr {RGB_PCC_THR})  "
        f"ref={r['ref_rgb_shape']} tt={r['tt_rgb_shape']}",
        flush=True,
    )
    assert r["ref_rgb_shape"] == r["tt_rgb_shape"], f"RGB shape mismatch {r['ref_rgb_shape']} vs {r['tt_rgb_shape']}"
    assert r["latent_pcc"] >= LATENT_PCC_THR, f"latent PCC {r['latent_pcc']:.6f} < {LATENT_PCC_THR}"
    assert r["rgb_pcc"] >= RGB_PCC_THR, f"RGB PCC {r['rgb_pcc']:.6f} < {RGB_PCC_THR}"


if __name__ == "__main__":
    r = _run()
    print(f"\ne2e latent PCC={r['latent_pcc']:.6f}  RGB PCC={r['rgb_pcc']:.6f}  rgb={r['tt_rgb_shape']}")
