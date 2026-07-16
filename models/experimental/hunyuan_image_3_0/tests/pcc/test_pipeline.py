# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Consolidated pipeline PCC tests:
#   - Transformer backbone (wte -> N layers -> ln_f)
#   - On-device denoise step (single device + mesh resident)
#   - End-to-end pipeline (opt-in random inputs; see HY_RUN_E2E_RANDOM)
#   - Special-token ordering gate for gen image inputs
#
# Lean ISL for backbone: S=1, 32 (fast); S=4096, 4160 (slow).
# Denoise step: GRID=8 S=128 (fast); GRID=64 S=4160 production (slow).
# Production CI gate (32L): see tests/run_pcc_production_slow.sh
#
# Run (fast):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_pipeline.py -m "not slow" -v
#
# E2E (opt-in; pytest.ini default timeout=300 is too short — pass --timeout):
#   HY_RUN_E2E_RANDOM=1 HY_NUM_LAYERS=32 HY_STEPS=8 \
#     python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_pipeline.py::test_e2e_pipeline \
#     -v -s --timeout=43200
# Long seq example (S = TEXT_PRE + GRID^2 + TEXT_POST = 12112):
#   HY_GRID=64 HY_TEXT_PRE=7984 HY_TEXT_POST=32 HY_RUN_E2E_RANDOM=1 HY_NUM_LAYERS=32 HY_STEPS=2 ...
#   (32L defaults HY_BASE_GUIDANCE=1 for densify-fair PCC; override to 5.0 to exercise CFG.)

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[5]
PCC_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PCC_DIR) not in sys.path:
    sys.path.insert(0, str(PCC_DIR))

import ttnn
from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive
from models.experimental.hunyuan_image_3_0.ref.attention.rms_norm import HunyuanRMSNorm
from models.experimental.hunyuan_image_3_0.ref.attention.rope_2d import build_batch_2d_rope
from models.experimental.hunyuan_image_3_0.ref.transformer_layer import HunyuanImage3DecoderLayer as RefLayer
from models.experimental.hunyuan_image_3_0.ref.vae.decoder import (
    Z_CHANNELS,
    load_decoder as load_ref_decoder,
    vae_decode_output_to_rgb,
)
from models.experimental.hunyuan_image_3_0.ref.weights import (
    load_prefixed_state_dict,
    load_tensors,
    resolve_base_model_dir,
)
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel
from models.experimental.hunyuan_image_3_0.tt.scheduler import HunyuanTtScheduler
from models.tt_dit.parallel.manager import CCLManager
from denoise_helpers import _forward_ref_layers, clear_ref_layer_cache
from pcc_common import (
    LEAN_ISL_CASES,
    PCC_BLOCK,
    PCC_DECODE_STACK,
    PIPELINE_LAYOUT_FAST,
    PIPELINE_LAYOUT_PROD,
    PRODUCTION_SEQ,
    pcc_metrics,
    transformer_cfg,
)
from pipeline_helpers import (
    bf16_layers_from_env,
    build_denoise_step_tt,
    e2e_pcc_thresholds,
    load_e2e_module,
    patch_embed_dims,
    pipeline_pcc_threshold,
    reference_denoise_step,
    reference_time_embed,
    resident_mesh_pcc_threshold,
    run_denoise_step_tt,
    weight_dtype_from_env,
)

BATCH = 1
NUM_LAYERS_BACKBONE = int(os.environ.get("HY_NUM_LAYERS", "2"))
NUM_LAYERS_STEP = int(os.environ.get("HY_NUM_LAYERS", "4"))
NUM_LAYERS_PRODUCTION = int(os.environ.get("HY_NUM_LAYERS", "32"))
# pytest.ini default timeout=300s kills long e2e (SIGTERM → "Terminated").
_E2E_TIMEOUT = 43200

BACKBONE_ISL_FAST = [(batch, seq_len, label) for batch, seq_len, label in LEAN_ISL_CASES if seq_len < 4096]
BACKBONE_ISL_SLOW = [(batch, seq_len, label) for batch, seq_len, label in LEAN_ISL_CASES if seq_len >= 4096]

DENOISE_LAYOUT_FAST = [("fast", PIPELINE_LAYOUT_FAST)]
DENOISE_LAYOUT_SLOW = [("prod", PIPELINE_LAYOUT_PROD)]


@pytest.fixture(scope="function")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="function")
def device_params():
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768}


def _backbone_run(device, seq_len: int, num_layers: int = NUM_LAYERS_BACKBONE):
    c = transformer_cfg()
    torch.manual_seed(0)
    input_ids = torch.randint(0, 130000, (BATCH, seq_len), dtype=torch.long)

    wte_w = load_tensors(resolve_base_model_dir(), ["model.wte.weight"])["model.wte.weight"]
    lnf_w = load_tensors(resolve_base_model_dir(), ["model.ln_f.weight"])["model.ln_f.weight"]

    def make_ref_layer(i):
        sd = load_prefixed_state_dict(resolve_base_model_dir(), f"model.layers.{i}.")
        layer = RefLayer(
            hidden_size=c["H"],
            num_attention_heads=c["HEADS"],
            num_key_value_heads=c["KV"],
            attention_head_dim=c["HD"],
            num_experts=c["E"],
            moe_topk=c["K"],
            moe_intermediate_size=c["MOE_INTER"],
            num_shared_expert=c["NUM_SHARED"],
            use_mixed_mlp_moe=c["MIXED"],
            norm_topk_prob=c["NORM_TOPK"],
            use_qk_norm=c["QKN"],
            rms_norm_eps=c["EPS"],
            layer_idx=i,
        )
        layer.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
        layer.eval()
        return layer

    ref_layers = [make_ref_layer(i) for i in range(num_layers)]
    ref_lnf = HunyuanRMSNorm(c["H"], eps=c["EPS"])
    ref_lnf.load_state_dict({"weight": lnf_w.float()})
    ref_lnf.eval()

    cos, sin = build_batch_2d_rope(seq_len, c["HD"], image_infos=None)
    mask_add = to_additive(build_attention_mask(seq_len, image_slices=None, bsz=BATCH), dtype=torch.float32)
    with torch.no_grad():
        h = torch.nn.functional.embedding(input_ids, wte_w.float())
        for layer in ref_layers:
            h = layer(h, attention_mask=mask_add, custom_pos_emb=(cos, sin))
        ref_out = ref_lnf(h)

    layer_loader = lambda i: {
        f"model.layers.{i}.{k}": v
        for k, v in load_prefixed_state_dict(resolve_base_model_dir(), f"model.layers.{i}.").items()
    }
    tt_model = HunyuanTtModel(
        device,
        num_layers=num_layers,
        hidden_size=c["H"],
        num_heads=c["HEADS"],
        num_kv_heads=c["KV"],
        head_dim=c["HD"],
        num_experts=c["E"],
        moe_topk=c["K"],
        use_qk_norm=c["QKN"],
        use_mixed_mlp_moe=c["MIXED"],
        norm_topk_prob=c["NORM_TOPK"],
        rms_norm_eps=c["EPS"],
        stream_experts=True,
        layer_loader=layer_loader,
        embed_state_dict={"model.wte.weight": wte_w},
        norm_state_dict={"model.ln_f.weight": lnf_w},
        apply_final_norm=True,
    )
    ids_tt = ttnn.from_torch(
        input_ids.to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_tt = tt_model(ids_tt, seq_len=seq_len, image_infos=None, attention_mask=None)
    tt_out = ttnn.to_torch(out_tt)[..., : c["H"]]
    ids_tt.deallocate(True)
    return pcc_metrics(ref_out, tt_out, PCC_BLOCK)


def _denoise_step_run(device, layout: dict, num_layers: int = NUM_LAYERS_STEP, mesh_composer=None):
    c = transformer_cfg()
    down_sd = load_prefixed_state_dict(resolve_base_model_dir(), "patch_embed.")
    up_sd = load_prefixed_state_dict(resolve_base_model_dir(), "final_layer.")
    h = c["H"]
    weight_dtype = weight_dtype_from_env()
    thr = pipeline_pcc_threshold(num_layers, weight_dtype)

    torch.manual_seed(0)
    latent_ch, _, _ = patch_embed_dims(down_sd)
    grid = layout["grid"]
    s = layout["seq_len"]
    latent = torch.randn(BATCH, latent_ch, grid, grid)
    text_embeds = torch.randn(BATCH, s, h) * 0.02
    timesteps = torch.rand(BATCH)
    t_emb1 = reference_time_embed("time_embed", h, timesteps)
    t_emb2 = reference_time_embed("time_embed_2", h, timesteps)

    ref_pred = reference_denoise_step(c, layout, num_layers, latent, t_emb1, t_emb2, text_embeds, down_sd, up_sd, BATCH)

    if mesh_composer is not None:
        ccl = CCLManager(device, num_links=1, topology=ttnn.Topology.Linear)
        step = build_denoise_step_tt(
            device,
            c,
            layout,
            num_layers,
            down_sd,
            up_sd,
            weight_dtype=ttnn.bfloat8_b,
            stream_experts=False,
            ccl_manager=ccl,
            expert_mesh_axis=1,
            tp_axis=1,
            tp_factor=2,
            sp_axis=0,
            sp_factor=2,
        )
        thr = resident_mesh_pcc_threshold()
    else:
        step = build_denoise_step_tt(
            device,
            c,
            layout,
            num_layers,
            down_sd,
            up_sd,
            weight_dtype=weight_dtype,
            bf16_layers=bf16_layers_from_env(),
        )

    pred = run_denoise_step_tt(step, device, layout, latent, text_embeds, t_emb1, t_emb2, BATCH, mesh_composer)
    p, d = pcc_metrics(ref_pred, pred, thr)
    return p, d, thr


# ---------------------------------------------------------------------------
# Backbone (test_model.py)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("batch,seq_len,label", BACKBONE_ISL_FAST)
def test_backbone_pcc(device, batch, seq_len, label):
    assert batch == 1
    p, d = _backbone_run(device, seq_len)
    thr = PCC_DECODE_STACK if seq_len == 1 else PCC_BLOCK
    print(f"backbone [{label}] S={seq_len} layers={NUM_LAYERS_BACKBONE}: PCC={p:.8f}  max|diff|={d:.6f}  thr={thr}")
    assert p >= thr


@pytest.mark.slow
@pytest.mark.parametrize("batch,seq_len,label", BACKBONE_ISL_SLOW)
def test_backbone_large_isl_pcc(device, batch, seq_len, label):
    assert batch == 1
    p, d = _backbone_run(device, seq_len)
    assert p >= PCC_BLOCK


# ---------------------------------------------------------------------------
# Denoise step — single device (test_pipeline_step.py)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tag,layout", DENOISE_LAYOUT_FAST)
def test_denoise_step_pcc(device, tag, layout):
    p, d, thr = _denoise_step_run(device, layout)
    print(
        f"denoise step [{tag}] GRID={layout['grid']} S={layout['seq_len']} "
        f"layers={NUM_LAYERS_STEP}: PCC={p:.8f}  max|diff|={d:.6f}  thr={thr}"
    )
    assert p >= thr


@pytest.mark.slow
@pytest.mark.parametrize("tag,layout", DENOISE_LAYOUT_SLOW)
def test_denoise_step_production_pcc(device, tag, layout):
    p, d, thr = _denoise_step_run(device, layout)
    assert p >= thr


@pytest.mark.slow
def test_backbone_production_32l_pcc(device):
    """32-layer backbone hidden PCC at production ISL S=4160 (real weights, full HF width)."""
    p, d = _backbone_run(device, PRODUCTION_SEQ, num_layers=NUM_LAYERS_PRODUCTION)
    print(f"backbone production 32L S={PRODUCTION_SEQ}: PCC={p:.8f}  max|diff|={d:.6f}  thr={PCC_BLOCK}")
    assert p >= PCC_BLOCK


@pytest.mark.slow
def test_backbone_production_32l_decode_pcc(device):
    """32-layer chained backbone (not teacher-forced) at decode S=1."""
    p, d = _backbone_run(device, 1, num_layers=NUM_LAYERS_PRODUCTION)
    print(f"backbone production 32L decode S=1: PCC={p:.8f}  max|diff|={d:.6f}  thr={PCC_DECODE_STACK}")
    assert p >= PCC_DECODE_STACK


@pytest.mark.slow
def test_denoise_step_production_32l_pcc(device):
    """32-layer denoise step at GRID=64 / S=4160 (production layout, real checkpoint weights)."""
    layout = PIPELINE_LAYOUT_PROD
    p, d, thr = _denoise_step_run(device, layout, num_layers=NUM_LAYERS_PRODUCTION)
    print(
        f"denoise step production 32L GRID={layout['grid']} S={layout['seq_len']}: "
        f"PCC={p:.8f}  max|diff|={d:.6f}  thr={thr}"
    )
    assert p >= thr


# ---------------------------------------------------------------------------
# Denoise step — mesh resident (test_pipeline_step_resident.py)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("tag,layout", DENOISE_LAYOUT_FAST)
def test_denoise_step_resident_mesh(mesh_device, tag, layout):
    mesh_device.enable_program_cache()
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    p, d, thr = _denoise_step_run(mesh_device, layout, mesh_composer=composer)
    logger.info(f"resident denoise step [{tag}] PCC={p:.6f}  thr={thr}")
    assert p >= thr


# ---------------------------------------------------------------------------
# E2E pipeline — random latent/text embeds; opt-in only (HY_RUN_E2E_RANDOM=1)
# ---------------------------------------------------------------------------
def _host_e2e_reference(e2e, c, down_sd, up_sd, init_latent, text_embeds, text_embeds_uncond=None, cfg_guidance=1.0):
    """Host denoise+VAE reference. Streams MoE layers when NUM_LAYERS>8 (resident 32L OOMs)."""
    from models.experimental.hunyuan_image_3_0.ref.image_gen.patch_embed import UNetDown as RefDown, UNetUp as RefUp
    from models.experimental.hunyuan_image_3_0.ref.image_gen.timestep_embedder import TimestepEmbedder as RefTimeEmbed

    B, GRID, S = e2e.B, e2e.GRID, e2e.S
    IMG_SLICE = e2e.IMG_SLICE
    NUM_LAYERS, STEPS, SCALING = e2e.NUM_LAYERS, e2e.STEPS, e2e.SCALING
    LATENT, HID, HSZ = patch_embed_dims(down_sd)
    H = c["H"]
    # Keys expected by denoise_helpers._make_ref_layer / _forward_ref_layers.
    c_ref = transformer_cfg()
    stream_layers = NUM_LAYERS > 8
    if stream_layers:
        clear_ref_layer_cache()

    print(
        f"[e2e ref] S={S} GRID={GRID} layers={NUM_LAYERS} steps={STEPS} "
        f"stream_layers={stream_layers} CFG={cfg_guidance}",
        flush=True,
    )

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
    cos, sin = build_batch_2d_rope(S, c["HD"], image_infos=[[(IMG_SLICE, (GRID, GRID))]])
    mask = to_additive(build_attention_mask(S, image_slices=[IMG_SLICE], bsz=B), dtype=torch.float32)

    do_cfg = text_embeds_uncond is not None and cfg_guidance != 1.0

    def _pred(lat, te_embeds, e1, e2_):
        img, th, tw = rd(lat, e1)
        h = te_embeds.clone()
        h[:, IMG_SLICE, :] = img
        h = _forward_ref_layers(c_ref, h, NUM_LAYERS, mask, cos, sin, stream_layers=stream_layers)
        return ru(h[:, IMG_SLICE, :], e2_, th, tw)

    lat = init_latent.clone()
    for i, t in enumerate(timesteps):
        print(f"[e2e ref] step {i + 1}/{STEPS} t={float(t):.4f}", flush=True)
        tv = torch.tensor([float(t)] * B)
        with torch.no_grad():
            e1, e2_ = te1r(tv), te2r(tv)
            pred = _pred(lat, text_embeds, e1, e2_)
            if do_cfg:
                pred_uncond = _pred(lat, text_embeds_uncond, e1, e2_)
                pred = pred_uncond + cfg_guidance * (pred - pred_uncond)
        lat = lat + float(sigmas[i + 1] - sigmas[i]) * pred

    if stream_layers:
        clear_ref_layer_cache()
        gc.collect()

    with torch.no_grad():
        ref_img = load_ref_decoder()((lat.float() / SCALING).unsqueeze(2))
    ref_rgb = vae_decode_output_to_rgb(ref_img)
    return lat, ref_rgb


@pytest.mark.slow
@pytest.mark.e2e_random_inputs
@pytest.mark.timeout(_E2E_TIMEOUT)
@pytest.mark.skipif(
    os.environ.get("HY_RUN_E2E_RANDOM", "0") != "1",
    reason="E2E with random latent/text inputs is opt-in; set HY_RUN_E2E_RANDOM=1",
)
def test_e2e_pipeline():
    """Integration PCC with random activations — not a module-weight gate. Opt-in only."""
    # 32L vs fp32 host: demo defaults (resident bf8 + CFG=5) drift to ~0.74 latent PCC.
    # Prefer the production densify loop setup (CFG off) unless the user overrides.
    # For tighter match set HY_WEIGHT_DTYPE=bf16 (streams experts; slower).
    if int(os.environ.get("HY_NUM_LAYERS", "2")) > 8:
        os.environ.setdefault("HY_BASE_GUIDANCE", "1.0")
    e2e = load_e2e_module()
    weight_dtype = ttnn.bfloat8_b if os.environ.get("HY_WEIGHT_DTYPE", "bf8") == "bf8" else ttnn.bfloat16

    c = e2e._cfg()
    assert e2e.S <= c["MAX_SEQ"], f"seq_len {e2e.S} exceeds max_position_embeddings {c['MAX_SEQ']}"
    print(
        f"[e2e] starting host ref then TT denoise+VAE: "
        f"S={e2e.S} GRID={e2e.GRID} layers={e2e.NUM_LAYERS} steps={e2e.STEPS} "
        f"dtype={'bf8' if weight_dtype == ttnn.bfloat8_b else 'bf16'} "
        f"BASE_GUIDANCE={e2e.BASE_GUIDANCE}",
        flush=True,
    )
    down_sd = e2e._load_prefix("patch_embed")
    up_sd = e2e._load_prefix("final_layer")
    LATENT, _, _ = patch_embed_dims(down_sd)
    assert LATENT == Z_CHANNELS

    cfg_distilled, use_meanflow = e2e._model_flags()
    base_cfg = (not (cfg_distilled or use_meanflow)) and e2e.BASE_GUIDANCE != 1.0
    cfg_guidance = e2e.BASE_GUIDANCE if base_cfg else 1.0
    latent_thr, rgb_thr = e2e_pcc_thresholds(e2e.NUM_LAYERS, e2e.STEPS, weight_dtype, cfg_guidance)

    torch.manual_seed(0)
    init_latent = torch.randn(e2e.B, LATENT, e2e.GRID, e2e.GRID)
    text_embeds = torch.randn(e2e.B, e2e.S, c["H"]) * 0.02
    text_embeds_uncond = torch.randn(e2e.B, e2e.S, c["H"]) * 0.02 if base_cfg else None

    ref_latent, ref_rgb = _host_e2e_reference(
        e2e,
        c,
        down_sd,
        up_sd,
        init_latent,
        text_embeds,
        text_embeds_uncond=text_embeds_uncond,
        cfg_guidance=cfg_guidance,
    )
    gc.collect()

    tt_latent = e2e.run_denoise(
        c,
        down_sd,
        up_sd,
        init_latent,
        text_embeds,
        text_embeds_uncond=text_embeds_uncond,
        cfg_guidance=cfg_guidance,
    )
    tt_rgb = e2e.run_vae_decode(tt_latent)

    latent_pcc, _ = pcc_metrics(ref_latent, tt_latent, latent_thr)
    rgb_pcc, _ = pcc_metrics(ref_rgb, tt_rgb, rgb_thr)
    latent_dmax = (ref_latent.float() - tt_latent.float()).abs().max().item()

    print(
        f"\n[e2e] grid={e2e.GRID} S={e2e.S} layers={e2e.NUM_LAYERS} steps={e2e.STEPS}\n"
        f"  latent PCC={latent_pcc:.6f} (thr {latent_thr})  max|Δ|={latent_dmax:.4e}\n"
        f"  RGB    PCC={rgb_pcc:.6f} (thr {rgb_thr})  "
        f"ref={tuple(ref_rgb.shape)} tt={tuple(tt_rgb.shape)}"
    )
    assert tuple(ref_rgb.shape) == tuple(tt_rgb.shape)
    assert latent_pcc >= latent_thr
    assert rgb_pcc >= rgb_thr


def test_gen_special_token_order():
    """Guards gen_special_token_indices() vs tokenizer canonical layout (demo/e2e.py)."""
    import dataclasses

    e2e = load_e2e_module()
    from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import (
        get_gen_image_slice,
        prepare_gen_image_inputs,
    )
    from models.experimental.hunyuan_image_3_0.ref.tokenizer.hunyuan_tokenizer import HunyuanTokenizer

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

    for cfg_distilled in (False, True):
        img_start, g_ts, g_guid, g_tr = canonical(cfg_distilled, False)
        ts, guid, tr, n = e2e.gen_special_token_indices(img_start, cfg_distilled, False)
        assert ts == g_ts
        assert guid == g_guid
        assert tr == g_tr

    img_start = 32
    ts, guid, tr, n = e2e.gen_special_token_indices(img_start, True, True)
    assert n == 3 and ts < guid < tr and tr == img_start - 1 and ts == img_start - 3
    ts, guid, tr, n = e2e.gen_special_token_indices(img_start, False, True)
    assert n == 2 and guid is None and ts < tr and tr == img_start - 1
