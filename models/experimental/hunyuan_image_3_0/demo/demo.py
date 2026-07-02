# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end HunyuanImage-3.0 text-to-image on Tenstorrent.
#
#   prompt --[ optional AR recaption: text-sampling loop on the resident backbone +
#              LM head -> <recaption>...</recaption> cot_text ]
#          --HunyuanTokenizer (cot_text injected as the assistant turn)--> input_ids
#          --HunyuanTtWte--> text embeddings (cond / uncond rows for CFG)
#   noise latent --[ patch_embed -> RESIDENT bf8 sharded backbone -> final_layer ]
#                  x N scheduler steps with CFG (denoise_loop on the 2x2 sp0tp1 mesh)
#          --> denoised latent
#          --VAE decode (TTNN, on device, 2x2 H/W-spatial-parallel)--> RGB image
#
# Optional AR recaption (HY_RECAPTION=1): rewrites the prompt via the text-sampling loop
# (ref/generate.py / tt/generate.py) before image gen. Upstream base uses bot_task=image
# and skips recaption by default; enable only when you want recaption/think explicitly.
#
# The whole pipeline runs on the TT mesh: the backbone with the model that fits
# DRAM (bf8 + 4-way expert sharding, first/last 3 layers bf16), and the VAE decode
# sharded H/W-spatial across the 2x2 mesh (each device a 512x512 quadrant of the
# 1024x1024 image; validated end-to-end vs the fp32 reference — see
# MEMORY_FIT_PLAN.md). No host round-trip for the VAE.
#
# Run:
#   HY_STEPS=8 HY_NUM_LAYERS=32 python_env/bin/python \
#     models/experimental/hunyuan_image_3_0/demo/demo.py "a photo of a cat"
#
# Optional recaption (off by default; slow — full forward per AR token):
#   HY_RECAPTION=1 HY_BOT_TASK=recaption HY_STEPS=8 python_env/bin/python \
#     models/experimental/hunyuan_image_3_0/demo/demo.py "a photo of a cat"

import os, sys, json, time
from pathlib import Path
import torch
from safetensors import safe_open

ROOT = str(Path(__file__).resolve().parents[4])  # tt-metal repo root (robust to checkout location)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.experimental.hunyuan_image_3_0.ref.weights import ensure_base_weights

WEIGHTS = ensure_base_weights()
HUNYUAN = str(WEIGHTS)
if HUNYUAN not in sys.path:
    sys.path.insert(0, HUNYUAN)

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer
from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import (
    prepare_gen_image_inputs,
    prepare_recaption_ar_bundle,
)
from models.experimental.hunyuan_image_3_0.ref.generate import SamplingConfig
from models.experimental.hunyuan_image_3_0.ref.image_processor import HunyuanImage3ImageProcessor
from models.experimental.hunyuan_image_3_0.ref.recaption import (
    default_recaption_sampling_config,
    system_prompt_for_bot_task,
)
from models.experimental.hunyuan_image_3_0.ref.system_prompt import get_system_prompt
from models.experimental.hunyuan_image_3_0.tt.attention.mask import build_attention_mask_tt
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel, default_bf16_layers
from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown, HunyuanTtUNetUp
from models.experimental.hunyuan_image_3_0.tt.image_gen.timestep_embedder import HunyuanTtTimestepEmbedder
from models.experimental.hunyuan_image_3_0.tt.lm_head import HunyuanTtLMHead
from models.experimental.hunyuan_image_3_0.tt.pipeline import HunyuanTtDenoiseStep, denoise_loop, decode_latent
from models.experimental.hunyuan_image_3_0.tt.recaption import run_recaption_on_device
from models.experimental.hunyuan_image_3_0.tt.scheduler import HunyuanTtScheduler
from models.experimental.hunyuan_image_3_0.tt.wte import HunyuanTtWte

PROMPT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HY_PROMPT", "a photo of a cat, studio lighting")
STEPS = int(os.environ.get("HY_STEPS", "8"))
NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "32"))
GUIDANCE = float(os.environ.get("HY_GUIDANCE", "5.0"))
SEED = int(os.environ.get("HY_SEED", "0"))
# Image side in pixels; grid = IMAGE_SIZE / 16 (vae_downsample_factor), and the
# backbone seq_len = text + grid^2. Raise this to scale seq_len toward the config's
# max_position_embeddings (the run asserts the cap). Default 1024 -> 64x64 grid.
# When set, overrides the recaption-resolved size. The VAE is rebuilt for the grid.
IMAGE_SIZE = int(os.environ["HY_IMAGE_SIZE"]) if os.environ.get("HY_IMAGE_SIZE") else None
SCALING = 0.562679178327931
_DEMO_DIR = Path(__file__).resolve().parent
_PKG_DIR = _DEMO_DIR.parent
OUT_PNG = os.environ.get("HY_OUT", str(_PKG_DIR / "output.png"))

# AR recaption (text-sampling loop) — off by default (upstream base bot_task=image).
RECAPTION = os.environ.get("HY_RECAPTION", "0") != "0"
BOT_TASK = os.environ.get("HY_BOT_TASK", "recaption")  # recaption | think | think_recaption
RECAPTION_LAYERS = int(os.environ.get("HY_RECAPTION_LAYERS", str(NUM_LAYERS)))
MAX_NEW_TOKENS = int(os.environ.get("HY_MAX_NEW_TOKENS", "512"))
# Sampling knobs (defaults match Instruct generation_config: temp 0.6 / top-k 1024 / top-p 0.95).
_SAMPLE_DEFAULTS = default_recaption_sampling_config()
DO_SAMPLE = os.environ.get("HY_DO_SAMPLE", "1") != "0"
TEMPERATURE = float(os.environ.get("HY_TEMPERATURE", str(_SAMPLE_DEFAULTS.temperature)))
TOP_K = int(os.environ.get("HY_TOP_K", str(_SAMPLE_DEFAULTS.top_k)))
TOP_P = float(os.environ.get("HY_TOP_P", str(_SAMPLE_DEFAULTS.top_p)))
REP_PENALTY = float(os.environ.get("HY_REP_PENALTY", str(_SAMPLE_DEFAULTS.repetition_penalty)))

_INDEX = WEIGHTS / "model.safetensors.index.json"
# Persist pre-tilized mesh weights across runs (see README § Weight cache).
if not os.environ.get("TT_DIT_CACHE_DIR"):
    os.environ["TT_DIT_CACHE_DIR"] = str(Path.home() / ".cache" / "tt-dit")
if not _INDEX.is_file():
    raise SystemExit(
        f"Base HunyuanImage-3 weights not found at {_INDEX}\n"
        f"Download with: hf download tencent/HunyuanImage-3.0\n"
        f"Or set HUNYUAN_MODEL_DIR to a checkpoint dir containing model.safetensors.index.json"
    )
_WMAP = json.load(open(_INDEX))["weight_map"]
_OPEN = {}

# --- lightweight per-stage timing -------------------------------------------
_TIMINGS = []


def _mark(name, since):
    """Record elapsed since `since`, print it, and return a fresh timestamp."""
    dt = time.time() - since
    _TIMINGS.append((name, dt))
    print(f"[timing] {name}: {dt:.1f}s", flush=True)
    return time.time()


def _print_timing_summary(total):
    print("\n==================== T2I TIMING SUMMARY ====================", flush=True)
    acc = 0.0
    for name, dt in _TIMINGS:
        acc += dt
        print(f"[timing] {name:34s} {dt:8.1f}s  ({100 * dt / total:5.1f}%)", flush=True)
    other = total - acc
    if abs(other) > 0.5:
        print(f"[timing] {'(unaccounted)':34s} {other:8.1f}s  ({100 * other / total:5.1f}%)", flush=True)
    print(f"[timing] {'TOTAL':34s} {total:8.1f}s", flush=True)
    print("============================================================", flush=True)


def _load(key):
    shard = _WMAP[key]
    f = _OPEN.get(shard) or _OPEN.setdefault(shard, safe_open(WEIGHTS / shard, framework="pt"))
    return f.get_tensor(key)


def _load_prefix(prefix):
    return {k[len(prefix) + 1 :]: _load(k) for k in _WMAP if k.startswith(prefix + ".")}


def _cfg():
    c = json.load(open(WEIGHTS / "config.json"))
    first = lambda v: v if isinstance(v, int) else v[0]
    return dict(
        H=c["hidden_size"],
        HEADS=c["num_attention_heads"],
        KV=c.get("num_key_value_heads", c["num_attention_heads"]),
        HD=c.get("attention_head_dim", c["hidden_size"] // c["num_attention_heads"]),
        E=first(c["num_experts"]),
        K=first(c["moe_topk"]),
        NORM=c.get("norm_topk_prob", True),
        MIXED=c.get("use_mixed_mlp_moe", True),
        QKN=c.get("use_qk_norm", True),
        EPS=c.get("rms_norm_eps", 1e-5),
        MAX_SEQ=int(c["max_position_embeddings"]),
    )


def _pe_dims(down_sd):
    hid, latent = down_sd["model.0.weight"].shape[:2]
    hsz = down_sd["model.1.in_layers.2.weight"].shape[0]
    return int(latent), int(hid), int(hsz)


def _parse_bf16_layers(num_layers: int) -> set[int]:
    if os.environ.get("HY_BF16_LAYERS"):
        return {int(s) for s in os.environ["HY_BF16_LAYERS"].split(",") if s.strip() != ""}
    return default_bf16_layers(num_layers)


def _build_backbone(mesh_device, ccl, c, *, num_layers, apply_final_norm, embed_sd=None, norm_sd=None):
    """Resident bf8 backbone on the 2x2 mesh (4-way expert shard, first/last 3 layers bf16).

    ``apply_final_norm=True`` + ``embed_sd``/``norm_sd`` builds the LM-backbone variant
    (device wte embed + ln_f) the AR recaption stage needs; the denoise path passes
    ``apply_final_norm=False`` and feeds pre-embedded hidden states instead.
    """
    layer_loader = lambda i: {f"model.layers.{i}.{k}": v for k, v in _load_prefix(f"model.layers.{i}").items()}
    bf16_layers = _parse_bf16_layers(num_layers)
    print(f"[demo] building resident backbone ({num_layers} layers, bf8 + bf16 layers {sorted(bf16_layers)}) ...")
    return HunyuanTtModel(
        mesh_device,
        num_layers=num_layers,
        hidden_size=c["H"],
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
        embed_state_dict=embed_sd,
        norm_state_dict=norm_sd,
        apply_final_norm=apply_final_norm,
        weight_dtype=ttnn.bfloat8_b,
        ccl_manager=ccl,
        expert_mesh_axis=1,
        tp_axis=1,  # TP=2 on axis 1: column-parallel qkv, row-parallel o_proj
        tp_factor=2,
        sp_axis=0,  # SP=2 on axis 0: sequence sharded across rows (gather-KV attn)
        sp_factor=2,
        bf16_layers=bf16_layers,
        model_cache_name="hunyuan-image-3.0",
    )


def _run_recaption(c, tok, proc, wte, prompt, generator):
    """AR recaption: rewrite the prompt with the text-sampling loop on a resident backbone.

    Opens its own 2x2 mesh (LM backbone + head are freed before the denoise backbone is
    built, mirroring demo_i2i.py). Returns ``(cot_text, image_size)``.
    """
    sp_key, sp_sub = system_prompt_for_bot_task(BOT_TASK)
    recap_system = get_system_prompt(sp_key, sp_sub)
    recap_bundle = prepare_recaption_ar_bundle(
        tok,
        prompt,
        proc,
        wte,
        bot_task=BOT_TASK,
        system_prompt=recap_system,
        sequence_template="pretrain",
        generator=generator,
    )
    prefix_len = int(recap_bundle.input_ids.shape[1])
    config = SamplingConfig(
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=REP_PENALTY,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    print(
        f"[demo] recaption stage (bot_task={BOT_TASK}, system={sp_key}/{sp_sub}) "
        f"prefix_len={prefix_len} layers={RECAPTION_LAYERS} max_new_tokens={MAX_NEW_TOKENS} "
        f"sample(do_sample={DO_SAMPLE}, temp={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P}, rep={REP_PENALTY})",
        flush=True,
    )

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    recap_mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2), l1_small_size=32768)
    try:
        recap_mesh.enable_program_cache()
        recap_ccl = CCLManager(recap_mesh, num_links=1, topology=ttnn.Topology.Linear)

        def rep(t):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=recap_mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(recap_mesh),
            )

        t0 = time.time()
        recap_backbone = _build_backbone(
            recap_mesh,
            recap_ccl,
            c,
            num_layers=RECAPTION_LAYERS,
            apply_final_norm=True,
            embed_sd={"model.wte.weight": wte},
            norm_sd={"model.ln_f.weight": _load("model.ln_f.weight")},
        )
        print(f"[demo] recaption backbone ready ({time.time() - t0:.0f}s); loading LM head ...", flush=True)
        lm_head = HunyuanTtLMHead(recap_mesh, {"lm_head.weight": _load("lm_head.weight")})
        recap_result = run_recaption_on_device(
            recap_backbone,
            lm_head,
            recap_mesh,
            recap_bundle,
            tok,
            BOT_TASK,
            proc,
            wte,
            image_size=1024,
            config=config,
            generator=generator,
            replicate_to_mesh=rep,
        )
        return recap_result.cot_text[0], recap_result.image_size
    finally:
        ttnn.close_mesh_device(recap_mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    t_start = time.time()
    t = t_start
    print(f"[demo] prompt={PROMPT!r}  steps={STEPS}  layers={NUM_LAYERS}  guidance={GUIDANCE}  recaption={RECAPTION}")
    c = _cfg()
    H = c["H"]
    down_sd, up_sd = _load_prefix("patch_embed"), _load_prefix("final_layer")
    LATENT, HID, HSZ = _pe_dims(down_sd)

    tok = HunyuanTokenizer.from_pretrained()
    wte = _load("model.wte.weight").float()
    proc = HunyuanImage3ImageProcessor(json.load(open(WEIGHTS / "config.json")))
    generator = torch.Generator().manual_seed(SEED)
    t = _mark("1_setup_weights_tokenizer", t)

    # 0) optional AR recaption: rewrite the prompt with the text-sampling loop
    #    (temperature/top-k/top-p/repetition penalty) on a resident backbone + LM head.
    cot_text, image_size = None, 1024
    if RECAPTION:
        cot_text, image_size = _run_recaption(c, tok, proc, wte, PROMPT, generator)
        print(f"[demo] recaption cot_text:\n{cot_text}\n[demo] resolved image_size={image_size}")
        t = _mark("2_recaption_ar", t)
    if IMAGE_SIZE is not None:  # explicit HY_IMAGE_SIZE overrides the resolved size
        image_size = IMAGE_SIZE
        print(f"[demo] HY_IMAGE_SIZE override -> image_size={image_size}")

    # 1) tokenize -> input_ids (cond row 0, uncond row 1) + contiguous image span.
    #    cot_text (when present) is injected as the assistant turn before the gen block.
    bundle = prepare_gen_image_inputs(tok, PROMPT, image_size=image_size, cot_text=cot_text)
    ids = bundle.input_ids  # [2, S]
    S = bundle.seq_len
    span = bundle.rope_image_info[0][0][0]
    grid = bundle.rope_image_info[0][0][1]  # (64, 64)
    print(f"[demo] seq_len={S} image_span={span} grid={grid}  (max_position_embeddings={c['MAX_SEQ']})")
    assert S <= c["MAX_SEQ"], f"seq_len {S} (image_size {image_size}) exceeds max_position_embeddings {c['MAX_SEQ']}"

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    # 2x2 mesh (full QB2): SP=axis0 / TP=axis1 layout. Phase 1 wires 4-way expert
    # parallelism across BOTH axes (16 experts/device, ~19GB bf8) so the 80GB model
    # fits; attention/dense stay replicated until the TP and SP phases land.
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

        # 2) text embeddings (on-device HunyuanTtWte) for cond/uncond rows.
        wte_tt = HunyuanTtWte(
            mesh_device,
            wte,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        emb = wte_tt.embedding_torch(ids)  # [2, S, H]

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
        backbone = _build_backbone(mesh_device, ccl, c, num_layers=NUM_LAYERS, apply_final_norm=False)
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
            img_slice=span,
            grid_hw=grid,
            seq_len=S,
        )

        # Attention mask built entirely on device (TTNN ops, replicated across the mesh)
        # — no host torch build + upload. Causal text + bidirectional image span.
        mask_tt = build_attention_mask_tt(mesh_device, S, image_slices=[span], bsz=1, dtype=ttnn.bfloat16)
        image_infos = [[(span, grid)]]

        def cond_dict(row):
            return dict(
                text_pre=rep(emb[row : row + 1, : span.start, :]),
                text_post=rep(emb[row : row + 1, span.stop :, :]),
                image_infos=image_infos,
                attention_mask=mask_tt,
                batch=1,
            )

        cond, uncond = cond_dict(0), cond_dict(1)

        torch.manual_seed(SEED)
        init_latent = torch.randn(1, LATENT, grid[0], grid[1])

        sched = HunyuanTtScheduler(mesh_device)
        sched.set_timesteps(STEPS)
        t = _mark("4_build_denoise_mesh_backbone", t)
        print(f"[demo] denoising {STEPS} steps (CFG={GUIDANCE}) on resident backbone ...")
        latent = denoise_loop(
            step,
            sched,
            init_latent,
            time_embed=te1,
            time_embed_2=te2,
            cond=cond,
            uncond=uncond,
            guidance_scale=GUIDANCE,
            mesh_device=mesh_device,
        )
        print(f"[demo] denoised latent {tuple(latent.shape)}  (finite={bool(torch.isfinite(latent).all())})")
        t = _mark("5_denoise_loop", t)
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    # VAE decode on device (TTNN), H/W-spatial-parallel on the 2x2 mesh: H->axis0,
    # W->axis1. Convs keep a neighbor-pad halo; GroupNorm/attention gather to full
    # spatial. Shrinks the conv im2col 4x vs the replicated path.
    print("[demo] VAE decode (TTNN, on device, 2x2 H/W-spatial-parallel) ...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    vae_mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))
    try:
        vae_mesh.enable_program_cache()
        vae_ccl = CCLManager(vae_mesh, num_links=1, topology=ttnn.Topology.Linear)
        img = decode_latent(
            vae_mesh,
            latent,
            scaling_factor=SCALING,
            grid_hw=grid,  # rebuild the VAE for the actual latent grid (1024 -> 64x64)
            ccl_manager=vae_ccl,
            h_mesh_axis=0,
            w_mesh_axis=1,
        )  # [1, 3, grid*16, grid*16] in [0,1]
    finally:
        ttnn.close_mesh_device(vae_mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    t = _mark("6_vae_decode", t)
    img = img[0]  # [3, 1024, 1024]

    from PIL import Image

    arr = (img.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    out_path = Path(OUT_PNG)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)
    print(f"[demo] saved image -> {out_path}")
    t = _mark("7_save_png", t)
    _print_timing_summary(time.time() - t_start)


if __name__ == "__main__":
    main()
