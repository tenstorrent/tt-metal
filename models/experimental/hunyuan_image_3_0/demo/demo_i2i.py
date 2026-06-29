# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end HunyuanImage-3.0 Instruct image-to-image on Tenstorrent.
#
# Full upstream flow (``bot_task=think_recaption``):
#   cond image(s) + prompt
#     → on-device AR think/recaption (``run_recaption_on_device``)
#     → ``cot_text`` injected into I2I denoise bundle
#     → denoise_loop + CFG → VAE decode → PNG
#
# Direct edit (``bot_task=image``) skips the AR recaption stage.
#
# Run (Instruct-Distil, 8-step meanflow, no CFG):
#   HY_DISTIL=1 HY_NUM_LAYERS=32 python_env/bin/python \
#     models/experimental/hunyuan_image_3_0/demo/demo_i2i.py \
#     --distil --prompt "make the sky sunset orange" --cond /path/to/image.png
#
# Multi-image:
#   --cond img1.png img2.png
#
# Auto output resolution (matches cond aspect via ratio token):
#   --image-size auto --infer-align-image-size
#
# Host recaption fallback (requires NVIDIA CUDA): HY_RECAPTION_DEVICE=0
# On TT-only systems, host recaption auto-falls back to the on-device path.

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[4]
HUNYUAN = Path(os.environ.get("HUNYUAN_UPSTREAM", "/home/iguser/tt-ign/HunyuanImage-3.0"))
for p in (str(ROOT), str(HUNYUAN)):
    if p not in sys.path:
        sys.path.insert(0, p)

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from models.experimental.hunyuan_image_3_0.ref.image_processor import HunyuanImage3ImageProcessor
from models.experimental.hunyuan_image_3_0.ref.image_gen.model_loaders import (
    load_aligner,
    load_patch_embed,
    load_siglip2_vision,
    load_timestep_embedder,
)
from dataclasses import replace

from models.experimental.hunyuan_image_3_0.ref.recaption import (
    default_recaption_sampling_config,
    run_recaption,
    system_prompt_for_bot_task,
)
from models.experimental.hunyuan_image_3_0.ref.tokenizer import (
    HunyuanTokenizer,
    build_i2i_cfg_conds,
    enrich_bundle_attention,
    prepare_i2i_denoise_bundle,
    prepare_recaption_ar_bundle,
)
from models.experimental.hunyuan_image_3_0.ref.weights import (
    INSTRUCT_DISTIL_MODEL_DIR,
    INSTRUCT_MODEL_DIR,
    load_tensors,
)
from models.experimental.hunyuan_image_3_0.tt.denoise_cond import upload_denoise_cond
from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown, HunyuanTtUNetUp
from models.experimental.hunyuan_image_3_0.tt.image_gen.timestep_embedder import HunyuanTtTimestepEmbedder
from models.experimental.hunyuan_image_3_0.tt.lm_head import HunyuanTtLMHead
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel
from models.experimental.hunyuan_image_3_0.tt.pipeline import HunyuanTtDenoiseStep, decode_latent, denoise_loop
from models.experimental.hunyuan_image_3_0.tt.recaption import run_recaption_on_device
from models.experimental.hunyuan_image_3_0.tt.scheduler import HunyuanTtScheduler

from models.experimental.hunyuan_image_3_0.ref.system_prompt import get_system_prompt

STEPS_ENV = os.environ.get("HY_STEPS")
NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "32"))
RECAPTION_LAYERS = int(os.environ.get("HY_RECAPTION_LAYERS", str(NUM_LAYERS)))
RECAPTION_KV = os.environ.get("HY_RECAPTION_KV", "1") != "0"
GUIDANCE = float(os.environ.get("HY_GUIDANCE", "2.5"))
SEED = int(os.environ.get("HY_SEED", "42"))
VIT_LAYERS = int(os.environ.get("HY_VIT_LAYERS", "27"))
MAX_NEW_TOKENS = int(os.environ.get("HY_MAX_NEW_TOKENS", "512"))
RECAPTION_ON_DEVICE = os.environ.get("HY_RECAPTION_DEVICE", "1") != "0"
USE_DISTIL = os.environ.get("HY_DISTIL", "0") == "1"
SCALING = 0.562679178327931


def _resolve_model_dir(distil: bool) -> Path:
    if distil:
        return INSTRUCT_DISTIL_MODEL_DIR
    return INSTRUCT_MODEL_DIR


def _default_steps(model_dir: Path, *, distil: bool) -> int:
    gen_cfg = model_dir / "generation_config.json"
    if gen_cfg.is_file():
        return int(json.load(open(gen_cfg)).get("diff_infer_steps", 8 if distil else 50))
    return 8 if distil else 50


def _model_flags(model_dir: Path) -> tuple[bool, bool]:
    cfg = json.load(open(model_dir / "config.json"))
    return bool(cfg.get("cfg_distilled", False)), bool(cfg.get("use_meanflow", False))


class _WeightLoader:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._wmap = json.load(open(model_dir / "model.safetensors.index.json"))["weight_map"]
        self._open: dict = {}

    def load(self, key):
        shard = self._wmap[key]
        f = self._open.get(shard) or self._open.setdefault(shard, safe_open(self.model_dir / shard, framework="pt"))
        return f.get_tensor(key)

    def load_prefix(self, prefix):
        return {k[len(prefix) + 1 :]: self.load(k) for k in self._wmap if k.startswith(prefix + ".")}


def _use_tt_recaption() -> bool:
    """Run AR recaption on Tenstorrent unless host CUDA recaption was explicitly requested."""
    if RECAPTION_ON_DEVICE:
        return True
    if not torch.cuda.is_available():
        print(
            "[demo_i2i] HY_RECAPTION_DEVICE=0 needs CUDA for host recaption; "
            "no NVIDIA GPU found — using Tenstorrent device instead",
            flush=True,
        )
        return True
    return False


def _cfg(model_dir: Path):
    c = json.load(open(model_dir / "config.json"))
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
    )


def _pe_dims(down_sd):
    hid, latent = down_sd["model.0.weight"].shape[:2]
    hsz = down_sd["model.1.in_layers.2.weight"].shape[0]
    return int(latent), int(hid), int(hsz)


def _parse_image_size(raw: str) -> str | int:
    if raw == "auto":
        return "auto"
    try:
        return int(raw)
    except ValueError:
        return raw


def _load_cond_images(proc, paths: list[str], *, infer_align: bool) -> tuple[list, list[Image.Image]]:
    random_crop = "resize" if infer_align else "center"
    cond_list = []
    pil_list = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        pil_list.append(img)
        cond, _ = proc.get_image_with_size(img, return_type="vae_vit", random_crop=random_crop)
        cond_list.append(cond)
    return cond_list, pil_list


def _build_backbone(
    mesh_device, ccl, c, weights: _WeightLoader, *, num_layers: int, apply_final_norm: bool, sp_factor: int = 2
):
    layer_loader = lambda i: {f"model.layers.{i}.{k}": v for k, v in weights.load_prefix(f"model.layers.{i}").items()}
    bf16_layers = {0, 1, 2, 3, num_layers - 4, num_layers - 3, num_layers - 2, num_layers - 1}
    norm_sd = {"model.ln_f.weight": weights.load("model.ln_f.weight")} if apply_final_norm else None
    embed_sd = {"model.wte.weight": weights.load("model.wte.weight")}
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
        tp_axis=1,
        tp_factor=2,
        sp_axis=0,
        sp_factor=sp_factor,
        bf16_layers=bf16_layers,
    )


def _run_recaption_host(model_dir, weights, prompt, bot_task, system_prompt, pil_images, image_size):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Host recaption requires an NVIDIA GPU (CUDA). "
            "On Tenstorrent-only systems omit HY_RECAPTION_DEVICE=0 or set HY_RECAPTION_DEVICE=1."
        )
    from transformers import AutoModelForCausalLM

    kwargs = dict(
        attn_implementation="sdpa",
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
        moe_impl="eager",
        moe_drop_tokens=True,
    )
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
    model.load_tokenizer(str(model_dir))
    image_arg = pil_images[0] if len(pil_images) == 1 else pil_images
    result = run_recaption(
        model,
        prompt,
        bot_task=bot_task,
        system_prompt=system_prompt,
        image=image_arg,
        image_size=image_size,
        seed=SEED,
        max_new_tokens=MAX_NEW_TOKENS,
        verbose=1,
    )
    return result.cot_text[0], result.image_size


def main():
    parser = argparse.ArgumentParser(description="HunyuanImage-3.0 Instruct I2I on Tenstorrent")
    parser.add_argument("--prompt", default=os.environ.get("HY_PROMPT", "make the sky more dramatic"))
    parser.add_argument(
        "--cond",
        nargs="+",
        default=[p for p in os.environ.get("HY_COND", "").split(",") if p],
        help="One or more conditional RGB image paths",
    )
    parser.add_argument(
        "--bot-task",
        default=os.environ.get("HY_BOT_TASK", "think_recaption"),
        choices=("image", "recaption", "think_recaption"),
    )
    parser.add_argument("--image-size", default=os.environ.get("HY_IMAGE_SIZE", "1024"))
    parser.add_argument(
        "--infer-align-image-size",
        action="store_true",
        default=os.environ.get("HY_INFER_ALIGN", "0") == "1",
    )
    parser.add_argument(
        "--distil",
        action="store_true",
        default=USE_DISTIL,
        help="Use HunyuanImage-3.0-Instruct-Distil (8-step meanflow, cfg_distilled)",
    )
    parser.add_argument("--out", default=os.environ.get("HY_OUT", str(ROOT / "hy_i2i.png")))
    args = parser.parse_args()

    model_dir = _resolve_model_dir(args.distil)
    if not (model_dir / "model.safetensors.index.json").is_file():
        label = "Instruct-Distil" if args.distil else "Instruct"
        raise SystemExit(f"{label} weights not found under {model_dir}")

    cfg_distilled, use_meanflow = _model_flags(model_dir)
    if args.distil and not cfg_distilled:
        print(f"[demo_i2i] warning: --distil but config.json has cfg_distilled=false ({model_dir})")

    steps = int(STEPS_ENV) if STEPS_ENV is not None else _default_steps(model_dir, distil=args.distil)
    weights = _WeightLoader(model_dir)

    if not args.cond:
        raise SystemExit("Provide --cond /path/to/image.png or set HY_COND")

    image_size = _parse_image_size(args.image_size)
    cot_text = None

    variant = "Instruct-Distil" if args.distil or cfg_distilled else "Instruct"
    print(
        f"[demo_i2i] model={model_dir}  variant={variant}  bot_task={args.bot_task}  "
        f"image_size={image_size}  steps={steps}  cond={args.cond!r}"
    )

    c = _cfg(model_dir)
    H = c["H"]
    down_sd, up_sd = weights.load_prefix("patch_embed"), weights.load_prefix("final_layer")
    LATENT, HID, HSZ = _pe_dims(down_sd)

    proc = HunyuanImage3ImageProcessor(json.load(open(model_dir / "config.json")))
    cond_list, pil_list = _load_cond_images(proc, args.cond, infer_align=args.infer_align_image_size)
    cond_for_bundle = cond_list if len(cond_list) > 1 else cond_list[0]

    tok = HunyuanTokenizer.from_model_dir(model_dir, sequence_template="instruct")
    wte = load_tensors(model_dir, ["model.wte.weight"])["model.wte.weight"]
    gen = torch.Generator().manual_seed(SEED)

    patch_embed_ref = load_patch_embed(model_dir)
    time_embed_ref = load_timestep_embedder("time_embed", model_dir)
    timestep_emb_ref = load_timestep_embedder("timestep_emb", model_dir)
    vision_ref = load_siglip2_vision(model_dir, num_layers=VIT_LAYERS)
    aligner_ref = load_aligner(model_dir)

    if args.bot_task != "image":
        sp_key, sp_sub = system_prompt_for_bot_task(args.bot_task)
        recap_system = get_system_prompt(sp_key, sp_sub)
        print(f"[demo_i2i] recaption stage (bot_task={args.bot_task}) ...")

        if _use_tt_recaption():
            print("[demo_i2i] opening recaption mesh (2x2) ...", flush=True)
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            recap_mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2), l1_small_size=32768)
            try:
                recap_mesh.enable_program_cache()
                recap_ccl = CCLManager(recap_mesh, num_links=1, topology=ttnn.Topology.Linear)
                print("[demo_i2i] building recaption AR bundle (host VAE/ViT encode) ...", flush=True)

                def rep(t):
                    return ttnn.from_torch(
                        t,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=recap_mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(recap_mesh),
                    )

                recap_bundle = prepare_recaption_ar_bundle(
                    tok,
                    args.prompt,
                    proc,
                    wte,
                    cond_images=cond_for_bundle,
                    bot_task=args.bot_task,
                    system_prompt=recap_system,
                    sequence_template="instruct",
                    patch_embed=patch_embed_ref,
                    time_embed=time_embed_ref,
                    timestep_emb=timestep_emb_ref,
                    vision_model=vision_ref,
                    aligner=aligner_ref,
                    model_dir=model_dir,
                    generator=gen,
                )
                prefix_len = int(recap_bundle.input_ids.shape[1])
                print(
                    f"[demo_i2i] recaption prefix_len={prefix_len} layers={RECAPTION_LAYERS} "
                    f"max_new_tokens={MAX_NEW_TOKENS} kv_cache={RECAPTION_KV}",
                    flush=True,
                )
                t0 = time.time()
                recap_sp = 1 if RECAPTION_KV else 2
                print(
                    f"[demo_i2i] loading recaption backbone ({RECAPTION_LAYERS} layers, sp={recap_sp}) ...", flush=True
                )
                recap_backbone = _build_backbone(
                    recap_mesh,
                    recap_ccl,
                    c,
                    weights,
                    num_layers=RECAPTION_LAYERS,
                    apply_final_norm=True,
                    sp_factor=recap_sp,
                )
                print(f"[demo_i2i] recaption backbone ready ({time.time() - t0:.0f}s)", flush=True)
                print("[demo_i2i] loading LM head ...", flush=True)
                lm_head = HunyuanTtLMHead(recap_mesh, {"lm_head.weight": weights.load("lm_head.weight")})
                recap_config = replace(default_recaption_sampling_config(), max_new_tokens=MAX_NEW_TOKENS)
                recap_result = run_recaption_on_device(
                    recap_backbone,
                    lm_head,
                    recap_mesh,
                    recap_bundle,
                    tok,
                    args.bot_task,
                    proc,
                    wte,
                    image_size=image_size,
                    config=recap_config,
                    generator=gen,
                    replicate_to_mesh=rep,
                )
                cot_text = recap_result.cot_text[0]
                if recap_result.image_size != image_size:
                    image_size = recap_result.image_size
            finally:
                ttnn.close_mesh_device(recap_mesh)
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        else:
            cot_text, image_size = _run_recaption_host(
                model_dir, weights, args.prompt, args.bot_task, recap_system, pil_list, image_size
            )

        print(f"[demo_i2i] cot_text:\n{cot_text}\n[demo_i2i] resolved image_size={image_size}")

    denoise_system = get_system_prompt("en_unified", "image")
    print("[demo_i2i] building I2I denoise bundle ...")
    bundle = prepare_i2i_denoise_bundle(
        tok,
        args.prompt,
        cond_for_bundle,
        proc,
        wte,
        patch_embed=patch_embed_ref,
        time_embed=time_embed_ref,
        timestep_emb=timestep_emb_ref,
        vision_model=vision_ref,
        aligner=aligner_ref,
        image_size=image_size,
        sequence_template="instruct",
        system_prompt=denoise_system,
        cot_text=cot_text,
        generator=gen,
    )
    cond_row, uncond_row = build_i2i_cfg_conds(bundle, wte, proc)
    enrich_bundle_attention(bundle, proc)
    attn_spans = bundle.full_attn_slices[0]
    img_slice = cond_row["gen_slice"]
    grid = cond_row["gen_hw"]
    seq_len = bundle.seq_len
    print(f"[demo_i2i] seq_len={seq_len} gen_span={img_slice} grid={grid}")

    print("[demo_i2i] opening denoise mesh (2x2) ...", flush=True)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 2), l1_small_size=32768)
    try:
        mesh_device.enable_program_cache()
        ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
        print("[demo_i2i] building patch_embed + final_layer ...", flush=True)

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
        print(f"[demo_i2i] building denoise backbone ({NUM_LAYERS} layers) ...", flush=True)
        t0 = time.time()
        backbone = _build_backbone(mesh_device, ccl, c, weights, num_layers=NUM_LAYERS, apply_final_norm=False)
        print(f"[demo_i2i] denoise backbone ready ({time.time() - t0:.0f}s)", flush=True)
        print("[demo_i2i] building timestep embedders ...", flush=True)
        te1 = HunyuanTtTimestepEmbedder(
            mesh_device,
            H,
            {f"time_embed.{k}": v for k, v in weights.load_prefix("time_embed").items()},
            "time_embed",
        )
        te2 = HunyuanTtTimestepEmbedder(
            mesh_device,
            H,
            {f"time_embed_2.{k}": v for k, v in weights.load_prefix("time_embed_2").items()},
            "time_embed_2",
        )
        tt_timestep_emb = HunyuanTtTimestepEmbedder(
            mesh_device,
            H,
            {f"timestep_emb.{k}": v for k, v in weights.load_prefix("timestep_emb").items()},
            "timestep_emb",
        )
        tt_guidance_emb = (
            HunyuanTtTimestepEmbedder(
                mesh_device,
                H,
                {f"guidance_emb.{k}": v for k, v in weights.load_prefix("guidance_emb").items()},
                "guidance_emb",
            )
            if cfg_distilled
            else None
        )
        tt_timestep_r_emb = (
            HunyuanTtTimestepEmbedder(
                mesh_device,
                H,
                {f"timestep_r_emb.{k}": v for k, v in weights.load_prefix("timestep_r_emb").items()},
                "timestep_r_emb",
            )
            if use_meanflow
            else None
        )
        step = HunyuanTtDenoiseStep(
            mesh_device,
            patch_embed=patch_embed,
            backbone=backbone,
            final_layer=final_layer,
            img_slice=img_slice,
            grid_hw=grid,
            seq_len=seq_len,
        )

        print("[demo_i2i] uploading cond/uncond to device (base_embeds + TT mask) ...", flush=True)
        cond_tt = upload_denoise_cond(mesh_device, cond_row, replicate_fn=rep, seq_len=seq_len, attn_spans=attn_spans)
        uncond_tt = (
            upload_denoise_cond(mesh_device, uncond_row, replicate_fn=rep, seq_len=seq_len, attn_spans=attn_spans)
            if uncond_row is not None
            else None
        )

        torch.manual_seed(SEED + 1)
        init_latent = torch.randn(1, LATENT, grid[0], grid[1])
        sched = HunyuanTtScheduler(mesh_device)
        sched.set_timesteps(steps)
        mode = "distil" if cfg_distilled else "CFG"
        print(
            f"[demo_i2i] denoising {steps} steps ({mode}, guidance={GUIDANCE}, seq_len={seq_len}) ...",
            flush=True,
        )
        latent = denoise_loop(
            step,
            sched,
            init_latent,
            time_embed=te1,
            time_embed_2=te2,
            cond=cond_tt,
            uncond=uncond_tt,
            guidance_scale=GUIDANCE,
            tt_timestep_emb=tt_timestep_emb,
            tt_guidance_emb=tt_guidance_emb,
            tt_timestep_r_emb=tt_timestep_r_emb,
            cfg_distilled=cfg_distilled,
            use_meanflow=use_meanflow,
            mesh_device=mesh_device,
        )
        print(f"[demo_i2i] denoised latent {tuple(latent.shape)}")
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    print("[demo_i2i] VAE decode (TTNN spatial) ...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    vae_mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))
    try:
        vae_mesh.enable_program_cache()
        vae_ccl = CCLManager(vae_mesh, num_links=1, topology=ttnn.Topology.Linear)
        img = decode_latent(
            vae_mesh,
            latent,
            scaling_factor=SCALING,
            ccl_manager=vae_ccl,
            h_mesh_axis=0,
            w_mesh_axis=1,
        )
    finally:
        ttnn.close_mesh_device(vae_mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    arr = (img[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    Image.fromarray(arr).save(args.out)
    print(f"[demo_i2i] saved -> {args.out}")


if __name__ == "__main__":
    main()
