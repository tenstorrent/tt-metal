# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end HunyuanImage-3.0 image-to-image (Instruct edit) on Tenstorrent.
#
#   cond image + prompt --prepare_i2i_denoise_bundle--> inputs_embeds + multi-span mask
#   noise latent --[ patch_embed -> resident backbone -> final_layer ] x N steps + CFG
#   --> denoised latent --VAE decode--> RGB image
#
# Run:
#   HY_STEPS=8 HY_NUM_LAYERS=32 python_env/bin/python \
#     models/experimental/hunyuan_image_3_0/demo/demo_i2i.py \
#     --prompt "make the sky sunset orange" --cond /path/to/image.png

from __future__ import annotations

import argparse
import json
import os
import sys
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
from models.experimental.hunyuan_image_3_0.ref.tokenizer import (
    HunyuanTokenizer,
    build_i2i_cfg_conds,
    prepare_i2i_denoise_bundle,
)
from models.experimental.hunyuan_image_3_0.ref.weights import (
    INSTRUCT_MODEL_DIR,
    load_tensors,
)
from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown, HunyuanTtUNetUp
from models.experimental.hunyuan_image_3_0.tt.image_gen.timestep_embedder import HunyuanTtTimestepEmbedder
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel
from models.experimental.hunyuan_image_3_0.tt.pipeline import HunyuanTtDenoiseStep, decode_latent, denoise_loop
from models.experimental.hunyuan_image_3_0.tt.scheduler import HunyuanTtScheduler
from models.experimental.hunyuan_image_3_0.tests.pcc.i2i_helpers import ref_timestep_emb

from hunyuan_image_3.system_prompt import get_system_prompt

STEPS = int(os.environ.get("HY_STEPS", "8"))
NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "32"))
GUIDANCE = float(os.environ.get("HY_GUIDANCE", "2.5"))
SEED = int(os.environ.get("HY_SEED", "42"))
IMAGE_SIZE = int(os.environ.get("HY_I2I_SIZE", "1024"))
VIT_LAYERS = int(os.environ.get("HY_VIT_LAYERS", "27"))
SCALING = 0.562679178327931
MODEL_DIR = INSTRUCT_MODEL_DIR

_WMAP = json.load(open(MODEL_DIR / "model.safetensors.index.json"))["weight_map"]
_OPEN: dict = {}


def _load(key):
    shard = _WMAP[key]
    f = _OPEN.get(shard) or _OPEN.setdefault(shard, safe_open(MODEL_DIR / shard, framework="pt"))
    return f.get_tensor(key)


def _load_prefix(prefix):
    return {k[len(prefix) + 1 :]: _load(k) for k in _WMAP if k.startswith(prefix + ".")}


def _cfg():
    c = json.load(open(MODEL_DIR / "config.json"))
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=os.environ.get("HY_PROMPT", "make the sky more dramatic"))
    parser.add_argument("--cond", default=os.environ.get("HY_COND", ""), help="Conditional RGB image path")
    parser.add_argument("--out", default=os.environ.get("HY_OUT", str(ROOT / "hy_i2i.png")))
    args = parser.parse_args()

    if not args.cond:
        raise SystemExit("Provide --cond /path/to/image.png or set HY_COND")
    if not (MODEL_DIR / "model.safetensors.index.json").is_file():
        raise SystemExit(
            f"Instruct weights not found under {MODEL_DIR}. "
            "Download HunyuanImage-3.0-Instruct or set HUNYUAN_INSTRUCT_MODEL_DIR."
        )

    print(
        f"[demo_i2i] model={MODEL_DIR}  prompt={args.prompt!r}  cond={args.cond!r}  steps={STEPS}  guidance={GUIDANCE}"
    )
    c = _cfg()
    H = c["H"]
    down_sd, up_sd = _load_prefix("patch_embed"), _load_prefix("final_layer")
    LATENT, HID, HSZ = _pe_dims(down_sd)

    proc = HunyuanImage3ImageProcessor(json.load(open(MODEL_DIR / "config.json")))
    cond_img = Image.open(args.cond).convert("RGB")
    if cond_img.size != (IMAGE_SIZE, IMAGE_SIZE):
        print(f"[demo_i2i] resizing cond {cond_img.size} -> ({IMAGE_SIZE}, {IMAGE_SIZE}) (ratio tokens are [0, 32))")
        cond_img = cond_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    cond, _ = proc.get_image_with_size(cond_img, return_type="vae_vit")

    tok = HunyuanTokenizer.from_model_dir(MODEL_DIR, sequence_template="instruct")
    system_prompt = get_system_prompt("en_unified", "image")
    wte = load_tensors(MODEL_DIR, ["model.wte.weight"])["model.wte.weight"]
    gen = torch.Generator().manual_seed(SEED)
    print("[demo_i2i] building host I2I bundle (instruct template + cond encode + multi-span mask) ...")
    bundle = prepare_i2i_denoise_bundle(
        tok,
        args.prompt,
        cond,
        proc,
        wte,
        patch_embed=load_patch_embed(MODEL_DIR),
        time_embed=load_timestep_embedder("time_embed", MODEL_DIR),
        timestep_emb=load_timestep_embedder("timestep_emb", MODEL_DIR),
        vision_model=load_siglip2_vision(MODEL_DIR, num_layers=VIT_LAYERS),
        aligner=load_aligner(MODEL_DIR),
        image_size=IMAGE_SIZE,
        sequence_template="instruct",
        system_prompt=system_prompt,
        generator=gen,
    )
    cond_row, uncond_row = build_i2i_cfg_conds(bundle, wte, proc)
    img_slice = cond_row["gen_slice"]
    grid = cond_row["gen_hw"]
    seq_len = bundle.seq_len
    print(f"[demo_i2i] seq_len={seq_len} gen_span={img_slice} grid={grid}")

    timestep_emb = ref_timestep_emb(H)

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
        layer_loader = lambda i: {f"model.layers.{i}.{k}": v for k, v in _load_prefix(f"model.layers.{i}").items()}
        bf16_layers = {0, 1, 2, 3, NUM_LAYERS - 4, NUM_LAYERS - 3, NUM_LAYERS - 2, NUM_LAYERS - 1}
        print(f"[demo_i2i] building resident backbone ({NUM_LAYERS} layers) ...")
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
            img_slice=img_slice,
            grid_hw=grid,
            seq_len=seq_len,
        )

        cond_tt = dict(cond_row)
        cond_tt["attention_mask"] = rep(cond_row["attention_mask"].reshape(1, 1, seq_len, seq_len).to(torch.bfloat16))
        uncond_tt = None
        if uncond_row is not None:
            uncond_tt = dict(uncond_row)
            uncond_tt["attention_mask"] = rep(
                uncond_row["attention_mask"].reshape(1, 1, seq_len, seq_len).to(torch.bfloat16)
            )

        torch.manual_seed(SEED + 1)
        init_latent = torch.randn(1, LATENT, grid[0], grid[1])
        sched = HunyuanTtScheduler(mesh_device)
        sched.set_timesteps(STEPS)
        print(f"[demo_i2i] denoising {STEPS} steps (CFG={GUIDANCE}) ...")
        latent = denoise_loop(
            step,
            sched,
            init_latent,
            time_embed=te1,
            time_embed_2=te2,
            cond=cond_tt,
            uncond=uncond_tt,
            guidance_scale=GUIDANCE if uncond_tt is not None else 1.0,
            timestep_emb=timestep_emb,
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
