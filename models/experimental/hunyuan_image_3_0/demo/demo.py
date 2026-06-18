# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end HunyuanImage-3.0 text-to-image on Tenstorrent.
#
#   prompt --HunyuanTokenizer--> input_ids
#          --wte--> text embeddings (cond / uncond rows for CFG)
#   noise latent --[ patch_embed -> RESIDENT bf8 sharded backbone -> final_layer ]
#                  x N scheduler steps with CFG (denoise_loop on the (1,4) mesh)
#          --> denoised latent
#          --VAE decode (PyTorch reference, CPU)--> RGB image  [*]
#
# [*] The TTNN VAE decode is the one unbuilt block (full-res OOM; HW-parallel is
#     future work — see MEMORY_FIT_PLAN.md). We fall back to the fp32 reference
#     VAE on CPU so the pipeline produces a viewable image end to end. Everything
#     else runs on the TT mesh with the model that fits DRAM (bf8 + 4-way expert
#     sharding, first/last 4 layers bf16).
#
# Run:
#   HY_STEPS=8 HY_NUM_LAYERS=32 python_env/bin/python \
#     models/experimental/hunyuan_image_3_0/demo/demo.py "a photo of a cat"

import os, sys, json, glob
import torch
from safetensors import safe_open

ROOT = "/home/iguser/Christy/tt-metal"
HUNYUAN = "/home/iguser/Christy/tt-metal/HunyuanImage-3.0"
WEIGHTS = "/home/iguser/Christy/HunyuanImage-3"
for p in (ROOT, HUNYUAN):
    if p not in sys.path:
        sys.path.insert(0, p)

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer
from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import prepare_gen_image_inputs
from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel
from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown, HunyuanTtUNetUp
from models.experimental.hunyuan_image_3_0.tt.image_gen.timestep_embedder import HunyuanTtTimestepEmbedder
from models.experimental.hunyuan_image_3_0.tt.pipeline import HunyuanTtDenoiseStep, denoise_loop, decode_latent
from models.experimental.hunyuan_image_3_0.tt.scheduler import HunyuanTtScheduler

PROMPT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HY_PROMPT", "a photo of a cat, studio lighting")
STEPS = int(os.environ.get("HY_STEPS", "8"))
NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "32"))
GUIDANCE = float(os.environ.get("HY_GUIDANCE", "5.0"))
SEED = int(os.environ.get("HY_SEED", "0"))
SCALING = 0.562679178327931
OUT_PNG = os.environ.get("HY_OUT", "/home/iguser/Christy/tt-metal/hy_t2i.png")

_WMAP = json.load(open(glob.glob(f"{WEIGHTS}/*.index.json")[0]))["weight_map"]
_OPEN = {}


def _load(key):
    shard = _WMAP[key]
    f = _OPEN.get(shard) or _OPEN.setdefault(shard, safe_open(f"{WEIGHTS}/{shard}", framework="pt"))
    return f.get_tensor(key)


def _load_prefix(prefix):
    return {k[len(prefix) + 1 :]: _load(k) for k in _WMAP if k.startswith(prefix + ".")}


def _cfg():
    c = json.load(open(f"{WEIGHTS}/config.json"))
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
    print(f"[demo] prompt={PROMPT!r}  steps={STEPS}  layers={NUM_LAYERS}  guidance={GUIDANCE}")
    c = _cfg()
    H = c["H"]
    down_sd, up_sd = _load_prefix("patch_embed"), _load_prefix("final_layer")
    LATENT, HID, HSZ = _pe_dims(down_sd)

    # 1) tokenize -> input_ids (cond row 0, uncond row 1) + contiguous image span.
    tok = HunyuanTokenizer.from_pretrained()
    bundle = prepare_gen_image_inputs(tok, PROMPT, image_size=1024)
    ids = bundle.input_ids  # [2, S]
    S = bundle.seq_len
    span = bundle.rope_image_info[0][0][0]
    grid = bundle.rope_image_info[0][0][1]  # (64, 64)
    print(f"[demo] seq_len={S} image_span={span} grid={grid}")

    # 2) text embeddings (host wte lookup — exact) for cond/uncond rows.
    wte = _load("model.wte.weight").float()
    emb = torch.nn.functional.embedding(ids, wte)  # [2, S, H]

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), l1_small_size=32768)
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
        print(f"[demo] building resident backbone ({NUM_LAYERS} layers, bf8 + bf16 layers {sorted(bf16_layers)}) ...")
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
            img_slice=span,
            grid_hw=grid,
            seq_len=S,
        )

        mask = to_additive(build_attention_mask(S, image_slices=[span], bsz=1), dtype=torch.float32).reshape(1, 1, S, S)
        mask_tt = rep(mask)
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
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    # VAE decode on device (TTNN). Fresh mesh context so the backbone DRAM is
    # freed first; the decoder's full-res convs/depth-to-space are H-chunked.
    print("[demo] VAE decode (TTNN, on device) ...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    vae_mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        vae_mesh.enable_program_cache()
        img = decode_latent(vae_mesh, latent, scaling_factor=SCALING)  # [1, 3, 1024, 1024] in [0,1]
    finally:
        ttnn.close_mesh_device(vae_mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    img = img[0]  # [3, 1024, 1024]

    from PIL import Image

    arr = (img.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    Image.fromarray(arr).save(OUT_PNG)
    print(f"[demo] saved image -> {OUT_PNG}")


if __name__ == "__main__":
    main()
