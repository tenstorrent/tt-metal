# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end denoise STEP driven by the RESIDENT sharded backbone on the (1,4)
# mesh: patch_embed + final_layer + norms run replicated, the MoE backbone runs
# expert-parallel (bf8, sharded). Same composition as test_pipeline_step.py but
# on the mesh with the fitting model — proves HunyuanTtDenoiseStep works with the
# resident backbone (the model that actually fits in DRAM). See MEMORY_FIT_PLAN.md.
#
# Run:
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_pipeline_step_resident.py -v -s --timeout=1800

import os, sys, json, glob, gc
import torch
from safetensors import safe_open
import pytest
from loguru import logger

ROOT = "/home/iguser/Christy/tt-metal"
HUNYUAN = "/home/iguser/Christy/tt-metal/HunyuanImage-3.0"
WEIGHTS = "/home/iguser/Christy/HunyuanImage-3"
for p in (ROOT, HUNYUAN):
    if p not in sys.path:
        sys.path.insert(0, p)

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from models.experimental.hunyuan_image_3_0.ref.transformer_layer import HunyuanImage3DecoderLayer as RefLayer
from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive
from models.experimental.hunyuan_image_3_0.ref.image_gen.patch_embed import UNetDown as RefDown, UNetUp as RefUp
from models.experimental.hunyuan_image_3_0.ref.image_gen.timestep_embedder import TimestepEmbedder as RefTimeEmbed
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel
from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown, HunyuanTtUNetUp
from models.experimental.hunyuan_image_3_0.tt.pipeline import HunyuanTtDenoiseStep
from hunyuan_image_3.modeling_hunyuan_image_3 import build_batch_2d_rope

B = 1
TEXT_PRE, TEXT_POST = 32, 32
GRID = 8
N_IMG = GRID * GRID
S = TEXT_PRE + N_IMG + TEXT_POST
IMG_START = TEXT_PRE
IMG_SLICE = slice(IMG_START, IMG_START + N_IMG)
NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "4"))
PCC_THR = 0.98  # bf8 sharded experts at low depth

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
        MI=first(c["moe_intermediate_size"]),
        NSH=first(c["num_shared_expert"]),
        NORM=c.get("norm_topk_prob", True),
        MIXED=c.get("use_mixed_mlp_moe", True),
        QKN=c.get("use_qk_norm", True),
        EPS=c.get("rms_norm_eps", 1e-5),
    )


def _pe_dims(down_sd):
    hid, latent = down_sd["model.0.weight"].shape[:2]
    hsz = down_sd["model.1.in_layers.2.weight"].shape[0]
    return int(latent), int(hid), int(hsz)


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def _time_embed(prefix, H, t):
    te = RefTimeEmbed(H).eval()
    te.load_state_dict({k: v.float() for k, v in _load_prefix(prefix).items()}, strict=True)
    with torch.no_grad():
        return te(t)


def _reference(c, latent, t1, t2, text_embeds, down_sd, up_sd):
    LATENT, HID, HSZ = _pe_dims(down_sd)
    rd = RefDown(1, LATENT, HSZ, HID, HSZ).eval()
    ru = RefUp(1, HSZ, HSZ, HID, LATENT, out_norm=True).eval()
    rd.load_state_dict({k: v.float() for k, v in down_sd.items()}, strict=True)
    ru.load_state_dict({k: v.float() for k, v in up_sd.items()}, strict=True)
    cos, sin = build_batch_2d_rope(S, c["HD"], image_infos=[[(IMG_SLICE, (GRID, GRID))]])
    mask = to_additive(build_attention_mask(S, image_slices=[IMG_SLICE], bsz=B), dtype=torch.float32)
    with torch.no_grad():
        img, th, tw = rd(latent, t1)
        h = text_embeds.clone()
        h[:, IMG_SLICE, :] = img
        for i in range(NUM_LAYERS):
            sd = _load_prefix(f"model.layers.{i}")
            layer = RefLayer(
                hidden_size=c["H"],
                num_attention_heads=c["HEADS"],
                num_key_value_heads=c["KV"],
                attention_head_dim=c["HD"],
                num_experts=c["E"],
                moe_topk=c["K"],
                moe_intermediate_size=c["MI"],
                num_shared_expert=c["NSH"],
                use_mixed_mlp_moe=c["MIXED"],
                norm_topk_prob=c["NORM"],
                use_qk_norm=c["QKN"],
                rms_norm_eps=c["EPS"],
                layer_idx=i,
            )
            layer.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
            layer.eval()
            h = layer(h, attention_mask=mask, custom_pos_emb=(cos, sin))
            del layer
            gc.collect()
        pred = ru(h[:, IMG_SLICE, :], t2, th, tw)
    return pred


@pytest.fixture(scope="function")
def device_params(request):
    # l1_small_size for the conv2d (halo/sliding-window scratch) in patch_embed.
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768}


# Recommended full-QB2 layout: 2x2 mesh, SP=2 (axis 0) + TP=2 (axis 1) + 4-way EP.
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_resident_denoise_step(mesh_device):
    mesh_device.enable_program_cache()
    c = _cfg()
    down_sd = _load_prefix("patch_embed")
    up_sd = _load_prefix("final_layer")
    H = c["H"]
    LATENT, HID, HSZ = _pe_dims(down_sd)

    torch.manual_seed(0)
    latent = torch.randn(B, LATENT, GRID, GRID)
    text_embeds = torch.randn(B, S, H) * 0.02
    timesteps = torch.rand(B)
    t1 = _time_embed("time_embed", H, timesteps)
    t2 = _time_embed("time_embed_2", H, timesteps)
    ref_pred = _reference(c, latent, t1, t2, text_embeds, down_sd, up_sd)

    def _rep(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    t1_tt = _rep(t1.reshape(1, 1, B, H))
    t2_tt = _rep(t2.reshape(1, 1, B, H))

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
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
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

    pre_tt = _rep(text_embeds[:, :IMG_START, :])
    post_tt = _rep(text_embeds[:, IMG_START + N_IMG :, :])
    mask_add = to_additive(build_attention_mask(S, image_slices=[IMG_SLICE], bsz=B), dtype=torch.float32)
    mask_tt = _rep(mask_add.reshape(B, 1, S, S))
    image_infos = [[(IMG_SLICE, (GRID, GRID))]]

    pred_tt = step(
        latent,
        text_pre=pre_tt,
        text_post=post_tt,
        t_emb1=t1_tt,
        t_emb2=t2_tt,
        image_infos=image_infos,
        attention_mask=mask_tt,
        batch=B,
    )
    pred = ttnn.to_torch(pred_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    pred = pred[:B].reshape(B, GRID, GRID, LATENT).permute(0, 3, 1, 2)

    pcc = _pcc(ref_pred, pred)
    logger.info(f"resident denoise step (mesh, bf8 experts, {NUM_LAYERS} layers) PCC={pcc:.6f}")
    assert pcc >= PCC_THR, f"PCC {pcc:.6f} < {PCC_THR}"
