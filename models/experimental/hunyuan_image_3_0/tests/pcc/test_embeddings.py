# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Consolidated PCC tests for embedding / conditioning modules:
#   patch_embed (UNetDown) + final_layer (UNetUp), timestep embedders, WTE.
#
# Real checkpoint weights throughout. Lean latent grids: GRID=8 (fast), GRID=64 (slow).
#
# Run (fast):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_embeddings.py -m "not slow" -v
# Full:
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_embeddings.py -v -s

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

ROOT = Path(__file__).resolve().parents[5]
PCC_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PCC_DIR) not in sys.path:
    sys.path.insert(0, str(PCC_DIR))

import ttnn
from models.experimental.hunyuan_image_3_0.ref.image_gen.patch_embed import UNetDown as RefDown, UNetUp as RefUp
from models.experimental.hunyuan_image_3_0.ref.image_gen.timestep_embedder import TimestepEmbedder
from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer
from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import prepare_gen_image_inputs
from models.experimental.hunyuan_image_3_0.ref.weights import (
    load_prefixed_state_dict,
    load_tensors,
    resolve_base_model_dir,
)
from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown, HunyuanTtUNetUp
from models.experimental.hunyuan_image_3_0.tt.image_gen.timestep_embedder import HunyuanTtTimestepEmbedder
from models.experimental.hunyuan_image_3_0.tt.wte import HunyuanTtWte
from pcc_common import (
    PATCH_GRID_FAST,
    PATCH_GRID_SLOW,
    PCC_BLOCK,
    PCC_STRICT,
    TIMESTEP_EMBEDDER_PREFIXES,
    TIMESTEP_RTL,
    load_config,
    pcc_metrics,
)

PATCH_SIZE = 1
BATCH = 1
WTE_PROMPT = "a photo of a cat"
WTE_IMAGE_SIZE = 1024
WTE_PRODUCTION_IMAGE_SIZE = 1024  # 64×64 image tokens @ 1024² latent
WTE_FULL_GRID_TOKENS = 64 * 64


@pytest.fixture(scope="function")
def device():
    """Function-scoped device so mesh WTE can open a 2×2 mesh after single-device cases."""
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


def _patch_embed_dims(down_sd: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    latent = int(down_sd["model.0.weight"].shape[1])
    hid = int(down_sd["model.1.in_layers.2.weight"].shape[1])
    hsz = int(down_sd["model.1.in_layers.2.weight"].shape[0])
    return latent, hid, hsz


def _load_patch_weights():
    model_dir = resolve_base_model_dir()
    down_sd = load_prefixed_state_dict(model_dir, "patch_embed.")
    up_sd = load_prefixed_state_dict(model_dir, "final_layer.")
    return down_sd, up_sd


def _patch_embed_run(device, grid: int, *, seed: int = 0) -> tuple[float, float]:
    down_sd, up_sd = _load_patch_weights()
    latent, hid, hsz = _patch_embed_dims(down_sd)

    ref_down = RefDown(PATCH_SIZE, latent, hsz, hid, hsz).eval()
    ref_up = RefUp(PATCH_SIZE, hsz, hsz, hid, latent, out_norm=True).eval()
    ref_down.load_state_dict({k: v.float() for k, v in down_sd.items()}, strict=True)
    ref_up.load_state_dict({k: v.float() for k, v in up_sd.items()}, strict=True)

    torch.manual_seed(seed)
    x = torch.randn(BATCH, latent, grid, grid)
    t = torch.randn(BATCH, hsz)

    with torch.no_grad():
        ref_seq, th, tw = ref_down(x, t)
        ref_lat = ref_up(ref_seq, t, th, tw)

    tt_down = HunyuanTtUNetDown(
        device,
        {f"patch_embed.{k}": v for k, v in down_sd.items()},
        in_channels=latent,
        hidden_channels=hid,
        out_channels=hsz,
    )
    tt_up = HunyuanTtUNetUp(
        device,
        {f"final_layer.{k}": v for k, v in up_sd.items()},
        in_channels=hsz,
        hidden_channels=hid,
        out_channels=latent,
    )
    t_tt = ttnn.from_torch(t.reshape(1, 1, BATCH, hsz), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    seq_tt, h2, w2 = tt_down(x, t_tt)
    lat_tt, h3, w3 = tt_up(seq_tt, t_tt, h2, w2, B=BATCH)

    seq_t = ttnn.to_torch(seq_tt).reshape(BATCH, th * tw, hsz)
    lat_t = ttnn.to_torch(lat_tt).reshape(BATCH, h3, w3, latent).permute(0, 3, 1, 2)

    p_down, _ = pcc_metrics(ref_seq, seq_t, PCC_BLOCK)
    p_up, _ = pcc_metrics(ref_lat, lat_t, PCC_BLOCK)
    return p_down, p_up


def _timestep_run(device, prefix: str, *, seed: int = 0) -> tuple[float, float, float]:
    h = load_config()["hidden_size"]
    model_dir = resolve_base_model_dir()
    sd = load_prefixed_state_dict(model_dir, f"{prefix}.")

    ref = TimestepEmbedder(h)
    ref.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    ref.eval()

    torch.manual_seed(seed)
    t = torch.rand(8)
    with torch.no_grad():
        ref_out = ref(t)

    tt = HunyuanTtTimestepEmbedder(device, h, {f"{prefix}.{k}": v for k, v in sd.items()}, prefix)
    out_tt = tt(t)
    tt_out = ttnn.to_torch(out_tt)[..., :h]
    tt.deallocate()
    out_tt.deallocate()

    p, d = pcc_metrics(ref_out, tt_out, PCC_STRICT)
    rel = d / (ref_out.float().abs().max().item() + 1e-9)
    return p, d, rel


@pytest.fixture(scope="function")
def device_params():
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


# ---------------------------------------------------------------------------
# patch_embed + final_layer (real weights)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("grid,label", PATCH_GRID_FAST)
def test_patch_embed_pcc(device, grid, label):
    p_down, p_up = _patch_embed_run(device, grid)
    print(f"patch_embed [{label}]: UNetDown PCC={p_down:.8f}  UNetUp PCC={p_up:.8f}")
    assert p_down >= PCC_BLOCK and p_up >= PCC_BLOCK


@pytest.mark.slow
@pytest.mark.parametrize("grid,label", PATCH_GRID_SLOW)
def test_patch_embed_large_grid_pcc(device, grid, label):
    p_down, p_up = _patch_embed_run(device, grid)
    assert p_down >= PCC_BLOCK and p_up >= PCC_BLOCK


@pytest.mark.slow
def test_patch_embed_production_64_pcc(device):
    """Production GRID=64 (1024² latent) patch_embed + final_layer — CI gate."""
    p_down, p_up = _patch_embed_run(device, 64)
    print(f"patch_embed production GRID=64: UNetDown PCC={p_down:.8f}  UNetUp PCC={p_up:.8f}")
    assert p_down >= PCC_BLOCK and p_up >= PCC_BLOCK


# ---------------------------------------------------------------------------
# Timestep embedders (real weights; all three prefixes)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("prefix", TIMESTEP_EMBEDDER_PREFIXES)
def test_timestep_embedder_pcc(device, prefix):
    p, d, rel = _timestep_run(device, prefix)
    print(f"{prefix}: PCC={p:.8f}  max|diff|={d:.6f}  rel={rel:.4%}")
    assert p >= PCC_STRICT and rel <= TIMESTEP_RTL


@pytest.mark.slow
@pytest.mark.parametrize("prefix", TIMESTEP_EMBEDDER_PREFIXES)
def test_timestep_embedder_full_schedule_pcc(device, prefix):
    """Timestep embedders over the full 50-step FlowMatch schedule (production denoise)."""
    from models.experimental.hunyuan_image_3_0.ref.scheduler import FlowMatchDiscreteScheduler

    h = load_config()["hidden_size"]
    model_dir = resolve_base_model_dir()
    sd = load_prefixed_state_dict(model_dir, f"{prefix}.")
    ref = TimestepEmbedder(h)
    ref.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    ref.eval()

    sched = FlowMatchDiscreteScheduler(shift=3.0, reverse=True, solver="euler")
    sched.set_timesteps(50)
    t = sched.timesteps.detach().float().cpu().reshape(-1)
    assert t.numel() == 50
    with torch.no_grad():
        ref_out = ref(t)

    tt = HunyuanTtTimestepEmbedder(device, h, {f"{prefix}.{k}": v for k, v in sd.items()}, prefix)
    out_tt = tt(t)
    tt_out = ttnn.to_torch(out_tt)[..., :h]
    tt.deallocate()
    out_tt.deallocate()

    p, d = pcc_metrics(ref_out, tt_out, PCC_STRICT)
    rel = d / (ref_out.float().abs().max().item() + 1e-9)
    print(f"{prefix} full schedule S=50: PCC={p:.8f}  max|diff|={d:.6f}  rel={rel:.4%}")
    assert p >= PCC_STRICT and rel <= TIMESTEP_RTL
    assert ref_out.shape == (50, h)


# ---------------------------------------------------------------------------
# Word-token embedding table (real weights + tokenizer bundle)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_wte_pcc(mesh_device):
    model_dir = resolve_base_model_dir()
    if not (model_dir / "model.safetensors.index.json").is_file():
        pytest.skip(f"weights not found at {model_dir}")

    mesh_device.enable_program_cache()
    tok = HunyuanTokenizer.from_pretrained()
    bundle = prepare_gen_image_inputs(tok, WTE_PROMPT, image_size=WTE_IMAGE_SIZE)
    input_ids = bundle.input_ids

    wte_w = load_tensors(model_dir, ["model.wte.weight"])["model.wte.weight"]
    ref = F.embedding(input_ids, wte_w.float())

    wte_tt = HunyuanTtWte(mesh_device, wte_w, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    out = wte_tt.embedding_torch(input_ids)

    p, d = pcc_metrics(ref, out, PCC_STRICT)
    logger.info(f"HunyuanTtWte PCC={p:.8f} shape={tuple(out.shape)} max|diff|={d:.6f}")
    assert p >= PCC_STRICT


@pytest.mark.slow
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_wte_production_pcc(mesh_device):
    """Production WTE at real T2I bundle with full 64×64 image token grid (1024²)."""
    model_dir = resolve_base_model_dir()
    if not (model_dir / "model.safetensors.index.json").is_file():
        pytest.skip(f"weights not found at {model_dir}")

    mesh_device.enable_program_cache()
    tok = HunyuanTokenizer.from_pretrained()
    bundle = prepare_gen_image_inputs(tok, WTE_PROMPT, image_size=WTE_PRODUCTION_IMAGE_SIZE, cfg_factor=1)
    gen_slice = bundle.gen_image_slices[0][0]
    n_img = gen_slice.stop - gen_slice.start
    assert n_img == WTE_FULL_GRID_TOKENS, f"expected {WTE_FULL_GRID_TOKENS} image tokens, got {n_img}"
    input_ids = bundle.input_ids

    wte_w = load_tensors(model_dir, ["model.wte.weight"])["model.wte.weight"]
    ref = F.embedding(input_ids, wte_w.float())

    wte_tt = HunyuanTtWte(mesh_device, wte_w, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))
    out = wte_tt.embedding_torch(input_ids)

    p, d = pcc_metrics(ref, out, PCC_STRICT)
    logger.info(
        f"HunyuanTtWte production S={bundle.seq_len} PCC={p:.8f} " f"shape={tuple(out.shape)} max|diff|={d:.6f}"
    )
    assert p >= PCC_STRICT
