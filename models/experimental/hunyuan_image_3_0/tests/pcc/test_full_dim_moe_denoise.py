# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Remaining full-dimension gates:
#   1) MoE router + expert FFN + MoE layer at max-context S≈22784
#   2) Multi-step diffusion loop at production GRID=64 / 32L / 50-step schedule
#
# Run this file alone (from tt-metal repo root). Must pass --timeout (pytest.ini default is 300s):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_full_dim_moe_denoise.py \
#     -v -s --timeout=43200
#
# MoE only:
#   .../test_full_dim_moe_denoise.py -k moe_max_context -v -s --timeout=43200
#
# Denoise loop only (HY_STEPS=5 for a shorter check):
#   HY_NUM_LAYERS=32 HY_STEPS=50 python_env/bin/python -m pytest \
#     .../test_full_dim_moe_denoise.py -k denoise_loop_production -v -s --timeout=43200

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest
import torch
import ttnn

ROOT = Path(__file__).resolve().parents[5]
PCC_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PCC_DIR) not in sys.path:
    sys.path.insert(0, str(PCC_DIR))

from models.experimental.hunyuan_image_3_0.ref.weights import load_prefixed_state_dict, resolve_base_model_dir
from denoise_helpers import (
    clear_ref_layer_cache,
    production_loop_pcc_threshold,
    reference_loop,
    run_denoise_loop_tt,
)
from pcc_common import (
    MOE_SET_MATCH,
    MOE_WEIGHT_PCC,
    PCC_STRICT,
    PIPELINE_LAYOUT_PROD,
    max_seq_tile_aligned,
    pcc_metrics,
    transformer_cfg,
)
from test_moe import _expert_ffn_run, _moe_layer_run, _router_run

BATCH = 1
NUM_LAYERS_PRODUCTION = 32
STEPS_PRODUCTION = 50
# Global pytest.ini timeout=300s is far too short for 32L×50 / max-context MoE.
_LONG_TIMEOUT = 43200


@pytest.fixture(scope="function")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


# ---------------------------------------------------------------------------
# MoE @ max-context S=22784
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.timeout(_LONG_TIMEOUT)
def test_moe_router_max_context_pcc(device):
    """Top-k router at tile-aligned max_position_embeddings (S≈22784)."""
    seq_len = max_seq_tile_aligned()
    effective_rate, weight_pcc, genuine = _router_run(device, seq_len)
    print(
        f"MoE router max-context S={seq_len}: effective={100 * effective_rate:.2f}%  "
        f"weight_PCC={weight_pcc:.8f}  genuine_mismatch={genuine}"
    )
    assert genuine == 0
    assert effective_rate >= MOE_SET_MATCH
    assert weight_pcc >= MOE_WEIGHT_PCC


@pytest.mark.slow
@pytest.mark.timeout(_LONG_TIMEOUT)
@pytest.mark.parametrize("module", ["expert_ffn", "moe_layer"])
def test_moe_module_max_context_pcc(device, module):
    """Expert FFN + full MoE layer at max-context S≈22784 (full E=64 / topk=8 / H=4096)."""
    seq_len = max_seq_tile_aligned()
    if module == "expert_ffn":
        p, d = _expert_ffn_run(device, seq_len)
    else:
        p, d = _moe_layer_run(device, seq_len)
    print(f"MoE {module} max-context S={seq_len}: PCC={p:.8f}  max|diff|={d:.6f}")
    assert p >= PCC_STRICT


# ---------------------------------------------------------------------------
# Multi-step denoise @ GRID=64 / 32L / 50-step FlowMatch schedule
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.timeout(_LONG_TIMEOUT)
def test_denoise_loop_production_32l_pcc():
    """Production diffusion loop: GRID=64 (S=4160), 32L backbone, full 50-step schedule.

    Host ref streams one MoE layer at a time (caching all 32 OOMs). Device is opened
    only after the CPU reference finishes so peak RAM stays bounded.
    """
    num_layers = int(os.environ.get("HY_NUM_LAYERS", str(NUM_LAYERS_PRODUCTION)))
    steps = int(os.environ.get("HY_STEPS", str(STEPS_PRODUCTION)))
    if num_layers != NUM_LAYERS_PRODUCTION:
        pytest.skip(f"requires HY_NUM_LAYERS={NUM_LAYERS_PRODUCTION}, got {num_layers}")

    layout = PIPELINE_LAYOUT_PROD
    c = transformer_cfg()
    down_sd = load_prefixed_state_dict(resolve_base_model_dir(), "patch_embed.")
    up_sd = load_prefixed_state_dict(resolve_base_model_dir(), "final_layer.")
    h = c["H"]
    grid = layout["grid"]
    s = layout["seq_len"]
    assert grid == 64 and s == 4160, f"expected production layout, got GRID={grid} S={s}"

    from pipeline_helpers import patch_embed_dims

    latent_ch, _, _ = patch_embed_dims(down_sd)
    thr = production_loop_pcc_threshold(num_layers, steps)

    torch.manual_seed(0)
    init_latent = torch.randn(BATCH, latent_ch, grid, grid)
    text_embeds = torch.randn(BATCH, s, h) * 0.02

    clear_ref_layer_cache()
    print(
        f"[denoise loop] CPU ref start: GRID={grid} S={s} layers={num_layers} steps={steps} "
        f"(streaming MoE layers to avoid host OOM)",
        flush=True,
    )
    ref_final = reference_loop(
        c,
        layout,
        num_layers,
        init_latent,
        text_embeds,
        down_sd,
        up_sd,
        steps,
        BATCH,
        stream_layers=True,
        progress=True,
    )
    clear_ref_layer_cache()
    gc.collect()

    print("[denoise loop] opening device for TT path", flush=True)
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        tt_final = run_denoise_loop_tt(
            device, layout, num_layers, init_latent, text_embeds, steps, c, down_sd, up_sd, mesh=False
        )
    finally:
        ttnn.close_device(device)

    p, d = pcc_metrics(ref_final, tt_final, thr)
    print(f"denoise loop production 32L GRID={grid} S={s} steps={steps}: " f"PCC={p:.8f}  max|diff|={d:.6f}  thr={thr}")
    assert p >= thr
