# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tier-1 up-front precompile validation on the real SDXL BasicTransformerBlock.

This is a *complex* layer: self-attention + cross-attention + GEGLU feedforward
+ three layernorms -> many distinct programs (qkv/out matmuls x2, scaled-dot-product
attention/softmax, layernorms, geglu matmuls + gelu, adds). It validates that the
device-op funnel hook captures a real model block's full program set and that a
post-precompile run is correct.

Built offline with random weights (no SDXL/UNet download, ~10GB avoided): a
diffusers BasicTransformerBlock supplies both the torch reference and a matching
state_dict for the real TtBasicTransformerBlock. Shapes and DRAM/L1 input memory
configs mirror the real PCC test (test_module_tt_transformerblock.py, the
down_blocks.1 / 1024x1024 case).

Lives under the SDXL test tree so it inherits the conftest's device fixture
(which injects SDXL_L1_SMALL_SIZE). Run on a FRESH cache so the parallel compile
does real work:

    TT_METAL_CACHE=/tmp/sdxl_$$ scripts/run_safe_pytest.sh \
        models/demos/stable_diffusion_xl_base/tests/pcc/test_up_front_compile_transformerblock.py
"""

import os
import time

import torch
from diffusers.models.attention import BasicTransformerBlock

import ttnn
from models.common.utility_functions import torch_random
from models.demos.stable_diffusion_xl_base.tt.model_configs import load_model_optimisations
from models.demos.stable_diffusion_xl_base.tt.tt_transformerblock import TtBasicTransformerBlock
from tests.ttnn.utils_for_testing import assert_with_pcc

IMAGE_RES = (1024, 1024)
INPUT_SHAPE = (1, 4096, 640)
ENCODER_SHAPE = (1, 77, 2048)
QUERY_DIM = 640
NUM_HEADS = 10
OUT_DIM = 640
MODULE_PATH = "down_blocks.1.attentions.0.transformer_blocks.0"


def _build_block(device):
    """Return (tt_block, torch_in, torch_enc, torch_out) — real TtBasicTransformerBlock, random weights."""
    torch.manual_seed(0)
    torch_block = BasicTransformerBlock(
        dim=QUERY_DIM,
        num_attention_heads=NUM_HEADS,
        attention_head_dim=QUERY_DIM // NUM_HEADS,
        cross_attention_dim=ENCODER_SHAPE[-1],
        activation_fn="geglu",
    ).eval()
    # Prefix the random state_dict with the SDXL module path the Tt module expects.
    state_dict = {f"{MODULE_PATH}.{k}": v for k, v in torch_block.state_dict().items()}

    torch_in = torch_random(INPUT_SHAPE, -0.1, 0.1, dtype=torch.float32)
    torch_enc = torch_random(ENCODER_SHAPE, -0.1, 0.1, dtype=torch.float32)
    with torch.no_grad():
        torch_out = torch_block(torch_in, None, torch_enc).unsqueeze(0)

    model_config = load_model_optimisations(IMAGE_RES)
    tt_block = TtBasicTransformerBlock(device, state_dict, MODULE_PATH, model_config, QUERY_DIM, NUM_HEADS, OUT_DIM)
    return tt_block, torch_in, torch_enc, torch_out


def _make_inputs(device, torch_in, torch_enc):
    """Fresh ttnn inputs per forward (the block deallocates its input tensor)."""
    tt_enc = ttnn.from_torch(
        torch_enc, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_in = ttnn.from_torch(
        torch_in.unsqueeze(0),
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return tt_in, tt_enc


def test_sdxl_transformerblock_up_front_compile(device):
    tt_block, torch_in, torch_enc, torch_out = _build_block(device)

    # --- Phase 1: collect (NO_DISPATCH) — capture the whole block's program set ---
    ttnn.graph.up_front_clear()
    ttnn.graph.up_front_begin_collect()
    try:
        ti, te = _make_inputs(device, torch_in, torch_enc)
        tt_block.forward(ti, None, te)
    finally:
        ttnn.graph.up_front_end_collect()
    n_collected = ttnn.graph.up_front_num_collected()
    n_unique = ttnn.graph.up_front_num_unique()
    print(f"\nSDXL block collect: {n_collected} ops -> {n_unique} unique programs")
    assert n_collected >= 8, "expected a rich program set (2 attentions + ff + 3 layernorms)"

    # --- Phase 2: parallel compile — warm the kernel cache (local executor or remote farm) ---
    req_workers = int(os.environ.get("UP_FRONT_WORKERS", "4"))
    num_programs, num_errors, workers, wall = ttnn.graph.up_front_compile(device, req_workers)
    print(
        f"SDXL block parallel compile: {num_programs} programs in {wall:.2f}s (workers={workers}, errors={num_errors})"
    )
    assert num_errors == 0, "parallel compile reported errors"

    # --- Phase 3: warm forward — must be correct vs the torch reference ---
    ti, te = _make_inputs(device, torch_in, torch_enc)
    t0 = time.perf_counter()
    out = tt_block.forward(ti, None, te)
    ttnn.synchronize_device(device)
    print(f"SDXL block warm forward: {(time.perf_counter() - t0) * 1000:.1f} ms")

    _, pcc_msg = assert_with_pcc(torch_out, ttnn.to_torch(out), 0.99)
    print(f"SDXL block warm-forward PCC: {pcc_msg}")


def test_sdxl_transformerblock_cold_baseline(device):
    """A/B baseline: NO precompile — time the cold-inline first forward (compiles serially).
    Run in its own fresh-cache invocation and compare its forward time to the up-front path's."""
    tt_block, torch_in, torch_enc, torch_out = _build_block(device)

    ti, te = _make_inputs(device, torch_in, torch_enc)
    t0 = time.perf_counter()
    out = tt_block.forward(ti, None, te)
    ttnn.synchronize_device(device)
    print(f"\nSDXL block cold-inline first forward: {time.perf_counter() - t0:.2f}s")

    _, pcc_msg = assert_with_pcc(torch_out, ttnn.to_torch(out), 0.99)
    print(f"SDXL block cold-baseline PCC: {pcc_msg}")
