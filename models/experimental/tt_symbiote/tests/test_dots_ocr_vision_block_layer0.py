# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Unit test for a single vision transformer block (TTNNDotsVisionBlock, layer 0).

Targets vision_tower.block_stack.layers[0]:
    norm1  -> LayerNormDeviceOperation   (BF16 x BF16 => BF16)
    attn   -> QKV 12288x1536x4608        (LoFi BF16 x BFP8 => BFP8)
               nlp_create_heads           (BFP8 => BFP8)
               rotary_embedding x2        (BFP8 x BF16 => BFP8)
               typecast BFP8->BFP4 (V)
               SDPA                       (BFP8 x BFP8 => BFP8)
               nlp_concat_heads
               O-proj 12288x1536x1536    (LoFi BFP8 x BFP8 => BF16)
    norm2  -> LayerNormDeviceOperation
    mlp    -> gate 12288x1536x4224       (LoFi BF16 x BFP8 => BFP8)
               up   12288x1536x4224       (LoFi BF16 x BFP8 => BFP8)
               silu * mul                 (BFP8 x BFP8 => BFP8)
               down 12288x4224x1536      (LoFi BFP8 x BFP8 => BFP8)
    residual adds (BFP8)

Input: [1, 1, SEQ_LEN, 1536] BFP8 DRAM.
Default SEQ_LEN = 12288 (matches perf-report image size).
Override with DOTS_OCR_VISION_SEQ_LEN environment variable.

Three-pass trace lifecycle (mirrors TTNNDotsOCRPipeline.warmup):
  pass 1 -> eager warm-up (weight cache written/read from disk)
  pass 2 -> ttnn.begin_trace_capture -> block.forward -> ttnn.end_trace_capture
  pass 3 -> ttnn.execute_trace (wall-clock measured)

Note: rot_mats (cos/sin) must be built *outside* the trace.  Using
``cu_seqlens=None`` with a seq_len that is already 32-aligned (12288) means
``_sdpa_padded_with_key_mask`` never issues a host->device write, so the
forward is fully traceable without the ``Writes are not supported during
trace capture`` crash.

Run::

    TT_SYMBIOTE_RUN_MODE=TRACED MESH_DEVICE=T3K DOTS_OCR_PARALLELISM=DP \\
        pytest -s \\
        models/experimental/tt_symbiote/tests/test_dots_ocr_vision_block_layer0.py

    # Single-device (N150):
    MESH_DEVICE=N150 pytest -s \\
        models/experimental/tt_symbiote/tests/test_dots_ocr_vision_block_layer0.py

    # Custom sequence length:
    DOTS_OCR_VISION_SEQ_LEN=8192 MESH_DEVICE=T3K \\
        pytest -s \\
        models/experimental/tt_symbiote/tests/test_dots_ocr_vision_block_layer0.py
"""

from __future__ import annotations

import os
import time

import pytest
import torch
import ttnn

from models.experimental.tt_symbiote.modules.dots_ocr_vision import (
    TTNNDotsVisionBlock,
    TTNNDotsVision2DRoPE,
)
from models.experimental.tt_symbiote.utils.device_management import set_device

# ---------------------------------------------------------------------------
# Device resolution (mirrors the rest of the test suite)
# ---------------------------------------------------------------------------

MESH_DEVICE_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}

DOTS_OCR_DP_MESH_DEVICE_MAP = {
    "N300": (2, 1),
    "T3K": (8, 1),
}

DOTS_OCR_MODEL_ID = "rednote-hilab/dots.ocr"


def _resolve_mesh_device_shape():
    mesh_device = os.environ.get("MESH_DEVICE")
    if os.environ.get("DOTS_OCR_PARALLELISM", "").upper() == "DP":
        return DOTS_OCR_DP_MESH_DEVICE_MAP.get(
            mesh_device, MESH_DEVICE_MAP.get(mesh_device, len(ttnn.get_device_ids()))
        )
    return MESH_DEVICE_MAP.get(mesh_device, len(ttnn.get_device_ids()))


def _dots_ocr_mesh_num_devices():
    sh = _resolve_mesh_device_shape()
    if isinstance(sh, int):
        return max(1, int(sh))
    if isinstance(sh, (tuple, list)):
        return int(sh[0]) * int(sh[1]) if len(sh) >= 2 else int(sh[0])
    return 1


def _dots_ocr_device_params():
    dp = {"trace_region_size": 300000000, "num_command_queues": 1}
    if _dots_ocr_mesh_num_devices() > 1:
        dp["fabric_config"] = ttnn.FabricConfig.FABRIC_1D_RING
    else:
        dp["fabric_config"] = ttnn.FabricConfig.DISABLED
    return dp


# ---------------------------------------------------------------------------
# Model path
# ---------------------------------------------------------------------------


def _resolve_model_path() -> str:
    env_path = os.environ.get("DOTS_OCR_MODEL_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(DOTS_OCR_MODEL_ID)
    except Exception:
        return DOTS_OCR_MODEL_ID


DOTS_OCR_LOCAL_PATH = _resolve_model_path()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_seq_len() -> int:
    return int(os.environ.get("DOTS_OCR_VISION_SEQ_LEN", "12288"))


def _load_hf_vision_block(block_idx: int = 0):
    """Load the HF model and extract one vision block (CPU, bfloat16).

    dots.ocr (Qwen2.5-VL) exposes the visual encoder as ``hf_model.vision_tower``
    (confirmed in ``TTNNDotsOCRPipeline.from_hf_model``).  The individual
    transformer blocks are under ``.blocks`` or ``.layers``.
    """
    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        DOTS_OCR_LOCAL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    vision_tower = getattr(hf_model, "vision_tower", None)
    if vision_tower is None:
        raise RuntimeError("Could not find vision_tower in HF model")
    blocks = getattr(vision_tower, "blocks", None) or getattr(vision_tower, "layers", None)
    if not blocks:
        raise RuntimeError("No vision blocks found in hf_model.vision_tower")
    return blocks[block_idx]


def _grid_thw_for_seq_len(seq_len: int) -> torch.Tensor:
    """Find a balanced T=1, H×W grid that produces exactly seq_len tokens.

    Both H and W must be divisible by spatial_merge_size=2.  Searches from
    the square root downward for the largest even W that divides seq_len.
    Falls back to [[1, 1, seq_len]] (may cause RoPE to fail if seq_len is
    odd; caller should ensure seq_len is reasonable).
    """
    import math

    for w in range(int(math.isqrt(seq_len)), 0, -1):
        if seq_len % w == 0 and w % 2 == 0:
            h = seq_len // w
            if h % 2 == 0:
                return torch.tensor([[1, h, w]], dtype=torch.int64)
    return torch.tensor([[1, 1, seq_len]], dtype=torch.int64)


def _build_synthetic_input(
    seq_len: int,
    hidden_size: int,
    device,
    dtype: ttnn.DataType = ttnn.bfloat8_b,
) -> ttnn.Tensor:
    """Random [1, 1, seq_len, hidden_size] BFP8 DRAM tensor on device."""
    num_dev = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
    mapper = ttnn.ReplicateTensorToMesh(device) if num_dev > 1 else None
    x = torch.randn(1, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    return ttnn.from_torch(
        x,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", [_dots_ocr_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [_resolve_mesh_device_shape()], indirect=True)
def test_dots_ocr_vision_block_layer0_traced(mesh_device):
    """Three-pass traced unit test for vision_tower.block_stack.layers[0].

    Pass 1 (warm-up, eager):
        Loads BFP4/BFP8 weight flatbuffers from cache or writes them on
        first run (~6 s one-time cost).  Verifies output shape.

    Pass 2 (trace capture):
        ``ttnn.begin_trace_capture`` -> ``block.forward`` ->
        ``ttnn.end_trace_capture``.  All device ops are recorded into
        ``trace_id``.  The rot_mats (cos/sin) tensors are built *before*
        capture so no host->device transfer fires inside the trace.

    Pass 3 (trace replay, measured):
        ``ttnn.execute_trace`` replays the recorded op sequence.
        Wall-clock printed at the end.
    """
    seq_len = _default_seq_len()
    hidden_size = 1536
    num_heads = 12
    head_dim = hidden_size // num_heads  # 128
    spatial_merge_size = 2

    # ------------------------------------------------------------------
    # 1. Build TTNNDotsVisionBlock from the HF checkpoint
    # ------------------------------------------------------------------
    print(f"\n[vision-block-0] loading vision_tower.blocks[0] from {DOTS_OCR_LOCAL_PATH}")
    hf_block = _load_hf_vision_block(block_idx=0)

    block = TTNNDotsVisionBlock.from_torch(hf_block, hidden_size=hidden_size, num_heads=num_heads)

    # Set the module name to match the production pipeline so on-disk weight
    # caches (BFP4/BFP8 flatbuffers) are shared with full pipeline runs.
    block._unique_name = "vision_tower.blocks[0]"
    block.override_children_module_names()

    # ``set_device`` mirrors ``TTNNDotsOCRPipeline._set_device_and_preprocess``:
    # it recurses through every child TTNNModule (norm1, attn, mlp, …) and
    # calls ``to_device`` on each, so their ``device`` property is non-None
    # before ``move_weights_to_device`` asserts on it.
    set_device(block, mesh_device, register_forward_hook=False)
    block.preprocess_weights()
    block.move_weights_to_device()

    # ------------------------------------------------------------------
    # 2. Build 2D RoPE rotation matrices (outside trace capture)
    # ------------------------------------------------------------------
    grid_thw = _grid_thw_for_seq_len(seq_len)
    print(f"[vision-block-0] seq_len={seq_len}, grid_thw={grid_thw.tolist()}")

    rope = TTNNDotsVision2DRoPE(
        device=mesh_device,
        head_dim=head_dim,
        spatial_merge_size=spatial_merge_size,
    )
    rot_mats, _cu_seqlens = rope.build(grid_thw, seq_len)

    # ------------------------------------------------------------------
    # Timing helper
    # ------------------------------------------------------------------
    def _ts(label: str, t0: float) -> float:
        ttnn.synchronize_device(mesh_device)
        dt = time.time() - t0
        print(f"[vision-block-0] {label}: {dt:.3f} s")
        return time.time()

    # ------------------------------------------------------------------
    # Pass 1: eager warm-up
    # ------------------------------------------------------------------
    t0 = time.time()
    print("[vision-block-0] pass 1: eager warm-up")
    x1 = _build_synthetic_input(seq_len, hidden_size, mesh_device)
    out1 = block.forward(x1, rot_mats=rot_mats)
    t0 = _ts("pass 1 done", t0)

    assert list(out1.shape) == [1, 1, seq_len, hidden_size], (
        f"pass 1 output shape mismatch: expected [1, 1, {seq_len}, {hidden_size}], " f"got {list(out1.shape)}"
    )
    ttnn.deallocate(x1)
    ttnn.deallocate(out1)

    # ------------------------------------------------------------------
    # Pass 2: trace capture
    # ------------------------------------------------------------------
    print("[vision-block-0] pass 2: trace capture")
    x2 = _build_synthetic_input(seq_len, hidden_size, mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out2 = block.forward(x2, rot_mats=rot_mats)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    t0 = _ts("pass 2 done (trace captured)", t0)

    # ------------------------------------------------------------------
    # Pass 3: trace replay (measured)
    # ------------------------------------------------------------------
    print("[vision-block-0] pass 3: trace replay (measured)")
    ttnn.synchronize_device(mesh_device)
    start = time.time()
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = (time.time() - start) * 1000.0

    print(f"\n{'='*60}")
    print(f"dots.ocr Vision Block 0 – single-layer traced replay")
    print(f"{'='*60}")
    print(f"Mesh:       {tuple(mesh_device.shape) if hasattr(mesh_device, 'shape') else mesh_device}")
    print(f"seq_len:    {seq_len}")
    print(f"hidden_dim: {hidden_size}")
    print(f"num_heads:  {num_heads}")
    print(f"head_dim:   {head_dim}")
    print(f"grid_thw:   {grid_thw.tolist()}")
    print(f"Replay:     {elapsed_ms:.3f} ms")
    print(f"{'='*60}\n")

    assert list(out2.shape) == [1, 1, seq_len, hidden_size], (
        f"pass 2 output shape mismatch: expected [1, 1, {seq_len}, {hidden_size}], " f"got {list(out2.shape)}"
    )

    ttnn.release_trace(mesh_device, trace_id)
    ttnn.deallocate(x2)
    ttnn.deallocate(out2)
