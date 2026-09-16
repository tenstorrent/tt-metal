# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Demo-faithful inputs and shared device setup for the Qwen3-TTS test suites.

The demo prompt / reference clip the PCC tests measure against, the device
open/close pair the fixtures use, and the prefill-bucket padding and Talker KV
allocation that mirror what ``demo_full_ttnn_tts.py`` does at run time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple

import torch

import ttnn

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REF_AUDIO = REPO_ROOT / "models/demos/qwen3_tts/demo/jim_reference.wav"
DEFAULT_REF_TEXT_PATH = DEFAULT_REF_AUDIO.with_suffix(".txt")
DEFAULT_TARGET_TEXT = (
    "Good morning. Today is a beautiful day for a walk in the park, with bright sun "
    "and a gentle breeze through the trees."
)
DEFAULT_HF_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
_TRACE_REGION = 200_000_000
_L1_SMALL = 32768
_TRACED_PREFILL_BUCKETS = (32, 64, 128)
_TILE = 32


def hf_id() -> str:
    return os.environ.get("HF_MODEL") or os.environ.get("QWEN3_TTS_HF_ID", DEFAULT_HF_ID)


def demo_ref_text() -> str:
    explicit = os.environ.get("QWEN3_TTS_PROFILE_REF_TEXT")
    if explicit is not None:
        return explicit.strip()
    if DEFAULT_REF_TEXT_PATH.is_file():
        return DEFAULT_REF_TEXT_PATH.read_text().strip()
    raise RuntimeError("Set QWEN3_TTS_PROFILE_REF_TEXT or place a .txt next to jim_reference.wav")


def demo_target_text() -> str:
    return os.environ.get("QWEN3_TTS_PROFILE_TARGET_TEXT", DEFAULT_TARGET_TEXT)


def open_profile_device() -> Tuple[Any, Any]:
    mesh_shape = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"))
    if mesh_shape is None:
        device = ttnn.open_device(
            device_id=0,
            l1_small_size=_L1_SMALL,
            trace_region_size=_TRACE_REGION,
        )
        device.enable_program_cache()
        return device, None
    if mesh_shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape),
        l1_small_size=_L1_SMALL,
        trace_region_size=_TRACE_REGION,
    )
    device.enable_program_cache()
    return device, mesh_shape


def close_profile_device(device, mesh_shape) -> None:
    if mesh_shape is None:
        ttnn.close_device(device)
        return
    ttnn.close_mesh_device(device)
    if mesh_shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _padded_max_talker_seq(padded_seq_len: int, max_new_tokens: int = 256) -> int:
    raw = padded_seq_len + max_new_tokens + 16
    return ((raw + _TILE - 1) // _TILE) * _TILE


def pad_inputs_to_demo_bucket(device, inputs_embeds_tt, real_seq_len: int, talker_h: int) -> Tuple[ttnn.Tensor, int]:
    """STEP 1 bucket padding from ``generate_codes_ttnn``."""
    if real_seq_len <= _TRACED_PREFILL_BUCKETS[-1]:
        padded_seq_len = next(b for b in _TRACED_PREFILL_BUCKETS if b >= real_seq_len)
    else:
        from models.demos.qwen3_tts.tt.server import get_padded_prefill_len

        padded_seq_len = get_padded_prefill_len(real_seq_len)

    if padded_seq_len > real_seq_len:
        pad_len = padded_seq_len - real_seq_len
        pad_zeros = ttnn.from_torch(
            torch.zeros(1, 1, pad_len, talker_h, dtype=torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        inputs_embeds_tt = ttnn.concat([inputs_embeds_tt, pad_zeros], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(pad_zeros)
    return inputs_embeds_tt, padded_seq_len


def allocate_talker_kv(device, model, padded_seq_len: int, max_new_tokens: int = 256):
    from models.demos.qwen3_tts.tt.server import allocate_kv_cache

    head_dim = model.talker_config.head_dim
    max_talker_seq_len = _padded_max_talker_seq(padded_seq_len, max_new_tokens)
    talker_kv_caches = allocate_kv_cache(
        device=device,
        num_layers=model.talker_config.num_hidden_layers,
        batch_size=1,
        num_kv_heads=model.talker_config.num_key_value_heads,
        max_seq_len=max_talker_seq_len,
        head_dim=head_dim,
    )
    return talker_kv_caches, max_talker_seq_len


# ── Demo trace capture ────────────────────────────────────────────────────────
# The demo replays Metal traces; an untraced pass has the same kernel graph but
# host-bound op-to-op gaps that swamp the report. These helpers capture the exact
# traces ``generate_codes_ttnn`` captures, so a profiling window can replay one.
