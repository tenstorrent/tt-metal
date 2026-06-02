# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Find the maximum supported context length for Mistral-Small-4-119B on the
current mesh.

The model has NO architectural context limit — `max_seq_len` is a runtime knob
(default 4096). The real ceiling is per-device DRAM, dominated by the
*replicated* KV cache (each chip holds the full cache). This tool:

  1. Opens the mesh exactly like the demo (honours $MESH_DEVICE).
  2. Reports per-device total DRAM and the analytical KV bytes/token.
  3. Empirically binary-searches the largest `max_seq_len` whose full
     36-layer × (K+V) replicated cache actually allocates without OOM.

Use --reserve-gb to model the weights + activation headroom that the real model
consumes per device (paste the figure you measure after a real load), so the
empirical number reflects what's left for the cache in practice.

Usage:
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    MESH_DEVICE=P150x8 python models/experimental/mistral_small_4_119b/find_max_context.py
    MESH_DEVICE=P150x8 python models/experimental/mistral_small_4_119b/find_max_context.py --reserve-gb 18
"""

import argparse

import torch
from loguru import logger

import ttnn
from models.experimental.mistral_small_4_119b.constants import (
    EXPECTED_NUM_LAYERS,
    HF_MODEL_ID,
    KV_LORA_RANK,
    QK_ROPE_HEAD_DIM,
    VISION_NUM_LAYERS,
)
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param

BF16_BYTES = 2
_MiB = 1024 * 1024
_GiB = 1024 * 1024 * 1024


def _open_mesh():
    """Mirror tt_demo_agent._open_mesh_device (fabric for multi-chip)."""
    rows, cols = mesh_device_request_param()
    fabric_set = False
    if (rows, cols) != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT)
        fabric_set = True
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(rows, cols),
        dispatch_core_config=ttnn.DispatchCoreConfig(),
        num_command_queues=1,
    )
    logger.info(f"Opened {rows}×{cols} mesh ({device.get_num_devices()} chips)")
    return device, fabric_set


def _close_mesh(device, fabric_set):
    try:
        ttnn.close_mesh_device(device)
    finally:
        if fabric_set:
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception:  # noqa: BLE001
                pass


def _dram_view(mesh_device) -> tuple[int, int] | None:
    """(total, free) DRAM bytes per chip. Cache is DRAM-interleaved across all
    banks, so the per-chip budget is num_banks × per-bank capacity. On a mesh the
    view reports one chip's banks (layout is identical across chips)."""
    try:
        mv = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
        return mv.num_banks * mv.total_bytes_per_bank, mv.num_banks * mv.total_bytes_free_per_bank
    except Exception as e:  # noqa: BLE001
        logger.warning(f"get_memory_view failed ({e}); skipping DRAM report")
        return None


_KVPE_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 320 — MLA latent cache width


def _kv_bytes_per_token(num_layers: int) -> int:
    """MLA latent (kvpe) cache bytes added per token, per device (replicated).

    Latent caching stores a single shared [kv_latent ‖ k_rope] vector per token —
    one "KV head" — instead of expanded per-head K/V. ~25× smaller than the old
    expanded cache.
    """
    return num_layers * _KVPE_DIM * BF16_BYTES


def _try_allocate(mesh_device, max_seq_len: int, num_layers: int) -> bool:
    """Allocate the real per-layer MLA latent (kvpe) cache at max_seq_len. True if it fits.

    Mirrors allocate_kv_cache: one [1, 1, padded_seq, KVPE_DIM] replicated tensor
    per layer. Reuses one zeroed host tensor across all layers so host RAM stays
    flat; each as_tensor still produces a distinct per-device buffer, so device
    DRAM footprint matches the real model exactly.
    """
    padded = ((max_seq_len + 31) // 32) * 32
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    kvpe_host = torch.zeros(1, 1, padded, _KVPE_DIM, dtype=torch.bfloat16)
    allocated = []
    try:
        for _ in range(num_layers):
            allocated.append(
                ttnn.as_tensor(
                    kvpe_host,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mapper,
                )
            )
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug(f"alloc failed @ {max_seq_len}: {e}")
        return False
    finally:
        for t in allocated:
            ttnn.deallocate(t)


def _reserve(mesh_device, reserve_gb: float):
    """Pin a per-device placeholder to model weights + activation headroom."""
    if reserve_gb <= 0:
        return None
    elems = int(reserve_gb * _GiB / BF16_BYTES)
    width = 4096
    rows = max(1, elems // width)
    rows = ((rows + 31) // 32) * 32
    host = torch.zeros(1, 1, rows, width, dtype=torch.bfloat16)
    t = ttnn.as_tensor(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    logger.info(f"Reserved ~{rows * width * BF16_BYTES / _GiB:.2f} GiB/device placeholder")
    return t


def _binary_search(mesh_device, num_layers: int, lo: int, hi_cap: int) -> int:
    """Find the largest max_seq_len that allocates. Doubles to bound, then bisects."""
    if not _try_allocate(mesh_device, lo, num_layers):
        logger.error(f"Even {lo} tokens does not fit — reduce --reserve-gb or layers.")
        return 0
    # Grow until failure to bound the upper edge.
    hi = lo
    nxt = lo * 2
    while nxt <= hi_cap and _try_allocate(mesh_device, nxt, num_layers):
        logger.info(f"  fits: {nxt}")
        hi = nxt
        nxt *= 2
    if nxt > hi_cap and _try_allocate(mesh_device, hi_cap, num_layers):
        return hi_cap
    lo_ok, hi_bad = hi, min(nxt, hi_cap)
    # Bisect on the 32-token grid.
    while hi_bad - lo_ok > 32:
        mid = ((lo_ok + hi_bad) // 2 // 32) * 32
        if mid <= lo_ok:
            break
        if _try_allocate(mesh_device, mid, num_layers):
            logger.info(f"  fits: {mid}")
            lo_ok = mid
        else:
            logger.info(f"  OOM:  {mid}")
            hi_bad = mid
    return lo_ok


def _load_unified_model(mesh_device, num_text_layers: int, num_vision_layers: int, load_seq: int):
    """Load the real vision+projector+text model onto the mesh, exactly like
    tt_demo_agent, at a small max_seq_len (so its own KV cache is negligible).
    Returns the orchestrator (kept alive so its DRAM stays resident)."""
    from transformers import AutoConfig
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding

    from models.experimental.mistral_small_4_119b.tt.mistral3_for_conditional_generation_unified import (
        TtMistral3ForConditionalGenerationUnified,
    )
    from models.experimental.mistral_small_4_119b.tt_demo_agent import (
        _precompute_rope_table,
        _state_dict_prefixes,
    )
    from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

    cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    text_cfg = cfg.text_config
    for attr in ("attn_implementation", "_attn_implementation"):
        if hasattr(text_cfg, attr):
            setattr(text_cfg, attr, "eager")
    image_token_id = int(getattr(cfg, "image_token_index", 10))

    logger.info(f"Loading state dict (text_layers={num_text_layers}, vision_layers={num_vision_layers})…")
    state_dict = load_hf_state_dict_filtered(
        HF_MODEL_ID, _state_dict_prefixes(num_text_layers, True, num_vision_layers)
    )
    cos_full, sin_full = _precompute_rope_table(Mistral4RotaryEmbedding, text_cfg, load_seq)

    logger.info("Building unified orchestrator + loading vision/projector/text weights…")
    orch = TtMistral3ForConditionalGenerationUnified(
        mesh_device=mesh_device,
        state_dict=state_dict,
        text_config=text_cfg,
        image_token_id=image_token_id,
        num_text_layers=num_text_layers,
        num_vision_layers=num_vision_layers,
        max_seq_len=load_seq,
        vision_dtype=ttnn.bfloat8_b,
    )
    orch.load_text()  # forces _load_vision_and_text (vision + projector + text resident)
    orch.cache_rope_tables(cos_full, sin_full)
    del state_dict
    return orch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-layers", type=int, default=EXPECTED_NUM_LAYERS, help="decoder layers (default 36)")
    ap.add_argument("--reserve-gb", type=float, default=0.0, help="per-device weights+activation headroom to reserve")
    ap.add_argument("--lo", type=int, default=4096, help="known-good lower bound to start from")
    ap.add_argument("--hi-cap", type=int, default=262144, help="ceiling to probe (HF model max is 256k)")
    ap.add_argument("--no-search", action="store_true", help="print analytical budget only; skip device probing")
    ap.add_argument(
        "--load-model",
        action="store_true",
        help="load the real vision+text model first, then measure the cache that fits on top",
    )
    ap.add_argument("--n-text-layers", type=int, default=EXPECTED_NUM_LAYERS, help="text layers to load (default 36)")
    ap.add_argument("--n-vision-layers", type=int, default=VISION_NUM_LAYERS, help="vision layers to load (default 24)")
    ap.add_argument("--load-seq", type=int, default=512, help="tiny max_seq_len to load the model at (its own cache)")
    args = ap.parse_args()

    bpt = _kv_bytes_per_token(args.num_layers)
    logger.info("=" * 64)
    logger.info(f"KV cache (replicated, per device): {bpt / _MiB:.4f} MiB / token")
    logger.info(
        f"  = {args.num_layers} layers × kvpe ({KV_LORA_RANK} latent + {QK_ROPE_HEAD_DIM} rope) × bf16, 1 shared KV head"
    )
    logger.info("Analytical per-device cache footprint:")
    for s in (4096, 32768, 131072, 262144, 524288, 1048576):
        logger.info(f"  {s:>7} tokens → {s * bpt / _GiB:7.2f} GiB")
    logger.info("=" * 64)

    if args.no_search:
        return

    mesh_device, fabric_set = _open_mesh()
    orch = None
    try:
        if args.load_model:
            orch = _load_unified_model(mesh_device, args.n_text_layers, args.n_vision_layers, args.load_seq)

        view = _dram_view(mesh_device)
        if view:
            total, free = view
            tag = "after model load" if args.load_model else "empty device"
            logger.info(f"Per-device DRAM: {total / _GiB:.2f} GiB total, {free / _GiB:.2f} GiB free ({tag})")
            if args.load_model:
                logger.info(
                    f"  → weights+vision resident: ~{(total - free) / _GiB:.2f} GiB/device "
                    f"(text_layers={args.n_text_layers}, vision_layers={args.n_vision_layers})"
                )
            usable = free - int(args.reserve_gb * _GiB)
            logger.info(
                f"Theoretical max (≈{usable / _GiB:.1f} GiB usable / {bpt / _MiB:.4f} MiB/tok): "
                f"~{max(0, int(usable / bpt)):,} tokens"
            )

        placeholder = _reserve(mesh_device, args.reserve_gb)
        logger.info(f"Binary-searching empirical max (lo={args.lo}, cap={args.hi_cap})…")
        best = _binary_search(mesh_device, args.num_layers, args.lo, args.hi_cap)
        if placeholder is not None:
            ttnn.deallocate(placeholder)

        logger.info("=" * 64)
        if args.load_model:
            logger.info(f"REAL MAX max_seq_len (model loaded, reserve={args.reserve_gb} GiB): {best:,} tokens")
            logger.info("This is the largest --max-seq-len you can load the model with. Note: leave")
            logger.info("headroom for prefill activation peaks — derate ~10-15% for safety.")
            logger.info("Sequence budget = text_tokens + Σ image_tokens + max_new_tokens must fit this.")
            logger.info("Per full-res 1540×1540 image ≈ 3025 tokens; tokens/image = (H×W)/784.")
        else:
            logger.info(f"EMPIRICAL MAX max_seq_len (cache-only, empty device): {best:,} tokens")
            logger.info("This ignores model weights. Use --load-model for the real number.")
        logger.info(f"  cache footprint at that length: {best * bpt / _GiB:.2f} GiB/device")
        logger.info("=" * 64)
    finally:
        _close_mesh(mesh_device, fabric_set)


if __name__ == "__main__":
    main()
