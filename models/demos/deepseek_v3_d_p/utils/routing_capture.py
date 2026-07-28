# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Env-gated routing capture for the DeepSeek/Kimi chunked prefill tests.

Enable with TT_DS_CAPTURE_ROUTING=1. When off every function here is a no-op with
zero device work, so the (tiny) gated call sites are safe to leave in place.

Two things are captured, both keyed by MoE layer:

  * per-(chunk, layer, group) load  -- from expert_token_counts (post routing_setup,
    padding already applied). Used to rank heavy vs balanced dispatch groups.
  * raw gate top-k expert IDs        -- global IDs in [0, num_routed_experts). Accumulated
    per layer across chunks and written to an expert_routing.safetensors that
    load_captured_routing() / test_dispatch_combine can consume directly.

"group" == Galaxy dispatch group == mesh column. On kimi-mesh-8x4 there are
num_dispatch_groups=4 groups of dispatch_group_size=8 chips; the 256 routed experts
are assigned contiguously, so group g owns global expert IDs
[g*experts_per_group, (g+1)*experts_per_group) and per-group load is the number of
routed tokens landing in that block -- the same "in-col share" metric the perf test
ranks (layer, col) pairs by, now with a chunk axis.

Usage
-----
1) chunk loop (test_prefill_transformer_chunked.py), before each forward:
       routing_capture.set_chunk(c)
2) inside TtMoe.forward, after the routing_setup call:
       routing_capture.capture(
           tt_expert_token_counts, indices, self.layer_idx, self.mesh_device,
           self.num_routed_experts, self.num_dispatch_groups)
3) after the chunk loop:
       routing_capture.dump_top(4)
       routing_capture.save_safetensors("captured_expert_routing.safetensors")
"""

import os

import torch

_ENABLED = os.getenv("TT_DS_CAPTURE_ROUTING", "0").lower() in ("1", "true", "yes")
_load_records: list[dict] = []  # {chunk, layer, group, load}
_indices_by_layer: dict[int, list[torch.Tensor]] = {}  # layer -> [ (tokens, top_k) per chunk ]
_chunk: int = -1


def enabled() -> bool:
    return _ENABLED


def set_chunk(c: int) -> None:
    global _chunk
    _chunk = c


def _compose_ep(tt_tensor, mesh_device):
    """Read a device tensor to host with the same composer TtMoe uses for its debug
    token-count readback (MeshComposerConfig(dims=[1, 0]))."""
    import ttnn

    t4d = ttnn.unsqueeze_to_4D(tt_tensor)
    composer = ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig(dims=[1, 0]))
    return ttnn.to_torch(t4d, mesh_composer=composer)


def capture(
    tt_expert_token_counts,
    tt_gate_indices,
    layer_idx: int,
    mesh_device,
    num_routed_experts: int,
    num_dispatch_groups: int,
) -> None:
    """Record per-group load (from counts) and raw gate indices (for the safetensor)."""
    if not _ENABLED:
        return
    from loguru import logger

    experts_per_group = num_routed_experts // num_dispatch_groups

    # --- per-group load from expert_token_counts (group-major order) ---
    counts = _compose_ep(tt_expert_token_counts, mesh_device).squeeze().flatten().to(torch.int64)
    if counts.numel() >= num_routed_experts:
        for g in range(num_dispatch_groups):
            load = int(counts[g * experts_per_group : (g + 1) * experts_per_group].sum().item())
            _load_records.append({"chunk": _chunk, "layer": layer_idx, "group": g, "load": load})
    else:
        logger.warning(
            f"[routing_capture] layer {layer_idx}: got {counts.numel()} counts, expected >= {num_routed_experts}; load skipped."
        )

    # --- raw gate indices for the safetensor ---
    # indices are replicated across the group (mesh-col) axis and sharded across the SP (token,
    # mesh-row) axis. The [1,0] compose stacks num_dispatch_groups identical copies of the full
    # token set; fold the replicated group axis out and keep one copy.
    idx = _compose_ep(tt_gate_indices, mesh_device).squeeze().to(torch.int32).flatten()
    top_k = num_routed_experts  # placeholder; real top_k derived below
    per_group = idx.numel() // num_dispatch_groups
    idx = idx[:per_group].reshape(-1)  # one replica
    # top_k is the last gate dim; recover it from the original tensor's trailing shape.
    top_k = int(tt_gate_indices.shape[-1])
    if idx.numel() % top_k != 0:
        logger.warning(
            f"[routing_capture] layer {layer_idx}: index numel {idx.numel()} not divisible by top_k {top_k}; indices skipped."
        )
        return
    idx = idx.reshape(-1, top_k)
    _indices_by_layer.setdefault(layer_idx, []).append(idx.cpu())


def dump_top(top_n: int = 4) -> list[dict]:
    """Log the full per-(chunk, layer, group) table and the top-N hottest triples."""
    if not _ENABLED:
        return []
    from loguru import logger

    total = sum(r["load"] for r in _load_records) or 1
    ranked = sorted(_load_records, key=lambda r: r["load"], reverse=True)
    logger.info("[routing_capture] per-(chunk, layer, group) routing load:")
    for r in ranked:
        logger.info(
            f"  chunk={r['chunk']:>2} layer={r['layer']:>2} group={r['group']} "
            f"load={r['load']:>8} ({100.0 * r['load'] / total:.2f}%)"
        )
    top = ranked[:top_n]
    logger.info(f"[routing_capture] TOP {top_n} most-loaded (chunk, layer, group):")
    for rank, r in enumerate(top, 1):
        logger.info(f"  #{rank}: chunk={r['chunk']} layer={r['layer']} group={r['group']} load={r['load']}")
    return top


def save_safetensors(path: str) -> None:
    """Write accumulated raw gate indices as expert_ids_layer_<L> tensors.

    Output matches load_captured_routing's schema: one (total_tokens, top_k) int32 tensor
    per captured MoE layer, values = Galaxy-global expert IDs. Feed this file to
    test_dispatch_combine via TT_DS_USE_CAPTURED_INDICES / captured_indices_path, setting
    seq_len_per_chip = total_tokens / dispatch_group_size for the run you captured.
    """
    if not _ENABLED:
        return
    from loguru import logger
    from safetensors.torch import save_file

    tensors = {}
    for layer, chunks in sorted(_indices_by_layer.items()):
        tensors[f"expert_ids_layer_{layer}"] = torch.cat(chunks, dim=0).to(torch.int32).contiguous()
    if not tensors:
        logger.warning("[routing_capture] no indices captured; nothing to save.")
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    save_file(tensors, path)
    any_key = next(iter(tensors))
    logger.info(
        f"[routing_capture] wrote {len(tensors)} layers to {path} "
        f"(e.g. {any_key} shape={tuple(tensors[any_key].shape)})"
    )
