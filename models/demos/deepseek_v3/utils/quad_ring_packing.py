# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for packing MoE expert weights into the quad-ring tile layout.

Used by both the runtime weight-conversion path (``Experts._convert_weights_quad_ring``)
and the offline checkpoint-preparation script (``save_quad_ring_hf_checkpoint``). Kept in
its own module so callers don't transitively import the heavier ``hf_model_utils`` graph.
"""

from __future__ import annotations

import torch


def quad_ring_shared_expert_to_device_map(
    n_routed_experts: int, n_shared_experts: int, n_devices: int
) -> dict[int, list[int]]:
    """Policy: a single shared expert sits at id ``n_routed + n_shared - 1`` and replicates everywhere."""
    shared_expert_id = n_routed_experts + n_shared_experts - 1
    return {shared_expert_id: list(range(n_devices))}


def default_quad_ring_ring2cores() -> dict[int, tuple[int, int, int]]:
    """Static ring2cores mirror of ``determine_compute_matmul_cores()`` for the 12-DRAM-bank layout.

    Use this when no live ``mesh_device`` is available (e.g. offline checkpoint prep).
    """
    matmul_pad_cores = {1, 2, 4, 5, 7, 8, 10, 11}
    return {ring_pos: (ring_pos, ring_pos, 1 if ring_pos in matmul_pad_cores else 0) for ring_pos in range(12)}


def prepare_quad_ring_packed_experts(
    *,
    routed_gate: torch.Tensor,
    routed_up: torch.Tensor,
    routed_down: torch.Tensor,
    num_routed_experts: int,
    num_devices: int,
    hidden_size: int,
    shared_gate: torch.Tensor | None = None,
    shared_up: torch.Tensor | None = None,
    shared_down: torch.Tensor | None = None,
    num_shared_experts: int = 0,
    shared_expert_ids_to_devices: dict[int, list[int]] | None = None,
    ring2cores: dict[int, tuple[int, int, int]] | None = None,
    num_layers: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack routed (and optional shared) expert weights into the quad-ring DRAM-bank layout.

    Inputs are the stacked HF tensors:
      - ``routed_*`` rank 3, shape ``(n_routed, *, *)`` (the HF "experts_stacked" layout).
      - ``shared_*`` rank 2, shape ``(out, in)`` — HF stores the shared MLP as a single
        unindexed module. Only ``num_shared_experts == 1`` is currently supported; pass
        ``num_shared_experts=0`` (and leave ``shared_*`` ``None``) for the routed-only path.

    Returns ``(prepared_w0_w1, prepared_w2)``, each shaped for ``shard_and_save`` consumption
    by the quad-ring expert weight config.
    """
    from ttnn._experimental.moe_compute_utils import (
        add_shared_expert_weights,
        get_shared_experts_per_device,
        prepare_w0_w1_tensor_for_moe_compute,
        prepare_w2_tensor_for_moe_compute,
    )

    if num_routed_experts % num_devices != 0:
        raise ValueError(
            f"n_routed_experts ({num_routed_experts}) must be divisible by num_devices ({num_devices}) "
            "for quad-ring expert preparation."
        )
    num_routed_experts_per_device = num_routed_experts // num_devices

    # Routed: (n_routed, out, in) -> (1, n_routed, in, out)
    w0 = routed_gate.unsqueeze(0).transpose(-1, -2)
    w1 = routed_up.unsqueeze(0).transpose(-1, -2)
    w2 = routed_down.unsqueeze(0).transpose(-1, -2)

    if w0.shape[-2] != hidden_size:
        raise ValueError(
            f"Expected gate expert input hidden size {hidden_size}, got {w0.shape[-2]} "
            f"(tensor shape: {tuple(w0.shape)})"
        )
    matmul_N = w0.shape[-1]

    if num_shared_experts > 0:
        if shared_gate is None or shared_up is None or shared_down is None or shared_expert_ids_to_devices is None:
            raise ValueError(
                "num_shared_experts > 0 requires shared_gate / shared_up / shared_down tensors "
                "and a shared_expert_ids_to_devices mapping."
            )
        # The HF shared MLP is unindexed. ``quad_ring_shared_expert_to_device_map`` emits a single
        # entry today; mirror the same constraint the runtime path asserts.
        if num_shared_experts != 1 or len(shared_expert_ids_to_devices) != 1:
            raise ValueError(
                f"Only n_shared_experts=1 is supported here (got {num_shared_experts}); "
                "the fat shared-MLP -> per-expert split is not implemented."
            )
        (shared_expert_id,) = shared_expert_ids_to_devices.keys()

        # Shared: (out, in) -> (1, 1, in, out)
        shared_w0 = {shared_expert_id: shared_gate.unsqueeze(0).unsqueeze(0).transpose(-1, -2)}
        shared_w1 = {shared_expert_id: shared_up.unsqueeze(0).unsqueeze(0).transpose(-1, -2)}
        shared_w2 = {shared_expert_id: shared_down.unsqueeze(0).unsqueeze(0).transpose(-1, -2)}

        w0, w1, w2 = add_shared_expert_weights(
            w0, w1, w2, shared_w0, shared_w1, shared_w2, shared_expert_ids_to_devices, num_devices
        )

        num_shared_experts_per_device = get_shared_experts_per_device(shared_expert_ids_to_devices, num_devices)[0]
    else:
        num_shared_experts_per_device = 0

    num_total_experts_per_device = num_routed_experts_per_device + num_shared_experts_per_device
    num_total_experts_on_devices = num_total_experts_per_device * num_devices

    if ring2cores is None:
        ring2cores = default_quad_ring_ring2cores()

    prepared_w0_w1: list[torch.Tensor] = []
    prepared_w2: list[torch.Tensor] = []
    for i in range(0, num_total_experts_on_devices, num_total_experts_per_device):
        prepared_w0_w1.append(
            prepare_w0_w1_tensor_for_moe_compute(
                w0[:, i : i + num_total_experts_per_device, :, :],
                w1[:, i : i + num_total_experts_per_device, :, :],
                num_layers,
                num_total_experts_per_device,
                hidden_size,
                matmul_N,
                ring2cores,
            )
        )
        prepared_w2.append(
            prepare_w2_tensor_for_moe_compute(
                w2[:, i : i + num_total_experts_per_device, :, :],
                num_layers,
                num_total_experts_per_device,
                matmul_N,
                hidden_size,
                ring2cores,
            )
        )

    return (
        torch.cat(prepared_w0_w1, dim=2).contiguous(),
        torch.cat(prepared_w2, dim=2).contiguous(),
    )
