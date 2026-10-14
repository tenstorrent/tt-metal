# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""What dispatch_fabric2d and combine_fabric2d need from the mesh, the fabric and the model.

Shared by TtDispatch2dModule, TtCombine2dModule and TtMoe, and by init_helpers.moe_fabric_payload_size,
which sizes the fabric packet a model's mesh opens with before any of them is built.
"""

import ttnn
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

# Must match FORWARDING_METADATA_SIZE in the kernel interface headers of dispatch_fabric2d and
# combine_fabric2d: the routing tail each forwarded packet carries on top of its token bytes.
FABRIC2D_FORWARDING_METADATA_BYTES = 64
# Largest top-k dispatch_fabric2d compiles for (a static_assert in dispatch_fabric2d_routing_index.hpp).
# combine_fabric2d has no such limit.
DISPATCH_FABRIC2D_MAX_TOPK = 8
BF16_BYTES = 2
# The largest DRAM alignment (Blackhole). dispatch_fabric2d rounds each token up to a multiple of this;
# combine_fabric2d requires the token to already be one.
DRAM_ALIGNMENT_BYTES = 64


def fabric2d_packet_bytes(emb_dim: int) -> int:
    """Bytes dispatch_fabric2d sends in one fabric packet: one whole token of emb_dim bf16 values plus the
    forwarding metadata."""
    # Rounded up as dispatch_fabric2d rounds it.
    token_page_bytes = (emb_dim * BF16_BYTES + DRAM_ALIGNMENT_BYTES - 1) // DRAM_ALIGNMENT_BYTES * DRAM_ALIGNMENT_BYTES
    return token_page_bytes + FABRIC2D_FORWARDING_METADATA_BYTES


def combine_fabric2d_payload_size(emb_dim: int) -> int:
    """The smallest fabric payload combine_fabric2d, standalone or overlapped with the routed expert, runs
    on: a bf16 token that does not fit one packet goes as two halves, the routing tail on the second."""
    return emb_dim * BF16_BYTES // 2 + FABRIC2D_FORWARDING_METADATA_BYTES


def fabric2d_payload_size(model_cfg) -> int:
    """Fabric max payload that fits dispatch_fabric2d's whole-token packet for this model.

    For op tests only. A model's mesh opens with init_helpers.moe_fabric_payload_size, half a token: a
    whole-token payload leaves the fabric routers fewer buffer slots and regresses the MLA CCLs.
    """
    routed_emb_dim = getattr(model_cfg, "ROUTED_EXPERT_HIDDEN_SIZE", None) or model_cfg.EMB_SIZE
    return max(model_cfg.FABRIC_PAYLOAD_SIZE, fabric2d_packet_bytes(routed_emb_dim))


def check_fabric2d_setup(mesh_device, cluster_axis: int, emb_dim: int, num_experts_per_tok: int | None = None) -> None:
    """Raise ValueError if dispatch_fabric2d / combine_fabric2d cannot run on this mesh and fabric.

    Both ops send bf16 tokens one hop at a time around a ring on `cluster_axis`. Both check the ring
    size and the packet size when they launch. dispatch_fabric2d also checks that the axis wraps;
    combine_fabric2d does not, so this is its only ring check.

    Pass `num_experts_per_tok` for dispatch_fabric2d only: it checks the top-k limit, which combine does
    not have, and a payload of a whole token, where combine needs only half of one.
    """
    if num_experts_per_tok is not None and num_experts_per_tok > DISPATCH_FABRIC2D_MAX_TOPK:
        raise ValueError(
            f"dispatch_fabric2d supports top-k up to {DISPATCH_FABRIC2D_MAX_TOPK}, "
            f"got num_experts_per_tok={num_experts_per_tok}"
        )
    extent = mesh_device.shape[cluster_axis]
    # Even, because the ops split the traffic to the opposite chip across both directions.
    if extent < 4 or extent % 2 != 0:
        raise ValueError(
            f"fabric2d dispatch/combine need an even ring of at least 4 chips on cluster_axis {cluster_axis}, "
            f"got {extent}"
        )
    fabric_config = ttnn.get_fabric_config()
    # FABRIC_1D_RING wraps axis 0 too, but the ops build 2D fabric routes, so it does not work.
    wraps = (
        fabric_config != ttnn.FabricConfig.FABRIC_1D_RING
        and per_axis_topology(fabric_config)[cluster_axis] == ttnn.Topology.Ring
    )
    if not wraps:
        raise ValueError(
            f"fabric2d dispatch/combine need a 2D fabric that wraps cluster_axis {cluster_axis} into a ring, "
            f"got {fabric_config}"
        )
    max_payload = ttnn.get_tt_fabric_max_payload_size_bytes()
    if num_experts_per_tok is None:
        packet_bytes = combine_fabric2d_payload_size(emb_dim)
        if max_payload < packet_bytes:
            raise ValueError(
                f"combine_fabric2d sends half a {emb_dim * BF16_BYTES} B token plus "
                f"{FABRIC2D_FORWARDING_METADATA_BYTES} B of forwarding metadata ({packet_bytes} B) per packet, but "
                f"the fabric max payload is {max_payload} B. Open the mesh with moe_fabric_payload_size."
            )
        return
    packet_bytes = fabric2d_packet_bytes(emb_dim)
    if max_payload < packet_bytes:
        raise ValueError(
            f"dispatch_fabric2d sends a whole {packet_bytes - FABRIC2D_FORWARDING_METADATA_BYTES} B token plus "
            f"{FABRIC2D_FORWARDING_METADATA_BYTES} B of forwarding metadata ({packet_bytes} B) in one packet, but "
            f"the fabric max payload is {max_payload} B. A model mesh stays at half a token, since a whole-token "
            f"payload regresses the MLA CCLs, so dispatch_fabric2d cannot run in a model until it splits a token "
            f"into two packets as combine_fabric2d does. Op tests can open the mesh with fabric2d_payload_size."
        )
