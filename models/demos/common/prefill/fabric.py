# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Payload sizing for prefill MoE dispatch/combine.

Keep dimension calculations import-safe: model configs are also imported by host-only
producers. Architecture detection is deferred until the fabric router is configured.
"""


MOE_FABRIC_DTYPE_SIZE_BYTES = 2
MOE_FABRIC_HEADER_SIZE_BYTES = 64
FABRIC_MAX_PAYLOAD_SIZE_BYTES = {"wormhole_b0": 7616, "blackhole": 15232}


def moe_fabric_payload_size(embedding_size: int) -> int:
    """Bytes needed for one BF16 embedding row and its routing header.

    Every prefill MoE model and test derives its router payload from this formula
    so dispatch, ordinary combine, and combine_fabric2d use one consistent size.
    """
    if embedding_size <= 0:
        raise ValueError("Embedding size must be positive")
    return embedding_size * MOE_FABRIC_DTYPE_SIZE_BYTES + MOE_FABRIC_HEADER_SIZE_BYTES


def limit_fabric_payload_size(payload_size: int, arch: str) -> int:
    """Cap a requested payload at Metal's architecture-specific fabric limit.

    Limits mirror FabricEriscDatamoverBuilder in
    tt_metal/fabric/erisc_datamover_builder.hpp. Larger rows are fragmented by
    the ordinary dispatch/combine send helpers.
    """
    if payload_size <= 0:
        raise ValueError("Fabric payload size must be positive")
    return min(payload_size, FABRIC_MAX_PAYLOAD_SIZE_BYTES[arch])


def create_fabric_router_config(max_payload_size: int):
    """Create a router config for the requested bytes on the current architecture."""
    import ttnn

    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = limit_fabric_payload_size(max_payload_size, ttnn.get_arch_name())
    return config
