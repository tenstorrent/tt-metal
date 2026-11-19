# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The fabric the Galaxy tests open the mesh with, picked by environment (set before ``open_mesh_device``).

ML4_FABRIC        ring1d (default): FABRIC_1D_RING
                  torus_xy: FABRIC_2D_TORUS_XY with RELAXED_INIT (the 8x4 profile of GLM-5.3 and Kimi)
                  fabric2d: plain FABRIC_2D
ML4_FABRIC_PAYLOAD  max fabric packet payload in bytes (default 12288: the CCLs' 4096 B pages travel 3 to a packet
                  instead of 1 in the default 4352 B packet; 0 = the fabric's default)
"""

import os

DEFAULT_PAYLOAD = 12288
RUNNER_DEFAULT_PAYLOAD = 4352  # the fabric's default packet, which payload 0 asks for

_CONFIGS = {
    "ring1d": "FABRIC_1D_RING",
    "torus_xy": "FABRIC_2D_TORUS_XY",
    "fabric2d": "FABRIC_2D",
}


def payload() -> int:
    """ML4_FABRIC_PAYLOAD (default 12288; 0 = the fabric's default packet)."""
    return int(os.environ.get("ML4_FABRIC_PAYLOAD", str(DEFAULT_PAYLOAD)))


def set_from_env() -> dict:
    """Sets the fabric config from ML4_FABRIC / ML4_FABRIC_PAYLOAD; returns what was set (for the result json)."""
    import ttnn

    name = os.environ.get("ML4_FABRIC", "ring1d")
    payload_bytes = payload()
    cfg = getattr(ttnn.FabricConfig, _CONFIGS[name])
    if name == "ring1d" and not payload_bytes:
        ttnn.set_fabric_config(cfg)
    else:
        router = None
        if payload_bytes:
            router = ttnn._ttnn.fabric.FabricRouterConfig()
            router.max_packet_payload_size_bytes = payload_bytes
        args = [
            cfg,
            ttnn.FabricReliabilityMode.RELAXED_INIT if name != "ring1d" else ttnn.FabricReliabilityMode.STRICT_INIT,
            None,
            ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
            ttnn.FabricManagerMode.DEFAULT,
        ]
        if router is not None:
            args.append(router)
        ttnn.set_fabric_config(*args)
    return {"fabric": name, "fabric_payload": payload_bytes or None}
