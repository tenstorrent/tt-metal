# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Mesh-device open/close for the standalone FLUX.2 server.

This mirrors the device-open recipe used by ``conftest.py`` (the ``set_fabric``
helper and the ``mesh_device`` fixture) so the server's fabric configuration is
byte-for-byte consistent with the tested path — but inlines the small helpers
instead of importing the conftest, which is fragile to import outside pytest.

CRITICAL fabric/topology consistency: the device fabric mode
(Linear -> ``FABRIC_1D``, Ring -> ``FABRIC_1D_RING``) MUST match the CCL topology
the pipeline uses. We open the device with the matching fabric dict here and pass
the same ``topology`` explicitly into ``create_pipeline`` so they can never disagree.

All ``ttnn`` imports are lazy (inside functions) so this module imports cleanly on
a machine with no device.
"""

from __future__ import annotations

import os

from loguru import logger

from .config import ServerConfig


# --- Inlined from conftest.py:set_fabric (lines 477-516) ----------------------
def _get_default_fabric_tensix_config():
    import ttnn

    return ttnn.FabricTensixConfig.DISABLED


def _set_fabric(
    fabric_config,
    reliability_mode=None,
    fabric_tensix_config=None,
    fabric_manager=None,
    fabric_router_config=None,
):
    import ttnn

    if fabric_config:
        if reliability_mode is None:
            reliability_mode = ttnn.FabricReliabilityMode.STRICT_INIT
        if fabric_tensix_config is None:
            fabric_tensix_config = _get_default_fabric_tensix_config()
        if fabric_manager is None:
            fabric_manager = ttnn.FabricManagerMode.DEFAULT
        if fabric_router_config is not None:
            ttnn.set_fabric_config(
                fabric_config,
                reliability_mode,
                None,
                fabric_tensix_config,
                ttnn.FabricUDMMode.DISABLED,
                fabric_manager,
                fabric_router_config,
            )
        else:
            ttnn.set_fabric_config(
                fabric_config,
                reliability_mode,
                None,
                fabric_tensix_config,
                ttnn.FabricUDMMode.DISABLED,
                fabric_manager,
            )


def _reset_fabric(fabric_config):
    import ttnn

    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _fabric_params(topology: str) -> dict:
    """Return the fabric device-params dict for a topology (mirrors
    ``models/tt_dit/utils/test.py`` ``line_params`` / ``ring_params``).

    Defined locally rather than imported so this module does not depend on
    ``utils/test`` importing cleanly (it touches ttnn at module scope).
    """
    import ttnn

    if topology == "ring":
        return {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


def warn_if_watcher_set() -> None:
    """``TT_METAL_WATCHER`` overflows the fabric-router kernel-config
    buffer on multi-chip fabric. The launcher unsets it; warn loudly if it leaks
    through (e.g. when launched without ``run_server.sh``)."""
    if os.environ.get("TT_METAL_WATCHER"):
        logger.warning(
            "TT_METAL_WATCHER is set — this overflows the fabric-router kernel-config "
            "buffer on multi-chip fabric and will likely crash device init. Unset it "
            "(run_server.sh does this automatically)."
        )


def open_mesh(cfg: ServerConfig):
    """Open the mesh device with the fabric matching ``cfg.topology``.

    Returns a tuple ``(mesh, fabric_config)``; ``fabric_config`` must be handed
    back to :func:`close_mesh` so the fabric can be reset on teardown.
    """
    import ttnn
    from tests.scripts.common import get_updated_device_params

    warn_if_watcher_set()

    rows, cols = cfg.mesh_shape
    device_params = dict(_fabric_params(cfg.topology))

    # Ring + traced needs a large trace region (mirrors the conftest path).
    if cfg.traced and cfg.topology == "ring":
        device_params["trace_region_size"] = 300_000_000

    # The VAE decoder uses conv2d, which needs L1 small buffers (matches the flux2 tests).
    device_params["l1_small_size"] = 65536

    updated = get_updated_device_params(device_params)
    fabric_config = updated.pop("fabric_config", None)
    fabric_tensix_config = updated.pop("fabric_tensix_config", None)
    reliability_mode = updated.pop("reliability_mode", None)
    fabric_manager = updated.pop("fabric_manager", None)
    fabric_router_config = updated.pop("fabric_router_config", None)

    logger.info(f"Opening mesh {rows}x{cols} with topology={cfg.topology} " f"(fabric_config={fabric_config})")
    _set_fabric(
        fabric_config,
        reliability_mode,
        fabric_tensix_config,
        fabric_manager,
        fabric_router_config,
    )
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols), **updated)
    logger.info(f"Mesh device opened: shape={tuple(mesh.shape)}")
    return mesh, fabric_config


def close_mesh(mesh, fabric_config) -> None:
    """Close all submeshes, the parent mesh, then reset the fabric."""
    import ttnn

    if mesh is not None:
        try:
            for submesh in mesh.get_submeshes():
                ttnn.close_mesh_device(submesh)
            ttnn.close_mesh_device(mesh)
            logger.info("Mesh device closed.")
        except Exception as exc:  # noqa: BLE001 — teardown must be best-effort
            logger.warning(f"Error closing mesh device: {exc}")
    _reset_fabric(fabric_config)
