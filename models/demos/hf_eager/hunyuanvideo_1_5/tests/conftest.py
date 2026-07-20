# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Local `mesh_device` fixture override for the HunyuanVideo-1.5 mesh tests.

WHY THIS EXISTS
---------------
These tests (`test_hunyuan_video15_transformer_block_mesh`, `test_stage2b_gen_qb2`)
request a 4-device mesh via `@pytest.mark.parametrize("mesh_device", [4])`. The
root `mesh_device` fixture services that by calling
`ttnn.open_mesh_device(MeshShape(1, 4))` DIRECTLY. On the dedicated 4-chip QB2 the
tests were authored for that is the whole board, so FABRIC_1D trains fine. But on a
larger Blackhole Galaxy (e.g. the 32-chip 8x4 system) a direct small-submesh open
FAILS to initialize fabric -- "Fabric Router Sync: Timeout ... Ethernet handshake"
-- because **fabric cannot be initialized on only a subset of the system's
devices** (see the identical note in `models/common/tests/conftest.py`).

The fix, mirroring that same battle-tested conftest: open the FULL system mesh
(which trains every fabric link) and then `create_submesh(...)` the requested view.

This override is a strict superset of the root behavior and is GUARDED: when the
system mesh already equals the requested size (a real 4-chip QB2), it opens
directly exactly as before -- so it changes nothing on the original target.
"""
import os

import pytest

from tests.scripts.common import get_updated_device_params


def _requested_shape(param):
    """Map the fixture param to the requested (rows, cols), matching the root
    `mesh_device` fixture: an int N -> (1, N); a 2-tuple -> that grid."""
    if isinstance(param, (tuple, list)):
        assert len(param) == 2, "mesh grid shape must have exactly two elements"
        return (int(param[0]), int(param[1]))
    return (1, int(param))


def _pick_parent_shape(system_shape, requested_shape):
    """Choose the full-mesh orientation whose FIRST submesh equals the request.
    Mirrors `models/common/tests/conftest.py::_pick_parent_shape_for_submesh`."""
    if requested_shape == (1, 1):
        return (1, 1)
    sys_n = system_shape[0] * system_shape[1]
    req_n = requested_shape[0] * requested_shape[1]
    if req_n == sys_n:  # request uses every device -> open the mesh in that view
        return requested_shape
    if requested_shape[0] <= system_shape[0] and requested_shape[1] <= system_shape[1]:
        return system_shape
    rotated = (system_shape[1], system_shape[0])
    if requested_shape[0] <= rotated[0] and requested_shape[1] <= rotated[1]:
        return rotated
    pytest.skip(
        f"requested submesh {requested_shape} does not fit within system mesh "
        f"{system_shape} (or rotated {rotated}) with default offset"
    )


@pytest.fixture(scope="function")
def mesh_device(request, silicon_arch_name, device_params):
    import ttnn

    request.node.pci_ids = ttnn.get_pcie_device_ids()

    try:
        param = request.param
    except (ValueError, AttributeError):
        param = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape().mesh_size()

    req_shape = _requested_shape(param)
    # Optional override for bring-up experiments (e.g. an 8-chip (1,8) DiT for
    # higher frame counts): HY_MESH="1,8" or "1x8" or "8". Backward compatible --
    # inactive unless set, so committed parametrize decorators are unaffected.
    # NOTE: for the DiT's flat head-TP, use (1,8) NOT (2,4): the row-parallel
    # all-reduce is on mesh_axis=1, so all TP devices must lie on axis=1. A (2,4)
    # split puts 4 devices on axis=1 and silently drops the other row's heads ->
    # correct-looking PCC on one block but NOISE end-to-end. (1,8) uses the rotated
    # (4,8) parent so all 8 devices are on axis=1.
    _mesh_env = os.environ.get("HY_MESH")
    if _mesh_env:
        _parts = [int(x) for x in _mesh_env.replace("x", ",").split(",") if x.strip()]
        req_shape = _requested_shape(tuple(_parts) if len(_parts) == 2 else _parts[0])
    sys_shape = tuple(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
    parent_shape = _pick_parent_shape(sys_shape, req_shape)

    updated_device_params = get_updated_device_params(device_params)
    updated_device_params.pop("require_exact_physical_num_devices", False)
    fabric_config = updated_device_params.pop("fabric_config", None)
    updated_device_params.pop("fabric_tensix_config", None)
    updated_device_params.pop("reliability_mode", None)
    updated_device_params.pop("fabric_manager", None)
    updated_device_params.pop("fabric_router_config", None)

    # Fabric must be set BEFORE opening the mesh. Use the same defaults as
    # `models/common/tests/conftest.py` (STRICT_INIT, tensix DISABLED).
    if fabric_config and parent_shape != (1, 1):
        ttnn.set_fabric_config(
            fabric_config, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
        )

    parent_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*parent_shape), **updated_device_params)
    submesh_device = None
    # Stash the VAE submesh on the vae_decoder MODULE (not this conftest): pytest
    # loads conftest under a private name, so a conftest global isn't visible to the
    # test via the package path, but the module below is imported identically by both.
    from models.demos.hf_eager.hunyuanvideo_1_5.tt import qwen_encoder as _qe
    from models.demos.hf_eager.hunyuanvideo_1_5.tt import vae_decoder as _vd

    _vd.HY_VAE_SUBMESH = None
    _qe.HY_QWEN_SUBMESH = None
    if req_shape != parent_shape:
        # Fabric is live on the FULL parent; the submesh is just a device-subset
        # view over already-trained links.
        submesh_device = parent_device.create_submesh(ttnn.MeshShape(*req_shape))
        want_vae = os.environ.get("HY_TT_VAE", "0") == "1"
        want_qwen = os.environ.get("HY_TT_QWEN", "0") == "1"
        rr, rc = req_shape
        # All-four-modules-on-device 3-way split: pack VAE and Qwen into single ROWS
        # below the DiT so all fit on one board. Qwen is capped at TP=4 (28 attn / 4
        # kv heads) so it takes a (1,4) row; VAE takes a (1,rc) row (tile-sharded).
        # e.g. DiT sp=2 (2,8) rows 0-1 -> VAE (1,8) row 2 -> Qwen (1,4) row 3.
        if want_vae and want_qwen and rr + 2 <= parent_shape[0] and parent_shape[1] >= 4:
            _vd.HY_VAE_SUBMESH = parent_device.create_submesh(ttnn.MeshShape(1, rc), offset=ttnn.MeshCoordinate(rr, 0))
            _qe.HY_QWEN_SUBMESH = parent_device.create_submesh(
                ttnn.MeshShape(1, 4), offset=ttnn.MeshCoordinate(rr + 1, 0)
            )
        elif want_vae and want_qwen and rr + 1 <= parent_shape[0] and parent_shape[1] >= 8:
            # Only ONE row left below the DiT (e.g. DiT sp=3 (3,8) uses rows 0-2):
            # put VAE (1,4) and Qwen (1,4) SIDE-BY-SIDE in that row -- all disjoint,
            # no chip is in two mesh contexts (overlapping compute deadlocks ttnn).
            _vd.HY_VAE_SUBMESH = parent_device.create_submesh(ttnn.MeshShape(1, 4), offset=ttnn.MeshCoordinate(rr, 0))
            _qe.HY_QWEN_SUBMESH = parent_device.create_submesh(ttnn.MeshShape(1, 4), offset=ttnn.MeshCoordinate(rr, 4))
        else:
            # Single extra stage: carve a same-shape submesh on the next block of chips.
            if want_vae and rr * 2 <= parent_shape[0]:
                _vd.HY_VAE_SUBMESH = parent_device.create_submesh(
                    ttnn.MeshShape(*req_shape), offset=ttnn.MeshCoordinate(rr, 0)
                )
            if want_qwen and rr * 3 <= parent_shape[0]:
                _qe.HY_QWEN_SUBMESH = parent_device.create_submesh(
                    ttnn.MeshShape(*req_shape), offset=ttnn.MeshCoordinate(rr * 2, 0)
                )
    yielded = submesh_device if submesh_device is not None else parent_device
    from loguru import logger

    logger.debug(
        f"hunyuan mesh_device: system={sys_shape} parent={parent_shape} requested={req_shape} "
        f"-> {yielded.get_num_devices()} devices {list(yielded.get_device_ids())}"
    )

    yield yielded

    _vd.HY_VAE_SUBMESH = None
    _qe.HY_QWEN_SUBMESH = None
    if submesh_device is not None:
        ttnn.close_mesh_device(submesh_device)
    for sm in parent_device.get_submeshes():
        ttnn.close_mesh_device(sm)
    ttnn.close_mesh_device(parent_device)
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    del parent_device
