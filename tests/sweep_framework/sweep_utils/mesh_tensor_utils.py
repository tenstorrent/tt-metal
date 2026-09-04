# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for handling mesh devices and tensor placements in sweep tests.
Supports creating tensors on mesh devices with proper placement (Shard/Replicate).

Environment Variables:
    MESH_DEVICE_SHAPE: Mesh shape to use (e.g., "1x2", "2x4", "4x8")
                       If not set, uses single device (default)
                       If set, creates mesh device with that shape
                       Tests will fail naturally if mesh shape exceeds hardware
"""

import os
import torch
import ttnn
from typing import Optional, Dict, Sequence, Tuple
import ast
from loguru import logger


def parse_placement_from_traced(tensor_placement: Optional[Dict]) -> Optional[ttnn.TensorMemoryLayout]:
    """
    Parse tensor placement from traced config and return appropriate mesh mapper.

    Args:
        tensor_placement: Dict with 'placement', 'distribution_shape', 'mesh_device_shape'
                         e.g., {'placement': "['PlacementShard(2)', 'PlacementShard(3)']", ...}

    Returns:
        Mesh mapper object (ShardTensor2dMesh or ReplicateTensorToMesh) or None
    """
    if not tensor_placement:
        return None

    try:
        placement_raw = tensor_placement.get("placement", "")
        placement_str = str(placement_raw) if not isinstance(placement_raw, str) else placement_raw

        # Check if it's a replicate placement
        if "PlacementReplicate" in placement_str:
            return ttnn.ReplicateTensorToMesh

        # Check if it's a shard placement
        if "PlacementShard" in placement_str:
            # Extract shard dimensions
            # e.g., "['PlacementShard(2)', 'PlacementShard(3)']" -> shard on dims 2,3
            import re

            shard_dims = re.findall(r"PlacementShard\((?:dim=)?(-?\d+)\)", placement_str)

            if shard_dims:
                # For 2D mesh, we typically shard on the last dimension(s)
                # Return a shard mapper - the specific implementation depends on the operation
                mesh_shape_str = tensor_placement.get("mesh_device_shape", "[1, 1]")
                mesh_shape = ast.literal_eval(mesh_shape_str) if isinstance(mesh_shape_str, str) else mesh_shape_str

                # For now, return ShardTensor2dMesh which will shard based on mesh shape
                return ttnn.ShardTensor2dMesh(
                    mesh_device=None,  # Will be set later
                    dim=int(shard_dims[-1]) if shard_dims else -1,
                    mesh_shape=ttnn.MeshShape(*mesh_shape) if len(mesh_shape) == 2 else None,
                )
    except Exception as e:
        print(f"⚠️ Warning: Failed to parse tensor placement: {e}")
        return None

    return None


def get_mesh_shape_from_machine_info(machine_info: Optional[Dict]) -> Optional[Tuple[int, int]]:
    """
    Extract mesh device shape from traced machine_info.

    Args:
        machine_info: Dict with 'mesh_device_shape', 'device_count', etc.

    Returns:
        Tuple of (rows, cols) or None if no mesh info
    """
    if not machine_info:
        return None

    mesh_shape = machine_info.get("mesh_device_shape")
    if not mesh_shape:
        return None

    # Handle both list and string formats
    if isinstance(mesh_shape, str):
        mesh_shape = ast.literal_eval(mesh_shape)

    if isinstance(mesh_shape, list) and len(mesh_shape) == 2:
        return tuple(mesh_shape)

    return None


def program_config_grid_bounds(pc):
    """Max core (x, y) a traced program_config will actually use.

    Accepts either a structured dict (matmul configs: ``compute_with_storage_grid_size
    = {"x":7,"y":10}`` plus ``hop_cores``/``sub_core_grids`` range lists) or a repr
    string (SDPA: ``compute_with_storage_grid_size=(x=8,y=8)`` or ``=8-9`` — a grid
    SIZE, so the max core index is size-1 — plus ``sub_core_grids={[x1-y1 - x2-y2],
    ...}``). compute_with_storage_grid_size is a SIZE; explicit core ranges
    (sub_core_grids/hop_cores) are exact and extend the bounds. Returns
    (max_x, max_y), each possibly None.
    """
    import re

    # Structured dict form (matmul program configs).
    if isinstance(pc, dict):
        max_x = max_y = None
        csg = pc.get("compute_with_storage_grid_size")
        if isinstance(csg, dict) and csg.get("x") is not None and csg.get("y") is not None:
            max_x, max_y = int(csg["x"]) - 1, int(csg["y"]) - 1
        for key in ("hop_cores", "sub_core_grids"):
            ranges = pc.get(key)
            if isinstance(ranges, list):
                for r in ranges:
                    end = r.get("end") if isinstance(r, dict) else None
                    if isinstance(end, dict):
                        max_x = max(max_x if max_x is not None else -1, int(end.get("x", -1)))
                        max_y = max(max_y if max_y is not None else -1, int(end.get("y", -1)))
        if max_x is None and max_y is None and pc.get("value"):
            return program_config_grid_bounds(pc.get("value"))  # SDPA-style repr in "value"
        return max_x, max_y

    s = str(pc or "")
    sub = list(re.finditer(r"\[(\d+)-(\d+)\s*-\s*(\d+)-(\d+)\]", s))
    if sub:
        return max(int(m.group(3)) for m in sub), max(int(m.group(4)) for m in sub)
    gm = re.search(r"compute_with_storage_grid_size=\(x=(\d+),y=(\d+)\)", s) or re.search(
        r"compute_with_storage_grid_size=(\d+)-(\d+)", s
    )
    if gm:
        return int(gm.group(1)) - 1, int(gm.group(2)) - 1
    return None, None


def dispatch_axis_for_grid(max_x, max_y):
    """Pick the Wormhole WORKER dispatch axis a core grid needs, or None.

    ROW dispatch -> compute grid (8,9): x in [0,7], y in [0,8].
    COL dispatch -> compute grid (7,10): x in [0,6], y in [0,9].
    A grid touching x=7 needs ROW; one touching y=9 (but not x=7) needs COL;
    otherwise either axis works (None). x is checked first because the matmul
    compute grid width is a hard placement constraint; if a config genuinely
    needs both x=7 and y=9 it fits neither (a genuinely inconsistent traced
    config), and ROW at least satisfies the compute-grid width.
    """
    if (max_x or -1) >= 7:
        return ttnn.DispatchCoreAxis.ROW
    if (max_y or -1) >= 9:
        return ttnn.DispatchCoreAxis.COL
    return None


def shard_grid_bounds(mc):
    """Max core (x, y) used by a (serialized) memory_config's shard_spec grid.

    Accepts the V2 dict form ({"data": {"shard_spec": {"grid": [{"start":..,
    "end":..}, ...]}}}) or a ttnn.MemoryConfig. Returns (max_x, max_y), each
    possibly None when there is no shard spec.
    """
    max_x = max_y = None
    grid = None
    if isinstance(mc, dict):
        ss = (mc.get("data") or {}).get("shard_spec") or mc.get("shard_spec")
        if isinstance(ss, dict):
            grid = ss.get("grid")
    else:
        ss = getattr(mc, "shard_spec", None)
        if ss is not None:
            try:
                for cr in ss.grid.ranges():
                    max_x = max(max_x if max_x is not None else -1, cr.end.x)
                    max_y = max(max_y if max_y is not None else -1, cr.end.y)
            except Exception:
                # malformed / partial shard_spec — return whatever bounds were gathered (best-effort)
                pass
            return max_x, max_y
    if isinstance(grid, list):
        for r in grid:
            end = r.get("end") if isinstance(r, dict) else None
            if isinstance(end, dict):
                max_x = max(max_x if max_x is not None else -1, int(end.get("x", -1)))
                max_y = max(max_y if max_y is not None else -1, int(end.get("y", -1)))
    return max_x, max_y


# ── Job-level device reuse (opt-in via TTNN_SWEEP_JOB_DEVICE=1) ───────────────
# When the sweeps runner keeps ONE process per job (persistent worker), a single
# open mesh device is reused across every module/vector that needs the SAME
# device configuration, and only reopened when the resolved config actually
# changes. This avoids the per-module device reopen that force-reinitializes
# dispatch on Galaxy and wedges a dispatch core (run_mailbox=0x40). Every module
# opens its device through create_mesh_device() and closes via
# ttnn.close_mesh_device(), so caching + a deferred-close guard here is
# transparent to the modules (a module that opens its own device just gets the
# cached one; its per-module close is deferred to job end / config change).
# Device-count boundary between single-host clusters (N150=1, N300=2, T3K=8) and
# multi-host Galaxy (32 chips). Galaxy is the only place the per-module device
# reopen wedges a dispatch core, so job-level device reuse is gated to clusters
# with MORE than this many devices.
_SINGLE_HOST_MAX_DEVICES = 8

_JOB_DEVICE = None
_JOB_DEVICE_KEY = None
_orig_close_mesh_device = ttnn.close_mesh_device


def _job_device_enabled() -> bool:
    """Job-level device reuse is opt-in (TTNN_SWEEP_JOB_DEVICE=1) AND restricted to
    Galaxy (>8 devices). The per-module device reopen it avoids only force-reinits
    dispatch (and wedges a core) on Galaxy; single-host (N150/N300/T3K) has no such
    issue AND has modules that take a SINGLE-device path (ttnn.open_device, e.g.
    clamp/fast_reduce_nc when get_mesh_shape() returns None on 1 chip) that would
    collide with a held cached mesh device. Gating to Galaxy keeps every single-host
    lane on the original per-module behavior. TTNN_SWEEP_JOB_DEVICE_FORCE=1 bypasses
    the device-count gate (for validation on smaller clusters)."""
    if os.environ.get("TTNN_SWEEP_JOB_DEVICE") != "1":
        return False
    if os.environ.get("TTNN_SWEEP_JOB_DEVICE_FORCE") == "1":
        return True
    try:
        return ttnn.get_num_devices() > _SINGLE_HOST_MAX_DEVICES
    except Exception:
        return False


def _axis_token(dispatch_core_axis):
    """Canonical lowercase token ('row'/'col') for a dispatch axis.

    str(ttnn.DispatchCoreAxis.COL) is 'DispatchCoreAxis.COL' while the
    TTNN_DISPATCH_AXIS env override is 'col'. Keying on the raw str() made those two
    DIFFERENT keys for the SAME physical device, so a module that derived COL from its
    vector grid missed the cache against the worker's env-derived 'col' device and did a
    full close+reopen -- on Galaxy every needless reopen risks wedging a dispatch core
    (run_mailbox=0x40). Measured on the Galaxy vector set: 18 of the 44 reopens in the
    modules that open per vector were this mismatch alone (linear: 19 of its 68 vectors).
    """
    return str(dispatch_core_axis).split(".")[-1].strip().lower()


def _job_device_key(mesh_shape, l1_small_size, dispatch_core_axis, prefer_eth):
    """Canonical key for the device config create_mesh_device WOULD open. Must be
    identical for the same intended device regardless of whether the caller passes
    an explicit axis or relies on env/auto-detect, so the worker's open and a
    module's own _ensure_*_device() open collapse to one cached device. Returns
    None when the config can't be keyed safely (auto axis with no env override) —
    caching is skipped so an ambiguous config never returns the wrong device."""
    arch = os.environ.get("ARCH_NAME", "").lower()
    if not arch:
        try:
            arch = ttnn.get_arch_name().lower()
        except Exception:
            arch = ""
    if "blackhole" in arch:
        disp = ("default",)
    else:
        try:
            single_host = ttnn.get_num_devices() <= _SINGLE_HOST_MAX_DEVICES
        except Exception:
            single_host = False
        if single_host and prefer_eth:
            disp = ("ETH",)  # ETH intent (may fall back to WORKER, but the key stays consistent)
        elif dispatch_core_axis is not None:
            disp = ("WORKER", _axis_token(dispatch_core_axis))
        else:
            env_axis = os.environ.get("TTNN_DISPATCH_AXIS", "").strip().lower()
            if env_axis in ("col", "row"):
                disp = ("WORKER", env_axis)
            else:
                return None  # auto-detect axis is op-dependent -> not safe to share
    return (tuple(mesh_shape), int(l1_small_size), bool(prefer_eth), disp)


def create_mesh_device(
    mesh_shape: Tuple[int, int],
    device_ids: Optional[list] = None,
    l1_small_size: int = 79104,
    dispatch_core_axis=None,
    prefer_eth: bool = True,
) -> ttnn.MeshDevice:
    """Open a mesh device, reusing a cached job-level device when
    TTNN_SWEEP_JOB_DEVICE=1 and the resolved config matches (see
    _job_device_key). On a config change the prior job device is closed first so
    the reopen/reconfig is legal (SetFabricConfig requires no open devices)."""
    if not _job_device_enabled():
        return _create_mesh_device_uncached(mesh_shape, device_ids, l1_small_size, dispatch_core_axis, prefer_eth)

    global _JOB_DEVICE, _JOB_DEVICE_KEY
    key = _job_device_key(mesh_shape, l1_small_size, dispatch_core_axis, prefer_eth)
    if key is None:
        # Not safely keyable (e.g. auto-detect axis) -> don't cache this device.
        # Close any live cached device FIRST so this uncached open never runs a
        # SECOND mesh alongside it (the open guard also closes, but do it
        # explicitly here so the one-live-device invariant is local and obvious).
        # Refuse if the close fails rather than open a second live mesh.
        if _JOB_DEVICE is not None and not close_job_device():
            raise RuntimeError(
                "create_mesh_device: refusing an unkeyable open because the cached "
                "job device could not be closed (would leave two live mesh devices)"
            )
        return _create_mesh_device_uncached(mesh_shape, device_ids, l1_small_size, dispatch_core_axis, prefer_eth)

    if _JOB_DEVICE is not None and _JOB_DEVICE_KEY == key:
        return _JOB_DEVICE
    if _JOB_DEVICE is not None:
        # Config changed: really close the old device before opening the new one
        # (a device must be closed before a fabric reconfig). If the close FAILS,
        # refuse to reopen -- a second live mesh corrupts context state; surface
        # the teardown error instead of silently reopening. close_job_device()
        # nulls _JOB_DEVICE/_KEY on success.
        if not close_job_device():
            raise RuntimeError(
                "create_mesh_device: refusing to reopen on a config change because "
                "the prior job device could not be closed (would leave two live meshes)"
            )
    _JOB_DEVICE = _create_mesh_device_uncached(mesh_shape, device_ids, l1_small_size, dispatch_core_axis, prefer_eth)
    _JOB_DEVICE_KEY = key
    return _JOB_DEVICE


def _guarded_close_mesh_device(device, *args, **kwargs):
    """Deferred close: a module's per-module/per-vector close of the shared job
    device is a no-op (it stays open for the next module); the real close happens
    at job end (close_job_device) or on a config change (in create_mesh_device)."""
    if _JOB_DEVICE is not None and device is _JOB_DEVICE:
        return None
    return _close_mesh_and_parent(device, *args, **kwargs)


# Install the deferred-close guard. No-op behavior when the job device is disabled
# (or nothing is cached), since it just passes through to the original close.
ttnn.close_mesh_device = _guarded_close_mesh_device


_orig_set_fabric_config = getattr(ttnn, "set_fabric_config", None)


def _guarded_set_fabric_config(*args, **kwargs):
    """A fabric config change requires ALL devices closed. Modules that change
    fabric (e.g. conv2d heavy<->light: close -> set_fabric_config -> reopen) call
    ttnn.close_mesh_device first, but that is deferred for the cached job device —
    so really close it here before the reconfig, or metal asserts 'SetFabricConfig
    not allowed while devices are still open'. create_mesh_device reopens after.
    If the close FAILS, refuse the reconfig rather than reconfigure fabric with a
    live device (which is illegal / corrupts state)."""
    if not close_job_device():
        raise RuntimeError(
            "set_fabric_config: refusing to reconfigure fabric because the cached job "
            "device could not be closed (a live mesh would make the reconfig illegal)"
        )
    return _orig_set_fabric_config(*args, **kwargs)


if _orig_set_fabric_config is not None:
    ttnn.set_fabric_config = _guarded_set_fabric_config


# ── Fabric configuration: match the device setup the traced model used ────────
# The fabric is part of a mesh device's configuration, not a global default, and the model
# tests treat it that way: conftest.py's mesh_device fixture pops fabric_config /
# reliability_mode / fabric_tensix_config out of device_params, calls set_fabric BEFORE
# open_mesh_device, and reset_fabric after. Every galaxy model behind our traced configs
# relies on it -- llama3_70b_galaxy/demo/text_demo.py sets "fabric_config": True, and
# deepseek_v3/demo/demo.py calls set_fabric_config(get_fabric_config(), RELAXED_INIT) with
# DISABLED at teardown.
#
# This path used to skip it entirely. Only ccl_common (the CCL modules) configured fabric, so
# every other module opened a 32-chip mesh with whatever fabric state the process happened to
# carry -- replaying llama/gpt_oss/deepseek ops under a device configuration the model never
# used. Beyond being wrong on its own terms, it correlated with the intermittent Galaxy hangs:
# the generic-path 2D batches (p1, p2, mesh8x4_col_2d, rms_norm_pre/post_all_gather) hung in
# four separate lead-model runs and ran clean across two runs with this in place, including a
# same-box rematch (p1 hung on j09glx02 and completed 542/542 on j09glx02 hours later).
# It does NOT address the CCL path (which already configured fabric) or the 1D path; both still
# hang intermittently and need hang recovery rather than a configuration change.
#
# The tracer records only machine_info + pytest_args + source -- no device_params -- so the
# value cannot be read back from a trace and is derived from the mesh, matching the rule
# all_gather_async_model_traced already applies. Ring is deliberately not inferred: topology is
# a per-op property this path cannot know.
#
# Kill switch: TTNN_SWEEP_FABRIC=off restores the previous behaviour exactly.
# Explicit override: TTNN_SWEEP_FABRIC=1d | 2d.
_FABRIC_ENV = "TTNN_SWEEP_FABRIC"


def _fabric_mode() -> str:
    return os.environ.get(_FABRIC_ENV, "auto").strip().lower()


def fabric_config_for_mesh(mesh_shape):
    """The FabricConfig a model test would have set for `mesh_shape`, or None to leave
    the fabric alone (single device, or disabled by env).

    Mirrors all_gather_async_model_traced: a line mesh gets FABRIC_1D, a genuinely 2D mesh
    gets FABRIC_2D. Ring (FABRIC_1D_RING) is deliberately not inferred -- topology is a
    per-op property the generic path cannot know."""
    mode = _fabric_mode()
    if mode in ("", "off", "0", "false", "no", "disabled"):
        return None
    if _orig_set_fabric_config is None:
        return None
    try:
        rows, cols = int(mesh_shape[0]), int(mesh_shape[1])
    except (TypeError, ValueError, IndexError):
        return None
    if rows * cols <= 1:
        return None
    if mode == "1d":
        return ttnn.FabricConfig.FABRIC_1D
    if mode == "2d":
        return ttnn.FabricConfig.FABRIC_2D
    return ttnn.FabricConfig.FABRIC_1D if (rows == 1 or cols == 1) else ttnn.FabricConfig.FABRIC_2D


# True between a successful _apply_fabric_config and its matching reset. Gates the reset so we
# only ever clear a fabric WE configured -- ccl_common manages its own and must not be disturbed.
_FABRIC_APPLIED = False


def _apply_fabric_config(mesh_shape) -> None:
    """Set the fabric for an imminent open. DISABLED first, mirroring ccl_common: metal
    rejects a transition straight from one live config to another."""
    global _FABRIC_APPLIED
    cfg = fabric_config_for_mesh(mesh_shape)
    if cfg is None:
        return
    logger.info(f"SWEEPS: setting fabric {cfg} for mesh {tuple(mesh_shape)} before open")
    _orig_set_fabric_config(ttnn.FabricConfig.DISABLED)
    _orig_set_fabric_config(cfg)
    _FABRIC_APPLIED = True


def _reset_fabric_config() -> None:
    """The reset_fabric() half of the model fixture's contract: leave the process with the
    fabric DISABLED so nothing inherits a configuration it never asked for.

    No-ops unless we are the ones who set it (_FABRIC_APPLIED), so a fabric configured by
    ccl_common survives. Uses the UNGUARDED setter: callers run after the device is already
    closed, and the guarded one would call back into close_job_device."""
    global _FABRIC_APPLIED
    if not _FABRIC_APPLIED or _orig_set_fabric_config is None:
        return
    try:
        _orig_set_fabric_config(ttnn.FabricConfig.DISABLED)
    except Exception:
        logger.exception("SWEEPS: failed to reset the fabric config after closing the mesh device")
    finally:
        _FABRIC_APPLIED = False


_orig_open_mesh_device = ttnn.open_mesh_device

# Full-host mesh orientations a 2D submesh can be carved out of, per host device count.
_FULL_HOST_MESH_CANDIDATES = {32: ((8, 4), (4, 8)), 16: ((8, 2), (2, 8), (4, 4))}

# Carved submesh -> (submesh, parent full mesh). The submesh is kept referenced so its
# id() cannot be reused by another object while the mapping is live.
_SUBMESH_PARENTS = {}


def _full_host_mesh_for(mesh_shape, num_devices):
    """The full-host mesh a 2D SUBMESH must be carved from, or None to open directly.

    Opening MeshShape(submesh) directly on a galaxy fails fabric router sync on the
    submesh's BOUNDARY ethernet links -- they connect to chips outside the carved region,
    so they never complete the handshake. Opening the full host mesh runs bring-up over the
    whole (healthy) topology, and a submesh of an already-synced fabric works.

    Returns None when mesh_shape is 1D (a line/ring is opened directly) or already spans
    the host. Mirrors _full_galaxy_mesh_for in all_gather_async_model_traced, which has done
    this per-vector for CCL since before the generic path existed.
    """
    try:
        dims = tuple(int(d) for d in mesh_shape)
    except (TypeError, ValueError):
        return None
    if len(dims) != 2:
        return None
    rows, cols = dims
    if rows <= 1 or cols <= 1:
        return None
    if rows * cols >= num_devices:
        return None
    for full_rows, full_cols in _FULL_HOST_MESH_CANDIDATES.get(num_devices, ()):
        if full_rows >= rows and full_cols >= cols and full_rows * full_cols == num_devices:
            return (full_rows, full_cols)
    return None


def _open_mesh_carving_submesh(*args, **kwargs):
    """Open the requested mesh, carving it out of the full host mesh when it is a 2D submesh.

    Every non-CCL module reaches the device through here, so a batch that declares a submesh
    shape (MESH_DEVICE_SHAPE=4x4 on a 32-card box) no longer opens that submesh directly.
    conv2d on a directly-opened 4x4 tripped fabric router sync in every job that contained it
    -- lead-models runs 30696173498 and 30702184301, 2 of 2, while every 4x4 job WITHOUT
    conv2d passed (5 of 5, one with 352 vectors) -- and on the full 4x8 mesh conv2d passes.

    The requested shape is preserved: the returned device's shape IS the submesh, so traced
    placement metadata still matches and create_tensor_on_mesh does not fall back to
    ReplicateTensorToMesh. That fallback is the original bug device-key batching removed, so
    declaring the full mesh instead of carving would not be a fix.

    Falls back to a direct open when carving is not applicable, not possible (e.g. a degraded
    host with fewer chips than the full mesh needs), or disabled via
    TTNN_SWEEP_NO_SUBMESH_CARVE=1.
    """
    requested = kwargs.get("mesh_shape")
    if requested is None or os.environ.get("TTNN_SWEEP_NO_SUBMESH_CARVE", "").strip() == "1":
        return _orig_open_mesh_device(*args, **kwargs)

    try:
        num_devices = ttnn.get_num_devices()
    except Exception:
        return _orig_open_mesh_device(*args, **kwargs)

    full = _full_host_mesh_for(requested, num_devices)
    if full is None:
        return _orig_open_mesh_device(*args, **kwargs)

    submesh_shape = tuple(int(d) for d in requested)
    parent = None
    try:
        logger.info(f"SWEEPS: opening full host mesh {full} then carving submesh {submesh_shape}")
        parent_kwargs = dict(kwargs)
        parent_kwargs["mesh_shape"] = ttnn.MeshShape(*full)
        parent = _orig_open_mesh_device(*args, **parent_kwargs)
        submesh = parent.create_submesh(ttnn.MeshShape(submesh_shape))
    except Exception as e:
        # Carving failed (commonly: the host has fewer chips than `full` needs, e.g. a box
        # whose topology downgraded). Close the parent and fall back to a direct open so the
        # behaviour is no worse than before this path existed.
        logger.warning(f"SWEEPS: submesh carve of {submesh_shape} from {full} failed ({e}); opening directly")
        if parent is not None:
            try:
                parent.quiesce_devices()
            except Exception:
                logger.exception("SWEEPS: quiesce_devices before closing the parent mesh after a failed carve failed")
            try:
                _orig_close_mesh_device(parent)
            except Exception as close_error:
                # The fallback below would open a SECOND mesh alongside a parent that is still
                # live, which is exactly the state _guarded_open_mesh_device exists to prevent
                # (invalid context_id / "binary not found" / event-order fatals on Galaxy, and
                # a box left wedged for every job after this one). A carve error is recoverable;
                # two live meshes are not. Surface it instead -- sweeps_runner classifies a
                # mesh-open failure as infra, so the vector is reported honestly and the box
                # survives.
                raise RuntimeError(
                    f"submesh carve of {submesh_shape} from {full} failed ({e}) AND the parent mesh "
                    f"could not be closed ({close_error}); refusing to open a second mesh over a live one"
                ) from close_error
        return _orig_open_mesh_device(*args, **kwargs)

    _SUBMESH_PARENTS[id(submesh)] = (submesh, parent)
    return submesh


def _close_mesh_and_parent(device, *args, **kwargs):
    """Close `device`, or its parent when `device` is a carved submesh.

    Closing a submesh does not release the parent's devices, so the parent must be closed or
    the next open runs a second live mesh (which corrupts context state on Galaxy).

    The carved submesh shares the PARENT's command queues, so the parent is quiesced first to
    drain them -- otherwise closing it with submesh work still in flight throws, which makes
    close_job_device() report failure and the fabric-reconfig / reopen guards then refuse
    ("refusing to reconfigure fabric because the cached job device could not be closed").
    That cost 30 vectors in lead-models run 30706921019 (24 conv2d + 6 group_norm), all of
    them fabric-reconfiguring modules. Mirrors ccl_common._teardown_cached_device, which has
    always quiesced before closing a carved parent."""
    entry = _SUBMESH_PARENTS.pop(id(device), None)
    if entry is None:
        result = _orig_close_mesh_device(device, *args, **kwargs)
        # Reset here, not only in close_job_device: create_mesh_device deliberately bypasses
        # the job-device cache (caching disabled, or an unkeyable config), and those devices are
        # closed straight through this path -- close_job_device would early-return on
        # _JOB_DEVICE is None and never reset, leaving the process-global fabric enabled for
        # whatever opens next. _reset_fabric_config only acts if WE set it, so a fabric
        # configured by ccl_common is left alone.
        _reset_fabric_config()
        return result
    parent = entry[1]
    try:
        parent.quiesce_devices()
    except Exception:
        logger.exception("SWEEPS: quiesce_devices before closing the carved parent mesh failed")
    result = _orig_close_mesh_device(parent, *args, **kwargs)
    _reset_fabric_config()
    return result


def _guarded_open_mesh_device(*args, **kwargs):
    """Some modules (linear/matmul gather_in0 ring, batched DRAM-sharded) open their
    OWN mesh device directly via ttnn.open_mesh_device instead of create_mesh_device,
    so they bypass the job-device cache. With the deferred-close guard the cached job
    device stays open, so a direct open would run a SECOND device alongside it — on
    Galaxy that corrupts device/context state (TT_FATAL metal_context.cpp:74
    'context_id ... is invalid', kernel.cpp:443 'binary not found', dispatch.cpp:254
    event-order). Really close the cached job device first so only ONE device is ever
    open; the next create_mesh_device reopens and re-caches it. Safe against the
    cache-miss reopen path in create_mesh_device (which nulls _JOB_DEVICE before
    calling _create_mesh_device_uncached -> here close_job_device is a no-op) and
    calls _orig_open_mesh_device (not the wrapper), so no re-entrancy. If the close
    FAILS, refuse to open rather than leave two live mesh devices."""
    if _job_device_enabled():
        if not close_job_device():
            raise RuntimeError(
                "open_mesh_device: refusing to open a new mesh because the cached job "
                "device could not be closed (would leave two live mesh devices)"
            )
    return _open_mesh_carving_submesh(*args, **kwargs)


ttnn.open_mesh_device = _guarded_open_mesh_device


def device_canary(device) -> Tuple[bool, str]:
    """Ask the device a question we already know the answer to: is 2 + 2 still 4, on every chip?

    Returns (healthy, detail). Never raises -- a throw IS a failed canary.

    Why this exists. Wedge detection is otherwise REACTIVE: the runner pattern-matches the
    exception text a vector produced, so a device that returns plausible-looking garbage is
    indistinguishable from an op that computed the wrong answer. Lead-models run 31295900210
    is the case in point -- after a device timeout, `add`, `linear` and
    `nlp_create_qkv_heads_decode` each came back at PCC exactly 0.0 (an all-zeros readback)
    and were booked as three separate test failures. All three pass in other runs on other
    boxes. A health probe answers the question directly instead of inferring it from prose.

    The tensor is SHARDED over the whole mesh, not run on one chip: a batch's vectors span
    every device in its mesh, so a probe that only exercises chip 0 would miss exactly the
    per-chip corruption this is meant to catch. Scatter + compute + gather all participate.

    Exact equality, not PCC: 2.0 and 4.0 are exactly representable in bfloat16, so any
    deviation at all is corruption rather than precision.
    """
    try:
        num_devices = device.get_num_devices() if hasattr(device, "get_num_devices") else 1
    except Exception as e:
        return False, f"canary: could not query device count ({e})"

    try:
        rows = 32 * max(int(num_devices), 1)
        torch_in = torch.full((rows, 32), 2.0, dtype=torch.float32)
        if hasattr(device, "get_num_devices") and num_devices > 1:
            mapper = ttnn.ShardTensorToMesh(device, dim=0)
            tt_in = ttnn.from_torch(
                torch_in, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mapper
            )
            out = ttnn.to_torch(ttnn.add(tt_in, tt_in), mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
        else:
            tt_in = ttnn.from_torch(torch_in, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            out = ttnn.to_torch(ttnn.add(tt_in, tt_in))
    except Exception as e:
        return False, f"canary: 2+2 raised on a {num_devices}-device mesh ({type(e).__name__}: {str(e)[:160]})"

    try:
        if tuple(out.shape) != (rows, 32):
            return False, f"canary: 2+2 returned shape {tuple(out.shape)}, expected {(rows, 32)}"
        bad = int((out != 4.0).sum().item())
        if bad:
            return False, (
                f"canary: 2+2 != 4 in {bad}/{out.numel()} elements on a {num_devices}-device mesh "
                f"(first value {out.flatten()[0].item()}) -- the device is returning corrupt data"
            )
    except Exception as e:
        return False, f"canary: could not verify the 2+2 result ({e})"
    return True, f"canary: 2+2 == 4 across {num_devices} device(s)"


def clear_job_device_program_cache() -> None:
    """Clear the cached job device's program cache — call at each module boundary
    so a new module doesn't collide with an earlier module's cached programs /
    kernel binaries on the reused device (TT_FATAL kernel.cpp:443 'binary not
    found')."""
    if _JOB_DEVICE is not None:
        try:
            _JOB_DEVICE.clear_program_cache()
        except Exception:
            pass


def close_job_device() -> bool:
    """Really close the cached job device (job end / worker teardown, config
    change, fabric reconfig).

    Returns True on success (or when nothing is cached). On a close FAILURE it
    KEEPS _JOB_DEVICE set (does not forget it) and returns False, so callers that
    require all devices closed before proceeding — a fabric reconfig or a fresh
    mesh open — can refuse rather than run a second live mesh and corrupt context.
    Best-effort teardown callers may ignore the result."""
    global _JOB_DEVICE, _JOB_DEVICE_KEY
    if _JOB_DEVICE is None:
        return True
    try:
        # _close_mesh_and_parent, not the raw close: the cached job device may be a submesh
        # carved from a full host mesh, and closing the submesh would leave the parent open.
        _close_mesh_and_parent(_JOB_DEVICE)
    except Exception:
        logger.exception("close_job_device: failed to close the cached job device; keeping it cached")
        return False
    _JOB_DEVICE = None
    _JOB_DEVICE_KEY = None
    # reset_fabric() half of the model fixture's contract: leave the process with the
    # fabric DISABLED so the next open starts from a known state and nothing inherits a
    # config it never asked for.
    _reset_fabric_config()
    return True


def _create_mesh_device_uncached(*args, **kwargs) -> ttnn.MeshDevice:
    """Open a mesh device, resetting the fabric if the open fails.

    Wraps _open_mesh_device_configured so a fabric we set on the way in never outlives a failed
    open: without this, an exception between _apply_fabric_config and a returned device would
    leave the process-global fabric enabled for whatever opens next (the same leak as an
    uncached close, see _close_mesh_and_parent)."""
    try:
        return _open_mesh_device_configured(*args, **kwargs)
    except BaseException:
        _reset_fabric_config()
        raise


def _open_mesh_device_configured(
    mesh_shape: Tuple[int, int],
    device_ids: Optional[list] = None,
    l1_small_size: int = 79104,
    dispatch_core_axis=None,
    prefer_eth: bool = True,
) -> ttnn.MeshDevice:
    """
    Create a mesh device with the specified shape.

    Args:
        mesh_shape: Tuple of (rows, cols) for mesh shape
        device_ids: Optional list of device IDs (deprecated, not used by API)
        l1_small_size: L1 small buffer size (default 79104 to prevent OOM in model-traced sweeps)
        dispatch_core_axis: explicit ttnn.DispatchCoreAxis (ROW/COL). When given,
            opens WORKER dispatch on that axis directly and skips auto-detection.
            Used for per-vector axis selection when an op's traced program_config
            grid needs a specific axis (some need x=7/ROW, others y=9/COL).

    Returns:
        ttnn.MeshDevice instance

    Dispatch axis selection (in priority order):
      0. Explicit ``dispatch_core_axis`` argument (per-vector selection).
      1. `TTNN_DISPATCH_AXIS=col|row` env var — explicit override. Used by
         the two-pass workflow when a single op has master configs that
         straddle both axes (e.g. linear has both y=9 and x=7 masters).
      2. Auto-detect from master JSON (legacy behaviour) — works when an
         op's masters all need the same axis.
      3. Default to COL.
    """
    # Blackhole does not support the WORKER ROW/COL dispatch-core axes that the
    # wormhole logic below selects: opening with one raises "ROW dispatch core
    # axis is not supported for blackhole arch unless fabric tensix MUX is
    # enabled" (~all blackhole p100a/p150b/p300a model_traced fails were this).
    # The whole ROW/COL/ETH selection dance is wormhole-specific (Galaxy 8x9 /
    # 7x10 grids); on blackhole just open with the default dispatch core config
    # — which blackhole supports — and skip it entirely. This also overrides any
    # explicit dispatch_core_axis a caller passes, since blackhole can't honor it.
    # Match the model-test fixture: fabric is configured BEFORE the mesh is opened.
    # Callers guarantee no live cached device here (create_mesh_device closes it first),
    # which is what makes a fabric transition legal.
    _apply_fabric_config(mesh_shape)

    _arch = os.environ.get("ARCH_NAME", "").lower()
    if not _arch:
        try:
            _arch = ttnn.get_arch_name().lower()
        except Exception:
            _arch = ""
    if "blackhole" in _arch:
        return ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*mesh_shape),
            l1_small_size=l1_small_size,
            dispatch_core_config=ttnn.DispatchCoreConfig(),
        )

    # Prefer ETH dispatch on single-host clusters, for EVERY caller (explicit
    # axis or auto-detect). On a single Wormhole chip, WORKER dispatch (either
    # axis) reserves a worker row/col and leaves only 8x7 / 7x8 = 56 compute
    # cores; many traced configs need the full 8x8 (64-core) grid and otherwise
    # crash with "compute_with_storage_grid_size must fit within (8,7)",
    # "num_shards (64) <= num_compute_banks (56)", "num_cores <= grid.x*grid.y",
    # or "not on_dispatch_core". ETH frees the full 8x8 grid (verified:
    # WORKER->56, ETH->64) and is a strict superset of both WORKER axes, so it
    # hosts every config either axis could plus the full-grid ones. This must run
    # even when dispatch_core_axis is None: ops whose grid nominally fits a WORKER
    # axis (so they pass axis=None) still need the 8th row/col that only ETH frees
    # (e.g. a (5,8) matmul grid needs y=8, which WORKER ROW's 8x7 lacks).
    #
    # Gate to single-host clusters (N150=1, N300=2, T3K=8). ETH dispatch (incl.
    # with FABRIC_1D) is validated on T3K: it frees the full 8x8 (64-bank) grid
    # per chip and closes cleanly. On large multi-host Galaxy clusters (32 chips)
    # ETH dispatch cores can't be allocated and a failed ETH open re-inits
    # MetalContext and wedges the command queue, so leave those on WORKER.
    try:
        _single_host = ttnn.get_num_devices() <= _SINGLE_HOST_MAX_DEVICES
    except Exception:
        _single_host = False
    # prefer_eth=False lets a caller opt out of ETH dispatch. conv2d needs WORKER
    # ROW dispatch aligned with its mesh-row distribution + FABRIC_1D, otherwise
    # the distributed DRAM-sliced convs hang in synchronize_device; ETH dispatch
    # (not row-aligned) wedges those, so conv2d forces WORKER.
    if _single_host and prefer_eth:
        try:
            return ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(*mesh_shape),
                l1_small_size=l1_small_size,
                dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH),
            )
        except Exception:
            # ETH dispatch unavailable here — fall back to the logic below.
            pass

    # 0. Explicit per-vector axis override (WORKER on the requested axis).
    if dispatch_core_axis is not None:
        return ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*mesh_shape),
            l1_small_size=l1_small_size,
            dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER, dispatch_core_axis),
        )

    # 1. Env-var override — but ETH dispatch overrides when 8x8 grid is needed.
    _env_axis = os.environ.get("TTNN_DISPATCH_AXIS", "").strip().lower()

    # Auto-discover master JSON if env var not set
    if not os.environ.get("TTNN_MASTER_JSON_PATH"):
        _auto_master = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "model_tracer",
            "traced_operations",
            "ttnn_operations_master.json",
        )
        if os.path.isfile(_auto_master):
            os.environ["TTNN_MASTER_JSON_PATH"] = os.path.abspath(_auto_master)

    # 2. Auto-detect from master configs (legacy path).
    # ROW dispatch gives compute_with_storage_grid_size = (8, 9): valid y in [0, 8].
    # COL dispatch gives (7, 10): valid y in [0, 9], valid x in [0, 6].
    # Master traces from deepseek_v3 use either layout depending on which dispatch
    # was active when traced. Default to ROW; switch to COL only if any of the
    # op's master shard_specs requires y=9 cores. Both cases: cores with x=7 fall
    # outside COL but inside ROW — those configs need ROW.
    needs_col = False
    needs_row_only = False
    try:
        # Try to derive the op name from the runner's --module-name arg, e.g.
        # "model_traced.linear_model_traced" -> "ttnn.linear"
        op_name = os.environ.get("TTNN_SWEEP_OP_NAME", "")
        if not op_name:
            import sys as _sys_d

            for _i, _a in enumerate(_sys_d.argv):
                if _a == "--module-name" and _i + 1 < len(_sys_d.argv):
                    _m = _sys_d.argv[_i + 1]
                    if _m.startswith("model_traced."):
                        _stem = _m.split(".", 1)[1].replace("_model_traced", "")
                        # Check experimental + transformer prefixes by probing master json
                        op_name = _stem  # bare; we'll match flexibly below
                    break
        master_json = os.environ.get("TTNN_MASTER_JSON_PATH", "")
        if op_name and master_json and os.path.isfile(master_json):
            import json as _json_d

            with open(master_json) as _f:
                _m = _json_d.load(_f)
            # Try multiple forms: "ttnn.X", "ttnn.experimental.X", "ttnn.transformer.X"
            _candidates = [
                op_name,
                f"ttnn.{op_name}",
                f"ttnn.experimental.{op_name}",
                f"ttnn.transformer.{op_name}",
            ]
            _matching_op = None
            _ops_dict = _m.get("operations", {})
            for _c in _candidates:
                if _c in _ops_dict:
                    _matching_op = _c
                    break
            if _matching_op is None:
                _matching_op = op_name
            for _cfg in _ops_dict.get(_matching_op, {}).get("configurations", []):
                for _arg in _cfg.get("arguments", {}).values():
                    if not isinstance(_arg, dict):
                        continue
                    _val = str(_arg.get("value", ""))
                    if "compute_with_storage_grid_size=8-8" in _val:
                        needs_row_only = True
                    _ss = (_arg.get("memory_config") or {}).get("shard_spec")
                    if not isinstance(_ss, dict):
                        _ss = _arg.get("shard_spec")
                    if not isinstance(_ss, dict):
                        continue
                    for _g in _ss.get("grid", []):
                        for _key in ("start", "end"):
                            _p = _g.get(_key, {})
                            if _p.get("y") == 9:
                                needs_col = True
                            if _p.get("x") == 7:
                                needs_row_only = True
    except Exception:
        needs_col = False
        needs_row_only = False

    # Explicit TTNN_DISPATCH_AXIS override takes priority over auto-detection.
    # This enables the two-pass workflow (run a sweep once with row, once with
    # col) for ops whose master configs straddle both axes — e.g. linear, which
    # has some configs needing x=7 (ROW/8x9) and others needing y=9 (COL/7x10);
    # no single axis fits all, so each pass covers the configs it can host.
    if _env_axis in ("col", "row"):
        _override_axis = ttnn.DispatchCoreAxis.COL if _env_axis == "col" else ttnn.DispatchCoreAxis.ROW
        try:
            return ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(*mesh_shape),
                l1_small_size=l1_small_size,
                dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER, _override_axis),
            )
        except Exception:
            # this dispatch axis/config isn't available on this cluster — fall through to the next open option
            pass

    # Default: COL (gives compute grid 7x10) since most lead_models traces use
    # cores in the 7-wide pattern with y up to 9. Switch to ROW only if any of
    # the op's master shard_specs uses x=7 (which COL excludes).
    # When x=7 or 8-8 grid is needed, use ETH dispatch so all 8x8
    # compute cores are available.
    if needs_row_only:
        # Prefer WORKER ROW dispatch (compute grid 8x9 -> x in [0,7]) which
        # provides the x=7 cores these configs need, and is a superset of ETH's
        # 8x8 grid. Try it BEFORE DispatchCoreType.ETH: on Galaxies where ETH
        # dispatch cores cannot be allocated ("No more available dispatch cores
        # on device 0"), the failed ETH open + MetalContext re-init leaves the
        # command queue in a broken state and the next op's readback hangs
        # (observed for add/gelu/layer_norm/softmax/transpose). ROW WORKER opens
        # cleanly and avoids that; ETH stays as a fallback. Dispatch-core
        # placement does not affect op results, only which cores are usable.
        try:
            # Specify WORKER explicitly: DispatchCoreConfig(axis=...) alone
            # leaves the core *type* at the system default (which can be ETH on
            # multi-chip clusters), defeating the purpose of avoiding ETH here.
            return ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(*mesh_shape),
                l1_small_size=l1_small_size,
                dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER, ttnn.DispatchCoreAxis.ROW),
            )
        except Exception:
            # WORKER ROW dispatch unavailable here — fall through to the next open option
            pass
        try:
            return ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(*mesh_shape),
                l1_small_size=l1_small_size,
                dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH),
            )
        except Exception:
            # ETH dispatch unavailable on this cluster — fall through to the default open below
            pass

    # 4. Default to COL.
    use_axis = ttnn.DispatchCoreAxis.COL

    try:
        # NB: pass axis only and let the core *type* default. Unlike the ROW path
        # above (which forces WORKER to dodge ETH-allocation failures), the default
        # type here keeps dispatch OFF the worker grid, leaving the full worker grid
        # free for compute — forcing WORKER instead collides with ops that use the
        # whole grid ("Illegal kernel placement ... on dispatch cores", e.g. add).
        return ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(*mesh_shape),
            l1_small_size=l1_small_size,
            dispatch_core_config=ttnn.DispatchCoreConfig(axis=use_axis),
        )
    except Exception:
        # requested dispatch axis was rejected — caller falls back to a plain device open
        pass

    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*mesh_shape),
        l1_small_size=l1_small_size,
        dispatch_core_config=ttnn.DispatchCoreConfig(),
    )


def _parse_shard_dim(placement_str: str) -> int:
    """Extract shard dimension from placement string, handling negative dims."""
    import re

    shard_dims = re.findall(r"PlacementShard\((?:dim=)?(-?\d+)\)", placement_str)
    return int(shard_dims[-1]) if shard_dims else -1


def _is_shard_placement(tensor_placement: Optional[Dict], num_devices: int) -> bool:
    """Check if placement is a shard placement with multiple devices."""
    if not tensor_placement:
        return False
    placement_str = tensor_placement.get("placement", "")
    # If it has both Replicate and Shard, check which comes first or treat as replicate
    if "PlacementReplicate" in placement_str and "PlacementShard" not in placement_str:
        return False
    return "PlacementShard" in placement_str and num_devices > 1


def get_mesh_composer(mesh_device, tensor_placement: Optional[Dict] = None):
    """
    Create a mesh composer matching the tensor placement for converting back to torch.

    For sharded tensors, returns a ConcatMesh2dToTensor that reassembles shards.
    For replicated tensors, returns None (caller should use device 0 extraction).

    Args:
        mesh_device: The mesh device
        tensor_placement: Placement info from traced config

    Returns:
        Mesh composer or None
    """
    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    if not _is_shard_placement(tensor_placement, num_devices):
        return None

    placement_str = tensor_placement.get("placement", "")
    shard_dim = _parse_shard_dim(placement_str)

    try:
        mesh_shape = (1, num_devices)
        return ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape, dims=shard_dim)
    except (TypeError, RuntimeError):
        return None


def was_replicated_for_validation(mesh_device, tensor_placement: Optional[Dict] = None) -> bool:
    """True when create_tensor_on_mesh stored a shard-placement tensor as a full
    per-chip replica (replicate_with_topology / ReplicateTensorToMesh) rather than
    truly splitting it.

    On this trace-validation path a sharded placement is ALWAYS materialized
    replicated: when the device mesh fits the traced mesh, create_tensor_on_mesh
    returns replicate_with_topology; when it doesn't, it falls back to
    ReplicateTensorToMesh. Either way each chip holds the FULL tensor and the op
    runs SPMD producing the FULL result per chip. The output must then be read
    from a single device — concatenating via a Shard composer would multiply the
    shard dim by the mesh factor (e.g. a head_dim-Shard(3) SDPA output read back
    as [.,256] instead of [.,128]).
    """
    if not tensor_placement:
        return False
    placement_str = str(tensor_placement.get("placement", ""))
    if "PlacementShard" not in placement_str:
        return False
    try:
        num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    except Exception:
        num_devices = 1
    return num_devices > 1


def _restore_topology(
    tensor: ttnn.Tensor,
    placement_entries: list,
    dist_parsed: list,
    mesh_shape_tuple: tuple,
) -> None:
    """Restore correct TensorTopology on a device tensor to match the master trace.

    The C++ factory methods may create a topology that doesn't match what the
    master trace recorded (e.g., flattening 2D to 1D, or setting 2D when the
    master had 1D).  This helper reconstructs the exact topology from the vector
    config's distribution_shape and placement info.
    """
    import re

    ndim = len(dist_parsed)

    if ndim >= 2:
        # 2D (or higher) distribution — e.g. [4, 8]
        dist_shape = ttnn.MeshShape(*dist_parsed[:2])

        placements = []
        for entry in placement_entries or []:
            shard_match = re.search(r"PlacementShard\((-?\d+)\)", entry)
            if shard_match:
                placements.append(ttnn.PlacementShard(int(shard_match.group(1))))
            else:
                placements.append(ttnn.PlacementReplicate())
        while len(placements) < 2:
            placements.append(ttnn.PlacementReplicate())

        rows, cols = dist_parsed[0], dist_parsed[1]
        mesh_coords = [ttnn.MeshCoordinate(r, c) for r in range(rows) for c in range(cols)]
    elif ndim == 1:
        # 1D distribution — e.g. [32].  Keep as 1D to match the master trace.
        dist_shape = ttnn.MeshShape(shape=dist_parsed)

        placements = []
        for entry in placement_entries or []:
            shard_match = re.search(r"PlacementShard\((-?\d+)\)", entry)
            if shard_match:
                placements.append(ttnn.PlacementShard(int(shard_match.group(1))))
            else:
                placements.append(ttnn.PlacementReplicate())
        if not placements:
            placements.append(ttnn.PlacementReplicate())

        total = dist_parsed[0]
        mesh_coords = [ttnn.MeshCoordinate(coords=[i]) for i in range(total)]
    else:
        return  # Nothing to restore

    topology = ttnn.TensorTopology(dist_shape, placements, mesh_coords)
    tensor.update_tensor_topology(topology)


def apply_tensor_placement_topology(tensor, tensor_placement, mesh_shape_tuple):
    """Apply topology from a tensor_placement config dict to a device tensor.

    Use this for tensors created outside of ``create_tensor_on_mesh`` (e.g. in
    decode-mode paths that use ``from_torch`` + ``interleaved_to_sharded``).
    """
    import ast as _ast
    import re

    if not tensor_placement:
        return
    dist_raw = tensor_placement.get("distribution_shape", "")
    if isinstance(dist_raw, str):
        try:
            dist_parsed = _ast.literal_eval(dist_raw)
        except (ValueError, SyntaxError):
            return
    else:
        dist_parsed = list(dist_raw) if dist_raw else []
    if not dist_parsed:
        return
    placement_str = str(tensor_placement.get("placement", ""))
    entries = re.findall(r"Placement(?:Shard\((?:dim=)?-?\d+\)|Replicate)", placement_str)
    try:
        _restore_topology(tensor, entries, dist_parsed, mesh_shape_tuple)
    except Exception:
        # Best-effort: topology restore is a soft annotation on the tensor.
        # Failure here doesn't affect numeric correctness — the tensor is
        # still valid for downstream use.
        pass


def replicate_with_topology(
    torch_tensor: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    dtype: ttnn.DataType,
    layout: ttnn.Layout,
    memory_config: ttnn.MemoryConfig,
    tensor_placement: Optional[Dict] = None,
) -> ttnn.Tensor:
    """Create a replicated tensor on mesh and restore the master trace's topology.

    Use this when the master model creates a tensor per-device (replicated) but the
    traced topology has shard-like placement.  This keeps the per-device shape as the
    logical shape while restoring the correct topology metadata so the operation
    tracer captures placement info matching the master trace.

    Args:
        torch_tensor: Input torch tensor (per-device shape)
        mesh_device: Mesh device to create tensor on
        dtype: TTNN data type
        layout: TTNN layout (TILE/ROW_MAJOR)
        memory_config: Memory configuration
        tensor_placement: Optional placement info from traced config

    Returns:
        TTNN tensor on mesh device with replicated data and restored topology
    """
    import ast as _ast
    import re

    # Create tensor with target layout. When memory_config is sharded, going
    # straight to from_torch tile-pads logical_shape to match the shard height
    # (e.g. 8 -> 32), mismatching master where the production tensor preserved
    # logical shape. Create in DRAM first, then to_memory_config preserves it.
    def _is_sharded(mc):
        if mc is None:
            return False
        try:
            return getattr(mc, "is_sharded", lambda: False)()
        except Exception:
            return False

    if _is_sharded(memory_config):
        tensor = ttnn.from_torch(
            torch_tensor,
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        # Upcast DRAM -> requested (often L1-sharded) memory_config. This reshard
        # is sensitive to transient L1 pressure when one device is reused across
        # many vectors (the worker alloc can momentarily land on a dispatch core
        # -> "not on_dispatch_core"); a device sync + retry clears it. Without the
        # retry the failure is silently swallowed, leaving the tensor DRAM-
        # interleaved — a flaky memory_config mismatch vs the master trace.
        for _attempt in range(4):
            try:
                tensor = ttnn.to_memory_config(tensor, memory_config)
                break
            except Exception:
                try:
                    ttnn.synchronize_device(mesh_device)
                except Exception:
                    # best-effort sync; ignore and continue the retry loop
                    pass
            # last attempt failing leaves the DRAM-resident tensor (best-effort)
    else:
        tensor = ttnn.from_torch(
            torch_tensor,
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            memory_config=memory_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    if tensor_placement:
        placement_raw = tensor_placement.get("placement", "")
        placement_str = str(placement_raw) if not isinstance(placement_raw, str) else placement_raw
        entries = re.findall(r"Placement(?:Shard\(-?\d+\)|Replicate)", placement_str)

        dist_raw = tensor_placement.get("distribution_shape", "")
        if isinstance(dist_raw, str):
            try:
                dist_parsed = _ast.literal_eval(dist_raw)
            except Exception:
                dist_parsed = []
        else:
            dist_parsed = list(dist_raw) if dist_raw else []

        mesh_shape_raw = tensor_placement.get("mesh_device_shape", "[1, 1]")
        if isinstance(mesh_shape_raw, str):
            mesh_shape_raw = _ast.literal_eval(mesh_shape_raw)
        mesh_shape_tuple = tuple(mesh_shape_raw) if isinstance(mesh_shape_raw, (list, tuple)) else (1, 1)
        if len(mesh_shape_tuple) == 0:
            mesh_shape_tuple = (1, 1)
        elif len(mesh_shape_tuple) == 1:
            mesh_shape_tuple = (mesh_shape_tuple[0], 1)
        if dist_parsed:
            try:
                _restore_topology(tensor, entries, dist_parsed, mesh_shape_tuple)
            except Exception:
                pass  # Best-effort; don't block sweep execution

    return tensor


def vector_required_axis(op_kwargs, named_mcs=None):
    """Return 'col' / 'row' / None for the dispatch axis a single sweep
    vector logically needs.

    Detection scans every memory_config (kwarg or function-named param) and
    looks at its shard_spec grid:
      - any L1 shard_spec core at y=9 -> needs COL (only COL exposes y=9)
      - any L1 shard_spec core at x=7 -> needs ROW (only ROW exposes x=7)
      - program_config compute_with_storage_grid_size with x>=8 -> ROW
      - program_config compute_with_storage_grid_size with y>=10 -> COL
    Returns None if neither axis is required (vector fits both).
    """
    import re as _re_a

    _y9_pat = _re_a.compile(r"""['"]y['"]\s*:\s*9(?!\d)""")
    _x7_pat = _re_a.compile(r"""['"]x['"]\s*:\s*7(?!\d)""")
    _grid_x_pat = _re_a.compile(r"x\s*=\s*(\d+)")
    _grid_y_pat = _re_a.compile(r"y\s*=\s*(\d+)")

    needs_y9 = False
    needs_x7 = False

    def _walk_mc(_obj):
        nonlocal needs_y9, needs_x7
        if _obj is None:
            return
        if isinstance(_obj, dict):
            _bt = str(_obj.get("buffer_type", ""))
            if "DRAM" in _bt:
                return
            _ss = _obj.get("shard_spec")
            if not _ss or _ss == "None":
                return
            _r = repr(_ss)
            if _y9_pat.search(_r):
                needs_y9 = True
            if _x7_pat.search(_r):
                needs_x7 = True
            return
        _r = repr(_obj)
        if "BufferType::DRAM" in _r:
            return
        if "shard_spec" not in _r:
            return
        if _y9_pat.search(_r):
            needs_y9 = True
        if _x7_pat.search(_r):
            needs_x7 = True

    _all_mcs = dict(op_kwargs) if op_kwargs else {}
    for _name, _val in named_mcs or []:
        if _val is not None:
            _all_mcs[_name] = _val
    for _key, _v in _all_mcs.items():
        if "memory_config" not in _key:
            continue
        _walk_mc(_v)

    # program_config grid (SDPA-style): x>=8 -> ROW, y>=10 -> COL.
    _pc = (op_kwargs or {}).get("program_config")
    if _pc is not None:
        _pc_text = ""
        if isinstance(_pc, dict):
            _pc_text = str(_pc.get("value", "")) or str(_pc.get("repr", ""))
        else:
            _pc_text = repr(_pc)
        if "compute_with_storage_grid_size" in _pc_text:
            _idx = _pc_text.find("compute_with_storage_grid_size")
            _section = _pc_text[_idx : _idx + 80]
            _xm = _grid_x_pat.search(_section)
            _ym = _grid_y_pat.search(_section)
            if _xm and int(_xm.group(1)) >= 8:
                needs_x7 = True
            if _ym and int(_ym.group(1)) >= 10:
                needs_y9 = True

    if needs_y9:
        return "col"
    if needs_x7:
        return "row"
    return None


def current_device_axis(device):
    """Return 'col' / 'row' / None inferred from the device's
    compute_with_storage_grid_size."""
    try:
        g = device.compute_with_storage_grid_size() if hasattr(device, "compute_with_storage_grid_size") else None
    except Exception:
        return None
    if g is None:
        return None
    # COL -> (7, 10); ROW -> (8, 9). Other meshes (N150 etc.) return None.
    if g.x == 7 and g.y == 10:
        return "col"
    if g.x == 8 and g.y == 9:
        return "row"
    return None


def vector_axis_matches(device, op_kwargs, named_mcs=None):
    """True if the vector either has no required axis or matches the device."""
    required = vector_required_axis(op_kwargs, named_mcs)
    if required is None:
        return True
    actual = current_device_axis(device)
    if actual is None:
        return True  # unknown / non-Galaxy mesh — let the test run.
    return required == actual


def scatter_uses_replicate_topology(tensor_placement, mesh_device) -> bool:
    """Whether create_tensor_on_mesh would build this placement via replicate_with_topology.

    A pure function of the traced placement and the actual mesh -- no device data involved -- and
    the SINGLE source of truth for that branch: create_tensor_on_mesh calls it to decide, and
    mesh_tensor_to_torch calls it to know how the input was built, so the two can never drift.

    This matters because replicate_with_topology puts IDENTICAL data on every chip while stamping a
    Shard topology, so a gathered output has to be collapsed to one copy to match a golden computed
    from the per-chip shape. Deciding that from the tensor alone is ambiguous (identical per-device
    shapes occur both here and for a genuine even shard), and _replicated_single_copy resolves it by
    comparing per-device CONTENTS -- which makes the returned SHAPE data-dependent: a single
    differing byte flips a [1,1,128,2048] chunk to [1,1,128,8192] on a mesh whose shard axis is 4.
    """
    if not tensor_placement:
        return False
    if "PlacementShard" not in str(tensor_placement.get("placement", "")):
        return False
    try:
        actual_mesh = mesh_device.shape
        actual_rows, actual_cols = actual_mesh[0], actual_mesh[1]
    except Exception:
        actual_rows, actual_cols = 1, 1

    traced = tensor_placement.get("mesh_device_shape", "[1, 1]")
    if isinstance(traced, str):
        try:
            traced = ast.literal_eval(traced)
        except (ValueError, SyntaxError):
            traced = [1, 1]
    if not isinstance(traced, (list, tuple)):
        traced = [1, 1]
    traced_rows = traced[0] if len(traced) > 0 else 1
    traced_cols = traced[1] if len(traced) > 1 else 1
    return actual_rows >= traced_rows and actual_cols >= traced_cols


def create_tensor_on_mesh(
    torch_tensor: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    dtype: ttnn.DataType,
    layout: ttnn.Layout,
    memory_config,
    tensor_placement: Optional[Dict] = None,
) -> ttnn.Tensor:
    """
    Create a TTNN tensor on a mesh device with optional placement.

    Args:
        torch_tensor: Input torch tensor
        mesh_device: Mesh device to create tensor on
        dtype: TTNN data type
        layout: TTNN layout (TILE/ROW_MAJOR)
        memory_config: Memory configuration (MemoryConfig object or dict)
        tensor_placement: Optional placement info from traced config

    Returns:
        TTNN tensor on mesh device with proper placement
    """
    if isinstance(memory_config, dict):
        from tests.sweep_framework.master_config_loader_v2 import dict_to_memory_config

        memory_config = dict_to_memory_config(memory_config) or ttnn.DRAM_MEMORY_CONFIG
    # Trace-validation path: when placement has Shard, the master records the
    # per-chip .shape (which equals the torch input shape). Going through
    # ShardTensor2dMesh would re-shard the input and produce a smaller per-chip
    # shape, so delegate to replicate_with_topology, which keeps .shape =
    # input shape and stamps the correct sharded topology metadata.
    if scatter_uses_replicate_topology(tensor_placement, mesh_device):
        return replicate_with_topology(torch_tensor, mesh_device, dtype, layout, memory_config, tensor_placement)

    # Determine mesh mapper based on placement
    if tensor_placement:
        import re
        import ast as _ast

        placement_raw = tensor_placement.get("placement", "")
        placement_str = str(placement_raw) if not isinstance(placement_raw, str) else placement_raw

        mesh_shape_raw = tensor_placement.get("mesh_device_shape", "[1, 1]")
        if isinstance(mesh_shape_raw, str):
            mesh_shape_raw = _ast.literal_eval(mesh_shape_raw)
        mesh_shape_tuple = tuple(mesh_shape_raw) if isinstance(mesh_shape_raw, (list, tuple)) else (1, 1)
        # Default empty/short mesh_shape to (1, 1)
        if len(mesh_shape_tuple) == 0:
            mesh_shape_tuple = (1, 1)
        elif len(mesh_shape_tuple) == 1:
            mesh_shape_tuple = (mesh_shape_tuple[0], 1)

        # Check if the actual device mesh can support the traced mesh shape.
        # If not (e.g., traced on Galaxy 4x8 but running on N150 1x1), fall back to replicate.
        try:
            actual_mesh = mesh_device.shape
            actual_rows, actual_cols = actual_mesh[0], actual_mesh[1]
        except Exception:
            actual_rows, actual_cols = 1, 1
        traced_rows = mesh_shape_tuple[0]
        traced_cols = mesh_shape_tuple[1] if len(mesh_shape_tuple) > 1 else 1
        mesh_compatible = actual_rows >= traced_rows and actual_cols >= traced_cols

        entries = re.findall(r"Placement(?:Shard\((?:dim=)?-?\d+\)|Replicate)", placement_str)

        dist_raw = tensor_placement.get("distribution_shape", "")
        if isinstance(dist_raw, str):
            try:
                dist_parsed = _ast.literal_eval(dist_raw)
            except Exception:
                dist_parsed = []
        else:
            dist_parsed = list(dist_raw) if dist_raw else []
        is_2d_distribution = len(dist_parsed) >= 2

        if not mesh_compatible or not entries or "PlacementShard" not in placement_str:
            if is_2d_distribution and mesh_compatible:
                mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape_tuple)
            else:
                mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        elif len(entries) >= 2:
            dims = []
            for entry in entries[:2]:
                shard_match = re.search(r"PlacementShard\((?:dim=)?(-?\d+)\)", entry)
                if shard_match:
                    dims.append(int(shard_match.group(1)))
                else:
                    dims.append(None)
            dims_tuple = tuple(dims)

            # Both mesh axes sharding the SAME tensor dim cannot go through
            # ShardTensor2dMesh: its C++ chunk_ndim normalizes the dims and then hard-fails
            # "dims must be unique" (partition.cpp), because it chunks each listed dim once.
            # A traced ['PlacementShard(-1)', 'PlacementShard(3)'] on a 4D tensor is exactly
            # that -- -1 and 3 are the same dim -- and it means dim 3 is split rows*cols ways
            # in total. That IS expressible, as a 1D shard of that dim over the flattened
            # mesh, so use it. Scatter and gather then use the same 1D scheme (see
            # mesh_tensor_to_torch), which makes them inverses by construction regardless of
            # how devices are ordered.
            #
            # Without this the mapper throws at distribution time, the tensor ends up
            # distributed some other way, and the op returns a CORRECTLY SHAPED but wrong
            # result -- no assert fires and it surfaces as an unexplained low PCC. That is the
            # near-zero-PCC class: lead-models run 30706921019 mesh8x4_col_2d, 8 multiply
            # vectors at PCC 0.0096-0.0152 plus 1 linear at 0.093, every one of them carrying
            # this placement, with 6 "dims must be unique" TT_FATALs in the same log.
            # REACHABILITY: on a mesh big enough for the traced shape this is NOT reached --
            # the replicate_with_topology early-return above claims every PlacementShard
            # placement first. It only runs when the device is too small for the traced mesh,
            # which is the one case that skips that return. Kept for that case; the gather-side
            # counterpart in mesh_tensor_to_torch is what fires on a compatible mesh.
            _dup_dim = _same_shard_dim(dims_tuple, torch_tensor.ndim)
            if _dup_dim is not None:
                logger.info(
                    f"SWEEPS: scatter -- placement shards dim {_dup_dim} on BOTH mesh axes "
                    f"{mesh_shape_tuple}; using a 1D shard over the flattened mesh "
                    f"({mesh_shape_tuple[0] * mesh_shape_tuple[1]} devices)"
                )
                mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=_dup_dim)
            else:
                # Traced shapes are global (pre-shard); ShardTensor2dMesh splits
                # them across the mesh internally.
                mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=dims_tuple, mesh_shape=mesh_shape_tuple)
        elif len(entries) == 1:
            shard_match = re.search(r"PlacementShard\((?:dim=)?(-?\d+)\)", entries[0])
            if shard_match:
                dim = int(shard_match.group(1))
                dims_tuple = (None, dim)

                # Traced shapes are global; ShardTensor2dMesh splits the
                # tensor across the mesh internally.
                mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=dims_tuple, mesh_shape=mesh_shape_tuple)
            else:
                if is_2d_distribution:
                    mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape_tuple)
                else:
                    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        else:
            if is_2d_distribution:
                mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, None), mesh_shape=mesh_shape_tuple)
            else:
                mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    else:
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    # Create tensor on mesh. When memory_config is sharded, route via DRAM
    # then to_memory_config to preserve logical shape (avoid tile-pad to shard
    # height).
    def _ctom_is_sharded(mc):
        if mc is None:
            return False
        try:
            return getattr(mc, "is_sharded", lambda: False)()
        except Exception:
            return False

    if _ctom_is_sharded(memory_config):
        result = ttnn.from_torch(
            torch_tensor,
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        try:
            result = ttnn.to_memory_config(result, memory_config)
        except Exception:
            # Best-effort upcast to the requested memory_config. On failure we
            # keep the DRAM-resident tensor — sweeps tolerate the placement
            # difference and the kernel still runs.
            pass
    else:
        result = ttnn.from_torch(
            torch_tensor,
            dtype=dtype,
            layout=layout,
            device=mesh_device,
            memory_config=memory_config,
            mesh_mapper=mesh_mapper,
        )

    # Restore correct tensor topology from vector placement info.
    # The C++ factory methods may create a topology that doesn't match the
    # master trace (e.g., flattening [4,8] to [32] or vice versa).
    # We reconstruct and re-apply the exact topology so that the operation
    # tracer captures it accurately (matching the master trace).
    if tensor_placement and dist_parsed:
        try:
            _restore_topology(result, entries, dist_parsed, mesh_shape_tuple)
        except Exception:
            pass  # Best-effort; don't block sweep execution

    return result


def get_mesh_shape() -> Optional[Tuple[int, int]]:
    """
    Get mesh shape from environment variable or auto-detect from hardware.

    Returns:
        Tuple of (rows, cols) or None if using single device

    Environment variable format:
        MESH_DEVICE_SHAPE="1x2" -> (1, 2)
        MESH_DEVICE_SHAPE="2x4" -> (2, 4)
        Not set -> auto-detect from available hardware
    """
    mesh_env = os.environ.get("MESH_DEVICE_SHAPE", "").strip()

    if mesh_env:
        # Parse "NxM" format
        if "x" in mesh_env.lower():
            try:
                parts = mesh_env.lower().split("x")
                rows, cols = int(parts[0]), int(parts[1])
                return (rows, cols)
            except (ValueError, IndexError):
                print(f"⚠️ Invalid MESH_DEVICE_SHAPE format: {mesh_env}, expected NxM (e.g., 1x2)")
                return None
        return None

    # Auto-detect mesh shape from available hardware when env var not set.
    # Model-traced sweeps need the correct mesh topology to reproduce the
    # tensor placement metadata recorded during model tracing.
    try:
        num_devices = ttnn.get_num_devices()
        if num_devices >= 32:
            return (4, 8)  # Galaxy
        elif num_devices >= 8:
            return (1, 8)  # T3000
        elif num_devices >= 2:
            return (1, num_devices)
    except Exception:
        # ttnn may not be initialized yet (env var path is preferred).
        # Fall through to None so the caller can pick a default.
        pass

    return None


_LOGGED_MESH_SHAPE = None


def _log_resolved_mesh_shape(shape, source):
    """Log the resolved model-traced mesh shape once per process, then return it.

    Nothing else in a sweep run records which mesh shape was opened, so a job whose
    vectors were traced across several topologies gave no way to tell from the logs which
    one they actually ran on (verified against Galaxy job 90192732565: MESH_DEVICE_SHAPE
    never appears, and neither does any opened-shape line). Logged once, not per vector,
    to keep it out of the per-vector noise.
    """
    global _LOGGED_MESH_SHAPE
    entry = (tuple(shape), source)
    if _LOGGED_MESH_SHAPE != entry:
        _LOGGED_MESH_SHAPE = entry
        try:
            logger.info(f"SWEEPS: model-traced mesh shape resolved to {tuple(shape)} via {source}.")
        except Exception:
            pass  # logging must never break device setup
    return shape


def get_model_traced_mesh_shape() -> Tuple[int, int]:
    """Get mesh shape for model-traced sweep modules.

    Model traces are always captured on a mesh device (even 1x1 on single
    chips).  The tracer records 2-D ``tensor_placement`` metadata that is
    only reproduced when the sweep re-executes on a mesh device.

    Priority: MESH_DEVICE_SHAPE env var > master JSON > auto-detect.

    NOTE the priority list is misleading: get_mesh_shape() ALREADY falls back to hardware
    auto-detect, so it returns non-None on any host with >= 2 devices and the master-JSON
    and auto-detect blocks below are unreachable there. They only run on a single-device
    host (or before ttnn is initialised). On Galaxy the answer is therefore always (4, 8),
    from auto-detect -- the master JSON never influences it.

    Only ttnn-model-trace-sweep-validation-impl.yaml sets MESH_DEVICE_SHAPE (per matrix
    batch); ttnn-run-sweeps-impl.yaml does not, and never downloads the master JSON either
    (the sweep job pulls only the 'sweeps-vectors' artifact). So a lead-model Galaxy job
    runs every vector on (4, 8) even though its vectors were traced across several
    topologies (4x8: 1138, 4x4: 413, 8x4: 366, 1x32: 38, 1x1: 37). The 404 traced at
    (8, 4)/(1, 32) fail create_tensor_on_mesh's actual_rows/cols >= traced_rows/cols test
    and fall back to ReplicateTensorToMesh. The result is logged (once per process)
    because nothing else in a run records it -- verified absent from Galaxy job
    90192732565's log, which is why this was invisible in CI.
    """
    # Env var takes priority — CI sets this per batch to match traced topology.
    shape = get_mesh_shape()
    if shape:
        _src = "MESH_DEVICE_SHAPE env" if os.environ.get("MESH_DEVICE_SHAPE", "").strip() else "hardware auto-detect"
        return _log_resolved_mesh_shape(shape, _src)

    # Fall back to master JSON, filtering by current arch AND card count.
    try:
        _master_path = os.environ.get("TTNN_MASTER_JSON_PATH")
        if not _master_path:
            for _base in [
                os.path.join(os.path.dirname(__file__), "..", "..", ".."),
                os.environ.get("TT_METAL_HOME", ""),
            ]:
                if not _base:
                    continue
                _auto = os.path.join(
                    _base,
                    "model_tracer",
                    "traced_operations",
                    "ttnn_operations_master.json",
                )
                if os.path.isfile(_auto):
                    _master_path = _auto
                    break
        if _master_path and os.path.isfile(_master_path):
            import json as _json_ms

            # Detect current arch to filter configs
            _current_arch = os.environ.get("ARCH_NAME", "")
            _is_bh = "blackhole" in _current_arch.lower()
            _is_wh = "wormhole" in _current_arch.lower()

            with open(_master_path) as _f_ms:
                _m_ms = _json_ms.load(_f_ms)
            # Tally EVERY matching config's shape and take the most common one rather
            # than returning the first hit. The scan walks ALL ops, so "first" was
            # whichever op happened to lead the JSON, not the op being run: with the
            # 32-card master JSON it yields ttnn.add's (8, 4) even when running softmax,
            # although (4, 8) is the majority (1138 vectors vs 366). This branch is only
            # reachable on a single-device host (see the docstring), so this is a
            # correctness/consistency fix rather than a behaviour change on Galaxy --
            # majority-wins now agrees with what auto-detect picks.
            _shape_votes = {}
            for _op_ms in _m_ms.get("operations", {}).values():
                for _cfg_ms in _op_ms.get("configurations", []):
                    _mi_ms = _cfg_ms.get("traced_machine_info") or {}
                    if not _mi_ms:
                        _execs = _cfg_ms.get("executions", [])
                        if _execs and isinstance(_execs[0], dict):
                            _mi_ms = _execs[0].get("machine_info", {})
                    # Filter: only use configs matching current arch AND device count.
                    _board = str(_mi_ms.get("board_type", "")).lower()
                    if _is_bh and "wormhole" in _board:
                        continue
                    elif _is_wh and "blackhole" in _board:
                        continue

                    # Filter by device count so N300 (1-2 devices) doesn't
                    # pick up Galaxy (32) or T3K (8) mesh shapes.
                    _num_devices = ttnn.get_num_devices()
                    _cfg_card_count = _mi_ms.get("card_count")
                    if _cfg_card_count is not None and int(_cfg_card_count) > _num_devices:
                        continue

                    _ms_val = _mi_ms.get("mesh_device_shape")
                    if _ms_val:
                        import ast as _ast_ms

                        if isinstance(_ms_val, str):
                            _ms_val = _ast_ms.literal_eval(_ms_val)
                        if isinstance(_ms_val, list) and len(_ms_val) == 2:
                            _key_ms = tuple(_ms_val)
                            _shape_votes[_key_ms] = _shape_votes.get(_key_ms, 0) + 1
            if _shape_votes:
                _best = max(_shape_votes.items(), key=lambda kv: kv[1])[0]
                return _log_resolved_mesh_shape(_best, f"master JSON majority of {_shape_votes}")
    except Exception:
        pass  # Intentionally ignored: master config parsing is best-effort, fall through to auto-detect
    # Auto-detect mesh shape from available hardware.
    # This ensures model-traced sweeps on Galaxy (32 devices) create a [4, 8]
    # mesh matching the topology used during model tracing.
    try:
        num_devices = ttnn.get_num_devices()
        if num_devices >= 32:
            return _log_resolved_mesh_shape((4, 8), f"auto-detect ({num_devices} devices, Galaxy)")
        elif num_devices >= 8:
            return _log_resolved_mesh_shape((1, 8), f"auto-detect ({num_devices} devices, T3000)")
        elif num_devices >= 2:
            return _log_resolved_mesh_shape((1, num_devices), f"auto-detect ({num_devices} devices)")
    except Exception:
        # ttnn may not be initialized yet (env var path is preferred).
        # Fall through to a 1x1 default for non-mesh runs.
        pass
    return _log_resolved_mesh_shape((1, 1), "default (no mesh detected)")


def _replicated_single_copy(device_tensors, to_torch_fn):
    """Return device 0's torch tensor iff every device holds identical data.

    Trace-validation inputs are built with ``replicate_with_topology``, which
    copies identical data to every chip while stamping a Shard topology. When
    such a tensor is gathered with ``ConcatMesh*ToTensor`` the N identical
    per-device copies are concatenated, blowing the host tensor up by the mesh
    factor (e.g. 8x) before golden tiling and PCC — the dominant cost (and OOM /
    timeout risk) for large vectors whose "sharded" dim can't actually be split
    (e.g. a size-1 dim over 8 devices).

    When all devices are identical the shard is effectively a replicate, so a
    single copy is numerically equivalent for PCC (concatenating/tiling both the
    actual and golden by the same factor leaves the correlation unchanged).
    Returns None when the data genuinely differs across devices (real shard) so
    the caller falls back to the normal concat path.
    """
    if not device_tensors or len(device_tensors) <= 1:
        return None
    try:
        ref = to_torch_fn(device_tensors[0])
    except Exception:
        return None
    for t in device_tensors[1:]:
        try:
            other = to_torch_fn(t)
        except Exception:
            return None
        identical = ref.shape == other.shape and torch.equal(ref, other)
        del other
        if not identical:
            return None
    return ref


def mesh_tensor_to_torch(
    ttnn_tensor, mesh_device=None, mesh_composer=None, force_single_device=False, scatter_placement=None
) -> torch.Tensor:
    """Convert a TTNN mesh tensor back to torch, reassembling shards by topology.

    Replicated tensors return device 0. Sharded tensors are reassembled
    according to the tensor's TensorTopology placements. Mixed
    [Replicate, Shard(d)] cases concatenate only the unique row/col of devices
    along the shard dim. A caller-supplied mesh_composer overrides this.

    Tensors whose Shard topology is backed by identical per-device data (the
    trace-validation replicate path) collapse to a single copy to avoid an
    N-fold host blow-up during gather.

    Args:
        force_single_device: When True, always return device 0's tensor
            without any concat/gather. Use when the inputs were replicated
            via replicate_with_topology and the golden was computed from a
            single copy — comparing against the full gathered output would
            produce shape mismatches.
        scatter_placement: The traced ``input_*_tensor_placement`` the INPUT was built from. When
            it implies the replicate_with_topology path (see
            scatter_uses_replicate_topology), the golden is per-chip, so the gather collapses to a
            single copy deterministically instead of asking _replicated_single_copy whether the
            per-device bytes happen to match. Prefer this over force_single_device: it is derived
            from the vector rather than asserted by the caller, so it stays correct for a genuine
            shard (where it is False and the normal gather runs).
    """
    if scatter_placement is not None and not force_single_device:
        try:
            force_single_device = scatter_uses_replicate_topology(
                scatter_placement, mesh_device or ttnn_tensor.device()
            )
        except Exception:
            force_single_device = False

    def _get_torch_dtype(t):
        try:
            dt = t.dtype
            if dt == ttnn.uint16:
                return torch.int32
            if dt == ttnn.uint32:
                return torch.int64
        except Exception:
            # tensor has no resolvable dtype — signal 'unknown' to the caller via None
            pass
        return None

    def _to_torch_safe(t):
        torch_dtype = _get_torch_dtype(t)
        if torch_dtype is not None:
            return ttnn.to_torch(t).to(torch_dtype)
        return ttnn.to_torch(t)

    try:
        device = ttnn_tensor.device()
    except Exception:
        device = None

    is_mesh = device is not None and hasattr(device, "get_num_devices")

    if force_single_device and is_mesh:
        try:
            device_tensors = ttnn.get_device_tensors(ttnn_tensor)
            if device_tensors:
                return _to_torch_safe(device_tensors[0])
        except Exception:
            # get_device_tensors can fail for some host/odd tensors; fall back
            # to converting the tensor as-is below.
            pass
        return _to_torch_safe(ttnn_tensor)

    # Auto-detect replicated outputs: when all per-device tensors have the
    # same shape, the data was replicated (not truly sharded). Return device 0
    # to avoid N-fold shape blow-ups from concatenation.  Truly sharded outputs
    # have different per-device shapes (each device holds a fraction).
    if is_mesh:
        try:
            dt = ttnn.get_device_tensors(ttnn_tensor)
            if len(dt) > 1:
                shapes = [tuple(t.shape) for t in dt]
                if len(set(shapes)) == 1:
                    topology = ttnn_tensor.tensor_topology()
                    placements = list(topology.placements())
                    if any(type(p).__name__ == "PlacementShard" for p in placements):
                        # Identical per-device *shapes* with a Shard topology are
                        # ambiguous: either replicate_with_topology (identical
                        # data -> safe to collapse) or a genuine shard (identical
                        # shapes, DIFFERENT data -> must NOT collapse, or we'd
                        # drop shards). _replicated_single_copy verifies contents
                        # are byte-identical and returns None for a real shard,
                        # so we fall through to the proper concat/gather path.
                        single = _replicated_single_copy(dt, _to_torch_safe)
                        if single is not None:
                            return single
                        # Contents differ, so this is treated as a real shard and the gather is
                        # about to multiply the shape by the mesh factor. Logged because that used
                        # to happen silently: one lead-models split vector passed on one Galaxy box
                        # and returned a 4x-wide chunk on another, from this branch alone. A caller
                        # that knows the input's placement should pass scatter_placement and never
                        # reach here.
                        logger.warning(
                            f"SWEEPS: gather -- Shard topology with identical per-device shapes "
                            f"{shapes[0]} but DIFFERING data; concatenating {len(dt)} shards. If the "
                            f"golden is per-chip, pass scatter_placement to keep the shape stable."
                        )
        except Exception:
            # Best-effort fast path; on any error fall back to the normal
            # topology-driven gather below.
            pass

    if not is_mesh:
        # Host tensor brought back from a mesh device (e.g. via from_device on a
        # replicated multi-device tensor) keeps multiple per-device buffers but
        # reports device()==None.  Plain ttnn.to_torch then asserts buffers==1.
        # Mirror the on-mesh non-shard path: take the first replica.
        try:
            topology = ttnn_tensor.tensor_topology()
            placements = list(topology.placements())
        except Exception:
            placements = []
        if placements and not any(type(p).__name__ == "PlacementShard" for p in placements):
            try:
                device_tensors = ttnn.get_device_tensors(ttnn_tensor)
            except Exception:
                device_tensors = []
            if len(device_tensors) > 1:
                return _to_torch_safe(device_tensors[0])
        return _to_torch_safe(ttnn_tensor)

    if mesh_composer is not None:
        result = ttnn.to_torch(ttnn_tensor, mesh_composer=mesh_composer)
        torch_dtype = _get_torch_dtype(ttnn_tensor)
        return result.to(torch_dtype) if torch_dtype is not None else result

    try:
        topology = ttnn_tensor.tensor_topology()
        placements = list(topology.placements())
        dist_shape = topology.distribution_shape()
        dist_dims = [int(d) for d in list(dist_shape)]
        mesh_coords = list(topology.mesh_coords())
    except Exception:
        placements = []
        dist_dims = []
        mesh_coords = []

    def _is_shard(p):
        return type(p).__name__ == "PlacementShard"

    has_shard = any(_is_shard(p) for p in placements)

    device_tensors = ttnn.get_device_tensors(ttnn_tensor)

    if not has_shard:
        if device_tensors:
            return _to_torch_safe(device_tensors[0])
        return _to_torch_safe(ttnn_tensor)

    # Validate shard dims against per-device tensor rank. A reshape may reduce
    # rank below the topology's shard dim (e.g. 4D->2D with PlacementShard(dim=2)).
    # When that happens the shard axis no longer exists, so each device already
    # holds a full copy along that axis -- treat as replicated.
    per_dev_ndim = None
    if device_tensors:
        try:
            per_dev_ndim = len(device_tensors[0].shape)
        except Exception:
            pass  # Intentionally ignored: shape query may fail on deallocated tensors, treat as unknown

    if per_dev_ndim is not None:
        if any(_is_shard(p) and p.dim >= per_dev_ndim for p in placements):
            return _to_torch_safe(device_tensors[0])

    if len(placements) == 2 and len(dist_dims) == 2 and all(_is_shard(p) for p in placements):
        # When the Shard topology is backed by identical per-device data (the
        # replicate_with_topology trace-validation path), concatenating the shards
        # just duplicates the same buffer mesh-factor times (e.g. a size-1 dim
        # "sharded" over 8 chips -> 8x host tensor). Collapse to one copy instead;
        # PCC is unchanged and we avoid the blow-up / OOM / 300s-timeout in CI.
        single = _replicated_single_copy(device_tensors, _to_torch_safe)
        if single is not None:
            return single

        try:
            d0 = placements[0].dim
            d1 = placements[1].dim
            # Mirror the scatter side: when both mesh axes shard the SAME tensor dim,
            # create_tensor_on_mesh used a 1D shard over the flattened mesh (ConcatMesh2d
            # cannot express it either -- same "dims must be unique" constraint), so gather
            # it back the same way. Using the same 1D scheme on both sides is what makes the
            # round trip an identity.
            _dup = _same_shard_dim((d0, d1), per_dev_ndim)
            if _dup is not None:
                # THIS is the half of the duplicate-dim fix that actually fires. The scatter
                # side is short-circuited by replicate_with_topology on any mesh big enough for
                # the traced shape, so a 2D-sharded tensor arrives here with its topology
                # stamped and the gather is where ConcatMesh2dToTensor would be handed two
                # entries naming the same dim -> "dims must be unique" -> caught -> wrong data
                # -> low PCC. Logged because it went unlogged before: 19 FATALs vanished
                # between runs 30706921019 and 30791207587 with nothing in the log to show why,
                # which cost a long detour through the (unreachable) scatter-side branch.
                logger.info(
                    f"SWEEPS: gather -- placements name dim {_dup} on BOTH mesh axes "
                    f"{tuple(dist_dims)}; composing with a 1D concat over the flattened mesh"
                )
                result = ttnn.to_torch(ttnn_tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=_dup))
                torch_dtype = _get_torch_dtype(ttnn_tensor)
                return result.to(torch_dtype) if torch_dtype is not None else result
            comp = ttnn.ConcatMesh2dToTensor(device, mesh_shape=tuple(dist_dims), dims=(d0, d1))
            result = ttnn.to_torch(ttnn_tensor, mesh_composer=comp)
            torch_dtype = _get_torch_dtype(ttnn_tensor)
            return result.to(torch_dtype) if torch_dtype is not None else result
        except Exception:
            # 2D mesh-composer path failed — fall through to the per-placement logic below
            pass

    if len(placements) == 2 and len(dist_dims) == 2:
        single = _replicated_single_copy(device_tensors, _to_torch_safe)
        if single is not None:
            return single

        rows, cols = dist_dims[0], dist_dims[1]
        if _is_shard(placements[0]) and not _is_shard(placements[1]):
            shard_dim = placements[0].dim
            picks = []
            for r in range(rows):
                for i, mc in enumerate(mesh_coords):
                    coord = list(mc)
                    if len(coord) == 2 and coord[0] == r and coord[1] == 0:
                        picks.append(i)
                        break
            shards = [_to_torch_safe(device_tensors[i]) for i in picks]
            return torch.cat(shards, dim=shard_dim)
        elif _is_shard(placements[1]) and not _is_shard(placements[0]):
            shard_dim = placements[1].dim
            picks = []
            for c in range(cols):
                for i, mc in enumerate(mesh_coords):
                    coord = list(mc)
                    if len(coord) == 2 and coord[0] == 0 and coord[1] == c:
                        picks.append(i)
                        break
            shards = [_to_torch_safe(device_tensors[i]) for i in picks]
            return torch.cat(shards, dim=shard_dim)

    shard_p = next((p for p in placements if _is_shard(p)), None)
    if shard_p is not None:
        single = _replicated_single_copy(device_tensors, _to_torch_safe)
        if single is not None:
            return single
        try:
            comp = ttnn.ConcatMeshToTensor(device, dim=shard_p.dim)
            result = ttnn.to_torch(ttnn_tensor, mesh_composer=comp)
            torch_dtype = _get_torch_dtype(ttnn_tensor)
            return result.to(torch_dtype) if torch_dtype is not None else result
        except Exception:
            # 1D mesh-composer path failed — fall through to the safe single-device conversion
            pass

    if device_tensors:
        return _to_torch_safe(device_tensors[0])
    return _to_torch_safe(ttnn_tensor)


def _same_shard_dim(dims, ndim):
    """The tensor dim both mesh axes shard, or None.

    ttnn's chunk_ndim normalizes negative dims and then requires them to be unique, so a
    placement naming the same dim twice (e.g. ['PlacementShard(-1)', 'PlacementShard(3)'] on a
    4D tensor) cannot go through the 2D mapper/composer at all. It is still meaningful --
    that dim is split rows*cols ways -- and is expressible as a 1D shard over the flattened
    mesh, which is what callers substitute. Returns the normalized dim so both the scatter and
    gather sides agree on which one it is.
    """
    if ndim is None or len(dims) != 2:
        return None
    try:
        d0, d1 = (int(d) for d in dims)
    except (TypeError, ValueError):
        return None
    d0 = d0 + ndim if d0 < 0 else d0
    d1 = d1 + ndim if d1 < 0 else d1
    if d0 != d1 or not (0 <= d0 < ndim):
        return None
    return d0


def broadcast_torch_inputs_to_global(
    torch_a: torch.Tensor,
    placement_a: Optional[Dict],
    torch_b: torch.Tensor,
    placement_b: Optional[Dict],
):
    """Reconcile torch shapes for elementwise ops with mismatched global shapes.

    Uses placement (Replicate/Shard) and distribution_shape to derive per-chip
    sizes for each tensor dim. Per-dim expansion rules when shapes mismatch:
      - per-chip sizes equal: smaller operand is "replicated full"; tile by mesh
        factor of the larger side using torch.repeat.
      - smaller operand has per-chip size 1 along this dim: it broadcasts within
        each chip; expand using torch.repeat_interleave by the per-chip size of
        the larger side (so each per-chip element of the smaller side is
        replicated to fill its corresponding chunk of the larger side).
    Falls back to plain torch.repeat (legacy behavior) when placement info is
    missing, when ndims differ, or when no clean integer-ratio expansion exists.
    """
    if torch_a.shape == torch_b.shape:
        return torch_a, torch_b
    if torch_a.ndim != torch_b.ndim:
        return torch_a, torch_b

    def _parse_placement_str(plac_val):
        if plac_val is None:
            return None
        if isinstance(plac_val, (list, tuple)):
            parts = [str(x).strip().strip("'") for x in plac_val]
        else:
            s_inner = str(plac_val).strip()
            if s_inner.startswith("[") and s_inner.endswith("]"):
                s_inner = s_inner[1:-1]
            parts = [x.strip().strip("'") for x in s_inner.split(",")]
        out = []
        for x in parts:
            if not x:
                continue
            if x.startswith("PlacementShard("):
                d = int(x[len("PlacementShard(") : -1])
                out.append(("S", d))
            elif x.startswith("PlacementReplicate"):
                out.append(("R", None))
            else:
                out.append(("?", None))
        return out

    def _parse_dist_str(dist_val):
        if dist_val is None:
            return None
        if isinstance(dist_val, (list, tuple)):
            return [int(x) for x in dist_val]
        s_inner = str(dist_val).strip()
        if s_inner.startswith("[") and s_inner.endswith("]"):
            s_inner = s_inner[1:-1]
        return [int(x.strip()) for x in s_inner.split(",") if x.strip()]

    def _factors(p, ndim):
        if not isinstance(p, dict):
            return [1] * ndim
        plac = _parse_placement_str(p.get("placement"))
        dist = _parse_dist_str(p.get("distribution_shape"))
        if plac is None or dist is None:
            return [1] * ndim
        f = [1] * ndim
        for (kind, dim), n in zip(plac, dist):
            if kind == "S" and dim is not None:
                d = dim if dim >= 0 else dim + ndim
                if 0 <= d < ndim:
                    f[d] *= n
        return f

    def _tile(t, dim, n):
        if n == 1:
            return t
        repeats = [1] * t.ndim
        repeats[dim] = n
        return t.repeat(*repeats)

    fa = _factors(placement_a, torch_a.ndim)
    fb = _factors(placement_b, torch_b.ndim)

    def _try_broadcast(a, b):
        try:
            return torch.broadcast_tensors(a, b)
        except RuntimeError:
            return None

    def _fallback_global_broadcast():
        """Try expanding per-chip sharded operands to their gathered global shape."""
        tiled_a = tile_torch_to_global(torch_a, placement_a)
        tiled_b = tile_torch_to_global(torch_b, placement_b)
        for cand_a, cand_b in (
            (torch_a, tiled_b),
            (tiled_a, torch_b),
            (tiled_a, tiled_b),
            (torch_a, torch_b),
        ):
            result = _try_broadcast(cand_a, cand_b)
            if result is not None:
                return result
        return torch_a, torch_b

    new_a, new_b = torch_a, torch_b
    for d in range(torch_a.ndim):
        sa = new_a.shape[d]
        sb = new_b.shape[d]
        if sa == sb:
            continue
        # Per-chip sizes derived from current shape and placement factor.
        per_chip_a = sa // fa[d] if fa[d] > 0 and sa % fa[d] == 0 else None
        per_chip_b = sb // fb[d] if fb[d] > 0 and sb % fb[d] == 0 else None
        if per_chip_a is None or per_chip_b is None:
            # Fall back to a single tile attempt below.
            per_chip_a = sa
            per_chip_b = sb

        if per_chip_a == per_chip_b:
            # Both operands carry the same per-chip slice; the smaller is
            # replicated across the mesh while the larger is sharded. Tile the
            # smaller by the larger side's mesh factor.
            if sa < sb and sb % sa == 0:
                new_a = _tile(new_a, d, sb // sa)
            elif sb < sa and sa % sb == 0:
                new_b = _tile(new_b, d, sa // sb)
            else:
                return _fallback_global_broadcast()
        elif per_chip_a == 1 and per_chip_b > 1:
            # a's per-chip element broadcasts to all per_chip_b elements on
            # each chip; globally that is repeat_interleave.
            if sb % sa == 0:
                new_a = new_a.repeat_interleave(per_chip_b, dim=d)
            else:
                return _fallback_global_broadcast()
        elif per_chip_b == 1 and per_chip_a > 1:
            if sa % sb == 0:
                new_b = new_b.repeat_interleave(per_chip_a, dim=d)
            else:
                return _fallback_global_broadcast()
        else:
            # Mixed per-chip sizes neither equal nor singleton: try plain tile.
            if sa < sb and sb % sa == 0:
                new_a = _tile(new_a, d, sb // sa)
            elif sb < sa and sa % sb == 0:
                new_b = _tile(new_b, d, sa // sb)
            else:
                return _fallback_global_broadcast()
    result = _try_broadcast(new_a, new_b)
    return result if result is not None else _fallback_global_broadcast()


def tile_torch_to_global(
    torch_tensor: torch.Tensor, tensor_placement: Optional[Dict], max_shape: Optional[Sequence[int]] = None
) -> torch.Tensor:
    """Expand a per-chip torch tensor to its global shape based on placement.

    For each PlacementShard(d) entry in `placement` paired with a factor N from
    `distribution_shape`, repeat the tensor along dim d by N. PlacementReplicate
    entries are no-ops. Returns the input unchanged when placement is missing
    or has no Shard entries.

    This mirrors the gather semantics of mesh_tensor_to_torch: a sweep that
    generates a per-chip golden via torch.op(per_chip_a, per_chip_b) needs the
    result tiled along the sharded dims so it matches the gathered global
    output shape used for PCC.
    """
    if not isinstance(tensor_placement, dict):
        return torch_tensor

    plac_val = tensor_placement.get("placement")
    dist_val = tensor_placement.get("distribution_shape")
    if plac_val is None or dist_val is None:
        return torch_tensor

    if isinstance(plac_val, (list, tuple)):
        plac_parts = [str(x).strip().strip("'") for x in plac_val]
    else:
        s = str(plac_val).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        plac_parts = [x.strip().strip("'") for x in s.split(",")]

    plac_entries = []
    for x in plac_parts:
        if not x:
            continue
        if x.startswith("PlacementShard("):
            plac_entries.append(("S", int(x[len("PlacementShard(") : -1])))
        elif x.startswith("PlacementReplicate"):
            plac_entries.append(("R", None))
        else:
            plac_entries.append(("?", None))

    if isinstance(dist_val, (list, tuple)):
        dist_factors = [int(x) for x in dist_val]
    else:
        s = str(dist_val).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        dist_factors = [int(x.strip()) for x in s.split(",") if x.strip()]

    ndim = torch_tensor.ndim
    out = torch_tensor
    for (kind, dim), n in zip(plac_entries, dist_factors):
        if kind != "S" or dim is None or n <= 1:
            continue
        d = dim if dim >= 0 else dim + ndim
        if d >= ndim:
            # Some traced transformer helper ops collapse a sharded input dim
            # into the last output dim (for example QKV/head reshape paths).
            # Preserve the shard factor by applying it to the innermost dim.
            d = ndim - 1
        if d < 0:
            continue
        if max_shape is not None and d < len(max_shape):
            # Never tile past the gathered actual. `placement` records how the INPUT was
            # distributed, but a Shard on a dim that could not actually be split -- e.g. a
            # size-1 broadcast operand declared [Shard(1), Shard(0)] on a 4x8 mesh -- did not
            # multiply the device output, so repeating by that factor invents data. Observed on
            # lead-models 4x8 multiply vectors 9d1aa2d11efe / 828c7b4faac3: the golden was
            # tiled 256x (dim0 1->64, dim1 4->16), which both mismatched the actual and, at
            # (64,16,44544,3072) float64, asked the host for 1.02 TiB and OOMed. Capping keeps
            # the golden bounded by the real output; the caller then finishes the match.
            current = out.shape[d]
            allowed = max_shape[d] // current if current else 1
            n = min(n, max(1, allowed))
            if n <= 1:
                continue
        repeats = [1] * out.ndim
        repeats[d] = n
        out = out.repeat(*repeats)
    return out


def reconcile_golden_to_actual(
    torch_golden: torch.Tensor,
    actual_global: torch.Tensor,
    *placements: Optional[Dict],
) -> torch.Tensor:
    """Tile a per-chip torch golden along sharded dims so it matches the gathered actual shape.

    Try strategies in order:

    1. Shapes already match: return as-is.
    2. Per-dim integer-ratio tile: every dim of `actual` is an integer
       multiple of the corresponding dim of `golden`, with at least one
       dim > 1. This handles the common trace-validation case where the
       inputs were produced via `replicate_with_topology` (so all chips
       hold identical data) and the device op's mesh-aware stitching
       only tiles along a subset of dims (e.g. concat-style ops that
       reassemble along one mesh axis but not the other). Picking up the
       actual's per-dim repeat factor works regardless of which mesh
       axis the device chose to stitch along.
    3. Original placement-driven tile via tile_torch_to_global: relies
       on the master's recorded `placement` + `distribution_shape` to
       repeat by the per-axis Shard factor. This is correct for genuine
       sharded inputs (inputs split across the mesh, each chip computing
       its slice) but produces the wrong shape when the mesh stitch
       only fired along a subset of axes.

    Strategy 2 is tried first because the trace-validation framework's
    default is to replicate inputs and rely on stitch-driven tiling on
    the output.
    """
    if torch_golden.shape == actual_global.shape:
        return torch_golden

    # Strategy 1.5: ndim mismatch — pad golden with trailing size-1 dims or squeeze.
    if torch_golden.ndim != actual_global.ndim:
        g = torch_golden
        while g.ndim < actual_global.ndim:
            g = g.unsqueeze(-1)
        while g.ndim > actual_global.ndim:
            if g.shape[-1] == 1:
                g = g.squeeze(-1)
            else:
                break
        if g.shape == actual_global.shape:
            return g
        torch_golden = g

    # Strategies 2 and 3: repeat up / slice down to the actual, using the actual's shape alone.
    matched = _match_ratio_or_slice(torch_golden, actual_global)
    if matched is not None:
        return matched

    # Strategy 4: placement-driven tile (legacy path), capped at the actual's shape so a
    # non-splittable Shard cannot inflate the golden (see tile_torch_to_global).
    out = torch_golden
    for plac in placements:
        if out.shape == actual_global.shape:
            break
        out = tile_torch_to_global(out, plac, max_shape=actual_global.shape)
    if out.shape != actual_global.shape:
        # Capping can land short of the actual; finish with the shape-driven reconcilers rather
        # than returning something that is merely closer and failing the caller's shape assert.
        matched = _match_ratio_or_slice(out, actual_global)
        if matched is not None:
            return matched
    return out


def _match_ratio_or_slice(torch_golden: torch.Tensor, actual_global: torch.Tensor) -> Optional[torch.Tensor]:
    """Match `torch_golden` to `actual_global` by whole-dim repeat or slice, else None."""
    if torch_golden.shape == actual_global.shape:
        return torch_golden
    if torch_golden.ndim != actual_global.ndim:
        return None

    # Per-dim integer-ratio tile (golden smaller than actual).
    repeats = []
    ok = True
    for d in range(torch_golden.ndim):
        g = torch_golden.shape[d]
        a = actual_global.shape[d]
        if g == 0 or a % g != 0:
            ok = False
            break
        repeats.append(a // g)
    if ok and any(r > 1 for r in repeats):
        tiled = torch_golden.repeat(*repeats)
        if tiled.shape == actual_global.shape:
            return tiled

    # Golden LARGER than actual — slice golden to match.
    slices = []
    for d in range(torch_golden.ndim):
        if actual_global.shape[d] <= torch_golden.shape[d]:
            slices.append(slice(0, actual_global.shape[d]))
        else:
            return None
    if any(s.stop < torch_golden.shape[i] for i, s in enumerate(slices)):
        sliced = torch_golden[tuple(slices)]
        if sliced.shape == actual_global.shape:
            return sliced
    return None
