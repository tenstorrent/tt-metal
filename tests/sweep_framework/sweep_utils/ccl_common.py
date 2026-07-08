# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import atexit
import os
from contextlib import contextmanager
from itertools import count, takewhile
from math import ceil, prod

import ttnn

from loguru import logger
from tests.scripts.common import get_updated_device_params


def mesh_shape_iterator(num_devices, limit=None):
    if num_devices == 1:
        return []

    assert num_devices % 2 == 0
    for r in takewhile(lambda x: x <= num_devices, (2**i for i in (range(limit) if limit else count()))):
        yield (num_devices // r, r)


# ── Open-mesh cache (group-by-fabric optimization) ───────────────────────────
# Opening a mesh + bringing up the fabric control plane is the dominant per-vector
# cost for CCL sweeps — each device_context() previously tore the mesh + fabric
# down and re-initialized them for EVERY vector ("reinitializing control plane",
# ~13s/config for FABRIC_2D on galaxy). Cache the opened mesh and reuse it across
# consecutive device_context() calls that share the same
# (mesh_shape, fabric_config, device_params, full_mesh_shape); only re-init when
# the key changes. The cache is invalidated (device closed, fabric reset) when the
# key changes, when the in-context body raises/times out (so a possibly-wedged
# device is never reused), and at process exit. Set
# TTNN_SWEEP_DISABLE_DEVICE_CACHE=1 to fall back to open/close-per-vector.
_DEVICE_CACHE = {"key": None, "mesh_device": None, "parent_device": None}


def _device_cache_key(mesh_shape, fabric_config, device_params, full_mesh_shape):
    return (
        tuple(mesh_shape),
        str(fabric_config),
        tuple(sorted((str(k), str(v)) for k, v in (device_params or {}).items())),
        tuple(full_mesh_shape) if full_mesh_shape else None,
    )


def _teardown_cached_device():
    """Close the cached mesh/parent device (if any) and restore fabric to DISABLED."""
    parent = _DEVICE_CACHE.get("parent_device")
    mesh = _DEVICE_CACHE.get("mesh_device")
    _DEVICE_CACHE["key"] = None
    _DEVICE_CACHE["mesh_device"] = None
    _DEVICE_CACHE["parent_device"] = None
    try:
        if parent is not None:
            # The carved submesh shares the parent's command queue; quiesce drains
            # all CQs so the close is clean (see open path note below).
            try:
                parent.quiesce_devices()
            except Exception:
                logger.opt(exception=True).warning("quiesce_devices during teardown failed")
            ttnn.close_mesh_device(parent)
        elif mesh is not None:
            ttnn.close_mesh_device(mesh)
    except Exception:
        logger.opt(exception=True).warning("close_mesh_device during cached teardown failed")
    finally:
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            # best-effort; fabric may already be disabled during teardown
            pass


# Close the last cached device gracefully when the (child) process exits.
atexit.register(_teardown_cached_device)


@contextmanager
def device_context(mesh_shape, fabric_config, device_params=None, full_mesh_shape=None, disable_cache=False):
    """Open a mesh (or a submesh of the full galaxy) with the given fabric config.

    full_mesh_shape: when set and != mesh_shape, open the FULL galaxy mesh first
    (so fabric bring-up runs over the whole, healthy ethernet topology) and then
    carve `mesh_shape` out of it via create_submesh. Opening MeshShape(submesh)
    DIRECTLY on a galaxy fails fabric router sync on the submesh's boundary
    ethernet links (they connect to chips outside the carved region); a submesh of
    the already-synced full-mesh fabric works.

    The opened mesh is cached and reused across consecutive calls with the same
    (mesh_shape, fabric_config, device_params, full_mesh_shape) to avoid
    re-initializing the fabric control plane on every vector (the dominant CCL
    per-config cost). See the _DEVICE_CACHE note above; opt out with
    TTNN_SWEEP_DISABLE_DEVICE_CACHE=1.
    """
    device_params = device_params or {}
    cache_enabled = (not disable_cache) and os.environ.get(
        "TTNN_SWEEP_DISABLE_DEVICE_CACHE", ""
    ).strip().lower() not in ("1", "true", "yes")
    key = _device_cache_key(mesh_shape, fabric_config, device_params, full_mesh_shape)

    # Fast path: reuse the cached device when the config matches — no teardown, no
    # fabric re-init, no re-open.
    if cache_enabled and _DEVICE_CACHE["key"] == key and _DEVICE_CACHE["mesh_device"] is not None:
        logger.info("Reusing cached device (same mesh_shape/fabric_config)")
        try:
            yield _DEVICE_CACHE["mesh_device"], None
        except BaseException:
            # Body failed/timed out on this device — drop it so the next vector
            # opens a fresh one (never reuse a possibly-wedged device).
            _teardown_cached_device()
            raise
        return

    # Key changed (or caching off / first call): tear down any cached device, then
    # open a fresh one.
    #
    # NOTE: do NOT clear the persisted kernel cache here. Wiping
    # ~/.cache/tt-metal-cache mid-run also deletes the base *firmware* objects
    # (which define globals like my_x/my_y/noc_*_num_issued, built once at the
    # first device open). The control-plane reinit on a fabric change rebuilds the
    # fabric_erisc_router kernel but NOT that firmware, so the fresh kernel fails
    # to link ("undefined reference to noc_posted_writes_num_issued ...",
    # build.cpp:67 "Failed to generate binaries for fabric_erisc_router") -> the op
    # hangs -> FAIL_CRASH_HANG (observed on T3K 2x4 all_gather, run 27607824383).
    # Cross-process stale-ELF staleness is handled by the workflow's
    # "Clear stale kernel cache" step, which runs BEFORE this process starts so the
    # firmware rebuilds cleanly at device open.
    _teardown_cached_device()
    parent_device = None
    mesh_device = None
    try:
        logger.info("Setting up device")
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        ttnn.set_fabric_config(fabric_config)
        if full_mesh_shape is not None and tuple(full_mesh_shape) != tuple(mesh_shape):
            logger.info(f"Opening full mesh {tuple(full_mesh_shape)} then carving submesh {tuple(mesh_shape)}")
            parent_device = ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(full_mesh_shape), **get_updated_device_params(device_params)
            )
            mesh_device = parent_device.create_submesh(ttnn.MeshShape(mesh_shape))
        else:
            mesh_device = ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(mesh_shape), **get_updated_device_params(device_params)
            )
    except AssertionError as e:
        logger.error(f"Device error: {e}")
        # Clean up a partially-opened device; do not cache, restore fabric.
        try:
            if parent_device is not None:
                ttnn.close_mesh_device(parent_device)
            elif mesh_device is not None:
                ttnn.close_mesh_device(mesh_device)
        except Exception:
            # best-effort teardown; a close failure must not mask the test result
            pass
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            # best-effort; fabric may already be disabled during teardown
            pass
        yield None, f"Device error {e}"
        return

    if not cache_enabled:
        # Original open/close-per-vector behavior.
        try:
            yield mesh_device, None
        finally:
            logger.info("Tearing down device")
            try:
                if parent_device is not None:
                    try:
                        parent_device.quiesce_devices()
                    except Exception:
                        logger.opt(exception=True).warning("quiesce_devices during teardown failed")
                    ttnn.close_mesh_device(parent_device)
                elif mesh_device:
                    ttnn.close_mesh_device(mesh_device)
            finally:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        return

    # Cache the freshly-opened device and keep it open after a successful body.
    _DEVICE_CACHE["key"] = key
    _DEVICE_CACHE["mesh_device"] = mesh_device
    _DEVICE_CACHE["parent_device"] = parent_device
    try:
        yield mesh_device, None
    except BaseException:
        # Failure/timeout — never reuse a possibly-wedged device.
        _teardown_cached_device()
        raise
    # Success: leave the device open/cached for the next vector (the whole point).


def get_serializable_shard_specs(
    input_shape, input_cores, input_strategy, output_shape, output_cores, output_strategy, valid_tensor_shapes
):
    return {
        "input": {
            "shape": input_shape,
            "cores": input_cores,
            "strategy": input_strategy,
        },
        "output": {"shape": output_shape, "cores": output_cores, "strategy": output_strategy},
        "valid_tensor_shapes": valid_tensor_shapes,
    }


def validate_serializable_shard_spec(input_shape, serializable_shard_specs, dim, cluster_size, scatter_gather=None):
    if serializable_shard_specs is None:
        return True

    if not tuple(input_shape) in list(map(tuple, serializable_shard_specs["valid_tensor_shapes"])):
        return False

    if scatter_gather == "scatter":
        sg_factor = 1 / cluster_size
    elif scatter_gather == "gather":
        sg_factor = cluster_size
    else:
        sg_factor = 1

    output_shape = [int(d * sg_factor) if i == dim else d for i, d in enumerate(input_shape)]
    output_cores = prod(serializable_shard_specs["output"]["cores"])
    idx = -1 if serializable_shard_specs["output"]["strategy"] == "w" else -2

    return output_shape[idx] % output_cores == 0


TILE_SIZE = 32


def _parse_serializable_shard_spec(serializable_shard_spec, mem_layout, output_shape, tile_size=TILE_SIZE):
    assert len(serializable_shard_spec) == 3

    shape, cores, strategy = tuple(serializable_shard_spec.values())

    if strategy == "w":
        strategy, layout = ttnn.ShardStrategy.WIDTH, ttnn.TensorMemoryLayout.WIDTH_SHARDED
    elif strategy == "h":
        strategy, layout = ttnn.ShardStrategy.HEIGHT, ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    else:
        raise RuntimeError("Ivalid shard strategy option")

    if shape is None:
        core_grid = ttnn.CoreGrid(**dict(zip(("x", "y"), cores)))
        shard_spec = ttnn.create_sharded_memory_config(output_shape, core_grid, strategy).shard_spec
        if mem_layout == ttnn.TILE_LAYOUT:
            shard_spec.shape = [ceil(s / TILE_SIZE) * TILE_SIZE for s in shard_spec.shape]
        return shard_spec, layout
    else:
        core_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores[0] - 1, cores[1] - 1))}
        )
        return ttnn.ShardSpec(core_grid, shape, ttnn.ShardOrientation.ROW_MAJOR), layout


def get_mem_configs(buffer_type, serializable_shard_specs, mem_layout, output_shape):
    if serializable_shard_specs is None:
        return ttnn.MemoryConfig(buffer_type=buffer_type), ttnn.MemoryConfig(buffer_type=buffer_type)
    else:
        input_spec, input_layout = _parse_serializable_shard_spec(serializable_shard_specs["input"], mem_layout, None)
        input_config = ttnn.MemoryConfig(input_layout, buffer_type, input_spec)

        output_spec, output_layout = _parse_serializable_shard_spec(
            serializable_shard_specs["output"], mem_layout, output_shape
        )
        output_config = ttnn.MemoryConfig(output_layout, buffer_type, output_spec)

        assert input_layout == output_layout

    return input_config, output_config
