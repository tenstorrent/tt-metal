# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from loguru import logger

import ttnn

from ..layers.module import Module
from . import walltime

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

CACHE_DICT_FILE = "cache_dict.json"


def _first_param_device(module: Module):
    """Find the mesh device backing the first Parameter under ``module`` (recursive)."""
    for _, parameter in module.named_parameters():
        return parameter.device
    for _, child in module.named_children():
        device = _first_param_device(child)
        if device is not None:
            return device
    return None


def _w7_h2d_probe(tt_model: Module, cache_dir: Path, subfolder: str) -> None:
    """Empirically characterize the cached H2D weight-load path (W7 mission).

    Entirely gated by ``LTX_W7_PROBE`` (default off) so golden runs are byte-identical —
    when the flag is unset this function returns immediately and touches nothing.

    When enabled it loads a sample of the largest cached ``.tensorbin`` shards and times:
      * host-only serial/parallel (device=None)  -> mmap+flatbuffer parse cost + GIL behaviour
      * device serial (host+H2D)                 -> the production path's effective GB/s
      * device parallel W=2/4/8 (ThreadPool)     -> does concurrent Python-issued H2D beat serial?
    All probe tensors are deallocated before returning; the real load is untouched.
    """
    flag = os.environ.get("LTX_W7_PROBE", "0")
    if flag in ("0", "", "false", "False"):
        return
    want = os.environ.get("LTX_W7_PROBE_SUBFOLDER", "transformer")
    if want and subfolder != want:
        return

    import glob
    from concurrent.futures import ThreadPoolExecutor

    raw_load = ttnn._ttnn.tensor.load_tensor_flatbuffer

    try:
        device = _first_param_device(tt_model)
        if device is None:
            logger.warning("W7-PROBE: no device found on model; skipping")
            return

        files = sorted(
            glob.glob(str(cache_dir / "*.tensorbin")),
            key=os.path.getsize,
            reverse=True,
        )
        if not files:
            logger.warning(f"W7-PROBE: no .tensorbin under {cache_dir}; skipping")
            return

        n = int(os.environ.get("LTX_W7_PROBE_N", "24"))
        sample = files[:n]
        total_bytes = sum(os.path.getsize(p) for p in sample)
        gb = total_bytes / 1e9
        logger.info(
            f"W7-PROBE: subfolder={subfolder} sample={len(sample)}/{len(files)} files "
            f"total={gb:.3f} GB device_shape={tuple(device.shape)}"
        )

        # Warm the page cache uniformly so every timed phase hits RAM and we isolate the
        # transfer cost from cold-disk reads.
        for p in sample:
            with open(p, "rb") as fh:
                while fh.read(1 << 24):
                    pass

        def _rate(t: float) -> float:
            return gb / t if t > 0 else float("nan")

        def _dealloc(tensors) -> None:
            for t in tensors:
                try:
                    ttnn.deallocate(t)
                except Exception:  # noqa: BLE001 - best-effort cleanup
                    pass

        # (1) host-only serial: mmap + flatbuffer parse, NO H2D.
        t = time.monotonic()
        hosts = [raw_load(p, None) for p in sample]
        t_host = time.monotonic() - t
        del hosts

        # (2) host-only parallel(8): does the parse release the GIL?
        t = time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as ex:
            hosts = list(ex.map(lambda p: raw_load(p, None), sample))
        t_host_par = time.monotonic() - t
        del hosts

        # (3) device serial: full host + H2D (production path).
        t = time.monotonic()
        devs = [raw_load(p, device) for p in sample]
        ttnn.synchronize_device(device)
        t_dev = time.monotonic() - t
        _dealloc(devs)
        del devs

        logger.info(f"W7-PROBE: host_only_serial {t_host:8.3f}s {_rate(t_host):7.2f} GB/s (parse, GIL held)")
        logger.info(f"W7-PROBE: host_only_par8   {t_host_par:8.3f}s {_rate(t_host_par):7.2f} GB/s (parse concurrent)")
        logger.info(f"W7-PROBE: device_serial    {t_dev:8.3f}s {_rate(t_dev):7.2f} GB/s (host+H2D, baseline)")
        implied = t_dev - t_host
        logger.info(
            f"W7-PROBE: implied_H2D      {implied:8.3f}s {_rate(implied):7.2f} GB/s (device_serial - host_serial)"
        )

        # (4) device parallel: does concurrent Python-issued H2D beat serial?
        for w in (2, 4, 8):
            t = time.monotonic()
            with ThreadPoolExecutor(max_workers=w) as ex:
                devs = list(ex.map(lambda p: raw_load(p, device), sample))
            ttnn.synchronize_device(device)
            t_par = time.monotonic() - t
            _dealloc(devs)
            del devs
            speedup = t_dev / t_par if t_par > 0 else float("nan")
            logger.info(
                f"W7-PROBE: device_par{w:<2d}     {t_par:8.3f}s {_rate(t_par):7.2f} GB/s ({speedup:4.2f}x vs serial)"
            )

        ttnn.synchronize_device(device)
        logger.info("W7-PROBE: done; probe tensors deallocated")
    except Exception as err:  # noqa: BLE001 - probe must never break the real run / md5
        logger.warning(f"W7-PROBE: failed ({type(err).__name__}: {err}); continuing with normal load")


def _wc_derep_probe(tt_model: Module, cache_dir: Path, subfolder: str) -> None:
    """Microbench (WC / ITER60): can the SP-replicated DiT H2D be de-replicated
    *net-positively* and byte-identically?  Gated by ``LTX_DEREP_PROBE`` (default off);
    when unset it returns immediately and is byte-identical / zero-overhead.

    W7 showed the cached transformer weights are TP-sharded but **SP-replicated**, so the
    stored 37 GB is H2D'd ~4x (once per SP device). W7's recommended fix is to H2D each TP
    shard **once** (the SP=0 column, 1x bytes) then replicate across the SP axis on the
    intra-mesh fabric instead of over PCIe. Every clean Python re-shard primitive
    (``distribute_tensor``/``aggregate_tensor``) instead *untilizes* the full logical tensor
    (``MeshToTensor``/``TensorToMesh`` -> ``to_vector`` in ``distributed_tensor.cpp``), a
    ~37 GB host-CPU cost that dwarfs the ~9 s H2D saving. This probe measures the
    **tile-preserving CLEAN chain** (``allocate_tensor_on_device`` full-mesh w/ 0 H2D ->
    ``copy_host_to_device_tensor`` of ONLY the SP=0 column w/ 1x H2D -> ``point_to_point``
    scatter across the SP axis) to check (a) it actually cuts H2D ~1/SP and (b) it reproduces
    the replicated bytes exactly, before any real load path is changed. Probe-only: it loads
    and frees its own tensors; the timed real load below is untouched, so golden md5 is preserved.
    """
    flag = os.environ.get("LTX_DEREP_PROBE", "0")
    if flag in ("0", "", "false", "False"):
        return
    want = os.environ.get("LTX_DEREP_PROBE_SUBFOLDER", "transformer")
    if want and subfolder != want:
        return

    import glob

    try:
        import torch

        device = _first_param_device(tt_model)
        if device is None:
            logger.warning("DEREP-PROBE: no device found on model; skipping")
            return
        mesh = list(device.shape)
        # cache key CP1_0_TP2_0_SP4_1 => SP is mesh axis 1 (size 4), TP is axis 0 (size 2).
        sp_axis = int(os.environ.get("LTX_DEREP_SP_AXIS", "1"))
        if sp_axis >= len(mesh) or mesh[sp_axis] <= 1:
            logger.warning(f"DEREP-PROBE: sp_axis={sp_axis} not shardable for mesh {tuple(mesh)}; skipping")
            return
        sp = mesh[sp_axis]
        logger.info(f"DEREP-PROBE: mesh={tuple(mesh)} sp_axis={sp_axis} sp={sp}")

        files = sorted(glob.glob(str(cache_dir / "*.tensorbin")), key=os.path.getsize, reverse=True)
        if not files:
            logger.warning(f"DEREP-PROBE: no .tensorbin under {cache_dir}; skipping")
            return
        n = int(os.environ.get("LTX_DEREP_PROBE_N", "2"))
        for path in files[:n]:
            _derep_probe_one(device, mesh, sp_axis, sp, path, torch)
        ttnn.synchronize_device(device)
        logger.info("DEREP-PROBE: done; probe tensors deallocated")
    except Exception as err:  # noqa: BLE001 - probe must never break the real run / md5
        logger.warning(f"DEREP-PROBE: failed ({type(err).__name__}: {err}); continuing with normal load")


def _sp_scatter_inplace(F, rows: int, cols: int, sp_axis: int) -> int:
    """In-place device->device broadcast of the SP=0 column/row to the other SP coords of the
    full-mesh tensor ``F`` via ``ttnn.point_to_point`` (the deepseek ``mesh_scatter`` pattern).
    Returns the number of point_to_point ops issued."""
    topo = ttnn.Topology.Linear
    n = 0
    if sp_axis == 1:  # SP is the column axis: broadcast col 0 -> cols 1..cols-1, per row.
        for r in range(rows):
            src = ttnn.MeshCoordinate(r, 0)
            for c in range(1, cols):
                ttnn.point_to_point(F, src, ttnn.MeshCoordinate(r, c), output_tensor=F, topology=topo)
                n += 1
    else:  # SP is the row axis: broadcast row 0 -> rows 1..rows-1, per col.
        for c in range(cols):
            src = ttnn.MeshCoordinate(0, c)
            for r in range(1, rows):
                ttnn.point_to_point(F, src, ttnn.MeshCoordinate(r, c), output_tensor=F, topology=topo)
                n += 1
    return n


def _derep_probe_one(device, mesh, sp_axis: int, sp: int, path: str, torch) -> None:
    """One-tensor de-rep microbench: inspect, baseline (4x) load, then the CLEAN de-rep chain
    (allocate full-mesh w/ 0 H2D -> write only the SP=0 column w/ 1x H2D -> point_to_point
    scatter across the SP axis) with per-coord byte-verify + end-to-end timing vs baseline.
    Fully logged so a partial failure still tells us which building block is missing."""

    def _dealloc(*ts):
        for t in ts:
            try:
                ttnn.deallocate(t)
            except Exception:  # noqa: BLE001
                pass

    name = Path(path).name
    fsz = os.path.getsize(path)

    # (0) INSPECT the on-disk host tensor.
    host = ttnn.load_tensor(path, device=None)
    hshards = ttnn.get_device_tensors(host)
    logger.info(
        f"DEREP-PROBE [{name}] file={fsz / 1e6:.1f}MB host_shape={tuple(host.shape)} "
        f"layout={host.layout} dtype={host.dtype} n_host_shards={len(hshards)} "
        f"shard0_shape={tuple(hshards[0].shape)}"
    )

    # (1) BASELINE: the production replicated load (4x H2D). Record golden per-coord bytes.
    ttnn.synchronize_device(device)
    t = time.monotonic()
    ref = ttnn.load_tensor(path, device=device)
    ttnn.synchronize_device(device)
    t_base = time.monotonic() - t
    ref_shards = ttnn.get_device_tensors(ref)
    gold = [ttnn.to_torch(s) for s in ref_shards]
    logger.info(
        f"DEREP-PROBE [{name}] BASELINE(4x) {t_base:.3f}s  n_coords={len(ref_shards)} "
        f"coord_shape={tuple(ref_shards[0].shape)}  (ideal 1x = {t_base / sp:.3f}s)"
    )

    # ---- CLEAN de-rep candidate: allocate full-mesh (0 H2D) + write ONLY the SP=0 column
    # (1x H2D) + on-device point_to_point scatter across the SP axis, then per-coord byte-verify.
    # This uses only exposed Python APIs (allocate_tensor_on_device / copy_host_to_device_tensor /
    # point_to_point) and never creates a submesh, so it has no teardown lifecycle hazard.
    rows = mesh[0]
    cols = mesh[1] if len(mesh) > 1 else 1
    mesh_size = rows * cols
    # Row-major coord index -> (row, col). SP=0 column (sp_axis==1) is col 0 => idx r*cols;
    # SP=0 row (sp_axis==0) is row 0 => idx c.
    if sp_axis == 1:
        sp0_idx = [r * cols for r in range(rows)]
    else:
        sp0_idx = list(range(cols))
    F = None
    if len(hshards) != mesh_size:
        logger.warning(
            f"DEREP-CLEAN [{name}] n_host_shards={len(hshards)} != mesh_size={mesh_size}; skipping clean chain"
        )
    else:
        try:
            memcfg = ref.memory_config()
            F = ttnn.allocate_tensor_on_device(hshards[0].shape, host.dtype, host.layout, device, memcfg)
            Fshards = ttnn.get_device_tensors(F)
            if len(Fshards) != mesh_size:
                logger.warning(f"DEREP-CLEAN [{name}] allocate gave {len(Fshards)} coords (want {mesh_size}); skipping")
            else:
                # (1x H2D) write ONLY the SP=0 column coords into the pre-allocated full-mesh tensor.
                ttnn.synchronize_device(device)
                t = time.monotonic()
                for i in sp0_idx:
                    ttnn.copy_host_to_device_tensor(hshards[i], Fshards[i])
                ttnn.synchronize_device(device)
                t_1x = time.monotonic() - t
                # (scatter) point_to_point the SP=0 column to the other SP coords, in place on F.
                ttnn.synchronize_device(device)
                t = time.monotonic()
                n_p2p = _sp_scatter_inplace(F, rows, cols, sp_axis)
                ttnn.synchronize_device(device)
                t_sc = time.monotonic() - t
                # (verify) every coord of F must be byte-identical to the golden 4x load.
                F2 = ttnn.get_device_tensors(F)
                ok = all(torch.equal(ttnn.to_torch(F2[i]), gold[i]) for i in range(mesh_size))
                t_derep = t_1x + t_sc
                sp_speed = t_base / t_derep if t_derep > 0 else float("nan")
                logger.info(
                    f"DEREP-CLEAN [{name}] 1xH2D(SP0={len(sp0_idx)}coords)={t_1x:.3f}s + "
                    f"scatter({n_p2p}xp2p)={t_sc:.3f}s = {t_derep:.3f}s vs baseline(4x) {t_base:.3f}s "
                    f"({sp_speed:.2f}x)  BYTE_IDENTICAL={ok}"
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"DEREP-CLEAN [{name}] FAILED: {type(e).__name__}: {e}")

    _dealloc(ref, host)
    if F is not None:
        _dealloc(F)


def _load_torch_with_logging(tt_model: Module, get_torch_state_dict: Callable[[], dict], subfolder: str) -> None:
    """Bracket the otherwise-silent source-weight read with logs. ``get_torch_state_dict``
    reads tens of GB of safetensors off disk; without this the load looks hung."""
    logger.info(f"{subfolder}: reading source weights from disk...")
    t = time.monotonic()
    state = get_torch_state_dict()
    logger.info(f"{subfolder}: read {len(state)} tensors in {time.monotonic() - t:.0f}s; converting to device...")
    tt_model.load_torch_state_dict(state)


class MissingCacheError(Exception):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def __str__(self) -> str:
        return f"cache does not exist at '{self.path}'"


def config_id(parallel_config):
    config_id = ""
    for n, v in parallel_config._asdict().items():
        if v is not None:
            config_id += f"{''.join([w[0].upper() for w in n.split('_')])}{v.factor}_{v.mesh_axis}_"
    return config_id


def cache_dir_is_set() -> bool:
    return _cache_root() is not None


def load_model(
    tt_model: Module,
    *,
    model_name: str,
    subfolder: str,
    parallel_config: NamedTuple,
    mesh_shape: Sequence[int],
    dtype: str = "bf16",
    is_fsdp: bool = False,
    get_torch_state_dict: Callable[[], dict] | None = None,
    create_cache: bool = True,
) -> None:
    """
    Load model weights from cache or PyTorch state dict.

    Attempts to load from cache first. If the cache does not exist, loads from PyTorch state dict
    (if provided) and optionally creates the cache. Raises `MissingCacheError` if neither is
    available. Finally, any module that needs to be offloaded is taken care of.

    Args:
        `tt_model`: TT model instance to load weights into.
        `model_name`: Model name (e.g., "flux1-dev", "stable-diffusion-3.5").
        `subfolder`: Subfolder within model cache directory (e.g., "transformer", "vae").
        `parallel_config`: Parallelism configuration (tensor/sequence parallel).
        `mesh_shape`: Device mesh shape.
        `dtype`: Data type for cached weights (default: "bf16").
        `is_fsdp`: Whether FSDP is used (default: False).
        `get_torch_state_dict`: Optional callable returning PyTorch state dict. Enables lazy
            evaluation - PyTorch model only loads if the cache does not exist. If `None`, cache
            must exist or `MissingCacheError` is raised.
        `create_cache`: Create cache after loading from PyTorch (default: True).

    Raises:
        `MissingCacheError`: Cache does not exist and `get_torch_state_dict` is `None`.
        `RuntimeError`: `TT_DIT_CACHE_DIR` is not set and `get_torch_state_dict` is `None`.
    """
    if tt_model.is_loaded():
        return

    t0 = time.monotonic()
    cache_dir = model_cache_dir(
        model_name=model_name,
        subfolder=subfolder,
        parallel_config=parallel_config,
        mesh_shape=mesh_shape,
        dtype=dtype,
        is_fsdp=is_fsdp,
        required=get_torch_state_dict is None,
    )

    if cache_dir is None:
        assert get_torch_state_dict is not None

        # cache_dir is None only when TT_DIT_CACHE_DIR is unset and the caller made the cache
        # optional (get_torch_state_dict provided). That silently turns the one-time weight
        # conversion into a per-run tax: nothing is ever persisted, so every process reconverts
        # from the raw checkpoint. Warn loudly — the info-level breadcrumb below is easy to miss.
        logger.warning(
            f"{model_name}/{subfolder}: TT_DIT_CACHE_DIR is unset — converting weights from the raw "
            f"checkpoint and NOT caching them; every run reconverts. Set TT_DIT_CACHE_DIR to persist "
            f"converted device weights."
        )
        logger.info(f"{subfolder}: no device cache (set TT_DIT_CACHE_DIR to cache converted weights).")
        with walltime.timed("weight_load", f"{model_name}/{subfolder}", cached=False):
            _load_torch_with_logging(tt_model, get_torch_state_dict, subfolder)
        ttnn.distributed_context_barrier()
        logger.info(f"{subfolder}: loaded in {time.monotonic() - t0:.0f}s")
        return

    if Path(cache_dir).is_dir():
        logger.info(f"{subfolder}: loading cached device weights from '{cache_dir}'...")
        _w7_h2d_probe(tt_model, Path(cache_dir), subfolder)
        _wc_derep_probe(tt_model, Path(cache_dir), subfolder)
        with walltime.timed("weight_load", f"{model_name}/{subfolder}", cached=True):
            tt_model.load(cache_dir)
        ttnn.distributed_context_barrier()
        logger.info(f"{subfolder}: loaded from cache in {time.monotonic() - t0:.0f}s")
        return

    if get_torch_state_dict is None:
        raise MissingCacheError(cache_dir)

    logger.info(f"{subfolder}: device cache miss at '{cache_dir}'.")
    with walltime.timed("weight_load", f"{model_name}/{subfolder}", cached=False):
        _load_torch_with_logging(tt_model, get_torch_state_dict, subfolder)

    # If distributed, ensure that all processes have completed the check whether cache_dir exists,
    # before any rank might proceed to create that dir to save.
    ttnn.distributed_context_barrier()

    if create_cache:
        ts = time.monotonic()
        logger.info(f"{subfolder}: writing device cache to '{cache_dir}'...")
        tt_model.save(cache_dir)
        logger.info(f"{subfolder}: cached to disk in {time.monotonic() - ts:.0f}s")

    logger.info(f"{subfolder}: ready in {time.monotonic() - t0:.0f}s")


def model_cache_dir(
    *,
    model_name: str,
    subfolder: str,
    parallel_config: NamedTuple,
    mesh_shape: Sequence[int],
    dtype: str = "bf16",
    is_fsdp: bool = False,
    required: bool = True,
) -> Path | None:
    cache_dir = _cache_root()
    if cache_dir is None:
        if required:
            msg = "Cache is required. Set the TT_DIT_CACHE_DIR environment variable."
            raise RuntimeError(msg)
        return None

    parallel_key = config_id(parallel_config)
    mesh_key = "x".join(str(x) for x in mesh_shape)

    key = f"{parallel_key}mesh{mesh_key}_{dtype}"
    if is_fsdp:
        key += "_FSDP"

    return Path(cache_dir) / model_name / subfolder / key


def _cache_root() -> str | None:
    return os.environ.get("TT_DIT_CACHE_DIR")
