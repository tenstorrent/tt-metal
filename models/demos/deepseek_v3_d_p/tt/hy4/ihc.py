# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview iHC (independent Hyper-Connections, #65891) in the prefill stack.

The device modules are the decode ones (models/experimental/hy4_preview/tt/ihc.py: ``TtIHCWrap``, ``TtIHCHead``,
``ihc_expand``). This file is the glue the prefill block and transformer need around them:

* ``ihc_enabled`` / ``ihc_config``: the switch (``model_cfg.ENABLE_IHC``) and the iHC hyperparameters from the model
  config;
* ``ihc_plan``: the mesh plan. iHC is per token, so it only needs to know where the residual's hidden dimension lives:
  the block's TP axis (axis 1, 4 chips, on the 8x4 mesh in both #65884's layout and the dev SP8 x TP4 one). The sequence split on the other
  axis never meets iHC's one all-reduce per site, which runs inside each group of TP chips;
* the weight cache: the iHC parameters are tiny (per site ``fn`` [8, 24576] + ``base`` [8] + ``scale`` [2], fp32, 0.8 MB)
  and ``TtIHCWrap`` / ``TtIHCHead`` upload them from torch tensors themselves, so they are cached as fp32 torch files
  next to the tensorbins (``layer_{i}.ihc.pt``: both sites of a layer; ``ihc_head.pt``) rather than as tensorbins;
* ``streams_to_host``: device streams back to ``[S, n, C]`` in the reference order.

Residual contract with iHC on: between blocks the residual is the n packed streams, per chip
``[1, 1, S / sp, n * hidden / tp]`` fp32 (each chip's own hidden slice of every stream, chip-major then stream, the
order ``ihc_expand`` of the per-chip embedding produces). Each sublayer still sees ``[1, 1, S / sp, hidden / tp]``
BF16, so attention / MoE / FFN are unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

import ttnn
from models.experimental.hy4_preview.reference.ihc import IHCConfig
from models.experimental.hy4_preview.tt.common import MeshPlan

# Streams stay fp32 end to end (HF computes the iHC parametrization and both mixes in fp32; the golden holds fp32
# streams). The sublayers run BF16.
STREAM_DTYPE = ttnn.float32
SUBLAYER_DTYPE = ttnn.bfloat16

# Checkpoint names (tencent/Hy4-preview).
SITES = {"attn": "hc_attn_layer", "ffn": "hc_mlp_layer"}
HEAD_PREFIX = "model.hc_head"


def ihc_enabled(model_cfg) -> bool:
    return bool(getattr(model_cfg, "ENABLE_IHC", False))


def ihc_config(config, model_cfg) -> IHCConfig:
    """iHC hyperparameters: hidden and the RMS eps from the HF config, the rest from the model config."""
    return IHCConfig(
        dim=int(config.hidden_size),
        n=int(model_cfg.HC_MULT),
        eps=float(model_cfg.HC_EPS),
        norm_eps=float(config.rms_norm_eps),
        magnitude=float(model_cfg.HC_MAGNITUDE),
    )


class PrefillMeshPlan(MeshPlan):
    """``MeshPlan`` whose all-reduce runs on the prefill's persistent CCL semaphores (``TT_CCL``).

    ``ttnn.all_reduce`` (what ``MeshPlan.all_reduce`` calls) takes the composite all-gather path for iHC's narrow
    ``[1, 1, T, 2n + 1]`` mixes and creates its global semaphores per program in L1_SMALL. The prefill opens the mesh
    with 1536 B of L1_SMALL (``GLM_L1_SMALL_SIZE``), which the MLA's own collectives fill: the first iHC all-reduce
    then leaves sparse MLA's all-to-all with no room (TT_FATAL, "Not enough space to allocate ... L1_SMALL").
    Here the sum is an ``all_gather_async`` onto a new leading dim with ``TT_CCL``'s double-buffered semaphores, then a
    local fp32 sum over the gathered copies -- the same order on every chip, so the mixes stay identical across the group.
    """

    def __init__(self, mesh_device, hidden_axis, num_links=None, topology=None):
        super().__init__(mesh_device, hidden_axis=hidden_axis, num_links=num_links, topology=topology)
        from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl

        self.tt_ccl = get_tt_ccl(mesh_device)

    def all_reduce(self, x, axis, memory_config=None):
        if self._tp(axis) == 1:
            return x
        gathered = ttnn.experimental.all_gather_async(
            ttnn.reshape(x, [1, 1, x.shape[-2], x.shape[-1]]),
            dim=1,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=axis),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=axis),
            num_links=self.num_links or 1,
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
            topology=self.topology or ttnn.Topology.Linear,
            cluster_axis=axis,
        )  # [1, tp, T, k]
        x.deallocate(True)
        T, k = gathered.shape[-2], gathered.shape[-1]
        parts = [ttnn.slice(gathered, [0, i, 0, 0], [1, i + 1, T, k]) for i in range(self._tp(axis))]
        gathered.deallocate(True)
        out = parts[0]
        for part in parts[1:]:  # elementwise fp32 adds, in chip order on every chip
            out = ttnn.add(out, part)
        return out


def ihc_plan(mesh_device, tp_axis: int, num_links: Optional[int] = None, topology=None) -> MeshPlan:
    """The hidden dimension is split over the block's TP axis; the other axis (sequence) is irrelevant to iHC."""
    return PrefillMeshPlan(mesh_device, hidden_axis=tp_axis, num_links=num_links, topology=topology)


# ----------------------------------------------------------------------------------------------- checkpoint
def layer_ihc_keys(layer_idx: int) -> dict:
    """``{site: {"fn"|"base"|"scale": checkpoint key}}`` for one decoder layer."""
    p = f"model.layers.{layer_idx}"
    return {site: {f: f"{p}.{name}.hc_pre.hc_{f}" for f in ("fn", "base", "scale")} for site, name in SITES.items()}


HEAD_KEYS = {f: f"{HEAD_PREFIX}.hc_head_{f}" for f in ("fn", "base", "scale")}


def read_layer_ihc(ckpt, layer_idx: int) -> dict:
    """``{"attn": {"fn", "base", "scale"}, "ffn": {...}}`` fp32 from a checkpoint reader with ``get(key)``."""
    return {site: {f: ckpt.get(k).float() for f, k in keys.items()} for site, keys in layer_ihc_keys(layer_idx).items()}


def read_head_ihc(ckpt) -> dict:
    return {f: ckpt.get(k).float() for f, k in HEAD_KEYS.items()}


# ----------------------------------------------------------------------------------------------- weight cache
def layer_cache_file(cache_path, layer_idx: int) -> Path:
    return Path(cache_path) / f"layer_{layer_idx}.ihc.pt"


def head_cache_file(cache_path) -> Path:
    return Path(cache_path) / "ihc_head.pt"


def layer_cache_complete(cache_path, layer_idx: int) -> bool:
    return cache_path is not None and layer_cache_file(cache_path, layer_idx).is_file()


def head_cache_complete(cache_path) -> bool:
    return cache_path is not None and head_cache_file(cache_path).is_file()


def resolve_weights(weights: Optional[dict], cache_file: Optional[Path], what: str) -> dict:
    """``weights`` when given (and written to ``cache_file`` when there is one), else read from ``cache_file``."""
    if weights:
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".pt.tmp")
            torch.save(_fp32(weights), tmp)
            tmp.replace(cache_file)
        return _fp32(weights)
    if cache_file is None or not cache_file.is_file():
        raise FileNotFoundError(
            f"{what}: no iHC weights passed and no cache file at {cache_file}; rebuild the 'ihc' weight category "
            f"from the checkpoint"
        )
    return torch.load(cache_file, map_location="cpu", weights_only=True)


def _fp32(d):
    if isinstance(d, dict):
        return {k: _fp32(v) for k, v in d.items()}
    return d.float().contiguous()


# ----------------------------------------------------------------------------------------------- host readback
def streams_to_host(t: ttnn.Tensor, mesh_device, *, sp_axis: int, tp_axis: int, n: int) -> torch.Tensor:
    """Packed device streams ``[1, 1, S / sp, n * C / tp]`` -> ``[S, n, C]`` fp32 in the reference order."""
    dims = [None, None]
    dims[sp_axis], dims[tp_axis] = 2, 3
    host = ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape))
    ).float()
    S = host.shape[-2]
    tp = mesh_device.shape[tp_axis]
    # [S, tp * n * C/tp], chip-major then stream -> [S, n, C]
    return host.reshape(S, tp, n, -1).permute(0, 2, 1, 3).reshape(S, n, -1)
