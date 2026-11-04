# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase 0 of the persistent-stack port: the reference's decode layer megakernels (models/autoports/
zai_org_glm_5_3_flash: TPDenseKDALayer / TPSparseKDALayer, program "layer" = Stage 7a's MkLayer, the whole layer in ONE
program per chip) built from OUR HF weights and timed on this tt-metal.

    GLM53_MESH_SHAPE=1,1 GLM53_MK_LAYER=0 GLM53_MK_VTP=8 pytest .../test_mk_layer_probe.py -s   # dense KDA layer
    GLM53_MESH_SHAPE=1,1 GLM53_MK_LAYER=4 GLM53_MK_VTP=8 ...                                   # sparse KDA layer

One chip with ``vtp=8`` runs exactly a TP8 chip's program (TP8-sized shards, no fabric phase: the reference's
"nofab" path); a (2,4) / (1,8) mesh runs the TP program with the in-program fabric all-reduce. Prints the traced
per-call time, the eager wall time and the reference's per-phase probe (``decode_probe``).
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt.config import DENSE_KDA, SPARSE_KDA, load_hf_text_config
from models.autoports.zai_org_glm_5_3_flash.tt.mk_layer import decode_probe
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import OptimizedDecoderConfig
from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import layer_kind
from models.autoports.zai_org_glm_5_3_flash.tt.tp8_layer import TPDenseKDALayer
from models.autoports.zai_org_glm_5_3_flash.tt.weights import WeightSource, cache_dtype
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tt.mesh import mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "1,1").split(","))
LAYER = int(os.environ.get("GLM53_MK_LAYER", "0"))  # 0-2 dense KDA, 3 + 4k DSA, the rest sparse KDA
VTP = int(os.environ.get("GLM53_MK_VTP", "8"))  # TP the shards are sized for (timing: a TP8 chip's program on 1 chip)
B = int(os.environ.get("GLM53_MK_B", "1"))  # user slots
T = int(os.environ.get("GLM53_MK_T", "1"))  # live rows per launch (<= 4)
N_TRACE = int(os.environ.get("GLM53_MK_N", "8"))
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
DRAM = ttnn.DRAM_MEMORY_CONFIG


class RawHFSource(WeightSource):
    """``host(layer, name)`` / ``device(layer, name, mesh)`` over our raw HF layer dicts (``load_layer_raw``), in the
    reference's tile-cache layout: linear weights [in, out], the hyper-connection ``fn`` [16384, 24], conv taps [4, C]
    per q / k / v, per-expert ``mlp.experts.{e}.gate_up_proj`` [H, 2I] = [gate | up] and ``down_proj`` [I, H]."""

    def __init__(self, snapshot, layers):
        self.snapshot = snapshot
        self.headers = shard_headers(snapshot)
        self.raw = {i: load_layer_raw(snapshot, i, self.headers) for i in layers}
        self.cache: dict[tuple[int, str], torch.Tensor] = {}

    def host(self, layer_idx: int, name: str) -> torch.Tensor:
        key = (layer_idx, name)
        if key in self.cache:
            return self.cache[key]
        raw = self.raw[layer_idx]
        if name.startswith("mlp.experts.") and name.endswith(("gate_up_proj", "down_proj")):
            e = name.split(".")[2]
            if name.endswith("gate_up_proj"):
                g = raw[f"mlp.experts.{e}.gate_proj.weight"].t()  # HF [I, H] -> [H, I]
                u = raw[f"mlp.experts.{e}.up_proj.weight"].t()
                v = torch.cat([g, u], dim=-1)  # [H, 2I]
            else:
                v = raw[f"mlp.experts.{e}.down_proj.weight"].t()  # HF [H, I] -> [I, H]
        else:
            if name not in raw:
                raise KeyError(f"layer {layer_idx}: {name} not in the raw HF dict ({sorted(raw)[:8]}...)")
            v = raw[name]
            if name in ("hc_attn_fn", "hc_ffn_fn"):
                v = v.t()
            elif "conv1d" in name:
                v = v.reshape(v.shape[0], -1).t()  # [C, 1, 4] -> [4, C]
            elif v.dim() == 2 and name.endswith(".weight"):
                v = v.t()  # [out, in] -> [in, out]
        self.cache[key] = v.contiguous()
        return self.cache[key]

    def device(self, layer_idx: int, name: str, mesh_device) -> ttnn.Tensor:
        t = self.host(layer_idx, name)
        dtype = cache_dtype(name)
        return ttnn.from_torch(
            t.to(torch.float32 if dtype == ttnn.float32 else torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )


def _build_layer(mesh_device, cfg, ocfg, source, kind: str):
    if kind == DENSE_KDA:
        return TPDenseKDALayer(mesh_device, cfg, LAYER, source, ocfg, b=B, probe=True, vtp=VTP)
    if kind == SPARSE_KDA:
        from models.autoports.zai_org_glm_5_3_flash.tt.tp8_sparse import TPSparseKDALayer, build_moe

        t0 = time.perf_counter()
        moe = build_moe(mesh_device, cfg, source, LAYER, ocfg)
        logger.info(
            f"MoE (router, shared expert, {cfg.n_routed_experts} routed experts) built in {time.perf_counter() - t0:.0f} s"
        )
        return TPSparseKDALayer(mesh_device, cfg, LAYER, source, ocfg, b=B, moe=moe, probe=True, vtp=VTP)
    pytest.skip(f"layer {LAYER} is {kind}: the DSA layer program is phase 0 step 4")


# the stack's kernel binaries need a kernel-config ring above the default 69 KB: the ring is the L1 below the
# allocatable region, so the device opens with the reference's smaller worker_l1_size (1,264,512)
WORKER_L1 = int(os.environ.get("GLM53_MK_WORKER_L1", "1264512"))


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params",
    [mesh_device_params(*MESH, trace_region_size=TRACE_MB << 20, worker_l1_size=WORKER_L1)],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_mk_layer_probe(mesh_device):
    snapshot = default_snapshot_dir()
    cfg = load_hf_text_config(str(snapshot))
    ocfg = OptimizedDecoderConfig()
    kind = layer_kind(cfg, LAYER)
    source = RawHFSource(snapshot, [LAYER])
    t0 = time.perf_counter()
    layer = _build_layer(mesh_device, cfg, ocfg, source, kind)
    logger.info(f"{kind} layer {LAYER} built in {time.perf_counter() - t0:.1f} s: mesh {MESH}, vtp {VTP}, b {B}")
    state = layer.attn.allocate_state(B)
    active = layer.masks(torch.ones(B))
    torch.manual_seed(0)
    x = layer.replicated((torch.randn(1, 1, T, 4 * cfg.hidden_size) * 0.1).float(), ttnn.float32)
    # warm (compiles every kernel)
    t0 = time.perf_counter()
    s1, out, details, _ = layer(x, state.C, active, state.S, t=T)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"first call (compile + run) {time.perf_counter() - t0:.1f} s; out {tuple(out.shape)} {out.dtype}")
    # eager wall per call
    t0 = time.perf_counter()
    for _ in range(3):
        layer(x, state.C, active, state.S, t=T)
    ttnn.synchronize_device(mesh_device)
    eager_us = (time.perf_counter() - t0) / 3 * 1e6
    # the probe of the last eager call
    marks = decode_probe(ttnn.to_torch(layer.mk.probe), layer.mk.readers.assign) if layer.mk.probe is not None else {}
    # traced: N calls in one trace, replayed
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(N_TRACE):
        layer(x, state.C, active, state.S, t=T)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    reps = 5
    for _ in range(reps):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    traced_us = (time.perf_counter() - t0) / (reps * N_TRACE) * 1e6
    ttnn.release_trace(mesh_device, tid)
    logger.info(
        f"MK_LAYER_PROBE: mesh {MESH} {kind} layer {LAYER} vtp {VTP} t {T}: traced {traced_us:.1f} us per layer call, eager {eager_us:.1f} us"
    )
    logger.info(f"MK_LAYER_PROBE marks (us from the earliest, medians per role): {marks}")
    assert traced_us > 0
