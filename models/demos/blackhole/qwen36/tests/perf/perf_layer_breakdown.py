# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer-type prefill/decode cost at a given TP, for the GDN-vs-FA breakdown table.

Times a model built from N layers of ONE type and reports the SLOPE between two layer
counts, so any model-level fixed cost (embedding, final norm, LM head, trace setup)
cancels and what is left is the marginal cost of one GDN layer / one full-attention
layer.

  QWEN_BD_TYPE=gdn|fa   which layer type to measure
  QWEN_BD_ISL=4096      prompt length
  MESH_DEVICE=P150|P150x4|P150x8
"""

import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import model_path
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE

GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]
FA_LAYERS = [3, 7, 11, 15, 19, 23]
BLOCK = 64


def _mesh_shape():
    n = {"P150": 1, "P150x4": 4, "P150x8": 8}.get(os.environ.get("MESH_DEVICE"), 8)
    return n


def _open(n_dies):
    if n_dies > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    system = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
    size = system.mesh_size()
    if size > n_dies:
        shape = ttnn.MeshShape(size // n_dies, n_dies) if size % n_dies == 0 else system
        parent = ttnn.open_mesh_device(
            mesh_shape=shape, trace_region_size=64 * 1024 * 1024, l1_small_size=GDN_CONV1D_L1_SMALL_SIZE
        )
        return parent, parent.create_submeshes(ttnn.MeshShape(1, n_dies))[0]
    m = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, n_dies), trace_region_size=64 * 1024 * 1024, l1_small_size=GDN_CONV1D_L1_SMALL_SIZE
    )
    return m, m


def _close(owner):
    for sub in owner.get_submeshes():
        for leaf in sub.get_submeshes():
            ttnn.close_mesh_device(leaf)
        ttnn.close_mesh_device(sub)
    ttnn.close_mesh_device(owner)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _time_prefill_decode(mesh, idxs, T, decode_steps=8):
    model = Qwen36Model.from_pretrained(
        mesh, max_batch_size=1, max_seq_len=T + 512, layer_indices=idxs, hf_model=model_path()
    )
    nb = (T // BLOCK) + 8
    page_table = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
    model.allocate_kv_caches((nb, model.args.n_local_kv_heads, BLOCK, model.args.head_dim), ttnn.bfloat16, batch_size=1)
    tokens = torch.randint(0, model.args.vocab_size, (1, T), dtype=torch.long)

    model.reset_tp()
    out = model.prefill_traced_chunked(tokens, page_table, actual_len=T)  # warmup
    ttnn.deallocate(out)
    ttnn.synchronize_device(mesh)

    # min of N replays: a single wall-clock sample is swamped by host noise -- the first
    # attempt produced a NEGATIVE slope (3 FA layers 38.53 ms vs 6 layers 30.36 ms).
    best = float("inf")
    for _ in range(5):
        model.reset_tp()
        t0 = time.time()
        out = model.prefill_traced_chunked(tokens, page_table, actual_len=T)
        ttnn.synchronize_device(mesh)
        best = min(best, (time.time() - t0) * 1e3)
        ttnn.deallocate(out)
    return best, model


def test_layer_breakdown():
    kind = os.environ.get("QWEN_BD_TYPE", "gdn")
    T = int(os.environ.get("QWEN_BD_ISL", "4096"))
    # A GDN-only model allocates no paged KV cache, so prefill_traced_chunked's
    # get_block_size(kv_cache) -> kv_cache[0][0] raises IndexError. Keep ONE full-attention
    # layer resident in both builds; it is constant so it cancels in the slope.
    if kind == "gdn":
        pool, n_lo, n_hi, keep = GDN_LAYERS, 4, 8, [FA_LAYERS[0]]
    else:
        pool, n_lo, n_hi, keep = FA_LAYERS, 2, 5, []
    n_dies = _mesh_shape()

    res = {}
    for n in (n_lo, n_hi):
        owner, mesh = _open(n_dies)
        try:
            pf, _ = _time_prefill_decode(mesh, sorted(pool[:n] + keep), T)
            res[n] = pf
            logger.info(f"[breakdown] {kind} TP={n_dies} layers={n}: prefill {pf:.2f} ms")
        finally:
            _close(owner)

    slope = (res[n_hi] - res[n_lo]) / (n_hi - n_lo)
    logger.info(
        f"[breakdown] RESULT {kind} TP={n_dies} ISL={T}: {slope:.3f} ms per {kind.upper()} layer "
        f"(from {n_lo}->{n_hi} layers: {res[n_lo]:.2f} -> {res[n_hi]:.2f} ms)"
    )
