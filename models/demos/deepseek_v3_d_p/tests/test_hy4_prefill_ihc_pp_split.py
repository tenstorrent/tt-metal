# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-preview prefill with iHC split into two pipeline stages vs one stage, layers 0..4 of the 4K golden.

The cross-stage residual under iHC is the 4 packed fp32 streams, per chip ``[1, 1, S/sp, 4 * hidden/tp]``, not one
bf16 hidden: the first stage expands the embedding and returns streams (no iHC head), the next stage takes them as is
(no second expand) and runs the head only if it has a tail. This drives ``TtPrefillTransformer`` as two pipeline ranks
in one process on the same mesh (stage 1 = layers 0..1, stage 2 = layers 2..4, the boundary carrying layer 1's top-k
for the shared layers 2..4 as the runner does), and compares against the single-stage run:

  * the handoff tensor: shape / dtype are what the runner's D2D activation and the runtime's placeholder are sized for
    (``d2d_activation_width(hidden, streams=4) / tp``, fp32), and it equals the single stage's streams after layer 1;
  * the streams at every iHC site and the KV / index-K of every layer, two-stage vs single-stage (and both vs golden).

The D2D socket transport itself (two meshes, fp32 payload) is covered by ``test_d2d_socket_sync.py``'s iHC case.
Layout: the dev ``glm_prefill_sp8_tp4`` (8x4 mesh, SP 8 on axis 0, TP 4 on axis 1).
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import models.demos.deepseek_v3_d_p.tests.test_hy4_prefill_device_smoke as smoke
import ttnn
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.hy4 import ihc as hy4_ihc

chunked = smoke.chunked
LAYERS = 5
BOUNDARY = 2  # stage 1 = layers [0, 2), stage 2 = [2, 5)
SEQ_CACHE = 2 * smoke.CHUNK_SIZE

pytestmark = smoke.pytestmark


def _pcc(a, b) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


class _Streams:
    """Host copies of the streams leaving every iHC site (patched ``TtPrefillBlock._ihc_post``), in run order."""

    def __init__(self, monkeypatch, mesh_device, sp_axis):
        from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock

        self.out: list = []
        orig = TtPrefillBlock._ihc_post

        def post(wrap, sublayer_out, streams, p, free_streams):
            new = orig(wrap, sublayer_out, streams, p, free_streams)
            ttnn.synchronize_device(mesh_device)
            self.out.append(hy4_ihc.streams_to_host(new, mesh_device, sp_axis=sp_axis, tp_axis=1 - sp_axis, n=4))
            return new

        monkeypatch.setattr(TtPrefillBlock, "_ihc_post", staticmethod(post))


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, sp_axis",
    [
        pytest.param(
            smoke.MESH,
            torus_xy_device_params(
                fabric_payload_size=chunked.moe_fabric_payload(GLM53Config),
                l1_small_size=chunked.GLM_L1_SMALL_SIZE,
                trace_region_size=chunked.GLM_TRACE_REGION_SIZE,
            ),
            2,
            0,
            id="dev-sp8tp4-torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["hy4_preview"], indirect=True, ids=["hy4"])
@pytest.mark.timeout(0)
def test_hy4_prefill_ihc_pp_split(variant, config_only, mesh_device, device_params, num_links, sp_axis, monkeypatch):
    from models.demos.common.prefill.runners.runner_utils import d2d_activation_width
    from models.demos.deepseek_v3_d_p.tt.hy4 import layout as hy4_layout
    from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank
    from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache, init_mla_kv_cache

    monkeypatch.setenv(hy4_layout.LAYOUT_ENV, hy4_layout.DEV_LAYOUT)
    sp, tp = smoke.MESH[sp_axis], smoke.MESH[1 - sp_axis]
    tp_axis = 1 - sp_axis
    S = smoke.CHUNK_SIZE
    model_cfg = variant.model_config
    fmt = chunked.GLM_KV_CACHE_FORMAT
    cache = smoke._mesh_cache_dir(mesh_device.get_num_devices())
    config = copy.deepcopy(config_only)
    config.max_seq_len = SEQ_CACHE
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and rope_scaling.get("factor", 1.0) == 1.0:
        rope_scaling["original_max_position_embeddings"] = SEQ_CACHE
    topology = per_axis_topology(device_params["fabric_config"])
    ids = torch.tensor(json.loads((smoke._trace_dir() / "metadata.json").read_text())["token_ids"][:S])

    def build(first, n, is_first, is_last):
        return TtPrefillTransformer(
            mesh_device=mesh_device,
            config=config,
            model_cfg=model_cfg,
            state_dict={},  # cache-only: test_hy4_prefill_ihc_layers.py builds the missing iHC / sink pieces
            num_layers=n,
            seq_len=S,
            max_seq_len=SEQ_CACHE,
            dispatch_buffer_capacity_factor=8,
            num_links=num_links,
            topology=topology,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            gate_fallback_mode=GateComputeMode.DEVICE_FP32,
            weight_cache_path=cache,
            is_chunked=True,
            slot_num=1,
            kv_only_last_layer=is_last,
            routing_use_l1_small_for_semaphores=True,
            tp_shard_kv=True,
            sparse_kv_cache_format=fmt,
            first_layer_idx=first,
            is_first_rank=is_first,
            is_last_rank=is_last,
        )

    def caches(first, n):
        kv = init_mla_kv_cache(
            cache_format=fmt,
            hf_config=config,
            mesh_device=mesh_device,
            seq_len=SEQ_CACHE,
            mesh_shape=list(mesh_device.shape),
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            num_kvpe_cache_layers=n,
            num_users=1,
        )
        n_idx = full_indexer_rank(config, first + n) - full_indexer_rank(config, first)
        idx = (
            init_kvpe_cache(
                kvpe_cache_head_dim=config.index_head_dim,
                mesh_device=mesh_device,
                seq_len=SEQ_CACHE,
                mesh_shape=list(mesh_device.shape),
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                num_kvpe_cache_layers=n_idx,
                num_users=1,
                dtype=ttnn.bfloat8_b,
            )
            if n_idx
            else None
        )
        return kv, idx

    def tokens():
        return ttnn.from_torch(
            ids.reshape(sp, 1, S // sp),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, None)),
        )

    def fwd(model, x, kv, idx, **kw):
        return model.forward(
            x, kv, actual_isl=S, actual_start=0, actual_end=S, cache_user_id=0, index_kv_cache=idx, **kw
        )

    def kv_host(kv, slot):
        cache_now, _ = chunked.gather_cache_natural(kv, mesh_device, True, sp_axis=sp_axis)
        return cache_now[slot].float()

    rec = _Streams(monkeypatch, mesh_device, sp_axis)

    # All three models are built before anything runs and released only at the end: sparse MLA's overlap semaphores
    # live on the mesh's shared TT_CCL, and releasing one model then building the next would allocate them again in
    # the prefill's 1536 B of L1_SMALL.
    one = build(0, LAYERS, True, True)
    s1 = build(0, BOUNDARY, True, False)
    s2 = build(BOUNDARY, LAYERS - BOUNDARY, False, True)

    # --- single stage ---
    kv1, idx1 = caches(0, LAYERS)
    fwd(one, tokens(), kv1, idx1)
    ttnn.synchronize_device(mesh_device)
    single = list(rec.out)
    single_kv = [kv_host(kv1, i) for i in range(LAYERS)]
    del kv1, idx1
    rec.out.clear()

    # --- two stages ---
    assert s1.forwards_indexer_indices and s2.receives_indexer_indices
    kva, idxa = caches(0, BOUNDARY)
    kvb, idxb = caches(BOUNDARY, LAYERS - BOUNDARY)
    h = fwd(s1, tokens(), kva, idxa)
    seed = s1.last_indexer_indices
    streams, dtype = hy4_ihc.residual_layout(model_cfg)
    want = (1, 1, S // sp, d2d_activation_width(config.hidden_size, streams=streams) // tp)
    assert (tuple(h.shape), h.dtype) == (want, dtype), f"handoff {tuple(h.shape)} {h.dtype}, sized for {want} {dtype}"
    ttnn.synchronize_device(mesh_device)
    handoff = hy4_ihc.streams_to_host(h, mesh_device, sp_axis=sp_axis, tp_axis=tp_axis, n=streams)
    fwd(s2, h, kvb, idxb, seed_indexer_indices=seed)
    ttnn.synchronize_device(mesh_device)
    split = list(rec.out)
    split_kv = [kv_host(kva, i) for i in range(BOUNDARY)] + [kv_host(kvb, i) for i in range(LAYERS - BOUNDARY)]
    for m in (one, s1, s2):
        m.release_sub_device_managers()

    names = [f"L{i // 2}.{('attn', 'ffn')[i % 2]}" for i in range(len(single))]
    assert len(split) == len(single) == 2 * (LAYERS - 1), (len(split), len(single))
    rows = {
        "handoff (streams after L1.ffn)": (
            _pcc(single[2 * BOUNDARY - 1], handoff),
            (single[2 * BOUNDARY - 1] - handoff).abs().max().item(),
        )
    }
    for name, a, b in zip(names, single, split):
        rows[f"streams {name}"] = (_pcc(a, b), (a - b).abs().max().item())
    for i, (a, b) in enumerate(zip(single_kv, split_kv)):
        rows[f"KV slot {i}"] = (_pcc(a, b), (a - b).abs().max().item())
    logger.info(
        f"Hy4 + iHC, 2 pipeline stages (layers [0,{BOUNDARY}) | [{BOUNDARY},{LAYERS})) vs 1 stage, layout glm_prefill_sp8_tp4:"
    )
    for k, (p, d) in rows.items():
        logger.info(f"  {k:32s} pcc={p:.8f} max|d|={d:.3e}")
    if os.environ.get("HY4_IHC_REPORT"):
        Path(os.environ["HY4_IHC_REPORT"]).write_text(json.dumps(rows, indent=1))
    bad = [f"{k} {p:.6f}" for k, (p, _) in rows.items() if p < 0.99999]
    assert not bad, f"two-stage run differs from one stage: {bad}"
