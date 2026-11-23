# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview attention sublayer on the SC1 2D tensor-parallel layout vs the fp32 HF ``hy_v4`` reference (#65885).

``ttMLA2DTP`` (``tt/mla/mla_2d.py``) with REAL Hy4 weights, every projection sharded once over the 32 chips. Input and
output are the prefill block's residual layout ``[S / 8, 6144 / 4]`` per chip (sequence over SP, hidden over TP), the
same as the 2D shared expert / dense FFN of #65890 and iHC (#65891).

Mesh: the physical 8x4 torus (``single_bh_galaxy_torus_xy``, dims [8, 4]) with SP 8 on axis 0 (sequence and heads, 8
per chip) and TP 4 on axis 1 (hidden): #65884's 4x8 placement (hidden over its 4 rows, heads over its 8 columns) with
its axes named by the fabric; no logical (4, 8) view is opened.

Layers 0 (full indexer) and 2 (shared), both fed the top-k: the golden layer-0 top-k at 4K, a sliding causal window at
longer lengths (seeded random tokens, no golden). ttMLA2DTP does not build the DSA indexer yet (#65887). Input: the layer-0 sublayer input ``hn = input_layernorm(embed)``, as in
``test_hy4_attention.py`` (exact at layer 0, no iHC needed). Device weights come from the production Hy4 reader (RoPE
frames reordered), the reference reads the raw HF tensors.

Env: ``HY4_HF_MODEL``, ``HY4_GOLDEN_DIR``, ``HY4_ATTN_SEQ_LENS`` (default 4096), ``HY4_ATTN_CHUNK`` (default 4096).
"""

from __future__ import annotations

import copy
import time

import pytest
import torch
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.demos.deepseek_v3_d_p.reference.hy4_attention import Hy4AttnDims
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.test_hy4_attention import (
    ATTN_PCC,
    CHECKPOINT,
    CHUNK,
    FULL_LAYER,
    L1_SMALL,
    SEQ_LENS,
    SHARED_LAYER,
    _check_layer,
    golden_tensor,
    hf_attention_weights,
    host_indices_to_device,
    layer0_hidden,
    sliding_topk,
    ttmla_weights,
)
from models.demos.deepseek_v3_d_p.tt.mla.mla_2d import MLA_2D_TP_FLAG, mla_class, ttMLA2DTP
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_mla_kv_cache

pytestmark = [
    pytest.mark.skipif(not is_blackhole(), reason="DSA ops (sparse SDPA) are Blackhole-only"),
    pytest.mark.skipif(not (CHECKPOINT / "model.safetensors.index.json").is_file(), reason="no Hy4 checkpoint"),
]

SP_AXIS = 0  # 8 chips: the residual's sequence, the 8-head blocks of q_b / the gate / o_proj
TP_AXIS = 1  # 4 chips: the residual's hidden, q_b's K


def run_2d_attention(config, weights, mesh_device, hn, layer_idx, topology, topk):
    """Chunked prefill of ``hn`` through one ttMLA2DTP layer fed ``topk`` ``[S, K]``. Returns ``[S, hidden]`` fp32."""
    seq_len = hn.shape[0]
    chunk = min(seq_len, CHUNK)
    assert seq_len % chunk == 0
    mesh_shape = list(mesh_device.shape)
    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.SCALED_FP8,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=1,
        tp_axis=TP_AXIS,
    )
    cls = mla_class(config)
    assert cls is ttMLA2DTP
    mla = cls(
        config,
        weights,
        mesh_device,
        layer_idx=layer_idx,
        seq_len=seq_len,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        is_balanced=False,
        topology=topology,
        is_chunked=True,
        active_seq_len=chunk,
        layer_num=1,
        sparse_kv_cache_format=kvpe_cache.format,
    )
    assert mla.attn_sink is not None and mla._use_gate and mla.heads_per_chip == 2
    rope = RotarySetup(config, mesh_device, sp_axis=SP_AXIS, is_balanced=False)
    rope_tensors = rope.get_rope_tensors_indexed(cache_seq_len_global=seq_len, chunk_size_global=chunk)
    # The prefill block's residual layout (also TtMlp2D's, #65890): sequence over SP, hidden over TP.
    dims = [None, None]
    dims[SP_AXIS], dims[TP_AXIS] = 2, 3
    t0 = time.time()
    outs = []
    for start in range(0, seq_len, chunk):
        tt_hidden = ttnn.from_torch(
            hn[start : start + chunk].view(1, 1, chunk, -1),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
        )
        out = mla.forward(
            hidden_states=tt_hidden,
            rope_tensors=rope_tensors,
            kvpe_cache=kvpe_cache,
            actual_start=start,
            indexer_indices=host_indices_to_device(topk[start : start + chunk], mesh_device, SP_AXIS),
        )
        sp, tp = mesh_device.shape[SP_AXIS], mesh_device.shape[TP_AXIS]
        assert tuple(out.shape) == (1, 1, chunk // sp, config.hidden_size // tp)
        outs.append(
            ttnn.to_torch(
                out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=mesh_device.shape)
            ).float()[0, 0]
        )
        ttnn.deallocate(out)
        ttnn.deallocate(tt_hidden)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"layer {layer_idx}: 2D TP attention, {len(outs)} chunk(s), {time.time() - t0:.1f}s (includes compile)")
    del mla, kvpe_cache
    return torch.cat(outs)


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((8, 4), torus_xy_device_params(l1_small_size=L1_SMALL), id="tp32_2d")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=[f"s{s}" for s in SEQ_LENS])
@pytest.mark.timeout(0)
def test_hy4_attention_2d_tp_vs_hf(mesh_device, device_params, seq_len):
    from models.demos.deepseek_v3_d_p.reference.hy4_config import hy4_hf_config
    from models.demos.deepseek_v3_d_p.tt.runners.hy4_checkpoint import Hy4Checkpoint

    config = hy4_hf_config(max_seq=seq_len)
    config.max_seq_len = seq_len
    setattr(config, MLA_2D_TP_FLAG, True)
    # No TtIndexer on this layout yet (#65887): every layer is built as a shared layer and fed the top-k.
    config.indexer_types = ["shared"] * len(config.indexer_types)
    dims = Hy4AttnDims.from_config(config)
    topology = per_axis_topology(device_params["fabric_config"])
    ckpt = Hy4Checkpoint(CHECKPOINT)
    hn, tokens = layer0_hidden(ckpt, seq_len, config.rms_norm_eps)
    topk = None
    if tokens is not None:
        topk = golden_tensor(
            f"dsa/dsa_topk_indices_layer_{FULL_LAYER}", f"dsa_topk_indices_layer_{FULL_LAYER}", seq_len
        )
    topk = topk.long() if topk is not None else sliding_topk(seq_len, config.index_topk)
    logger.info(f"S={seq_len}: top-k from {'the golden layer 0' if tokens is not None else 'a sliding causal window'}")

    results = {}
    for kind, layer in (("full", FULL_LAYER), ("shared", SHARED_LAYER)):
        out = run_2d_attention(
            config,
            ttmla_weights(ckpt, layer, copy.copy(config), False),
            mesh_device,
            hn,
            layer,
            topology,
            topk,
        )
        w = hf_attention_weights(ckpt, layer, owns_indexer=False)
        results[kind], _ = _check_layer(f"layer {layer} ({kind}), 2D TP", out, hn, w, topk, dims)
        del w, out

    logger.info(f"Hy4 2D TP attention sublayer PCC (mesh {tuple(mesh_device.shape)}, S={seq_len}): {results}")
    for kind, p in results.items():
        assert p >= ATTN_PCC, f"{kind} layer 2D TP attention PCC {p:.6f} < {ATTN_PCC}"
