# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview DSA indexer + attention on the SC1 2D tensor-parallel layout vs the HF ``hy_v4`` golden (#65887).

``ttMLA2DTP`` (``tt/mla/mla_2d.py``) on the full-indexer layers 0, 1 and 5 with REAL Hy4 weights, building its own
``TtIndexer2D`` (``tt/mla/indexer_2d.py``: every indexer weight sharded once over the 32 chips) and attending over the
top-k THAT indexer picks. Mesh: the physical 8x4 torus, SP 8 on axis 0 (sequence, heads), TP 4 on axis 1 (hidden);
input and output the residual ``[S / 8, 6144 / 4]`` per chip. One 4096-token chunk of the 4K golden prompt.

Scored per layer:
  * the index-key cache (read back, de-Hadamarded, block-cyclic order undone) against the golden ``indexer_k_layer_L``
    (device frame: rope block first), rope and nope halves;
  * the device top-k against the golden ``dsa_topk_indices_layer_L``: the mean fraction of each row's golden set the
    device also picked, rows >= index_topk (below that both select every causal key);
  * the attention output against the fp32 HF reference (``reference/hy4_attention.py``) attending over the DEVICE
    top-k, so the PCC isolates the attention math from top-k ties.

Layer inputs: the sublayer input ``hn = input_layernorm(hc_pre(streams))`` from the fp32 reference streams
(``HY4_IHC_REF_STREAMS``: embedding for layer 0, ``streams_L{i-1}_ffn`` before layer i), so each layer is checked on
the reference's own input (no upstream device error). ``dev_1d`` runs the same through ttMLA + TtIndexer on the dev
layout (``glm_prefill_sp8_tp4``: wq_b replicated) for an apples-to-apples baseline.

Env: ``HY4_HF_MODEL``, ``HY4_GOLDEN_DIR``, ``HY4_IHC_REF_STREAMS``, ``HY4_IDX_LAYERS`` (default 0,1,5),
``HY4_IDX_REPORT`` (optional JSON path).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.demos.deepseek_v3_d_p.reference.hy4_attention import Hy4AttnDims, rms_norm
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.test_hy4_attention import (
    ATTN_PCC,
    CHECKPOINT,
    GOLDEN,
    L1_SMALL,
    _check_layer,
    _topk_agreement,
    device_indices_to_host,
    golden_tensor,
    hf_attention_weights,
    ttmla_weights,
)
from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank, normalized_hadamard_matrix
from models.demos.deepseek_v3_d_p.tt.mla.indexer_2d import TtIndexer2D
from models.demos.deepseek_v3_d_p.tt.mla.mla_2d import MLA_2D_TP_FLAG, mla_class, ttMLA2DTP
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import cache_half_pccs, gather_cache_natural, unrotate_cache_layer

REF_DIR = Path(os.environ.get("HY4_IHC_REF_STREAMS", "/mnt/tt-data/dwadhwani/hy4-golden/hy4-4k-ropeint-streams"))
LAYERS = [int(x) for x in os.environ.get("HY4_IDX_LAYERS", "0,1,5").split(",")]
S = 4096
SP_AXIS, TP_AXIS = 0, 1
INDEX_K_PCC = 0.99
TOPK_AGREEMENT = 0.95

pytestmark = [
    pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only"),
    pytest.mark.skipif(not (CHECKPOINT / "model.safetensors.index.json").is_file(), reason="no Hy4 checkpoint"),
    pytest.mark.skipif(not (GOLDEN / "metadata.json").is_file(), reason=f"no Hy4 golden at {GOLDEN}"),
    pytest.mark.skipif(not (REF_DIR / "streams_embed.pt").is_file(), reason=f"no reference streams at {REF_DIR}"),
]


def layer_input(ckpt, layer: int, eps: float) -> torch.Tensor:
    """``input_layernorm(hc_pre(streams))`` of ``layer`` from the reference streams entering it, bf16-rounded."""
    from models.demos.deepseek_v3_d_p.tt.hy4 import ihc as hy4_ihc
    from models.experimental.hy4_preview.reference.ihc import IHCConfig, IHCWrap

    streams = torch.load(
        REF_DIR / ("streams_embed.pt" if layer == 0 else f"streams_L{layer - 1}_ffn.pt"),
        map_location="cpu",
        weights_only=True,
    )[:S]
    w = hy4_ihc.read_layer_ihc(ckpt, layer)["attn"]
    ref = IHCWrap(IHCConfig()).eval()
    ref.fn.data, ref.base.data, ref.scale.data = w["fn"], w["base"], w["scale"]
    with torch.no_grad():
        y, _ = ref.hc_pre(streams.unsqueeze(0))
    hn = rms_norm(y[0].float(), ckpt.get(f"model.layers.{layer}.input_layernorm.weight").float(), eps)
    return hn.to(torch.bfloat16)


def run_layer(config, weights, mesh_device, hn, layer, topology):
    """One chunk of ``hn`` through the layer's MLA with its own indexer. Returns (output [S, hidden] fp32, the device
    top-k [S, K], the index-key cache row block [S, D] in natural order, de-Hadamarded, device frame)."""
    mesh_shape = list(mesh_device.shape)
    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.SCALED_FP8,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=S,
        mesh_shape=mesh_shape,
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=1,
        tp_axis=TP_AXIS,
    )
    index_kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.index_head_dim,
        mesh_device=mesh_device,
        seq_len=S,
        mesh_shape=mesh_shape,
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=sum(t == "full" for t in config.indexer_types),
        num_users=1,
        dtype=ttnn.bfloat8_b,
        tp_axis=TP_AXIS,
    )
    mla = mla_class(config)(
        config,
        weights,
        mesh_device,
        layer_idx=layer,
        seq_len=S,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        is_balanced=False,
        topology=topology,
        is_chunked=True,
        active_seq_len=S,
        layer_num=1,
        sparse_kv_cache_format=kvpe_cache.format,
    )
    if isinstance(mla, ttMLA2DTP):
        idx = mla._indexer
        assert isinstance(idx, TtIndexer2D)
        # One whole index head per chip; wk / weights_proj K over all 32 chips (nothing replicated per row).
        assert tuple(idx._idx_wq_b.shape) == (config.q_lora_rank, config.index_head_dim)
        assert tuple(idx._idx_wk.shape) == (config.hidden_size // 32, config.index_head_dim)
        assert tuple(idx._idx_wproj.shape) == (config.hidden_size // 32, config.index_n_heads)
    rope = RotarySetup(config, mesh_device, sp_axis=SP_AXIS, is_balanced=False)
    rope_tensors = rope.get_rope_tensors_indexed(cache_seq_len_global=S, chunk_size_global=S)
    dims = [None, None]
    dims[SP_AXIS], dims[TP_AXIS] = 2, 3
    tt_hidden = ttnn.from_torch(
        hn.view(1, 1, S, -1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
    )
    t0 = time.time()
    out, idx_tt = mla.forward(
        hidden_states=tt_hidden,
        rope_tensors=rope_tensors,
        kvpe_cache=kvpe_cache,
        actual_start=0,
        return_indexer_indices=True,
        index_kv_cache=index_kv_cache,
    )
    sp, tp = mesh_device.shape[SP_AXIS], mesh_device.shape[TP_AXIS]
    assert tuple(out.shape) == (1, 1, S // sp, config.hidden_size // tp)
    out_host = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=mesh_device.shape)
    ).float()[0, 0]
    topk = device_indices_to_host(idx_tt, mesh_device, SP_AXIS, S)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"layer {layer}: {type(mla).__name__} + {type(mla._indexer).__name__}, {time.time() - t0:.1f}s")
    cache, stripes = gather_cache_natural(index_kv_cache, mesh_device, tp_shard_kv=True, sp_axis=SP_AXIS)
    slot = full_indexer_rank(config, layer)
    k = unrotate_cache_layer(cache[slot], blockcyclic_positions(stripes, S, S), S)
    k = k.float() @ normalized_hadamard_matrix(config.index_head_dim).float()
    ttnn.deallocate(out)
    ttnn.deallocate(tt_hidden)
    del mla, kvpe_cache, index_kv_cache
    return out_host, topk, k


@pytest.mark.parametrize(
    "mesh_device,device_params,layout",
    [
        pytest.param((8, 4), torus_xy_device_params(l1_small_size=L1_SMALL), "tp32_2d", id="tp32_2d"),
        pytest.param((8, 4), torus_xy_device_params(l1_small_size=L1_SMALL), "dev_1d", id="dev_1d"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(0)
def test_hy4_indexer_2d_vs_golden(mesh_device, device_params, layout):
    from models.demos.deepseek_v3_d_p.reference.hy4_config import hy4_hf_config
    from models.demos.deepseek_v3_d_p.tt.runners.hy4_checkpoint import Hy4Checkpoint

    config = hy4_hf_config(max_seq=S)
    config.max_seq_len = S
    if layout == "tp32_2d":
        setattr(config, MLA_2D_TP_FLAG, True)
    assert all(config.indexer_types[i] == "full" for i in LAYERS), f"{LAYERS} must be full-indexer layers"
    dims = Hy4AttnDims.from_config(config)
    topology = per_axis_topology(device_params["fabric_config"])
    ckpt = Hy4Checkpoint(CHECKPOINT)
    results = {}
    for layer in LAYERS:
        hn = layer_input(ckpt, layer, config.rms_norm_eps)
        out, topk, k = run_layer(config, ttmla_weights(ckpt, layer, config, True), mesh_device, hn, layer, topology)
        r = {}
        gold_k = golden_tensor(f"dsa/indexer_k_layer_{layer}", f"indexer_k_layer_{layer}", S)
        r["index_k_rope"], r["index_k_nope"] = cache_half_pccs(
            gold_k.float(), k.to(torch.bfloat16).float(), config.index_head_dim // 2, pe_interleave=False
        )
        gold_topk = golden_tensor(f"dsa/dsa_topk_indices_layer_{layer}", f"dsa_topk_indices_layer_{layer}", S)
        r["topk_agreement"] = _topk_agreement(topk, gold_topk.long(), config.index_topk)
        w = hf_attention_weights(ckpt, layer, owns_indexer=False)
        r["attn_pcc"], _ = _check_layer(f"layer {layer} ({layout})", out, hn, w, topk, dims)
        # The reference attention over the GOLDEN top-k, for scale: how far the top-k difference moves the output.
        r["attn_pcc_vs_ref_on_golden_topk"], _ = _check_layer(
            f"layer {layer} ({layout}) vs ref on golden top-k", out, hn, w, gold_topk.long(), dims
        )
        logger.info(f"[{layout}] layer {layer}: " + ", ".join(f"{k_}={v:.6f}" for k_, v in r.items()))
        results[layer] = r
        del w, out, hn

    logger.info(f"Hy4 indexer ({layout}, mesh {tuple(mesh_device.shape)}, S={S}):")
    logger.info("layer  index_k_rope  index_k_nope  topk_agree  attn_pcc  attn_vs_golden_topk")
    for layer, r in results.items():
        logger.info(
            f"{layer:5d}  {r['index_k_rope']:12.6f}  {r['index_k_nope']:12.6f}  {r['topk_agreement']:10.4f}  "
            f"{r['attn_pcc']:8.6f}  {r['attn_pcc_vs_ref_on_golden_topk']:8.6f}"
        )
    if os.environ.get("HY4_IDX_REPORT"):
        Path(os.environ["HY4_IDX_REPORT"]).write_text(json.dumps({"layout": layout, "layers": results}, indent=1))
    bad = []
    for layer, r in results.items():
        bad += [f"L{layer} {k_}={r[k_]:.4f}" for k_ in ("index_k_rope", "index_k_nope") if r[k_] < INDEX_K_PCC]
        if r["topk_agreement"] < TOPK_AGREEMENT:
            bad.append(f"L{layer} topk_agreement={r['topk_agreement']:.4f}")
        if r["attn_pcc"] < ATTN_PCC:
            bad.append(f"L{layer} attn_pcc={r['attn_pcc']:.6f}")
    assert not bad, f"below the floors (index-K {INDEX_K_PCC}, top-k {TOPK_AGREEMENT}, attention {ATTN_PCC}): {bad}"
