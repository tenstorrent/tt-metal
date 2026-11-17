# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview attention sublayer on device vs the fp32 HF ``hy_v4`` reference (#65921).

One ttMLA forward with REAL Hy4 weights (projections + DSA sparse MLA + learnable per-head sink + sigmoid
output gate + o_proj) against ``reference/hy4_attention.py``, on a full-indexer layer (the device indexer
picks the top-k) and a shared layer (the full layer's top-k injected). The reference attends over the
top-k the device picked, so the PCC isolates the attention math from top-k ties.

Input: the layer-0 sublayer input of the 4K golden prompt (#65919). At layer 0 the four iHC streams are
identical copies of the embedding, so the iHC pre-mix is a positive per-token multiple of the embedding and
``input_layernorm`` removes it: ``hn = input_layernorm(embed(tokens))`` (exact up to the norm's eps). That is
what the TT stack (no iHC yet, #65891) feeds layer 0, so this layer's attention is checked against HF's. Longer
prompts (32K) use seeded random tokens through the same embedding + norm.

Weights go to ttMLA in its RoPE frame: Hy4 ropes half-split, ttMLA interleaved, so the q/k RoPE columns (and the
indexer's) are permuted here; scores are unchanged. This is done in the test, not via the checkpoint reader,
so the check does not depend on where that permutation lands. Also logged (not asserted): the PCC against the
reference without the sink / without the gate, and the device top-k's agreement with the golden's.

Env: ``HY4_HF_MODEL`` (default /mnt/tt-data/Hy4-Preview), ``HY4_GOLDEN_DIR`` (default the 4K v2 golden),
``HY4_ATTN_SEQ_LENS`` (comma list, default 4096).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger
from ttnn.device import is_blackhole

import ttnn
from models.demos.deepseek_v3_d_p.reference.hy4_attention import Hy4AttnDims, hy4_attention, rms_norm
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.indexer import indexer_layer_is_reused, num_full_indexer_layers
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache

CHECKPOINT = Path(os.environ.get("HY4_HF_MODEL", "/mnt/tt-data/Hy4-Preview"))
GOLDEN = Path(os.environ.get("HY4_GOLDEN_DIR", "/mnt/tt-data/dwadhwani/hy4-golden/hy4-bf16-longbench-4k-v2"))
SEQ_LENS = [int(s) for s in os.environ.get("HY4_ATTN_SEQ_LENS", "4096").split(",")]
ATTN_PCC = 0.995  # GLM-5.3's MLA threshold (Hy4Adapter.mla_pcc_threshold)
FULL_LAYER, SHARED_LAYER = 0, 2  # Hy4 indexer_types: 0, 1 full; 2, 3, 4 shared

pytestmark = [
    pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only"),
    pytest.mark.skipif(not (CHECKPOINT / "model.safetensors.index.json").is_file(), reason="no Hy4 checkpoint"),
]


# ---------------------------------------------------------------------------------------------- frames
def half_split_to_interleaved_perm(d: int) -> torch.Tensor:
    """``x[..., perm]``: half-split RoPE pairs ``(i, i + d/2)`` -> interleaved ``(2i, 2i + 1)``, same frequencies."""
    h = d // 2
    return torch.stack([torch.arange(h), torch.arange(h) + h], dim=-1).reshape(d)


def index_k_device_perm(index_head_dim: int, rope_dim: int) -> torch.Tensor:
    """Hy4 index key ``[nope | rope(half-split)]`` -> TtIndexer's ``[rope(interleaved) | nope]``."""
    nope = index_head_dim - rope_dim
    return torch.cat([nope + half_split_to_interleaved_perm(rope_dim), torch.arange(nope)])


def hf_attention_weights(ckpt, layer: int, owns_indexer: bool) -> dict:
    """HF-named fp32 weights of one Hy4 attention layer (reference input)."""
    a = f"model.layers.{layer}.self_attn."
    names = [
        "q_a_proj.weight",
        "q_a_layernorm.weight",
        "q_b_proj.weight",
        "kv_a_proj_with_mqa.weight",
        "kv_a_layernorm.weight",
        "kv_b_proj.weight",
        "o_proj.weight",
        "linear_gate.weight",
    ]
    if owns_indexer:
        names += [f"indexer.{n}" for n in ("wq_b.weight", "wk.weight", "k_norm.weight", "k_norm.bias")]
        names += ["indexer.weights_proj.weight"]
    w = {n: ckpt._read(a + n).float() for n in names}
    w["learnable_sink_param"] = ckpt.get(a + "learnable_sink_param").float()
    return w


def ttmla_weights(w: dict, config, owns_indexer: bool) -> dict:
    """ttMLA canonical weight dict in ttMLA's RoPE frame (see module docstring)."""
    H, N, R, L = config.num_attention_heads, config.qk_nope_head_dim, config.qk_rope_head_dim, config.kv_lora_rank
    rp = half_split_to_interleaved_perm(R)
    kv = w["kv_a_proj_with_mqa.weight"]
    qb = w["q_b_proj.weight"].reshape(H, N + R, -1)
    bf = lambda t: t.to(torch.bfloat16).contiguous()  # noqa: E731
    out = {
        "q_a_proj.weight": bf(w["q_a_proj.weight"]),
        "q_a_layernorm.weight": w["q_a_layernorm.weight"],
        "q_b_proj.weight": bf(torch.cat([qb[:, :N], qb[:, N:][:, rp]], dim=1).reshape(H * (N + R), -1)),
        "kv_a_proj_with_mqa.weight": bf(torch.cat([kv[:L], kv[L:][rp]])),
        "kv_a_layernorm.weight": w["kv_a_layernorm.weight"],
        "kv_b_proj.weight": bf(w["kv_b_proj.weight"]),
        "o_proj.weight": bf(w["o_proj.weight"]),
        "g_proj.weight": bf(w["linear_gate.weight"]),
        "attn_sink.weight": w["learnable_sink_param"],
    }
    if owns_indexer:
        d = config.index_head_dim
        p = index_k_device_perm(d, R)
        wq = w["indexer.wq_b.weight"]
        out.update(
            {
                "indexer.wq_b.weight": bf(wq.reshape(-1, d, wq.shape[-1])[:, p].reshape(wq.shape)),
                "indexer.wk.weight": bf(w["indexer.wk.weight"][p]),
                "indexer.k_norm.weight": w["indexer.k_norm.weight"][p].contiguous(),
                "indexer.k_norm_bias.weight": w["indexer.k_norm.bias"][p].contiguous(),
                "indexer.weights_proj.weight": w["indexer.weights_proj.weight"],
            }
        )
    return out


def layer0_hidden(ckpt, seq_len: int, eps: float) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``hn = input_layernorm(embed(tokens))`` (bf16-rounded, as the device gets it) and the golden token ids
    when the golden covers ``seq_len`` (else seeded random tokens, returned as None)."""
    import json

    meta = GOLDEN / "metadata.json"
    tokens = None
    if meta.is_file():
        ids = json.loads(meta.read_text())["token_ids"]
        if len(ids) >= seq_len:
            tokens = torch.tensor(ids[:seq_len])
    golden_tokens = tokens is not None
    if tokens is None:
        tokens = torch.randint(0, 120000, (seq_len,), generator=torch.Generator().manual_seed(0))
    emb = ckpt._read("model.embed_tokens.weight")[tokens].float()
    hn = rms_norm(emb, ckpt.get("model.layers.0.input_layernorm.weight").float(), eps)
    return hn.to(torch.bfloat16), (tokens if golden_tokens else None)


def golden_tensor(sub: str, name: str, seq_len: int) -> torch.Tensor | None:
    from safetensors import safe_open

    f = GOLDEN / sub / f"rows_00000000_{seq_len:08d}.safetensors"
    if not f.is_file():
        return None
    with safe_open(str(f), framework="pt") as h:
        return h.get_tensor(name)


# ---------------------------------------------------------------------------------------------- device
def device_indices_to_host(indices: ttnn.Tensor, mesh_device, sp_axis: int, seq_len: int) -> torch.Tensor:
    """ttMLA's top-k ``[1, 1, S/(sp*tp), K]`` per chip (sequence sharded over SP, then over TP by the
    head-to-sequence reshard) -> ``[S, K]`` int64 natural positions, -1 = masked."""
    rows, cols = mesh_device.shape
    shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(indices)]
    tp = cols if sp_axis == 0 else rows
    blocks = [None] * (rows * cols)
    for i, t in enumerate(shards):
        r, c = divmod(i, cols)
        sp_i, tp_i = (r, c) if sp_axis == 0 else (c, r)
        blocks[sp_i * tp + tp_i] = t.reshape(-1, t.shape[-1])
    idx = torch.cat(blocks).to(torch.int64) & 0xFFFFFFFF
    assert idx.shape[0] == seq_len, idx.shape
    return torch.where(idx >= seq_len, torch.full_like(idx, -1), idx)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def run_device_attention(config, weights, mesh_device, hn, layer_idx, topology, sp_axis, inject=None):
    seq_len = hn.shape[0]
    tp_axis = 1 - sp_axis
    mesh_shape = list(mesh_device.shape)
    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.SCALED_FP8,  # GLM-5.3 / Hy4 adapter's production KVPE format
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        tp_axis=tp_axis,
    )
    mla = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=layer_idx,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        topology=topology,
        is_chunked=True,
        active_seq_len=seq_len,
        layer_num=1,
        sparse_kv_cache_format=kvpe_cache.format,
    )
    assert mla.attn_sink is not None and mla._use_gate
    rope = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    rope_tensors = rope.get_rope_tensors_indexed(cache_seq_len_global=seq_len, chunk_size_global=seq_len)
    index_kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.index_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_full_indexer_layers(config) or 1,
        num_users=1,
        dtype=ttnn.bfloat8_b,
        tp_axis=tp_axis,
    )
    shard_dims = [None, None]
    shard_dims[tp_axis], shard_dims[sp_axis] = -1, -2
    tt_hidden = ttnn.from_torch(
        hn.view(1, 1, seq_len, -1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )
    t0 = time.time()
    out, indices = mla.forward(
        hidden_states=tt_hidden,
        rope_tensors=rope_tensors,
        kvpe_cache=kvpe_cache,
        actual_start=0,
        indexer_indices=inject,
        return_indexer_indices=True,
        index_kv_cache=index_kv_cache,
    )
    ttnn.synchronize_device(mesh_device)
    logger.info(f"layer {layer_idx}: device attention forward {time.time() - t0:.1f}s (includes compile)")
    out_host = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape)
    ).float()[0, 0]
    return out_host, indices


def _check_layer(name, dev_out, hn, w_hf, topk, dims):
    t0 = time.time()
    ref, kvpe = hy4_attention(hn.float(), w_hf, topk, dims, block=64)
    logger.info(f"{name}: CPU reference {time.time() - t0:.1f}s")
    p = pcc(dev_out, ref)
    diag = {}
    for tag, kw in (("no_sink", {"use_sink": False}), ("no_gate", {"use_gate": False})):
        other, _ = hy4_attention(hn.float(), w_hf, topk, dims, block=64, **kw)
        diag[tag] = (pcc(dev_out, other), pcc(ref, other))
    logger.info(
        f"{name}: PCC(device, HF ref) = {p:.6f} | vs ref without sink: device {diag['no_sink'][0]:.6f}, "
        f"ref {diag['no_sink'][1]:.6f} | vs ref without gate: device {diag['no_gate'][0]:.6f}, "
        f"ref {diag['no_gate'][1]:.6f} | rel max err {(dev_out - ref).abs().max() / ref.abs().max():.4f}"
    )
    return p, kvpe


def _topk_agreement(dev: torch.Tensor, gold: torch.Tensor, start: int) -> float:
    """Mean fraction of each row's golden top-k set that the device also picked, over rows >= ``start``
    (rows below index_topk select every causal key on both sides)."""
    fr = []
    for i in range(start, dev.shape[0]):
        g = set(gold[i][gold[i] >= 0].tolist())
        d = set(dev[i][dev[i] >= 0].tolist())
        fr.append(len(g & d) / max(1, len(g)))
    return sum(fr) / max(1, len(fr))


# ---------------------------------------------------------------------------------------------- test
@pytest.mark.parametrize(
    "mesh_device,device_params,sp_axis",
    [
        # GLM-5.3's prefill layout (default): SP 8 x TP 4, 16 heads per chip for the projections.
        pytest.param((8, 4), torus_xy_device_params(l1_small_size=1152), 0, id="sp8xtp4"),
        # decode-aligned head split (open layout decision, tt/hy4/layout.py): SP 4 x TP 8, 8 heads per chip.
        pytest.param((4, 8), torus_xy_device_params(l1_small_size=1152), 0, id="sp4xtp8"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=[f"s{s}" for s in SEQ_LENS])
@pytest.mark.timeout(0)
def test_hy4_attention_sublayer_vs_hf(mesh_device, device_params, sp_axis, seq_len):
    from models.demos.deepseek_v3_d_p.reference.hy4_config import hy4_hf_config
    from models.demos.deepseek_v3_d_p.tt.runners.hy4_checkpoint import Hy4Checkpoint

    config = hy4_hf_config(max_seq=seq_len)
    config.max_seq_len = seq_len
    assert config.mla_learnable_sink and config.mla_use_output_gate
    assert not indexer_layer_is_reused(config, FULL_LAYER) and indexer_layer_is_reused(config, SHARED_LAYER)
    dims = Hy4AttnDims.from_config(config)
    assert dims.softmax_scale == pytest.approx(ttMLA.softmax_scale(config))
    topology = per_axis_topology(device_params["fabric_config"])
    ckpt = Hy4Checkpoint(CHECKPOINT)
    hn, tokens = layer0_hidden(ckpt, seq_len, config.rms_norm_eps)
    results = {}

    # Full-indexer layer: the device indexer picks the top-k; the reference attends over the same rows.
    w0 = hf_attention_weights(ckpt, FULL_LAYER, owns_indexer=True)
    out0, idx_tt = run_device_attention(
        config, ttmla_weights(w0, config, True), mesh_device, hn, FULL_LAYER, topology, sp_axis
    )
    topk = device_indices_to_host(idx_tt, mesh_device, sp_axis, seq_len)
    results["full"], kvpe0 = _check_layer(f"layer {FULL_LAYER} (full)", out0, hn, w0, topk, dims)
    if tokens is not None:
        gold_topk = golden_tensor(
            f"dsa/dsa_topk_indices_layer_{FULL_LAYER}", f"dsa_topk_indices_layer_{FULL_LAYER}", seq_len
        )
        gold_kv = golden_tensor(f"kv_cache/layer_{FULL_LAYER}", f"kv_post_transform_layer_{FULL_LAYER}", seq_len)
        if gold_kv is not None:
            L = config.kv_lora_rank
            logger.info(
                f"layer {FULL_LAYER}: reference KVPE vs golden (checks hn = layer-0 sublayer input): "
                f"latent {pcc(kvpe0[:, :L], gold_kv[:, :L]):.6f} k_pe {pcc(kvpe0[:, L:], gold_kv[:, L:]):.6f}"
            )
        if gold_topk is not None and seq_len > config.index_topk:
            agree = _topk_agreement(topk, gold_topk.long(), config.index_topk)
            logger.info(f"layer {FULL_LAYER}: device top-k vs golden, rows >= {config.index_topk}: {agree:.4f}")
    del w0

    # Shared layer: no indexer of its own; reuses the full layer's top-k, as in the model.
    w2 = hf_attention_weights(ckpt, SHARED_LAYER, owns_indexer=False)
    out2, _ = run_device_attention(
        config, ttmla_weights(w2, config, False), mesh_device, hn, SHARED_LAYER, topology, sp_axis, inject=idx_tt
    )
    results["shared"], _ = _check_layer(f"layer {SHARED_LAYER} (shared)", out2, hn, w2, topk, dims)

    logger.info(f"Hy4 attention sublayer PCC (mesh {tuple(mesh_device.shape)}, S={seq_len}): {results}")
    for kind, p in results.items():
        assert p >= ATTN_PCC, f"{kind} layer attention PCC {p:.6f} < {ATTN_PCC}"
