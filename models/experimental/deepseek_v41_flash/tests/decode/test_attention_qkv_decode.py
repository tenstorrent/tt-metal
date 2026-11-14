# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``ttnn.experimental.deepseek.attention_qkv_decode`` against torch, on random bf4 weights.

The reference uses the weights as the device holds them (bf4 round-tripped), so the comparison measures the
op's own error: bf16 activations, the LoFi bf4 matmuls, the bf16 RMSNorm statistics and the SFPU RoPE.
``heads=16`` is a chip's share at TP4, ``heads=64`` the whole model on one chip.

``test_attention_layer_fused_qkv*`` decode ``DeepSeekV41Attention`` layers on random weights with and without
the op (``DEEPSEEK_V41_FUSED_QKV``): on one chip, on a 1x4 stage, and on a 1x4 stage with the prefetcher.

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_attention_qkv_decode.py
"""

import contextlib
import json
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR
from models.experimental.deepseek_v41_flash.tt.decode import attention as attention_module
from models.experimental.deepseek_v41_flash.tt.decode.attention import (
    DECODE_LAYOUTS,
    DeepSeekV41Attention,
    build_decode_caches,
)
from models.experimental.deepseek_v41_flash.tt.decode.decode_prefetch import make_prefetch_buffers
from models.experimental.deepseek_v4_flash.tt.common import width_sharded_l1_config
from models.experimental.deepseek_v4_flash.tt.decode.attention import (
    _apply_rope,
    _decode_activation,
    _interleaved_rotate_matrix,
    _update_cache_at,
)
from models.experimental.deepseek_v4_flash.tt.layers import LinearDecode
from models.experimental.deepseek_v4_flash.tt.weight_cache import _load_weight
from tests.ttnn.unit_tests.operations.prefetcher_common import tensor_prefetcher_session

K = 5120
Q_LORA = 1280
HEAD_DIM = 512
ROPE_DIM = 64
EPS = 1e-6
NUM_DRAM_BANKS = 8
NUM_QB_CORES = 64
# The bf4 matmuls accumulate K in a 16-bit DST; q chains two of them (today's ops score ~0.9985 on q too).
Q_PCC = 0.998
KV_PCC = 0.999

GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 9))})
KV_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(7, 9))})
Q_A_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(11, 9))})


def _bf4(w: torch.Tensor) -> torch.Tensor:
    """``w`` as a bfloat4_b TILE tensor holds it."""
    return ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()


def _dram_sharded_weight(w: torch.Tensor, slab_width: int, device, dtype=ttnn.bfloat4_b) -> ttnn.Tensor:
    """``w`` ``[K, N]`` as ``dtype`` TILE, ND-sharded in DRAM one ``[K, slab_width]`` slab per core."""
    memory_config = ttnn.MemoryConfig(
        ttnn.BufferType.DRAM,
        ttnn.NdShardSpec(
            ttnn.Shape([w.shape[0], slab_width]),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(NUM_DRAM_BANKS - 1, 0))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )
    return ttnn.from_torch(
        w.reshape(1, 1, *w.shape),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def _gamma(gamma: torch.Tensor, grid: ttnn.CoreRangeSet, device) -> ttnn.Tensor:
    """``gamma`` ``[N]`` as a bfloat16 1x32 TILE ``[1, N]``, WIDTH_SHARDED one tile per core of ``grid``."""
    return ttnn.from_torch(
        gamma.reshape(1, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((1, ttnn.TILE_SIZE)),
        device=device,
        memory_config=ttnn.create_sharded_memory_config(
            (1, ttnn.TILE_SIZE),
            core_grid=grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )


def _rms_norm(x: torch.Tensor, gamma: torch.Tensor | None = None) -> torch.Tensor:
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    return y if gamma is None else y * gamma


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved RoPE of the trailing ``ROPE_DIM`` channels of each row of ``x`` ``[rows, HEAD_DIM]``."""
    nope, rope = x[:, :-ROPE_DIM], x[:, -ROPE_DIM:]
    pairs = rope.reshape(rope.shape[0], -1, 2)
    rotated = torch.stack([-pairs[..., 1], pairs[..., 0]], dim=-1).reshape_as(rope)
    return torch.cat([nope, rope * cos + rotated * sin], dim=-1)


def _setup(device, heads: int, cache_rows: int, seed: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(1, K)
    w_q_a = torch.randn(K, Q_LORA) / K**0.5
    w_kv = torch.randn(K, HEAD_DIM) / K**0.5
    w_q_b = torch.randn(Q_LORA, heads * HEAD_DIM) / Q_LORA**0.5
    q_a_gamma = 1 + 0.1 * torch.randn(Q_LORA)
    kv_gamma = 1 + 0.1 * torch.randn(HEAD_DIM)
    angles = torch.rand(ROPE_DIM // 2) * 6.28
    cos = angles.cos().repeat_interleave(2).reshape(1, 1, 1, ROPE_DIM)
    sin = angles.sin().repeat_interleave(2).reshape(1, 1, 1, ROPE_DIM)
    cache = torch.randn(1, 1, cache_rows, HEAD_DIM)

    host = dict(
        x=x.bfloat16().float(),
        w_q_a=_bf4(w_q_a),
        w_kv=_bf4(w_kv),
        w_q_b=_bf4(w_q_b),
        q_a_gamma=q_a_gamma.bfloat16().float(),
        kv_gamma=kv_gamma.bfloat16().float(),
        cos=cos.bfloat16().float().reshape(-1),
        sin=sin.bfloat16().float().reshape(-1),
        cache=cache.bfloat16().float(),
        raw=dict(w_q_a=w_q_a, w_kv=w_kv, w_q_b=w_q_b, q_a_gamma=q_a_gamma, kv_gamma=kv_gamma),
    )
    tt = dict(
        tokens=ttnn.from_torch(
            x.expand(GRID.num_cores(), K).reshape(1, 1, GRID.num_cores(), K),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(GRID, [1, K], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        ),
        w_q_a=_dram_sharded_weight(w_q_a, ttnn.TILE_SIZE, device),
        w_kv=_dram_sharded_weight(w_kv, ttnn.TILE_SIZE, device),
        w_q_b=_dram_sharded_weight(w_q_b, heads * HEAD_DIM // NUM_QB_CORES, device),
        q_a_gamma=_gamma(q_a_gamma, Q_A_GRID, device),
        kv_gamma=_gamma(kv_gamma, KV_GRID, device),
        cos=ttnn.from_torch(cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        sin=ttnn.from_torch(sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        kv_cache=ttnn.from_torch(cache, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    )
    return host, tt


def _reference(host: dict, heads: int):
    q_a = host["x"] @ host["w_q_a"]
    q = _rms_norm(q_a, host["q_a_gamma"]) @ host["w_q_b"]
    q = _rope(q.reshape(heads, HEAD_DIM), host["cos"], host["sin"])
    kv = _rms_norm(host["x"] @ host["w_kv"], host["kv_gamma"])
    kv = _rope(kv, host["cos"], host["sin"])
    return q, kv.reshape(-1)


class _TodayQkv:
    """The decode attention's path to SDPA today, on the same weights: q_a (RMSNorm fused, mcast onto q_b's
    grid), q_b and kv (RMSNorm fused) ``matmul_decode``s, ``fused_partial_rope`` on q and kv and
    ``paged_update_cache``, as ``DeepSeekV41Attention`` builds and calls them."""

    def __init__(self, host: dict, device, heads: int):
        raw = host["raw"]
        self.q_a_proj = LinearDecode(
            raw["w_q_a"].T.contiguous(),
            device,
            dtype=ttnn.bfloat4_b,
            **DECODE_LAYOUTS["q_a_proj"],
            rectangle_b_grid=True,
            use_rm_hs=True,
            dram_sharded_weight=True,
        )
        self.kv_proj = LinearDecode(
            raw["w_kv"].T.contiguous(),
            device,
            dtype=ttnn.bfloat4_b,
            **DECODE_LAYOUTS["kv_proj"],
            use_rm_hs=True,
            dram_sharded_weight=True,
        )
        self.kv_proj.set_output_core_grid(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        )
        self.q_b_proj = LinearDecode(
            raw["w_q_b"].T.contiguous(),
            device,
            dtype=ttnn.bfloat4_b,
            **{**DECODE_LAYOUTS["q_b_proj"], "N": heads * HEAD_DIM},
            rectangle_b_grid=True,
            use_rm_hs=True,
            dram_sharded_weight=True,
        )
        self.q_a_proj.set_output_core_grid(self.q_b_proj.b_core_grid())
        self.q_a_proj.enable_fused_rms_norm(EPS, raw["q_a_gamma"])
        self.kv_proj.enable_fused_rms_norm(EPS, raw["kv_gamma"])
        self.rot = _load_weight(_interleaved_rotate_matrix(ROPE_DIM), device)

    def __call__(self, tt: dict, cache_pos: ttnn.Tensor) -> ttnn.Tensor:
        tokens = tt["tokens"]
        q_a = self.q_a_proj(_decode_activation(self.q_a_proj, tokens))
        q = self.q_b_proj(_decode_activation(self.q_b_proj, q_a))
        ttnn.deallocate(q_a)
        kv = self.kv_proj(_decode_activation(self.kv_proj, tokens))
        q = _apply_rope(q, tt["cos"], tt["sin"], self.rot, ROPE_DIM, head_dim=HEAD_DIM)
        kv = _apply_rope(kv, tt["cos"], tt["sin"], self.rot, ROPE_DIM)
        _update_cache_at(tt["kv_cache"], kv, cache_pos)
        ttnn.deallocate(kv)
        return q


def _cache_pos(pos: int, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )


def _run(tt: dict, cache_pos: ttnn.Tensor, **comp) -> ttnn.Tensor:
    return ttnn.experimental.deepseek.attention_qkv_decode(
        tt["tokens"],
        tt["w_q_a"],
        tt["w_kv"],
        tt["w_q_b"],
        tt["q_a_gamma"],
        tt["kv_gamma"],
        tt["cos"],
        tt["sin"],
        tt["kv_cache"],
        cache_pos,
        epsilon=EPS,
        **comp,
    )


@pytest.mark.parametrize("heads", (16, 64), ids=("tp4", "tp1"))
# Across a face (16) and a tile row (32), the window's last row, and a later session's slot of a shared ring.
@pytest.mark.parametrize("positions", [(0, 5, 17, 31, 32, 127, 300, 511)], ids=["rows"])
@torch.no_grad()
def test_attention_qkv_decode(device, heads, positions):
    cache_rows = 512
    host, tt = _setup(device, heads, cache_rows)
    expected_q, expected_kv = _reference(host, heads)
    expected_cache = host["cache"].clone()
    for pos in positions:
        q = _run(tt, _cache_pos(pos, device))
        assert list(q.shape) == [1, 1, heads, HEAD_DIM]
        got_q = ttnn.to_torch(q).float().reshape(heads, HEAD_DIM)
        passing, pcc = comp_pcc(expected_q, got_q, Q_PCC)
        logger.info(f"heads {heads} pos {pos}: q PCC {pcc}")
        assert passing, f"pos {pos}: q PCC {pcc} < {Q_PCC}"
        rope_pcc = comp_pcc(expected_q[:, -ROPE_DIM:], got_q[:, -ROPE_DIM:], Q_PCC)
        assert rope_pcc[0], f"pos {pos}: q rope channels PCC {rope_pcc[1]} < {Q_PCC}"

        expected_cache[0, 0, pos] = expected_kv
        got_cache = ttnn.to_torch(tt["kv_cache"]).float()
        passing, pcc = comp_pcc(expected_kv, got_cache[0, 0, pos], KV_PCC)
        logger.info(f"heads {heads} pos {pos}: kv PCC {pcc}")
        assert passing, f"pos {pos}: kv PCC {pcc} < {KV_PCC}"
        rope_pcc = comp_pcc(expected_kv[-ROPE_DIM:], got_cache[0, 0, pos, -ROPE_DIM:], KV_PCC)
        assert rope_pcc[0], f"pos {pos}: kv rope channels PCC {rope_pcc[1]} < {KV_PCC}"
        others = torch.ones(cache_rows, dtype=torch.bool)
        others[pos] = False
        # Every other row is the bf16 the ring held, untouched by the tile read-modify-write.
        assert torch.equal(got_cache[0, 0, others], expected_cache[0, 0, others]), f"pos {pos}: other rows changed"
        expected_cache[0, 0, pos] = got_cache[0, 0, pos]
        ttnn.deallocate(q)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 1 << 20}], indirect=True)
@torch.no_grad()
def test_attention_qkv_decode_trace(device):
    """One trace replayed at several positions: the position is read on device, so only cache_pos changes."""
    heads, cache_rows = 16, 128
    host, tt = _setup(device, heads, cache_rows)
    expected_q, expected_kv = _reference(host, heads)
    cache_pos = _cache_pos(0, device)
    ttnn.deallocate(_run(tt, cache_pos))

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    q = _run(tt, cache_pos)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    for pos in (3, 64, 127):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
            cache_pos,
        )
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        q_pcc = comp_pcc(expected_q, ttnn.to_torch(q).float().reshape(heads, HEAD_DIM), Q_PCC)
        kv_pcc = comp_pcc(expected_kv, ttnn.to_torch(tt["kv_cache"]).float()[0, 0, pos], KV_PCC)
        logger.info(f"trace pos {pos}: q PCC {q_pcc[1]}, kv PCC {kv_pcc[1]}")
        assert q_pcc[0] and kv_pcc[0], f"trace pos {pos}: q PCC {q_pcc[1]}, kv PCC {kv_pcc[1]}"
    ttnn.release_trace(device, trace_id)


@pytest.mark.parametrize("heads", (16, 64), ids=("tp4", "tp1"))
@torch.no_grad()
def test_attention_qkv_decode_vs_today(device, heads):
    """Both paths against the reference: the fused op must be as accurate as today's ops."""
    pos = 77
    host, tt = _setup(device, heads, 128)
    today = _TodayQkv(host, device, heads)
    expected_q, expected_kv = _reference(host, heads)
    pccs = {}
    for name, run in (("fused", _run), ("today", today)):
        q = run(tt, _cache_pos(pos, device))
        got_q = ttnn.to_torch(q).float().reshape(heads, HEAD_DIM)
        got_kv = ttnn.to_torch(tt["kv_cache"]).float()[0, 0, pos]
        pccs[name] = (comp_pcc(expected_q, got_q)[1], comp_pcc(expected_kv, got_kv)[1])
        logger.info(f"heads {heads} {name}: q PCC {pccs[name][0]}, kv PCC {pccs[name][1]}")
        ttnn.deallocate(q)
    for i, what in enumerate(("q", "kv")):
        fused, ref = pccs["fused"][i], pccs["today"][i]
        assert 1 - fused <= 1.5 * (1 - ref), f"{what}: fused PCC {fused} well below today's {ref}"


# A rank's compressor columns: kv + gate at TP4 (ratio 2), kv alone at TP4 (ratio 1), kv + gate on one chip.
@pytest.mark.parametrize("n_comp", (256, 128, 1024), ids=("tp4_ratio2", "tp4_ratio1", "tp1_ratio2"))
@torch.no_grad()
def test_attention_qkv_decode_compressor_columns(device, n_comp):
    """``w_comp``'s columns of the tokens row into ``comp_output``, beside an unchanged q and kv."""
    heads, pos = 16, 77
    host, tt = _setup(device, heads, 128)
    w_comp = torch.randn(K, n_comp) / K**0.5
    tt_w_comp = _dram_sharded_weight(w_comp, ttnn.TILE_SIZE, device, dtype=ttnn.bfloat8_b)
    w_comp = ttnn.to_torch(tt_w_comp).float().reshape(K, n_comp)
    comp_output = ttnn.from_torch(
        torch.zeros(1, 1, 1, n_comp),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    expected_q, expected_kv = _reference(host, heads)
    for step in range(2):
        q = _run(tt, _cache_pos(pos + step, device), w_comp=tt_w_comp, comp_output=comp_output)
        got_q = ttnn.to_torch(q).float().reshape(heads, HEAD_DIM)
        got_kv = ttnn.to_torch(tt["kv_cache"]).float()[0, 0, pos + step]
        got_comp = ttnn.to_torch(comp_output).float().reshape(-1)
        q_pcc, kv_pcc = comp_pcc(expected_q, got_q, Q_PCC), comp_pcc(expected_kv, got_kv, KV_PCC)
        comp_pcc_ = comp_pcc(host["x"].reshape(-1) @ w_comp, got_comp, KV_PCC)
        logger.info(f"n_comp {n_comp} step {step}: q PCC {q_pcc[1]}, kv PCC {kv_pcc[1]}, comp PCC {comp_pcc_[1]}")
        assert q_pcc[0] and kv_pcc[0], f"q PCC {q_pcc[1]}, kv PCC {kv_pcc[1]}"
        assert comp_pcc_[0], f"compressor columns PCC {comp_pcc_[1]} < {KV_PCC}"
        ttnn.deallocate(q)


def _text_config():
    """The checkpoint's ``text_config`` as :func:`load_config` gives it, from ``config.json`` alone."""
    configs = sorted(DEFAULT_MODEL_DIR.glob("snapshots/*/config.json"))
    if not configs:
        return None
    cfg = json.loads(configs[0].read_text())["text_config"]
    cfg["layer_types"] = ["sliding_attention" if r == 0 else "compressed_attention" for r in cfg["compress_ratios"]]
    cfg["num_local_experts"] = cfg["n_routed_experts"]
    return SimpleNamespace(**cfg)


def _random_attention_weights(config, layer: int) -> dict:
    """Random weights of every tensor layer ``layer``'s attention reads, at the checkpoint's shapes."""
    d, h, dh = config.hidden_size, config.num_attention_heads, config.head_dim
    groups, o_rank, q_lora = config.o_groups, config.o_lora_rank, config.q_lora_rank
    weights = {
        "q_a_proj.weight": torch.randn(q_lora, d) / d**0.5,
        "q_a_norm.weight": 1 + 0.1 * torch.randn(q_lora),
        "q_b_proj.weight": torch.randn(h * dh, q_lora) / q_lora**0.5,
        "kv_proj.weight": torch.randn(dh, d) / d**0.5,
        "kv_norm.weight": 1 + 0.1 * torch.randn(dh),
        "o_a_proj.weight": torch.randn(groups * o_rank, h * dh // groups) / (h * dh // groups) ** 0.5,
        "o_b_proj.weight": torch.randn(d, groups * o_rank) / (groups * o_rank) ** 0.5,
        "sinks": torch.randn(h),
    }
    if layer in config.kv_source_layer_ids:
        ih = config.index_head_dim
        weights |= {
            "compressor.wkv.weight": torch.randn(dh, d) / d**0.5,
            "compressor.wgate.weight": torch.randn(dh, d) / d**0.5,
            "compressor.norm.weight": 1 + 0.1 * torch.randn(dh),
            "indexer.wk.weight": torch.randn(ih, dh) / dh**0.5,
            "indexer.k_norm.weight": 1 + 0.1 * torch.randn(ih),
        }
    if layer in config.index_source_layer_ids:
        nh, ih = config.index_n_heads, config.index_head_dim
        weights |= {
            "indexer.wq_b.weight": torch.randn(nh * ih, q_lora) / q_lora**0.5,
            "indexer.weights_proj.weight": torch.randn(nh, d) / d**0.5,
        }
    return weights


def _decode_both(
    attns: dict,
    caches: dict,
    layers: tuple,
    hidden,
    angles,
    hidden_config,
    device,
    use_prefetcher: bool = False,
):
    """Each position through ``layers`` in order, on both builds, comparing their block outputs."""
    for pos in range(hidden.shape[1]):
        for j, i in enumerate(layers):
            x = ttnn.from_torch(
                hidden[j, pos].reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=hidden_config,
            )
            outs = {}
            for fused in attns:
                attn = attns[fused][i]
                inputs = attn.decode_inputs(pos, angles.cos(), angles.sin())
                use_fused = not attn.is_index_source or not inputs.get("select") or bool(inputs.get("dense"))
                if fused and attn.is_index_source:
                    if pos == 0 and attn.compress_ratio > 1:
                        assert not inputs.get("select")
                        assert use_fused
                    elif (pos + 1) // attn.compress_ratio <= attn.index_topk:
                        assert inputs.get("select")
                        assert bool(inputs.get("dense")) == attention_module.DENSE_SELECT
                        assert use_fused == attention_module.DENSE_SELECT
                if use_prefetcher:
                    attn.prefetch_weights(fused_qkv=use_fused)
                out = attn.decode(x, scache=caches[fused][i], **inputs)
                outs[fused] = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
            passing, pcc = comp_pcc(outs[False], outs[True], 0.99)
            logger.info(f"layer {i} pos {pos}: fused vs separate output PCC {pcc}")
            assert passing, f"layer {i} pos {pos}: fused vs separate output PCC {pcc}"


def _check_layer_fused_qkv(
    device,
    tp_size: int,
    layers: tuple,
    monkeypatch,
    use_prefetcher: bool = False,
    dense_select: bool = True,
    separate_column_compressor: bool = True,
):
    """``layers`` on random weights, in order from position 0 past the ring wrap, with and without
    attention_qkv_decode: every layer's block output and ring must agree. ``use_prefetcher`` streams both
    builds' ring weights through one set of rings, as a stage device does. ``separate_column_compressor=False``
    gives the separate build the bfloat16 row-parallel compressor instead of the bfloat8_b column split."""
    config = _text_config()
    seq_len = config.sliding_window + 4
    torch.manual_seed(0)
    weights = {i: _random_attention_weights(config, i) for i in layers}
    prefetch_buffers = make_prefetch_buffers(device, ttnn.bfloat4_b, tp_size=tp_size) if use_prefetcher else None
    monkeypatch.setattr(attention_module, "DENSE_SELECT", dense_select)
    attns, caches = {}, {}
    for fused in (True, False):
        monkeypatch.setattr(attention_module, "_FUSED_QKV", fused)
        monkeypatch.setattr(attention_module, "COLUMN_COMPRESSOR", fused or separate_column_compressor)
        attns[fused] = {
            i: DeepSeekV41Attention(
                config,
                i,
                weights[i],
                device,
                weight_dtype=ttnn.bfloat4_b,
                tp_size=tp_size,
                use_prefetcher=use_prefetcher,
                prefetch_buffers=prefetch_buffers,
            )
            for i in layers
        }
        caches[fused] = build_decode_caches(device, config, layers, seq_len)
    for i in layers:
        assert attns[True][i].fused_qkv
        assert not attns[False][i].fused_qkv
        # At TP4 a q_b slab fits the shared ring, so the fused op streams all three weights.
        assert (attns[True][i].fused_pipes is not None) == (use_prefetcher and attns[True][i].fused_qkv)

    angles = torch.arange(seq_len)[:, None] * torch.rand(config.qk_rope_head_dim // 2)
    hidden = torch.randn(len(layers), seq_len, config.hidden_size)
    hidden_config = width_sharded_l1_config(1, config.hidden_size, device, tile_height=1)
    session = tensor_prefetcher_session(device) if use_prefetcher else contextlib.nullcontext()
    with session:
        if use_prefetcher:
            ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        _decode_both(attns, caches, layers, hidden, angles, hidden_config, device, use_prefetcher=use_prefetcher)
    for i in layers:
        for attr in ("kv", "comp_kv", "sel_tile"):
            if getattr(caches[True][i], attr) is None:
                continue
            bufs = {
                fused: ttnn.to_torch(ttnn.get_device_tensors(getattr(caches[fused][i], attr))[0]).float()
                for fused in caches
            }
            passing, pcc = comp_pcc(bufs[False], bufs[True], 0.999)
            logger.info(f"layer {i}: {attr} PCC {pcc}")
            assert passing, f"layer {i}: {attr} PCC {pcc}"


# Window only (0); ratio 2, its kv + index source (2, fused when not select or dense) and a consumer (3).
FUSED_QKV_LAYERS = pytest.mark.parametrize("layers", ((0,), (2, 3)), ids=("window", "ratio2"))
INDEX_SOURCE_FUSED_LAYERS = pytest.mark.parametrize("layers", ((2,), (2, 3)), ids=("index_source", "ratio2"))


@pytest.mark.skipif(_text_config() is None, reason=f"no V4.1-Flash config.json under {DEFAULT_MODEL_DIR}")
@FUSED_QKV_LAYERS
@torch.no_grad()
def test_attention_layer_fused_qkv(device, layers, monkeypatch):
    _check_layer_fused_qkv(device, 1, layers, monkeypatch)


@pytest.mark.skipif(_text_config() is None, reason=f"no V4.1-Flash config.json under {DEFAULT_MODEL_DIR}")
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["8x4"])
@FUSED_QKV_LAYERS
@torch.no_grad()
def test_attention_layer_fused_qkv_tp4(mesh_device, layers, monkeypatch):
    """The same on one 1x4 stage of the Galaxy, as the full-model decode carves it."""
    if tuple(mesh_device.shape) != (8, 4):
        pytest.skip(f"needs an 8x4 mesh, got {tuple(mesh_device.shape)}")
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
    _check_layer_fused_qkv(submesh, 4, layers, monkeypatch)


@pytest.mark.skipif(_text_config() is None, reason=f"no V4.1-Flash config.json under {DEFAULT_MODEL_DIR}")
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["8x4"])
@FUSED_QKV_LAYERS
@torch.no_grad()
def test_attention_layer_fused_qkv_tp4_prefetcher(mesh_device, layers, monkeypatch):
    """The same with the DRISC prefetcher on, as the full-model decode runs: the fused layers' o_a / o_b and
    the separate build's every projection stream through the rings, which must stay in step."""
    if tuple(mesh_device.shape) != (8, 4):
        pytest.skip(f"needs an 8x4 mesh, got {tuple(mesh_device.shape)}")
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
    if not ttnn.experimental.is_tensor_prefetcher_supported(submesh):
        pytest.skip("needs the tensor prefetcher (programmable DRAM cores)")
    _check_layer_fused_qkv(submesh, 4, layers, monkeypatch, use_prefetcher=True)


@pytest.mark.skipif(_text_config() is None, reason=f"no V4.1-Flash config.json under {DEFAULT_MODEL_DIR}")
@INDEX_SOURCE_FUSED_LAYERS
@pytest.mark.parametrize("dense_select", (True, False), ids=("dense", "sparse"))
@torch.no_grad()
def test_attention_layer_fused_qkv_index_source(device, layers, dense_select, monkeypatch):
    """Index source layer 2 (and consumer 3) under dense (fused for pos >= 0) vs sparse select (fused at pos=0,
    separate _qkv + select_kv for pos >= 1)."""
    _check_layer_fused_qkv(device, 1, layers, monkeypatch, dense_select=dense_select)


@pytest.mark.skipif(_text_config() is None, reason=f"no V4.1-Flash config.json under {DEFAULT_MODEL_DIR}")
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["8x4"])
@pytest.mark.parametrize("dense_select", (True, False), ids=("dense", "sparse"))
@torch.no_grad()
def test_attention_layer_column_compressor_tp4(mesh_device, dense_select, monkeypatch):
    """Kv source 2 and its consumer on a 1x4 stage: the bfloat8_b column-split compressor (inside
    attention_qkv_decode on its one-row steps) against the bfloat16 row-parallel one."""
    if tuple(mesh_device.shape) != (8, 4):
        pytest.skip(f"needs an 8x4 mesh, got {tuple(mesh_device.shape)}")
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
    _check_layer_fused_qkv(submesh, 4, (2, 3), monkeypatch, dense_select=dense_select, separate_column_compressor=False)


@pytest.mark.skipif(_text_config() is None, reason=f"no V4.1-Flash config.json under {DEFAULT_MODEL_DIR}")
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["8x4"])
@INDEX_SOURCE_FUSED_LAYERS
@pytest.mark.parametrize("dense_select", (True, False), ids=("dense", "sparse"))
@torch.no_grad()
def test_attention_layer_fused_qkv_index_source_tp4_prefetcher(mesh_device, layers, dense_select, monkeypatch):
    """The same on a 1x4 stage with the DRISC prefetcher, verifying ring sync across fused/unfused steps."""
    if tuple(mesh_device.shape) != (8, 4):
        pytest.skip(f"needs an 8x4 mesh, got {tuple(mesh_device.shape)}")
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
    if not ttnn.experimental.is_tensor_prefetcher_supported(submesh):
        pytest.skip("needs the tensor prefetcher (programmable DRAM cores)")
    _check_layer_fused_qkv(submesh, 4, layers, monkeypatch, use_prefetcher=True, dense_select=dense_select)


@pytest.mark.parametrize("path", ("fused", "today"))
@pytest.mark.parametrize("heads", (16, 64), ids=("tp4", "tp1"))
@torch.no_grad()
def test_attention_qkv_decode_perf(device, heads, path):
    """A few warm dispatches of either path to time under Tracy:

    python -m tracy -r -p -v -m pytest models/experimental/deepseek_v41_flash/tests/decode/test_attention_qkv_decode.py -k perf
    """
    host, tt = _setup(device, heads, 128)
    run = _run if path == "fused" else _TodayQkv(host, device, heads)
    cache_pos = _cache_pos(77, device)
    for _ in range(10):
        ttnn.deallocate(run(tt, cache_pos))
    ttnn.synchronize_device(device)
