# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``ttnn.experimental.deepseek.attention_qkv_decode`` on a block of ``h`` token rows (DSpark verify), per row
against torch.

A block is the tokens replica as ``(h, 32)``-tiled ``[h, K]`` shards. Row ``j`` must be what a one-token call
computes for token ``j`` at its own RoPE row: per-row RMSNorms of q_a and kv, q ``[1, 1, H * h, Dh]`` ROW_MAJOR
head-major (row ``head * h + j``), the kv rows into rows ``0..h-1`` of the staging tensor ``[1, 1, 32, Dh]``
(not the ring), and the compressor columns as an ``(h, 32)``-tiled ``[1, 1, h, N_c]``.

The perf test times s = 1 (today's ROW_MAJOR path) against s = 4 at a TP4 chip's shapes; profile it with::

    python -m tracy -r -p -v --no-web-server -t 8123 -o <out> -m pytest <this file> -k perf -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v41_flash.tests.decode.test_attention_qkv_decode import (
    EPS,
    GRID,
    HEAD_DIM,
    K,
    KV_GRID,
    NUM_QB_CORES,
    Q_A_GRID,
    Q_LORA,
    ROPE_DIM,
    _bf4,
    _cache_pos,
    _dram_sharded_weight,
    _gamma,
    _rms_norm,
    _rope,
    _setup,
)

# As the one-row test: q chains two bf4 matmuls with a 16-bit DST.
Q_PCC = 0.998
KV_PCC = 0.999
# kv's 64 rope channels alone: bf16 noise on few values; today's one-row path scores the same (0.998..0.9996) here.
KV_ROPE_PCC = 0.998
# Block row j against today's one-row call on the same token and angles.
VS_ONE_ROW_PCC = 0.9999


def _block(x: torch.Tensor, rows: int, device, memory_config=None) -> ttnn.Tensor:
    """``x`` ``[rows, W]`` as a bfloat16 ``(rows, 32)`` TILE ``[1, 1, rows, W]``."""
    return ttnn.from_torch(
        x.reshape(1, 1, rows, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        tile=ttnn.Tile((rows, ttnn.TILE_SIZE)),
        device=device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )


def _setup_rows(device, heads: int, rows: int, seed: int = 0, n_comp: int = 0):
    torch.manual_seed(seed)
    x = torch.randn(rows, K)
    w_q_a = torch.randn(K, Q_LORA) / K**0.5
    w_kv = torch.randn(K, HEAD_DIM) / K**0.5
    w_q_b = torch.randn(Q_LORA, heads * HEAD_DIM) / Q_LORA**0.5
    q_a_gamma = 1 + 0.1 * torch.randn(Q_LORA)
    kv_gamma = 1 + 0.1 * torch.randn(HEAD_DIM)
    # Row j: position p + j, a different angle per row.
    angles = (torch.arange(rows)[:, None] + 1.0) * torch.rand(ROPE_DIM // 2) * 3.0
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    staging = torch.randn(1, 1, 32, HEAD_DIM)

    host = dict(
        x=x.bfloat16().float(),
        w_q_a=_bf4(w_q_a),
        w_kv=_bf4(w_kv),
        w_q_b=_bf4(w_q_b),
        q_a_gamma=q_a_gamma.bfloat16().float(),
        kv_gamma=kv_gamma.bfloat16().float(),
        cos=cos.bfloat16().float(),
        sin=sin.bfloat16().float(),
        staging=staging.bfloat16().float(),
    )
    tt = dict(
        tokens=ttnn.from_torch(
            x.repeat(GRID.num_cores(), 1).reshape(1, 1, GRID.num_cores() * rows, K),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile((rows, ttnn.TILE_SIZE)),
            device=device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(GRID, [rows, K], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        ),
        w_q_a=_dram_sharded_weight(w_q_a, ttnn.TILE_SIZE, device),
        w_kv=_dram_sharded_weight(w_kv, ttnn.TILE_SIZE, device),
        w_q_b=_dram_sharded_weight(w_q_b, heads * HEAD_DIM // NUM_QB_CORES, device),
        q_a_gamma=_gamma(q_a_gamma, Q_A_GRID, device),
        kv_gamma=_gamma(kv_gamma, KV_GRID, device),
        cos=_block(cos, rows, device),
        sin=_block(sin, rows, device),
        staging=ttnn.from_torch(staging, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        cache_pos=_cache_pos(0, device),
    )
    if n_comp:
        w_comp = torch.randn(K, n_comp) / K**0.5
        tt["w_comp"] = _dram_sharded_weight(w_comp, ttnn.TILE_SIZE, device, dtype=ttnn.bfloat8_b)
        host["w_comp"] = ttnn.to_torch(tt["w_comp"]).float().reshape(K, n_comp)
        tt["comp_output"] = _block(torch.zeros(rows, n_comp), rows, device)
    return host, tt


def _reference_rows(host: dict, heads: int):
    """Per row j: q ``[heads, Dh]`` and kv ``[Dh]``, as a one-token step at row j's angles computes them."""
    qs, kvs = [], []
    for j in range(host["x"].shape[0]):
        x = host["x"][j : j + 1]
        cos, sin = host["cos"][j], host["sin"][j]
        q = _rms_norm(x @ host["w_q_a"], host["q_a_gamma"]) @ host["w_q_b"]
        qs.append(_rope(q.reshape(heads, HEAD_DIM), cos, sin))
        kv = _rms_norm(x @ host["w_kv"], host["kv_gamma"])
        kvs.append(_rope(kv, cos, sin).reshape(-1))
    return qs, kvs


def _run_rows(tt: dict, memory_config=None, comp: bool = False) -> ttnn.Tensor:
    extra = dict(w_comp=tt["w_comp"], comp_output=tt["comp_output"]) if comp else {}
    return ttnn.experimental.deepseek.attention_qkv_decode(
        tt["tokens"],
        tt["w_q_a"],
        tt["w_kv"],
        tt["w_q_b"],
        tt["q_a_gamma"],
        tt["kv_gamma"],
        tt["cos"],
        tt["sin"],
        tt["staging"],
        tt["cache_pos"],
        epsilon=EPS,
        memory_config=memory_config,
        **extra,
    )


def _sdpa_q_config(heads: int, rows: int) -> ttnn.MemoryConfig:
    """q HEIGHT_SHARDED 32 rows per core (8 heads x 4 rows at s = 4), over the first cores of row 0."""
    per_core = 32
    ncores = heads * rows // per_core
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ncores - 1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [per_core, HEAD_DIM], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _single_row(host: dict, tt: dict, j: int, device):
    """Row j through today's one-row path (ROW_MAJOR tokens, ring at row 0): q ``[heads, Dh]``, kv ``[Dh]``."""
    x = host["x"][j : j + 1]
    tokens = ttnn.from_torch(
        x.expand(GRID.num_cores(), K).reshape(1, 1, GRID.num_cores(), K),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(GRID, [1, K], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )
    table = lambda t: ttnn.from_torch(
        t.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ring = ttnn.from_torch(torch.zeros(1, 1, 32, HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    q = ttnn.experimental.deepseek.attention_qkv_decode(
        tokens,
        tt["w_q_a"],
        tt["w_kv"],
        tt["w_q_b"],
        tt["q_a_gamma"],
        tt["kv_gamma"],
        table(host["cos"][j]),
        table(host["sin"][j]),
        ring,
        _cache_pos(0, device),
        epsilon=EPS,
    )
    heads = q.shape[-2]
    return ttnn.to_torch(q).float().reshape(heads, HEAD_DIM), ttnn.to_torch(ring).float().reshape(32, HEAD_DIM)[0]


def _check(host, tt, q, heads: int, rows: int, label: str, comp: bool = False, device=None):
    expected_q, expected_kv = _reference_rows(host, heads)
    assert list(q.shape) == [1, 1, heads * rows, HEAD_DIM]
    got_q = ttnn.to_torch(q).float().reshape(heads, rows, HEAD_DIM)
    got_staging = ttnn.to_torch(tt["staging"]).float().reshape(32, HEAD_DIM)
    got_comp = ttnn.to_torch(tt["comp_output"]).float().reshape(rows, -1) if comp else None
    for j in range(rows):
        q_pcc = comp_pcc(expected_q[j], got_q[:, j], Q_PCC)
        rope_pcc = comp_pcc(expected_q[j][:, -ROPE_DIM:], got_q[:, j, -ROPE_DIM:], Q_PCC)
        kv_pcc = comp_pcc(expected_kv[j], got_staging[j], KV_PCC)
        kv_rope_pcc = comp_pcc(expected_kv[j][-ROPE_DIM:], got_staging[j, -ROPE_DIM:], KV_ROPE_PCC)
        # Per-row scale: catches a statistic shared or swapped between rows, which PCC alone would not.
        q_scale = (got_q[:, j].norm() / expected_q[j].norm()).item()
        kv_scale = (got_staging[j].norm() / expected_kv[j].norm()).item()
        msg = (
            f"{label} row {j}: q PCC {q_pcc[1]:.5f} (rope {rope_pcc[1]:.5f}, |q| ratio {q_scale:.4f}), "
            f"kv PCC {kv_pcc[1]:.5f} (rope {kv_rope_pcc[1]:.5f}, |kv| ratio {kv_scale:.4f})"
        )
        if comp:
            c_pcc = comp_pcc(host["x"][j] @ host["w_comp"], got_comp[j], KV_PCC)
            msg += f", comp PCC {c_pcc[1]:.5f}"
            assert c_pcc[0], msg
        if device is not None:
            # Against today's one-row path on the same row: the block must compute what a one-token step does.
            one_q, one_kv = _single_row(host, tt, j, device)
            vs_q, vs_kv = comp_pcc(one_q, got_q[:, j])[1], comp_pcc(one_kv, got_staging[j])[1]
            one_q_scale = (one_q.norm() / expected_q[j].norm()).item()
            one_kv_scale = (one_kv.norm() / expected_kv[j].norm()).item()
            msg += (
                f"; vs one-row path: q PCC {vs_q:.5f}, kv PCC {vs_kv:.5f} "
                f"(one-row |q| ratio {one_q_scale:.4f}, |kv| ratio {one_kv_scale:.4f})"
            )
            assert vs_q >= VS_ONE_ROW_PCC and vs_kv >= VS_ONE_ROW_PCC, msg
        logger.info(msg)
        assert q_pcc[0] and rope_pcc[0] and kv_pcc[0] and kv_rope_pcc[0], msg
        # A statistic shared or swapped between rows would be off by far more than bf16's bias.
        assert abs(q_scale - 1) < 0.03 and abs(kv_scale - 1) < 0.03, msg
    # The staging tensor's other rows are untouched.
    assert torch.equal(got_staging[rows:], host["staging"].reshape(32, HEAD_DIM)[rows:]), f"{label}: rows >= h changed"


@pytest.mark.parametrize("rows", (4, 2, 8))
@pytest.mark.parametrize("heads", (16, 64), ids=("tp4", "tp1"))
@torch.no_grad()
def test_attention_qkv_decode_rows(device, heads, rows):
    host, tt = _setup_rows(device, heads, rows)
    for step in range(2):
        q = _run_rows(tt)
        _check(host, tt, q, heads, rows, f"heads {heads} h {rows} step {step}", device=device if step == 0 else None)
        ttnn.deallocate(q)


@pytest.mark.parametrize("rows", (4, 2, 8))
@torch.no_grad()
def test_attention_qkv_decode_rows_commit_rows(device, rows):
    """``commit_rows``: the block's kv rows also ROW_MAJOR on one core, bit-identical to the staging rows."""
    host, tt = _setup_rows(device, 16, rows)
    commit = ttnn.from_torch(
        torch.zeros(1, 1, rows, HEAD_DIM),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                [rows, HEAD_DIM],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    for step in range(2):  # the second call runs from the program cache
        q = ttnn.experimental.deepseek.attention_qkv_decode(
            tt["tokens"],
            tt["w_q_a"],
            tt["w_kv"],
            tt["w_q_b"],
            tt["q_a_gamma"],
            tt["kv_gamma"],
            tt["cos"],
            tt["sin"],
            tt["staging"],
            tt["cache_pos"],
            epsilon=EPS,
            commit_rows=commit,
        )
        _check(host, tt, q, 16, rows, f"commit_rows h {rows} step {step}")
        ttnn.deallocate(q)
        got = ttnn.to_torch(commit).float().reshape(rows, HEAD_DIM)
        staging = ttnn.to_torch(tt["staging"]).float().reshape(32, HEAD_DIM)[:rows]
        assert torch.equal(got, staging), (got - staging).abs().max()


@torch.no_grad()
def test_attention_qkv_decode_rows_sdpa_layout(device):
    """q straight into flash_mqa_decode's s = 4 layout: 32 rows (8 heads x 4 rows) per core."""
    heads, rows = 16, 4
    host, tt = _setup_rows(device, heads, rows, seed=1)
    q = _run_rows(tt, memory_config=_sdpa_q_config(heads, rows))
    _check(host, tt, q, heads, rows, "sdpa layout")


@pytest.mark.parametrize("n_comp", (256, 128), ids=("tp4_ratio2", "tp4_ratio1"))
@pytest.mark.parametrize("rows", (4, 8))
@torch.no_grad()
def test_attention_qkv_decode_rows_compressor_columns(device, n_comp, rows):
    heads = 16
    host, tt = _setup_rows(device, heads, rows, seed=2, n_comp=n_comp)
    for step in range(2):
        q = _run_rows(tt, comp=True)
        _check(host, tt, q, heads, rows, f"n_comp {n_comp} h {rows} step {step}", comp=True)
        ttnn.deallocate(q)


def _run_one(tt: dict, cache_pos, comp=None):
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
        **(comp or {}),
    )


@pytest.mark.parametrize("which", ("s1", "s4"))
@pytest.mark.parametrize("n_comp", (0, 256), ids=("plain", "comp"))
@torch.no_grad()
def test_attention_qkv_decode_rows_perf(device, n_comp, which):
    """s = 1 (today's one-row path) and s = 4 at a TP4 chip's shapes (16 heads, D 5120, q_lora 1280, Dh 512),
    weights from DRAM, each after a warmup call and its signpost."""
    from tracy import signpost

    heads, iters = 16, 20
    _, tt1 = _setup(device, heads, 128)
    cache_pos = _cache_pos(77, device)
    comp1 = None
    if n_comp and which == "s1":
        w_comp = _dram_sharded_weight(torch.randn(K, n_comp) / K**0.5, ttnn.TILE_SIZE, device, dtype=ttnn.bfloat8_b)
        comp_row = ttnn.from_torch(
            torch.zeros(1, 1, 1, n_comp),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        comp1 = dict(w_comp=w_comp, comp_output=comp_row)
    if which == "s1":
        runs = {"s1": lambda: _run_one(tt1, cache_pos, comp1)}
    else:
        _, tt4 = _setup_rows(device, heads, 4, n_comp=n_comp)
        runs = {
            "s4": lambda: _run_rows(tt4, comp=bool(n_comp)),
            "s4_sdpa": lambda: _run_rows(tt4, memory_config=_sdpa_q_config(heads, 4), comp=bool(n_comp)),
        }
    for name, run in runs.items():
        ttnn.deallocate(run())
        ttnn.synchronize_device(device)
        signpost(f"qkv_{name}_{'comp' if n_comp else 'plain'}")
        for _ in range(iters):
            ttnn.deallocate(run())
        ttnn.synchronize_device(device)


@pytest.mark.parametrize("which", ("s1", "s4"))
@torch.no_grad()
def test_attention_qkv_decode_rows_perf_prefetcher(device, which):
    """The same timing with the weights streamed through the model's tensor-prefetcher rings (TP4 ring sizes, on
    one chip), as the decode stage runs the op: one queued request per weight per call."""
    from tracy import signpost

    from models.experimental.deepseek_v41_flash.tt.decode.decode_prefetch import (
        FUSED_KV_RING,
        FUSED_Q_A_RING,
        make_prefetch_buffers,
        prefetch_rings,
    )
    from tests.ttnn.unit_tests.operations.prefetcher_common import tensor_prefetcher_session

    if not ttnn.experimental.is_tensor_prefetcher_supported(device):
        pytest.skip("needs the tensor prefetcher (programmable DRAM cores)")
    heads, iters = 16, 20
    buffers = make_prefetch_buffers(device, ttnn.bfloat4_b, tp_size=4)
    pipes = {
        "q_a": buffers[FUSED_Q_A_RING],
        "kv": buffers[FUSED_KV_RING],
        "q_b": buffers[prefetch_rings(4)["q_b_proj"].key],
    }
    if which == "s1":
        _, tt = _setup(device, heads, 128)
        cache_pos = _cache_pos(77, device)
        args = lambda: (tt["tokens"], tt["kv_cache"], cache_pos, None)
    else:
        _, tt = _setup_rows(device, heads, 4)
        args = lambda: (tt["tokens"], tt["staging"], tt["cache_pos"], _sdpa_q_config(heads, 4))

    def run():
        for name, weight in (("q_a", tt["w_q_a"]), ("kv", tt["w_kv"]), ("q_b", tt["w_q_b"])):
            ttnn.experimental.queue_tensor_prefetcher_request(device, [(weight, 1)], prefetcher_pipes=pipes[name])
        tokens, cache, pos, mem = args()
        return ttnn.experimental.deepseek.attention_qkv_decode(
            tokens,
            tt["w_q_a"],
            tt["w_kv"],
            tt["w_q_b"],
            tt["q_a_gamma"],
            tt["kv_gamma"],
            tt["cos"],
            tt["sin"],
            cache,
            pos,
            epsilon=EPS,
            memory_config=mem,
            prefetcher_pipes=[*pipes["q_a"], *pipes["kv"], *pipes["q_b"]],
        )

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        ttnn.deallocate(run())
        ttnn.synchronize_device(device)
        signpost(f"qkv_prefetch_{which}")
        for _ in range(iters):
            ttnn.deallocate(run())
        ttnn.synchronize_device(device)
