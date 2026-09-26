# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Routed-expert FFN DRAM bandwidth at MiMo-V2 shapes (H 4096, I 2048, bf4 weights, bf8 activations), 2x2.

The op is weight-bandwidth bound for any realistic load: at bf4 (0.5625 B/element) the ridge point of an 88-core
LoFi grid against ~500 GB/s DRAM is ~270 tokens per expert, and real routing averages 40 (640 tok/chip) to 128
(2048 tok/chip). So the number that matters is weight bytes read / op time, per chip.

Each case builds a synthetic dispatch result directly (no routing, no dispatch): ``active`` of the 64 local experts
per chip get ``tokens`` rows each, the rest get 0. Weights are MiMo's real layer-1 experts, loaded cache-only from
``MIMO_TTNN_CACHE`` (build it once with any decoder-layer test and the env var set). Paths:
  unified  unified_routed_expert_moe only
  fused    moe_fused_swiglu only (AI-CodeGen)
  hybrid   experts with <= MIMO_RE_HYBRID_THRESHOLD (default 64) tokens -> fused, the rest -> unified
``test_linear_reference`` times a plain ttnn.linear that reads the same bf4 bytes (M=32, one fused
[4096, 64 * 3 * 2048 / 3] weight), as the "ordinary DRAM-bound matmul" reference.

Tags ``expert_{path}_a{active}_t{tokens}`` / ``linear_ref_M{m}``; summarise with analyze_expert_bw.py.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, extract_mesh_config, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS
from models.demos.mimo_v2_d_p.tt.weight_cache import cache_dir

try:
    from tracy import signpost
except ImportError:  # pragma: no cover
    signpost = lambda *a, **k: None


def _env_list(name, default, cast=str):
    return [cast(v) for v in os.environ.get(name, default).split(",") if v]


PATHS = _env_list("MIMO_EXPERT_PATHS", "unified,fused,hybrid")
ACTIVE = _env_list("MIMO_EXPERT_ACTIVE", "64,32,16", int)
TOKENS = _env_list("MIMO_EXPERT_TOKENS", "32,64,128,256", int)
ITERS = int(os.environ.get("MIMO_EXPERT_ITERS", "3"))
HYBRID_T = int(os.environ.get("MIMO_RE_HYBRID_THRESHOLD", "64"))
LAYER = int(os.environ.get("MIMO_EXPERT_LAYER", "1"))
STATS_PATH = Path(os.environ.get("MIMO_EXPERT_STATS", "generated/mimo_expert_bw/cases.jsonl"))
BF4_BYTES = 0.5625  # 16 x 4-bit mantissa+sign + one shared 8-bit exponent per 16 values


def _log_case(stats):
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("a") as f:
        f.write(json.dumps(stats) + "\n")


@pytest.mark.timeout(1800)
@MESH_PARAMS
@pytest.mark.parametrize("tokens", TOKENS, ids=lambda t: f"t{t}")
@pytest.mark.parametrize("active", ACTIVE, ids=lambda a: f"a{a}")
@pytest.mark.parametrize("path", PATHS)
def test_expert_bw(mesh_device, device_params, path, active, tokens):
    cfg = MiMoTextConfig.from_json()
    E, H, I = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
    sp, tp = tuple(mesh_device.shape)
    n_dev = sp * tp
    epc = E // n_dev
    wdir = cache_dir(mesh_device)
    prefix = f"L{LAYER}.experts"
    if wdir is None:
        pytest.skip("set MIMO_TTNN_CACHE (expert weights are loaded cache-only)")
    from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker

    init_checker(wdir)
    if not TtRoutedExpert.check_cache_complete(wdir, prefix, epc, ttnn.bfloat4_b):
        pytest.skip(f"no complete bf4 {prefix} cache under {wdir}; run a decoder-layer test with MIMO_TTNN_CACHE first")

    mc = extract_mesh_config(mesh_device)
    gidx_host = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=epc, dispatch_group_size=mc.dispatch_group_size, num_dispatch_groups=mc.num_dispatch_groups
    )  # (group, chip, local) -> global id
    gidx = ttnn.from_torch(
        gidx_host,
        mesh_mapper=get_ep_mesh_mapper(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint32,
    )
    gidx = ttnn.squeeze(ttnn.squeeze(gidx, 0), 0)
    threshold = {"unified": None, "fused": tokens, "hybrid": HYBRID_T}[path]
    max_tok = tokens  # per-expert capacity; with the fused path it makes threshold >= max_tokens -> fused only
    expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=epc,
        global_expert_idx_table=gidx,
        emb_dim=H,
        hidden_dim=I,
        max_tokens=max_tok,
        torch_weights=None,
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        activation=ttnn.RoutedExpertActivation.Silu,
        weight_cache_path=wdir,
        cache_name_prefix=prefix,
        hybrid_token_threshold=threshold,
    )

    # Synthetic dispatch result, per chip: the first `active` local experts own `tokens` rows each, tile-aligned
    # regions packed in local order; every other expert has count 0.
    tok_pad = (tokens + 31) // 32 * 32
    buf_rows = epc * tok_pad
    counts = torch.zeros(sp, tp, E, dtype=torch.int32)
    regions = torch.zeros(sp, tp, E, dtype=torch.int32)
    g = gidx_host  # [col (dispatch group), row (chip in group), local expert]
    for r in range(sp):
        for c in range(tp):
            ids = g[c, r].long()
            counts[r, c, ids[:active]] = tokens
            regions[r, c, ids] = torch.arange(epc, dtype=torch.int32) * tok_pad
    shard = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=(0, 1))
    tt_counts = ttnn.from_torch(
        counts.reshape(sp, tp, 1, E),
        mesh_mapper=shard,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    tt_regions = ttnn.from_torch(
        regions.reshape(sp, tp, 1, E),
        mesh_mapper=shard,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    tt_counts = ttnn.reshape(tt_counts, (1, E))
    tt_regions = ttnn.reshape(tt_regions, (1, E))
    x = ttnn.from_torch(
        torch.randn(buf_rows, H) * 0.1,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tag = f"expert_{path}_a{active}_t{tokens}"
    w_bytes = active * 3 * H * I * BF4_BYTES
    _log_case(
        {
            "tag": tag,
            "path": path,
            "active": active,
            "tokens": tokens,
            "weight_bytes": w_bytes,
            "act_bytes": 2 * active * tokens * H * 1.0625,
            "flops": active * tokens * 6 * H * I,
        }
    )
    for it in range(1 + ITERS):
        ttnn.synchronize_device(mesh_device)
        if it:
            signpost(f"{tag}_start")
        y = expert(x, tt_counts, tt_regions)
        ttnn.synchronize_device(mesh_device)
        if it:
            signpost(f"{tag}_end")
        if y.buffer_address() != x.buffer_address():  # the unified op on a TILE buffer writes in place and returns x
            y.deallocate(True)
    logger.info(f"ran {tag}: {w_bytes / 1e6:.0f} MB of bf4 weights per chip")


@pytest.mark.timeout(1800)
@MESH_PARAMS
@pytest.mark.parametrize("m", _env_list("MIMO_LINEAR_REF_M", "32,128", int), ids=lambda m: f"M{m}")
def test_linear_reference(mesh_device, device_params, m):
    """A plain interleaved-DRAM ttnn.linear reading the same bf4 bytes as 64 experts' gate+up+down (~906 MB/chip)."""
    cfg = MiMoTextConfig.from_json()
    H, I = cfg.hidden_size, cfg.moe_intermediate_size
    n = 64 * 3 * I  # [4096, 393216] bf4 = the weight bytes of 64 experts
    w = ttnn.from_torch(
        torch.randn(1, 1, H, n) * 0.02,
        device=mesh_device,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x = ttnn.from_torch(
        torch.randn(1, 1, m, H),
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    tag = f"linear_ref_M{m}"
    _log_case(
        {
            "tag": tag,
            "path": "linear",
            "active": 64,
            "tokens": m,
            "weight_bytes": H * n * BF4_BYTES,
            "act_bytes": m * (H + n) * 1.0625,
            "flops": 2 * m * H * n,
        }
    )
    for it in range(1 + ITERS):
        ttnn.synchronize_device(mesh_device)
        if it:
            signpost(f"{tag}_start")
        y = ttnn.linear(x, w, dtype=ttnn.bfloat8_b, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(mesh_device)
        if it:
            signpost(f"{tag}_end")
        y.deallocate(True)
    logger.info(f"ran {tag}")


def _dram_sharded_weight(mesh_device, k, n, dtype):
    """[1, 1, k, n] weight WIDTH_SHARDED over the DRAM banks (one shard per bank, n padded to banks * 32)."""
    banks = mesh_device.dram_grid_size().x
    n_pad = -(-n // (32 * banks)) * 32 * banks
    spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
        (k, n_pad // banks),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)
    return ttnn.from_torch(
        torch.randn(1, 1, k, n) * 0.02,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=mem,
    )


def _dram_sharded_matmul(mesh_device, m, k, n, grid):
    """(in0 L1 width-sharded memcfg, out memcfg, program config) for a DRAM-sharded [m, k] x [k, n] matmul on grid."""
    cores = grid.x * grid.y
    in0_mem = ttnn.create_sharded_memory_config(
        (1, 1, m, k), core_grid=grid, strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    out_mem = ttnn.create_sharded_memory_config(
        (1, 1, m, n), core_grid=grid, strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    kt = k // 32 // cores
    in0_block_w = max(d for d in range(1, kt + 1) if kt % d == 0 and d <= 8)
    pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w, per_core_M=m // 32, per_core_N=-(-n // (32 * cores)), fused_activation=None
    )
    return in0_mem, out_mem, pc


@pytest.mark.timeout(1800)
@MESH_PARAMS
@pytest.mark.parametrize("cores", _env_list("MIMO_DSH_CORES", "8,16,32,64", int), ids=lambda c: f"c{c}")
@pytest.mark.parametrize("shape", ["big", "experts"])
def test_linear_dram_sharded_reference(mesh_device, device_params, shape, cores):
    """DRAM-sharded ttnn.linear (each core reads its own bank) over the same ~906 MB/chip of bf4 as 64 experts.
    big: one [4096, 393216] matmul (the streaming ceiling). experts: 64 x (gate+up [4096, 4096] + down
    [2048, 4096]) as separate calls, the expert-shaped version (includes per-op overhead)."""
    cfg = MiMoTextConfig.from_json()
    H, I = cfg.hidden_size, cfg.moe_intermediate_size
    m = 32
    grid = ttnn.CoreGrid(y=cores // 8, x=8)
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    if shape == "big":
        mats = [(H, 8 * 3 * I, 8)]
    else:
        mats = [(H, 2 * I, 64), (I, H, 64)]  # gate+up fused, then down; 64 experts' worth each
    plans = []
    for k, n, reps in mats:
        in0_mem, out_mem, pc = _dram_sharded_matmul(mesh_device, m, k, n, grid)
        w = _dram_sharded_weight(mesh_device, k, n, ttnn.bfloat4_b)
        x = ttnn.from_torch(
            torch.randn(1, 1, m, k),
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=in0_mem,
        )
        plans.append((x, w, pc, out_mem, reps))
    w_bytes = sum(k * n * reps for k, n, reps in mats) * BF4_BYTES
    tag = f"linear_dsh_{shape}_c{cores}"
    _log_case(
        {
            "tag": tag,
            "path": "linear_dram_sharded",
            "active": 64,
            "tokens": m,
            "weight_bytes": w_bytes,
            "act_bytes": 0,
            "flops": sum(2 * m * k * n * reps for k, n, reps in mats),
        }
    )
    for it in range(1 + ITERS):
        ttnn.synchronize_device(mesh_device)
        if it:
            signpost(f"{tag}_start")
        for x, w, pc, out_mem, reps in plans:
            for _ in range(reps):
                y = ttnn.linear(
                    x, w, program_config=pc, memory_config=out_mem, dtype=ttnn.bfloat8_b, compute_kernel_config=ckc
                )
                y.deallocate(True)
        ttnn.synchronize_device(mesh_device)
        if it:
            signpost(f"{tag}_end")
    logger.info(f"ran {tag}: {w_bytes / 1e6:.0f} MB")
