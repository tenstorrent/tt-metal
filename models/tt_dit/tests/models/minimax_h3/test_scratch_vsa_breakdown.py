# EXPLORATION (not for merge): one MiniMax-H3 transformer block, dense ring SDPA vs VSA (sparse_ring_sdpa), at the
# ref2va campaign's request layouts, with per-section timings. Written on WH 4x8 to find where VSA loses; meant to be
# rerun on BH 4x8 for comparison. Handoff, run list and WH reference numbers: VSA_BH_COMPARISON.md (same dir).
#
#   WH: TIME_VSA=1 pytest -s models/tt_dit/tests/models/minimax_h3/test_scratch_vsa_breakdown.py -k "vsa- and fsdp1 and uc4_10s"
#   BH: TIME_VSA=1 pytest -s models/tt_dit/tests/models/minimax_h3/test_scratch_vsa_breakdown.py -k "vsa- and 4x8sp1tp0nl2 and uc4_10s"
#   (the BH 4x8 fixture row is fsdp0 / 2 links, so "fsdp1" selects nothing there; use "dense-" / "vsa-" in -k:
#   plain "vsa" also matches the file name)
#
# Knobs (env):
#   SCRATCH_SPARSITY=0.9          VSA sparsity (0.0 keeps every block)
#   SCRATCH_K_CHUNK=256           sparse op k_chunk_size
#   SCRATCH_GRID_X / _Y           sparse op worker grid
#   SCRATCH_QBPP=n                sparse op q_blocks_per_pass
#   SCRATCH_DENSE_Q / _K          dense ring SDPA q / k chunk sizes (e.g. 64 / 256 = the sparse op's Q geometry)
#   SCRATCH_DENSE_OWN_RUNG=1      dense uses its own rung instead of VSA's padded rung
#   SCRATCH_EXTRA_RUNGS=a,b       extra rungs added to the ladder
#   MINIMAX_H3_VSA_INTERLEAVE=4   spread prefix (dense) tiles through each shard in 4-block groups (vsa_geometry.py)
#
# WH 4x8 reference (UC4 10s, rung 106496, attention op ms): dense q256/k512 150.5, dense q64/k256 403.4,
# sparse keep-all 204.4, sparse s0.9 + INTERLEAVE=4 112.8; passes (QBPP 32..7) no effect.
import os
import statistics
import time

import pytest
import torch
from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM, MiniMaxH3RotaryPosEmbed
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock as TorchMiniMaxH3Block
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.attention_minimax_h3 import prepare_rope_tables
from ....models.transformers.minimax_h3.transformer_block_minimax_h3 import MiniMaxH3TransformerBlock
from ....models.transformers.minimax_h3.vsa_prepass import MiniMaxH3VSAConfig, VSAPrepass
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.pipeline_minimax_h3 import select_bucket
from ....pipelines.minimax_h3.vsa_geometry import build_vsa_geometry, vsa_min_padded_len
from ....utils.tensor import bf16_tensor_2dshard, from_torch
from .common import GALAXY_RING, REAL_BLOCK_CONFIG, packed_layout
from .test_transformer_minimax_h3 import (
    ATTENTION_HEAD_DIM,
    HIDDEN_SIZE,
    ROPE_FREQ_DIM,
    ROPE_THETA,
    TIME_EMBED_DIM,
    TT_BLOCK_CONFIG,
    _packed_sizes,
    logical_length_tensor,
    upload_rope,
)

# The served ladder (server log "Initializing bucket buffers"), not the full test ladder.
MINIMAX_H3_BUCKET_LADDER_4X8 = [
    24064,
    28416,
    33536,
    39680,
    46848,
    55296,
    65280,
    76800,
    90368,
    106496,
    125440,
    147712,
    173824,
    204544,
    240640,
    283136,
    326656,
]

# (prefix text rows, condition rows, target seconds) from the campaign's "packed sequence" lines.
CASES = {
    "uc4_5s_54211": (3397, 13104, 5.0),
    "uc7_5s_81640": (6634, 37296, 5.0),
    "uc5_5s_96515": (11429, 47376, 5.0),
    "uc4_10s_89887": (3397, 13104, 10.0),
    "uc4_15s_125563": (3397, 13104, 15.0),
}


@GALAXY_RING
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("attention", ["dense", "vsa"])
def test_scratch_vsa_breakdown(
    mesh_device, sp_axis, tp_axis, num_links, is_fsdp, topology, case, attention, reset_seeds
):
    sp, tp = tuple(mesh_device.shape)[sp_axis], tuple(mesh_device.shape)[tp_axis]
    text, cond, dur = CASES[case]
    s = _packed_sizes(dur, 0)
    num_video, num_audio = s["num_video"], s["num_audio"]
    frame = s["grid_h"] * s["grid_w"]
    grid = (num_video // frame, s["grid_h"], s["grid_w"])
    seq_len = text + cond + num_audio + num_video
    segments = (text, cond, num_audio)
    ladder = sorted(
        set(MINIMAX_H3_BUCKET_LADDER_4X8) | {int(r) for r in os.environ.get("SCRATCH_EXTRA_RUNGS", "").split(",") if r}
    )
    rung = select_bucket(vsa_min_padded_len(segments, grid, sp), ladder)
    dense_rung = select_bucket(seq_len, MINIMAX_H3_BUCKET_LADDER_4X8)
    logger.info(
        f"{case}: seq {seq_len}, video {num_video} ({num_video / seq_len:.0%}), vsa rung {rung}, dense rung {dense_rung}"
    )

    if attention == "dense" and os.environ.get("SCRATCH_DENSE_OWN_RUNG"):
        rung = dense_rung
    position_ids, tags, timestep_indices = packed_layout(
        text + cond, num_audio, num_video, (s["grid_h"], s["grid_w"]), rung
    )
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + tags.clamp(min=0)
    geometry = None
    if attention == "vsa":
        geometry = build_vsa_geometry(segments, grid, sp_factor=sp, padded_len=rung)
        position_ids = geometry.permute_metadata(position_ids[:seq_len])
        adaln_indices = geometry.permute_metadata(adaln_indices[:seq_len])
        logger.info(f"VSA geometry: {geometry.n_real_tiles} tiles in {geometry.n_blocks} blocks")

    rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=ROPE_FREQ_DIM, rope_theta=ROPE_THETA)
    with torch.no_grad():
        rope_cos, rope_sin = rope(position_ids)
    rope_cos, rope_sin = prepare_rope_tables(rope_cos, rope_sin, ATTENTION_HEAD_DIM)

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    pc = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp),
        cfg_parallel=None,
    )
    block = MiniMaxH3TransformerBlock(
        **TT_BLOCK_CONFIG,
        rotary_dim=2 * 3 * ROPE_FREQ_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=pc,
        is_fsdp=is_fsdp,
        vsa_config=MiniMaxH3VSAConfig(sparsity=float(os.environ.get("SCRATCH_SPARSITY", "0.9")))
        if geometry is not None
        else None,
    )
    tb = TorchMiniMaxH3Block(**REAL_BLOCK_CONFIG).to(torch.float32)
    block.load_torch_state_dict(tb.state_dict())
    del tb

    k_override = int(os.environ.get("SCRATCH_K_CHUNK", "0"))
    gx_override = int(os.environ.get("SCRATCH_GRID_X", "0"))
    gy_override = int(os.environ.get("SCRATCH_GRID_Y", "0"))
    if k_override or gx_override or gy_override:
        base = block.attn._vsa_program_config()
        g = base.compute_with_storage_grid_size
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx_override or g.x, gy_override or g.y),
            q_chunk_size=base.q_chunk_size,
            k_chunk_size=k_override or base.k_chunk_size,
            exp_approx_mode=False,
        )
        block.attn._vsa_program_config = lambda: cfg
        logger.info(f"VSA program config override: grid {cfg.compute_with_storage_grid_size}, k {cfg.k_chunk_size}")

    dq, dk = int(os.environ.get("SCRATCH_DENSE_Q", "0")), int(os.environ.get("SCRATCH_DENSE_K", "0"))
    orig_cfg = block.attn._sdpa_program_config

    def dense_cfg(seq_local, *, ring, windowed=False):
        base = orig_cfg(seq_local, ring=ring, windowed=windowed)
        if not (dq or dk):
            logger.info(f"dense SDPA config: local {seq_local} q {base.q_chunk_size} k {base.k_chunk_size}")
            return base
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=base.compute_with_storage_grid_size,
            q_chunk_size=dq or base.q_chunk_size,
            k_chunk_size=dk or base.k_chunk_size,
            exp_approx_mode=False,
        )
        logger.info(f"dense SDPA config override: local {seq_local} q {cfg.q_chunk_size} k {cfg.k_chunk_size}")
        return cfg

    block.attn._sdpa_program_config = dense_cfg

    qbpp = int(os.environ.get("SCRATCH_QBPP", "0"))
    if qbpp:
        orig_sparse = ttnn.transformer.sparse_ring_sdpa
        monkeypatch_target = ttnn.transformer

        def sparse_with_qbpp(*a, **k):
            return orig_sparse(*a, q_blocks_per_pass=qbpp, **k)

        monkeypatch_target.sparse_ring_sdpa = sparse_with_qbpp
        logger.info(f"sparse_ring_sdpa q_blocks_per_pass={qbpp}")

    logical_n = seq_len
    if geometry is not None:
        prepass = VSAPrepass(
            rung=rung,
            sp_factor=sp,
            sparsity=float(os.environ.get("SCRATCH_SPARSITY", "0.9")),
            num_heads=REAL_BLOCK_CONFIG["num_attention_heads"] // tp,
            head_dim=ATTENTION_HEAD_DIM,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            sp_axis=sp_axis,
        )
        prepass.update(geometry, traced=False)
        block.attn.set_vsa_prepass(prepass)
        logical_n = geometry.logical_n

    x = bf16_tensor_2dshard(
        torch.randn(1, 1, rung, HIDDEN_SIZE), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3}
    )
    temb = from_torch(torch.randn(1, 1, 2, TIME_EMBED_DIM), device=mesh_device, dtype=ttnn.float32)
    adaln = from_torch(
        adaln_indices.to(torch.int32).reshape(1, 1, 1, rung),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.Layout.ROW_MAJOR,
        mesh_axes=[..., None, sp_axis],
    )
    rc, rs = upload_rope(rope_cos, rope_sin, mesh_device=mesh_device, sp_axis=sp_axis)
    ln = logical_length_tensor(mesh_device, logical_n)

    timings: dict[str, float] = {}
    orig = block.attn.forward
    block.attn.forward = lambda *a, **k: orig(*a, timings=timings, **k)

    def run(timed: bool):
        if not timed:
            block.attn.forward = orig
        out = block(x, ln, temb=temb, adaln_indices=adaln, rope_cos=rc, rope_sin=rs)
        ttnn.synchronize_device(mesh_device)
        block.attn.forward = lambda *a, **k: orig(*a, timings=timings, **k)
        return out

    ttnn.deallocate(run(False))  # compile
    blocks_ms, per = [], {}
    for _ in range(3):  # untimed block wall (no per-section syncs)
        t0 = time.perf_counter()
        ttnn.deallocate(run(False))
        blocks_ms.append((time.perf_counter() - t0) * 1e3)
    for _ in range(3):
        timings.clear()
        ttnn.deallocate(run(True))
        for k, v in timings.items():
            per.setdefault(k, []).append(v)
    lines = [
        f"RESULT {case} {attention} s{os.environ.get('SCRATCH_SPARSITY', '0.9')} k{os.environ.get('SCRATCH_K_CHUNK', 'def')} gx{os.environ.get('SCRATCH_GRID_X', 'def')} dq{dq or 'def'} dk{dk or 'def'} qbpp{qbpp or 'def'} rung {rung}: block {statistics.median(blocks_ms):.1f} ms (unsynced)"
    ]
    for k, v in per.items():
        lines.append(f"RESULT   {k:40s} {statistics.median(v):8.1f} ms")
    lines.append(f"RESULT   {'attention sections total':40s} {sum(statistics.median(v) for v in per.values()):8.1f} ms")
    logger.info("\n" + "\n".join(lines))
