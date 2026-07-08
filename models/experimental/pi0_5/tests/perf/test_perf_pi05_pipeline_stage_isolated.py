# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-stage wall-clock with NO D2D: each (5,5,4,4) denoise stage timed ALONE on one chip.

Answers "how long does each stage take e2e if no D2D exists?". Each of the four pipeline stages
is built standalone on a single (1,1) mesh -- L1-pinned TTNNPi05DenoiseExpertBlock, concat-KV,
per-step mods + phantom mask bound exactly as the streamed pipeline binds them -- with NO bridges
and NO SocketTransport. So forward() here is pure stage compute: embed (stage0 only), the stage's
N expert layers (attention over the 1024-token concat-KV prefix + MLP), and final-norm+project
(last stage only). Zero inter-stage hops, zero velocity-wrap, zero fabric.

Single-chip => NO fabric and NO multi-chip profiler-hang risk (unlike the streamed walltime test).

Reports, per stage, BOTH:
  - eager forward wall ms (includes per-op host dispatch), and
  - traced replay wall ms (host dispatch removed -- the apples-to-apples number vs the streamed
    e2e replay; CLAUDE.md device-time is tracy-only, so these are WALL-clock, not device-time).
The sum of the four traced per-stage times is the no-D2D e2e compute (the stages run serially at
batch=1); the streamed e2e replay minus this sum bounds the D2D + velocity-wrap + Euler overhead.

Synthetic weights -> needs NO checkpoint, only one Blackhole chip. ZERO tt_symbiote imports.

Env knobs:
  PI05_STAGE_AH      (default 50)   action_horizon (50 -> suffix_len 64 M=2; 10 -> 32 M=1, which
                                    activates the tuned matmul table)
  PI05_STAGE_PREFIX  (default 1024) VLM prefix length (concat-KV depth)
  PI05_STAGE_REPS    (default 20)   timed reps
  PI05_STAGE_WARMUP  (default 3)    warm-up reps (untimed)

Run:  pytest models/experimental/pi0_5/tests/perf/test_perf_pi05_pipeline_stage_isolated.py -s
"""
from __future__ import annotations

import os
import statistics
import time

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

SEED = 42
_ACTION_DIM = 32
_AH = int(os.environ.get("PI05_STAGE_AH", "10"))
_PREFIX_LEN = int(os.environ.get("PI05_STAGE_PREFIX", "1024"))
_REPS = int(os.environ.get("PI05_STAGE_REPS", "20"))
_WARMUP = int(os.environ.get("PI05_STAGE_WARMUP", "3"))
_TRACE_REGION = 134_217_728

# (label, n_layers, lo, is_first, is_last) -- the (5,5,4,4) pipeline, layer ranges over the 18.
_STAGES = [
    ("stage0 (5L, embed)", 5, 0, True, False),
    ("stage1 (5L, mid)", 5, 5, False, False),
    ("stage2 (4L, mid)", 4, 10, False, False),
    ("stage3 (4L, project)", 4, 14, False, True),
]


def _expert_block_w(W, mlp_dim, head_dim, num_heads, num_kv_heads):
    qkv_out, kv_out = num_heads * head_dim, num_kv_heads * head_dim
    return {
        "input_layernorm.dense.weight": torch.randn(3 * W, W) * 0.02,
        "input_layernorm.dense.bias": torch.randn(3 * W) * 0.02,
        "post_attention_layernorm.dense.weight": torch.randn(3 * W, W) * 0.02,
        "post_attention_layernorm.dense.bias": torch.randn(3 * W) * 0.02,
        "self_attn.q_proj.weight": torch.randn(qkv_out, W) * 0.02,
        "self_attn.k_proj.weight": torch.randn(kv_out, W) * 0.02,
        "self_attn.v_proj.weight": torch.randn(kv_out, W) * 0.02,
        "self_attn.o_proj.weight": torch.randn(W, qkv_out) * 0.02,
        "mlp.gate_proj.weight": torch.randn(mlp_dim, W) * 0.02,
        "mlp.up_proj.weight": torch.randn(mlp_dim, W) * 0.02,
        "mlp.down_proj.weight": torch.randn(W, mlp_dim) * 0.02,
    }


def _suffix_w(W, action_dim):
    return {
        "action_in_proj.weight": torch.randn(W, action_dim) * 0.02,
        "action_in_proj.bias": torch.randn(W) * 0.02,
        "action_out_proj.weight": torch.randn(action_dim, W) * 0.02,
        "action_out_proj.bias": torch.randn(action_dim) * 0.02,
        "time_mlp_in.weight": torch.randn(W, W) * 0.02,
        "time_mlp_in.bias": torch.randn(W) * 0.02,
        "time_mlp_out.weight": torch.randn(W, W) * 0.02,
        "time_mlp_out.bias": torch.randn(W) * 0.02,
    }


def _build_isolated_stage(
    mesh,
    ref_blocks,
    ref_suffix,
    final_w,
    final_b,
    cond_torch,
    prefix_kv,
    mask_torch,
    config,
    suffix_cfg,
    *,
    is_first,
    is_last,
    position_offset,
    prefix_len,
    suffix_len,
):
    """Build ONE stage on `mesh` with concat-KV + mods + mask bound -- no bridges, no transport."""
    from models.experimental.pi0_5.tt.tt_pipeline._device import set_device
    from models.experimental.pi0_5.tt.tt_pipeline.denoise_block import TTNNPi05DenoiseExpertBlock
    from models.experimental.pi0_5.tt.tt_pipeline.denoise_pipeline import (
        TTNNPi05DenoisePipelineStage,
        _kv_dtype,
        _to_dram,
    )
    from models.experimental.pi0_5.tt.tt_pipeline.modeling.suffix import TTNNPi05SuffixEmbedding

    _L1 = ttnn.L1_MEMORY_CONFIG
    _DRAM = ttnn.DRAM_MEMORY_CONFIG
    ec = config.expert_config

    tt_blocks = [TTNNPi05DenoiseExpertBlock.from_torch(b, ec) for b in ref_blocks]
    suffix = TTNNPi05SuffixEmbedding.from_torch(ref_suffix, suffix_cfg) if (is_first or is_last) else None
    stage = TTNNPi05DenoisePipelineStage(
        blocks=tt_blocks,
        suffix=suffix,
        is_first=is_first,
        is_last=is_last,
        expert_config=ec,
        max_seq_len=config.max_seq_len,
        rope_base=ec.rope_base,
        eps_expert=ec.rms_norm_eps,
        expert_width=ec.width,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        position_offset=position_offset,
        action_horizon=suffix_cfg.action_horizon,
        use_concat_kv=True,
    )
    if is_last:
        stage._raw_final_norm_mod_w = final_w
        stage._raw_final_norm_mod_b = final_b
    set_device(stage, mesh)  # preprocess + move_weights (RoPE table, final-mod for last stage)

    kvd = _kv_dtype()
    stage._prefix_kv = []
    for j in range(len(tt_blocks)):
        pk, pv = prefix_kv[j]
        pk_dev = ttnn.from_torch(pk, dtype=kvd, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=_L1)
        pv_dev = ttnn.from_torch(pv, dtype=kvd, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=_L1)
        stage._prefix_kv.append((pk_dev, pv_dev))

    cond_dev = ttnn.from_torch(cond_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
    stage._precomputed_block_mods = [_to_dram(blk.precompute_mods(cond_dev)) for blk in stage.blocks]
    if is_last and stage._tt_final_mod_w is not None:
        stage._precomputed_final_mod = stage._precompute_final_mod(cond_dev)
    ttnn.deallocate(cond_dev)
    stage._attention_mask = ttnn.from_torch(
        mask_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=_DRAM
    )
    return stage


def _time_eager(stage, inp, mesh):
    for _ in range(_WARMUP):
        out = stage.forward(inp)
        ttnn.deallocate(out)
    ttnn.synchronize_device(mesh)
    times = []
    for _ in range(_REPS):
        t0 = time.perf_counter()
        out = stage.forward(inp)
        ttnn.synchronize_device(mesh)
        times.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(out)
    return times


def _time_traced(stage, inp, mesh):
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = stage.forward(inp)  # captured once; replays write into this output buffer
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(_WARMUP):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    times = []
    for _ in range(_REPS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        times.append((time.perf_counter() - t0) * 1e3)
    ttnn.release_trace(mesh, tid)
    _ = out
    return times


def test_isolated_stage_walltime():
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig, SuffixConfig
    from models.experimental.pi0_5.reference.torch_gemma import AdaRMSGemmaBlock, apply_rotary_emb, precompute_freqs_cis
    from models.experimental.pi0_5.reference.torch_suffix import Pi0_5SuffixEmbedding
    from models.experimental.pi0_5.tt.tt_pipeline import euler_schedule, perf_suffix_len

    if ttnn.get_num_devices() < 1:
        pytest.skip("need >=1 chip")

    ah = _AH
    suffix_len = perf_suffix_len(ah)
    config = Pi0_5ModelConfig(action_horizon=ah, num_denoising_steps=5)
    ec = config.expert_config
    W, head_dim, n_kv = ec.width, ec.head_dim, ec.num_kv_heads
    suffix_cfg = SuffixConfig(action_dim=_ACTION_DIM, action_horizon=ah, expert_width=W, pi05=True)

    torch.manual_seed(SEED)
    bw = [_expert_block_w(W, ec.mlp_dim, head_dim, ec.num_heads, n_kv) for _ in range(18)]
    ref_blocks = [AdaRMSGemmaBlock(ec, bw[i], i) for i in range(18)]
    ref_suffix = Pi0_5SuffixEmbedding(suffix_cfg, _suffix_w(W, _ACTION_DIM))
    final_w = torch.randn(3 * W, W) * 0.02
    final_b = torch.randn(3 * W) * 0.02
    timesteps, _ = euler_schedule(5)
    cond_torch = ref_suffix.embed_timestep_adarms(torch.tensor([timesteps[0]]))

    cos, sin = precompute_freqs_cis(head_dim, config.max_seq_len, base=ec.rope_base)
    pid_pre = torch.arange(_PREFIX_LEN).unsqueeze(0)
    mask = torch.zeros(1, 1, suffix_len, _PREFIX_LEN + suffix_len)  # zeros = golden parity (iter-3 fix)
    torch.manual_seed(SEED + 200)
    prefix_kv = []
    for _ in range(18):
        k = torch.randn(1, n_kv, _PREFIX_LEN, head_dim) * 0.1
        v = torch.randn(1, n_kv, _PREFIX_LEN, head_dim) * 0.1
        k_roped, _ = apply_rotary_emb(k, k.clone(), cos, sin, position_ids=pid_pre)
        prefix_kv.append((k_roped, v))

    results = []
    for label, nL, lo, is_first, is_last in _STAGES:
        mesh = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=_TRACE_REGION
        )
        try:
            stage = _build_isolated_stage(
                mesh,
                ref_blocks[lo : lo + nL],
                ref_suffix,
                final_w,
                final_b,
                cond_torch,
                prefix_kv[lo : lo + nL],
                mask,
                config,
                suffix_cfg,
                is_first=is_first,
                is_last=is_last,
                position_offset=_PREFIX_LEN,
                prefix_len=_PREFIX_LEN,
                suffix_len=suffix_len,
            )
            if is_first:
                inp = ttnn.from_torch(
                    torch.randn(1, suffix_len, _ACTION_DIM),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            else:
                inp = ttnn.from_torch(
                    torch.randn(1, suffix_len, W) * 0.5,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            eager = _time_eager(stage, inp, mesh)
            traced = _time_traced(stage, inp, mesh)
            results.append((label, nL, statistics.median(eager), statistics.median(traced)))
        finally:
            ttnn.close_mesh_device(mesh)

    M = suffix_len // 32
    print(
        f"\n[isolated-stage walltime  prefix={_PREFIX_LEN} ah={ah} suffix_len={suffix_len} M={M}  no D2D, single chip]"
    )
    print(f"{'stage':<24}{'layers':>7}{'eager ms':>12}{'traced ms':>12}{'traced/layer':>14}")
    tot_e = tot_t = 0.0
    for label, nL, e, t in results:
        print(f"{label:<24}{nL:>7}{e:>12.3f}{t:>12.3f}{t / nL:>14.3f}")
        tot_e += e
        tot_t += t
    print(f"{'SUM (no-D2D e2e/step)':<24}{18:>7}{tot_e:>12.3f}{tot_t:>12.3f}")
    print(
        "Note: WALL-clock (not tracy device-time). 'traced' removes host dispatch and is the "
        "apples-to-apples figure vs the streamed e2e replay; (streamed e2e/step - traced SUM) "
        "bounds the D2D hops + velocity-wrap + Euler-integrate overhead."
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-xvs"]))
