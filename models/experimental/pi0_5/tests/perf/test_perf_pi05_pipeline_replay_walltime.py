# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Replay wall-clock latency for the standalone tt_pipeline fused N-step denoise loop trace.

Standalone port of tt_symbiote's
``tests/experimental/pipelined_pi05/Tier4/test_replay_walltime_pipelined_pi05.py``.
ZERO tt_symbiote imports -- drives the in-repo ``tt/tt_pipeline`` driver only.

Production config: 1024-token VLM prefix, 5-step Euler denoise, 30 timed replays, 4-way split
(5,5,4,4). Synthetic weights (reference ``AdaRMSGemmaBlock`` / ``Pi0_5SuffixEmbedding``), so the
test needs NO checkpoint -- only hardware. Weights-on-L1 + the tuned matmul/SDPA configs are the
``denoise_block.py`` defaults.

``replay()`` wall time = host dispatch + on-device exec + a single synchronize + action readback.

Config is env-overridable:
  PI05_WALLTIME_PREFIX   (default 1024)     VLM prefix length
  PI05_WALLTIME_AH       (default 10)       action_horizon (10 -> suffix_len 32, m_tiles 1, which
                                            ACTIVATES the tuned matmul table; 50 -> 64, m_tiles 2)
  PI05_WALLTIME_STEPS    (default 5)        Euler steps
  PI05_WALLTIME_REPLAYS  (default 30)       timed replays
  PI05_WALLTIME_WARMUP   (default 3)        warm-up replays (untimed)
  PI05_WALLTIME_NSUB     (default 4)        submeshes (4 -> (5,5,4,4); skips if > #chips)
  PI05_WALLTIME_CEILING  (default 40.0)     median-latency regression ceiling (ms)
  PI05_WALLTIME_DRAIN    (default "stage0") drain mode ("stage0" | "all"). NOTE: the timed
                                            driver.replay() ALWAYS single-drains stage0 (the SC5
                                            win -- the velocity-wrap lands the final x_t on stage0,
                                            so syncing stage0 transitively gates the pipeline);
                                            this knob only sets the capture-time drain in
                                            stream_euler, kept consistent at "stage0".

NOTE (this build): the always-on real-time device profiler can emit "Skipping zone with end <
start" and HANG on REPEATED multi-chip replay drains (see tt_pipeline/PORT_NOTES.md "no-tracy
mandate"). If this test hangs on the box, lower PI05_WALLTIME_REPLAYS and `tt-smi -r`; that hang
is the documented environmental profiler limitation, not a port defect. NEVER run this under
``python -m tracy``.

Run:  pytest models/experimental/pi0_5/tests/perf/test_perf_pi05_pipeline_replay_walltime.py -s
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

# Reuse the reset/retry fabric harness (sibling tests/pcc) for stability across many replays.
sys.path.insert(0, str(Path(__file__).parent.parent / "pcc"))
from _fabric_harness import close_parent as _close_parent  # noqa: E402
from _fabric_harness import open_parent_with_retry as _open_parent  # noqa: E402

SEED = 42
_ACTION_DIM = 32
_PREFIX_LEN = int(os.environ.get("PI05_WALLTIME_PREFIX", "1024"))
_ACTION_HORIZON = int(os.environ.get("PI05_WALLTIME_AH", "10"))
_N_STEPS = int(os.environ.get("PI05_WALLTIME_STEPS", "5"))
_N_REPLAYS = int(os.environ.get("PI05_WALLTIME_REPLAYS", "30"))
_N_WARMUP = int(os.environ.get("PI05_WALLTIME_WARMUP", "3"))
_N_SUBMESHES = int(os.environ.get("PI05_WALLTIME_NSUB", "4"))
_LATENCY_CEILING_MS = float(os.environ.get("PI05_WALLTIME_CEILING", "40.0"))
_DRAIN = os.environ.get("PI05_WALLTIME_DRAIN", "stage0")
# PI05_WALLTIME_DECODE_ALL=1 routes the denoise block's 5 projection matmuls (QKV, o, MLP
# gate/up/down) through ttnn.matmul_decode (partial-width-sharded) instead of ttnn.linear.
_DECODE_ALL = os.environ.get("PI05_WALLTIME_DECODE_ALL", "0").lower() in ("1", "true", "yes", "on")

# 4-way (5,5,4,4); 8-way (2,2,2,3,3,2,2,2). Both sum to the 18 expert layers.
_SPLITS = {4: (5, 5, 4, 4), 8: (2, 2, 2, 3, 3, 2, 2, 2)}


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


def _build_inputs(config, suffix_cfg, ah, suffix_len, n_steps):
    from models.experimental.pi0_5.common.configs import GemmaConfig as RefGemmaConfig
    from models.experimental.pi0_5.reference.torch_gemma import (
        AdaRMSGemmaBlock,
        apply_rotary_emb,
        precompute_freqs_cis,
    )
    from models.experimental.pi0_5.reference.torch_suffix import Pi0_5SuffixEmbedding
    from models.experimental.pi0_5.tt.tt_pipeline import euler_schedule

    ec = config.expert_config
    W, head_dim, num_kv_heads = ec.width, ec.head_dim, ec.num_kv_heads
    ref_ec = RefGemmaConfig.gemma_300m()

    torch.manual_seed(SEED)
    bw = [_expert_block_w(W, ec.mlp_dim, head_dim, ec.num_heads, num_kv_heads) for _ in range(18)]
    ref_blocks = [AdaRMSGemmaBlock(ref_ec, bw[i], i) for i in range(18)]
    ref_suffix = Pi0_5SuffixEmbedding(suffix_cfg, _suffix_w(W, _ACTION_DIM))
    final_mod_w = torch.randn(3 * W, W) * 0.02
    final_mod_b = torch.randn(3 * W) * 0.02

    torch.manual_seed(SEED + 100)
    x_t = torch.randn(1, suffix_len, _ACTION_DIM) * 0.5
    x_t[:, ah:, :] = 0.0
    timesteps, _ = euler_schedule(n_steps)
    conds = [ref_suffix.embed_timestep_adarms(torch.tensor([timesteps[i]])) for i in range(n_steps)]

    cos, sin = precompute_freqs_cis(head_dim, config.max_seq_len, base=ec.rope_base)
    pid_pre = torch.arange(_PREFIX_LEN).unsqueeze(0)
    # All-zeros additive mask: golden parity. The iter-2 `-1e4` phantom band on pad-suffix KEY
    # positions was the multi-stage PCC bug (PORT_NOTES "iter-3 ROOT CAUSE"); zeros is correct and
    # also leaves replay wall-time unchanged.
    mask = torch.zeros(1, 1, suffix_len, _PREFIX_LEN + suffix_len)

    torch.manual_seed(SEED + 200)
    prefix_kv = []
    for _ in range(18):
        k = torch.randn(1, num_kv_heads, _PREFIX_LEN, head_dim) * 0.1
        v = torch.randn(1, num_kv_heads, _PREFIX_LEN, head_dim) * 0.1
        k_roped, _ = apply_rotary_emb(k, k.clone(), cos, sin, position_ids=pid_pre)
        prefix_kv.append((k_roped, v))
    return ref_blocks, final_mod_w, final_mod_b, ref_suffix, x_t, conds, prefix_kv, mask


def test_replay_walltime():
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig, SuffixConfig
    from models.experimental.pi0_5.tt.tt_pipeline import (
        TTNNPi05DenoiseExpertBlock,
        build_denoise_loop_pipeline,
        perf_suffix_len,
    )
    from models.experimental.pi0_5.tt.tt_pipeline._d2d_pipeline import Pipeline
    from models.experimental.pi0_5.tt.tt_pipeline import denoise_block as _db

    _db.DECODE_ALL = _DECODE_ALL  # route projection matmuls through matmul_decode when set

    if _N_SUBMESHES not in _SPLITS:
        pytest.skip(f"no layer split defined for n_submeshes={_N_SUBMESHES} (have {sorted(_SPLITS)})")

    ah = _ACTION_HORIZON
    suffix_len = perf_suffix_len(ah)
    config = Pi0_5ModelConfig(action_horizon=ah, num_denoising_steps=_N_STEPS)
    ec = config.expert_config
    suffix_cfg = SuffixConfig(action_dim=_ACTION_DIM, action_horizon=ah, expert_width=ec.width, pi05=True)
    ref_blocks, fmw, fmb, ref_suffix, x_t, conds, prefix_kv, mask = _build_inputs(
        config, suffix_cfg, ah, suffix_len, _N_STEPS
    )

    parent = _open_parent(_N_SUBMESHES)
    drv = None
    try:
        drv = build_denoise_loop_pipeline(
            ref_blocks,
            fmw,
            fmb,
            ref_suffix,
            config,
            suffix_cfg,
            parent,
            adarms_cond_per_step=conds,
            prefix_kv_cache=prefix_kv,
            prefix_len=_PREFIX_LEN,
            suffix_len=suffix_len,
            attention_mask_torch=mask,
            position_offset=_PREFIX_LEN,
            num_steps=_N_STEPS,
            action_horizon=ah,
            splits=_SPLITS[_N_SUBMESHES],
            block_cls=TTNNPi05DenoiseExpertBlock,
            use_concat_kv=True,
            drain=_DRAIN,
        )
        drv.stream_euler(x_t, capture=True)  # capture the fused per-submesh loop trace
        for _ in range(_N_WARMUP):
            drv.replay()  # warm-up / stabilize

        times = []
        for _ in range(_N_REPLAYS):
            t0 = time.perf_counter()
            drv.replay()  # replay_loop + single synchronize + action readback
            times.append((time.perf_counter() - t0) * 1e3)
        md = statistics.median(times)
        print(
            f"\n[replay-walltime n_sub={_N_SUBMESHES} prefix={_PREFIX_LEN} N={_N_STEPS} "
            f"M={suffix_len // 32} drain={_DRAIN} decode_all={_DECODE_ALL}] "
            f"min={min(times):.3f} median={md:.3f} mean={statistics.mean(times):.3f} ms "
            f"({md / _N_STEPS:.3f} ms/step) over {_N_REPLAYS} reps"
        )
        assert md < _LATENCY_CEILING_MS, f"{_N_STEPS}-step replay {md:.2f} ms exceeds {_LATENCY_CEILING_MS} ms ceiling"
    finally:
        if drv is not None:
            drv.close()
        Pipeline.release_all()
        _close_parent(parent)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))
