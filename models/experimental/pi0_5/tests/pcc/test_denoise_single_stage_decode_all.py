# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""ONE stage of the 8-chip (NSUB=8) decode_all denoise pipeline, on a SINGLE Blackhole chip.

The pi0.5 denoise action-expert is 18 AdaRMS Gemma layers partitioned across NSUB submeshes
(one stage per chip). For NSUB=8 the layer split is ``(2, 2, 2, 3, 3, 2, 2, 2)`` (the
``_SPLITS`` table in ``test_perf_pi05_pipeline_replay_walltime.py``), so an interior stage holds
2-3 of the 18 layers. This test builds ONE such stage (default: the heaviest 3-layer stage,
index 3, covering layers 6,7,8) on ONE 1x1 submesh, runs its forward with
``PI05_WALLTIME_DECODE_ALL`` semantics (the 5 projection matmuls -- QKV/o/MLP gate/up/down --
routed through ``ttnn.matmul_decode`` partial-width-sharded instead of ``ttnn.linear``), asserts
PCC >= 0.99 vs the torch ``AdaRMSGemmaBlock`` reference for exactly that stage's layers, and
measures stage latency (eager + traced-replay).

A single stage has NO cross-chip hop -> zero D2D. Synthetic seeded weights (the reference
``AdaRMSGemmaBlock``), so no checkpoint is needed -- HARDWARE ONLY. The inputs mirror the
production single-layer config the tuned-matmul table targets:
prefix=1024, action_horizon=10 -> suffix_len 32 (m_tiles=1, which activates the tuned
``_DENOISE_TUNE_TABLE`` and the matmul_decode path), bf8_b prefix KV, all-zeros golden-parity mask.

NO TRACY: the always-on device profiler can HANG on repeated replay drains (see
tt_pipeline/PORT_NOTES.md no-tracy mandate). This test uses plain wall-clock timing only;
NEVER run it under ``python -m tracy``. (Single chip is the safe case, but the rule stands.)

Run:
  pytest models/experimental/pi0_5/tests/pcc/test_denoise_single_stage_decode_all.py -s
Parametrizable by stage index:
  PI05_STAGE_IDX (default 3 -> the heaviest 3-layer stage in the 8-way split)
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

# Reuse the reset/retry fabric harness (sibling tests/pcc).
sys.path.insert(0, str(Path(__file__).parent))
from _fabric_harness import close_parent as _close_parent  # noqa: E402
from _fabric_harness import open_parent_with_retry as _open_parent  # noqa: E402

SEED = 42
_ACTION_DIM = 32
_PREFIX_LEN = int(os.environ.get("PI05_WALLTIME_PREFIX", "1024"))
_ACTION_HORIZON = int(os.environ.get("PI05_WALLTIME_AH", "10"))
_N_REPLAYS = int(os.environ.get("PI05_STAGE_REPLAYS", "30"))
_N_WARMUP = int(os.environ.get("PI05_STAGE_WARMUP", "3"))
_PCC_THRESHOLD = float(os.environ.get("PI05_STAGE_PCC", "0.99"))
_DECODE_ALL = os.environ.get("PI05_WALLTIME_DECODE_ALL", "1").lower() in ("1", "true", "yes", "on")

# 8-way (2,2,2,3,3,2,2,2) -- sums to the 18 expert layers (matches the walltime _SPLITS[8]).
_NSUB = 8
_SPLITS_8 = (2, 2, 2, 3, 3, 2, 2, 2)
# Default to the heaviest (3-layer) stage. The first 3-layer stage is index 3 (layers 6,7,8).
_STAGE_IDX = int(os.environ.get("PI05_STAGE_IDX", "3"))


def _bounds(splits):
    bounds, acc = [], 0
    for sp in splits:
        bounds.append((acc, acc + sp))
        acc += sp
    return bounds


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


def _build_inputs(config, suffix_len, ah):
    """Synthetic seeded inputs identical in construction to the walltime perf test."""
    from models.experimental.pi0_5.common.configs import GemmaConfig as RefGemmaConfig
    from models.experimental.pi0_5.reference.torch_gemma import (
        AdaRMSGemmaBlock,
        apply_rotary_emb,
        precompute_freqs_cis,
    )

    ec = config.expert_config
    W, head_dim, num_kv_heads = ec.width, ec.head_dim, ec.num_kv_heads
    ref_ec = RefGemmaConfig.gemma_300m()

    torch.manual_seed(SEED)
    bw = [_expert_block_w(W, ec.mlp_dim, head_dim, ec.num_heads, num_kv_heads) for _ in range(18)]
    ref_blocks = [AdaRMSGemmaBlock(ref_ec, bw[i], i) for i in range(18)]

    # adarms_cond: timestep-conditioning vector [1, W]. Use a single Euler step t=1.0 equivalent
    # (a seeded random cond is sufficient -- the stage forward is conditioning-vector-agnostic).
    torch.manual_seed(SEED + 1)
    adarms_cond = torch.randn(1, W) * 0.1

    torch.manual_seed(SEED + 100)
    # Hidden state entering the stage (already-embedded suffix activations -- this is an INTERIOR
    # stage, so it consumes a hidden, not the raw action x_t).
    hidden = torch.randn(1, suffix_len, W) * 0.5
    hidden[:, ah:, :] = 0.0  # pad-suffix rows zeroed (golden-parity; final slice discards them)

    cos, sin = precompute_freqs_cis(head_dim, config.max_seq_len, base=ec.rope_base)
    pid_pre = torch.arange(_PREFIX_LEN).unsqueeze(0)
    mask = torch.zeros(1, 1, suffix_len, _PREFIX_LEN + suffix_len)  # all-zeros = golden-parity no-op

    torch.manual_seed(SEED + 200)
    prefix_kv = []
    for _ in range(18):
        k = torch.randn(1, num_kv_heads, _PREFIX_LEN, head_dim) * 0.1
        v = torch.randn(1, num_kv_heads, _PREFIX_LEN, head_dim) * 0.1
        k_roped, _ = apply_rotary_emb(k, k.clone(), cos, sin, position_ids=pid_pre)
        prefix_kv.append((k_roped, v))
    return ref_blocks, adarms_cond, hidden, prefix_kv, mask, cos, sin


def _torch_stage_oracle(ref_blocks, lo, hi, hidden, adarms_cond, prefix_kv, mask, cos, sin, suffix_len):
    """Run the torch AdaRMSGemmaBlock reference for layers [lo:hi] -- mirrors the device stage's
    use_concat_kv forward: hidden in, per-layer prefix KV as past_key_value, suffix queries roped
    at positions [prefix_len : prefix_len+suffix_len], all-zeros mask."""
    # Suffix queries occupy absolute positions prefix_len..prefix_len+suffix_len-1 (the device
    # slices RoPE cos/sin at position_offset=prefix_len; the prefix KV is roped at 0..prefix_len-1).
    pid_suf = torch.arange(_PREFIX_LEN, _PREFIX_LEN + suffix_len).unsqueeze(0)
    h = hidden
    with torch.no_grad():
        for i in range(lo, hi):
            h, _ = ref_blocks[i].forward(
                h,
                cos,
                sin,
                adarms_cond,
                attention_mask=mask,
                position_ids=pid_suf,
                past_key_value=prefix_kv[i],
                use_cache=False,
            )
    return h


def _compute_pcc(a, b):
    t1, t2 = a.flatten().float(), b.flatten().float()
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    cov = torch.mean((t1 - m1) * (t2 - m2))
    return (cov / (s1 * s2)).item()


def test_denoise_single_stage_decode_all():
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig, SuffixConfig
    from models.experimental.pi0_5.tt.tt_pipeline import denoise_block as _db
    from models.experimental.pi0_5.tt.tt_pipeline import perf_suffix_len
    from models.experimental.pi0_5.tt.tt_pipeline._d2d_pipeline import Pipeline
    from models.experimental.pi0_5.tt.tt_pipeline._device import set_device
    from models.experimental.pi0_5.tt.tt_pipeline.denoise_block import TTNNPi05DenoiseExpertBlock
    from models.experimental.pi0_5.tt.tt_pipeline.denoise_pipeline import (
        TTNNPi05DenoisePipelineStage,
        _to_dram,
    )

    # Route the 5 projection matmuls through ttnn.matmul_decode (partial-width-sharded).
    _db.DECODE_ALL = _DECODE_ALL

    # Instrument matmul_decode to PROVE whether the decode_all path fired on HW.
    _md_calls = {"n": 0}
    _orig_md = ttnn.matmul_decode

    def _counting_md(*a, **k):
        _md_calls["n"] += 1
        return _orig_md(*a, **k)

    ttnn.matmul_decode = _counting_md

    bounds = _bounds(_SPLITS_8)
    assert 0 <= _STAGE_IDX < _NSUB, f"stage idx {_STAGE_IDX} out of range [0,{_NSUB})"
    lo, hi = bounds[_STAGE_IDX]
    n_layers = hi - lo
    layer_idxs = list(range(lo, hi))

    ah = _ACTION_HORIZON
    suffix_len = perf_suffix_len(ah)  # 10 -> 32 (m_tiles=1; activates the tuned matmul_decode path)
    config = Pi0_5ModelConfig(action_horizon=ah, num_denoising_steps=5)
    ec = config.expert_config
    suffix_cfg = SuffixConfig(action_dim=_ACTION_DIM, action_horizon=ah, expert_width=ec.width, pi05=True)

    ref_blocks, adarms_cond, hidden, prefix_kv, mask, cos, sin = _build_inputs(config, suffix_len, ah)

    # Torch oracle for exactly this stage's layers.
    h_oracle = _torch_stage_oracle(ref_blocks, lo, hi, hidden, adarms_cond, prefix_kv, mask, cos, sin, suffix_len)

    from models.experimental.pi0_5.tt.tt_pipeline.mesh_carve import carve_n_submeshes

    # The (1,1) parent fails the FABRIC_1D handshake (bus error on open) -- the proven regime
    # opens the FULL (1,4) ring parent and carves submeshes from it (see pi05-pipeline-bringup
    # note: "parent mesh must be the full (1,4) ring; a partial parent fails fabric handshake").
    # We carve ONE 1x1 submesh and use only that single chip (zero D2D for one stage).
    _PARENT_N = ttnn.get_num_devices()
    parent = _open_parent(_PARENT_N)
    submesh = None
    drv_stage = None
    try:
        # Carve ONE 1x1 submesh (single chip) from the full-ring parent.
        submesh = carve_n_submeshes(parent, 1)[0]

        # Build ONE interior stage (is_first=False, is_last=False -> pure expert chain, no embed /
        # no final-norm-project), holding exactly this stage's layers, concat-KV path.
        tt_blocks = [TTNNPi05DenoiseExpertBlock.from_torch(ref_blocks[i], ec) for i in layer_idxs]
        stage = TTNNPi05DenoisePipelineStage(
            blocks=tt_blocks,
            suffix=None,
            is_first=False,
            is_last=False,
            expert_config=ec,
            max_seq_len=config.max_seq_len,
            rope_base=ec.rope_base,
            eps_expert=ec.rms_norm_eps,
            expert_width=ec.width,
            prefix_len=_PREFIX_LEN,
            suffix_len=suffix_len,
            position_offset=_PREFIX_LEN,
            action_horizon=ah,
            use_concat_kv=True,
        )
        drv_stage = stage
        set_device(stage, submesh)

        # Bind runtime state (mirrors denoise_pipeline._bind_stage_runtime for one stage):
        # bf8_b concat prefix KV (one (pk, pv) per local block), precomputed per-block mods, mask.
        stage._prefix_kv = []
        for j, gi in enumerate(layer_idxs):
            pk, pv = prefix_kv[gi]
            pk_dev = ttnn.from_torch(
                pk, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=submesh, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            pv_dev = ttnn.from_torch(
                pv, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=submesh, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            stage._prefix_kv.append((pk_dev, pv_dev))

        cond_dev = ttnn.from_torch(adarms_cond, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=submesh)
        stage._precomputed_block_mods = [_to_dram(blk.precompute_mods(cond_dev)) for blk in stage.blocks]
        ttnn.deallocate(cond_dev)
        stage._attention_mask = ttnn.from_torch(
            mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=submesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # --- forward (eager) + PCC ---
        h_dev_in = ttnn.from_torch(
            hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=submesh, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        _md_before = _md_calls["n"]
        out = stage.forward(h_dev_in)
        ttnn.synchronize_device(submesh)
        h_dev = ttnn.to_torch(out)
        _md_fired = _md_calls["n"] - _md_before

        pcc = _compute_pcc(h_oracle, h_dev)
        # Compare on the active (non-pad) suffix rows too (the rows that actually carry signal).
        pcc_active = _compute_pcc(h_oracle[:, :ah, :], h_dev[:, :ah, :])
        print(
            f"\n[single-stage decode_all={_DECODE_ALL}] stage_idx={_STAGE_IDX} layers={layer_idxs} "
            f"({n_layers} of 18) prefix={_PREFIX_LEN} suffix={suffix_len} m_tiles={suffix_len // 32}\n"
            f"  PCC(full)={pcc:.6f}  PCC(active rows [:{ah}])={pcc_active:.6f}\n"
            f"  matmul_decode calls in one forward = {_md_fired}  (decode_all fired = {_md_fired > 0})"
        )

        # --- perf: eager ---
        def _eager_once():
            hi_in = ttnn.from_torch(
                hidden,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=submesh,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            o = stage.forward(hi_in)
            ttnn.synchronize_device(submesh)
            ttnn.deallocate(o)

        for _ in range(_N_WARMUP):
            _eager_once()
        eager_times = []
        for _ in range(_N_REPLAYS):
            t0 = time.perf_counter()
            _eager_once()
            eager_times.append((time.perf_counter() - t0) * 1e3)
        eager_med = statistics.median(eager_times)

        # --- perf: traced replay (single chip -> clean, no multi-chip drain hang) ---
        traced_med = None
        traced_err = None
        try:
            x_persist = ttnn.from_torch(
                hidden,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=submesh,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            # Warm-up (kernel compile) before capture.
            _ = stage.forward(x_persist)
            ttnn.synchronize_device(submesh)
            tid = ttnn.begin_trace_capture(submesh, cq_id=0)
            out_t = stage.forward(x_persist)
            ttnn.end_trace_capture(submesh, tid, cq_id=0)
            ttnn.synchronize_device(submesh)
            for _ in range(_N_WARMUP):
                ttnn.execute_trace(submesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(submesh)
            traced_times = []
            for _ in range(_N_REPLAYS):
                t0 = time.perf_counter()
                ttnn.execute_trace(submesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(submesh)
                traced_times.append((time.perf_counter() - t0) * 1e3)
            traced_med = statistics.median(traced_times)
            ttnn.release_trace(submesh, tid)
            # Verify traced output still matches the oracle (numerics unchanged by tracing).
            pcc_traced = _compute_pcc(h_oracle, ttnn.to_torch(out_t))
            print(f"  traced-replay PCC(full)={pcc_traced:.6f}")
        except Exception as e:  # noqa: BLE001
            traced_err = repr(e)

        print(
            f"  latency  eager: min={min(eager_times):.3f} median={eager_med:.3f} ms"
            + (
                f"   traced: median={traced_med:.3f} ms"
                if traced_med is not None
                else f"   traced: SKIPPED ({traced_err})"
            )
            + f"   (over {_N_REPLAYS} reps)"
        )

        assert pcc >= _PCC_THRESHOLD, f"stage PCC {pcc:.6f} < {_PCC_THRESHOLD}"
        if _DECODE_ALL:
            assert _md_fired > 0, "decode_all set but matmul_decode never fired"
        else:
            assert _md_fired == 0, (
                f"decode_all=0 but matmul_decode fired {_md_fired}x -- expected the ttnn.linear "
                f"projection path (incl. concat_heads_matmul linear o-proj), not matmul_decode"
            )
    finally:
        ttnn.matmul_decode = _orig_md
        if submesh is not None:
            try:
                ttnn.close_mesh_device(submesh)
            except Exception:
                pass
        Pipeline.release_all()
        _close_parent(parent)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))
