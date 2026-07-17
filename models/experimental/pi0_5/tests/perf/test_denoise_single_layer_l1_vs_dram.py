# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single AdaRMS expert layer (idx 6): L1 decode_all vs production-tuned DRAM ExpertChunkSlice — PCC + traced-replay wall-clock."""
from __future__ import annotations

import os
import re
import statistics


def _apply_production_env_defaults():
    """setdefault the validated production tuning flags (read at import/__init__ time) so the
    DRAM path runs its most-tuned config. Shell exports still win; PI05_NO_PROD_ENV=1 skips."""
    if os.environ.get("PI05_NO_PROD_ENV", "").lower() in ("1", "true", "yes", "on"):
        return
    root = os.environ.get("TT_METAL_HOME") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), *([os.pardir] * 4))
    )
    envf = os.path.join(root, "_bench_runs", "pi05_production.env")
    if not os.path.exists(envf):
        return
    with open(envf) as f:
        for line in f:
            m = re.match(r"\s*export\s+([A-Z0-9_]+)=(\S+)", line)
            if m and m.group(1) != "PI05_CHECKPOINT_DIR":
                os.environ.setdefault(m.group(1), m.group(2))


_apply_production_env_defaults()

import pytest  # noqa: E402
import torch  # noqa: E402
from models.experimental.pi0_5.tt.tt_pipeline.denoise_pipeline import perf_suffix_len
from models.experimental.pi0_5.tt.tile_config import TILE_HEIGHT, from_torch_pi05

ttnn = pytest.importorskip("ttnn")

SEED = 42
_LO, _HI = 6, 7
_PREFIX_LEN = 1024
_ACTION_HORIZON = 10
_PCC = 0.99
_DEVICE_PARAMS = {"trace_region_size": 134_217_728, "l1_small_size": 24576}


def _compute_pcc(a, b):
    t1, t2 = a.flatten().float(), b.flatten().float()
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    cov = torch.mean((t1 - torch.mean(t1)) * (t2 - torch.mean(t2)))
    return (cov / (s1 * s2)).item()


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
    from models.experimental.pi0_5.common.configs import GemmaConfig as RefGemmaConfig
    from models.experimental.pi0_5.reference.torch_gemma import (
        AdaRMSGemmaBlock,
        apply_rotary_emb,
        precompute_freqs_cis,
    )

    ec = config.expert_config
    W, head_dim, num_kv_heads = ec.width, ec.head_dim, ec.num_kv_heads
    torch.manual_seed(SEED)
    bw = [_expert_block_w(W, ec.mlp_dim, head_dim, ec.num_heads, num_kv_heads) for _ in range(_HI)]
    ref_blocks = [AdaRMSGemmaBlock(RefGemmaConfig.gemma_300m(), bw[i], i) for i in range(_HI)]
    torch.manual_seed(SEED + 1)
    adarms_cond = torch.randn(1, W) * 0.1
    torch.manual_seed(SEED + 100)
    hidden = torch.randn(1, suffix_len, W) * 0.5
    hidden[:, ah:, :] = 0.0
    cos, sin = precompute_freqs_cis(head_dim, config.max_seq_len, base=ec.rope_base)
    pid_pre = torch.arange(_PREFIX_LEN).unsqueeze(0)
    mask = torch.zeros(1, 1, suffix_len, _PREFIX_LEN + suffix_len)
    torch.manual_seed(SEED + 200)
    prefix_kv = []
    for _ in range(_HI):
        k = torch.randn(1, num_kv_heads, _PREFIX_LEN, head_dim) * 0.1
        v = torch.randn(1, num_kv_heads, _PREFIX_LEN, head_dim) * 0.1
        k_roped, _ = apply_rotary_emb(k, k.clone(), cos, sin, position_ids=pid_pre)
        prefix_kv.append((k_roped, v))
    return bw, ref_blocks, adarms_cond, hidden, prefix_kv, mask, cos, sin


def _torch_oracle(ref_blocks, hidden, adarms_cond, prefix_kv, mask, cos, sin, suffix_len):
    pid_suf = torch.arange(_PREFIX_LEN, _PREFIX_LEN + suffix_len).unsqueeze(0)
    h = hidden
    with torch.no_grad():
        for i in range(_LO, _HI):
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


def _chunk_weights(per_layer_w):
    out = {}
    for i in range(_LO, _HI):
        for k, v in per_layer_w[i].items():
            out[f"model.layers.{i}." + k] = v.clone()
    return out


def _build_l1(submesh, ref_blocks, ec, config, suffix_len, ah, adarms_cond, prefix_kv, mask):
    import models.experimental.pi0_5.tt.tt_pipeline.denoise_block as _db
    from models.experimental.pi0_5.tt.tt_pipeline._device import set_device
    from models.experimental.pi0_5.tt.tt_pipeline.denoise_block import TTNNPi05DenoiseExpertBlock
    from models.experimental.pi0_5.tt.tt_pipeline.denoise_pipeline import TTNNPi05DenoisePipelineStage, _to_dram

    _db.DECODE_ALL = True
    tt_blocks = [TTNNPi05DenoiseExpertBlock.from_torch(ref_blocks[i], ec) for i in range(_LO, _HI)]
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
    set_device(stage, submesh)
    dev = []
    stage._prefix_kv = []
    for gi in range(_LO, _HI):
        pk, pv = prefix_kv[gi]
        pk_dev = from_torch_pi05(pk, dtype=ttnn.bfloat8_b, device=submesh, memory_config=ttnn.L1_MEMORY_CONFIG)
        pv_dev = from_torch_pi05(pv, dtype=ttnn.bfloat8_b, device=submesh, memory_config=ttnn.L1_MEMORY_CONFIG)
        stage._prefix_kv.append((pk_dev, pv_dev))
        dev += [pk_dev, pv_dev]
    cond_dev = from_torch_pi05(adarms_cond, dtype=ttnn.bfloat16, device=submesh)
    stage._precomputed_block_mods = [_to_dram(blk.precompute_mods(cond_dev)) for blk in stage.blocks]
    ttnn.deallocate(cond_dev)
    stage._attention_mask = from_torch_pi05(
        mask, dtype=ttnn.bfloat16, device=submesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    dev.append(stage._attention_mask)
    return dev, lambda x: stage.forward(x)


def _precomputed_mods(submesh, expert_config, chunk_weights, adarms_cond):
    W = expert_config.width
    cond = adarms_cond.to(torch.bfloat16)

    def _up(t2d):
        return from_torch_pi05(
            t2d.unsqueeze(1).contiguous(),
            dtype=ttnn.bfloat16,
            device=submesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    out = []
    for i in range(_LO, _HI):
        p = f"model.layers.{i}."
        fused_w = (
            torch.cat(
                [
                    chunk_weights[f"{p}input_layernorm.dense.weight"],
                    chunk_weights[f"{p}post_attention_layernorm.dense.weight"],
                ],
                dim=0,
            )
            .contiguous()
            .to(torch.bfloat16)
        )
        fused_b = (
            torch.cat(
                [
                    chunk_weights[f"{p}input_layernorm.dense.bias"],
                    chunk_weights[f"{p}post_attention_layernorm.dense.bias"],
                ],
                dim=0,
            )
            .contiguous()
            .to(torch.bfloat16)
        )
        mod = torch.nn.functional.linear(cond, fused_w, fused_b)
        s = [mod[:, j * W : (j + 1) * W] for j in range(6)]
        out.append((_up(s[0] + 1.0), _up(s[1]), _up(s[2]), _up(s[3] + 1.0), _up(s[4]), _up(s[5])))
    return out


def _build_dram(submesh, expert_config, chunk_weights, config, suffix_len, ah, adarms_cond, prefix_kv, mask):
    from models.experimental.pi0_5.tt.tt_bh_glx.expert_slice import ExpertChunkSlice
    from models.experimental.pi0_5.tt.tt_pipeline.denoise_pipeline import _slice_rope

    slice_ = ExpertChunkSlice(
        expert_config=expert_config,
        expert_weights=chunk_weights,
        submesh=submesh,
        layer_range=(_LO, _HI),
        max_seq_len=config.max_seq_len,
    )
    cos_suf, sin_suf = _slice_rope(slice_.cos_meta, slice_.sin_meta, suffix_len, _PREFIX_LEN)
    cond_dev = from_torch_pi05(adarms_cond, dtype=ttnn.bfloat16, device=submesh)
    block_mods = _precomputed_mods(submesh, expert_config, chunk_weights, adarms_cond)
    dev = [cond_dev]
    for tup in block_mods:
        dev += list(tup)
    prefix_kv_for_chunk = []
    for gi in range(_LO, _HI):
        pk, pv = prefix_kv[gi]
        pk_dev = from_torch_pi05(pk, dtype=ttnn.bfloat8_b, device=submesh, memory_config=ttnn.L1_MEMORY_CONFIG)
        pv_dev = from_torch_pi05(pv, dtype=ttnn.bfloat8_b, device=submesh, memory_config=ttnn.L1_MEMORY_CONFIG)
        prefix_kv_for_chunk.append((pk_dev, pv_dev))
        dev += [pk_dev, pv_dev]
    attn_mask_dev = from_torch_pi05(mask, dtype=ttnn.bfloat16, device=submesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    dev.append(attn_mask_dev)

    def _fwd(x):
        return slice_.forward(
            x,
            cond_dev,
            prefix_kv_for_chunk,
            attention_mask=attn_mask_dev,
            position_ids=None,
            cos_override=cos_suf,
            sin_override=sin_suf,
            precomputed_mods=block_mods,
        )

    return dev, _fwd


def _time_replay(submesh, forward_fn, x):
    import time

    _ = forward_fn(x)
    ttnn.synchronize_device(submesh)
    tid = ttnn.begin_trace_capture(submesh, cq_id=0)
    forward_fn(x)
    ttnn.end_trace_capture(submesh, tid, cq_id=0)
    ttnn.synchronize_device(submesh)
    try:
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            ttnn.execute_trace(submesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(submesh)
            times.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(times)
    finally:
        try:
            ttnn.release_trace(submesh, tid)
        except Exception:
            pass


def _setup(mesh):
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig

    ah = _ACTION_HORIZON
    suffix_len = perf_suffix_len(ah, TILE_HEIGHT)
    config = Pi0_5ModelConfig(action_horizon=ah, num_denoising_steps=5)
    bw, ref_blocks, adarms_cond, hidden, prefix_kv, mask, cos, sin = _build_inputs(config, suffix_len, ah)
    h_oracle = _torch_oracle(ref_blocks, hidden, adarms_cond, prefix_kv, mask, cos, sin, suffix_len)
    return config, ah, suffix_len, bw, ref_blocks, adarms_cond, hidden, prefix_kv, mask, h_oracle


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_dram_single_layer_pcc(mesh_device):
    from models.experimental.pi0_5.common.configs import GemmaConfig

    config, ah, suffix_len, bw, ref_blocks, adarms_cond, hidden, prefix_kv, mask, h_oracle = _setup(mesh_device)
    dev, dram_fwd = _build_dram(
        mesh_device, GemmaConfig.gemma_300m(), _chunk_weights(bw), config, suffix_len, ah, adarms_cond, prefix_kv, mask
    )
    try:
        xi = from_torch_pi05(
            hidden,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        out = dram_fwd(xi)
        ttnn.synchronize_device(mesh_device)
        h_dev = ttnn.to_torch(out)
        pcc = _compute_pcc(h_oracle[:, :ah, :], h_dev[:, :ah, :])
        print(f"\n[DRAM single-layer] PCC(active)={pcc:.6f}")
        assert pcc >= _PCC, f"DRAM PCC {pcc:.6f} < {_PCC}"
    finally:
        for t in dev:
            try:
                ttnn.deallocate(t)
            except Exception:
                pass


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_l1_single_layer_pcc(mesh_device):
    config, ah, suffix_len, bw, ref_blocks, adarms_cond, hidden, prefix_kv, mask, h_oracle = _setup(mesh_device)
    dev, l1_fwd = _build_l1(
        mesh_device, ref_blocks, config.expert_config, config, suffix_len, ah, adarms_cond, prefix_kv, mask
    )
    print(f"Hidden shape: {hidden.shape}")
    try:
        xi = from_torch_pi05(
            hidden,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        out = l1_fwd(xi)
        ttnn.synchronize_device(mesh_device)
        h_dev = ttnn.to_torch(out)
        pcc = _compute_pcc(h_oracle[:, :ah, :], h_dev[:, :ah, :])
        print(f"\n[L1 single-layer] PCC(active)={pcc:.6f}")
        assert pcc >= _PCC, f"L1 PCC {pcc:.6f} < {_PCC}"
    finally:
        for t in dev:
            try:
                ttnn.deallocate(t)
            except Exception:
                pass


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_walltime_l1_vs_dram_single_layer(mesh_device):
    from models.experimental.pi0_5.common.configs import GemmaConfig

    config, ah, suffix_len, bw, ref_blocks, adarms_cond, hidden, prefix_kv, mask, _ = _setup(mesh_device)

    l1_dev, l1_fwd = _build_l1(
        mesh_device, ref_blocks, config.expert_config, config, suffix_len, ah, adarms_cond, prefix_kv, mask
    )
    x = from_torch_pi05(hidden, dtype=ttnn.bfloat16, device=mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG)
    l1_dev.append(x)
    try:
        l1_med = _time_replay(mesh_device, l1_fwd, x)
    finally:
        for t in l1_dev:
            try:
                ttnn.deallocate(t)
            except Exception:
                pass
        ttnn.synchronize_device(mesh_device)

    dram_dev, dram_fwd = _build_dram(
        mesh_device, GemmaConfig.gemma_300m(), _chunk_weights(bw), config, suffix_len, ah, adarms_cond, prefix_kv, mask
    )
    x = from_torch_pi05(hidden, dtype=ttnn.bfloat16, device=mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG)
    dram_dev.append(x)
    try:
        dram_med = _time_replay(mesh_device, dram_fwd, x)
    finally:
        for t in dram_dev:
            try:
                ttnn.deallocate(t)
            except Exception:
                pass

    print(
        f"\n[walltime single-layer] L1={l1_med:.3f} ms  DRAM={dram_med:.3f} ms  "
        f"DRAM-L1={dram_med - l1_med:+.3f} ms ({'DRAM slower' if dram_med > l1_med else 'L1 slower'})"
    )
