# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single action-expert (denoise) block perf: L1 vs DRAM weights / biases / acts.

Measures the cost of one AdaRMSGemmaBlockTTNN forward (Gemma-300M expert,
width=1024, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256) on a
single chip. Random weights — kernel cost, not correctness.

The expert MLP is much smaller than the Gemma-2B prefill MLP (mlp_dim 4096
vs 16384), so the matmul kernel CB region is small and L1 weights fit above
the default 24KB reservation. All five variants are measurable.

Variants:
  V0  weights DRAM   biases DRAM   activations DRAM
  V1  weights L1     biases DRAM   activations DRAM
  V2  weights L1     biases L1     activations DRAM
  V3  weights L1     biases L1     activations L1
  V4  weights DRAM   biases DRAM   activations L1

Run:
    PI0_EXPERT_BENCH=1 pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_expert_block_l1_vs_dram.py
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.configs import GemmaConfig
from models.experimental.pi0_5.tt.ttnn_common import tensor_1d_to_2d_ttnn
from models.experimental.pi0_5.tt.ttnn_gemma import (
    AdaRMSGemmaBlockTTNN,
    precompute_freqs_cis_meta_format,
)

REAL_CKPT = os.environ.get("PI0_EXPERT_BENCH_CHECKPOINT", "/home/tt-admin/pi05_cache/pi05_libero_upstream")


BENCH_ENABLED = os.environ.get("PI0_EXPERT_BENCH") == "1"
pytestmark = pytest.mark.skipif(
    not BENCH_ENABLED,
    reason="set PI0_EXPERT_BENCH=1 to run the expert single-block bench",
)

# Gemma-300M expert block shape (Option C denoise, 3 layers / chip in paired).
BATCH = 1
S = int(os.environ.get("PI0_EXPERT_BENCH_S", "32"))  # action_horizon=10 padded to 32
WIDTH = 1024
MLP_DIM = 4096
NUM_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 256
MAX_SEQ_LEN = 512  # need room for prefix_kv during cross-attn (we skip kv though)

NUM_WARMUP = int(os.environ.get("PI0_EXPERT_BENCH_WARMUP", "10"))
NUM_ITER = int(os.environ.get("PI0_EXPERT_BENCH_ITER", "100"))
NUM_LAYERS = int(os.environ.get("PI0_EXPERT_BENCH_LAYERS", "2"))
VARIANTS = os.environ.get("PI0_EXPERT_BENCH_VARIANTS", "V0,V1,V2,V3,V4").split(",")


def _make_expert_config() -> GemmaConfig:
    return GemmaConfig(
        width=WIDTH,
        depth=1,
        mlp_dim=MLP_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
    )


def _to_l1(t):
    if t is None:
        return None
    if t.memory_config().buffer_type == ttnn.BufferType.L1:
        return t
    new_t = ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(t)
    return new_t


def _upload(t: torch.Tensor, device, dtype, memory_config) -> "ttnn.Tensor":
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)


def _real_or_random_expert_layer(idx: int = 0) -> Dict[str, torch.Tensor]:
    """Load layer-`idx` expert weights from the real checkpoint; fall back to random.

    Returns: dict with attn/mlp matmul keys, plus pre-fused adaRMS modulation
    weight/bias (concat of input_layernorm.dense.* and
    post_attention_layernorm.dense.* along dim 0 — shape [6*W, W] and [6*W]).
    """
    if Path(REAL_CKPT, "model.safetensors").exists():
        from models.experimental.pi0_5.common import Pi0_5WeightLoader

        loader = Pi0_5WeightLoader(REAL_CKPT)
        ae = loader.get_action_expert_weights()
        pfx = f"model.layers.{idx}."
        in_ln_w = ae[pfx + "input_layernorm.dense.weight"]
        post_ln_w = ae[pfx + "post_attention_layernorm.dense.weight"]
        in_ln_b = ae[pfx + "input_layernorm.dense.bias"]
        post_ln_b = ae[pfx + "post_attention_layernorm.dense.bias"]
        out = {
            "self_attn.q_proj.weight": ae[pfx + "self_attn.q_proj.weight"].clone(),
            "self_attn.k_proj.weight": ae[pfx + "self_attn.k_proj.weight"].clone(),
            "self_attn.v_proj.weight": ae[pfx + "self_attn.v_proj.weight"].clone(),
            "self_attn.o_proj.weight": ae[pfx + "self_attn.o_proj.weight"].clone(),
            "mlp.gate_proj.weight": ae[pfx + "mlp.gate_proj.weight"].clone(),
            "mlp.up_proj.weight": ae[pfx + "mlp.up_proj.weight"].clone(),
            "mlp.down_proj.weight": ae[pfx + "mlp.down_proj.weight"].clone(),
            "adarms_mod.weight_fused_2d": torch.cat([in_ln_w, post_ln_w], dim=0).contiguous(),  # [6W, W]
            "adarms_mod.bias_fused": torch.cat([in_ln_b, post_ln_b], dim=0).contiguous(),  # [6W]
        }
        print(f"   [real weights] expert layer {idx} from {REAL_CKPT}")
        return out
    print(f"   [random weights] expert layer {idx} missing, using torch.randn")
    torch.manual_seed(idx)
    s = 0.02
    return {
        "self_attn.q_proj.weight": torch.randn(NUM_HEADS * HEAD_DIM, WIDTH) * s,
        "self_attn.k_proj.weight": torch.randn(NUM_KV_HEADS * HEAD_DIM, WIDTH) * s,
        "self_attn.v_proj.weight": torch.randn(NUM_KV_HEADS * HEAD_DIM, WIDTH) * s,
        "self_attn.o_proj.weight": torch.randn(WIDTH, NUM_HEADS * HEAD_DIM) * s,
        "mlp.gate_proj.weight": torch.randn(MLP_DIM, WIDTH) * s,
        "mlp.up_proj.weight": torch.randn(MLP_DIM, WIDTH) * s,
        "mlp.down_proj.weight": torch.randn(WIDTH, MLP_DIM) * s,
        "adarms_mod.weight_fused_2d": torch.randn(6 * WIDTH, WIDTH) * s,
        "adarms_mod.bias_fused": torch.randn(6 * WIDTH) * s,
    }


def _build_expert_block(
    device,
    weights_mc: "ttnn.MemoryConfig",
    biases_mc: "ttnn.MemoryConfig",
    raw: Optional[Dict[str, torch.Tensor]] = None,
    layer_idx: int = 0,
) -> AdaRMSGemmaBlockTTNN:
    if raw is None:
        raw = _real_or_random_expert_layer(layer_idx)

    w_q = raw["self_attn.q_proj.weight"]
    w_k = raw["self_attn.k_proj.weight"]
    w_v = raw["self_attn.v_proj.weight"]
    w_o = raw["self_attn.o_proj.weight"]
    w_gate = raw["mlp.gate_proj.weight"]
    w_up = raw["mlp.up_proj.weight"]
    w_down = raw["mlp.down_proj.weight"]
    mod_w = raw["adarms_mod.weight_fused_2d"]
    mod_b = raw["adarms_mod.bias_fused"]
    # The AdaRMSGemmaBlockTTNN does not consume input_layernorm.weight or
    # post_attention_layernorm.weight (it uses the adaRMS modulation instead),
    # so we skip them. Keep harmless placeholders for the dict-build below.
    in_ln = torch.zeros(WIDTH)
    post_ln = torch.zeros(WIDTH)

    wq = _upload(w_q.T.contiguous(), device, ttnn.bfloat8_b, weights_mc)
    wk = _upload(w_k.T.contiguous(), device, ttnn.bfloat8_b, weights_mc)
    wv = _upload(w_v.T.contiguous(), device, ttnn.bfloat8_b, weights_mc)
    fused_wqkv = ttnn.concat([wq, wk, wv], dim=-1, memory_config=weights_mc)
    ttnn.deallocate(wq)
    ttnn.deallocate(wk)
    ttnn.deallocate(wv)

    weights: Dict[str, "ttnn.Tensor"] = {
        "self_attn.wqkv": fused_wqkv,
        "self_attn.o_proj.weight": _upload(w_o.T.contiguous(), device, ttnn.bfloat8_b, weights_mc),
        "mlp.gate_proj.weight": _upload(w_gate.T.contiguous(), device, ttnn.bfloat8_b, weights_mc),
        "mlp.up_proj.weight": _upload(w_up.T.contiguous(), device, ttnn.bfloat8_b, weights_mc),
        "mlp.down_proj.weight": _upload(w_down.T.contiguous(), device, ttnn.bfloat8_b, weights_mc),
        "input_layernorm.weight": _upload((in_ln + 1.0).reshape(1, -1).contiguous(), device, ttnn.bfloat16, biases_mc),
        "post_attention_layernorm.weight": _upload(
            (post_ln + 1.0).reshape(1, -1).contiguous(), device, ttnn.bfloat16, biases_mc
        ),
        "adarms_mod.weight": _upload(mod_w.T.contiguous(), device, ttnn.bfloat16, weights_mc),
        "adarms_mod.bias": tensor_1d_to_2d_ttnn(mod_b, device, dtype=ttnn.bfloat16),
    }
    # tensor_1d_to_2d_ttnn defaults vary by helper; force the bias placement.
    weights["adarms_mod.bias"] = (
        _to_l1(weights["adarms_mod.bias"])
        if biases_mc.buffer_type == ttnn.BufferType.L1
        else weights["adarms_mod.bias"]
    )

    cos_meta, sin_meta = precompute_freqs_cis_meta_format(HEAD_DIM, MAX_SEQ_LEN, device)
    cfg = _make_expert_config()
    block = AdaRMSGemmaBlockTTNN(cfg, weights, layer_idx=layer_idx, device=device, cos_meta=cos_meta, sin_meta=sin_meta)
    block._bench_weights = weights
    block._bench_cos = cos_meta
    block._bench_sin = sin_meta
    return block


def _build_expert_stack(
    device,
    weights_mc: "ttnn.MemoryConfig",
    biases_mc: "ttnn.MemoryConfig",
    raw_layers: List[Dict[str, torch.Tensor]],
) -> List[AdaRMSGemmaBlockTTNN]:
    return [
        _build_expert_block(device, weights_mc, biases_mc, raw=raw_layers[i], layer_idx=i)
        for i in range(len(raw_layers))
    ]


def _free_block(block: AdaRMSGemmaBlockTTNN) -> None:
    for t in (block._bench_cos, block._bench_sin):
        try:
            ttnn.deallocate(t)
        except RuntimeError:
            pass
    for t in block._bench_weights.values():
        try:
            ttnn.deallocate(t)
        except RuntimeError:
            pass


def _forward_expert_stack(blocks, h_in, mask_tt, adarms_tt) -> "ttnn.Tensor":
    h = h_in
    h_owned = False
    for blk in blocks:
        out, _ = blk.forward(
            h,
            blk._bench_cos,
            blk._bench_sin,
            adarms_tt,
            attention_mask=mask_tt,
            position_ids=None,
            past_key_value=None,
            use_cache=False,
        )
        if h_owned:
            ttnn.deallocate(h)
        h = out
        h_owned = True
    return h


def _time_forward(
    device, blocks: List[AdaRMSGemmaBlockTTNN], activations_mc: "ttnn.MemoryConfig"
) -> Tuple[float, float, float, float]:
    hidden_host = torch.randn(BATCH, S, WIDTH) * 0.5
    adarms_host = torch.randn(BATCH, 1, WIDTH) * 0.1

    adarms_tt = _upload(adarms_host, device, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG)
    mask_tt = _upload(torch.zeros(BATCH, 1, S, S), device, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG)

    def upload_hidden():
        return _upload(hidden_host, device, ttnn.bfloat16, activations_mc)

    for _ in range(NUM_WARMUP):
        h = upload_hidden()
        out = _forward_expert_stack(blocks, h, mask_tt, adarms_tt)
        ttnn.deallocate(h)
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    samples: List[float] = []
    for _ in range(NUM_ITER):
        h = upload_hidden()
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = _forward_expert_stack(blocks, h, mask_tt, adarms_tt)
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - t0) * 1000)
        ttnn.deallocate(h)
        ttnn.deallocate(out)

    ttnn.deallocate(adarms_tt)
    ttnn.deallocate(mask_tt)
    return (
        statistics.mean(samples),
        statistics.stdev(samples) if len(samples) > 1 else 0.0,
        min(samples),
        max(samples),
    )


VARIANT_SPECS = {
    "V0": ("DRAM", "DRAM", "DRAM"),
    "V1": ("L1", "DRAM", "DRAM"),
    "V2": ("L1", "L1", "DRAM"),
    "V3": ("L1", "L1", "L1"),
    "V4": ("DRAM", "DRAM", "L1"),
}


def _mc(label: str) -> "ttnn.MemoryConfig":
    return ttnn.L1_MEMORY_CONFIG if label == "L1" else ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_expert_block_l1_vs_dram(device):
    print("\n" + "=" * 80)
    print(
        f"  EXPERT STACK L1 vs DRAM  shape=(B={BATCH}, S={S}, W={WIDTH}, mlp_dim={MLP_DIM})  "
        f"layers={NUM_LAYERS}  warmup={NUM_WARMUP}  iter={NUM_ITER}  variants={VARIANTS}"
    )
    print("=" * 80)

    # Load weights for every layer once and reuse across variants.
    raw_layers = [_real_or_random_expert_layer(i) for i in range(NUM_LAYERS)]

    results: List[Tuple[str, str, float, float, float]] = []
    for name in VARIANTS:
        if name not in VARIANT_SPECS:
            print(f"   skipping unknown variant {name!r}")
            continue
        w_lbl, b_lbl, a_lbl = VARIANT_SPECS[name]
        print(f"\n>> {name}  w={w_lbl},b={b_lbl},a={a_lbl}")
        blocks = _build_expert_stack(device, _mc(w_lbl), _mc(b_lbl), raw_layers)
        try:
            mean, stdev, mn, mx = _time_forward(device, blocks, _mc(a_lbl))
            print(f"   mean={mean:.2f} ms  stdev={stdev:.3f}  min={mn:.2f}  max={mx:.2f}")
            results.append((name, f"w={w_lbl},b={b_lbl},a={a_lbl}", mean, stdev, mn))
        finally:
            for blk in blocks:
                _free_block(blk)
            ttnn.synchronize_device(device)

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    base = next((r for r in results if r[0] == "V0"), None)
    print(f"  {'variant':<6}  {'placement':<28}  {'mean ms':>9}  {'stdev':>7}  {'min':>7}  {'speedup':>8}")
    for name, place, mean, stdev, mn in results:
        speedup = (base[2] / mean) if (base is not None and mean > 0) else 0.0
        print(f"  {name:<6}  {place:<28}  {mean:>9.2f}  {stdev:>7.3f}  {mn:>7.2f}  {speedup:>8.2f}x")
    print("=" * 80)
