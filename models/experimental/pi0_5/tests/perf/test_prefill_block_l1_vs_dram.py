# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single Gemma-2B VLM block perf: L1 vs DRAM weights / biases / activations.

Measures the cost of one VLM transformer-block forward (the Gemma-2B layer
used in Option C prefill) under different memory-config combinations on a
single chip. Uses the real pi05_libero_upstream layer-0 weights when the
checkpoint is present; falls back to random weights otherwise (perf is
shape-dependent, not value-dependent, so this is mostly a sanity affordance).

Variants (matrix of placement choices):
  V0  weights DRAM   biases DRAM   activations DRAM   (baseline)
  V1  weights L1     biases DRAM   activations DRAM
  V2  weights L1     biases L1     activations DRAM
  V3  weights L1     biases L1     activations L1     (all-L1)
  V4  weights DRAM   biases DRAM   activations L1     (input-only L1)

Reported per variant:
  mean ms / iter, stdev, min, speed-up vs V0.

Run:
    PI0_PREFILL_BENCH=1 pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_prefill_block_l1_vs_dram.py

Env knobs (optional):
    PI0_PREFILL_BENCH_S=512        prefix sequence length (must be tile-aligned)
    PI0_PREFILL_BENCH_WARMUP=5     warmup iterations per variant
    PI0_PREFILL_BENCH_ITER=20      timed iterations per variant
    PI0_PREFILL_BENCH_VARIANTS=V0,V1,V2,V3,V4   comma-list to subset variants
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
from models.experimental.pi0_5.tt.ttnn_gemma import (
    GemmaBlockTTNN,
    precompute_freqs_cis_meta_format,
)

REAL_CKPT = os.environ.get("PI0_PREFILL_BENCH_CHECKPOINT", "/home/tt-admin/pi05_cache/pi05_libero_upstream")


BENCH_ENABLED = os.environ.get("PI0_PREFILL_BENCH") == "1"
pytestmark = pytest.mark.skipif(
    not BENCH_ENABLED,
    reason="set PI0_PREFILL_BENCH=1 to run the single-block L1-vs-DRAM bench",
)

# Gemma-2B VLM block shape (Option C prefill, depth=18 with 1 layer / chip).
BATCH = 1
S = int(os.environ.get("PI0_PREFILL_BENCH_S", "512"))
WIDTH = 2048
MLP_DIM = 16384
NUM_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 256
MAX_SEQ_LEN = 512
RMS_NORM_EPS = 1e-6

NUM_WARMUP = int(os.environ.get("PI0_PREFILL_BENCH_WARMUP", "10"))
NUM_ITER = int(os.environ.get("PI0_PREFILL_BENCH_ITER", "100"))
NUM_LAYERS = int(os.environ.get("PI0_PREFILL_BENCH_LAYERS", "2"))
VARIANTS = os.environ.get("PI0_PREFILL_BENCH_VARIANTS", "V0,V1,V2,V3,V4").split(",")


# Dtype knobs for the matmul weights. Default MLP dtype is bf4_b — this is
# the smallest tile size that lets L1 weights fit at TP=1 (the bf8_b production
# default triggers a static-CB clash; see project_pi05_single_layer_l1_dram_perf
# memory for the analysis). PI0_PREFILL_BENCH_MLP_DTYPE / PI0_PREFILL_BENCH_ATTN_DTYPE
# accept "bf16", "bf8", or "bf4".
def _dtype_from_env(name: str, default: str) -> "ttnn.DataType":
    label = os.environ.get(name, default).lower()
    if label in ("bf4", "bfloat4_b", "bfloat4"):
        return ttnn.bfloat4_b
    if label in ("bf16", "bfloat16"):
        return ttnn.bfloat16
    return ttnn.bfloat8_b


MLP_DTYPE = _dtype_from_env("PI0_PREFILL_BENCH_MLP_DTYPE", "bf4")
ATTN_DTYPE = _dtype_from_env("PI0_PREFILL_BENCH_ATTN_DTYPE", "bf8")


def _make_gemma_config() -> GemmaConfig:
    return GemmaConfig(
        width=WIDTH,
        depth=1,
        mlp_dim=MLP_DIM,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
    )


def _upload(t: torch.Tensor, device, dtype, memory_config) -> "ttnn.Tensor":
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def _to_mem(t: "ttnn.Tensor", target: "ttnn.MemoryConfig") -> "ttnn.Tensor":
    """Migrate an already-on-device tensor; no-op if already at target."""
    if t is None:
        return None
    if t.memory_config().buffer_type == target.buffer_type:
        return t
    new_t = ttnn.to_memory_config(t, target)
    ttnn.deallocate(t)
    return new_t


def _real_or_random_vlm_layer(idx: int) -> Dict[str, torch.Tensor]:
    """Return layer-`idx` VLM weights from the real checkpoint, or random fallback."""
    if Path(REAL_CKPT, "model.safetensors").exists():
        from models.experimental.pi0_5.common import Pi0_5WeightLoader

        loader = Pi0_5WeightLoader(REAL_CKPT)
        lang = loader.get_vlm_language_weights()
        pfx = f"model.layers.{idx}."
        out: Dict[str, torch.Tensor] = {}
        for k in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ):
            out[k] = lang[pfx + k].clone()
        print(f"   [real weights] loaded layer {idx} from {REAL_CKPT}")
        return out
    # Fallback to random — same shape contract.
    print(f"   [random weights] checkpoint missing for layer {idx}, using torch.randn")
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
        "input_layernorm.weight": torch.randn(WIDTH) * 0.1,
        "post_attention_layernorm.weight": torch.randn(WIDTH) * 0.1,
    }


def _build_random_block(
    device,
    weights_mc: "ttnn.MemoryConfig",
    biases_mc: "ttnn.MemoryConfig",
    raw: Optional[Dict[str, torch.Tensor]] = None,
    layer_idx: int = 0,
) -> GemmaBlockTTNN:
    """Build a GemmaBlockTTNN with real or random weights at the requested placements."""
    if raw is None:
        raw = _real_or_random_vlm_layer(layer_idx)
    w_q = raw["self_attn.q_proj.weight"]
    w_k = raw["self_attn.k_proj.weight"]
    w_v = raw["self_attn.v_proj.weight"]
    w_o = raw["self_attn.o_proj.weight"]
    w_gate = raw["mlp.gate_proj.weight"]
    w_up = raw["mlp.up_proj.weight"]
    w_down = raw["mlp.down_proj.weight"]
    in_ln = raw["input_layernorm.weight"]
    post_ln = raw["post_attention_layernorm.weight"]

    # QKV fusion: upload each .T, concat along last dim. (Matches
    # _load_vlm_block_weights_l1.)
    wq = _upload(w_q.T.contiguous(), device, ATTN_DTYPE, weights_mc)
    wk = _upload(w_k.T.contiguous(), device, ATTN_DTYPE, weights_mc)
    wv = _upload(w_v.T.contiguous(), device, ATTN_DTYPE, weights_mc)
    fused_wqkv = ttnn.concat([wq, wk, wv], dim=-1, memory_config=weights_mc)
    ttnn.deallocate(wq)
    ttnn.deallocate(wk)
    ttnn.deallocate(wv)

    weights: Dict[str, "ttnn.Tensor"] = {
        "self_attn.wqkv": fused_wqkv,
        "self_attn.o_proj.weight": _upload(w_o.T.contiguous(), device, ATTN_DTYPE, weights_mc),
        "mlp.gate_proj.weight": _upload(w_gate.T.contiguous(), device, MLP_DTYPE, weights_mc),
        "mlp.up_proj.weight": _upload(w_up.T.contiguous(), device, MLP_DTYPE, weights_mc),
        "mlp.down_proj.weight": _upload(w_down.T.contiguous(), device, MLP_DTYPE, weights_mc),
        # Norms are 1-D — bring to 2-D [1, W] for tile layout. Gemma adds +1
        # offset to RMSNorm scales; the loader pre-bakes that.
        "input_layernorm.weight": _upload((in_ln + 1.0).reshape(1, -1).contiguous(), device, ttnn.bfloat16, biases_mc),
        "post_attention_layernorm.weight": _upload(
            (post_ln + 1.0).reshape(1, -1).contiguous(), device, ttnn.bfloat16, biases_mc
        ),
    }

    cos_meta, sin_meta = precompute_freqs_cis_meta_format(HEAD_DIM, MAX_SEQ_LEN, device)
    cfg = _make_gemma_config()
    block = GemmaBlockTTNN(cfg, weights, layer_idx=layer_idx, device=device, cos_meta=cos_meta, sin_meta=sin_meta)
    # Stash so we can free between variants.
    block._bench_weights = weights
    block._bench_cos = cos_meta
    block._bench_sin = sin_meta
    return block


def _build_block_stack(
    device,
    weights_mc: "ttnn.MemoryConfig",
    biases_mc: "ttnn.MemoryConfig",
    raw_layers: List[Dict[str, torch.Tensor]],
) -> List[GemmaBlockTTNN]:
    """Build NUM_LAYERS independent GemmaBlockTTNN with their own weights."""
    return [
        _build_random_block(device, weights_mc, biases_mc, raw=raw_layers[i], layer_idx=i)
        for i in range(len(raw_layers))
    ]


def _free_block(block: GemmaBlockTTNN) -> None:
    """Deallocate every uploaded tensor from a benchmark block."""
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


def _forward_stack(
    blocks: List[GemmaBlockTTNN],
    h_in: "ttnn.Tensor",
    mask_tt: "ttnn.Tensor",
) -> "ttnn.Tensor":
    """Run h_in through every block in sequence, deallocating intermediates."""
    h = h_in
    h_owned = False  # caller owns h_in
    for blk in blocks:
        out, _ = blk.forward(
            h,
            cos=blk._bench_cos,
            sin=blk._bench_sin,
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
    device,
    blocks: List[GemmaBlockTTNN],
    activations_mc: "ttnn.MemoryConfig",
) -> Tuple[float, float, float, float]:
    """Returns (mean_ms, stdev_ms, min_ms, max_ms) over NUM_ITER stack forwards."""
    hidden_host = torch.randn(BATCH, S, WIDTH) * 0.5
    mask_host = torch.zeros(BATCH, 1, S, S)
    mask_tt = _upload(mask_host, device, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG)

    for _ in range(NUM_WARMUP):
        h = _upload(hidden_host, device, ttnn.bfloat16, activations_mc)
        out = _forward_stack(blocks, h, mask_tt)
        ttnn.deallocate(h)
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    samples: List[float] = []
    for _ in range(NUM_ITER):
        h = _upload(hidden_host, device, ttnn.bfloat16, activations_mc)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = _forward_stack(blocks, h, mask_tt)
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - t0) * 1000)
        ttnn.deallocate(h)
        ttnn.deallocate(out)

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


def _dump_placement(block: GemmaBlockTTNN, label: str) -> None:
    """Print the actual buffer_type of every block weight/bias.

    Run before AND after one forward pass to catch any silent migration.
    """
    print(f"   [placement {label}]")

    def fmt(t) -> str:
        if t is None:
            return "None"
        bt = t.memory_config().buffer_type
        return "L1" if bt == ttnn.BufferType.L1 else "DRAM" if bt == ttnn.BufferType.DRAM else str(bt)

    attn = block.attention
    mlp = block.mlp
    rows = [
        ("self_attn.wqkv", attn.wqkv),
        ("self_attn.o_proj", attn.o_proj),
        ("mlp.gate_proj", mlp.gate_proj),
        ("mlp.up_proj", mlp.up_proj),
        ("mlp.down_proj", mlp.down_proj),
        ("input_layernorm.weight", block.input_layernorm_weight),
        ("post_attention_layernorm.weight", block.post_attention_layernorm_weight),
    ]
    for name, t in rows:
        print(f"      {name:<40} {fmt(t)}")


def _mc(label: str) -> "ttnn.MemoryConfig":
    return ttnn.L1_MEMORY_CONFIG if label == "L1" else ttnn.DRAM_MEMORY_CONFIG


# 24 KB l1_small_size — matches the production Option C path. With bf8_b MLP
# weights the matmul kernel's static CB region (~694 KB / core) overlaps the
# allocator's user-L1 region and L1 weights TT_THROW with a CB clash. The
# default MLP_DTYPE is bf4_b which halves the weight CB tile and clears the
# overlap; see project_pi05_single_layer_l1_dram_perf memory for the analysis.
# Set PI0_PREFILL_BENCH_MLP_DTYPE=bf8 to reproduce the original clash.
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_prefill_block_l1_vs_dram(device):
    def _dtype_label(d):
        return {ttnn.bfloat4_b: "bf4", ttnn.bfloat8_b: "bf8", ttnn.bfloat16: "bf16"}.get(d, str(d))

    print("\n" + "=" * 80)
    print(f"  PREFILL STACK L1 vs DRAM  shape=(B={BATCH}, S={S}, W={WIDTH})  layers={NUM_LAYERS}")
    print(
        f"  warmup={NUM_WARMUP}  iter={NUM_ITER}  variants={VARIANTS}  "
        f"mlp_dtype={_dtype_label(MLP_DTYPE)}  attn_dtype={_dtype_label(ATTN_DTYPE)}"
    )
    print("=" * 80)

    # Load weights for every layer once and reuse for every variant.
    raw_layers = [_real_or_random_vlm_layer(i) for i in range(NUM_LAYERS)]

    results: List[Tuple[str, str, float, float, float, float]] = []
    for name in VARIANTS:
        if name not in VARIANT_SPECS:
            print(f"   skipping unknown variant {name!r}")
            continue
        w_lbl, b_lbl, a_lbl = VARIANT_SPECS[name]
        print(f"\n>> {name}  weights={w_lbl}  biases={b_lbl}  activations={a_lbl}")
        blocks = _build_block_stack(device, _mc(w_lbl), _mc(b_lbl), raw_layers)
        try:
            _dump_placement(blocks[0], f"{name} layer 0 after build")
            mean, stdev, mn, mx = _time_forward(device, blocks, _mc(a_lbl))
            print(f"   mean={mean:.2f} ms  stdev={stdev:.3f}  min={mn:.2f}  max={mx:.2f}")
            results.append((name, f"w={w_lbl},b={b_lbl},a={a_lbl}", mean, stdev, mn, mx))
        finally:
            for blk in blocks:
                _free_block(blk)
            ttnn.synchronize_device(device)

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    base = next((r for r in results if r[0] == "V0"), None)
    print(f"  {'variant':<6}  {'placement':<28}  {'mean ms':>9}  {'stdev':>7}  {'min':>7}  {'speedup':>8}")
    for r in results:
        name, place, mean, stdev, mn, mx = r
        speedup = (base[2] / mean) if (base is not None and mean > 0) else 0.0
        print(f"  {name:<6}  {place:<28}  {mean:>9.2f}  {stdev:>7.3f}  {mn:>7.2f}  {speedup:>8.2f}x")
    print("=" * 80)
