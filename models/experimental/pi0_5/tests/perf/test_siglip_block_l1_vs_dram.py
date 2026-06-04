# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single SigLIP transformer block perf: L1 vs DRAM weights / biases / activations.

Measures the cost of one SigLIP encoder layer (hidden=1152, intermediate=4304,
num_heads=16, head_dim=72→padded to 96) forward on a single chip. Random
weights — we're measuring kernel cost, not correctness.

The SigLIP MLP is ~4× smaller than the Gemma-2B prefill MLP (4304 vs 16384
output dim), so the matmul kernel's static CB region is small enough that
L1 weights fit above the default `l1_small_size=24KB` reservation. All five
variants should be measurable.

Variants:
  V0  weights DRAM   biases DRAM   activations DRAM
  V1  weights L1     biases DRAM   activations DRAM
  V2  weights L1     biases L1     activations DRAM
  V3  weights L1     biases L1     activations L1
  V4  weights DRAM   biases DRAM   activations L1

Run:
    PI0_SIGLIP_BENCH=1 pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_siglip_block_l1_vs_dram.py
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.configs import SigLIPConfig
from models.experimental.pi0_5.tt.ttnn_siglip import SigLIPBlockTTNN

REAL_CKPT = os.environ.get("PI0_SIGLIP_BENCH_CHECKPOINT", "/home/tt-admin/pi05_cache/pi05_libero_upstream")


BENCH_ENABLED = os.environ.get("PI0_SIGLIP_BENCH") == "1"
pytestmark = pytest.mark.skipif(
    not BENCH_ENABLED,
    reason="set PI0_SIGLIP_BENCH=1 to run the SigLIP single-block bench",
)

# SigLIP-27 layer shape used by pi0.5.
BATCH = 1
S = int(os.environ.get("PI0_SIGLIP_BENCH_S", "256"))  # 224x224 image → 16x16 = 256 patches
HIDDEN = 1152
INTERMEDIATE = 4304
NUM_HEADS = 16
HEAD_DIM = 72  # padded to 96 inside SigLIPAttentionTTNN

NUM_WARMUP = int(os.environ.get("PI0_SIGLIP_BENCH_WARMUP", "10"))
NUM_ITER = int(os.environ.get("PI0_SIGLIP_BENCH_ITER", "100"))
NUM_LAYERS = int(os.environ.get("PI0_SIGLIP_BENCH_LAYERS", "2"))
VARIANTS = os.environ.get("PI0_SIGLIP_BENCH_VARIANTS", "V0,V1,V2,V3,V4").split(",")


def _make_siglip_config() -> SigLIPConfig:
    return SigLIPConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=1,
        num_attention_heads=NUM_HEADS,
        image_size=224,
        patch_size=14,
    )


def _to_l1(t: "ttnn.Tensor") -> "ttnn.Tensor":
    """Migrate a tensor to L1; no-op when already L1."""
    if t is None:
        return None
    if t.memory_config().buffer_type == ttnn.BufferType.L1:
        return t
    new_t = ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(t)
    return new_t


def _real_or_random_siglip_layer(idx: int = 0) -> Dict[str, torch.Tensor]:
    """Load layer-`idx` SigLIP weights from the real checkpoint; fall back to random."""
    if Path(REAL_CKPT, "model.safetensors").exists():
        from models.experimental.pi0_5.common import Pi0_5WeightLoader

        loader = Pi0_5WeightLoader(REAL_CKPT)
        vision = loader.get_vlm_vision_weights()
        pfx = f"vision_model.encoder.layers.{idx}."
        # The block reads these key suffixes directly out of the torch dict.
        suffixes = [
            "layer_norm1.weight",
            "layer_norm1.bias",
            "layer_norm2.weight",
            "layer_norm2.bias",
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.out_proj.weight",
            "self_attn.out_proj.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ]
        out = {s: vision[pfx + s].clone() for s in suffixes}
        print(f"   [real weights] SigLIP layer {idx} from {REAL_CKPT}")
        return out
    print(f"   [random weights] SigLIP layer {idx} missing, using torch.randn")
    torch.manual_seed(idx)
    scale = 0.02
    return {
        "layer_norm1.weight": torch.randn(HIDDEN) * 0.1 + 1.0,
        "layer_norm1.bias": torch.randn(HIDDEN) * 0.05,
        "layer_norm2.weight": torch.randn(HIDDEN) * 0.1 + 1.0,
        "layer_norm2.bias": torch.randn(HIDDEN) * 0.05,
        "self_attn.q_proj.weight": torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN) * scale,
        "self_attn.q_proj.bias": torch.randn(NUM_HEADS * HEAD_DIM) * scale,
        "self_attn.k_proj.weight": torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN) * scale,
        "self_attn.k_proj.bias": torch.randn(NUM_HEADS * HEAD_DIM) * scale,
        "self_attn.v_proj.weight": torch.randn(NUM_HEADS * HEAD_DIM, HIDDEN) * scale,
        "self_attn.v_proj.bias": torch.randn(NUM_HEADS * HEAD_DIM) * scale,
        "self_attn.out_proj.weight": torch.randn(HIDDEN, NUM_HEADS * HEAD_DIM) * scale,
        "self_attn.out_proj.bias": torch.randn(HIDDEN) * scale,
        "mlp.fc1.weight": torch.randn(INTERMEDIATE, HIDDEN) * scale,
        "mlp.fc1.bias": torch.randn(INTERMEDIATE) * scale,
        "mlp.fc2.weight": torch.randn(HIDDEN, INTERMEDIATE) * scale,
        "mlp.fc2.bias": torch.randn(HIDDEN) * scale,
    }


def _migrate_block_weights(block: SigLIPBlockTTNN, weights_to_l1: bool, biases_to_l1: bool) -> None:
    """Migrate the block's already-uploaded tensors to L1 according to flags."""
    # Norm weights/biases — count as 'biases' (small).
    if biases_to_l1:
        block.ln1_weight = _to_l1(block.ln1_weight)
        block.ln1_bias = _to_l1(block.ln1_bias)
        block.ln2_weight = _to_l1(block.ln2_weight)
        block.ln2_bias = _to_l1(block.ln2_bias)
        block.attention.bqkv = _to_l1(getattr(block.attention, "bqkv", None))
        block.attention.bo = _to_l1(getattr(block.attention, "bo", None))
        block.mlp.fc1_bias = _to_l1(getattr(block.mlp, "fc1_bias", None))
        block.mlp.fc2_bias = _to_l1(getattr(block.mlp, "fc2_bias", None))
    if weights_to_l1:
        block.attention.wqkv = _to_l1(block.attention.wqkv)
        block.attention.wo = _to_l1(block.attention.wo)
        block.mlp.fc1_weight = _to_l1(block.mlp.fc1_weight)
        block.mlp.fc2_weight = _to_l1(block.mlp.fc2_weight)


def _forward_stack(blocks: List[SigLIPBlockTTNN], h_in: "ttnn.Tensor") -> "ttnn.Tensor":
    """Run h_in through every block in sequence."""
    h = h_in
    h_owned = False
    for blk in blocks:
        out = blk.forward(h)
        if h_owned:
            ttnn.deallocate(h)
        h = out
        h_owned = True
    return h


def _time_forward(
    device, blocks: List[SigLIPBlockTTNN], activations_mc: "ttnn.MemoryConfig"
) -> Tuple[float, float, float, float]:
    hidden_host = torch.randn(BATCH, S, HIDDEN) * 0.5

    def upload_hidden() -> "ttnn.Tensor":
        return ttnn.from_torch(
            hidden_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=activations_mc
        )

    for _ in range(NUM_WARMUP):
        h = upload_hidden()
        out = _forward_stack(blocks, h)
        ttnn.deallocate(h)
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    samples: List[float] = []
    for _ in range(NUM_ITER):
        h = upload_hidden()
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = _forward_stack(blocks, h)
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - t0) * 1000)
        ttnn.deallocate(h)
        ttnn.deallocate(out)

    return (
        statistics.mean(samples),
        statistics.stdev(samples) if len(samples) > 1 else 0.0,
        min(samples),
        max(samples),
    )


VARIANT_SPECS = {
    "V0": (False, False, "DRAM"),
    "V1": (True, False, "DRAM"),
    "V2": (True, True, "DRAM"),
    "V3": (True, True, "L1"),
    "V4": (False, False, "L1"),
}


def _mc(label: str) -> "ttnn.MemoryConfig":
    return ttnn.L1_MEMORY_CONFIG if label == "L1" else ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_siglip_block_l1_vs_dram(device):
    print("\n" + "=" * 80)
    print(
        f"  SIGLIP STACK L1 vs DRAM  shape=(B={BATCH}, S={S}, H={HIDDEN})  "
        f"layers={NUM_LAYERS}  warmup={NUM_WARMUP}  iter={NUM_ITER}  variants={VARIANTS}"
    )
    print("=" * 80)

    cfg = _make_siglip_config()
    raw_layers = [_real_or_random_siglip_layer(i) for i in range(NUM_LAYERS)]

    def _free_one(block: SigLIPBlockTTNN) -> None:
        for attr in (
            block.ln1_weight,
            block.ln1_bias,
            block.ln2_weight,
            block.ln2_bias,
            getattr(block.attention, "wqkv", None),
            getattr(block.attention, "bqkv", None),
            getattr(block.attention, "wo", None),
            getattr(block.attention, "bo", None),
            getattr(block.mlp, "fc1_weight", None),
            getattr(block.mlp, "fc1_bias", None),
            getattr(block.mlp, "fc2_weight", None),
            getattr(block.mlp, "fc2_bias", None),
        ):
            if attr is None:
                continue
            try:
                ttnn.deallocate(attr)
            except RuntimeError:
                pass

    results: List[Tuple[str, str, float, float, float]] = []
    for name in VARIANTS:
        if name not in VARIANT_SPECS:
            print(f"   skipping unknown variant {name!r}")
            continue
        weights_l1, biases_l1, a_lbl = VARIANT_SPECS[name]
        place_str = f"w={'L1' if weights_l1 else 'DRAM'},b={'L1' if biases_l1 else 'DRAM'},a={a_lbl}"
        print(f"\n>> {name}  {place_str}")

        blocks: List[SigLIPBlockTTNN] = []
        try:
            for raw in raw_layers:
                blk = SigLIPBlockTTNN(cfg, raw, device)
                _migrate_block_weights(blk, weights_l1, biases_l1)
                blocks.append(blk)
            mean, stdev, mn, mx = _time_forward(device, blocks, _mc(a_lbl))
            print(f"   mean={mean:.2f} ms  stdev={stdev:.3f}  min={mn:.2f}  max={mx:.2f}")
            results.append((name, place_str, mean, stdev, mn))
        finally:
            for blk in blocks:
                _free_one(blk)
            ttnn.synchronize_device(device)

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    base = next((r for r in results if r[0] == "V0"), None)
    print(f"  {'variant':<6}  {'placement':<30}  {'mean ms':>9}  {'stdev':>7}  {'min':>7}  {'speedup':>8}")
    for name, place, mean, stdev, mn in results:
        speedup = (base[2] / mean) if (base is not None and mean > 0) else 0.0
        print(f"  {name:<6}  {place:<30}  {mean:>9.2f}  {stdev:>7.3f}  {mn:>7.2f}  {speedup:>8.2f}x")
    print("=" * 80)
