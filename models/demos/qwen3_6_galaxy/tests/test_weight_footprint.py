# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-block weight DRAM footprint tests for qwen3.6-galaxy.

These tests catch the class of bug where a Galaxy block uploads a weight via
``ReplicateTensorToMesh`` when it should be sharded.  Single-layer PCC tests
pass even when weights are wrongly replicated; this test would have caught the
original 64-layer OOM at block-write time.

Footprint accounting
--------------------
For every ttnn.Tensor stored on the block, we ask the on-device buffer how
much DRAM it consumes per physical chip and sum it.  A replicated weight
contributes its full size on every chip; a sharded weight contributes
size/n_dev_along_shard_axis.

Thresholds
----------
The thresholds are computed from the BF16 + row-sharded plan documented in
ARCHITECTURE.md and budgeted in the architecture skill:

- MLP per layer per chip (gate+up+down sharded over 8 rows):    ~67 MB
- Full-attention per layer per chip (wqkvg+wo sharded over 8):  ~26 MB
- LM head (sharded over 4 cols, vocab dim):                     ~637 MB

We use generous thresholds (~2× expected) so the test catches REPLICATED
weights (which would be 8× or 32× larger) without being brittle to minor
implementation choices like fused vs split QKV.

Run
---
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    python3 -m pytest models/demos/qwen3_6_galaxy/tests/test_weight_footprint.py -x -s
"""
from __future__ import annotations

import json
import pathlib

import pytest

_SNAPSHOT_DIR = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)


# ---------------------------------------------------------------------------
# Mesh fixture (shared)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh_8x4():
    """Open the 8×4 BH GLX mesh."""
    import ttnn

    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# ---------------------------------------------------------------------------
# Weight loaders
# ---------------------------------------------------------------------------


def _load_layer_weights(layer_idx: int) -> dict:
    from safetensors.torch import load_file as load_st

    with open(_SNAPSHOT_DIR / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    pfx = f"model.language_model.layers.{layer_idx}"
    keys = [k for k in weight_map if k.startswith(pfx + ".")]
    files = sorted({weight_map[k] for k in keys})
    raw = {}
    for fn in files:
        shard = load_st(str(_SNAPSHOT_DIR / fn))
        for k in keys:
            if k in shard:
                raw[k] = shard[k].float()
    return {k[len(pfx) + 1 :]: v for k, v in raw.items()}


def _load_global_weights() -> dict:
    from safetensors.torch import load_file as load_st

    with open(_SNAPSHOT_DIR / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    needed = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]
    files = sorted({weight_map[k] for k in needed if k in weight_map})
    raw = {}
    for fn in files:
        shard = load_st(str(_SNAPSHOT_DIR / fn))
        for k in needed:
            if k in shard:
                raw[k] = shard[k].float()
    return {
        "tok_embeddings.weight": raw["model.language_model.embed_tokens.weight"],
        "norm.weight": raw["model.language_model.norm.weight"],
        "output.weight": raw["lm_head.weight"],
    }


# ---------------------------------------------------------------------------
# Footprint helper
# ---------------------------------------------------------------------------


def _bytes_per_dtype(dtype) -> float:
    """Bytes per element per device for a ttnn dtype.

    bfloat8_b/bfloat4_b: blocked formats — approximate to 1.0 / 0.5 bytes/elem.
    """
    import ttnn

    return {
        ttnn.bfloat16: 2.0,
        ttnn.bfloat8_b: 1.0625,  # 16 fp8 + 4 fp16 scale / 16 elems
        ttnn.bfloat4_b: 0.5625,
        ttnn.float32: 4.0,
        ttnn.uint32: 4.0,
        ttnn.uint16: 2.0,
        ttnn.uint8: 1.0,
    }.get(dtype, 2.0)


def _per_chip_bytes(t) -> int:
    """Return bytes one physical chip allocates for ttnn.Tensor `t`.

    Strategy: get the per-device tensor shape (= logical shape if replicated, or
    sharded slice if sharded), multiply by bytes-per-element of the dtype.

    Mesh-aware: ``ttnn.get_device_tensors(t)`` returns a list of per-device
    tensors. Take the first one's shape — for both replicated and sharded the
    per-chip allocation is the same shape across chips.
    """
    import ttnn

    try:
        per_dev = ttnn.get_device_tensors(t)[0]
        per_shape = per_dev.shape
    except Exception:
        per_shape = t.shape

    n_elem = 1
    for d in per_shape:
        n_elem *= d
    return int(round(n_elem * _bytes_per_dtype(t.dtype)))


def _sum_block_weight_bytes_per_device(block) -> tuple[int, list[str]]:
    """Return (total_bytes_per_device, breakdown_lines).

    For every ttnn.Tensor on `block`, sum the **per-physical-chip** allocation.
    Replicated weights count their full size; sharded weights count their
    per-chip slice.
    """
    import ttnn

    lines = []
    total = 0
    seen_ids = set()
    for name in dir(block):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(block, name)
        except Exception:
            continue
        candidates = []
        if isinstance(attr, ttnn.Tensor):
            candidates.append((name, attr))
        elif isinstance(attr, (list, tuple)):
            for i, x in enumerate(attr):
                if isinstance(x, ttnn.Tensor):
                    candidates.append((f"{name}[{i}]", x))
        for n, t in candidates:
            if id(t) in seen_ids:
                continue
            seen_ids.add(id(t))
            per_chip = _per_chip_bytes(t)
            total += per_chip
            lines.append(f"    {n:35s} logical={list(t.shape)} dtype={t.dtype} -> {per_chip/1e6:.2f} MB/chip")
    return total, lines


# ---------------------------------------------------------------------------
# Test A1: MLP per-layer per-device footprint
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_mlp_per_device_footprint_under_threshold(mesh_8x4):
    """A correctly-sharded MLP block uses < 100 MB/chip for one layer's weights.

    Replicated BF16: 3 × (5120×17408×2 B) = 534 MB/chip   ← current (wrong)
    Sharded /8 BF16: 3 × (5120×2176×2 B)  =  67 MB/chip   ← correct
    Threshold: 100 MB/chip catches replication, allows sharded plan.
    """
    from models.demos.qwen3_6_galaxy.tt.llama_mlp import TtQwen36MLP
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh_8x4)
    sd = _load_layer_weights(0)  # any layer's MLP weights

    mlp = TtQwen36MLP(mesh_device=mesh_8x4, args=args, state_dict=sd)

    total, lines = _sum_block_weight_bytes_per_device(mlp)
    print(f"\n[MLP] per-chip weight DRAM: {total/1e6:.1f} MB")
    for line in lines:
        print(line)

    assert total < 100 * 1e6, (
        f"MLP weights consume {total/1e6:.1f} MB/chip — likely REPLICATED. "
        f"Expected ≤100 MB/chip (sharded BF16: ~67 MB). "
        f"Use ShardTensor2dMesh with cluster_axis=0 (rows) on the intermediate dim."
    )


# ---------------------------------------------------------------------------
# Test A2: Full-attention per-layer per-device footprint
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_attention_per_device_footprint_under_threshold(mesh_8x4):
    """A correctly-sharded full_attention block uses < 60 MB/chip for one layer.

    Replicated BF16: wqkvg 147 MB + wo 31 MB = 178 MB/chip   ← current (wrong)
    Sharded /8 BF16: wqkvg ~18 MB + wo ~8 MB =  ~26 MB/chip  ← correct
    Threshold: 60 MB/chip catches replication, allows sharded plan + KV cache scratch.
    """
    from models.demos.qwen3_6_galaxy.tt.llama_attention import TtQwen36GatedAttention
    from models.demos.qwen3_6_galaxy.tt.llama_rope import Qwen36RopeSetup
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh_8x4)

    # Find first full_attention layer
    full_layer_idx = next(i for i, lt in enumerate(args.linear_attention_pattern) if lt == "full_attention")
    print(f"[attention footprint] using layer {full_layer_idx} (first full_attention)")

    sd = _load_layer_weights(full_layer_idx)
    # Strip "self_attn." prefix the attention block expects
    sd_attn = {k[len("self_attn.") :]: v for k, v in sd.items() if k.startswith("self_attn.")}

    rope_setup = Qwen36RopeSetup(
        mesh_device=mesh_8x4, args=args, batch_size=args.max_batch_size, max_seq_len=args.max_seq_len
    )
    attn = TtQwen36GatedAttention(
        mesh_device=mesh_8x4, args=args, state_dict=sd_attn, layer_num=full_layer_idx, rope_setup=rope_setup
    )

    total, lines = _sum_block_weight_bytes_per_device(attn)
    # KV cache (layer_past) is an unavoidable per-chip cost but counted as a weight here;
    # subtract it for the matmul-weight threshold check.
    print(f"\n[Attention] per-chip total tensor DRAM (incl KV): {total/1e6:.1f} MB")
    for line in lines:
        print(line)

    # Sum only the matmul weights (wqkvg, wo) — KV cache is separate.
    matmul_bytes = 0
    for name in ("wqkvg", "wo"):
        t = getattr(attn, name, None)
        if t is not None:
            matmul_bytes += _per_chip_bytes(t)
    print(f"[Attention] per-chip matmul-weight DRAM (wqkvg+wo): {matmul_bytes/1e6:.1f} MB")

    assert matmul_bytes < 60 * 1e6, (
        f"Attention matmul weights consume {matmul_bytes/1e6:.1f} MB/chip — likely REPLICATED. "
        f"Expected ≤60 MB/chip (sharded BF16: ~26 MB). "
        f"Use ShardTensor2dMesh for wqkvg and wo on the head/output dim."
    )


# ---------------------------------------------------------------------------
# Test A3: LM head per-device footprint
# ---------------------------------------------------------------------------


@pytest.mark.hardware
def test_lm_head_per_device_footprint_under_threshold(mesh_8x4):
    """A correctly-sharded LM head uses < 800 MB/chip.

    Replicated BF16: 248832 × 5120 × 2 B    = 2548 MB/chip   ← current (wrong)
    Sharded /4 BF16: 248832 × 5120 × 2 / 4  =  637 MB/chip   ← correct
    Threshold: 800 MB/chip catches replication.
    """
    from models.demos.qwen3_6_galaxy.tt.llama_model import TtQwen36Transformer
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh_8x4)
    gw = _load_global_weights()
    # Load only 1 decoder layer to keep the test cheap — we're only inspecting the head.
    layer0 = _load_layer_weights(0)
    model = TtQwen36Transformer(
        mesh_device=mesh_8x4, args=args, global_weights=gw, layers_weights=[layer0], num_layers=1
    )

    head = model.lm_head_weight
    per_chip = _per_chip_bytes(head)
    print(f"\n[LM head] logical={list(head.shape)} dtype={head.dtype} -> {per_chip/1e6:.1f} MB/chip")

    assert per_chip < 800 * 1e6, (
        f"LM head consumes {per_chip/1e6:.1f} MB/chip — REPLICATED into a 248K-vocab head. "
        f"Expected ≤800 MB/chip (sharded /4 cols BF16: ~637 MB). "
        f"Shard the vocab dim across cluster_axis=1."
    )
