# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Verify the merged-gate (.gm) transformer weight cache against a fresh fold of the checkpoint.

Every gate-merge test folds the gate into the projection FRESH from safetensors. The pipeline does
not: it loads pre-folded tensors from TT_DIT_CACHE_DIR/<ckpt>.gm/transformer/<key>/*.tensorbin. That
disk cache is therefore the one thing the whole passing test suite never exercises — and the merged
pipeline deviates from the standalone one (frame-PCC 0.868) while every test says the fold is exact.

So: read the cached device shards back and compare them, shard by shard, against what the current
_prepare_torch_state / _fold_gate_into_projection actually produce. Requires a device only because
ttnn.load_tensor needs a cluster descriptor; no compute is done on it.

    python -m models.tt_dit.tests.models.ltx.verify_gm_cache
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
from safetensors import safe_open

import ttnn
from models.tt_dit.models.transformers.ltx.attention_ltx import LTXAttention

TP = 4
SP = 8
MESH = (4, 8)  # tp_axis=0, sp_axis=1
NUM_HEADS = 32
VIDEO_DIM = 4096
AUDIO_DIM = 2048

# (module, is_self, dim, query_input_dim) for the six gated attentions.
ATTNS = [
    ("attn1", True, VIDEO_DIM, VIDEO_DIM),
    ("attn2", False, VIDEO_DIM, VIDEO_DIM),
    ("audio_attn1", True, AUDIO_DIM, AUDIO_DIM),
    ("audio_attn2", False, AUDIO_DIM, AUDIO_DIM),
    ("audio_to_video_attn", False, AUDIO_DIM, VIDEO_DIM),
    ("video_to_audio_attn", False, AUDIO_DIM, AUDIO_DIM),
]


def fresh_fold(state: dict[str, torch.Tensor], is_self: bool, dim: int) -> dict[str, torch.Tensor]:
    """Run production's _prepare_torch_state (pure torch) on a raw checkpoint substate."""
    stub = SimpleNamespace(
        dim=dim,
        num_heads=NUM_HEADS,
        head_dim=dim // NUM_HEADS,
        n_local_heads=NUM_HEADS // TP,
        is_self=is_self,
        base_chunks=3 if is_self else 1,
        merge_gate=True,
        parallel_config=SimpleNamespace(tensor_parallel=SimpleNamespace(factor=TP)),
    )
    stub._fold_gate_into_projection = lambda s, p: LTXAttention._fold_gate_into_projection(stub, s, p)
    LTXAttention._prepare_torch_state(stub, state)
    return state


def main() -> int:
    ckpt = os.environ["LTX_CHECKPOINT"]
    root = os.environ["TT_DIT_CACHE_DIR"]
    base = os.path.basename(ckpt).removesuffix(".safetensors")
    gm = os.path.join(root, f"{base}.gm", "transformer", "CP1_0_TP4_0_SP8_1_mesh4x8_bf16")
    print(f"checkpoint : {ckpt}")
    print(f"gm cache   : {gm}")
    assert os.path.isdir(gm), gm

    blocks = [int(b) for b in os.environ.get("VERIFY_BLOCKS", "0,1,47").split(",")]
    bad = []

    with safe_open(ckpt, framework="pt") as f:
        keys = list(f.keys())
        for blk in blocks:
            pre = f"model.diffusion_model.transformer_blocks.{blk}."
            for name, is_self, dim, q_in in ATTNS:
                sub_pre = pre + name + "."
                state = {k[len(sub_pre) :]: f.get_tensor(k) for k in keys if k.startswith(sub_pre)}
                if "to_gate_logits.weight" not in state:
                    continue
                folded = fresh_fold(state, is_self, dim)
                proj = "to_qkv" if is_self else "to_q"

                for suffix in ("weight", "bias"):
                    key = f"{proj}.{suffix}"
                    if key not in folded:
                        continue
                    t = folded[key]
                    # ColParallelLinear._prepare_torch_state: weight -> [in, out]; bias -> [1, out].
                    exp = t.transpose(0, 1).contiguous() if suffix == "weight" else t.reshape(1, -1)

                    path = os.path.join(gm, f"transformer_blocks.{blk}.{name}.{key}.tensorbin")
                    if not os.path.exists(path):
                        bad.append(f"block{blk}.{name}.{key}: MISSING from cache")
                        continue
                    shards = ttnn.get_device_tensors(ttnn.load_tensor(path, device=None))

                    out_per_dev = exp.shape[-1] // TP
                    n_bad = 0
                    worst = 0.0
                    for d, sh in enumerate(shards):
                        tp = d // SP  # mesh 4x8: tp_axis=0 is the slow axis
                        got = ttnn.to_torch(sh).float().reshape(exp.shape[0], -1)
                        want = exp[:, tp * out_per_dev : (tp + 1) * out_per_dev].to(torch.bfloat16).float()
                        if not torch.equal(got, want):
                            n_bad += 1
                            worst = max(worst, (got - want).abs().max().item())
                    status = "OK" if n_bad == 0 else f"MISMATCH {n_bad}/{len(shards)} shards, max_abs={worst:.4g}"
                    print(f"  block{blk:<3d} {name:<22s} {key:<14s} {tuple(exp.shape)}  {status}")
                    if n_bad:
                        bad.append(f"block{blk}.{name}.{key}: {status}")

    print()
    if bad:
        print(f"FAIL: the .gm cache does NOT match a fresh fold ({len(bad)} tensors)")
        for b in bad:
            print("   " + b)
        return 1
    print("PASS: every cached .gm projection tensor is bit-identical to a fresh fold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
