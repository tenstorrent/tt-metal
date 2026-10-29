# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gradient check of the TP=4 LoRA paths (transposed frozen weights, fused update, column- and row-parallel tails)
against the single-chip path on an identical model.

Run once per mesh with the same seed; the second run compares its adapter gradients and loss with the first's:

    python tp_gradient_check.py --tp 1 --out single.npz
    python tp_gradient_check.py --tp 4 --out tp4.npz --compare single.npz

A 1-layer Mistral at the real shapes, every parameter assigned from a seeded numpy init (sharded with the
parameter's own mapper under TP), LoRA B non-zero so dA is non-trivial, fixed token ids; the loss and the gathered
dA / dB of every adapter are compared with PCC and relative L2 error at the harness bars.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import zlib

import numpy as np

import ttnn
import ttml
from ttml.common.utils import get_loss_over_devices
from ttml.models import EmbeddingPlacement
from ttml.models.mistral import MistralConfig
from ttml.models.mistral import Mistral
from ttml.modules import LoraConfig, LoraModel, fuse_lora_into_frozen_weights, transpose_frozen_weights

HF = {
    "model_type": "mistral",
    "hidden_size": 5120,
    "intermediate_size": 32768,
    "num_hidden_layers": 1,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 131072,
    "rms_norm_eps": 1e-5,
    "rope_theta": 1e8,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "hidden_act": "silu",
}
SEQ, RANK, ALPHA = 1024, 16, 32.0
TARGETS = [r"attention\.qkv_linear$", r"attention\.out_linear$", r"mlp\.w_gate_up$", r"mlp\.w2$"]


# Fused parameters whose rows are rank-interleaved under TP (the loader's ``_assemble``): on chip c the local shard
# reads [Q_c | K_c | V_c] / [gate_c | up_c], so the global array is [Q_0 | K_0 | V_0 | Q_1 | ...]. The seeded
# single-chip-order array must be permuted into that order (and gathered gradients permuted back).
INTERLEAVED = {
    "attention/qkv_linear/weight": (4096, 1024, 1024),
    "attention/qkv_linear/lora_B": (4096, 1024, 1024),
    "mlp/w_gate_up/weight": (32768, 32768),
    "mlp/w_gate_up/lora_B": (32768, 32768),
}


def block_rows_of(name: str):
    for key, rows in INTERLEAVED.items():
        if name.endswith(key):
            return rows
    return None


def to_tp_order(arr: np.ndarray, block_rows, tp: int) -> np.ndarray:
    blocks = np.split(arr, np.cumsum(block_rows)[:-1], axis=2)
    per_block = [np.split(b, tp, axis=2) for b in blocks]
    return np.concatenate([slices[rank] for rank in range(tp) for slices in per_block], axis=2)


def from_tp_order(arr: np.ndarray, block_rows, tp: int) -> np.ndarray:
    shard_rows = sum(block_rows) // tp
    recovered = [[] for _ in block_rows]
    for rank in range(tp):
        shard = arr[:, :, rank * shard_rows : (rank + 1) * shard_rows]
        offset = 0
        for i, rows in enumerate(block_rows):
            recovered[i].append(shard[:, :, offset : offset + rows // tp])
            offset += rows // tp
    return np.concatenate([np.concatenate(parts, axis=2) for parts in recovered], axis=2)


def seeded_value(name: str, shape, rng: np.random.Generator) -> np.ndarray:
    fan_in = shape[-1]
    if name.endswith("/gamma"):
        return rng.normal(1.0, 0.1, shape).astype(np.float32)
    if "lora_A" in name:
        return rng.uniform(-1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in), shape).astype(np.float32)
    if "lora_B" in name:
        return rng.normal(0.0, 0.02, shape).astype(np.float32)  # non-zero: dA must be non-trivial
    if "tok_emb" in name or name.endswith("fc/weight"):
        return rng.normal(0.0, 0.02, shape).astype(np.float32)
    return rng.normal(0.0, 1 / math.sqrt(fan_in), shape).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--out", required=True)
    p.add_argument("--compare", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-fuse", action="store_true")
    p.add_argument("--no-transpose", action="store_true", help="composite-phase path: weights [out, in], LoraApply")
    p.add_argument(
        "--sgd-lr",
        type=float,
        default=0.0,
        help="after the backward, take one SGD step and re-evaluate the loss on the same batch",
    )
    args = p.parse_args()

    use_tp = args.tp > 1
    if use_tp:
        mesh = ttml.Mesh((1, args.tp), ("dp", "tp"))
        ttml.open_device_mesh(mesh)
    else:
        ttml.autograd.AutoContext.get_instance().open_device()
    config = MistralConfig.from_hf(HF, max_position_embeddings=SEQ)
    if use_tp:
        config = dataclasses.replace(config, use_tp=True, embedding_placement=EmbeddingPlacement.VocabParallel)
    model = LoraModel(Mistral(config), LoraConfig(rank=RANK, alpha=ALPHA, target_modules=TARGETS))

    # Identical parameters on every mesh: the global array is generated per name from one seed and sharded by the
    # parameter's own mapper (the loader's path), so a 4-chip model holds the same weights as the 1-chip one.
    params = model.parameters()
    for name in sorted(params):
        tensor = params[name]
        sharding = ttml.Sharding.from_tensor(tensor)
        shape = list(tensor.shape())
        placements = sharding.placements
        if use_tp and placements:
            axis = mesh.axis_index("tp")
            placement = placements[axis] if axis < len(placements) else None
            if isinstance(placement, ttnn.PlacementShard):
                shape[placement.dim] *= args.tp
        value = seeded_value(name, tuple(shape), np.random.default_rng(zlib.crc32(name.encode()) + args.seed))
        rows = block_rows_of(name)
        if use_tp and rows is not None:
            value = to_tp_order(value, rows, args.tp)
        ttml.autograd.assign_numpy(tensor, value, mapper=sharding.derive_mapper())
    converted = [] if args.no_transpose else transpose_frozen_weights(model)
    fused = [] if (args.no_fuse or args.no_transpose) else fuse_lora_into_frozen_weights(model)
    print(f"tp {args.tp}: transposed {len(converted)}, fused {len(fused)}")
    model.train()

    rng = np.random.default_rng(args.seed)
    ids = rng.integers(0, HF["vocab_size"], (1, 1, 1, SEQ), dtype=np.uint32)
    targets = rng.integers(0, HF["vocab_size"], (1, SEQ), dtype=np.uint32)
    x = ttml.autograd.Tensor.from_numpy(ids, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
    y = ttml.autograd.Tensor.from_numpy(targets, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
    logits = model(x, None)
    if use_tp:
        loss = ttml.ops.distributed.vocab_parallel_cross_entropy_loss(logits, y, cluster_axis=mesh.axis_index("tp"))
    else:
        loss = ttml.ops.loss.cross_entropy_loss(logits, y, ttml.ops.ReduceType.MEAN)
    loss.backward(False)
    loss_val = float(get_loss_over_devices(loss)) if use_tp else float(loss.to_numpy(ttnn.DataType.FLOAT32).item())

    out = {"loss": np.array([loss_val])}
    for name, tensor in params.items():
        if tensor.get_requires_grad():
            sharding = ttml.Sharding.from_tensor(tensor)
            grad = ttml.autograd.create_tensor(tensor.get_grad(), False)
            if use_tp:
                # Control: the parameter gathered the same way must equal the seeded value on every mesh.
                rows = block_rows_of(name)
                gathered_param = sharding.gather(tensor).astype(np.float32)
                gathered_grad = sharding.gather(grad).astype(np.float32)
                if rows is not None:
                    gathered_param = from_tp_order(gathered_param, rows, args.tp)
                    gathered_grad = from_tp_order(gathered_grad, rows, args.tp)
                out["param:" + name] = gathered_param
                out[name] = gathered_grad
            else:
                out["param:" + name] = tensor.to_numpy(ttnn.DataType.FLOAT32)
                out[name] = grad.to_numpy(ttnn.DataType.FLOAT32)
    np.savez(args.out, **out)
    print(f"saved {len(out) - 1} gradients, loss {loss_val:.4f}")

    if args.sgd_lr > 0:
        # Descent check: loss(p - lr g) - loss(p) must be about -lr * |g|^2 if g is the gradient of this loss.
        g2 = sum(
            float(np.square(out[k].astype(np.float64)).sum())
            for k in out
            if k != "loss" and not k.startswith(("param:", "raw:"))
        )
        ttml.autograd.AutoContext.get_instance().reset_graph()
        opt = ttml.optimizers.SGD(params, ttml.optimizers.SGDConfig.make(args.sgd_lr, 0.0, 0.0, 0.0, False))
        opt.step()
        logits2 = model(x, None)
        if use_tp:
            loss2 = ttml.ops.distributed.vocab_parallel_cross_entropy_loss(
                logits2, y, cluster_axis=mesh.axis_index("tp")
            )
        else:
            loss2 = ttml.ops.loss.cross_entropy_loss(logits2, y, ttml.ops.ReduceType.MEAN)
        loss2_val = (
            float(get_loss_over_devices(loss2)) if use_tp else float(loss2.to_numpy(ttnn.DataType.FLOAT32).item())
        )
        print(
            f"descent check: loss {loss_val:.4f} -> {loss2_val:.4f} after one SGD step (lr {args.sgd_lr}); measured drop {loss_val - loss2_val:+.4f}, first-order prediction {args.sgd_lr * g2:+.4f}"
        )

    if args.compare:
        ref = np.load(args.compare)
        print("tensor | shape | PCC | rel err")
        bad = []
        for name in sorted(out):
            if name.startswith("raw:") or name not in ref:
                continue
            a, b = out[name].astype(np.float64).ravel(), ref[name].astype(np.float64).ravel()
            if a.shape != b.shape:
                print(f"  {name:55s} shape {out[name].shape} vs ref {ref[name].shape}")
                bad.append(name)
                continue
            pcc = 1.0 if name == "loss" else float(np.corrcoef(a, b)[0, 1])
            rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
            bar = (
                (0.0, 1e-2)
                if name == "loss"
                else (
                    (0.999999, 1e-6)
                    if name.startswith("param:")
                    else ((0.98, 8e-2) if "attention" in name else (0.99, 5e-2))
                )
            )
            ok = pcc >= bar[0] and rel <= bar[1]
            print(f"  {name:55s} {str(out[name].shape):18s} {pcc:.6f} {rel:.2e} {'ok' if ok else 'FAIL'}")
            if not ok:
                bad.append(name)
        print("PASS" if not bad else f"FAIL: {bad}")
    ttml.autograd.AutoContext.get_instance().reset_graph()
    if use_tp:
        ttml.close_device_mesh()
    else:
        ttml.autograd.AutoContext.get_instance().close_device()


if __name__ == "__main__":
    main()
