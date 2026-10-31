# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Numpy-only loading of the diffusers MiniMax-H3 transformer checkpoint into the ttml model, with the
host-side transforms the device layout needs (q/k rope pair order, per-shard SwiGLU halves, TP shards)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import ml_dtypes
import numpy as np

from .packing import qk_channel_permutation

TIME_EMBEDDER_KEYS = (
    "time_embedder.linear_1.weight",
    "time_embedder.linear_1.bias",
    "time_embedder.linear_2.weight",
    "time_embedder.linear_2.bias",
)


def _attn_plan(prefix_hf: str, prefix_tt: str) -> dict[str, tuple[tuple[str, ...], int | None]]:
    return {
        f"{prefix_tt}attn/to_q/weight": ((f"{prefix_hf}attn.to_q.weight",), 2),
        f"{prefix_tt}attn/to_k/weight": ((f"{prefix_hf}attn.to_k.weight",), 2),
        f"{prefix_tt}attn/to_v/weight": ((f"{prefix_hf}attn.to_v.weight",), 2),
        f"{prefix_tt}attn/to_out/weight": ((f"{prefix_hf}attn.to_out.0.weight",), 3),
        f"{prefix_tt}attn/norm_q/weight": ((f"{prefix_hf}attn.norm_q.weight",), None),
        f"{prefix_tt}attn/norm_k/weight": ((f"{prefix_hf}attn.norm_k.weight",), None),
        f"{prefix_tt}norm1/weight": ((f"{prefix_hf}norm1.weight",), None),
        f"{prefix_tt}norm2/weight": ((f"{prefix_hf}norm2.weight",), None),
        f"{prefix_tt}ff/proj/weight": ((f"{prefix_hf}ff.net.0.proj.weight",), 2),
        f"{prefix_tt}ff/out/weight": ((f"{prefix_hf}ff.net.2.weight",), 3),
    }


def weight_plan(config, root: str) -> dict[str, tuple[tuple[str, ...], int | None]]:
    """ttml param name -> (HF key(s), TP-sharded tensor dim or None for replicated).

    Only parameters the device model holds; `time_embedder.*` stays on the host in fp32."""
    plan = {
        f"{root}/proj_in/weight": (("proj_in.weight",), None),
        f"{root}/proj_in/bias": (("proj_in.bias",), None),
        f"{root}/audio_proj_in/weight": (("audio_proj_in.weight",), None),
        f"{root}/audio_proj_in/bias": (("audio_proj_in.bias",), None),
        f"{root}/context_embedder/weight": (("context_embedder.weight",), None),
        f"{root}/context_embedder/bias": (("context_embedder.bias",), None),
        f"{root}/refiner_norm/weight": (("token_refiner.final_norm.weight",), None),
        f"{root}/norm_out/weight": (("norm_out.norm.weight",), None),
        f"{root}/norm_out_linear/weight": (("norm_out.linear.weight",), 2),
        f"{root}/norm_out_linear/bias": (("norm_out.linear.bias",), 3),
        f"{root}/proj_out/weight": (("proj_out.weight",), None),
        f"{root}/proj_out/bias": (("proj_out.bias",), None),
        f"{root}/audio_proj_out/weight": (("audio_proj_out.weight",), None),
        f"{root}/audio_proj_out/bias": (("audio_proj_out.bias",), None),
    }
    for i in range(config.num_refiner_layers):
        plan.update(_attn_plan(f"token_refiner.refiner_blocks.{i}.", f"{root}/refiner_blocks/{i}/"))
    for i in range(config.num_layers):
        plan.update(_attn_plan(f"transformer_blocks.{i}.", f"{root}/blocks/{i}/"))
        plan[f"{root}/blocks/{i}/adaln/weight"] = ((f"transformer_blocks.{i}.adaln_proj.linear.weight",), 2)
        plan[f"{root}/blocks/{i}/adaln/bias"] = ((f"transformer_blocks.{i}.adaln_proj.linear.bias",), 3)
    return plan


def permute_qk_rows(w: np.ndarray, num_heads: int, head_dim: int, rotary_dim: int) -> np.ndarray:
    """Reorder output rows (or a per-head 1-D weight) into the adjacent-pair rotary layout."""
    perm = qk_channel_permutation(head_dim, rotary_dim)
    if w.ndim == 1:
        return w.reshape(-1, head_dim)[:, perm].reshape(-1)
    return w.reshape(num_heads, head_dim, -1)[:, perm, :].reshape(w.shape)


def unpermute_qk_rows(w: np.ndarray, num_heads: int, head_dim: int, rotary_dim: int) -> np.ndarray:
    """Inverse of `permute_qk_rows`, for exporting LoRA B rows of to_q / to_k back to the diffusers layout."""
    perm = qk_channel_permutation(head_dim, rotary_dim)
    inv = np.argsort(perm)
    if w.ndim == 1:
        return w.reshape(-1, head_dim)[:, inv].reshape(-1)
    return w.reshape(num_heads, head_dim, -1)[:, inv, :].reshape(w.shape)


def interleave_halves_per_shard(w: np.ndarray, tp: int) -> np.ndarray:
    """[value (F) | gate (F)] rows -> [gate_s0 | value_s0 | gate_s1 | ...]: each TP shard holds both halves, gate
    first as `swiglu_packed` reads them."""
    rows = w.shape[0]
    half = rows // 2
    per = half // tp
    value, gate = w[:half], w[half:]
    return np.concatenate(
        [np.concatenate([gate[s * per : (s + 1) * per], value[s * per : (s + 1) * per]]) for s in range(tp)]
    )


def prepare_weight(name: str, arrays: tuple[np.ndarray, ...], config, tp: int) -> np.ndarray:
    """One ttml parameter's global 4-D bf16 array from its HF array(s)."""
    w = arrays[0]
    if name.endswith("attn/to_q/weight") or name.endswith("attn/to_k/weight"):
        w = permute_qk_rows(w, config.num_attention_heads, config.attention_head_dim, config.rotary_dim)
    elif name.endswith("attn/norm_q/weight") or name.endswith("attn/norm_k/weight"):
        w = permute_qk_rows(w, 1, config.attention_head_dim, config.rotary_dim)
    elif name.endswith("ff/proj/weight"):
        w = interleave_halves_per_shard(w, tp)
    w = w.reshape(1, 1, 1, -1) if w.ndim == 1 else w.reshape(1, 1, *w.shape)
    return np.ascontiguousarray(w.astype(ml_dtypes.bfloat16, copy=False))


def read_time_embedder(weights_dir: Path) -> dict[str, np.ndarray]:
    """The fp32 timestep MLP, kept on the host."""
    from safetensors import safe_open

    weights_dir = Path(weights_dir)
    index = json.loads((weights_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for key in TIME_EMBEDDER_KEYS:
        with safe_open(str(weights_dir / index[key]), framework="np") as f:
            out[key] = f.get_tensor(key).astype(np.float32)
    return out


def load_transformer(model, weights_dir: Path, only: tuple[str, ...] | None = None, verbose: bool = True) -> int:
    """Fill the ttml model from the HF `transformer/` shards; numpy only, one shard open at a time.

    `only`: load just the parameters whose ttml name (after the root) starts with one of these prefixes."""
    import ttnn
    import ttml
    from safetensors import safe_open

    weights_dir = Path(weights_dir)
    index = json.loads((weights_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    params = model.parameters()
    root = next(iter(params)).split("/", 1)[0]
    mesh = ttml.maybe_mesh() if hasattr(ttml, "maybe_mesh") else None
    tp = mesh.axis_size("tp") if (mesh is not None and model.config.use_tp) else 1
    device = ttml.autograd.AutoContext.get_instance().get_device()
    replicate = ttml.core.distributed.replicate_tensor_to_mesh_mapper(device)

    plan = weight_plan(model.config, root)
    if only is not None:
        plan = {n: v for n, v in plan.items() if any(n[len(root) + 1 :].startswith(p) for p in only)}
    missing = [n for n in plan if n not in params]
    if missing:
        raise KeyError(f"model lacks {missing[:4]}... ; model has {sorted(params)[:4]}...")

    by_shard: dict[str, list] = {}
    for name, (hf_keys, tdim) in plan.items():
        by_shard.setdefault(index[hf_keys[0]], []).append((name, hf_keys, tdim))
    loaded = 0
    timing = {"read": 0.0, "prepare": 0.0, "upload": 0.0}
    for shard, entries in sorted(by_shard.items()):
        with safe_open(str(weights_dir / shard), framework="np") as f:
            for name, hf_keys, tdim in entries:
                t0 = time.perf_counter()
                arrays = tuple(f.get_tensor(k) for k in hf_keys)
                t1 = time.perf_counter()
                w = prepare_weight(name, arrays, model.config, tp)
                expected = list(params[name].shape())
                if tdim is not None and tp > 1:
                    expected[tdim] *= tp
                if list(w.shape) != expected:
                    raise ValueError(f"{name}: checkpoint {list(w.shape)} vs model {expected}")
                mapper = mesh.axis_mapper("tp", tdim=tdim) if (tdim is not None and tp > 1) else replicate
                t2 = time.perf_counter()
                params[name].assign(
                    ttml.autograd.Tensor.from_numpy(w, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)
                )
                t3 = time.perf_counter()
                timing["read"] += t1 - t0
                timing["prepare"] += t2 - t1
                timing["upload"] += t3 - t2
                loaded += 1
        t4 = time.perf_counter()
        ttnn.synchronize_device(device)  # uploads are queued; charge the wait to "upload", not the next shard
        timing["upload"] += time.perf_counter() - t4
        if verbose:
            print(
                f"  {shard}: {len(entries)} params ({loaded}/{len(plan)}) | read {timing['read']:.0f} s "
                f"prepare {timing['prepare']:.0f} s upload {timing['upload']:.0f} s",
                flush=True,
            )
    return loaded
