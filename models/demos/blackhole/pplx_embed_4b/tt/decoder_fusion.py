# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fuse each decoder layer's residual adds with the RMSNorm that follows them (batched prefill).

Per layer the stock ``TransformerBlock.forward`` runs ``ttnn.add`` (post-attention residual)
then ``ff_norm``, and ``ttnn.add`` (post-MLP residual) whose result the *next* layer's
``attention_norm`` consumes. ``custom_ops.fused_add_rmsnorm`` produces the sum and the
normalised tensor in one pass, so both pairs collapse into one op each (72 launches and one
DRAM pass over ``[M, dim]`` per pair removed). Installed by wrapping ``layer.forward``: during
the wrapped call ``ttnn.add`` is intercepted for the two residual-shaped adds and the norm
attributes hand back the precomputed result. bs=1 (M=512 rows) keeps the stock ops: with 16
tile-rows only 16 cores work and the fused op is slower there (62.8 vs 52.9 us standalone).
Enable with ``QWEN_FUSED_ADD_NORM=1``.
"""
import os

import ttnn
from models.demos.blackhole.pplx_embed_4b.tt.custom_ops.fused_add_rmsnorm import (
    fused_add_rmsnorm,
    fused_add_rmsnorm_split,
    make_add_norm_constants,
    pick_split,
    supported,
)
from models.tt_transformers.tt.common import Mode

# Flattened rows below which the stock add + norm are kept (bs1 ISL512 = 512 rows).
_MIN_ROWS = int(
    os.getenv("QWEN_FUSED_ADD_NORM_MIN_ROWS", "8192")
)  # flattened M: bs16+ at ISL512 (bs8 measured +1.7%, bs1 slower standalone)


def _gamma(distributed_norm):
    return ttnn.to_torch(distributed_norm.norm.weight).flatten()


def install_decoder_fusion(model) -> int:
    """Wrap every layer of ``model``; returns the number of layers wrapped."""
    layers = model.layers
    device = model.mesh_device
    consts = []
    for layer in layers:
        eps = layer.ff_norm.norm.eps
        consts.append(
            (
                make_add_norm_constants(_gamma(layer.ff_norm), eps, device),
                make_add_norm_constants(_gamma(layer.attention_norm), layer.attention_norm.norm.eps, device),
            )
        )
    # id(residual out) -> (residual out, normalised tensor) for the next layer's attention_norm.
    # The tensor itself is kept so a recycled id() can never match a stale entry; layer 0
    # clears whatever a previous forward left behind.
    stash = {}
    verify = os.getenv("QWEN_FUSED_ADD_NORM_VERIFY", "0") == "1"
    for i, layer in enumerate(layers):
        next_attn = consts[i + 1][1] if i + 1 < len(layers) else None
        next_layer = layers[i + 1] if i + 1 < len(layers) else None
        _wrap_layer(
            layer,
            consts[i][0],
            next_attn,
            stash,
            is_first=(i == 0),
            verify=(verify, i, next_layer),
            model_args=getattr(model, "args", None),
        )
    return len(layers)


def _pcc(t, r):
    import torch

    a, b = ttnn.to_torch(t).float().flatten(), ttnn.to_torch(r).float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _forward_resid_sharded(layer, orig_forward, model_args, has_next_layer, x, args, kwargs):
    """bs1 (few rows, no fused add+norm): write the two residual adds straight in the block-shard layout the
    prefill RMSNorm reads (10x8 grid), so the interleaved-to-sharded op in front of each norm becomes a no-op
    (``to_memory_config`` hands the tensor back). ``ttnn.add`` with one block-sharded and one interleaved
    input is bit-identical to add + I2S (standalone 11.3 vs 10.7 + 8.7 us). The last layer keeps its
    interleaved output for the final norm / pooling path. Opt out: QWEN_BS1_RESID_SHARDED=0."""
    cfg = model_args.get_prefill_block_sharded_norm_config(int(x.padded_shape[-2]))
    if cfg is None:
        return orig_forward(x, *args, **kwargs)
    shard_mc = cfg["sharded_input_mem_cfg"]
    shape = list(x.padded_shape)
    calls = [0]
    orig_add = ttnn.add

    def add_wrapper(a, b, *a_args, **a_kwargs):
        if (
            a_args
            or not hasattr(a, "padded_shape")
            or not hasattr(b, "padded_shape")
            or list(a.padded_shape) != shape
            or list(b.padded_shape) != shape
            or any(k not in ("memory_config", "dtype") for k in a_kwargs)
        ):
            return orig_add(a, b, *a_args, **a_kwargs)
        calls[0] += 1
        if calls[0] == 1 or (calls[0] == 2 and has_next_layer):
            a_kwargs = dict(a_kwargs, memory_config=shard_mc)
        return orig_add(a, b, **a_kwargs)

    ttnn.add = add_wrapper
    try:
        return orig_forward(x, *args, **kwargs)
    finally:
        ttnn.add = orig_add


def _wrap_layer(layer, ff_consts, next_attn_consts, stash, is_first=False, verify=(False, 0, None), model_args=None):
    orig_forward = layer.forward
    orig_ff_norm = layer.ff_norm
    orig_attn_norm = layer.attention_norm
    do_verify, layer_idx, next_layer = verify

    def forward(x, *args, **kwargs):
        mode = kwargs.get("mode", args[4] if len(args) > 4 else "decode")  # (current_pos, rot_g, rot_l, user_id, mode)
        is_prefill = mode == Mode.PREFILL or mode == "prefill"
        if not is_prefill or not supported(x, x):
            return orig_forward(x, *args, **kwargs)
        rows = int(x.padded_shape[-2]) * int(x.padded_shape[-3]) * int(x.padded_shape[0])
        fuse = None
        if rows >= _MIN_ROWS:
            # QWEN_FUSED_ADD_NORM_R >= 2: split every row over R cores (multi-wave row-split kernel). With
            # 128-512 tile-rows on 120 cores the row-granular kernel leaves most cores idle in the last
            # wave; R=4-5 balances it: standalone M=4096 184 -> 141 us, M=8192 320 -> 242, M=16384 597 -> 459
            # (stock add + rms_norm -> split), vs 178 / 299 / 530 for the row-granular fused kernel.
            R_env = int(os.getenv("QWEN_FUSED_ADD_NORM_R", "0") or 0)
            if R_env >= 2:
                fuse = lambda a, b, consts, dt, mc: fused_add_rmsnorm_split(
                    a, b, *consts, R=R_env, sum_dtype=dt, memory_config=mc
                )
            else:
                fuse = lambda a, b, consts, dt, mc: fused_add_rmsnorm(a, b, *consts, sum_dtype=dt, memory_config=mc)
        elif os.getenv("QWEN_FUSED_ADD_NORM_SPLIT", "0") == "1":  # probe: +5% e2e at bs1, see NEGATIVE_RESULTS 34
            # Few rows (bs1: 16 tile-rows): split each row over R cores with a partial-sum exchange.
            grid = x.device().compute_with_storage_grid_size()
            R = pick_split(rows // 32, int(x.padded_shape[-1]) // 32, int(grid.x) * int(grid.y))
            if R >= 2:
                fuse = lambda a, b, consts, dt, mc: fused_add_rmsnorm_split(
                    a, b, *consts, R=R, sum_dtype=dt, memory_config=mc
                )
        if fuse is None:
            if model_args is not None and os.getenv("QWEN_BS1_RESID_SHARDED", "1") == "1":
                return _forward_resid_sharded(
                    layer, orig_forward, model_args, next_attn_consts is not None, x, args, kwargs
                )
            return orig_forward(x, *args, **kwargs)
        if is_first:
            stash.clear()
        shape = list(x.padded_shape)
        pending = {}
        calls = [0]
        orig_add = ttnn.add

        def add_wrapper(a, b, *a_args, **a_kwargs):
            if (
                a_args
                or not hasattr(a, "padded_shape")
                or not hasattr(b, "padded_shape")
                or list(a.padded_shape) != shape
                or list(b.padded_shape) != shape
                or not supported(a, b)
                or any(k not in ("memory_config", "dtype") for k in a_kwargs)
            ):
                return orig_add(a, b, *a_args, **a_kwargs)
            calls[0] += 1
            mc, dt = a_kwargs.get("memory_config"), a_kwargs.get("dtype")
            if calls[0] == 1:  # post-attention residual -> feeds ff_norm
                s, n = fuse(a, b, ff_consts, dt, mc)
                if do_verify:
                    s_ref = orig_add(a, b, *a_args, **a_kwargs)
                    n_ref = orig_ff_norm(s_ref, mode)
                    print(
                        f"[verify L{layer_idx} add1] pcc sum={_pcc(s, s_ref):.6f} norm={_pcc(n, n_ref):.6f} "
                        f"dtypes fused={s.dtype}/{n.dtype} ref={s_ref.dtype}/{n_ref.dtype}",
                        flush=True,
                    )
                pending[id(s)] = (s, n)
                return s
            if calls[0] == 2 and next_attn_consts is not None:  # post-MLP residual -> next attention_norm
                s, n = fuse(a, b, next_attn_consts, dt, mc)
                if do_verify and next_layer is not None:
                    s_ref = orig_add(a, b, *a_args, **a_kwargs)
                    n_ref = next_layer.attention_norm(s_ref, mode)
                    print(
                        f"[verify L{layer_idx} add2] pcc sum={_pcc(s, s_ref):.6f} norm={_pcc(n, n_ref):.6f} "
                        f"dtypes fused={s.dtype}/{n.dtype} ref={s_ref.dtype}/{n_ref.dtype}",
                        flush=True,
                    )
                stash[id(s)] = (s, n)
                return s
            return orig_add(a, b, *a_args, **a_kwargs)

        def ff_norm_wrapper(h, *n_args, **n_kwargs):
            hit = pending.pop(id(h), None)
            if hit is not None and hit[0] is h:
                return hit[1]
            return orig_ff_norm(h, *n_args, **n_kwargs)

        def attn_norm_wrapper(h, *n_args, **n_kwargs):
            hit = stash.pop(id(h), None)
            if hit is not None and hit[0] is h:
                return hit[1]
            return orig_attn_norm(h, *n_args, **n_kwargs)

        ttnn.add = add_wrapper
        layer.ff_norm = ff_norm_wrapper
        layer.attention_norm = attn_norm_wrapper
        try:
            return orig_forward(x, *args, **kwargs)
        finally:
            ttnn.add = orig_add
            layer.ff_norm = orig_ff_norm
            layer.attention_norm = orig_attn_norm

    layer.forward = forward
