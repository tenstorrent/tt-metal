# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Named workarounds (and, for bisecting, host fallbacks) installed around ttnn ops for one test."""
import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class Workaround:
    name: str
    target: str
    reason: str
    remove_when: str
    applies: Callable
    rewrite: Callable


def resolve(target):
    import ttnn

    parts = target.split(".")
    assert parts[0] == "ttnn", target
    parent = ttnn
    for p in parts[1:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _fp32_to_bf16_on_device(args, kwargs):
    import ttnn

    src = args[0] if args else kwargs.get("tensor")
    return (
        isinstance(src, torch.Tensor)
        and src.dtype == torch.float32
        and kwargs.get("dtype") == ttnn.bfloat16
        and kwargs.get("device") is not None
    )


def _cast_source_to_bf16(original, args, kwargs):
    if args:
        return original(args[0].to(torch.bfloat16), *args[1:], **kwargs)
    return original(**{**kwargs, "tensor": kwargs["tensor"].to(torch.bfloat16)})


def _fewer_cores_than_heads(args, kwargs):
    x, heads = (args[0] if args else kwargs.get("input_tensor")), kwargs.get("num_heads")
    if x is None or heads is None:
        return False
    g = x.device().compute_with_storage_grid_size()
    return g.x * g.y < heads


def _merge_heads_on_device(original, args, kwargs):
    """[1, batch, heads, head_dim] -> [1, 1, batch(padded to a tile), heads*head_dim] without per-head sharding."""
    import ttnn

    x, heads = (args[0] if args else kwargs["input_tensor"]), kwargs["num_heads"]
    batch, head_dim = int(x.shape[1]), int(x.shape[3])
    if x.is_sharded():
        x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.untilize(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.reshape(x, (1, 1, batch, heads * head_dim))  # row-major: heads are contiguous per user
    pad_to = -(-batch // 32) * 32
    if pad_to != batch:
        x = ttnn.pad(x, [(0, 0), (0, 0), (0, pad_to - batch), (0, 0)], value=0.0)
    return ttnn.tilize(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _small_grid_sharded_linear(args, kwargs):
    x = args[0] if args else kwargs.get("input_tensor_a")
    if x is None or kwargs.get("program_config") is not None or not x.is_sharded():
        return False
    g = x.device().compute_with_storage_grid_size()
    return g.x * g.y < 64


def _linear_from_interleaved(original, args, kwargs):
    import ttnn

    x = ttnn.sharded_to_interleaved(args[0] if args else kwargs.pop("input_tensor_a"), ttnn.DRAM_MEMORY_CONFIG)
    want = kwargs.get("memory_config")
    out = original(x, *args[1:], **{**kwargs, "memory_config": ttnn.DRAM_MEMORY_CONFIG})
    if want is None or not want.is_sharded():
        return out
    # Callers rely on the sharded output (they sharded_to_interleaved it and free the source), but ttnn's default
    # matmul cannot fill in a missing shard spec, so complete it here.
    return ttnn.to_memory_config(out, want if want.shard_spec is not None else _with_shard_spec(out, want))


def _with_shard_spec(t, want):
    """`want` (sharded, no shard spec) with a spec spreading `t` over as many cores as divide its tiles."""
    import ttnn

    grid = t.device().compute_with_storage_grid_size()
    dims = list(t.padded_shape)
    w, h = dims[-1], math.prod(dims[:-1])
    width = want.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert width or want.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED, want
    n = _largest_divisor((w if width else h) // ttnn.TILE_SIZE, grid.x * grid.y)
    shard = [h, w // n] if width else [h // n, w]
    cores = ttnn.num_cores_to_corerangeset(n, grid, row_wise=True)
    spec = ttnn.ShardSpec(cores, shard, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(want.memory_layout, want.buffer_type, spec)


def _small_grid_wide_untilize(args, kwargs):
    x = args[0] if args else kwargs.get("input_tensor")
    if x is None or x.is_sharded() or not kwargs.get("use_multicore", True):
        return False
    g = x.device().compute_with_storage_grid_size()
    # A full tile row of more than 128 bf16 tiles (in + out CBs, 512 KiB) leaves little L1 for live buffers.
    return g.x * g.y < 64 and x.padded_shape[-1] // 32 > 128


def _untilize_single_core(original, args, kwargs):
    return original(*args, **{**kwargs, "use_multicore": False})


def _largest_divisor(n, at_most):
    return next(d for d in range(min(n, at_most), 0, -1) if n % d == 0)


WORKAROUNDS = [
    Workaround(
        name="host_cast_fp32_upload",
        target="ttnn.from_torch",
        reason="fp32->bf16 uploads tilize in fp32 on device, unpacking fp32 to SrcA (#57780; QUASAR_GAPS Q3/S2)",
        remove_when="#57780 fixed: fp32 tilize unpacks to DEST on Quasar (and ttsim WH accepts it)",
        applies=_fp32_to_bf16_on_device,
        rewrite=_cast_source_to_bf16,
    ),
    Workaround(
        name="concat_heads_decode_small_grid",
        target="ttnn.experimental.nlp_concat_heads_decode",
        reason="the op places one head per core and needs num_heads (32) cores; the emulator has 2 (QUASAR_GAPS G1)",
        remove_when="nlp_concat_heads_decode packs several heads per core",
        applies=_fewer_cores_than_heads,
        rewrite=_merge_heads_on_device,
    ),
    Workaround(
        name="small_grid_unshard_linear",
        target="ttnn.linear",
        reason="decode norms emit L1-sharded activations; with ttnn's default program those collide with the "
        "matmul's L1 buffers on a 2-core grid (QUASAR_GAPS G4)",
        remove_when="decode norms and linears on small grids are configured interleaved end to end",
        applies=_small_grid_sharded_linear,
        rewrite=_linear_from_interleaved,
    ),
    Workaround(
        name="small_grid_untilize_single_core",
        target="ttnn.untilize",
        reason="multicore untilize sizes its CBs for a full tile row without counting live L1 buffers; the "
        "151936-wide decode logits then clash with them on a 2-core grid (QUASAR_GAPS G5)",
        remove_when="untilize's L1 check accounts for allocated L1 buffers or splits wide rows",
        applies=_small_grid_wide_untilize,
        rewrite=_untilize_single_core,
    ),
]


class OverrideSession:
    def __init__(self, mesh_device, host_ops, disable_wa):
        self.mesh_device = mesh_device
        self.host_ops = tuple(host_ops)
        self.disable_wa = set(disable_wa)
        self.hits = Counter()
        self.host_ops_active = []

    def install(self, monkeypatch):
        unknown = self.disable_wa - {w.name for w in WORKAROUNDS}
        if unknown:
            raise KeyError(f"unknown workaround(s): {sorted(unknown)}")
        for wa in WORKAROUNDS:
            if wa.name not in self.disable_wa:
                self._install_workaround(monkeypatch, wa)

    def _install_workaround(self, monkeypatch, wa):
        parent, attr = resolve(wa.target)
        original = getattr(parent, attr)

        def wrapper(*args, **kwargs):
            if wa.applies(args, kwargs):
                self.hits[f"wa:{wa.name}"] += 1
                return wa.rewrite(original, args, kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(parent, attr, wrapper)
