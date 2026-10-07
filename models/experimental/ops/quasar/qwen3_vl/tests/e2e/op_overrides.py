# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Named workarounds (and, for bisecting, host fallbacks) installed around ttnn ops for one test."""
import inspect
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


def _on_quasar(args, kwargs):
    import ttnn

    x = next((a for a in [*args, *kwargs.values()] if isinstance(a, ttnn.Tensor)), None)
    return x is not None and x.device().arch() == ttnn.device.Arch.QUASAR


def _to_experimental_quasar(name):
    """Call the stop-gap Quasar port of a base op (ttnn.experimental.quasar.<name>) with the same arguments."""

    def rewrite(original, args, kwargs):
        import ttnn

        op = ttnn.experimental.quasar
        for part in name.split("."):  # e.g. "transformer.scaled_dot_product_attention"
            op = getattr(op, part)
        return op(*args, **kwargs)

    return rewrite


def _quasar_fp32_dest_acc(args, kwargs):
    ckc = kwargs.get("compute_kernel_config")
    return ckc is not None and getattr(ckc, "fp32_dest_acc_en", False) and _on_quasar(args, kwargs)


def _with_bf16_dest_acc(original, args, kwargs):
    import ttnn

    ckc = kwargs["compute_kernel_config"]
    bf16 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ckc.math_fidelity,
        math_approx_mode=ckc.math_approx_mode,
        fp32_dest_acc_en=False,
        packer_l1_acc=ckc.packer_l1_acc,
    )
    return original(*args, **{**kwargs, "compute_kernel_config": bf16})


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
    Workaround(
        name="quasar_experimental_add",
        target="ttnn.add",
        reason="base binary_ng builds Gen1 DataMovementKernels, which Quasar rejects; the stop-gap "
        "ttnn.experimental.quasar.add runs (QUASAR_GAPS Q7)",
        remove_when="base ttnn.add (binary_ng) is ported to Quasar",
        applies=_on_quasar,
        rewrite=_to_experimental_quasar("add"),
    ),
    Workaround(
        name="quasar_experimental_sdpa",
        target="ttnn.transformer.scaled_dot_product_attention",
        reason="base prefill SDPA builds Gen1 DataMovementKernels, which Quasar rejects; the stop-gap "
        "ttnn.experimental.quasar.transformer.scaled_dot_product_attention runs (QUASAR_GAPS Q8)",
        remove_when="base ttnn.transformer.scaled_dot_product_attention is ported to Quasar",
        applies=_on_quasar,
        rewrite=_to_experimental_quasar("transformer.scaled_dot_product_attention"),
    ),
    Workaround(
        name="quasar_rms_norm_bf16_dest",
        target="ttnn.rms_norm",
        reason="models/common/rmsnorm.py builds its own config with fp32_dest_acc_en=True (tt_transformers never passes "
        "False), bypassing the bf16-dest policy; craq-sim cannot unpack bf16 to Tf32 for it (QUASAR_GAPS S4)",
        remove_when="craq-sim implements bf16->Tf32 unpack, or RMSNorm takes the model's dest-acc setting",
        applies=_quasar_fp32_dest_acc,
        rewrite=_with_bf16_dest_acc,
    ),
]


@dataclass(frozen=True)
class HostFallback:
    """Runs `target` on the host with torch (bisecting only: a run using one is DIAGNOSTIC, never PASS)."""

    target: str
    source: str  # "golden" (ttnn golden function), "graph_case" (qwen3_vl_ops reference) or "hand"
    torch_fn: Callable  # (host args, host kwargs) -> tensor, list of tensors, or None when updating in place
    inplace_arg: int | None = None  # index of the device tensor the op updates in place


_GOLDENS: dict = {}  # target -> golden function, looked up on the real op before a fallback replaces it


def golden_of(target):
    import ttnn

    if target not in _GOLDENS:
        _GOLDENS[target] = ttnn.get_golden_function(getattr(*resolve(target)))
    return _GOLDENS[target]


def _golden(target, prep=lambda targs, tkwargs: (targs, tkwargs)):
    def fn(targs, tkwargs):
        g = golden_of(target)
        targs, tkwargs = prep(targs, tkwargs)
        params = inspect.signature(g).parameters
        return g(*targs, **{k: v for k, v in tkwargs.items() if k in params})

    return fn


def _flat_affine(targs, tkwargs):
    """Device norm weights are row-major [1, 1, dim / 32, 32]; the goldens want [dim]."""
    dim = targs[0].shape[-1]
    flat = lambda t: t.reshape(-1)[:dim] if isinstance(t, torch.Tensor) else t
    targs = [targs[0], *map(flat, targs[1:])]
    return targs, {k: flat(v) if k in ("weight", "bias") else v for k, v in tkwargs.items()}


def _graph_case(ref_name, out_shapes):
    def fn(targs, tkwargs):
        from models.experimental.ops.quasar.tests.qwen3_vl_ops import graph_case as G

        case = {
            "kwargs": {k: {"v": v} for k, v in tkwargs.items()},
            "outs": [{"shape": s} for s in out_shapes(targs, tkwargs)],
        }
        out = getattr(G, ref_name)({str(i): t for i, t in enumerate(targs)}, tkwargs, case)
        if out is None:
            raise RuntimeError(f"{ref_name} does not model these shapes")
        return out

    return fn


def _head_dim(x, kw):
    nh = kw["num_heads"]
    return x.shape[-1] // (nh + 2 * kw.get("num_kv_heads", nh))


def _qkv_shapes(targs, kw):
    nh = kw["num_heads"]
    nkv, hd, s = kw.get("num_kv_heads", nh), _head_dim(targs[0], kw), targs[0].shape[-2]
    return [(1, nh, s, hd), (1, nkv, s, hd), (1, nkv, s, hd)]


def _qkv_decode_shapes(targs, kw):
    nh = kw["num_heads"]
    nkv, hd, b = kw.get("num_kv_heads", nh), _head_dim(targs[0], kw), targs[0].shape[-2]
    return [(1, b, nh, hd), (1, b, nkv, hd), (1, b, nkv, hd)]


def _rope_llama(targs, kw):
    """x * cos + (x @ T) * sin, with the 32x32 transformation matrix T repeated along the head dim."""
    x, cos, sin, trans = targs[:4]
    t = trans.reshape(-1, trans.shape[-2], trans.shape[-1])[0, :32, :32].float()
    big = torch.block_diag(*([t] * (x.shape[-1] // 32)))
    return x.float() * cos.float() + (x.float() @ big) * sin.float()


def _paged_update(targs, kw):
    """Write each user's new K/V row at its position: [1, batch, kv_heads, hd] -> cache[block, head, row, hd]."""
    cache, upd = targs[0], targs[1]
    idxs, pt = kw["update_idxs_tensor"], kw["page_table"]
    bs = cache.shape[2]
    for b, pos in enumerate(int(p) for p in idxs.reshape(-1).tolist()):
        if pos < 0:
            continue
        cache[int(pt[b, pos // bs]), :, pos % bs, :] = upd[0, b, : cache.shape[1], :]


def _paged_fill(targs, kw):
    """Write a user's prefill K/V [1, kv_heads, seq, hd] into the blocks its page table names."""
    cache, x = targs[0], targs[1]
    pt = targs[2] if len(targs) > 2 else kw["page_table"]
    b, bs = int(kw.get("batch_idx", 0)), cache.shape[2]
    for s in range(x.shape[2]):
        cache[int(pt[b, s // bs]), :, s % bs, :] = x[0, :, s, :]


FALLBACKS = {
    f.target: f
    for f in [
        HostFallback("ttnn.linear", "golden", _golden("ttnn.linear")),
        HostFallback("ttnn.rms_norm", "golden", _golden("ttnn.rms_norm", _flat_affine)),
        HostFallback("ttnn.layer_norm", "golden", _golden("ttnn.layer_norm", _flat_affine)),
        HostFallback("ttnn.add", "golden", _golden("ttnn.add")),
        HostFallback("ttnn.multiply", "golden", _golden("ttnn.multiply")),
        HostFallback(
            "ttnn.transformer.scaled_dot_product_attention",
            "golden",
            _golden("ttnn.transformer.scaled_dot_product_attention"),
        ),
        HostFallback(
            "ttnn.transformer.paged_scaled_dot_product_attention_decode",
            "golden",
            _golden("ttnn.transformer.paged_scaled_dot_product_attention_decode"),
        ),
        HostFallback("ttnn.experimental.minimal_matmul", "graph_case", _graph_case("_ref_matmul", lambda a, k: [()])),
        HostFallback(
            "ttnn.experimental.nlp_create_qkv_heads", "graph_case", _graph_case("_ref_create_qkv_heads", _qkv_shapes)
        ),
        HostFallback(
            "ttnn.experimental.nlp_create_qkv_heads_decode",
            "graph_case",
            _graph_case("_ref_create_qkv_heads_decode", _qkv_decode_shapes),
        ),
        HostFallback(
            "ttnn.experimental.nlp_concat_heads", "graph_case", _graph_case("_ref_concat_heads", lambda a, k: [()])
        ),
        HostFallback(
            "ttnn.experimental.nlp_concat_heads_decode",
            "graph_case",
            _graph_case("_ref_concat_heads_decode", lambda a, k: [()]),
        ),
        HostFallback("ttnn.experimental.rotary_embedding_llama", "hand", _rope_llama),
        HostFallback("ttnn.experimental.paged_update_cache", "hand", _paged_update, inplace_arg=0),
        HostFallback("ttnn.experimental.paged_fill_cache", "hand", _paged_fill, inplace_arg=0),
    ]
}

# target -> where it matched the real op (PCC >= 0.999 on every captured case, bf16, HiFi4, bf16 dest acc).
# Certified before the 2026-10-06 rebase onto main 6a3ecc02796 (as af544975e4d; 37be61a6ea7 after it).
CERTIFIED: dict = {
    "ttnn.linear": "test_fallbacks.py[test_linear-*] (12 cases) @ ttsim WH 37be61a6ea7",
    "ttnn.rms_norm": "test_fallbacks.py[test_rms_norm-*] (8 cases) @ ttsim WH 37be61a6ea7",
    "ttnn.layer_norm": "test_fallbacks.py[test_layer_norm-*] (3 cases) @ ttsim WH 37be61a6ea7",
    "ttnn.add": "test_fallbacks.py[test_add-*] (9 cases) @ ttsim WH 37be61a6ea7",
    "ttnn.multiply": "test_fallbacks.py[test_multiply-*] (3 cases) @ ttsim WH 37be61a6ea7",
    "ttnn.transformer.scaled_dot_product_attention": "test_fallbacks.py[test_scaled_dot_product_attention-*] (2 cases) @ ttsim WH 37be61a6ea7",
    "ttnn.transformer.paged_scaled_dot_product_attention_decode": "test_fallbacks.py[test_paged_scaled_dot_product_attention_decode-*] (1 cases) @ ttsim WH 37be61a6ea7",
    "ttnn.experimental.minimal_matmul": "test_fallbacks.py[test_minimal_matmul-*] (2 cases) @ ttsim WH 37be61a6ea7",
    "ttnn.experimental.nlp_create_qkv_heads": "test_fallbacks.py[test_nlp_create_qkv_heads-*] (2 cases) @ ttsim WH 37be61a6ea7",
    "ttnn.experimental.nlp_create_qkv_heads_decode": "test_fallbacks.py[test_nlp_create_qkv_heads_decode-*] (1 cases) @ ttsim WH 37be61a6ea7",
    "ttnn.experimental.nlp_concat_heads": "test_fallbacks.py[test_nlp_concat_heads-*] (2 cases) @ ttsim WH 37be61a6ea7",
    "ttnn.experimental.nlp_concat_heads_decode": "test_fallbacks.py[test_nlp_concat_heads_decode-*] (1 cases) @ ttsim WH 37be61a6ea7",
    "ttnn.experimental.rotary_embedding_llama": "test_fallbacks.py[test_rotary_embedding_llama-*] (5 cases) @ ttsim WH 37be61a6ea7",
    "ttnn.experimental.paged_update_cache": "test_fallbacks.py[test_paged_update_cache-*] (1 cases) @ ttsim WH 37be61a6ea7",
    "ttnn.experimental.paged_fill_cache": "test_fallbacks.py[test_paged_fill_cache-*] (1 cases) @ ttsim WH 37be61a6ea7",
}


def _short(target):
    return target.rsplit(".", 1)[-1]


class OverrideSession:
    def __init__(self, mesh_device, host_ops, disable_wa, allow_uncertified=False):
        self.mesh_device = mesh_device
        self.host_ops = tuple(host_ops)
        self.disable_wa = set(disable_wa)
        self.allow_uncertified = allow_uncertified
        self.hits = Counter()
        self.host_ops_active = []

    def _selected_fallbacks(self):
        if self.host_ops == ("all",):
            return list(FALLBACKS.values())
        by_short = {_short(t): f for t, f in FALLBACKS.items()}
        unknown = [n for n in self.host_ops if n not in FALLBACKS and n not in by_short]
        if unknown:
            raise KeyError(f"no host fallback for {unknown}; known: {sorted(by_short)}")
        return [FALLBACKS.get(n) or by_short[n] for n in self.host_ops]

    def install(self, monkeypatch):
        unknown = self.disable_wa - {w.name for w in WORKAROUNDS}
        if unknown:
            raise KeyError(f"unknown workaround(s): {sorted(unknown)}")
        fallbacks = self._selected_fallbacks()
        uncertified = [f.target for f in fallbacks if f.target not in CERTIFIED]
        if uncertified and not self.allow_uncertified:
            raise RuntimeError(
                f"host fallback(s) {uncertified} not certified against the real op (run test_fallbacks.py on WH/BH, "
                "or pass --qwen-allow-uncertified)"
            )
        for fb in fallbacks:  # goldens hang off the real ops, so look them up before any wrapper replaces one
            if fb.source == "golden":
                golden_of(fb.target)
        for wa in WORKAROUNDS:
            if wa.name not in self.disable_wa:
                self._install_workaround(monkeypatch, wa)
        for fb in fallbacks:  # after the workarounds: a host op replaces the whole op
            self._install_fallback(monkeypatch, fb)

    def _install_fallback(self, monkeypatch, fb):
        import ttnn

        from models.experimental.ops.quasar.qwen3_vl.tests.e2e.recorder import to_host

        parent, attr = resolve(fb.target)
        self.host_ops_active.append(fb.target)
        conv = lambda v: to_host(v) if isinstance(v, ttnn.Tensor) else v

        def wrapper(*args, **kwargs):
            self.hits[f"host:{fb.target}"] += 1
            targs, tkw = [conv(a) for a in args], {k: conv(v) for k, v in kwargs.items()}
            out = fb.torch_fn(targs, tkw)
            if fb.inplace_arg is not None:
                dev = args[fb.inplace_arg]
                host = ttnn.from_torch(targs[fb.inplace_arg], dtype=dev.dtype, layout=dev.layout)
                ttnn.copy_host_to_device_tensor(host, dev)
                return None
            ref = next(a for a in args if isinstance(a, ttnn.Tensor))
            want = kwargs.get("memory_config")

            def upload(t):
                r = ttnn.from_torch(
                    t.to(torch.bfloat16),
                    dtype=kwargs.get("dtype") or ref.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                if want is None or not want.is_sharded():
                    return r
                return ttnn.to_memory_config(r, want if want.shard_spec is not None else _with_shard_spec(r, want))

            return [upload(t) for t in out] if isinstance(out, (list, tuple)) else upload(out)

        monkeypatch.setattr(parent, attr, wrapper)

    def _install_workaround(self, monkeypatch, wa):
        parent, attr = resolve(wa.target)
        original = getattr(parent, attr)

        def wrapper(*args, **kwargs):
            if wa.applies(args, kwargs):
                self.hits[f"wa:{wa.name}"] += 1
                return wa.rewrite(original, args, kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(parent, attr, wrapper)
