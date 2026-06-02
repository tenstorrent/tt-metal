# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ACE Qwen3 prefill: route activations through L1 (LoFi + ``bfloat8_b`` via :mod:`math_perf_env`).

``tt_transformers`` defaults prefill residuals and matmul ``in0`` to DRAM. Tracy reports large
``MinimalMatmulDeviceOperation`` / ``MatmulDeviceOperation (in0:dram_interleaved)`` buckets for
conditioning; this module patches the loaded Qwen stack so prefill uses ``L1_MEMORY_CONFIG``
without editing upstream ``tt_transformers`` sources.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import ttnn
from models.tt_transformers.tt.common import Mode

from .math_perf_env import ace_step_linear_l1_memory_config

# ModelArgs getters that return ``DRAM_MEMORY_CONFIG`` for ``Mode.PREFILL``.
_PREFILL_DRAM_MEMCFG_GETTERS: tuple[str, ...] = (
    "get_residual_mem_config",
    "get_mlp_input_mem_config",
    "get_mlp_ff1_3_mem_config",
    "get_mlp_ff2_mem_config",
    "get_mlp_output_mem_config",
    "get_attn_input_mem_config",
    "get_attn_qkv_mm_mem_config",
    "get_attn_qkv_all_reduce_output_mem_config",
    "get_attn_create_head_input_mem_config",
    "get_attn_create_head_output_mem_config",
    "get_attn_sdpa_output_mem_config",
    "get_attn_concat_heads_output_mem_config",
    "get_attn_all_gather_output_mem_config",
    "get_attn_wo_output_mem_config",
    "get_attn_dense_output_mem_config",
    "get_attn_all_reduce_output_mem_config",
    "get_attn_gather_users_mem_config",
)


def _is_dram_mc(mc: Any, dram_mc: Any) -> bool:
    return mc is not None and dram_mc is not None and mc == dram_mc


def _ensure_l1(ttnn_module: Any, tensor: Any, *, l1_mc: Any) -> Any:
    if tensor is None or l1_mc is None or not hasattr(ttnn_module, "to_memory_config"):
        return tensor
    if tensor.memory_config() == l1_mc:
        return tensor
    return ttnn_module.to_memory_config(tensor, l1_mc)


def _swap_dram_kwarg(kwargs: dict, *, dram_mc: Any, l1_mc: Any) -> None:
    mc = kwargs.get("memory_config")
    if _is_dram_mc(mc, dram_mc):
        kwargs["memory_config"] = l1_mc
    imc = kwargs.get("intermediate_memory_config")
    if _is_dram_mc(imc, dram_mc):
        kwargs["intermediate_memory_config"] = l1_mc


def _ensure_l1_arg(arg: Any, *, l1_mc: Any) -> Any:
    """Move a tensor or a list of tensors (e.g. ``ttnn.concat`` inputs) to L1."""
    if isinstance(arg, list):
        return [_ensure_l1(ttnn, t, l1_mc=l1_mc) for t in arg]
    return _ensure_l1(ttnn, arg, l1_mc=l1_mc)


def _ensure_l1_first_arg(args: tuple, *, l1_mc: Any) -> tuple:
    if not args:
        return args
    return (_ensure_l1_arg(args[0], l1_mc=l1_mc),) + args[1:]


def _patch_lru_cached_getter(model_args: Any, name: str, *, dram_mc: Any, l1_mc: Any) -> None:
    original: Callable = getattr(model_args, name)

    if hasattr(original, "cache_clear"):
        original.cache_clear()

    @functools.wraps(original)
    def patched(*args, **kwargs):
        mode = args[0] if args else kwargs.get("mode")
        out = original(*args, **kwargs)
        if mode == Mode.PREFILL and _is_dram_mc(out, dram_mc):
            return l1_mc
        return out

    if hasattr(original, "cache_clear"):
        patched = functools.lru_cache(maxsize=None)(patched)  # type: ignore[assignment]

    setattr(model_args, name, patched)


def _patch_mlp_ff2_all_reduce_getter(model_args: Any, *, dram_mc: Any, l1_mc: Any) -> None:
    name = "get_mlp_ff2_all_reduce_mem_config"
    original: Callable = getattr(model_args, name)

    @functools.wraps(original)
    def patched(mode, tensor):
        out = original(mode, tensor)
        if mode == Mode.PREFILL and _is_dram_mc(out, dram_mc):
            return l1_mc
        return out

    setattr(model_args, name, patched)


def ace_step_patch_model_args_prefill_l1(model_args: Any) -> None:
    """Force L1 interleaved for all ``ModelArgs`` prefill activation memory getters."""
    l1_mc = ace_step_linear_l1_memory_config(ttnn)
    dram_mc = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    if l1_mc is None or dram_mc is None:
        return

    for name in _PREFILL_DRAM_MEMCFG_GETTERS:
        if hasattr(model_args, name):
            _patch_lru_cached_getter(model_args, name, dram_mc=dram_mc, l1_mc=l1_mc)

    if hasattr(model_args, "get_mlp_ff2_all_reduce_mem_config"):
        _patch_mlp_ff2_all_reduce_getter(model_args, dram_mc=dram_mc, l1_mc=l1_mc)


def _patch_distributed_norm_prefill_l1(norm_mod: Any, *, l1_mc: Any) -> None:
    """Prefill ``DistributedNorm`` uses DRAM for ``to_memory_config`` / gather — switch to L1."""
    orig_forward = norm_mod.forward

    def forward(x, mode: Mode, norm_config=None):
        if getattr(norm_mod, "TG", False) or mode != Mode.PREFILL:
            return orig_forward(x, mode, norm_config)

        sharded_output_config = norm_config.get("sharded_output_config") if norm_config else None
        input_mem_cfg = l1_mc

        if norm_mod.args.is_multichip and not norm_mod.args.is_distributed_norm(mode):
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=norm_mod.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=norm_mod.tt_ccl.get_num_links(1),
                topology=norm_mod.args.ccl_topology(),
                memory_config=input_mem_cfg,
                barrier_semaphore=norm_mod.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
                subdevice_id=norm_mod.prefetcher.worker_sub_device_id if norm_mod.prefetcher is not None else None,
            )
        else:
            x = ttnn.to_memory_config(x, input_mem_cfg)

        x = norm_mod.norm(x, mode=mode, in_sharded=False, out_sharded=False, norm_config=norm_config)

        if norm_mod.args.is_distributed_norm(mode) and norm_mod.enable_all_gather:
            x = ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=norm_mod.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=norm_mod.tt_ccl.get_num_links(1),
                topology=norm_mod.args.ccl_topology(),
                memory_config=x.memory_config(),
                barrier_semaphore=norm_mod.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
        return x

    norm_mod.forward = forward  # type: ignore[method-assign]


def _patch_embd_l1_output(embd_module: Any, *, l1_mc: Any) -> None:
    """Prefill-only L1 embedding output; decode passes an explicit ``memory_config`` — leave it."""

    orig_forward = embd_module.forward

    def forward(x, memory_config=None):
        if memory_config is not None:
            return orig_forward(x, memory_config=memory_config)
        out = orig_forward(x, memory_config=l1_mc)
        return _ensure_l1(ttnn, out, l1_mc=l1_mc)

    embd_module.forward = forward  # type: ignore[method-assign]


def _patch_decoder_l1_residual(decoder_layer: Any, *, l1_mc: Any) -> None:
    orig_forward = decoder_layer.forward

    def forward(x, *args, mode="decode", **kwargs):
        if mode == Mode.PREFILL:
            x = _ensure_l1(ttnn, x, l1_mc=l1_mc)
        return orig_forward(x, *args, mode=mode, **kwargs)

    decoder_layer.forward = forward  # type: ignore[method-assign]


# Tuned ``MinimalMatmulConfig`` for the two SLOW Qwen3 prefill ``minimal_matmul`` ops at
# seq_len=256 (M=256). Both the fused QKV (K=1024, N=4096) and MLP down/FF2 (K=3072, N=1024)
# matmuls win on the SAME block/grid: M_block=2, K_block=8, N_block=8 on a 8x4 (32-core) grid.
# Device sweep (BF16/BFP8 x BFP8, LoFi, in-process kernel-duration, PCC 0.9999/0.9998):
#   qkv  : stock g8x10 m8k8n8 = 45.45us  ->  g8x4 m2k8n8 = 38.69us  (~1.17x)
#   ff2  : stock ~39us         ->  g8x4 m2k8n8 = 32.02us  (~1.22x)
# Stock returns an 80-core (8x10) MinimalMatmulConfig with a literal "FIXME: optimize this
# config for prefill" in tt_transformers/model_config.py; this overrides it WITHOUT editing
# upstream.  Gated to 128 < seq_len <= 256 (the conditioning/caption prefill band we measured);
# any other seq_len defers to the stock getter.
_ACE_STEP_QWEN_PREFILL_MM_TUNED = dict(M_block_size=2, K_block_size=8, N_block_size=8)
_ACE_STEP_QWEN_PREFILL_MM_GRID = (8, 4)
_ACE_STEP_QWEN_PREFILL_MM_SEQ_MAX = 256


def _ace_step_tuned_minimal_matmul_config():
    cfg_cls = getattr(ttnn, "MinimalMatmulConfig", None)
    coord = getattr(ttnn, "CoreCoord", None)
    if cfg_cls is None or coord is None:
        return None
    gx, gy = _ACE_STEP_QWEN_PREFILL_MM_GRID
    return cfg_cls(compute_with_storage_grid_size=coord(gx, gy), **_ACE_STEP_QWEN_PREFILL_MM_TUNED)


def _patch_prefill_minimal_matmul_getter(model_args: Any, name: str) -> None:
    """Override a ``get_*_program_config`` getter to return the tuned config for PREFILL seq<=256."""
    original: Callable | None = getattr(model_args, name, None)
    if original is None:
        return
    tuned = _ace_step_tuned_minimal_matmul_config()
    if tuned is None:
        return

    @functools.wraps(original)
    def patched(mode, seq_len=1, *args, **kwargs):
        if mode == Mode.PREFILL and 128 < int(seq_len) <= _ACE_STEP_QWEN_PREFILL_MM_SEQ_MAX:
            return tuned
        return original(mode, seq_len, *args, **kwargs)

    setattr(model_args, name, patched)


def ace_step_apply_qwen_prefill_matmul_configs(model_args: Any) -> None:
    """Pin tuned ``MinimalMatmulConfig`` for the SLOW Qwen3 prefill qkv + FF2 matmuls (seq<=256)."""
    _patch_prefill_minimal_matmul_getter(model_args, "get_attn_qkv_program_config")
    _patch_prefill_minimal_matmul_getter(model_args, "get_mlp_ff2_prg_config")


def ace_step_apply_qwen_prefill_l1(tt_model: Any, model_args: Any) -> None:
    """Patch a loaded ``tt_transformers`` Qwen model for L1 prefill activations."""
    l1_mc = ace_step_linear_l1_memory_config(ttnn)
    if l1_mc is None:
        return

    ace_step_patch_model_args_prefill_l1(model_args)
    ace_step_apply_qwen_prefill_matmul_configs(model_args)

    _patch_embd_l1_output(tt_model.embd, l1_mc=l1_mc)
    for layer in tt_model.layers:
        _patch_decoder_l1_residual(layer, l1_mc=l1_mc)
        if hasattr(layer, "attention_norm"):
            _patch_distributed_norm_prefill_l1(layer.attention_norm, l1_mc=l1_mc)
        if hasattr(layer, "ff_norm"):
            _patch_distributed_norm_prefill_l1(layer.ff_norm, l1_mc=l1_mc)

    if hasattr(tt_model, "norm"):
        _patch_distributed_norm_prefill_l1(tt_model.norm, l1_mc=l1_mc)


@contextmanager
def ace_step_qwen_prefill_l1_op_context() -> Iterator[None]:
    """During Qwen prefill, redirect hard-coded ``DRAM_MEMORY_CONFIG`` TTNN calls to L1."""
    l1_mc = ace_step_linear_l1_memory_config(ttnn)
    dram_mc = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
    if l1_mc is None or dram_mc is None:
        yield
        return

    saved: dict[str, Any] = {}
    experimental = getattr(ttnn, "experimental", None)
    tile_layout = getattr(ttnn, "TILE_LAYOUT", None)

    def _wrap_l1_compute(name: str, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _swap_dram_kwarg(kwargs, dram_mc=dram_mc, l1_mc=l1_mc)
            if name.endswith("minimal_matmul") and kwargs.get("memory_config") is None:
                kwargs["memory_config"] = l1_mc
            return fn(*args, **kwargs)

        return wrapper

    def _wrap_l1_unary(name: str, fn: Callable) -> Callable:
        """Matmul/SDPA/concat paths: L1 ``in0`` + L1 output when callers pass DRAM defaults."""

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            args = _ensure_l1_first_arg(args, l1_mc=l1_mc)
            _swap_dram_kwarg(kwargs, dram_mc=dram_mc, l1_mc=l1_mc)
            if kwargs.get("memory_config") is None:
                kwargs["memory_config"] = l1_mc
            out = fn(*args, **kwargs)
            return _ensure_l1(ttnn, out, l1_mc=l1_mc)

        return wrapper

    def _wrap_to_layout(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(tensor, layout, *args, **kwargs):
            _swap_dram_kwarg(kwargs, dram_mc=dram_mc, l1_mc=l1_mc)
            out = fn(tensor, layout, *args, **kwargs)
            if tile_layout is not None and layout == tile_layout:
                out = _ensure_l1(ttnn, out, l1_mc=l1_mc)
            return out

        return wrapper

    for op_name in ("linear", "add", "mul", "multiply", "pad", "reshape", "permute", "typecast"):
        if hasattr(ttnn, op_name):
            saved[f"ttnn.{op_name}"] = getattr(ttnn, op_name)
            setattr(ttnn, op_name, _wrap_l1_compute(op_name, getattr(ttnn, op_name)))

    if hasattr(ttnn, "embedding"):
        _orig_embedding = ttnn.embedding
        saved["ttnn.embedding"] = _orig_embedding

        @functools.wraps(_orig_embedding)
        def _embedding(*args, **kwargs):
            _swap_dram_kwarg(kwargs, dram_mc=dram_mc, l1_mc=l1_mc)
            if kwargs.get("memory_config") is None:
                kwargs["memory_config"] = l1_mc
            return _orig_embedding(*args, **kwargs)

        ttnn.embedding = _embedding

    if hasattr(ttnn, "to_layout"):
        saved["ttnn.to_layout"] = ttnn.to_layout
        ttnn.to_layout = _wrap_to_layout(ttnn.to_layout)

    if hasattr(ttnn, "concat"):
        saved["ttnn.concat"] = ttnn.concat
        ttnn.concat = _wrap_l1_unary("concat", ttnn.concat)

    transformer = getattr(ttnn, "transformer", None)
    if transformer is not None:
        for op_name in ("scaled_dot_product_attention", "chunked_scaled_dot_product_attention"):
            if hasattr(transformer, op_name):
                key = f"transformer.{op_name}"
                saved[key] = getattr(transformer, op_name)
                setattr(transformer, op_name, _wrap_l1_unary(op_name, getattr(transformer, op_name)))

    if experimental is not None:
        for op_name in ("minimal_matmul", "nlp_create_qkv_heads", "all_gather_async"):
            if hasattr(experimental, op_name):
                key = f"experimental.{op_name}"
                saved[key] = getattr(experimental, op_name)
                setattr(experimental, op_name, _wrap_l1_compute(op_name, getattr(experimental, op_name)))
        if hasattr(experimental, "nlp_concat_heads"):
            saved["experimental.nlp_concat_heads"] = experimental.nlp_concat_heads
            experimental.nlp_concat_heads = _wrap_l1_unary("nlp_concat_heads", experimental.nlp_concat_heads)

    tt_ccl_mod = None
    try:
        from models.tt_transformers.tt import ccl as tt_ccl_mod

        if hasattr(tt_ccl_mod, "tt_all_reduce"):
            saved["tt_all_reduce"] = tt_ccl_mod.tt_all_reduce
            tt_ccl_mod.tt_all_reduce = _wrap_l1_compute("tt_all_reduce", tt_ccl_mod.tt_all_reduce)
    except ImportError:
        tt_ccl_mod = None

    try:
        yield
    finally:
        for key, fn in saved.items():
            if key.startswith("ttnn."):
                setattr(ttnn, key.split(".", 1)[1], fn)
            elif key.startswith("experimental.") and experimental is not None:
                setattr(experimental, key.split(".", 1)[1], fn)
            elif key.startswith("transformer.") and transformer is not None:
                setattr(transformer, key.split(".", 1)[1], fn)
            elif key == "tt_all_reduce" and tt_ccl_mod is not None:
                tt_ccl_mod.tt_all_reduce = fn


__all__ = [
    "ace_step_apply_qwen_prefill_l1",
    "ace_step_patch_model_args_prefill_l1",
    "ace_step_qwen_prefill_l1_op_context",
]
