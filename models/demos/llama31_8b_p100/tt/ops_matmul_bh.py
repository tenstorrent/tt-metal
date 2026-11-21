# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pathlib

import ttnn

OPS = (
    "matmul_bh",
    "matmul_bh_covers",
    "minimal_matmul_bh",
    "minimal_matmul_bh_covers",
    "qkv_heads_bh",
    "qkv_heads_bh_covers",
)
_covers = {}


def reset():
    _covers.clear()


def _op(name):
    return getattr(ttnn.experimental.llama31_8b, name)


def mlp_gate_up(args, state_dict, state_dict_prefix, layer_num, cache_file):
    if state_dict:
        return state_dict[f"{state_dict_prefix}.w1.weight"], state_dict[f"{state_dict_prefix}.w3.weight"]
    if cache_file is not None:
        cache_file = pathlib.Path(cache_file)
        if any(cache_file.parent.glob(cache_file.name + "_*.tensorbin")):
            return None
    from models.ttt_compat.tt.load_checkpoints import load_hf_state_dict_filtered

    keys = [f"model.layers.{layer_num}.mlp.gate_proj.weight", f"model.layers.{layer_num}.mlp.up_proj.weight"]
    weights = load_hf_state_dict_filtered(args.CKPT_DIR, keys)
    return weights[keys[0]], weights[keys[1]]


def linear(
    input_tensor_a,
    input_tensor_b,
    *,
    memory_config=None,
    dtype=None,
    program_config=None,
    compute_kernel_config=None,
    core_grid=None,
    global_cb=None,
    sub_device_id=None,
    **kwargs,
):
    shape = input_tensor_a.shape
    if core_grid is None and global_cb is None and sub_device_id is None and not kwargs and len(shape) >= 3:
        key = ("matmul_bh", id(input_tensor_b), tuple(shape), input_tensor_a.dtype)
        hit = _covers.get(key)
        if hit is None:
            hit = _covers[key] = (
                _op("matmul_bh_covers")(
                    input_tensor_a,
                    input_tensor_b,
                    memory_config=memory_config,
                    dtype=dtype,
                    program_config=program_config,
                    compute_kernel_config=compute_kernel_config,
                ),
                input_tensor_b,
            )
        if hit[0]:
            return _op("matmul_bh")(
                input_tensor_a,
                input_tensor_b,
                memory_config=memory_config,
                dtype=dtype,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
            )
    return ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        memory_config=memory_config,
        dtype=dtype,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        core_grid=core_grid,
        global_cb=global_cb,
        sub_device_id=sub_device_id,
        **kwargs,
    )


def minimal_matmul(input_tensor, weight_tensor, *, config=None, dtype=None, compute_kernel_config=None, **kwargs):
    shape = input_tensor.shape
    if not kwargs and len(shape) >= 3:
        key = ("minimal_matmul_bh", id(weight_tensor), tuple(shape), input_tensor.dtype)
        hit = _covers.get(key)
        if hit is None:
            hit = _covers[key] = (
                _op("minimal_matmul_bh_covers")(
                    input_tensor, weight_tensor, config=config, dtype=dtype, compute_kernel_config=compute_kernel_config
                ),
                weight_tensor,
            )
        if hit[0]:
            return _op("minimal_matmul_bh")(
                input_tensor, weight_tensor, config=config, dtype=dtype, compute_kernel_config=compute_kernel_config
            )
    return ttnn.experimental.minimal_matmul(
        input_tensor, weight_tensor, config=config, dtype=dtype, compute_kernel_config=compute_kernel_config, **kwargs
    )


def qkv_heads(
    input_tensor,
    weight_tensor,
    *,
    num_heads,
    num_kv_heads,
    config=None,
    dtype=None,
    compute_kernel_config=None,
    **kwargs,
):
    shape = input_tensor.shape
    if kwargs or len(shape) < 3:
        return None
    key = ("qkv_heads_bh", id(weight_tensor), tuple(shape), input_tensor.dtype, num_heads, num_kv_heads)
    args = dict(
        config=config,
        dtype=dtype,
        compute_kernel_config=compute_kernel_config,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    hit = _covers.get(key)
    if hit is None:
        hit = _covers[key] = (_op("qkv_heads_bh_covers")(input_tensor, weight_tensor, **args), weight_tensor)
    return tuple(_op("qkv_heads_bh")(input_tensor, weight_tensor, **args)) if hit[0] else None


def swiglu(
    input_tensor, gate_up_tensor, *, memory_config=None, dtype=None, program_config=None, compute_kernel_config=None
):
    shape = input_tensor.shape
    if gate_up_tensor is None or len(shape) < 3:
        return None
    args = dict(
        memory_config=memory_config,
        dtype=dtype,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    key = ("matmul_bh", id(gate_up_tensor), tuple(shape), input_tensor.dtype)
    hit = _covers.get(key)
    if hit is None:
        hit = _covers[key] = (_op("matmul_bh_covers")(input_tensor, gate_up_tensor, **args), gate_up_tensor)
    return _op("matmul_bh")(input_tensor, gate_up_tensor, **args) if hit[0] else None
