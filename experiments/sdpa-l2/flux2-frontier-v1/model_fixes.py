# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Experiment-only isolation of FLUX.2 fixes already present on newer main.

The default residual repair uses an unfused addcmul to isolate missing math
from the fused kernel. The optional fused path matches main's non-Ring fix
for the FLUX.2 projections tested here; neither path changes SDPA numerics.
"""

import ttnn

from models.tt_dit.layers.linear import maybe_cast_activation, resolve_output_dtype
from models.tt_dit.utils.matmul import get_matmul_config, get_matmul_core_grid


def install(transformer, *, residual=False, per_head_norm=False, fused=False):
    for block in transformer.transformer_blocks:
        if per_head_norm:
            block.attn.per_head_norm = True
        if not residual:
            continue
        for projection in (block.attn.to_out, block.attn.to_add_out):
            if projection.ccl_manager.topology == ttnn.Topology.Ring:
                continue
            original = projection.forward

            def forward(*args, _original=original, _projection=projection, **kwargs):
                a = kwargs.pop("addcmul_a", None)
                b = kwargs.pop("addcmul_b", None)
                scalar = kwargs.pop("addcmul_scalar", 1.0)
                assert (a is None) == (b is None)
                assert scalar == 1.0, "This diagnostic only covers FLUX.2's unit scalar"
                if a is not None and fused:
                    # Same non-Ring operation/config selection as main's #55225
                    # fix, restricted to the FLUX.2 projections under diagnosis.
                    layer = _projection
                    assert layer.fsdp_mesh_axis is None
                    assert layer.chunks is None and layer.activation_fn is None
                    assert layer.fused_activation_fn is None and not layer.fuse_swiglu
                    x = args[0] if args else kwargs["x"]
                    x = maybe_cast_activation(x, layer.activation_dtype)
                    dtype = kwargs.get("dtype")
                    if layer.pin_output_bf16:
                        dtype = resolve_output_dtype(dtype, x)
                    weight = layer.weight.data
                    if x.padded_shape[-1] != weight.padded_shape[-2]:
                        x = layer.ccl_manager.all_gather_persistent_buffer(
                            x,
                            dim=-1,
                            mesh_axis=kwargs["parallel_config"].tensor_parallel.mesh_axis,
                            use_hyperparams=True,
                        )
                    config = get_matmul_config(
                        x.padded_shape[-2],
                        x.padded_shape[-1],
                        weight.padded_shape[-1],
                        get_matmul_core_grid(layer.mesh_device),
                        kwargs.get("default_block_size"),
                    )
                    return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                        x,
                        weight,
                        scalar,
                        a,
                        b,
                        bias_tensor=layer.bias.data if layer.bias is not None else None,
                        config=config,
                        compute_kernel_config=kwargs.get("compute_kernel_config") or layer.compute_config,
                        dtype=dtype,
                    )
                out = _original(*args, **kwargs)
                return ttnn.addcmul(a, out, b) if a is not None else out

            projection.forward = forward
