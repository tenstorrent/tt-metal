# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Explicit gather-KV integration; not an overlapped ring-attention kernel."""

import functools

import ttnn

import device_attention as kernel


class FrontierAttention:
    def __init__(self, variant, capture=None):
        if variant not in kernel.VARIANTS:
            raise ValueError(f"Unknown attention variant {variant}")
        self.variant = variant
        self.capture = capture
        self.calls = {}
        self.transport = {}

    def install(self, transformer):
        for family, blocks in (
            ("dual", transformer.transformer_blocks),
            ("single", transformer.single_transformer_blocks),
        ):
            for index, block in enumerate(blocks):
                name = f"{family}.{index}"
                block.attn._attention_override = functools.partial(self.run, name)

    def run(self, name, attn, q, k, v, joint_q, joint_k, joint_v, logical_n, logical_l):
        device = attn.mesh_device
        sp = attn.parallel_config.sequence_parallel
        if tuple(device.shape) != (2, 4) or sp.factor != 2 or sp.mesh_axis != 0:
            raise ValueError("This first model adapter is qualified only for the 2x4 SP2/TP4 mesh")
        if q.shape[2] * sp.factor != logical_n:
            raise ValueError("Padded or mismatched logical spatial length is unsupported")
        split = q.shape[2]
        if logical_l:
            if not attn.shard_prompt or joint_q.shape[2] * sp.factor != logical_l:
                raise ValueError("Joint prompt must be sequence-sharded without padding")
            q, k, v = [ttnn.concat(parts, dim=2) for parts in ((q, joint_q), (k, joint_k), (v, joint_v))]
        call = self.calls.get(name, 0)
        self.calls[name] = call + 1
        if self.capture is not None:
            self.capture(name, call, attn, q, k, v)
        hardware = device.compute_with_storage_grid_size()
        cores = hardware.x * (hardware.y - 1)
        q = kernel.prepare(device, q, self.variant, is_q=True, cores=cores)
        k = kernel.prepare(device, k, self.variant, is_q=False, cores=cores)
        v = kernel.prepare(device, v, self.variant, is_q=False, cores=cores)
        # Quantize once on each owning rank. All-gather forwards the prepared
        # storage format directly; no BF16 expansion or requantization here.
        k = attn.ccl_manager.all_gather(k, dim=2, mesh_axis=sp.mesh_axis, use_hyperparams=True)
        v = attn.ccl_manager.all_gather(v, dim=2, mesh_axis=sp.mesh_axis, use_hyperparams=True)
        self.transport[name] = dict(k_dtype=str(k.get_dtype()), v_dtype=str(v.get_dtype()), kv_shape=list(k.shape))
        out = kernel.attention(device, q, k, v, self.variant, max_cores=cores)
        if not logical_l:
            return out, None
        end = list(out.shape)
        end[2] = split
        spatial = ttnn.slice(out, [0, 0, 0, 0], end)
        prompt = ttnn.slice(out, [0, 0, split, 0], list(out.shape))
        return spatial, prompt
