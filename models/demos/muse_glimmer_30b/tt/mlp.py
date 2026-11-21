# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Muse-Glimmer dense SwiGLU MLP, tensor parallel: column-parallel gate and up, row-parallel down + all_reduce."""

import ttnn
from models.demos.muse_glimmer_30b.tt.attention import MuseLinear, dram_sharded_enabled
from models.demos.muse_glimmer_30b.tt.common import as_tt, decode_memcfg, is_decode, shard_mapper
from models.demos.muse_glimmer_30b.tt.model_config import LOW_PRECISION
from models.demos.muse_glimmer_30b.tt.weight_packing import pack_down, pack_gate_up, pack_gate_up_swiglu

PREFILL_CHUNK = 2048


class MuseGlimmerMLP:
    """down(silu(gate(x)) * up(x)). Input/output replicated [1, 1, M, H]; the input is not deallocated."""

    def __init__(self, args, mesh_device, ccl, layer_idx, state_dict, use_weight_cache=True):
        self.args = args
        self.mesh_device = mesh_device
        self.ccl = ccl
        self.layer_idx = layer_idx
        tp = args.tp
        self.i_local = args.intermediate_local
        self.i_pad = args.intermediate_local_padded  # zero-padded per-rank intermediate (gate/up/down use it)
        dtype = args.mlp_weight_dtype
        hid = args.hidden_size
        sharded = dram_sharded_enabled(hid, 2 * self.i_pad, dtype) and dram_sharded_enabled(self.i_pad, hid, dtype)
        tag = "v3ws" if sharded else "v3"
        tag_down = "v2ws" if sharded else "v2"

        def cache(name, t):
            if not use_weight_cache:
                return None
            return args.cache_file(f"layers.{layer_idx}.mlp.{name}", dtype, t)

        # Fused per-rank [gate | up] weight [H, 2*I_pad]; both MLP matmuls share one path so caches never mix.
        self.w_gate_up = MuseLinear(
            pack_gate_up(state_dict["mlp.gate_proj.weight"], state_dict["mlp.up_proj.weight"], tp, pad_to=self.i_pad),
            mesh_device,
            shard_dim=-1,
            k=hid,
            n=2 * self.i_pad,
            dtype=dtype,
            cache_file_name=cache("w_gate_up", tag),
            sharded=sharded,
            group="mlp",
            readers=3,
            block=1,
        )
        self.w_down = MuseLinear(
            pack_down(state_dict["mlp.down_proj.weight"], tp, pad_to=self.i_pad),
            mesh_device,
            shard_dim=-2,
            k=self.i_pad,
            n=hid,
            dtype=dtype,
            cache_file_name=cache("w_down", tag_down),
            sharded=sharded,
            group="mlp",
            readers=2,
            block=2,
        )
        # Prefill from 512 rows fuses the SwiGLU into the gate/up minimal_matmul. The kernel needs gate/up tile
        # pairs, so this is a second copy of the gate/up weight in that layout.
        self.w_swiglu = None
        if sharded:
            self.w_swiglu = as_tt(
                pack_gate_up_swiglu(
                    state_dict["mlp.gate_proj.weight"], state_dict["mlp.up_proj.weight"], tp, pad_to=self.i_pad
                ),
                mesh_device,
                dtype=dtype,
                mesh_mapper=shard_mapper(mesh_device, -1),
                cache_file_name=cache("w_swiglu", "v1"),
            )

    def _forward_chunk(self, x):
        decode = is_decode(x)
        if not decode and self.w_swiglu is not None and x.shape[-2] >= 512:
            h = ttnn.experimental.minimal_matmul(
                x,
                self.w_swiglu,
                fuse_swiglu=True,
                config=self.w_gate_up._prefill_cfg(x.shape[-2]),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b if LOW_PRECISION else None,
                compute_kernel_config=self.w_gate_up.hifi_cfg,
            )
            return self._down(h, decode)
        silu = [ttnn.UnaryOpType.SILU]
        gu = self.w_gate_up.decode(x) if decode else self.w_gate_up(x)
        s = gu.shape
        gate = ttnn.slice(gu, [0, 0, 0, 0], [s[0], s[1], s[2], self.i_pad])
        up = ttnn.slice(gu, [0, 0, 0, self.i_pad], [s[0], s[1], s[2], 2 * self.i_pad])
        ttnn.deallocate(gu)
        if decode:
            h = ttnn.mul(gate, up, input_tensor_a_activations=silu)
        else:
            h = ttnn.mul(gate, up, input_tensor_a_activations=silu, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return self._down(h, decode)

    def _down(self, h, decode):
        out = self.w_down.decode(h) if decode else self.w_down(h)
        ttnn.deallocate(h)
        if decode:
            reduced = self.ccl.all_reduce(out, memory_config=decode_memcfg(self.args.hidden_size, self.mesh_device))
        else:
            reduced = self.ccl.all_reduce(out)
        if reduced is not out:
            ttnn.deallocate(out)
        return reduced

    def __call__(self, x):
        m = x.shape[-2]
        if m <= PREFILL_CHUNK:
            return self._forward_chunk(x)
        # Prefill: bound the [M, 2*I_local] intermediate by processing row chunks.
        outs = []
        for start in range(0, m, PREFILL_CHUNK):
            end = min(start + PREFILL_CHUNK, m)
            xc = ttnn.slice(x, (0, 0, start, 0), (x.shape[0], x.shape[1], end, x.shape[3]))
            outs.append(self._forward_chunk(xc))
            ttnn.deallocate(xc)
        out = ttnn.concat(outs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for o in outs:
            ttnn.deallocate(o)
        return out
