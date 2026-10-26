# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Muse-Glimmer dense SwiGLU MLP, tensor parallel: column-parallel gate and up, row-parallel down + all_reduce."""

import ttnn
from models.demos.muse_glimmer_30b.tt.attention import MuseLinear, dram_sharded_enabled
from models.demos.muse_glimmer_30b.tt.common import decode_memcfg, is_decode
from models.demos.muse_glimmer_30b.tt.weight_packing import pack_down, pack_gate_up

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

    def _forward_chunk(self, x):
        decode = is_decode(x)
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
        out = self.w_down.decode(h) if decode else self.w_down(h)
        ttnn.deallocate(h)
        if decode:
            reduced = self.ccl.all_reduce(out, memory_config=decode_memcfg(self.args.hidden_size))
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
