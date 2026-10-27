# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Muse-Glimmer dense SwiGLU MLP, tensor parallel: column-parallel gate and up, row-parallel down + all_reduce."""

import ttnn
from models.demos.muse_glimmer_30b.tt.attention import MuseLinear, dram_sharded_enabled
from models.demos.muse_glimmer_30b.tt.weight_packing import pack_column_parallel, pack_down

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
        sharded = dram_sharded_enabled(hid, self.i_pad, dtype) and dram_sharded_enabled(self.i_pad, hid, dtype)
        tag = "v3ws" if sharded else "v3"
        tag_down = "v2ws" if sharded else "v2"

        def cache(name, t):
            if not use_weight_cache:
                return None
            return args.cache_file(f"layers.{layer_idx}.mlp.{name}", dtype, t)

        # Separate gate / up weights (each [H, I_pad] per rank) so each matmul is a small DRAM-sharded kernel.
        # All three share the same path so caches never mix; if either shape is ineligible, all fall back.
        self.w_gate = MuseLinear(
            pack_column_parallel(state_dict["mlp.gate_proj.weight"], tp, pad_to=self.i_pad),
            mesh_device,
            shard_dim=-1,
            k=hid,
            n=self.i_pad,
            dtype=dtype,
            cache_file_name=cache("w_gate", tag),
            sharded=sharded,
            group="mlp",
        )
        self.w_up = MuseLinear(
            pack_column_parallel(state_dict["mlp.up_proj.weight"], tp, pad_to=self.i_pad),
            mesh_device,
            shard_dim=-1,
            k=hid,
            n=self.i_pad,
            dtype=dtype,
            cache_file_name=cache("w_up", tag),
            sharded=sharded,
            group="mlp",
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
        )

    def _forward_chunk(self, x):
        gate = self.w_gate(x)
        up = self.w_up(x)
        h = ttnn.mul(
            gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        out = self.w_down(h)
        ttnn.deallocate(h)
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
