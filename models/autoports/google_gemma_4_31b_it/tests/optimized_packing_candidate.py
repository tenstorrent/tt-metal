"""Same-policy projection packing with split costs included in the layer trace."""

import math

import ttnn
from models.autoports.google_gemma_4_31b_it.tests.optimized_layout_candidate import LayoutCandidate


class PackingCandidate(LayoutCandidate):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        obj = super().from_state_dict(state_dict, **kwargs)
        obj.pack_attention = obj.geometry.get("pack_attention", False) and bool(obj.sliding_window)
        obj.pack_mlp = obj.geometry.get("pack_mlp", False)
        obj.split_pending = {}
        for enabled, names, target in [
            (obj.pack_attention, ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "self_attn.qkv_proj"),
            (obj.pack_mlp, ["mlp.gate_proj", "mlp.up_proj"], "mlp.gate_up_proj"),
        ]:
            if not enabled:
                continue
            weights = [ttnn.to_memory_config(obj.weights[n + ".weight"], ttnn.DRAM_MEMORY_CONFIG) for n in names]
            packed = ttnn.concat(weights, dim=-1)
            k, n = tuple(packed.shape)[-2:]
            grid = obj.mesh_device.dram_grid_size()
            cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(cores, (k, math.ceil(n / (32 * obj.banks)) * 32), ttnn.ShardOrientation.ROW_MAJOR),
            )
            obj.weights[target + ".weight"] = ttnn.to_memory_config(packed, mem)
            for name in names:
                del obj.weights[name + ".weight"]
        return obj

    def _linear(self, x, name):
        if name in self.split_pending:
            return self.split_pending.pop(name)
        if self.pack_attention and name == "self_attn.q_proj":
            packed = self._linear(x, "self_attn.qkv_proj")
            widths = [self.nq * self.head_dim, self.nkv * self.head_dim, self.nkv * self.head_dim]
            names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
        elif self.pack_mlp and name == "mlp.gate_proj":
            packed = self._linear(x, "mlp.gate_up_proj")
            widths = [self.config.intermediate_size] * 2
            names = ["mlp.gate_proj", "mlp.up_proj"]
        else:
            return super()._linear(x, name)
        if packed.is_sharded():
            packed = ttnn.to_memory_config(packed, ttnn.L1_MEMORY_CONFIG)
        offset = 0
        for key, width in zip(names, widths):
            begin = [0] * len(packed.shape)
            end = list(packed.shape)
            begin[-1] = offset
            end[-1] = offset + width
            self.split_pending[key] = ttnn.slice(packed, begin, end)
            offset += width
        return self.split_pending.pop(name)
