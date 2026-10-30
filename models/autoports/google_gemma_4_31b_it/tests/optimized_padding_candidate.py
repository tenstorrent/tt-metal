"""Zero output-channel padding to make two-reader O/down matmuls legal."""

import torch

import ttnn
from models.autoports.google_gemma_4_31b_it.tests.optimized_topology_candidate import TopologyCandidate


class PaddingCandidate(TopologyCandidate):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        roles = kwargs.get("geometry", {}).get("output_padding", [])
        padded = {}
        for key, value in state_dict.items():
            if any(key.endswith(role + ".weight") for role in roles):
                extra = (
                    kwargs.get("geometry", {}).get("output_width", ((value.shape[0] + 511) // 512) * 512)
                    - value.shape[0]
                )
                value = torch.nn.functional.pad(value, (0, 0, 0, extra))
            padded[key] = value
        obj = super().from_state_dict(padded, **kwargs)
        obj.output_padding = set(roles)
        return obj

    def _linear(self, x, name):
        if name not in self.output_padding:
            return super()._linear(x, name)
        carry = self.carry
        self.carry = False
        try:
            out = super()._linear(x, name)
        finally:
            self.carry = carry
        if out.is_sharded():
            out = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG if self.decoding else ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.slice(out, [0] * len(out.shape), [*tuple(out.shape)[:-1], self.hidden])
        if self.decoding and carry:
            out = ttnn.to_memory_config(out, self._working_memory(out, self.geometry.get("residual_cores", 8)))
        return out
