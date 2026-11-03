# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""EP prefill banks plus TP indexed decode; both allocated before capture."""

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.candidate_multichip_allreduce import MultichipDecoder as TP
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.candidate_multichip_ep import MultichipDecoder as EP


class MultichipDecoder(EP):
    @classmethod
    def from_state_dict(cls, sd, **kw):
        self = super().from_state_dict(sd, **kw)
        # Materialize the TP banks explicitly; other constants are unchanged.
        if self.sparse:
            import torch

            sd = {k.removeprefix(f"model.layers.{self.layer_idx}."): v for k, v in sd.items()}
            prefix = "mlp.experts."
            banks = {}
            for name in ("gate_up", "down_proj"):
                shards = []
                for rank in range(4):
                    rows = []
                    for e in range(100):
                        if name == "gate_up":
                            rows.append(
                                torch.cat(
                                    [
                                        sd[f"{prefix}{e}.{p}_proj.weight"].T.chunk(4, dim=-1)[rank]
                                        for p in ("gate", "up")
                                    ],
                                    dim=-1,
                                )
                            )
                        else:
                            rows.append(sd[f"{prefix}{e}.down_proj.weight"].T.chunk(4, dim=-2)[rank])
                    shards.append(torch.stack(rows).unsqueeze(0))
                banks["experts." + name] = ttnn.from_torch(
                    torch.cat(shards, dim=0),
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
                    dtype=ttnn.bfloat4_b,
                    layout=ttnn.TILE_LAYOUT,
                )
            self.ep_banks = {name: self.weights[name] for name in banks}
            self.weights.update(banks)
        return self

    def _experts(self, x, *, logits, value=False, valid_tokens=None):
        if value or x.shape[-2] == 32 or valid_tokens is not None:
            return TP._experts(self, x, logits=logits, value=value, valid_tokens=valid_tokens)
        saved = {name: self.weights[name] for name in self.ep_banks}
        self.weights.update(self.ep_banks)
        try:
            return EP._experts(self, x, logits=logits, value=value, valid_tokens=valid_tokens)
        finally:
            self.weights.update(saved)
