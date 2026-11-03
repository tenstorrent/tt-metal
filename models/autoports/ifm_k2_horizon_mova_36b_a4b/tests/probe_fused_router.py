# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Same-logit real-weight control for the rejected BF16 generalized router."""

import json

import torch

import ttnn

from ..tt.functional_decoder import FunctionalDecoder
from .candidate_generalized_router import FusedDecoder as Candidate
from .fused_reference import ROOT, pcc, reference


def main():
    torch.set_num_threads(8)
    torch.manual_seed(42)
    c, hf, rope, sd = reference(3)
    x = (torch.randn(1, 256, 2560) * 0.1).bfloat16()
    with torch.no_grad():
        normed = hf.input_layernorm(x)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0])
    try:
        model = Candidate.from_state_dict(sd, hf_config=c, layer_idx=3, mesh_device=mesh)
        tx = ttnn.from_torch(normed.unsqueeze(0), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        def cpu(t):
            return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

        results = []
        for name, k, e in [("self_attn.v_router", 4, 64), ("mlp.gate", 8, 100)]:
            # The MLP control intentionally uses the same fixed real normalized
            # activations, isolating routing arithmetic, not a full-layer oracle.
            logits = model._linear(tx, name)
            old_linear = model._linear
            model._linear = lambda x, n: logits
            reference_routes = cpu(FunctionalDecoder._route(model, tx, name, k, e))
            candidate_routes = cpu(model._route(tx, name, k, e))
            model._linear = old_linear
            rows_a = reference_routes.reshape(256, e)
            rows_b = candidate_routes.reshape(256, e)
            changed = ((rows_a != 0) != (rows_b != 0)).any(dim=-1)
            # Same selected indices, recomputed FP32 scores: isolates selection
            # from candidate normalization, using the identical device logits.
            scores = cpu(ttnn.sigmoid(ttnn.typecast(logits, ttnn.float32)))[..., :e].reshape(256, e)
            controlled = scores * (rows_b != 0)
            controlled = (controlled / controlled.sum(-1, keepdim=True) * 2.5).bfloat16()
            results.append(
                dict(
                    router=name,
                    logit_source="one shared TT linear output",
                    tokens=256,
                    changed_expert_sets=int(changed.sum()),
                    route_pcc=pcc(rows_a, rows_b),
                    candidate_vs_same_selection_fp32_weights_pcc=pcc(rows_b, controlled),
                    bias_min=float(sd[name + ".bias"].min()),
                    bias_max=float(sd[name + ".bias"].max()),
                )
            )
            print(results[-1], flush=True)
        (ROOT / "doc/fused_decoder/router_probe.json").write_text(json.dumps(results, indent=2))
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
