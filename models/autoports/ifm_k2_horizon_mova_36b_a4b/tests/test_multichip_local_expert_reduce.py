# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight regression: all four TP partial expert sums survive local reduce."""

import hashlib
import json

import torch

import ttnn

from ..tt.multichip_decoder import MultichipDecoder
from .fused_reference import ROOT, pcc, reference
from .multichip_activations import recorded_inputs


def main():
    torch.set_num_threads(8)
    c, _, _, sd = reference(3)
    x = recorded_inputs(3, 32, 1).reshape(1, 1, 32, 2560)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    original = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused
    try:
        model = MultichipDecoder.from_state_dict(sd, hf_config=c, layer_idx=3, mesh_device=mesh)
        tx = ttnn.from_torch(
            x, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        norm = model._norm(tx, "post_attention_layernorm")
        logits = model._linear(norm, "mlp.shared_router")[..., 512:]
        observed = {}

        def intercept(activ, ids, mapping, **kwargs):
            assert kwargs["cluster_axis"] == 1
            observed["activ"] = activ
            observed["scores"] = kwargs["scores_tensor"]
            observed["bad"] = original(activ, ids, mapping, **{**kwargs, "cluster_axis": 0})[0]
            result = original(activ, ids, mapping, **kwargs)
            observed["good"] = result[0]
            return result

        ttnn.experimental.deepseek_moe_fast_reduce_nc_fused = intercept
        rows = []

        def ranks(v):
            return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(v)]

        for batch in (2, 8, 32):
            result = model._experts_expanded(norm, logits=logits, valid_tokens=batch)
            activ, scores, bad, good = [ranks(observed[name]) for name in ("activ", "scores", "bad", "good")]
            for rank in range(4):
                weighted = (
                    activ[rank].float().reshape(100, 32, 2560) * scores[rank].float().reshape(32, 100).T.unsqueeze(-1)
                ).sum(dim=0)
                expected = weighted[:batch]
                actual = good[rank].float().reshape(32, 2560)[:batch]
                score = pcc(actual, expected)
                assert score >= 0.999
                old = bad[rank].float().reshape(32, 2560)[:batch]
                assert torch.count_nonzero(actual) > 0
                if rank:
                    assert torch.count_nonzero(old) == 0
                else:
                    assert torch.equal(old, actual)
                rows.append(
                    dict(
                        batch=batch,
                        rank=rank,
                        pcc_to_explicit_local_sum=score,
                        old_l2=old.norm().item(),
                        fixed_l2=actual.norm().item(),
                    )
                )
            result.deallocate(True)
            observed.clear()
        root = ROOT / "doc/multichip_decoder"
        (root / "local_expert_reduce_regression.json").write_text(
            json.dumps(
                dict(
                    status="pass",
                    rows=rows,
                    source_sha256=hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
                ),
                indent=2,
            )
            + "\n"
        )
        print("PASS", rows, flush=True)
    finally:
        ttnn.experimental.deepseek_moe_fast_reduce_nc_fused = original
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
