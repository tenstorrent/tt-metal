# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real recorded activations: BFP4 final outputs aggregate through legal BF16 inputs."""

import json
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn

from ..tt.multichip_decoder import MultichipDecoder
from ..tt.precision import layer_options, load_precision
from .fused_reference import pcc, reference
from .multichip_activations import recorded_inputs

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/datatype_sweep"


def main():
    torch.set_num_threads(8)
    c, _, _, state = reference(3)
    policy = load_precision(DOC / "configs/expert_activations_bfp4.json")
    options = layer_options(policy, 3)
    # Exercise the optional EP MLP BFP4 boundary too, although the coarse candidate retains BFP8 there.
    options["ep_expert_activation"] = "bfloat4_b"
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    rows = []
    original = ttnn.experimental.fast_reduce_nc
    fused = ttnn.experimental.deepseek_moe_fast_reduce_nc_fused
    cast = ttnn.typecast
    phase = [""]

    def ranks(x):
        return [ttnn.to_torch(v).float() for v in ttnn.get_device_tensors(x)]

    def observe(x, *args, **kw):
        assert x.dtype in (ttnn.bfloat16, ttnn.bfloat8_b), x.dtype
        out = original(x, *args, **kw)
        if kw.get("dims") == [1]:
            for rank, (inp, result) in enumerate(zip(ranks(x), ranks(out))):
                score = pcc(inp.sum(dim=1, keepdim=True), result)
                assert score >= 0.999, (phase[0], rank, score)
                rows.append(
                    dict(phase=phase[0], operation="fast_reduce_nc", rank=rank, input_dtype=str(x.dtype), pcc=score)
                )
        return out

    def observe_fused(x, ids, mapping, **kw):
        assert x.dtype == ttnn.bfloat16 and kw["cluster_axis"] == 1
        out = fused(x, ids, mapping, **kw)
        for rank, (inp, scores, result) in enumerate(zip(ranks(x), ranks(kw["scores_tensor"]), ranks(out[0]))):
            expected = (inp.reshape(100, 32, 2560) * scores.reshape(32, 100).T.unsqueeze(-1)).sum(0)
            score = pcc(expected, result.reshape(32, 2560))
            assert score >= 0.999, (phase[0], rank, score)
            rows.append(dict(phase=phase[0], operation="fused_reduce", rank=rank, input_dtype=str(x.dtype), pcc=score))
        return out

    casts = []

    def observe_cast(x, dtype, *a, **kw):
        if x.dtype == ttnn.bfloat4_b and dtype == ttnn.bfloat16:
            casts.append(dict(phase=phase[0], shape=list(x.shape)))
        return cast(x, dtype, *a, **kw)

    try:
        layer = MultichipDecoder.from_state_dict(state, hf_config=c, layer_idx=3, mesh_device=mesh, options=options)
        for tokens in (32, 64):
            x = recorded_inputs(3, 1, tokens).reshape(1, 1, tokens, 2560)
            tx = ttnn.from_torch(
                x,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            norm = layer._norm(tx, "post_attention_layernorm")
            logits = layer._linear(norm, "mlp.shared_router")[..., 512:]
            attn = layer._norm(tx, "input_layernorm")
            packed = layer._linear(attn, "attention_packed")
            value_logits = packed[..., 2304:]
            with (
                patch.object(ttnn.experimental, "fast_reduce_nc", observe),
                patch.object(ttnn.experimental, "deepseek_moe_fast_reduce_nc_fused", observe_fused),
                patch.object(ttnn, "typecast", observe_cast),
            ):
                if tokens == 32:
                    for batch in (1, 2, 8, 32):
                        phase[0] = f"indexed_or_expanded_b{batch}"
                        layer._experts(norm, logits=logits, valid_tokens=batch)
                else:
                    phase[0] = "ep_mlp"
                    layer._experts_ep(norm, logits)
                    phase[0] = "ep_value"
                    layer._experts_ep_value(attn, value_logits)
                    phase[0] = "expanded_value_fallback"
                    layer._experts_expanded(attn, logits=value_logits, value=True)
            ttnn.synchronize_device(mesh)
        assert {r["phase"] for r in casts} == {
            "indexed_or_expanded_b1",
            "indexed_or_expanded_b2",
            "indexed_or_expanded_b8",
            "indexed_or_expanded_b32",
            "ep_mlp",
            "ep_value",
            "expanded_value_fallback",
        }
        (DOC / "debug/expert_aggregation.json").write_text(
            json.dumps(dict(status="pass", rows=rows, promotions=casts, options=options), indent=2) + "\n"
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
