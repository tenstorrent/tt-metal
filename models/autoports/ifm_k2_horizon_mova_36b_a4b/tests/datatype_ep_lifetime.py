# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare EP weighting buffer reuse with the original out-of-place arithmetic."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn

from ..tt.multichip_decoder import MultichipDecoder
from ..tt.precision import layer_options, load_precision
from .fused_reference import reference
from .multichip_activations import recorded_inputs

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/datatype_sweep"


def main():
    torch.set_num_threads(8)
    config, _, _, state = reference(3)
    policy = load_precision(DOC / "configs/expert_gate_up_bfp8_LoFi.json")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    rows = []
    multiply = ttnn.mul

    def out_of_place(*a, **kw):
        kw.pop("output_tensor", None)
        return multiply(*a, **kw)

    def read(tensor):
        return [ttnn.to_torch(t).clone() for t in ttnn.get_device_tensors(tensor)]

    try:
        layer = MultichipDecoder.from_state_dict(
            state, hf_config=config, layer_idx=3, mesh_device=mesh, options=layer_options(policy, 3)
        )
        source = recorded_inputs(3, 1, 64).reshape(1, 1, 64, 2560)
        for dtype in ("bfloat8_b", "bfloat4_b"):
            layer.options["ep_expert_activation"] = dtype
            for tokens in (64, 4096):
                # Repeated real activations isolate the same weighting arithmetic at the failing shape.
                x = ttnn.from_torch(
                    source.repeat(1, 1, tokens // 64, 1),
                    device=mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )
                norm = layer._norm(x, "post_attention_layernorm")
                logits = layer._linear(norm, "mlp.shared_router")[..., 512:]
                with patch.object(ttnn, "mul", out_of_place):
                    expected_tt = layer._experts_ep(norm, logits)
                expected = read(expected_tt)
                del expected_tt
                actual = layer._experts_ep(norm, logits)
                for a, b in zip(read(actual), expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                del actual
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                captured = layer._experts_ep(norm, logits)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                for replay in range(3):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                    for rank, (a, b) in enumerate(zip(read(captured), expected)):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                        rows.append(
                            dict(tokens=tokens, ep_output_dtype=dtype, replay=replay, rank=rank, bitwise_equal=True)
                        )
                ttnn.release_trace(mesh, tid)
                del captured, expected, norm, logits, x
        (DOC / "debug/ep_lifetime.json").write_text(
            json.dumps(
                dict(
                    status="pass",
                    source_sha256=hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
                    input_provenance="Recorded layer3 B1/S64 activations, repeated to4096 for allocation-shape coverage; actual checkpoint weights",
                    oracle="Same quantized matmuls and weighting; original out-of-place multiplication",
                    rows=rows,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
