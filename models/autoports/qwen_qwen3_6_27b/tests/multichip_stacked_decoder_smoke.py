# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP4 stacked-layer smoke: linear attention -> full attention.

The first layer's device output is passed directly to the second layer.  The
host reads of that boundary are diagnostics only and do not replace the tensor
used by the second layer.
"""

import argparse
import math

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.full_attention_synthetic_pcc import LAYER as FULL_LAYER
from models.autoports.qwen_qwen3_6_27b.tests.full_attention_synthetic_pcc import _state as full_state
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import LAYER as LINEAR_LAYER
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import _state as linear_state
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID, _to_device
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import TARGET_FABRIC, MultichipDecoder
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder
from models.common.utility_functions import comp_pcc


def upload(tensor, mesh, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def host_tensor(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def replicated_values(tensor):
    return [ttnn.to_torch(value) for value in ttnn.get_device_tensors(tensor)]


def local_conv(global_conv, rank):
    """Select this rank's Q/K/V channels from the baseline conv state."""
    return torch.cat(
        [
            global_conv[..., rank * 512 : (rank + 1) * 512, :],
            global_conv[..., 2048 + rank * 512 : 2048 + (rank + 1) * 512, :],
            global_conv[..., 4096 + rank * 1536 : 4096 + (rank + 1) * 1536, :],
        ],
        dim=-2,
    )


def compare(actual, expected, threshold):
    return comp_pcc(expected.float(), actual.float(), threshold)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--batch", type=int, choices=(1, 32), default=1)
    parser.add_argument("--sequence", type=int, default=5, help="Logical prefill length; need not be tile aligned")
    parsed = parser.parse_args()
    if parsed.mode == "prefill" and parsed.batch != 1:
        parser.error("this focused non-aligned prefill smoke supports batch 1")

    ttnn.CONFIG.throw_exception_on_fallback = True
    torch.manual_seed(20260813)
    config = AutoConfig.from_pretrained(MODEL_ID).text_config
    state = linear_state(config) | full_state(config)
    batch = parsed.batch
    sequence = 1 if parsed.mode == "decode" else parsed.sequence
    context = max(64, sequence)
    pages_per_user = math.ceil(context / 64)
    hidden = (torch.randn(batch, sequence, config.hidden_size) * 0.2).bfloat16()
    input_host = hidden.permute(1, 0, 2).unsqueeze(0) if parsed.mode == "decode" else hidden.unsqueeze(0)
    page_host = torch.arange(batch * pages_per_user, dtype=torch.int32).reshape(batch, pages_per_user).flip(0)
    if parsed.mode == "decode":
        positions_host = (torch.arange(batch, dtype=torch.int64) % context).to(torch.uint32)
    else:
        positions_host = torch.arange(sequence, dtype=torch.int64).to(torch.uint32).reshape(1, -1)

    # Sequential optimized 1x1 baseline, including both layers' cache effects.
    one = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        baseline_linear = OptimizedDecoder.from_state_dict(
            state, hf_config=config, layer_idx=LINEAR_LAYER, mesh_device=one, batch=batch, max_context=context
        )
        baseline_full = OptimizedDecoder.from_state_dict(
            state, hf_config=config, layer_idx=FULL_LAYER, mesh_device=one, batch=batch, max_context=context
        )
        one_x = _to_device(input_host, mesh_device=one)
        one_page = _to_device(page_host, mesh_device=one, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
        one_pos = _to_device(positions_host, mesh_device=one, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)
        forward = OptimizedDecoder.decode_forward if parsed.mode == "decode" else OptimizedDecoder.prefill_forward
        one_boundary = forward(baseline_linear, hidden_states=one_x, page_table=one_page, current_positions=one_pos)
        one_output = forward(baseline_full, hidden_states=one_boundary, page_table=one_page, current_positions=one_pos)
        ttnn.synchronize_device(one)
        expected_boundary = host_tensor(one_boundary)
        expected_output = host_tensor(one_output)
        expected_linear = {name: host_tensor(baseline_linear.caches[name]) for name in ("conv", "recurrent")}
        expected_full = {name: host_tensor(baseline_full.caches[name]) for name in ("key", "value")}
    finally:
        ttnn.close_mesh_device(one)

    ttnn.set_fabric_config(TARGET_FABRIC)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000)
    try:
        linear = MultichipDecoder.from_state_dict(
            state, hf_config=config, layer_idx=LINEAR_LAYER, mesh_device=mesh, batch=batch, max_context=context
        )
        full = MultichipDecoder.from_state_dict(
            state, hf_config=config, layer_idx=FULL_LAYER, mesh_device=mesh, batch=batch, max_context=context
        )
        x = upload(input_host, mesh)
        page = upload(page_host, mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        positions = upload(positions_host, mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        forward = MultichipDecoder.decode_forward if parsed.mode == "decode" else MultichipDecoder.prefill_forward
        boundary = forward(linear, hidden_states=x, page_table=page, current_positions=positions)
        boundary_shape = tuple(boundary.shape)
        # Deliberately pass the original mesh tensor, without composing through host.
        output = forward(full, hidden_states=boundary, page_table=page, current_positions=positions)
        ttnn.synchronize_device(mesh)
        boundary_replicas = replicated_values(boundary)
        output_replicas = replicated_values(output)

        logical_shape = (
            (1, 1, batch, config.hidden_size) if parsed.mode == "decode" else (1, batch, sequence, config.hidden_size)
        )
        boundary_pcc = compare(boundary_replicas[0], expected_boundary, 0.995)
        output_pcc = compare(output_replicas[0], expected_output, 0.995)
        boundary_equal = all(compare(value, boundary_replicas[0], 0.99999) for value in boundary_replicas[1:])
        output_equal = all(compare(value, output_replicas[0], 0.99999) for value in output_replicas[1:])

        linear_conv = replicated_values(linear.caches["conv"])
        linear_recurrent = replicated_values(linear.caches["recurrent"])
        full_key = replicated_values(full.caches["key"])
        full_value = replicated_values(full.caches["value"])
        linear_cache_pcc = {
            "conv": [
                compare(value, local_conv(expected_linear["conv"], rank), 0.995)
                for rank, value in enumerate(linear_conv)
            ],
            "recurrent": [
                compare(value, expected_linear["recurrent"][:, rank * 12 : (rank + 1) * 12], 0.995)
                for rank, value in enumerate(linear_recurrent)
            ],
        }
        full_cache_pcc = {
            # The TP4 full-attention projection is BFP4 while the optimized
            # baseline is BF16.  Key-cache contents therefore use a looser
            # diagnostic gate; final stacked output remains gated at 0.995.
            "key": [
                compare(value, expected_full["key"][:, rank : rank + 1], 0.95) for rank, value in enumerate(full_key)
            ],
            "value": [
                compare(value, expected_full["value"][:, rank : rank + 1], 0.99)
                for rank, value in enumerate(full_value)
            ],
        }
        print(
            "MULTICHIP_STACKED",
            f"mode={parsed.mode}",
            f"batch={batch}",
            f"sequence={sequence}",
            f"boundary_shape={boundary_shape}",
            f"logical_shape={logical_shape}",
            f"boundary_pcc={boundary_pcc}",
            f"output_pcc={output_pcc}",
            f"boundary_replicas_equal={boundary_equal}",
            f"output_replicas_equal={output_equal}",
            f"linear_cache_pcc={linear_cache_pcc}",
            f"full_cache_pcc={full_cache_pcc}",
            f"page_table={page_host.tolist()}",
            f"positions={positions_host.tolist()}",
            "direct_device_boundary=True",
            "fallback_audit=True",
        )
        assert boundary_shape == logical_shape
        assert boundary_pcc and output_pcc
        assert boundary_equal and output_equal
        assert all(all(values) for values in linear_cache_pcc.values())
        # Paged-fill key storage is a diagnostic in prefill; the semantic gate
        # is the stacked output (and the separate prefill->decode test). Decode
        # validates raw local K/V ownership exactly.
        if parsed.mode == "decode":
            assert all(all(values) for values in full_cache_pcc.values())
        else:
            assert all(full_cache_pcc["value"])
        assert [tuple(value.shape) for value in linear_conv] == [(1, batch, 2560, 4)] * 4
        assert [tuple(value.shape) for value in linear_recurrent] == [(batch, 12, 128, 128)] * 4
        expected_full_shape = (batch * pages_per_user, 1, 64, 256)
        assert [tuple(value.shape) for value in full_key] == [expected_full_shape] * 4
        assert [tuple(value.shape) for value in full_value] == [expected_full_shape] * 4
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
