# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused TP4 linear-attention decode comparison to optimized 1x1 TTNN."""

import argparse

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import LAYER, _state
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


def upload_fractured(tensor, mesh):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def args(mesh, batch):
    page = upload(
        torch.arange(batch, dtype=torch.int32).reshape(batch, 1), mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    pos = upload(torch.zeros(batch, dtype=torch.uint32), mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    return page, pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--sequence", type=int, default=5)
    parsed = parser.parse_args()
    ttnn.CONFIG.throw_exception_on_fallback = True
    torch.manual_seed(20260813)
    config = AutoConfig.from_pretrained(MODEL_ID).text_config
    state = _state(config)
    batch = 1
    sequence = 1 if parsed.mode == "decode" else parsed.sequence
    hidden = (torch.randn(batch, sequence, config.hidden_size) * 0.2).bfloat16()

    one = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    baseline = OptimizedDecoder.from_state_dict(
        state, hf_config=config, layer_idx=LAYER, mesh_device=one, batch=batch, max_context=64
    )
    x = _to_device(hidden.unsqueeze(0), mesh_device=one)
    page, pos = args(one, batch)
    if parsed.mode == "decode":
        expected_tt = baseline.decode_forward(hidden_states=x, page_table=page, current_positions=pos)
    else:
        pos = _to_device(
            torch.arange(sequence, dtype=torch.int64).to(torch.uint32).reshape(1, -1),
            mesh_device=one,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
        )
        expected_tt = baseline.prefill_forward(hidden_states=x, page_table=page, current_positions=pos)
    ttnn.synchronize_device(one)
    expected = ttnn.to_torch(ttnn.get_device_tensors(expected_tt)[0])
    ttnn.close_mesh_device(one)

    ttnn.set_fabric_config(TARGET_FABRIC)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000)
    try:
        decoder = MultichipDecoder.from_state_dict(
            state, hf_config=config, layer_idx=LAYER, mesh_device=mesh, batch=batch, max_context=64
        )
        x = upload(hidden.unsqueeze(0), mesh)
        page, pos = args(mesh, batch)
        if parsed.mode == "decode":
            output = decoder.decode_forward(hidden_states=x, page_table=page, current_positions=pos)
        else:
            pos = upload(
                torch.arange(sequence, dtype=torch.int64).to(torch.uint32).reshape(1, -1),
                mesh,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            output = decoder.prefill_forward(hidden_states=x, page_table=page, current_positions=pos)
        ttnn.synchronize_device(mesh)
        replicas = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output)]
        passed, message = comp_pcc(expected.float(), replicas[0].float(), 0.995)
        replica_pass = all(comp_pcc(replicas[0].float(), value.float(), 0.99999)[0] for value in replicas[1:])
        state_shapes = [tuple(t.shape) for t in ttnn.get_device_tensors(decoder.caches["recurrent"])]
        conv_shapes = [tuple(t.shape) for t in ttnn.get_device_tensors(decoder.caches["conv"])]
        print(
            "MULTICHIP_LINEAR",
            f"mode={parsed.mode}",
            f"sequence={sequence}",
            message,
            f"replicas_equal={replica_pass}",
            f"state_shapes={state_shapes}",
            f"conv_shapes={conv_shapes}",
            "fallback_audit=True",
        )
        assert passed, message
        assert replica_pass
        assert state_shapes == [(batch, 12, 128, 128)] * 4
        assert conv_shapes == [(1, batch, 2560, 4)] * 4
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
