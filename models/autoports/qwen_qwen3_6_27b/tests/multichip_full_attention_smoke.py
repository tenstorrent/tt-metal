# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused TP4 full-attention hardware smoke and baseline comparison."""

import argparse
import math

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.full_attention_synthetic_pcc import LAYER, _state
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID, _to_device
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import TARGET_FABRIC, MultichipDecoder
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("decode", "prefill", "prefill_decode"), default="decode")
    parser.add_argument("--sequence", type=int, default=33)
    parser.add_argument("--batch", type=int, choices=(1, 32), default=1)
    parsed = parser.parse_args()
    ttnn.CONFIG.throw_exception_on_fallback = True
    torch.manual_seed(20260813)
    config = AutoConfig.from_pretrained(MODEL_ID).text_config
    state = _state(config)
    batch = parsed.batch
    sequence = 1 if parsed.mode == "decode" else parsed.sequence
    context = max(128 if batch > 1 else 64, sequence)
    hidden = (
        torch.randn(batch, sequence, config.hidden_size) * 0.2 + torch.arange(batch).reshape(-1, 1, 1) * 0.01
    ).bfloat16()
    pages_per_user = math.ceil(context / 64)
    page_host = torch.arange(batch * pages_per_user, dtype=torch.int32).reshape(batch, pages_per_user).flip(0)
    decode_positions = (
        (torch.arange(batch, dtype=torch.int64) % context).to(torch.uint32)
        if batch > 1
        else torch.zeros(1, dtype=torch.uint32)
    )
    input_host = hidden.permute(1, 0, 2).unsqueeze(0) if parsed.mode == "decode" else hidden.unsqueeze(0)
    continuation = (torch.randn(1, 1, batch, config.hidden_size) * 0.2).bfloat16()

    # Serialized 1x1 optimized baseline.
    from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder

    one = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    baseline = OptimizedDecoder.from_state_dict(
        state, hf_config=config, layer_idx=LAYER, mesh_device=one, batch=batch, max_context=context
    )
    x = _to_device(input_host, mesh_device=one)
    page = _to_device(page_host, mesh_device=one, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
    pos = _to_device(decode_positions, mesh_device=one, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)
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
        if parsed.mode == "prefill_decode":
            next_x = _to_device(continuation, mesh_device=one)
            next_pos = _to_device(
                torch.full((batch,), sequence, dtype=torch.uint32),
                mesh_device=one,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
            )
            expected_tt = baseline.decode_forward(hidden_states=next_x, page_table=page, current_positions=next_pos)
    ttnn.synchronize_device(one)
    expected = ttnn.to_torch(ttnn.get_device_tensors(expected_tt)[0])
    expected_caches = {
        name: ttnn.to_torch(ttnn.get_device_tensors(baseline.caches[name])[0]) for name in ("key", "value")
    }
    ttnn.close_mesh_device(one)

    # Target 1x4 ring.
    ttnn.set_fabric_config(TARGET_FABRIC)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000)
    try:
        decoder = MultichipDecoder.from_state_dict(
            state, hf_config=config, layer_idx=LAYER, mesh_device=mesh, batch=batch, max_context=context
        )
        x = upload(input_host, mesh)
        page = upload(page_host, mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pos = upload(decode_positions, mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
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
            if parsed.mode == "prefill_decode":
                next_x = upload(continuation, mesh)
                next_pos = upload(
                    torch.full((batch,), sequence, dtype=torch.uint32),
                    mesh,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                output = decoder.decode_forward(hidden_states=next_x, page_table=page, current_positions=next_pos)
        ttnn.synchronize_device(mesh)
        replicas = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output)]
        passed, message = comp_pcc(expected.float(), replicas[0].float(), 0.995)
        replica_pcc = all(comp_pcc(replicas[0].float(), value.float(), 0.99999)[0] for value in replicas[1:])
        cache_shapes = [tuple(t.shape) for t in ttnn.get_device_tensors(decoder.caches["key"])]
        cache_pcc = {}
        for name in ("key", "value"):
            local = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(decoder.caches[name])]
            cache_pcc[name] = [
                comp_pcc(expected_caches[name][:, rank : rank + 1].float(), value.float(), 0.999)[0]
                for rank, value in enumerate(local)
            ]
        expected_cache_shape = (batch * pages_per_user, 1, 64, 256)
        print(
            "MULTICHIP_FULL",
            f"mode={parsed.mode}",
            f"batch={batch}",
            f"sequence={sequence}",
            message,
            f"replicas_equal={replica_pcc}",
            f"cache_shapes={cache_shapes}",
            f"cache_pcc={cache_pcc}",
            f"page_table={page_host.tolist()}",
            f"positions={decode_positions.tolist()}",
            "fallback_audit=True",
        )
        assert passed, message
        assert replica_pcc
        assert cache_shapes == [expected_cache_shape] * 4
        if parsed.mode == "decode":
            assert all(all(values) for values in cache_pcc.values())
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
