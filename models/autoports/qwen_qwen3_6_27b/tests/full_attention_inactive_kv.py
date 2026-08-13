# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Minimal official-weight TP4 B2 proof for inactive paged-KV preservation."""

from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_REVISION
from models.autoports.qwen_qwen3_6_27b.tt.model import SnapshotReader
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import TARGET_FABRIC, MultichipDecoder


SNAPSHOT = Path("/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots") / MODEL_REVISION


def upload(mesh, value, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        value.to(torch.bfloat16 if dtype == ttnn.bfloat16 else (torch.int32 if dtype == ttnn.int32 else torch.uint32)).contiguous(),
        device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), dtype=dtype,
        layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def snapshot(decoder):
    return {
        name: [ttnn.to_torch(shard).clone() for shard in ttnn.get_device_tensors(decoder.caches[name])]
        for name in ("key", "value")
    }


def main():
    torch.manual_seed(20260813)
    ttnn.set_fabric_config(TARGET_FABRIC)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=134_217_728)
    try:
        config = AutoConfig.from_pretrained(SNAPSHOT, local_files_only=True).text_config
        decoder = MultichipDecoder.from_state_dict(
            SnapshotReader(SNAPSHOT), hf_config=config, layer_idx=3, mesh_device=mesh,
            batch=2, max_context=64, page_size=64, candidate="default",
        )
        page = upload(mesh, torch.tensor([[0], [1]]), ttnn.int32)
        positions = upload(mesh, torch.tensor([7, 11]), ttnn.uint32)
        active = upload(mesh, torch.tensor([1, 0]), ttnn.bfloat16)
        hidden = upload(
            mesh, torch.randn(1, 1, 2, config.hidden_size, dtype=torch.bfloat16) * 0.03,
            layout=ttnn.TILE_LAYOUT,
        )
        before = snapshot(decoder)
        decoder.decode_forward(
            hidden_states=hidden, page_table=page, current_positions=positions, active_mask=active,
        )
        ttnn.synchronize_device(mesh)
        after = snapshot(decoder)
        active_changed = False
        for name in ("key", "value"):
            for rank, (old, new) in enumerate(zip(before[name], after[name])):
                assert torch.equal(old[1], new[1]), f"inactive {name} changed on rank {rank}"
                active_changed |= not torch.equal(old[0], new[0])
        assert active_changed, "active slot K/V did not change"
        print("FULL_ATTENTION_INACTIVE_KV_OK inactive_exact=true active_changed=true", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
