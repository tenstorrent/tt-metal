# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reduced TP4 evidence for mixed prompt lengths and inactive decode slots."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import default_snapshot, MODEL_REVISION
from models.autoports.qwen_qwen3_6_27b.tt.model import SnapshotReader
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import TARGET_FABRIC, MultichipDecoder
from models.common.utility_functions import comp_pcc


SNAPSHOT = default_snapshot()


def _upload(mesh, value, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        value, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT if value.ndim <= 2 else ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _metadata(mesh, lengths, physical_len):
    batch = len(lengths)
    masks, selector_chunks = [], []
    lengths = torch.tensor(lengths)
    for start in range(0, physical_len, 64):
        chunk_len = min(64, physical_len - start)
        active_len = torch.clamp(lengths - start, min=0, max=chunk_len)
        mask = (torch.arange(chunk_len).reshape(1, chunk_len) < active_len.reshape(batch, 1)).to(torch.bfloat16)
        masks.append(_upload(mesh, mask))
        selectors = []
        for lane in range(4):
            selector = torch.zeros((batch, chunk_len + 4), dtype=torch.bfloat16)
            selector[torch.arange(batch), active_len + lane] = 1
            selectors.append(_upload(mesh, selector))
        selector_chunks.append(selectors)
    return masks, selector_chunks


def _host_shards(tensor):
    return [ttnn.to_torch(part).float() for part in ttnn.get_device_tensors(tensor)]


def _snapshot(decoder):
    return {name: _host_shards(decoder.caches[name]) for name in ("conv", "recurrent")}


def _reset(decoder):
    for name in ("conv", "recurrent"):
        ttnn.multiply(decoder.caches[name], 0.0, output_tensor=decoder.caches[name])


def _compare(actual, expected, label, threshold=0.9999):
    for name in ("conv", "recurrent"):
        for rank, (got, want) in enumerate(zip(actual[name], expected[name])):
            passed, message = comp_pcc(want, got, threshold)
            print(label, name, "rank", rank, message)
            assert passed, f"{label} {name} rank {rank}: {message}"


@torch.no_grad()
def run():
    torch.manual_seed(20260813)
    ttnn.set_fabric_config(TARGET_FABRIC)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=134_217_728)
    try:
        config = AutoConfig.from_pretrained(SNAPSHOT, local_files_only=True).text_config
        decoder = MultichipDecoder.from_state_dict(
            SnapshotReader(SNAPSHOT), hf_config=config, layer_idx=0, mesh_device=mesh,
            batch=2, max_context=64, page_size=64, candidate="default",
        )
        page_table = _upload(mesh, torch.tensor([[0], [1]], dtype=torch.int32), dtype=ttnn.int32)
        positions5 = _upload(mesh, torch.arange(65, dtype=torch.int64).to(torch.uint32).repeat(2, 1), dtype=ttnn.uint32)
        hidden5_host = torch.randn(1, 2, 65, config.hidden_size, dtype=torch.bfloat16) * 0.03
        hidden5 = _upload(mesh, hidden5_host)

        mask, selectors = _metadata(mesh, [65, 63], 65)
        decoder.prefill_forward(
            hidden_states=hidden5, page_table=page_table, current_positions=positions5,
            sequence_mask=mask, conv_state_selectors=selectors,
        )
        mixed = _snapshot(decoder)

        _reset(decoder)
        mask, selectors = _metadata(mesh, [65, 65], 65)
        decoder.prefill_forward(
            hidden_states=hidden5, page_table=page_table, current_positions=positions5,
            sequence_mask=mask, conv_state_selectors=selectors,
        )
        uniform5 = _snapshot(decoder)

        _reset(decoder)
        hidden3 = _upload(mesh, hidden5_host[:, :, :63])
        positions3 = _upload(mesh, torch.arange(63, dtype=torch.int64).to(torch.uint32).repeat(2, 1), dtype=ttnn.uint32)
        mask, selectors = _metadata(mesh, [63, 63], 63)
        decoder.prefill_forward(
            hidden_states=hidden3, page_table=page_table, current_positions=positions3,
            sequence_mask=mask, conv_state_selectors=selectors,
        )
        uniform3 = _snapshot(decoder)

        expected = {}
        for name in ("conv", "recurrent"):
            expected[name] = []
            for five, three in zip(uniform5[name], uniform3[name]):
                batch_dim = 1 if name == "conv" else 0
                expected[name].append(torch.cat([five.narrow(batch_dim, 0, 1), three.narrow(batch_dim, 1, 1)], dim=batch_dim))
        _compare(mixed, expected, "MIXED_PREFILL")

        before = _snapshot(decoder)
        token = _upload(mesh, torch.randn(1, 1, 2, config.hidden_size, dtype=torch.bfloat16) * 0.03)
        pos = _upload(mesh, torch.tensor([63, 63], dtype=torch.uint32), dtype=ttnn.uint32)
        inactive = _upload(mesh, torch.zeros(2, dtype=torch.bfloat16))
        decoder.decode_forward(hidden_states=token, page_table=page_table, current_positions=pos, active_mask=inactive)
        after = _snapshot(decoder)
        _compare(after, before, "INACTIVE_DECODE", threshold=0.999999)

        one_active = _upload(mesh, torch.tensor([1, 0], dtype=torch.bfloat16))
        decoder.decode_forward(hidden_states=token, page_table=page_table, current_positions=pos, active_mask=one_active)
        one_active_after = _snapshot(decoder)
        for name in ("conv", "recurrent"):
            batch_dim = 1 if name == "conv" else 0
            for rank, (got, want) in enumerate(zip(one_active_after[name], after[name])):
                inactive_got = got.narrow(batch_dim, 1, 1)
                inactive_want = want.narrow(batch_dim, 1, 1)
                assert torch.equal(inactive_got, inactive_want), f"slot 1 changed: {name} rank {rank}"
                active_got = got.narrow(batch_dim, 0, 1)
                active_want = want.narrow(batch_dim, 0, 1)
                assert not torch.equal(active_got, active_want), f"slot 0 did not advance: {name} rank {rank}"
        print("ACTIVE_INACTIVE_DECODE_PASS")
        print("MIXED_PROMPT_STATE_PASS")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    run()
