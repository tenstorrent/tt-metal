# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device coverage for the SP-sequence/TP-width DFlash accumulator."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.gpt_oss_d_p.tt.attention.kv_cache import allocate_kv_cache
from models.demos.gpt_oss_d_p.tt.dflash import (
    DFlashPrefillConfig,
    TtDFlashFeatureAccumulator,
    reference_accumulate_reduced_hidden,
)
from models.demos.gpt_oss_d_p.tt.dflash_kv import DFlashKVConfig, TtDFlashKVBuilder, reference_dflash_kv


@pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2")
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_dflash_accumulator_sp_sequence_tp_width_sharded(mesh_device, reset_seeds):
    rows, cols = tuple(mesh_device.shape)
    hidden = 64
    seq = 64
    targets = (1, 3, 5, 7, 9)
    generator = torch.Generator().manual_seed(31)
    fc = torch.randn(hidden, len(targets) * hidden, generator=generator, dtype=torch.bfloat16) * 0.02
    activations = {
        layer_id: torch.randn(1, 1, seq, hidden, generator=generator, dtype=torch.bfloat16) for layer_id in targets
    }
    cfg = DFlashPrefillConfig(
        checkpoint_path=Path("/synthetic/not-read"),
        hidden_size=hidden,
        num_target_layers=12,
        target_layer_ids=targets,
    )
    accumulator = TtDFlashFeatureAccumulator(
        mesh_device,
        cfg,
        fc,
        sp_axis=0,
        tp_axis=1,
        dtype=ttnn.bfloat16,
    )

    def to_device(x):
        return ttnn.from_torch(
            x,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                mesh_shape=tuple(mesh_device.shape),
                dims=(2, None),
            ),
        )

    tt_activations = {layer_id: to_device(value) for layer_id, value in activations.items()}

    # First prove one synthetic FC target projection in isolation.
    one_cfg = DFlashPrefillConfig(
        checkpoint_path=Path("/synthetic/not-read"),
        hidden_size=hidden,
        num_target_layers=12,
        target_layer_ids=(targets[0],),
    )
    one = TtDFlashFeatureAccumulator(
        mesh_device,
        one_cfg,
        fc[:, :hidden],
        sp_axis=0,
        tp_axis=1,
        dtype=ttnn.bfloat16,
    )
    one.tap(tt_activations[targets[0]], targets[0])
    tt_one = one.export()
    one_host = ttnn.to_torch(
        tt_one,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device,
            mesh_shape=tuple(mesh_device.shape),
            dims=(2, 3),
        ),
    ).float()
    one_reference = activations[targets[0]].float() @ fc[:, :hidden].T.float()
    passing, pcc = comp_pcc(one_reference, one_host, 0.99)
    assert passing, f"DFlash one-target projection PCC={pcc}"
    ttnn.deallocate(tt_one)

    for layer_id in targets:
        accumulator.tap(tt_activations[layer_id], layer_id)
    tt_result = accumulator.export()
    ttnn.synchronize_device(mesh_device)

    # The result remains seq/SP x feature/TP sharded.  Reconstruct both axes
    # solely for the test's torch comparison.
    assert tuple(ttnn.get_device_tensors(tt_result)[0].shape)[-2:] == (seq // rows, hidden // cols)
    result = ttnn.to_torch(
        tt_result,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device,
            mesh_shape=tuple(mesh_device.shape),
            dims=(2, 3),
        ),
    ).float()
    reference = reference_accumulate_reduced_hidden(
        activations,
        fc,
        target_layer_ids=targets,
    ).float()
    passing, pcc = comp_pcc(reference, result, 0.99)
    assert passing, f"DFlash five-target accumulator PCC={pcc}"

    for tensor in tt_activations.values():
        ttnn.deallocate(tensor)
    ttnn.deallocate(tt_result)


@pytest.mark.requires_mesh_topology(mesh_shape=(4, 8), topology="mesh-4x8")
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    ids=["line"],
    indirect=True,
)
def test_dflash_kv_tail_all_layers_offsets_padding_and_two_slots(mesh_device, device_params, reset_seeds, expect_error):
    # hidden/TP=36 is deliberately not tile-aligned, matching GPT-OSS's
    # 2880/8=360 logical shard and exercising TTNN's physical padding.
    hidden, layers, heads, head_dim = 288, 8, 8, 64
    chunk, max_seq, users = 128, 256, 2
    generator = torch.Generator().manual_seed(47)
    config = DFlashKVConfig(
        checkpoint_path=Path("/synthetic/not-read"),
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_key_value_heads=heads,
        head_dim=head_dim,
        rms_norm_eps=1e-5,
        rope_theta=150000.0,
        yarn_factor=32.0,
        yarn_orig_max_pos=4096,
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0,
    )
    weights = {"hidden_norm.weight": torch.ones(hidden)}
    for layer_idx in range(layers):
        weights[f"layers.{layer_idx}.self_attn.k_proj.weight"] = (
            torch.randn(heads * head_dim, hidden, generator=generator) * 0.02
        )
        weights[f"layers.{layer_idx}.self_attn.v_proj.weight"] = (
            torch.randn(heads * head_dim, hidden, generator=generator) * 0.02
        )
        weights[f"layers.{layer_idx}.self_attn.k_norm.weight"] = torch.ones(head_dim)

    builder = TtDFlashKVBuilder(
        mesh_device,
        config,
        weights,
        max_seq_len=max_seq,
        chunk_sizes=(chunk,),
        sp_axis=0,
        tp_axis=1,
        topology=ttnn.Topology.Linear,
    )
    cache = allocate_kv_cache(
        mesh_device,
        num_layers=layers,
        max_seq_len=max_seq,
        sp_axis=0,
        num_users=users,
        head_dim=head_dim,
    )

    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3))

    def to_device(value):
        return ttnn.from_torch(
            value,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    chunks = [
        torch.randn(1, 1, chunk, hidden, generator=generator, dtype=torch.bfloat16),
        torch.randn(1, 1, chunk, hidden, generator=generator, dtype=torch.bfloat16),
        torch.randn(1, 1, chunk, hidden, generator=generator, dtype=torch.bfloat16),
    ]
    invalid = to_device(chunks[0])
    with expect_error(ValueError, "slot_id"):
        builder.forward(invalid, cache, slot_id=users, actual_start=0, actual_end=chunk, chunk_size=chunk)
    with expect_error(ValueError, "aligned"):
        builder.forward(invalid, cache, slot_id=0, actual_start=1, actual_end=chunk, chunk_size=chunk)
    with expect_error(ValueError, "real range"):
        builder.forward(invalid, cache, slot_id=0, actual_start=0, actual_end=chunk + 1, chunk_size=chunk)
    with expect_error(ValueError, "capacity"):
        builder.forward(invalid, cache, slot_id=0, actual_start=max_seq, actual_end=max_seq + 1, chunk_size=chunk)
    ttnn.deallocate(invalid)

    # slot 0: two chunks with partial tails; slot 1: an independent full first chunk.
    calls = ((0, 0, 100, chunks[0]), (0, 128, 228, chunks[1]), (1, 0, 128, chunks[2]))
    acknowledgements = []
    for slot, start, end, host_hidden in calls:
        tt_hidden = to_device(host_hidden)
        builder.forward(
            tt_hidden,
            cache,
            slot_id=slot,
            actual_start=start,
            actual_end=end,
            chunk_size=chunk,
            on_layer_complete=acknowledgements.append,
            layer_ack_base=36,
        )
        ttnn.deallocate(tt_hidden)
    ttnn.synchronize_device(mesh_device)
    assert acknowledgements == list(range(36, 44)) * len(calls)

    positions = blockcyclic_positions(4, chunk, max_seq)
    natural_rows = torch.argsort(positions)

    def gather(tensor, slot, layer):
        batch_idx = slot * layers + layer
        device_tensors = ttnn.get_device_tensors(tensor)
        per_head = []
        for head in range(heads):
            raw = torch.cat(
                [ttnn.to_torch(device_tensors[row * heads + head])[batch_idx, 0] for row in range(4)], dim=0
            )
            per_head.append(raw[natural_rows])
        return torch.stack(per_head).unsqueeze(0).float()

    references = [reference_dflash_kv(value, weights, config, start_pos=start) for _, start, _, value in calls]
    for layer in range(layers):
        slot0_k, slot0_v = gather(cache.k, 0, layer), gather(cache.v, 0, layer)
        slot1_k, slot1_v = gather(cache.k, 1, layer), gather(cache.v, 1, layer)
        for actual, expected in (
            (slot0_k[:, :, :100], references[0][0][layer].reshape(1, heads, chunk, head_dim)[:, :, :100]),
            (slot0_v[:, :, :100], references[0][1][layer].reshape(1, heads, chunk, head_dim)[:, :, :100]),
            (
                slot0_k[:, :, 128:228],
                references[1][0][layer].reshape(1, heads, chunk, head_dim)[:, :, :100],
            ),
            (
                slot0_v[:, :, 128:228],
                references[1][1][layer].reshape(1, heads, chunk, head_dim)[:, :, :100],
            ),
            (slot1_k[:, :, :128], references[2][0][layer].reshape(1, heads, chunk, head_dim)),
            (slot1_v[:, :, :128], references[2][1][layer].reshape(1, heads, chunk, head_dim)),
        ):
            passing, pcc = comp_pcc(expected.float(), actual, 0.999)
            assert passing, f"DFlash layer {layer} context KV PCC={pcc}"
        assert torch.count_nonzero(slot0_k[:, :, 100:128]) == 0
        assert torch.count_nonzero(slot0_v[:, :, 228:]) == 0
        assert torch.count_nonzero(slot1_k[:, :, 128:]) == 0
        assert torch.count_nonzero(slot1_v[:, :, 128:]) == 0
