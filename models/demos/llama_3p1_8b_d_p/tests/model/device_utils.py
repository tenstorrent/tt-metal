# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared model-device checks; host metrics and policy live in tests.utils."""

import torch

import ttnn
from models.demos.llama_3p1_8b_d_p.tests import utils
from models.demos.llama_3p1_8b_d_p.tests.utils import check_metric
from models.demos.llama_3p1_8b_d_p.tests.utils import positions as sp_positions
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import allocate_kv_cache


def hidden_limits(num_layers, dtype):
    return utils.hidden_limits(num_layers, dtype == ttnn.bfloat16)


def kv_limits(layer_idx, dtype):
    return utils.kv_limits(layer_idx, dtype == ttnn.bfloat16)


def cache_positions(sp):
    return torch.tensor([p for p in range(2048) if (p // 256) % 4 == sp])


def seed_cache(mesh_device, model, dtype):
    cache = allocate_kv_cache(mesh_device, model.mesh_config, cache_dtype=dtype)
    for name, sign in (("k", 1), ("v", -1)):
        previous = getattr(cache, name)
        sentinel = (torch.arange(64).reshape(64, 1, 1, 1) % 7 + 1).expand(64, 1, 512, 128) * (sign / 8)
        setattr(
            cache,
            name,
            ttnn.from_torch(
                sentinel.contiguous(),
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=previous.memory_config(),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
        )
        previous.deallocate(True)
    return cache


def check_hidden(hidden, expected, *, start, end, limits, label, records, enforce=True):
    assert tuple(hidden.shape) == (1, 1, 256, 4096)
    assert hidden.dtype == ttnn.bfloat16 and hidden.layout == ttnn.TILE_LAYOUT
    assert hidden.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    for chip, shard in enumerate(ttnn.get_device_tensors(hidden)):
        positions = sp_positions(start, chip // 8)
        valid = positions < end
        actual = ttnn.to_torch(shard)[0, 0]
        assert torch.isfinite(actual).all()
        if valid.any():
            check_metric(
                expected[positions[valid]], actual[valid], limits, f"{label} chip={chip}", records, enforce=enforce
            )


def check_cache(cache, before, reference, *, slot, start, end, records, enforce_accumulated=True):
    layers = reference["layers"]
    for name, tensor, snapshots in zip(("k", "v"), (cache.k, cache.v), before):
        for chip, (shard, snapshot) in enumerate(zip(ttnn.get_device_tensors(tensor), snapshots)):
            actual = ttnn.to_torch(shard)
            assert torch.isfinite(actual).all()
            sp, head = divmod(chip, 8)
            positions = cache_positions(sp)
            written = (positions >= start) & (positions < end)
            padding = (positions >= end) & (positions < (end + 31) // 32 * 32)
            untouched = ~(written | padding)
            changed_planes = set(range(slot * 32, slot * 32 + len(layers)))
            for plane in range(64):
                if plane not in changed_planes:
                    assert torch.equal(actual[plane], snapshot[plane]), (name, chip, plane, "other plane")
                    continue
                layer_idx = plane - slot * 32
                assert torch.equal(actual[plane, 0, untouched], snapshot[plane, 0, untouched]), (
                    name,
                    chip,
                    plane,
                    "outside chunk",
                )
                assert torch.count_nonzero(actual[plane, 0, padding]) == 0, (name, chip, plane, "padding")
                if written.any():
                    check_metric(
                        layers[layer_idx][name][head, positions[written]],
                        actual[plane, 0, written],
                        kv_limits(layer_idx, cache.k.dtype),
                        f"cache {name} slot={slot} layer={layer_idx} chip={chip} head={head}",
                        records,
                        enforce=enforce_accumulated or layer_idx == 0,
                    )


def check_logits(logits, reference, *, start, end, num_layers, dtype, records):
    assert tuple(logits.shape) == (1, 1, 256, 16032)
    shards = [ttnn.to_torch(shard)[0, 0].float() for shard in ttnn.get_device_tensors(logits)]
    assert all(torch.isfinite(shard).all() for shard in shards)
    positions = reference["logit_positions"]
    selected = positions[(positions >= start) & (positions < end)]
    assert selected.numel(), "every tested chunk needs reference logit positions"
    expected_rows, actual_rows = [], []
    lookup = {int(position): row for row, position in enumerate(positions)}
    for absolute in selected.tolist():
        sp = (absolute // 256) % 4
        row = sp_positions(start, sp).tolist().index(absolute)
        # TP columns are exact adjacent 16032-wide vocabulary intervals. No logits from padded
        # query rows or a different SP owner may enter token comparisons.
        actual_rows.append(torch.cat([shards[sp * 8 + tp][row] for tp in range(8)]))
        expected_rows.append(reference["logits"][lookup[absolute]])
    expected, actual = torch.stack(expected_rows), torch.stack(actual_rows)
    assert actual.shape[1] == 128256
    limits = hidden_limits(num_layers, dtype)
    return utils.check_logits_rows(expected, actual, selected, start=start, end=end, limits=limits, records=records)
