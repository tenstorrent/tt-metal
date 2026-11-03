# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.scripts.prepare_expert_placement import prepare, shard_hashes, weight_path
from models.demos.deepseek_v3_d_p.tt.moe.expert_placement import PLACEMENT_FILE, load_placement, validate_mapping


def permutation():
    generator = torch.Generator().manual_seed(1618)
    return torch.cat([torch.randperm(64, generator=generator) + 64 * group for group in range(4)]).tolist()


def marker(mapping):
    inverse = [0] * 256
    for logical, physical in enumerate(mapping):
        inverse[physical] = logical
    return {
        "version": 1,
        "layer": 3,
        "num_experts": 256,
        "complete": True,
        "kind": "bijective_within_tp_group",
        "mesh_shape": [8, 4],
        "expert_dtype": "BFLOAT4_B",
        "logical_to_physical": mapping,
        "physical_to_logical": inverse,
    }


def test_placement_marker_fails_closed(tmp_path, expect_error):
    mesh = SimpleNamespace(arch=lambda: ttnn.Arch.BLACKHOLE, get_num_devices=lambda: 32, shape=[8, 4])
    kwargs = dict(
        num_experts=256,
        top_k=8,
        experts_per_chip=8,
        dispatch_group_size=8,
        num_dispatch_groups=4,
        expert_dtype=ttnn.bfloat4_b,
    )
    load = lambda **kw: load_placement(tmp_path, 3, mesh, **(kwargs | kw))
    assert load(cache_only=True) is None
    mapping = permutation()
    data = marker(mapping)
    path = tmp_path / PLACEMENT_FILE.format(layer=3)
    path.write_text(json.dumps(data))
    assert load(cache_only=True) == mapping
    with expect_error(ValueError, "cache-only"):
        load(cache_only=False)
    for invalid in [
        dict(num_experts=128),
        dict(top_k=4),
        dict(experts_per_chip=4),
        dict(dispatch_group_size=4),
        dict(num_dispatch_groups=8),
    ]:
        with expect_error(ValueError, "SP=8/TP=4"):
            load(cache_only=True, **invalid)
    with expect_error(ValueError, "BF4"):
        load(cache_only=True, expert_dtype=ttnn.bfloat8_b)
    data["complete"] = False
    path.write_text(json.dumps(data))
    with expect_error(ValueError, "Incomplete"):
        load(cache_only=True)
    data = marker(mapping)
    data["physical_to_logical"][0], data["physical_to_logical"][1] = (
        data["physical_to_logical"][1],
        data["physical_to_logical"][0],
    )
    path.write_text(json.dumps(data))
    with expect_error(ValueError, "inverse"):
        load(cache_only=True)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "cross_group", "boolean"])
def test_mapping_must_retain_each_expert_in_its_group(kind, expect_error):
    mapping = list(range(256))
    if kind == "missing":
        mapping.pop()
    elif kind == "duplicate":
        mapping[1] = mapping[0]
    elif kind == "cross_group":
        mapping[0], mapping[64] = mapping[64], mapping[0]
    else:
        mapping[0] = False
    with expect_error(ValueError, "GLM placement"):
        validate_mapping(mapping)


def test_repacking_preserves_every_quantized_shard(tmp_path, expect_error):
    source, target = tmp_path / "original", tmp_path / "placed"
    physical = permutation()
    inverse = [0] * 256
    for p, e in enumerate(physical):
        inverse[e] = p
    plan = {
        "kind": "bijective_within_tp_group",
        "layers": {"3": {"enabled": True, "physical_to_logical": physical, "logical_to_physical": inverse}},
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    original_hashes = {}
    for projection in ("gate", "up", "down"):
        for slot in range(8):
            shards = []
            for row in range(8):
                for col in range(4):
                    expert = (col * 8 + row) * 8 + slot
                    values = torch.randn(
                        (1, 1, 32, 32),
                        generator=torch.Generator().manual_seed(
                            123 + expert + 1000 * ("gate", "up", "down").index(projection)
                        ),
                    )
                    shards.append(ttnn.from_torch(values, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT))
            tensor = ttnn.from_host_shards(shards, ttnn.MeshShape(8, 4))
            path = source / weight_path(3, slot, projection)
            path.parent.mkdir(parents=True, exist_ok=True)
            ttnn.dump_tensor(path, tensor, mode=ttnn.DumpTensorMode.LOCAL)
            original_hashes[(projection, slot)] = shard_hashes(path)
    # An unrelated cache file must stay unchanged and linked to the source.
    extra = source / "unchanged.tensorbin"
    extra.write_bytes(b"unchanged cached payload")
    with expect_error(ValueError, "separate"):
        prepare(source, source / "nested", plan_path)
    prepare(source, target, plan_path)
    assert (target / "unchanged.tensorbin").is_symlink()
    assert (target / "unchanged.tensorbin").read_bytes() == extra.read_bytes()
    saved = json.loads((target / PLACEMENT_FILE.format(layer=3)).read_text())
    assert saved["complete"] is True and saved["logical_to_physical"] == inverse
    for projection in ("gate", "up", "down"):
        for slot in range(8):
            result = shard_hashes(target / weight_path(3, slot, projection))
            for row in range(8):
                for col in range(4):
                    logical = physical[(col * 8 + row) * 8 + slot]
                    assert (
                        result[(row, col)]
                        == original_hashes[(projection, logical % 8)][(logical // 8 % 8, logical // 64)]
                    )
    with expect_error(ValueError, "destination must not exist"):
        prepare(source, target, plan_path)
