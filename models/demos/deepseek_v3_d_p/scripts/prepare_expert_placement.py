# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a separate placed BF4 expert cache without changing any quantized shard.

Source and destination are the actual <model>_bh_32dev/8x4 cache directories (GLM-5.3, Kimi-K3, ...);
the expert count, and so experts per chip, comes from the plan.
The plan uses kind=bijective_within_tp_group and global layer-number keys.
Keep source available: unchanged tensors are symlinked. Completion markers enable
matching router translation and reject partially prepared caches.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.expert_placement import NUM_DEVICES, PLACEMENT_FILE, validate_mapping


def shard_hashes(path):
    # Read the documented Tensor/TensorShard/InlineFileStorage flatbuffer schema.
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        h = f.read(header_size)

        def u32(p):
            return struct.unpack_from("<I", h, p)[0]

        def field(t, n):
            v = t - struct.unpack_from("<i", h, t)[0]
            sz = struct.unpack_from("<H", h, v)[0]
            off = struct.unpack_from("<H", h, v + 4 + 2 * n)[0] if 4 + 2 * n < sz else 0
            return t + off if off else None

        def target(p):
            return p + u32(p)

        root = u32(0)
        vec = target(field(root, 2))
        count = u32(vec)
        result = {}
        for i in range(count):
            t = target(vec + 4 + 4 * i)
            assert h[field(t, 0)] == 1
            data = target(field(t, 1))
            offset, size = struct.unpack_from("<QQ", h, data)
            coord = target(field(t, 2))
            cv = target(field(coord, 0))
            dims = u32(cv)
            coords = tuple(u32(cv + 4 + 4 * j) for j in range(dims))
            assert coords not in result
            f.seek(8 + header_size + offset)
            remain = size
            digest = hashlib.sha256()
            while remain:
                b = f.read(min(remain, 4 * 1024 * 1024))
                assert b
                digest.update(b)
                remain -= len(b)
            result[coords] = digest.hexdigest()
    return result


def weight_path(layer, slot, projection):
    return Path(f"layer_{layer}.routed_expert.local_{slot}_{projection}_dtype_BFLOAT4_B_layout_TILE.tensorbin")


def selected_layers(plan):
    if plan.get("kind") != "bijective_within_tp_group" or not isinstance(plan.get("layers"), dict):
        raise ValueError("Expected a bijective_within_tp_group placement plan with layers")
    selected, num_experts = {}, None
    for key, entry in plan["layers"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid layer entry: {key}")
        layer = int(key)
        if str(layer) != key or layer < 0 or type(entry.get("enabled")) is not bool:
            raise ValueError(f"Invalid layer entry: {key}")
        physical = validate_mapping(entry.get("physical_to_logical"), num_experts)
        num_experts = len(physical)
        inverse = [0] * num_experts
        for slot, logical in enumerate(physical):
            inverse[logical] = slot
        if "logical_to_physical" in entry and validate_mapping(entry["logical_to_physical"], num_experts) != inverse:
            raise ValueError(f"Inconsistent inverse for layer {layer}")
        if not entry["enabled"]:
            if physical != list(range(num_experts)):
                raise ValueError(f"Disabled layer {layer} must use the identity permutation")
            continue
        selected[layer] = (physical, inverse)
    if not selected:
        raise ValueError("The plan enables no layers")
    return selected, num_experts


def prepare(source, destination, plan_path):
    source = source.resolve()
    destination = destination.resolve()
    if source == destination or source in destination.parents:
        raise ValueError("Destination must be separate from the original cache")
    if not source.is_dir() or destination.exists():
        raise ValueError("Source must be a cache directory and destination must not exist")
    if list(source.glob("layer_*.expert_placement.json")):
        raise ValueError("Source already has placement markers; use the original logical-order cache")
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    selected, num_experts = selected_layers(plan)
    per_chip = num_experts // NUM_DEVICES
    moved = {
        weight_path(layer, slot, proj)
        for layer in selected
        for slot in range(per_chip)
        for proj in ("gate", "up", "down")
    }
    for relative in moved:
        if not (source / relative).is_file():
            raise FileNotFoundError(source / relative)
    destination.mkdir(parents=True, exist_ok=False)
    markers = {}
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    # Fail closed if a partially prepared cache is opened. Completion is per layer.
    for layer, (physical, inverse) in selected.items():
        marker = destination / PLACEMENT_FILE.format(layer=layer)
        marker.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "layer": layer,
            "num_experts": num_experts,
            "complete": False,
            "kind": plan["kind"],
            "mesh_shape": [8, 4],
            "expert_dtype": "BFLOAT4_B",
            "physical_to_logical": physical,
            "logical_to_physical": inverse,
            "source_cache": str(source),
            "plan_sha256": plan_hash,
            "files": {},
        }
        marker.write_text(json.dumps(data, indent=2) + "\n")
        markers[layer] = (marker, data)
    # Link every unchanged file, not only tensorbins: some models' caches also carry completion
    # markers (Kimi-K3's hidden .layer_N.complete) that their loaders check before reading.
    for path in source.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative not in moved:
            target.symlink_to(path)
    for layer, (permutation, _) in sorted(selected.items()):
        marker, data = markers[layer]
        for projection in ("gate", "up", "down"):
            relatives = [weight_path(layer, slot, projection) for slot in range(per_chip)]
            originals = [ttnn.load_tensor(source / relative) for relative in relatives]
            shards = [ttnn.get_device_tensors(tensor) for tensor in originals]
            hashes = [shard_hashes(source / relative) for relative in relatives]
            if any(len(part) != 32 for part in shards) or any(len(part) != 32 for part in hashes):
                raise ValueError("Expected 32 host shards for each expert weight file")
            for slot, relative in enumerate(relatives):
                ordered, expected = [], {}
                for row in range(8):
                    for col in range(4):
                        physical = (col * 8 + row) * per_chip + slot
                        logical = permutation[physical]
                        old_slot, old_row, old_col = logical % per_chip, logical // per_chip % 8, logical // (8 * per_chip)
                        ordered.append(shards[old_slot][old_row * 4 + old_col])
                        expected[(row, col)] = hashes[old_slot][(old_row, old_col)]
                merged = ttnn.from_host_shards(ordered, ttnn.MeshShape(8, 4))
                merged.update_tensor_topology(originals[slot].tensor_topology())
                if merged.shape != originals[slot].shape or merged.dtype != ttnn.bfloat4_b:
                    raise ValueError("Expert cache shape or precision changed")
                target = destination / relative
                temporary = target.with_name(target.stem + ".partial.tensorbin")
                ttnn.dump_tensor(temporary, merged, mode=ttnn.DumpTensorMode.LOCAL)
                actual = shard_hashes(temporary)
                if actual != expected:
                    raise ValueError(f"Quantized weight bytes changed: {relative}")
                check = ttnn.load_tensor(temporary)
                if check.tensor_topology() != originals[slot].tensor_topology():
                    raise ValueError(f"Host shard topology changed: {relative}")
                temporary.replace(target)
                data["files"][relative.name] = {str(coord): digest for coord, digest in actual.items()}
                del check, merged, ordered
            del originals, shards, hashes
        data["complete"] = True
        temporary_marker = marker.with_suffix(".partial.json")
        temporary_marker.write_text(json.dumps(data, indent=2) + "\n")
        temporary_marker.replace(marker)
        print(f"Layer {layer}: {3 * per_chip} files, {3 * per_chip * NUM_DEVICES} quantized shards verified", flush=True)
    print(f"Prepared {len(selected)} placed layers in {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.destination, args.plan)


if __name__ == "__main__":
    main()
