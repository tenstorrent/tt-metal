# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Exact cache order and ownership regression on a two-plane Galaxy cache."""

import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.llama_3p1_8b_d_p.tt.attention import FullCausalAttention
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import PrefillGeometry

MESH = (4, 8)


def _prefixes(capacity):
    # Large capacity needs one full readback and one padded continuation; small cases
    # cover awkward stripe/tile tails without duplicating large transfers.
    ends = (
        (1025, capacity)
        if capacity == 131072
        else tuple(end for end in (1, 255, 257, 1025, capacity - 33, capacity - 1) if end <= capacity)
    )
    return [(end, ((end + 31) // 32) * 32) for end in ends]


def _addresses(value):
    return tuple(int(part.buffer_address()) for part in ttnn.get_device_tensors(value))


def _hash_decoded(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _two_plane_cache(mesh, capacity, dtype):
    # Only two batch planes are allocated. Packed row order is rank-major, independently
    # defined by the absolute stripe owner; the operation receives batch_index 0 or 1.
    positions = [p for rank in range(4) for p in range(capacity) if (p // 256) % 4 == rank]
    pos = torch.tensor(positions, dtype=torch.int64)[:, None]
    dim = torch.arange(128, dtype=torch.int64)[None, :]
    packed = torch.empty((2, 8, capacity, 128), dtype=torch.bfloat16)
    for slot in (0, 1):
        for head in range(8):
            code = (pos * 193 + head * 619 + dim * 71 + slot * 89) % 263
            packed[slot, head] = ((code.float() - 131) / 128).to(torch.bfloat16)
    return ttnn.from_torch(
        packed,
        device=mesh,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=MESH, dims=(2, 1)),
    )


# Prove exact all-chip order, padded prefixes and storage ownership with two distinct slots.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(MESH, id="galaxy-4x8")], indirect=True)
@pytest.mark.parametrize(
    "capacity", [1024, 2048, 8192, 131072], ids=["capacity-1024", "capacity-2048", "capacity-8192", "capacity-131072"]
)
@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bfp8"])
def test_cache_reorder_preserves_values_and_storage(mesh_device, capacity, cache_dtype, tmp_path):
    assert tuple(mesh_device.shape) == MESH and mesh_device.get_num_devices() == 32
    mesh_device.enable_program_cache()
    ctx = SimpleNamespace(max_seq_len=capacity, geometry=PrefillGeometry(capacity))
    directory = Path(os.environ.get("REORDER_EVIDENCE_DIR", str(tmp_path))) / f"{capacity}-{cache_dtype}"
    directory.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "component_two_plane_gather_reorder_exact_decoded_value_parity",
        "capacity": capacity,
        "dtype": str(cache_dtype),
        "slots": [0, 1],
        "batch_planes": 2,
        "case_passed": False,
        "full_model_accepted": False,
        "packed_byte_identity_checked": False,
        "counts": {},
        "readback": {"calls": 0, "elements": 0, "decoded_bytes": 0},
        "prefixes": [{"actual_end": end, "logical_n": length} for end, length in _prefixes(capacity)],
        "attention_sha256": hashlib.sha256(Path(inspect.getfile(FullCausalAttention)).read_bytes()).hexdigest(),
    }
    cache, gather, live = None, None, []
    finished, cleaned = False, False

    def read(part):
        result = ttnn.to_torch(part)
        report["readback"]["calls"] += 1
        report["readback"]["elements"] += result.numel()
        report["readback"]["decoded_bytes"] += result.numel() * result.element_size()
        return result

    def check(category, actual, expected, **identity):
        report["last_check"] = dict(category=category, **identity)
        assert actual.shape == expected.shape, report["last_check"]
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), report["last_check"]
        assert torch.equal(actual.float(), expected.float()), report["last_check"]
        report["counts"][category] = report["counts"].get(category, 0) + 1

    def free(value):
        value.deallocate(True)
        live[:] = [item for item in live if item is not value]

    try:
        cache = _two_plane_cache(mesh_device, capacity, cache_dtype)
        assert tuple(cache.shape) == (2, 1, capacity // 4, 128)
        expected = torch.empty((2, 8, capacity, 128), dtype=torch.float32)
        before = {}
        parts = ttnn.get_device_tensors(cache)
        assert len(parts) == 32
        for chip, part in enumerate(parts):
            value = read(part)
            assert torch.isfinite(value).all()
            before[chip] = _hash_decoded(value)
            rank, head = divmod(chip, 8)
            positions = [p for p in range(capacity) if (p // 256) % 4 == rank]
            for slot in (0, 1):
                expected[slot, head, positions] = value[slot, 0].float()
        gather = ttnn.empty(
            (1, 1, capacity, 128),
            device=mesh_device,
            dtype=cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        persistent = _addresses(gather)
        for actual_end, length in _prefixes(capacity):
            held = []
            full_large = capacity == 131072 and length == capacity
            methods = [("production", FullCausalAttention._gather_and_reorder)]
            if not full_large:
                methods.append(("repeat", FullCausalAttention._gather_and_reorder))
            for slot in (0, 1):
                outputs = []
                for name, method in methods:
                    value = method(ctx, cache, gather, batch_index=slot, logical_n=length)
                    live.append(value)
                    outputs.append(value)
                    assert value.dtype == cache_dtype and value.layout == ttnn.TILE_LAYOUT
                    assert tuple(value.shape) == (1, 1, length, 128)
                    assert all(a != b for a, b in zip(_addresses(value), persistent))
                    if name == "production":
                        # Read the first result once, after the other slot reuses persistent gather.
                        held.append((slot, value))
                    else:
                        for chip, part in enumerate(ttnn.get_device_tensors(value)):
                            check(
                                "output",
                                read(part),
                                expected[slot : slot + 1, chip % 8 : chip % 8 + 1, :length],
                                method=name,
                                slot=slot,
                                actual_end=actual_end,
                                logical_n=length,
                                chip=chip,
                            )
                assert all(
                    len({addresses[chip] for addresses in map(_addresses, outputs)}) == len(methods)
                    for chip in range(32)
                ), "simultaneously live outputs share storage"
                for (name, _), value in zip(methods, outputs):
                    if name != "production":
                        free(value)
            gc.collect()
            for slot, value in held:
                for chip, part in enumerate(ttnn.get_device_tensors(value)):
                    check(
                        "retained_after_reuse",
                        read(part),
                        expected[slot : slot + 1, chip % 8 : chip % 8 + 1, :length],
                        slot=slot,
                        actual_end=actual_end,
                        logical_n=length,
                        chip=chip,
                    )
                free(value)
            # An owning small slice proves persistent storage survives caller deallocation
            # without rereading the entire configured capacity.
            assert _addresses(gather) == persistent
            probe = ttnn.slice(gather, [0, 0, 0, 0], [1, 1, 32, 128])
            live.append(probe)
            assert all(a != b for a, b in zip(_addresses(probe), persistent))
            for chip, part in enumerate(ttnn.get_device_tensors(probe)):
                check(
                    "borrowed_gather_alive",
                    read(part),
                    expected[1:2, chip % 8 : chip % 8 + 1, :32],
                    logical_n=length,
                    chip=chip,
                )
            free(probe)
        for chip, part in enumerate(ttnn.get_device_tensors(cache)):
            assert _hash_decoded(read(part)) == before[chip], chip
            report["counts"]["cache_unchanged"] = report["counts"].get("cache_unchanged", 0) + 1
        points = len(_prefixes(capacity))
        assert report["counts"] == {
            "output": (points - int(capacity == 131072)) * 64,
            "retained_after_reuse": points * 64,
            "borrowed_gather_alive": points * 32,
            "cache_unchanged": 32,
        }
        finished = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            for value in live:
                value.deallocate(True)
            if gather is not None:
                gather.deallocate(True)
            if cache is not None:
                cache.deallocate(True)
            ttnn.synchronize_device(mesh_device)
            cleaned = True
        except BaseException as error:
            report["cleanup_error"] = repr(error)
            raise
        finally:
            report["case_passed"] = finished and cleaned and "error" not in report
            (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
