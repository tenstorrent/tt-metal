# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Saved reference outputs for the MoE block's C++ ops, compared bit-exact against the C++ op on the same inputs.

Layout (root: $MOE_BLOCK_REFS; unset: no reference data - list_cases() is empty and the reference-data tests skip,
so CI needs no saved cases; the lead's cases are at /data/mstaletovic/jobs/port/refs):

    <root>/<op>/<case>/meta.json        what the case is: op, case, mesh_shape, seed, model, config (the op's
                                        arguments), inputs, outputs, tensors
    <root>/<op>/<case>/chip_<r>_<c>.pt  torch.save({name: torch.Tensor}) of chip (r, c): every input and output of
                                        that chip, exactly as ttnn.to_torch reads the chip's shard back (the device
                                        tensor's per-chip view, dtype and layout as on device, no padding removed)

A device test loads the case, puts each chip's inputs on its chip (to_mesh), runs the op and compares each output per
chip with assert_bit_exact. This module imports ttnn only inside the functions that need it. Tests under this
directory import it as `import ref_harness` (conftest.py puts this directory on sys.path).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

REFS_ROOT: Path | None = Path(os.environ["MOE_BLOCK_REFS"]) if os.environ.get("MOE_BLOCK_REFS") else None


def refs_available(root: Path | None = None) -> bool:
    """True when a reference root is given or $MOE_BLOCK_REFS is set (reference-data tests skip otherwise)."""
    return (root or REFS_ROOT) is not None


def _root(root: Path | None) -> Path:
    r = root or REFS_ROOT
    if r is None:
        raise RuntimeError("no reference root: set MOE_BLOCK_REFS (or pass root=)")
    return Path(r)


def case_dir(op: str, case: str, root: Path | None = None) -> Path:
    return _root(root) / op / case


def load_case(op: str, case: str, root: Path | None = None):
    """(meta, per_chip): per_chip[(r, c)][name] -> torch.Tensor."""
    d = case_dir(op, case, root)
    meta = json.loads((d / "meta.json").read_text())
    R, C = meta["mesh_shape"]
    per_chip = {(r, c): torch.load(d / f"chip_{r}_{c}.pt") for r in range(R) for c in range(C)}
    return meta, per_chip


def list_cases(op: str, root: Path | None = None) -> list[str]:
    """The saved cases of op; empty without a reference root."""
    if not refs_available(root):
        return []
    d = _root(root) / op
    return sorted(p.name for p in d.iterdir() if (p / "meta.json").exists()) if d.exists() else []


def from_mesh(tensor, mesh_shape: tuple[int, int]) -> dict[tuple[int, int], torch.Tensor]:
    """Per-chip host copies of a mesh tensor (row-major chip order)."""
    import ttnn

    R, C = mesh_shape
    shards = ttnn.get_device_tensors(tensor)
    assert len(shards) == R * C, (len(shards), mesh_shape)
    return {(i // C, i % C): ttnn.to_torch(s) for i, s in enumerate(shards)}


def to_mesh(per_chip: dict, name: str, mesh_device, *, dtype, layout, memory_config=None):
    """A mesh tensor whose chip (r, c) holds per_chip[(r, c)][name] (shards stacked in row-major chip order along a new
    leading dim, sharded over the mesh, the leading dim then dropped)."""
    import ttnn

    R, C = tuple(mesh_device.shape)
    stacked = torch.cat([per_chip[(i // C, i % C)][name].unsqueeze(0) for i in range(R * C)], dim=0)
    t = ttnn.from_torch(
        stacked,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(mesh_device, dim=0),
    )
    return ttnn.reshape(t, list(t.shape)[1:])


def assert_bit_exact(actual: dict, expected: dict, names: list[str] | None = None):
    """actual / expected: per_chip[(r, c)][name]. Every named tensor equal bit for bit on every chip."""
    bad = []
    for ch in sorted(expected):
        for name in names or sorted(expected[ch]):
            a, e = actual[ch][name], expected[ch][name]
            if a.shape != e.shape or a.dtype != e.dtype:
                bad.append(f"chip {ch} {name}: {a.dtype}{list(a.shape)} vs expected {e.dtype}{list(e.shape)}")
                continue
            ab, eb = _bits(a), _bits(e)
            diff = (ab != eb).nonzero()
            if diff.numel():
                first = tuple(diff[0].tolist())
                bad.append(
                    f"chip {ch} {name}: {diff.shape[0]} elements differ, first at {first}: {a[first]} vs {e[first]}"
                )
    assert not bad, "not bit-exact:\n" + "\n".join(bad)


def _bits(t: torch.Tensor) -> torch.Tensor:
    views = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16}
    return t.view(views[t.dtype]) if t.dtype in views else t
