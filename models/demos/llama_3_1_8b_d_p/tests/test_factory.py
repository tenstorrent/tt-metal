# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers, spec constants and mesh parametrization for the device test suites.

The FIXTURES live in ``tests/conftest.py`` — pytest only auto-discovers fixtures from a conftest.

Borrowed from `minimax_m3/tests/test_factory.py`, narrowed to this bring-up's one mesh.

The target is **(8, 4) — SP=8 rows x TP=4 cols**, which is the whole 32-chip Blackhole Galaxy the
spec asks for. Bring-up goes straight to that mesh: there is no single-card step at any point, so
every module PCC test exercises sharding and collectives from the first one and the "worked on one
card, broke on the mesh" class of bug does not arise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
SPEC_PATH = Path(__file__).parent.parent / "llama_3_1_8b.spec.json"
TARGET_MESH = (8, 4)


def spec() -> dict:
    """The binding prefill spec. Every value in it is respected exactly, at every stage."""
    with open(SPEC_PATH) as f:
        return json.load(f)


SPEC = spec()
PCC_TARGET = SPEC["acceptance"]["pcc_target"]  # 0.99 — what every component aims for
PCC_LOWER_BOUND = SPEC["acceptance"]["pcc_lower_bound"]  # 0.85 — the assert in every test
TP = SPEC["parallelism"]["tp"]
SP = SPEC["parallelism"]["sp"]
MAX_SEQ_LEN = SPEC["shapes"]["max_seq_len"]
CHUNK_SIZE = SPEC["shapes"]["chunk_size"]

_DTYPES = {
    "bfloat16": ttnn.bfloat16,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
    "float32": ttnn.float32,
}


def spec_dtype(group: str, override: str | None = None) -> ttnn.DataType:
    """A dtype from the spec's `dataformats`. An empty or absent override inherits the default."""
    block = SPEC["dataformats"][group]
    name = override or block.get("default")
    return _DTYPES[name]


ACT_DTYPE = spec_dtype("activations")  # bfloat16
KV_DTYPE = spec_dtype("kv_cache")  # bfloat8_b
WEIGHT_DTYPE = spec_dtype("weights")  # bfloat8_b


def parametrize_target_mesh(mesh_shapes=None):
    """Paired `(mesh_device, device_params)` parametrization for this package's device tests.

    Defaults to the target (8, 4). Fabric is `FABRIC_1D` unless `LLAMA31_8B_FABRIC_TORUS=1`, because
    this Galaxy has no wrap-around links — see `conftest.py`, which picks the matching mesh-graph
    descriptor before the cluster initialises. Skips cleanly on a system with too few devices rather
    than failing, so the suite is still collectable on a smaller box.
    """
    shapes = list(mesh_shapes or [TARGET_MESH])
    num_devices = ttnn.get_num_devices()
    fits = [s for s in shapes if s[0] * s[1] <= num_devices]
    torus = os.getenv("LLAMA31_8B_FABRIC_TORUS") == "1"
    fabric = ttnn.FabricConfig.FABRIC_1D_RING if torus else ttnn.FabricConfig.FABRIC_1D

    if not fits:
        params = [
            pytest.param(
                shapes[0],
                {"fabric_config": None, "trace_region_size": 100000000},
                id=f"{shapes[0][0]}x{shapes[0][1]}",
                marks=pytest.mark.skip(
                    reason=f"needs {shapes[0][0] * shapes[0][1]} devices, this system has {num_devices}"
                ),
            )
        ]
    else:
        params = [
            pytest.param(
                shape,
                {"fabric_config": fabric, "trace_region_size": 100000000},
                id=f"{shape[0]}x{shape[1]}",
            )
            for shape in fits
        ]

    def decorator(func):
        return pytest.mark.parametrize("mesh_device, device_params", params, indirect=True)(func)

    return decorator


def sp_shard_rope(mesh_device, mesh_config, table) -> "ttnn.Tensor":
    """Put a `[1, 1, seq, head_dim]` rope table on the mesh, SHARDED on the SP axis.

    SP row r holds sequence positions `[r*seq_local, (r+1)*seq_local)`, so it needs exactly those
    rows of cos/sin. **A replicated full-length table is silently wrong**: the device cannot know its
    own SP offset, and `rotary_embedding_llama` has no prefill-mode check that the table length
    matches the input, so every row ends up rotating at positions `[0, seq_local)`.

    It fails quietly, too — the whole-model LOGITS check still reads PCC 1.0 with replicated tables,
    while the per-layer KV check drops to 0.59. That asymmetry is why the KV comparison, not the
    logits, is the artifact this bring-up is graded on.
    """
    dims = [None, None]
    dims[mesh_config.sp_axis] = 2
    return ttnn.from_torch(
        table,
        device=mesh_device,
        dtype=ACT_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
    )


def comp_pcc(golden: torch.Tensor, got: torch.Tensor) -> float:
    """Pearson correlation between a torch golden and a device read-back, in fp32."""
    a = golden.detach().float().flatten()
    b = got.detach().float().flatten()
    assert a.shape == b.shape, f"shape mismatch: golden {tuple(golden.shape)} vs device {tuple(got.shape)}"
    a, b = a - a.mean(), b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.allclose(a, b) else 0.0
    # Clamped: on near-identical tensors fp32 rounding can push the ratio a hair past 1.0, and a
    # reported "PCC 1.000356" in a results table is just noise dressed as precision.
    return min(1.0, max(-1.0, float((a @ b) / denom)))


def assert_pcc(name: str, golden: torch.Tensor, got: torch.Tensor, topology: str = "linear") -> float:
    """Assert at the spec's `pcc_lower_bound`, and say when a component is under `pcc_target`.

    The two numbers come from the spec and apply unchanged to every component test. Below the lower
    bound the component is NOT accepted and the stage does not pass. Between the two the test
    passes, but the measured value and the reason belong in the README's PCC table — so this logs
    loudly rather than passing quietly.
    """
    from loguru import logger

    pcc = comp_pcc(golden, got)
    logger.info(f"[{name}] PCC = {pcc:.6f} (target {PCC_TARGET}, bound {PCC_LOWER_BOUND}, {topology})")
    assert pcc >= PCC_LOWER_BOUND, (
        f"{name}: PCC {pcc:.6f} is below the spec's pcc_lower_bound {PCC_LOWER_BOUND} - "
        "the component is not accepted and the stage does not pass"
    )
    if pcc < PCC_TARGET:
        logger.warning(
            f"{name}: PCC {pcc:.6f} is below pcc_target {PCC_TARGET}. Accept only after every "
            f"correctness knob has been tried; record the value and the reason in README.md."
        )
    return pcc


def to_torch_replicated(tt_tensor, mesh_device) -> torch.Tensor:
    """Read back a tensor that is replicated across the mesh, taking device 0's copy."""
    return ttnn.to_torch(ttnn.get_device_tensors(tt_tensor)[0])


def to_torch_sharded(tt_tensor, mesh_device, dims) -> torch.Tensor:
    """Read back a 2D-sharded tensor by concatenating along `dims = (rows_dim, cols_dim)`."""
    rows, cols = tuple(mesh_device.shape)
    return ttnn.to_torch(
        tt_tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(dims), mesh_shape=(rows, cols)),
    )
