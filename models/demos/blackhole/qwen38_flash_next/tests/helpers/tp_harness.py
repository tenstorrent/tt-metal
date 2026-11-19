# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device side of the tiered unit tests: one checkpoint layer's TTNN components on the shared ``mesh_device`` fixture,
the ``tt/`` torch references beside them, and the transfers between the two.  The component tests
(``test_*_component.py``) prove those references against the pinned Transformers modules on the CPU; the ``*_tp``
tests are the same layers on the mesh.

Env: MODEL_WEIGHTS_DIR (or QWEN38_CHECKPOINT, or HF_MODEL resolved through the local Hugging Face cache) names the
checkpoint.  QWEN38_CACHE_ROOT is needed only for the routed experts: the BF4 cache under ``<root>/caches/bf4-experts``.
GDN and QSA weights are converted into ``<root>/caches/<label>/components`` (label QWEN38_CACHE_LABEL, default
``c32768-tests``) when the root is set and into the test's temporary directory otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.config import Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.tools.live_decode_construction import prepare_live_decode
from models.demos.blackhole.qwen38_flash_next.tools.runtime_admission import git_identity, sha256_of
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_DEFAULT_QSA_CACHE_CAPACITY, Qwen38TTNNBuilder
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    FABRIC_CONFIG,
    MESH_SHAPE,
    Qwen38MeshContract,
    TensorPlacement,
    replicate_tensor_2d_mesh_mapper,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.model import Qwen38TTNNRoPE

HIDDEN_SIZE = 2560
RESIDUAL_BRANCHES = 4
# open_partition_b_mesh's parameters (tools/qwen38_chat_session.py) with the second command queue of the MTP form.
DEVICE_PARAMS = {
    "l1_small_size": 24576,
    "num_command_queues": 2,
    "fabric_config": FABRIC_CONFIG["config"],
    "reliability_mode": FABRIC_CONFIG["reliability_mode"],
    "fabric_tensix_config": FABRIC_CONFIG["fabric_tensix_config"],
    "trace_region_size": 0,
}


def pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Pearson correlation of the flattened tensors in FP32 (the tiered CI's PCC); both must be finite."""

    actual = actual.detach().to(torch.float32).flatten()
    expected = expected.detach().to(torch.float32).flatten()
    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert torch.isfinite(actual).all(), "device output has a non-finite value"
    assert torch.isfinite(expected).all(), "reference output has a non-finite value"
    return float(torch.corrcoef(torch.stack((actual, expected)))[0, 1])


class Qwen38TPHarness:
    """One admitted mesh: the checkpoint with its placement and mesh contract, the host-device transfers in the
    model's tensor placements, and the provenance-bound builder for a layer with routed experts."""

    def __init__(self, mesh_device, tmp_path: Path, checkpoint_root: Path) -> None:
        self.mesh_device = mesh_device
        self.checkpoint = Qwen38Checkpoint(checkpoint_root)
        self.config = self.checkpoint.config
        self.contract = Qwen38MeshContract(tuple(int(device_id) for device_id in mesh_device.get_device_ids()))
        self.placement = Qwen38Placement(self.config, mesh_shape=MESH_SHAPE, physical_ids=self.contract.physical_ids)
        self.tt_metal_sha = git_identity(Path(ttnn.__file__).resolve().parents[2])["head"]
        self.topology = ttnn.Topology.Linear
        # The model's per-position RoPE uploader (ttnn/model.py Qwen38TTNNRoPE): cos/sin for the position, and at
        # positions 3 mod 4 the block-start pair QSA needs to close the four-token index block.
        self.rope = Qwen38TTNNRoPE(mesh_device, self.contract, self.config)
        cache_root = os.environ.get("QWEN38_CACHE_ROOT")
        self.caches = None if not cache_root else Path(cache_root) / "caches"
        self.cache_label = os.environ.get("QWEN38_CACHE_LABEL") or f"c{RESIDENT_DEFAULT_QSA_CACHE_CAPACITY}-tests"
        self.component_cache_root = tmp_path if self.caches is None else self.caches / self.cache_label / "components"

    def upload_sharded(self, host: torch.Tensor):
        """``[1,B,R,2560]`` to the HIDDEN_SHARDED placement: each device its 640-column slice, BF16 DRAM tiles.
        Block activations are ``[1,1,R,2560]``; the layer's branch-major residual is ``[1,4,1,2560]``."""

        assert host.ndim == 4 and host.shape[0] == 1 and host.shape[3] == HIDDEN_SIZE, tuple(host.shape)
        tensor = ttnn.from_torch(
            host.to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
        )
        self.contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        return tensor

    def upload_replicated(self, host: torch.Tensor):
        tensor = ttnn.from_torch(
            host.to(torch.bfloat16).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_tensor_2d_mesh_mapper(self.mesh_device),
        )
        self.contract.validate_tensor(tensor, placement=TensorPlacement.REPLICATED)
        return tensor

    def download_sharded(self, tensor) -> torch.Tensor:
        """The four device shards concatenated on the hidden axis, FP32."""

        return torch.cat([ttnn.to_torch(shard).to(torch.float32) for shard in ttnn.get_device_tensors(tensor)], dim=3)

    def download_replicated(self, tensor) -> torch.Tensor:
        replicas = [ttnn.to_torch(local) for local in ttnn.get_device_tensors(tensor)]
        assert all(torch.equal(replica, replicas[0]) for replica in replicas[1:]), "replicas differ across the mesh"
        return replicas[0]

    def build_layer(self, layer_index: int):
        """The provenance-bound builder and one built backbone layer.

        Skips when QWEN38_CACHE_ROOT is unset or the layer's BF4 record is not in the cache.  The builder is
        ``construct_live_decode``'s, streamed instead of resident: the layer's experts are loaded for
        the call and released after it (the device component gate's configuration), so no preload of all 48 layers.
        """

        if self.caches is None:
            pytest.skip(
                "QWEN38_CACHE_ROOT unset: the routed experts need the BF4 expert cache under caches/bf4-experts"
            )
        extension = Path(ttnn._ttnn.__file__).resolve()
        prepared = prepare_live_decode(
            checkpoint_root=self.checkpoint.root,
            component_cache_root=self.component_cache_root,
            routed_bf4_scratch_root=self.caches / "bf4-experts",
            model_io_cache_root=self.caches / self.cache_label / "model-io",
            tt_metal_sha=self.tt_metal_sha,
            runtime_extension=extension,
            runtime_sha256=sha256_of(extension),
            physical_ids=self.contract.physical_ids,
        )
        builder = Qwen38TTNNBuilder(
            checkpoint=prepared.checkpoint,
            placement=prepared.placement,
            mesh_device=self.mesh_device,
            mesh_contract=prepared.mesh_contract,
            provenance=prepared.provenance,
            cache_roots=prepared.cache_roots,
            collective_topology=self.topology,
            expert_residency="streamed",
            qsa_cache_capacity=prepared.qsa_cache_capacity,
        )
        if builder.bf4_cache.verify_layer("backbone", layer_index) is None:
            pytest.skip(
                f"BF4 expert layer {layer_index} is not in the cache under {self.caches / 'bf4-experts'}; "
                "the chat server converts it on its first start"
            )
        return builder, builder.build_backbone_layer(layer_index)


# Transfers and checks of the fused-kernel device tests (test_*_device.py).


def upload(
    mesh, host: torch.Tensor, dtype, *, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, shard_dim=None
):
    """``host`` replicated over the mesh or split on ``shard_dim`` (device i its i-th slice); plain on a one-device
    submesh.  uint32 goes through int64, as ``qsa._upload_uint32`` converts it."""

    if mesh.get_num_devices() == 1:
        mapper = None
    elif shard_dim is None:
        mapper = replicate_tensor_2d_mesh_mapper(mesh)
    else:
        mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=MESH_SHAPE, dims=(None, shard_dim))
    if dtype == ttnn.uint32:
        host = host.to(torch.int64).to(torch.uint32)
    return ttnn.from_torch(
        host.contiguous(), dtype=dtype, layout=layout, device=mesh, memory_config=memory_config, mesh_mapper=mapper
    )


def download_shards(tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


def download_replica(tensor) -> torch.Tensor:
    """The host copy of a replicated mesh tensor; every device must hold the same values."""

    replicas = download_shards(tensor)
    for device, replica in enumerate(replicas[1:], start=1):
        assert_tensors_equal(replica, replicas[0], f"replica on device {device} versus device 0")
    return replicas[0]


def uint32_bits(tensor: torch.Tensor) -> torch.Tensor:
    """32-bit integers as their unsigned values in int64, so signed and unsigned views compare equal."""

    return tensor.to(torch.int64) & 0xFFFFFFFF


def assert_tensors_equal(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    """``torch.equal``; a mismatch reports the count of differing elements and the first one's flat index and values."""

    assert actual.shape == expected.shape, f"{label}: shape {tuple(actual.shape)}, expected {tuple(expected.shape)}"
    if torch.equal(actual, expected):
        return
    differs = (actual != expected).flatten()
    first = int(differs.int().argmax())
    raise AssertionError(
        f"{label}: {int(differs.sum())} of {differs.numel()} elements differ, first at flat index {first}: "
        f"{actual.flatten()[first].item()}, expected {expected.flatten()[first].item()}"
    )


def assert_shards_equal(actual, expected, label: str) -> None:
    """Each device shard of mesh tensor ``actual`` equal to ``expected``'s, a mesh tensor or its list of host shards."""

    expected_shards = expected if isinstance(expected, list) else download_shards(expected)
    for device, (got, want) in enumerate(zip(download_shards(actual), expected_shards, strict=True)):
        assert_tensors_equal(got, want, f"{label}, device {device}")


def assert_program_cache_stable(mesh, cached: int | None) -> int:
    """The mesh's program cache entry count, asserted equal to ``cached`` (the first seed's) when that is set."""

    count = mesh.num_program_cache_entries()
    assert cached is None or count == cached, f"program cache {count} entries, first seed {cached}"
    return count


def replay_trace(mesh, run, check, owned: list) -> None:
    """``run()`` captured once and replayed three times, ``check(outputs)`` after each; nothing compiles."""

    cached = mesh.num_program_cache_entries()
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        outputs = run()
    finally:
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
    owned.extend(outputs)
    try:
        for _ in range(3):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            check(outputs)
        count = mesh.num_program_cache_entries()
        assert count == cached, f"program cache {count} entries after the replays, {cached} before the capture"
    finally:
        ttnn.release_trace(mesh, trace)
