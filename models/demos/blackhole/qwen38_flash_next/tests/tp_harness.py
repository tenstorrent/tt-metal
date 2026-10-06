# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device side of the tiered unit tests: one checkpoint layer's TTNN components on the shared ``mesh_device`` fixture,
the ``tt/`` torch references beside them, and the transfers between the two.  The component tests
(``test_*_component.py``) prove those references against the pinned Transformers modules on the CPU; the ``*_tp``
tests are the same layers on the mesh.

Env: MODEL_WEIGHTS_DIR (or QWEN38_CHECKPOINT, or HF_MODEL resolved through the local Hugging Face cache) names the
checkpoint.  QWEN38_CACHE_ROOT is needed only for the routed experts: the BF4 cache under ``<root>/caches/bf4-experts``,
or the CPU-staged corpus QWEN38_BF4_CORPUS with QWEN38_BF4_CORPUS_VERIFICATION.  GDN and QSA weights are converted
into ``<root>/caches/<label>/components`` when the root is set and into the test's temporary directory otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
from models.demos.blackhole.qwen38_flash_next.config import Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.tools.live_decode_diagnostic import (
    Qwen38DiagnosticBF4Cache,
    prepare_live_decode_diagnostic,
)
from models.demos.blackhole.qwen38_flash_next.tools.runtime_admission import git_identity, sha256_of
from models.demos.blackhole.qwen38_flash_next.ttnn.bf4 import Qwen38BF4Streamer
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
# The demo's mesh parameters (demo/text_demo.py DEVICE_PARAMS: the vLLM launch's tt config).
DEVICE_PARAMS = {
    "l1_small_size": 24576,
    "num_command_queues": 2,
    "fabric_config": FABRIC_CONFIG["config"],
    "reliability_mode": FABRIC_CONFIG["reliability_mode"],
    "fabric_tensix_config": FABRIC_CONFIG["fabric_tensix_config"],
    "trace_region_size": 0,
}


def checkpoint_root() -> Path:
    for name in ("MODEL_WEIGHTS_DIR", "QWEN38_CHECKPOINT"):
        if os.environ.get(name):
            return Path(os.environ[name])
    if os.environ.get("HF_MODEL"):  # the snapshot in the local Hugging Face cache, offline
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(os.environ["HF_MODEL"], local_files_only=True))
    pytest.fail("set MODEL_WEIGHTS_DIR (or QWEN38_CHECKPOINT) to the checkpoint directory or HF_MODEL to its id")


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

    def __init__(self, mesh_device, tmp_path: Path) -> None:
        self.mesh_device = mesh_device
        self.checkpoint = Qwen38Checkpoint(checkpoint_root())
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
        # The adapter's default label: the demo and these tests share one component namespace.
        self.cache_label = os.environ.get("QWEN38_CACHE_LABEL") or f"c{RESIDENT_DEFAULT_QSA_CACHE_CAPACITY}-vllm"
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
        ``construct_live_decode_diagnostic``'s, streamed instead of resident: the layer's experts are loaded for
        the call and released after it (the device component gate's configuration), so no preload of all 48 layers.
        """

        if self.caches is None:
            pytest.skip(
                "QWEN38_CACHE_ROOT unset: the routed experts need the BF4 expert cache under caches/bf4-experts"
            )
        corpus_root = os.environ.get("QWEN38_BF4_CORPUS")
        corpus_verification = os.environ.get("QWEN38_BF4_CORPUS_VERIFICATION")
        extension = Path(ttnn._ttnn.__file__).resolve()
        prepared = prepare_live_decode_diagnostic(
            checkpoint_root=self.checkpoint.root,
            component_cache_root=self.component_cache_root,
            routed_bf4_scratch_root=self.caches / "bf4-experts",
            model_io_cache_root=self.caches / self.cache_label / "model-io",
            tt_metal_sha=self.tt_metal_sha,
            runtime_extension=extension,
            runtime_sha256=sha256_of(extension),
            physical_ids=self.contract.physical_ids,
            bf4_corpus_root=None if corpus_root is None else Path(corpus_root),
            bf4_corpus_verification=None if corpus_verification is None else Path(corpus_verification),
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
        if prepared.corpus is not None:
            builder.bf4_cache = Qwen38DiagnosticBF4Cache(
                production_cache=builder.bf4_cache,
                corpus=prepared.corpus,
                artifact_signatures=prepared.artifact_signatures,
                compatibility=prepared.bf4_consumer_compatibility,
                mesh_device=self.mesh_device,
            )
            builder.expert_streamer = Qwen38BF4Streamer(builder.bf4_cache, self.mesh_device)
        if builder.bf4_cache.verify_layer("backbone", layer_index) is None:
            pytest.skip(
                f"BF4 expert layer {layer_index} is not in the cache under {self.caches / 'bf4-experts'}; "
                "the chat server or the demo converts it on its first start"
            )
        return builder, builder.build_backbone_layer(layer_index)
