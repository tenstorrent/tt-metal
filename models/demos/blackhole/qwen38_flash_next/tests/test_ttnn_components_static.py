# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device placement gates for the Qwen3.8 TTNN component path."""

import ast
import dataclasses
import inspect
import re
import textwrap
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.checkpoint import (
    CHECKPOINT_FILE_MANIFEST_SHA256,
    CHECKPOINT_TENSOR_MANIFEST_SHA256,
    INDEX_SHA256,
    PINNED_CHECKPOINT_REVISION,
)
from models.demos.blackhole.qwen38_flash_next.config import CONFIG_SHA256
from models.demos.blackhole.qwen38_flash_next.ttnn import final_mixer as final_mixer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import gr as gr_module
from models.demos.blackhole.qwen38_flash_next.ttnn import layer as layer_module
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn import moe as moe_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp as mtp_module
from models.demos.blackhole.qwen38_flash_next.ttnn import ple as ple_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import (
    BACKBONE_PLAN,
    TARGET_OBJECT_GRAPH,
    Qwen38BuildProvenance,
    validate_builder_constructor_contract,
    validate_builder_static_contract,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import Qwen38IOCacheIdentity
from models.demos.blackhole.qwen38_flash_next.ttnn.gdn import Qwen38TTNNGDN, validate_gdn_static_contract
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import (
    Qwen38TTNNDecoderLayer,
    Qwen38TTNNDecoderLayerGenericState,
    Qwen38TTNNDecoderLayerSnapshot,
    Qwen38TTNNLayerNamespace,
    Qwen38TTNNLayerType,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import Qwen38TTNNPLEResult as LayerPLEResult
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import validate_layer_static_contract
from models.demos.blackhole.qwen38_flash_next.ttnn.model import (
    EXPECTED_CONFIG_SHA256,
    Qwen38TTNNGenericDecodeOutput,
    Qwen38TTNNModelPoisonedError,
    Qwen38TTNNPreparedDecodeInputs,
    Qwen38TTNNRoPEInputs,
    Qwen38TTNNRoPETable,
    Qwen38TTNNTextModel,
    Qwen38TTNNTextModelGenericState,
    _host_rope,
    _inverse_frequency,
    validate_model_static_contract,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.ple import (
    EMBEDDING_WIDTH,
    HIDDEN_SIZE,
    LOCAL_HIDDEN_SIZE,
    RESIDUAL_BRANCHES,
    RESIDUAL_WIDTH,
    Qwen38TTNNPLE,
    Qwen38TTNNPLEResult,
    _prepare_key_weight,
    validate_ple_static_contract,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import (
    MAX_SELECTED_TOKENS,
    SPARSE_INDEX_CAPACITY,
    Qwen38TTNNQSA,
    qsa_selection_geometry,
)


def test_device_component_constants_are_exact() -> None:
    validate_gdn_static_contract()
    validate_ple_static_contract()
    validate_layer_static_contract()
    validate_model_static_contract()
    validate_builder_static_contract()
    validate_builder_constructor_contract()


def test_gdn_local_reshape_retags_the_logical_head_shard(monkeypatch) -> None:
    class _Topology:
        def distribution_shape(self):
            return (1, 4)

        def mesh_coords(self):
            return ((0, 0), (0, 1), (0, 2), (0, 3))

    class _Reference:
        def tensor_topology(self):
            return _Topology()

    class _Tensor:
        updated = None

        def update_tensor_topology(self, topology):
            self.updated = topology

    monkeypatch.setattr(gdn_module.ttnn, "PlacementReplicate", lambda: "replicate")
    monkeypatch.setattr(gdn_module.ttnn, "PlacementShard", lambda dim: ("shard", dim))
    monkeypatch.setattr(
        gdn_module.ttnn,
        "TensorTopology",
        lambda distribution, placements, coords: (tuple(distribution), tuple(placements), tuple(coords)),
    )

    tensor = _Tensor()
    gdn_module._retag_head_shard_after_reshape(tensor, reference=_Reference(), shard_dim=2)
    assert tensor.updated == (
        (1, 4),
        ("replicate", ("shard", 2)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
    )


def test_gdn_state_uses_seven_distinct_device_native_zero_allocations(monkeypatch, expect_error) -> None:
    class PlacementReplicate:
        pass

    class PlacementShard:
        def __init__(self, dim):
            self.dim = dim

    class FakeTopology:
        def __init__(self, distribution, placements, coordinates):
            self.distribution = tuple(distribution)
            self._placements = tuple(placements)
            self.coordinates = tuple(coordinates)

        def distribution_shape(self):
            return self.distribution

        def placements(self):
            return self._placements

        def mesh_coords(self):
            return self.coordinates

    class FakeLocal:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype
            self.layout = tile_layout

        @staticmethod
        def memory_config():
            return dram_memory

    class FakeTensor:
        def __init__(self, identity, shape, dtype, *, invalid_topology=False):
            self.identity = identity
            self.shape = shape
            self.dtype = dtype
            placements = (
                (PlacementReplicate(),)
                if invalid_topology
                else (
                    PlacementReplicate(),
                    PlacementReplicate(),
                )
            )
            self.topology = FakeTopology((1, 4), placements, ((0, 0), (0, 1), (0, 2), (0, 3)))
            self.locals = tuple(FakeLocal(shape, dtype) for _ in range(4))

        def tensor_id(self):
            return self.identity

        def tensor_topology(self):
            return self.topology

        def update_tensor_topology(self, topology):
            self.topology = topology

    class FakeContract:
        def __init__(self):
            self.validated = []

        @staticmethod
        def validate_mesh(mesh):
            assert mesh is fake_mesh

        def validate_tensor(self, tensor, *, placement, shard_dim):
            assert placement is gdn_module.TensorPlacement.HEAD_SHARDED
            assert shard_dim in (1, 3)
            placements = tensor.tensor_topology().placements()
            assert tuple(type(value).__name__ for value in placements) == (
                "PlacementReplicate",
                "PlacementShard",
            )
            assert placements[1].dim == shard_dim
            self.validated.append(tensor.tensor_id())

    fake_mesh = object()
    float32 = object()
    bfloat16 = object()
    tile_layout = object()
    dram_memory = object()
    created = []
    deallocated = []
    invalid_at = None

    def moreh_full(shape, value, mesh, *, dtype, layout, memory_config):
        assert value == 0.0
        assert mesh is fake_mesh
        assert layout is tile_layout
        assert memory_config is dram_memory
        identity = len(created) + 1
        tensor = FakeTensor(identity, tuple(shape), dtype, invalid_topology=identity == invalid_at)
        created.append(tensor)
        return tensor

    monkeypatch.setattr(gdn_module.ttnn, "float32", float32)
    monkeypatch.setattr(gdn_module.ttnn, "bfloat16", bfloat16)
    monkeypatch.setattr(gdn_module.ttnn, "TILE_LAYOUT", tile_layout)
    monkeypatch.setattr(gdn_module.ttnn, "DRAM_MEMORY_CONFIG", dram_memory)
    monkeypatch.setattr(gdn_module.ttnn, "PlacementReplicate", PlacementReplicate)
    monkeypatch.setattr(gdn_module.ttnn, "PlacementShard", PlacementShard)
    monkeypatch.setattr(gdn_module.ttnn, "TensorTopology", FakeTopology)
    monkeypatch.setattr(gdn_module.ttnn, "moreh_full", moreh_full)
    monkeypatch.setattr(gdn_module.ttnn, "get_device_tensors", lambda tensor: tensor.locals)
    monkeypatch.setattr(gdn_module.ttnn, "deallocate", lambda tensor: deallocated.append(tensor.tensor_id()))
    monkeypatch.setattr(
        gdn_module.ttnn,
        "from_torch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("raw H2D is forbidden")),
    )
    monkeypatch.setattr(
        gdn_module.torch,
        "zeros",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("host zero is forbidden")),
    )

    contract = FakeContract()
    state = gdn_module.Qwen38TTNNGDNState.allocate(fake_mesh, contract, layer_index=0)
    owned = (state.recurrent, *state.conv, state.zero_recurrent, state.zero_conv)
    assert len(owned) == len({tensor.tensor_id() for tensor in owned}) == 7
    assert [tensor.shape for tensor in owned] == [
        (1, 12, 128, 128),
        (1, 1, 1, 2560),
        (1, 1, 1, 2560),
        (1, 1, 1, 2560),
        (1, 1, 1, 2560),
        (1, 12, 128, 128),
        (1, 1, 1, 2560),
    ]
    assert [tensor.dtype for tensor in owned] == [float32] + [bfloat16] * 4 + [float32, bfloat16]
    assert len(contract.validated) == 14
    state.deallocate()
    assert deallocated == list(range(1, 8))

    created.clear()
    deallocated.clear()
    invalid_at = 4
    with expect_error(RuntimeError, "not initially replicated"):
        gdn_module.Qwen38TTNNGDNState.allocate(fake_mesh, FakeContract(), layer_index=0)
    assert deallocated == [4, 1, 2, 3]


def test_qsa_flat_zero_topology_is_canonicalized_to_explicit_2d(monkeypatch) -> None:
    class PlacementReplicate:
        pass

    class _Topology:
        def __init__(self):
            self.coordinates = ((0, 0), (0, 1), (0, 2), (0, 3))

        def distribution_shape(self):
            return (4,)

        def mesh_coords(self):
            return self.coordinates

        def placements(self):
            return (PlacementReplicate(),)

    class _Tensor:
        updated = None

        def __init__(self, topology):
            self.topology = topology

        def tensor_topology(self):
            return self.topology

        def update_tensor_topology(self, topology):
            self.updated = topology

    class _Contract:
        mesh_shape = (1, 4)

        def __init__(self):
            self.validated = []

        def validate_tensor(self, tensor, *, placement):
            self.validated.append((tensor, placement))

    monkeypatch.setattr(qsa_module.ttnn, "PlacementReplicate", PlacementReplicate)
    monkeypatch.setattr(qsa_module.ttnn, "MeshShape", lambda *shape: ("MeshShape", *shape))
    monkeypatch.setattr(
        qsa_module.ttnn,
        "TensorTopology",
        lambda distribution, placements, coords: (distribution, tuple(placements), coords),
    )
    topology = _Topology()
    tensor = _Tensor(topology)
    contract = _Contract()

    qsa_module._canonicalize_flat_replicated_topology(tensor, contract)

    assert tensor.updated[0] == ("MeshShape", 1, 4)
    assert tuple(type(value).__name__ for value in tensor.updated[1]) == (
        "PlacementReplicate",
        "PlacementReplicate",
    )
    assert tensor.updated[2] == ((0, 0), (0, 1), (0, 2), (0, 3))
    assert tensor.updated[2] is topology.coordinates
    assert contract.validated == [(tensor, qsa_module.TensorPlacement.REPLICATED)]


def test_qsa_flat_zero_topology_rejects_unexpected_distribution_and_coordinates(expect_error) -> None:
    class PlacementReplicate:
        pass

    class _Topology:
        def __init__(self, distribution, coordinates):
            self.distribution = distribution
            self.coordinates = coordinates

        def distribution_shape(self):
            return self.distribution

        def mesh_coords(self):
            return self.coordinates

        def placements(self):
            return (PlacementReplicate(),)

    class _Tensor:
        def __init__(self, topology):
            self.topology = topology

        def tensor_topology(self):
            return self.topology

        def update_tensor_topology(self, _topology):
            raise AssertionError("unexpected flat topology must fail before retagging")

    contract = type("Contract", (), {"mesh_shape": (1, 4)})()
    cases = (
        _Topology((2, 2), ((0, 0), (0, 1), (0, 2), (0, 3))),
        _Topology((4,), ((0, 0), (0, 1), (0, 3), (0, 2))),
    )
    for topology in cases:
        with expect_error(RuntimeError, "not the exact flat replicated TP4 form"):
            qsa_module._canonicalize_flat_replicated_topology(_Tensor(topology), contract)


def test_gated_residual_uses_the_runtime_stats_dtype_contract() -> None:
    from models.demos.blackhole.qwen38_flash_next.ttnn.builder import Qwen38TTNNBuilder
    from models.demos.blackhole.qwen38_flash_next.ttnn.gr import Qwen38TTNNGatedResidual

    source = inspect.getsource(Qwen38TTNNGatedResidual._normalize)
    assert "dtype=ttnn.bfloat16" in source
    assert "dtype=ttnn.float32" not in source
    assert "stats = ttnn.pad(" not in source
    assert "expected [1,4,1,128]" in source
    post_call = source.split("unit = ttnn.rms_norm_post_all_gather(", 1)[1].split(")", 1)[0]
    assert "weight=" not in post_call
    assert "normalized_ws = ttnn.multiply(" in source
    assert "self.norm_scale_flat" in source
    assert "memory_config=self.down_inject_act_memory_config" in source
    assert "gathered_stats = ttnn.all_gather(" in source
    assert "cluster_axis=TP_AXIS" in source
    assert "_deallocate(stats)" in source
    assert source.index("gathered_stats = ttnn.all_gather(") < source.index("unit = ttnn.rms_norm_post_all_gather(")
    assert source.index("_deallocate(stats)") < source.index("unit = ttnn.rms_norm_post_all_gather(")
    assert source.index("unit = ttnn.rms_norm_post_all_gather(") < source.index("normalized_ws = ttnn.multiply(")
    builder_source = inspect.getsource(Qwen38TTNNBuilder._build_gr)
    assert "tt_ccl=self.tt_ccl" in builder_source


def test_final_mixer_avoids_one_page_standard_all_gather_stats() -> None:
    from models.demos.blackhole.qwen38_flash_next.ttnn.final_mixer import Qwen38TTNNFinalMixer

    source = inspect.getsource(Qwen38TTNNFinalMixer._normalize)
    assert "dtype=ttnn.bfloat16" in source
    assert "dtype=ttnn.float32" not in source


def test_ple_distributed_group_norm_uses_bf16_stats_contract() -> None:
    source = inspect.getsource(Qwen38TTNNPLE._distributed_group_norm)
    assert "dtype=ttnn.float32" not in source
    assert source.count("dtype=ttnn.bfloat16") == 2
    pre_call = source.split("stats = ttnn.rms_norm_pre_all_gather(", 1)[1].split(")", 1)[0]
    post_call = source.split("normalized = ttnn.rms_norm_post_all_gather(", 1)[1].split(")", 1)[0]
    assert "dtype=ttnn.bfloat16" in pre_call
    assert "dtype=ttnn.bfloat16" in post_call


def test_ple_gate_gathers_global_norms_before_local_fp32_dot() -> None:
    source = inspect.getsource(Qwen38TTNNPLE._gate)
    assert source.count("ttnn.all_gather(") == 2
    assert "ttnn.all_reduce(" not in source
    assert "mark_local_partial(" not in source
    for name in ("key_global", "query_global"):
        gather = source.split(f"{name} = ttnn.all_gather(", 1)[1].split("\n        )", 1)[0]
        assert "dim=3" in gather
        assert "cluster_axis=TP_AXIS" in gather
        assert "memory_config=ttnn.DRAM_MEMORY_CONFIG" in gather
        assert "topology=" not in gather
    assert "global_shape = (1, 1, RESIDUAL_BRANCHES, HIDDEN_SIZE)" in source
    assert source.count("placement=TensorPlacement.REPLICATED") == 2
    assert source.index("query_global = ttnn.all_gather(") < source.index("_deallocate(key_norm, query_norm)")
    assert source.index("_deallocate(key_norm, query_norm)") < source.index("key_fp32 = ttnn.typecast(key_global")
    assert source.index("query_fp32 = ttnn.typecast(query_global") < source.index(
        "_deallocate(key_global, query_global)"
    )
    assert source.index("products = ttnn.multiply(key_fp32, query_fp32") < source.index("unscaled_gate = ttnn.sum(")
    assert source.index("unscaled_gate = ttnn.sum(") < source.index(
        "gate = ttnn.multiply(unscaled_gate, HIDDEN_SIZE**-0.5"
    )
    assert source.index("gate = ttnn.multiply(unscaled_gate, HIDDEN_SIZE**-0.5") < source.index(
        "_deallocate(unscaled_gate)"
    )

    base = torch.arange(RESIDUAL_BRANCHES * LOCAL_HIDDEN_SIZE, dtype=torch.float32).reshape(
        1, 1, RESIDUAL_BRANCHES, LOCAL_HIDDEN_SIZE
    )
    key_shards = tuple(((base + rank).remainder(7) - 3).to(torch.bfloat16) for rank in range(4))
    query_shards = tuple(((base + 2 * rank).remainder(5) - 2).to(torch.bfloat16) for rank in range(4))
    gathered_dot = (torch.cat(key_shards, dim=3).float() * torch.cat(query_shards, dim=3).float()).sum(
        dim=3, keepdim=True
    )
    shard_dot = sum(
        (key.float() * query.float()).sum(dim=3, keepdim=True) for key, query in zip(key_shards, query_shards)
    )
    torch.testing.assert_close(
        gathered_dot * HIDDEN_SIZE**-0.5,
        shard_dot * HIDDEN_SIZE**-0.5,
        rtol=0.0,
        atol=0.0,
    )


def test_builder_owns_the_exact_ordered_48_layer_graph() -> None:
    assert len(BACKBONE_PLAN) == len(TARGET_OBJECT_GRAPH.layers) == 48
    assert [slot.layer_index for slot in BACKBONE_PLAN] == list(range(48))
    assert [slot.layer_index for slot in BACKBONE_PLAN if slot.attention == "qsa"] == list(range(3, 48, 4))
    assert [slot.layer_index for slot in BACKBONE_PLAN if slot.has_ple] == [1]
    assert {node.bf4_streamer_owner for node in TARGET_OBJECT_GRAPH.layers} == {"target-shared-single-slot"}


def test_build_and_io_cache_provenance_bind_both_checkpoint_manifests() -> None:
    provenance = Qwen38BuildProvenance(
        checkpoint_revision=PINNED_CHECKPOINT_REVISION,
        checkpoint_index_sha256=INDEX_SHA256,
        checkpoint_config_sha256=CONFIG_SHA256,
        checkpoint_file_manifest_sha256=CHECKPOINT_FILE_MANIFEST_SHA256,
        checkpoint_hash_manifest_sha256=CHECKPOINT_TENSOR_MANIFEST_SHA256,
        tt_metal_sha="1" * 40,
        ttnn_runtime_sha256="2" * 64,
    )
    io_identity = Qwen38IOCacheIdentity(
        checkpoint_revision=PINNED_CHECKPOINT_REVISION,
        checkpoint_config_sha256=CONFIG_SHA256,
        checkpoint_file_manifest_sha256=CHECKPOINT_FILE_MANIFEST_SHA256,
        checkpoint_hash_manifest_sha256=CHECKPOINT_TENSOR_MANIFEST_SHA256,
        tt_metal_revision="1" * 40,
        ttnn_runtime_sha256="2" * 64,
        mesh_shape=(1, 4),
        physical_ids=(0, 1, 2, 3),
    )
    assert provenance.checkpoint_file_manifest_sha256 == io_identity.checkpoint_file_manifest_sha256
    assert provenance.checkpoint_hash_manifest_sha256 == io_identity.checkpoint_hash_manifest_sha256
    for field in ("checkpoint_file_manifest_sha256", "checkpoint_hash_manifest_sha256"):
        try:
            dataclasses.replace(provenance, **{field: "0" * 64})
        except ValueError:
            pass
        else:
            raise AssertionError(f"builder provenance accepted a changed {field}")
        try:
            dataclasses.replace(io_identity, **{field: "0" * 64})
        except ValueError:
            pass
        else:
            raise AssertionError(f"I/O cache identity accepted a changed {field}")


def test_qsa_selection_geometry_covers_complete_blocks_and_causal_tail() -> None:
    assert qsa_selection_geometry(1).tail_count == 1
    assert qsa_selection_geometry(3).tail_count == 3
    closed = qsa_selection_geometry(4)
    assert (closed.complete_blocks, closed.complete_token_count, closed.tail_count) == (1, 4, 0)
    long = qsa_selection_geometry(262144)
    assert (long.selected_blocks, long.complete_token_count, long.tail_count) == (512, 2048, 0)
    assert (MAX_SELECTED_TOKENS, SPARSE_INDEX_CAPACITY) == (2055, 2080)


def test_decoder_layer_uses_canonical_fixed_address_ple_contract() -> None:
    assert LayerPLEResult is Qwen38TTNNPLEResult
    fields = tuple(field.name for field in dataclasses.fields(Qwen38TTNNDecoderLayerSnapshot))
    assert "source_ple" in fields
    token_annotation = inspect.signature(Qwen38TTNNDecoderLayer.forward_decode).parameters["token_id"].annotation
    assert "torch.Tensor" in str(token_annotation)


def test_ple_key_packing_keeps_every_branch_hidden_shard_on_its_owner() -> None:
    source = torch.zeros((RESIDUAL_WIDTH, EMBEDDING_WIDTH), dtype=torch.bfloat16)
    input_column = 17
    sentinels: dict[tuple[int, int, int], float] = {}
    for device_index in range(4):
        for branch in range(RESIDUAL_BRANCHES):
            local_hidden = 11 + branch
            source_row = branch * HIDDEN_SIZE + device_index * LOCAL_HIDDEN_SIZE + local_hidden
            value = float(1 + device_index * RESIDUAL_BRANCHES + branch)
            source[source_row, input_column] = value
            sentinels[(device_index, branch, local_hidden)] = value

    packed = _prepare_key_weight(source).reshape(EMBEDDING_WIDTH, RESIDUAL_WIDTH)
    for (device_index, branch, local_hidden), expected in sentinels.items():
        packed_column = (
            device_index * (RESIDUAL_BRANCHES * LOCAL_HIDDEN_SIZE) + branch * LOCAL_HIDDEN_SIZE + local_hidden
        )
        assert float(packed[input_column, packed_column]) == expected


def test_qsa_output_projection_sums_bf16_partials_in_fp32_reduce_scatter() -> None:
    """The QSA o_proj boundary sums the four BF16 partials in FP32 and rounds once.

    ttnn.reduce_scatter accumulates in its operand dtype at every hop, so a
    BF16 operand is rounded three times across the four-coordinate line.  The
    decode sequence must widen the BF16 partial to FP32 after the DRAM-sharded
    o_proj linear, reduce in FP32, and narrow the H/4 shard to BF16 exactly
    once, with placement metadata repaired after every placement-changing or
    local-only operation.  The pre-collective BF16 partial is unchanged.
    """

    decode = inspect.getsource(qsa_module.Qwen38TTNNQSA.forward_decode)
    start = decode.index("local_partial_ws = ttnn.linear(")
    stop = decode.index("_deallocate(full_hidden)", start)
    source = decode[start:stop]
    assert re.findall(r"\bttnn\.(\w+)\(", source) == [
        "linear",
        "to_memory_config",
        "typecast",
        "reduce_scatter",
        "typecast",
    ]
    assert re.findall(r"\bttnn\.(float32|bfloat16)\b", source) == ["float32", "float32", "bfloat16", "bfloat16"]
    assert source.count("mark_local_partial(") == 2
    assert source.count("mark_collective_shard(") == 1
    assert source.count("_retag_tensor(") == 1
    assert source.count("validate_tensor(") == 1
    assert source.count("_deallocate(") == 5
    widen = source.split("local_partial_fp32 = ttnn.typecast(", 1)[1].split(")", 1)[0]
    assert "local_partial, ttnn.float32" in widen
    assert "memory_config=ttnn.DRAM_MEMORY_CONFIG" in widen
    reduce = source.split("output_fp32 = ttnn.reduce_scatter(", 1)[1].split("\n        )", 1)[0]
    assert "local_partial_fp32," in reduce
    assert "dim=3" in reduce
    assert "cluster_axis=TP_AXIS" in reduce
    assert "memory_config=ttnn.DRAM_MEMORY_CONFIG" in reduce
    assert "topology=self.collective_topology" in reduce
    # The runtime derives HiFi4 + FP32 DEST accumulation from the FLOAT32 operand
    # (ccl resolve_fp32_acc_compute_kernel_config); an explicit compute config
    # would only move the collective off the direct reduce_scatter path.
    assert "compute_kernel_config" not in reduce
    narrow = source.split("output = ttnn.typecast(", 1)[1].split(")", 1)[0]
    assert "output_fp32, ttnn.bfloat16" in narrow
    assert "memory_config=ttnn.DRAM_MEMORY_CONFIG" in narrow
    assert source.count("dtype=ttnn.float32") == 0
    order = (
        "local_partial_ws = ttnn.linear(",
        "_deallocate(attention_ws)",
        "local_partial = ttnn.to_memory_config(local_partial_ws, ttnn.DRAM_MEMORY_CONFIG)",
        "_deallocate(local_partial_ws)",
        "mark_local_partial(\n            local_partial,",
        "local_partial_fp32 = ttnn.typecast(",
        "_deallocate(local_partial)",
        "mark_local_partial(\n            local_partial_fp32,",
        "output_fp32 = ttnn.reduce_scatter(",
        "_deallocate(local_partial_fp32)",
        "mark_collective_shard(\n            output_fp32,",
        "output_fp32.dtype != ttnn.float32",
        'raise RuntimeError(f"QSA output reduce_scatter must sum FP32 partials, got {tensor_metadata(output_fp32)}")',
        "output = ttnn.typecast(",
        "_retag_tensor(output, reference=output_fp32, shard_dim=3)",
        "_deallocate(output_fp32)",
        '_require_shape(output, (1, 1, 1, HIDDEN_SIZE // TP_SIZE), "QSA output projection")',
        "output.dtype != ttnn.bfloat16",
        'raise RuntimeError(f"QSA output projection must return BF16, got {tensor_metadata(output)}")',
        "validate_tensor(output, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)",
    )
    positions = [source.index(marker) for marker in order]
    assert positions == sorted(positions)
    assert len(set(positions)) == len(positions)
    # The MTP layer builds the same Qwen38TTNNQSA class, so it shares this
    # boundary through forward_decode rather than a second collective call.
    assert decode.count("ttnn.reduce_scatter(") == 1


@pytest.mark.parametrize(
    "from_checkpoint",
    [
        final_mixer_module.Qwen38TTNNFinalMixerWeights.from_checkpoint,
        gdn_module.Qwen38TTNNGDNWeights.from_checkpoint,
        gr_module.Qwen38TTNNGatedResidualWeights.from_checkpoint,
        moe_module.Qwen38TTNNMoEWeights.from_checkpoint,
        mtp_module.Qwen38TTNNMTPInputWeights.from_checkpoint,
        ple_module.Qwen38TTNNPLEWeights.from_checkpoint,
        qsa_module.Qwen38TTNNQSAWeights.from_checkpoint,
    ],
    ids=lambda function: function.__qualname__.split(".")[0],
)
def test_weight_loader_closures_are_called_with_their_full_signature(from_checkpoint) -> None:
    """Every call to a closure defined inside a weight loader supplies all of its required arguments.

    The loaders run only on hardware, so a closure whose signature grew a parameter
    fails there, at model build; this check gives the same TypeError without a device.
    """

    tree = ast.parse(textwrap.dedent(inspect.getsource(from_checkpoint)))
    closures = {
        node.name: node.args
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name != from_checkpoint.__name__
    }
    checked = 0
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in closures):
            continue
        if any(isinstance(arg, ast.Starred) for arg in call.args) or any(kw.arg is None for kw in call.keywords):
            continue
        signature = closures[call.func.id]
        positional = signature.posonlyargs + signature.args
        supplied_keywords = {keyword.arg for keyword in call.keywords}
        required_positional = positional[: len(positional) - len(signature.defaults)]
        missing = [arg.arg for arg in required_positional[len(call.args) :] if arg.arg not in supplied_keywords]
        missing += [
            arg.arg
            for arg, default in zip(signature.kwonlyargs, signature.kw_defaults)
            if default is None and arg.arg not in supplied_keywords
        ]
        where = f"{from_checkpoint.__qualname__} line {call.lineno}: {call.func.id}()"
        assert not missing, f"{where} missing {missing}"
        assert signature.vararg is not None or len(call.args) <= len(positional), f"{where} has too many arguments"
        checked += 1
    assert checked > 0


def _calls_named(tree: ast.AST, attribute: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == attribute
    ]


def test_moe_dense_linears_share_one_width_sharded_hidden() -> None:
    """The router and every shared-expert linear read the five-core shard the hidden all-gather writes."""

    init_tree = ast.parse(textwrap.dedent(inspect.getsource(moe_module.Qwen38TTNNMoE.__init__)))
    grids = [
        next(keyword.value.value for keyword in call.keywords if keyword.arg == "num_cores")
        for call in ast.walk(init_tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "dram_sharded_matmul_configs"
    ]
    assert grids == [5, 5, 5, 5, 5]  # router, shared gate/up, shared down, shared scalar, fused gate|up|scalar

    gather = inspect.getsource(moe_module.Qwen38TTNNMoE._all_gather_hidden)
    gathers = _calls_named(ast.parse(textwrap.dedent(gather)), "all_gather")
    assert len(gathers) == 1
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in gathers[0].keywords}
    assert keywords["memory_config"] == "self.hidden_gather_memory_config"
    init = inspect.getsource(moe_module.Qwen38TTNNMoE.__init__)
    assert (
        "self.hidden_gather_memory_config = (\n"
        "            self.hidden_act_memory_config if self.row_contract.row_tiles == 1 else ttnn.DRAM_MEMORY_CONFIG\n"
        "        )"
    ) in init

    forward = inspect.getsource(moe_module.Qwen38TTNNMoE.forward)
    assert not _calls_named(ast.parse(textwrap.dedent(forward)), "to_memory_config")
    assert "hidden_ws" not in forward
    # the composed route and shared partial keep the gathered shard as their input (the else branch of the MoE dense
    # composite's switch, one indentation level in since 2026-09-26)
    assert (
        'self._route(\n                    temporaries["full_hidden"], hidden_tiles=temporaries["hidden_tiles"], phase_observer=observe\n                )'
        in forward
    )
    assert (
        'self._shared_partial(\n                    hidden_sharded, temporaries["full_hidden"], temporaries["hidden_tiles"]\n                )'
        in forward
    )
    # The gathered shard is also the routed untilize's input, so it is released
    # only after the reduce-scatter enqueue.
    assert forward.index("self._routed_partial(") < forward.index(
        'release_many("local_sum", "full_hidden", "routing_tiles")'
    )
    assert forward.index("tt_all_reduce(") < forward.index('release_many("local_sum", "full_hidden", "routing_tiles")')

    for helper in (moe_module.Qwen38TTNNMoE._route, moe_module.Qwen38TTNNMoE._shared_partial):
        source = inspect.getsource(helper)
        assert "to_memory_config(full_hidden" not in source
        assert "hidden_ws" not in source

    shared = inspect.getsource(moe_module.Qwen38TTNNMoE._shared_partial)
    assert "partial = ttnn.linear(" in shared
    assert "to_memory_config(partial" not in shared
    assert "gated_partials.append(ttnn.mul(partial, scalar_gate, memory_config=ttnn.DRAM_MEMORY_CONFIG))" in shared


def test_qsa_projections_read_the_width_sharded_hidden_the_gather_writes() -> None:
    """The QSA all-gather writes the eight-core activation shard; no interleaved-to-sharded copy follows it."""

    gather = inspect.getsource(qsa_module.Qwen38TTNNQSA._all_gather_hidden)
    gathers = _calls_named(ast.parse(textwrap.dedent(gather)), "all_gather")
    assert len(gathers) == 1
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in gathers[0].keywords}
    assert keywords["memory_config"] == "self.hidden_act_memory_config"
    for body in (qsa_module.Qwen38TTNNQSA.forward_decode, qsa_module.Qwen38TTNNQSA.forward_decode_generic):
        source = inspect.getsource(body)
        assert "to_memory_config(full_hidden" not in source and "hidden_ws" not in source
        assert source.index("self._index_projection(full_hidden, cos, sin)") < source.index(
            "self._main_projection(full_hidden, cos, sin)"
        )
        # The shard is released with the gathered hidden after the output projection.
        assert source.index("self._main_projection(full_hidden, cos, sin)") < source.index("_deallocate(full_hidden)")
    for helper in (qsa_module.Qwen38TTNNQSA._index_projection, qsa_module.Qwen38TTNNQSA._main_projection):
        source = inspect.getsource(helper)
        assert "to_memory_config(full_hidden" not in source and "hidden_ws" not in source
        linears = _calls_named(ast.parse(textwrap.dedent(source)), "linear")
        assert linears and all(ast.unparse(call.args[0]) == "full_hidden" for call in linears)


def test_gdn_gate_product_is_written_in_the_out_projection_layout() -> None:
    source = inspect.getsource(gdn_module.Qwen38TTNNGDN._gate)
    assert "gated = ttnn.multiply(normalized, sigmoid_bf16, memory_config=self.out_proj_act_memory_config)" in source
    assert "to_memory_config(gated" not in source
    out_project = inspect.getsource(gdn_module.Qwen38TTNNGDN._out_project)
    assert "to_memory_config(gated" not in out_project
    assert "partial_ws = ttnn.linear(\n            gated," in out_project


# --- resident RoPE tables for the position-generic body -------------------------


PINNED_CONFIG = SimpleNamespace(
    config_sha256=EXPECTED_CONFIG_SHA256,
    hidden_size=2560,
    residual_branches=4,
    vocab_size=248320,
    num_hidden_layers=48,
    layer_types=("linear_attention", "linear_attention", "linear_attention", "full_attention") * 12,
    max_position_embeddings=262144,
    qsa_rope_dim=64,
    index_compress_ratio=4,
    rope_theta=10_000_000,
)


class _FakeRoPETensor:
    _next_id = 1

    def __init__(self, shape, dtype, layout, *, padded_shape=None, host=None, source=None) -> None:
        self.shape = tuple(shape)
        self.padded_shape = tuple(padded_shape) if padded_shape is not None else tuple(shape)
        self.dtype = dtype
        self.layout = layout
        self.host = host
        self.source = source
        self._id = _FakeRoPETensor._next_id
        _FakeRoPETensor._next_id += 1

    def tensor_id(self) -> int:
        return self._id


def _tile_padded(shape: tuple[int, ...]) -> tuple[int, ...]:
    """TILE padded shape: the last two dims rounded up to the 32x32 tile."""

    return tuple(shape[:-2]) + tuple(-(-dim // ttnn.TILE_SIZE) * ttnn.TILE_SIZE for dim in shape[-2:])


def _patch_rope_lookup_ops(monkeypatch, calls: list) -> None:
    """ttnn stand-ins whose output metadata follows the C++ rules the table lookup relies on."""

    def embedding(indices, weight, **kwargs):
        calls.append(("embedding", indices, weight, kwargs))
        # ttnn/cpp/ttnn/operations/embedding/embedding.cpp:38-39,70-74: the result is
        # [sentence, hidden] for rank-1 indices and [batch, sentence, hidden] otherwise
        # (never the device op's rank-4 shape); the fused TILE program allocates whole
        # 32-row tiles, so 32 indices fill one tile and the padded shape equals it.
        hidden = weight.shape[-1]
        shape = (
            (indices.shape[-1], hidden) if len(indices.shape) == 1 else (indices.shape[0], indices.shape[-1], hidden)
        )
        return _FakeRoPETensor(shape, kwargs["dtype"], kwargs["layout"], padded_shape=_tile_padded(shape))

    def unsqueeze_to_4d(tensor):
        calls.append(("unsqueeze_to_4D", tensor))
        # ttnn/cpp/ttnn/operations/core/core.cpp:20-30: rank 4 is returned as is; a
        # lower rank is reshaped to logical_shape.to_rank(4) / padded_shape.to_rank(4).
        assert len(tensor.shape) <= 4
        lead = (1,) * (4 - len(tensor.shape))
        return _FakeRoPETensor(
            lead + tensor.shape, tensor.dtype, tensor.layout, padded_shape=lead + tensor.padded_shape, source=tensor
        )

    def reshape(tensor, *shapes):
        calls.append(("reshape", tensor, shapes))
        # ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.hpp:21-27: the
        # (logical_shape, padded_shape) overload takes both from its arguments; the
        # one-shape form derives the padding from the layout (tile-aligned for TILE).
        logical = tuple(shapes[0])
        if len(shapes) == 2:
            padded = tuple(shapes[1])
        else:
            padded = logical if tensor.layout == ttnn.ROW_MAJOR_LAYOUT else _tile_padded(logical)
        return _FakeRoPETensor(logical, tensor.dtype, tensor.layout, padded_shape=padded, source=tensor)

    monkeypatch.setattr(model_module.ttnn, "embedding", embedding)
    monkeypatch.setattr(model_module.ttnn, "unsqueeze_to_4D", unsqueeze_to_4d)
    monkeypatch.setattr(model_module.ttnn, "reshape", reshape)
    monkeypatch.setattr(model_module.ttnn, "Shape", lambda shape: tuple(shape))


class _NoopContract:
    def validate_mesh(self, mesh_device) -> None:
        del mesh_device

    def validate_tensor(self, tensor, **kwargs) -> None:
        del tensor, kwargs


def test_rope_table_rows_equal_the_per_position_host_rope_bitwise(monkeypatch) -> None:
    uploads = []

    def from_torch(host, **kwargs):
        uploads.append((host, kwargs))
        return _FakeRoPETensor(tuple(host.shape), kwargs["dtype"], kwargs["layout"], host=host)

    monkeypatch.setattr(model_module.ttnn, "from_torch", from_torch)
    monkeypatch.setattr(model_module, "replicate_tensor_2d_mesh_mapper", lambda mesh_device: "replicate")
    context = 4096

    table = Qwen38TTNNRoPETable.build("mesh", _NoopContract(), PINNED_CONFIG, context)

    assert len(uploads) == 2
    inverse_frequency = _inverse_frequency(PINNED_CONFIG)
    expected_cos = torch.cat([_host_rope(position, inverse_frequency)[0] for position in range(context)], dim=2)
    expected_sin = torch.cat([_host_rope(position, inverse_frequency)[1] for position in range(context)], dim=2)
    for (host, kwargs), expected in zip(uploads, (expected_cos, expected_sin)):
        assert host.dtype == torch.bfloat16 and tuple(host.shape) == (1, 1, context, 64)
        assert torch.equal(host, expected)
        assert kwargs["dtype"] == ttnn.bfloat16 and kwargs["layout"] == ttnn.ROW_MAJOR_LAYOUT
        assert kwargs["device"] == "mesh" and kwargs["memory_config"] == ttnn.DRAM_MEMORY_CONFIG
    assert table.cos_table.host is uploads[0][0] and table.sin_table.host is uploads[1][0]
    assert table.allocated_context == context
    host_rows = table.host_rows(4095)
    assert torch.equal(host_rows["cos"], expected_cos[:, :, 4095:4096]) and torch.equal(
        host_rows["sin"], expected_sin[:, :, 4095:4096]
    )
    assert torch.equal(host_rows["block_start_cos"], expected_cos[:, :, 4092:4093])
    assert torch.equal(host_rows["block_start_sin"], expected_sin[:, :, 4092:4093])
    build = inspect.getsource(Qwen38TTNNRoPETable.build)
    assert "_host_rope(position, inverse_frequency) for position in range(allocated_context)" in build
    assert "torch.outer(torch.arange" not in build


@pytest.mark.parametrize("context", (0, 33, 262144 + 32, True, 4096.0))
def test_rope_table_rejects_non_tile_or_out_of_range_contexts(expect_error, monkeypatch, context) -> None:
    monkeypatch.setattr(model_module.ttnn, "from_torch", lambda *args, **kwargs: pytest.fail("no upload expected"))
    with expect_error((TypeError, ValueError), match="RoPE table context"):
        Qwen38TTNNRoPETable.build("mesh", _NoopContract(), PINNED_CONFIG, context)


def test_rope_table_rows_are_four_embedding_lookups_viewed_as_tile_rows(monkeypatch) -> None:
    cos_table = _FakeRoPETensor((1, 1, 4096, 64), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    sin_table = _FakeRoPETensor((1, 1, 4096, 64), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    table = Qwen38TTNNRoPETable(cos_table, sin_table, 4096, _inverse_frequency(PINNED_CONFIG), _NoopContract())
    calls = []
    _patch_rope_lookup_ops(monkeypatch, calls)
    index_row = _FakeRoPETensor((1, 1, 1, 32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    block_start_row = _FakeRoPETensor((1, 1, 1, 32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    rope = table.rows(index_row, block_start_row)

    assert isinstance(rope, Qwen38TTNNRoPEInputs) and rope.position is None and rope.active
    embeddings = [call for call in calls if call[0] == "embedding"]
    assert len(embeddings) == 4
    assert [call[2] for call in embeddings] == [cos_table, sin_table, cos_table, sin_table]
    assert [call[1].source for call in embeddings] == [index_row, index_row, block_start_row, block_start_row]
    assert all(call[1].shape == (1, 1, 32) and call[1].layout == ttnn.ROW_MAJOR_LAYOUT for call in embeddings)
    for call in embeddings:
        assert call[3] == {
            "layout": ttnn.TILE_LAYOUT,
            "dtype": ttnn.bfloat16,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        }
    # The op hands back [1, 32, 64] (4x p150 failure of 39ca9542); every lookup
    # unsqueezes it once and views the 32-row tile as one row.
    unsqueezes = [call for call in calls if call[0] == "unsqueeze_to_4D"]
    assert len(unsqueezes) == 4
    assert all(call[1].shape == (1, 32, 64) and call[1].padded_shape == (1, 32, 64) for call in unsqueezes)
    views = [call for call in calls if call[0] == "reshape" and len(call[2]) == 2]
    assert [call[2] for call in views] == [((1, 1, 1, 64), (1, 1, 32, 64))] * 4
    assert all(call[1].shape == (1, 1, 32, 64) and call[1].padded_shape == (1, 1, 32, 64) for call in views)
    assert [call[1].source for call in views] == [call[1] for call in unsqueezes]
    for row in (rope.cos, rope.sin, rope.block_start_cos, rope.block_start_sin):
        assert row.shape == (1, 1, 1, 64) and row.padded_shape == (1, 1, 32, 64)
        assert row.dtype == ttnn.bfloat16 and row.layout == ttnn.TILE_LAYOUT
    source = inspect.getsource(Qwen38TTNNRoPETable)
    assert source.count("ttnn.embedding(") == 1
    assert "ttnn.unsqueeze_to_4D(looked_up) if len(looked_up.shape) == 3 else looked_up" in source
    assert "_padded_shape(rows) != padded" in source and "_padded_shape(row) != padded" in source
    assert "ttnn.reshape(rows, ttnn.Shape((1, 1, 1, QSA_ROPE_DIM)), ttnn.Shape(padded))" in source
    assert "ttnn.reshape(index_row, (1, 1, ttnn.TILE_SIZE))" in source
    assert "Qwen38TTNNRoPEInputs(None, *looked_up)" in source
    assert "for_position" not in source and "from_torch" not in inspect.getsource(Qwen38TTNNRoPETable.rows)


def test_rope_table_lookup_reports_actual_and_expected_metadata(expect_error, monkeypatch) -> None:
    table = Qwen38TTNNRoPETable(
        _FakeRoPETensor((1, 1, 4096, 64), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        _FakeRoPETensor((1, 1, 4096, 64), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        4096,
        _inverse_frequency(PINNED_CONFIG),
        _NoopContract(),
    )
    calls = []
    _patch_rope_lookup_ops(monkeypatch, calls)
    # A ROW_MAJOR lookup result (as if layout=TILE were ignored) must name what came back.
    monkeypatch.setattr(
        model_module.ttnn,
        "embedding",
        lambda indices, weight, **kwargs: _FakeRoPETensor((1, 32, 64), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    )
    index_row = _FakeRoPETensor((1, 1, 1, 32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    with pytest.raises(RuntimeError) as error:  # allow-pytest.raises: inspect the captured exception object
        table.rows(index_row, index_row)
    message = str(error.value)
    assert message.startswith("QSA RoPE cos lookup must be BF16 TILE [1, 1, 32, 64] backed by [1, 1, 32, 64], got ")
    assert "shape=[1, 1, 32, 64] padded_shape=[1, 1, 32, 64]" in message
    assert "(ttnn.embedding returned shape=[1, 32, 64] padded_shape=[1, 32, 64]" in message
    # A wrongly shaped index row is reported the same way.
    with expect_error(RuntimeError, match=r"must be UINT32 ROW_MAJOR \[1,1,1,32\], got shape=\[1, 1, 32\]"):
        table.rows(_FakeRoPETensor((1, 1, 32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), index_row)


def test_prepared_inputs_carry_optional_host_rope_and_position() -> None:
    fields = {field.name: field for field in dataclasses.fields(Qwen38TTNNPreparedDecodeInputs)}
    assert fields["rope"].type == "Qwen38TTNNRoPEInputs | None"
    assert fields["position"].type == "int | None"
    assert dataclasses.fields(Qwen38TTNNRoPEInputs)[0].type == "int | None"

    class FakePLEInput:
        active = True

        def release(self) -> None:
            self.active = False

    prepared = Qwen38TTNNPreparedDecodeInputs(
        input_token_id=17,
        position=None,
        host_token=torch.tensor([[17]]),
        residual_sharded=None,
        rope=None,
        ple=FakePLEInput(),
        device_token=object(),
    )
    prepared.release()
    assert prepared.active is False and prepared.ple.active is False


# --- position-generic decoder layer -----------------------------------------------


GENERIC_LAYER_PHASES = [
    "before-ple",
    "after-ple",
    "before-attention-gr-read",
    "after-attention-gr-read",
    "before-gdn",
    "after-gdn",
    "before-attention-gr-write",
    "after-attention-gr-write",
    "before-mlp-gr-read",
    "after-mlp-gr-read",
    "before-expert-stream-acquire",
    "after-expert-stream-acquire",
    "before-moe-forward",
    "after-moe-forward",
    "before-expert-stream-release",
    "after-expert-stream-release",
    "before-mlp-gr-write",
    "after-mlp-gr-write",
]


def _generic_layer(monkeypatch, *, layer_type: Qwen38TTNNLayerType, operations: list, ple: bool):
    class FakeGR:
        def __init__(self, label: str) -> None:
            self.label = label

        def read(self, residual):
            operations.append((f"{self.label}-read", residual))
            return object(), SimpleNamespace(residual=object(), injection=object())

        def write(self, hidden, state):
            operations.append((f"{self.label}-write", hidden))
            return f"{self.label}-written"

    class FakeStreamer:
        class Lease:
            def __enter__(self):
                operations.append(("expert-stream-acquire", None))
                return (object(), object())

            def __exit__(self, error_type, error, traceback) -> None:
                del error_type, error, traceback
                operations.append(("expert-stream-release", None))

        def layer(self, layer_index: int, *, namespace: str):
            assert namespace == "backbone"
            return self.Lease()

    def moe_forward(hidden, packed_w01, packed_w2, *, return_routing, phase_observer=None):
        del packed_w01, packed_w2, phase_observer
        operations.append(("moe-forward", hidden, return_routing))
        return SimpleNamespace(hidden_sharded="moe-hidden", routing=None)

    layer = Qwen38TTNNDecoderLayer.__new__(Qwen38TTNNDecoderLayer)
    layer.namespace = Qwen38TTNNLayerNamespace.BACKBONE
    layer.layer_index = 1 if layer_type is Qwen38TTNNLayerType.GDN else 3
    layer.layer_type = layer_type
    layer.attention = (Qwen38TTNNGDN if layer_type is Qwen38TTNNLayerType.GDN else Qwen38TTNNQSA).__new__(
        Qwen38TTNNGDN if layer_type is Qwen38TTNNLayerType.GDN else Qwen38TTNNQSA
    )
    layer.attention_gr = FakeGR("attention-gr")
    layer.mlp = SimpleNamespace(forward=moe_forward)
    layer.mlp_gr = FakeGR("mlp-gr")
    layer.expert_streamer = FakeStreamer()
    layer.ple = object() if ple else None
    layer._validate_residual = lambda residual, label: None
    layer._validate_block = lambda hidden, label: None
    layer._validate_generic_state = lambda state: operations.append(("validate-generic-state", state))
    layer._apply_ple = lambda residual, state, *, token_id=None, prepared_ple=None, release_input=True: (
        operations.append(("ple", token_id, prepared_ple, release_input)) or residual,
        state.ple,
    )
    monkeypatch.setattr(layer_module, "_deallocate_unique", lambda *tensors: operations.append(("deallocate", tensors)))
    return layer


def test_generic_layer_state_has_no_host_position() -> None:
    assert tuple(field.name for field in dataclasses.fields(Qwen38TTNNDecoderLayerGenericState)) == (
        "namespace",
        "layer_index",
        "attention",
        "ple",
    )
    signature = inspect.signature(Qwen38TTNNDecoderLayer.forward_decode_generic)
    assert tuple(signature.parameters) == (
        "self",
        "residual_sharded",
        "state",
        "prepared_ple",
        "rope",
        "qsa_position",
        "release_input",
        "phase_observer",
    )
    assert all(
        signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in ("prepared_ple", "rope", "qsa_position", "release_input", "phase_observer")
    )
    # The per-position body and the PLE helper keep consuming their input by default.
    assert signature.parameters["release_input"].default is True
    assert inspect.signature(Qwen38TTNNDecoderLayer._apply_ple).parameters["release_input"].default is True
    assert "release_input" not in inspect.getsource(Qwen38TTNNDecoderLayer.forward_decode)


def test_generic_gdn_layer_runs_the_checkpoint_sequence_in_place_with_a_context_free_ple_row(
    expect_error, monkeypatch
) -> None:
    operations: list = []
    phases: list[str] = []
    layer = _generic_layer(monkeypatch, layer_type=Qwen38TTNNLayerType.GDN, operations=operations, ple=True)
    attention_state = SimpleNamespace(layer_index=1)
    layer.attention.forward_decode = lambda hidden, state: (
        operations.append(("gdn-forward", hidden, state)) or SimpleNamespace(hidden_sharded="gdn-hidden", state=state)
    )
    ple_state = SimpleNamespace(token_context=torch.tensor([[5, 6]]))
    state = Qwen38TTNNDecoderLayerGenericState(Qwen38TTNNLayerNamespace.BACKBONE, 1, attention_state, ple_state)
    prepared_ple = SimpleNamespace(source_token_context=None)

    residual = layer.forward_decode_generic(
        "residual-in", state, prepared_ple=prepared_ple, rope=None, qsa_position=None, phase_observer=phases.append
    )

    assert residual == "mlp-gr-written"
    assert phases == GENERIC_LAYER_PHASES
    assert [operation[0] for operation in operations] == [
        "validate-generic-state",
        "ple",
        "attention-gr-read",
        "gdn-forward",
        "deallocate",
        "attention-gr-write",
        "deallocate",
        "mlp-gr-read",
        "expert-stream-acquire",
        "moe-forward",
        "expert-stream-release",
        "deallocate",
        "mlp-gr-write",
        "deallocate",
    ]
    assert operations[1] == ("ple", None, prepared_ple, True)
    assert operations[3][2] is attention_state
    assert operations[5][1] == "gdn-hidden"
    assert operations[9][2] is False
    # The caller owns the n-gram context in the generic body; the PLE state's
    # host bookkeeping is pinned to the context-free prepared row.
    assert ple_state.token_context is None

    with expect_error(ValueError, match="without a host n-gram context"):
        layer.forward_decode_generic(
            "residual-in",
            state,
            prepared_ple=SimpleNamespace(source_token_context=torch.tensor([[1, 2]])),
            rope=None,
            qsa_position=None,
        )
    with expect_error(ValueError, match="requires its prepared persistent row"):
        layer.forward_decode_generic("residual-in", state, prepared_ple=None, rope=None, qsa_position=None)


def test_generic_qsa_layer_passes_device_rope_rows_and_position_inputs_to_the_generic_qsa_body(
    expect_error, monkeypatch
) -> None:
    operations: list = []
    layer = _generic_layer(monkeypatch, layer_type=Qwen38TTNNLayerType.QSA, operations=operations, ple=False)
    attention_state = SimpleNamespace(layer_index=3)
    qsa_calls = []

    def forward_decode_generic(hidden, state, **kwargs):
        qsa_calls.append((hidden, state, kwargs))
        return "qsa-hidden"

    layer.attention.forward_decode_generic = forward_decode_generic
    state = Qwen38TTNNDecoderLayerGenericState(Qwen38TTNNLayerNamespace.BACKBONE, 3, attention_state, None)
    rope = SimpleNamespace(cos="cos", sin="sin", block_start_cos="block-cos", block_start_sin="block-sin")
    qsa_position = object()

    residual = layer.forward_decode_generic(
        "residual-in", state, prepared_ple=None, rope=rope, qsa_position=qsa_position
    )

    assert residual == "mlp-gr-written"
    assert len(qsa_calls) == 1 and qsa_calls[0][1] is attention_state
    assert qsa_calls[0][2] == {
        "cos": "cos",
        "sin": "sin",
        "block_start_cos": "block-cos",
        "block_start_sin": "block-sin",
        "position": qsa_position,
    }
    assert ("attention-gr-write", "qsa-hidden") in operations
    with expect_error(ValueError, match="device RoPE rows and QSA position inputs"):
        layer.forward_decode_generic("residual-in", state, prepared_ple=None, rope=rope, qsa_position=None)

    source = inspect.getsource(Qwen38TTNNDecoderLayer.forward_decode_generic)
    for forbidden in (
        "state.position",
        "next_position",
        "Qwen38TTNNDecoderLayerState(",
        "reuse_selection",
        "retain_input_state",
        "snapshot",
        "%",
        "//",
    ):
        assert forbidden not in source
    assert "self.attention.forward_decode_generic(" in source
    assert "return_routing=False" in source
    class_source = inspect.getsource(Qwen38TTNNDecoderLayer)
    assert class_source.count("self._route_through_gr_and_moe(") == 2


def test_generic_layer_state_lifecycle_is_in_place_and_per_component(monkeypatch) -> None:
    operations: list = []
    layer = _generic_layer(monkeypatch, layer_type=Qwen38TTNNLayerType.GDN, operations=operations, ple=True)
    attention_state = SimpleNamespace(
        layer_index=1,
        reset_inplace=lambda: operations.append(("gdn-reset", None)),
        deallocate=lambda: operations.append(("gdn-deallocate", None)),
    )
    ple_state = SimpleNamespace(
        reset_inplace=lambda: operations.append(("ple-reset", None)),
        deallocate=lambda: operations.append(("ple-deallocate", None)),
    )
    state = Qwen38TTNNDecoderLayerGenericState(Qwen38TTNNLayerNamespace.BACKBONE, 1, attention_state, ple_state)

    layer.reset_generic_state_inplace(state)
    layer.release_generic_state(state)

    assert [operation[0] for operation in operations] == [
        "validate-generic-state",
        "gdn-reset",
        "ple-reset",
        "validate-generic-state",
        "gdn-deallocate",
        "ple-deallocate",
    ]
    reset = inspect.getsource(Qwen38TTNNDecoderLayer.reset_generic_state_inplace)
    assert "state.attention.reset_inplace()" in reset
    assert "self.attention.reset_generic_state_inplace(state.attention)" in reset
    assert "state.ple.reset_inplace()" in reset
    assert "allocate" not in reset and "reset_state(" not in reset
    allocate = inspect.getsource(Qwen38TTNNDecoderLayer.allocate_generic_state)
    assert "self.attention.allocate_generic_state()" in allocate and "self.attention.allocate_state()" in allocate
    release = inspect.getsource(Qwen38TTNNDecoderLayer._release_generic_attention_state)
    assert "self.attention.release_generic_state(attention_state)" in release


# --- position-generic text model body ---------------------------------------------


def _generic_model(monkeypatch, log: list):
    class FakeLayer:
        def __init__(self, index: int) -> None:
            self.layer_index = index
            self.layer_type = Qwen38TTNNLayerType.QSA if index % 4 == 3 else Qwen38TTNNLayerType.GDN
            self.attention = SimpleNamespace(allocated_compressed_blocks=8192)

        def validate_generic_state(self, state) -> None:
            assert state.layer_index == self.layer_index

        def allocate_generic_state(self):
            log.append(("allocate", self.layer_index))
            return SimpleNamespace(namespace=Qwen38TTNNLayerNamespace.BACKBONE, layer_index=self.layer_index)

        def reset_generic_state_inplace(self, state) -> None:
            log.append(("reset", self.layer_index))

        def release_generic_state(self, state) -> None:
            log.append(("release", self.layer_index))

        def forward_decode_generic(self, residual, state, **kwargs):
            log.append(("forward", self.layer_index, residual, kwargs))
            return f"residual-{self.layer_index}"

    class FakePosition:
        scalar = "position-scalar"
        rope_shift = 0  # the rotary shift's host mirror (a text stand-in: 0)

        def index_row(self):
            log.append(("index_row",))
            return "index-row"

        def block_start_index_row(self, index_row):
            log.append(("block_start_index_row", index_row))
            return "block-start-row"

        def rope_index_row(self, index_row):
            log.append(("rope_index_row", index_row))
            return "rope-index-row"

        def rope_block_start_index_row(self, index_row):
            log.append(("rope_block_start_index_row", index_row))
            return "rope-block-start-row"

        def advance(self) -> None:
            log.append(("advance",))

        def reset(self, position: int, rope_shift: int | None = None) -> None:
            log.append(("position-reset", position))

        def deallocate(self) -> None:
            log.append(("position-deallocate",))

    class FakeRoPE:
        active = True
        cos = sin = block_start_cos = block_start_sin = "rope-row"

        def deallocate(self) -> None:
            log.append(("rope-deallocate",))

    class FakeQSAPosition:
        def deallocate(self) -> None:
            log.append(("qsa-position-deallocate",))

    owner = Qwen38TTNNTextModel.__new__(Qwen38TTNNTextModel)
    owner.layers = tuple(FakeLayer(index) for index in range(48))
    owner.mesh_contract = _NoopContract()
    owner.mesh_device = "mesh"
    owner._state_owner = object()
    owner._poisoned_error = None
    owner._active_snapshot = None
    owner._poisoned_device_owners = []
    owner.rope_table = SimpleNamespace(
        rows=lambda index_row, block_start_row: log.append(("rows", index_row, block_start_row)) or FakeRoPE()
    )
    owner.qsa_position_constants = "qsa-constants"
    owner._embed_residual_from_device_token = lambda token_row: log.append(("embed", token_row)) or "residual-in"
    owner._validate_residual = lambda tensor, label: None
    owner._validate_hidden = lambda tensor, label: None
    owner._validate_generic_head = lambda head: log.append(("validate-head", head.residual, head.active))
    owner.final_mixer = lambda residual: log.append(("final-mixer", residual)) or "hidden"
    owner.model_io = SimpleNamespace(
        lm_head=lambda hidden: log.append(("lm-head", hidden)) or SimpleNamespace(tensor="logits-tensor"),
        embedding=SimpleNamespace(validate_token_row=lambda row, label: None),
    )
    monkeypatch.setattr(
        model_module.qsa_module,
        "derive_qsa_position_inputs",
        lambda scalar, constants: log.append(("derive", scalar, constants)) or FakeQSAPosition(),
        raising=False,
    )
    monkeypatch.setattr(model_module, "_deallocate_unique", lambda *tensors: log.append(("deallocate", tensors)))
    layer_states = tuple(
        SimpleNamespace(
            namespace=Qwen38TTNNLayerNamespace.BACKBONE, layer_index=index, ple=SimpleNamespace(token_context=None)
        )
        for index in range(48)
    )
    position = FakePosition()
    monkeypatch.setattr(model_module, "Qwen38TTNNDevicePosition", type(position))
    state = Qwen38TTNNTextModelGenericState(position, layer_states, owner._state_owner)
    ple_input = SimpleNamespace(active=True)
    prepared = Qwen38TTNNPreparedDecodeInputs(
        input_token_id=17,
        position=None,
        host_token=torch.tensor([[17]]),
        residual_sharded=None,
        rope=None,
        ple=ple_input,
        device_token="token-row",
    )
    return owner, state, prepared


def test_generic_model_body_reads_the_position_first_derives_once_and_advances_last(monkeypatch) -> None:
    log: list = []
    owner, state, prepared = _generic_model(monkeypatch, log)

    output = owner.forward_decode_generic(prepared, state)

    assert isinstance(output, Qwen38TTNNGenericDecodeOutput) and output.logits.tensor == "logits-tensor"
    names = [entry[0] for entry in log]
    forwards = [entry for entry in log if entry[0] == "forward"]
    # HEAD: embed + layer 0 (no position read); TAIL: derive the position inputs, layers 1-47, advance last.
    head_names = ["embed", "forward", "validate-head"]
    derive_names = [
        "validate-head",
        "index_row",
        "rope_index_row",
        "rope_block_start_index_row",
        "rows",
        "deallocate",
        "derive",
    ]
    assert names[: len(head_names)] == head_names
    assert names[3:10] == derive_names
    assert names[10:57] == ["forward"] * 47
    assert names[57:] == [
        "qsa-position-deallocate",
        "rope-deallocate",
        "final-mixer",
        "deallocate",
        "lm-head",
        "deallocate",
        "advance",
    ]
    assert log[0] == ("embed", "token-row")
    assert log[2] == ("validate-head", "residual-0", True)
    assert log[5] == ("rope_index_row", "index-row")
    assert log[6] == ("rope_block_start_index_row", "index-row")
    assert log[7] == ("rows", "rope-index-row", "rope-block-start-row")
    assert log[8] == ("deallocate", ("index-row", "rope-index-row", "rope-block-start-row"))
    assert log[9] == ("derive", "position-scalar", "qsa-constants")
    assert [entry[1] for entry in forwards] == list(range(48))
    assert forwards[0][2] == "residual-in" and forwards[1][2] == "residual-0"
    # Layer 0 (HEAD) gets no PLE row, RoPE rows or position inputs; the TAIL layers share one of each.
    assert forwards[0][3] == {"prepared_ple": None, "rope": None, "qsa_position": None}
    tail_forwards = forwards[1:]
    qsa_positions = {id(entry[3]["qsa_position"]) for entry in tail_forwards}
    ropes = {id(entry[3]["rope"]) for entry in tail_forwards}
    assert len(qsa_positions) == 1 and len(ropes) == 1
    assert [entry[3]["prepared_ple"] is prepared.ple for entry in forwards] == [index == 1 for index in range(48)]
    # The fused body consumes the head residual in layer 1 exactly as before the split.
    assert [entry[3]["release_input"] for entry in tail_forwards] == [True] * 47
    assert log[-1] == ("advance",)
    assert not owner.poisoned

    released = []
    monkeypatch.setattr(model_module.ttnn, "deallocate", released.append)
    output.release_tensors()
    assert released == ["logits-tensor"] and not output.active


def test_generic_model_body_rejects_host_rope_or_position_and_poisons_on_a_layer_failure(
    expect_error, monkeypatch
) -> None:
    log: list = []
    owner, state, prepared = _generic_model(monkeypatch, log)
    with expect_error(TypeError, match="without host RoPE/position"):
        owner.forward_decode_generic(dataclasses.replace(prepared, position=3), state)
    with expect_error(TypeError, match="without host RoPE/position"):
        owner.forward_decode_generic(dataclasses.replace(prepared, rope=SimpleNamespace(active=True)), state)
    assert log == [] and not owner.poisoned

    def failing_forward(residual, state, **kwargs):
        raise RuntimeError("injected layer failure")

    owner.layers[40].forward_decode_generic = failing_forward
    with expect_error(Qwen38TTNNModelPoisonedError, match="poisoned after forward_decode_generic_tail; 40 layer"):
        owner.forward_decode_generic(prepared, state)
    assert owner.poisoned and ("advance",) not in log
    with expect_error(Qwen38TTNNModelPoisonedError):
        owner.reset_generic_state_inplace(state)


def test_generic_model_prologue_is_a_position_reset_plus_in_place_layer_resets(monkeypatch) -> None:
    log: list = []
    owner, state, _ = _generic_model(monkeypatch, log)

    owner.reset_generic_state_inplace(state)

    assert log == [("position-reset", 0), *(("reset", index) for index in range(48))]
    log.clear()
    owner.release_generic_state(state)
    assert log == [*(("release", index) for index in reversed(range(48))), ("position-deallocate",)]

    reset = inspect.getsource(Qwen38TTNNTextModel.reset_generic_state_inplace)
    assert "state.position.reset(0, 0)" in reset
    assert "layer.reset_generic_state_inplace(layer_state)" in reset
    for forbidden in ("allocate_state(", "reset_state(", "allocate_generic_state(", "snapshot"):
        assert forbidden not in reset


def test_generic_model_snapshot_copies_the_recurrent_buffers_and_restores_position_and_phases(
    expect_error, monkeypatch
) -> None:
    """The prompt-end snapshot (REPLY-TAIL-SPLICE-DESIGN-20260906): one buffer per GDN recurrent state and ring
    slot, PLE slot, QSA staging tile and raw-key ring (213 for the 48 layers; the caches are positional and not
    copied); the capture copies state -> snapshot, the restore copies back, resets the position and sets every GDN
    phase to position mod 4."""

    log: list = []
    owner, state, _ = _generic_model(monkeypatch, log)
    layers = []
    for index, layer_state in enumerate(state.layers):
        if owner.layers[index].layer_type is Qwen38TTNNLayerType.GDN:
            attention = SimpleNamespace(
                recurrent=f"recurrent-{index}", conv=tuple(f"conv-{index}-{slot}" for slot in range(4)), conv_phase=0
            )
        else:
            attention = SimpleNamespace(kv_staging=f"staging-{index}", raw_key_ring=f"ring-{index}")
        ple = (
            SimpleNamespace(conv=tuple(f"ple-{slot}" for slot in range(9)), token_context="stale")
            if index == 1
            else None
        )
        layers.append(SimpleNamespace(namespace=layer_state.namespace, layer_index=index, attention=attention, ple=ple))
    state = Qwen38TTNNTextModelGenericState(state.position, tuple(layers), owner._state_owner)
    monkeypatch.setattr(
        model_module.ttnn,
        "empty_like",
        lambda tensor, **kwargs: log.append(("empty_like", tensor, tuple(kwargs))) or f"snapshot:{tensor}",
        raising=False,
    )
    monkeypatch.setattr(model_module.ttnn, "copy", lambda source, target: log.append(("copy", source, target)))

    snapshot = owner.allocate_generic_snapshot(state)
    labels = [label for label, _, _ in snapshot.pairs]
    assert len(snapshot.pairs) == 36 * 5 + 12 * 2 + 9 == 213 and len(set(labels)) == 213
    assert [entry for entry in log if entry[0] == "empty_like"] == [
        ("empty_like", source, ("memory_config",)) for _, source, _ in snapshot.pairs
    ]
    assert labels[:7] == [
        "layer 0 GDN recurrent",
        *(f"layer 0 GDN conv[{slot}]" for slot in range(4)),
        "layer 1 GDN recurrent",
        "layer 1 GDN conv[0]",
    ]
    assert labels[10:19] == [f"layer 1 PLE conv[{slot}]" for slot in range(9)]
    assert labels[24:26] == ["layer 3 QSA KV staging", "layer 3 QSA raw-key ring"]
    assert all(target == f"snapshot:{source}" for _, source, target in snapshot.pairs)
    assert len(snapshot.gdn_states) == 36 and len(snapshot.ple_states) == 1 and not snapshot.captured
    with expect_error(RuntimeError, match="never captured"):
        owner.restore_generic_snapshot(snapshot, state)

    log.clear()
    owner.capture_generic_snapshot(state, snapshot, position=37)
    assert log == [("copy", source, target) for _, source, target in snapshot.pairs]
    assert snapshot.position == 37 and snapshot.captured

    log.clear()
    owner.restore_generic_snapshot(snapshot, state)
    assert log == [*(("copy", target, source) for _, source, target in snapshot.pairs), ("position-reset", 37)]
    assert {gdn.conv_phase for gdn in snapshot.gdn_states} == {37 % 4} and layers[1].ple.token_context is None
    with expect_error(ValueError, match="snapshot position must be an int"):
        owner.capture_generic_snapshot(state, snapshot, position=owner.allocated_context + 1)
    with expect_error(ValueError, match="not allocated by this model owner"):
        owner.restore_generic_snapshot(dataclasses.replace(snapshot, _owner=object()), state)

    # An MTP alignment layer's generic state rides along as an extra (QSA) layer.
    extra = owner.allocate_generic_snapshot(state, extra_layers=((owner.layers[3], layers[3]),))
    assert len(extra.pairs) == 215 and [label for label, _, _ in extra.pairs][-2:] == labels[24:26]
    log.clear()
    owner.release_generic_snapshot(snapshot)
    assert log == [("deallocate", tuple(target for _, _, target in snapshot.pairs))] and not snapshot.captured
    source = inspect.getsource(Qwen38TTNNTextModel.capture_generic_snapshot) + inspect.getsource(
        Qwen38TTNNTextModel.restore_generic_snapshot
    )
    for forbidden in ("synchronize", "to_torch", "from_torch", "empty_like", "deallocate"):
        assert forbidden not in source


def test_generic_model_body_source_has_no_host_position_and_advances_as_its_last_device_op() -> None:
    # The body is HEAD then TAIL; the pins below read the two halves in body order.
    fused = inspect.getsource(Qwen38TTNNTextModel.forward_decode_generic)
    assert fused.index("self.forward_decode_generic_head(prepared, state)") < fused.index(
        "self.forward_decode_generic_tail(head, prepared, state, return_logits=return_logits)"
    )
    assert "ttnn." not in fused and "state.position" not in fused
    head_source = inspect.getsource(Qwen38TTNNTextModel.forward_decode_generic_head)
    tail_source = inspect.getsource(Qwen38TTNNTextModel.forward_decode_generic_tail)
    source = inspect.getsource(Qwen38TTNNTextModel._require_generic_decode_inputs) + head_source + tail_source
    assert head_source.index("self._embed_residual_from_device_token(prepared.device_token)") < head_source.index(
        "layer.forward_decode_generic("
    )
    order = [
        "state.position.index_row()",
        "state.position.rope_index_row(index_row)",
        "state.position.rope_block_start_index_row(index_row)",
        "self.rope_table.rows(rope_index_row, rope_block_start_row)",
        "qsa_module.derive_qsa_position_inputs(state.position.scalar, self.qsa_position_constants)",
        "layer.forward_decode_generic(",
        "qsa_position.deallocate()",
        "rope.deallocate()",
        "self.final_mixer(residual)",
        "self.model_io.lm_head(hidden)",
        "state.position.advance()",
    ]
    indices = [tail_source.index(text) for text in order]
    assert indices == sorted(indices)
    assert source.count("derive_qsa_position_inputs(") == 1
    assert source.count("state.position.advance()") == 1
    tail = source[source.index("state.position.advance()") :]
    assert "ttnn." not in tail and "self." not in tail.split("except", 1)[0].replace("state.position.advance()", "")
    for forbidden in (
        "for_position",
        "snapshot_state",
        "restore_state",
        "synchronize",
        "to_torch",
        "from_torch",
        "state.position +",
        "prepared.position",
        "% ",
        "//",
        "Qwen38TTNNTextModelState(",
        "greedy_token(",
    ):
        assert forbidden not in source.replace("prepared.position is not None", "")
    prepare = inspect.getsource(Qwen38TTNNTextModel.prepare_generic_decode_inputs)
    assert "resident_lookup=True" in prepare and "rope=None" in prepare and "position=None" in prepare
    assert "for_position" not in prepare and "_embed_residual" not in prepare
    allocate = inspect.getsource(Qwen38TTNNTextModel.allocate_generic_state)
    assert "Qwen38TTNNRoPETable.build(" in allocate
    assert "qsa_module.Qwen38TTNNQSAPositionConstants.build(" in allocate
    assert "Qwen38TTNNDevicePosition.allocate(self.mesh_device, self.mesh_contract, position=0)" in allocate
    assert "layer.allocate_generic_state()" in allocate
    # The per-position paths are untouched: the eager body still uploads RoPE per position.
    assert "self.rope.for_position(state.position)" in inspect.getsource(Qwen38TTNNTextModel.prepare_decode_inputs)
