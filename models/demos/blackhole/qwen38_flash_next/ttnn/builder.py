# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed construction for the exact four-P150 Qwen3.8 target.

This module separates conversion from model construction.  A caller must:

1. supply the pinned checkpoint and every provenance digest explicitly;
2. open an already-leased physical ``1x4`` Blackhole mesh and name the
   collective topology selected by the topology gate;
3. stage each routed-expert layer independently with
   :meth:`Qwen38TTNNBuilder.stage_bf4_layer`; and
4. call :meth:`Qwen38TTNNBuilder.build_target` only after all 48 backbone
   artifacts are present and hash-valid.

Staging never retains packed routed weights: the converter returns one live
BF4_B layer, this owner validates it, and then immediately deallocates it.
The default completed target shares one single-slot
:class:`Qwen38BF4Streamer` across all layers.  An explicit resident policy
instead preloads exact TP4/EP4 BF4 pairs during component construction and
requires a bounded physical QSA-cache capacity.

The builder does not open devices, acquire locks, choose physical IDs, infer a
runtime revision, or choose a collective topology.  Those are launch-time
safety decisions and are mandatory constructor inputs.  Constructing this
object performs live mesh and DRAM-ring queries, so it must happen only inside
the full-lifetime exact-device lease.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import ttnn
from models.demos.blackhole.qwen38_flash_next.checkpoint import (
    CHECKPOINT_FILE_MANIFEST_SHA256,
    INDEX_SHA256,
    Qwen38Checkpoint,
)
from models.demos.blackhole.qwen38_flash_next.config import CONFIG_SHA256, LAYER_PATTERN, Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.tt.ple import Qwen38HostPLEEmbedding
from models.demos.blackhole.qwen38_flash_next.ttnn.bf4 import (
    BF4CacheIdentity,
    BF4CleanupError,
    BF4LayerRecord,
    Qwen38BF4Cache,
    Qwen38BF4ResidentSet,
    Qwen38BF4Streamer,
    bf4_converter_source_identity,
    qualify_live_bf4_ring,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE, Qwen38MeshContract, is_slab_rows
from models.demos.blackhole.qwen38_flash_next.ttnn.decode_matmul import (  # noqa: E402  (the placement source, READY)
    TWO_READER_QUALIFIED_DTYPES,
    DenseWeightPlan,
    decode_dram_workers_placement,
    default_decode_dram_workers,
    default_dense_weight_plan,
    dense_dtype_tag,
    qualify_decode_dram_workers,
    validate_decode_dram_workers,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import (
    PINNED_CHECKPOINT_REVISION,
    PINNED_TENSOR_MANIFEST_SHA256,
    Qwen38IOCache,
    Qwen38IOCacheIdentity,
    Qwen38TTNNEmbeddingSyncPolicy,
    Qwen38TTNNModelIO,
    Qwen38TTNNModelIOWeights,
    validate_terminal_architecture,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.final_mixer import Qwen38TTNNFinalMixer, Qwen38TTNNFinalMixerWeights
from models.demos.blackhole.qwen38_flash_next.ttnn.gdn import Qwen38TTNNGDN, Qwen38TTNNGDNWeights
from models.demos.blackhole.qwen38_flash_next.ttnn.gr import Qwen38TTNNGatedResidual, Qwen38TTNNGatedResidualWeights
from models.demos.blackhole.qwen38_flash_next.ttnn.layer import Qwen38TTNNDecoderLayer, Qwen38TTNNLayerNamespace
from models.demos.blackhole.qwen38_flash_next.ttnn.model import Qwen38TTNNTextModel
from models.demos.blackhole.qwen38_flash_next.ttnn.moe import (
    Qwen38TTNNMoE,
    Qwen38TTNNMoESyncPolicy,
    Qwen38TTNNMoEWeights,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.mtp import Qwen38TTNNMTPInput, Qwen38TTNNMTPInputWeights
from models.demos.blackhole.qwen38_flash_next.ttnn.ple import Qwen38TTNNPLE, Qwen38TTNNPLEWeights
from models.demos.blackhole.qwen38_flash_next.ttnn.prefill_dense import (
    Qwen38PrefillDensePolicy,
    Qwen38TTNNPrefillDense,
    admit_prefill_dense_dram,
    build_prefill_dense_weights,
    slab_working_set_bytes,
    weight_dtype_name,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import (
    CACHE_WRITE_ROWS,
    COMPRESS_RATIO,
    HEAD_DIM,
    INDEX_HEAD_DIM,
    MAX_CONTEXT,
    ROPE_DIM,
    Qwen38TTNNQSA,
    Qwen38TTNNQSAWeights,
    validate_qsa_cache_capacity,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.vision_residency import VISION_ROW_BUCKETS, Qwen38VisionResidency
from models.tt_transformers.tt.ccl import TT_CCL

TP_SIZE = 4
BACKBONE_LAYERS = 48
MTP_LAYERS = 1
PLE_LAYER = 1
HIDDEN_SIZE = 2560
VOCAB_SIZE = 248_320
ROUTED_EXPERTS = 512
EXPERTS_PER_DEVICE = 128
TOP_K = 10
BUILDER_FORMAT_VERSION = 1
# The resident build allocates one of four contexts: 32,768 (the default; every timing and bitwise pin is at this
# allocation), 65,536, 131,072 or 262,144 (every QSA op admitted at each by the long-context micro-test, the 4x p150 host
# 2026-09-03; the state grows by 1,088 bytes per token per QSA layer: 1.71 GB per device at 131,072, 3.43 GB at
# 262,144, which leaves about 2.7 GB and about 1 GB of the measured 4.4 GB headroom, so 262,144 is a single-user
# configuration).  RESIDENT_MAX_QSA_CACHE_CAPACITY keeps the default's historical name: demo/production.py and the
# 32k pins read it.
RESIDENT_DEFAULT_QSA_CACHE_CAPACITY = 32_768
RESIDENT_QSA_CACHE_CAPACITIES = (8_192, RESIDENT_DEFAULT_QSA_CACHE_CAPACITY, 65_536, 131_072, 262_144)
RESIDENT_MAX_QSA_CACHE_CAPACITY = RESIDENT_DEFAULT_QSA_CACHE_CAPACITY
# Admission bucket of the resident build per device beyond the 49 BF4 expert pairs at the default context: the
# non-expert weights, the persistent state, model I/O, allocator padding, transient output and MoE workspace.
RESIDENT_DEFAULT_RESERVE_BYTES_PER_DEVICE = 5 << 30
# The device position P must stay below the allocated context (RoPE table lookup, KV write); the consumed EOS step
# and a little slack are the headroom the chat session keeps.
RESIDENT_CONTEXT_HEADROOM = 64
QSA_LAYERS = BACKBONE_LAYERS // 4

AttentionKind = Literal["gdn", "qsa"]
Namespace = Literal["backbone", "mtp"]


@dataclass(frozen=True)
class Qwen38ResidentContext:
    """One resident build's allocated context and every size the builder, the chain and the admission derive from it.

    The QSA caches, the compressed-index cache (the generic state carries one extra tile: the fixed indexer window),
    the RoPE tables and the position constants all follow ``allocated_context``; the reserve grows by the state the
    larger context adds over the default, so the capacity admission stays the measured 5 GiB bucket plus that delta.
    """

    allocated_context: int = RESIDENT_DEFAULT_QSA_CACHE_CAPACITY

    def __post_init__(self) -> None:
        if type(self.allocated_context) is not int or self.allocated_context not in RESIDENT_QSA_CACHE_CAPACITIES:
            raise ValueError(
                f"resident allocated context must be one of {RESIDENT_QSA_CACHE_CAPACITIES}, "
                f"got {self.allocated_context!r}"
            )

    @classmethod
    def of(cls, allocated_context: int | None) -> "Qwen38ResidentContext":
        return cls() if allocated_context is None else cls(allocated_context)

    @property
    def label(self) -> str:
        return f"c{self.allocated_context}"

    @property
    def compressed_blocks(self) -> int:
        return self.allocated_context // COMPRESS_RATIO

    @property
    def indexer_window_rows(self) -> int:
        """Rows of the generic state's compressed-index cache: every block plus the fixed 32-row indexer window."""

        return self.compressed_blocks + ttnn.TILE_SIZE

    @property
    def rope_rows(self) -> int:
        return self.allocated_context

    @property
    def context_limit(self) -> int:
        return self.allocated_context - RESIDENT_CONTEXT_HEADROOM

    @property
    def packed_kv_cache_bytes(self) -> int:
        """One QSA layer's BF16 ``[1,1,allocated_context,2*HEAD_DIM]`` packed cache per device."""

        return self.allocated_context * 2 * HEAD_DIM * 2

    @property
    def compressed_index_cache_bytes(self) -> int:
        """One QSA layer's BF16 TILE ``[1,1,indexer_window_rows,INDEX_HEAD_DIM]`` generic compressed cache."""

        return self.indexer_window_rows * INDEX_HEAD_DIM * 2

    @property
    def qsa_generic_state_bytes(self) -> int:
        """One QSA layer's generic state per device: the two caches, the 32-row staging slab and the raw-key ring."""

        return (
            self.packed_kv_cache_bytes
            + self.compressed_index_cache_bytes
            + CACHE_WRITE_ROWS * 2 * HEAD_DIM * 2
            + CACHE_WRITE_ROWS * INDEX_HEAD_DIM * 2
        )

    @property
    def rope_table_bytes(self) -> int:
        """The cos and sin tables, BF16 ``[1,1,rope_rows,ROPE_DIM]`` each, replicated per device."""

        return 2 * self.rope_rows * ROPE_DIM * 2

    @property
    def context_state_bytes_per_device(self) -> int:
        """Every byte per device that scales with the allocated context: 12 QSA generic states and the RoPE tables."""

        return QSA_LAYERS * self.qsa_generic_state_bytes + self.rope_table_bytes

    @property
    def reserve_bytes_per_device(self) -> int:
        default = Qwen38ResidentContext()
        return RESIDENT_DEFAULT_RESERVE_BYTES_PER_DEVICE + (
            self.context_state_bytes_per_device - default.context_state_bytes_per_device
        )


class Qwen38ResidentBuildFailure(RuntimeError):
    """A resident build failed after another graph already published its owner."""

    def __init__(
        self,
        operation: str,
        cause: BaseException,
        *,
        unreleased_tensor_slots: tuple[tuple[str, int, int, bool, str | None], ...],
    ) -> None:
        self.operation = operation
        self.cause = cause
        self.unreleased_tensor_slots = unreleased_tensor_slots
        self.requires_process_termination = True
        super().__init__(
            f"{operation} failed after a resident graph was published: {type(cause).__name__}: {cause}; "
            f"resident BF4 cleanup was not attempted; unreleased_tensor_slots={unreleased_tensor_slots}; "
            "retire this builder and tear down its owning process/mesh"
        )


class _Qwen38LazyTTCCL:
    """Share one model-scoped CCL manager without allocating semaphores during build."""

    def __init__(self, mesh_device) -> None:
        self._mesh_device = mesh_device
        self._manager: TT_CCL | None = None

    def _resolve(self) -> TT_CCL:
        manager = self._manager
        if manager is None:
            manager = TT_CCL(self._mesh_device)
            self._manager = manager
        return manager

    def get_num_links(self, cluster_axis=None):
        return self._resolve().get_num_links(cluster_axis)

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
        return self._resolve().get_and_cycle_barrier_semaphore_handle(cluster_axis)

    def get_and_cycle_ag_semaphore_handles(self, cluster_axis=None):
        return self._resolve().get_and_cycle_ag_semaphore_handles(cluster_axis)

    def get_and_cycle_rs_semaphore_handles(self, cluster_axis=None):
        return self._resolve().get_and_cycle_rs_semaphore_handles(cluster_axis)


def _require_lower_hex(value: str, length: int, *, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase {length}-hex, got {value!r}")


def _identity_key(value: Any) -> str:
    return _payload_key(asdict(value))


def _payload_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _topology_identity(topology) -> str:
    if topology is None:
        raise ValueError("collective_topology must be supplied from the qualified physical topology gate")
    name = getattr(topology, "name", None)
    if callable(name):
        name = name()
    if not isinstance(name, str) or not name or len(name) > 128:
        raise ValueError(f"collective topology has no bounded enum name: {name!r}")
    return name


def _normalize_cache_root(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an explicit absolute path, got {path}")
    path = path.resolve()
    if path == Path(path.anchor):
        raise ValueError(f"{label} cannot be a filesystem root")
    return path


@dataclass(frozen=True)
class Qwen38LayerBuildSpec:
    """One immutable slot in the released 48-layer backbone."""

    layer_index: int
    attention: AttentionKind
    has_ple: bool


@dataclass(frozen=True)
class Qwen38LayerObjectGraph:
    """Pure no-device description of every required layer-owned component."""

    layer_index: int
    namespace: Namespace
    attention: AttentionKind
    attention_gr_block: Literal["attn"]
    moe_namespace: Namespace
    mlp_gr_block: Literal["mlp"]
    ple_layer: int | None
    bf4_streamer_owner: Literal["target-shared-single-slot"]


@dataclass(frozen=True)
class Qwen38TargetObjectGraph:
    """No-device ordinary-target graph used before any cache/device work."""

    model_io: Literal["vocab-row-sharded-untied"]
    layers: tuple[Qwen38LayerObjectGraph, ...]
    final_mixer_namespace: Literal["backbone"]


BACKBONE_PLAN = tuple(
    Qwen38LayerBuildSpec(
        layer_index=layer_index,
        attention="qsa" if layer_index % 4 == 3 else "gdn",
        has_ple=layer_index == PLE_LAYER,
    )
    for layer_index in range(BACKBONE_LAYERS)
)

TARGET_OBJECT_GRAPH = Qwen38TargetObjectGraph(
    model_io="vocab-row-sharded-untied",
    layers=tuple(
        Qwen38LayerObjectGraph(
            layer_index=spec.layer_index,
            namespace="backbone",
            attention=spec.attention,
            attention_gr_block="attn",
            moe_namespace="backbone",
            mlp_gr_block="mlp",
            ple_layer=spec.layer_index if spec.has_ple else None,
            bf4_streamer_owner="target-shared-single-slot",
        )
        for spec in BACKBONE_PLAN
    ),
    final_mixer_namespace="backbone",
)


@dataclass(frozen=True)
class Qwen38BuildProvenance:
    """Caller-supplied source/runtime identity; no value is inferred."""

    checkpoint_revision: str
    checkpoint_index_sha256: str
    checkpoint_config_sha256: str
    checkpoint_file_manifest_sha256: str
    checkpoint_hash_manifest_sha256: str
    tt_metal_sha: str
    ttnn_runtime_sha256: str
    format_version: int = BUILDER_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.checkpoint_revision != PINNED_CHECKPOINT_REVISION:
            raise ValueError(
                f"builder requires checkpoint {PINNED_CHECKPOINT_REVISION}, got {self.checkpoint_revision}"
            )
        expected = {
            "checkpoint_index_sha256": INDEX_SHA256,
            "checkpoint_config_sha256": CONFIG_SHA256,
            "checkpoint_file_manifest_sha256": CHECKPOINT_FILE_MANIFEST_SHA256,
            "checkpoint_hash_manifest_sha256": PINNED_TENSOR_MANIFEST_SHA256,
        }
        for name, required in expected.items():
            actual = getattr(self, name)
            _require_lower_hex(actual, 64, label=name)
            if actual != required:
                raise ValueError(f"{name} must be the pinned digest {required}, got {actual}")
        _require_lower_hex(self.tt_metal_sha, 40, label="tt_metal_sha")
        _require_lower_hex(self.ttnn_runtime_sha256, 64, label="ttnn_runtime_sha256")
        if isinstance(self.format_version, bool) or self.format_version != BUILDER_FORMAT_VERSION:
            raise ValueError(f"builder provenance format must be {BUILDER_FORMAT_VERSION}, got {self.format_version}")

    @property
    def key(self) -> str:
        return _identity_key(self)


@dataclass(frozen=True)
class Qwen38CacheRoots:
    """Three disjoint caller-owned cache roots for this host/run."""

    component_weights: Path
    routed_bf4: Path
    model_io: Path

    def __post_init__(self) -> None:
        names = ("component_weights", "routed_bf4", "model_io")
        normalized = tuple(_normalize_cache_root(getattr(self, name), label=name) for name in names)
        for name, path in zip(names, normalized):
            object.__setattr__(self, name, path)
        for left_index, left in enumerate(normalized):
            for right in normalized[left_index + 1 :]:
                if left == right or left in right.parents or right in left.parents:
                    raise ValueError(
                        "component, routed-BF4, and model-I/O cache roots must be disjoint; " f"got {left} and {right}"
                    )


@dataclass(frozen=True)
class Qwen38LiveBuildIdentity:
    """Complete provenance after binding source identity to the live mesh."""

    provenance: Qwen38BuildProvenance
    mesh_shape: tuple[int, int]
    physical_ids: tuple[int, int, int, int]
    collective_topology: str
    dram_bank_ring_order: tuple[int, ...]
    ring_size: int
    expert_residency: Literal["streamed", "resident"] = "streamed"
    qsa_cache_capacity: int = MAX_CONTEXT
    decode_dram_workers_per_bank: int = 1  # the decode linears' DRAM readers per bank (decode_matmul)

    def __post_init__(self) -> None:
        validate_decode_dram_workers(self.decode_dram_workers_per_bank)
        if not isinstance(self.provenance, Qwen38BuildProvenance):
            raise TypeError("live builder identity requires validated Qwen38BuildProvenance")
        if tuple(self.mesh_shape) != MESH_SHAPE:
            raise ValueError(f"live builder identity requires mesh {MESH_SHAPE}, got {self.mesh_shape}")
        if (
            len(self.physical_ids) != TP_SIZE
            or len(set(self.physical_ids)) != TP_SIZE
            or any(
                isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0
                for device_id in self.physical_ids
            )
        ):
            raise ValueError(f"live builder identity requires four distinct physical IDs, got {self.physical_ids}")
        if self.ring_size not in (7, 8):
            raise ValueError(f"live Blackhole DRAM ring must contain seven or eight workers, got {self.ring_size}")
        if any(isinstance(bank, bool) or not isinstance(bank, int) for bank in self.dram_bank_ring_order) or sorted(
            self.dram_bank_ring_order
        ) != list(range(self.ring_size)):
            raise ValueError("live builder identity has an invalid DRAM bank ring order")
        if not self.collective_topology:
            raise ValueError("live builder identity requires an explicit collective topology")
        if self.expert_residency not in {"streamed", "resident"}:
            raise ValueError(
                f"live builder identity expert residency must be streamed or resident, got {self.expert_residency!r}"
            )
        allocated_context = validate_qsa_cache_capacity(self.qsa_cache_capacity)
        if self.expert_residency == "resident" and allocated_context not in RESIDENT_QSA_CACHE_CAPACITIES:
            raise ValueError(
                "resident live builder identity requires qsa_cache_capacity in "
                f"{RESIDENT_QSA_CACHE_CAPACITIES}, got {allocated_context}"
            )

    @property
    def key(self) -> str:
        payload = asdict(self)
        if self.decode_dram_workers_per_bank == 1:
            del payload["decode_dram_workers_per_bank"]  # the one-reader caches keep the identity they were built under
        return _payload_key(payload)


@dataclass(frozen=True)
class Qwen38TTNNTargetComponents:
    """Inspectably owned pieces used to instantiate ordinary target decode."""

    identity: Qwen38LiveBuildIdentity
    bf4_cache: Qwen38BF4Cache
    io_cache: Qwen38IOCache
    expert_streamer: Qwen38BF4Streamer
    model_io: Qwen38TTNNModelIO
    layers: tuple[Qwen38TTNNDecoderLayer, ...]
    final_mixer: Qwen38TTNNFinalMixer

    def close_resident_experts(self) -> None:
        """Close the optional static BF4 owner before mesh teardown."""

        if not isinstance(self.expert_streamer, Qwen38BF4ResidentSet):
            raise RuntimeError("target components do not own resident BF4 experts")
        self.expert_streamer.close()


@dataclass(frozen=True)
class Qwen38TTNNBuiltTarget:
    """The exact ordinary-decode model plus its inspectable component owners."""

    model: Qwen38TTNNTextModel
    components: Qwen38TTNNTargetComponents


@dataclass(frozen=True)
class Qwen38TTNNMTPComponents:
    """Exact resident pieces for the released one-layer MTP draft stack.

    This remains a component bundle rather than a speculative decoder: draft
    recurrence, fixed-five-position target verification, and transactional
    commit/rollback belong to the later MTP state owner.
    """

    identity: Qwen38LiveBuildIdentity
    input_mixer: Qwen38TTNNMTPInput
    decoder_layer: Qwen38TTNNDecoderLayer
    final_mixer: Qwen38TTNNFinalMixer

    def close_resident_experts(self) -> None:
        """Close the target-shared static BF4 owner before mesh teardown."""

        owner = self.decoder_layer.expert_streamer
        if not isinstance(owner, Qwen38BF4ResidentSet):
            raise RuntimeError("MTP components do not own resident BF4 experts")
        owner.close()


def validate_builder_static_contract() -> None:
    """No-device proof of the hard-coded target construction plan."""

    if tuple(LAYER_PATTERN) != tuple(
        "full_attention" if spec.attention == "qsa" else "linear_attention" for spec in BACKBONE_PLAN
    ):
        raise RuntimeError("builder layer plan differs from the pinned Qwen4Exp configuration")
    if tuple(spec.layer_index for spec in BACKBONE_PLAN) != tuple(range(BACKBONE_LAYERS)):
        raise RuntimeError("builder layer indices are not the exact contiguous range 0..47")
    if sum(spec.attention == "gdn" for spec in BACKBONE_PLAN) != 36:
        raise RuntimeError("builder must contain exactly 36 GDN layers")
    if sum(spec.attention == "qsa" for spec in BACKBONE_PLAN) != 12:
        raise RuntimeError("builder must contain exactly 12 QSA layers")
    if tuple(spec.layer_index for spec in BACKBONE_PLAN if spec.has_ple) != (PLE_LAYER,):
        raise RuntimeError("builder must place PLE only at zero-based checkpoint layer 1")
    graph = TARGET_OBJECT_GRAPH
    if len(graph.layers) != BACKBONE_LAYERS:
        raise RuntimeError("builder object graph must contain exactly 48 target layers")
    for spec, node in zip(BACKBONE_PLAN, graph.layers):
        expected = (
            spec.layer_index,
            "backbone",
            spec.attention,
            "attn",
            "backbone",
            "mlp",
            spec.layer_index if spec.has_ple else None,
            "target-shared-single-slot",
        )
        actual = (
            node.layer_index,
            node.namespace,
            node.attention,
            node.attention_gr_block,
            node.moe_namespace,
            node.mlp_gr_block,
            node.ple_layer,
            node.bf4_streamer_owner,
        )
        if actual != expected:
            raise RuntimeError(f"builder layer {spec.layer_index} object graph drifted: {actual} != {expected}")
    if graph.model_io != "vocab-row-sharded-untied" or graph.final_mixer_namespace != "backbone":
        raise RuntimeError("builder model-I/O or terminal-mixer graph drifted")
    if (TP_SIZE, HIDDEN_SIZE, VOCAB_SIZE, ROUTED_EXPERTS, EXPERTS_PER_DEVICE, TOP_K) != (
        4,
        2560,
        248_320,
        512,
        128,
        10,
    ):
        raise RuntimeError("builder TP4/model/MoE geometry drifted from the pinned target")


def validate_builder_constructor_contract() -> None:
    """No-device guard that every composed public API still has its exact inputs."""

    required_parameters = {
        Qwen38TTNNGDNWeights.from_checkpoint: {
            "checkpoint",
            "mesh_device",
            "mesh_contract",
            "cache_root",
            "layer_index",
            "tt_metal_sha",
        },
        Qwen38TTNNGatedResidualWeights.from_checkpoint: {
            "checkpoint",
            "placement",
            "mesh_device",
            "mesh_contract",
            "cache_root",
            "layer_index",
            "block",
            "namespace",
            "tt_metal_sha",
        },
        Qwen38TTNNQSAWeights.from_checkpoint: {
            "checkpoint",
            "placement",
            "mesh_device",
            "mesh_contract",
            "cache_root",
            "layer_index",
            "tt_metal_sha",
        },
        Qwen38TTNNQSAWeights.from_mtp_checkpoint: {
            "checkpoint",
            "placement",
            "mesh_device",
            "mesh_contract",
            "cache_root",
            "mtp_layer_index",
            "tt_metal_sha",
        },
        Qwen38TTNNPLEWeights.from_checkpoint: {
            "checkpoint",
            "mesh_device",
            "mesh_contract",
            "cache_root",
            "tt_metal_sha",
        },
        Qwen38TTNNMoEWeights.from_checkpoint: {
            "checkpoint",
            "placement",
            "mesh_device",
            "mesh_contract",
            "cache_root",
            "layer_index",
            "namespace",
            "tt_metal_sha",
        },
        Qwen38TTNNFinalMixerWeights.from_checkpoint: {
            "checkpoint",
            "placement",
            "mesh_device",
            "mesh_contract",
            "cache_root",
            "namespace",
            "tt_metal_sha",
        },
        Qwen38TTNNModelIOWeights.from_checkpoint: {
            "checkpoint",
            "placement",
            "mesh_device",
            "mesh_contract",
            "cache",
        },
        Qwen38TTNNMTPInputWeights.from_checkpoint: {
            "checkpoint",
            "placement",
            "mesh_device",
            "mesh_contract",
            "cache_root",
            "tt_metal_sha",
            "ttnn_runtime_sha256",
        },
        Qwen38TTNNDecoderLayer: {
            "mesh_contract",
            "namespace",
            "layer_index",
            "attention",
            "attention_gr",
            "mlp",
            "mlp_gr",
            "expert_streamer",
            "ple",
        },
        Qwen38TTNNTextModel: {
            "config",
            "mesh_device",
            "mesh_contract",
            "model_io",
            "layers",
            "final_mixer",
        },
    }
    for constructor, required in required_parameters.items():
        actual = set(inspect.signature(constructor).parameters)
        missing = required - actual
        if missing:
            raise RuntimeError(f"builder dependency {constructor.__qualname__} is missing parameters {missing}")


def validate_checkpoint_and_placement(checkpoint: Qwen38Checkpoint, placement: Qwen38Placement) -> None:
    """Header/config-only gate shared by live construction and static tests."""

    validate_builder_static_contract()
    if not isinstance(checkpoint, Qwen38Checkpoint):
        raise TypeError("checkpoint must be the pinned, index-validating Qwen38Checkpoint")
    if not isinstance(placement, Qwen38Placement):
        raise TypeError("placement must be Qwen38Placement")
    if checkpoint.config != placement.config:
        raise ValueError("checkpoint and numerical placement configurations differ")
    config = checkpoint.config
    exact = {
        "config_sha256": CONFIG_SHA256,
        "hidden_size": HIDDEN_SIZE,
        "vocab_size": VOCAB_SIZE,
        "num_hidden_layers": BACKBONE_LAYERS,
        "layer_types": tuple(LAYER_PATTERN),
        "num_experts": ROUTED_EXPERTS,
        "top_k": TOP_K,
        "ple_checkpoint_layer": PLE_LAYER,
        "mtp_layers": MTP_LAYERS,
        "mtp_uses_shared_embeddings": True,
    }
    for name, expected in exact.items():
        actual = getattr(config, name)
        if actual != expected:
            raise ValueError(f"pinned builder requires config {name}={expected!r}, got {actual!r}")
    if tuple(placement.mesh_shape) != MESH_SHAPE:
        raise ValueError(f"numerical placement must be {MESH_SHAPE}, got {placement.mesh_shape}")
    expected_expert_ranges = tuple((index * EXPERTS_PER_DEVICE, (index + 1) * EXPERTS_PER_DEVICE) for index in range(4))
    if tuple(placement.expert_ranges) != expected_expert_ranges:
        raise ValueError(f"routed experts must be sharded as 128/card, got {tuple(placement.expert_ranges)}")
    validate_terminal_architecture(checkpoint)


class Qwen38TTNNBuilder:
    """One-shot, provenance-bound component builder for one admitted mesh."""

    # The prefill slab's dense-linear switches (ttnn/prefill_dense), read once per builder; the default is the wide grid
    # on today's arithmetic (prefill_dense.DEFAULTS).
    prefill_dense_policy: Qwen38PrefillDensePolicy = Qwen38PrefillDensePolicy()
    # The prefill slab the target will run (enable_prefill_slab, before the target is built); None = no slab, and
    # then the switches are inert: no resident prefill weight is built and nothing is admitted.
    prefill_slab_rows: int | None = None
    # The numbers the resident prefill weights were admitted on (prefill_dense.admit_prefill_dense_dram); None when
    # the policy allocates nothing or the build runs no slab.
    prefill_dense_admission: dict | None = None
    # The vision tower's residency in this process (enable_vision, before the chain is constructed); None = no
    # image path: nothing of the tower is read, allocated or compiled.
    vision_residency: Qwen38VisionResidency | None = None

    def __init__(
        self,
        *,
        checkpoint: Qwen38Checkpoint,
        placement: Qwen38Placement,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        provenance: Qwen38BuildProvenance,
        cache_roots: Qwen38CacheRoots,
        collective_topology,
        expert_residency: Literal["streamed", "resident"] = "streamed",
        qsa_cache_capacity: int = MAX_CONTEXT,
        decode_dram_workers_per_bank: int | None = None,
        dense_weight_plan: DenseWeightPlan | None = None,
    ) -> None:
        # The decode linears' DRAM readers per bank (decode_matmul): the serving default or QWEN38_DRAM_WORKERS
        # unless the caller names the count; the GDN weights, the QSA and the LM head (backbone and MTP) run it.
        decode_dram_workers_per_bank = (
            default_decode_dram_workers()
            if decode_dram_workers_per_bank is None
            else validate_decode_dram_workers(decode_dram_workers_per_bank)
        )
        # The resident dense matmul weights' dtype per module (decode_matmul: QWEN38_DENSE_WEIGHT_DTYPE, bf16 unless set).
        if dense_weight_plan is None:
            dense_weight_plan = default_dense_weight_plan()
        if not isinstance(dense_weight_plan, DenseWeightPlan):
            raise TypeError("dense_weight_plan must be a DenseWeightPlan")
        # The prefill slab's dense-linear switches (QWEN38_PREFILL_DENSE_*, ttnn/prefill_dense); an unknown value
        # raises here.
        self.prefill_dense_policy = Qwen38PrefillDensePolicy.from_environ()
        validate_checkpoint_and_placement(checkpoint, placement)
        if not isinstance(mesh_contract, Qwen38MeshContract):
            raise TypeError("mesh_contract must be Qwen38MeshContract")
        if not isinstance(provenance, Qwen38BuildProvenance):
            raise TypeError("provenance must be validated Qwen38BuildProvenance")
        if not isinstance(cache_roots, Qwen38CacheRoots):
            raise TypeError("cache_roots must be validated, disjoint Qwen38CacheRoots")
        if expert_residency not in {"streamed", "resident"}:
            raise ValueError(f"expert_residency must be 'streamed' or 'resident', got {expert_residency!r}")
        qsa_cache_capacity = validate_qsa_cache_capacity(qsa_cache_capacity)
        if expert_residency == "resident" and qsa_cache_capacity not in RESIDENT_QSA_CACHE_CAPACITIES:
            raise ValueError(
                f"resident expert policy requires qsa_cache_capacity in {RESIDENT_QSA_CACHE_CAPACITIES}, got "
                f"{qsa_cache_capacity}; other resident configurations lack measured DRAM headroom"
            )
        if tuple(placement.physical_ids) != mesh_contract.physical_ids:
            raise ValueError(
                f"placement physical order {placement.physical_ids} differs from admitted "
                f"{mesh_contract.physical_ids}"
            )
        if provenance.checkpoint_config_sha256 != checkpoint.config.config_sha256:
            raise ValueError("provenance config digest differs from the validated checkpoint")
        topology_identity = _topology_identity(collective_topology)
        mesh_contract.validate_mesh(mesh_device)
        if mesh_device.arch() != ttnn.Arch.BLACKHOLE:
            raise ValueError("Qwen3.8 four-P150 construction requires a Blackhole mesh")
        ring_order = qualify_live_bf4_ring(mesh_device)
        # Two readers per bank need one bank -> worker assignment on every device (decode_matmul): a mesh of
        # differently harvested dies runs one reader, and the one-reader caches keep their identity.
        decode_dram_workers_per_bank, decode_dram_workers_fallback = qualify_decode_dram_workers(
            mesh_device, decode_dram_workers_per_bank
        )
        if decode_dram_workers_per_bank != 1:
            # Two readers per bank serve the GDN, QSA and LM-head projections: their weight dtype must be one the
            # two-reader form was checked bitwise against one reader with (decode_matmul.TWO_READER_QUALIFIED_DTYPES).
            unqualified = [
                module
                for module in ("gdn", "qsa", "lm_head")
                if dense_weight_plan.dtype(module) not in TWO_READER_QUALIFIED_DTYPES
            ]
            if unqualified:
                decode_dram_workers_per_bank = 1
                decode_dram_workers_fallback = (
                    "one reader per DRAM bank: two readers are qualified for "
                    f"{sorted(dense_dtype_tag(d) for d in TWO_READER_QUALIFIED_DTYPES)} dense weights, not "
                    + ", ".join(f"{module}={dense_weight_plan.tag(module)}" for module in unqualified)
                )
        live_identity = Qwen38LiveBuildIdentity(
            provenance=provenance,
            mesh_shape=tuple(mesh_contract.mesh_shape),
            physical_ids=mesh_contract.physical_ids,
            collective_topology=topology_identity,
            dram_bank_ring_order=ring_order,
            ring_size=len(ring_order),
            expert_residency=expert_residency,
            qsa_cache_capacity=qsa_cache_capacity,
            decode_dram_workers_per_bank=decode_dram_workers_per_bank,
        )

        # The component and model-I/O caches receive a root namespaced by the complete builder identity (the
        # runtime digest, the live ring, the allocated context); component-specific paths add their own layer/block
        # key.  The routed BF4 cache is keyed by its own identity only (checkpoint, converter sources, mesh, ring): the
        # expert bytes depend neither on the allocated context nor on the rest of the runtime, so one conversion serves
        # every context and survives a runtime rebuild.
        component_cache_root = cache_roots.component_weights / live_identity.key
        routed_cache_root = cache_roots.routed_bf4
        io_cache_root = cache_roots.model_io / live_identity.key
        bf4_identity = BF4CacheIdentity(
            checkpoint_revision=provenance.checkpoint_revision,
            checkpoint_config_sha256=provenance.checkpoint_config_sha256,
            checkpoint_file_manifest_sha256=provenance.checkpoint_file_manifest_sha256,
            checkpoint_hash_manifest_sha256=provenance.checkpoint_hash_manifest_sha256,
            converter_sources=bf4_converter_source_identity(),
            mesh_shape=tuple(mesh_contract.mesh_shape),
            physical_ids=mesh_contract.physical_ids,
            ring_size=len(ring_order),
            dram_bank_ring_order=ring_order,
        )
        bf4_cache = Qwen38BF4Cache(routed_cache_root, bf4_identity, mesh_contract)
        io_identity = Qwen38IOCacheIdentity(
            checkpoint_revision=provenance.checkpoint_revision,
            checkpoint_config_sha256=provenance.checkpoint_config_sha256,
            checkpoint_file_manifest_sha256=provenance.checkpoint_file_manifest_sha256,
            checkpoint_hash_manifest_sha256=provenance.checkpoint_hash_manifest_sha256,
            tt_metal_revision=provenance.tt_metal_sha,
            ttnn_runtime_sha256=provenance.ttnn_runtime_sha256,
            mesh_shape=tuple(mesh_contract.mesh_shape),
            physical_ids=mesh_contract.physical_ids,
        )

        self.checkpoint = checkpoint
        self.placement = placement
        self.mesh_device = mesh_device
        self.mesh_contract = mesh_contract
        self.tt_ccl = _Qwen38LazyTTCCL(mesh_device)
        self.provenance = provenance
        self.cache_roots = cache_roots
        self.collective_topology = collective_topology
        self.expert_residency = expert_residency
        self.qsa_cache_capacity = qsa_cache_capacity
        self.decode_dram_workers_per_bank = decode_dram_workers_per_bank
        self.decode_dram_workers_fallback = decode_dram_workers_fallback  # None, or why the mesh runs one reader
        # the two-reader programs' shared placement source ("identical", or the reference coordinate + the dies whose
        # own optimal readers differ); None with one reader
        self.decode_dram_workers_placement = (
            decode_dram_workers_placement(mesh_device) if decode_dram_workers_per_bank != 1 else None
        )
        self.dense_weight_plan = dense_weight_plan
        self.identity = live_identity
        self.component_cache_root = component_cache_root
        self.bf4_cache = bf4_cache
        self.io_cache = Qwen38IOCache(io_cache_root, io_identity, mesh_contract)
        if expert_residency == "resident":
            self.expert_streamer = Qwen38BF4ResidentSet(bf4_cache, mesh_device)
        else:
            self.expert_streamer = Qwen38BF4Streamer(bf4_cache, mesh_device)
        self._host_ple: Qwen38HostPLEEmbedding | None = None
        self._building_target = False
        self._built_target = False
        self._target_components: Qwen38TTNNTargetComponents | None = None
        self._target_components_published = False
        self._building_mtp = False
        self._built_mtp = False
        self._resident_build_failed = False
        self._resident_build_failure: tuple[str, BaseException, bool, bool, str] | None = None

    def _resident_graph_is_published(self) -> bool:
        return self._target_components_published or self._built_target or self._built_mtp

    def _assert_resident_builder_usable(self) -> None:
        if self._resident_build_failed:
            raise RuntimeError("resident builder was retired after a resident build failure")

    @property
    def resident_build_failure(self) -> tuple[str, BaseException, bool, bool, str] | None:
        """Operation, cause, teardown requirement, publication, and owner cleanup."""

        return self._resident_build_failure

    def enable_vision(
        self, dram_view, *, reserved_bytes_per_bank: int = 0, buckets=VISION_ROW_BUCKETS
    ) -> Qwen38VisionResidency:
        """Give this process the vision tower: the checkpoint's 333 tower tensors read from the host now (BF16), the
        residency decided and made in the chain's warm hook (``vision_residency.vision_warm_hook`` goes to
        ``construct_chain(warm_hook=...)``, composed with any other hook): the admission on the live allocator, the
        weights resident, one forward per row bucket compiled before the captures.  ``dram_view`` reads the mesh's
        DRAM view (the server's symmetric probe bound to its route); ``reserved_bytes_per_bank`` is what the chain
        allocates after the hook (its traces).  Called at most once per builder; the tower is never rebuilt."""

        self._assert_resident_builder_usable()
        if self.vision_residency is not None:
            raise RuntimeError("the vision tower is already enabled on this builder")
        self.vision_residency = Qwen38VisionResidency(
            mesh_device=self.mesh_device,
            state_dict=self.checkpoint.vision_state_dict(),
            dram_view=dram_view,
            reserved_bytes_per_bank=reserved_bytes_per_bank,
            buckets=tuple(buckets),
        )
        return self.vision_residency

    def _close_resident_after_build_failure(self, operation: str, primary_error: BaseException) -> None:
        owner = self.expert_streamer
        if not isinstance(owner, Qwen38BF4ResidentSet):
            return
        published = self._resident_graph_is_published()
        self._resident_build_failed = True
        # Every caught build failure follows entry into a device-building
        # operation.  Publication controls whether this method may enqueue
        # resident releases; it does not make an unpublished partial build safe
        # to reuse or exempt it from process/mesh teardown.
        cleanup_status = "not_attempted_published_graph" if published else "pending_unpublished_cleanup"
        self._resident_build_failure = (operation, primary_error, True, published, cleanup_status)
        if published:
            raise Qwen38ResidentBuildFailure(
                operation,
                primary_error,
                unreleased_tensor_slots=owner.unreleased_tensor_slots,
            ) from primary_error
        if owner.closed:
            self._resident_build_failure = (operation, primary_error, True, False, "owner_already_closed")
            return
        try:
            owner.close()
        except BaseException as cleanup_error:
            self._resident_build_failure = (operation, primary_error, True, False, "cleanup_failed_unreleased")
            raise BF4CleanupError(
                f"{operation} failed and resident BF4 cleanup also failed; "
                f"unreleased_tensor_slots={owner.unreleased_tensor_slots}",
                primary_error=primary_error,
                cleanup_errors=(cleanup_error,),
            ) from primary_error
        self._resident_build_failure = (operation, primary_error, True, False, "cleanup_enqueued_unfenced")

    @staticmethod
    def _validate_layer_request(namespace: Namespace, layer_index: int) -> None:
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise TypeError(f"layer index must be an integer, got {layer_index!r}")
        if namespace == "backbone":
            if not 0 <= layer_index < BACKBONE_LAYERS:
                raise ValueError(f"backbone layer index must be in [0,{BACKBONE_LAYERS}), got {layer_index}")
        elif namespace == "mtp":
            if layer_index != 0:
                raise ValueError(f"the pinned MTP stack has only layer 0, got {layer_index}")
        else:
            raise ValueError(f"namespace must be 'backbone' or 'mtp', got {namespace!r}")

    def _validate_bf4_record(
        self,
        record: BF4LayerRecord,
        *,
        namespace: Namespace,
        layer_index: int,
    ) -> BF4LayerRecord:
        expected_ranges = tuple(tuple(pair) for pair in self.placement.expert_ranges)
        if (record.namespace, record.layer_index) != (namespace, layer_index):
            raise RuntimeError(
                f"BF4 record identity {(record.namespace, record.layer_index)} differs from "
                f"{(namespace, layer_index)}"
            )
        if record.expert_ranges != expected_ranges:
            raise RuntimeError(f"BF4 record expert ownership {record.expert_ranges} != {expected_ranges}")
        if record.ring_size != self.identity.ring_size:
            raise RuntimeError(f"BF4 record ring size {record.ring_size} differs from live {self.identity.ring_size}")
        return record

    def stage_bf4_layer(
        self,
        *,
        namespace: Namespace,
        layer_index: int,
    ) -> BF4LayerRecord:
        """Convert/cache one layer and release both returned BF4_B tensors."""

        self._validate_layer_request(namespace, layer_index)
        tensors: tuple[Any, Any] | None = None
        try:
            tensors = self.bf4_cache.convert_and_upload(
                self.checkpoint,
                self.placement,
                self.mesh_device,
                namespace=namespace,
                layer_index=layer_index,
            )
        finally:
            if tensors is not None:
                try:
                    ttnn.deallocate(tensors[0])
                finally:
                    ttnn.deallocate(tensors[1])
        record = self.bf4_cache.verify_layer(namespace, layer_index)
        if record is None:
            raise RuntimeError(f"BF4 conversion returned without manifesting {namespace} layer {layer_index}")
        return self._validate_bf4_record(record, namespace=namespace, layer_index=layer_index)

    def stage_backbone_bf4(self, layer_indices: Sequence[int]) -> tuple[BF4LayerRecord, ...]:
        """Stage an explicit, unique subset serially; there is no implicit all."""

        indices = tuple(layer_indices)
        if not indices:
            raise ValueError("at least one explicit backbone BF4 layer index is required")
        if len(set(indices)) != len(indices):
            raise ValueError(f"duplicate backbone BF4 layer indices are not allowed: {indices}")
        for index in indices:
            self._validate_layer_request("backbone", index)
        return tuple(self.stage_bf4_layer(namespace="backbone", layer_index=index) for index in indices)

    def require_bf4_layers(
        self,
        *,
        namespace: Namespace,
        layer_indices: Sequence[int],
    ) -> tuple[BF4LayerRecord, ...]:
        """Hash-validate every named cache artifact without loading it resident."""

        indices = tuple(layer_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("BF4 requirement needs a nonempty sequence of unique layer indices")
        records: list[BF4LayerRecord] = []
        missing: list[int] = []
        for layer_index in indices:
            self._validate_layer_request(namespace, layer_index)
            record = self.bf4_cache.verify_layer(namespace, layer_index)
            if record is None:
                missing.append(layer_index)
            else:
                records.append(self._validate_bf4_record(record, namespace=namespace, layer_index=layer_index))
        if missing:
            raise RuntimeError(
                f"BF4 cache is missing {namespace} layers {tuple(missing)}; "
                "stage each one explicitly before constructing device modules"
            )
        return tuple(records)

    def require_complete_backbone_bf4(self) -> tuple[BF4LayerRecord, ...]:
        return self.require_bf4_layers(namespace="backbone", layer_indices=range(BACKBONE_LAYERS))

    def _build_gr(
        self,
        *,
        namespace: Namespace,
        layer_index: int,
        block: Literal["attn", "mlp"],
    ) -> Qwen38TTNNGatedResidual:
        weights = Qwen38TTNNGatedResidualWeights.from_checkpoint(
            self.checkpoint,
            self.placement,
            self.mesh_device,
            self.mesh_contract,
            self.component_cache_root,
            namespace=namespace,
            layer_index=layer_index,
            block=block,
            tt_metal_sha=self.provenance.tt_metal_sha,
        )
        if (weights.namespace, weights.layer_index, weights.block) != (namespace, layer_index, block):
            raise RuntimeError("constructed GR weight identity differs from its requested checkpoint slot")
        return Qwen38TTNNGatedResidual(
            self.mesh_device,
            self.mesh_contract,
            weights,
            tt_ccl=self.tt_ccl,
            collective_topology=self.collective_topology,
            prefill_dense=self._prefill_dense(),
        )

    def _prefill_dense(self) -> Qwen38TTNNPrefillDense:
        """One module's prefill dense object: the builder's policy, its resident prefill weights attached by
        :meth:`_attach_prefill_dense_weights` once every other resident allocation is done."""

        return Qwen38TTNNPrefillDense(self.prefill_dense_policy, self.mesh_device)

    def enable_prefill_slab(self, rows: int) -> None:
        """Tell the builder the target will run a prefill slab of ``rows`` (``--prefill-slab``), before the target is
        built: the slab's dense-linear policy then builds its resident prefill weights behind the DRAM admission.
        Without this call the QWEN38_PREFILL_DENSE_* switches are inert (no slab linear runs), nothing is allocated
        and nothing is admitted: a build without a slab is today's build."""

        if not is_slab_rows(rows):
            raise ValueError(f"a prefill slab takes a multiple of 128 rows in 256..4096, got {rows!r}")
        if self._target_components is not None or self._built_target:
            raise RuntimeError("the prefill slab must be enabled before the target is built")
        self.prefill_slab_rows = rows

    def _attach_prefill_dense_weights(self, layers: Sequence[Qwen38TTNNDecoderLayer]) -> None:
        """The resident prefill-only weights of the slab's dense linears (the bfloat8_b copies under
        QWEN38_PREFILL_DENSE_DTYPE=bf8, the fused [k | v] and [gate | up] siblings under
        QWEN38_PREFILL_DENSE_GRID=wide, the default), only for a build that runs a slab (enable_prefill_slab):
        nothing without one, nothing under GRID=today with DTYPE=bf16.  Built after the decode weights and the
        resident experts so the DRAM admission sees the real free bytes; refused, with the numbers, when they plus
        the context-scaled state, the slab's measured working set and the margin exceed the free DRAM
        (prefill_dense.admit_prefill_dense_dram)."""

        policy = self.prefill_dense_policy
        if self.prefill_slab_rows is None or not policy.resident_weights:
            return
        try:
            context = Qwen38ResidentContext(self.qsa_cache_capacity)
        except ValueError:
            context = Qwen38ResidentContext()
        # The fused siblings in their module's dense weight format (the same tiles the separate linears read).
        weight_dtypes = {
            "qsa": weight_dtype_name(self.dense_weight_plan.dtype("qsa")),
            "moe": weight_dtype_name(self.dense_weight_plan.dtype("shared_expert")),
        }
        self.prefill_dense_admission = admit_prefill_dense_dram(
            self.mesh_device,
            plan_bytes=policy.plan_bytes_per_device(weight_dtypes),
            context_state_bytes=context.context_state_bytes_per_device,
            slab_working_set=slab_working_set_bytes(self.prefill_slab_rows),
        )
        for layer in layers:
            attention_module = "gdn" if isinstance(layer.attention, Qwen38TTNNGDN) else "qsa"
            for module, kind, block in (
                (layer.attention, attention_module, None),
                (layer.attention_gr, "gr", "attn"),
                (layer.mlp, "moe", None),
                (layer.mlp_gr, "gr", "mlp"),
            ):
                module.prefill_dense.attach(
                    build_prefill_dense_weights(
                        policy,
                        kind,
                        self.mesh_device,
                        self.mesh_contract,
                        self.checkpoint,
                        self.placement,
                        self.component_cache_root,
                        layer_index=layer.layer_index,
                        tt_metal_sha=self.provenance.tt_metal_sha,
                        block=block,
                        weight_dtype=weight_dtypes.get(kind, "bf16"),
                    )
                )

    def _build_moe(self, *, namespace: Namespace, layer_index: int) -> Qwen38TTNNMoE:
        weights = Qwen38TTNNMoEWeights.from_checkpoint(
            self.checkpoint,
            self.placement,
            self.mesh_device,
            self.mesh_contract,
            self.component_cache_root,
            namespace=namespace,
            layer_index=layer_index,
            tt_metal_sha=self.provenance.tt_metal_sha,
            shared_dtype=self.dense_weight_plan.dtype("shared_expert"),
        )
        return Qwen38TTNNMoE(
            self.mesh_device,
            self.mesh_contract,
            weights,
            tt_ccl=self.tt_ccl,
            collective_topology=self.collective_topology,
            synchronization_policy=(
                Qwen38TTNNMoESyncPolicy.RESIDENT_ASYNC
                if self.expert_residency == "resident"
                else Qwen38TTNNMoESyncPolicy.CORRECTNESS_FENCED
            ),
            prefill_dense=self._prefill_dense(),
        )

    def _build_backbone_attention(self, spec: Qwen38LayerBuildSpec) -> Qwen38TTNNGDN | Qwen38TTNNQSA:
        if spec.attention == "gdn":
            weights = Qwen38TTNNGDNWeights.from_checkpoint(
                self.checkpoint,
                self.mesh_device,
                self.mesh_contract,
                self.component_cache_root,
                layer_index=spec.layer_index,
                tt_metal_sha=self.provenance.tt_metal_sha,
                projection_dtype=self.dense_weight_plan.dtype("gdn"),
                decode_dram_workers_per_bank=self.decode_dram_workers_per_bank,
            )
            if weights.layer_index != spec.layer_index:
                raise RuntimeError("constructed GDN weights belong to another checkpoint layer")
            return Qwen38TTNNGDN(
                self.mesh_device,
                self.mesh_contract,
                weights,
                collective_topology=self.collective_topology,
                prefill_dense=self._prefill_dense(),
            )
        weights = Qwen38TTNNQSAWeights.from_checkpoint(
            self.checkpoint,
            self.placement,
            self.mesh_device,
            self.mesh_contract,
            self.component_cache_root,
            layer_index=spec.layer_index,
            tt_metal_sha=self.provenance.tt_metal_sha,
            weight_dtype=self.dense_weight_plan.dtype("qsa"),
        )
        if (weights.source_kind, weights.layer_index) != ("backbone", spec.layer_index):
            raise RuntimeError("constructed QSA weights belong to another checkpoint slot")
        return Qwen38TTNNQSA(
            self.mesh_device,
            self.mesh_contract,
            weights,
            layer_index=spec.layer_index,
            rms_norm_eps=self.checkpoint.config.rms_norm_eps,
            max_context=self.checkpoint.config.max_position_embeddings,
            allocated_context=self.qsa_cache_capacity,
            collective_topology=self.collective_topology,
            decode_dram_workers_per_bank=self.decode_dram_workers_per_bank,
            prefill_dense=self._prefill_dense(),
        )

    def _build_ple(self) -> Qwen38TTNNPLE:
        if self._host_ple is None:
            self._host_ple = Qwen38HostPLEEmbedding(self.checkpoint)
        weights = Qwen38TTNNPLEWeights.from_checkpoint(
            self.checkpoint,
            self.mesh_device,
            self.mesh_contract,
            self.component_cache_root,
            tt_metal_sha=self.provenance.tt_metal_sha,
        )
        if weights.layer_index != PLE_LAYER:
            raise RuntimeError(f"constructed PLE weights belong to layer {weights.layer_index}, expected 1")
        return Qwen38TTNNPLE(
            self.mesh_device,
            self.mesh_contract,
            self._host_ple,
            weights,
            collective_topology=self.collective_topology,
        )

    def build_backbone_layer(self, layer_index: int) -> Qwen38TTNNDecoderLayer:
        """Build one exact resident layer after its BF4 artifact is proven."""

        self._validate_layer_request("backbone", layer_index)
        self.require_bf4_layers(namespace="backbone", layer_indices=(layer_index,))
        return self._build_backbone_layer(BACKBONE_PLAN[layer_index])

    def _build_backbone_layer(self, spec: Qwen38LayerBuildSpec) -> Qwen38TTNNDecoderLayer:
        """Build a layer whose BF4 manifest was validated by the caller."""

        layer_index = spec.layer_index
        attention = self._build_backbone_attention(spec)
        attention_gr = self._build_gr(namespace="backbone", layer_index=layer_index, block="attn")
        mlp = self._build_moe(namespace="backbone", layer_index=layer_index)
        mlp_gr = self._build_gr(namespace="backbone", layer_index=layer_index, block="mlp")
        ple = self._build_ple() if spec.has_ple else None
        layer = Qwen38TTNNDecoderLayer(
            mesh_contract=self.mesh_contract,
            namespace=Qwen38TTNNLayerNamespace.BACKBONE,
            layer_index=layer_index,
            attention=attention,
            attention_gr=attention_gr,
            mlp=mlp,
            mlp_gr=mlp_gr,
            expert_streamer=self.expert_streamer,
            ple=ple,
        )
        if layer.layer_index != layer_index or layer.expert_streamer is not self.expert_streamer:
            raise RuntimeError("constructed backbone layer lost its checkpoint or shared-streamer identity")
        return layer

    def _build_model_io(self) -> Qwen38TTNNModelIO:
        weights = Qwen38TTNNModelIOWeights.from_checkpoint(
            self.checkpoint,
            self.placement,
            self.mesh_device,
            self.mesh_contract,
            self.io_cache,
            lm_head_dtype=self.dense_weight_plan.dtype("lm_head"),
        )
        return Qwen38TTNNModelIO(
            self.mesh_device,
            self.mesh_contract,
            weights,
            tt_ccl=self.tt_ccl,
            collective_topology=self.collective_topology,
            decode_dram_workers_per_bank=self.decode_dram_workers_per_bank,
            synchronization_policy=(
                Qwen38TTNNEmbeddingSyncPolicy.RESIDENT_ASYNC
                if self.expert_residency == "resident"
                else Qwen38TTNNEmbeddingSyncPolicy.CORRECTNESS_FENCED
            ),
        )

    def _build_final_mixer(self, *, namespace: Namespace) -> Qwen38TTNNFinalMixer:
        weights = Qwen38TTNNFinalMixerWeights.from_checkpoint(
            self.checkpoint,
            self.placement,
            self.mesh_device,
            self.mesh_contract,
            self.component_cache_root,
            namespace=namespace,
            tt_metal_sha=self.provenance.tt_metal_sha,
            weight_dtype=self.dense_weight_plan.dtype("final_mixer"),
        )
        if weights.namespace != namespace:
            raise RuntimeError(f"constructed final mixer belongs to {weights.namespace}, expected {namespace}")
        return Qwen38TTNNFinalMixer(
            self.mesh_device,
            self.mesh_contract,
            weights,
            collective_topology=self.collective_topology,
        )

    def build_target_components(self) -> Qwen38TTNNTargetComponents:
        """Build target components and close resident weights on any partial failure."""

        self._assert_resident_builder_usable()
        try:
            result = self._build_target_components()
            self._target_components_published = True
            return result
        except BaseException as error:
            self._close_resident_after_build_failure("target component construction", error)
            raise

    def _build_target_components(self) -> Qwen38TTNNTargetComponents:
        """Build all 48 target layers without instantiating model state."""

        if self._target_components is not None:
            return self._target_components
        self.require_complete_backbone_bf4()
        model_io = self._build_model_io()
        layers = tuple(self._build_backbone_layer(spec) for spec in BACKBONE_PLAN)
        if len(layers) != BACKBONE_LAYERS or any(layer.layer_index != slot for slot, layer in enumerate(layers)):
            raise RuntimeError("constructed target layer tuple is not the exact ordered 48-layer stack")
        if any(layer.expert_streamer is not self.expert_streamer for layer in layers):
            raise RuntimeError("constructed target layers do not share exactly one BF4 streamer")
        if tuple(index for index, layer in enumerate(layers) if layer.ple is not None) != (PLE_LAYER,):
            raise RuntimeError("constructed target does not place PLE only at checkpoint layer 1")
        final_mixer = self._build_final_mixer(namespace="backbone")
        if self.expert_residency == "resident":
            if not isinstance(self.expert_streamer, Qwen38BF4ResidentSet):
                raise RuntimeError("resident builder did not retain the exact BF4 resident owner")
            self.expert_streamer.preload_backbone()
        self._attach_prefill_dense_weights(layers)
        result = Qwen38TTNNTargetComponents(
            identity=self.identity,
            bf4_cache=self.bf4_cache,
            io_cache=self.io_cache,
            expert_streamer=self.expert_streamer,
            model_io=model_io,
            layers=layers,
            final_mixer=final_mixer,
        )
        self._target_components = result
        return result

    def build_target(self) -> Qwen38TTNNBuiltTarget:
        """Construct the ordinary 48-layer text model exactly once."""

        self._assert_resident_builder_usable()
        if self._built_target:
            raise RuntimeError("this builder already constructed its ordinary target")
        if self._building_target:
            raise RuntimeError("ordinary target construction is already active")
        self._building_target = True
        try:
            # Keep component ownership inside this atomic public build.  Only
            # the final Qwen38TTNNBuiltTarget return publishes the graph.
            components = self._build_target_components()
            model = Qwen38TTNNTextModel(
                config=self.checkpoint.config,
                mesh_device=self.mesh_device,
                mesh_contract=self.mesh_contract,
                model_io=components.model_io,
                layers=components.layers,
                final_mixer=components.final_mixer,
            )
            result = Qwen38TTNNBuiltTarget(model=model, components=components)
            self._built_target = True
            return result
        except BaseException as error:
            self._close_resident_after_build_failure("ordinary target construction", error)
            raise
        finally:
            self._building_target = False

    def build_mtp_components(self) -> Qwen38TTNNMTPComponents:
        """Build MTP components and close the shared resident owner on failure."""

        self._assert_resident_builder_usable()
        # Pure one-shot/precondition failures must not retire or mutate a live
        # previously published graph.
        if self._built_mtp:
            raise RuntimeError("this builder already constructed its MTP component set")
        if self._building_mtp:
            raise RuntimeError("MTP component construction is already active")
        try:
            return self._build_mtp_components()
        except BaseException as error:
            self._close_resident_after_build_failure("MTP component construction", error)
            raise

    def _build_mtp_components(self) -> Qwen38TTNNMTPComponents:
        """Build the exact input mixer, one draft layer, and final mixer."""

        if self._built_mtp:
            raise RuntimeError("this builder already constructed its MTP component set")
        if self._building_mtp:
            raise RuntimeError("MTP component construction is already active")
        self._building_mtp = True
        try:
            self.require_bf4_layers(namespace="mtp", layer_indices=(0,))
            input_weights = Qwen38TTNNMTPInputWeights.from_checkpoint(
                self.checkpoint,
                self.placement,
                self.mesh_device,
                self.mesh_contract,
                self.component_cache_root,
                tt_metal_sha=self.provenance.tt_metal_sha,
                ttnn_runtime_sha256=self.provenance.ttnn_runtime_sha256,
                projection_dtype=self.dense_weight_plan.dtype("mtp"),
            )
            input_mixer = Qwen38TTNNMTPInput(
                self.mesh_device,
                self.mesh_contract,
                input_weights,
                collective_topology=self.collective_topology,
            )
            attention_weights = Qwen38TTNNQSAWeights.from_mtp_checkpoint(
                self.checkpoint,
                self.placement,
                self.mesh_device,
                self.mesh_contract,
                self.component_cache_root,
                mtp_layer_index=0,
                tt_metal_sha=self.provenance.tt_metal_sha,
                weight_dtype=self.dense_weight_plan.dtype("qsa"),
            )
            if (attention_weights.source_kind, attention_weights.layer_index) != ("mtp", 0):
                raise RuntimeError("constructed MTP QSA weights do not belong to released layer 0")
            attention = Qwen38TTNNQSA(
                self.mesh_device,
                self.mesh_contract,
                attention_weights,
                layer_index=0,
                rms_norm_eps=self.checkpoint.config.rms_norm_eps,
                max_context=self.checkpoint.config.max_position_embeddings,
                allocated_context=self.qsa_cache_capacity,
                collective_topology=self.collective_topology,
                decode_dram_workers_per_bank=self.decode_dram_workers_per_bank,
            )
            decoder_layer = Qwen38TTNNDecoderLayer(
                mesh_contract=self.mesh_contract,
                namespace=Qwen38TTNNLayerNamespace.MTP,
                layer_index=0,
                attention=attention,
                attention_gr=self._build_gr(namespace="mtp", layer_index=0, block="attn"),
                mlp=self._build_moe(namespace="mtp", layer_index=0),
                mlp_gr=self._build_gr(namespace="mtp", layer_index=0, block="mlp"),
                expert_streamer=self.expert_streamer,
                ple=None,
            )
            if self.expert_residency == "resident":
                if not isinstance(self.expert_streamer, Qwen38BF4ResidentSet):
                    raise RuntimeError("resident builder did not retain the exact BF4 resident owner")
                self.expert_streamer.preload_mtp()
            result = Qwen38TTNNMTPComponents(
                identity=self.identity,
                input_mixer=input_mixer,
                decoder_layer=decoder_layer,
                final_mixer=self._build_final_mixer(namespace="mtp"),
            )
            self._built_mtp = True
            return result
        finally:
            self._building_mtp = False


validate_builder_static_contract()
validate_builder_constructor_contract()


__all__ = [
    "BACKBONE_PLAN",
    "TARGET_OBJECT_GRAPH",
    "Qwen38BuildProvenance",
    "Qwen38CacheRoots",
    "Qwen38LayerBuildSpec",
    "Qwen38LayerObjectGraph",
    "Qwen38LiveBuildIdentity",
    "Qwen38ResidentBuildFailure",
    "Qwen38ResidentContext",
    "RESIDENT_CONTEXT_HEADROOM",
    "RESIDENT_DEFAULT_QSA_CACHE_CAPACITY",
    "RESIDENT_DEFAULT_RESERVE_BYTES_PER_DEVICE",
    "RESIDENT_MAX_QSA_CACHE_CAPACITY",
    "RESIDENT_QSA_CACHE_CAPACITIES",
    "Qwen38TTNNBuilder",
    "Qwen38TTNNBuiltTarget",
    "Qwen38TTNNMTPComponents",
    "Qwen38TTNNTargetComponents",
    "Qwen38TargetObjectGraph",
    "validate_builder_static_contract",
    "validate_builder_constructor_contract",
    "validate_checkpoint_and_placement",
]
