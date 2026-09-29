# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exact Qwen3.8-Flash-Next TTNN decoder-layer orchestration.

This module owns composition and state lifetime, not component arithmetic.  A
decode token follows the checkpoint order exactly::

    PLE residual injection (checkpoint layer 1 only)
      -> attention GR read -> GDN or QSA -> attention GR write
      -> MLP GR read -> routed + dynamically-gated shared MoE -> MLP GR write

The public residual is branch-major and hidden sharded.  Its local shape is
``[1, 4, 1, 640]`` on every coordinate of the exact 1x4 mesh; the leading one
is the *global* batch and dimension one is the four hyper-connection branches.
No path in this file turns the four coordinate-local shards into four
replicated batch elements.

PLE remains a deliberately explicit dependency.  Its 51B-parameter n-gram
table is host resident, while its convolution rows are fixed-address device
state.  The layer performs the checkpoint's residual addition immediately
before the attention GR read and owns PLE snapshot lifetime alongside the
attention state.

Speculative verification has two different state mechanisms:

* GDN recurrent and convolution buffers mutate in place and are copied into a
  preallocated snapshot.
* QSA caches are append-only while state objects are immutable metadata views;
  rollback restores the prior view and deterministically overwrites rejected
  cache bytes on replay.

The QSA module owns its transient-tensor aliasing and therefore supplies the
``restore_state`` and ``commit_state`` operations used here.  Guessing which
view owns reused complete-block indices would make rejection cleanup unsafe.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.bf4 import Qwen38BF4Streamer
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    CHUNK_ROW_COUNTS,
    CHUNK_ROWS,
    LONG_CHUNK_ROWS,
    MESH_SHAPE,
    Qwen38MeshContract,
    TensorPlacement,
    is_slab_rows,
    require_lane_count,
    tensor_metadata,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.gdn import (
    CONV_KERNEL_SIZE,
    Qwen38TTNNGDN,
    Qwen38TTNNGDNRowsConstants,
    Qwen38TTNNGDNRowsResult,
    Qwen38TTNNGDNRowsState,
    Qwen38TTNNGDNSnapshot,
    Qwen38TTNNGDNState,
    Qwen38TTNNRowsSelectors,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.gr import (
    Qwen38TTNNGatedResidual,
    block_rows_shape,
    residual_rows_shape,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.moe import SUPPORTED_ROWS, Qwen38TTNNMoE, Qwen38TTNNRouting
from models.demos.blackhole.qwen38_flash_next.ttnn.ple import (
    Qwen38TTNNPLE,
    Qwen38TTNNPLELanesPreparedInput,
    Qwen38TTNNPLELanesState,
    Qwen38TTNNPLEPreparedInput,
    Qwen38TTNNPLEResult,
    Qwen38TTNNPLERowsPreparedInput,
    Qwen38TTNNPLERowsState,
    Qwen38TTNNPLESnapshot,
    Qwen38TTNNPLEState,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38TTNNQSA, Qwen38TTNNQSASelection, Qwen38TTNNQSAState

BACKBONE_LAYERS = 48
MTP_LAYERS = 1
HIDDEN_SIZE = 2560
TP_SIZE = 4
LOCAL_HIDDEN_SIZE = HIDDEN_SIZE // TP_SIZE
RESIDUAL_BRANCHES = 4
RESIDUAL_LOCAL_SHAPE = (1, RESIDUAL_BRANCHES, 1, LOCAL_HIDDEN_SIZE)
BLOCK_LOCAL_SHAPE = (1, 1, 1, LOCAL_HIDDEN_SIZE)
# The prefill chunk's 32-row forms of the two local shapes (branch-major residual rows, block rows).
RESIDUAL_ROWS_LOCAL_SHAPE = (1, RESIDUAL_BRANCHES, CHUNK_ROWS, LOCAL_HIDDEN_SIZE)
BLOCK_ROWS_LOCAL_SHAPE = (1, 1, CHUNK_ROWS, LOCAL_HIDDEN_SIZE)
PLE_CHECKPOINT_LAYER = 1  # zero based; the official documentation calls this layer 2
VOCAB_SIZE = 248320


class Qwen38TTNNLayerCleanupError(RuntimeError):
    """All semantic state work finished, but one or more owned releases failed."""

    def __init__(self, label: str, errors: list[BaseException], *, primary: BaseException | None = None) -> None:
        self.label = label
        self.errors = tuple(errors)
        self.primary = primary
        detail = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        super().__init__(f"{label} cleanup failed for {len(errors)} resource(s): {detail}")


def _run_cleanup_actions(
    label: str,
    actions: list[tuple[str, Any]],
    *,
    primary: BaseException | None = None,
) -> None:
    """Attempt every independent release and report all failures together."""

    errors = []
    for resource, action in actions:
        try:
            action()
        except BaseException as error:
            wrapped = RuntimeError(f"{resource}: {type(error).__name__}: {error}")
            wrapped.__cause__ = error
            errors.append(wrapped)
    if errors:
        raise Qwen38TTNNLayerCleanupError(label, errors, primary=primary) from primary


class Qwen38TTNNLayerNamespace(str, Enum):
    """Checkpoint namespaces accepted by the device layer.

    ``BACKBONE`` is the target model used for verification.  ``MTP`` is the
    released one-layer draft head.  These values intentionally match the BF4,
    GR, and MoE cache namespaces.
    """

    BACKBONE = "backbone"
    MTP = "mtp"


class Qwen38TTNNLayerType(str, Enum):
    GDN = "linear_attention"
    QSA = "full_attention"


@dataclass(frozen=True)
class Qwen38TTNNDecoderLayerState:
    """All mutable/cache state for one decoder layer and one global sequence."""

    namespace: Qwen38TTNNLayerNamespace
    layer_index: int
    position: int
    attention: Qwen38TTNNGDNState | Qwen38TTNNQSAState
    ple: Qwen38TTNNPLEState | None


@dataclass(frozen=True)
class Qwen38TTNNDecoderLayerGenericState:
    """Fixed-address state of one layer for the position-generic decode body.

    No host position: the body derives it on device.  GDN and PLE buffers are
    the same in-place tensors as :class:`Qwen38TTNNDecoderLayerState` holds;
    the QSA state is the QSA module's generic (in-place, fixed-shape) state.
    """

    namespace: Qwen38TTNNLayerNamespace
    layer_index: int
    attention: Qwen38TTNNGDNState | Any
    ple: Qwen38TTNNPLEState | None


@dataclass(frozen=True)
class Qwen38TTNNDecoderLayerChunkState:
    """Per-layer buffers of the prefill chunk body, beside the generic state (all fixed addresses).

    ``attention`` is the GDN rows state (the FIR history carried between chunks, the chunk's kept q|k|v rows,
    the chunk kernel's inputs, the output rows) or the QSA chunk state (the kept slab and raw keys of the
    hand-off); ``ple`` the PLE rows state on the PLE layer; ``moe`` a rows-``rows`` instance borrowing the
    layer's router and shared weights (its own [10,32,2560] combine buffer at 32 rows; the 128-row instances
    share one).  The 128-row state shares the GDN and PLE histories with the 32-row state it was allocated
    beside, so chunks of either row count carry one history.
    """

    namespace: Qwen38TTNNLayerNamespace
    layer_index: int
    attention: Qwen38TTNNGDNRowsState | Any
    ple: Qwen38TTNNPLERowsState | None
    moe: Qwen38TTNNMoE
    rows: int = CHUNK_ROWS


@dataclass(frozen=True)
class Qwen38TTNNDecoderLayerLaneState:
    """Fixed-address state of one layer for B batched decode lanes (lane u = row / batch index u).

    ``attention`` is the B-lane GDN state (recurrent ``[B,12,128,128]``, ring slots ``[1,1,B,2560]``, one shared
    phase) or the QSA lane state; ``ple`` the PLE lanes state on the PLE layer; ``moe`` a rows-B instance over the
    layer's router and shared weights (its own ``[10,B,2560]`` combine buffer); ``attention_rows`` the persistent
    ``[1,1,B,640]`` hidden-sharded rows a QSA layer's 32-row output is sliced into (None on GDN layers and at B = 32).
    """

    namespace: Qwen38TTNNLayerNamespace
    layer_index: int
    lanes: int
    attention: Qwen38TTNNGDNState | Any
    ple: Qwen38TTNNPLELanesState | None
    moe: Qwen38TTNNMoE
    attention_rows: Any | None


@dataclass
class Qwen38TTNNDecoderLayerSnapshot:
    """One-shot transaction snapshot used by speculative verification."""

    namespace: Qwen38TTNNLayerNamespace
    layer_index: int
    position: int
    source_attention: Qwen38TTNNGDNState | Qwen38TTNNQSAState
    attention: Qwen38TTNNGDNSnapshot | Qwen38TTNNQSAState | None
    source_ple: Qwen38TTNNPLEState | None
    ple: Qwen38TTNNPLESnapshot | None
    active: bool = True


@dataclass(frozen=True)
class Qwen38TTNNDecoderLayerAux:
    routing: Qwen38TTNNRouting | None
    selection: Qwen38TTNNQSASelection | None
    reused_qsa_selection: bool


@dataclass(frozen=True)
class Qwen38TTNNDecoderLayerResult:
    residual_sharded: Any
    state: Qwen38TTNNDecoderLayerState
    aux: Qwen38TTNNDecoderLayerAux


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def _tensor_key(tensor) -> tuple[str, int]:
    tensor_id = getattr(tensor, "tensor_id", None)
    if callable(tensor_id):
        tensor_id = tensor_id()
    return ("ttnn", int(tensor_id)) if tensor_id is not None else ("python", id(tensor))


def _deallocate_unique(*tensors) -> None:
    seen: set[tuple[str, int]] = set()
    for tensor in tensors:
        if tensor is None:
            continue
        key = _tensor_key(tensor)
        if key in seen:
            continue
        seen.add(key)
        ttnn.deallocate(tensor)


def _expected_layer_type(namespace: Qwen38TTNNLayerNamespace, layer_index: int) -> Qwen38TTNNLayerType:
    if namespace is Qwen38TTNNLayerNamespace.BACKBONE:
        if not 0 <= layer_index < BACKBONE_LAYERS:
            raise ValueError(f"backbone layer index must be in [0,{BACKBONE_LAYERS}), got {layer_index}")
        return Qwen38TTNNLayerType.QSA if layer_index % 4 == 3 else Qwen38TTNNLayerType.GDN
    if namespace is Qwen38TTNNLayerNamespace.MTP:
        if layer_index != 0:
            raise ValueError(f"the pinned MTP stack has only layer 0, got {layer_index}")
        return Qwen38TTNNLayerType.QSA
    raise ValueError(f"unsupported layer namespace {namespace!r}")


def _normalize_namespace(value: Qwen38TTNNLayerNamespace | str) -> Qwen38TTNNLayerNamespace:
    try:
        return Qwen38TTNNLayerNamespace(value)
    except ValueError as error:
        choices = tuple(item.value for item in Qwen38TTNNLayerNamespace)
        raise ValueError(f"layer namespace must be one of {choices}, got {value!r}") from error


class Qwen38TTNNDecoderLayer:
    """One exact true-global-B1 TTNN target or MTP decoder layer.

    Component weights are constructed outside this class so their conversion
    and cache provenance remains independently inspectable.  ``expert_streamer``
    is mandatory: each invocation loads one exact BF4_B routed-expert layer and
    releases that slot after the MoE operation.  There is no replicated-expert
    fallback.
    """

    _fused_apply_ple = None  # QWEN38_FUSED=ple binds ttnn/fused/ple's layer glue per instance; the body is the chain

    def __init__(
        self,
        *,
        mesh_contract: Qwen38MeshContract,
        namespace: Qwen38TTNNLayerNamespace | str,
        layer_index: int,
        attention: Qwen38TTNNGDN | Qwen38TTNNQSA,
        attention_gr: Qwen38TTNNGatedResidual,
        mlp: Qwen38TTNNMoE,
        mlp_gr: Qwen38TTNNGatedResidual,
        expert_streamer: Qwen38BF4Streamer,
        ple: Qwen38TTNNPLE | None = None,
    ) -> None:
        namespace = _normalize_namespace(namespace)
        expected_type = _expected_layer_type(namespace, layer_index)
        if expected_type is Qwen38TTNNLayerType.GDN:
            if not isinstance(attention, Qwen38TTNNGDN):
                raise TypeError(f"{namespace.value} layer {layer_index} requires GDN attention")
        elif not isinstance(attention, Qwen38TTNNQSA):
            raise TypeError(f"{namespace.value} layer {layer_index} requires QSA attention")

        expected_ple = namespace is Qwen38TTNNLayerNamespace.BACKBONE and layer_index == PLE_CHECKPOINT_LAYER
        if expected_ple and ple is None:
            raise ValueError("checkpoint layer 1 (one-indexed layer 2) requires the exact host-resident PLE callback")
        if not expected_ple and ple is not None:
            raise ValueError(f"PLE is not present in {namespace.value} layer {layer_index}")
        if ple is not None and int(ple.weights.layer_index) != PLE_CHECKPOINT_LAYER:
            raise ValueError(
                f"PLE callback belongs to layer {ple.weights.layer_index}, expected {PLE_CHECKPOINT_LAYER}"
            )

        component_contracts = {
            "attention": attention.mesh_contract,
            "attention_gr": attention_gr.mesh_contract,
            "mlp": mlp.mesh_contract,
            "mlp_gr": mlp_gr.mesh_contract,
            "expert_streamer": expert_streamer.cache.mesh_contract,
        }
        if ple is not None:
            component_contracts["ple"] = ple.mesh_contract
        for name, contract in component_contracts.items():
            if contract != mesh_contract:
                raise ValueError(f"{name} belongs to a different physical mesh contract")

        expected_namespace = namespace.value
        for name, module, block in (
            ("attention_gr", attention_gr, "attn"),
            ("mlp_gr", mlp_gr, "mlp"),
        ):
            weights = module.weights
            if weights.layer_index != layer_index or weights.block != block or weights.namespace != expected_namespace:
                raise ValueError(
                    f"{name} identity {(weights.namespace, weights.layer_index, weights.block)} does not match "
                    f"{(expected_namespace, layer_index, block)}"
                )
        if isinstance(attention, Qwen38TTNNGDN) and attention.weights.layer_index != layer_index:
            raise ValueError(f"GDN weights belong to layer {attention.weights.layer_index}, expected {layer_index}")
        if isinstance(attention, Qwen38TTNNQSA):
            if attention.layer_index != layer_index:
                raise ValueError(f"QSA module belongs to layer {attention.layer_index}, expected {layer_index}")
            if attention.weights.source_kind != expected_namespace:
                raise ValueError(
                    f"QSA weights belong to {attention.weights.source_kind!r}, expected {expected_namespace!r}"
                )

        self.mesh_contract = mesh_contract
        self.namespace = namespace
        self.layer_index = layer_index
        self.layer_type = expected_type
        self.attention = attention
        self.attention_gr = attention_gr
        self.mlp = mlp
        self.mlp_gr = mlp_gr
        self.expert_streamer = expert_streamer
        self.ple = ple
        # QWEN38_FUSED=ple: the fused PLE body with the layer's output permute + residual add folded into its conv program.
        from models.demos.blackhole.qwen38_flash_next.ttnn import fused as fused_kernels

        if ple is not None and fused_kernels.enabled("ple"):
            from models.demos.blackhole.qwen38_flash_next.ttnn.fused import ple as fused_ple

            self._fused_apply_ple = functools.partial(fused_ple.ple_layer_fused, self)

    def _validate_residual(self, residual, *, label: str) -> None:
        if (
            _shape(residual) != RESIDUAL_LOCAL_SHAPE
            or residual.dtype != ttnn.bfloat16
            or residual.layout != ttnn.TILE_LAYOUT
        ):
            raise ValueError(
                f"{label} must be global-B1 branch-major BF16 TILE {list(RESIDUAL_LOCAL_SHAPE)}, "
                f"got {tensor_metadata(residual)}"
            )
        self.mesh_contract.validate_tensor(residual, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)

    def _validate_block(self, hidden, *, label: str) -> None:
        if _shape(hidden) != BLOCK_LOCAL_SHAPE or hidden.dtype != ttnn.bfloat16 or hidden.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError(f"{label} must be BF16 TILE {list(BLOCK_LOCAL_SHAPE)}, got {tensor_metadata(hidden)}")
        self.mesh_contract.validate_tensor(hidden, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)

    def _validate_state(self, state: Qwen38TTNNDecoderLayerState) -> None:
        if not isinstance(state, Qwen38TTNNDecoderLayerState):
            raise TypeError(f"decoder layer requires Qwen38TTNNDecoderLayerState, got {type(state).__name__}")
        if state.namespace is not self.namespace or state.layer_index != self.layer_index:
            raise ValueError(
                f"state identity {(state.namespace, state.layer_index)} does not match "
                f"{(self.namespace.value, self.layer_index)}"
            )
        if state.position < 0:
            raise ValueError(f"layer position must be nonnegative, got {state.position}")
        if self.layer_type is Qwen38TTNNLayerType.GDN:
            if not isinstance(state.attention, Qwen38TTNNGDNState):
                raise TypeError("GDN layer received non-GDN attention state")
            if state.attention.layer_index != self.layer_index:
                raise ValueError("GDN state belongs to another layer")
            self.attention._validate_state(state.attention)
        else:
            if not isinstance(state.attention, Qwen38TTNNQSAState):
                raise TypeError("QSA layer received non-QSA attention state")
            if state.attention.layer_index != self.layer_index:
                raise ValueError("QSA state belongs to another layer")
            if state.attention.next_position != state.position:
                raise ValueError(
                    f"QSA metadata position {state.attention.next_position} does not match layer {state.position}"
                )
            self.attention._validate_state(state.attention)
        self._validate_ple_state(state.ple)

    def _validate_ple_state(self, ple_state: Qwen38TTNNPLEState | None) -> None:
        if self.ple is None:
            if ple_state is not None:
                raise ValueError("PLE state is present outside checkpoint layer 1")
        elif not isinstance(ple_state, Qwen38TTNNPLEState):
            raise TypeError("checkpoint layer 1 requires Qwen38TTNNPLEState")
        elif ple_state.mesh_contract != self.mesh_contract:
            raise ValueError("PLE state belongs to a different physical mesh contract")
        else:
            ple_state.validate()
        if (ple_state is not None) != (self.ple is not None):
            raise ValueError("PLE state presence does not match this layer's checkpoint placement")

    def _validate_generic_state(self, state: Qwen38TTNNDecoderLayerGenericState) -> None:
        if not isinstance(state, Qwen38TTNNDecoderLayerGenericState):
            raise TypeError(f"decoder layer requires Qwen38TTNNDecoderLayerGenericState, got {type(state).__name__}")
        if state.namespace is not self.namespace or state.layer_index != self.layer_index:
            raise ValueError(
                f"generic state identity {(state.namespace, state.layer_index)} does not match "
                f"{(self.namespace.value, self.layer_index)}"
            )
        if self.layer_type is Qwen38TTNNLayerType.GDN:
            if not isinstance(state.attention, Qwen38TTNNGDNState):
                raise TypeError("GDN layer received non-GDN generic attention state")
            self.attention._validate_state(state.attention)
        elif not isinstance(state.attention, qsa_module.Qwen38TTNNQSAGenericState):
            raise TypeError("QSA layer received non-QSA generic attention state")
        if state.attention.layer_index != self.layer_index:
            raise ValueError("generic attention state belongs to another layer")
        self._validate_ple_state(state.ple)

    def _validate_snapshot(self, snapshot: Qwen38TTNNDecoderLayerSnapshot) -> None:
        if not isinstance(snapshot, Qwen38TTNNDecoderLayerSnapshot):
            raise TypeError(f"decoder layer requires Qwen38TTNNDecoderLayerSnapshot, got {type(snapshot).__name__}")
        if not snapshot.active:
            raise RuntimeError("decoder-layer snapshot was already restored or committed")
        if snapshot.namespace is not self.namespace or snapshot.layer_index != self.layer_index:
            raise ValueError("decoder-layer snapshot belongs to another namespace or layer")
        if snapshot.position < 0:
            raise ValueError(f"decoder-layer snapshot position must be nonnegative, got {snapshot.position}")

        if isinstance(self.attention, Qwen38TTNNGDN):
            if not isinstance(snapshot.source_attention, Qwen38TTNNGDNState):
                raise TypeError("GDN snapshot source must be Qwen38TTNNGDNState")
            if not isinstance(snapshot.attention, Qwen38TTNNGDNSnapshot):
                raise TypeError("GDN layer snapshot must contain Qwen38TTNNGDNSnapshot")
            self.attention._validate_state(snapshot.source_attention)
            snapshot.attention.validate(self.mesh_contract)
        else:
            if not isinstance(snapshot.source_attention, Qwen38TTNNQSAState):
                raise TypeError("QSA snapshot source must be Qwen38TTNNQSAState")
            if not isinstance(snapshot.attention, Qwen38TTNNQSAState):
                raise TypeError("QSA layer snapshot must contain Qwen38TTNNQSAState")
            if snapshot.source_attention.next_position != snapshot.position:
                raise ValueError("QSA snapshot source position does not match decoder-layer snapshot")
            self.attention._validate_state(snapshot.source_attention)

        if self.ple is None:
            if snapshot.source_ple is not None or snapshot.ple is not None:
                raise ValueError("PLE snapshot state is present outside checkpoint layer 1")
        else:
            if not isinstance(snapshot.source_ple, Qwen38TTNNPLEState):
                raise TypeError("PLE snapshot source must be Qwen38TTNNPLEState")
            if not isinstance(snapshot.ple, Qwen38TTNNPLESnapshot):
                raise TypeError("checkpoint layer 1 snapshot must contain Qwen38TTNNPLESnapshot")
            if snapshot.source_ple.mesh_contract != self.mesh_contract:
                raise ValueError("PLE snapshot source belongs to a different physical mesh contract")
            snapshot.source_ple.validate()
            snapshot.ple.validate(self.mesh_contract)

    def validate_state(self, state: Qwen38TTNNDecoderLayerState) -> None:
        """Pure public state preflight used by the 48-layer owner."""

        self._validate_state(state)

    def validate_snapshot(self, snapshot: Qwen38TTNNDecoderLayerSnapshot) -> None:
        """Pure public snapshot identity/liveness preflight."""

        self._validate_snapshot(snapshot)

    def validate_generic_state(self, state: Qwen38TTNNDecoderLayerGenericState) -> None:
        """Pure public generic-state preflight used by the 48-layer owner."""

        self._validate_generic_state(state)

    def _validate_transaction_pair(
        self,
        current: Qwen38TTNNDecoderLayerState,
        snapshot: Qwen38TTNNDecoderLayerSnapshot,
        *,
        operation: str,
    ) -> None:
        """Validate a complete pair without copying, consuming, or releasing."""

        if operation not in {"restore", "commit"}:
            raise ValueError(f"unsupported decoder-layer transaction operation {operation!r}")
        self._validate_state(current)
        self._validate_snapshot(snapshot)
        if current.position < snapshot.position:
            raise ValueError(f"cannot {operation} a future decoder-layer snapshot")
        if current.ple is not snapshot.source_ple:
            raise ValueError(f"PLE {operation} target is not the snapshotted fixed-address state")
        if snapshot.ple is not None:
            if not snapshot.ple.captured:
                raise RuntimeError(f"cannot {operation} an uncaptured PLE snapshot")

        if isinstance(self.attention, Qwen38TTNNGDN):
            if current.attention is not snapshot.source_attention:
                raise ValueError(f"GDN {operation} target is not the snapshotted fixed-address state")
            if snapshot.attention.layer_index != self.layer_index:
                raise ValueError("GDN snapshot belongs to another layer")
            if not snapshot.attention.captured:
                raise RuntimeError(f"cannot {operation} an uncaptured GDN snapshot")
            return

        if snapshot.attention is not snapshot.source_attention:
            raise ValueError("QSA checkpoint is not the immutable source view recorded by the layer snapshot")
        if snapshot.attention.layer_index != self.layer_index or snapshot.attention.next_position != snapshot.position:
            raise ValueError("QSA checkpoint layer or position does not match the decoder-layer snapshot")
        # This reads only ownership metadata.  It must run for all 48 layers
        # before any model-level restore/commit consumes its first QSA view.
        self.attention._validate_state(snapshot.attention)
        self.attention._validate_transaction_pair(snapshot.attention, current.attention)

    def validate_restore_pair(
        self,
        current: Qwen38TTNNDecoderLayerState,
        snapshot: Qwen38TTNNDecoderLayerSnapshot,
    ) -> None:
        self._validate_transaction_pair(current, snapshot, operation="restore")

    def validate_commit_pair(
        self,
        current: Qwen38TTNNDecoderLayerState,
        snapshot: Qwen38TTNNDecoderLayerSnapshot,
    ) -> None:
        self._validate_transaction_pair(current, snapshot, operation="commit")

    @staticmethod
    def _release_attention_state(attention_module, attention_state) -> None:
        if isinstance(attention_module, Qwen38TTNNGDN):
            attention_state.deallocate()
        else:
            attention_module.release_state(attention_state)

    @staticmethod
    def _cleanup_snapshot_resources(
        snapshot: Qwen38TTNNDecoderLayerSnapshot,
        *,
        primary: BaseException | None = None,
    ) -> None:
        """Release every independent copy after semantic consumption.

        ``snapshot.active`` must already be false.  Failed resources remain
        referenced for diagnostics, but the snapshot can never be retried and
        therefore cannot double-free resources which were already released.
        """

        actions = []
        attention_snapshot = snapshot.attention
        ple_snapshot = snapshot.ple
        if isinstance(attention_snapshot, Qwen38TTNNGDNSnapshot):

            def release_attention() -> None:
                attention_snapshot.deallocate()
                snapshot.attention = None

            actions.append(("GDN snapshot", release_attention))
        if ple_snapshot is not None:

            def release_ple() -> None:
                ple_snapshot.deallocate()
                snapshot.ple = None

            actions.append(("PLE snapshot", release_ple))
        _run_cleanup_actions("decoder-layer snapshot", actions, primary=primary)

    def allocate_state(self) -> Qwen38TTNNDecoderLayerState:
        attention = self.attention.allocate_state()
        ple = None
        try:
            ple = None if self.ple is None else self.ple.allocate_state()
            result = Qwen38TTNNDecoderLayerState(self.namespace, self.layer_index, 0, attention, ple)
            self._validate_state(result)
            return result
        except BaseException as error:
            actions = []
            if ple is not None:
                actions.append(("PLE state", ple.deallocate))
            # Release in reverse allocation order.  The actions are independent,
            # so a failed PLE release cannot strand the attention state silently.
            actions.append(("attention state", lambda: self._release_attention_state(self.attention, attention)))
            _run_cleanup_actions(
                "decoder-layer state allocation",
                actions,
                primary=error,
            )
            raise

    def reset_state(self, state: Qwen38TTNNDecoderLayerState) -> Qwen38TTNNDecoderLayerState:
        """Reset one layer to position zero without retaining rejected state."""

        self._validate_state(state)
        if isinstance(self.attention, Qwen38TTNNGDN):
            state.attention.reset_inplace()
            attention = state.attention
        else:
            attention = self.attention.reset_state(state.attention)
        if state.ple is not None:
            state.ple.reset_inplace()
        ple = state.ple
        result = Qwen38TTNNDecoderLayerState(self.namespace, self.layer_index, 0, attention, ple)
        self._validate_state(result)
        return result

    def release_state(self, state: Qwen38TTNNDecoderLayerState) -> None:
        """Release a final state after all snapshots/views for it are dead."""

        self._validate_state(state)
        actions = [
            (
                "attention state",
                lambda: self._release_attention_state(self.attention, state.attention),
            )
        ]
        if state.ple is not None:
            actions.append(("PLE state", state.ple.deallocate))
        _run_cleanup_actions("decoder-layer state", actions)

    def allocate_generic_state(self) -> Qwen38TTNNDecoderLayerGenericState:
        """Allocate the fixed-address state of the position-generic body."""

        if isinstance(self.attention, Qwen38TTNNGDN):
            attention = self.attention.allocate_state()
        else:
            attention = self.attention.allocate_generic_state()
        ple = None
        try:
            ple = None if self.ple is None else self.ple.allocate_state()
            result = Qwen38TTNNDecoderLayerGenericState(self.namespace, self.layer_index, attention, ple)
            self._validate_generic_state(result)
            return result
        except BaseException as error:
            actions = []
            if ple is not None:
                actions.append(("PLE state", ple.deallocate))
            actions.append(("attention state", lambda: self._release_generic_attention_state(attention)))
            _run_cleanup_actions("decoder-layer generic state allocation", actions, primary=error)
            raise

    def _release_generic_attention_state(self, attention_state) -> None:
        if isinstance(self.attention, Qwen38TTNNGDN):
            attention_state.deallocate()
        else:
            self.attention.release_generic_state(attention_state)

    def reset_generic_state_inplace(self, state: Qwen38TTNNDecoderLayerGenericState) -> None:
        """Return every buffer to its position-zero contents without changing an address."""

        self._validate_generic_state(state)
        if isinstance(self.attention, Qwen38TTNNGDN):
            state.attention.reset_inplace()
        else:
            self.attention.reset_generic_state_inplace(state.attention)
        if state.ple is not None:
            state.ple.reset_inplace()

    def release_generic_state(self, state: Qwen38TTNNDecoderLayerGenericState) -> None:
        self._validate_generic_state(state)
        actions = [("attention state", lambda: self._release_generic_attention_state(state.attention))]
        if state.ple is not None:
            actions.append(("PLE state", state.ple.deallocate))
        _run_cleanup_actions("decoder-layer generic state", actions)

    def snapshot_state(self, state: Qwen38TTNNDecoderLayerState) -> Qwen38TTNNDecoderLayerSnapshot:
        """Capture a one-shot speculative transaction checkpoint."""

        self._validate_state(state)
        attention_snapshot = None
        try:
            if isinstance(self.attention, Qwen38TTNNGDN):
                attention_snapshot = state.attention.allocate_snapshot()
                state.attention.capture_into(attention_snapshot)
            else:
                checkpoint = getattr(self.attention, "checkpoint_state", None)
                if not callable(checkpoint):
                    raise RuntimeError("QSA checkpoint_state ownership operation is unavailable")
                attention_snapshot = checkpoint(state.attention)
        except BaseException as error:
            if isinstance(attention_snapshot, Qwen38TTNNGDNSnapshot):
                _run_cleanup_actions(
                    "decoder-layer attention snapshot creation",
                    [("GDN snapshot", attention_snapshot.deallocate)],
                    primary=error,
                )
            raise
        ple_snapshot = None
        try:
            if state.ple is not None:
                ple_snapshot = state.ple.allocate_snapshot()
                state.ple.capture_into(ple_snapshot)
        except BaseException as error:
            actions = []
            if ple_snapshot is not None:
                actions.append(("PLE snapshot", ple_snapshot.deallocate))
            if isinstance(attention_snapshot, Qwen38TTNNGDNSnapshot):
                actions.append(("GDN snapshot", attention_snapshot.deallocate))
            else:
                # ``checkpoint_state`` protects the existing immutable view.
                # With no branch view yet, restoring the checkpoint to itself
                # is the QSA-owned operation that releases that protection.
                actions.append(
                    (
                        "QSA checkpoint protection",
                        lambda: self.attention.restore_state(state.attention, attention_snapshot),
                    )
                )
            _run_cleanup_actions("decoder-layer snapshot creation", actions, primary=error)
            raise
        return Qwen38TTNNDecoderLayerSnapshot(
            namespace=self.namespace,
            layer_index=self.layer_index,
            position=state.position,
            source_attention=state.attention,
            attention=attention_snapshot,
            source_ple=state.ple,
            ple=ple_snapshot,
        )

    def restore_state(
        self,
        current: Qwen38TTNNDecoderLayerState,
        snapshot: Qwen38TTNNDecoderLayerSnapshot,
    ) -> Qwen38TTNNDecoderLayerState:
        """Reject a branch, restore the snapshot, and consume it."""

        self.validate_restore_pair(current, snapshot)

        if isinstance(self.attention, Qwen38TTNNGDN):
            # Keep the copy alive until PLE restore also succeeds.  A host-side
            # PLE failure can then be retried from the same active transaction.
            current.attention.restore_from(snapshot.attention)
            attention = current.attention
            if current.ple is not None:
                current.ple.restore_from(snapshot.ple)
        else:
            restore = getattr(self.attention, "restore_state", None)
            if not callable(restore):
                raise RuntimeError("QSA restore_state ownership operation is unavailable")
            # Restore fixed-address PLE bytes before consuming the QSA branch
            # view.  If the copy fails, the QSA checkpoint remains protected
            # and the complete transaction can be retried.
            if current.ple is not None:
                current.ple.restore_from(snapshot.ple)
            attention = restore(current.attention, snapshot.attention)
        ple = current.ple
        # Semantic consumption is final even if release later reports an
        # error.  Mark inactive first so no retry can double-consume QSA views.
        snapshot.active = False
        result = Qwen38TTNNDecoderLayerState(
            self.namespace,
            self.layer_index,
            snapshot.position,
            attention,
            ple,
        )
        try:
            self._validate_state(result)
        except BaseException as error:
            self._cleanup_snapshot_resources(snapshot, primary=error)
            raise
        self._cleanup_snapshot_resources(snapshot)
        return result

    def commit_state(
        self,
        current: Qwen38TTNNDecoderLayerState,
        snapshot: Qwen38TTNNDecoderLayerSnapshot,
    ) -> Qwen38TTNNDecoderLayerState:
        """Accept a branch, retain its state, and consume the snapshot."""

        self.validate_commit_pair(current, snapshot)
        if isinstance(self.attention, Qwen38TTNNGDN):
            attention = current.attention
        else:
            commit = getattr(self.attention, "commit_state", None)
            if not callable(commit):
                raise RuntimeError("QSA commit_state ownership operation is unavailable")
            attention = commit(snapshot.attention, current.attention)
        ple = current.ple
        snapshot.active = False
        result = Qwen38TTNNDecoderLayerState(
            self.namespace,
            self.layer_index,
            current.position,
            attention,
            ple,
        )
        try:
            self._validate_state(result)
        except BaseException as error:
            self._cleanup_snapshot_resources(snapshot, primary=error)
            raise
        self._cleanup_snapshot_resources(snapshot)
        return result

    def _apply_ple(
        self,
        residual,
        state: Qwen38TTNNDecoderLayerState,
        *,
        token_id: torch.Tensor | None,
        prepared_ple: Qwen38TTNNPLEPreparedInput | None = None,
        release_input: bool = True,
    ) -> tuple[Any, Qwen38TTNNPLEState | None]:
        if self._fused_apply_ple is not None:
            return self._fused_apply_ple(
                residual, state, token_id=token_id, prepared_ple=prepared_ple, release_input=release_input
            )
        if self.ple is None:
            if prepared_ple is not None:
                raise ValueError("prepared PLE input was supplied outside checkpoint layer 1")
            if not release_input:
                raise ValueError("only the PLE layer consumes its input residual directly and can retain it")
            return residual, None
        if prepared_ple is None and token_id is None:
            raise ValueError("the PLE layer requires one exact host token tensor")
        # PLE is qualified on the [1,1,4,640] branch-row layout; convert at its
        # once-per-token boundary instead of re-qualifying its internals.
        branch_rows = ttnn.permute(residual, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        result = (
            self.ple.forward_decode(branch_rows, token_id, state.ple)
            if prepared_ple is None
            else self.ple.forward_prepared(branch_rows, prepared_ple, state.ple)
        )
        if not isinstance(result, Qwen38TTNNPLEResult):
            raise TypeError("PLE callback must return Qwen38TTNNPLEResult")
        if result.state is not state.ple:
            raise RuntimeError("PLE decode replaced its fixed-address state")
        delta = ttnn.permute(result.residual_delta, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _deallocate_unique(branch_rows, result.residual_delta)
        self._validate_residual(delta, label="PLE residual delta")
        injected = ttnn.add(residual, delta, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # release_input=False keeps a caller-retained residual allocated (the HEAD/TAIL trace handoff).
        if release_input:
            _deallocate_unique(residual, delta)
        else:
            _deallocate_unique(delta)
        self._validate_residual(injected, label="PLE-injected residual")
        return injected, result.state

    def _route_through_gr_and_moe(
        self,
        attention_hidden,
        attention_gr_state,
        *,
        observe: Callable[[str], None],
        return_routing: bool,
        observe_moe: bool,
    ) -> tuple[Any, Any]:
        """Attention GR write -> MLP GR read -> streamed MoE -> MLP GR write; shared by both decode bodies."""

        self._validate_block(attention_hidden, label="attention output")

        observe("before-attention-gr-write")
        residual = self.attention_gr.write(attention_hidden, attention_gr_state)
        observe("after-attention-gr-write")
        _deallocate_unique(
            attention_hidden,
            attention_gr_state.residual,
            attention_gr_state.injection,
        )
        self._validate_residual(residual, label="post-attention residual")

        observe("before-mlp-gr-read")
        mlp_input, mlp_gr_state = self.mlp_gr.read(residual)
        observe("after-mlp-gr-read")
        self._validate_block(mlp_input, label="MLP GR read")

        observe("before-expert-stream-acquire")
        with self.expert_streamer.layer(self.layer_index, namespace=self.namespace.value) as packed_experts:
            observe("after-expert-stream-acquire")
            if not isinstance(packed_experts, tuple) or len(packed_experts) != 2:
                raise RuntimeError("BF4 streamer must yield exactly (packed_w0_w1, packed_w2)")
            observe("before-moe-forward")
            mlp_result = self.mlp.forward(
                mlp_input,
                packed_experts[0],
                packed_experts[1],
                return_routing=return_routing,
                phase_observer=observe if observe_moe else None,
            )
            observe("after-moe-forward")
            observe("before-expert-stream-release")
        observe("after-expert-stream-release")
        _deallocate_unique(mlp_input)
        self._validate_block(mlp_result.hidden_sharded, label="routed/shared MoE output")

        observe("before-mlp-gr-write")
        residual = self.mlp_gr.write(mlp_result.hidden_sharded, mlp_gr_state)
        observe("after-mlp-gr-write")
        _deallocate_unique(
            mlp_result.hidden_sharded,
            mlp_gr_state.residual,
            mlp_gr_state.injection,
        )
        self._validate_residual(residual, label="decoder-layer output residual")
        return residual, mlp_result

    def forward_decode_generic(
        self,
        residual_sharded,
        state: Qwen38TTNNDecoderLayerGenericState,
        *,
        prepared_ple: Qwen38TTNNPLEPreparedInput | None,
        rope,
        qsa_position,
        release_input: bool = True,
        phase_observer: Callable[[str], None] | None = None,
    ):
        """Advance one token through the position-generic body; every state buffer is updated in place.

        Same PLE -> attention GR -> attention -> attention GR -> MLP GR -> MoE
        -> MLP GR sequence as :meth:`forward_decode`, with no host position:
        ``rope`` holds the device-selected rows (:class:`Qwen38TTNNRoPEInputs`)
        and ``qsa_position`` the QSA module's device-derived position inputs.
        ``release_input=False`` leaves the input residual allocated after the
        PLE add consumed it (the HEAD/TAIL trace handoff the caller retains at a
        fixed address); only the PLE layer accepts it, every other layer's input
        is consumed by its attention GR write.  Returns the ``[1,4,1,640]``
        residual.
        """

        if phase_observer is not None and not callable(phase_observer):
            raise TypeError("decoder-layer phase observer must be callable")

        def observe(phase: str) -> None:
            if phase_observer is not None:
                phase_observer(phase)

        self._validate_residual(residual_sharded, label="decoder-layer residual")
        self._validate_generic_state(state)
        if self.ple is not None:
            if prepared_ple is None:
                raise ValueError("the generic PLE layer requires its prepared persistent row")
            if prepared_ple.source_token_context is not None:
                raise ValueError("the generic PLE row must be prepared without a host n-gram context")
            # The caller owns the n-gram context in the generic body (it derives
            # each token's row itself and rewrites prepared_ple.embedding_sharded
            # in place), so the PLE module's host bookkeeping is pinned to the
            # context-free prepared row at every position.
            state.ple.token_context = None
        if self.layer_type is Qwen38TTNNLayerType.QSA and (rope is None or qsa_position is None):
            raise ValueError("generic QSA decode requires device RoPE rows and QSA position inputs")

        observe("before-ple")
        residual, _ = self._apply_ple(
            residual_sharded, state, token_id=None, prepared_ple=prepared_ple, release_input=release_input
        )
        observe("after-ple")

        observe("before-attention-gr-read")
        attention_input, attention_gr_state = self.attention_gr.read(residual)
        observe("after-attention-gr-read")
        self._validate_block(attention_input, label="attention GR read")
        if isinstance(self.attention, Qwen38TTNNGDN):
            observe("before-gdn")
            attention_result = self.attention.forward_decode(attention_input, state.attention)
            observe("after-gdn")
            if attention_result.state is not state.attention:
                raise RuntimeError("generic GDN decode replaced its fixed-address state")
            attention_hidden = attention_result.hidden_sharded
        else:
            observe("before-qsa")
            attention_hidden = self.attention.forward_decode_generic(
                attention_input,
                state.attention,
                cos=rope.cos,
                sin=rope.sin,
                block_start_cos=rope.block_start_cos,
                block_start_sin=rope.block_start_sin,
                position=qsa_position,
            )
            observe("after-qsa")
        _deallocate_unique(attention_input)
        residual, _ = self._route_through_gr_and_moe(
            attention_hidden,
            attention_gr_state,
            observe=observe,
            return_routing=False,
            observe_moe=phase_observer is not None,
        )
        return residual

    def forward_decode(
        self,
        residual_sharded,
        state: Qwen38TTNNDecoderLayerState,
        *,
        token_id: torch.Tensor | None = None,
        prepared_ple: Qwen38TTNNPLEPreparedInput | None = None,
        cos=None,
        sin=None,
        block_start_cos=None,
        block_start_sin=None,
        reuse_qsa_selection: Qwen38TTNNQSASelection | None = None,
        retain_input_state: bool = False,
        return_routing: bool = False,
        phase_observer: Callable[[str], None] | None = None,
    ) -> Qwen38TTNNDecoderLayerResult:
        """Advance exactly one global token through one device layer.

        The input residual is consumed.  A speculative caller must first call
        :meth:`snapshot_state`; ``retain_input_state`` preserves QSA view
        transients but cannot itself copy GDN's or PLE's mutable buffers.  The
        first QSA call after a checkpoint must set ``retain_input_state=True``;
        the QSA ownership tracker rejects a destructive branch otherwise.
        """

        if phase_observer is not None and not callable(phase_observer):
            raise TypeError("decoder-layer phase observer must be callable")

        def observe(phase: str) -> None:
            if phase_observer is not None:
                phase_observer(phase)

        self._validate_residual(residual_sharded, label="decoder-layer residual")
        self._validate_state(state)
        if self.layer_type is Qwen38TTNNLayerType.GDN and reuse_qsa_selection is not None:
            raise ValueError("a GDN layer cannot reuse QSA selected indices")
        if self.layer_type is Qwen38TTNNLayerType.QSA and (cos is None or sin is None):
            raise ValueError("QSA decode requires current-position RoPE cos and sin")

        observe("before-ple")
        residual, ple_state = self._apply_ple(
            residual_sharded,
            state,
            token_id=token_id,
            prepared_ple=prepared_ple,
        )
        observe("after-ple")

        observe("before-attention-gr-read")
        attention_input, attention_gr_state = self.attention_gr.read(residual)
        observe("after-attention-gr-read")
        self._validate_block(attention_input, label="attention GR read")
        if isinstance(self.attention, Qwen38TTNNGDN):
            observe("before-gdn")
            attention_result = self.attention.forward_decode(attention_input, state.attention)
            observe("after-gdn")
            selection = None
        else:
            observe("before-qsa")
            attention_result = self.attention.forward_decode(
                attention_input,
                state.attention,
                cos=cos,
                sin=sin,
                block_start_cos=block_start_cos,
                block_start_sin=block_start_sin,
                position=state.position,
                reuse_selection=reuse_qsa_selection,
                retain_input_state=retain_input_state,
            )
            observe("after-qsa")
            selection = attention_result.selection
        _deallocate_unique(attention_input)
        residual, mlp_result = self._route_through_gr_and_moe(
            attention_result.hidden_sharded,
            attention_gr_state,
            observe=observe,
            return_routing=return_routing,
            observe_moe=phase_observer is not None,
        )

        observe("before-state-update")
        next_state = Qwen38TTNNDecoderLayerState(
            namespace=self.namespace,
            layer_index=self.layer_index,
            position=state.position + 1,
            attention=attention_result.state,
            ple=ple_state,
        )
        self._validate_state(next_state)
        observe("after-state-update")
        return Qwen38TTNNDecoderLayerResult(
            residual_sharded=residual,
            state=next_state,
            aux=Qwen38TTNNDecoderLayerAux(
                routing=mlp_result.routing,
                selection=selection,
                reused_qsa_selection=reuse_qsa_selection is not None,
            ),
        )

    # ------------------------------------------------------------------ rows path (MTP v2 verify)
    # Thin, typed entry points to the GDN and PLE multi-row paths.  They are the only way the layer
    # runs more than one row; the 1-row bodies above are untouched.  GR, MoE and QSA rows support
    # is not part of this step, so a whole-layer rows body is not offered yet.

    def _require_gdn(self, operation: str) -> Qwen38TTNNGDN:
        if not isinstance(self.attention, Qwen38TTNNGDN):
            raise TypeError(
                f"{operation} is a GDN-layer operation; layer {self.layer_index} is {self.layer_type.value}"
            )
        return self.attention

    def _require_ple(self, operation: str) -> Qwen38TTNNPLE:
        if self.ple is None:
            raise ValueError(f"{operation} is only available on checkpoint layer {PLE_CHECKPOINT_LAYER}")
        return self.ple

    def allocate_gdn_rows_state(self, constants: Qwen38TTNNGDNRowsConstants) -> Qwen38TTNNGDNRowsState:
        return self._require_gdn("allocate_gdn_rows_state").allocate_rows_state(constants)

    def forward_gdn_rows(
        self, attention_input_rows, state: Qwen38TTNNGDNState, rows_state: Qwen38TTNNGDNRowsState
    ) -> Qwen38TTNNGDNRowsResult:
        """``rows`` attention-GR-read rows ``[1,1,rows,640]`` -> GDN output rows; ``state`` is read only."""

        gdn = self._require_gdn("forward_gdn_rows")
        if not isinstance(state, Qwen38TTNNGDNState) or state.layer_index != self.layer_index:
            raise ValueError(f"forward_gdn_rows needs this layer's Qwen38TTNNGDNState, got {type(state).__name__}")
        return gdn.forward_rows(attention_input_rows, state, rows_state)

    def commit_gdn_rows(
        self,
        state: Qwen38TTNNGDNState,
        rows_state: Qwen38TTNNGDNRowsState,
        selectors: Qwen38TTNNRowsSelectors,
        *,
        step_on_full_rejection: bool = False,
        step_committed_rows: bool = False,
    ) -> None:
        self._require_gdn("commit_gdn_rows").commit_rows(
            state,
            rows_state,
            selectors,
            step_on_full_rejection=step_on_full_rejection,
            step_committed_rows=step_committed_rows,
        )

    def allocate_ple_rows_state(self, rows: int) -> Qwen38TTNNPLERowsState:
        return self._require_ple("allocate_ple_rows_state").allocate_rows_state(rows)

    def forward_ple_rows(
        self, residual_rows, prepared: Qwen38TTNNPLERowsPreparedInput, rows_state: Qwen38TTNNPLERowsState
    ):
        """PLE injection for ``rows`` residual rows ``[1,4,rows,640]`` (branch-major); the input is consumed."""

        ple = self._require_ple("forward_ple_rows")
        expected = (1, RESIDUAL_BRANCHES, rows_state.rows, LOCAL_HIDDEN_SIZE)
        if _shape(residual_rows) != expected or residual_rows.dtype != ttnn.bfloat16:
            raise ValueError(f"rows residual must be BF16 TILE {list(expected)}, got {tensor_metadata(residual_rows)}")
        self.mesh_contract.validate_tensor(residual_rows, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)
        # The PLE module owns the branch-row permute pair of the rows path (the layer permutes only in _apply_ple).
        injected = ple.inject_rows(residual_rows, prepared, rows_state)
        if _shape(injected) != expected:
            raise RuntimeError(f"PLE-injected rows have shape {_shape(injected)}, expected {expected}")
        return injected

    def commit_ple_rows(self, rows_state: Qwen38TTNNPLERowsState, selectors: Qwen38TTNNRowsSelectors) -> None:
        self._require_ple("commit_ple_rows").commit_rows(rows_state, selectors)

    # ------------------------------------------------------------------ prefill chunk (32 or 128 rows)
    # forward_chunk_generic is forward_decode_generic over the rows of one chunk: PLE rows (+ commit) ->
    # attention GR read_rows -> GDN forward_rows + commit_rows (the history carry) or QSA forward_chunk_generic
    # -> attention GR write_rows -> MLP GR read_rows -> the rows-32 / rows-128 MoE -> MLP GR write_rows.  The
    # chunk state holds what a chunk carries to the next one; the generic state is the decode's, updated in
    # place.  The 32-row form commits through the accept-scalar selectors (a full chunk or the padded tail);
    # the 128-row form always commits every row (no selectors).

    def _validate_residual_rows(self, residual, *, label: str, rows: int = CHUNK_ROWS) -> None:
        expected = residual_rows_shape(rows)
        if _shape(residual) != expected or residual.dtype != ttnn.bfloat16 or residual.layout != ttnn.TILE_LAYOUT:
            raise ValueError(
                f"{label} must be branch-major BF16 TILE {list(expected)}, got {tensor_metadata(residual)}"
            )
        self.mesh_contract.validate_tensor(residual, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)

    def _validate_block_rows(self, hidden, *, label: str, rows: int = CHUNK_ROWS) -> None:
        expected = block_rows_shape(rows)
        if _shape(hidden) != expected or hidden.dtype != ttnn.bfloat16 or hidden.layout != ttnn.TILE_LAYOUT:
            raise RuntimeError(f"{label} must be BF16 TILE {list(expected)}, got {tensor_metadata(hidden)}")
        self.mesh_contract.validate_tensor(hidden, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)

    def _validate_chunk_state(self, state: Qwen38TTNNDecoderLayerChunkState) -> None:
        if not isinstance(state, Qwen38TTNNDecoderLayerChunkState):
            raise TypeError(f"decoder layer requires Qwen38TTNNDecoderLayerChunkState, got {type(state).__name__}")
        if state.namespace is not self.namespace or state.layer_index != self.layer_index:
            raise ValueError(
                f"chunk state identity {(state.namespace, state.layer_index)} does not match "
                f"{(self.namespace.value, self.layer_index)}"
            )
        if state.rows not in CHUNK_ROW_COUNTS and not is_slab_rows(state.rows):
            raise ValueError(
                f"chunk state rows must be one of {CHUNK_ROW_COUNTS} or a slab row count, got {state.rows!r}"
            )
        if isinstance(self.attention, Qwen38TTNNGDN) != isinstance(state.attention, Qwen38TTNNGDNRowsState):
            raise TypeError(f"layer {self.layer_index} chunk attention state is {type(state.attention).__name__}")
        attention_rows = (
            state.attention.constants.rows
            if isinstance(state.attention, Qwen38TTNNGDNRowsState)
            else state.attention.rows
        )
        if attention_rows != state.rows:
            raise ValueError(f"chunk attention state holds {attention_rows} rows, the chunk state {state.rows}")
        if (self.ple is None) != (state.ple is None):
            raise ValueError("PLE rows state must be present exactly on the PLE layer")
        # A slab's PLE runs the 128-row pass per block: its rows state holds 128 rows.
        ple_rows = LONG_CHUNK_ROWS if is_slab_rows(state.rows) else state.rows
        if state.ple is not None and state.ple.rows != ple_rows:
            raise ValueError(f"PLE rows state holds {state.ple.rows} rows, the chunk state needs {ple_rows}")
        if state.moe.rows != state.rows or state.moe.weights is not self.mlp.weights:
            raise ValueError(f"chunk MoE must be a rows-{state.rows} instance over this layer's weights")

    def allocate_chunk_state(
        self,
        constants: Qwen38TTNNGDNRowsConstants,
        *,
        base: Qwen38TTNNDecoderLayerChunkState | None = None,
        local_combine_output=None,
        gdn_body: Qwen38TTNNGDNRowsState | None = None,
    ) -> Qwen38TTNNDecoderLayerChunkState:
        """Allocate this layer's chunk buffers before any capture: rows state, PLE rows state, the MoE instance.

        ``constants.rows`` picks the form.  The 128-row form and a slab need ``base``, this layer's 32-row chunk
        state, whose GDN and PLE histories they share, and ``local_combine_output``, the combine buffer shared by
        every layer's MoE instance of that form.  ``gdn_body`` (slab GDN layers after the first) is the rows state
        whose pass buffers this layer's rows state reuses.
        """

        rows = constants.rows
        if rows != CHUNK_ROWS and (base is None or base.rows != CHUNK_ROWS or local_combine_output is None):
            raise ValueError(
                f"the {rows}-row chunk state needs the layer's 32-row chunk state and the shared combine buffer"
            )
        if rows == CHUNK_ROWS and (base is not None or local_combine_output is not None):
            raise ValueError("the 32-row chunk state owns its histories and combine buffer")
        if gdn_body is not None and (not is_slab_rows(rows) or not isinstance(self.attention, Qwen38TTNNGDN)):
            raise ValueError("a shared GDN rows body is a slab GDN layer's option")
        if isinstance(self.attention, Qwen38TTNNGDN):
            attention = self.attention.allocate_rows_state(
                constants, history=None if base is None else base.attention.history, body=gdn_body
            )
        else:
            attention = self.attention.allocate_chunk_state(rows)
        ple = None
        moe = None
        try:
            if self.ple is not None:
                ple = self.ple.allocate_rows_state(
                    LONG_CHUNK_ROWS if is_slab_rows(rows) else rows, history=None if base is None else base.ple.history
                )
            moe = Qwen38TTNNMoE(
                self.mlp.mesh_device,
                self.mlp.mesh_contract,
                self.mlp.weights,
                tt_ccl=self.mlp.tt_ccl,
                collective_topology=self.mlp.collective_topology,
                rows=rows,
                synchronization_policy=self.mlp.synchronization_policy,
                local_combine_output=local_combine_output,
                admitted_rows=SUPPORTED_ROWS + ((rows,) if is_slab_rows(rows) else ()),
                prefill_dense=self.mlp.prefill_dense,  # the layer's policy and resident prefill weights, shared
            )
            result = Qwen38TTNNDecoderLayerChunkState(self.namespace, self.layer_index, attention, ple, moe, rows)
            self._validate_chunk_state(result)
            return result
        except BaseException as error:
            actions = []
            if moe is not None:
                actions.append(("chunk MoE buffers", moe.release_owned_buffers))
            if ple is not None:
                actions.append(("PLE rows state", ple.deallocate))
            actions.append(("chunk attention state", lambda: self._release_chunk_attention_state(attention)))
            _run_cleanup_actions("decoder-layer chunk state allocation", actions, primary=error)
            raise

    def _release_chunk_attention_state(self, attention_state) -> None:
        if isinstance(self.attention, Qwen38TTNNGDN):
            attention_state.deallocate()
        else:
            self.attention.release_chunk_state(attention_state)

    def release_chunk_state(self, state: Qwen38TTNNDecoderLayerChunkState) -> None:
        self._validate_chunk_state(state)
        actions = [
            ("chunk MoE buffers", state.moe.release_owned_buffers),
            ("chunk attention state", lambda: self._release_chunk_attention_state(state.attention)),
        ]
        if state.ple is not None:
            actions.append(("PLE rows state", state.ple.deallocate))
        _run_cleanup_actions("decoder-layer chunk state", actions)

    def reset_chunk_state_inplace(
        self, state: Qwen38TTNNDecoderLayerChunkState, generic_state: Qwen38TTNNDecoderLayerGenericState
    ) -> None:
        """Seed the chunk carry from the generic state at fixed addresses (after the generic reset, or after
        decode steps that ended at P % 32 == 0): the GDN history from the ring, the PLE history from the nine
        slots, the QSA kept buffers zeroed (hygiene; every chunk rewrites them).  The 128-row state shares its
        histories with the 32-row state, which seeds them; only its QSA buffers are zeroed here."""

        self._validate_chunk_state(state)
        self._validate_generic_state(generic_state)
        if isinstance(self.attention, Qwen38TTNNGDN):
            # A chunk starts at P % 32 == 0: the next token lands in slot 0.  The traced steps never advance the
            # host phase, so it is set here rather than read (a hand-off at P % 4 != 0 may have left it there).
            generic_state.attention.conv_phase = 0
            if state.rows == CHUNK_ROWS:
                self.attention.sync_rows_history_from_state(generic_state.attention, state.attention)
        else:
            for label, tensor in (
                ("QSA kept KV slab", state.attention.kept_kv),
                ("QSA kept raw keys", state.attention.kept_raw),
            ):
                zeroed = ttnn.fill(tensor, 0.0, output_tensor=tensor)
                if _tensor_key(zeroed) != _tensor_key(tensor):
                    raise RuntimeError(f"{label} reset was not in place")
        if state.ple is not None and state.rows == CHUNK_ROWS:
            state.ple.load_from_state(generic_state.ple)

    def _inject_ple_slab(self, residual_rows, prepared: Qwen38TTNNPLERowsPreparedInput, rows_state):
        """The slab's PLE: the 128-row chunk pass (inject + full commit, bitwise the long chunk) per 128-row block of
        the residual rows and the prepared rows, the injected blocks concatenated.  The PLE's gate keeps token-major
        ``[rows, 4, 2560]`` fp32 intermediates whose branch dim pads to a tile: at 2048 rows they would be 670 MB."""

        dram = ttnn.DRAM_MEMORY_CONFIG
        rows = _shape(residual_rows)[2]
        blocks = []
        for start in range(0, rows, LONG_CHUNK_ROWS):
            residual_block = ttnn.slice(
                residual_rows,
                (0, 0, start, 0),
                (1, RESIDUAL_BRANCHES, start + LONG_CHUNK_ROWS, LOCAL_HIDDEN_SIZE),
                memory_config=dram,
            )
            rows_block = ttnn.slice(
                prepared.embedding_rows,
                (0, 0, start, 0),
                (1, 1, start + LONG_CHUNK_ROWS, LOCAL_HIDDEN_SIZE),
                memory_config=dram,
            )
            rows_block.update_tensor_topology(prepared.embedding_rows.tensor_topology())
            block_prepared = Qwen38TTNNPLERowsPreparedInput(rows_block, (), ())
            blocks.append(self.ple.inject_rows(residual_block, block_prepared, rows_state))
            self.ple.commit_rows_full(rows_state)
            block_prepared.release()
        _deallocate_unique(residual_rows)
        residual = ttnn.concat(blocks, dim=2, memory_config=dram)
        _deallocate_unique(*blocks)
        return residual

    def forward_chunk_generic(
        self,
        residual_rows,
        generic_state: Qwen38TTNNDecoderLayerGenericState,
        chunk_state: Qwen38TTNNDecoderLayerChunkState,
        *,
        prepared_ple_rows: Qwen38TTNNPLERowsPreparedInput | None,
        rope_rows,
        qsa_chunk: qsa_module.Qwen38TTNNQSAChunkInputs | None,
        qsa_chunk_constants: qsa_module.Qwen38TTNNQSAChunkConstants | None,
        selectors: Qwen38TTNNRowsSelectors | None,
        gdn_step_anchor: bool = False,
    ):
        """Advance the rows of one chunk through this layer; the input rows are consumed.

        ``selectors`` (from the device accept scalar: 31 for a full chunk, r - 1 for the padded tail) drive
        the GDN and PLE history commits of the 32-row form, so one trace serves both; the 128-row form takes
        ``None`` and commits every row (``commit_rows_full``: the forward pass's own final state, no re-run).
        ``rope_rows`` holds the chunk's cos/sin tiles and ``qsa_chunk`` the derived chunk inputs; both are None
        on GDN layers' callers' side only by omission (they are shared by every layer).  ``gdn_step_anchor``
        commits the GDN state through the 1-row FP32 step arithmetic over the committed rows
        (``commit_rows(step_committed_rows=True)``: the committed state is the 1-row path's, not the chunk
        kernel's) at about 16 ops per row per GDN layer; the 128-row form does not offer it.  Returns the
        ``[1,4,rows,640]`` residual rows.
        """

        rows = chunk_state.rows
        if (selectors is None) != (rows != CHUNK_ROWS):
            raise ValueError(f"the {rows}-row chunk form {'takes no' if rows != CHUNK_ROWS else 'needs the'} selectors")
        if gdn_step_anchor and rows != CHUNK_ROWS:
            raise ValueError("the GDN step anchor is a 32-row chunk option")
        self._validate_residual_rows(residual_rows, label="chunk residual rows", rows=rows)
        self._validate_generic_state(generic_state)
        self._validate_chunk_state(chunk_state)
        if self.ple is not None:
            if prepared_ple_rows is None:
                raise ValueError("the PLE layer's chunk needs its prepared persistent PLE rows")
            generic_state.ple.token_context = None
            if is_slab_rows(rows):
                residual = self._inject_ple_slab(residual_rows, prepared_ple_rows, chunk_state.ple)
            else:
                residual = self.ple.inject_rows(residual_rows, prepared_ple_rows, chunk_state.ple)
                if selectors is None:
                    self.ple.commit_rows_full(chunk_state.ple)
                else:
                    self.ple.commit_rows(chunk_state.ple, selectors)
        else:
            if prepared_ple_rows is not None:
                raise ValueError("prepared PLE rows were supplied outside checkpoint layer 1")
            residual = residual_rows
        self._validate_residual_rows(residual, label="chunk PLE-injected residual rows", rows=rows)

        attention_input, attention_gr_state = self.attention_gr.read_rows(residual)
        self._validate_block_rows(attention_input, label="chunk attention GR read", rows=rows)
        if isinstance(self.attention, Qwen38TTNNGDN):
            result = self.attention.forward_rows(attention_input, generic_state.attention, chunk_state.attention)
            if selectors is None:
                self.attention.commit_rows_full(generic_state.attention, chunk_state.attention, result.final_state)
            else:
                self.attention.commit_rows(
                    generic_state.attention, chunk_state.attention, selectors, step_committed_rows=gdn_step_anchor
                )
                _deallocate_unique(result.final_state)
            attention_hidden = result.hidden_rows  # the persistent rows output: never deallocated here
            persistent_hidden = True
        else:
            if rope_rows is None or qsa_chunk is None or qsa_chunk_constants is None:
                raise ValueError("chunk QSA needs the RoPE rows, the chunk inputs and the chunk constants")
            attention_hidden = self.attention.forward_chunk_generic(
                attention_input,
                generic_state.attention,
                chunk_state.attention,
                cos=rope_rows.cos,
                sin=rope_rows.sin,
                block_start_cos=rope_rows.block_start_cos,
                block_start_sin=rope_rows.block_start_sin,
                chunk=qsa_chunk,
                constants=qsa_chunk_constants,
            )
            persistent_hidden = False
        _deallocate_unique(attention_input)
        self._validate_block_rows(attention_hidden, label="chunk attention output", rows=rows)

        residual = self.attention_gr.write_rows(attention_hidden, attention_gr_state)
        _deallocate_unique(
            None if persistent_hidden else attention_hidden,
            attention_gr_state.residual,
            attention_gr_state.injection,
        )
        self._validate_residual_rows(residual, label="chunk post-attention residual rows", rows=rows)

        mlp_input, mlp_gr_state = self.mlp_gr.read_rows(residual)
        self._validate_block_rows(mlp_input, label="chunk MLP GR read", rows=rows)
        with self.expert_streamer.layer(self.layer_index, namespace=self.namespace.value) as packed_experts:
            if not isinstance(packed_experts, tuple) or len(packed_experts) != 2:
                raise RuntimeError("BF4 streamer must yield exactly (packed_w0_w1, packed_w2)")
            mlp_result = chunk_state.moe.forward(mlp_input, packed_experts[0], packed_experts[1])
        _deallocate_unique(mlp_input)
        self._validate_block_rows(mlp_result.hidden_sharded, label="chunk routed/shared MoE output", rows=rows)

        residual = self.mlp_gr.write_rows(mlp_result.hidden_sharded, mlp_gr_state)
        _deallocate_unique(mlp_result.hidden_sharded, mlp_gr_state.residual, mlp_gr_state.injection)
        self._validate_residual_rows(residual, label="chunk decoder-layer output residual rows", rows=rows)
        return residual

    def finish_chunk_state_inplace(
        self,
        state: Qwen38TTNNDecoderLayerChunkState,
        generic_state: Qwen38TTNNDecoderLayerGenericState,
        *,
        prefilled: int,
        qsa_ring_select,
    ) -> None:
        """Eager hand-off after the last chunk of a prefill of ``prefilled`` positions: the decode's per-token
        buffers take what the chunk carry holds (the GDN ring slots and phase from the committed history, the PLE
        slots from the rows history, the QSA staging tile and raw-key ring from the kept slab and raw keys).  The
        committed GDN state and the QSA caches are already the decode's."""

        self._validate_chunk_state(state)
        self._validate_generic_state(generic_state)
        if state.rows != CHUNK_ROWS:
            raise ValueError(f"the hand-off reads the {CHUNK_ROWS}-row chunk state, got {state.rows} rows")
        if isinstance(self.attention, Qwen38TTNNGDN):
            # The next token lands in slot prefilled % 4; the history rows are the three before it, oldest first.
            generic_state.attention.conv_phase = prefilled % CONV_KERNEL_SIZE
            self.attention.sync_state_from_rows_history(state.attention, generic_state.attention)
        else:
            self.attention.handoff_chunk_state(
                generic_state.attention,
                state.attention,
                ring_select=qsa_ring_select,
                open_block=prefilled % CHUNK_ROWS != 0,
            )
        if state.ple is not None:
            state.ple.store_to_state(generic_state.ple)
            generic_state.ple.token_context = None  # the generic body's caller owns the n-gram context

    # ------------------------------------------------------------------ lanes path (batched decode)
    # forward_decode_lanes is forward_decode_generic over B lanes (row u = lane u): PLE lanes -> attention GR
    # read_rows -> GDN forward_decode_lanes or QSA forward_decode_lanes on the 32-row tile -> attention GR
    # write_rows -> MLP GR read_rows -> the rows-B MoE -> MLP GR write_rows.  The lane state is fixed-address and
    # updated in place; the 1-row bodies above are untouched.

    def _validate_rows(self, tensor, expected: tuple[int, ...], *, label: str) -> None:
        if _shape(tensor) != expected or tensor.dtype != ttnn.bfloat16 or tensor.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"{label} must be BF16 TILE {list(expected)}, got {tensor_metadata(tensor)}")
        self.mesh_contract.validate_tensor(tensor, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)

    def _validate_lane_state(self, state: Qwen38TTNNDecoderLayerLaneState) -> int:
        if not isinstance(state, Qwen38TTNNDecoderLayerLaneState):
            raise TypeError(f"decoder layer requires Qwen38TTNNDecoderLayerLaneState, got {type(state).__name__}")
        if state.namespace is not self.namespace or state.layer_index != self.layer_index:
            raise ValueError(
                f"lane state identity {(state.namespace, state.layer_index)} does not match "
                f"{(self.namespace.value, self.layer_index)}"
            )
        lanes = require_lane_count(state.lanes, label="decoder-layer lanes")
        if isinstance(self.attention, Qwen38TTNNGDN):
            if not isinstance(state.attention, Qwen38TTNNGDNState) or state.attention.batch_size != lanes:
                raise TypeError(f"layer {self.layer_index} lane attention state is not a {lanes}-lane GDN state")
            if state.attention_rows is not None:
                raise ValueError("a GDN layer's lane state carries no attention rows buffer")
        else:
            if not isinstance(state.attention, qsa_module.Qwen38TTNNQSALaneState) or state.attention.lanes != lanes:
                raise TypeError(f"layer {self.layer_index} lane attention state is not a {lanes}-lane QSA state")
            if (state.attention_rows is None) != (lanes == CHUNK_ROWS):
                raise ValueError("a QSA layer's lane state carries an attention rows buffer exactly below 32 lanes")
            if state.attention_rows is not None:
                self._validate_rows(
                    state.attention_rows, (1, 1, lanes, LOCAL_HIDDEN_SIZE), label="QSA lane attention rows"
                )
        if (self.ple is None) != (state.ple is None):
            raise ValueError("PLE lanes state must be present exactly on the PLE layer")
        if state.ple is not None and state.ple.lanes != lanes:
            raise ValueError(f"PLE lanes state holds {state.ple.lanes} lanes, expected {lanes}")
        if state.moe.rows != lanes or state.moe.weights is not self.mlp.weights:
            raise ValueError(f"lane MoE must be a rows-{lanes} instance over this layer's weights")
        return lanes

    def allocate_lane_state(self, lanes: int) -> Qwen38TTNNDecoderLayerLaneState:
        """Allocate this layer's B-lane buffers before any capture: the lane attention state, the PLE lanes state,
        the rows-B MoE and (a QSA layer below 32 lanes) the persistent attention rows."""

        lanes = require_lane_count(lanes, label="decoder-layer lanes")
        attention = self.attention.allocate_lane_state(lanes)
        ple = None
        moe = None
        attention_rows = None
        try:
            ple = None if self.ple is None else self.ple.allocate_lanes_state(lanes)
            moe = Qwen38TTNNMoE(
                self.mlp.mesh_device,
                self.mlp.mesh_contract,
                self.mlp.weights,
                tt_ccl=self.mlp.tt_ccl,
                collective_topology=self.mlp.collective_topology,
                rows=lanes,
                synchronization_policy=self.mlp.synchronization_policy,
            )
            if isinstance(self.attention, Qwen38TTNNQSA) and lanes < CHUNK_ROWS:
                mesh_device = self.attention.mesh_device
                attention_rows = ttnn.from_torch(
                    torch.zeros((1, 1, lanes, HIDDEN_SIZE), dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
                )
            result = Qwen38TTNNDecoderLayerLaneState(
                self.namespace, self.layer_index, lanes, attention, ple, moe, attention_rows
            )
            self._validate_lane_state(result)
            return result
        except BaseException as error:
            actions = []
            if attention_rows is not None:
                actions.append(("QSA lane attention rows", lambda: _deallocate_unique(attention_rows)))
            if moe is not None:
                actions.append(("lane MoE buffers", moe.release_owned_buffers))
            if ple is not None:
                actions.append(("PLE lanes state", ple.deallocate))
            actions.append(("lane attention state", lambda: self._release_lane_attention_state(attention)))
            _run_cleanup_actions("decoder-layer lane state allocation", actions, primary=error)
            raise

    def _release_lane_attention_state(self, attention_state) -> None:
        if isinstance(self.attention, Qwen38TTNNGDN):
            attention_state.deallocate()
        else:
            self.attention.release_lane_state(attention_state)

    def release_lane_state(self, state: Qwen38TTNNDecoderLayerLaneState) -> None:
        self._validate_lane_state(state)
        actions = [
            ("lane MoE buffers", state.moe.release_owned_buffers),
            ("lane attention state", lambda: self._release_lane_attention_state(state.attention)),
        ]
        if state.attention_rows is not None:
            actions.append(("QSA lane attention rows", lambda: _deallocate_unique(state.attention_rows)))
        if state.ple is not None:
            actions.append(("PLE lanes state", state.ple.deallocate))
        _run_cleanup_actions("decoder-layer lane state", actions)

    def reset_lane_state_inplace(self, state: Qwen38TTNNDecoderLayerLaneState) -> None:
        """Every lane back to its position-zero contents at every captured address (a new batch)."""

        self._validate_lane_state(state)
        if isinstance(self.attention, Qwen38TTNNGDN):
            state.attention.reset_inplace()
        else:
            self.attention.reset_lane_state_inplace(state.attention)
        if state.ple is not None:
            state.ple.reset_inplace()

    def reset_lane_inplace(self, state: Qwen38TTNNDecoderLayerLaneState, lane: int) -> None:
        """One lane back to position zero (keep-mask writes: the other lanes hold their state), no address change;
        the caller admits the lane at a step of its position's residue class."""

        lanes = self._validate_lane_state(state)
        if isinstance(lane, bool) or type(lane) is not int or not 0 <= lane < lanes:
            raise ValueError(f"lane must be an int in [0,{lanes}), got {lane!r}")
        if isinstance(self.attention, Qwen38TTNNGDN):
            state.attention.reset_lane_inplace(lane)
        else:
            self.attention.reset_lane_inplace(state.attention, lane)
        if state.ple is not None:
            state.ple.reset_lane_inplace(lane)

    def forward_decode_lanes(
        self,
        residual_lanes,
        state: Qwen38TTNNDecoderLayerLaneState,
        *,
        prepared_ple: Qwen38TTNNPLELanesPreparedInput | None,
        rope,
        qsa_lanes: qsa_module.Qwen38TTNNQSALaneInputs | None,
        qsa_constants: qsa_module.Qwen38TTNNQSAChunkConstants | None,
        qsa_lane_constants: qsa_module.Qwen38TTNNQSALaneConstants | None,
    ):
        """Advance one token per lane through this layer; the ``[1,4,B,640]`` input rows are consumed and every
        state buffer is updated in place.

        No host position: ``rope`` holds the per-lane RoPE rows (``Qwen38TTNNRoPEInputs`` of the 32-lane lookup)
        and ``qsa_lanes`` the derived lane inputs, both shared by every layer.  The caller owns the PLE n-gram
        contexts: the prepared lanes are looked up context-free and the state's contexts are pinned to None at every
        step (the generic body's rule).  Returns the ``[1,4,B,640]`` residual rows.
        """

        lanes = self._validate_lane_state(state)
        residual_shape = (1, RESIDUAL_BRANCHES, lanes, LOCAL_HIDDEN_SIZE)
        block_shape = (1, 1, lanes, LOCAL_HIDDEN_SIZE)
        self._validate_rows(residual_lanes, residual_shape, label="lane residual rows")
        if self.ple is not None:
            if prepared_ple is None:
                raise ValueError("the PLE layer's lanes need the prepared persistent PLE rows")
            if tuple(prepared_ple.source_contexts) != (None,) * lanes:
                raise ValueError("the lane PLE rows must be prepared without host n-gram contexts")
            state.ple.token_contexts = (None,) * lanes
            residual = self.ple.inject_lanes(residual_lanes, prepared_ple, state.ple)
        else:
            if prepared_ple is not None:
                raise ValueError("prepared PLE lanes were supplied outside checkpoint layer 1")
            residual = residual_lanes
        self._validate_rows(residual, residual_shape, label="lane PLE-injected residual rows")

        attention_input, attention_gr_state = self.attention_gr.read_rows(residual, flat_views=True)
        self._validate_rows(attention_input, block_shape, label="lane attention GR read")
        if isinstance(self.attention, Qwen38TTNNGDN):
            result = self.attention.forward_decode_lanes(attention_input, state.attention)
            if result.state is not state.attention:
                raise RuntimeError("lane GDN decode replaced its fixed-address state")
            attention_hidden = result.hidden_sharded
            persistent_hidden = False
        else:
            if rope is None or qsa_lanes is None or qsa_constants is None or qsa_lane_constants is None:
                raise ValueError("lane QSA needs the RoPE rows, the lane inputs, the chunk and the lane constants")
            # The QSA lane body runs on the 32-row tile (rows past the lane count idle): the lane rows are zero
            # padded inside their own tile (ttnn.pad returns a view of its input on this runtime, so it is never
            # released here) and the output's lane rows land in the persistent attention rows.
            if lanes == CHUNK_ROWS:
                hidden_rows = attention_input
            else:
                hidden_rows = ttnn.pad(
                    attention_input,
                    [(0, 0), (0, 0), (0, CHUNK_ROWS - lanes), (0, 0)],
                    0.0,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                hidden_rows.update_tensor_topology(attention_input.tensor_topology())
            output_rows = self.attention.forward_decode_lanes(
                hidden_rows,
                state.attention,
                cos=rope.cos,
                sin=rope.sin,
                block_start_cos=rope.block_start_cos,
                block_start_sin=rope.block_start_sin,
                lanes=qsa_lanes,
                constants=qsa_constants,
                lane_constants=qsa_lane_constants,
            )
            if lanes == CHUNK_ROWS:
                attention_hidden = output_rows
                persistent_hidden = False
            else:
                landed = ttnn.slice(
                    output_rows, (0, 0, 0, 0), (1, 1, lanes, LOCAL_HIDDEN_SIZE), output_tensor=state.attention_rows
                )
                if landed is not None and _tensor_key(landed) != _tensor_key(state.attention_rows):
                    raise RuntimeError("lane QSA output slice did not land in the persistent attention rows")
                _deallocate_unique(output_rows)
                attention_hidden = state.attention_rows
                persistent_hidden = True
        _deallocate_unique(attention_input)
        self._validate_rows(attention_hidden, block_shape, label="lane attention output")

        residual = self.attention_gr.write_rows(attention_hidden, attention_gr_state)
        _deallocate_unique(
            None if persistent_hidden else attention_hidden,
            attention_gr_state.residual,
            attention_gr_state.injection,
        )
        self._validate_rows(residual, residual_shape, label="lane post-attention residual rows")

        mlp_input, mlp_gr_state = self.mlp_gr.read_rows(residual, flat_views=True)
        self._validate_rows(mlp_input, block_shape, label="lane MLP GR read")
        with self.expert_streamer.layer(self.layer_index, namespace=self.namespace.value) as packed_experts:
            if not isinstance(packed_experts, tuple) or len(packed_experts) != 2:
                raise RuntimeError("BF4 streamer must yield exactly (packed_w0_w1, packed_w2)")
            mlp_result = state.moe.forward(mlp_input, packed_experts[0], packed_experts[1])
        _deallocate_unique(mlp_input)
        self._validate_rows(mlp_result.hidden_sharded, block_shape, label="lane routed/shared MoE output")

        residual = self.mlp_gr.write_rows(mlp_result.hidden_sharded, mlp_gr_state)
        _deallocate_unique(mlp_result.hidden_sharded, mlp_gr_state.residual, mlp_gr_state.injection)
        self._validate_rows(residual, residual_shape, label="lane decoder-layer output residual rows")
        return residual


def validate_layer_static_contract() -> None:
    """No-device guard for exact layer order and local shapes."""

    if (BACKBONE_LAYERS, MTP_LAYERS, HIDDEN_SIZE, TP_SIZE, RESIDUAL_BRANCHES) != (48, 1, 2560, 4, 4):
        raise RuntimeError("Qwen3.8 decoder-layer geometry drifted")
    if RESIDUAL_LOCAL_SHAPE != (1, 4, 1, 640) or BLOCK_LOCAL_SHAPE != (1, 1, 1, 640):
        raise RuntimeError("Qwen3.8 decoder-layer local shapes drifted")
    if tuple(_expected_layer_type(Qwen38TTNNLayerNamespace.BACKBONE, index).value for index in range(8)) != (
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ):
        raise RuntimeError("Qwen3.8 3xGDN+1xQSA layer pattern drifted")
    if _expected_layer_type(Qwen38TTNNLayerNamespace.MTP, 0) is not Qwen38TTNNLayerType.QSA:
        raise RuntimeError("Qwen3.8 MTP layer must use QSA")
