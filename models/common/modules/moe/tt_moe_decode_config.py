# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _is_subconfig(annotation: Any) -> bool:
    """True if the annotation is a `BaseModel` subclass.

    `issubclass` raises TypeError on generic aliases like `tuple[int, int]`
    (pydantic's BaseModel metaclass rejects them), so we guard it.
    """
    try:
        return issubclass(annotation, BaseModel)
    except TypeError:
        return False


from ttnn.operations.ccl import MoEActivationFunction

import ttnn


def _default_fast_reduce_output_memory_config() -> ttnn.MemoryConfig:
    """`fast_reduce_output_memory_config` from test_optimized_moe_decode_block.py."""
    return ttnn.MemoryConfig(
        ttnn.BufferType.L1,
        ttnn.NdShardSpec(
            ttnn.Shape([1, 32, 128]),
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(2, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(2, 5), ttnn.CoreCoord(2, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(3, 5), ttnn.CoreCoord(3, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(6, 0)),
                    ttnn.CoreRange(ttnn.CoreCoord(6, 5), ttnn.CoreCoord(6, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
                ]
            ),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


def _default_dispatch_input_expert_scores_memory_config(
    tokens_per_device: int, select_experts_k: int
) -> ttnn.MemoryConfig:
    """`dispatch_input_expert_scores_memory_config` from test_optimized_moe_decode_block.py.

    Height-sharded L1, shard shape `[1, select_experts_k]` (seq=1 for decode),
    spread over a core grid sized from `tokens_per_device`.
    """
    num_cores_y = min(8, tokens_per_device)
    num_cores_x = (tokens_per_device + num_cores_y - 1) // num_cores_y
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))}),
        [1, select_experts_k],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )


def _default_post_combine_tilize_memory_config(effective_experts_k: int) -> ttnn.MemoryConfig:
    """`post_combine_tilize_output_memory_config` from test_optimized_moe_decode_block.py.

    NdShard L1 with grid extending to `(6, effective_experts_k - 1)`.
    """
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.L1,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[32, 1024],
            grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, effective_experts_k - 1))}),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def _fill_default_if_missing(data: dict, sub_field_name: str, mem_field_name: str, default_value: Any) -> None:
    """Set `data[sub_field_name][mem_field_name] = default_value` only if not provided.

    Handles dict/BaseModel/None inputs for the sub-config. User-provided values are
    preserved; only None or missing keys get the default.
    """
    cur = data.get(sub_field_name)
    if cur is None:
        data[sub_field_name] = {mem_field_name: default_value}
    elif isinstance(cur, dict):
        if cur.get(mem_field_name) is None:
            cur[mem_field_name] = default_value
    elif isinstance(cur, BaseModel):
        if getattr(cur, mem_field_name, None) is None:
            data[sub_field_name] = cur.model_copy(update={mem_field_name: default_value})


class _TTOpKwargs(BaseModel):
    """Base for sub-configs whose `model_dump()` is splatted into a ttnn op call.

    `arbitrary_types_allowed` lets us hold ttnn objects (MemoryConfig, CoreRangeSet,
    enum values, etc.) as field values. `frozen` prevents accidental mutation after
    construction. `populate_by_name` lets users construct via either the field name
    or its `validation_alias` (used to receive renamed values from the parent).

    Shared fields adopted from `TTMoEDecodeConfig` are declared via `adopt_fields()`
    — the parent's pre-validator injects each name in that set into this sub-config's
    input. For fields whose name in the sub-config differs from the parent (e.g.
    reduce-scatter `cluster_axis` ← parent's `rs_cluster_axis`), declare the field
    with `Field(validation_alias='<parent_name>')` and list the parent name in
    `adopt_fields()`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, populate_by_name=True)

    @classmethod
    def adopt_fields(cls) -> set[str]:
        """Parent field/derived-value names this sub-config wants at validation."""
        return set()


class DispatchConfig(_TTOpKwargs):
    """Kwargs spread into `ttnn.experimental.all_to_all_dispatch_metadata`."""

    shared_expert_ids: Optional[list[int]] = None
    cluster_axis: int
    num_links: int
    drain_sync_tilizer_core: Optional[ttnn.CoreCoord] = None
    worker_mode: ttnn.WorkerMode
    dispatch_algorithm: ttnn.DispatchAlgorithm

    @classmethod
    def adopt_fields(cls) -> set[str]:
        return {"cluster_axis", "num_links"}


class ComputeConfig(_TTOpKwargs):
    """Kwargs spread into `ttnn.experimental.moe_compute`."""

    output_height_shard_dim: int
    cluster_axis: int
    mux_core_range_set: ttnn.CoreRangeSet
    has_bias: bool
    activation_type: MoEActivationFunction = MoEActivationFunction.SILU

    @classmethod
    def adopt_fields(cls) -> set[str]:
        return {"cluster_axis", "has_bias"}


class PostCombineTilizeConfig(_TTOpKwargs):
    """Kwargs spread into `ttnn.experimental.deepseek_moe_post_combine_tilize`
    (taken when `batch_per_device == ttnn.TILE_SIZE`).

    `output_memory_config` defaults to the data-dependent NdShard config from
    test_optimized_moe_decode_block.py (filled in by the parent validator using
    `effective_experts_k`).
    """

    output_memory_config: Optional[ttnn.MemoryConfig] = None


class TilizeWithValPaddingConfig(_TTOpKwargs):
    """Kwargs spread into `ttnn.tilize_with_val_padding`
    (taken when `batch_per_device != ttnn.TILE_SIZE`).

    `memory_config` shares the same default as `PostCombineTilizeConfig.output_memory_config`.
    """

    memory_config: Optional[ttnn.MemoryConfig] = None


class ReduceConfig(_TTOpKwargs):
    """Kwargs spread into `ttnn.experimental.deepseek_moe_fast_reduce_nc_fused`."""

    reduce_dim: int
    cluster_axis: int
    split_size: int
    output_memory_config: ttnn.MemoryConfig = Field(default_factory=_default_fast_reduce_output_memory_config)
    num_shared_experts: int
    shared_expert_scale: float

    @classmethod
    def adopt_fields(cls) -> set[str]:
        return {"cluster_axis", "split_size"}


class DeepseekMoEReduceScatterConfig(_TTOpKwargs):
    """Kwargs spread into `ttnn.experimental.deepseek_moe_reduce_scatter`.

    `cluster_axis` is adopted from the parent's `rs_cluster_axis`
    (= `1 - top.cluster_axis`) via a `validation_alias` — reduce-scatter runs on
    the replicated axis, not the dispatch axis.
    """

    output_memory_config: ttnn.MemoryConfig = Field(default_factory=lambda: ttnn.DRAM_MEMORY_CONFIG)
    dim: int
    num_links: int
    topology: ttnn.Topology
    cluster_axis: int = Field(validation_alias="rs_cluster_axis")

    @classmethod
    def adopt_fields(cls) -> set[str]:
        return {"num_links", "topology", "rs_cluster_axis"}


class ReduceScatterConfig(_TTOpKwargs):
    """Kwargs spread into `ttnn.reduce_scatter` (generic fallback path).

    `cluster_axis` is adopted from the parent's `rs_cluster_axis` via a
    `validation_alias`, same reasoning as the deepseek path.
    """

    dim: int
    math_op: Any = None
    num_links: int
    cluster_axis: int = Field(validation_alias="rs_cluster_axis")
    topology: ttnn.Topology
    memory_config: Optional[ttnn.MemoryConfig] = None

    @classmethod
    def adopt_fields(cls) -> set[str]:
        return {"num_links", "topology", "rs_cluster_axis"}


class StateConfig(_TTOpKwargs):
    """Routing / weight-prep params for `_TTMoEDecodeState`."""

    mesh_shape: tuple[int, int]
    cluster_axis: int
    has_bias: bool
    expert_mapping: list[int]
    shared_expert_ids_to_devices: Optional[dict[int, list[int]]] = None

    @classmethod
    def adopt_fields(cls) -> set[str]:
        return {"mesh_shape", "cluster_axis", "has_bias"}


class BuffersConfig(_TTOpKwargs):
    """Sizing and placement params for `_TTMoEDecodeBuffers` allocation."""

    mesh_shape: tuple[int, int]
    cluster_axis: int
    batch_per_device: int
    hidden_size: int
    effective_experts_k: int
    shard_dim: int = 0
    compute_tilize_drain_core: ttnn.CoreCoord = Field(default_factory=lambda: ttnn.CoreCoord(6, 9))

    @classmethod
    def adopt_fields(cls) -> set[str]:
        return {"mesh_shape", "cluster_axis", "batch_per_device", "hidden_size", "effective_experts_k"}


class TTMoEDecodeConfig(BaseModel):
    """Top-level config for `TTMoEDecode`.

    Source of truth for shared and derived fields. The pre-validator builds a
    dict of all parent-visible values (real top-level fields plus derived ones
    like `rs_cluster_axis`, `split_size`, `effective_experts_k`) and asks each
    sub-config — via its `adopt_fields()` classmethod — which of those it wants
    injected before pydantic validates it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    # Mesh / shape
    mesh_shape: tuple[int, int]
    cluster_axis: int
    batch_per_device: int
    hidden_size: int
    select_experts_k: int
    num_shared_experts: int
    has_bias: bool = False

    # Shared ccl kwargs adopted by sub-configs
    num_links: int
    topology: ttnn.Topology

    # Top-level memory configs
    dispatch_input_memory_config: ttnn.MemoryConfig = Field(default_factory=lambda: ttnn.L1_MEMORY_CONFIG)
    dispatch_input_expert_scores_memory_config: Optional[ttnn.MemoryConfig] = None

    # Per-op sub-configs
    dispatch: DispatchConfig
    compute: ComputeConfig
    post_combine_tilize: PostCombineTilizeConfig
    tilize_with_val_padding: TilizeWithValPaddingConfig
    reduce: ReduceConfig
    deepseek_moe_reduce_scatter: DeepseekMoEReduceScatterConfig
    reduce_scatter: ReduceScatterConfig
    state: StateConfig
    buffers: BuffersConfig

    # Top-level fields exposed to sub-configs as-is (passthrough); their values
    # are copied straight from input. Derived/special-case values are computed
    # from these in `_adoptable()` below.
    _ADOPTABLE_PASSTHROUGH: ClassVar[tuple[str, ...]] = (
        "mesh_shape",
        "cluster_axis",
        "batch_per_device",
        "hidden_size",
        "select_experts_k",
        "num_shared_experts",
        "has_bias",
        "num_links",
        "topology",
    )

    @classmethod
    def _adoptable(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Full lookup of values sub-configs may adopt by name."""
        return {
            **{name: data[name] for name in cls._ADOPTABLE_PASSTHROUGH if name in data},
            # has_bias has a top-level default; pydantic only fills it in *after*
            # this mode='before' validator, so apply the default ourselves here.
            "has_bias": data.get("has_bias", False),
            # derived
            "rs_cluster_axis": 1 - data["cluster_axis"],
            "split_size": data["hidden_size"] // data["mesh_shape"][1 - data["cluster_axis"]],
            "effective_experts_k": data["select_experts_k"] + data["num_shared_experts"],
        }

    @model_validator(mode="before")
    @classmethod
    def _adopt_shared_into_subconfigs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # bail if any field needed to build the adoptable lookup is missing —
        # let pydantic raise about it instead
        if any(data.get(name) is None for name in cls._ADOPTABLE_PASSTHROUGH if name != "has_bias"):
            return data

        adoptable = cls._adoptable(data)

        for field_name, field_info in cls.model_fields.items():
            annotation = field_info.annotation
            if not _is_subconfig(annotation):
                continue
            if not hasattr(annotation, "adopt_fields"):
                continue

            additions = {name: adoptable[name] for name in annotation.adopt_fields() if name in adoptable}
            if not additions:
                continue

            cur = data.get(field_name)
            if isinstance(cur, BaseModel):
                data[field_name] = {**cur.model_dump(), **additions}
            elif isinstance(cur, dict):
                data[field_name] = {**cur, **additions}
            # else: leave it; pydantic will produce a typed error

        # Fill in data-dependent memory config defaults where the user didn't provide one.
        if data.get("dispatch_input_expert_scores_memory_config") is None:
            data["dispatch_input_expert_scores_memory_config"] = _default_dispatch_input_expert_scores_memory_config(
                tokens_per_device=adoptable["batch_per_device"],  # seq=1 for decode
                select_experts_k=adoptable["select_experts_k"],
            )

        post_combine_default = _default_post_combine_tilize_memory_config(adoptable["effective_experts_k"])
        _fill_default_if_missing(data, "post_combine_tilize", "output_memory_config", post_combine_default)
        _fill_default_if_missing(data, "tilize_with_val_padding", "memory_config", post_combine_default)

        return data
