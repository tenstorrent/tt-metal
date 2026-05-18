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


import ttnn
from models.common.modules.moe.tt_moe_decode_config_schemas import (
    ActivationFunction,
    CoreCoord,
    CoreRangeSet,
    DispatchAlgorithm,
    MemoryConfig,
    Topology,
    WorkerMode,
)


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
    drain_sync_tilizer_core: Optional[CoreCoord] = None
    worker_mode: WorkerMode = ttnn.WorkerMode.DIRECT
    dispatch_algorithm: DispatchAlgorithm = ttnn.DispatchAlgorithm.SPARSE_MCAST_SHORTEST_PATH

    @classmethod
    def adopt_fields(cls) -> set[str]:
        return {"cluster_axis", "num_links"}


class ComputeConfig(_TTOpKwargs):
    """Kwargs spread into `ttnn.experimental.moe_compute`."""

    output_height_shard_dim: int = 4
    cluster_axis: int
    mux_core_range_set: CoreRangeSet = Field(
        default_factory=lambda: ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 3))})
    )
    has_bias: bool
    activation_type: ActivationFunction
    intermediate_size: int

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

    output_memory_config: Optional[MemoryConfig] = None


class TilizeWithValPaddingConfig(_TTOpKwargs):
    """Kwargs spread into `ttnn.tilize_with_val_padding`
    (taken when `batch_per_device != ttnn.TILE_SIZE`).

    `memory_config` shares the same default as `PostCombineTilizeConfig.output_memory_config`.
    """

    memory_config: Optional[MemoryConfig] = None


class ReduceConfig(_TTOpKwargs):
    """Kwargs spread into `ttnn.experimental.deepseek_moe_fast_reduce_nc_fused`."""

    reduce_dim: int = 0
    cluster_axis: int
    split_size: int
    output_memory_config: MemoryConfig = Field(default_factory=_default_fast_reduce_output_memory_config)
    num_shared_experts: int
    shared_expert_scale: float

    @classmethod
    def adopt_fields(cls) -> set[str]:
        return {"cluster_axis", "split_size", "num_shared_experts"}


class DeepseekMoEReduceScatterConfig(_TTOpKwargs):
    """Kwargs spread into `ttnn.experimental.deepseek_moe_reduce_scatter`.

    `cluster_axis` is adopted from the parent's `rs_cluster_axis`
    (= `1 - top.cluster_axis`) via a `validation_alias` — reduce-scatter runs on
    the replicated axis, not the dispatch axis.
    """

    output_memory_config: MemoryConfig = Field(default_factory=lambda: ttnn.DRAM_MEMORY_CONFIG)
    dim: int = -1
    num_links: int
    topology: Topology
    cluster_axis: int = Field(validation_alias="rs_cluster_axis")

    @classmethod
    def adopt_fields(cls) -> set[str]:
        return {"num_links", "topology", "rs_cluster_axis"}


class ReduceScatterConfig(_TTOpKwargs):
    """Kwargs spread into `ttnn.reduce_scatter` (generic fallback path).

    `cluster_axis` is adopted from the parent's `rs_cluster_axis` via a
    `validation_alias`, same reasoning as the deepseek path.
    """

    dim: int = -1
    math_op: Any = None
    num_links: int
    cluster_axis: int = Field(validation_alias="rs_cluster_axis")
    topology: Topology
    memory_config: Optional[MemoryConfig] = None

    @classmethod
    def adopt_fields(cls) -> set[str]:
        return {"num_links", "topology", "rs_cluster_axis"}


class ExpertStateConfig(_TTOpKwargs):
    """Routing / weight-prep params for `_TTMoEDecodeState`."""

    mesh_shape: tuple[int, int]
    cluster_axis: int
    has_bias: bool
    expert_mapping: list[int]
    num_routed_experts: int
    num_shared_experts: int
    shared_expert_ids_to_devices: Optional[dict[int, list[int]]] = None

    @classmethod
    def adopt_fields(cls) -> set[str]:
        return {"mesh_shape", "cluster_axis", "has_bias", "num_routed_experts", "num_shared_experts"}


class BuffersConfig(_TTOpKwargs):
    """Sizing and placement params for `_TTMoEDecodeBuffers` allocation."""

    mesh_shape: tuple[int, int]
    cluster_axis: int
    batch_per_device: int
    hidden_size: int
    effective_experts_k: int
    shard_dim: int = 0
    compute_tilize_drain_core: CoreCoord = Field(default_factory=lambda: ttnn.CoreCoord(6, 9))

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
    num_routed_experts: int
    num_shared_experts: int
    has_bias: bool

    # Shared ccl kwargs adopted by sub-configs
    num_links: int = 4
    topology: Topology = ttnn.Topology.Ring

    # Top-level memory configs
    dispatch_input_memory_config: MemoryConfig = Field(default_factory=lambda: ttnn.L1_MEMORY_CONFIG)
    dispatch_input_expert_scores_memory_config: Optional[MemoryConfig] = None

    # Per-op sub-configs
    dispatch: DispatchConfig
    compute: ComputeConfig
    post_combine_tilize: PostCombineTilizeConfig
    tilize_with_val_padding: TilizeWithValPaddingConfig
    reduce: ReduceConfig
    deepseek_moe_reduce_scatter: DeepseekMoEReduceScatterConfig
    reduce_scatter: ReduceScatterConfig
    experts: ExpertStateConfig
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
        "num_routed_experts",
        "num_shared_experts",
        "has_bias",
        "num_links",
        "topology",
    )

    @classmethod
    def _adoptable(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Full lookup of values sub-configs may adopt by name.

        For passthrough fields, fall back to the field's declared default when
        missing — pydantic only fills defaults *after* this mode='before'
        validator, so we apply them ourselves so sub-config adoption sees them.
        """
        resolved: dict[str, Any] = {}
        for name in cls._ADOPTABLE_PASSTHROUGH:
            finfo = cls.model_fields[name]
            if name in data and data[name] is not None:
                resolved[name] = data[name]
            elif finfo.default_factory is not None:
                resolved[name] = finfo.default_factory()
            elif not finfo.is_required():
                resolved[name] = finfo.default
            # else: required & missing — bail check upstream returned already

        return {
            **resolved,
            # derived
            "rs_cluster_axis": 1 - resolved["cluster_axis"],
            "split_size": resolved["hidden_size"] // resolved["mesh_shape"][1 - resolved["cluster_axis"]],
            "effective_experts_k": resolved["select_experts_k"] + resolved["num_shared_experts"],
        }

    @model_validator(mode="before")
    @classmethod
    def _adopt_shared_into_subconfigs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # Sub-configs whose required fields are entirely either (a) adopted from
        # the parent or (b) defaulted at the sub-config level can be omitted from
        # input — slot in an empty dict so the adoption + default-fill logic below
        # has something to populate.
        for field_name in (
            "dispatch",
            "compute",
            "post_combine_tilize",
            "tilize_with_val_padding",
            "deepseek_moe_reduce_scatter",
            "reduce_scatter",
            "buffers",
        ):
            data.setdefault(field_name, {})

        # bail if any *truly required* passthrough is missing — let pydantic raise
        if any(data.get(name) is None for name in cls._ADOPTABLE_PASSTHROUGH if cls.model_fields[name].is_required()):
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

    # ---- YAML round-trip ----

    def to_yaml(self, *, exclude_defaults: bool = True, exclude_none: bool = True, **dump_kwargs) -> str:
        """Serialize to a minimal YAML string.

        Defaults to skipping fields equal to their default and `None` fields, so
        the output only carries values the user actually configured. ttnn objects
        (MemoryConfig, CoreCoord, etc.) are serialized via their `to_json`
        representation; enums by their name.
        """
        import yaml

        return yaml.safe_dump(
            self.model_dump(mode="json", exclude_defaults=exclude_defaults, exclude_none=exclude_none, **dump_kwargs),
            sort_keys=False,
        )

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "TTMoEDecodeConfig":
        """Load a config from YAML.

        The pre-validators on each field re-hydrate ttnn objects from their dict
        / string representations via `from_json` (and enums by their name).
        """
        import yaml

        return cls.model_validate(yaml.safe_load(yaml_str))
