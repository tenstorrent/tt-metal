# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from math import prod
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


_POST_COMBINE_TILIZE_MAX_CORES_X = 7  # original (deepseek) used 7 cores @ 1024 wide for hidden=7168
_POST_COMBINE_TILIZE_SHARD_WIDTH_MULTIPLE = (
    128  # kernel reads 4 tiles wide; shard_width must be a multiple of 4*TILE_SIZE
)

# RS-axis size that selects the fused deepseek_moe_reduce_scatter path (consumes a list of
# N pre-split outputs from fast_reduce_nc_fused). Any other size falls through to a single
# input ttnn.reduce_scatter, so fast_reduce should produce one output instead of N.
_DEEPSEEK_RS_DP_DIM = 8


def _default_post_combine_tilize_memory_config(
    effective_experts_k: int, hidden_size: int
) -> Optional[ttnn.MemoryConfig]:
    """`post_combine_tilize_output_memory_config` from test_optimized_moe_decode_block.py.

    NdShard L1 grid is `[num_cores_x, effective_experts_k]`. `num_cores_x` and the
    inner shard width are chosen so the shards evenly tile `hidden_size` and the
    width is a multiple of 128 (the kernel's 4-tile read granularity). Prefer the
    widest grid (≤ MAX_CORES_X) that satisfies both. Matches the deepseek default
    of 7×1024 for hidden=7168.

    Returns None when no `num_cores_x` ∈ [1, MAX_CORES_X] yields a width that is
    a multiple of 128 — caller is expected to fall back to the
    `tilize_with_val_padding` path. For gpt_oss (hidden=2880) no split works,
    so this returns None.
    """
    num_cores_x = None
    for n in range(_POST_COMBINE_TILIZE_MAX_CORES_X, 0, -1):
        if hidden_size % n != 0:
            continue
        width = hidden_size // n
        if width % _POST_COMBINE_TILIZE_SHARD_WIDTH_MULTIPLE == 0:
            num_cores_x = n
            break
    if num_cores_x is None:
        return None
    shard_width = hidden_size // num_cores_x
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.L1,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[ttnn.TILE_SIZE, shard_width],
            grid=ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, effective_experts_k - 1))}
            ),
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
    expert_mapping: list[int] | str
    num_routed_experts: int
    num_shared_experts: int
    shared_expert_ids_to_devices: Optional[dict[int, list[int]] | str] = None

    @model_validator(mode="before")
    @classmethod
    def _expand_convenience_keywords(cls, data: Any) -> Any:
        """Expand the string-form convenience keywords for `expert_mapping` and
        `shared_expert_ids_to_devices` into their concrete dict/list forms.

        Runs before pydantic validates the field types, so we can replace strings
        in the input dict without bumping into the frozen-model assignment ban.
        """
        if not isinstance(data, dict):
            return data

        mesh_shape = data.get("mesh_shape")
        cluster_axis = data.get("cluster_axis")
        num_routed = data.get("num_routed_experts")
        num_shared = data.get("num_shared_experts", 0)
        if mesh_shape is None or cluster_axis is None or num_routed is None:
            return data  # let pydantic raise about missing fields

        num_devices = prod(mesh_shape)
        if num_routed % num_devices != 0:
            raise ValueError(
                f"num_routed_experts ({num_routed}) must be evenly divisible " f"by num_devices ({num_devices})"
            )

        # expert_mapping: "sequential" → linearized mesh-coord per expert, following
        # the convention in `test_optimized_moe_decode_block.get_linearized_mesh_coord`.
        # cluster_axis=1 (row-major dispatch): linear_id = expert_id // experts_per_device.
        # cluster_axis=0 (column-major dispatch): adjacent experts straddle clusters —
        #   linear_id = device_within_cluster * num_replicated + cluster_id.
        mapping = data.get("expert_mapping")
        if isinstance(mapping, str):
            if mapping == "sequential":
                experts_per_device = num_routed // num_devices
                if cluster_axis == 1:
                    data["expert_mapping"] = [e // experts_per_device for e in range(num_routed)]
                elif cluster_axis == 0:
                    num_dispatch = mesh_shape[0]
                    num_replicated = num_devices // num_dispatch
                    experts_per_cluster = num_routed // num_replicated
                    new_mapping = []
                    for e in range(num_routed):
                        cluster_id = e // experts_per_cluster
                        expert_within_cluster = e % experts_per_cluster
                        device_within_cluster = expert_within_cluster // experts_per_device
                        new_mapping.append(device_within_cluster * num_replicated + cluster_id)
                    data["expert_mapping"] = new_mapping
                else:
                    raise ValueError(f"Unsupported cluster_axis={cluster_axis} for sequential expert_mapping")
            else:
                raise ValueError(f"Unknown expert_mapping keyword: {mapping!r}")

        # shared_expert_ids_to_devices: "fully_replicated" → {num_routed+i: list(range(num_devices)) for i ...}
        shared = data.get("shared_expert_ids_to_devices")
        if isinstance(shared, str):
            if num_shared <= 0:
                raise ValueError(f"shared_expert_ids_to_devices={shared!r} given but num_shared_experts={num_shared}")
            if shared == "fully_replicated":
                data["shared_expert_ids_to_devices"] = {
                    num_routed + i: list(range(num_devices)) for i in range(num_shared)
                }
            else:
                raise ValueError(f"Unknown shared_expert_ids_to_devices keyword: {shared!r}")
        elif isinstance(shared, dict):
            if num_shared != len(shared):
                raise ValueError(
                    f"shared_expert_ids_to_devices has {len(shared)} entries but " f"num_shared_experts={num_shared}"
                )

        return data

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

    # Post-combine path selection. True → fused `deepseek_moe_post_combine_tilize`
    # (requires `batch_per_device == TILE_SIZE` and a valid sharded memory config).
    # False → generic `tilize_with_val_padding` fallback. Auto-computed in the
    # pre-validator from those preconditions when left as None.
    use_post_combine_tilize: Optional[bool] = None

    # Number of outputs fast_reduce_nc_fused will produce. Equals the RS-axis size
    # only when the deepseek RS path is taken; otherwise 1 (downstream ttnn.reduce_scatter
    # consumes a single tensor and splits internally). Auto-computed in pre-validator.
    num_fast_reduce_outputs: Optional[int] = None

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

        # Pick num_fast_reduce_outputs and split_size based on which post-compute RS
        # path forward() will take:
        #   - rs_axis == _DEEPSEEK_RS_DP_DIM → deepseek_moe_reduce_scatter consumes the
        #     full list, so fast_reduce pre-splits into N=rs_axis_size outputs. Each
        #     per-output split must be TILE_SIZE-aligned (fast_reduce's constraint).
        #   - otherwise → ttnn.reduce_scatter (or SKIP) takes a single tensor and does
        #     the per-device split internally. fast_reduce produces N=1 wide output
        #     covering the whole hidden dim; padding it to a multiple of
        #     num_replicated * TILE_SIZE keeps each post-RS device chunk tile-aligned.
        num_replicated = resolved["mesh_shape"][1 - resolved["cluster_axis"]]
        if num_replicated == _DEEPSEEK_RS_DP_DIM:
            num_fast_reduce_outputs = num_replicated
            align_unit = ttnn.TILE_SIZE
        else:
            num_fast_reduce_outputs = 1
            align_unit = ttnn.TILE_SIZE * num_replicated
        pre_split_chunk = resolved["hidden_size"] // num_fast_reduce_outputs
        split_size = ((pre_split_chunk + align_unit - 1) // align_unit) * align_unit

        return {
            **resolved,
            # derived
            "rs_cluster_axis": 1 - resolved["cluster_axis"],
            "split_size": split_size,
            "num_fast_reduce_outputs": num_fast_reduce_outputs,
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

        post_combine_default = _default_post_combine_tilize_memory_config(
            adoptable["effective_experts_k"], adoptable["hidden_size"]
        )
        # Only fill the post_combine_tilize default if we actually have one — otherwise
        # leave the field None and let the tilize_with_val_padding fallback own it.
        if post_combine_default is not None:
            _fill_default_if_missing(data, "post_combine_tilize", "output_memory_config", post_combine_default)
            _fill_default_if_missing(data, "tilize_with_val_padding", "memory_config", post_combine_default)

        # Auto-pick the post-combine path when the user didn't pin it.
        # `deepseek_moe_post_combine_tilize` requires batch_per_device == TILE_SIZE AND
        # a sharded memory config (either default-computable or user-supplied).
        if data.get("use_post_combine_tilize") is None:
            user_pct = data.get("post_combine_tilize")
            user_mc_present = (isinstance(user_pct, dict) and user_pct.get("output_memory_config") is not None) or (
                isinstance(user_pct, BaseModel) and getattr(user_pct, "output_memory_config", None) is not None
            )
            data["use_post_combine_tilize"] = adoptable["batch_per_device"] == ttnn.TILE_SIZE and (
                post_combine_default is not None or user_mc_present
            )

        if data.get("num_fast_reduce_outputs") is None:
            data["num_fast_reduce_outputs"] = adoptable["num_fast_reduce_outputs"]

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
