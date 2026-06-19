# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from math import prod
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


def _is_subconfig(annotation: Any) -> bool:
    """True if the annotation is a `BaseModel` subclass.

    `issubclass` raises TypeError on generic aliases like `tuple[int, int]`
    (pydantic's BaseModel metaclass rejects them), so we guard it.
    """
    try:
        return issubclass(annotation, BaseModel)
    except TypeError:
        return False


from ttnn.experimental.moe_compute_utils import get_tilize_drain_core

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
_WH_MAX_CORE_GRID_Y = 9


def _default_post_combine_tilize_memory_config(
    effective_experts_k: int, hidden_size: int
) -> Optional[ttnn.MemoryConfig]:
    """`post_combine_tilize_output_memory_config` from test_optimized_moe_decode_block.py.

    Each expert is one shard-row, `num_cores_x` cores wide (the hidden split). Rows
    stack down a column up to the device height (`_WH_MAX_CORE_GRID_Y + 1`); when
    `effective_experts_k` exceeds that, the overflow wraps into additional
    column-bands of width `num_cores_x`. `num_cores_x` is reduced as needed so the
    bands (`num_cores_x * num_bands`) fit the device width. `num_cores_x` and the
    inner shard width are chosen so the shards evenly tile `hidden_size` and the
    width is a multiple of 128 (the kernel's 4-tile read granularity). Prefer the
    widest grid that satisfies all of these.

    Returns None when no `num_cores_x` yields a width that is a multiple of 128 —
    caller is expected to fall back to the `tilize_with_val_padding` path.
    """

    usable_rows = _WH_MAX_CORE_GRID_Y  # 0-indexed inclusive max → row count
    usable_cols = _POST_COMBINE_TILIZE_MAX_CORES_X
    num_bands = (effective_experts_k + usable_rows - 1) // usable_rows
    # Each band is num_cores_x wide; all bands must fit the compute width side by side.
    max_cores_x = usable_cols // num_bands

    num_cores_x = None
    for n in range(max_cores_x, 0, -1):
        if hidden_size % n != 0:
            continue
        width = hidden_size // n
        if width % _POST_COMBINE_TILIZE_SHARD_WIDTH_MULTIPLE == 0:
            num_cores_x = n
            break
    if num_cores_x is None:
        return None
    shard_width = hidden_size // num_cores_x

    # Lay experts into column-bands: band b holds experts [b*usable_dim, ...), stacked
    # down y, at x-offset b*num_cores_x. A single band (effective_experts_k ≤ usable_rows)
    # reproduces the original single-rectangle grid exactly.
    core_ranges = []
    for band in range(num_bands):
        first_expert = band * usable_rows
        band_height = min(usable_rows, effective_experts_k - first_expert)
        x0 = band * num_cores_x
        core_ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(x0, 0),
                ttnn.CoreCoord(x0 + num_cores_x - 1, band_height - 1),
            )
        )

    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.L1,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[ttnn.TILE_SIZE, shard_width],
            grid=ttnn.CoreRangeSet(core_ranges),
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
    # all_to_all_dispatch_metadata auto-detects num_links when None.
    num_links: Optional[int] = None
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
    # Splatted into moe_compute as `num_shared_experts_per_device`. This is the *physical*
    # per-device shared-expert count, derived by the parent from `shared_expert_ids_to_devices`
    # (see TTMoEDecodeConfig._shared_experts_per_device) — NOT the logical `num_shared_experts`.
    num_shared_experts_per_device: int

    @classmethod
    def adopt_fields(cls) -> set[str]:
        return {"cluster_axis", "has_bias", "num_shared_experts_per_device"}


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
    # num_links is auto-detected by the op when None; topology must be Ring (op requirement).
    num_links: Optional[int] = None
    topology: Topology = ttnn.Topology.Ring
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
    # ttnn.reduce_scatter accepts None for both (auto-determines num_links / fabric topology),
    # so leave these unset rather than forcing a value the caller may not know.
    num_links: Optional[int] = None
    cluster_axis: int = Field(validation_alias="rs_cluster_axis")
    topology: Optional[Topology] = None
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
    # Per-arch tilize drain/sync core for the MoE compute op (WH (6,9) / BH (10,9)).
    # Resolved from the current arch via the commonized moe_compute helper so the
    # default tracks the hardware rather than hardcoding the WH coordinate; callers
    # can still pin an explicit core.
    compute_tilize_drain_core: CoreCoord = Field(default_factory=get_tilize_drain_core)

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
    # (requires `batch_per_device == TILE_SIZE` .
    # False → generic `tilize_with_val_padding` fallback. Auto-computed in the pre-validator from preconditions.
    use_post_combine_tilize: Optional[bool] = None

    # Number of outputs fast_reduce_nc_fused will produce. Equals the RS-axis size
    # only when the deepseek RS path is taken; otherwise 1 (downstream ttnn.reduce_scatter
    # consumes a single tensor and splits internally). Auto-computed in pre-validator.
    num_fast_reduce_outputs: Optional[int] = None

    # Shared ccl kwargs adopted by sub-configs
    num_links: Optional[int] = None
    # Runtime fabric topology (caller-detected). Required: it drives both the fast-reduce
    # output layout and the reduce-scatter op selection, so we never silently guess it.
    topology: Topology

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

    @staticmethod
    def _shared_experts_per_device(shared_expert_ids_to_devices: Any, num_devices: int, num_shared_experts: int) -> int:
        """Number of *physical* shared experts resident on each device.

        This is what `moe_compute` wants as `num_shared_experts_per_device`, and it is
        NOT the logical `num_shared_experts`. It is derived from
        `shared_expert_ids_to_devices` (global shared-expert id → list of hosting device
        linear ids): fully-replicated placement puts every shared expert on every device
        (per-device count == num_shared_experts), while a distributed placement (each
        shared expert on a subset of devices) yields a smaller, uniform per-device count.
        e.g. 2 shared experts each residing on half the devices → 1 per device.
        """
        if not num_shared_experts:
            return 0
        # None or the "fully_replicated" convenience keyword (not yet expanded at this
        # point): every shared expert resides on every device.
        if shared_expert_ids_to_devices is None or isinstance(shared_expert_ids_to_devices, str):
            return num_shared_experts
        counts = [0] * num_devices
        for devices in shared_expert_ids_to_devices.values():
            for device_id in devices:
                counts[device_id] += 1
        if len(set(counts)) != 1:
            raise ValueError(
                "shared_expert_ids_to_devices must place an equal number of shared experts on every "
                f"device to derive num_shared_experts_per_device; got per-device counts {counts}"
            )
        return counts[0]

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
            elif not finfo.is_required() and finfo.default is not None:
                # A None default means "unset" — don't adopt it, or it would clobber the
                # sub-config's own default (e.g. DeepseekMoEReduceScatterConfig.topology=Ring)
                # with None and fail that sub-config's type check.
                resolved[name] = finfo.default
            # else: required & missing — bail check upstream returned already

        # Pick num_fast_reduce_outputs and split_size based on which post-compute RS
        # path forward() will take:
        #   - rs_axis == _DEEPSEEK_RS_DP_DIM AND topology is Ring → deepseek_moe_reduce_scatter
        #     consumes the full list, so fast_reduce pre-splits into N=rs_axis_size outputs.
        #     Each per-output split must be TILE_SIZE-aligned (fast_reduce's constraint).
        #   - otherwise → ttnn.reduce_scatter (or SKIP) takes a single tensor and does
        #     the per-device split internally. fast_reduce produces N=1 wide output
        #     covering the whole hidden dim; padding it to a multiple of
        #     num_replicated * TILE_SIZE keeps each post-RS device chunk tile-aligned.
        # deepseek_moe_reduce_scatter only supports Ring, so on any other topology (or when
        # the caller left topology unset → None) we must use the generic single-output layout.
        num_replicated = resolved["mesh_shape"][1 - resolved["cluster_axis"]]
        if num_replicated == _DEEPSEEK_RS_DP_DIM and resolved.get("topology") == ttnn.Topology.Ring:
            num_fast_reduce_outputs = num_replicated
            align_unit = ttnn.TILE_SIZE
        else:
            num_fast_reduce_outputs = 1
            align_unit = ttnn.TILE_SIZE * num_replicated
        pre_split_chunk = resolved["hidden_size"] // num_fast_reduce_outputs
        split_size = ((pre_split_chunk + align_unit - 1) // align_unit) * align_unit

        # Derive the *physical* per-device shared-expert count moe_compute wants from the
        # experts sub-config's device assignment. Use only `.values()` (device-id lists),
        # so the dict/str/None and JSON-stringified-key forms all work uniformly.
        experts_data = data.get("experts")
        if isinstance(experts_data, BaseModel):
            shared_expert_ids_to_devices = getattr(experts_data, "shared_expert_ids_to_devices", None)
        elif isinstance(experts_data, dict):
            shared_expert_ids_to_devices = experts_data.get("shared_expert_ids_to_devices")
        else:
            shared_expert_ids_to_devices = None
        num_shared_experts_per_device = cls._shared_experts_per_device(
            shared_expert_ids_to_devices, prod(resolved["mesh_shape"]), resolved["num_shared_experts"]
        )

        return {
            **resolved,
            # derived
            "rs_cluster_axis": 1 - resolved["cluster_axis"],
            "split_size": split_size,
            "num_fast_reduce_outputs": num_fast_reduce_outputs,
            "effective_experts_k": resolved["select_experts_k"] + resolved["num_shared_experts"],
            "num_shared_experts_per_device": num_shared_experts_per_device,
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
        else:
            raise ValidationError("post combine tilize is not user configurable")

        if data.get("num_fast_reduce_outputs") is None:
            data["num_fast_reduce_outputs"] = adoptable["num_fast_reduce_outputs"]

        # Auto-fill `dispatch.shared_expert_ids` when shared experts exist. The dispatch
        # kernel needs this list to identify which expert ids are shared (broadcast to
        # the local replica) vs. routed (cross-cluster send). Without it, dispatch
        # treats shared-expert ids as routed, mis-routes them, and stomps memory.
        # By contract (enforced in `map_shared_experts`), shared expert ids are always
        # `[num_routed, num_routed + num_shared)`, regardless of device assignment, so
        # we derive them directly rather than depending on the experts sub-config
        # convenience-keyword expansion (which runs after this validator).
        num_shared = adoptable.get("num_shared_experts", 0)
        if num_shared > 0:
            num_routed = adoptable["num_routed_experts"]
            shared_ids = list(range(num_routed, num_routed + num_shared))
            dispatch_data = data.get("dispatch")
            if isinstance(dispatch_data, dict) and dispatch_data.get("shared_expert_ids") is None:
                dispatch_data["shared_expert_ids"] = shared_ids
            elif isinstance(dispatch_data, BaseModel) and getattr(dispatch_data, "shared_expert_ids", None) is None:
                data["dispatch"] = dispatch_data.model_copy(update={"shared_expert_ids": shared_ids})

        return data

    # ---- Test-only mesh slicing ----

    def with_mesh_shape(self, target_mesh_shape: tuple[int, int]) -> "TTMoEDecodeConfig":
        """Return a new config sized for a column of the original mesh.

        Lets a config that targets a full e.g. 16x4 mesh run on a single column (16x1)
        so correctness tests can run on smaller hardware. The `cluster_axis` dim must
        match the original; the replicated dim must be a divisor of the original.
        Total expert count scales down proportionally so each device keeps the same
        expert load (`num_routed_experts / num_devices` is invariant).

        Derived fields (split_size, num_fast_reduce_outputs, the sub-configs' adopted
        values, data-dependent memory-config defaults, etc.) are recomputed by
        round-tripping through `model_validate` with the new mesh and expert count.
        """
        if tuple(target_mesh_shape) == self.mesh_shape:
            return self

        if target_mesh_shape[self.cluster_axis] != self.mesh_shape[self.cluster_axis]:
            raise ValueError(
                f"cluster_axis dim must be preserved when slicing: "
                f"target[{self.cluster_axis}]={target_mesh_shape[self.cluster_axis]} != "
                f"self[{self.cluster_axis}]={self.mesh_shape[self.cluster_axis]}"
            )
        other = 1 - self.cluster_axis
        if self.mesh_shape[other] % target_mesh_shape[other] != 0:
            raise ValueError(
                f"replicated dim must be a divisor of the original: "
                f"self[{other}]={self.mesh_shape[other]} not divisible by "
                f"target[{other}]={target_mesh_shape[other]}"
            )

        new_num_routed = self.num_routed_experts * target_mesh_shape[other] // self.mesh_shape[other]
        if new_num_routed % prod(target_mesh_shape) != 0:
            raise ValueError(
                f"scaled num_routed_experts ({new_num_routed}) is not divisible by "
                f"target num_devices ({prod(target_mesh_shape)})"
            )

        data = self.model_dump(mode="json", exclude_defaults=True, exclude_none=True)
        data["mesh_shape"] = list(target_mesh_shape)
        data["num_routed_experts"] = new_num_routed
        # Force re-derivation of mesh-derived top-level fields.
        self._drop_derived_fields(data)
        # Re-derive expert routing — the dumped expert_mapping is the expanded
        # "sequential" list for the old mesh and won't address the sliced experts.
        # Explicit (non-sequential) mappings aren't supported by the slicer.
        experts_data = data.setdefault("experts", {})
        if isinstance(experts_data, dict):
            experts_data["expert_mapping"] = "sequential"
            if self.num_shared_experts > 0:
                experts_data["shared_expert_ids_to_devices"] = "fully_replicated"
        # Clear the auto-filled `dispatch.shared_expert_ids` so it re-derives against
        # the new num_routed_experts.
        if isinstance(data.get("dispatch"), dict):
            data["dispatch"].pop("shared_expert_ids", None)
        return type(self).model_validate(data)

    @staticmethod
    def _drop_derived_fields(data: dict) -> None:
        """In-place: remove top-level fields the pre-validator derives (and rejects as
        input) so they re-derive when dumped data is fed back through `model_validate`.
        """
        for derived in ("num_fast_reduce_outputs", "use_post_combine_tilize"):
            data.pop(derived, None)

    # ---- YAML round-trip ----

    def to_yaml(self, *, exclude_defaults: bool = True, exclude_none: bool = True, **dump_kwargs) -> str:
        """Serialize to a minimal YAML string.

        Defaults to skipping fields equal to their default and `None` fields, so
        the output only carries values the user actually configured. ttnn objects
        (MemoryConfig, CoreCoord, etc.) are serialized via their `to_json`
        representation; enums by their name.
        """
        import yaml

        data = self.model_dump(mode="json", exclude_defaults=exclude_defaults, exclude_none=exclude_none, **dump_kwargs)
        self._drop_derived_fields(data)
        return yaml.safe_dump(data, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str: str, topology: ttnn.Topology) -> "TTMoEDecodeConfig":
        """Load a config from YAML.

        `topology` is the runtime fabric topology (the caller detects it from the
        device). It overrides any `topology` authored in the YAML, since the layout
        and reduce-scatter path depend on the actual hardware, not the saved value.

        The pre-validators on each field re-hydrate ttnn objects from their dict
        / string representations via `from_json` (and enums by their name).
        """
        import yaml

        data = yaml.safe_load(yaml_str)
        data["topology"] = topology
        return cls.model_validate(data)
