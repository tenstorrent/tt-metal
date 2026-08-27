# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTTv2 rotary setup for the canonical Wormhole Galaxy (8, 4) mesh."""

from dataclasses import dataclass, replace
from typing import Any

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight
from models.common.tensor_utils import TILE_SIZE, get_rot_transformation_mat, parse_shard_dims_from_mesh_mapper_config

_GALAXY_MESH_SHAPE = (8, 4)


@dataclass(frozen=True)
class RotarySetup2DConfig:
    """Static RoPE data and Galaxy decode placement.

    Llama scaling and Qwen theta choices are represented by the supplied
    cos/sin tables. The scalar fields retain their provenance for validation
    and serialization without introducing model-family branches.
    """

    cos_matrix: LazyWeight
    sin_matrix: LazyWeight
    max_batch_size: int
    head_dim: int | None = None
    mesh_device: ttnn.MeshDevice | None = None
    users_per_column: int = 8
    use_qk_fused: bool = False
    rope_theta: float = 10000.0
    rope_scaling_factor: float | None = None
    original_context_len: int | None = None
    datatype: ttnn.DataType = ttnn.bfloat16
    core_grid: Any | None = None
    batch_grid: Any | None = None
    decode_index_mapper_config: ttnn.MeshMapperConfig | None = None
    decode_trans_mat_memcfg: ttnn.MemoryConfig | None = None
    decode_cos_sin_memcfg: ttnn.MemoryConfig | None = None
    prefill_cos_sin_memcfg: ttnn.MemoryConfig | None = None
    _decode_trans_mat: LazyWeight | None = None
    _prefill_trans_mat: LazyWeight | None = None

    def is_resolved(self) -> bool:
        optional = {"rope_scaling_factor", "original_context_len"}
        return all(getattr(self, field) is not None for field in self.__dataclass_fields__ if field not in optional)


class RotarySetup2D(LightweightModule):
    """Owns lazy RoPE tables and transformation matrices for Galaxy."""

    def __init__(
        self,
        cos_matrix: LazyWeight,
        sin_matrix: LazyWeight,
        max_batch_size: int,
        *,
        core_grid: Any | None = None,
        batch_grid: Any | None = None,
    ):
        super().__init__()
        self.config = _resolve_rope2d_config(
            RotarySetup2DConfig(
                cos_matrix,
                sin_matrix,
                max_batch_size,
                core_grid=core_grid,
                batch_grid=batch_grid,
            )
        )
        self._device_weights_loaded = False

    @classmethod
    def from_config(cls, config: RotarySetup2DConfig):
        instance = object.__new__(cls)
        super(RotarySetup2D, instance).__init__()
        instance.config = _resolve_rope2d_config(config)
        instance._device_weights_loaded = False
        return instance

    def load_device_weights(self) -> None:
        if self._device_weights_loaded:
            return
        self.cos_matrix = self.config.cos_matrix.get_device_weight()
        self.sin_matrix = self.config.sin_matrix.get_device_weight()
        self.cos_matrix_prefill = _materialize_table_copy(self.config.cos_matrix)
        self.sin_matrix_prefill = _materialize_table_copy(self.config.sin_matrix)
        self.transformation_mat = self.config._decode_trans_mat.get_device_weight()
        self.transformation_mat_prefill = self.config._prefill_trans_mat.get_device_weight()
        self._device_weights_loaded = True

    def get_rot_idxs(self, position_idxs: "torch.Tensor", on_host: bool = False) -> ttnn.Tensor:
        return prepare_rot_idxs(self.config, position_idxs, on_host=on_host)

    def decode_forward(self, rot_idxs: "ttnn.Tensor | torch.Tensor") -> list[ttnn.Tensor]:
        self.load_device_weights()
        owns_rot_idxs = not isinstance(rot_idxs, ttnn.Tensor)
        if owns_rot_idxs:
            rot_idxs = prepare_rot_idxs(self.config, rot_idxs)
        else:
            _validate_decode_indices(rot_idxs, self.config)
        cos = sin = None
        try:
            cos = ttnn.embedding(
                rot_idxs,
                self.cos_matrix,
                layout=ttnn.TILE_LAYOUT,
                memory_config=self.config.decode_cos_sin_memcfg,
            )
            sin = ttnn.embedding(
                rot_idxs,
                self.sin_matrix,
                layout=ttnn.TILE_LAYOUT,
                memory_config=self.config.decode_cos_sin_memcfg,
            )
            rows = self.config.users_per_column * (2 if self.config.use_qk_fused else 1)
            cos = _reshape_decode_embedding(cos, rows)
            sin = _reshape_decode_embedding(sin, rows)
            return [ttnn.unsqueeze_to_4D(cos), ttnn.unsqueeze_to_4D(sin)]
        except BaseException:
            if cos is not None:
                ttnn.deallocate(cos)
            if sin is not None:
                ttnn.deallocate(sin)
            raise
        finally:
            if owns_rot_idxs:
                ttnn.deallocate(rot_idxs)

    def prefill_forward(self, start_pos: int, seq_len: int) -> list[ttnn.Tensor]:
        self.load_device_weights()
        if start_pos < 0 or seq_len <= 0:
            raise ValueError("RotarySetup2D prefill range must be non-negative and non-empty")
        end_pos = start_pos + seq_len
        if end_pos > self.config.cos_matrix.source.shape[-2]:
            raise ValueError(f"RotarySetup2D range [{start_pos}:{end_pos}) exceeds the table")
        cos_view = self.cos_matrix_prefill[:, :, start_pos:end_pos, :]
        sin_view = self.sin_matrix_prefill[:, :, start_pos:end_pos, :]
        cos = ttnn.clone(cos_view, memory_config=self.config.prefill_cos_sin_memcfg)
        try:
            sin = ttnn.clone(sin_view, memory_config=self.config.prefill_cos_sin_memcfg)
        except BaseException:
            ttnn.deallocate(cos)
            raise
        return [cos, sin]

    def forward(self, mode: str, **kwargs) -> list[ttnn.Tensor]:
        if mode == "decode":
            return self.decode_forward(**kwargs)
        if mode == "prefill":
            return self.prefill_forward(**kwargs)
        raise ValueError(f"Unknown mode: {mode!r}; expected 'decode' or 'prefill'")

    def get_both_trans_mats(self) -> dict[str, ttnn.Tensor]:
        self.load_device_weights()
        return {"decode": self.transformation_mat, "prefill": self.transformation_mat_prefill}

    def release(self) -> None:
        failures = []
        for weight in (
            self.config.cos_matrix,
            self.config.sin_matrix,
            self.config._decode_trans_mat,
            self.config._prefill_trans_mat,
        ):
            if weight._value is not None:
                try:
                    ttnn.deallocate(weight._value)
                except BaseException as error:
                    failures.append(error)
                weight._value = None
        for name in (
            "cos_matrix",
            "sin_matrix",
            "cos_matrix_prefill",
            "sin_matrix_prefill",
            "transformation_mat",
            "transformation_mat_prefill",
        ):
            if hasattr(self, name):
                if name in ("cos_matrix_prefill", "sin_matrix_prefill"):
                    try:
                        ttnn.deallocate(getattr(self, name))
                    except BaseException as error:
                        failures.append(error)
                delattr(self, name)
        self._device_weights_loaded = False
        if failures:
            raise failures[0]


def _materialize_table_copy(table: LazyWeight) -> ttnn.Tensor:
    """Return a second, independent device copy of a resolved RoPE table.

    Written from the host source rather than with an on-device ``ttnn.clone``.
    The two are numerically identical - same source tensor, same dtype, same
    layout, same DRAM placement - but they are not interchangeable on Galaxy:

    ``ttnn.clone`` compiles a program over the *full* compute grid, and a
    program may only touch cores that belong to the loaded sub-device manager.
    ``load_device_weights`` is lazy, so on the Galaxy decode path it first runs
    inside ``RotarySetup2D.decode_forward`` - by which time the Galaxy
    prefetcher has loaded its sender/worker sub-device partition, which does not
    cover the whole grid. The clone then aborts with

        TT_FATAL ... Kernel group cores do not match sub device cores
                     for programmable core type TENSIX

    and, because the abort happens inside a multi-sub-device program, it leaves
    the mesh un-drainable: teardown blocks forever in
    ``FDMeshCommandQueue::~FDMeshCommandQueue``. Sealing the Galaxy prefetcher
    loads the decode partition and leaves it loaded, so there is no later moment
    at which a full-grid clone would be safe either.

    A host-to-device write compiles no program and is therefore legal under any
    sub-device manager. This is the only lazy device-weight loader among the 2D
    modules that ran a compute op; the rest already only write.

    No cache entry is requested: this copy must not collide with the decode
    table's cache key, and it is cheap to rewrite.

    **The copy is tilized, and the table it copies is not.** The two modes read
    the table through different ops, and each op accepts exactly one layout:

    * decode calls ``ttnn.embedding(rot_idxs, cos_matrix, layout=TILE)``, which
      requires a **row-major** weight table and *produces* tilized cos/sin;
    * prefill slices the table directly and hands the slice to
      ``ttnn.experimental.rotary_embedding_llama``, which requires **tilized**
      cos/sin:

          TT_FATAL ... cos tensor to rotary embedding must be tilized
          (rotary_embedding_llama_device_operation.cpp:51, cos.layout() == TILE)

      measured on `(8, 4)` at prefill 128 with the row-major copy this used to
      make. The qualified 1D reference agrees: ``get_prefill_rot_mat`` in
      ``models/tt_transformers/tt/common.py`` writes its cos/sin with
      ``layout=ttnn.TILE_LAYOUT``.

    So there is no layout that serves both, and this is not a configuration
    choice - it is one legal layout per consumer. Tilizing here rather than at
    the slice keeps it a host-side write, which is the whole point of this
    function: a device-side ``ttnn.tilize`` would compile a full-grid program
    under the decode sub-device partition and abort exactly as the clone did.

    A tilized slice constrains prefill's ``start_pos`` to a multiple of the tile
    height, which every prefill and chunked-prefill start in this package already
    is.
    """

    return LazyWeight(
        source=table.source,
        device=table.device,
        dtype=table.dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper_config=table.mesh_mapper_config,
        pad_value=table.pad_value,
    ).get_device_weight()


def _resolve_rope2d_config(config: RotarySetup2DConfig) -> RotarySetup2DConfig:
    mesh_device = config.mesh_device or config.cos_matrix.device or ttnn.GetDefaultDevice()
    if mesh_device is None:
        raise ValueError("RotarySetup2D mesh_device must be provided")
    for name, weight in (("cos", config.cos_matrix), ("sin", config.sin_matrix)):
        if weight.device is not None and weight.device is not mesh_device:
            raise ValueError(f"RotarySetup2D {name} table belongs to a different mesh")
    _validate_galaxy_mesh(mesh_device)

    cos_shape = tuple(config.cos_matrix.source.shape)
    sin_shape = tuple(config.sin_matrix.source.shape)
    if cos_shape != sin_shape or len(cos_shape) != 4 or cos_shape[:2] != (1, 1):
        raise ValueError("RotarySetup2D cos/sin tables must have identical shape [1, 1, max_seq_len, head_dim]")
    head_dim = config.head_dim or cos_shape[-1]
    if head_dim != cos_shape[-1] or head_dim % TILE_SIZE:
        raise ValueError(f"RotarySetup2D head_dim must match the table and be tile aligned, got {head_dim}")
    if config.max_batch_size != 32 or config.users_per_column != 8:
        raise ValueError("RotarySetup2D requires batch 32 arranged as 8 users per Galaxy column")
    if config.rope_theta <= 0 or (config.rope_scaling_factor is not None and config.rope_scaling_factor <= 0):
        raise ValueError("RotarySetup2D RoPE theta and scaling factor must be positive")
    if config._decode_trans_mat is not None or config._prefill_trans_mat is not None:
        raise ValueError("RotarySetup2D transformation matrices are module-owned")

    if config.core_grid is None or config.batch_grid is None:
        raise ValueError("RotarySetup2D requires explicit fabric-safe core_grid and batch_grid resources")
    core_grid = config.core_grid
    rows_per_column = config.users_per_column * (2 if config.use_qk_fused else 1)
    batch_grid = config.batch_grid
    if not callable(getattr(batch_grid, "num_cores", None)) or batch_grid.num_cores() != rows_per_column:
        raise ValueError(f"RotarySetup2D batch_grid must contain exactly {rows_per_column} cores")
    decode_trans_memcfg = config.decode_trans_mat_memcfg or ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, TILE_SIZE),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    decode_cos_sin_memcfg = config.decode_cos_sin_memcfg or ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    table_kwargs = dict(
        device=mesh_device,
        dtype=config.datatype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper_config=None,
    )
    cos = resolve_lazy_weight(config.cos_matrix, **table_kwargs)
    sin = resolve_lazy_weight(config.sin_matrix, **table_kwargs)
    decode_trans = LazyWeight(
        source=get_rot_transformation_mat(dhead=TILE_SIZE).repeat(1, 1, rows_per_column, 1),
        device=mesh_device,
        dtype=config.datatype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=decode_trans_memcfg,
        mesh_mapper_config=None,
    )
    prefill_trans = LazyWeight(
        # `TILE_SIZE`, not `head_dim`. `rotary_embedding_llama` applies the
        # transformation one tile at a time and validates it:
        #     TT_FATAL ... Transformation matrix must have 4th dim equal to
        #                  TILE_WIDTH
        #     (rotary_embedding_llama_device_operation.cpp:194,
        #      trans_mat.logical_shape()[-1] == TILE_WIDTH)
        # measured on `(8, 4)` at prefill 128 with `head_dim = 128`. The helper
        # says the same thing in its own docstring - "dhead: Matrix dimension.
        # Must equal TILE_SIZE" - and the qualified reference forces it:
        # `get_rot_transformation_mat` in `models/tt_transformers/tt/common.py`
        # opens with `dhead = 32  # ROPE op uses a single tile`.
        source=get_rot_transformation_mat(dhead=TILE_SIZE),
        device=mesh_device,
        dtype=config.datatype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper_config=None,
    )
    decode_index_mapper = config.decode_index_mapper_config or ttnn.MeshMapperConfig(
        placements=[ttnn.PlacementReplicate(), ttnn.PlacementShard(0)],
        mesh_shape_override=ttnn.MeshShape(*_GALAXY_MESH_SHAPE),
    )
    if parse_shard_dims_from_mesh_mapper_config(decode_index_mapper) != [0] or "PlacementReplicate" not in repr(
        decode_index_mapper
    ):
        raise ValueError("RotarySetup2D decode indices must be replicated over rows and sharded on axis 0 over columns")
    resolved = replace(
        config,
        cos_matrix=cos,
        sin_matrix=sin,
        mesh_device=mesh_device,
        head_dim=head_dim,
        core_grid=core_grid,
        batch_grid=batch_grid,
        decode_index_mapper_config=decode_index_mapper,
        decode_trans_mat_memcfg=decode_trans_memcfg,
        decode_cos_sin_memcfg=decode_cos_sin_memcfg,
        prefill_cos_sin_memcfg=config.prefill_cos_sin_memcfg or ttnn.DRAM_MEMORY_CONFIG,
        _decode_trans_mat=decode_trans,
        _prefill_trans_mat=prefill_trans,
    )
    if not resolved.is_resolved():
        raise ValueError("RotarySetup2D config did not resolve completely")
    return resolved


def prepare_rot_idxs(
    config: RotarySetup2DConfig,
    position_idxs: "torch.Tensor",
    on_host: bool = False,
) -> ttnn.Tensor:
    import torch

    if not config.is_resolved():
        raise ValueError("RotarySetup2D config must be resolved before preparing decode indices")
    if not isinstance(position_idxs, torch.Tensor) or position_idxs.ndim != 1:
        raise TypeError("RotarySetup2D position indices must be a one-dimensional torch tensor")
    if position_idxs.numel() != config.max_batch_size:
        raise ValueError(f"RotarySetup2D requires {config.max_batch_size} decode position indices")
    if position_idxs.is_floating_point() or position_idxs.is_complex():
        raise TypeError("RotarySetup2D position indices must have an integer dtype")
    if torch.any(position_idxs < 0):
        raise ValueError("RotarySetup2D position indices must be non-negative")
    table_len = config.cos_matrix.source.shape[-2]
    if torch.any(position_idxs >= table_len):
        raise ValueError(f"RotarySetup2D position indices must be less than table length {table_len}")

    position_idxs = position_idxs.reshape(-1, config.users_per_column)
    if config.use_qk_fused:
        position_idxs = position_idxs.repeat(1, 2)
    position_idxs = position_idxs.reshape(-1, 1)
    position_idxs = torch.nn.functional.pad(position_idxs, (0, TILE_SIZE - 1), value=0)
    mapper = ttnn.create_mesh_mapper(config.mesh_device, config.decode_index_mapper_config)
    kwargs = dict(
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=mapper,
    )
    if not on_host:
        kwargs.update(device=config.mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.as_tensor(position_idxs, **kwargs)


def _validate_decode_indices(rot_idxs: ttnn.Tensor, config: RotarySetup2DConfig) -> None:
    expected_rows = config.users_per_column * (2 if config.use_qk_fused else 1)
    if tuple(rot_idxs.shape) != (expected_rows, TILE_SIZE):
        raise ValueError(f"RotarySetup2D decode indices must have shape ({expected_rows}, {TILE_SIZE})")
    if rot_idxs.dtype != ttnn.uint32:
        raise TypeError("RotarySetup2D decode indices must have uint32 dtype")


def _reshape_decode_embedding(tensor: ttnn.Tensor, rows: int) -> ttnn.Tensor:
    return ttnn.reshape(
        tensor,
        ttnn.Shape((rows, 1, tensor.shape[-1])),
        ttnn.Shape((rows, TILE_SIZE, tensor.shape[-1])),
    )


def _validate_galaxy_mesh(mesh_device) -> None:
    if tuple(mesh_device.shape) != _GALAXY_MESH_SHAPE:
        raise ValueError(f"RotarySetup2D requires logical mesh shape (8, 4), got {tuple(mesh_device.shape)}")
    if mesh_device.get_num_devices() != 32:
        raise ValueError("RotarySetup2D requires exactly 32 devices")
    if mesh_device.arch() != ttnn.device.Arch.WORMHOLE_B0:
        raise ValueError("RotarySetup2D supports Wormhole only")
