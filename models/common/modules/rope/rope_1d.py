# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2-style Rotary Position Embedding (RoPE) setup for 1D-topology devices.

RotarySetup1D takes pre-computed cos/sin rotation matrices (as LazyWeight) and
provides efficient position-based lookup at runtime.

Core API:
  - decode_forward(rot_idxs): Look up cos/sin rotation matrices from ttnn.Tensor indices
  - prefill_forward(start_pos, seq_len): Slice cos/sin matrices for prefill
  - get_both_trans_mats(): Decode + prefill transformation matrices (read-only)

Backward-compatible API (will retire with TTTv1):
  - get_rot_idxs(position_idxs): Convert torch position indices to ttnn tensor
  - get_rot_mats(position_idxs): Combined get_rot_idxs + decode_forward

Helper:
  - prepare_rot_idxs(config, position_idxs, on_host): Standalone torch→ttnn index conversion

Note: torch is used at construction time for building transformation matrices.
      The decode_forward() and prefill_forward() methods use only ttnn ops.
"""

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple, Union

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight
from models.common.tensor_utils import TILE_SIZE, get_rot_transformation_mat
from models.common.utility_functions import nearest_32

# =============================================================================
# Config dataclass
# =============================================================================


@dataclass
class Rope1DConfig:
    """
    Configuration for RotarySetup1D.

    Simple usage:
        rope = RotarySetup1D(cos_lw, sin_lw, max_batch_size=32)

    Power API:
        config = Rope1DConfig(cos_lw, sin_lw, max_batch_size=32,
                              device=mesh_device, use_qk_fused=True)
        rope = RotarySetup1D.from_config(config)
    """

    # Required: cos/sin embedding lookup tables as LazyWeight
    # source shape: [1, 1, max_seq_len, head_dim]
    cos_matrix: LazyWeight
    sin_matrix: LazyWeight

    # Required scalars
    max_batch_size: int

    # Optional (derived from weights if None)
    head_dim: int | None = None  # derived from cos_matrix.source.shape[-1]
    device: Any | None = None  # derived from cos_matrix.device
    use_qk_fused: bool = False
    datatype: ttnn.DataType = ttnn.bfloat16

    # Derived (resolved by _resolve_rope_config, None = auto)
    batch_size_per_device_group: int | None = None
    core_grid: Any | None = None
    batch_grid: Any | None = None
    decode_trans_mat_mem_config: Optional[ttnn.MemoryConfig] = None
    cos_sin_shard_mem_config: Optional[ttnn.MemoryConfig] = None

    # Internal: transformation matrix LazyWeights (built by _resolve_rope_config)
    _decode_trans_mat: Optional[LazyWeight] = None
    _prefill_trans_mat: Optional[LazyWeight] = None


# =============================================================================
# RotarySetup1D
# =============================================================================


class RotarySetup1D(LightweightModule):
    """
    Rotary position embedding setup for non-TG (1D) devices.

    Takes pre-computed cos/sin lookup tables as LazyWeight and provides
    efficient position-based rotation matrix lookup at runtime.

    Simple API:
        rope = RotarySetup1D(cos_lw, sin_lw, max_batch_size=32)
        rot_idxs = prepare_rot_idxs(rope.config, position_idxs)
        [cos, sin] = rope.decode_forward(rot_idxs)
        [cos, sin] = rope.prefill_forward(start_pos=0, seq_len=128)

    Power API:
        config = Rope1DConfig(cos_lw, sin_lw, max_batch_size=32, use_qk_fused=True)
        rope = RotarySetup1D.from_config(config)

    Backward-compatible API:
        rope.get_rot_mats(position_idxs)  # accepts torch.Tensor or ttnn.Tensor
        rope.get_rot_idxs(position_idxs, on_host=True)
    """

    def __init__(
        self,
        cos_matrix: LazyWeight,
        sin_matrix: LazyWeight,
        max_batch_size: int,
    ) -> None:
        """
        Simple API — takes cos/sin as LazyWeight, derives all other config.

        Args:
            cos_matrix: LazyWeight with source shape [1, 1, max_seq_len, head_dim].
            sin_matrix: LazyWeight with source shape [1, 1, max_seq_len, head_dim].
            max_batch_size: Maximum batch size for decode mode.
        """
        super().__init__()
        config = Rope1DConfig(
            cos_matrix=cos_matrix,
            sin_matrix=sin_matrix,
            max_batch_size=max_batch_size,
        )
        self.config = _resolve_rope_config(config)
        self._device_weights_loaded = False

    @classmethod
    def from_config(cls, config: Rope1DConfig):
        """Power API — construct from config dataclass with full customization."""
        instance = object.__new__(cls)
        super(RotarySetup1D, instance).__init__()
        instance.config = _resolve_rope_config(config)
        instance._device_weights_loaded = False
        return instance

    def load_device_weights(self):
        """Load cos/sin and transformation matrices to device. Called lazily on first use."""
        if self._device_weights_loaded:
            return

        cfg = self.config

        self.cos_matrix = cfg.cos_matrix.get_device_weight()
        self.sin_matrix = cfg.sin_matrix.get_device_weight()
        self.transformation_mat = cfg._decode_trans_mat.get_device_weight()
        self.transformation_mat_prefill = cfg._prefill_trans_mat.get_device_weight()

        self._device_weights_loaded = True

    # ---- Core API ----

    def decode_forward(self, rot_idxs: ttnn.Tensor) -> List[ttnn.Tensor]:
        """Look up cos/sin rotation matrices for given position indices (decode mode).

        Args:
            rot_idxs: ttnn.Tensor of shape [1, batch], dtype uint32.

        Returns:
            [cos, sin] — height-sharded rotation matrices.
        """
        self.load_device_weights()
        cfg = self.config

        # Send to device if needed
        if rot_idxs.device != cfg.device:
            rot_idxs = ttnn.to_device(rot_idxs, cfg.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Embedding lookup: [1, batch] → [1, batch, head_dim]
        cos = ttnn.embedding(rot_idxs, self.cos_matrix, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rot_idxs, self.sin_matrix, layout=ttnn.TILE_LAYOUT)

        # Reshape: [1, batch, head_dim] → [1, 1, batch, head_dim] → [1, batch, 1[32], head_dim]
        cos = ttnn.unsqueeze_to_4D(cos)
        sin = ttnn.unsqueeze_to_4D(sin)
        cos = ttnn.transpose(cos, 1, 2)
        sin = ttnn.transpose(sin, 1, 2)

        # Trim to actual batch size.
        # When batch_size_per_device_group is not a multiple of TILE_SIZE (e.g., batch=1),
        # the embedding produces nearest_32(batch) rows that need trimming.
        # When tile-aligned (e.g., batch=32), this is a no-op slice.
        cos = cos[:, : cfg.batch_size_per_device_group, :, :]
        sin = sin[:, : cfg.batch_size_per_device_group, :, :]

        # Shard to batch cores
        cos = ttnn.interleaved_to_sharded(cos, cfg.cos_sin_shard_mem_config)
        sin = ttnn.interleaved_to_sharded(sin, cfg.cos_sin_shard_mem_config)

        return [cos, sin]

    def prefill_forward(self, start_pos: int, seq_len: int, pad_to: int | None = None) -> List[ttnn.Tensor]:
        """Slice cos/sin matrices for prefill mode.

        This replaces the duplicated cos/sin slicing logic found in every model's
        prepare_inputs_prefill(). Models no longer need to reach into
        rope_setup.cos_matrix / sin_matrix directly.

        Args:
            start_pos: Starting position in the sequence.
            seq_len: Number of positions to slice (typically the padded input length S).
            pad_to: If set, zero-pad the sequence dim to this length (for SDPA alignment).
                    Ignored when pad_to <= seq_len.

        Returns:
            [cos_slice, sin_slice] — interleaved DRAM tensors,
            shape [1, 1, seq_len (or pad_to), head_dim].
        """
        self.load_device_weights()

        mat_len = self.cos_matrix.shape[2]
        end_pos = start_pos + seq_len
        assert mat_len >= end_pos, f"Requested range [{start_pos}:{end_pos}) exceeds cos/sin table length {mat_len}"

        cos_slice = self.cos_matrix[:, :, start_pos:end_pos, :]
        sin_slice = self.sin_matrix[:, :, start_pos:end_pos, :]

        if pad_to is not None and pad_to > seq_len:
            pad_len = pad_to - seq_len
            padding = [(0, 0)] * 4
            padding[2] = (0, pad_len)
            cos_slice = ttnn.pad(cos_slice, padding=padding, value=0.0)
            sin_slice = ttnn.pad(sin_slice, padding=padding, value=0.0)

        return [cos_slice, sin_slice]

    def forward(self, mode: str, **kwargs) -> List[ttnn.Tensor]:
        """Dispatch to decode_forward or prefill_forward based on mode.

        Args:
            mode: "decode" or "prefill".
            **kwargs: Forwarded to the underlying method.
                decode:  rot_idxs (ttnn.Tensor)
                prefill: start_pos (int), seq_len (int), pad_to (int | None)
        """
        if mode == "decode":
            return self.decode_forward(**kwargs)
        else:
            return self.prefill_forward(**kwargs)

    def get_both_trans_mats(self) -> Dict[str, ttnn.Tensor]:
        """Return both decode and prefill transformation matrices."""
        self.load_device_weights()
        return {"decode": self.transformation_mat, "prefill": self.transformation_mat_prefill}

    # ---- Backward-compatible API (will retire with TTTv1) ----

    def get_rot_idxs(self, position_idxs: "torch.Tensor", on_host: bool = False) -> ttnn.Tensor:
        """Convert torch position indices to ttnn tensor. Delegates to prepare_rot_idxs()."""
        return prepare_rot_idxs(self.config, position_idxs, on_host=on_host)

    def get_rot_mats(
        self,
        position_idxs: "Union[torch.Tensor, ttnn.Tensor]",
        return_rot_idxs: bool = False,
    ) -> "Union[List[ttnn.Tensor], Tuple[List[ttnn.Tensor], ttnn.Tensor]]":
        """Look up cos/sin from torch or ttnn position indices.

        For new code, prefer: rot_idxs = prepare_rot_idxs(...); rope.decode_forward(rot_idxs)
        """
        import torch

        if isinstance(position_idxs, torch.Tensor):
            rot_idxs = self.get_rot_idxs(position_idxs)
        else:
            rot_idxs = position_idxs

        cos_sin = self.decode_forward(rot_idxs)

        if return_rot_idxs:
            return cos_sin, rot_idxs
        return cos_sin

    # ---- Factory for TTTv1 backward compatibility ----

    @classmethod
    def from_model_args(cls, device: Any, args, model_name: str = "unknown"):
        """Factory method for backward compatibility with ModelArgs.

        Args:
            device: MeshDevice or single device.
            args: Model arguments (ModelArgs instance).
            model_name: Model name for logging (default "unknown").
        """
        if args.is_galaxy:
            raise ValueError("RotarySetup1D cannot be used for Galaxy devices.")

        from models.tt_transformers.tt.common import rope_scaling_model_factory

        rope_scaling = None
        if hasattr(args, "rope_scaling_params") and args.rope_scaling_params is not None:
            rope_scaling = rope_scaling_model_factory(
                args.rope_scaling_params, getattr(args, "original_max_context_len", None)
            )

        # Compute cos/sin torch tensors using TTTv1 reference
        from models.tt_transformers.tt.rope import compute_gather_cos_sin

        cos_torch, sin_torch = compute_gather_cos_sin(
            dhead=args.head_dim, end=2 * args.max_seq_len, theta=args.rope_theta, rope_scaling=rope_scaling
        )

        # Wrap in LazyWeight
        cos_lw = LazyWeight(source=cos_torch, device=device)
        sin_lw = LazyWeight(source=sin_torch, device=device)

        config = Rope1DConfig(
            cos_matrix=cos_lw,
            sin_matrix=sin_lw,
            max_batch_size=args.max_batch_size,
            head_dim=args.head_dim,
            device=device,
            use_qk_fused=getattr(args, "use_qk_fused", False),
            datatype=ttnn.bfloat16,
        )
        return cls.from_config(config)


# =============================================================================
# Config resolver
# =============================================================================


def _resolve_rope_config(config: Rope1DConfig) -> Rope1DConfig:
    """Resolve None fields in Rope1DConfig to sensible defaults.

    Fills in:
      - head_dim: derived from cos_matrix.source.shape[-1]
      - device: derived from cos_matrix.device if None
      - batch_size_per_device_group, core_grid, batch_grid
      - decode_trans_mat_mem_config, cos_sin_shard_mem_config
      - cos_matrix/sin_matrix LazyWeight fields: device, memory_config, layout, dtype
      - _decode_trans_mat, _prefill_trans_mat: internal transformation matrix LazyWeights
    """
    to_set = {}

    # --- Phase 1: Foundational fields ---

    head_dim = config.head_dim
    if head_dim is None:
        head_dim = config.cos_matrix.source.shape[-1]
        to_set["head_dim"] = head_dim

    device = config.device
    if device is None:
        device = config.cos_matrix.device
    if device is None:
        device = ttnn.GetDefaultDevice()
    if config.device is None:
        to_set["device"] = device

    assert device is not None, "device must be available at this point!"

    # --- Phase 2: Derived scalars ---

    batch_size_per_device_group = config.batch_size_per_device_group
    if batch_size_per_device_group is None:
        batch_size_per_device_group = config.max_batch_size * 2 if config.use_qk_fused else config.max_batch_size
        to_set["batch_size_per_device_group"] = batch_size_per_device_group

    core_grid = config.core_grid
    if core_grid is None:
        core_grid = device.compute_with_storage_grid_size()
        to_set["core_grid"] = core_grid

    batch_grid = config.batch_grid
    if batch_grid is None:
        batch_grid = ttnn.num_cores_to_corerangeset(batch_size_per_device_group, core_grid, row_wise=True)
        to_set["batch_grid"] = batch_grid

    # --- Phase 3: Memory configs ---

    if config.decode_trans_mat_mem_config is None:
        to_set["decode_trans_mat_mem_config"] = ttnn.create_sharded_memory_config(
            shape=(TILE_SIZE, TILE_SIZE),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    if config.cos_sin_shard_mem_config is None:
        to_set["cos_sin_shard_mem_config"] = ttnn.create_sharded_memory_config(
            shape=(TILE_SIZE, head_dim),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

    # --- Phase 4: Resolve LazyWeights ---

    to_set["cos_matrix"] = resolve_lazy_weight(
        config.cos_matrix,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper_config=None,  # replicated
        layout=ttnn.TILE_LAYOUT,
        dtype=config.datatype,
    )
    to_set["sin_matrix"] = resolve_lazy_weight(
        config.sin_matrix,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper_config=None,  # replicated
        layout=ttnn.TILE_LAYOUT,
        dtype=config.datatype,
    )

    # --- Phase 5: Transformation matrix LazyWeights ---

    decode_trans_mat_mem_config = to_set.get("decode_trans_mat_mem_config", config.decode_trans_mat_mem_config)

    assert config._decode_trans_mat is None, "_decode_trans_mat is internal and should not be set by the user"
    decode_trans_mat_source = get_rot_transformation_mat(dhead=TILE_SIZE).repeat(
        1,
        1,
        batch_size_per_device_group,
        1,
    )
    to_set["_decode_trans_mat"] = LazyWeight(
        source=decode_trans_mat_source,
        device=device,
        memory_config=decode_trans_mat_mem_config,
        mesh_mapper_config=None,  # replicated
        layout=ttnn.TILE_LAYOUT,
        dtype=config.datatype,
    )

    assert config._prefill_trans_mat is None, "_prefill_trans_mat is internal and should not be set by the user"
    prefill_trans_mat_source = get_rot_transformation_mat(dhead=head_dim)
    to_set["_prefill_trans_mat"] = LazyWeight(
        source=prefill_trans_mat_source,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper_config=None,  # replicated
        layout=ttnn.TILE_LAYOUT,
        dtype=config.datatype,
    )

    return replace(config, **to_set)


# =============================================================================
# Standalone helper function
# =============================================================================


def prepare_rot_idxs(
    config: Rope1DConfig,
    position_idxs: "torch.Tensor",
    on_host: bool = False,
) -> ttnn.Tensor:
    """Convert torch position indices to a TTNN rotation index tensor.

    This is the standalone equivalent of RotarySetup1D.get_rot_idxs().
    Useful for pre-allocating host tensors for trace mode.

    Args:
        config: Resolved Rope1DConfig (device must be set).
        position_idxs: 1D torch tensor of position indices, shape [batch].
        on_host: If True, return tensor on host (for later copy_host_to_device_tensor).

    Returns:
        ttnn tensor of shape [1, nearest_32(batch)], dtype uint32.
    """
    import torch

    assert isinstance(position_idxs, torch.Tensor), "Position ids must be a torch tensor"
    assert len(position_idxs.shape) == 1, "position idxs must be a [batch] tensor"
    assert config.device is not None, "config.device must be set (use a resolved config)"
    assert config.batch_size_per_device_group is not None, "config must be resolved (_resolve_rope_config)"

    if config.use_qk_fused:
        position_idxs = position_idxs.repeat(2)
        assert (
            position_idxs.shape[0] == config.batch_size_per_device_group
        ), "Position idxs must match batch_size_per_device_group"

    batch = position_idxs.shape[0]
    position_idxs = position_idxs.reshape(1, batch)
    assert torch.min(position_idxs) >= 0, "Position idxs must be non-negative"

    # Pad to nearest tile boundary
    pad_size = nearest_32(batch) - batch
    position_idxs = torch.nn.functional.pad(position_idxs, (0, pad_size), "constant", 0)

    device = config.device

    if on_host:
        rot_idxs = ttnn.as_tensor(
            position_idxs,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(device),
        )
    else:
        rot_idxs = ttnn.as_tensor(
            position_idxs,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(device),
        )

    return rot_idxs
