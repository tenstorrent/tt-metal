# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Ministral3 RMSNorm wrapper (attention_norm / ffn_norm meta keys).

from __future__ import annotations

import os

import ttnn
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.common import Mode

_TILE = 32
_PREFILL_NORM_M_CAP = 128  # match TtMinistralMLP; keeps sharded norm CBs within L1


def _distributed_rmsnorm_shard_count(mesh_device) -> int:
    mesh_shape = list(mesh_device.shape)
    return mesh_shape[1] if mesh_shape[1] > 1 else mesh_shape[0]


def validate_distributed_rmsnorm_gamma(torch_weight, dim: int, mesh_device, *, tile: int = _TILE) -> None:
    """Fail fast if gamma tile layout or mesh shard count is incompatible with ShardTensor2dMesh."""
    expected_shape = (1, 1, dim // tile, tile)
    assert tuple(torch_weight.shape) == expected_shape, (
        f"Distributed RMSNorm gamma shape mismatch: expected {expected_shape} "
        f"(dim={dim}, tile={tile}), got {tuple(torch_weight.shape)}. "
        "Gamma must be reshaped to [1, 1, hidden_tiles, tile] before mesh sharding."
    )
    n_shards = _distributed_rmsnorm_shard_count(mesh_device)
    hidden_tiles = dim // tile
    assert hidden_tiles % n_shards == 0, (
        f"Distributed RMSNorm gamma cannot shard evenly: hidden_tiles={hidden_tiles} "
        f"(dim={dim}), n_shards={n_shards} (mesh_shape={list(mesh_device.shape)}). "
        f"Require (dim // {tile}) % n_shards == 0."
    )


def _block_sharded_prefill_norm_enabled() -> bool:
    for key in (
        "TT_MINISTRAL3_BLOCK_SHARDED_NORM",
        "TT_MINISTRAL3_HEIGHT_SHARDED_NORM",
        "TT_MINISTRAL3_WIDTH_SHARD_NORM",
    ):
        val = os.environ.get(key)
        if val is not None:
            return val.strip().lower() not in ("0", "false", "no")
    return True


def _prefill_chunk_seq(full_seq_len: int) -> tuple[int, int]:
    if full_seq_len <= _PREFILL_NORM_M_CAP:
        return 1, full_seq_len
    chunk = _PREFILL_NORM_M_CAP
    while chunk > 1 and full_seq_len % chunk != 0:
        chunk -= 1
    return full_seq_len // chunk, chunk


def _padded_seq_len(seq_len: int) -> int:
    return ((int(seq_len) + _TILE - 1) // _TILE) * _TILE


def _block_shard_grid(args, seq_len: int) -> ttnn.CoreGrid:
    row_tiles = _padded_seq_len(seq_len) // _TILE
    col_tiles = int(args.dim) // _TILE
    rows, cols = args.find_prefill_grid(row_tiles, col_tiles)
    return ttnn.CoreGrid(y=rows, x=cols)


def ministral_prefill_block_shard_mem_cfg(args, seq_len: int) -> ttnn.MemoryConfig:
    """BLOCK-sharded L1 layout shared by prefill RMSNorm output and QKV/FF1 matmul in0."""
    return _block_sharded_norm_configs(args, seq_len)[0]


def ministral_prefill_block_shard_grid(args, seq_len: int) -> ttnn.CoreGrid:
    return _block_shard_grid(args, seq_len)


def _block_sharded_norm_configs(
    args, seq_len: int
) -> tuple[ttnn.MemoryConfig, ttnn.LayerNormShardedMultiCoreProgramConfig]:
    padded_seq = _padded_seq_len(seq_len)
    dim = int(args.dim)
    grid = _block_shard_grid(args, seq_len)
    row_tiles = padded_seq // _TILE
    col_tiles = dim // _TILE
    per_core_m = row_tiles // grid.y
    per_core_k = col_tiles // grid.x

    sharded_mem_cfg = ttnn.create_sharded_memory_config(
        (1, 1, padded_seq, dim),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    subblock_w = 4
    while subblock_w > 0:
        if per_core_k % subblock_w == 0:
            break
        subblock_w -= 1

    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        subblock_w=subblock_w,
        block_h=per_core_m,
        block_w=per_core_k,
        inplace=False,
    )
    return sharded_mem_cfg, program_config


def _is_dram_interleaved_tile(x: ttnn.Tensor) -> bool:
    mc = x.memory_config()
    return (
        not mc.is_sharded()
        and mc.buffer_type == ttnn.BufferType.DRAM
        and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
        and x.get_layout() == ttnn.TILE_LAYOUT
    )


def _owned_dram_interleaved_tile(x: ttnn.Tensor) -> ttnn.Tensor:
    """Owned DRAM TILE buffer (to_memory_config, not clone) for block-sharded norm input."""
    if _is_dram_interleaved_tile(x):
        return x
    mc = x.memory_config()
    if mc.is_sharded():
        x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
    else:
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    if x.get_layout() != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return x


def _prefill_output_dram_tile(x: ttnn.Tensor) -> ttnn.Tensor:
    if x.memory_config().is_sharded():
        return ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
    if _is_dram_interleaved_tile(x):
        return x
    return ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)


def _interleaved_tile_to_block_shard(x: ttnn.Tensor, sharded_mem_cfg: ttnn.MemoryConfig) -> ttnn.Tensor:
    """Shard activations without Clone; prefer to_memory_config over interleaved_to_sharded when TILE."""
    if x.memory_config().is_sharded() and x.memory_config() == sharded_mem_cfg:
        return x
    if _is_dram_interleaved_tile(x):
        return ttnn.to_memory_config(x, sharded_mem_cfg)
    return ttnn.interleaved_to_sharded(x, sharded_mem_cfg)


def ministral_prefill_block_shard_add_eligible(seq_len: int) -> bool:
    """Block-sharded L1 residual adds (same eligibility as block-sharded prefill RMSNorm)."""
    if not _block_sharded_prefill_norm_enabled():
        return False
    return int(seq_len) <= _PREFILL_NORM_M_CAP


def ministral_prefill_is_block_sharded_l1(x: ttnn.Tensor) -> bool:
    mc = x.memory_config()
    return (
        mc.is_sharded()
        and mc.buffer_type == ttnn.BufferType.L1
        and mc.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
    )


def ministral_prefill_prepare_block_shard_input(x: ttnn.Tensor, args, seq_len: int) -> ttnn.Tensor:
    """DRAM interleaved or mismatched shard -> BLOCK-sharded L1 (no-op if already on target memcfg)."""
    sharded_mem_cfg = ministral_prefill_block_shard_mem_cfg(args, seq_len)
    return _interleaved_tile_to_block_shard(x, sharded_mem_cfg)


def ministral_prefill_block_shard_norm_config(
    args, seq_len: int, *, output_mem_cfg: ttnn.MemoryConfig | None = None
) -> dict:
    """norm_config for RMSNorm.forward(in_sharded=True) on block-sharded prefill activations."""
    sharded_mem_cfg, program_config = _block_sharded_norm_configs(args, seq_len)
    out_cfg = output_mem_cfg if output_mem_cfg is not None else sharded_mem_cfg
    return {
        "sharded_program_config": program_config,
        "sharded_output_config": out_cfg,
    }


class TtMinistralRMSNorm(RMSNorm):
    # post_attention=False → attention_norm; True → ffn_norm.

    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        tt_ccl,
        *,
        post_attention: bool = False,
        weight_key: str | None = None,
    ):
        self.args = args
        self.post_attention = post_attention
        if weight_key is None:
            weight_key = "ffn_norm" if post_attention else "attention_norm"
        super().__init__(
            device=mesh_device,
            dim=args.dim,
            eps=args.norm_eps,
            state_dict=state_dict,
            weight_key=weight_key,
            layer_num=None,
            state_dict_prefix=args.get_state_dict_prefix("", layer_num),
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            is_distributed=None,
            add_unit_offset=args.rms_norm_add_unit_offset,
            ccl_topology=args.ccl_topology(),
            tt_ccl=tt_ccl,
        )
        self.is_distributed = args.is_distributed_norm
        if self.is_distributed:
            self.weight_distributed = self._load_distributed_weight(
                mesh_device,
                args,
                state_dict,
                weight_key,
                layer_num,
                None if args.dummy_weights else weight_cache_path,
            )

    @staticmethod
    def _load_distributed_weight(
        mesh_device,
        args,
        state_dict,
        weight_key: str,
        layer_num: int | None,
        weight_cache_path,
    ):
        weight_name = f"{args.get_state_dict_prefix('', layer_num)}{weight_key}.weight"
        torch_weight = (
            state_dict[weight_name].unsqueeze(0).view(1, 1, args.dim).reshape([1, 1, args.dim // _TILE, _TILE])
        )
        if args.rms_norm_add_unit_offset:
            torch_weight = torch_weight + 1.0

        validate_distributed_rmsnorm_gamma(torch_weight, args.dim, mesh_device)

        mesh_shape = list(mesh_device.shape)
        shard_dims = (None, 2) if mesh_shape[1] > 1 else (2, None)
        cache_file_name = (
            None
            if weight_cache_path is None
            else weight_cache_path / f"{weight_name}_distributed_{mesh_shape[0]}x{mesh_shape[1]}"
        )
        return ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_file_name,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=mesh_shape),
        )

    def _block_sharded_norm_chunk(
        self,
        x: ttnn.Tensor,
        seq_len: int,
        *,
        keep_sharded_output: bool = False,
        output_mem_cfg: ttnn.MemoryConfig | None = None,
    ) -> ttnn.Tensor:
        sharded_mem_cfg, program_config = _block_sharded_norm_configs(self.args, seq_len)
        if x.memory_config().is_sharded() and x.memory_config() == sharded_mem_cfg:
            x_sharded = x
        else:
            # Materialize slice/views; do not deallocate caller tensor (residual is still live).
            x_owned = _owned_dram_interleaved_tile(x)
            x_sharded = _interleaved_tile_to_block_shard(x_owned, sharded_mem_cfg)
            if x_owned is not x:
                ttnn.deallocate(x_owned)
        out = ttnn.rms_norm(
            x_sharded,
            epsilon=self.eps,
            weight=self.weight,
            program_config=program_config,
            memory_config=sharded_mem_cfg,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        if x_sharded is not x:
            ttnn.deallocate(x_sharded)
        if keep_sharded_output:
            if output_mem_cfg is not None and out.memory_config() != output_mem_cfg:
                out = ttnn.to_memory_config(out, output_mem_cfg)
            return out
        return _prefill_output_dram_tile(out)

    def _forward_block_sharded_prefill(
        self,
        x: ttnn.Tensor,
        *,
        keep_sharded_output: bool = False,
        output_mem_cfg: ttnn.MemoryConfig | None = None,
    ) -> ttnn.Tensor:
        full_seq_len = int(x.shape[-2])
        n_chunks, chunk_seq = _prefill_chunk_seq(full_seq_len)
        if n_chunks == 1:
            return self._block_sharded_norm_chunk(
                x,
                full_seq_len,
                keep_sharded_output=keep_sharded_output,
                output_mem_cfg=output_mem_cfg,
            )

        feat = int(x.shape[-1])
        parts: list[ttnn.Tensor] = []
        for start in range(0, full_seq_len, chunk_seq):
            end = start + chunk_seq
            sl = ttnn.slice(
                x,
                (0, 0, start, 0),
                (1, 1, end, feat),
                memory_config=x.memory_config(),
            )
            parts.append(
                self._block_sharded_norm_chunk(
                    sl,
                    chunk_seq,
                    keep_sharded_output=False,
                    output_mem_cfg=None,
                )
            )
            ttnn.deallocate(sl)

        out = parts[0]
        for part in parts[1:]:
            prev = out
            out = ttnn.concat([prev, part], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(prev)
            ttnn.deallocate(part)
        if keep_sharded_output and output_mem_cfg is not None:
            return _interleaved_tile_to_block_shard(out, output_mem_cfg)
        return out

    def forward(
        self,
        x: ttnn.Tensor,
        mode: Mode | str,
        in_sharded=False,
        out_sharded=False,
        norm_config=None,
    ) -> ttnn.Tensor:
        if isinstance(mode, str):
            try:
                mode = Mode(mode)
            except ValueError:
                raise ValueError(f"Invalid mode: {mode}")
        elif not isinstance(mode, Mode):
            raise ValueError(f"Invalid mode: {mode}")

        distributed = self.is_distributed and self.is_distributed(mode)
        requested_output_sharded_cfg = None
        if out_sharded and norm_config is not None:
            requested_output_sharded_cfg = norm_config.get("sharded_output_config")

        if mode == Mode.PREFILL and not distributed and _block_sharded_prefill_norm_enabled() and not in_sharded:
            return self._forward_block_sharded_prefill(
                x,
                keep_sharded_output=out_sharded,
                output_mem_cfg=requested_output_sharded_cfg,
            )

        return super().forward(
            x,
            mode,
            in_sharded=in_sharded,
            out_sharded=out_sharded,
            norm_config=norm_config,
        )


__all__ = [
    "TtMinistralRMSNorm",
    "validate_distributed_rmsnorm_gamma",
    "ministral_prefill_block_shard_add_eligible",
    "ministral_prefill_block_shard_mem_cfg",
    "ministral_prefill_block_shard_grid",
    "ministral_prefill_block_shard_norm_config",
    "ministral_prefill_is_block_sharded_l1",
    "ministral_prefill_prepare_block_shard_input",
]
