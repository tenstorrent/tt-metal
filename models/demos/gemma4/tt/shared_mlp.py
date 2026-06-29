# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Shared/Dense MLP with GeGLU activation.

Each decoder layer has BOTH a shared MLP and routed MoE experts.
Architecture: down_proj(GELU(gate_proj(x)) * up_proj(x))
intermediate_size = 2112, no bias.

HF weight shapes:
  gate_proj.weight: [intermediate_size, hidden_size] = [2112, 2816]
  up_proj.weight:   [intermediate_size, hidden_size] = [2112, 2816]
  down_proj.weight: [hidden_size, intermediate_size] = [2816, 2112]
"""

import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.gemma4.tt.ccl import ccl_allreduce
from models.demos.gemma4.utils.general_utils import get_cache_file_name


def _tiles(dim):
    return (dim + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE


def find_largest_divisor(n, max_divisor=8):
    for i in range(max_divisor, 0, -1):
        if n % i == 0:
            return i
    return 1


def _best_core_grid(n_tiles, max_x, max_y):
    """Largest grid (cx<=max_x, cy<=max_y) whose core count divides n_tiles.

    Exact division keeps per_core_N * num_cores == n_tiles (no over-allocation),
    mirroring experts/_build_sparse_matmul_config.
    """
    best_cores, best = 1, (1, 1)
    for cy in range(1, max_y + 1):
        for cx in range(1, max_x + 1):
            cores = cx * cy
            if n_tiles % cores == 0 and cores > best_cores:
                best_cores, best = cores, (cx, cy)
    return best


def _decode_linear_1d_config(m_tiles, k_tiles, n_tiles, grid_size, is_bh, return_output_memcfg=False):
    """1D (N-parallel) matmul program config for a decode-mode Linear.

    Decode has small M, so we use 1d splitting across N

    Constraints satisfied here:
      - in0_block_w divides k_tiles (K streamed in even chunks)
      - per_core_N * num_cores == n_tiles (grid chosen to divide n_tiles)
      - out_subblock_w divides per_core_N; out_subblock_h divides per_core_M
      - out_subblock_h * out_subblock_w <= arch cap (4 Blackhole / 8 Wormhole)

    When ``return_output_memcfg`` is set, also returns a WIDTH-sharded L1
    MemoryConfig laid out on the *same* (cx, cy) grid the matmul uses, with each
    core holding ``per_core_N`` output tiles. This lets the matmul write its
    result straight into L1 (skipping the DRAM write).
    """
    cx, cy = _best_core_grid(n_tiles, grid_size.x, grid_size.y)
    num_cores = cx * cy

    per_core_M = m_tiles
    per_core_N = n_tiles // num_cores

    # Must divide k_tiles exactly; derive straight from k_tiles rather than the
    # tt_transformers k_tiles//num_cores form, which isn't guaranteed to divide.
    in0_block_w = find_largest_divisor(k_tiles)

    max_subblock = 4 if is_bh else 8
    out_subblock_w = find_largest_divisor(per_core_N, max_subblock)
    out_subblock_h = find_largest_divisor(per_core_M, max(1, max_subblock // out_subblock_w))

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cx, cy),
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    if not return_output_memcfg:
        return program_config

    output_memcfg = ttnn.create_sharded_memory_config(
        shape=(per_core_M * ttnn.TILE_SIZE, per_core_N * ttnn.TILE_SIZE),
        core_grid=ttnn.CoreGrid(x=cx, y=cy),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return program_config, output_memcfg


class SharedMLP:
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        mesh_config,
        ccl_manager=None,
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
    ):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size

        tp = mesh_config.tp if mesh_config else 1
        self.tp = tp
        tp_suffix = f"_tp{tp}" if tp > 1 else ""

        # Per-device intermediate width, tile-aligned. With TP the raw per-device
        # split (e.g. 2112/8 = 264) is not a multiple of TILE_SIZE, so the fused
        # [gate | up] slab would have its split point land mid-tile — slicing it
        # then mixes gate/up lanes within a tile and tanks PCC. We pad each
        # per-device gate/up half (and, symmetrically, the down_proj input rows)
        # up to a tile multiple so the split lands on a tile boundary and every
        # matmul's N/K stays tile-aligned. Mirrors experts/weights.py.
        split = self.intermediate_size // tp
        self.padded_split = ((split + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        self._intermediate_pad = self.padded_split - split if tp > 1 else 0

        # Tag the cache filenames with the weight dtype so that flipping a
        # SharedMLP weight's dtype (e.g. bf16 → bfp8 for DRAM-pressure relief)
        # doesn't collide with a previously-cached file that holds the same
        # logical weight at a different dtype. The rest of the model's cache
        # entries are unaffected and stay reusable across runs.
        _dtype_str = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8"}[dtype]
        dtype_suffix = f"_{_dtype_str}"

        if tp > 1:
            col_mapper = mesh_config.column_parallel(mesh_device)
            row_mapper = mesh_config.row_parallel(mesh_device)
        else:
            col_mapper = None
            row_mapper = None

        if state_dict:
            # Fuse gate_proj + up_proj into a single column-parallel weight so the
            # forward runs ONE matmul producing a [gate | up] slab
            gate_w = state_dict["gate_proj.weight"]  # [intermediate_size, hidden_size]
            up_w = state_dict["up_proj.weight"]  # [intermediate_size, hidden_size]
            pad = self._intermediate_pad
            if tp > 1:
                fused_list = []
                for i in range(tp):
                    wg_chunk = torch.chunk(gate_w, tp, dim=0)[i].transpose(-2, -1)  # [hidden, split]
                    wu_chunk = torch.chunk(up_w, tp, dim=0)[i].transpose(-2, -1)  # [hidden, split]
                    if pad:
                        # Pad each half's N dim with zeros so the slab is
                        # [gate(padded_split) | up(padded_split)] per device
                        wg_chunk = torch.nn.functional.pad(wg_chunk, (0, pad))
                        wu_chunk = torch.nn.functional.pad(wu_chunk, (0, pad))
                    fused_list.append(torch.cat([wg_chunk, wu_chunk], dim=-1))  # [hidden, 2*padded_split]
                gate_up_proj_weight = torch.cat(fused_list, dim=-1).unsqueeze(0).unsqueeze(0)
            else:
                gate_up_proj_weight = (
                    torch.cat([gate_w.transpose(-2, -1), up_w.transpose(-2, -1)], dim=-1).unsqueeze(0).unsqueeze(0)
                )

            # need to pad down_proj so it matches the padding for the gate+up proj
            down_proj_weight = state_dict["down_proj.weight"].transpose(-2, -1)  # [intermediate, hidden]
            if tp > 1 and pad:
                down_chunks = torch.chunk(down_proj_weight, tp, dim=-2)  # each [split, hidden]
                down_chunks = [
                    torch.nn.functional.pad(c, (0, 0, 0, pad)) for c in down_chunks
                ]  # [padded_split, hidden]
                down_proj_weight = torch.cat(down_chunks, dim=-2)  # [tp*padded_split, hidden]
            down_proj_weight = down_proj_weight.unsqueeze(0).unsqueeze(0)
        else:
            gate_up_proj_weight = None
            down_proj_weight = None

        # gate+up fused: column-parallel (shard output dim across TP devices)
        self.gate_up_proj = ttnn.as_tensor(
            gate_up_proj_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_up_proj.weight{tp_suffix}{dtype_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # down: row-parallel (shard input dim, allreduce after)
        self.down_proj = ttnn.as_tensor(
            down_proj_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=row_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"down_proj.weight{tp_suffix}{dtype_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, hidden_states, is_decode=True):
        """
        GeGLU MLP forward with TP support.

        gate+up are fused (column-parallel), down is row-parallel + allreduce.
        In decode the two matmuls use tuned 1D program configs; prefill falls
        back to the ttnn-chosen default (program_config=None).
        """
        gate_up_pc = down_pc = down_out_memcfg = None
        if is_decode:
            grid = self.mesh_device.compute_with_storage_grid_size()
            is_bh = is_blackhole()
            m_tiles = _tiles(hidden_states.shape[-2])
            gate_up_pc = _decode_linear_1d_config(
                m_tiles, _tiles(self.gate_up_proj.shape[-2]), _tiles(self.gate_up_proj.shape[-1]), grid, is_bh
            )
            # Request a matching L1 width-sharded output config for down_proj so
            # the matmul writes its result into L1 instead of DRAM
            down_pc, down_out_memcfg = _decode_linear_1d_config(
                m_tiles,
                _tiles(self.down_proj.shape[-2]),
                _tiles(self.down_proj.shape[-1]),
                grid,
                is_bh,
                return_output_memcfg=True,
            )

        # Fused gate+up projection: one matmul produces the [gate | up] slab.
        gate_up = ttnn.linear(hidden_states, self.gate_up_proj, program_config=gate_up_pc)

        # Split the slab into gate / up halves and fuse GeGLU into the multiply:
        # fast-approx GELU on the gate half (operand a) only, then elementwise * up.
        # Matches the original fast_and_approximate_mode=True GeGLU semantics.
        split = self.padded_split
        gate = gate_up[..., :split]
        up = gate_up[..., split:]
        hidden = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, True)])
        gate_up.deallocate(True)

        # output = hidden @ down_proj
        output = ttnn.linear(hidden, self.down_proj, program_config=down_pc, memory_config=down_out_memcfg)
        hidden.deallocate(True)

        # Allreduce after row-parallel down_proj
        if self.mesh_config is not None and self.mesh_config.tp > 1:
            output = ccl_allreduce(output, self.mesh_config, self.ccl_manager)

        return output
