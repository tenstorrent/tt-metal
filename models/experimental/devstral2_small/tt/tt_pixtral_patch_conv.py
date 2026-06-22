# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Pixtral patch embedding as a host-unfold + matmul (non-overlapping patch conv).

from __future__ import annotations

import torch
import torch.nn.functional as F

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole
from models.experimental.devstral2_small.devstral_utils.pixtral_seq_chunk import vision_seq_memcfg

# Patch-projection matmul chunking. _MM_CHUNK rows = Mt=32 = the 8x8 w19 sweep shape (per_core_M=4,
# in0_block_w=19 fits L1). Larger images are split into _MM_CHUNK-row chunks; the single-chunk path
# pads M to a multiple of _MM_ALIGN (=8 tiles) so Mt stays divisible by the grid for small/odd images.
_MM_CHUNK = 1024
_MM_ALIGN = 256


class TtPixtralPatchConv(LightweightModule):
    """Pixtral patch conv via meta keys ``{prefix}_linear.*``; torch ``[N,C,H,W]`` → ``[N,patches,out]``.

    The patch conv is non-overlapping (``kernel_size == stride``), i.e. a patchify followed by a
    linear projection. We patchify on the HOST with ``F.unfold`` and project with a single
    ``ttnn.matmul``. This removes the device-side NCHW→NHWC permute, the ``ttnn.fold``, and the conv
    input tilize (the ``Permute``/``Fold``/``TilizeWithValPadding`` ops in the device trace).

    Correctness: ``F.unfold`` emits patch elements in ``(c, kh, kw)`` order along the gather axis,
    which is exactly the flatten order of the conv weight ``[out, in, kh, kw]``. So the matmul weight
    ``w.reshape(out, in*kh*kw).T`` reproduces ``F.conv2d`` to within bf16 reduction noise, with no
    dependence on ``ttnn.fold``'s internal channel layout.
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        dtype,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        weight = state_dict[f"{state_dict_prefix}_linear.weight"]
        if weight.ndim == 2:
            weight = weight.T.reshape(out_channels, in_channels, kernel_size, kernel_size)
        # [out, in, kh, kw] -> [in*kh*kw, out]: the matmul weight (K=in*kh*kw, N=out). The K-flatten
        # order (c, kh, kw) matches F.unfold's gather order. K (=588 here) tile-pads to 608 with zeros.
        weight_2d = weight.reshape(out_channels, in_channels * kernel_size * kernel_size).t().contiguous()

        self._mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        self.weight = ttnn.from_torch(
            weight_2d,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._mesh_mapper,
            pad_value=0.0,
        )

        self.bias = None
        if bias:
            bias_t = state_dict[f"{state_dict_prefix}_linear.bias"].reshape(1, 1, 1, out_channels)
            self.bias = ttnn.from_torch(
                bias_t,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self._mesh_mapper,
            )

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @staticmethod
    def _best_subblock(per_core_m: int, per_core_n: int, max_tiles: int) -> tuple[int, int]:
        """Largest (h, w) dividing the per-core block with h*w <= the DST tile budget."""
        best_h, best_w = 1, 1
        for h in range(1, per_core_m + 1):
            if per_core_m % h:
                continue
            for w in range(1, per_core_n + 1):
                if per_core_n % w:
                    continue
                if h * w <= max_tiles and h * w > best_h * best_w:
                    best_h, best_w = h, w
        return best_h, best_w

    def _projection_program_config(
        self, chunk_m: int, k: int, n: int, fuse_batch: bool
    ) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
        """2D block-mcast config for ONE matmul chunk of the patch projection.

        Sweep finding (tests/matmul/test_matmul_1024x608x1024_sweep.py): the op was core-starved
        (the baseline ran on 16 cores at 36% FLOP util), so the first lever is the full 8x8 grid.
        ``chunk_m`` is the M of a single matmul (the forward chunks the patch sequence to ``_MM_CHUNK``
        so per_core_M stays small); at chunk_m=1024 -> Mt=32, with Nt=32 both powers of two the grid
        divides cleanly (8x8, per_core 4x4). Kt=608/32=19 is PRIME, so in0_block_w must be 1 or 19;
        the whole-K block (19) avoids the inner-K reload loop and fits L1 at per_core_M=4.
        fp32_dest_acc_en=True caps the out_subblock to 4 tiles on Blackhole.
        """
        grid = self.mesh_device.compute_with_storage_grid_size()
        mt, kt, nt = (chunk_m + 31) // 32, (k + 31) // 32, (n + 31) // 32
        grid_x = max(d for d in range(1, min(nt, int(grid.x)) + 1) if nt % d == 0)
        grid_y = max(d for d in range(1, min(mt, int(grid.y)) + 1) if mt % d == 0)
        per_core_m = mt // grid_y
        per_core_n = nt // grid_x
        # in0_block_w divides Kt. Prefer a small even divisor; for a prime Kt (19 here) the only
        # divisors are 1 and Kt, so fall back to the whole-K block.
        small = [d for d in (8, 4, 2) if kt % d == 0]
        in0_block_w = small[0] if small else kt
        max_dst = 4 if is_blackhole() else 8  # fp32_dest_acc_en=True halves the DST budget
        out_subblock_h, out_subblock_w = self._best_subblock(per_core_m, per_core_n, max_dst)
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=fuse_batch,
        )

    def forward(self, x) -> ttnn.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"TtPixtralPatchConv expects a host torch.Tensor [N,C,H,W]; got {type(x).__name__}")
        batch_size = int(x.shape[0])

        # Host patchify: [N,C,H,W] -> [N, in*k*k, L] -> [N, L, in*k*k], L = (H/s)*(W/s) patches.
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        patches = patches.transpose(1, 2).contiguous()
        num_patches = int(patches.shape[1])
        k_dim = int(patches.shape[2])
        m = batch_size * num_patches

        # Chunk the patch rows so a single matmul's per_core_M stays bounded (a large image -> a huge,
        # possibly prime Mt would either land on grid_y=1 with a per_core_M that OOMs L1, or force
        # in0_block_w=1). Each chunk is _MM_CHUNK rows = Mt=32 = the 8x8 w19 sweep shape. Rows are
        # independent (one patch per row), so chunking is correctness-neutral; pad with zero rows
        # (host) and trim them off the output. The single-chunk path pads M to _MM_ALIGN so Mt stays
        # divisible by the 8-row grid even for small/odd images.
        if m > _MM_CHUNK:
            m_pad = ((m + _MM_CHUNK - 1) // _MM_CHUNK) * _MM_CHUNK
            n_chunks, chunk_m, fuse_batch = m_pad // _MM_CHUNK, _MM_CHUNK, False
        else:
            m_pad = ((m + _MM_ALIGN - 1) // _MM_ALIGN) * _MM_ALIGN
            n_chunks, chunk_m, fuse_batch = 1, m_pad, True

        act = patches.reshape(1, 1, m, k_dim)
        if m_pad != m:
            act = F.pad(act, (0, 0, 0, m_pad - m))
        act = act.reshape(1, n_chunks, chunk_m, k_dim).contiguous()

        # Tilize on host (no device=) then upload; the projection consumes a tiled tensor directly.
        # Sweep winner "2D l1/dram/l1 8x8 w19": in0 is L1-interleaved (weights are DRAM, set in
        # __init__; see _projection_program_config). vision_seq_memcfg keeps in0/out in L1 at the
        # 1024-patch shape and falls back to DRAM only for oversized images.
        act_mem = vision_seq_memcfg(m_pad, k_dim)
        act_tt = ttnn.from_torch(
            act,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._mesh_mapper,
            pad_value=0.0,
        )
        act_tt = ttnn.to_device(act_tt, self.mesh_device, memory_config=act_mem)

        out_mem = vision_seq_memcfg(m_pad, self.out_channels)
        program_config = self._projection_program_config(chunk_m, k_dim, self.out_channels, fuse_batch)
        output = ttnn.linear(
            act_tt,
            self.weight,
            bias=self.bias,
            dtype=ttnn.bfloat16,
            memory_config=out_mem,
            compute_kernel_config=self.compute_kernel_config,
            program_config=program_config,
        )
        ttnn.deallocate(act_tt)

        output = ttnn.reshape(output, (1, 1, m_pad, self.out_channels))
        if m_pad != m:
            output = output[:, :, :m, :]
        return ttnn.reshape(output, (batch_size, num_patches, self.out_channels))


__all__ = ["TtPixtralPatchConv"]
