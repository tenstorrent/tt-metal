# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN vision patch embed for dots.ocr.

DotsPatchEmbed = Conv2d(3 -> 1536, kernel 14x14, stride 14) + RMSNorm(eps=1e-5).

The HF preprocessor hands the vision tower PRE-FLATTENED patches of shape
[num_patches, C * T * P * P] (T = temporal_patch_size = 1). Because the conv
stride equals the kernel size, each patch is convolved exactly once and the
convolution is mathematically a single linear projection with weight
``proj.weight.view(embed_dim, C * P * P)`` — the same trick used by ViT-style
TTNN ports (cf. reference_impl models/demos/qwen25_vl/tt/model.py, whose
patch-embed conv likewise collapses onto a matmul over flattened patches).

Parallelism plan (ARCHITECTURE.md): placement=replicate — the vision tower is
run-once per input, all weights are ``ReplicateTensorToMesh`` and the output
stays replicated, so the handoff into the column-parallel decoder needs no CCL.
On a single device the mesh_mapper degenerates gracefully (1x1 mesh).
"""

import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32


def _largest_divisor_leq(n: int, cap: int) -> int:
    """Largest d such that d | n and d <= cap (>= 1)."""
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


class TtVisionPatchEmbed(LightweightModule):
    """dots.ocr DotsPatchEmbed: flattened patches -> linear(conv) -> RMSNorm.

    Args:
        mesh_device: ttnn mesh device handle (all weights replicated).
        state_dict: {"proj.weight": [E, C, P, P], "proj.bias": [E],
                     "norm.weight": [E]} torch tensors.
        dtype: on-device weight/activation dtype.
        eps: RMSNorm epsilon (dots.ocr vision uses 1e-5).
    """

    def __init__(self, mesh_device, state_dict, dtype=ttnn.bfloat16, eps=1e-5, weight_dtype=None):
        super().__init__()
        self.mesh_device = mesh_device
        self.eps = eps

        proj_w = state_dict["proj.weight"]  # [E, C, P, P]
        embed_dim = proj_w.shape[0]
        in_features = proj_w.shape[1] * proj_w.shape[2] * proj_w.shape[3]
        self.embed_dim = embed_dim
        self.in_features = in_features
        # bf8b-first single-pass directive (mirrors vision_mlp/vision_attention/
        # patch_merger): the run-once bf16 tower takes a bfloat8_b projection
        # weight + HiFi2 (fp32 dest acc kept); the fp32 high-precision path is
        # untouched. The bias stays at the activation dtype (block-fp8 biases
        # cost PCC for no bandwidth win).
        self.high_precision = dtype == ttnn.float32
        if weight_dtype is None:
            weight_dtype = dtype if self.high_precision else ttnn.bfloat8_b
        self.weight_dtype = weight_dtype
        # temporal_patch_size == 1: input [N, C*P*P] flattening matches the
        # conv kernel's (C, kh, kw) flattening, so conv == x @ W_flat.T + b.
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        self.proj_weight = ttnn.from_torch(
            proj_w.reshape(embed_dim, in_features).T.contiguous(),
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        self.proj_bias = ttnn.from_torch(
            state_dict["proj.bias"].reshape(1, embed_dim),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        # ttnn.rms_norm gamma format: [1, 1, E//32, 32] in ROW_MAJOR for
        # bf16. The ROW_MAJOR gamma path is bf16-only (an fp32 ROW_MAJOR
        # gamma is misread on device, PCC ~0) — fp32 gammas use TILE
        # [1, 1, 1, E] instead.
        if dtype == ttnn.float32:
            gamma, gamma_layout = state_dict["norm.weight"].reshape(1, 1, 1, embed_dim), ttnn.TILE_LAYOUT
        else:
            gamma, gamma_layout = (
                state_dict["norm.weight"].reshape(1, 1, embed_dim // TILE, TILE),
                ttnn.ROW_MAJOR_LAYOUT,
            )
        self.norm_weight = ttnn.from_torch(
            gamma,
            dtype=dtype,
            layout=gamma_layout,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [num_patches, C*P*P] TILE_LAYOUT, replicated across the mesh.

        Returns: [num_patches, embed_dim], replicated.
        """
        # Occupancy-driven path (tracy, BH 11x10 grid): the interleaved
        # rms_norm fell back to the row-per-core kernel (28/110 cores, 25%
        # occupancy, ~37 us) and the L1-interleaved matmul ran on a
        # hardcoded 10x10 grid. Fix: the projection matmul writes a
        # BLOCK_SHARDED L1 output via a 2D-mcast program config and the
        # sharded rms_norm consumes the shard in place, so the norm runs on
        # the full shard grid instead of one core per tile row. Measured
        # (fp32, [896, 588] production shape, traced replay): 104.2 ->
        # 86.4 us/iter; in0_block_w = full inner K (19 tiles) beat
        # in0_block_w=1 by another 8%. Grid is derived from the queried
        # compute grid + tile divisibility, with the interleaved path as
        # fallback for shapes that don't divide.
        #
        # SIZE GATE (document scale): the sharded recipe only wins — and only
        # COMPILES — when the per-core circular buffers fit L1. At the 11k-token
        # document shape ([11264, 588], M=352 tiles) the largest dividing grid
        # is 8x8, per_core_M=44, and the full-K mcast matmul CBs grow to
        # 3 150 336 B > 1.5 MB L1 (TT_FATAL, program.cpp:1326). The estimate
        # below (in0 block per_m*K + in1 block K*per_n + 2x out shard
        # per_m*per_n, in tile bytes) reproduces that measurement (~3.0 MB at
        # the doc shape, ~0.98 MB at the measured-good fp32 gate shape), so
        # shapes that blow the budget take the interleaved path. That is also
        # the RIGHT path at document scale: with m_tiles (352) >= grid cores
        # (110) the interleaved rms_norm occupies the full grid — the
        # 28-row under-occupancy that motivated the sharded recipe is a
        # small-shape artifact. Measured at [11264, 588] bf16 (traced
        # replay): interleaved matmul + LN run at full-grid occupancy; see
        # forward-docstring numbers in the optimization notes.
        m_tiles = (x.padded_shape[-2] + TILE - 1) // TILE
        n_tiles = self.embed_dim // TILE
        k_tiles = (self.in_features + TILE - 1) // TILE
        grid = self.mesh_device.compute_with_storage_grid_size()
        gx = _largest_divisor_leq(n_tiles, grid.x)
        gy = _largest_divisor_leq(m_tiles, grid.y)
        per_m, per_n = m_tiles // gy, n_tiles // gx
        tile_bytes = TILE * TILE * (4 if x.dtype == ttnn.float32 else 2)
        cb_estimate = (per_m * k_tiles + k_tiles * per_n + 2 * per_m * per_n) * tile_bytes
        CB_BUDGET = 1_300_000  # conservative slice of the 1.5 MB per-core L1

        if gx * gy > 1 and x.padded_shape[-2] % TILE == 0 and cb_estimate <= CB_BUDGET:
            shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
            block_mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(shard_grid, [per_m * TILE, per_n * TILE], ttnn.ShardOrientation.ROW_MAJOR),
            )
            mm_pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                in0_block_w=k_tiles,
                out_subblock_h=1,
                # fp32 dest acc halves the dest budget: subblock <= 4 tiles.
                out_subblock_w=_largest_divisor_leq(per_n, 4),
                per_core_M=per_m,
                per_core_N=per_n,
                transpose_mcast=False,
                fused_activation=None,
            )
            ln_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(gx, gy),
                subblock_w=_largest_divisor_leq(per_n, 4),
                block_h=per_m,
                block_w=per_n,
                inplace=False,
            )
            # Explicit precision floor: the program-config matmul/LN paths do
            # not inherit the interleaved defaults, and the downstream
            # 42-block tower sits at razor-thin 0.99 PCC margin — HiFi4 +
            # fp32 dest acc keeps the sharded path's accumulation at least
            # as precise as the interleaved path it replaced. The bf8b
            # weight path runs HiFi2 (the extra two LoFi passes only refine
            # bits bf8b weights don't carry), matching the sibling blocks.
            compute_cfg = ttnn.init_device_compute_kernel_config(
                self.mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4
                if self.weight_dtype != ttnn.bfloat8_b
                else ttnn.MathFidelity.HiFi2,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            h = ttnn.linear(
                x,
                self.proj_weight,
                bias=self.proj_bias,
                program_config=mm_pc,
                memory_config=block_mc,
                compute_kernel_config=compute_cfg,
            )
            n = ttnn.rms_norm(
                h,
                epsilon=self.eps,
                weight=self.norm_weight,
                program_config=ln_pc,
                memory_config=block_mc,
                compute_kernel_config=compute_cfg,
            )
            ttnn.deallocate(h)
            # L1-resident handoff (batch-1 contract): the embed stream is at
            # most [896, 1536] fp32 ≈ 5.5 MB interleaved across the grid;
            # landing it in L1 instead of DRAM saved ~9 us/iter measured
            # (s2i_DRAM 85.8 -> s2i_L1 76.7 us traced replay) and the tower's
            # first norm reads it from L1.
            out = ttnn.sharded_to_interleaved(n, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(n)
            return out

        # Fallback: interleaved matmul on the full queried grid + fused norm.
        # Document scale ([11264, 588] bf16, traced replay): matmul 100/110
        # cores (91%), rms_norm 110/110 — the heuristic path is already at
        # the occupancy bar, so only precision/placement are pinned here.
        # Output placement is size-gated: at document scale the [11264, 1536]
        # bf16 output is ~33 MB and pinning it to L1 leaves too little head-
        # room for the fp32-dest-acc matmul's interm CBs (measured CB/L1-
        # buffer clash at 1 224 704 + 157 KB overrun, program.cpp:1335); the
        # tower stream is DRAM at this scale anyway (recorded L1-into-matmul
        # stall hazard). Small (gate) shapes keep the measured L1 handoff.
        h = ttnn.linear(
            x,
            self.proj_weight,
            bias=self.proj_bias,
            core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        out = ttnn.rms_norm(h, epsilon=self.eps, weight=self.norm_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(h)
        return out
