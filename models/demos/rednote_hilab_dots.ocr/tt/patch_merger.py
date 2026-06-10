# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN patch merger for dots.ocr.

DotsPatchMerger (modeling_dots_vision):
``LayerNorm(eps=1e-6) -> view(-1, C*m^2) -> Linear -> GELU -> Linear``
with hidden 1536, spatial_merge_size 2, so 1536 -> view 6144 -> 6144 -> 1536.
All four affine params present (ln_q weight+bias, both Linear biases).

Structure mirrors reference_impl models/demos/qwen25_vl/tt/patch_merger.py:
norm -> ROW_MAJOR reshape workaround (tilized ttnn.reshape hang, tt-metal
issue #29932) -> linear -> gelu -> linear. Differences vs qwen25_vl: dots.ocr
uses LayerNorm with bias (qwen used RMSNorm) and biased Linears, so we pass
``bias=`` to ttnn.linear and use ttnn.layer_norm (TILE gamma/beta per
models/tt_transformers/tt/multimodal/llama_layernorm.py).

KB ttnn_gelu cited: standalone exact ttnn.gelu(fast_and_approximate_mode=False)
after ttnn.linear replaces the torch linear->gelu subsequence (entry notes that
fusing the activation into the matmul cost PCC).

Occupancy REDO (production posture: bf16 tower, document image => padded_seq
11264, 1x4 BH mesh, queried grid 11x10=110; tracy under --traced, metal trace
replay): block kernel time 3541 -> 2362 us/device (-33%), traced wall
3.44 -> 2.77 ms/call (the merger runs once per image). Levers, one per
measurement: (1) bf8b-first single-pass directive — bfloat8_b weights + HiFi2
(fp32 dest acc kept) on both projections, matmuls 2416 -> 1825 us, gate PCC
0.999992 -> 0.999915; (2) up-projection explicit 2D-mcast config (heuristic
ibw=1 @46% FPU 1513us -> ibw=16 sb1x3 out_block 9x9 952us @73% FPU, 110/110
cores); (3) down-projection 11x10 ibw=16 sb3x1 DRAM-direct (heuristic-with-
10x10-grid+L1+copy 509+34us -> 274us @100/110 — N_t=48/5-per-core covers 10
columns, an N-divisibility ceiling). Waved with evidence/recorded hazards:
exact-erf GELU 337us @110/110 (fuse-into-matmul costs PCC per KB ttnn_gelu);
LayerNorm 240us @110/110 (sharded-LN i2s/s2i bounce loses, vision_rmsnorm
A/B); untilize/reshape/tilize 558us @104-110/110 (tilized ttnn.reshape hang,
issue #29932 — the ROW_MAJOR round-trip is the qwen25_vl recipe and runs at
full grid). Gates: gate-shape PCC 0.999915, document-shape PCC 0.999889 vs
fp32 torch, e2e WER parity re-run.

Parallelism plan (ARCHITECTURE.md): vision tower placement=replicate — all
weights ``ReplicateTensorToMesh`` on the 1x4 mesh, activations stay replicated,
no CCL. On a single device the mesh_mapper degenerates gracefully.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32


class TtPatchMerger(LightweightModule):
    """dots.ocr patch merger: LayerNorm -> view(-1, dim*m^2) -> Linear -> GELU -> Linear.

    Args:
        mesh_device: ttnn mesh device handle (weights replicated).
        state_dict: {"ln_q.weight": [dim], "ln_q.bias": [dim],
            "mlp.0.weight": [dim*m^2, dim*m^2], "mlp.0.bias": [dim*m^2],
            "mlp.2.weight": [out, dim*m^2], "mlp.2.bias": [out]} torch tensors
            (HF keys vision_tower.merger.*).
        spatial_merge_size: spatial merge factor m (default 2).
        eps: LayerNorm epsilon (DotsPatchMerger hard-codes 1e-6).
        dtype: on-device weight dtype.
    """

    def __init__(self, mesh_device, state_dict, spatial_merge_size=2, eps=1e-6, dtype=ttnn.bfloat16, weight_dtype=None):
        super().__init__()
        self.mesh_device = mesh_device
        self.eps = eps
        dim = state_dict["ln_q.weight"].shape[0]
        self.merged_dim = dim * spatial_merge_size**2
        # bf8b-first on the single-pass vision path (occupancy REDO): the
        # run-once bf16 tower takes bfloat8_b weights + HiFi2 (fp32 dest acc
        # kept) on both projections — matmuls 2416 -> 1825 us/device at the
        # 11264-row document shape. The fp32 high-precision path
        # (tests/test_vision_transformer.py legacy posture) keeps fp32
        # weights + HiFi4 untouched.
        self.high_precision = dtype == ttnn.float32
        if weight_dtype is None:
            weight_dtype = dtype if self.high_precision else ttnn.bfloat8_b

        replicate = lambda t, layout, w_dtype=dtype: ttnn.from_torch(
            t,
            dtype=w_dtype,
            layout=layout,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # LayerNorm gamma/beta: [1, TILE, dim] TILE_LAYOUT, per
        # models/tt_transformers/tt/multimodal/llama_layernorm.py.
        norm_param = lambda name: replicate(state_dict[name].view(1, 1, dim).expand(1, TILE, dim), ttnn.TILE_LAYOUT)
        self.norm_weight = norm_param("ln_q.weight")
        self.norm_bias = norm_param("ln_q.bias")

        # Linear weights transposed [out, in] -> [in, out] for x @ W^T; biases [1, out].
        as_weight = lambda name: replicate(
            state_dict[name].transpose(-2, -1).contiguous(), ttnn.TILE_LAYOUT, weight_dtype
        )
        # Biases stay at the activation dtype: bf8b's shared-exponent blocks
        # are lossy for a [1, N] vector and save no matmul time.
        as_bias = lambda name: replicate(state_dict[name].reshape(1, -1), ttnn.TILE_LAYOUT)
        self.w1 = as_weight("mlp.0.weight")
        self.b1 = as_bias("mlp.0.bias")
        self.w2 = as_weight("mlp.2.weight")
        self.b2 = as_bias("mlp.2.bias")

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4 if weight_dtype != ttnn.bfloat8_b else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _mm_program_config(self, x, w):
        """2D-mcast matmul config for the merged-token projections.

        Occupancy REDO A/B at the production document shape (queried grid
        11x10=110; [2816,6144] bf16 act x bf8b weight, HiFi2+fp32-acc;
        scratch sweep, preallocated DRAM-interleaved operands):

        - up 6144->6144 (M_t=88, K_t=192, N_t=192): heuristic 11x10 ibw=1
          sb3x1 1.731 -> ibw=16 sb1x3 out_block 9x9 1.084 ms/iter (ibw 4/6/12
          measured 1.26/1.14/1.13; ibw>=6 with full out_block and ibw>=24
          CB-overflow program.cpp:1326 -> out_block_w halved).
        - down 6144->1536 (N_t=48): hardcoded-10x10 core_grid+L1+copy recipe
          0.366+0.034 copy -> 11x10 ibw=16 sb3x1 per_core 9x5 DRAM-direct
          0.314 ms/iter (8x10 2D variants 0.35-0.39; heuristic 0.49).

        Size-gated to the measured envelope: small shapes (the 896-row PCC
        gate posture, per_core_M < 2) and K not divisible by ibw fall back
        to the heuristic (None).
        """
        m_tiles = 1
        for i in range(len(x.padded_shape) - 1):
            m_tiles *= x.padded_shape[i]
        m_tiles //= TILE
        k_tiles = x.padded_shape[-1] // TILE
        n_tiles = w.padded_shape[-1] // TILE
        grid = self.mesh_device.compute_with_storage_grid_size()
        per_m = -(-m_tiles // grid.y)  # ceil
        per_n = -(-n_tiles // grid.x)
        if per_m < 2 or per_m > 12 or k_tiles % 16:
            return None  # off the measured envelope -> heuristic
        wide = n_tiles >= 96  # up-projection: halve out_block_w for CB fit
        if wide and (per_n % 6 or per_m % 3):
            return None
        if not wide and per_m % 3:
            return None
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            in0_block_w=16,
            out_subblock_h=1 if wide else 3,
            out_subblock_w=3 if wide else 1,
            out_block_h=per_m,
            out_block_w=per_n // 2 if wide else per_n,
            per_core_M=per_m,
            per_core_N=per_n,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [seq, dim] (or [..., seq, dim]) TILE_LAYOUT, replicated across the mesh.

        Returns: [seq / m^2, out_dim], replicated.
        """
        x = ttnn.layer_norm(
            x,
            epsilon=self.eps,
            weight=self.norm_weight,
            bias=self.norm_bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Merge m^2 adjacent patch rows into the feature dim: [seq, dim] ->
        # [seq/m^2, dim*m^2]. Tilized ttnn.reshape can hang (issue #29932) —
        # use the qwen25_vl ROW_MAJOR round-trip workaround.
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (-1, self.merged_dim))
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Up-projection: explicit 2D-mcast config at the document envelope
        # (heuristic picked in0_block_w=1 at 46% FPU util — 1513us; ibw=16
        # with halved out_block_w 1084us, see _mm_program_config). Off the
        # envelope (896-row gate / fp32 legacy shapes) the heuristic stands
        # (its 96-core DRAM fused-bias path measured best there: forcing
        # core_grid 10x10 + L1 was 431.9 -> 448.8us at fp32 [224,6144]).
        h = ttnn.linear(
            x,
            self.w1,
            bias=self.b1,
            program_config=self._mm_program_config(x, self.w1),
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        # KB ttnn_gelu: exact (erf) GELU, standalone after the linear.
        # Writes DRAM so the next matmul is DRAM-fed (large L1-interleaved
        # matmul operands stall the kernel, see vision_block notes).
        h = ttnn.gelu(h, fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        down_pc = self._mm_program_config(h, self.w2)
        if down_pc is not None:
            # Document envelope: 11x10 ibw=16 sb3x1 DRAM-direct (314us; also
            # drops the 34us L1->DRAM copy the legacy recipe needed).
            out = ttnn.linear(
                h,
                self.w2,
                bias=self.b2,
                program_config=down_pc,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(h)
            return out
        # Legacy/gate-shape path (fp32 [224,6144] tracy A/B): the heuristic
        # ran at only 48 cores (249.6us); core_grid 10x10 + L1 output engages
        # the fused-bias L1 matmul variant on a fuller grid (142.4us @ 70
        # cores). Grid clamped to the queried size for harvested parts.
        grid = self.mesh_device.compute_with_storage_grid_size()
        out = ttnn.linear(
            h,
            self.w2,
            bias=self.b2,
            core_grid=ttnn.CoreGrid(y=min(10, grid.y), x=min(10, grid.x)),
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(h)
        # Block-output contract: hand the (small, [seq/m^2, out]) result back
        # in DRAM interleaved like the other tower blocks.
        out_dram = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        return out_dram
