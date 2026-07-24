# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN MLP for Swin-L blocks.
Adapted from models/experimental/swin_s/tt/tt_mlp.py.
Initial version: generic (no hardcoded sharding configs).
"""

import ttnn


# Per-(K, N) tuned 2D-mcast program_config for fc1. Only applied when the runtime input
# shape's leading dims match the canonical Swin-L shape that produces the same
# effective M_tiles used by the kernel. ttnn.linear's leading-dim handling is rank-dependent
# (it pads the last spatial dim to TILE), so M_tiles_effective = Z * ceil(Y/TILE).
_TUNED_FC1_PROGRAM_CONFIGS = {
    # Stage 2: input (1, 40, 40, 768) -> Z=40, Y=64[40], so M_tiles=40*2=80. weight (768, 3072).
    # Standalone sweep: default LoFi+gelu+8x8 = 518us; tuned 2D 8x8 pcm=10 ibw=8 sub=2x4 = 218us (-58%).
    # 18 Stage-2 calls/iter -> ~5.4 ms iter savings.
    (768, 3072): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=8,
        out_subblock_h=2,
        out_subblock_w=4,
        per_core_M=10,
        per_core_N=12,
        transpose_mcast=False,
        fused_activation=(ttnn.UnaryOpType.GELU, True),
    ),
    # Stage 1: input (1, 80, 80, 384) -> Z=80, Y=96[80], M_tiles=80*3=240. weight (384, 1536).
    # Standalone sweep: default = 716us; tuned 2D 8x8 pcm=30 ibw=12 sub=1x6 = 340us (-52%).
    # 2 Stage-1 calls/iter -> ~0.75 ms iter savings.
    (384, 1536): ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=12,
        out_subblock_h=1,
        out_subblock_w=6,
        per_core_M=30,
        per_core_N=6,
        transpose_mcast=False,
        fused_activation=(ttnn.UnaryOpType.GELU, True),
    ),
    # Stage 0: input (1, 160, 160, 192) -> M_tiles=800, K=6, N=24.
    # 2D mcast doesn't fit (would require pcm=100 -> L1 overflow). 1D mcast distributes
    # M=800 tiles across 64 cores (pcm=13 with one core handling the remainder).
    # Standalone sweep: default 2D = 1281us; tuned 1D pcm=13 pcn=24 ibw=2 sub=1x8 = 550us (-57%).
    # 2 Stage-0 calls/iter -> ~1.5 ms iter savings.
    (192, 768): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=8,
        per_core_M=13,
        per_core_N=24,
        fuse_batch=True,
        fused_activation=(ttnn.UnaryOpType.GELU, True),
        mcast_in0=False,
        gather_in0=False,
        hop_cores=ttnn.CoreRangeSet([]),
        num_global_cb_receivers=0,
        untilize_out=False,
    ),
}

# Each (K, N) entry's expected M_tiles_effective for the canonical input shape.
_EXPECTED_M_TILES = {
    (768, 3072): 80,    # Stage 2: 40 * 2
    (384, 1536): 240,   # Stage 1: 80 * 3
    (192, 768): 800,    # Stage 0: 160 * 5
}

# Per-(K, N) tuned program_config for fc2. Same shape logic as fc1.
_TUNED_FC2_PROGRAM_CONFIGS = {
    # Stage 0: input (1, 160, 160, 768) -> M_tiles=800, K=24, N=6.
    # Standalone sweep: default = 582us; tuned 1D pcm=13 pcn=6 ibw=3 sub=1x6 = 381us (-35%).
    # 2 Stage-0 calls/iter -> ~0.4 ms iter savings.
    (768, 192): ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        in0_block_w=3,
        out_subblock_h=1,
        out_subblock_w=6,
        per_core_M=13,
        per_core_N=6,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=False,
        hop_cores=ttnn.CoreRangeSet([]),
        num_global_cb_receivers=0,
        untilize_out=False,
    ),
}

_EXPECTED_M_TILES_FC2 = {
    (768, 192): 800,    # Stage 0
}


class TtSwinMLP:
    """Two-layer MLP with GELU activation."""

    def __init__(self, device, parameters, dim, mlp_ratio=4.0):
        self.device = device
        self.parameters = parameters
        self.hidden_dim = int(dim * mlp_ratio)
        self.dim = dim
        # Pre-resolve fc1 program_config off the static (K, N) shape.
        w = parameters["fc1"]["weight"]
        K_w = int(w.shape[-2])
        N_w = int(w.shape[-1])
        self._fc1_pcfg = _TUNED_FC1_PROGRAM_CONFIGS.get((K_w, N_w))
        self._fc1_expected_Z_x_Y_tiles = _EXPECTED_M_TILES.get((K_w, N_w))
        # Pre-resolve fc2 program_config off the static (K, N) shape.
        w2 = parameters["fc2"]["weight"]
        K2_w = int(w2.shape[-2])
        N2_w = int(w2.shape[-1])
        self._fc2_pcfg = _TUNED_FC2_PROGRAM_CONFIGS.get((K2_w, N2_w))
        self._fc2_expected_Z_x_Y_tiles = _EXPECTED_M_TILES_FC2.get((K2_w, N2_w))

    def __call__(self, input_tensor):
        # Decide whether the runtime shape matches the tuned config's M_tiles.
        use_tuned = False
        if self._fc1_pcfg is not None:
            shp = input_tensor.shape
            if len(shp) == 4:
                Z = int(shp[1])
                Y = int(shp[2])
                Y_tiles = (Y + 31) // 32
                use_tuned = (Z * Y_tiles) == self._fc1_expected_Z_x_Y_tiles

        # fc1 + GELU. Output dtype=bfloat8_b halves the (B*nW, S, 4*C) write/read
        # bandwidth for fc2 -- same trick ViT uses on every linear in the encoder.
        if use_tuned:
            output = ttnn.linear(
                input_tensor,
                self.parameters["fc1"]["weight"],
                bias=self.parameters["fc1"]["bias"],
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=True,
                ),
                program_config=self._fc1_pcfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
            )
        else:
            output = ttnn.linear(
                input_tensor,
                self.parameters["fc1"]["weight"],
                bias=self.parameters["fc1"]["bias"],
                activation="gelu",
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=True,
                ),
                core_grid=ttnn.CoreGrid(y=8, x=8),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
            )

        # fc2 — same shape-guarded tuned path as fc1.
        use_tuned_fc2 = False
        if self._fc2_pcfg is not None:
            shp2 = output.shape
            if len(shp2) == 4:
                Z2 = int(shp2[1])
                Y2 = int(shp2[2])
                Y2_tiles = (Y2 + 31) // 32
                use_tuned_fc2 = (Z2 * Y2_tiles) == self._fc2_expected_Z_x_Y_tiles
        if use_tuned_fc2:
            return ttnn.linear(
                output,
                self.parameters["fc2"]["weight"],
                bias=self.parameters["fc2"]["bias"],
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=True,
                ),
                program_config=self._fc2_pcfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
            )
        return ttnn.linear(
            output,
            self.parameters["fc2"]["weight"],
            bias=self.parameters["fc2"]["bias"],
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            ),
            core_grid=ttnn.CoreGrid(y=8, x=8),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
        )
