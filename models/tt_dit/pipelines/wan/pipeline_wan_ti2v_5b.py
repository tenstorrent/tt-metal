# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 TI2V-5B pipeline variant for single BH Galaxy (4x8).

Dense 5B + Wan2.2-VAE (48-ch residual decoder, patch_size=2).
The DiT matmul tables are keyed on the shapes and grids the model actually
requests and carry swept blockings (see ``_register_5b_matmul_tables``).
"""

import ttnn
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline
from models.tt_dit.utils.conv3d import _BLOCKINGS, register_conv3d_configs
from models.tt_dit.utils.matmul import FusedMMRSConfig, register_fused_mmrs_configs, register_matmul_configs

_5B_CHECKPOINT = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def _register_5b_matmul_tables() -> None:
    """Wan2.2 TI2V-5B DiT matmul blockings on one BH Galaxy 4x8 (SP=8 on axis 1, TP=4 on axis 0).

    Per device the DiT sees N = dim/TP = 768, qkv = 3*dim/TP = 2304, ffn/TP = 3584 and proj_out
    = 48*2*2 = 192, at M = the SP-local, tile-padded token count::

        1280x704 / 81f  -> 18480 tokens -> M = 2336 per SP rank
         832x480 / 81f  ->  8190 tokens -> M = 1024 per SP rank
        1280x704 / 121f -> 27280 tokens -> M = 3424  (not tabled; takes the rule engines below)

    Which table a call site reads is decided by the op it lands on, not by the model's grid:

    * ColumnParallelLinear on the TP ring -> ``all_gather_minimal_matmul_async``. Every Wan call
      site leaves ``force_transpose=True``, so ``get_agmm_config`` keys the lookup on
      ``agmm_worker_grid(12x10, transpose=True)`` = **12x9**, with the *global* K (3072). Shapes
      absent from that table go to the AGMM v3 rule engine when M > N and to the warned 8x8x8
      default otherwise -- which is where ff1 at both resolutions and qkv at 480p were landing.
    * ``RowParallelLinear.forward_fused_addcmul`` (ff2) -> fused MM+RS, keyed by the full 12x10
      device grid with the *per-device* K (3584); the matmul runs on 12x8 above the reduce-scatter.
    * Plain ``Linear`` / ``minimal_matmul_split`` (proj_out, cross-attention to_kv) ->
      ``get_matmul_core_grid``, clamped to **11x10** on a Galaxy.

    The earlier version of this table registered everything under "11x10" at M in (2368, 2720,
    3488, 9472, 13952), none of which the model requests, so no entry was ever reached.

    Blockings are the winners of the per-shape sweep in ``sweep_mm_block_sizes.py`` (the eleven 5B
    shapes listed there; the orchestrator prints each winner as a PASTE line in this file's tuple
    format)::

        pytest models/tt_dit/utils/sweep_mm_block_sizes.py::test_mm_sweep -sv --timeout=0 \\
          -k "bh_4x8_sp1_tp0 and (3072_2304 or 3072_768 or 3072_3584 or 3584_3072 \\
                                  or 3072_192 or 512_3072_1536)"

    An entry marked PRE-SWEEP is not swept yet and reproduces exactly what the lookup resolved to
    before the table existed, so it changes the key, not the kernel. Note the sweep compiles one
    program per combo and the kernel JIT cache keeps every one (~12 GB per shape under
    ``~/.cache/tt-metal-cache``, on top of ~3 GB of profiler capture); budget disk accordingly.
    """
    dim, ffn = 3072, 14336
    dim_tp = dim // 4
    ffn_tp = ffn // 4
    qkv_tp = (dim * 3) // 4
    proj_out_n = 48 * 2 * 2
    prompt_seq = 512
    m_720p, m_480p = 2336, 1024

    # Swept 2026-09-22 on this Galaxy (sweep_mm_block_sizes.py, DEVICE KERNEL DURATION, ~330-380
    # L1-feasible combos per shape). "was" is the blocking the lookup resolved to before the table
    # existed: the AGMM v3 / MMRS v2.3 rule pick where one fired, the warned 8x8x8 default otherwise.
    register_matmul_configs(
        {
            # AGMM (ColumnParallelLinear, TP ring): global K, 12x9 worker grid.
            "12x9": {
                # attn1.to_qkv (chunks=3, approx math)
                (m_480p, dim, qkv_tp): (3, 8, 16, (1, 4)),  # 149.5 us; was (8, 8, 8) 160.2 us
                (m_720p, dim, qkv_tp): (7, 6, 8, (1, 4)),  # 275.8 us; was v3 (4, 8, 8) 315.3 us
                # attn1.to_out (+ fused addcmul residual, approx math)
                (m_480p, dim, dim_tp): (3, 8, 3, (1, 3)),  # 108.5 us; the v3 pick, confirmed optimal
                (m_720p, dim, dim_tp): (7, 6, 3, (1, 3)),  # 229.2 us; was v3 (4, 8, 3) 236.0 us
                # ffn.ff1 (fused gelu_tanh). N=3584 is 112 tiles over 9 columns -> 13 per core.
                (m_480p, dim, ffn_tp): (3, 6, 13, (3, 1)),  # 203.8 us; was (8, 8, 8) 239.3 us
                (m_720p, dim, ffn_tp): (7, 6, 13, (1, 1)),  # 352.1 us; was (8, 8, 8) 400.7 us
            },
            # Plain minimal_matmul / minimal_matmul_split: Galaxy-clamped 11x10 grid.
            "11x10": {
                # proj_out (6 N tiles)
                (m_480p, dim, proj_out_n): (3, 8, 2, (3, 1)),  # 33.7 us; was (8, 8, 2) 35.4 us
                (m_720p, dim, proj_out_n): (4, 8, 4, (2, 2)),  # 70.2 us; was (8, 8, 2) 74.7 us
                # attn2.to_kv over the padded prompt (chunks=2, approx math); same at both resolutions.
                (prompt_seq, dim, 2 * dim_tp): (8, 8, 8, (2, 2)),  # PRE-SWEEP: was the warned default
            },
        }
    )
    # ffn.ff2: fused matmul + reduce-scatter + addcmul. Keyed by the full device grid; per-device K
    # (112 tiles). Both M blocks leave >= 2 blocks per core, so both take the windowed L1 handoff.
    register_fused_mmrs_configs(
        {
            ttnn.CoreCoord(12, 10): {
                (m_480p, ffn_tp, dim): FusedMMRSConfig(
                    ttnn.CoreCoord(12, 8), 3, 7, 14, 3, 1, None, 1
                ),  # 210.4 us; was v2.3 (2, 4, 6) 270.5 us
                (m_720p, ffn_tp, dim): FusedMMRSConfig(
                    ttnn.CoreCoord(12, 8), 6, 2, 8, 2, 2, None, 1
                ),  # 357.7 us; was v2.3 (6, 4, 6) 410.6 us
            }
        }
    )


def _register_5b_conv3d_tables() -> None:
    """BH Galaxy 4x8, Wan2.2 TI2V-5B VAE decoder conv3d blockings.

    Swept via bruteforce_conv3d_sweep.py (bh_4x8_5b_480p_t7). Keyed by
    (C_in, C_out, kernel_size) -> (C_in_block, C_out_block, T_out_block,
    H_out_block, W_out_block). Without these every 5B VAE conv3d fell to the
    worst-case (<=256, 32, 1, 1, 1) fallback. For channel-keys used at multiple
    T/H/W the blocking chosen is L1-safe for (and measured on) the LARGEST 480p
    variant of that key.
    """
    register_conv3d_configs(
        {
            (64, 1024, (3, 3, 3)): (64, 256, 1, 8, 4),  # conv_in
            (1024, 1024, (3, 3, 3)): (128, 64, 7, 16, 2),  # res_deep (t9/t16); sized for t16
            (1024, 2048, (3, 1, 1)): (512, 128, 3, 8, 4),  # tconv (t9/t16); conservative C_out
            (1024, 1024, (1, 3, 3)): (256, 128, 1, 16, 2),  # spatial_deep/spatial_mid; sized for mid
            (1024, 512, (3, 3, 3)): (64, 256, 2, 16, 2),  # up_512
            (512, 512, (3, 3, 3)): (64, 256, 2, 16, 2),  # res_512
            (512, 512, (1, 3, 3)): (256, 128, 1, 16, 2),  # spatial_512
            (512, 256, (3, 3, 3)): (64, 256, 2, 8, 4),  # up_256
            (256, 256, (3, 3, 3)): (64, 256, 2, 8, 4),  # res_256
            (256, 12, (3, 3, 3)): (128, 32, 4, 8, 4),  # conv_out
        }
    )


_register_5b_matmul_tables()
_register_5b_conv3d_tables()

# 720p exact-shape conv3d winners (measured on 4x8, 720p t=7; keyed by full shape tuple so the
# 480p channel-keyed table is untouched). Complete (13/13 swept via bruteforce_conv3d_sweep.py
# bh_4x8_5b_720p_t7); every 720p VAE conv3d now has a speed-tuned exact-shape entry.
_BLOCKINGS.update(
    {
        (4, 8, 64, 1024, (3, 3, 3), 9, 11, 10): (64, 256, 1, 4, 8),  # conv_in
        (4, 8, 1024, 1024, (3, 3, 3), 16, 22, 20): (128, 64, 7, 8, 4),  # res_deep_t16
        (4, 8, 1024, 1024, (3, 3, 3), 9, 11, 10): (128, 64, 7, 4, 8),  # res_deep_t9
        (4, 8, 1024, 1024, (1, 3, 3), 14, 22, 20): (256, 128, 1, 4, 8),  # spatial_deep
        (4, 8, 1024, 1024, (1, 3, 3), 28, 44, 40): (256, 128, 1, 4, 8),  # spatial_mid
        (4, 8, 1024, 2048, (3, 1, 1), 16, 22, 20): (512, 256, 2, 8, 4),  # tconv_t16
        (4, 8, 1024, 2048, (3, 1, 1), 9, 11, 10): (512, 128, 7, 4, 8),  # tconv_t9
        (4, 8, 1024, 512, (3, 3, 3), 30, 44, 40): (64, 256, 2, 4, 8),  # up_512
        (4, 8, 512, 512, (3, 3, 3), 30, 44, 40): (64, 256, 2, 4, 8),  # res_512
        (4, 8, 512, 512, (1, 3, 3), 28, 88, 80): (256, 128, 1, 4, 8),  # spatial_512
        (4, 8, 512, 256, (3, 3, 3), 30, 88, 80): (64, 256, 2, 4, 8),  # up_256
        (4, 8, 256, 256, (3, 3, 3), 30, 88, 80): (64, 256, 2, 4, 8),  # res_256
        (4, 8, 256, 12, (3, 3, 3), 30, 88, 80): (128, 32, 3, 4, 8),  # conv_out
    }
)


class WanTI2V5BPipeline(WanPipeline):
    @classmethod
    def _config_overrides(cls) -> dict[str, object]:
        return {
            "model_type": "ti2v",
            "checkpoint_name": _5B_CHECKPOINT,
            "boundary_ratio": None,
            # 4x8 BH preset defaults vae_t_chunk_size=None (full-T), which OOMs DRAM
            # for long clips (e.g. 121f needs a ~14GB VAE activation). Chunk the temporal
            # VAE decode (feat_cache carries causal-conv state across chunks).
            "vae_t_chunk_size": 7,
            # Retune ring SDPA for the 5B's geometry without moving the 14B, which shares
            # WanAttention.sdpa_chunk_size_map at the same (is_blackhole, sp, tp) key.
            #
            # Per device the 5B self-attention is M=2336 over 6 local heads (24/TP4), and
            # work items (B*NH*ceil(M/q)) are spread flat over the SDPA worker grid. The
            # inherited q=128 gives ceil(2336/128)=19 chunks x 6 = 114 items, just over the
            # grid, so it pays a second scheduling round for a 3.6% overshoot; q=160 gives
            # 15 x 6 = 90 in a single round. Measured on a 4x8 BH Galaxy with Tracy
            # (DEVICE KERNEL DURATION, mean of 32 device instances per config):
            #
            #   q\k        128      256      512
            #   128     1954.4   1619.2   1480.0   <- inherited
            #   160     1579.0   1174.8   1068.4   <- -27.8%
            #   192     1480.7   1248.9   1183.6
            #   224     2028.6   1457.4   1260.9
            #   256     1837.0   1537.4   1433.0
            #
            # k=512 wins at every q, so the larger k chunk is worth more than the K-padding
            # it wastes. Note this key is not resolution-aware: 480p (M=1024) already fits
            # one round at either value, so it is along for the ride -- gate both resolutions.
            "sdpa_chunk_size_overrides": {(True, 8, 4): (160, 512)},
            # NOTE: TI2V-5B uses per-token (expanded) timesteps for true image
            # conditioning, but the tt _step path only plumbs a scalar timestep and
            # this base pipeline runs T2V with an all-ones mask (per-token == scalar).
            # Keep False until 2-D timestep is plumbed through combined_step.
            "expand_timesteps": False,
        }
