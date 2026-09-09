# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 TI2V-5B pipeline variant for single BH Galaxy (4x8).

Dense 5B + Wan2.2-VAE (48-ch residual decoder, patch_size=2).
Matmul tables below are first-cut 14B 11x10 blocking remapped to
dim=3072 / ffn=14336 / TP=4. Sweep before claiming perf.
"""

import ttnn
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline
from models.tt_dit.utils.matmul import FusedMMRSConfig, register_fused_mmrs_configs, register_matmul_configs

_5B_CHECKPOINT = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def _register_5b_matmul_tables() -> None:
    """BH Galaxy 4x8, SP=8, TP=4, matmul grid clamped to 11x10 like 14B.

    Per-device N: dim/TP=768, ffn/TP=3584, qkv/TP=2304.
    M is SP-local tokens: 14B 720p/81f is 9472; 5B 720p/121f is ~13952;
    WAN5B_SMOKE 21f is ~2720. Blocking copied from nearest 14B 11x10 rows —
    not a sweep.
    """
    dim, ffn = 3072, 14336
    dim_tp = dim // 4
    ffn_tp = ffn // 4
    qkv_tp = (dim * 3) // 4
    seqs = (2368, 2720, 3488, 9472, 13952)
    kn = {
        (dim, 64): (2, 32, 2, (2, 2)),
        (dim, qkv_tp): (16, 4, 4, (1, 4)),
        (dim, dim_tp): (16, 8, 4, (1, 4)),
        (dim, ffn_tp): (16, 8, 4, (1, 4)),
        (ffn_tp, dim): (16, 3, 4, (1, 4)),
        (dim, dim): (8, 8, 8, (1, 4)),
    }
    entries = {}
    for m in seqs:
        for (k, n), cfg in kn.items():
            m_block, k_block, n_block, sub = cfg
            if m <= 3488:
                m_block = min(m_block, 8)
            entries[(m, k, n)] = (m_block, k_block, n_block, sub)
    register_matmul_configs({"11x10": entries})
    register_fused_mmrs_configs(
        {
            ttnn.CoreCoord(12, 10): {
                (9472, ffn_tp, dim): FusedMMRSConfig(ttnn.CoreCoord(12, 8), 8, 3, 8, 2, 2, None, 1, 5),
                (9472 // 4, ffn_tp, dim): FusedMMRSConfig(ttnn.CoreCoord(12, 8), 6, 3, 8, 2, 2, None, 1, 5),
                (13952, ffn_tp, dim): FusedMMRSConfig(ttnn.CoreCoord(12, 8), 8, 4, 8, 2, 2, None, 1, 5),
                (2720, ffn_tp, dim): FusedMMRSConfig(ttnn.CoreCoord(12, 8), 4, 4, 8, 2, 2, None, 1, 5),
            }
        }
    )


_register_5b_matmul_tables()


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
            # NOTE: TI2V-5B uses per-token (expanded) timesteps for true image
            # conditioning, but the tt _step path only plumbs a scalar timestep and
            # this base pipeline runs T2V with an all-ones mask (per-token == scalar).
            # Keep False until 2-D timestep is plumbed through combined_step.
            "expand_timesteps": False,
        }
