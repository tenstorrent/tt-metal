# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""FIBO-owned matmul blocking configs.

These were previously inlined into ``models/transformers/transformer_bria_fibo.py``, including the
SmolLM3 text-encoder and VAE-decoder shapes that the transformer does not own. They are split by
owning module here and registered into the shared ``utils/matmul.py`` grid tables at import of that
module, so neither the shared file nor the transformer carries the data:

* :func:`register_fibo_dit_matmul_configs` -- the DiT (``transformer_bria_fibo``)
* :func:`register_fibo_smollm3_matmul_configs` -- the SmolLM3 text encoder (``smollm3.model_smollm3``)
* :func:`register_fibo_vae_matmul_configs` -- the VAE decoder (``models.vae.vae_bria_fibo``)

All were tuned via ``models/tt_dit/utils/sweep_mm_block_sizes.py``. Every FIBO matmul takes the
non-AGMM minimal_matmul path, so a per-shape MinimalMatmulConfig replaces the generic (8,8,8)
fallback. Keyed by (M, K, N) under the runtime grid; registration is additive (``update`` merges by
shape) so it cannot affect other models, whose (M, K, N) differ. Two DiT (M, K, N) collide across use
cases (proj_mlp "plain" vs ff1 "plain_gelu"); the block is chosen to minimize the op-count-weighted
total (proj_mlp runs ~38x/forward vs ff1 ~8x). Regenerate with the relevant sweep if shapes change.
"""
from ...utils.matmul import register_matmul_configs

# DiT denoise blockings. Grid key -> (M, K, N) -> (M_block, K_block, N_block, (subblock_h, subblock_w)).
_DIT_MATMUL_CONFIGS = {
    "12x10": {
        # FIBO denoise on the 2x2 Blackhole mesh (sp=2/tp=2), 12x10 compute grid.
        (2048, 7680, 3072): (4, 5, 14, (2, 2)),  # single proj_out spatial
        (2048, 3072, 6144): (4, 4, 16, (2, 2)),  # ff1 (gelu) / proj_mlp spatial (weighted pick)
        (2048, 6144, 3072): (4, 4, 10, (2, 2)),  # ff2 spatial
        (2048, 3072, 4608): (4, 4, 15, (4, 1)),  # to_qkv spatial
        (2048, 3072, 1536): (6, 3, 5, (3, 1)),  # attn to_out spatial
        (2048, 3072, 64): (3, 8, 2, (3, 1)),  # final proj_out
        (2048, 64, 1536): (16, 2, 5, (4, 1)),  # x_embedder
        (128, 7680, 3072): (2, 8, 14, (2, 2)),  # single proj_out prompt
        (128, 3072, 6144): (4, 3, 16, (2, 2)),  # proj_mlp / ff1 prompt (weighted pick)
        (128, 6144, 3072): (2, 4, 8, (2, 2)),  # ff2 prompt
        (128, 3072, 4608): (2, 3, 15, (1, 3)),  # to_qkv prompt
        (128, 3072, 1536): (2, 8, 4, (2, 2)),  # attn to_add_out prompt
        (128, 4096, 1536): (2, 8, 4, (2, 2)),  # context_embedder
        (128, 2048, 1536): (2, 8, 4, (2, 2)),  # caption_projection
        (32, 3072, 9216): (4, 4, 8, (2, 2)),  # norm1 modulation
        (32, 3072, 6144): (2, 2, 16, (2, 2)),  # time_embed_out
        (32, 3072, 4608): (2, 3, 14, (2, 2)),  # single time_embed
        (32, 3072, 3072): (2, 4, 14, (2, 2)),  # timestep_embedder linear_2
        (32, 256, 3072): (2, 2, 8, (2, 2)),  # timestep_embedder linear_1
        # --- FIBO denoise on the 4x8 Galaxy (sp=4/tp=8): M=1024 spatial (4096/sp), M=128 prompt,
        # M=32 timestep; N/K are tp=8-sharded so they differ from the 2x2/tp=2 shapes above. Swept
        # 2026-07-15 via sweep_mm_block_sizes.py (bh_4x8_fibo, 12x10). The 4 tp-independent shapes
        # (32,256,3072 / 32,3072,3072 / 32,3072,6144 / 128,2048,1536) already hit the 2x2 12x10
        # entries above, so only the 14 new (tp-dependent) shapes are added here. ns = HiFi2 per-op.
        (1024, 1536, 3072): (8, 3, 10, (2, 2)),  # dual ff.ff2 spatial — 70330 ns
        (1024, 1920, 3072): (10, 3, 12, (2, 2)),  # single proj_out spatial — 79605 ns
        (1024, 3072, 1152): (8, 6, 4, (2, 2)),  # to_qkv spatial — 65895 ns
        (1024, 3072, 1536): (4, 6, 4, (2, 2)),  # dual ff.ff1 / proj_mlp spatial — 101395 ns
        (1024, 3072, 384): (3, 6, 2, (3, 1)),  # attn to_out spatial — 40553 ns
        (1024, 3072, 64): (3, 8, 2, (3, 1)),  # final proj_out — 34773 ns
        (1024, 64, 384): (3, 2, 2, (3, 1)),  # x_embedder — 5864 ns
        (128, 1536, 3072): (2, 4, 14, (2, 2)),  # dual ff_context.ff2 prompt — 47116 ns
        (128, 1920, 3072): (2, 4, 10, (2, 2)),  # single proj_out prompt twin — 56373 ns
        (128, 3072, 1152): (2, 8, 4, (2, 2)),  # to_qkv prompt — 38160 ns
        (128, 3072, 384): (2, 8, 2, (2, 2)),  # attn to_add_out prompt — 21230 ns
        (128, 4096, 384): (2, 8, 2, (2, 2)),  # context_embedder — 25822 ns
        (32, 3072, 1152): (2, 8, 6, (2, 2)),  # single-block time_embed — 37902 ns
        (32, 3072, 2304): (2, 6, 6, (2, 2)),  # norm1 modulation — 64667 ns
        # DiT prompt-branch matmuls at M=864 (long structured-JSON caption -> 833 tokens tile-padded
        # to 864; the committed fibo_vlm_prompt.json). The prompt branch runs UNPADDED at the true
        # token length (encoder unpads at text_encoder.py:160), so a long prompt lands at M=864 --
        # distinct from the M=128 (~128-token) / M=32 (short) twins above. Swept 2026-07-16 via
        # sweep_mm_block_sizes.py (bh_4x8_fibo, 12x10, all 7 measured 0-OOM). ns = HiFi2 per-op.
        (864, 4096, 384): (3, 8, 2, (3, 1)),  # context_embedder — 49220 ns
        (864, 2048, 1536): (3, 4, 4, (1, 4)),  # caption_projection — 46424 ns
        (864, 3072, 1152): (3, 4, 4, (1, 4)),  # to_qkv prompt (chunks=3, approx) — 52626 ns
        (864, 3072, 384): (3, 8, 2, (3, 1)),  # attn to_add_out prompt (addcmul, approx) — 39992 ns
        (864, 3072, 1536): (3, 6, 4, (1, 4)),  # ff_context.ff1 prompt (fused GELU) — 81504 ns
        (864, 1536, 3072): (3, 4, 8, (1, 4)),  # ff_context.ff2 prompt — 61876 ns
        (864, 1920, 3072): (3, 4, 16, (1, 4)),  # single proj_out prompt — 71226 ns
        # DiT prompt-branch matmuls at M=1024 (keep_padding=True: the JSON caption's 833 tokens pad
        # to the fixed 1024 bucket instead of the true M=864). The 5 shared (K,N) shapes reuse the
        # M=1024 spatial entries above; only these 2 are prompt-only (context_embedder K=4096,
        # caption_projection K=2048 have no spatial twin). Seeded from the M=864 winners (same K,N)
        # -- RE-SWEEP at M=1024 via sweep_mm_block_sizes.py (bh_4x8_fibo, 12x10).
        (1024, 4096, 384): (3, 8, 2, (3, 1)),  # context_embedder prompt (seed from M=864)
        (1024, 2048, 1536): (3, 4, 4, (1, 4)),  # caption_projection prompt (seed from M=864)
        # DiT prompt-branch matmuls at M=256: the short/empty CFG NEGATIVE branch under the 256
        # encoder bucket (keep_padding pads it to the 256 bucket). Distinct from the M=1024
        # padded-positive, M=864 unpadded-JSON, and M=128/M=32 twins. Swept 2026-07-23 via
        # sweep_mm_block_sizes.py (bh_4x8_fibo, 12x10; all 7 measured 0-OOM). M_block=2 throughout
        # (M=256 = 8 tiles); sb (2,2) forced by fp32 dest. ns = HiFi2 per-op.
        (256, 4096, 384): (2, 8, 4, (2, 2)),  # context_embedder prompt — 26480 ns
        (256, 2048, 1536): (2, 8, 8, (2, 2)),  # caption_projection prompt — 35288 ns
        (256, 3072, 1152): (2, 8, 4, (2, 2)),  # to_qkv prompt (chunks=3, approx) — 38632 ns
        (256, 3072, 384): (2, 8, 2, (2, 2)),  # attn to_add_out prompt (addcmul, approx) — 21512 ns
        (256, 3072, 1536): (2, 6, 4, (2, 2)),  # ff_context.ff1 prompt (fused GELU) — 65167 ns
        (256, 1536, 3072): (2, 4, 8, (2, 2)),  # ff_context.ff2 prompt — 47950 ns
        (256, 1920, 3072): (2, 4, 10, (2, 2)),  # single proj_out prompt — 56791 ns
        # DiT prompt-branch matmuls at M=32 (short / empty-CFG-uncond prompt; the M=128 twins are
        # for a ~128-token prompt). The 2 small-N shapes swept cleanly; the 4 large-N shapes hit a
        # profiler-buffer failure at M=32, so they reuse their M=128 prompt winners (M=32 is 1 tile
        # -> only M_block differs and it clamps; far better than the generic fallback). 2026-07-15.
        (32, 3072, 384): (2, 8, 2, (2, 2)),  # to_add_out prompt (M=32) — 21061 ns
        (32, 4096, 384): (2, 8, 4, (2, 2)),  # context_embedder (M=32) — 25547 ns
        (32, 1536, 3072): (2, 4, 14, (2, 2)),  # ff_context.ff2 prompt (M=32, reuse M=128)
        (32, 1920, 3072): (2, 4, 10, (2, 2)),  # single proj_out prompt (M=32, reuse M=128)
        (32, 2048, 1536): (2, 8, 4, (2, 2)),  # caption_projection (M=32, reuse M=128)
        (32, 3072, 1536): (2, 8, 4, (2, 2)),  # ff_context.ff1 prompt (M=32, reuse M=128)
    },
    # FIBO denoise on the 4x8 Galaxy at 11x10 (the historical Galaxy grid clamp). FIBO registered
    # nothing at 11x10 before, so all 19 4x8 shapes are added. Kept as a fallback for when the
    # matmul core grid is clamped back to 11x10 (see get_matmul_core_grid in utils/matmul.py).
    # Swept 2026-07-15, same run as the 12x10 block above.
    "11x10": {
        (1024, 1536, 3072): (8, 4, 10, (2, 2)),  # dual ff.ff2 spatial — 78747 ns
        (1024, 1920, 3072): (12, 4, 10, (2, 2)),  # single proj_out spatial — 90564 ns
        (1024, 3072, 1152): (8, 6, 4, (2, 2)),  # to_qkv spatial — 65969 ns
        (1024, 3072, 1536): (4, 3, 5, (4, 1)),  # dual ff.ff1 / proj_mlp spatial — 114275 ns
        (1024, 3072, 384): (3, 6, 2, (3, 1)),  # attn to_out spatial — 40002 ns
        (1024, 3072, 64): (3, 8, 2, (3, 1)),  # final proj_out — 34826 ns
        (1024, 64, 384): (3, 2, 2, (3, 1)),  # x_embedder — 5742 ns
        (128, 1536, 3072): (2, 3, 10, (2, 2)),  # dual ff_context.ff2 prompt — 51235 ns
        (128, 1920, 3072): (2, 4, 10, (2, 2)),  # single proj_out prompt twin — 61632 ns
        (128, 3072, 1152): (2, 8, 4, (2, 2)),  # to_qkv prompt — 44027 ns
        (128, 3072, 384): (2, 8, 2, (2, 2)),  # attn to_add_out prompt — 28460 ns
        (128, 4096, 384): (2, 8, 2, (2, 2)),  # context_embedder — 35620 ns
        (32, 3072, 1152): (2, 8, 6, (2, 2)),  # single-block time_embed — 43811 ns
        (32, 3072, 2304): (2, 6, 8, (2, 2)),  # norm1 modulation — 71665 ns
        (128, 2048, 1536): (2, 4, 6, (2, 2)),  # caption_projection — 39750 ns
        (128, 3072, 1536): (2, 6, 5, (2, 1)),  # dual ff_context.ff1 prompt — 76822 ns
        # DiT prompt-branch matmuls at M=1024 (keep_padding=True, 11x10 fallback grid). The 5 shared
        # (K,N) shapes reuse the M=1024 spatial entries above; only these 2 are prompt-only. Seeded
        # from the 12x10 M=864 winners -- RE-SWEEP at M=1024 (bh_4x8_fibo, 11x10).
        (1024, 4096, 384): (3, 8, 2, (3, 1)),  # context_embedder prompt (seed from M=864)
        (1024, 2048, 1536): (3, 4, 4, (1, 4)),  # caption_projection prompt (seed from M=864)
        # NOTE (2026-07-23): the 256 encoder bucket adds a short/empty CFG negative DiT prompt
        # branch at M=256. The 7 M=256 prompt (K,N) shapes were swept and registered for the 12x10
        # runtime grid above; this 11x10 dormant-fallback grid was NOT swept, so M=256 falls back to
        # the generic matmul config here (correct, not perf-tuned). RE-SWEEP at 11x10 if the grid is
        # ever clamped back (see get_matmul_core_grid).
        (32, 256, 3072): (2, 2, 10, (2, 2)),  # timestep_embedder linear_1 — 15618 ns
        (32, 3072, 3072): (2, 4, 16, (2, 2)),  # timestep_embedder linear_2 — 89023 ns
        (32, 3072, 6144): (2, 8, 12, (2, 2)),  # time_embed_out — 164954 ns
        # DiT prompt-branch matmuls at M=32 (11x10 fallback grid), same rationale as the 12x10 block.
        (32, 3072, 384): (2, 8, 2, (2, 2)),  # to_add_out prompt (M=32) — 28426 ns
        (32, 4096, 384): (4, 16, 2, (2, 2)),  # context_embedder (M=32) — 35392 ns
        (32, 3072, 1536): (2, 6, 5, (2, 1)),  # ff_context.ff1 prompt (M=32) — 76036 ns
        (32, 1536, 3072): (2, 3, 10, (2, 2)),  # ff_context.ff2 prompt (M=32, reuse M=128)
        (32, 1920, 3072): (2, 4, 10, (2, 2)),  # single proj_out prompt (M=32, reuse M=128)
        (32, 2048, 1536): (2, 4, 6, (2, 2)),  # caption_projection (M=32, reuse M=128)
        # --- FIBO DiT DENOISE on the 2x2 BH DEV mesh (sp=2/tp=2) at its ACTUAL 11x10 grid.
        # This board's compute_with_storage_grid_size() reports 11x10 (harvested column) and
        # get_matmul_core_grid does NOT clamp for a 4-device mesh, so the 2x2 denoise matmuls run at
        # 11x10 and MISS the 2x2 configs registered above under "12x10" (same for the encoder/VAE
        # tables below). Swept 2026-07-23 via sweep_mm_block_sizes.py (bh_2x2, 11x10) from the "No
        # known best blocking ... on 11x10" warnings in
        # test_fibo_pipeline_perf_breakdown_json[mesh_device0]. All plain except the DiT FFN ff1
        # (M,3072,6144) fused-exact-GELU (registered as the binding L1 case; proj_mlp twin is
        # plain). ns = HiFi2 per-op device kernel duration.
        # DiT denoise spatial (M=2048 = 4096 seq / sp2):
        (2048, 3072, 4608): (4, 6, 15, (4, 1)),  # to_qkv spatial — 310080 ns
        (2048, 3072, 6144): (8, 3, 6, (2, 2)),  # ff.ff1 spatial (fused GELU; proj_mlp twin) — 726364 ns
        (2048, 6144, 3072): (4, 6, 5, (4, 1)),  # ff.ff2 spatial — 410618 ns
        (2048, 7680, 3072): (4, 10, 5, (4, 1)),  # single-block proj_out spatial — 480209 ns
        (2048, 3072, 1536): (6, 3, 5, (3, 1)),  # attn to_out spatial — 106198 ns
        (2048, 3072, 64): (3, 8, 2, (3, 1)),  # final proj_out — 62730 ns
        (2048, 64, 1536): (16, 2, 6, (2, 2)),  # x_embedder (in_channels 48->64) — 25864 ns
        # DiT denoise prompt (M=1024 pad bucket / M=256 negative bucket):
        (1024, 3072, 4608): (6, 4, 8, (2, 2)),  # to_qkv prompt — 175164 ns
        (1024, 3072, 6144): (4, 3, 6, (2, 2)),  # ff_context.ff1 prompt (fused GELU; proj_mlp twin) — 375314 ns
        (1024, 4096, 1536): (8, 4, 5, (4, 1)),  # context_embedder prompt — 90365 ns
        (1024, 6144, 3072): (4, 6, 5, (4, 1)),  # ff_context.ff2 prompt — 225166 ns
        (1024, 7680, 3072): (4, 8, 5, (4, 1)),  # single-block proj_out prompt — 272160 ns
        (256, 3072, 4608): (4, 3, 16, (2, 2)),  # to_qkv prompt (M=256) — 129434 ns
        (256, 3072, 6144): (2, 32, 2, (2, 2)),  # ff_context.ff1 prompt (M=256, fused GELU; twin) — 198330 ns
        (256, 3072, 1536): (2, 8, 6, (2, 2)),  # attn to_add_out prompt (M=256) — 53370 ns
        (256, 4096, 1536): (2, 8, 5, (2, 1)),  # context_embedder prompt (M=256) — 67512 ns
        (256, 2048, 1536): (2, 4, 5, (2, 1)),  # caption_projection (M=256) — 39370 ns
        (256, 6144, 3072): (4, 4, 10, (2, 2)),  # ff_context.ff2 prompt (M=256) — 162870 ns
        (256, 7680, 3072): (4, 20, 6, (2, 2)),  # single-block proj_out prompt (M=256) — 199946 ns
        # DiT timestep / AdaLN modulation (M=32):
        (32, 3072, 4608): (2, 6, 8, (2, 2)),  # single-block time_embed — 128392 ns
        (32, 3072, 9216): (2, 16, 4, (2, 2)),  # norm1/norm1_context AdaLN modulation — 239496 ns
    },
}

# SmolLM3 text-encoder (tensor-parallel) blockings. SmolLM3 has no matmul registration of its own
# and FIBO is its only user, so its configs live in the FIBO table (additive; the K=2048/1376/5504
# keys are distinct from the DiT's).
_SMOLLM3_MATMUL_CONFIGS = {
    "12x10": {
        # tp=8 on the 4x8 Galaxy. M=32 (short prompt, one tile), K=2048=hidden. Swept 2026-07-15
        # (bh_4x8_fibo). Longer prompts give larger M -> a follow-up.
        (32, 2048, 256): (2, 8, 4, (2, 2)),  # kv proj (grouped-query) — 16107 ns
        (32, 2048, 512): (2, 8, 2, (2, 2)),  # q proj / attn out — 21747 ns
        (32, 2048, 1376): (2, 8, 4, (2, 2)),  # MLP gate/up proj — 33838 ns
        (32, 1376, 2048): (2, 43, 2, (2, 2)),  # MLP down proj (RowParallel) — 49293 ns
    },
    "11x10": {
        # tp=8 at 11x10 (the dormant Galaxy fallback grid). (32,2048,256) produced no OK sweep rows
        # at 11x10, so it reuses its 12x10 winner (same matmul, adjacent grid; beats the generic
        # default).
        (32, 2048, 256): (2, 8, 4, (2, 2)),  # kv proj (reused 12x10 winner)
        (32, 2048, 512): (2, 8, 2, (2, 2)),  # q proj / attn out — 21460 ns
        (32, 2048, 1376): (2, 8, 4, (2, 2)),  # MLP gate/up proj — 33524 ns
        (32, 1376, 2048): (2, 43, 2, (2, 2)),  # MLP down proj — 47616 ns
        # tp=2 on the 2x2 BH dev mesh at its actual 11x10 grid (see the DiT 11x10 note above).
        # M = token bucket 128/512. Swept 2026-07-23 (bh_2x2, 11x10). ns = HiFi2 per-op.
        (128, 2048, 1024): (2, 8, 4, (2, 2)),  # o_proj — 27144 ns
        (128, 2048, 5504): (2, 2, 16, (2, 2)),  # MLP gate/up proj — 106854 ns
        (128, 5504, 2048): (2, 4, 6, (2, 2)),  # MLP down proj — 105921 ns
        (512, 2048, 1024): (2, 4, 6, (2, 2)),  # o_proj — 31728 ns
        (512, 2048, 1536): (2, 4, 5, (2, 1)),  # qkv_proj (grouped-query) — 41340 ns
        (512, 2048, 5504): (6, 4, 8, (2, 2)),  # MLP gate/up proj — 111325 ns
        (512, 5504, 2048): (2, 4, 7, (2, 1)),  # MLP down proj — 108604 ns
    },
}

# VAE decoder blockings (mid-block attention M=4096; 1x1 convs lowered to matmul). 2x2 BH dev mesh
# at its actual 11x10 grid (see the DiT 11x10 note above). Swept 2026-07-23 (bh_2x2, 11x10).
_VAE_MATMUL_CONFIGS = {
    "11x10": {
        (4096, 1024, 1024): (14, 4, 4, (2, 2)),  # mid attn proj (dim 1024) — 75292 ns
        (4096, 1024, 3072): (14, 8, 6, (2, 2)),  # mid attn to_qkv (1024->3*1024) — 186970 ns
        (128, 1024, 512): (2, 4, 2, (2, 2)),  # conv_shortcut 1x1 (1024->512) — 14232 ns
        (256, 512, 256): (4, 8, 2, (2, 2)),  # conv_shortcut 1x1 (512->256) — 7254 ns
        (32, 64, 64): (2, 2, 2, (2, 2)),  # small 1x1 conv/proj (64-ch) — 3511 ns
    },
}

_registered: set[str] = set()


def _register_once(key: str, configs: dict) -> None:
    if key in _registered:
        return
    register_matmul_configs(configs)
    _registered.add(key)


def register_fibo_dit_matmul_configs() -> None:
    """Idempotently inject the FIBO DiT blockings into the shared matmul tables."""
    _register_once("dit", _DIT_MATMUL_CONFIGS)


def register_fibo_smollm3_matmul_configs() -> None:
    """Idempotently inject the FIBO SmolLM3 text-encoder blockings into the shared matmul tables."""
    _register_once("smollm3", _SMOLLM3_MATMUL_CONFIGS)


def register_fibo_vae_matmul_configs() -> None:
    """Idempotently inject the FIBO VAE-decoder blockings into the shared matmul tables."""
    _register_once("vae", _VAE_MATMUL_CONFIGS)
