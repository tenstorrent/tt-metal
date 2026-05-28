#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tracy harness for the SeamlessM4T-v2 ``CodeHifiGanVocoder`` block.

The code HiFi-GAN vocoder synthesises a 16 kHz waveform from a sequence
of RVQ unit-token ids. It is the FINAL stage of the T2ST and S2ST
audio-out pipelines (one ``synthesize()`` call per utterance).

Per-call structure (see ``tt/code_hifigan_vocoder.py``):

  1. ``ttnn.embedding`` on unit ids (B, T_in) -> (B, T_in, C_u)
  2. Duration predictor (TTNN VariancePredictor) (B, T_in)
  3. host_repeat_interleave (acknowledged host-side, per HF spec)
  4. host concat([lang, hidden, spkr]) -> upload [B, in_dim, T_up]
  5. ``HifiGanVocoder``: conv_pre -> 5x [LeakyReLU, ConvTranspose1d,
     MRF (3 x HifiGanResidualBlock summed/scaled)] -> LeakyReLU(0.01)
     -> conv_post -> tanh -> waveform [B, T_out]

The HiFi-GAN core is the dominant chunk of device time (conv + transpose
conv + residual blocks). This harness times the FULL ``CodeHifiGanVocoder``
forward end-to-end, but tracy attributes per-op device kernel duration so we
can see the conv breakdown in the CSV.

Production shapes (SeamlessM4T-v2-Large):

  - unit_vocab=10000, unit_embed_dim=1280
  - lang_embed_dim=256, spkr_embed_dim=256
  - in_dim = 1280 + 256 + 256 = 1792 channels into conv_pre
  - upsample_initial_channel C0 = 512
  - upsample_rates = (5, 4, 4, 2, 2), upsample_kernel_sizes = (11, 8, 8, 4, 4)
  - resblock_kernel_sizes = (3, 7, 11); dilations all (1, 3, 5)

Representative T_in: production utterances after T2U sampling are
~50-200 unit tokens. We default to T=64 (tile-aligned) with average
``dur_out`` per token = 4 -> T_up = 256, T_out = 256 * 320 = 81920
samples (~5.1 s @ 16 kHz). This is in the production envelope and
makes the conv stack the dominant cost.

Run untraced (host dispatch dominates the dispatch numbers; conv kernel
breakdown in CSV is still meaningful per-op):

    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) && \\
        export PYTHONPATH=$(pwd) && export ARCH_NAME=blackhole
    python models/demos/facebook_seamless_m4t_v2_large/tt/profile_code_hifigan_vocoder.py

Run under tracy (production path -- code_hifigan_vocoder runs one-shot per
synthesize call, NOT wrapped in metal trace by upstream callers, so the
"traced" flag here re-captures inside the harness to demonstrate the
per-op device kernel breakdown):

    python -m tracy -p -v -r --no-device-data-capture \\
        -o generated/profiler/reports/code_hifigan_vocoder_traced \\
        models/demos/facebook_seamless_m4t_v2_large/tt/profile_code_hifigan_vocoder.py --traced

The CSV at ``generated/profiler/.logs/cpp_device_perf_report.csv`` is the
authoritative artifact for hotspot analysis.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import List

import torch

import ttnn
from models.demos.facebook_seamless_m4t_v2_large.tt.code_hifigan_vocoder import CodeHifiGanVocoder

# Production shapes (SeamlessM4T-v2-Large).
UNIT_VOCAB = 10000
UNIT_EMBED_DIM = 1280
LANG_EMBED_DIM = 256
SPKR_EMBED_DIM = 256
NUM_LANGS = 36
NUM_SPKRS = 200
T_IN_DEFAULT = 64  # tile-aligned representative unit-token sequence length.
DUR_PER_TOKEN = 4  # average; -> T_up = 256, ~5.1 s @ 16 kHz.
UPSAMPLE_INITIAL_CHANNEL = 512
UPSAMPLE_RATES = (5, 4, 4, 2, 2)
UPSAMPLE_KERNEL_SIZES = (11, 8, 8, 4, 4)
RESBLOCK_KERNEL_SIZES = (3, 7, 11)
RESBLOCK_DILATION_SIZES = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
LEAKY_RELU_SLOPE = 0.1
VARIANCE_PREDICTOR_KERNEL_SIZE = 3
T2U_PAD_TOKEN_ID = 1
BATCH = 1


def _ln_state(dim: int) -> dict:
    return {"weight": torch.ones(dim), "bias": torch.zeros(dim)}


def _make_state_dict(seed: int = 0) -> dict:
    """Synthesize a representative state_dict matching CodeHifiGanVocoder."""

    # Unit / speaker / language embedding tables.
    g_emb = torch.Generator().manual_seed(seed)
    unit_emb = {"weight": torch.randn(UNIT_VOCAB, UNIT_EMBED_DIM, generator=g_emb) * 0.02}
    spkr_emb = {"weight": torch.randn(NUM_SPKRS, SPKR_EMBED_DIM, generator=g_emb) * 0.02}
    lang_emb = {"weight": torch.randn(NUM_LANGS, LANG_EMBED_DIM, generator=g_emb) * 0.02}

    # Duration predictor (VariancePredictor).
    def _vp_conv(seed_offset):
        g = torch.Generator().manual_seed(seed + seed_offset)
        # conv1d weight: [out_C, in_C, K]
        return {
            "weight": torch.randn(UNIT_EMBED_DIM, UNIT_EMBED_DIM, VARIANCE_PREDICTOR_KERNEL_SIZE, generator=g)
            * (1.0 / (UNIT_EMBED_DIM * VARIANCE_PREDICTOR_KERNEL_SIZE) ** 0.5),
            "bias": torch.zeros(UNIT_EMBED_DIM),
        }

    dur_predictor = {
        "conv1": _vp_conv(1),
        "ln1": _ln_state(UNIT_EMBED_DIM),
        "conv2": _vp_conv(2),
        "ln2": _ln_state(UNIT_EMBED_DIM),
        "proj": {
            "weight": torch.randn(1, UNIT_EMBED_DIM, generator=torch.Generator().manual_seed(seed + 3))
            * (1.0 / UNIT_EMBED_DIM**0.5),
            "bias": torch.zeros(1),
        },
    }

    # HiFi-GAN core.
    in_dim_vocoder = LANG_EMBED_DIM + UNIT_EMBED_DIM + SPKR_EMBED_DIM  # 1792
    g_pre = torch.Generator().manual_seed(seed + 4)
    conv_pre = {
        "weight": torch.randn(UPSAMPLE_INITIAL_CHANNEL, in_dim_vocoder, 7, generator=g_pre)
        * (1.0 / (in_dim_vocoder * 7) ** 0.5),
        "bias": torch.zeros(UPSAMPLE_INITIAL_CHANNEL),
    }
    # conv_post is [1, last_C, 7] where last_C is final upsampler out.
    last_C = UPSAMPLE_INITIAL_CHANNEL // (2 ** len(UPSAMPLE_RATES))  # 512/32 = 16
    g_post = torch.Generator().manual_seed(seed + 5)
    conv_post = {
        "weight": torch.randn(1, last_C, 7, generator=g_post) * (1.0 / (last_C * 7) ** 0.5),
        "bias": torch.zeros(1),
    }

    # Upsamplers: ConvTranspose1d weight is [in_C, out_C, K].
    upsampler = []
    cur_in = UPSAMPLE_INITIAL_CHANNEL
    for i, (rate, ks) in enumerate(zip(UPSAMPLE_RATES, UPSAMPLE_KERNEL_SIZES)):
        cur_out = cur_in // 2
        g_up = torch.Generator().manual_seed(seed + 100 + i)
        upsampler.append(
            {
                "weight": torch.randn(cur_in, cur_out, ks, generator=g_up) * (1.0 / (cur_in * ks) ** 0.5),
                "bias": torch.zeros(cur_out),
            }
        )
        cur_in = cur_out

    # Resblocks: 5 stages * 3 kernels = 15 blocks. Each has convs1 + convs2
    # (3 dilated convs + 3 dilation=1 convs) at the upsampled out channels.
    resblocks = []
    cur_C = UPSAMPLE_INITIAL_CHANNEL // 2  # 256 after first upsample
    for i in range(len(UPSAMPLE_RATES)):
        for j, k in enumerate(RESBLOCK_KERNEL_SIZES):
            convs1, convs2 = [], []
            for di_idx, _d in enumerate(RESBLOCK_DILATION_SIZES[j]):
                gw1 = torch.Generator().manual_seed(seed + 1000 + i * 100 + j * 10 + di_idx)
                gw2 = torch.Generator().manual_seed(seed + 2000 + i * 100 + j * 10 + di_idx)
                convs1.append(
                    {
                        "weight": torch.randn(cur_C, cur_C, k, generator=gw1) * (1.0 / (cur_C * k) ** 0.5),
                        "bias": torch.zeros(cur_C),
                    }
                )
                convs2.append(
                    {
                        "weight": torch.randn(cur_C, cur_C, k, generator=gw2) * (1.0 / (cur_C * k) ** 0.5),
                        "bias": torch.zeros(cur_C),
                    }
                )
            resblocks.append({"convs1": convs1, "convs2": convs2})
        cur_C = cur_C // 2  # next stage halves channels

    hifi_gan = {
        "conv_pre": conv_pre,
        "upsampler": upsampler,
        "resblocks": resblocks,
        "conv_post": conv_post,
    }

    return {
        "unit_embedding": unit_emb,
        "speaker_embedding": spkr_emb,
        "language_embedding": lang_emb,
        "dur_predictor": dur_predictor,
        "hifi_gan": hifi_gan,
    }


def _make_inputs(t_in: int, seed: int = 11):
    g = torch.Generator().manual_seed(seed)
    # Unit ids in [2, vocab); 0 == pad in HF spec but here we keep them all valid.
    input_ids = torch.randint(2, UNIT_VOCAB, (BATCH, t_in), generator=g, dtype=torch.long)
    speaker_id = torch.tensor([[3]], dtype=torch.long)
    lang_id = torch.tensor([[5]], dtype=torch.long)
    # Dur predictor's output drives the upsample factor. Since we use a random
    # state_dict, dur_predictor's outputs are not under our control, so we
    # don't try to pin T_up directly -- we just let the block run.
    return input_ids, speaker_id, lang_id


def _run_block(block: CodeHifiGanVocoder, input_ids, speaker_id, lang_id, n_iter: int):
    times: List[float] = []
    # Warmup -- compiles all kernels into the program cache.
    _ = block(input_ids=input_ids, speaker_id=speaker_id, lang_id=lang_id)
    ttnn.synchronize_device(block.device)

    for _ in range(n_iter):
        t0 = time.perf_counter()
        out = block(input_ids=input_ids, speaker_id=speaker_id, lang_id=lang_id)
        ttnn.synchronize_device(block.device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        # Force a host read to ensure the device actually flushed.
        _ = ttnn.to_torch(out)
    return times


def _stat(name: str, xs: List[float]) -> str:
    if not xs:
        return f"{name}=<empty>"
    return (
        f"{name}: n={len(xs)}, min={min(xs):.3f}, "
        f"p50={statistics.median(xs):.3f}, mean={statistics.mean(xs):.3f}, "
        f"max={max(xs):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traced",
        action="store_true",
        help=(
            "Tracy convention: profile the production path -- code_hifigan_vocoder is "
            "exercised one-shot per synthesize call (NOT under metal trace upstream), "
            "so the 'traced' flag here just signals the tracy harness has set this run "
            "up as the canonical per-op device kernel breakdown."
        ),
    )
    parser.add_argument("--n-iter", type=int, default=8, help="Steady-state iteration count.")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=T_IN_DEFAULT, help="Number of unit tokens (T_in).")
    args = parser.parse_args()

    print(
        f"[profile_code_hifigan_vocoder] T_in={args.seq_len} "
        f"upsample_rates={UPSAMPLE_RATES} "
        f"in_dim_vocoder={LANG_EMBED_DIM + UNIT_EMBED_DIM + SPKR_EMBED_DIM} "
        f"C0={UPSAMPLE_INITIAL_CHANNEL} traced={args.traced} n_iter={args.n_iter}"
    )

    # The vocoder allocates a lot of weight tensors at init; keep l1_small modest.
    device = ttnn.open_device(
        device_id=args.device_id,
        l1_small_size=32768,
    )
    try:
        sd = _make_state_dict()
        block = CodeHifiGanVocoder(
            device=device,
            state_dict=sd,
            pad_token_id=T2U_PAD_TOKEN_ID,
            variance_predictor_kernel_size=VARIANCE_PREDICTOR_KERNEL_SIZE,
            upsample_rates=UPSAMPLE_RATES,
            upsample_kernel_sizes=UPSAMPLE_KERNEL_SIZES,
            resblock_kernel_sizes=RESBLOCK_KERNEL_SIZES,
            resblock_dilation_sizes=RESBLOCK_DILATION_SIZES,
            leaky_relu_slope=LEAKY_RELU_SLOPE,
            weight_dtype=ttnn.bfloat16,
        )

        input_ids, speaker_id, lang_id = _make_inputs(args.seq_len)
        times = _run_block(block, input_ids, speaker_id, lang_id, args.n_iter)
        print(_stat("[code_hifigan_vocoder] step_ms", times))
        print("[profile_code_hifigan_vocoder] DONE")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
