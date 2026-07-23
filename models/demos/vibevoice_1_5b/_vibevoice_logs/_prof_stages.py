# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-stage device-synced micro-profile of the VibeVoice per-frame compute (current bf16 build).

Builds the real pipeline once, then times each hot per-frame stage in isolation on device with a
sync after each (median of many iters, after warm-up): the S-step diffusion loop, the vocoder
(acoustic decoder), the semantic encoder, and the LM decode step. Also times the individual
conv sub-stubs the vocoder composes (SConvTranspose1d / SConv1d / Block1D) so we know which
op to attack for the layout-churn rewrite. No trace — this is raw eager op timing to attribute cost.
"""
import sys
import time

import torch

sys.path.insert(0, "/local/ttuser/teja/tt-metal")
import ttnn
from models.demos.vibevoice_1_5b.tt import pipeline as P
from models.demos.vibevoice_1_5b.tt._golden import reference as R

N, S = 4, 20
ITERS = 20


def _sync(device):
    ttnn.synchronize_device(device)


def _time(device, fn, iters=ITERS):
    fn()
    _sync(device)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1e3  # median ms


def main():
    model = R.load_reference_model()
    processor = R.build_processor()
    tok = processor.tokenizer
    inputs = dict(R.make_inputs(processor, "Speaker 0: Hello there, this is a test.", R.default_voice_sample()))
    inputs["noises"] = R.make_noises(N + 2, int(model.config.acoustic_vae_dim))

    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    try:
        pipe = P.VibeVoiceTTS(device, model, N=N, S=S, use_trace=False)
        H, AV = pipe.hidden, pipe.acoustic_vae

        # a representative condition (from a real prefill) + noise, on device
        embeds = pipe._prefill_embeds(inputs)
        Lp = int(embeds.shape[1])
        C = ((Lp + N + 8 + 31) // 32) * 32
        kvb = [
            (
                P._tt(torch.zeros(1, pipe.num_kv_heads, C, pipe.head_dim), device=device),
                P._tt(torch.zeros(1, pipe.num_kv_heads, C, pipe.head_dim), device=device),
            )
            for _ in range(pipe.num_layers)
        ]
        hidden = pipe.qwen(inputs_embeds=embeds, kv_buffers=kvb)
        cond = ttnn.reshape(ttnn.slice(hidden, [0, Lp - 1, 0], [1, Lp, H]), [1, H])
        noise = P._tt(inputs["noises"][0].to(torch.float32), device=device)

        pipe._dpm_schedule(S)
        latent = pipe._sample_latent(cond, noise)  # [1,AV] on device
        scaled = ttnn.reshape(ttnn.subtract(ttnn.multiply(latent, 1.0 / pipe.scaling), pipe.bias), [1, AV, 1])
        audio = pipe.decoder(scaled)  # [1,1,3200]

        # ── time the per-frame stages ──────────────────────────────────────────
        t_diff = _time(device, lambda: pipe._sample_latent(cond, noise))
        t_voc = _time(device, lambda: pipe.decoder(scaled))
        t_sem = _time(device, lambda: pipe.semantic_model(audio))
        emb1 = P._tt(torch.zeros(1, 1, H), device=device)
        cos_b, sin_b = pipe._rope(Lp, 1)
        cos_t, sin_t = P._tt(cos_b, device=device), P._tt(sin_b, device=device)
        dm = torch.zeros(1, 1, 1, C)
        mask_t = P._tt(dm, device=device)
        oh = P._tt(pipe._onehot(Lp, C), device=device)
        t_lm = _time(
            device,
            lambda: pipe.qwen(
                inputs_embeds=emb1, kv_buffers=kvb, ext_cos=cos_t, ext_sin=sin_t, ext_mask=mask_t, write_onehot=oh
            ),
        )

        total = t_diff + t_voc + t_sem + t_lm
        print("================= per-frame stage timing (eager, device-synced, ms) =================", flush=True)
        print(f"diffusion loop (S={S}) : {t_diff:8.2f}  ({100*t_diff/total:4.1f}%)", flush=True)
        print(f"vocoder (decoder)      : {t_voc:8.2f}  ({100*t_voc/total:4.1f}%)", flush=True)
        print(f"semantic encoder       : {t_sem:8.2f}  ({100*t_sem/total:4.1f}%)", flush=True)
        print(f"LM decode step         : {t_lm:8.2f}  ({100*t_lm/total:4.1f}%)", flush=True)
        print(f"sum                    : {total:8.2f}", flush=True)

        # ── time the vocoder's constituent sub-stubs (attribute the layout-churn) ──
        print("\n================= vocoder sub-stub timing (single call, ms) =================", flush=True)
        m = model.model.acoustic_tokenizer.decoder
        from models.demos.vibevoice_1_5b._stubs.block1_d import build as _bblk
        from models.demos.vibevoice_1_5b._stubs.s_conv1d import build as _bsc
        from models.demos.vibevoice_1_5b._stubs.s_conv_transpose1d import build as _bsct

        x = scaled
        for i, layer_seq in enumerate(m.upsample_layers):
            layer = layer_seq[0]
            kind = type(layer).__name__
            fwd = _bsct(device, layer) if kind == "SConvTranspose1d" else _bsc(device, layer)
            xin = x
            t = _time(device, lambda: fwd(xin), iters=8)
            x = fwd(x)
            T = int(x.shape[-1])
            print(f"  upsample[{i}] {kind:18s}: {t:7.2f} ms  -> T_out={T}", flush=True)
            for j, blk in enumerate(m.stages[i]):
                bf = _bblk(device, blk)
                xin2 = x
                tb = _time(device, lambda: bf(xin2), iters=8)
                x = bf(x)
                print(f"      block1d[{i}.{j}]           : {tb:7.2f} ms  T={int(x.shape[-1])}", flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
