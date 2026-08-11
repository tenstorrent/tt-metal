# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Phase B (python_env, TT): FULL GPT on device — prefill + KV-cached decode.

Prefills coqui's real prompt [cond, start_text, text, stop_text, START_AUDIO] into the
KV cache on TT, then autoregressively generates audio codes with the traced TTNNGPTTracedDecoder.
Sampling (mel_head + repetition penalty + argmax) is done on the host for now (the plan is
to move it onto the device later). The latents produced at each generated position are the
vocoder input (gpt_latents)."""
import argparse, os

import torch
import ttnn

from models.experimental.xtts_v2.reference.xtts_gpt_ref import load_gen_head
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt import preprocess_gpt_parameters
from models.experimental.xtts_v2.tt.ttnn_xtts_gpt_decode import TTNNGPTTracedDecoder
from models.experimental.xtts_v2.tt.ttnn_xtts_cond import (
    LATENTS,
    TTNNConditioningEncoder,
    TTNNPerceiver,
    preprocess_encoder_parameters,
    preprocess_perceiver_parameters,
)
from models.experimental.xtts_v2.tt.ttnn_xtts_speaker import TTNNSpeakerEncoder, preprocess_speaker_parameters


def block2_speaker_embedding(dev, work):
    """Run Block 2 (ResNet speaker encoder) on TT from the reference logmel; returns
    speaker_embedding [1,512,1] (torch). The mel/STFT front-end stays on CPU (phase A)."""
    logmel = torch.load(os.path.join(work, "speaker_logmel.pt")).float()  # [1,64,T]
    model = TTNNSpeakerEncoder(dev, preprocess_speaker_parameters(dev))
    logmel_tt = ttnn.from_torch(logmel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    emb = model(logmel_tt)
    if isinstance(emb, tuple):
        emb = emb[0]
    return ttnn.to_torch(emb).to(torch.float32).reshape(1, 512, 1)


def block1_gpt_cond_latent(dev, work):
    """Run Block 1 (conditioning encoder + Perceiver) on TT from the reference mel;
    returns gpt_cond_latent [1,32,1024] (torch)."""
    mel = torch.load(os.path.join(work, "cond_mel_in.pt")).float()  # [1,80,T]
    T = mel.shape[2]
    S = ((T + 31) // 32) * 32
    mel_f = torch.nn.functional.pad(mel.permute(0, 2, 1).contiguous(), (0, 0, 0, S - T))  # [1,S,80]
    enc = TTNNConditioningEncoder(dev, preprocess_encoder_parameters(dev, dtype=ttnn.float32), t_real=T, s_pad=S)
    perc = TTNNPerceiver(dev, preprocess_perceiver_parameters(dev, dtype=ttnn.float32))
    km = torch.zeros(1, 1, 1, LATENTS + S)
    km[:, :, :, LATENTS + T :] = -1e9
    km_tt = ttnn.from_torch(km, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    frames = enc(ttnn.from_torch(mel_f, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev))
    return ttnn.to_torch(perc(frames, km_tt)).to(torch.float32)  # [1,32,1024]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--max-new", type=int, default=605)
    args = ap.parse_args()

    prefix = torch.load(os.path.join(args.work, "prefix_emb.pt")).float()  # [1, P, 1024]
    meta = torch.load(os.path.join(args.work, "gen_meta.pt"))
    start_audio, stop_audio = meta["start_audio"], meta["stop_audio"]
    penalty = meta["repetition_penalty"]
    heads = load_gen_head()
    mel_emb, mel_pos = heads["mel_emb"], heads["mel_pos"]
    mh_w, mh_b = heads["mel_head_w"], heads["mel_head_b"]
    P = prefix.shape[1]
    # BUG-1 fixed: the decoder rounds max_seq up to an even tile count internally, so we can
    # size the cache to the model's true limit (gpt_max_audio_tokens=605) and let generation
    # stop naturally on stop_audio -- no longer capped to coqui's oracle sequence length.
    GPT_MAX_AUDIO = 605
    max_seq = P + 1 + GPT_MAX_AUDIO  # prefill(P) + START(1) + up to 605 codes
    max_new = GPT_MAX_AUDIO
    S_hint = torch.load(os.path.join(args.work, "emb.pt")).shape[1]  # coqui length, for reference only
    print(
        f"[B] prefix P={P} coqui_S={S_hint} max_seq={max_seq} max_new={max_new} "
        f"start_audio={start_audio} stop_audio={stop_audio} penalty={penalty}"
    )

    # coqui's default decode strategy: stochastic sampling (do_sample=True) with repetition
    # penalty + temperature + top-k + top-p, then a multinomial draw — NOT greedy argmax.
    # (Greedy was only for golden-exact validation; it collapses into repetition attractors on
    # long free-running generations. Sampling injects randomness so no deterministic fixed point
    # can trap it.) Params are coqui's Xtts.inference defaults.
    TEMPERATURE, TOP_K, TOP_P = 0.75, 50, 0.85
    torch.manual_seed(0)  # reproducible draws

    def sample(latent, seen):
        logits = (latent @ mh_w.t() + mh_b)[0, 0].clone().float()  # [1026]
        for tok in seen:  # repetition penalty (HF applies this first)
            logits[tok] = logits[tok] / penalty if logits[tok] > 0 else logits[tok] * penalty
        logits = logits / TEMPERATURE  # temperature
        if TOP_K and TOP_K < logits.numel():  # top-k
            kth = torch.topk(logits, TOP_K).values[-1]
            logits[logits < kth] = float("-inf")
        if TOP_P < 1.0:  # top-p (nucleus)
            sl, si = torch.sort(logits, descending=True)
            drop = torch.softmax(sl, dim=-1).cumsum(dim=-1) > TOP_P
            drop[1:] = drop[:-1].clone()
            drop[0] = False  # always keep the top-1
            logits[si[drop]] = float("-inf")
        return int(torch.multinomial(torch.softmax(logits, dim=-1), 1))

    # trace_region_size: the traced GPT decoder captures its per-step graph into a trace.
    dev = ttnn.open_device(device_id=0, l1_small_size=65536, trace_region_size=60_000_000)
    try:
        dev.enable_program_cache()

        # --- Block 2 on TT: speaker d-vector from the reference logmel (feeds the vocoder) ---
        spk_tt = block2_speaker_embedding(dev, args.work)  # [1,512,1]
        spk_ref = torch.load(os.path.join(args.work, "speaker_embedding.pt")).float()
        num = (spk_ref.flatten() * spk_tt.flatten()).sum()
        den = spk_ref.flatten().norm() * spk_tt.flatten().norm()
        print(f"[B] Block2 speaker_embedding on TT {tuple(spk_tt.shape)}  cos-vs-coqui={(num/den).item():.5f}")
        torch.save(spk_tt, os.path.join(args.work, "speaker_embedding_tt.pt"))

        # --- Block 1 on TT: recompute gpt_cond_latent and splice into the prompt prefix ---
        cond_tt = block1_gpt_cond_latent(dev, args.work)  # [1,32,1024]
        cond_ref = prefix[:, :LATENTS, :]
        num = (cond_ref.flatten() * cond_tt.flatten()).sum()
        den = cond_ref.flatten().norm() * cond_tt.flatten().norm()
        print(f"[B] Block1 gpt_cond_latent on TT {tuple(cond_tt.shape)}  cos-vs-coqui-prefix={(num/den).item():.5f}")
        prefix[:, :LATENTS, :] = cond_tt  # use TT-computed conditioning in the prompt

        dec = TTNNGPTTracedDecoder(dev, preprocess_gpt_parameters(dev, dtype=ttnn.bfloat16), max_seq=max_seq)
        # One-shot parallel prefill of the prompt [cond, start_text, text, stop_text] (positions
        # 0..P-1 via fill_cache) instead of P single-token steps; capture AFTER prefill (it leaves
        # the prefilled cache intact). Decode then starts at position P (the START_AUDIO token).
        dec.reset_caches()
        dec.prefill(prefix.contiguous())  # prefix: torch [1, P, 1024]
        dec.capture()  # compile + capture the per-step decode graph into a trace

        pos_ctr = [P]  # first decode step (START_AUDIO) is at position P

        def step(emb_1x1):
            # driver owns the host<->device I/O; the decoder exposes only step_device (device in/out)
            emb_dev = ttnn.from_torch(
                emb_1x1.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                mesh_mapper=dec.mesh_mapper,
            )
            lat = ttnn.to_torch(dec.step_device(emb_dev, pos_ctr[0])).float()  # torch [1,1,1024]
            pos_ctr[0] += 1
            return lat

        # START_AUDIO (mel_pos[0]) at position P -> predicts code_0.
        last = step((mel_emb[start_audio] + mel_pos[0]).view(1, 1, -1))

        # Decode: last (START_AUDIO) predicts code_0; then feed each code at mel_pos[j+1].
        seen = {start_audio}
        code = sample(last, seen)
        codes, vlat = [code], []
        for j in range(max_new):
            inp = (mel_emb[codes[-1]] + mel_pos[j + 1]).view(1, 1, -1)
            lat = step(inp)
            vlat.append(lat)  # vocoder latent for codes[-1]
            seen.add(codes[-1])
            nxt = sample(lat, seen)
            if nxt == stop_audio:
                break
            codes.append(nxt)

        gpt_latents = torch.cat(vlat, dim=1)  # [1, T, 1024]

        # --- Block 4 on TT: HiFi-GAN vocoder (guarded until the module exists) ---
        # NOTE: the exact TTNNHifiganGenerator call/layout may need reconciling with the
        # agent's final interface. Falls back to coqui's vocoder (phase C) if unavailable.
        try:
            import torch.nn.functional as F
            from models.experimental.xtts_v2.tt.ttnn_xtts_hifigan import (
                TTNNHifiganGenerator,
                preprocess_hifigan_parameters,
            )

            # Build the generator input z from gpt_latents the way HifiDecoder.forward does:
            # two linear time-resizes (host, cheap). ar_mel_length_compression=1024 (gpt code
            # stride), output_hop_length=256, input_sr=22050, output_sr=24000.
            AR_COMP, HOP, ISR, OSR = 1024, 256, 22050, 24000
            z = F.interpolate(gpt_latents.transpose(1, 2), scale_factor=AR_COMP / HOP, mode="linear")
            z = F.interpolate(z, scale_factor=OSR / ISR, mode="linear")  # [1, 1024, L]

            # Interface (see tests/test_hifigan_pcc.py): z as NHWC fp32 ROW_MAJOR [1,1,L,1024];
            # g as fp32 TILE [1,512]; fp32 activations (bf16 tops out ~0.96 on the waveform).
            L = z.shape[-1]
            z_nhwc = z.permute(0, 2, 1).reshape(1, 1, L, 1024)
            voc = TTNNHifiganGenerator(dev, preprocess_hifigan_parameters(dev))
            z_tt = ttnn.from_torch(z_nhwc, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            g_tt = ttnn.from_torch(spk_tt.reshape(1, 512), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            wav = ttnn.to_torch(voc(z_tt, g_tt)).to(torch.float32).reshape(1, 1, -1)  # [1, 1, N]
            torch.save(wav, os.path.join(args.work, "vocoder_wav_tt.pt"))
            print(f"[B] Block4 HiFi-GAN on TT -> wav {tuple(wav.shape)}")
        except ImportError:
            print("[B] TT HiFi-GAN (Block 4) not available yet -> vocoder falls back to coqui (phase C)")
    finally:
        ttnn.close_device(dev)

    torch.save(gpt_latents, os.path.join(args.work, "gpt_latents_tt.pt"))
    print(f"[B] generated {len(codes)} codes, gpt_latents {tuple(gpt_latents.shape)}")
    print(f"[B] tt codes first16: {codes[:16]}")

    ref = os.path.join(args.work, "audio_codes.pt")
    if os.path.exists(ref):
        cq = torch.load(ref).flatten().tolist()
        lead = 0
        for a, b in zip(codes, cq):
            if a == b:
                lead += 1
            else:
                break
        print(f"[B] coqui codes first16: {cq[:16]}")
        print(f"[B] leading-match vs coqui greedy: {lead}/{min(len(codes), len(cq))}  (coqui len={len(cq)})")
    print("[B] wrote gpt_latents_tt.pt")


if __name__ == "__main__":
    main()
