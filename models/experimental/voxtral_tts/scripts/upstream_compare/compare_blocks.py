"""Numerical PCC of our Voxtral-TTS reference blocks against vLLM-Omni's own nn.Modules (CPU).

Run with cmp_venv (has einops + real torch) and the repo on PYTHONPATH:
    PYTHONPATH=$TT_METAL_HOME ./cmp_venv/bin/python compare_blocks.py
"""

import json
import os
import pathlib
import sys
import types

import torch

import upstream_loader as UL

REPO = os.environ.get("TT_METAL_HOME") or str(pathlib.Path(__file__).resolve().parents[4])
CKPT = f"{REPO}/models/experimental/voxtral_tts/reference/weights/consolidated.safetensors"
PARAMS = f"{REPO}/models/experimental/voxtral_tts/reference/weights/params.json"

sys.path.insert(0, REPO)
from models.experimental.voxtral_tts.reference import voxtral_codec_ref as C  # noqa: E402
from models.experimental.voxtral_tts.reference import voxtral_flow_ref as F  # noqa: E402
from models.experimental.voxtral_tts.reference.voxtral_common_ref import SafeTensors, pcc  # noqa: E402

RESULTS = []


def report(name, a, b, gate=0.9999):
    p = pcc(a, b)
    mx = (a.float() - b.float()).abs().max().item()
    ok = p >= gate
    RESULTS.append((name, p, mx, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name:52s} PCC {p:.8f}  maxabs {mx:.3e}")
    return ok


def load_params():
    with open(PARAMS) as f:
        d = json.load(f)
    args = d["multimodal"]["audio_model_args"]
    # upstream's parser injects this default when params.json omits it
    args["acoustic_transformer_args"].setdefault("n_decoding_steps", 7)
    return d, args


# =======================================================================================
# Block 2 — flow-matching acoustic transformer
# =======================================================================================
def block2(gen, audio_args):
    print("\n=== BLOCK 2: flow-matching acoustic transformer ===")
    up = gen.FlowMatchingAudioTransformer(dict(audio_args))
    st = SafeTensors(CKPT)
    sd = st.prefixed("acoustic_transformer.", torch.float32)
    sd.pop("time_embedding.inv_freq", None)
    missing, unexpected = up.load_state_dict(sd, strict=False)
    # inv_freq is a buffer upstream computes itself; everything else must be present
    missing = [m for m in missing if "inv_freq" not in m]
    print(f"[b2] upstream load: {len(sd)} tensors | missing {missing} | unexpected {list(unexpected)}")
    assert not missing and not unexpected
    up = up.float().eval()

    ours = F.load_flow_state(CKPT)
    B = 4
    torch.manual_seed(11)
    h = torch.randn(B, 3072)

    with torch.no_grad():
        # 1) semantic head (+ masking + argmax)
        up_logit = up.semantic_codebook_output(h).float()
        our_logit = torch.nn.functional.linear(h, ours["semantic_codebook_output.weight"])
        report("b2 semantic logits (raw head)", our_logit, up_logit)
        # masked argmax must agree exactly (it picks the code)
        up_masked = up_logit.clone()
        up_masked[:, up._empty_audio_token_id] = -float("inf")
        up_masked[:, (len(gen.AudioSpecialTokens) + up.model_args.semantic_codebook_size):] = -float("inf")
        up_code = up_masked.argmax(-1, keepdim=True)
        our_code = F.semantic_code(h, ours)
        same = bool((up_code == our_code).all())
        RESULTS.append(("b2 semantic code argmax (exact)", 1.0 if same else 0.0, 0.0, same))
        print(f"{'PASS' if same else 'FAIL'}  {'b2 semantic code argmax (exact)':52s} {up_code.flatten().tolist()}")

        # 2) time embedding
        t = torch.tensor([0.375])
        up_te = up.time_embedding(t.view(-1, 1).repeat(B, 1))
        our_te = F.time_embedding(t.view(1, 1).repeat(B, 1), ours["time_embedding.inv_freq"])
        report("b2 time embedding (sinusoidal)", our_te, up_te)

        # 3) one velocity evaluation (the unit a TTNN trace captures)
        x_t = torch.randn(B, 36)
        up_v = up._predict_velocity(x_t=x_t, llm_output=h, t_emb=up_te)
        our_v = F.predict_velocity(x_t, h, our_te, ours)
        report("b2 predict_velocity (3-token seq, 3 layers)", our_v, up_v)

        # 4) full frame: 7 Euler steps + CFG + FSQ. Seed so both draw the same x_0.
        cfg = torch.full((B,), 1.2)
        torch.manual_seed(4242)
        up_frame = up(llm_hidden=h, cfg_alpha=cfg)
        torch.manual_seed(4242)
        x0 = torch.randn(B, up.model_args.n_acoustic_codebook)
        our_frame = F.reference_frame(h, ours, cfg_alpha=1.2, x_0=x0)
        same = bool((up_frame == our_frame).all())
        RESULTS.append(("b2 full frame, 37 codes (EXACT ints)", 1.0 if same else 0.0, 0.0, same))
        detail = ("identical" if same else
                  f"{(up_frame != our_frame).sum().item()} of {up_frame.numel()} codes differ")
        print(f"{'PASS' if same else 'FAIL'}  {'b2 full frame, 37 codes (EXACT ints)':52s} {detail}")
        if not same:
            print("      upstream:", up_frame[0, :8].tolist())
            print("      ours    :", our_frame[0, :8].tolist())


# =======================================================================================
# Block 3 — codec decoder
# =======================================================================================
def block3(tokmod, full_params):
    print("\n=== BLOCK 3: codec decoder ===")
    audio_cfg = {
        "codec_args": full_params["multimodal"]["audio_tokenizer_args"],
        "audio_model_args": full_params["multimodal"]["audio_model_args"],
    }
    cfg = types.SimpleNamespace(
        audio_config=audio_cfg,
        text_config=types.SimpleNamespace(hidden_size=full_params["dim"]),
    )
    vllm_config = types.SimpleNamespace(model_config=types.SimpleNamespace(hf_config=cfg))
    up = tokmod.VoxtralTTSAudioTokenizer(vllm_config=vllm_config)

    st = SafeTensors(CKPT)
    sd = st.prefixed("audio_tokenizer.", torch.float32)
    # upstream's load_weights remaps this one in from a different checkpoint prefix; it belongs to
    # the ENCODE side (37-codebook input embedding) so it is irrelevant to the decode comparison,
    # but load_state_dict wants it present.
    sd["audio_token_embedding.embeddings.weight"] = st.get(
        "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight", torch.float32
    )
    missing, unexpected = up.load_state_dict(sd, strict=False)
    # the encoder is absent from the release, so its params are legitimately missing
    enc_missing = [m for m in missing if m.startswith(("input_proj.", "encoder_blocks."))]
    other_missing = [m for m in missing if m not in enc_missing]
    print(f"[b3] upstream load: {len(sd)} tensors | encoder missing {len(enc_missing)} (expected) "
          f"| other missing {other_missing} | unexpected {list(unexpected)}")
    assert not other_missing and not unexpected
    up = up.float().eval()

    ours = C.load_codec_state(CKPT)
    torch.manual_seed(7)
    codes = C.make_synthetic_codes(n_frames=20)

    with torch.no_grad():
        # 1) quantizer decode (semantic lookup ++ FSQ rescale)
        up_lat = up.quantizer.decode(codes, torch.float32)
        our_lat = C.quantizer_decode(codes, ours)
        report("b3 quantizer.decode -> latents [1,292,T]", our_lat, up_lat)

        # 2) full decode to waveform
        up_wav = up.decode(codes, dtype=torch.float32)
        our_wav = C.reference_decode(codes, ours)
        print(f"[b3] shapes upstream {tuple(up_wav.shape)} ours {tuple(our_wav.shape)}")
        report("b3 full decode -> waveform @ 24 kHz", our_wav, up_wav)

        # 3) per-stage bisect so a failure localises
        x = C.causal_conv1d(our_lat, ours["decoder_blocks.0.conv.weight"], 3, 1, "replicate")
        up_x = up.decoder_blocks[0](up_lat)
        report("b3   stage 0: CausalConv1d(292->1024,k3)", x, up_x)
        emb_up = up_x.transpose(1, 2).contiguous()
        emb_our = x.permute(0, 2, 1)
        for stage, (tf_i, conv_i) in enumerate(zip((1, 3, 5, 7), (2, 4, 6, None))):
            emb_up = up.decoder_blocks[tf_i](emb_up)
            emb_our = C.codec_transformer(emb_our, ours, tf_i, 2, C.decoder_window_sizes()[stage])
            report(f"b3   stage {tf_i}: Transformer(2L, window "
                   f"{C.decoder_window_sizes()[stage]})", emb_our, emb_up)
            if conv_i is not None:
                emb_up = up.decoder_blocks[conv_i](emb_up.transpose(1, 2)).transpose(1, 2)
                emb_our = C.causal_conv_transpose1d(
                    emb_our.permute(0, 2, 1), ours[f"decoder_blocks.{conv_i}.conv.weight"], 4, 2
                ).permute(0, 2, 1)
                report(f"b3   stage {conv_i}: CausalConvTranspose1d(k4,s2)", emb_our, emb_up)


def block1_input(tokmod, full_params):
    """The 37-codebook frame embedding — Block 1's INPUT side, and the feedback path of the
    generation loop. Upstream's MultiVocabEmbeddings offsets each codebook into one flat table
    and sums; a wrong offset would misread the table with no crash."""
    print("\n=== BLOCK 1 (input side): 37-codebook frame embedding ===")
    from models.experimental.voxtral_tts.reference import voxtral_backbone_ref as B
    from models.experimental.voxtral_tts.reference.voxtral_common_ref import N_AUDIO_SPECIAL, NUM_CODEBOOKS

    args = full_params["multimodal"]["audio_model_args"]
    up = tokmod.MultiVocabEmbeddings(audio_model_args=dict(args), embedding_dim=full_params["dim"])
    W = SafeTensors(CKPT).get("mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight", torch.float32)
    up.embeddings.weight.data.copy_(W)
    up = up.float().eval()

    same_off = bool((up.offsets.long() == B.codebook_offsets()).all())
    RESULTS.append(("b1-in codebook offsets (exact)", 1.0 if same_off else 0.0, 0.0, same_off))
    print(f"{'PASS' if same_off else 'FAIL'}  {'b1-in codebook offsets (exact)':52s} "
          f"{B.codebook_offsets()[:4].tolist()}...")

    torch.manual_seed(5)
    T = 6
    frames = torch.cat([torch.randint(0, 8192, (T, 1)),
                        torch.randint(0, 21, (T, NUM_CODEBOOKS - 1))], dim=1) + N_AUDIO_SPECIAL
    w = {"audio_embeddings": W}
    with torch.no_grad():
        up_emb = up(frames.t().unsqueeze(0)).sum(dim=1).squeeze(0)  # BxCBxL -> sum over codebooks
        report("b1-in embed_frames (batched) ", B.embed_frames(w, frames)[0], up_emb)
        single = torch.cat([B.embed_frame(w, frames[i])[0] for i in range(T)], dim=0)
        report("b1-in embed_frame (per-frame, decode loop)", single, up_emb)


def main():
    full_params, audio_args = load_params()
    gen = UL.load_generation()
    tokmod = UL.load_tokenizer_module()
    block1_input(tokmod, full_params)
    block2(gen, audio_args)
    block3(tokmod, full_params)

    print("\n=== SUMMARY ===")
    n_ok = sum(1 for _, _, _, ok in RESULTS if ok)
    for name, p, mx, ok in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:52s} {p:.8f}")
    print(f"  {n_ok}/{len(RESULTS)} checks pass")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
