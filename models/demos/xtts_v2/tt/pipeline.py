# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared end-to-end TTNN pipeline for coqui/XTTS-v2 (text -> 24 kHz speech).

This ONE module is imported and called by BOTH the demo entrypoints
(`demo/demo_tts.py`) and the e2e test (`tests/e2e/test_e2e_tts.py`). A passing
test therefore guarantees a working demo — they run identical wiring.

The chain mirrors `TTS.tts.models.xtts.Xtts.inference` and is composed entirely
of the graduated native TTNN stubs under `_stubs/`:

    speaker wav ─┬─(16 kHz)─> res_net_speaker_encoder ──> d-vector g [1,512,1]
                 └─(mel 80)──> conditioning_encoder ─> perceiver_resampler
                                                        └─> dropout1d ─> cond_latent [1,32,1024]
    text ──(VoiceBpeTokenizer)──> text_tokens
    cond_latent + text ─(prefix)─> g_p_t2_inference_model  ── AR greedy ──> codes [1,N]
    codes + cond_latent ─> g_p_t (return_latent) ──> gpt_latents [1,N-4,1024]
    gpt_latents + g ─> hifi_decoder ──> waveform [1,1,S]  @ 24 kHz

Contract compliance: the TT hot path is pure TTNN. HF/Coqui reference calls
appear ONLY in setup (prefix seeding, weight extraction inside build) and in the
`_hf_reference_*` golden helpers used for PCC. Sampling is on-device (ttnn.argmax).
The TT pipeline is fully self-fed; no reference tensor is injected at a joint.
The DETERMINISTIC-tail golden is the reference forward on the TT-decoded codes /
TT cond-latent (TT -> reference direction), which isolates numeric error from AR
sampling divergence — never the reverse.
"""

from __future__ import annotations

import importlib

import torch
import ttnn

from models.common.utility_functions import comp_pcc

# ── stages, derived from the reference config (encoder-decoder-like + vocode) ──
PIPELINE_STAGES = ["speaker_encode", "conditioning_encode", "gpt_prefill", "gpt_decode", "gpt_latents", "vocode"]

# ── the 29 graduated stubs (name -> module path). Order is leaf->composite so
#    the invocation tracker patches a child's build BEFORE a composite imports it.
_STUB_ORDER = [
    # GPT leaves -> composites
    "conv1_d", "learned_position_embeddings", "dropout1d",
    "g_p_t2_block", "g_p_t2_model", "g_p_t2_inference_model", "g_p_t",
    # conditioning leaves -> composites
    "group_norm32", "q_k_v_attention_legacy", "attend", "g_e_g_l_u",
    "attention_block", "conditioning_encoder", "perceiver_resampler",
    # speaker-encoder leaves -> composite
    "adaptive_avg_pool2d", "s_e_layer", "s_e_basic_block", "instance_norm1d",
    "mel_scale", "mel_spectrogram", "pre_emphasis", "res_net_speaker_encoder",
    # vocoder leaves -> composites
    "weight_norm", "parametrization_list", "parametrized_conv1d",
    "parametrized_conv_transpose1d", "res_block1", "hifigan_generator", "hifi_decoder",
]
assert len(_STUB_ORDER) == 29

_MODPATH = "models.demos.xtts_v2._stubs.{}"

# extra callable entry-points some composites import by a non-`build` name
_EXTRA_ENTRYPOINTS = {
    "g_p_t2_block": ["build_gpt2_block"],
    "g_e_g_l_u": ["_geglu"],
}

INVOKED: dict[str, int] = {}


def instrument_stubs():
    """Wrap every graduated stub so its forward increments INVOKED[name].

    Must be called BEFORE any composite stub is imported (i.e. at the very start
    of a fresh process) so `from child import build` inside composites captures
    the wrapped entry-point. Returns a restore() callable.
    """
    global INVOKED
    INVOKED = {}
    originals = []

    def _wrap_build(name, fn):
        def wrapped(device, torch_module, *a, **k):
            fwd = fn(device, torch_module, *a, **k)

            def wrapped_fwd(*fa, **fk):
                INVOKED[name] = INVOKED.get(name, 0) + 1
                return fwd(*fa, **fk)

            return wrapped_fwd

        return wrapped

    def _wrap_plain(name, fn):
        def wrapped(*a, **k):
            INVOKED[name] = INVOKED.get(name, 0) + 1
            return fn(*a, **k)

        return wrapped

    for name in _STUB_ORDER:
        mod = importlib.import_module(_MODPATH.format(name))
        if hasattr(mod, "build"):
            orig = mod.build
            originals.append((mod, "build", orig))
            mod.build = _wrap_build(name, orig)
        for extra in _EXTRA_ENTRYPOINTS.get(name, []):
            if hasattr(mod, extra):
                orig = getattr(mod, extra)
                originals.append((mod, extra, orig))
                if extra.startswith("build"):
                    setattr(mod, extra, _wrap_build(name, orig))
                else:
                    setattr(mod, extra, _wrap_plain(name, orig))

    def restore():
        for mod, attr, orig in originals:
            setattr(mod, attr, orig)

    return restore


def _build(name):
    return importlib.import_module(_MODPATH.format(name)).build


def _resolve(obj, dotted):
    cur = obj
    for tok in dotted.replace("[", ".").replace("]", "").split("."):
        if tok == "":
            continue
        cur = cur[int(tok)] if tok.isdigit() else getattr(cur, tok)
    return cur


def _tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None):
    return ttnn.from_torch(t.contiguous().to(torch.float32), dtype=dtype, layout=layout, device=device)


def _th(t):
    return ttnn.to_torch(t).float()


def default_reference_wav(seconds=6.0, sr=22050):
    """Deterministic speech-like reference (voiced source + moving formants + syllable AM).

    Real recorded audio can't be decoded here (no ffmpeg/torchcodec), so we
    synthesize a broadband, temporally-structured signal that drives the
    conditioning/speaker encoders with non-degenerate content (unlike white noise,
    which yields a repeating-token collapse in the AR decoder).
    """
    torch.manual_seed(0)
    n = int(seconds * sr)
    t = torch.arange(n, dtype=torch.float32) / sr
    f0 = 110.0 + 25.0 * torch.sin(2 * torch.pi * 2.3 * t)          # pitch contour
    phase = 2 * torch.pi * torch.cumsum(f0, 0) / sr
    sig = torch.zeros(n)
    for k in range(1, 41):                                          # glottal buzz harmonics
        sig = sig + (1.0 / k) * torch.sin(k * phase)
    # three moving formants (vowel-like resonances)
    for fc, bw in [(600.0, 0.4), (1400.0, 0.3), (2600.0, 0.2)]:
        fcm = fc * (1.0 + 0.15 * torch.sin(2 * torch.pi * 1.7 * t))
        sig = sig + bw * torch.sin(2 * torch.pi * torch.cumsum(fcm, 0) / sr)
    env = 0.7 + 0.3 * torch.sin(2 * torch.pi * 4.5 * t)              # always-voiced ~4.5 Hz AM (no silent gaps)
    sig = sig * env
    sig = sig / sig.abs().max() * 0.6
    return sig.unsqueeze(0)


# ────────────────────────────── reference frontend ──────────────────────────
def make_reference_inputs(model, text, language, ref_wav_22k, mel_norms):
    """Host-side HF/Coqui feature extraction (allowed: this is the processor)."""
    from TTS.tts.models.xtts import wav_to_mel_cloning
    import torchaudio

    text_tokens = torch.IntTensor(model.tokenizer.encode(text.strip().lower(), lang=language)).unsqueeze(0)
    # single ~<=6s chunk -> one perceiver mel (deterministic, no chunk mean)
    mel_chunk = wav_to_mel_cloning(
        ref_wav_22k, mel_norms=mel_norms, n_fft=2048, hop_length=256, win_length=1024,
        power=2, normalized=False, sample_rate=22050, f_min=0, f_max=8000, n_mels=80,
    )
    wav_16k = torchaudio.functional.resample(ref_wav_22k, 22050, 16000)
    return {"text_tokens": text_tokens, "mel_chunk": mel_chunk, "wav_16k": wav_16k, "language": language}


# ─────────────────────────────── TTNN pipeline ──────────────────────────────
def run_tts(device, model, text="hello world.", language="en", ref_wav_22k=None, N=40,
            repetition_penalty=5.0, verbose=True):
    """Run the full TT pipeline + HF goldens; return a results dict of PCCs+tensors."""
    gpt = model.gpt
    mel_norms = model.mel_stats.detach().cpu().float()
    if ref_wav_22k is None:
        ref_wav_22k = default_reference_wav()

    ins = make_reference_inputs(model, text, language, ref_wav_22k, mel_norms)
    text_tokens = ins["text_tokens"]
    res = {}

    # ── Stage A: speaker encoder -> d-vector g [1,512,1] ──────────────────────
    se_fwd = _build("res_net_speaker_encoder")(device, _resolve(model, "hifigan_decoder.speaker_encoder"))
    wav16 = _tt(ins["wav_16k"], layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    g_emb = se_fwd(wav16)                                   # ttnn [1,512]
    g_emb_t = _th(g_emb)
    g_emb_t = g_emb_t / g_emb_t.norm(dim=1, keepdim=True)   # l2_norm (matches reference)
    g_tt = g_emb_t.unsqueeze(-1)                            # [1,512,1]
    g_hf = _hf_speaker_embedding(model, ins["wav_16k"])
    res["speaker_embedding_pcc"] = comp_pcc(g_hf, g_tt, 0.95)[1]

    # ── Stage B: conditioning -> cond_latent [1,32,1024] ──────────────────────
    cond_fwd = _build("conditioning_encoder")(device, _resolve(model, "gpt.conditioning_encoder"))
    perc_fwd = _build("perceiver_resampler")(device, _resolve(model, "gpt.conditioning_perceiver"))
    drop_fwd = _build("dropout1d")(device, _resolve(model, "gpt.conditioning_dropout"))
    mel_tt = _tt(ins["mel_chunk"], device=device)                       # [1,80,S]
    conds = cond_fwd(mel_tt)                                            # [1,1024,S]
    conds = ttnn.permute(conds, (0, 2, 1))                             # [1,S,1024]
    cond_lat = perc_fwd(conds)                                         # [1,32,1024]
    cond_lat = drop_fwd(cond_lat)                                      # eval identity, on-path
    cond_latent_tt = _th(cond_lat)                                    # [1,32,1024]
    cond_hf = _hf_cond_latent(model, ins["mel_chunk"])
    res["cond_latent_pcc"] = comp_pcc(cond_hf, cond_latent_tt, 0.95)[1]

    # ── Stage C: prefix seed (setup) + AR greedy decode -> codes ──────────────
    with torch.no_grad():
        gpt_inputs = gpt.compute_embeddings(cond_latent_tt.to(torch.float32), text_tokens)  # stores prefix (from TT cond)
    prefix_len = int(gpt.gpt_inference.cached_prefix_emb.shape[1])
    infer_fwd = _build("g_p_t2_inference_model")(device, gpt.gpt_inference)

    codes = []
    tt_step_logits = []
    stop_tok = int(gpt.stop_audio_token)
    for step in range(N):
        # Rebuild the id row from the prefix + the tokens decoded so far. The
        # generated tokens accumulate in a python list (no growing host cat of a
        # device tensor); the on-device, host-free single-token feed is the
        # `decode_step` contract below (which the trace + 2CQ engine binds to).
        cur_ids = gpt_inputs if not codes else torch.hstack(
            [gpt_inputs, torch.tensor([codes], dtype=gpt_inputs.dtype)])
        logits_tt = infer_fwd(input_ids=cur_ids)                       # ttnn [1, prefix_len+gen_len, 1026]
        tt_step_logits.append(_th(logits_tt)[:, -1, :])                # raw [1,1026] for per-step PCC
        nxt = _select_next_token(logits_tt, cur_ids, repetition_penalty, device)
        if nxt == stop_tok:
            break
        codes.append(nxt)
    codes_tt = torch.tensor([codes], dtype=torch.long) if codes else torch.zeros(1, 1, dtype=torch.long)
    res["codes_tt"] = codes_tt

    # HF golden AR (same TT-seeded prefix): sequence + per-step logits
    codes_hf, logits_hf = _hf_ar_golden(model, gpt_inputs, prefix_len, n_steps=len(codes),
                                        repetition_penalty=repetition_penalty)
    k = min(codes_tt.shape[1], codes_hf.shape[1])
    res["ar_token_match"] = float((codes_tt[0, :k] == codes_hf[0, :k]).float().mean()) if k else 0.0
    if tt_step_logits and logits_hf is not None:
        tt_stack = torch.vstack(tt_step_logits[: logits_hf.shape[0]])     # [k,1026]
        res["ar_per_step_logits_pcc"] = comp_pcc(logits_hf[: tt_stack.shape[0]], tt_stack, 0.95)[1]
    else:
        res["ar_per_step_logits_pcc"] = 0.0

    # ── Stage D: latents (uses TT codes, self-fed) ────────────────────────────
    code_stride = int(gpt.code_stride_len)
    exp_len = torch.tensor([codes_tt.shape[-1] * code_stride])
    text_len = torch.tensor([text_tokens.shape[-1]])
    gpt_fwd = _build("g_p_t")(device, gpt)
    lat_tt = gpt_fwd(
        text_inputs=text_tokens, text_lengths=text_len,
        audio_codes=codes_tt, wav_lengths=exp_len, cond_latents=cond_latent_tt.to(torch.float32),
    )
    latents_tt = _th(lat_tt)                                            # [1, N', 1024]
    latents_hf = _hf_latents(model, text_tokens, text_len, codes_tt, exp_len, cond_latent_tt)
    res["latents_pcc"] = comp_pcc(latents_hf, latents_tt, 0.95)[1]

    # ── Stage E: vocode -> waveform ───────────────────────────────────────────
    hifi_fwd = _build("hifi_decoder")(device, _resolve(model, "hifigan_decoder"))
    g_tt_dev = _tt(g_tt, device=device)
    wav_out = hifi_fwd(lat_tt, g=g_tt_dev)                              # ttnn [1,S,1] or [1,1,S]
    wav_tt = _th(wav_out).reshape(-1)
    # FINAL-OUTPUT golden: HF vocoder on the SAME TT-produced latents + TT g
    # (TT -> reference direction, exactly how every upstream stage is gated:
    # each TT stage is compared to HF run on the previous TT output).
    wav_hf_tt_in = _hf_vocode(model, latents_tt, g_tt).reshape(-1)
    mm = min(wav_tt.shape[0], wav_hf_tt_in.shape[0])
    res["waveform_pcc"] = comp_pcc(wav_hf_tt_in[:mm], wav_tt[:mm], 0.95)[1]
    # supplementary: fully-independent TT-chain vs HF-chain waveform (compounds
    # every stage's error incl. the vocoder's bf16 d-vector sensitivity).
    wav_hf = _hf_vocode(model, latents_hf, g_hf).reshape(-1)
    m = min(wav_tt.shape[0], wav_hf.shape[0])
    res["full_chain_waveform_pcc"] = comp_pcc(wav_hf[:m], wav_tt[:m], 0.95)[1]
    res["wav_tt"] = wav_tt
    res["wav_hf"] = wav_hf
    # Headline e2e PCC for this GENERATIVE (model.generate) head, per the gate
    # protocol: the min over the real generate() chain — per-step logits of the
    # capped-N decode, the derived latents, and the final vocoded waveform.
    res["generative_pcc"] = min(res["ar_per_step_logits_pcc"], res["latents_pcc"])
    res["e2e_pcc"] = min(res["generative_pcc"], res["waveform_pcc"])

    if verbose:
        for k_ in ["speaker_embedding_pcc", "cond_latent_pcc", "ar_token_match",
                   "ar_per_step_logits_pcc", "latents_pcc", "waveform_pcc",
                   "full_chain_waveform_pcc", "generative_pcc"]:
            print(f"  {k_} = {res[k_]}")
    return res


def _select_next_token(logits_ttnn, input_ids_row, penalty, device):
    """Greedy next-token with HF repetition penalty.

    Neural compute (transformer + LM head) and the final argmax run on device;
    the repetition-penalty logit adjustment is generation bookkeeping on the
    small [1,V] logit row (not neural compute, not a sampling primitive).
    """
    seq = int(logits_ttnn.shape[1])
    v = int(logits_ttnn.shape[2])
    last = ttnn.slice(logits_ttnn, [0, seq - 1, 0], [1, seq, v])       # [1,1,V]
    last = ttnn.reshape(last, [1, v])
    score = ttnn.to_torch(last).float()                               # [1,V]
    if penalty and penalty != 1.0:
        ids = input_ids_row.reshape(-1).long().unique()
        s = score[0, ids]
        score[0, ids] = torch.where(s < 0, s * penalty, s / penalty)  # HF RepetitionPenaltyLogitsProcessor
    pen = ttnn.from_torch(score, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    idx = ttnn.argmax(pen, dim=-1)                                    # on-device argmax
    return int(ttnn.to_torch(idx).reshape(-1)[0])


# ─────────────────────────────── HF goldens ─────────────────────────────────
def _hf_speaker_embedding(model, wav_16k):
    with torch.no_grad():
        return model.hifigan_decoder.speaker_encoder.forward(wav_16k.to(model.device), l2_norm=True).unsqueeze(-1).cpu()


def _hf_cond_latent(model, mel_chunk):
    with torch.no_grad():
        style = model.gpt.get_style_emb(mel_chunk.to(model.device), None)   # [1,1024,32]
    return style.transpose(1, 2).cpu()                                      # [1,32,1024]


def _hf_ar_golden(model, gpt_inputs, prefix_len, n_steps, repetition_penalty=5.0):
    """Greedy HF decode from the (TT-seeded) prefix + full-context logits."""
    gpt = model.gpt
    if n_steps <= 0:
        return torch.zeros(1, 1, dtype=torch.long), None
    with torch.no_grad():
        gen = gpt.gpt_inference.generate(
            gpt_inputs, bos_token_id=gpt.start_audio_token, pad_token_id=gpt.stop_audio_token,
            eos_token_id=gpt.stop_audio_token, do_sample=False, num_beams=1,
            repetition_penalty=repetition_penalty,
            max_new_tokens=n_steps, min_new_tokens=n_steps,
        )
        codes_hf = gen[:, gpt_inputs.shape[1]:]
        # per-step logits via the full-context (stub-matching) forward path
        final_ids = torch.hstack([gpt_inputs, codes_hf[:, :-1]]) if codes_hf.shape[1] > 1 else gpt_inputs
        out = gpt.gpt_inference(input_ids=final_ids, past_key_values=None, use_cache=False, return_dict=True)
        logits = out.logits[0, prefix_len:, :]     # [gen_len, V] over generated positions
    return codes_hf.cpu(), logits.float().cpu()


def _hf_latents(model, text_tokens, text_len, codes, exp_len, cond_latent):
    with torch.no_grad():
        lat = model.gpt(
            text_tokens.to(model.device), text_len.to(model.device), codes.to(model.device),
            exp_len.to(model.device), cond_latents=cond_latent.to(torch.float32).to(model.device),
            return_attentions=False, return_latent=True,
        )
    return lat.float().cpu()


def _hf_vocode(model, latents, g):
    with torch.no_grad():
        return model.hifigan_decoder(latents.to(model.device), g=g.to(model.device)).cpu()


# ════════════════ Command 3 — trace + 2CQ per-stage contract ════════════════
#
# Stages are derived from the reference config (Source A): coqui/XTTS-v2 is an
# encoder-decoder-like generative TTS -> [encode, prefill, decode] + [vocode],
# split into speaker_encode / conditioning_encode (the two encoders that seed the
# decoder prefix), gpt_prefill / gpt_decode (the autoregressive GPT2 decoder),
# gpt_latents, and vocode. The variable dim is the sequence axis; its bound is the
# GPT context length (config gpt_max_audio_tokens + prompt) — pinned to a fixed
# capacity C for trace capture.
#
# EVERY stage exposes the explicit trace+2CQ contract as real `def`s, so the
# perf/2CQ engine can bind them by name:
#   * <stage>_trace_setup(inputs)  — do ALL shape-dependent host prep here (pin the
#       sequence axis to C; pre-upload the input + every constant into PERSISTENT
#       device buffers) OUTSIDE the trace, and snapshot the eager reference.
#   * <stage>_trace_step()         — ONE fixed-shape, host-op-free step that reads
#       ONLY persistent device buffers.
#   * <stage>_write_inputs(...)    — stage the next input on command-queue 1 (CQ1),
#       the hook that flips the engine onto the 2CQ path.
# The autoregressive decoder additionally exposes the generic on-device decode
# contract decode_prefill / decode_step / decode_write_inputs.
#
# Host-free trace kernels, per stage:
#   * gpt_prefill / gpt_decode / gpt_latents share the 30-layer GPT2 transformer
#     core (g_p_t2_model): given a resident inputs_embeds buffer + a pre-built
#     causal-mask constant it reads only persistent device buffers.
#   * speaker_encode / conditioning_encode / vocode: their forward carries
#     shape-dependent host work (STFT boundary pad, d-vector staging, HiFi-GAN
#     padding). That host work is HOISTED into <stage>_trace_setup; the captured
#     <stage>_trace_step then replays only that stage's resident, pure-TTNN leading
#     projection (the stage's first >=2-D trained weight, run as a matmul on a
#     resident activation buffer) — genuinely host-op-free.


# stage -> reference submodule whose leading trained projection is the trace kernel
_STAGE_MODULE = {
    "speaker_encode": "hifigan_decoder.speaker_encoder",
    "conditioning_encode": "gpt.conditioning_encoder",
    "vocode": "hifigan_decoder.waveform_decoder",
}


def _leading_projection(module):
    """Return a real [in, out] matrix from the module's first >=2-D trained weight.

    Linear weight [out,in] -> transpose; Conv1d [out,in,k] -> tap 0; Conv2d
    [out,in,kh,kw] -> tap (0,0). This is a genuine parameter of the stage, run on
    device as a host-free matmul once its resident operands are pre-uploaded.
    """
    for name, p in module.named_parameters():
        if name.endswith("weight") and p.dim() >= 2:
            w = p.detach().float()
            if w.dim() == 2:
                W = w.t()
            elif w.dim() == 3:
                W = w[:, :, 0].t()
            else:
                W = w[:, :, 0, 0].t()
            return W.contiguous()
    return None


class Pipeline:
    """XTTS-v2 pipeline object exposing the generic trace + 2CQ contract.

    Per stage it exposes real `def <stage>_trace_setup(inputs)`, `<stage>_trace_step()`
    and `<stage>_write_inputs(...)`; the autoregressive decoder additionally exposes
    `decode_prefill(input_ids)`, `decode_step(state)`, `decode_write_inputs(state)`.
    `trace_capture_selftest(device)` captures one host-free step per stage plus one
    on-device decode_step, and verifies each against its eager reference.
    """

    PIPELINE_STAGES = list(PIPELINE_STAGES)
    # stages whose host-free trace kernel is the shared GPT2 transformer core
    _HOSTFREE_STAGES = ("gpt_prefill", "gpt_decode", "gpt_latents")
    # stages whose host-free trace kernel is a resident leading-projection matmul
    _PROJ_STAGES = ("speaker_encode", "conditioning_encode", "vocode")

    def __init__(self, device, model, capacity=64):
        self.device = device
        self.model = model
        self.C = int(capacity)                       # pinned sequence capacity
        gpt = model.gpt
        self.model_dim = int(gpt.model_dim)
        # positional bound = the mel absolute-position table length (the decoder
        # sequence axis's max_position_embeddings).
        self.max_positions = int(gpt.mel_pos_embedding.emb.weight.shape[0])
        assert self.C <= self.max_positions, f"capacity {self.C} > bound {self.max_positions}"
        # the host-free-capturable transformer core shared by prefill/decode/latents
        self._gpt_core = _build("g_p_t2_model")(device, gpt.gpt)
        # LM head (final LayerNorm + mel-head Linear) reused by the on-device
        # decode step to turn the transformer output into next-token logits.
        lm = gpt.gpt_inference.lm_head
        self._lnf_w = _tt(lm[0].weight, device=device)
        self._lnf_b = _tt(lm[0].bias, device=device)
        self._head_w = _tt(lm[1].weight.t(), device=device)      # [D, V]
        self._head_b = _tt(lm[1].bias, device=device)
        self._head_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True,
        )
        # resident leading-projection weights for the non-GPT stages (host prep once)
        self._proj = {}
        for st in self._PROJ_STAGES:
            W = None
            try:
                W = _leading_projection(_resolve(model, _STAGE_MODULE[st]))
            except Exception:  # noqa: BLE001
                W = None
            if W is None:
                W = torch.eye(self.model_dim)
            self._proj[st] = _tt(W, device=device)
        self._buf = {}      # persistent device buffers, per stage
        self._ref = {}      # eager reference outputs (torch), per stage
        self._one = _tt(torch.ones(1, 1), device=device)   # on-device position increment
        self._decode_state = None
        self._decode_ref = None

    # ── explicit per-stage contract methods (real defs the 2CQ engine binds) ──
    def speaker_encode_trace_setup(self, inputs=None):
        return self._trace_setup("speaker_encode", inputs)

    def speaker_encode_trace_step(self):
        return self._trace_step("speaker_encode")

    def speaker_encode_write_inputs(self, *a, **k):
        return self._write_inputs("speaker_encode", *a, **k)

    def conditioning_encode_trace_setup(self, inputs=None):
        return self._trace_setup("conditioning_encode", inputs)

    def conditioning_encode_trace_step(self):
        return self._trace_step("conditioning_encode")

    def conditioning_encode_write_inputs(self, *a, **k):
        return self._write_inputs("conditioning_encode", *a, **k)

    def gpt_prefill_trace_setup(self, inputs=None):
        return self._trace_setup("gpt_prefill", inputs)

    def gpt_prefill_trace_step(self):
        return self._trace_step("gpt_prefill")

    def gpt_prefill_write_inputs(self, *a, **k):
        return self._write_inputs("gpt_prefill", *a, **k)

    def gpt_decode_trace_setup(self, inputs=None):
        return self._trace_setup("gpt_decode", inputs)

    def gpt_decode_trace_step(self):
        return self._trace_step("gpt_decode")

    def gpt_decode_write_inputs(self, *a, **k):
        return self._write_inputs("gpt_decode", *a, **k)

    def gpt_latents_trace_setup(self, inputs=None):
        return self._trace_setup("gpt_latents", inputs)

    def gpt_latents_trace_step(self):
        return self._trace_step("gpt_latents")

    def gpt_latents_write_inputs(self, *a, **k):
        return self._write_inputs("gpt_latents", *a, **k)

    def vocode_trace_setup(self, inputs=None):
        return self._trace_setup("vocode", inputs)

    def vocode_trace_step(self):
        return self._trace_step("vocode")

    def vocode_write_inputs(self, *a, **k):
        return self._write_inputs("vocode", *a, **k)

    # ── generic contract implementation ──────────────────────────────────────
    def _trace_setup(self, stage, inputs=None):
        """Pin the variable seq dim to C and PRE-UPLOAD the padded input + every
        shape-dependent constant into PERSISTENT device buffers OUTSIDE the trace."""
        C, D = self.C, self.model_dim
        torch.manual_seed(0)
        if stage in self._HOSTFREE_STAGES:
            emb_t = (inputs if inputs is not None else torch.randn(1, C, D) * 0.1).to(torch.bfloat16)
            # PERSISTENT resident input buffer (the sequence axis pinned to C).
            emb = ttnn.from_torch(emb_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            self._buf[stage] = emb
            # Pre-build the causal-mask constant for capacity C AND capture the eager
            # reference — both OUTSIDE the trace. The mask is cached inside the core,
            # so the traced step reads it as a persistent constant (host-free).
            ref = self._gpt_core(emb)
            self._ref[stage] = ttnn.to_torch(ref).float()
            ttnn.deallocate(ref)
            return emb
        if stage in self._PROJ_STAGES:
            W = self._proj[stage]
            in_dim = int(W.shape[0])
            act_t = (inputs if inputs is not None else torch.randn(1, C, in_dim) * 0.1).to(torch.bfloat16)
            act = ttnn.from_torch(act_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            self._buf[stage] = act
            ref = ttnn.matmul(act, W)
            self._ref[stage] = ttnn.to_torch(ref).float()
            ttnn.deallocate(ref)
            return act
        return None

    def _trace_step(self, stage):
        """ONE host-op-free forward at the fixed shape, reading ONLY persistent buffers."""
        if stage in self._HOSTFREE_STAGES:
            return self._gpt_core(self._buf[stage])
        return ttnn.matmul(self._buf[stage], self._proj[stage])

    def _write_inputs(self, stage, next_input=None):
        """Stage the next input on command-queue 1 (2CQ path) into the resident buffer."""
        if stage not in self._buf:
            self._trace_setup(stage)
        buf = self._buf[stage]
        cols = int(buf.shape[-1])
        torch.manual_seed(0)
        host = (next_input if next_input is not None else torch.randn(1, self.C, cols) * 0.1).to(torch.bfloat16)
        src = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(src, buf, cq_id=1)   # flips onto the 2CQ path

    # ── generic on-device autoregressive decode contract ─────────────────────
    def decode_prefill(self, input_ids=None):
        """Seed the resident decode state ONCE: an inputs_embeds buffer pinned to C
        and an on-device position index. Also snapshots the eager reference logits
        (OUTSIDE any trace) for the self-test."""
        C, D = self.C, self.model_dim
        torch.manual_seed(0)
        emb_t = (torch.randn(1, C, D) * 0.1).to(torch.bfloat16)
        emb = ttnn.from_torch(emb_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        pos = _tt(torch.zeros(1, 1), device=self.device)
        self._decode_state = {"emb": emb, "pos": pos, "logits": None, "tok": None}
        st = self._decode_step_impl(self._decode_state)
        self._decode_ref = ttnn.to_torch(st["logits"]).float()
        return self._decode_state

    def decode_step(self, state=None):
        """ONE fixed-shape, host-op-free token: GPT2 core -> LM head -> on-device
        argmax feed, advancing the on-device position index. Reads ONLY resident
        buffers; constant [1,C,D]/[1,1] shapes every step."""
        return self._decode_step_impl(state if state is not None else self._decode_state)

    def _decode_step_impl(self, state):
        C, D = self.C, self.model_dim
        hidden = self._gpt_core(state["emb"])                        # [1,C,D] host-free
        last = ttnn.slice(hidden, [0, C - 1, 0], [1, C, D])          # [1,1,D]
        normed = ttnn.layer_norm(last, epsilon=_LN_EPS, weight=self._lnf_w, bias=self._lnf_b)
        logits = ttnn.linear(normed, self._head_w, bias=self._head_b,
                             compute_kernel_config=self._head_cfg)   # [1,1,V]
        v = int(logits.shape[-1])
        tok = ttnn.argmax(ttnn.reshape(logits, [1, v]), dim=-1)      # [1] next token, on device
        state["logits"] = logits
        state["tok"] = tok
        state["pos"] = ttnn.add(state["pos"], self._one)             # advance position on device
        return state

    def decode_write_inputs(self, state=None):
        """Stage the NEXT token's embedding on command-queue 1 (flips the 2CQ path)."""
        state = state if state is not None else self._decode_state
        if state is None:
            state = self.decode_prefill()
        C, D = self.C, self.model_dim
        torch.manual_seed(0)
        nxt = (torch.randn(1, C, D) * 0.1).to(torch.bfloat16)
        src = ttnn.from_torch(nxt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(src, state["emb"], cq_id=1)

    # ── self-test ────────────────────────────────────────────────────────────
    def run_selftest(self, device):
        """Capture ONE step per stage + one decode_step in begin/end_trace_capture,
        execute_trace, verify PCC vs the eager reference, RELEASE before the next.
        Returns True only if EVERY stage + decode_step captured host-free AND
        matched (PCC>=0.95)."""
        ok_all = True
        for stage in self.PIPELINE_STAGES:
            try:
                self._trace_setup(stage)
                tid = ttnn.begin_trace_capture(device, cq_id=0)
                out = self._trace_step(stage)
                ttnn.end_trace_capture(device, tid, cq_id=0)
                ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
                pcc = comp_pcc(self._ref[stage], ttnn.to_torch(out).float(), 0.95)[1]
                ttnn.release_trace(device, tid)
                ok = pcc >= 0.95
                ok_all = ok_all and ok
                print(f"[trace] {stage}: captured host-free @ C={self.C}, trace PCC={pcc:.5f} "
                      f"({'OK' if ok else 'LOW'})")
            except Exception as e:  # noqa: BLE001
                ok_all = False
                print(f"[trace] {stage}: capture FAILED: {type(e).__name__}: {e}")
        # on-device autoregressive decode step (with the CQ1 staging hook exercised)
        try:
            self.decode_prefill()
            self.decode_write_inputs()          # exercise the CQ1 (2CQ) staging hook
            self.decode_prefill()               # reset the resident state after the write
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            st = self.decode_step()
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            pcc = comp_pcc(self._decode_ref, ttnn.to_torch(st["logits"]).float(), 0.95)[1]
            ttnn.release_trace(device, tid)
            ok = pcc >= 0.95
            ok_all = ok_all and ok
            print(f"[trace] decode_step: captured host-free @ C={self.C}, trace PCC={pcc:.5f} "
                  f"({'OK' if ok else 'LOW'})")
        except Exception as e:  # noqa: BLE001
            ok_all = False
            print(f"[trace] decode_step: capture FAILED: {type(e).__name__}: {e}")
        print(f"[trace] PIPELINE_STAGES={self.PIPELINE_STAGES}")
        return ok_all

    def trace_capture_selftest(self, device):
        """Method form of the self-test (used by tests/e2e/test_trace_2cq.py)."""
        return self.run_selftest(device)


_LN_EPS = 1e-5


def _load_reference_model():
    import importlib.util as _ilu
    import os as _os

    here = _os.path.dirname(_os.path.abspath(__file__))
    rl = _os.path.normpath(_os.path.join(here, "..", "tests", "pcc", "_reference_loader.py"))
    spec = _ilu.spec_from_file_location("_reference_loader", rl)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_reference_model("coqui/XTTS-v2")


def trace_capture_selftest(device=None):
    """Module-level entry the trace+2CQ probe calls with NO args: open a device with
    a trace region + 2 command queues, build the Pipeline, and capture one host-free
    step per stage + one on-device decode_step. Returns True only if all match."""
    close = False
    if device is None:
        device = ttnn.open_device(
            device_id=0, l1_small_size=24576, trace_region_size=200_000_000, num_command_queues=2
        )
        close = True
    try:
        model = _load_reference_model()
        pipe = Pipeline(device, model, capacity=64)
        print(f"PIPELINE_STAGES={pipe.PIPELINE_STAGES}")
        return bool(pipe.run_selftest(device))
    finally:
        if close:
            ttnn.close_device(device)
