# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The HF golden chains for `mistralai/Voxtral-4B-TTS-2603` -- Source A's reference output.

THIS MODULE IS NOT THE PIPELINE. It is the reference side of the PCC comparison, and nothing in
`tt/pipeline.py`'s forward path imports it. It is the one place HF submodules may be called
(allowed usage 3: "HF calls inside a `_hf_reference_<task>()` helper").

WHY NOT `model.generate()`. Source A ships no generation loop for the TTS chain: the serving
loop lives in vLLM-Omni, which is not one of this run's two permitted sources. And
`generate()` on this checkpoint would drive the TIED TEXT head, which cannot emit audio codes at
all -- it is not the model's task. So the golden runs the reference's OWN submodules
(`hf.model.layers`, `hf.acoustic_transformer.*`, `hf.audio_tokenizer.*`) through the chain the
checkpoint's architecture dictates, and the TT pipeline runs the identical chain in ttnn. Call 2's
golden is one plain causal-LM forward, `hf(input_ids).logits`.

WHY THE NOISE IS AN ARGUMENT. `FlowMatchingAudioTransformer.decode_one_frame` draws
`x_0 = torch.randn(...)` INSIDE the module, so its output is a function of the RNG rather than of
its inputs -- any PCC against it would be measuring the random draw. The noise is drawn ONCE on
the host and handed to BOTH sides, which is what a flow-matching sampler takes as input anyway.
`reference_frame_matches_module()` PROVES this rewrite is the reference's own arithmetic and not a
paraphrase of it: it re-seeds, draws the same first tensor the module would draw, and asserts the
two frame outputs are integer-identical.
"""
from __future__ import annotations

import torch

from models.demos.voxtral_4b_tts_2603.tt import common

# Bumped whenever a chain in this module changes what it RETURNS. It goes into every golden cache
# key, so a cached reference can never outlive the code that produced it -- a stale golden silently
# compared against new arithmetic is the failure mode that once had a PCC gate testing another
# checkout for a whole run.
CHAIN_VERSION = 4


def _new_cache(hf_model):
    from transformers import DynamicCache

    return DynamicCache(config=hf_model.config)


# ----------------------------------------------------------------------------------------
# Call 1 -- text to speech
# ----------------------------------------------------------------------------------------


def acoustic_frame(hf_model, llm_hidden, x0, cfg_alpha):
    """ONE frame of 37 audio codes, the reference's arithmetic with `x_0` supplied.

    Line for line `FlowMatchingAudioTransformer.forward` + `decode_one_frame`, over the
    reference's own submodules, with the internal `torch.randn` replaced by `x0`.
    Returns ``(frame [B,37] int64, diagnostics)``.
    """
    at = hf_model.acoustic_transformer
    batch = llm_hidden.shape[0]

    semantic_logit = at.semantic_codebook_output(llm_hidden).float()
    # Keep the head's raw output before the -inf mask: the masked copy is what the argmax reads,
    # but a numeric comparison has to be made on finite values.
    semantic_raw = semantic_logit.clone()
    semantic_logit[:, at._empty_audio_token_id] = -float("inf")
    semantic_logit[:, (n_special_tokens() + at.model_args.semantic_codebook_size) :] = -float("inf")
    semantic_code = semantic_logit.argmax(dim=-1, keepdim=True)

    code = semantic_code.squeeze(1)
    should_decode = code != at._end_audio_token_id

    timesteps = at._timesteps.to(dtype=llm_hidden.dtype, device=llm_hidden.device)
    t_emb_table = at.time_embedding(timesteps.view(-1, 1)).to(llm_hidden.dtype)
    t_proj_table = at.time_projection(t_emb_table)
    dts = timesteps[1:] - timesteps[:-1]

    llm_batched = torch.cat([llm_hidden, torch.zeros_like(llm_hidden)], dim=0)
    llm_proj_batched = at.llm_projection(llm_batched)
    alpha = cfg_alpha.to(dtype=llm_hidden.dtype, device=llm_hidden.device).unsqueeze(1)

    sampled = at._noise_scale * x0.to(dtype=llm_hidden.dtype, device=llm_hidden.device)
    velocities = []
    for i in range(len(timesteps) - 1):
        t_proj = t_proj_table[i].unsqueeze(0).expand(batch, -1)
        v_all = at._predict_velocity(
            x_t=torch.cat([sampled, sampled], dim=0),
            llm_proj=llm_proj_batched,
            t_proj=torch.cat([t_proj, t_proj], dim=0),
        )
        v_t, uncond = v_all[:batch], v_all[batch:]
        v_t = alpha * v_t + (1 - alpha) * uncond
        velocities.append(v_t.clone())
        sampled = sampled + v_t * dts[i]

    sampled = torch.clamp(sampled, -1, 1)
    scaled = ((sampled + 1) / 2) * (at.acoustic_embeddings_levels - 1)
    out_codes = scaled.round().long()
    out_codes[~should_decode] = at._empty_audio_token_id
    acoustic = out_codes + n_special_tokens()

    frame = torch.concatenate([semantic_code, acoustic], dim=1)
    return frame, {
        "semantic_logits": semantic_logit,
        "semantic_logits_raw": semantic_raw,
        "velocities": velocities,
        "x_final": sampled,
    }


def acoustic_spread_under_matmul_noise(hf_model, llm_hidden, x0, cfg_alpha, rel_eps, draws=4, seed=0):
    """Per-element std of the reference's `x_final` when every Linear carries `rel_eps` noise.

    The flow sampler runs 7 Euler steps under classifier-free guidance at alpha=3, and some output
    elements are ill-conditioned: a perturbation the size of ONE device matmul's rounding moves them
    by several 0.1-wide code bins. Whether the reference's own rounding of such an element is a
    DECISION at all depends on the arithmetic precision available, so this measures it: the
    reference's own chain, every `nn.Linear` output perturbed by `rel_eps * rms(row) * N(0, 1)`,
    `draws` times. The noise is scaled by the output ROW's RMS, not by each element, because that is
    how a matmul's rounding behaves -- a K-long dot product's error is set by the magnitudes it sums,
    so a small output element carries the same absolute error as its large neighbours -- and it is
    the quantity `rel_eps` is measured as (RMS error / RMS output). Returns ``[B, 36]`` std in x
    units. Hooks are always removed.
    """
    at = hf_model.acoustic_transformer
    gen = torch.Generator().manual_seed(seed)

    def noisy(_module, _inputs, out):
        scale = out.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return out + rel_eps * scale * torch.randn(out.shape, generator=gen, dtype=out.dtype)

    hooks = [m.register_forward_hook(noisy) for m in at.modules() if isinstance(m, torch.nn.Linear)]
    try:
        outs = []
        for _ in range(int(draws)):
            with torch.no_grad():
                _, diag = acoustic_frame(hf_model, llm_hidden, x0, cfg_alpha)
            outs.append(diag["x_final"])
    finally:
        for h in hooks:
            h.remove()
    return torch.stack(outs).std(dim=0)


def n_special_tokens() -> int:
    """`len(AudioSpecialTokens)` -- read off the reference loader's own enum, never re-listed.

    This is the offset between an emitted audio token and the codec's code space: the acoustic
    transformer shifts its codes up by it, `MultiVocabEmbeddings` expects them shifted, and the
    codec's quantizer expects them unshifted.
    """
    return len(common._reference_loader_module().AudioSpecialTokens.all_special_tokens())


def reference_frame_matches_module(hf_model, llm_hidden, cfg_alpha, seed: int = 0):
    """Prove `acoustic_frame` IS the reference's arithmetic, not a paraphrase of it.

    Seed, call the UNMODIFIED module (which draws its own `x_0`); re-seed identically and draw
    the same first tensor the module would have drawn; feed that to `acoustic_frame`. The two
    frames must be integer-identical.
    """
    at = hf_model.acoustic_transformer
    batch = llm_hidden.shape[0]

    torch.manual_seed(seed)
    with torch.no_grad():
        codes_module = at(llm_hidden, cfg_alpha=cfg_alpha)

    torch.manual_seed(seed)
    x0 = torch.randn(batch, at.model_args.n_acoustic_codebook, dtype=llm_hidden.dtype, device=llm_hidden.device)
    with torch.no_grad():
        codes_helper, _ = acoustic_frame(hf_model, llm_hidden, x0, cfg_alpha)

    ok = bool(torch.equal(codes_module, codes_helper))
    return ok, codes_module, codes_helper, x0


def draw_noise(batch: int, n_acoustic: int, max_frames: int, seed: int = 0, dtype=torch.float32):
    """The sampler's noise input: one `[B, 36]` draw per frame, drawn ONCE on the host.

    Both the TT pipeline and the golden are handed this same tensor. `torch.randn` here is input
    preparation, the same category as `torch.arange` -- it is not host compute in the forward path.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(max_frames, batch, n_acoustic, generator=generator, dtype=dtype)


def reference_prefill(hf_model, input_ids, audio_mask=None, voice_embedding=None):
    """The reference's prefill over the (voiced) prompt: ``(prefill_hidden, kv_cache)``.

    Both reference arms start from this identical state, so a caller running both can compute it
    once and hand it to each via `prefill=` -- on CPU it is the single most expensive step.
    """
    batch = int(input_ids.shape[0])
    cache = _new_cache(hf_model)
    with torch.no_grad():
        if voice_embedding is None:
            out = hf_model.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        else:
            # The speaker's voice replaces the EMBEDDINGS of the prompt's `[AUDIO]` placeholders;
            # ids, positions and the causal mask stay the prompt's own.
            embeds = hf_model.model.embed_tokens(input_ids).clone()
            embeds[audio_mask] = voice_embedding.to(embeds.dtype).repeat(batch, 1)
            out = hf_model.model(inputs_embeds=embeds, past_key_values=cache, use_cache=True)
    return out.last_hidden_state, cache


def hf_reference_text_to_speech(
    hf_model,
    input_ids,
    x0,
    cfg_alpha,
    max_frames,
    fed_codes=None,
    fed_hidden=None,
    audio_mask=None,
    voice_embedding=None,
    prefill=None,
):
    """Source A's golden for Call 1: tokenized text -> a 24 kHz waveform.

    Runs the real chain -- text backbone (KV-cached), then per frame the acoustic sampler, the
    audio-token feedback embedding, and one more backbone step -- and finally the codec. Stops on
    the model's own `end_audio` token once every row has emitted it, bounded by `max_frames`.

    `fed_codes [B, 37, T]` puts the reference on a GIVEN trajectory: each step still computes the
    reference's own frame from its own hidden state (that is what the comparison reads), but the
    frame EMBEDDED and fed to the next backbone step is taken from `fed_codes` instead. Passing
    the TT pipeline's own emitted codes makes the reference answer the well-posed question "given
    exactly the context this pipeline produced, what does torch compute?" -- every stage is then
    compared on identical inputs. Nothing is ever spliced into the TT side: the TT chain runs
    free either way, so no wiring bug can hide (a TT stage fed the wrong thing still lands on a
    different hidden state than the reference computed from the same codes).

    `fed_hidden [B, dim, T]` does the same for the ACOUSTIC stage's input. The backbone still
    computes and returns its own hidden at every frame -- that is what `llm_hiddens` is compared
    against -- but the frame handed to `acoustic_frame` comes from `fed_hidden`, so the flow
    sampler and the semantic head are scored on the hidden they actually consumed rather than on
    a different one. Without it the acoustic comparison measures the BACKBONE's error amplified
    by the sampler: classifier-free guidance runs at alpha=3 (`v = 3*v_cond - 2*v_uncond`), which
    measured a 9.4x amplification on this checkpoint -- a per-frame hidden at PCC 0.999163
    (relative 0.041) came out as `x_final` at PCC 0.9259 (relative 0.385) even though the sampler
    itself is exact on identical inputs (velocity PCC 1.000000, semantic logits 1.000000, 99.3%
    of codes bit-identical). Amplification is the reference's own behaviour, not the port's.
    """
    at = hf_model.acoustic_transformer
    codec = hf_model.audio_tokenizer
    offset = common.n_audio_special_tokens(hf_model)
    stop_id = common.audio_stop_token_id(hf_model)
    batch, prompt_len = int(input_ids.shape[0]), int(input_ids.shape[1])

    if prefill is None:
        prefill_hidden, cache = reference_prefill(hf_model, input_ids, audio_mask, voice_embedding)
    else:
        import copy

        # A private copy: the decode loop appends to the cache, and the other arm needs it pristine.
        prefill_hidden, cache = prefill[0], copy.deepcopy(prefill[1])
    llm_hidden = prefill_hidden[:, -1]

    if fed_codes is not None:
        # The trajectory is given, so its length is too -- the stop rule already ran on the side
        # that produced it and forcing the reference past that point would compare different
        # contexts, which is the exact thing this mode exists to avoid.
        max_frames = int(fed_codes.shape[-1])

    frames, hiddens, diags = [], [llm_hidden.clone()], []
    finished = torch.zeros(batch, dtype=torch.bool)
    position = prompt_len
    stop_reason = f"max_frames={max_frames}"
    end_frame = [-1] * batch

    for step in range(max_frames):
        # The backbone's own hidden still drives `llm_hiddens`; only what the ACOUSTIC stage is
        # scored on changes, so each stage is measured on the input it actually consumed.
        acoustic_in = llm_hidden if fed_hidden is None else fed_hidden[..., step].to(llm_hidden.dtype)
        with torch.no_grad():
            frame, diag = acoustic_frame(hf_model, acoustic_in, x0[step], cfg_alpha)
        frames.append(frame)
        diags.append(diag)
        newly = (frame[:, 0] == stop_id) & ~finished
        for r in torch.nonzero(newly).reshape(-1).tolist():
            end_frame[r] = step
        finished |= frame[:, 0] == stop_id
        if fed_codes is None and bool(finished.all()):
            stop_reason = f"every row emitted end_audio (id {stop_id}) at frame {step}"
            break
        if step + 1 == max_frames:
            break
        if fed_codes is not None:
            frame = fed_codes[..., step].to(frame.dtype)
        with torch.no_grad():
            # `MultiVocabEmbeddings` broadcasts its per-codebook offsets as `[1, 37, 1]`, so it
            # wants `[B, 37, T]` -- a frame has to carry its length-1 time axis. Summing over the
            # codebook axis is the checkpoint's `input_embedding_concat_type: "sum"`.
            embeds = codec.audio_token_embedding(frame.unsqueeze(-1)).sum(dim=1)
            out = hf_model.model(
                inputs_embeds=embeds,
                past_key_values=cache,
                use_cache=True,
                position_ids=torch.full((batch, 1), position, dtype=torch.long),
            )
        llm_hidden = out.last_hidden_state[:, -1]
        hiddens.append(llm_hidden.clone())
        position += 1

    codes = torch.stack(frames, dim=-1)
    # The waveform is the codec's rendering of the codes the chain actually RAN on: its own when
    # free-running, the given trajectory when aligned. Rendering the reference's own codes in
    # aligned mode would compare two different code sequences and measure nothing about the codec.
    rendered = codes if fed_codes is None else fed_codes[..., : codes.shape[-1]].to(codes.dtype)
    with torch.no_grad():
        # Clamped at 0 exactly as the TT codec stage does: a row's `end_audio` frame (and what it
        # generated after it) is still in the batch block, and 1 - offset is not a codebook index.
        waveform = codec((rendered - offset).clamp(min=0))

    return {
        "input_ids": input_ids,
        "prefill_hidden": prefill_hidden,
        "llm_hiddens": hiddens,
        "codes": codes,
        "fed_codes": rendered,
        "waveform": waveform,
        "diagnostics": diags,
        "frames_decoded": codes.shape[-1],
        "stop_reason": stop_reason,
        "end_frame": end_frame,
        "sampling_rate": int(codec.sampling_rate),
    }


# ----------------------------------------------------------------------------------------
# Call 2 -- text continuation
# ----------------------------------------------------------------------------------------


def hf_reference_text_continuation(hf_model, input_ids):
    """Source A's golden for Call 2: the causal LM's teacher-forced next-token prediction.

    ONE plain `hf_model(input_ids)` forward -- not `generate()`. `logits[:, s]` is the reference's
    next-token distribution after tokens `0..s`, `next_tokens[:, s]` its greedy pick: the same
    quantities `pipeline.run_text_continuation` returns, over every position of every row.
    """
    with torch.no_grad():
        logits = hf_model(input_ids=input_ids).logits.float()
    return {"logits": logits, "next_tokens": logits.argmax(dim=-1)}
