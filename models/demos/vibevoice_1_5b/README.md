# VibeVoice-1.5B — end-to-end TTNN pipeline (text → speech)

`microsoft/VibeVoice-1.5B` brought up on Tenstorrent hardware. The model is a
text-to-speech stack: a **Qwen2** language-model backbone drives an autoregressive
loop that, at each speech step, runs a **DDPM diffusion head** to produce one
acoustic-VAE latent, decodes it to audio with the **acoustic tokenizer decoder**,
re-encodes it with the **semantic tokenizer**, and feeds the two **speech
connectors**' embeddings back into the LM.

## Call 1 — `tts` (text + voice sample → 24 kHz speech)

Single task head (all 19 graduated modules share one chained forward, so there is
one pipeline and one demo). Input is built by the HF `VibeVoiceProcessor` from a
script + a reference voice sample; output is the concatenated per-frame waveform.

The one shared forward lives in **`tt/pipeline.py::run_tts`** and is imported by
BOTH the demo and the e2e test (a green test guarantees a working demo).

### Chain (all 19 graduated stubs, on device)

```
voice sample ─> vibe_voice_acoustic_tokenizer_model(encode)      # tokenizer_encoder → block1_d→{convlayer,f_f_n}, s_conv1d→norm_conv1d
                 └─(+bias)*scaling─> speech_connector(acoustic) ─┐  (tokenizer_decoder→s_conv_transpose1d→norm_conv_transpose1d also fires)
input_ids ─> embed_tokens ──────────────────────────────────────├─> inputs_embeds
for N diffusion frames:                                          ┘
  inputs_embeds ─> qwen2_model (→ qwen2_decoder_layer) ─> hidden ─> constrained argmax
  hidden[-1] ─(cond)─> S × vibe_voice_diffusion_head (→ timestep_embedder, head_layer→feed_forward_network, final_layer) + DDPM ─> latent
  latent ─(/scaling-bias)─> tokenizer_decoder ─> audio chunk (3200 samp)
  audio chunk ─> vibe_voice_semantic_tokenizer_model(encode) ─> semantic
  speech_connector(latent) + speech_connector(semantic) ─> feedback embed
  inputs_embeds <- concat(inputs_embeds, feedback)        # self-fed, no reference injected
waveform = concat(audio chunks)
```

### Golden / validation

The reference is a faithful reimplementation of
`VibeVoiceForConditionalGenerationInference.generate()` (`tt/reference.py::hf_reference_tts`):
greedy decode under the speech-token constraint, `cfg_scale=1.0` (so the CFG
negative branch is a mathematical no-op and is omitted), the LM in full-recompute
causal mode, and BOTH sides capped to the same horizon (N diffusion frames, S DDPM
steps). The DDPM initial noise is shared verbatim, so the diffusion is deterministic
and TT-vs-HF differences are pure numeric error. Metric: `comp_pcc` on the final
waveform (Gate 3 ≥ 0.95), plus per-stage PCC.

## Run

```bash
# e2e gate (Gate 1 native / Gate 2 all-19-invoked / Gate 3 PCC>=0.95)
./python_env/bin/python -m pytest models/demos/vibevoice_1_5b/tests/e2e/test_e2e_tts.py -s

# demo (writes a .wav, prints token schedule + e2e PCC)
./python_env/bin/python -m models.demos.vibevoice_1_5b.demo.demo_tts \
    --text "Speaker 0: Hello there." --frames 6 --ddpm-steps 5 --out /tmp/vibevoice_tt.wav
```

`VIBEVOICE_E2E_N` / `VIBEVOICE_E2E_S` override the horizon / DDPM steps for the test.

## Graduated modules (19, all invoked)

`vibe_voice_acoustic_tokenizer_model`, `vibe_voice_semantic_tokenizer_model`,
`tokenizer_encoder`, `tokenizer_decoder`, `block1_d`, `convlayer`, `f_f_n`,
`s_conv1d`, `norm_conv1d`, `s_conv_transpose1d`, `norm_conv_transpose1d`,
`qwen2_model`, `qwen2_decoder_layer`, `speech_connector`,
`vibe_voice_diffusion_head`, `timestep_embedder`, `head_layer`,
`feed_forward_network`, `final_layer`.

Four parents were refactored to COMPOSE their graduated child stub instead of
inlining it, so every one of the 19 genuinely executes in the real forward path
(block1_d→convlayer+f_f_n, s_conv1d→norm_conv1d, head_layer→feed_forward_network,
qwen2_model→qwen2_decoder_layer). Each still passes its per-component PCC test.

## PCC numbers

Printed by the e2e test as `e2e PCC=…` (and per-stage PCC). See the final summary
in the bring-up report for the measured value.
