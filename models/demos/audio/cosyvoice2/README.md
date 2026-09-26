# CosyVoice2 on Tenstorrent (TTNN)

TTNN bring-up of [CosyVoice2](https://github.com/QwenAudio/CosyVoice) (FunAudioLLM's
CosyVoice repo now redirects here), Alibaba's second-generation streaming TTS model,
for [tenstorrent/tt-metal#54104](https://github.com/tenstorrent/tt-metal/issues/54104).

## Status

Bring-up in progress. Phase 1 (HiFT vocoder iSTFT) is the only piece implemented so far.

| stage | status |
|---|---|
| HiFT vocoder: inverse STFT (`tt/hifigan/istft.py`) | implemented, PCC-validated on device against synthetic wide-dynamic-range inputs |
| HiFT vocoder: upsample/resblock/NSF stack | not started |
| LLM backbone (Qwen2-0.5B) | not started |
| Flow-matching decoder | not started |
| Full pipeline / streaming | not started |

## Architecture

Three components, as in CosyVoice1, but not architecturally identical to it:

1. **LLM backbone**: Qwen2-0.5B, text -> semantic tokens (`speech_token_size: 6561`).
2. **Flow-matching decoder**: `CausalMaskedDiffWithXvec` / `CausalConditionalCFM` --
   chunk-aware and causal (`static_chunk_size`, `pre_lookahead_len: 3`), token-to-mel.
3. **HiFT vocoder**: mel-to-waveform. Plain (non-causal) `HiFTGenerator`, 3-stage
   upsampling (`upsample_rates: [8, 5, 3]`, one source/NSF branch per stage), 24 kHz
   output, ending in an inverse STFT.

Confirmed against the actual upstream training config
(`examples/libritts/cosyvoice2/conf/cosyvoice2.yaml` in
[QwenAudio/CosyVoice](https://github.com/QwenAudio/CosyVoice)) rather than assumed:
the iSTFT contract itself (`n_fft: 16, hop_len: 4`) is identical to CosyVoice1's, so
[the iSTFT-as-matmul identity](#why-the-vocoder-head-is-not-an-fft-problem) carries
over unchanged even though the upsample topology around it does not. Streaming
causality lives in the flow decoder, not the vocoder -- the vocoder receives
already-chunked mel frames and processes them with an ordinary non-causal stack.

## Why the vocoder head is not an FFT problem

TTNN has no FFT of any kind. HiFT ends in an inverse STFT, so the vocoder looks
unportable at first glance.

It isn't, because `n_fft = 16`. At that size the inverse DFT of 9 one-sided bins is a
fixed 16x9 real matrix pair, smaller than a single 32x32 tile -- a matmul. Windowing
and overlap-add fuse into one transposed convolution with a diagonal kernel, since OLA
(`out[t*hop+j] += frame[j,t]*w[j]`) and `conv_transpose1d`
(`out[o,t*s+k] += in[i,t]*W[i,o,k]`) are the same operation. NOLA normalisation
depends only on frame count, so it is a precomputed constant multiply.

Net: `matmul + conv_transpose2d + multiply`. All ops TTNN already has. See
`tt/hifigan/istft.py` for the derivation and `tests/pcc/test_istft.py` for the checks.

This module is unmodified in substance from the equivalent identity in an unmerged
CosyVoice1 TTNN port
([tenstorrent/tt-metal#52540](https://github.com/tenstorrent/tt-metal/pull/52540)),
which validated it to PCC 1.0 (fp32) / 0.9999765688 (bf16) against a real vocoder's
captured magnitude/phase spanning 14 decades of dynamic range. That specific numeric
claim has not been re-measured here yet -- see [Known gaps](#known-gaps) -- but nothing
in the identity is version-specific, so there is no reason to expect it to move.

## Running

```bash
# host tier: no device, ~1s
pytest models/demos/audio/cosyvoice2/tests/pcc/test_istft.py -k "not device"

# device tier: needs /dev/tenstorrent
pytest models/demos/audio/cosyvoice2/tests/pcc/test_istft.py -k device
```

## Known gaps

- **No golden validation yet.** `tests/pcc/test_istft.py`'s device tests check the
  identity against synthetic magnitude/phase spanning 14 decades, not a real
  CosyVoice2 vocoder's captured tensors -- there is no CosyVoice2-0.5B checkpoint or
  golden-capture tooling in this environment yet. Once a checkpoint is available, add
  a `scripts/gen_golden.py`-equivalent capture and a golden-based PCC test alongside
  the synthetic one, following the pattern in the CosyVoice1 port referenced above.
- **RNG-mid-forward.** Not yet relevant to `istft.py` itself (it draws no randomness),
  but `ConditionalCFM`, `SineGen`, and `SourceModuleHnNSF` all do, and CosyVoice2 has a
  third NSF source branch versus CosyVoice1's two. Seeding alone will not make TTNN and
  PyTorch comparable for those modules -- every random draw needs to be captured as a
  named golden array and fed in explicitly during PCC validation. Relevant starting at
  the upsample/NSF stack (next piece of Phase 1), not yet for this module.
- **`ttnn.cumsum` bf16 accuracy.** Not used by `istft.py`. Will matter once `SineGen`'s
  phase integration is ported -- use `ttnn.cumsum(dtype=ttnn.float32)` explicitly there.
- **HiFTGenerator vs CausalHiFTGenerator.** The upstream training config instantiates
  the plain (non-causal) `HiFTGenerator` for the vocoder, confirmed from
  `examples/libritts/cosyvoice2/conf/cosyvoice2.yaml`. Worth re-confirming directly
  against whichever released checkpoint bring-up eventually targets, since a released
  checkpoint's config could in principle differ from the training config in the repo.
