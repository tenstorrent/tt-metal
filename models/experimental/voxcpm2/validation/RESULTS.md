# VoxCPM2 component accuracy

All 16 TT replay cases covering 14 implemented component types passed the stated
PCC >= 0.99 criterion. These tests replay native CUDA component inputs into TT
components. They do not qualify integrated autoregressive TT speech generation.
The [machine-readable results](results/2026-10-01-component-pcc.json) record the
checkpoint hashes, input shapes/dtypes, events, hardware and error metrics.

## Conditions

- Model: `openbmb/VoxCPM2`, checkpoint revision `32279effe8c19989596f05d353d1447f51d9e915`.
- Official code: OpenBMB/VoxCPM revision `f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629`.
- Tested implementation: `b3bec4eb2977efcfac561b8394fa6b7d808f0502`, including the device fixes described below. Earlier unchanged components were measured while this change was in progress.
- Native CUDA reference: local RTX A6000 GPU 2, Torch 2.8.0+cu128, compile disabled, TF32 disabled, retries disabled.
- TT: `f02cs02`, physical Blackhole card 5, BDF `0000:c1:00.0`, filtered logical device 0.
- Runtime: installed `ttnn==0.65.1rc17.dev6333+h3.3`, own model venv, Torch 2.8.0+cpu. No fresh TT-Metal build was performed for this test.
- TT storage: BF16; compute: HiFi4, FP32 destination accumulation where supported.
- Text: “The quick brown fox jumps over the lazy dog.” Seed 42, CFG 2.0, 10 diffusion steps per audio patch, maximum 32 patches, batch 1. The DiT estimator receives the native two-item CFG batch.
- Native text-only generation produced 25 patches / 4.0 seconds at 48 kHz. A second native generation used that audio as a voice reference to exercise AudioVAE encoding.
- Component correctness only. Cold launch timings include possible compilation and are not a steady-state performance qualification.

## Results

| Component / native event | PCC | Relative RMS error |
| --- | ---: | ---: |
| Base LM prefill / 0 | 0.9999093 | 1.3470% |
| Residual LM prefill / 0 | 0.9999660 | 0.8423% |
| Local patch encoder / 1 | 0.9999862 | 0.5287% |
| Encoder-to-LM projection / 1 | 0.9999994 | 0.1073% |
| FSQ / 1 | 0.9999977 | 0.2237% |
| Fusion projection / 1 | 0.9999999 | 0.0507% |
| LM-to-DiT projection / 0 | 0.9999994 | 0.1105% |
| Residual-to-DiT projection / 0 | 0.9999994 | 0.1120% |
| Stop projection / 0 | 0.9999995 | 0.0918% |
| Stop activation / 0 | 1.0000000 | 0.0284% |
| Stop logits / 0 | 1.0000000 | 0.0000% |
| Local DiT / 0 | 0.9999566 | 0.9957% |
| Local DiT / 4 | 0.9999425 | 1.1138% |
| Local DiT / 8 | 0.9999570 | 0.9339% |
| AudioVAE encoder / 0, voice-reference capture | 0.9991306 | 4.1925% |
| AudioVAE decoder / 0, complete 4.0 s waveform | 0.9997781 | 7.4962% |

The three local DiT events span the first generated patch's denoising schedule.
The local-encoder event uses a real generated patch rather than only its zero
initialization. The codec encoder processes real reference audio; the decoder
processes all native generated latents. Codec output tensors were saved to disk.
The TT waveform is a codec replay from CUDA latents, not a fully TT-generated
utterance. Passing PCC does not mean every tensor has a 1% relative or absolute
error bound, or that final integrated audio quality has been demonstrated.

## Issues found and fixed

1. The installed TTNN `Shape` class supports integer indexing but not Python slicing; local DiT now converts it to a tuple before slicing.
2. The official timestep embedding rounds frequency construction and both angle multiplications in the input timestep dtype. The previous FP32 implementation removed native BF16 rounding boundaries. The port now retains them, and the corrected DiT was tested at three native timestep events.
3. Codec convolution defaults exceeded L1 circular-buffer capacity. Explicit 32-row activation blocks bounded that allocation. Its cached reader tables also exhausted a 32 KiB L1-small region; the complete encoder/decoder tests passed with a 256 KiB reservation.
4. The original dependency ranges selected a newer Torch/CUDA combination. The validated CUDA environment now pins Torch and Torchaudio 2.8.0, and the dependency lock is retained.

Every device run acquired the existing card lock and shared reset coordination
lock, verified ownership, and held an explicit project reservation. Each device
closed cleanly and the reservation was released. No cards were reset and other
users' workloads were left running.

## Reproduction and remaining work

Use the CUDA capture and component replay commands in the [model README](../README.md),
with the pinned checkpoint above. Replay event 1 for the local encoder, FSQ,
encoder-to-LM and fusion projections; events 0, 4 and 8 for local DiT; event 0
for the other components. Use the native voice-reference capture for
`audio_vae.encode`. Open the codec device with `--l1-small-size 262144`.

The own model environment's full unittest discovery passed 30 tests, including
a configured real-device local encoder replay, with no skipped tests. Host-only
runs without hardware configuration still skip that device test explicitly.

Still unimplemented or unqualified: AR KV-cache decoding, diffusion solver and
complete TT generation loop, integrated speech quality, streaming, multilingual
coverage, additional voices/sample rates and steady-state performance. FP32 codec
storage was not tested. These selected component cases do not establish the
behavior of every input length or every denoising step.

## Cleanup regression — 2026-10-07

All 16 original component/event pairs passed PCC >= 0.99 again after formatting,
unused-import and documentation cleanup. The same reserved physical Blackhole
card 5, BF16/HiFi4 configuration and installed runtime were used. The configured
unittest suite passed all 30 tests with no skips. The local host run passed 29
tests with the device test explicitly skipped. Locked reference dependency
installation, compilation and applicable repository pre-commit hooks also
passed. Python computational ASTs were unchanged by cleanup, excluding
docstrings/import declarations; no computation or precision policy was altered.
The [regression summary](results/2026-10-07-cleanup-regression.json) records
tested source-file hashes and metrics. Its parent revision is `04ba2da9`;
the cleanup working tree is identified by those hashes.

The dependency lock was narrowed to Linux x86-64 and Python 3.10–3.12, covering
the tested hosts, to stay below the repository's 500 KiB file-size limit. This
does not add hardware or Python-version coverage beyond the measured cases.

A separate fresh seed-43 native CUDA generation produced a complete four-second
48 kHz waveform for the same fox sentence, using ten diffusion timesteps per
patch. TT decoded its CUDA-generated latents with PCC 0.999825983 and relative
RMS error 6.8647%; both WAVs were saved for listening comparison. The
[audio replay record](results/2026-10-07-audio-decode-pcc.json) retains the
checkpoint and numerical conditions. This audio case preceded cosmetic cleanup
and used implementation `04ba2da9`; the complete original-case regression above
ran after cleanup. Neither run qualifies integrated TT synthesis.

Fresh current-main TTNN builds, upstream CI/post-commit regressions and
maintainer/codeowner acceptance have not been completed. Successful tests on
the installed runtime must not be presented as those checks passing.
