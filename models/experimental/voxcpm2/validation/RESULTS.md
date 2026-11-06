# VoxCPM2 component accuracy

The sections below retain historical measurements and their original acceptance criteria. Current qualification uses a fresh source build and explicit RMS bounds; the decoder precision and CI changes are described in the 2026-10-16 review follow-up at the end of this report.

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

## PyTorch 2.10 compatibility — 2026-10-07

Torch and Torchaudio were upgraded together from 2.8.0 to 2.10.0, and the model
dependency lock was regenerated. No model or validation Python computation was
changed; implementation revision remains `6c2f371a`. Isolated environments were
used so the original qualification environments remain available.

Native, uncompiled CUDA generation ran on RTX A6000 GPU 2 with Torch/Torchaudio
2.10.0+cu128, Python 3.12, and the same pinned official source and checkpoint.
The fox sentence, seed 42, CFG 2.0, ten diffusion steps per patch and 32-patch cap
produced four seconds at 48 kHz. A second run used that audio as voice reference
and produced 3.68 seconds. Both WAVs were finite and non-silent. Fresh component
captures from these runs were replayed on the same reserved Blackhole p150b.

The TT compatibility environment used Python 3.10, Torch 2.10.0+cpu and the
existing TTNN 0.65.1rc17.dev6333+h3.3 wheel. The wheel's declared NumPy < 2
constraint required NumPy 1.26.4 rather than the old environment's 2.2.6; the
new environment passed `uv pip check`. CUDA dependencies also passed that check.
TT computation retained BF16, HiFi4, FP32 destination accumulation and the
256 KiB L1-small reservation. No fresh TT-Metal build was performed.

All 16 cases covering 14 component types passed PCC >= 0.99 against the fresh
2.10 CUDA oracle. All 30 configured TT tests passed with no skips; host-only
discovery passed 29 tests and explicitly skipped the device test. The
[machine-readable upgrade record](results/2026-10-07-torch210-pcc.json) includes
all case metrics, environment versions and capture-manifest hashes.

| Component | PCC | Relative RMS error |
| --- | ---: | ---: |
| Base LM prefill | 0.9999093 | 1.3470% |
| Residual LM prefill | 0.9999660 | 0.8423% |
| Local patch encoder | 0.9999826 | 0.5983% |
| FSQ | 0.9999813 | 0.6156% |
| Local DiT, three estimator calls | 0.9999425–0.9999566 | 0.9677–1.1138% |
| AudioVAE encoder | 0.9995424 | 3.0457% |
| AudioVAE decoder, complete 4 s waveform | 0.9997835 | 7.6411% |

The native same-seed CUDA waveform is not stable sample-by-sample across Torch
2.8 and 2.10: waveform PCC is 0.632118, relative RMS difference 86.2285%, with
the same output length. This comparison measures cross-version free-running
generation, not matched-input component correctness. The component gates above
use fresh 2.10 native inputs and outputs. These tests establish execution and
the stated component accuracy; they do not establish bitwise cross-version
generation, subjective speech quality or integrated TT synthesis. Different
native trajectories also mean changes in component metrics between the two
qualification runs must not be described as accuracy improvements.


## Source-build review follow-up — 2026-10-16

The complete component suite passed on a fresh build of the submitted native source tree: **58 tests passed, including all 16 device cases across 14 component types, with zero failures or skips**. Pytest 9 additionally reports 49 successful unittest subtests; these are not 49 additional device cases. Suite elapsed time was **131.99 seconds** on the reserved Blackhole p150b, within the registered 10-minute job budget. This is correctness-suite wall time with possible JIT compilation, not steady-state inference performance.

The [machine-readable qualification](results/2026-10-16-source-build-pcc.json) includes exact source-file SHA256 values, checkpoint identity, both pinned CUDA manifest hashes, native binary SHA256, runtime versions and every measured case. The native tree was cloned at `ec74d3be21e8dabe457e87df464a047c07c115e2`; native code was unchanged by this follow-up, and the patched model/validation Python sources are identified by their hashes. This qualifies the submitted branch's native sources rather than a rebase onto the latest upstream main.

### Runtime and reproduction

- Source build: `CMAKE_BUILD_PARALLEL_LEVEL=12 ./build_metal.sh --release --disable-profiler --without-distributed`, with the branch's submodules and bundled SFPI.
- Python 3.10.12, Torch 2.11.0+cpu, Torchvision 0.26.0+cpu, NumPy 1.26.4, pytest 9.0.3. All installed dependencies pass `uv pip check`.
- TTNN imports from the newly built source checkout. Its shallow-clone distribution version is `0.1.dev1+ggec74d3b`; that metadata string is not an upstream release identifier. The recorded source/binary hashes establish provenance.
- Hardware: one Blackhole p150b, BDF `0000:c1:00.0`, UMD ID 6, `/dev/tenstorrent/5`, filtered logical ID 0; 256 KiB L1-small reservation. UMD IDs and Linux device nodes are different numbering systems.
- CUDA oracle: the existing native Torch 2.10 text and voice captures, with unchanged source/checkpoint pins, fox sentence, seed 42 and generation settings from the preceding section. No new Torch 2.11 CUDA synthesis is claimed.
- Precision: transformer components retain BF16/HiFi4/FP32 destination accumulation. Codec activations, nonlinear operations, residual additions and outputs use FP32, while convolution inputs/weights use BF16 and FP32 accumulation. The report exposes both storage and convolution operand dtypes.

After the build, requirements installation and device reservation described in the README, set the three pinned asset environment variables plus `VOXCPM2_REQUIRE_DEVICE_TESTS=1` and run:

```bash
pytest models/experimental/voxcpm2/tests --device-id 0 --timeout=600
```

The measured local invocation additionally selected a report directory and JUnit output; its exact command is in the JSON record. If adding a custom `--basetemp`, create its parent directory first. Tests use the repository `device` fixture, which owns opening and closing the device; replay does not close a supplied fixture-owned device.

### Current accuracy

| Component / event | Activation/output storage | PCC | Relative RMS error |
| --- | --- | ---: | ---: |
| audio_vae.decode / 0 | float32 | 0.9999540 | 1.2561% |
| audio_vae.encode / 0 | float32 | 0.9997892 | 2.0619% |
| base_lm.forward / 0 | bfloat16 | 0.9998486 | 1.7401% |
| enc_to_lm_proj.forward / 1 | bfloat16 | 0.9999996 | 0.0903% |
| feat_decoder.estimator.forward / 0 | bfloat16 | 0.9999566 | 0.9888% |
| feat_decoder.estimator.forward / 4 | bfloat16 | 0.9999525 | 0.9823% |
| feat_decoder.estimator.forward / 8 | bfloat16 | 0.9999574 | 0.9247% |
| feat_encoder.forward / 1 | bfloat16 | 0.9999820 | 0.6069% |
| fsq_layer.forward / 1 | bfloat16 | 0.9999813 | 0.6156% |
| fusion_concat_proj.forward / 1 | bfloat16 | 0.9999996 | 0.0920% |
| lm_to_dit_proj.forward / 0 | bfloat16 | 0.9999994 | 0.1105% |
| res_to_dit_proj.forward / 0 | bfloat16 | 0.9999994 | 0.1120% |
| residual_lm.forward / 0 | bfloat16 | 0.9999690 | 0.8194% |
| stop_actn.forward / 0 | bfloat16 | 1.0000000 | 0.0284% |
| stop_head.forward / 0 | bfloat16 | 1.0000000 | 0.0000% |
| stop_proj.forward / 0 | bfloat16 | 0.9999995 | 0.0918% |

The gate requires PCC ≥ 0.99 and relative RMS ≤ 2% for ordinary components, ≤ 5% for encoding and ≤ 3% for decoding. Stop-head logits also require maximum absolute error ≤ 0.001; the measured error is zero. PCC on the two-element stop output alone is not evidence of amplitude accuracy. These are component acceptance limits, not universal 1% elementwise bounds.

### Decoder investigation and correction

A fit of the historical waveform against its matched CUDA output gives gain 1.0730594 and offset 0.00020768. After removing the diagnostic fit, residual relative RMS is 2.23%; total error is 7.64%. This confirms a substantial gain component, rather than merely uncorrelated waveform noise. No fitted correction is applied to production/replay outputs.

The [decoder stage record](results/2026-10-16-decoder-stages.json) checks each stage independently from its native input and cumulatively through the TT decoder. Error grows through multiple upsampling/residual blocks. Controlled precision trials on the fresh build produced:

| Decoder policy | PCC | Relative RMS error | Fitted gain |
| --- | ---: | ---: | ---: |
| BF16 storage throughout | 0.9997971 | 7.2997% | 1.0697287 |
| FP32 storage and FP32 convolution operands | 0.9999368 | 4.9449% | 0.9517222 |
| FP32 storage and BF16 convolution operands | 0.9999540 | 1.2561% | 0.9917985 |

FP32 pointwise/residual storage avoids repeated BF16 rounding, while explicitly rounded BF16 convolution operands avoid truncating arbitrary FP32 operand mantissas on the matrix path. The latter behavior is documented in this tree's FP32 depthwise tests. The final policy keeps native bias handling and causal padding, and uses TTNN operations for all activation work. This is a measured precision choice, not a mathematical error bound for every utterance.

Long FP32 transpose convolutions initially exceeded L1 during automatic slicing. Explicit DRAM width slices bound input length per slice to approximately 4096 samples; the complete decoder and full suite then ran without allocation failures. Prepared-weight caches also include input dtype, layout and memory configuration rather than only batch/length.

A second, independent seed-43 capture also passed the same 3% decoder RMS gate: **PCC 0.9999751, relative RMS 1.0615%, fitted gain 0.9920224**. This replay uses the saved native Torch 2.8 CUDA capture and its exact captured latents, not cross-version waveform matching. Both tested waveforms are four seconds at 48 kHz.

### CI and integrity changes

The former single environment-selected unittest is replaced by 16 explicit pytest cases. The two oracle manifest hashes are pinned; the encoder uses its separate voice-reference capture. Missing assets skip in ordinary local runs and fail in CI under `VOXCPM2_REQUIRE_DEVICE_TESTS=1`; changed manifests fail in either mode. The suite is registered for `bh_p150`, tier 3. Repository matrix generation and all 276 time-budget buckets pass without increasing budgets.

The nested standalone project/lock is replaced by runtime and separate CUDA-reference requirements. Runtime requirements permit CI's Torch 2.11 and constrain NumPy below 2. The checkpoint loader only selects weight files present in the verified manifest, with regressions covering unrecorded safetensors overrides for both model and codec. Missing installed-distribution metadata no longer loses a completed replay report, and source paths plus measured Python hashes are recorded. Derived implementations include OpenBMB attribution.

Remaining maintainer actions: provision the approximately 5.4 GiB pinned checkpoint/capture set under the documented read-only CI path and assign a notification owner. Local source-build results do not imply that upstream CI has those assets or has run this new suite. Current-main rebase testing, additional devices/shapes/voices/languages, integrated TT synthesis and steady-state performance remain outside this qualification. The card reservation was released after all jobs closed; no resets, firmware changes or other users' workload changes were made.
