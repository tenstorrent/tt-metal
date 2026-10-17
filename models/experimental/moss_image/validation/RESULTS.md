# MOSS-Image accuracy and runtime evidence

## Qualification status

The initial Base/SFT implementation completes independent text-to-image generation on one and two Blackhole p150b cards. Component PCC checks pass the stated component thresholds. **Final integrated image accuracy is not qualified at PCC 0.99: all four initial images fall below that value.** Component replay success is not an end-to-end accuracy pass.

The readings below were collected on 2026-10-16 using a native TT-Metal source build reported as `0.1.dev1+ggec74d3b`, before migration into this current-main branch. They are historical bring-up evidence, not a claim that the migrated branch has passed on its current native runtime. Current-main build and device qualification must be reported separately.

Raw measurements and recorded source hashes:

- [Component results](results/2026-10-16-components-historical.json).
- [Independent integrated results, all 50 latent comparisons, environment and source hashes](results/2026-10-16-integrated-historical.json).

## Workload and reference

Both checkpoints use the official MOSS-Image architecture: Qwen3.5-9B prompt encoder, 42-layer MOSS diffusion transformer, and FLUX.2 VAE. Base and SFT share the text encoder, tokenizer, scheduler and VAE files; only the transformer weights differ. The native CUDA oracle executes the reviewed official pipeline rather than a reimplemented approximate reference.

| Setting | Value |
|---|---|
| Prompt | `the quick brown fox jumps over the lazy dog` |
| Negative prompt | Empty string |
| Batch / resolution | 1 / 1024 × 1024 |
| Steps / seed | 50 / 42 |
| Guidance / schedule shift | 3.0 / 5.0 |
| Negative branch skip | Transformer block 22, zero-based, all steps |
| Initial noise | FP32 CPU generator; independently generated TT/CUDA tensors match bitwise |
| Schedule | Official MOSS nearest-grid scheduler; TT/CUDA sigmas and timesteps match bitwise |
| Transformer/text precision | BF16, TT HiFi4 with FP32 destination accumulation |
| CFG / Euler | FP32 on TT |
| TT placement | Prompt encoder on first card; DiT on one card or head/channel TP across two cards; VAE replicated |
| CUDA | NVIDIA RTX A6000; PyTorch 2.11.0 CUDA 13.0; Transformers 5.17.0; Diffusers 0.40.0 |
| TT host | PyTorch 2.11.0 CPU; no CUDA execution in TT generation |

Pinned checkpoints: Base `1ceede80badddd6a7812069619edaa5633587192`, SFT `fc5de0d56fc910cbca6f6625a70f5f582b26deb1`. Official pipeline SHA256: `1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31`. Full consumed checkpoint hashes are recorded in the original capture/comparison manifests.

Relative RMS means `||TT − CUDA||₂ / ||CUDA||₂`. It is a fraction in JSON and a percentage in the tables below. PCC is centered Pearson correlation. Neither metric alone bounds every element's error.

## Component accuracy

The prompt encoder starts from token IDs and performs all learned operations on TT. Its 66 layer/final-output comparisons pass the encoder thresholds, PCC ≥ 0.99 and relative RMS ≤ 3%. Positive final conditioning: PCC **0.999645114**, relative RMS **2.6647%**, maximum absolute error **1.6875**. Negative final conditioning: PCC **0.999555002**, relative RMS **2.9837%**, maximum absolute error **1.625**. These weights are shared between Base and SFT.

The following DiT tests run a complete first-step transformer forward with CUDA-captured input latents and conditioning. They isolate DiT accuracy and are explicitly **activation-injected component tests**. The component bounds are PCC ≥ 0.99 and relative RMS ≤ 5%.

| Checkpoint | Cards | Positive velocity PCC | Positive relative RMS | Negative velocity PCC | Negative relative RMS |
|---|---:|---:|---:|---:|---:|
| Base | 1 | 0.999949567 | 1.0362% | 0.999952063 | 1.0152% |
| Base | 2 | 0.999950289 | 1.0518% | 0.999951938 | 1.0505% |
| SFT | 1 | 0.999938225 | 1.1344% | 0.999949494 | 1.0575% |
| SFT | 2 | 0.999938594 | 1.1369% | 0.999948561 | 1.1085% |

All four positive outputs and the first three negative outputs have maximum absolute error 0.0625; SFT two-card negative is 0.078125. These are measured output errors, not an absolute-tolerance guarantee.

The VAE replay uses CUDA final latents to isolate inverse batch normalization, latent unpacking and decoding. At 1024 × 1024, Base one-card decoder PCC is **0.999990095**, relative RMS **0.5440%**, maximum absolute error **0.029943**. The other checkpoint/card combinations have similar PCC around 0.999990; see the raw records for exact values and per-replica readings. VAE acceptance is PCC ≥ 0.99 and relative RMS ≤ 3%. The isolated FP32 guidance/Euler arithmetic test matches CUDA bitwise.

## Independent integrated accuracy

These runs start from the raw prompt and a fresh seeded Gaussian tensor. Captured reference activations are used only for post-run comparison. They run all 50 steps and decode the resulting TT latent through the TT VAE.

| Checkpoint | Cards | Final latent PCC | Final latent relative RMS | Decoded pixel PCC | Decoded pixel relative RMS |
|---|---:|---:|---:|---:|---:|
| Base | 1 | 0.971663038 | 23.6873% | 0.974538317 | 18.6630% |
| Base | 2 | 0.978349201 | 20.7069% | 0.981509305 | 15.9628% |
| SFT | 1 | 0.860320519 | 52.7332% | 0.850775170 | 56.7092% |
| SFT | 2 | 0.862588197 | 52.2494% | 0.857495285 | 55.2366% |

The images are coherent fox illustrations/posters. Base has similar layout across CUDA and TT, with changed outlines/geometry. SFT changes the fox's mouth, background details and title lettering. **None of these raw-prompt outputs, including the native CUDA references, depicts a fox jumping over a dog.** Prompt interpretation is therefore a separate issue from TT numerical fidelity.

The error grows throughout denoising, before decode. For example, SFT one-card latent relative RMS is 0.0144% after step 1, 0.4522% after step 10, 6.2638% after step 25, 28.3019% after step 40, and 52.7332% after step 50. Latent PCC first drops below 0.99 at step 34 (SFT one card), 33 (SFT two cards), 43 (Base one card), and 44 (Base two cards). The raw JSON contains every step.

This establishes progressive denoising trajectory divergence. A high component PCC does not imply that a nonlinear 50-step feedback process preserves the same final image. The controlled experiment below isolates the encoder's contribution.

Unlike the earlier VoxCPM decoder investigation, a simple gain/offset discrepancy does not explain this result. Fitting `TT = gain × CUDA + offset` for diagnosis only leaves Base one-card pixel residual relative RMS 18.6133% (original 18.6630%) and SFT one-card 52.5350% (original 56.7092%). No fitted correction is applied to any output. Complete fits and normalization definitions are in the integrated JSON.

## Conditioning ablation, 2026-10-17

The official CUDA transformer, VAE, schedule, initial noise, guidance and weights were retained unchanged. Only the positive and negative conditioning tensors were replaced with the independently generated TT text encoder outputs. Each checkpoint ran 50 denoising steps from fresh seed-42 noise. This is an **activation-injected causal diagnostic**, not an independent integrated qualification or a proposed production fallback. All four original TT runs have bitwise identical conditioning tensors for each branch.

| Checkpoint | Backend after conditioning | Final latent PCC vs original CUDA | Final latent relative RMS | Pixel PCC vs original CUDA | Pixel relative RMS |
|---|---|---:|---:|---:|---:|
| Base | Official CUDA with TT conditioning | 0.979350284 | 20.2517% | 0.983491025 | 14.9612% |
| SFT | Official CUDA with TT conditioning | 0.873899875 | 50.0500% | 0.872724321 | 51.7234% |

Changing the prompt conditioning alone reproduces a similarly sized final difference, including the large SFT drift. The text encoder is therefore a demonstrated major contributor; this conclusion does not rely on an assumed diffusion sensitivity. The ordinary DiT replay errors also matter: full TT versus CUDA using identical TT conditioning still gives pixel PCC 0.986525830 and relative RMS 13.6089% for Base, and PCC 0.942911097 / relative RMS 34.1532% for SFT. Those error magnitudes are not additive, and this experiment does not establish an exact percentage decomposition of causes.

Encoder error grows gradually across its 32 layers, rather than appearing at one discrete broken layer: positive layer-output relative RMS is 0.210% at layer 0, 0.958% at layer 7, 1.587% at layer 15, 1.967% at layer 23, and 2.322% at layer 31, reaching 2.665% after final normalization. Positive per-token final errors range from 0.899% to 6.670%; negative errors range from 1.256% to 4.929%. A 3% global encoder relative-RMS allowance is consequently too loose to establish final-image agreement for this workload, even though global PCC exceeds 0.9995.

The next accuracy work should tighten prompt encoder arithmetic against layer/component captures and separately measure teacher-forced DiT predictions later in the schedule. No final-image accuracy pass is asserted until the complete independent runs are repeated. [Raw conditioning ablation readings](results/2026-10-17-conditioning-ablation.json) preserve the diagnostic scope, hashes and paired comparisons. Both diagnostic requests took approximately 69 seconds each on the RTX A6000; these are instrumented diagnostic wall times, not a TT performance comparison.

## Teacher-forced late denoising, SFT, 2026-10-17

One reusable single-card TT DiT was evaluated at selected points of the original CUDA trajectory. Both the latent and conditioning were supplied from the corresponding CUDA captures. This separates local DiT arithmetic error from independently accumulated latent/encoder differences. All consumed transformer and artifact hashes were checked. These are historical-native-runtime diagnostic results, not current-main hardware qualification.

| Zero-based step | Positive PCC | Positive relative RMS | Negative PCC | Negative relative RMS | Guided velocity PCC | Guided velocity relative RMS |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.999938225 | 1.1344% | 0.999949494 | 1.0575% | 0.999490934 | 3.1886% |
| 10 | 0.999739301 | 2.2872% | 0.999803753 | 1.9900% | 0.997964457 | 6.3766% |
| 25 | 0.999803273 | 2.0732% | 0.999797965 | 2.1119% | 0.998458428 | 5.5838% |
| 40 | 0.999705540 | 2.6877% | 0.999712750 | 2.6459% | 0.998002990 | 6.4601% |
| 49 | 0.999531405 | 3.7014% | 0.999551020 | 3.5620% | 0.996670474 | 8.5703% |

Guided velocity is calculated on the host from the captured TT branch outputs in FP32, as `negative + 3 × (positive − negative)`, and compared with native CUDA guidance. It measures amplification of branch errors rather than requalifying the TT scheduler. Positive/negative error can combine and amplify under CFG; checking only unguided branch PCC or only the first step misses this effect. There is no abrupt late-layer failure, but local velocity relative error increases across the trajectory. [Raw teacher-forced readings and source hashes](results/2026-10-17-teacherforce-SFT.json) contain maximum absolute errors as well.

A separate step-49 CUDA replay captured blocks 0, 21 and 41, including leaf component inputs/outputs, attention QKV, and global conditioning/time/head projections, for further localization. Both positive and negative final predictions match the original step-49 CUDA capture **bitwise**. The consumed DiT checkpoint files and parent capture inputs were independently hash-verified. The external tensor set occupies 4.726 GiB; it is not checked into Git. [Oracle provenance and checks](results/2026-10-17-SFT-step49-oracle.json) identify the full artifact-manifest hash and source. This is a component replay oracle, not a new full generation.

## CUDA accumulation policy control, 2026-10-17

The initial CUDA capture used PyTorch's default BF16 matmul reduction policy. A separate experiment disables `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction` explicitly, retaining the same official model, raw prompts, checkpoint files, seed, resolution, schedule and 50 steps. Each run executes independently through the native CUDA encoder, transformer and VAE. The original reference remains preserved. Initial noise and schedule tensors match bitwise between the two policy settings.

| Checkpoint | Comparison | Final pixel PCC | Final pixel relative RMS |
|---|---|---:|---:|
| Base | CUDA policy False vs original CUDA | 0.926343215 | 32.0824% |
| SFT | CUDA policy False vs original CUDA | 0.971529729 | 24.1479% |
| Base | Historical TT one card vs CUDA policy False | 0.913187540 | 34.6133% |
| Base | Historical TT two cards vs CUDA policy False | 0.919327146 | 33.3803% |
| SFT | Historical TT one card vs CUDA policy False | 0.853823079 | 55.3163% |
| SFT | Historical TT two cards vs CUDA policy False | 0.859712913 | 54.0207% |

Disabling reduced-precision reductions changes native CUDA conditioning by 1.7555% positive / 2.1785% negative relative RMS and produces a substantial final-image difference even without TT. This demonstrates sensitivity to arithmetic policy, but **does not fix the TT/CUDA discrepancy**. It is not evidence that either backend's output can be declared accurate merely by choosing a different reference. Precision flags must be recorded and kept explicit, and both the historical and controlled results must remain visible. [Raw precision-policy comparison](results/2026-10-17-cuda-precision-policy.json) includes conditioning, first prediction, final latent and final pixel readings, plus the new reference manifest hashes.

## Proven arithmetic differences and candidate corrections

Fixed-input replays identified concrete arithmetic mismatches, beyond the general observation of denoising sensitivity. The following results are diagnostic alternatives on the historical native runtime. They do not assert that these changes have passed a corrected independent integrated run or current-main device suite. [Raw arithmetic diagnostic evidence](results/2026-10-17-arithmetic-diagnostics.json) preserves individual readings and source-file hashes.

**BF16 pointwise rounding:** direct TT BF16 addition differs from PyTorch at exact halfway values. For example, `1 + 0.00390625` produces `1.0078125` on the tested direct TT path and `1.0` with PyTorch's nearest-even rounding. TT FP32 computation followed by BF16 conversion produces `1.0`, matching PyTorch. The paired positive/negative test values in `rounding_ties.json` establish this as an arithmetic-contract difference rather than a random generation discrepancy. This statement applies to the measured direct pointwise path, not every TT conversion or matmul implementation.

Applying FP32 intermediates with BF16 conversion at the corresponding reference boundaries makes the following encoder operations bitwise equal on captured native inputs:

| Encoder component | Original relative RMS, layer 0 / layer 1 | Corrected fixed-input relative RMS |
|---|---:|---:|
| Convolution SiLU | 0.0239% / 0.1790% | 0% / 0% |
| SwiGLU | 0.1327% / 0.1602% | 0% / 0% |
| Attention residual addition | 0.1167% / 0.1740% | 0% / 0% |
| MLP residual addition | 0.0626% / 0.1601% | 0% / 0% |

The DiT's bounded `_situ` activation also differs on native inputs: original relative RMS is 0.371%, 0.535%, and 0.392% at Base blocks 0, 21 and 41. FP32 pointwise intermediates with explicit reference BF16 boundaries match the saved CUDA output bitwise across all 6.29 million compared values. Native TT rotary application has roughly 0.27–0.33% relative RMS on captured Q/K inputs; explicit split-half arithmetic with the reference rounding boundaries matches all six complete Q/K tensors bitwise. The host RoPE constants were checked against the actual captured CUDA tensors and match bitwise, so this is not an incorrect positional formula.

**GDN scan arithmetic:** using fixed native scan inputs, the initial chunk implementation has relative RMS approximately 0.683% and 0.744% in encoder layers 0 and 1. A diagnostic recurrence with more precise pointwise operations reduces those values to 0.0000195% and 0.0002249%; respectively 99.978% and 99.986% of BF16 elements match exactly. This proves an avoidable scan arithmetic contribution, while leaving the production implementation/performance choice subject to separate qualification.

Combining encoder scan and pointwise alternatives improves full positive conditioning from PCC 0.999645 / relative RMS 2.665% to **PCC 0.999839 / relative RMS 1.794%**, and negative conditioning from PCC 0.999555 / relative RMS 2.984% to **PCC 0.999671 / relative RMS 2.569%**. Remaining encoder discrepancy is still material; isolated-operation equality must not be mistaken for a completely matched encoder.

**Biased linear epilogues:** the initial DiT rounds a matmul to BF16 before adding the BF16 bias. Retaining the matmul result in FP32 through the bias addition and casting afterward better matches the reference epilogue. On fixed native inputs:

| Projection | Original separate BF16 bias relative RMS | FP32 matmul + bias, then BF16 relative RMS | Original / corrected maximum absolute error |
|---|---:|---:|---:|
| Shared block modulation | 0.2519% | 0.0444% | 0.125 / 0.015625 |
| Video head modulation | 0.2580% | 0.0231% | 0.03125 / 0.00390625 |
| Video embedder | 0.2898% | 0.1053% | 0.125 / 0.125 |

Modulation affects every transformer layer, so reducing this error is relevant even though projection PCC was already high. A partial DiT candidate using corrected pointwise/rotary arithmetic and a more precise attention exponential reduces SFT step-49 positive/negative relative RMS from 3.701% / 3.562% to approximately 2.54% / 2.53%; guided error remains about 7.57%.

Adding the corrected bias epilogues produces the following matched-input SFT results:

| Zero-based step | Positive PCC / relative RMS | Negative PCC / relative RMS | Guided PCC / relative RMS |
|---:|---|---|---|
| 0 | 0.999962039 / 0.9754% | 0.999973694 / 0.8790% | 0.999628911 / 2.7338% |
| 49 | 0.999739410 / 2.2844% | 0.999732635 / 2.3162% | 0.997277882 / 7.3669% |

These are activation-injected teacher-forced component measurements, using a 32-token attention query/key chunk diagnostic. They improve on the original implementation, but late-step branch error remains above 2% and guided error remains substantial. This is **not a resolved final-image accuracy result**. Attention chunk selection, the final production implementation and independent integrated runs require their own results before a passing claim.

## Runtime boundaries

These are single qualification requests with compilation/cache effects, transfers and artifact writes; they are **not steady-state benchmark numbers**. They should not be compared directly against the heavily instrumented CUDA capture wall times.

| Checkpoint | Cards | Prompt encoding | Denoising sum | VAE construction + decode | In-process total |
|---|---:|---:|---:|---:|---:|
| Base | 1 | 51.357 s | 267.237 s | 1.854 s | 322.956 s |
| Base | 2 | 51.199 s | 182.362 s | 13.481 s | 249.637 s |
| SFT | 1 | 51.897 s | 267.894 s | 1.720 s | 324.200 s |
| SFT | 2 | 51.349 s | 182.882 s | 2.044 s | 239.110 s |

The in-process timer starts after runtime import and ends after saving the image, before final device closure. Encoder timing excludes encoder construction. Denoising sums synchronized per-step timings, including first-step compilation/upload and first-prediction captures, but excludes the subsequent latent save/status writes. VAE timing includes construction, decode and result transfer. These boundaries explain why column sums need not equal the total. The Base two-card VAE measurement includes a larger cold compilation cost.

## Failures preserved during bring-up

An initial two-card attempt failed opening its mesh with `Physical chip id 0 not found in control plane chip mapping`, before accuracy measurements. Retrying on a verified connected pair produced the passing two-card component replay. An earlier one-card DiT test was deliberately aborted after an unfiltered encoder launch attempted the same reserved card; its empty result set is not counted as a pass. Early text encoder accuracy failures were superseded by the FP32 normalization version recorded here; their unsuccessful results must not be presented as passing current implementation evidence.

Current-main migration, fresh native build, latest CI registration, and any subsequent accuracy changes require their own dated result records. Hardware tests must fail on missing required assets, mismatch, nonfinite output, or failed metric bounds; host-only and asset-free checks cannot substitute for device qualification.
