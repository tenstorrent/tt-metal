# PLE34 precision boundary diagnosis

Read-only AutoFix hypothesis investigation, 2026-11-13. No target forwards, TTNN import, hardware access, tensor experiments, implementation edits, or gate changes were performed. This document is the only written artifact. Paths below are relative to the model directory unless explicitly absolute.

## Finding

There is no supported fused-arithmetic repair for recovering the exact checkpoint lookup from the existing packed input. The dominant discrepancy already exists in CPU HF when only that lookup is replaced by the actual cache bytes. The supplied `PackedEmbedding` object contains quantized codes and scales, not the discarded checkpoint residuals. Widening those values or changing later arithmetic cannot reconstruct information absent from that input.

This is **not a proof that the unchanged 0.995 gate is unattainable at the existing storage budget**. A new independently calibrated, same-format packed input is a legitimate, capacity-neutral candidate through the existing external input contract. Its success is unproved. The smallest credible candidate changes only the global 16-level weight codebook used during offline packing, optimizing the actual group10/scale8 reconstruction objective; it requires no edit to functional source and no new lookup representation. A complete new cache and all existing acceptance gates would still be required if its CPU screen passes.

If “fused-decoder only” additionally requires preserving the exact external cache identity, that candidate is outside the available boundary. Under that narrower interpretation, no demonstrated repair remains in `fused_decoder.py`; the stage stays failed, rather than waived.

## Evidence tying the failure to the external packed input

`AUTODEBUG.md`, `AUTOFIX.md`, and `doc/fused_decoder/ple34_packed_backend_v1.json` agree:

- Literal HF versus fused row5 whole PCC is 0.9944641348820654 and contribution PCC is 0.9936444270418033. The functional baseline fails almost identically. Writer-only, tail-only, prior v5, and combined fused outputs are byte-identical on complete prefill and all three decode outputs.
- Replacing only CPU HF lookup with cache-decoded BF16 values gives row5 whole 0.9946457606798581 and contribution 0.993853134747022. Twelve cache parity checks pass; the literal HF repeat is exact.
- Fused versus packed-HF passes every row, minimum whole 0.999462011875479 and contribution 0.9994260598093204. Row5 agreement itself is 0.9999013857432748 / 0.9998866820627059.
- Six FP64 projection controls retain the packed row5 failure (whole 0.9946348156447864). The literal router set contains expert88 where packed uses expert325. This is consistent with upstream quantization changing a near-cutoff route; it does not establish an incorrect router implementation.

The metrics do not prove all device arithmetic is exact. They establish that fixing only writer/tail arithmetic, or merely eliminating the observed device-versus-packed residual, does not address the CPU packed-only failure. None of these comparisons replaces literal HF acceptance.

## Actual representation and input boundary

`tt/functional_decoder.py:58-82` receives `packed_embedding` externally, requires its full 320,001,536-row domain for PLE, and retains it as `result.embedding`. `FusedDecoder.from_state_dict` delegates this setup through `**kwargs`; its default `fuse_ple=False` preserves the ordinary PLE component. The fused file does not receive exact embedding rows through this argument. The real probe constructs `PackedEmbedding.from_checkpoint(...)` before constructing the decoder.

`tt/packed_embedding_cpu.py:32-76` already permits arbitrary strictly increasing finite FLOAT32 16-level weight codebooks. For group10 it requires a 256-level, nonnegative, strictly increasing, BF16-exact scale LUT. `PackedEmbedding.from_checkpoint` reads these immutable values from the cache manifest and rejects inconsistent explicit overrides. A separately generated cache with new valid levels is therefore supported by existing factories; silently replacing a LUT while preserving old packed bytes is not an equivalent cache.

The active manifest at `/home/ttuser/qwen38-flash/packed_ple_lloyd_lsq_group10_scale8_wide/manifest.json` declares:

- Four-bit weights with a **learned symmetric Lloyd codebook**, not the legacy NF4 constants. The learned codebook has no exact zero level; zero groups can use zero scale.
- Five group-local least-squares iterations, group size10, sixteen scale bytes per160-value row; weight-code hash `59c463a9a75bec1e953be5a908eec43d0d0b8ff1cb68d3fd8238bbebaf41e68f` and scale-LUT hash `180b4e3289539394e3ece69d1268ec7e30a7c920c3055bfad89fa70253b5dbf6`.
- Twenty UINT32 weight words plus four UINT32 scale words per row: 80+16=96 bytes. Full streams total **30,720,147,456 bytes**.

The quantizer fits scales, rounds them to the nearest fixed LUT entry, then selects nearest weight levels using those reconstructed scales. The lookup multiplies FLOAT32 codebook values by BF16-exact scales in FLOAT32 and rounds the final product to BF16 (`tt/packed_embedding.py:251-282`). Raising the codebook or scale lookup precision cannot restore lost four-bit residuals: the weight codebook is already FLOAT32, and the scale table values are already exactly represented.

The original calibration uses8192 checkpoint rows,64 per shard, seed43191, disjoint from the long fixture and original short holdouts. `fit_codebook` in `tests/probe_ple_decoder_quantization.py:86-107` fits Lloyd centroids to **absmax-normalized** magnitudes. Production subsequently uses five-step LSQ scales and a quantized scale LUT. Thus the existing codebook is not proven optimal for the actual deployed reconstruction objective. This is a concrete opportunity for a general calibration experiment, not evidence that a different codebook passes PLE34.

`doc/functional_decoder/ple_scale8_cpu_report.md` and the active manifest rule out ordinary scale clipping as the obvious repair: the full-cache verification attributes all1440 below-range groups to90 padding rows beyond the valid hash domain; no valid-domain groups clip either end. The wide LUT spans0.0003147125244140625 through0.1689453125, while the full-table maximum fitted scale is0.08966350555419922. Merely expanding that LUT further spends more resolution without resolving known clipping. The documented uniform-scale alternative has worse independent training MSE (2.40541027e-7 versus2.34911795e-7); that alone says nothing decisive about decoder row gates.

## Supported memory limits versus unproved alternatives

The empty-device evidence reports34,178,731,008 allocatable DRAM bytes across8 banks. The older resident group10 model has541,166,080 bytes free (`doc/functional_decoder/ple_capacity_group10.json`). More relevant to current fused preparation, `doc/fused_decoder/integrated_writer_ple4768_v1.json` reports514,214,400 bytes before preparation and402,058,752 after preparation, before the full public input/output allocation. These are recorded measurements, not fresh measurements here.

| Representation | Embedding stream bytes | Consequence |
|---|---:|---|
| Current four-bit/group10/scale8 | 30,720,147,456 | Existing resident input |
| Five-bit weights, same scale bytes | 37,120,178,176 | Exceeds empty-device capacity before any decoder weights |
| Four-bit/group10/BF16 scales | 35,840,172,032 | Exceeds empty-device capacity before any decoder weights |
| Four-bit/group16/BF16 scales | 32,000,153,600 | Adds1,280,006,144 bytes, exceeding measured current free space before public I/O |
| Exact BF16 table | 102,400,491,520 | Exceeds single-device capacity by itself |

Group16/BF16 is factory-supported but is not a drop-in capacity-preserving upgrade for the measured model. A speculative group8/scale8 layout also adds1.28GB and is not supported by the existing format validator. Full eight-bit weights are still larger. Adding one uniform residual bit per checkpoint weight costs6,400,030,720 bytes before indexing or alignment.

These byte counts support rejecting those particular full-domain upgrades under the current resident contract. They **do not** establish a universal compression lower bound for this checkpoint or prove that every sparse correction scheme is impossible. Sparse residuals, vector quantization, entropy coding, new group layouts, or jointly repacking other weights would require new evidence, representation/identity handling, random-lookup implementation, setup peak accounting, and capacity revalidation. None is an established small repair. A token-selected exact-row table, dynamic host gather, cache eviction, or fallback would violate the requested contract. A static weight-only correction scheme could be general in principle, but no measured residual distribution or memory bound currently supports one.

## Smallest legitimate candidate and one discriminating CPU check

Candidate: keep group10, the existing wide BF16-exact scale LUT, five LSQ iterations, four-bit indices, stripe layout, and complete row domain. Refit only the eight positive symmetric weight levels against the actual production reconstruction MSE on a fixed independent checkpoint calibration set, then mirror them to make16 strictly increasing FLOAT32 levels. The existing codebook is the initial candidate. A deterministic bounded coordinate search or equivalent calibration procedure may evaluate proposals using the existing CPU quantizer; the objective includes LSQ, scale-byte quantization, and final BF16 rounding. Training must use checkpoint weights only, never router membership, activations, failed row IDs, HF outputs, or device errors. Freeze the candidate and its hash before inspecting decoder holdout results. New calibration rows must be disjoint from every evaluated hash ID, including PLE34, not merely the older holdouts.

This is a candidate external input change, not an in-place fused patch. It remains fully compatible with `quantization_spec`, `_quantize_rows`, and `PackedEmbedding.from_checkpoint`; it needs no modification of the functional runtime or cache format. It requires a new persistent cache directory and full-domain re-encoding. It must replace the resident table at setup, not coexist with another full table. Retuning just the decoder-side LUT against old bytes would bypass these guarantees and is not recommended.

**One discriminating CPU experiment, not run here:** in a test-only extension of `probe_ple_prefill_backend`, use the exact saved PLE34 prompt/token IDs and literal expected tensors; verify the old actual-cache control still reproduces its recorded failure. Then replace only lookup with the frozen candidate reconstructed by the unchanged production `_quantize_rows`, run the complete CPU decoder with all other weights/arithmetic unchanged, and report FP64 whole/contribution PCC for every34rows plus the three decode steps. Include the original33-token and independent65-token fixtures as untouched holdouts in the same experiment. Require the literal rerun to remain exact and the training/evaluation ID intersection to be empty. Log router sets diagnostically, without forcing routes or using them to select a codebook.

A failure rejects this particular codebook candidate. A pass would justify the cost of preparing the full new cache; it would **not** certify device correctness or all-context accuracy. Full-cache byte/manifest checks and actual TT validation must then preserve public4768/4767, PLEbatch5,262144 context, trace/reset/replay behavior, and every original0.995 gate. Same byte count reduces the capacity risk but does not replace those checks. A lower calibration MSE alone cannot justify adoption because route changes make decoder PCC nonmonotonic.

## Disposition

Do not edit the new writer/tail to compensate for PLE34. Keep its literal failure visible. A supported information boundary exists for the current fixed external cache, and specific higher-bit resident tables are physically excluded. A capacity-neutral recalibration is legal through the existing external input interface but remains an untested hypothesis, so neither a universal blocker nor a completed repair can honestly be claimed.
