# Long-context prefill and parallel migration

## Current scope

2K full-model prefill and its detailed KV checks are accepted and published.
The capacity implementation is published in commit `cee680d3`.

The user revised the plan on 17 September:
- Test 4K, 8K, 16K, 32K, 64K and 128K with the full 32-layer model.
- Confirm that each run finishes without errors.
- Measure warmed prompt wall time, throughput per user and each 1,024-token chunk.
- Show the next predicted word for two fixed book passages.
- Do not run golden KV comparisons at these larger lengths.
- Develop and test native migration on the prefill side in parallel, starting at 2K.
- Do not develop or test the decode side. Use passive buffers for native transport verification.

Use only the assigned Galaxies: **c04u14 for prefill validation/performance** and **b09u02 for native migration work**.
The old b07u08 allocation is released. The live page records current job IDs and lease times.
Keep SP=4, TP=8, two independent slots, BF16 weights/activations and BFP8_B cache.

## Milestones and present evidence

| Milestone | Current result | Next check |
|---|---|---|
| 2K full prefill and all-layer KV comparison | Passed and pushed; saved performance retained | No repeat for small or documentation-only changes |
| Capacity through 128K | Host geometry tests, 4K layer tests and 2K regressions pass; implementation pushed | Each requested length still needs live execution |
| 4K full model | Book benchmark passed: eight requests, 1,024 finite/repeated chip outputs, stable sources and clean close; median 3,869 / 3,857 tokens/s/user | Retain these measurements; advance to the next length |
| 8K | Book benchmark passed: eight requests, 2,048 finite/repeated chip outputs, stable sources and clean close; median 2,854 / 2,842 tokens/s/user | Retain these measurements; advance to 16K after resource review |
| 16K / 32K / 64K / 128K | Fixed book inputs ready; resource costs reviewed from source | Review the prior measured length and current memory/lease before each run |
| 2K migration runtime | Live H2D/readiness/table gate and local packed copy passed: four interleaved chunks, 128 acknowledgements, all 16 configs and 65,536 byte-identical destination pages | Build the native transfer dependencies, then test tt-d-gen transfer |
| Native KV transfer | Host contracts and local same-device packed copy pass; native-manager transfer remains untested | Build prerequisites, then two-host native transfer and exact destination-byte checks |
| Decode / SC4 handoff | Outside the current scope by user instruction | No decode preparation or tests |

The former larger-context golden-matrix plan is historical.
The remaining 4K held-out golden work was stopped at the user's request.
Its stopped run keeps its actual nonzero exit and is not called a complete matrix pass.
Completed baseline and boundary evidence is retained.

## What each performance number means

The measured path is embedding, all 32 decoder layers, final RMS normalization and the full vocabulary head.
It runs once per chunk.

| Measure | Meaning |
|---|---|
| Prompt wall | First token upload through the last completed model forward; includes upload, dispatch and synchronization |
| Throughput, tokens/s/user | Prompt token count divided by prompt wall, for one sequential user |
| Chunk forward wall | Synchronized full-model forward for one 1,024-token chunk |
| Chunk wall | Chunk upload, synchronization and forward together |
| Startup and warmup | Weight/cache loading and compilation, reported separately |
| Readback/check wall | Output transfer, finite/repeat checks, hashes and token ranking, outside prompt timing |

Run one warmup per slot, then three measured requests per slot.
The two slots run sequentially.
These are eager host-wall measurements, not concurrent serving or device-kernel timings.

## Book output check

Inputs are fixed excerpts from [Pride and Prejudice](https://www.gutenberg.org/ebooks/1342)
and [Great Expectations](https://www.gutenberg.org/ebooks/1400).
Each prompt has one BOS token and an exact number of book tokens.
It uses raw text continuation with the Instruct checkpoint.

Choose the endpoint before any model run.
Show the trailing prompt, predicted word, top five candidates and the book's actual next word.
A different but plausible continuation is an observation, not an automatic failure.
Finite and repeatable outputs remain required.
The test reads next-token logits; it does not append that token or run autoregressive decode.

## Native migration progression

1. Use the real H2D input service and the accepted 2K model.
2. Confirm device completion before layer-ready acknowledgements.
3. Export live source addresses and all physical device identities.
4. Prove that table reads match an independent live cache view; then read packed source pages.
5. Connect layer acknowledgements to the real native source client; transfer into passive destination allocations.
6. Compare every packed byte, across K/V, heads, layers, slots and positions.
7. Test continuation, slot isolation/reuse, failures, drain and source-buffer lifetime.
8. Qualify each requested capacity through 128K without larger-context golden KV comparisons.

The first live runtime test does not prove native byte transfer.
Native transfer does not by itself prove correct generated text.
The passive receiver runs no decoder model. Decoder compatibility and served output are outside this task.
No Llama-specific change to native manager copy semantics has been shown necessary.

## Resource checks

At 128K, two-slot KV uses 544 MiB per chip and retained logits use 1,002 MiB per chip.
These are source-derived payload counts, not measured peaks.
Eight full output checks process 250.5 GiB of cumulative host readback.
Current cache reordering also performs many small slices; timing must be measured at each length.

See the [detailed resource plan](/data/divanovic/llama31-8b-disagg/notes/task-9-long-context-execution-resource-plan.md) for formulas and source references.
Do not infer 128K execution or performance from capacity allocation or short-context results.

## Published measurement scope

See [prefill performance](docs/performance-prefill.md) for saved 2K results, verified 4K and 8K book measurements, all measured request/chunk samples and exact evidence hashes. Larger lengths remain pending; no full golden-matrix acceptance is implied.

The 2K runtime result uses a corrected direct-run lifecycle verifier. The original verifier expected a pytest-only log message. The correction checks the native 32-device fabric event, exact physical inventory and final close. The original result is preserved; no model rerun or native-manager pass is implied. See the [root verification](/data/divanovic/llama31-8b-disagg/evidence/task-10-native-migration/readiness-003-verifier-correction-001/root-verification.json).

The local copy gate verified all 65,536 BFP8 pages (285,212,672 bytes), including exponent data, in independent source and destination allocations. Source pages stayed unchanged. This uses TTNN copy, so it does not establish native-manager transport. See the [packed-copy root verification](/data/divanovic/llama31-8b-disagg/evidence/task-10-native-migration/packed-copy-launch-preparation-001/run/root-verification.json).

## Prefill migration acceptance plan

See [the required tests and completion criteria](docs/migration-prefill-tests.md). The guide maps GPT-OSS and shared prefill gates to Llama, distinguishes local copy from native transfer, and explains the native source registration and acknowledgement path. It includes current evidence and a two-prompt test story.
