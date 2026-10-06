# Recorded text inputs for public-contract checks

`batched.py`, `prefix_continuation.py`, and `request_reuse.py` accept
`--input-fixture`. They continue loading real checkpoint weights, and use the
same selected input tensors for HF and TT. Omitting the flag retains the
seeded Gaussian diagnostic mode. All existing PCC thresholds remain 0.995.

The shared CPU fixture loader checks model, revision, layer, recorded shape,
text provenance, finiteness, and exact BF16 transport rounding. Each result
records the fixture SHA-256, source provenance, selected rows, and position
policy. The optimized wrapper additionally records runtime SHA-256, precision
policy, and any candidate overrides.

| Contract | Selection from fixture `prefill` | Preserved checks |
| --- | --- | --- |
| Batch 32, default length33 | Slot `s` uses rows `[34*s, 34*s+33)` for prefill and row `34*s+33` for decode; 1,088 recorded rows total | Distinct slot inputs, disjoint randomly permuted pages, every slot's prefill/decode PCC, traced decode, exact repeat equality |
| Batch 32, `--heterogeneous-positions` | Slot `s` has prompt length/decode position `32+s`; windows consume consecutive prompt-plus-decode spans, starting at `33*s+s*(s-1)/2`; 1,552 recorded rows total | Distinct positions32..63, aligned and nonaligned prompts, independent per-slot HF caches/references, the same paged/traced/repeat checks |
| Prefix continuation | First 65 rows, partitioned at the original `[0,31)`, `[31,33)`, `[33,65)` boundaries | Nonaligned page updates, slot 1 execution, unchanged cached prefix and other slot, whole-output PCC |
| Request reuse | Request `i` starts at recorded row `128*i`; the next row after its prompt supplies decode | Original lengths `31,32,33,1023,1024,1025,2049,33,2047`, changing physical page ownership, one loaded layer/cache, one reused decode trace, prefill/decode PCC for every request |

Batch and request windows are rebased to positions starting at zero identically
for HF and TT. These are module contract checks using recorded text-derived
layer-boundary activations. They do not recompute preceding layers for each
sliced prompt. The two length-33 reuse requests receive different recorded
windows. Prefix continuation retains the original first-65 positions.

The optional heterogeneous batch mode leaves the default33-token behavior
unchanged. HF and TT receive each slot's valid prompt length; zero padding in
the host staging tensor is never included in that prompt. Both device position
buffers contain the per-slot length list before trace capture. Results record
`heterogeneous_positions`, `lengths`, each prefill row's `length` and each decode
row's `position`; the top-level `length` is a list in heterogeneous mode and
remains33 in the default mode.

Run from the repository root, with hardware access coordinated separately.
These examples exercise the runtime defaults. To validate a candidate before
promoting its defaults, append the same `--default-overrides '<JSON>'` used for
that candidate's headline and stress checks.

```bash
python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_optimized_contract \
  --contract batched --batch 32 --layer 0 \
  --input-fixture models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_text_layer0_4096_128.pt \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_batched_layer0.json

python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_optimized_contract \
  --contract batched --batch 32 --layer 5 \
  --input-fixture models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_text_layer5_4096_128.pt \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_batched_layer5.json

python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_optimized_contract \
  --contract batched --batch 32 --heterogeneous-positions --layer 0 \
  --input-fixture models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_text_layer0_4096_128.pt \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_batched_heterogeneous_layer0.json

python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_optimized_contract \
  --contract batched --batch 32 --heterogeneous-positions --layer 5 \
  --input-fixture models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_text_layer5_4096_128.pt \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_batched_heterogeneous_layer5.json

python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_optimized_contract \
  --contract prefix_continuation --layer 0 \
  --input-fixture models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_text_layer0_4096_128.pt \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_prefix_layer0.json

python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_optimized_contract \
  --contract prefix_continuation --layer 5 \
  --input-fixture models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_text_layer5_4096_128.pt \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_prefix_layer5.json

python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_optimized_contract \
  --contract request_reuse --layer 0 \
  --input-fixture models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_text_layer0_4096_128.pt \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_reuse_layer0.json

python_env/bin/python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_optimized_contract \
  --contract request_reuse --layer 5 \
  --input-fixture models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_text_layer5_4096_128.pt \
  --output models/autoports/google_gemma_4_26b_a4b_it/doc/optimized_decoder/actual_reuse_layer5.json
```

The prefix runner's cache allocator now reads `decoder.kv_cache_dtype`; its
previous reference to the undefined name `layer` prevented that runner from
reaching the contract checks.

Validation for this test-only change: Python syntax compilation and Black;
CPU-only execution of the new input-selection branches against both recorded
fixtures, including all 32 slots and all nine requests, exact source-row
equality, distinct repeated-length requests, and wrong-layer rejection. No
device execution or performance claim is part of this change.

`heterogeneous_batch_cpu_validation.json` additionally records CPU checks of
both recorded fixtures at B32/positions32..63, B1/position32, unchanged B32
constant-position fixture selection and unchanged seeded Gaussian inputs.
Per-slot HF call wiring, causal masks, cache continuity, decode-row selection
and both position buffers were checked with a lightweight oracle stub. These
checks neither execute real-weight HF forwards nor claim device correctness.
