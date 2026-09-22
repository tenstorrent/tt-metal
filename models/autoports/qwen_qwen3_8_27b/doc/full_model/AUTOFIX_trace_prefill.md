# AutoFix: prefill programs allocated behind live decode traces

## Starting evidence

- Diagnosis: `AUTODEBUG_trace.md`, findings 1 and 7.
- Coordinator's experiment: `tests/full_model_shapes.py` with `TT_METAL_TRACE_ALLOC_TRACKING=1`, using one persistent maximum-context cache.
- Evidence: `shapes.log:37-52` passes S1 and rejects S31 before decode replay with 63 still-live device buffers.
- Host-only log parsing confirms all 63 have `program_cache:` allocation contexts, including 14 slice programs. The final-token boundary and convolution-tail slices explicitly depend on logical length 31.

## Hypothesis experiment

**Hypothesis:** A new logical prefill signature compiles persistent program resources while the prior model/sampling traces remain live. Those new resources must be allocated before the next trace pair is captured.

**Prediction:** Same-signature repeat can pass while the first new logical length fails allocation tracking before replay, even after consumed prefill logits are released.

**Result:** Matches `shapes.log`: S1 passes, S31 fails with retained program-cache buffers. No unsafe replay was executed after the tracker rejected it.

**Verdict:** Verified mechanism. Source inspection locates the program-cache allocation contexts in `ttnn/api/ttnn/device_operation.hpp:367-385,428-438`.

## Minimal proposed repair

Track successful whole-request prefill signatures `(batch, page_table_shape, slot, absolute_start, logical_length, all_logits)` within the current cache binding. Release both decode traces before any device work for an unknown signature, mark the signature after the full prefill including LM head and slot writeback succeeds, and recapture the pair at the next decode. Known prefills and steady decode reuse traces. Clear the ledger on cache replacement/binding; enforce canonical bound cache/table specifications or include their specifications in the key.

The coordinator implemented this lifecycle while the diagnosis agent reviewed source. This agent changed documentation only. The fix does not skip the tracker, mark program-cache buffers corruptible, discard program caches, or precompile every advertised context length.

## Verification

- `shapes_final.json` passes allocation-tracked S1,31,32,33,4095,4096,4097 followed by known repeats S33,31,4097, all under the same cache binding. The executable assertions verify no additional captures on known repeats and exact position advancement.
- `contract_b32_final.json` passes all-logits mode, continuation31+2 beginning inside a page, and fixed-slot B32 prefill/writeback. B3 coverage is retained separately.
- `watcher_fixed.json` passes the stronger same-input physical-page-permutation logits oracle, unchanged/changed table counters, and the same continuation at PCC .9999673. All watcher checks remain enabled and the process exits0.

The source-only diagnosis agent ran no hardware. The coordinator ran the above verification. The prefill trace-lifecycle repair is verified; overall stage completion still requires the separate fullmodel gates and review.
