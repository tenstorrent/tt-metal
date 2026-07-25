# AutoDebug: GPT-OSS-20B layer-12 decode at position 256

## Verdict

The highest-confidence root cause is a TTNN rotary-embedding program-cache
defect exposed by making the emitted decode graph position-generic.
`FunctionalDecoder._decode_attention()` calls
`ttnn.experimental.rotary_embedding(..., cache_position)` for both Q and K
(`functional_decoder.py:388-393`). The rotary operation's program hash omits
`token_idx` (`rotary_embedding_device_operation.cpp:146-150`), even though the
program factory turns it into two scalar runtime arguments:

```text
cos_sin_offset   = token_idx % 32 * Wbytes
cos_sin_start_id = token_idx / 32 * Wt
```

Those values are installed in reader/writer runtime arguments
(`rotary_embedding_program_factory.cpp:385-420` and `:815-857`). The
operation uses `Buffer*` runtime arguments too, so the
`ProgramDescriptor` cache-hit fast path only patches buffer addresses
(`mesh_device_operation_adapter.hpp:584-600`). Rotary embedding declares no
`get_dynamic_runtime_args`, so the non-buffer token-position scalars remain
frozen at the first same-shape decode invocation.

The observed order is the signature this defect predicts:

1. Position 17 is the first decode and compiles Q and K rotary programs with
   row 17; PCC is 0.999317.
2. The position-256 decode has the same Q/K tensor specifications, hits those
   programs, and still selects rotary row 17.
3. For head dimension 64 (`Wt=2`, BF16 `Wbytes=128`), the correct runtime
   values change from `(cos_sin_start_id=0, cos_sin_offset=2176)` at position
   17 to `(16, 0)` at position 256. Neither is refreshed on the cache hit.
4. Q is therefore rotated at the wrong absolute position. K written to cache
   slot 256 is expected to be wrong for the same reason. Correctly prefetched
   keys 129..255 retain their absolute rotations, so the relative Q/K phases
   consumed by attention are inconsistent.

This is a concrete source-level correctness bug, not merely a speculative
model mismatch. Its causal connection to the reported PCC still needs the
single-op isolation below, because this investigation was explicitly
non-hardware.

The newly reported diagnostics strengthen this ranking:

- The key/value-cache PCC above 0.99994 is measured after prefill and compares
  TT positions 129:256, i.e. absolute positions 129..255. It does **not**
  inspect the newly rotary-transformed key written at absolute position 256,
  so it does not test or refute this defect.
- Replacing causal/sliding-window mode with an explicit full-cache mask still
  leaves position-256 post-attention PCC at 0.583.
- Removing the attention sink from both paths also leaves the failure. These
  results move native sliding-window masking and sink reduction below the
  upstream Q/K transformation.

## Evidence-ranked hypotheses

### 1. Stale rotary `token_idx` on program-cache hits — very high confidence

Direct evidence:

- The model passes the host scalar `cache_position` to Q and K rotary calls.
- `RotaryEmbeddingParams` contains `token_idx`
  (`rotary_embedding_device_operation_types.hpp:12-17`).
- `compute_program_hash()` hashes `seq_len`, output memory configuration, and
  the three tensors, but not `token_idx`.
- Both rotary program-factory branches derive reader/writer scalar runtime
  arguments from `token_idx`.
- The descriptor contains buffer bindings, selecting the cache-hit fast path.
  The adapter documentation explicitly says values omitted from a hash remain
  frozen unless dynamic runtime arguments are declared.
- `RotaryEmbeddingDeviceOperation` declares no dynamic-runtime-argument
  callback (`rotary_embedding_device_operation.hpp:13-27`).
- The first decode at 17 passes and a later same-shape decode at 256 fails,
  exactly matching a first-program-wins defect.

The emitted `g1_decode/main.py` does not exercise this case. It passes
already-selected cos/sin rows and a constant rotary token index of zero
(for example, lines 1268 and 1285). The functional translation switched to
full cos/sin caches plus a varying host token index, exposing the cache-key
hole.

Existing rotary tests do not cover this lifecycle. The decode test
parametrizes `token_idx` across separate pytest invocations
(`test_rotary_embedding.py:111-192`) and calls the operation only once per
case. It never calls the same-shaped program twice with two different token
indices while the program remains cached.

### 2. SDPA explicit attention-mask path at non-power-of-two cache length — low residual confidence

This remains worth isolating only if the rotary experiment is negative.
The explicit-mask result shows the failure persists outside native sliding
window calculation, but it still uses the same SDPA kernel with cache length
288 and the same attention-mask reader path. A standalone Q/K/V SDPA
comparison can decide it.

The straightforward native-window off-by-one is contradicted statically:
cache sequence 288 chooses a 32-token chunk
(`sdpa_decode.cpp:22-35,59-60`). At `cur_pos=256`, `window=128`,
`get_workload_for_core()` computes:

```text
exclusive end       257
unaligned start     129
aligned start       128
aligned end         288
processed interval  128..287
```

The first-chunk sliding mask removes position 128
(`rt_args_common.hpp:46-70`;
`dataflow_common.hpp:307-345`) and the causal mask removes 257..287, leaving
exactly the HF eligible positions 129..256. Moreover, the explicit mask
experiment bypasses that native window construction and still fails.

The SDPA current-position/address cache theory is also contradicted by the
implementation. Its Q/K/V/position/mask/sink addresses are deliberately
inserted as raw `uint32_t` runtime arguments
(`sdpa_decode_program_factory.cpp:903-915`), forcing the adapter's descriptor
rebuild path on a cache hit rather than leaving the position address stale.

### 3. Attention-sink reduction or scaling — very low confidence

The sink is applied after the chunk reduction in the decode kernel, so a
per-chunk duplication explanation was already weak. More decisively, the
reported sink-disabled A/B still fails. Do not change sink scaling or layout
as the next fix.

### 4. Cache fill/update index or HF retained-cache interpretation — very low confidence

Prefill cache PCC validates absolute positions 129..255, and the update call
uses the device position tensor to write slot 256. An update-index bug would
not explain a wrong Q, and changing the attention mask does not repair the
failure. The one remaining cache check of value is the decoded K row at slot
256, because the current cache comparison occurs before that update.

## Exact isolated experiments

Run these in order. The first experiment is the shortest decisive test.

### Experiment A: same-program rotary token-index reversal

Use one open device with program caching enabled and exact decode-like BF16
tensors: cos/sin `[1,1,288,64]`, a height-sharded tiled input of
`[1,1,64,64]` for Q, and output
`L1_HEIGHT_SHARDED_MEMORY_CONFIG`.

1. Call rotary at `token_idx=17`; compare to Torch row 17.
2. Without clearing the program cache, call the same operation/specification
   at `token_idx=256`; compare that output to both Torch row 256 and Torch row
   17.
3. Repeat after reopening/clearing the cache, but call 256 first and 17 second.
4. Repeat for the exact K shape `[1,1,8,64]`.

Predicted result if the diagnosis is correct:

- The first call in each sequence passes.
- The second call matches the **first call's token index**, not its own.
- Clearing/forcing a program miss before each call makes both positions pass.
- Program-cache entry count does not increase when only `token_idx` changes.

This test should compare tensors directly, not only an end-to-end PCC.

### Experiment B: decoder Q/K checkpoint

In the boundary test, retain device tensors immediately after each rotary call
and before `paged_update_cache`.

1. Compute Torch Q and K after input norm, projection, head split, and rotary.
2. Compare TT Q/K at position 17 to Torch row 17.
3. Compare TT Q/K at position 256 to both Torch row 256 and a deliberately
   row-17-rotated Torch tensor.
4. After the update, compare only TT cache row 256 against the decoded HF K.

Predicted result: position-256 TT Q and K agree with the row-17 transform.
This also explains why the existing prefill-cache PCC passes: it checks only
the rows created by the separate prefill rotary path.

### Experiment C: invocation-order test

On a fresh model/device program cache, run the boundary position-256 decode
before the position-17 decode, or make position 256 the only decode.

Predicted result: position 256 passes and the subsequent position 17 fails.
This is not a production workaround; it is an inexpensive cache-key
fingerprint.

### Experiment D: standalone SDPA fallback discriminator

Only if A-C disprove the rotary diagnosis, feed known Torch-generated,
correctly rotated Q/K/V directly to decode SDPA with the exact shapes
`Q=[1,64,1,64]`, `K/V=[1,8,288,64]`.

Compare:

1. `is_causal=False` with an explicit mask allowing exactly 129..256;
2. `is_causal=True`, `cur_pos=256`, `sliding_window_size=128`;
3. the same two cases with cache length padded to 512.

If only the 288-length cases fail, inspect the dynamic chunk/mask path. If
explicit-mask 288 passes with isolated correct Q/K/V, SDPA is exonerated and
the defect is upstream.

## Fixes to test in isolation

### Minimal correctness fix

Include `args.token_idx` in
`RotaryEmbeddingDeviceOperation::compute_program_hash()`. Also include
`args.compute_kernel_config`, which currently affects compilation but is
omitted from the same custom hash.

This is the smallest decisive patch and the best first A/B. It will create a
program per token index, so it is correct but undesirable for long-running
decode and tracing.

### Preferred production fix

Keep token index out of the program hash and implement rotary dynamic runtime
arguments. On every cache hit, recompute and patch, for every active core:

```text
reader cos_sin_start_id = token_idx / 32 * Wt
writer cos_sin_offset   = token_idx % 32 * Wbytes
```

Use the operation framework's `get_dynamic_runtime_args` mechanism so buffer
bindings and the fast cache-hit path remain intact. `compute_kernel_config`
must still be added to the hash because it changes compiled kernels rather
than runtime scalars.

### Model-local workaround

Preselect the one-position cos/sin tensors and call rotary with constant
`token_idx=0`, matching the emitted decode graph. Validate the resulting
slice/layout and program-cache behavior with the Q/K checkpoint before using
this as a temporary model fix. Clearing the program cache or hashing every
position is suitable for diagnosis, not serving.

## Required regression coverage

Add one test that performs two same-shape rotary decode calls with different
token indices in a single program-cache lifetime. Cover:

- forward order 17 then 256 and reverse order 256 then 17;
- exact head dimension 64 (`Wt=2`);
- height-sharded input/output matching GPT-OSS Q and K;
- BF16 cos/sin cache with a non-power-of-two logical cache length such as 288;
- direct Torch comparison of both outputs;
- for the preferred dynamic fix, stable program-cache entry count across
  positions.

Then rerun the real-weight boundary test with checkpoints for post-rotary Q,
decoded cache row 256, and post-attention output. Remove diagnostic sink
suppression and explicit-mask overrides once native
`is_causal=True, cur_pos=256, sliding_window_size=128` passes.

## Investigation limits

No Tenstorrent hardware command was run and no implementation file was
modified by this diagnostic task. The `$autodebug` launcher was attempted in
a fresh context, but its nested sandbox could not read the workspace in this
environment; the alternate backend was not authenticated. The source audit
was therefore completed in the already-fresh diagnostic agent and is left as
an evidence-ranked report rather than claiming hardware confirmation.
