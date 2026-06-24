# moe_compute + selective_reduce_combine deterministic deadlock

**Status:** root-caused to a reproducible trigger; two distinct defects identified.
**Severity:** hard multi-device hang (requires `tt-smi -r`), no error, no timeout.
**Assume the original reporter cannot reproduce** — this report gives a *deterministic* repro on
our cluster and isolates the mechanism so it can be understood and fixed from the code alone.

---

## 1. TL;DR

`ttnn.experimental.moe_compute` (fused with `selective_reduce_combine`) deadlocks on a (4,8) galaxy
mesh, `cluster_axis=0`, `FABRIC_1D_RING`. The deadlock is **deterministically triggered by freeing and
recreating the dispatch input tensors (`x/idx/scr`) before the op** — it has nothing to do with buffer
addresses, input values, op timing, or mux geometry.

The combine writer enters an **infinite token-search loop** at
`selective_reduce_combine/.../writer.cpp` (was line 271): for each expert it reads
`token_split_counts[e]` entries from `dense_token_maps` and searches the activations stream for each
token id. `dense_token_maps` is moe_compute's `-1`-padded expert→token (`e_t`) buffer, and
`token_split_counts` is the per-expert token count — both produced by moe_compute's tilize stage from
the dispatched data. When the inputs are recreated, the tilize **count over-runs the valid map
entries**, so the combine reads a sentinel `0xFFFFFFFF` (= -1, "no more tokens") — or a token whose
activation never arrived — as a real token id and searches forever. The loop's only exit was an
`ASSERT` that is **compiled out in release**, so the core never reaches the cross-device ring barrier
and the whole mesh deadlocks.

**This is now confirmed in-kernel** (DPRINT, dev 24, expert 0): the combine searched for
`missing_token_id=4294967295` (the -1 sentinel) and `missing_token_id=33` (a real but absent token).

Two separable bugs:
- **Bug A (robustness — FIXED + VALIDATED in this branch):** the bounded-search fix at the combine
  token search converts the unrecoverable silent hang into a logged, recoverable completion. With the
  fix, the previously-100%-hang repro now reports `SMOKE PASSED` and logs the desync. (Output is still
  numerically wrong for the dropped tokens — that needs Bug B.)
- **Bug B (root cause — still open):** moe_compute's tilize stage produces a `dense_token_counts` /
  `dense_token_maps` / activations triple that is internally inconsistent (count > valid map entries,
  and a counted token with no activation) when the dispatch inputs are recreated — ultimately because
  `all_to_all_dispatch_metadata` delivers incomplete data across the ring in that state. Inputs are
  allocator-state sensitive even though their **addresses and data are byte-identical** to a passing
  run.

---

## 2. Deterministic repro

Environment: `moe_compute_combine_deadlock_repro/moe_compute_smoke.py`, run in the dev container
(`tt-xla-ird-mvasiljev`), `source .env.sh` first. Reset between runs with `tt-smi -r` on the host and
allow a settle (~25–30 s) before reopening the device.

```bash
# PASS (baseline single-shot — the original repro path):
SMOKE_REALLOC=0       python3 moe_compute_smoke.py     # -> SMOKE PASSED

# HANG (free + recreate the dispatch inputs before the op):
SMOKE_REALLOC=inputs  python3 moe_compute_smoke.py     # -> hangs forever in moe_compute
```

The only difference between the two is the block guarded by `SMOKE_REALLOC` in `moe_compute_smoke.py`:
it deallocates `x/idx/scr` and reallocates them (re-upload from host). `SMOKE_REALLOC=inputs_clone`
(on-device `ttnn.clone`, no host re-upload) hangs identically; `SMOKE_REALLOC=prealloc` (recreate the
dispatch *output* buffers instead) **passes**.

Observed verdicts (each `tt-smi -r` + settle between runs):

| mode | what is recreated before the op | result |
|------|----------------------------------|--------|
| `0` (baseline) | nothing | **PASS** 3/3 |
| `inputs` | `x/idx/scr` freed + re-uploaded | **HANG** 3/3 |
| `inputs_clone` | `x/idx/scr` via on-device clone | **HANG** |
| `prealloc` | dispatch output buffers | **PASS** |
| `inputs` + `SMOKE_REALLOC_SYNC=1` | `x/idx/scr` + `synchronize_device` after | **HANG** 3/3 |

---

## 3. The hang (triaged signature)

From live `tt-triage` against the wedged runtime (BEFORE killing anything):
- Hung op: `MoEComputeDeviceOperation`, devices 16, 20, 24, 28.
- Stuck kernels (callstacks): `logs/livehang_triage_callstacks_20260624_172714.txt`.

Three interlocked stall points:

1. **`selective_reduce_combine/.../kernels/dataflow/writer.cpp:271`** — combine writer busy-waits in
   the token search:
   ```cpp
   const uint32_t st = dense_token_maps_l1_ptr[ ... ];   // target token id (from dispatch metadata)
   uint32_t guard = 0;
   while (expert_token_activations_ptr[0] != st) {        // line 271 — never matches
       expert_token_activations_ptr += activations_stride_elm;
       ASSERT(guard++ < global_num_tokens);               // line 273 — compiled out in release
   }
   ```
   `st` comes from the dispatch-produced `dense_token_maps`; the activations come from the matmul on
   the dispatched tokens. When the metadata is incomplete, `st` references a token that is not present
   in the activations → the loop never terminates.

2. **`moe_compute/.../kernels/dm1.cpp:359`** — the moe_compute data-mover waits for the combine to
   free its buffer (`noc_semaphore_wait(combine_semaphore_ptr, combine_semaphore_val)`); the combine is
   stuck at (1) and never signals.

3. **`selective_reduce_combine/.../writer.cpp:365`** — peer combine writers wait at the cross-device
   ring barrier for an atomic-increment that the stalled device(s) never send.

Causal chain: incomplete dispatch metadata → token search (1) spins forever → combine never reaches
barrier (3) and never frees its buffer → data-mover (2) deadlocks → whole-mesh hang.

---

## 4. What rules out the obvious red herrings

All captured on our cluster with `SMOKE_ADDRS=1` / `SMOKE_DUMP=1` (see `logs/` + INVESTIGATION_LOG.md).

- **Buffer address / memory map — NOT it.** `buffer_address()` of every tensor is byte-identical
  between a PASS run and a HANG run:
  ```
  MODE=0      (PASS): x=0x895180 idx=0x89a180 scr=0x89a1c0 sparse=0x886180 disp_idx=0x165c00 ...
  MODE=inputs (HANG): x=0x895180 idx=0x89a180 scr=0x89a1c0 sparse=0x886180 disp_idx=0x165c00 ...
  ```
  The allocator returns the freed slots to the exact same addresses. (And `inputs_clone`, which lands
  at *different* addresses, also hangs — so neither "same" nor "different" address is the trigger.)
- **Input data values — NOT it.** Same host tensors uploaded; identical bytes.
- **Write-before-read ordering / timing — NOT it.** `synchronize_device` immediately after the
  recreate (drains all pending work before the dispatch reads the inputs) does **not** fix it (HANG
  3/3). So it is not a drainable producer/consumer ordering race.
- **Mux core range geometry — NOT it.** Swept multiple `mux_core_range_set` geometries; no geometry
  flips the verdict (constraint: needs `num_links * neighbors` mux cores, else a clean `TT_FATAL`).
- **Core-placement counts — NOT it.** `moe_core_placement` core counts are identical PASS vs HANG.

What *is* the trigger: the **act of freeing + recreating the dispatch INPUT tensors** (by any method),
which perturbs allocator state. Recreating the dispatch *output* buffers does not trigger it.

---

## 5. Direct evidence of the cause (dispatch under-populates metadata)

`SMOKE_DUMP=1` reads `disp_idx` (the routing metadata that becomes `dense_token_maps`) back to host
right after the dispatch, before moe_compute:

```
disp_idx MODE=0      (PASS): sum=267801704  first16=[62,17,109,33,102,43,74,94, 16058,16083,15733,15995,48703,16074,48775,15888]
disp_idx MODE=inputs (HANG): sum=646176     first16=[106,71,44,20,23,156,56,7,  0,0,0,0,0,0,0,0]
```

Token 0 is populated in both, but in the HANG run the rest of `disp_idx` is **all zeros** — sum is
~400× smaller (sum is order-independent, so this is not a shard-reordering artifact). With identical
inputs and identical buffer addresses, `all_to_all_dispatch_metadata` **wrote far less routing metadata
in the HANG run**. That is exactly the condition under which the combine's token search at
`writer.cpp:271` looks for a token id that was never recorded.

Caveat: the host readback also touches uninitialized tile padding (`disp_scr` shows nan/garbage in both
runs), so treat the absolute values loosely; the reliable signal is the **populated-vs-zero structural
difference** and the **400× sum gap**.

---

## 6. The fix (Bug A) — implemented & validated

`ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/kernels/dataflow/writer.cpp`

The combine token search was changed from an unbounded loop whose only exit was a release-compiled-out
`ASSERT(guard++ < global_num_tokens)` to a **bounded** loop that, on overrun: logs the desynced
`(expert, dt, missing token id)` via `DPRINT`, fires `ASSERT(false)` (debug builds), and `break`s out
of the expert's token loop so the core still reaches the cross-device ring barrier instead of spinning
forever.

Validation (`SMOKE_REALLOC=inputs`, `TT_METAL_DPRINT_CORES=all TT_METAL_DPRINT_CHIPS=all`):
- Before fix: 100% HANG (machine wedged, needs `tt-smi -r`).
- After fix: **`SMOKE PASSED`** (op completes) and the kernel logs, on dev 24, expert 0:
  ```
  24:5-4:BR: selective_reduce_combine: token-map desync expert=0 dt=0 missing_token_id=4294967295 ...
  24:6-0:BR: selective_reduce_combine: token-map desync expert=0 dt=0 missing_token_id=33 ...
  24:6-4:BR: selective_reduce_combine: token-map desync expert=0 dt=5 missing_token_id=4294967295 ...
  ```
  `4294967295 = 0xFFFFFFFF = -1` is the e_t "no more tokens" sentinel; `33` is a real token whose
  activation was never delivered. This is direct, in-kernel proof of the count/map/activation desync.

NOTE: Bug A stops the hang but does NOT restore correctness — the dropped tokens make the output wrong.
A timeout / PCC check on the host would now catch the failure (instead of hanging the machine).

## 7. Bug B (root cause, still open) — where the desync is produced

The combine inputs are produced by moe_compute's **tilize stage**
(`moe_compute_program_factory.cpp`, `tensor_return_value`):
- `tensor_return_value[0]` → `tilize_per_expert_total_tokens_output_tensor` = `dense_token_counts`
  (the `token_split_counts[e]` the combine trusts).
- `tensor_return_value[2]` → `tilize_e_t_output_tensor` = `dense_token_maps`, the `[E,T]` expert→token
  buffer that is **`-1`-padded** ("capped by -1 to indicate no more tokens to send for this expert",
  `moe_compute_program_factory.cpp:571`).
- `tensor_return_value[1]` → `tilize_expert_activation_output_tensor` = the activations the combine
  searches.

The defect: when the dispatch inputs are recreated, the tilize stage emits a `(count, map,
activations)` triple where `count[e]` exceeds the number of valid (non-`-1`) map entries and/or counts
a token with no activation. Upstream, this is because `all_to_all_dispatch_metadata` delivers
incomplete tokens/metadata across the ring in that allocator state (input address+data identical to a
passing run, so it is NOT an address/data bug — see §4/§5).

Suggested fixes for the owners:
- Make the tilize stage guarantee `dense_token_counts[e]` == number of valid `dense_token_maps[e]`
  entries (and that every counted token has a delivered activation), or have the combine treat the
  `-1` sentinel as an end-of-list terminator rather than a token id to search for.
- Root-cause the dispatch's allocator-state sensitivity in
  `all_to_all_dispatch_metadata/device/all_to_all_dispatch_metadata_program_factory.cpp` +
  `kernels/dataflow/writer_all_to_all_dispatch_metadata.cpp` (the cross-ring metadata/token broadcast).

---

## 8. Artifacts (all under `moe_compute_combine_deadlock_repro/`)

- `moe_compute_smoke.py` — repro; knobs: `SMOKE_REALLOC`, `SMOKE_REALLOC_SYNC`, `SMOKE_ADDRS`, `SMOKE_DUMP`.
- `logs/fix_validate_20260624_202943.txt` — post-fix run: `SMOKE PASSED` + in-kernel DPRINT desync lines.
- `INVESTIGATION_LOG.md` — full timeline, every experiment, what was ruled out.
- `logs/livehang_triage_20260624_172714.txt` — hung-op + device signature.
- `logs/livehang_triage_callstacks_20260624_172714.txt` — full all-core callstacks (the deadlock chain).
- `logs/livehang_triage_callstacks_20260624_110018.txt` — earlier full callstack capture (same signature).
- `logs/ts_nosync_*`, `logs/ts_sync_*` — the sync A/B (ordering ruled out).
- Address + content captures recorded inline in `INVESTIGATION_LOG.md` (sections "Address capture",
  "Dispatch metadata content differs").

### Kernel source references
- `ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/kernels/dataflow/writer.cpp` (lines 271, 273, 365)
- `ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/kernels/dm1.cpp` (line 359)
- `ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_dispatch_metadata/device/all_to_all_dispatch_metadata_program_factory.cpp`
