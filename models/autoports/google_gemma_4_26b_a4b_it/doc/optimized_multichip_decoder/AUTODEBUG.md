# AutoDebug: decode-only packed dense BFP4 changes prefill PCC

## Scope and starting evidence

This is an inspection-only diagnosis. No implementation code was changed and no
TT hardware command was run. The repo-local AutoDebug runner was invoked against
this model, but its fresh Codex session could not execute or patch any file because
its nested sandbox failed during `bwrap` setup. The findings below were therefore
checked directly against the current checkout rather than copied from that failed
run.

The reported control uses BFP8 weights with
`GEMMA4_MULTICHIP_DRAM_SHARDED_ROLES=o_proj,packed_mlp_gate_up,mlp_down`.
The failing candidate additionally sets
`GEMMA4_MULTICHIP_DECODE_MLP_GATE_UP_WEIGHT_DTYPE=bfp4`.

The candidate artifacts record:

| Layer | Prefill PCC | Decode PCC | Threshold |
| --- | ---: | ---: | ---: |
| layer 0 sliding | 0.986111709 | 0.993460365 | 0.995 |
| layer 5 full | 0.992532577 | 0.994809513 | 0.995 |

These values establish the symptom, but not its cause. The PCC JSON files contain
no command, environment, checkout hash, or source hash. The timing JSON files in
the same candidate directory are not candidate provenance: their exact commands
name `test_functional_decoder.py::test_functional_decoder_perf_profile`, their
hash key is `functional_decoder_sha256`, and they do not record the BFP4 or
multichip environment. Do not use those timing files to infer the failing policy.

## Headline finding 1: ordinary `ttnn.typecast` source aliasing is contradicted by the implementation

`MultichipDecoder.from_state_dict` creates the prefill weight as
`decoder.packed_mlp_gate_up = ttnn.concat(...)`. For a decode dtype override it
then evaluates

```python
candidate_weight = ttnn.typecast(
    decoder.packed_mlp_gate_up,
    ttnn.bfloat4_b,
    memory_config=decoder.packed_mlp_gate_up.memory_config(),
)
```

and sends that result through `_dram_sharded_weight_and_config`.

The C++ implementation in
`ttnn/cpp/ttnn/operations/copy/typecast/typecast.cpp` calls
`ttnn::prim::typecast` without an optional output tensor. The device operation
therefore allocates and returns an output tensor; it is not an in-place cast of
the input. `_dram_sharded_weight_and_config` likewise returns
`ttnn.to_memory_config(weight, new_sharded_config)` and contains no explicit
deallocation or source mutation. The prefill branch in
`OptimizedDecoder._dense_mlp` selects `self.packed_mlp_gate_up` whenever
`_matrix_rows(x) > TILE_SIZE`; only the decode-sized branch selects
`self.decode_dram_weights["packed_mlp_gate_up"]`.

Consequently, a simple Python-object alias or intentional in-place typecast does
not complete the causal chain for degraded prefill. Treating "typecast mutates the
BFP8 source" as established would conflict with the checked code.

## Headline finding 2: the strongest remaining code-level hypothesis is temporary tensor lifetime / queued mesh conversion

The BFP4 override introduces a lifetime that the same-dtype control does not:

1. allocate an interleaved BFP4 mesh tensor with `typecast`;
2. enqueue/call `to_memory_config` to create the DRAM-sharded copy;
3. store only the sharded result;
4. let the local interleaved BFP4 `candidate_weight` lose its last Python
   reference at the next loop iteration or method return.

The original BFP8 packed tensor remains referenced, so this is not ordinary
source aliasing. However, an async command-queue lifetime/dependency defect at the
`typecast -> to_memory_config -> temporary destruction` boundary could corrupt
either allocation or conversion state, and allocator damage could affect later
prefill as well as decode. This hypothesis predicts:

- retaining the intermediate BFP4 tensor on the decoder, or synchronizing between
  typecast and sharding, restores prefill PCC without changing decode arithmetic;
- creating the BFP4 tensor directly in the final DRAM-sharded memory config also
  restores prefill if it removes the problematic temporary;
- host readback of the original BFP8 packed tensor differs only in the failing
  construction if corruption reaches the source/allocation.

This remains a hypothesis because proving it requires a device run. It is ranked
above numerical-instability explanations because a decode-only numerical change
cannot by itself alter the already-constructed prefill branch.

## Headline finding 3: candidate provenance is insufficient to exclude a harness/environment mix-up

The functional correctness harness constructs one decoder, runs prefill, then
decode, and writes PCC JSON without provenance. The stage artifact directory can
be redirected through `GEMMA4_MULTICHIP_ARTIFACT_DIR`, but the JSON does not record
that variable, the dtype override, DRAM roles, source hashes, or exact command.
The unrelated timing files make the directory look better-proven than it is.

Therefore a stale file, an inherited broader environment variable such as
`GEMMA4_MULTICHIP_MLP_WEIGHT_DTYPE=bfp4`, or a command that did not isolate the
decode override cannot currently be refuted. This hypothesis predicts that a
fresh minimal environment run either makes prefill match the BFP8 control or
produces a provenance-complete reproduction that clears this concern.

## Focused verify/refute experiments

Run each as an isolated A/B. Do not batch fixes.

### Experiment A: prove branch and tensor identities/dtypes

Add temporary diagnostics immediately after construction and at the packed MLP
selection point. Record, for each device tensor, logical/padded shape, dtype,
layout, memory config, buffer address, and whether `_matrix_rows(x) <= 32`.
Assert that prefill selects `decoder.packed_mlp_gate_up` with BFP8 and decode
selects the DRAM-sharded BFP4 tensor. Record the complete relevant environment.

Verdict rules:

- prefill selects BFP4: selection/configuration bug verified;
- prefill selects BFP8 but its address/dtype changes after candidate construction:
  allocation/lifetime corruption strongly verified;
- branch and metadata remain correct: continue to B/C; metadata alone does not
  prove tensor contents.

### Experiment B: source-content checksum/readback around each construction step

On a fresh decoder, read back each device shard of
`decoder.packed_mlp_gate_up` at four boundaries: after concat, after typecast,
after DRAM sharding, and after device synchronization. Compare bitwise and by PCC
to the after-concat BFP8 snapshot. Separately read back the BFP4 intermediate and
final sharded tensor and compare both to a host BFP4 quantization oracle.

This directly distinguishes source corruption, bad BFP4 conversion, bad sharding,
and a clean construction. Use the exact production mesh mapper and shapes; a
single-device or near-shape probe is not proof.

### Experiment C: lifetime controls

Repeat the failing correctness command with only one change at a time:

1. keep `candidate_weight` in a persistent decoder dictionary;
2. call `ttnn.synchronize_device(mesh_device)` between typecast and sharding;
3. typecast directly to the final DRAM-sharded memory config and skip the second
   conversion;
4. create the decode BFP4 packed weight independently from the host gate/up
   tensors, rather than deriving it from the prefill device tensor.

If (1) or (2) fixes prefill, the queued temporary-lifetime boundary is verified.
If only (3) fixes it, the two-step mesh conversion is implicated. If only (4)
fixes it, deriving two precision copies from shared device state is implicated.
If none fixes prefill, refute the lifetime hypothesis and inspect the harness and
other environment variables before changing kernels.

### Experiment D: fresh provenance-complete reproduction

Start from `env -i` plus only required runtime variables and the three control
roles. Run both layer kinds once with BFP8 decode gate/up, then add only
`GEMMA4_MULTICHIP_DECODE_MLP_GATE_UP_WEIGHT_DTYPE=bfp4`. Write to new directories.
Capture exact command, all `GEMMA4_*` variables, git SHA, source hashes, TTNN
extension hash, device IDs, and per-branch runtime dtype/memory metadata in both
PCC artifacts. Do not reuse the current timing JSON files.

### Experiment E: split prefill and decode construction/order

Run prefill-only on a newly constructed failing-policy decoder, then run
decode-only on another newly constructed decoder. Reverse the order on a third.
If prefill-only is already bad, construction is sufficient. If prefill is good
until decode executes, the issue is runtime mutation/cache/trace state rather than
load-time source corruption.

## Other potential issues, not headline causes

- The candidate loop permits duplicate roles, causing repeated conversion and
  replacement, but the reported role list contains each role once.
- Keeping both interleaved and DRAM-sharded precision copies increases DRAM use.
  Pure memory pressure should normally fail allocation rather than silently lower
  PCC, but Experiment B/C will expose reuse or premature-release behavior.
- Decode BFP4 may legitimately miss decode PCC due to quantization. That does not
  explain prefill drift and should be evaluated only after the construction-state
  issue is isolated.

## Recommended next action

Run Experiments A, B, and C first. The smallest likely implementation fix, if the
lifetime hypothesis verifies, is to construct the decode-only tensor directly in
its final memory config or retain the intermediate until synchronization. Do not
keep either change unless it restores the prefill BFP8 control and the original
failing check with provenance-complete evidence.
