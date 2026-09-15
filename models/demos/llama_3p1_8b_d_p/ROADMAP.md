<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llama-3.1-8B disaggregated prefill roadmap

This roadmap is the review and verification plan for Llama-3.1-8B-Instruct disaggregated prefill.
It is a recorded snapshot, not a claim that pending stages are complete. The live status page is
available at `http://127.0.0.1:8768/` only on the user's tunneled machine; it is not a public
service.

## Target and invariants

- Canonical work and external evidence live under `/data/divanovic/llama31-8b-disagg`.
- Prefill runs on one Blackhole Galaxy with SP=4 and TP=8. The existing Blaze decode engine runs
  on SC4, for five Galaxies total across the disaggregated system.
- Bring-up covers two independent slots and a 2,048-token logical limit including decode tokens.
- `PREFILL_CHUNK_SIZE=1024`, `PREFILL_MAX_SEQ_LEN=2048`, and `PREFILL_NUM_LAYERS=32` are pinned.
- A prefill transformer decoder layer processes a prompt chunk. It is distinct from the later
  autoregressive Blaze decode engine, which consumes the migrated cache one decode step at a time.
- Every implementation stage ends in a logical commit and review boundary. Its commit includes
  working tests and a plain-language comment before each test method explaining the behavior and
  expected result. Publication follows the stage's checks and review.

The model math is Llama-specific: Llama3-scaled frequencies, Meta-interleaved rotary coordinates,
RMSNorm, full causal grouped-query attention, and unclamped, bias-free dense SiLU SwiGLU. There are
no GPT-OSS YaRN frequencies, attention sinks, sliding windows, MoE experts/router, clamping, or
biases. Architecture-neutral indexed and collective mechanisms may be reused after their geometry
and data contract are checked; model-specific GPT-OSS behavior may not be inherited.

## Evidence rules for every stage

Each stage starts from documented inputs and setup, then compares against an independently derived
Hugging Face/PyTorch result or another independent contract oracle. A production helper cannot be
its own reference. Real checkpoint cases complement synthetic cases that isolate boundary and
numerical behavior.

Device comparisons must inspect every relevant chip, not just a composed tensor or the first
shard. Numerical checks include both correlation and a magnitude-sensitive metric such as
normalized L2 or maximum absolute error: high PCC alone can hide a uniform scale error. Thresholds
are chosen and justified from the operand dtype, cache dtype, and accumulation path before the run;
they are not relaxed to mask a coordinate, sharding, or precision defect. Exact invariants use exact
checks where appropriate.

Host tests and mocks establish math, validation, and control-plane behavior. They do not establish
hardware correctness. A required hardware case that is skipped, unavailable, or run on only part of
the chip set does not count as a pass. No future result or precise performance requirement is
claimed here; performance is measured after numerical correctness without inventing an unvalidated
target.

Completed host RoPE checks use exact equality for coordinate permutations, `rtol=1e-6` with
`atol=1e-8` or `1e-6` for frequency/table comparisons, and `rtol=atol=1e-5` for rotated Q/K. These
limits are appropriate for float32 evaluation of equivalent independently implemented formulas.
The device RoPE gate uses PCC ≥ 0.9999 and normalized L2 ≤ 0.01 because BF16 inputs, tables, and
outputs introduce rounding while coordinate and scale errors remain much larger. RMSNorm begins
with the same BF16 device limits, as specified in its stage brief. The MLP, attention, integrated
layer, full-model, and BF8_B-cache stages must lock their own correlation and magnitude limits in
their stage brief before the first hardware result, justified by an independent dtype-rounded
reference for that accumulation depth. Their limits are deliberately not guessed in this snapshot
and cannot be chosen after inspecting a failing result.

## Strict implementation and review sequence

The order below is also the intended logical commit/PR order. A stage begins only after its
predecessor's required checks and review are complete.

### 0. Environment and scaffold

**Input/setup.** Use the canonical checkout, source
`/data/divanovic/llama31-8b-disagg/tools/prefill_env.sh`, and confirm that
`$PREFILL_PYTHON`, TTNN bindings, and native libraries resolve from the build matching the checked
out source. Exercise the import-light adapter, bundled HF configuration, cache-path logic, and SP/TP
mesh validation in `tests/unit/test_scaffold.py`.

**Independent reference and expected behavior.** The bundled checkpoint JSON and explicit mesh
arithmetic are the references. Adapter import must not pull in model/device stacks; cache paths use
SP×TP without opening devices; invalid axes fail before sharding is derived.

**Bugs caught and pass criteria.** This catches stale native bindings, early device discovery,
incorrect cache identity, a retargeted checkpoint, and invalid mesh axes. All six scaffold tests
must pass in the source-matched environment. This gate prepares later tests but proves no model
kernel.

### 1. Host and device Llama3 RoPE

**Input/setup.** Use 128-dimensional Q/K heads, Llama3 scaling parameters from the checkpoint, and
the indexed operator used by chunked prefill. Host coverage is in `tests/unit/test_rope.py`; Galaxy
coverage is in `tests/unit/test_indexed_rope_vs_ref.py`.

The logical context limit is 2,048 tokens, while a 1,024-token physical chunk can begin at the last
valid tile. Padded physical reads therefore need tables through 3,072 positions
(`max_seq_len + chunk_size`, rounded to a global chunk). This table capacity does not extend the
logical context: the future runtime and cache writer must preserve the 2,048-token bound and write
only the logically valid tail.

**Independent reference and expected behavior.** Hugging Face Llama3 frequency math and rotary
application are the independent reference. Tests cover Q and K frames, boundary positions,
block-cyclic SP ownership, TP replication, changed indexed starts, program reuse, and padded reads
on all 32 chips.

**Bugs caught and pass criteria.** This catches GPT-OSS YaRN leakage, HF half-split versus Meta
adjacent-pair permutation, wrong Q/K conversion, stale runtime indices, incomplete tables, and SP/TP
ownership errors. Host math must meet the float32 tolerances above. Device BF16 comparisons require
PCC ≥ 0.9999 and normalized L2 ≤ 0.01, and every required start/head/chip case must run without
skips.

### 2. RMSNorm

**Input/setup.** Add `tt/rms_norm.py` and `tests/unit/test_rms_norm_vs_ref.py` (**planned**). Test a
full 4,096-wide hidden state per chip with SP row placement and TP replication, including normal,
zero, constant, tiny-magnitude, and real checkpoint gamma vectors. Exercise cached-program reuse
and preserve the residual input unchanged.

**Independent reference and expected behavior.** Use Transformers `LlamaRMSNorm` in float32 against
BF16-rounded inputs and the same gamma. Expected math is plain RMS normalization with epsilon
`1e-5` and weight multiplication, with no mean subtraction, fused residual, bias, or Q/K norm.

**Bugs caught and pass criteria.** Constant and tiny inputs expose LayerNorm substitution and a
missing/wrong epsilon; nonuniform and real gamma expose shape, replication, and address reuse bugs.
All chips must return finite outputs of the required shape/layout/dtype, meet PCC ≥ 0.9999 and
normalized L2 ≤ 0.01 for nonconstant output, produce exact zero for zero input, and leave input
unchanged. Those BF16 limits match the RoPE component gate and are tight enough to expose a wrong
epsilon or reduction while allowing normal operand rounding. Invalid gamma shapes and epsilon
values must fail clearly. The live L1 budget and launch allocation diagnostics are recorded before
the no-skip Galaxy gate.

### 3. Bias-free dense SiLU SwiGLU MLP

**Input/setup.** Add `tt/mlp.py` and `tests/unit/test_mlp_vs_ref.py` (**planned**). Cover synthetic
and real gate/up/down weights for 4,096→14,336→4,096, with TP8 column-parallel gate/up projections
and the required down-projection reduction across TP.

**Independent reference and expected behavior.** A standalone PyTorch calculation computes
`down(silu(gate(x)) * up(x))` from the same BF16-rounded input and weights. Expected output is the
full dense SwiGLU result, without biases, experts, routing, or activation clamping.

**Bugs caught and pass criteria.** Cases isolate swapped gate/up weights, missing SiLU, elementwise
product errors, wrong TP slices, and omitted/duplicated collective reduction. Synthetic and selected
real-weight outputs on every chip must meet precision-justified correlation and magnitude limits;
shape/layout, collective participation, program reuse, and invalid geometry checks must pass.

### 4. Q/K/V projections and bounded packed-cache writes

**Input/setup.** Add `tt/attention.py`, cache-writing support, and
`tests/unit/test_qkv_cache_vs_ref.py` (**planned**). Exercise all 32 query heads and eight KV heads,
TP8 ownership, multiple layers, both slots, tile-aligned nonzero starts, a full chunk, and a padded
tail at the logical limit.

**Independent reference and expected behavior.** Independent PyTorch linear projections plus the
approved Llama3 rotary reference produce Q, K, and V. A separately constructed address/ownership
oracle determines the packed cache destination. Only `[actual_start, actual_end)` may change; all
other slots, layers, heads, and positions remain untouched.

**Bugs caught and pass criteria.** This catches Q/K/V weight permutation, head reshaping errors,
wrong rotary frame, local/global start confusion, slot or layer cross-talk, and padded-tail writes.
Projection values and every written cache region must meet justified value checks on every chip;
sentinel checks must prove all out-of-range regions remain unchanged.

### 5. Full causal grouped-query attention

**Input/setup.** Extend the attention module and add `tests/unit/test_attention_vs_ref.py`
(**planned**). Cover chunk starts at zero and nonzero continuations, prompt lengths that end inside a
tile/chunk, two distinct slots, and KV history across SP4 with four Q heads sharing each KV head.

**Independent reference and expected behavior.** A direct PyTorch full-causal GQA calculation uses
the independently produced projections, rotary coordinates, causal mask, and prior KV. Each query
may attend to all valid history through its position and never to padded or future tokens.

**Bugs caught and pass criteria.** This catches accidental sliding windows or attention sinks,
wrong GQA head expansion, causal-mask offsets, SP ownership/collective errors, and continuation
cross-talk. Full outputs and attention-sensitive diagnostics across all chips, starts, heads, and
slots must satisfy justified correlation and magnitude limits with no skipped hardware cases.

### 6. One prefill transformer decoder layer

**Input/setup.** Add the decoder-layer composition and
`tests/unit/test_decoder_layer_vs_ref.py` (**planned**), combining input RMSNorm, GQA, residual,
post-attention RMSNorm, dense MLP, and the second residual. Use synthetic isolation cases and one
real Llama layer, with cache checks for K and V.

**Independent reference and expected behavior.** A standalone Hugging Face/PyTorch Llama decoder
layer, fed identical rounded inputs and weights, supplies hidden-state and cache references.

**Bugs caught and pass criteria.** This detects wrong normalization order, missing or overwritten
residuals, local/global layer indices, composition precision loss, and cache writes occurring at the
wrong stage. Every-chip hidden output and K/V must meet their separately justified correlation and
magnitude limits, while residual inputs and unrelated cache regions remain intact.

### 7. Complete 32-layer prefill

**Input/setup.** Add the model composition and `tests/unit/test_prefill_model_vs_ref.py`
(**planned**). Run real weights through all 32 layers on the SP4/TP8 Galaxy for representative
prompt lengths, boundaries, continuation starts, and two distinct slots.

**Independent reference and expected behavior.** The independent Hugging Face/PyTorch model and
CPU-generated per-layer K/V provide hidden-state and cache references. Expected behavior includes
correct layer ordering, final normalization/output contract, and valid K/V for every layer and head.

**Bugs caught and pass criteria.** This exposes errors that self-consistent component tests miss:
weight-to-layer mapping, accumulated precision drift, address aliasing, cross-layer contamination,
and slot contamination. Verify every layer/head/slot and every participating chip, both numerically
and with magnitude-sensitive errors. Record measured runtime after correctness; do not convert an
unvalidated performance aspiration into a pass condition.

### 8. Shared runtime, continuation, tail, and two-slot regressions

**Input/setup.** Register the adapter/runtime under `models/demos/common/prefill` and add
`tests/unit/test_prefill_runtime.py` plus focused engine tests (**planned**). Feed full 1,024-token
physical buffers with metadata `(slot_id, actual_start, actual_end)`, including a short tail,
non-chunk-aligned continuation bases, repeated program use, and different traces in two slots.

**Independent reference and expected behavior.** An independent chunk planner and global-position
oracle define logical tokens, SP encounter order, padding, and cache ranges. The model oracle from
stage 7 defines hidden/KV values. Exactly one globally indexed layer-completion signal is emitted
after each layer's KV is resident.

**Bugs caught and pass criteria.** This catches double reshuffling, ignored `actual_end`, treating a
physical table bound as a logical context extension, local layer acknowledgements, early acks,
wrong layer counts, stale program arguments, and two-slot cross-talk. Values, untouched sentinels,
ack order/count, continuation behavior, and all-chip ownership must pass on hardware; host/mock
control-plane tests do not replace that gate.

### 9. Migration, Blaze decode, and serving integration

**Input/setup.** Define the backend-independent cache/address-table contract first, then integrate
the existing SC4 Blaze decode and serving path. Native serving reuses tt-d-gen PR #772 or its merged
successor. A Llama-specific native manager rewrite is not assumed or required; all backends must
consume the same cache data format and address-table semantics. Planned coverage includes
`tests/unit/test_kv_chunk_table.py`, migration loopback tests, and end-to-end disaggregated serving
tests (**planned**, with final locations chosen in the owning repositories).

**Independent reference and expected behavior.** A separately calculated locator checks config
order, device/head ownership, bank offsets, layer range, position chunks, and slot strides. Source
golden K/V and an independently run decode supply the semantic reference beyond transport bytes.

**Bugs caught and pass criteria.** Byte comparison catches transport corruption but cannot prove
that two self-consistent endpoints chose the correct tensor coordinates or rotary frame. Gates must
verify every layer, head, config, and slot, then run actual migrated decode and compare its behavior
with the source-golden/reference decode. Two distinct slot traces are mandatory. Mock, loopback,
cross-endpoint, and serving gates run in increasing scope; no required hardware skip counts as pass.

## Recorded evidence at this snapshot

- The native build matching source `904cc323141` was available, and all six scaffold tests passed.
- Host RoPE: 11 tests were approved at commit `4bac36b20e5`.
- Indexed device RoPE at commit `68fe0cca2c9`: 17 host tests and the real-Galaxy device test passed
  with many Q/K, start-position, and shard cases and no skips. The final device test took 49.28 s;
  worst PCC was 0.9999936 and worst normalized L2 was 0.0031986.
- Independent Task 2 review approved commit `68fe0cca2c9` as spec-compliant, with no critical or
  important findings.
- There is no completed RMSNorm, MLP, full-model, runtime, migration, Blaze-handoff, or serving pass
  in this snapshot.

The evidence filenames are stored outside Git under
`/data/divanovic/llama31-8b-disagg/evidence`. References to those logs do not imply that the logs are
committed to this repository.

## Reproducing the current gates

Start from the canonical checkout and supplied environment:

```bash
cd /data/divanovic/llama31-8b-disagg
source tools/prefill_env.sh
cd repos/tt-metal
"$PREFILL_PYTHON" --version
```

The host-only RoPE test intentionally avoids root fixtures and third-party plugin loading:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PREFILL_PYTHON" -m pytest \
  --noconftest --rootdir=. -c /dev/null \
  models/demos/llama_3p1_8b_d_p/tests/unit/test_rope.py
```

The device test needs repository root fixtures, so it must not use `--noconftest`. Run these steps
from an Exabox login shell. First inspect the user's running jobs and identify the allocation that
the developer assigned to this test; do not select another user's job or preserve an expired job ID
as the permanent recipe:

```bash
squeue --me --states=RUNNING --format="%.18i %.9P %.8j %.2t %.10M %.6D %R"
```

Replace the marker below with that current allocation's numeric job ID. The subshell fails before
`srun` if the marker was not replaced, the value is empty/non-numeric, or the job is no longer
running:

```bash
(
  export PREFILL_SLURM_JOB_ID="REPLACE_WITH_CURRENT_ASSIGNED_JOB_ID"
  case "$PREFILL_SLURM_JOB_ID" in
    ''|*[!0-9]*)
      echo "Set PREFILL_SLURM_JOB_ID to the developer's current numeric Slurm allocation ID." >&2
      exit 2
      ;;
  esac
  if ! squeue --jobs "$PREFILL_SLURM_JOB_ID" --states=RUNNING --noheader --format="%A" \
    | grep -Fxq "$PREFILL_SLURM_JOB_ID"; then
    echo "Slurm job $PREFILL_SLURM_JOB_ID is not a current running allocation." >&2
    exit 2
  fi

  srun --overlap --jobid "$PREFILL_SLURM_JOB_ID" --nodes=1 --ntasks=1 --cpu-bind=none \
    bash -lc 'cd /data/divanovic/llama31-8b-disagg && \
    source tools/prefill_env.sh && \
    cd repos/tt-metal && \
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PREFILL_PYTHON" -m pytest \
      --rootdir=. -c /dev/null -v \
      models/demos/llama_3p1_8b_d_p/tests/unit/test_indexed_rope_vs_ref.py'
)
```

Future stage reports must record the exact allocation, interpreter/native source identity, command,
case-level numerical results, skips, and evidence filename used for that run.
