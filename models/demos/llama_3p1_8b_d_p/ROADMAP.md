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
- Every implementation stage ends in a logical commit and review boundary. The user accepted the
  reviewer role and deferred detailed independent review until the implementation works. Hardware
  tests, self-review, and repository hooks remain required. Each implementation commit includes
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

The order below is also the intended logical commit/PR order. A stage begins after its predecessor's
required checks pass, with the user-approved review arrangement above. Diagnostic comparisons may
combine modules to locate an error; they do not close a failed module gate.

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

**Input/setup.** Use `tt/rms_norm.py` and `tests/unit/test_rms_norm_vs_ref.py` (**published**). Test a
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

**Input/setup.** Use `tt/mlp.py` and `tests/unit/test_mlp_vs_ref.py` (**published**). Cover synthetic
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

**Input/setup.** Use `tt/qkv.py`, `tt/kv_cache.py`, `tests/unit/test_qkv_vs_ref.py`, and
`tests/unit/test_kv_cache.py` (**published**). Exercise all 32 query heads and eight KV heads,
TP8 ownership, multiple layers, both slots, tile-aligned nonzero starts, a full chunk, and a padded
tail at the logical limit.

**Independent reference and expected behavior.** Independent PyTorch linear projections plus the
approved Llama3 rotary reference produce Q, K, and V. A separately constructed address/ownership
oracle determines the packed cache destination. Write valid K/V only to `[actual_start, actual_end)`. Zero the unused rows through the end of the
last 32-token page. Preserve earlier history, later pages, and other slots and layers.

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

**Current candidate (16 September 2026).** Gather the selected packed K/V plane across SP4, restore
natural token order, and build an exact absolute-position causal mask on the device. Use supported
stock SDPA. Q, output, and many intermediates remain BF16. The attention destination, QK
intermediate, softmax sum, and exact position vectors use FP32. Cache storage is BF16 for diagnostics
and BF8_B for migration. This replaces the earlier indexed ring attention candidate, which failed a
real-token continuation check. The indexed ring FP32 guard remains intact.

**Required production tests.**

- Compare real checkpoint attention heads before O projection and outputs after O projection with
  independent HF-frame float32 mathematics. Keep PCC >= 0.999 / normalized L2 <= 0.03 for BF16
  cache and PCC >= 0.995 / normalized L2 <= 0.05 for BF8_B cache. These gates were selected before
  the experiments. BF8_B storage and the projection chain add rounding; the looser BF8_B gate does
  not permit coordinate or cache-address errors.
- Check exact gathered token/head order, selected slot/layer, 0/-infinity masks, numeric 0/1 query
  validity, and zero output on padded query rows. Test one-token, tile-crossing, SP-crossing,
  continuation, and terminal boundaries on every chip.
- Use a positive-value causal prefix-average fixture with pulses at page and SP boundaries.
  Omitted and shifted pulses must fail the same independent oracle. Include zero-value and
  future-tail poisoning cases so correlation cannot hide illegal reads.
  The original narrow-variance fixture has BF16-oracle baseline PCC 0.992345 at pulse 256 and
  0.992072 at pulse 1024, with normalized L2 about 0.0023. This is not a strict PCC ceiling and does
  not explain the observed device minimum PCC 0.97959. A stronger feature-variation fixture may
  improve conditioning, but thresholds stay unchanged.
- Replay A, B, then A with changed input and cache addresses. Verify unchanged input/cache data,
  refreshed runtime arguments, stable warmed program counts, and bounded persistent allocations.
  Invalid calls must fail before cache changes; a subsequent valid call must work.
- Compare the original ten synthetic intervals and both cache types with stock causal SDPA on the
  same exact inputs. The preselected parity gate is PCC >= 0.9999 / normalized L2 <= 0.01 per chip.
  This checks our gather/mask composition; it does not replace the independent real-weight oracle.
- Keep the stock streaming exponential-mode regression separate. It must detect the previously
  ignored `exp_approx_mode=False` flag and retain approximate-mode behavior.

**Known numerical limitation.** The hash fixture still fails its original float-reference limits.
At query position 256, both explicit-mask and stock causal FP32 attention have PCC 0.995757 and
normalized L2 0.092928. Their outputs at the failing row are identical. BF8_B cache compression
also causes large relative error in some synthetic cases before attention runs. Keep the original
fixtures, limits, and measured failures in executable characterization tools. Do not label them as
passing, silently relax their limits, or claim that one identified arithmetic operation explains all
cases. These periodic-hash failures remain characterization evidence after K512 acceptance.

### 6. One prefill transformer decoder layer

**Input/setup.** Add the decoder-layer composition and
`tests/unit/test_decoder_layer_vs_ref.py`, combining input RMSNorm, GQA, residual,
post-attention RMSNorm, dense MLP, and the second residual. Use synthetic isolation cases and real
Llama layers 0 and 13, with cache checks for K and V.
Share accepted attention resources across layers. Validate the request and cache before writing K/V.

**Preparation status (16 September 2026).** Root copied the exact three prepared files into the
canonical tree. Host checks pass, including independent Hugging Face decoder parity with maximum
absolute difference `1.1920928955078125e-07`. That value checks the host oracle and fixture only; it
is not device accuracy. The launch contract is open. Device validation is starting in this order:
`residual001`, `smoke002`, `full003`, then `Watcher004`. No decoder device result exists yet.

**Independent reference and expected behavior.** A standalone Hugging Face/PyTorch Llama decoder
layer, fed identical rounded inputs and weights, supplies hidden-state and cache references.

**Bugs caught and pass criteria.** This detects wrong normalization order, missing or overwritten
residuals, local/global layer indices, composition precision loss, and cache writes occurring at the
wrong stage. Every-chip hidden output and K/V must meet their separately justified correlation and
magnitude limits, while residual inputs and unrelated cache regions remain intact. The selected
layer-output gates are PCC >= 0.999 with normalized L2 <= 0.025 for BF16 cache. BF8_B uses
PCC >= 0.999 and normalized L2 <= 0.05. This strict BF16 gate was fixed before hardware execution.
Keep the published independent KV gates: PCC >= 0.9999 / normalized L2 <= 0.01 for BF16 and
PCC >= 0.999 / normalized L2 <= 0.02 for BF8_B. Use zero-branch and isolated-branch runs to prove
that residual magnitude cannot hide missing attention or MLP output. Check all tensors and metrics
for finiteness before aggregation. These limits are fixed before decoder implementation and testing.

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

## Recorded evidence at this snapshot (16 September 2026)

- The native build matching source `904cc323141` was available, and all six scaffold tests passed.
- Host RoPE: 11 tests were approved at commit `4bac36b20e5`.
- Indexed device RoPE at commit `68fe0cca2c9`: 17 host tests and the real-Galaxy device test passed
  with many Q/K, start-position, and shard cases and no skips. The final device test took 49.28 s;
  worst PCC was 0.9999936 and worst normalized L2 was 0.0031986.
- Independent Task 2 review approved commit `68fe0cca2c9` as spec-compliant, with no critical or
  important findings.
- RMSNorm is published at `1ff25da02ae`; MLP is published at `7b12bf9d576`. Their module hardware
  tests and review passed.
- QKV/cache writes are published at `4cf42fb0b9a`: all seven hardware tests passed, followed by
  five covering checks after hooks. The user accepted the reviewer role; independent review was
  deferred and is not claimed as complete.
- Isolated attention output projection passed. The exponential-mode regression passed.
- FP32 attention prototype: attempt 031 passed 12 exact boundary cases and all six original
  repeated-token stress cases. Worst head PCC / normalized L2 was 0.999785 / 0.020919 for BF16 and
  0.999651 / 0.032706 for BF8_B. These are prototype results.
- Attempt 032 retained the failed synthetic limits. Attempt 033 reproduced the worst failing row
  with stock causal attention. A completed diagnostic is not a numerical acceptance pass.
- Attempt 034 passed all 20 production-to-stock parity cases at the fixed per-chip gates. Attempt 035
  passed 20 real-weight and 12 token-stream cases at unchanged gates, but K128 pulse PCC failed. At
  that stage, production attention remained unaccepted. The original failures remain characterization.
  Host omit and shift mutations fail all eight exposed TP shards at each boundary. Attempt 036
  passed its first three stronger pulses but failed last-pulse PCC. Attempt 037 matched production
  and stock outputs exactly on all eight valid chips for both cache dtypes. Its source minimum PCC was
  0.9975303 for BF16 and 0.9974257 for BF8_B, so the source gate still failed. All 136,048 numeric
  values were finite, and devices closed cleanly. A bounded host replay of repeated BF16 partial-
  numerator rounding did not explain the error; error-direction cosine was weak or negative.
- Attempt 038 tested Q128/K512 directly. Both cache dtypes passed the original source and cache pulse
  gates, and production matched stock exactly on all eight valid chips. Production source hash is
  `ab1808733e9d5ed4a515cdd94c35c5b5cd848a157878de3020ac991fc7fc10e2`. Actual circular-buffer
  allocation is 1,241,088 bytes per core; the conservative L1 gate is 1,273,856 bytes per core.
- Attempt 041 passed the final production suite at unchanged numerical gates: actual exit 0, verified
  exit 0, eight tests, no skips, all 32 chips, and all 32 real-weight and token-stream cases. K512 pulse
  checks passed for both cache dtypes. Devices closed cleanly at 13:19:12.853 UTC. This supplies final
  Task 6 acceptance evidence. Original periodic-hash failures remain characterization evidence, and
  the exact internal cause is unresolved.
- Task 7 isolated preparation completed, and root copied the exact three files into the canonical
  tree. Host oracle parity reached maximum absolute difference `1.1920928955078125e-07`; this is not
  device accuracy. The launch contract is open. Device validation is starting with `residual001`,
  `smoke002`, `full003`, then `Watcher004`; no decoder device result exists yet.
- Decoder-layer, full-model, runtime, migration, Blaze-handoff, and serving gates have not run.

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
