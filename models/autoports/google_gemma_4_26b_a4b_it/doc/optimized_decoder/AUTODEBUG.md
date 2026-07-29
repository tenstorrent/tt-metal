# AutoDebug: optimized prefill logical-boundary failure

Date: 2026-07-29 UTC

Scope: source and existing artifacts only. This investigation did not run
pytest, TT hardware, `tt-smi`, reset, watcher, profiler, or any other device
command. It did not modify implementation or tests.

## Verdict

The shipped/default optimized decoder at source SHA256
`9da01f5f2571d0c0e3d1ac73297fb9d21908b254c4173aa50b8f466078ad74b2`
selected an unverified 128-token grouped sparse-MoE prefill path. That is the
cause at the smallest intervention boundary.

The default `prefill_expert_chunk_size=128` is wrong for the currently proven
sparse-prefill contract. A 32-token chunk makes every sparse invocation contain
one tile group. With the fast geometry otherwise unchanged
(`per_core_n=2`, gate `block_w=44`, down `block_w=11`), that single change
passes all 20 logical-boundary cases:

- sliding attention: minimum PCC `0.9953398667`;
- full attention: minimum PCC `0.9980481839`.

The smallest safe fix is therefore to set `prefill_expert_chunk_size=TILE_SIZE`
in both `OptimizedDecoder.__init__` and
`OptimizedDecoder.from_state_dict`, and to update the defaults assertion from
128 to 32. No block-width, `per_core_n`, tail-geometry, attention, cache, or
threshold change is needed.

That exact fix was applied and hardware-verified by the parent task while this
review was in progress. The current files and refreshed no-policy-override
artifacts show the fixed default. The deeper reason a grouped invocation is
numerically wrong is not yet localized between gate/up grouped-batch handling,
the down projection's multi-row work split, and their surrounding
transpose/reshape chain. It should be investigated as a separate TTNN
`sparse_matmul` issue; it is not necessary to hold the safe decoder fix.

## Provenance and state

The report analyzes the failing shipped source, not just the post-fix working
tree.

| State | Decoder SHA256 | Test SHA256 | Effective chunk | Evidence |
| --- | --- | --- | ---: | --- |
| failing shipped default | `9da01f5f...ad74b2` | `0c8a9fb9...195ba` | 128 | `candidate_runs/tail_aware_boundary_diagnostic.json` |
| same source, controlled passing A/B | `9da01f5f...ad74b2` | `0c8a9fb9...195ba` | 32 | `candidate_runs/chunk32_fast_boundary.json` |
| fixed no-policy-override default | `803f0e1945...f9e7e` | `829da22cc6...d3b41` | 32 | current `prefill_boundaries_*.json` |

The current source consequently shows `TILE_SIZE` at
`tt/optimized_decoder.py:257` and `:321`, and the current defaults test expects
32 at `tests/test_optimized_decoder.py:311`. The failing artifacts retain the
original constructor default of 128 in their stamped provenance.

The repository AutoDebug Codex backend could not enter its workspace because
the host disallows the requested bubblewrap namespace. The supported Claude
fallback emitted no report or progress during a bounded 18-minute run and was
terminated. The findings below were then completed by direct source/artifact
review, and every headline claim was rechecked against the code and the
controlled artifacts.

## Direct observations

### 1. Failure starts exactly when one sparse invocation contains more than one tile group

The boundary oracle pads every logical sequence to a multiple of 32 before the
optimized MoE. Under the shipped chunk size of 128:

| Layer kind | Logical length | Physical length | Groups in invocation | PCC |
| --- | ---: | ---: | ---: | ---: |
| sliding | 1 | 32 | 1 | 0.997323 |
| sliding | 31 | 32 | 1 | 0.996908 |
| sliding | 32 | 32 | 1 | 0.997454 |
| sliding | 33 | 64 | 2 | 0.994842 |
| sliding | 63 | 64 | 2 | 0.938407 |
| sliding | 64 | 64 | 2 | 0.935063 |
| sliding | 65 | 96 | 3 | 0.941738 |
| sliding | 1023 | 1024 | 8 × 4 | 0.888242 |
| sliding | 1024 | 1024 | 8 × 4 | 0.892437 |
| sliding | 1025 | 1056 | 8 × 4 + 1 | 0.892085 |
| full | 1 | 32 | 1 | 0.999875 |
| full | 31 | 32 | 1 | 0.998082 |
| full | 32 | 32 | 1 | 0.998209 |
| full | 33 | 64 | 2 | 0.994998 |
| full | 127 | 128 | 4 | 0.906667 |
| full | 128 | 128 | 4 | 0.900904 |
| full | 129 | 160 | 4 + 1 | 0.908236 |
| full | 1023 | 1024 | 8 × 4 | 0.903767 |
| full | 1024 | 1024 | 8 × 4 | 0.904291 |
| full | 1025 | 1056 | 8 × 4 + 1 | 0.904579 |

These values come from
`candidate_runs/tail_aware_boundary_diagnostic.json`, whose provenance records
the failing source hash and no correctness-policy override.

All one-group cases pass. The first two-group case is already below the
unchanged `0.995` bar for both layer kinds. Longer cases do not introduce a new
boundary; they merely contain larger or repeated grouped invocations.

### 2. Chunk size is isolated from sparse geometry

The available controlled contrasts cover both layer kinds and all ten
boundaries per kind:

| Effective policy | Sliding minimum | Full minimum | Verdict |
| --- | ---: | ---: | --- |
| chunk 128, shipped fast geometry | 0.888242 | 0.900904 | fail |
| chunk 128, legacy `pcn=11`, gate/down block 1 | 0.888421 | 0.900920 | fail |
| chunk 32, legacy `pcn=11`, gate/down block 1 | 0.995449 | 0.998082 | pass |
| chunk 32, fast `pcn=2`, gate 44/down 11 | 0.995340 | 0.998048 | pass |
| fixed chunk-32 default, fast geometry, no policy override | 0.995340 | 0.998048 | pass |

The strongest contrast is
`candidate_runs/chunk32_fast_boundary.json`: it uses the exact failing
decoder/test hashes and sets chunk size to 32 while redundantly setting
`per_core_n`, gate block width, and down block width to their shipped values.
It therefore does not rely on the earlier legacy geometry.

The current `prefill_boundaries_full_attention.json` and
`prefill_boundaries_sliding_attention.json` repeat those passing PCCs with
constructor default 32 and no `GEMMA4_OPT_PREFILL_*` override.

### 3. The source sends chunk size directly to the grouped sparse path

`OptimizedDecoder.from_state_dict` resolves
`GEMMA4_OPT_PREFILL_EXPERT_CHUNK_SIZE` over the constructor default
(`tt/optimized_decoder.py:380`), passes it through construction (`:414`), and
stores it on the decoder (`:436`).

`_moe_prefill_chunk` then:

1. splits physical tokens by that exact size (`:791-801`);
2. computes `groups = physical_chunk // 32` (`:805-808`);
3. reshapes gate/up input to `[1, groups, 32, 2816]` and repeats the all-expert
   sparsity descriptor across groups (`:816-825`);
4. performs grouped sparse gate and up projections, transposes them, and
   flattens group × tile rows back to physical sequence (`:840-859`);
5. performs the sparse down projection with input A sparse
   (`:861-872`);
6. applies routing, reduces experts, and concatenates chunks (`:873-878`).

Thus chunk 128 selects groups 2, 3, or 4 for physical lengths 64, 96, or 128.
Chunk 32 makes `groups=1` for every invocation without changing model math.

The sparsity/count pairs are source-consistent: gate/up repeat an all-ones
128-expert mask `groups` times and pass `nnz=128*groups`; down uses the base
128-entry all-ones mask and `nnz=128`. This is a numerical correctness
failure, not evidence of an `nnz` deadlock contract violation.

### 4. The canonical and functional preparation path specifies chunk 32

The functional decoder delegates MoE prefill to the canonical Gemma helper
(`tt/functional_decoder.py:1160-1167`).

That helper explicitly sets `PREFILL_CHUNK_SIZE = 32` and states why:
one group keeps sparse `num_blocks_y=1` and is the guaranteed-fit path
(`models/demos/gemma4/tt/experts/prefill.py:144-149`). It splits all longer
inputs into 32-token chunks before `_process_prefill_chunk`
(`prefill.py:181-204`).

The optimized default of 128 therefore replaced an established preparation
contract. Performance candidate artifacts existed for larger chunks, but the
full boundary evidence that passed before the default run used the explicit
chunk-32 override.

### 5. The test really covers the optimized common path and the complete matrix

`test_optimized_paged_prefill_logical_boundary_lengths` installs
`OptimizedDecoder` into the functional HF oracle and requires optimized
attention, dense MLP, MoE prefill, and MoE chunk methods to execute
(`tests/test_optimized_decoder.py:244-260`).

It intentionally lets every boundary run, records every PCC, then asserts the
minimum is at least `0.995` (`:250-265`). The two parameter values are sliding
layer 0 and full layer 5. The failure is therefore not a partial matrix, a
functional fallback, or an early-abort artifact.

Both layer kinds share `_moe_prefill_chunk`. Their different PCC magnitudes are
consistent with different attention outputs entering the same bad grouped MoE
path; the common transition at physical length 64 argues against separate
sliding/full attention bugs.

## Diagnosis

### Headline finding: the default crossed the proven intervention boundary

**Status: verified.**

The decoder's public setup default selected groups greater than one even though
the canonical sibling path fixes the sparse chunk to one group. The complete
same-source A/B shows that changing this setup value alone repairs every named
case. This explains:

- why 1/31/32 pass under the bad default;
- why both layer kinds first fail at logical 33 / physical 64;
- why legacy block geometry does not repair chunk 128;
- why both legacy and fast geometries pass at chunk 32;
- why long cases remain poor even when they end with a correct 32-token tail;
- why the current two-default fix produces the same PCCs as the controlled
  chunk-32 fast candidate.

No deeper implementation claim is needed to establish the safe decoder fix.

### Deeper hypothesis A: sparse down's multi-row work split is wrong for this exact configuration

**Status: plausible and highest-value follow-up; not proven.**

For groups greater than one, down input is
`[1, 128, 32*groups, 704]` with `is_input_a_sparse=True`,
`per_core_M=1`, and output width 2816. The C++ sparse factory derives
`num_blocks_y` from `Mt/per_core_M`
(`sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp:112-142`), so this is
the first projection where groups become multiple M work blocks.

Existing unit coverage for sparse input A uses `m=16`, tile height 16, and
`per_core_M=1`, so M remains one tile
(`tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py:426-504`).
The broad sparse sweep covers larger M but sets `per_core_M=m/32`, again making
one M work block. Neither matches `M>32` split across cores with
`per_core_M=1`, 128 all-active experts, and the Gemma dimensions/config.

This gap makes down the best first isolation target, but the full-layer PCC
does not prove down is the first wrong tensor.

### Deeper hypothesis B: gate/up grouped-batch enumeration or flattening is wrong

**Status: plausible; not proven.**

Gate and up use A shaped `[1, groups, 32, 2816]`, B shaped
`[1, 128, 2816, 704]`, repeated sparsity `[1,1,groups,128]`, and
`nnz=groups*128`. Their outputs are transposed across group/expert dimensions
and reshaped into `[1,128,32*groups,704]`.

A wrong sparse batch ordering, output ordering, transpose interpretation, or
reshape physical-layout assumption would be invisible at `groups=1` and would
corrupt exactly the observed multi-group cases. It must be tested before
assigning the bug to down.

### Lower-ranked hypothesis: a later grouped pointwise/reduction layout issue

**Status: possible but less likely.**

GeGLU, routing multiplication, fast expert reduction, or concatenation could
consume a grouped sparse result incorrectly. Chunk 32 exercises the same
operations and repeated concatenation successfully, so any such bug would
have to depend specifically on the provenance/layout of a multi-group sparse
result rather than on sequence concatenation generally.

## Refuted or demoted explanations

- **Fast block geometry is the root cause:** refuted. Chunk 128 still fails
  with legacy `pcn=11`/block-1 geometry; chunk 32 passes with fast
  `pcn=2`/44/11 geometry.
- **Tail geometry fixes short non-128 chunks:** refuted. Under the shipped
  default, physical lengths 64 and 96 use the tail branch
  (`pcn=11`, block 1) and still fail. The fixed chunk-32 default never needs
  the tail branch for a processed chunk and passes.
- **BF8/LoFi alone is too inaccurate:** refuted as a complete explanation.
  The same weights, fidelity, and fast geometry pass the entire matrix when
  grouped invocation is removed. A grouped-specific accumulation issue remains
  possible.
- **Sliding attention, full attention, paged cache, or page ordering is the
  common bug:** strongly demoted. The two attention paths differ, but the
  onset follows the shared MoE group count and disappears when only MoE chunk
  size changes.
- **Logical padding itself is corrupt:** demoted. Padding is unchanged in the
  passing A/B. Physical 32 passes, physical 64 is repaired by splitting into
  two physical-32 sparse calls, and returned logical slicing is unchanged.
- **`nnz` metadata disagrees with sparsity contents:** source-refuted for this
  all-ones path, as described above.

## Focused verify/refute experiments

The first experiment is complete; the remaining experiments require TT
hardware and were not run by this review.

### Completed: decoder-level single-variable A/B

- Failing: shipped source/test hashes, chunk 128, fast geometry.
- Passing: same hashes, chunk 32, same fast geometry.
- Result: all 20 cases move above `0.995`.
- Verdict: verifies the default/intervention boundary.

### Hardware follow-up 1: isolate gate and up

For `groups = 1, 2, 3, 4`, feed the same BF16 activation and BFP8 expert
weights to:

1. one grouped gate/up invocation;
2. `groups` independent one-tile invocations concatenated in logical sequence.

Compare raw sparse output before transpose, after transpose/reshape, and after
GeGLU. Use exact Gemma dimensions, all 128 experts, exact `nnz`, LoFi compute,
and the optimized program builder.

- Divergence before transpose proves the grouped sparse batch path.
- Equality before transpose but divergence after reshape proves the wrapper's
  output-layout assumption.
- Equality through GeGLU moves suspicion to down.

### Hardware follow-up 2: isolate down

Construct one identical down input and compare:

1. `[1,128,32*groups,704]` in one sparse call using
   `is_input_a_sparse=True`, `per_core_M=1`;
2. `groups` independent `[1,128,32,704]` calls concatenated in M.

Test groups 1/2/3/4, exact Gemma widths, BFP8 down weights, all 128 experts,
and the exact grids produced by `_optimized_sparse_prefill_config`.

If only the multi-row form diverges, add a TTNN regression at this exact
`M>32`, `per_core_M=1` boundary before changing the program factory/kernel.

### Hardware follow-up 3: locate the first bad full-decoder tensor

At physical lengths 64 and 128, compare grouped versus 32-chunk execution
after:

1. attention/residual;
2. router and MoE input norm;
3. gate;
4. up;
5. GeGLU;
6. down;
7. routing multiply/reduction.

Use per-token/per-group PCC and maximum error rather than only final layer PCC.
This distinguishes a reordered group from uniform numerical degradation.

### Hardware follow-up 4: precision only as a discriminator

After locating the first bad sparse projection, cross:

- BFP8 versus BF16 weights;
- LoFi versus HiFi2/HiFi4;
- grouped versus independent 32-token calls.

Do not accept higher fidelity as the default fix unless it repairs every
group count and remains within the expected performance envelope. The current
evidence already provides a smaller correctness-preserving preparation fix.

### Final regression gate after any deeper TTNN repair

Before considering chunk 64/96/128 safe again:

1. run the direct sparse-op regression at groups 1/2/3/4;
2. run both complete logical-boundary suites;
3. run non-aligned capacity and batch-2 prefill;
4. run the full optimized default suite with no policy-changing
   `GEMMA4_OPT_*` environment variables;
5. remeasure warmed prefill, separately from correctness/watcher runs.

## Smallest likely fix and current status

The safe patch is:

```python
# OptimizedDecoder.__init__
prefill_expert_chunk_size: int = TILE_SIZE

# OptimizedDecoder.from_state_dict
prefill_expert_chunk_size: int = TILE_SIZE
```

and:

```python
assert signature.parameters["prefill_expert_chunk_size"].default == 32
```

Keep the existing environment override for explicit experiments. Keep the fast
`per_core_n=2`, gate block 44, down block 11 defaults. Tail settings can remain
as opt-in/larger-chunk diagnostic policy, but they are not part of the safe
default path when every processed chunk is exactly 32 tokens.

The current working tree already contains these three changes, made outside
this inspection-only task. Refreshed default artifacts verify all 20 cases:

- current sliding PCC range: `0.9953398667` to `0.9984786570`;
- current full PCC range: `0.9980481839` to `0.9998492412`.

The parent task additionally reports that the clean default suite, watcher
gate, both boundary suites, and both capacity paths all pass after this
chunk-32 default change. Those are post-fix hardware results supplied to this
source-only review; this review did not execute them.

## Evidence/test gap that allowed the regression

The artifact stamper records constructor defaults and environment overrides,
but earlier boundary evidence was accepted from a candidate run whose
effective chunk was 32 while the shipped constructor default remained 128.
The defaults unit test then asserted 128 without connecting that policy to the
full default-path boundary gate. Documentation consequently claimed the
boundary suite passed even though the clean shipped default did not.

For final-stage evidence, reject or clearly label any correctness artifact
containing policy-changing `GEMMA4_OPT_*` overrides. A candidate artifact can
support an A/B, but the final claim must come from the clean default and its
stamped effective policy. The refreshed current artifacts now satisfy that
requirement.

## Final disposition

The decoder-level diagnosis is conclusive: restore the canonical one-tile
sparse-prefill chunk as the default. The implementation/test fix already
present in the working tree is the smallest verified correction.

The internal grouped sparse-matmul defect is real at the full-decoder
boundary, but its first divergent projection remains unresolved. Keep chunk
sizes above 32 experimental until the gate/up-versus-down isolation above
identifies and regression-tests the exact lower-level contract.
