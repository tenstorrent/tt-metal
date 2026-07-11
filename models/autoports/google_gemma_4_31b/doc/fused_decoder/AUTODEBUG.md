# AutoDebug: Stage 02 fused-decoder review failures

## Executive verdict

The independent `more-work-needed` verdict is supported by the delivered code
and evidence. No static inspection found a proven fused-decoder correctness
failure, but the current artifacts cannot establish the required mutable trace
semantics, long-prefill fusion exhaustion, or source-to-evidence provenance.

The findings below are hypotheses for the AutoFix verification loop, not verified
fixes. Their confidence ranking is:

1. **High — coverage defect:** fused traced decode has never been replayed after
   changing the captured token, RoPE position, and cache-update index.
2. **High — optimization-closure defect:** long prefill always runs a standalone
   GELU because of a local `m_tiles <= 4` guard. That guard is not a source-backed
   TTNN legality limit. A plausible explicit 1D configuration exists, but only a
   device admission/PCC/performance experiment may call it legal.
3. **High — provenance defect:** the delivered test file postdates every retained
   gate, and report destinations do not match delivered destinations.
4. **High — unsupported ordering claim:** the selected sliding-decode path leads
   the post-projection candidate by only 0.810 us in two non-interleaved,
   single-sample runs. Repeated A/B is required to claim it is faster.

No TT hardware was used and no implementation or test file was changed.

## Finding 1 — fused mutable-input trace semantics are untested

### Evidence and causal scope

`tests/test_fused_decoder.py:59-135` captures `FusedDecoder` with one token and
one position, then executes the trace eight times without changing any captured
allocation. This proves deterministic unchanged replay, but a graph that reads a
stale token, stale RoPE row, or stale cache index could pass.

The inherited batch-32 test also does not close the gap. At
`tests/test_functional_decoder.py:769-777`, it copies the original token and
original position values back into the same device allocations and checks an
identical result.

The exact regression already exists at
`tests/test_functional_decoder.py:400-524`. It:

- chooses random non-block-aligned positions or a 1023-to-1024 sliding-window
  boundary;
- captures with `token_a`, uint32 RoPE lookup position A, and int32 cache index A;
- copies distinct token/position/index B values into the same allocations;
- compares the replay to the second HF decode;
- requires the correct-output RMSE to beat the stale-output RMSE; and
- replays B again and requires bitwise determinism.

However, line 415 constructs `FunctionalDecoder`, and the fused suite never
wraps this test. That matters because `FusedDecoder._decode_attention`
(`tt/fused_decoder.py:252-424`) owns the affected graph: token-dependent QKV is
written to L1; `current_position` selects RoPE tables; `current_position_cache`
selects cache updates and paged SDPA; full attention uses
`paged_fused_update_cache`, while sliding attention uses two modulo-aware
`paged_update_cache` calls.

This finding explains an evidence gap, not an observed wrong output. Static code
shows the intended device tensors flow to all consumers, but unchanged replay
cannot prove that trace capture preserves those runtime bindings.

### Smallest verify/refute experiment

Add only a fused wrapper around the existing functional regression, following
the suite's established wrappers at `test_fused_decoder.py:240-254`: monkeypatch
`functional_tests.FunctionalDecoder = FusedDecoder`, then invoke
`test_changed_trace_buffers_random_and_boundaries`. Run its full four-case
matrix: sliding/full layer kinds times random/window-wrap positions.

Prediction: all four cases achieve the existing PCC threshold against HF, the B
replay is closer to reference B than stale replay A, and repeated B is bitwise
identical. Any failure should then be localized by mutating token, uint32 RoPE
position, and int32 cache position one at a time; this distinguishes stale input,
stale embedding lookup, and stale cache/SDPA indexing.

### Watcher implication and intervention boundary

Run the four-case fused mutation node under watcher after the ordinary focused
run. This path mutates indices consumed by paged cache writers and paged SDPA,
and trace enqueue completion is asynchronous unless explicitly synchronized.
The existing watcher log covers only unchanged replay and predates the delivered
test source, so it is not reusable. Retain the pytest log and watcher attach,
check, and detach log with no fatal/assert/NOC/L1/overflow/sanitizer finding.

If the wrapper passes, the smallest intervention is test/evidence only. If it
fails, do not alter all trace code at once: use the one-buffer-at-a-time ladder
above and repair the first stale binding in `_decode_attention` or its caller.

## Finding 2 — long-prefill GELU fusion remains open

### Confirmed branch and real geometry

`_FusedSharedMLP.__call__` at `tt/fused_decoder.py:47-71` constructs an explicit
1D multicast matmul with fused GELU only for `m_tiles <= 4`; otherwise it passes
no program config and calls standalone `ttnn.gelu`. `MLP_CHUNK = 4096` at
`tt/functional_decoder.py:37`, and the fused long-path loop at
`tt/fused_decoder.py:471-484` sends 4096 rows per MLP call. Thus normal long
chunks are exactly:

```text
M=4096, K=5376, N=21504
Mt=128, Kt=168, Nt=672 (32x32 tiles)
```

Both weights and activations are BF16 on this stage. The retained sequence-128
report proves the explicit 11x10 1D family fuses GELU for Mt=4, but it does not
exercise Mt=128. The rejected `activation=` candidate proves only that the
wrapper left a post-matmul unary when it generated the program config; it does
not reject an explicit config-level `fused_activation`.

### What source proves — and does not prove

The `m_tiles <= 4` condition is a local tuning guard, not a TTNN validator.
TTNN's block validator requires nonzero dimensions, block/subblock divisibility,
per-core/block divisibility, and a subblock fitting destination registers
(`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:119-193`).
The 1D factory also requires `Kt % in0_block_w == 0`
(`matmul_multicore_reuse_mcast_1d_program_factory.cpp:4866`). For `mcast_in0`,
the work mapping requires a single M work block, so `per_core_M=Mt`; it does
**not** require `out_block_h=per_core_M`.

The factory explicitly loops over
`per_core_M / out_block_h` and sizes its major input/output/intermediate CBs from
the smaller output block (`matmul_multicore_reuse_mcast_1d_program_factory.cpp:
146-204`). This is the important adaptation the current code omits. For BF16,
`in0_block_w=2`, and `out_block_w=7`, the four principal CBs total approximately:

| `out_block_h` | Approximate four-CB bytes |
|---:|---:|
| 1 | 94,208 |
| 2 | 131,072 |
| 4 | 204,800 |
| 128 (naive extension) | 4,775,936 |

Therefore naively setting `out_block_h=m_tiles=128` is clearly unsuitable, but
keeping `per_core_M=128` while using `out_block_h` 1, 2, or 4 is an intended
factory idiom. These numbers are not a total program/L1 admission proof: code,
other CBs, allocator state, architecture limits, and runtime correctness still
require a P150 experiment. No configuration below should be described as legal
until it compiles, runs, synchronizes, and passes PCC on the target.

The factory does support non-ReLU fused activations and selects its fused
bias/activation compute kernel (`matmul_multicore_reuse_mcast_1d_program_factory.cpp:
499-512,726-750`). Nightly activation coverage uses this 1D family with fused
fast GELU, but only through M <= 128 rows, so it is supporting idiom evidence,
not coverage of the real long shape.

### Minimal candidate matrix

Run candidates one at a time and retain only verified configurations:

| ID | MLP chunk | Candidate | Purpose |
|---|---:|---|---|
| B0 | 4096 | current auto matmul + unary GELU | baseline |
| F4 | 4096 | 1D 11x10; `in0_block_w=2`; subblock 1x7; block 4x7; `per_core_M=128`, `per_core_N=7`; fused fast GELU; `mcast_in0=True` | direct real-shape adaptation |
| C2048 | 2048 | best admitted F geometry, `per_core_M=64` | first chunk fallback |
| C1024 | 1024 | same, `per_core_M=32` | second chunk fallback |
| C128 | 128 | current proven Mt=4 geometry | fused control and chunk-overhead bound |

If F4 is rejected or slower, try `out_block_h=2` and then 1 before changing
families. Each divides `per_core_M`; `Kt=168` is divisible by block width 2;
`Nt=672` yields 96 seven-tile N blocks, within the 110-core rectangle. Those
facts satisfy visible algebraic constraints only. An explicit 2D multicast
family is a second-line experiment, not part of the first minimal matrix; its
exact grid/block/L1 geometry must be derived and admitted rather than guessed.

The 262113 tail is 4065 logical rows and tile-pads to the same Mt=128 class, so
test both aligned 262144 and nonaligned 262113 after selecting a candidate.

### Experiment ladder and intervention boundary

1. Run a single real-shape gate linear for each candidate; synchronize and
   record admission or the exact validator/allocator/runtime rejection.
2. Compare fused gate output against the current linear-plus-approximate-GELU
   output and require the stage PCC bar. Confirm the profiler has no standalone
   GELU row.
3. Run the complete real-shape MLP chunk and compare output/PCC; collect warmed
   per-chunk latency.
4. Measure warmed end-to-end long-path latency with identical inputs and both
   layer kinds. Validate aligned 262144 and nonaligned 262113 correctness,
   including the final padded chunk.
5. Keep the fastest correct chunk family, or retain B0 with the exact rejection
   or measured end-to-end regression artifacts.

The smallest likely code intervention, if F4 verifies, is to derive
`per_core_M` from Mt while keeping a small divisor `out_block_h`, rather than
raising/removing the guard with `out_block_h=m_tiles`. If no explicit family
admits or every correct family regresses end-to-end, the intervention is
documentation/evidence: record the exact blocker or distribution, not the
current unsupported prose assertion.

## Finding 3 — evidence provenance is stale and destination-ambiguous

### Evidence

The current hashes are:

```text
3ae979f9153386cdc7fc07e445be42209f7bb375e0a23c9ff2b5418c2cb7d845  tt/fused_decoder.py
62baf70fffdb441f1e5818001d452c014da546be545479835f23b6a417e1f8c0  tests/test_fused_decoder.py
```

But `tests/test_fused_decoder.py` has mtime 2026-07-11 08:37:12 UTC, after
`standard_suite.log` (08:29:04), refreshed watcher evidence (08:29:31), all
final Tracy files (08:32:22), and `long_nonaligned_262113.log` (08:36:23). The
hash appears only in prose; no run log binds it to the executed source. Because
the stage files are untracked, there is no versioned prior test blob with which
to prove that the late edit was import-only.

Furthermore, final `tracy/*/*/*console.log` files say reports were written under
`tracy_refresh`, while delivered files live under `tracy`. Candidate report text
for `activation_argument_not_fused...` and `folded_mlp_scalar_only...` names
canonical selected destinations rather than its candidate directory. Numeric
CSV recomputation may be sound, but the retained files do not establish their
copy/report lineage.

### Exact minimal cleanup and reruns

After the trace wrapper and any GELU decision are final:

1. Record `git HEAD`, UTC time, exact command, and SHA-256 for
   `tt/{fused_decoder.py,functional_decoder.py}` and
   `tests/{test_fused_decoder.py,test_functional_decoder.py}` at the start of
   every retained log. The functional files must be bound because fused tests
   execute inherited helpers and the mutation regression.
2. Rerun the complete standard fused suite against those hashes.
3. Run the new four-case mutable-trace node normally and under watcher; retain
   both logs and the watcher output tree.
4. Because no prior test blob can classify the late edit, conservatively refresh
   both 262144 and 262113 long gates against the final hashes. This is the
   smallest unambiguous remedy; mtime reasoning cannot recover missing lineage.
5. If implementation/performance-node code changes for the GELU experiment,
   regenerate all affected final profiler paths. Run `tt-perf-report` directly
   into the same final directory as its raw CSV and console log. Regenerate the
   two misleading candidate reports from their candidate raw CSVs into their
   own candidate directories.
6. Add a manifest mapping every command to input hashes, exit status, raw CSV,
   signposts, filtered CSV, rendered report, and artifact SHA-256. Prefer no
   copies. If staging/copying is unavoidable, record source and destination
   paths plus matching pre/post-copy hashes; do not retain console text that
   names a different destination without that manifest.

The intervention boundary is evidence generation and documentation unless a
hash-bound rerun fails. Existing artifacts need not be dismissed numerically,
but they cannot serve as the final provenance-bound gates.

## Finding 4 — the 0.810 us candidate ordering is noise-scale

The selected sliding decode recomputes to 2556.330 us and the post-projection
candidate to 2557.140 us: 0.810 us, or 0.032%. Each raw CSV contains one measured
signpost interval. The test performs one ordinary warm call, one blocking trace
replay, and one measured replay (`test_fused_decoder.py:201-212`). Candidate and
selected artifacts were collected at different times, not as interleaved A/B.
The work log also cites an older selected value of 2556.383 us, already showing
drift on the scale relevant to the claimed lead.

Repeated A/B is required **if the report continues to claim that the selected
path beats every correct candidate**. Use the same process and warmed regime,
randomized or ABBA ordering, at least ten measured samples per variant, and
report median plus spread/confidence interval. If the distributions overlap,
call the variants tied and choose based on structural simplicity or risk; do not
claim a performance win. This repetition is not required to prove functional
correctness, but it is required for the current ordering claim.

## Revalidation of headline claims

- Re-read `FusedDecoder._decode_attention`: both positions are device-tensor
  consumers, so no static stale scalar was found. Finding 1 remains a coverage
  hypothesis rather than an implementation accusation.
- Re-read TTNN validators and the 1D factory: no Mt<=4 contract exists, and
  `out_block_h` is explicitly allowed to divide `per_core_M`. Finding 2 remains
  headline-worthy, while every proposed configuration is labeled unverified.
- Rechecked current hashes, mtimes, report console destinations, and work-log
  values. Findings 3 and 4 directly match retained artifacts.
- None of these findings proves the model's passing PCC values wrong. They
  explain why the Stage 02 fusion/trace/evidence contract is not yet closed.

## Ranked AutoFix experiment plan

1. **Trace semantics:** add the one-line-style fused wrapper; run four mutable
   cases, then watcher; isolate individual buffers only on failure.
2. **Long GELU:** admission/PCC/op-row test B0 versus F4; try block heights 2/1
   only if needed; then C2048/C1024/C128 and long end-to-end selection.
3. **Evidence refresh:** freeze final source, bind all four source/test hashes,
   rerun standard, mutable watcher, both long gates, and affected profilers;
   generate destination-consistent reports and a manifest.
4. **Fragile candidate:** run interleaved repeated A/B before retaining the
   “beats every candidate” claim; otherwise report a tie.
