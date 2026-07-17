# AUTODEBUG: Gemma 4 31B dynamic decode batch crash

## Verdict

The crash is a deterministic host-side grid-planning bug in the generated
Gemma autoport. `MultichipDecoder._decode_attention_tp` assumes that every
active batch can be represented by one rectangular `CoreRange`. That is false
for active batch 19 on the measured P150 11x10 dispatch/storage worker grid,
so its divisor generator is empty and `max()` raises before any
attention-concat work is enqueued.

The minimal complete fix is the established Qwen autoport pattern: keep the
current rectangle for factorable batches, fall back to an exact row-wise
multi-range `CoreRangeSet` otherwise, and pass a full-grid `sub_core_grids` to
`nlp_concat_heads_decode` for those irregular inputs. Merely padding the core
allocation, or merely adding the multi-range fallback without
`sub_core_grids`, is not valid.

## Headline finding (high confidence)

### 1. Exact-rectangle selection has no candidate for batch 19

Runtime evidence is internally consistent:

- `models/autoports/google_gemma_4_31b/readiness_vllm/server.log` records
  `active_batch=19`, 19 running requests, and 13 requests that had just
  finished around lines 850-860.
- The traceback reaches
  `models/autoports/google_gemma_4_31b/tt/multichip_decoder.py:1019` and raises
  `ValueError: max() arg is an empty sequence` around server-log lines 901-905.
- The TTI command used `--batch_size 32` for the full 541-sample IFEval set in
  `.exp_run/tti-release/gemma4-31b-20260716/tti_release_final.log:68`; the
  failure is therefore the expected dynamic tail after individual completions,
  not an invalid request or context-length condition.

At `multichip_decoder.py:1016-1020`, the code searches for a divisor `x` such
that `batch_size % x == 0` and `batch_size // x <= grid.y`, then asks
`num_to_corerange` for one exact rectangle. A direct post-reset runtime query
of `mesh.compute_with_storage_grid_size()` on the reserved P150x4 mesh returned
`11x10`; that dispatch/storage grid is authoritative for this call. The SoC
descriptor lists 14 functional-worker coordinates in each of 10 rows
(`tt_metal/soc_descriptors/blackhole_140_arch.yaml:86-98`), but those physical
coordinates do not override the smaller runtime grid exposed to TTNN. For 19,
the only divisors are 1 and 19: `19x1` exceeds grid width and `1x19` exceeds grid
height. The filtered sequence is therefore empty.

`num_to_corerange` explicitly supports only a single exact rectangle and
asserts the factorization/bounds contract
(`models/tt_transformers/tt/model_config.py:4544-4597`). The current 1..32
matrix on the measured 11x10 worker grid is:

| Active batch | Current rectangle | Result |
|---:|---:|---|
| 1-11 | `batch x 1` | valid |
| 12 | `6x2` | valid |
| 13 | none | crash |
| 14 | `7x2` | valid |
| 15 | `5x3` | valid |
| 16 | `8x2` | valid |
| 17 | none | crash |
| 18 | `9x2` | valid |
| 19 | none | observed crash |
| 20 | `10x2` | valid |
| 21 | `7x3` | valid |
| 22 | `11x2` | valid |
| 23 | none | crash |
| 24 | `8x3` | valid |
| 25 | `5x5` | valid |
| 26 | none | crash |
| 27 | `9x3` | valid |
| 28 | `7x4` | valid |
| 29 | none | crash |
| 30 | `10x3` | valid |
| 31 | none | crash |
| 32 | `8x4` | valid |

Thus the same bug affects exactly active batches 13, 17, 19, 23, 26, 29, and
31 in the supported 1..32 range on the measured grid. In particular, the new
runtime evidence establishes 13 in addition to the observed 19; direct
factorization shows that 26 is unsupported by the same predicate as well.
Existing tests cover fixed batch 32 and a mocked
dynamic transition to batch 2, but do not exercise these grid geometries
(`tests/test_vllm_adapter_contract.py:166-189`).

## Complete causal chain

The intended invariant is one height-shard core per active user. It does not
require those cores to form one rectangle. TT Metal's
`num_cores_to_corerangeset(..., row_wise=True)` creates full rows plus a
partial final row (`tt_metal/common/work_split.cpp:80-147`). For batch 19 on an
11x10 grid this is exactly 19 cores: `(0,0)-(10,0)` plus `(0,1)-(7,1)`.

That fallback must be paired with the concat-op subcore path. The lowered
`nlp_concat_heads_decode` implementation:

1. Selects its subcore program factory whenever the input shard grid has more
   than one range (`nlp_concat_heads_decode_device_operation.cpp:133-145`).
2. Requires a provided `sub_core_grids` value in that mode and still requires
   the input grid to contain exactly one core per input user (lines 60-69).
3. Has a dedicated multi-range program factory that enumerates the actual
   input cores (`nlp_concat_heads_decode_subcoregrids_program_factory.cpp:69-83`).

Consequently, changing only `max(...)` to a row-wise fallback would advance to
a second validation failure because Gemma currently calls
`nlp_concat_heads_decode` without `sub_core_grids` at
`multichip_decoder.py:1030`.

Padding only the core allocation from 19 to 20 is also invalid: the concat op
requires input shard-grid core count to equal `input_shape[1]`, which is the
active user count. The slice at `multichip_decoder.py:1034-1037` removes the
concat op's output padding to 32; it does not authorize an extra input shard.
Padding the entire model batch/page-table/cache path would be a much larger
stateful change and is unnecessary.

## Minimal correctness-preserving fix

Use the already-proven sibling implementation from
`models/autoports/qwen_qwen3_4b/tt/optimized_decoder.py:32-58`, introduced by
commit `9019b51e2bd2a949e55a2664454a982f38fe14a9`:

1. Add an equivalent `_decode_head_core_grid(mesh_device, batch_size)` helper
   to the Gemma autoport. Preserve the current exact rectangle when a fitting
   divisor exists; otherwise return
   `ttnn.num_cores_to_corerangeset(batch_size, compute_grid, row_wise=True)`.
2. Add an equivalent `_decode_head_sub_core_grids(...)` helper. Return `None`
   for the normal single range starting at `(0,0)` and the full physical worker
   grid for a multi-range input.
3. Use the computed exact-batch core grid for `head_mem`.
4. Pass the matching `sub_core_grids` to
   `ttnn.experimental.nlp_concat_heads_decode`.

Qwen's TP4 multichip decoder uses this exact pair at
`models/autoports/qwen_qwen3_4b/tt/multichip_decoder.py:1033-1038`. It leaves
all already-working rectangular Gemma batches on their existing path and uses
the irregular-grid factory only for 13, 17, 19, 23, 26, 29, and 31 on the
measured runtime grid.

The same unsafe expression also exists in Gemma's single-chip
`optimized_decoder.py:670-674` and `fused_decoder.py:378-382`. Stage 11's
observed TP4 server imports `multichip_decoder.py`, so that is the immediate
root cause. A shared local helper should replace all duplicate sites to prevent
the same supported-batch defect in alternate generated autoport paths.

## Focused verification plan

No hardware reproduction is needed to prove the selection defect. The repair
should add cheap tests using a fake mesh whose
`compute_with_storage_grid_size()` returns `ttnn.CoreCoord(11, 10)`:

1. Parameterize every batch 1..32; assert core-grid construction never raises,
   stays within 11x10, and returns exactly `batch_size` cores.
2. Assert the rectangular dimensions listed above for all factorable cases.
3. Assert exact row-wise fallback ranges for 13 (`11+2`), 17 (`11+6`), 19
   (`11+8`), 23 (`22+1`), 26 (`22+4`), 29 (`22+7`), and 31 (`22+9`).
4. Assert the subcore helper returns `None` for rectangular cases and the full
   11x10 grid for exactly those seven multi-range cases.
5. Statically or with mocks assert the Gemma concat call receives the matching
   `sub_core_grids` argument.
6. After the CPU/static suite, run a bounded P150x4 decoder smoke for batches
   `[1, 13, 17, 19, 23, 26, 29, 31, 32]`, then repeat the 32-request staggered
   vLLM workload that previously transitioned to 19. Verify trace recapture,
   completion success, and the established output correctness/determinism gate.

## Remaining uncertainty

Static inspection proves the host exception and the required TTNN call
contract. Hardware verification is still required to validate numerical output
and trace replay for the newly exercised irregular-grid program factory on
Gemma's two attention kinds. No hardware, server, container, or production
source was modified during this autodebug investigation.
