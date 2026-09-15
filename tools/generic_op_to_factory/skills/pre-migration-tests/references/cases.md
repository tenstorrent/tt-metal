# Choose cases from the source operation

Create tests for the current generic operation before its host code changes.
Keep the tests small enough to run and strong enough to catch a bad translation.

## The minimum areas to cover

| Area | Cases and assertions |
| --- | --- |
| API | Defaults, keywords, scalar conversions, optional arguments and wrong types. Call the exact source API. |
| Validation | Each independent current rule: shapes, dimensions, dtypes, layout, storage/device, grid, shard and optional/output constraints. Test valid cases just inside the limits too. |
| Output | Return count/container, logical/padded shape, dtype, tile/layout, memory config, shard shape/orientation and device/mesh placement. Check every returned tensor. |
| Memory effects | Supplied output address identity, separate output storage when promised, allowed aliases, in-place results and untouched inputs/regions. |
| Illegal writes | Watcher-supported address/CB checks plus exact preservation of protected inputs, retained outputs and guard regions where real placement can be verified. Exercise tail work and fresh-buffer/cache/alias transitions. |
| Poisoning / initialization | Same valid inputs with different allowed padding/guard poison; prefilled overwrite-only outputs; independently expected zero regions; history-dependent results after prior work. Distinguish missing initialization from allocation overreads and record inaccessible scratch/CB coverage. |
| Planner | Each descriptor/program branch: minimum work, even split, uneven split, idle cores, partial final work, and offset/disjoint grids if supported. |
| Numerical result | Independent reference on first and repeated calls. Test patterns that reveal scale, indexing and padding errors. |
| Cache | Repeated calls with different live buffers; changed program-defining fields; changed runtime values; optional and alias transitions. |
| Integration | A useful output consumer. Add trace, queues, multi-device or persistent state only when the current operation supports them. |

Do not copy another operation's factory counts, grid restrictions or dtype
support. Use its tests for ideas. Existing helpers and source examples live in:

- `tests/ttnn/unit_tests/operations/eltwise/test_typecast_program_cache.py`
- `tests/ttnn/nightly/unit_tests/operations/rand/test_uniform.py`
- `tests/ttnn/unit_tests/operations/fused/test_softmax_program_cache.py`
- `tests/ttnn/unit_tests/operations/fused/test_group_norm.py`
- `tests/ttnn/unit_tests/gtests/test_work_split_tilize.cpp`
- `tests/tests_common/cache_entries_counter.py`

## Test the cache as a sequence

Read the generic descriptor hash and the source descriptor builder. Explain why
each field needs a new program or can change at runtime. Neither shapes nor
scalars have one universal cache rule.

| Sequence | What must be true |
| --- | --- |
| A → A | The second call reuses a compatible program and remains correct. |
| A → new A → new A | Same metadata, different data and live addresses. Every output uses the current inputs and output buffer. |
| A → B → A | Change one program-defining field, then return. Expect the source's justified specialization and reuse. |
| scalar a → b → a | The output follows each scalar. Expected hits/misses follow how the source passes that scalar to kernels. |
| absent → present → absent | Optional branches are correct; their buffers/values are current. |
| separate → shared → separate | Allowed buffer sharing works in both directions. Also start a sequence with sharing. |
| valid → invalid → valid | A warmed call still rejects invalid state, and the next valid call works. Confirm which checks/cache path the invalid call actually reaches. |

Keep earlier tensors alive while creating new ones and assert different device
addresses for the relevant buffers. A fresh Python object can still reuse the
same freed device address. Use different data each time; a stale read should
visibly fail the golden comparison. Bound retained allocations to the device's
memory budget. An allowed in-place output naturally shares its input address;
assert that identity rather than demanding all addresses be distinct.

Measure cache-count changes only around the selected operation, after setup and
before readback. Conversion and `to_torch` can launch other operations.
`CacheEntriesCounter.reset()` only clears its counter, not the program cache.
Start from a known cache state and verify caching is actually enabled.

Do not impose “exactly one entry” on every operation. No-ops can create zero;
composites can create several. Identify setup/helper entries and record the
reason for each expected count. Never accept equality of counts as the only
proof of correct reuse.

The Python source may rebuild a descriptor on every call even while reusing a
cached device program. A test demanding that Python planning disappear on hits
would test a future optimization, not the current source contract.

## Illegal writes: check ownership as well as addresses

Use two complementary layers; neither proves every device write was legal:

1. **Watcher instrumentation.** For the eventual safety run, enable
   `TT_METAL_WATCHER=1` before starting the Python process (one-second polling),
   not after importing TTNN or opening the device. Verify the evaluated branch's
   implementation, architecture support and effective settings. Keep relevant
   sanitizers enabled; record attach/detach evidence, complete logs and process
   status with the test results. A fault, abort, timeout or incomplete run is not
   a pass. Do not enable `DUMP_ALL`, test mode or fault suppression for acceptance.
   Watcher instruments supported NoC APIs; it is not a general device ASan or a
   tensor-ownership checker. In this checkout, `debug_valid_dram_addr` checks the
   device DRAM range, not individual tensor allocations. `debug_valid_cb_addr`
   detects local transfers starting inside an active CB and extending beyond it;
   it does not cover arbitrary CPU stores, all remote writes or every architecture
   (the implementation notes Quasar's DFBs are not covered by this CB check).
2. **Protected-memory assertions in Python.** Take independent snapshots of
   read-only inputs and unrelated live tensors. After each synchronized call,
   check those and all retained earlier outputs exactly. Test partial final
   tiles/blocks and new live buffers on hits, including legal alias transitions.
   Explicitly exclude regions the contract allows to change. For partial writes,
   initialize the untouched region with a diagnostic sentinel and compare it
   afterward. Do not require unspecified output padding to remain unchanged.

For true output-allocation overruns, prefix/suffix canaries need verified physical
placement around the actual destination on each relevant bank/shard. Use an
existing supported allocation/view or low-level test helper if available. A
Python slice may copy, and allocating another tensor afterward does not prove
adjacency. Do not pass fabricated device addresses, scan unallocated memory or
change the operation's accepted shape merely to create a guard. If this checkout
cannot safely expose guarded storage, record the red-zone check as unverified;
ordinary unchanged-tensor checks are still useful but are not red zones.

Read back the actual protected bytes/regions; a logical `to_torch` result may
omit padding and guard storage. Value comparisons cannot detect writes that
leave or restore the original bytes. No Watcher fault plus intact sentinels means
no violation detected in the exercised cases, not a universal memory-safety proof.

Watcher may perturb timing and compilation. Keep safety evidence separate from
performance measurements. Reuse selected acceptance cases rather than adding
another full golden run. The current driver does not set or attest a Watcher
profile: record that integration as pending, not satisfied by ordinary pytest
success. Do not work around it using unrecorded inherited environment variables.

Inspect these paths in the evaluated checkout, not just the tooling branch:

- `docs/source/tt-metalium/tools/watcher.rst`
- `tt_metal/hw/inc/internal/debug/sanitize.h`
- `tt_metal/llrt/rtoptions.cpp` (effective environment parsing)
- `tests/tt_metal/tt_metal/debug_tools/watcher/test_sanitize.cpp`
- `tests/ttnn/unit_tests/operations/toy_spec_mul/test_toy_spec_mul.py`
  (`test_cache_hit_refreshes_scalar_runtime_args` shows sentinel checks for a
  contractually untouched tail, not an allocation red zone).

## Memory poisoning and missing initialization

Here poisoning means filling safely owned test memory with diagnostic contents,
not marking it inaccessible as an address sanitizer would. Pick cases from these
distinct contracts:

| Target | Setup and assertion | What a failure suggests |
| --- | --- | --- |
| Input padding that must not affect logical results | Keep logical inputs/metadata fixed; vary only permitted padding contents. Verify physical padding contains the pattern and each logical result matches the independent reference. | Padding dependence or incorrect tail masking; not necessarily a read outside the allocation. |
| Input-allocation red zones | With a verified guarded-allocation helper, vary prefix/suffix bytes outside the tensor but inside owned backing storage. Preserve the real tensor extent and verify bank/shard placement. Check the result against the same reference. | Dependence on bytes beyond the tensor's allocation boundary. A failure still needs localization. |
| Output the operation promises to overwrite | If a supplied-output API exists, prefill it with distinct nonzero patterns before separate calls. Each call must produce the full expected result, including exact zeros where promised. Check regions promised untouched separately. | Missing stores, missing zero fill, or accidental accumulation into old output. |
| Private scratch / accumulators requiring initialization | Use a verified test hook to poison owned scratch before operation-owned initialization, never after that initialization. Otherwise use prior-work sequences and report them as indirect coverage only. | Dependence on stale scratch, missing accumulator reset or another history-sensitive defect. |

Use multiple deterministic, dtype-representable nonzero patterns with different
signs/positions, chosen not to match the expected contents. Verify patterns remain
distinct after conversion/quantization. NaN/Inf are optional only where the
contract and format permit them; do not use them as the sole poison. Packed
formats can share exponents between logical and padded values: verify the logical
device inputs really stay identical, not just the original CPU input.

Compare each run to an independent reference, not merely two runs to each other.
For deterministic outputs also check poison invariance; for legitimately
nondeterministic operations use the established invariants/tolerances. Check
promised zero-filled regions directly (not only PCC). Do not demand zero in
unspecified padding, poison caller-required zero inputs/padding, or treat a
legitimate accumulation/in-place state input as discardable output storage.

Exercise first calls and cache hits with tail/partial-block cases, different live
buffers, and prior-work A → B → A sequences with contrasting data/work sizes.
Reinitialize mutable inputs and outputs between poison variants, synchronize
fills and calls, and keep setup outside cache-count measurement. For missing
zero-outs, choose supported inputs whose independent reference includes zeros;
do not assume zero input implies zero output for every operation.

Use supported tensor/allocation helpers only. `ttnn.from_torch(..., pad_value=...)`
is a candidate for tiled input padding in this checkout; verify its actual
layout/dtype path and read back physical padding before crediting coverage. A
conversion that sanitizes/replaces poison makes the experiment ineffective.
For fresh internally allocated outputs or private CB/scratch without safe hooks,
report direct poisoning as unverified. Allocator churn alone does not prove
reuse of the target address. Never fill all L1/DRAM, freed/unallocated memory,
semaphores, allocator metadata or runtime-owned state. Do not change production
kernels to add hooks as part of this test-authoring skill; report that separate
implementation need.

`TT_METAL_CLEAR_L1` clears memory on device initialization in this checkout; it is
not poisoning and can mask reliance on zeroed startup state. Record its effective
setting rather than enabling it to make these tests pass. A passing poison test
means no forbidden dependence was observed: unused/masked out-of-bounds reads
can leave the result unchanged, and a failure does not by itself identify the
offending read. Pair this with Watcher where supported and retain the same
instrumentation limitations as for illegal writes.

## Inputs and references

- Keep shapes around the source's actual tile, row, block and core thresholds.
  Examples such as T−1/T/T+1 apply only where legal. Exercise smallest supported
  cases, practical model cases and important upper limits within a stated budget.
- Use coordinate patterns, impulses and distinct regions for indexing/copy ops.
  Test dirty padding where the source supports it; padding must not contaminate
  the logical output.
- For scalar handling, choose values that expose sign, conversion and precision
  errors. Test NaN/Inf/overflow only according to the API's contract.
- Compute the reference from matching input values and the existing agreed
  precision convention. Preserve existing tolerances. Exact copies and index
  outputs need exact comparison; PCC alone can hide scale or offset errors.
- Check error type and a useful diagnostic with the repository's `expect_error`
  fixture. Exact wording, exception precedence or custom conversion behavior
  deserves an assertion only when the source contract promises it.
- For mutable/stateful ops, save independent expected state before calling the
  operation. Check updated values and all regions that should remain unchanged.

## Source planner evidence

For each meaningful planner branch, capture or inspect what the Python host
builds: participating cores, work counts/offsets, kernels and compile arguments,
CB formats/page sizes/capacity, semaphores, runtime argument roles, and output
connections. Prefer assertions about these invariants over an opaque whole-object
dump. A scoped test spy may observe the real descriptor and then forward the real
dispatch; restore it afterward.

If retaining snapshots for future comparison, normalize addresses to tensor
role plus byte offset and preserve alias groups. Keep raw addresses separately
for fresh-buffer checks. Do not sort argument arrays or remove code-defining
values to make snapshots agree. A captured source descriptor is source evidence,
not an independent mathematical oracle or a native comparison.

Use host-only tests for real pure planner functions when possible. Mocked device
or descriptor tests cannot prove real allocation, synchronization or numerical
results. Keep that limitation visible in the handoff.

## Keep failures honest

Unsupported configurations get explicit rejection tests where the API promises
rejection. Future `TARGET` cells remain outside current successful support; do
not count their skips as tested behavior. If the source accepts a configuration
but produces wrong results, keep a failing correctness test and report a source
bug. If it crashes or hangs on a rejected case, report that separately from an
ordinary expected exception. Do not catch every exception and call it success.

Missing hardware does not justify replacing execution with mocks, adding blanket
skips or reporting readiness. Deliver the authored tests and exact pending
commands, with status `unverified`.
