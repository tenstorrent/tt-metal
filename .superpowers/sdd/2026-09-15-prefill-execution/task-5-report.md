# Task 5 report: Llama Q/K/V projection and packed KV cache

## Summary

PASS. The fixed 4x8 Galaxy Q/K/V projection and externally owned packed K/V cache are
implemented and validated on all 32 Blackhole chips. The final gate ran seven tests with no
skips: one structured and real checkpoint projection test, BF16/BF8_B allocation tests,
BF16/BF8_B exact placement tests, and BF16/BF8_B QKV-to-RoPE-to-cache composition tests.

Tested base commit: 7b12bf9d576dce3581f6aad19e44d6af43b0c45d.

Tested owned-file hashes:

- `tt/qkv.py`: cce1242a47f086faae7db51369c231b9ea80c97742a1e9c1b5e90adf40d99d82
- `tt/kv_cache.py`: 8db495f8ac8bc3c46db74d12dc50d7eb5a28a98169153871ec81791298600227
- `tests/unit/test_qkv_vs_ref.py`:
  45c9cd973364cda474cde37db7247661588efd37a8aab66fbf8af0c47a1fd2f6
- `tests/unit/test_kv_cache.py`:
  bfa7c1615c8f640d50bf1e2c382f7db06b06a91c241ce2a8ffbdfada338131b8

This report is included in the single Task 5 commit. The controller handoff records the final
commit SHA because a commit cannot contain its own hash.

## Changes

- Added `models/demos/llama_3p1_8b_d_p/tt/qkv.py`.
  - Validates the exact 4x8 mesh, SP4 axis 0, TP8 axis 1, device coverage, raw CPU HF tensor
    shapes, and BF16 TILE interleaved-DRAM activation contract.
  - Converts Q/K one 128-wide head at a time from HF to Meta adjacent-pair coordinates exactly
    once in the constructor. It keeps V raw and packs each TP group as Q_i(512)|K_i(128)|V_i(128)
    before column-parallel upload.
  - Runs the audited BF16 matmul and stock untied QKV split, then releases only the owned fused
    intermediate.
- Added `models/demos/llama_3p1_8b_d_p/tt/kv_cache.py`.
  - Allocates replicated zero K/V caches with per-device shape [64,1,512,128] and actual-bank
    DRAM NdShard [1,1,32,128], ROW_MAJOR, ROUND_ROBIN_1D.
  - Validates all eager scalar metadata, cache structure, and K/V inputs before mutation.
  - Uses the stock bounded writer with `valid_global=actual_end`, `cluster_axis=0`, and no
    `tp_axis`, followed by stock 32-token padding zeroing. It creates and releases staging copies
    only for cache dtype conversion.
- Added independent tests for structured and layer-0 checkpoint projection, allocator geometry,
  exact cache placement, program reuse with changed addresses, invalid/no-op metadata, and real
  QKV-to-indexed-RoPE-to-cache composition at [224,257) and [2016,2048).

The unimplemented adapter/runtime, `ModelArgs` loader, attention, decoder layer, full model,
migration, and Blaze code were not changed.

## Source and synchronization preflight

The required source audits were read before launch:

- `/data/divanovic/llama31-8b-disagg/evidence/qkv-split-typecast-preflight.md`
- `/data/divanovic/llama31-8b-disagg/evidence/kv-writer-preflight.md`
- `/data/divanovic/llama31-8b-disagg/evidence/kv-zeroing-preflight.md`
- `/data/divanovic/llama31-8b-disagg/evidence/mlp-matmul-preflight.md`

The matmul uses the same audited dense reuse/multicast path as Task 4 with three N columns. Its
four readiness/valid semaphores have matched participants, Blackhole multicast flushes precede
publication, cached addresses are overridden, and no dependency cycle was found. The QKV split
has balanced reader/writer traffic for Q16+K4+V4 pages per sequence-tile core, barriers before
CB publication/consumption, and no inter-core semaphore or fabric traffic. The BF16-to-BF8_B
typecast uses one page on each of 32 cores with balanced two-page input/output CB lifecycles.

The cache writer assigns four tiles to each of eight independent cores. Reader and writer use
the same work split, all reads/writes are barriered, and no TRISC, semaphore, or fabric dependency
exists. Padding zeroing has unconditional matched CB counts on reader, compute, and writer paths;
no-work chips follow the same counts, and the dependency graph is acyclic. The implementation
uses eager scalar metadata, avoiding the zeroing audit's unproven metadata-cache visibility path.

## Runtime configuration and L1

~~~text
mesh: 4x8, 32 chips; SP=4 axis 0; TP=8 axis 1
matmul: grid=3x8, per_core_M=1, per_core_N=8, in0_block_w=4,
        out_subblock_h=1, out_subblock_w=4, transpose_mcast=False,
        fused_activation=None, fuse_batch=False
compute: HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
split: num_heads=4, num_kv_heads=1, transpose_k_heads=False, kv_tied=False
cache: local [64,1,512,128], NdShard [1,1,32,128], actual DRAM banks,
       ROW_MAJOR, ROUND_ROBIN_1D
~~~

Named audited L1 maxima were 196,608 bytes/core for QKV matmul, 8,192 for QKV split,
6,272 for BF16-to-BF8_B typecast, 4,096/2,176 for the BF16/BF8_B writer, and
20,480/11,840 for BF16/BF8_B zeroing. Final live values before QKV execution:

~~~text
total_bytes_per_bank=1461248
largest_contiguous_bytes_free_per_bank=1461248
maximum_scoped_requirement=196608
headroom=1264640
~~~

## TDD and host evidence

RED command:

~~~bash
source /data/divanovic/llama31-8b-disagg/tools/prefill_env.sh
cd /data/divanovic/llama31-8b-disagg/repos/tt-metal
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 timeout 300 "$PREFILL_PYTHON" -m pytest \
  models/demos/llama_3p1_8b_d_p/tests/unit/test_qkv_vs_ref.py \
  models/demos/llama_3p1_8b_d_p/tests/unit/test_kv_cache.py \
  --collect-only -q --rootdir=. -c /dev/null
~~~

RED ran from 20:43:10 to 20:47:04 UTC and exited 2 with the two expected errors:

~~~text
ModuleNotFoundError: No module named 'models.demos.llama_3p1_8b_d_p.tt.qkv'
ModuleNotFoundError: No module named 'models.demos.llama_3p1_8b_d_p.tt.kv_cache'
no tests collected, 2 errors
~~~

Raw RED log:
`/data/divanovic/llama31-8b-disagg/evidence/task-5-qkv-cache/red.log`.

The same controlled collection command after implementation found exactly seven tests and exited
zero: `7 tests collected in 186.67s`. Raw host collection log:
`/data/divanovic/llama31-8b-disagg/evidence/task-5-qkv-cache/host-collect.log`.

## Galaxy command and final result

Allocation 105135 was RUNNING on `bh-glx-120-c03u14`. The final whole-process invocation held
the shared blocking lock:

~~~bash
srun --jobid=105135 --nodes=1 --ntasks=1 --exclusive --wait=0 bash -lc \
  'flock /data/divanovic/llama31-8b-disagg/tmp/prefill-device.lock \
   bash /data/divanovic/llama31-8b-disagg/tmp/task5_qkv_cache_run.sh' 2>&1 | \
  tee /data/divanovic/llama31-8b-disagg/evidence/task-5-qkv-cache/green-device-attempt7-final.log
~~~

The reviewed wrapper sourced `tools/prefill_env.sh`, selected the portable Python 3.10.19
environment and exact task-native build, disabled plugin autoload, and ran:

~~~bash
timeout 7200 "$PREFILL_PYTHON" -m pytest \
  models/demos/llama_3p1_8b_d_p/tests/unit/test_qkv_vs_ref.py \
  models/demos/llama_3p1_8b_d_p/tests/unit/test_kv_cache.py \
  --tb=native -p no:cacheprovider -v -s --rootdir=. -c /dev/null
~~~

Final result:

~~~text
TASK5_DEVICE_START_UTC=2026-09-15T21:27:46Z
7 passed, 1 warning in 85.15s (0:01:25)
Closing devices in cluster completed.
Cluster destructor completed.
TASK5_DEVICE_END_UTC=2026-09-15T21:29:41Z
TASK5_DEVICE_EXIT=0
~~~

There were no skips. Raw final GREEN log:
`/data/divanovic/llama31-8b-disagg/evidence/task-5-qkv-cache/green-device-attempt7-final.log`.

The repository hook then required replacing `pytest.raises` with the root `expect_error` fixture.
After formatting and that context-manager-only test change, the five affected hardware cases were
rerun; the two unchanged real composition cases remain covered by the seven-test run above:

~~~text
TASK5_EXPECT_ERROR_START_UTC=2026-09-15T21:53:44Z
collected 7 items / 2 deselected / 5 selected
5 passed, 2 deselected in 153.33s (0:02:33)
Closing devices in cluster completed.
Cluster destructor completed.
TASK5_EXPECT_ERROR_END_UTC=2026-09-15T21:56:56Z
TASK5_EXPECT_ERROR_EXIT=0
~~~

The log contains matching `[EXPECTED_ERROR BEGIN]` and `[EXPECTED_ERROR END]` records for every
converted assertion. Exact BF16 and BF8_B placement remained PCC 1.0/NL2 0 with exact untouched
and padding checks. Raw covering log:
`/data/divanovic/llama31-8b-disagg/evidence/task-5-qkv-cache/green-device-expect-error-rerun-final.log`.

### Numerical thresholds and final metrics

QKV and BF16 composition thresholds stayed PCC >= 0.9999 and normalized L2 <= 0.01.
BF8_B composition thresholds stayed PCC >= 0.999 and normalized L2 <= 0.02.

| Case | Minimum PCC | Maximum normalized L2 |
| --- | ---: | ---: |
| QKV structured synthetic | 0.9999991 | 0.0014718 |
| QKV real layer-0 checkpoint | 0.9999985 | 0.0017180 |
| QKV changed input | 0.9999988 | 0.0015881 |
| QKV exact zero | 1.0000000 | 0.0000000 |
| QKV structured return | 0.9999991 | 0.0014718 |
| BF16 exact tagged cache placement, written region only | 1.0000000 | 0.0000000 |
| BF8_B exact tagged cache placement, written region only | 1.0000000 | 0.0000000 |
| Real QKV + RoPE + BF16 cache | 0.9999940 | 0.0038669 |
| Real QKV + RoPE + BF8_B cache | 0.9999636 | 0.0094675 |

Placement uses 15-bit vectors built only from +/-32 and +/-64. The vectors uniquely encode all
eight heads and every generated physical source position through 3039. Those values are exactly
representable in both BF16 and BF8_B, so the exact equality result measures storage and placement
rather than input rounding. The oracle separately checked the written region, every untouched
cache element against +/-7 sentinels, and padding against exact zero on every chip. Thus the large
untouched cache volume cannot conceal an incorrect write. Composition metrics intentionally
measure projection, rotary, and BF8 storage rounding only over written elements; untouched
zero-initialized cache elements and padding were checked separately and exactly.

### Program-cache and address evidence

Both synthetic and real QKV weight instances were warmed before freezing the program count at six
entries; changed, zero, and return calls kept it at six. The first call used input address
`0x140080` and Q/K/V outputs `0x10c080`, `0x114080`, `0x116080` on each chip. Retained guards forced
the later calls to input `0x340080` and outputs `0x11c080`, `0x100080`, `0x106080`; the input/output
sets were disjoint and remained stable through changed and return calls.

For each cache dtype, three structural layer/range variants were warmed before freezing the
writer/typecast/zero program count. All eight placement scenarios then used separately retained
input allocations and a second disjoint cache allocation; varying slots, layers, starts, ends,
and physical-tail access left the count stable. Tests also proved K/V input contents and cache
handles unchanged. Final JIT summary was 87/87 hits with 2,864 build-once deduplications.

## Preserved attempts and resolutions

All attempts are retained under
`/data/divanovic/llama31-8b-disagg/evidence/task-5-qkv-cache/`:

- `green-device-attempt1.log`: pre-device wrapper argument corruption passed literal `+` paths;
  zero tests, exit 4. The owned wrapper was corrected and host collection rechecked its arguments.
- `green-device-attempt2.log`: 2 passed/5 failed, clean close. Four writer paths omitted required
  `valid_global=actual_end`; QKV froze its program count before warming the real weight instance.
  Production gained the required argument, and the test warmed both actual weights first.
- `green-device-attempt3.log`: 5 passed/2 failed, clean close. The shared verifier expected +/-7
  in composition caches that correctly began at zero. The verifier gained explicit sentinels.
- `green-device-attempt4.log`: pre-device `flock` tried to execute a non-executable wrapper;
  `Permission denied`, srun exit 69. Invocation changed to explicit `bash`; no device opened.
- `green-device-attempt5.log`: 7 passed and clean exit, but the BF8 placement fixture still
  measured source quantization. It was superseded by an exactly representable tagged fixture.
- `green-device-attempt6-final.log`: 6 passed/1 failed, clean close. The exact BF8 assertion found
  that V retained the old `+0.25` fixture offset. Removing that offset made K/V distinct by sign
  while preserving exact +/-32 and +/-64 values.
- `green-device-attempt7-final.log`: final exact-fixture code, 7 passed, clean close, exit zero.

No numerical tolerance, mesh geometry, cache geometry, or production algorithm was relaxed.

## Hooks and self-review

The prepared environment supplied these hook values exactly:

~~~text
PRE_COMMIT_HOME=/data/divanovic/.cache/pre-commit
TMPDIR=/data/divanovic/llama31-8b-disagg/tmp
PREFILL_PYTHON=/data/divanovic/llama31-8b-disagg/runtime/python_env/bin/python
~~~

The final hook logs are recorded at:

- `/data/divanovic/llama31-8b-disagg/evidence/task-5-qkv-cache/precommit-prepared-cache-code.log`
  (initial actionable failure requiring the `expect_error` conversion)
- `/data/divanovic/llama31-8b-disagg/evidence/task-5-qkv-cache/precommit-prepared-cache-code-final.log`
  (all applicable source/test hooks passed)
- `/data/divanovic/llama31-8b-disagg/evidence/task-5-qkv-cache/precommit-prepared-cache-report-final.log`
  (final report-inclusive run)

An earlier invocation mistakenly overrode `PRE_COMMIT_HOME` with
`/data/divanovic/llama31-8b-disagg/cache/pre-commit`. Its partially initialized, sparse/NUL-bearing
log is preserved as `precommit-initial.log`. After its verified task-owned parent stalled in a
cache-local checkout/virtualenv operation, the controller authorized SIGTERM of that exact parent.
No shared cache was cleaned, and all successful hook evidence uses the prepared environment above.

Self-review re-read all four owned source/test files against the brief. It confirmed per-head raw
HF Q/K conversion, TP group order, input and persistent-weight lifetime, exact program configs,
stock split/writer/zero arguments, all validation before mutation, actual DRAM bank geometry,
independent oracles, all-chip checks, physical tails, exact placement regions, changed addresses,
program reuse, and required explanatory test comments. The only production correction after first
hardware execution was restoring the brief's unconditional `valid_global=actual_end` writer
argument.

## Concerns and limitations

- The later `ModelArgs` loader must be changed to pass raw HF Q/K weights before integration; the
  constructor intentionally owns the only HF-to-Meta conversion. This update is deferred by the
  Task 5 brief.
- Pytest reports one installed Pydantic class-config deprecation from the shared environment. It
  does not affect these tests. Device clocks settled within the accepted 5% firmware range.
- These modules intentionally support only the fixed 4x8 Galaxy, 1024-token global chunks,
  two-user/32-layer/2048-token cache, and specified BF16 input layout. Attention and end-to-end
  integration remain later tasks.
