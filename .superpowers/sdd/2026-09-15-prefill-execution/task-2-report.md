# Task 2 report: indexed device RoPE

## Scope and base

- Repository: `/data/divanovic/llama31-8b-disagg/repos/tt-metal`
- Branch: `divanovic/llama31-8b-disagg`
- Task base and prerequisite host-RoPE commit: `4bac36b20e5fe8d94a22678838d867e27c36e7ff`
- Owned implementation: `models/demos/llama_3p1_8b_d_p/tt/rope.py`
- Owned tests: `models/demos/llama_3p1_8b_d_p/tests/unit/test_rope.py` and
  `models/demos/llama_3p1_8b_d_p/tests/unit/test_indexed_rope_vs_ref.py`
- Required synchronization audit read before hardware execution:
  `/data/divanovic/llama31-8b-disagg/evidence/rope-kernel-audit.md`

No normalization, MLP, attention, KV-cache migration, Blaze, metadata, or tracing code was changed.

## Changes

- Added host-side capacity calculation for indexed RoPE physical reads. With logical
  `max_seq_len=2048` and `chunk_size=1024`, the persistent table capacity is 3072 rows. This does
  not increase the logical serving limit.
- Added geometry validation for a positive 2-D mesh, SP axis 0/1, positive sequence/chunk limits,
  SP divisibility, and tile-aligned local chunks.
- Added a replicated BF16 single-tile transformation-matrix builder.
- Added persistent Llama3 cos/sin setup using the Task 1 HF-matched frequencies, Meta adjacent-pair
  layout, the shared architecture-neutral `models.common.utils.block_cyclic_reorder`, SP sequence
  sharding, TP replication, and DRAM placement.
- Added a narrow scalar indexed-apply helper that forwards `kv_actual_global` and `sp_axis` to the
  stock `ttnn.experimental.deepseek_prefill.rotary_embedding_indexed` operator.
- Kept all TTNN imports local so Task 1 host helpers remain importable without native TTNN.
- Added plain-language comments immediately before every test in the two owned test files. Each
  comment explains the behavior, expected result, and regression caught.

## Test-first evidence

The focused host tests were written before production code. The RED command was:

```bash
source /data/divanovic/llama31-8b-disagg/tools/prefill_env.sh
export PRE_COMMIT_HOME=/data/divanovic/.cache/pre-commit
cd /data/divanovic/llama31-8b-disagg/repos/tt-metal
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PREFILL_PYTHON" -m pytest \
  models/demos/llama_3p1_8b_d_p/tests/unit/test_rope.py \
  --noconftest --rootdir=. -c /dev/null -q
```

Result: 6 failed and 11 passed in 72.41 seconds. Every new failure was the intended missing-feature
failure: `indexed_rope_table_capacity` or `build_indexed_rope` did not exist. Log:
`/data/divanovic/llama31-8b-disagg/evidence/task-2-host-red.log`.

After the minimal implementation, the same focused host suite passed 17/17. After hook formatting,
the fresh final host run again passed 17/17 in 105.18 seconds. Final log:
`/data/divanovic/llama31-8b-disagg/evidence/task-2-host-final.log`.

The Galaxy test was also authored before implementation. Its independent oracle does not call the
production reorder, frequency, coordinate-conversion, or indexed-apply helpers to derive expected
outputs. It uses Hugging Face `LlamaRotaryEmbedding` and `apply_rotary_pos_emb` with full-precision
Llama3 math; test-local HF-to-Meta conversion; and explicit global-interval enumeration with owner
`(position // 256) % 4`, preserving per-owner encounter order.

## Galaxy command and result

The final numerical run used the assigned allocation and exact prepared environment:

```bash
srun --overlap --jobid 104525 --nodes=1 --ntasks=1 --cpu-bind=none \
  -w bh-glx-110-c07u20 bash /data/divanovic/llama31-8b-disagg/task-2-device-test.sh
```

The script sources `/data/divanovic/llama31-8b-disagg/tools/prefill_env.sh` and runs:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PREFILL_PYTHON" -m pytest \
  models/demos/llama_3p1_8b_d_p/tests/unit/test_indexed_rope_vs_ref.py \
  --rootdir=. -c /dev/null -q -s
```

Result: 1 passed, no skips, in 49.28 seconds on a real `(4,8)` Blackhole Galaxy. All 32 chips were
checked independently for Q `(1,32,1024,128)` and K `(1,8,1024,128)`, with BF16 inputs and tables.
The table shards were read back once and checked against independent HF cos/sin for SP ownership and
TP replication. Starts `0, 32, 224, 256, 768, 1024, 2016`, then `224` again, ran in one mesh session
at fixed shapes. The program-cache entry count stayed fixed after the first Q/K programs. Start 2016
checked every physical output row through position 3039; only `[2016,2048)` will be logically valid
to the later runtime. Start 1 raised the stock operator's expected tile-alignment error.

Final log: `/data/divanovic/llama31-8b-disagg/evidence/task-2-device.log`.

| start | Q min PCC | Q max normalized L2 | K min PCC | K max normalized L2 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.9999944 | 0.0029949 | 0.9999955 | 0.0030378 |
| 32 | 0.9999944 | 0.0029949 | 0.9999961 | 0.0030378 |
| 224 | 0.9999944 | 0.0030068 | 0.9999961 | 0.0030378 |
| 256 | 0.9999939 | 0.0030310 | 0.9999955 | 0.0030527 |
| 768 | 0.9999939 | 0.0030848 | 0.9999955 | 0.0031143 |
| 1024 | 0.9999939 | 0.0030954 | 0.9999946 | 0.0031225 |
| 2016 | 0.9999936 | 0.0031792 | 0.9999947 | 0.0031986 |
| 224 return | 0.9999944 | 0.0030068 | 0.9999961 | 0.0030378 |

Every result cleared the unchanged thresholds PCC >= 0.9999 and normalized L2 <= 0.01. The first
hardware run passed too, but emitted two unknown-marker warnings because `-c /dev/null` intentionally
disables marker registration. The unnecessary topology/timeout markers were removed and the final
run above has only the existing third-party Pydantic V2 deprecation warning from the environment; it
was recorded and not masked.

## Hooks and self-review

The required hook command was:

```bash
source /data/divanovic/llama31-8b-disagg/tools/prefill_env.sh
cd /data/divanovic/llama31-8b-disagg/repos/tt-metal
"$PREFILL_PYTHON" -m pre_commit run --files \
  models/demos/llama_3p1_8b_d_p/tt/rope.py \
  models/demos/llama_3p1_8b_d_p/tests/unit/test_rope.py \
  models/demos/llama_3p1_8b_d_p/tests/unit/test_indexed_rope_vs_ref.py \
  .superpowers/sdd/2026-09-15-prefill-execution/task-2-report.md
```

An initial direct `pre-commit` attempt reported that the executable was not on `PATH`; the prepared
Python environment's module entry point above was used instead. The first module run reformatted the
three Python files with Black and therefore exited 1; all other applicable hooks passed. A first
`git commit` attempt created no commit because its hook defaulted to the full login-home cache and
failed with `ENOSPC`. Setting the project-provided `PRE_COMMIT_HOME` above fixed that environment
error. The final run after formatting and requested comments passed every applicable hook from the
project cache.

Self-review checked `git diff --check`, import locality, capacity math, 4x8 mapper axes, Q/K head
slices on every chip, independent expected-position construction, full physical tail coverage,
runtime-argument reuse, threshold reporting, and scope. No generated files or unrelated source edits
were found.

## Limitations

- This is the first eager scalar-offset path. Metadata tensors and tracing remain intentionally
  deferred.
- `apply_indexed_rope` does not enforce the logical 2048-token limit. The future runtime owns that
  bound; the stock TTNN operator enforces physical table bounds.
- The kernel audit notes a pre-existing compute startup-order TODO (#52395). This task adds no kernel
  changes; the real numerical Galaxy result above is the applicable correctness evidence.
