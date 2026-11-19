# Recurrent value-column ownership candidate

CPU-prepared, opt-in experiment; no device result or speedup is claimed.
The preceding state-streaming smoke was exact but its supplied 132.719 versus
132.594 us timing difference is insufficient evidence to select it. This
candidate changes ownership instead, and does not include state streaming.

New artifacts:

- [Source and host-wrapper transformations](../../tt/recurrent_column_candidate.py)
- [Standalone probe](../../tests/probe_multichip_column_recurrent.py)

The matched baseline is `cumulative_exp_scalar`. The candidate keeps the same
math and four-K-row reduction order for each independently owned value column.
It changes only who owns the column, the corresponding state reads, and the
state/output writes. Model files, policy defaults and existing probes are
unchanged. The wrapper uses fixed private function namespaces, not global
writer replacement or per-call monkeypatching.

## CPU comparison before implementation

Contiguous work ranges are assigned in units of one column (`columns`) or two
adjacent columns (`pairs`). The latter is the first recommended B32 experiment:
it has the same maximum arithmetic work with slightly fewer duplicated heads
and Q/K preparations. A full head is four columns.

| B32 on 110 workers | Max columns/core | Total head fragments | Max fragments/core | Q/K preparations | Max Q/K preparations/core |
|---|---:|---:|---:|---:|---:|
| Existing head ownership | 16 | 384 | 4 | 164 | 2 |
| Pairs | 14 | 438 | 4 | 218 | 2 |
| Individual columns | 14 | 440 | 4 | 220 | 2 |

The 12.5% decrease in maximum column work is not a latency prediction. Pairs
increase total Q/K preparations by about 33%, plus duplicated V/gate reads and
per-head decay/normalization work. They can lose despite the improved balance.
This is why the experiment starts with a matched focused smoke.

At B1 existing head ownership uses 12 cores/four columns each; pairs use 24
cores/two columns and individual columns 48 cores/one column. Q/K preparation
counts increase from 12 to 24 or 48. At B3 pairs use 72 cores/two columns;
individual columns use 110 cores/at most two columns, with more duplicated
preparation. Both alternatives remain available to measure the tradeoff.

## Source and ownership contract

`column_partition` returns `(first_head, head_count, first_n, last_n_exclusive)`
per owner. Only the first/last head use boundary masks; interior heads use all
four columns. For a single-head fragment both masks apply to that same head.

- Reader runtime arguments append the two boundaries at indices 8/9; compute
  appends them at 2/3; writer at 4/5. Compile-time tensor-accessor offsets and
  every CB descriptor stay unchanged.
- Reader fetches only state pages `16*h + 4*m + n` for owned `n`, into their
  original CB5 offsets. Reading a whole shared head would race another owner's
  in-place persistence and waste bandwidth, so that shortcut is excluded.
- Compute changes only the bounds of its outer value-column loop. All helpers,
  m0..3 reduction/accumulation, BF16 native persistence rounding and FP32 live
  readout remain byte-identical after removing the two bounds declarations.
- CB5 reserves/publishes 16 slots per head fragment, CB11 likewise 16, and CB12
  four. Unowned slots are never loaded by compute or read by writer. This
  preserves the proven fixed-size FIFO protocol and avoids changing ring
  reserve/pop semantics while testing ownership.
- Writer persists only owned matrix pages. Useful output writes own distinct
  64-byte face-row ranges. Last-head zero-padding writes are restricted to the
  same owned `n` range; otherwise two final-head fragments would duplicate
  writes. Writer retains its original final NoC barrier before popping CBs.
- Grouped Q/K still prepares at each core's first head or global triple start,
  and releases at triple/core end. Multiple owners of one head independently
  prepare identical Q/K. Gate caching retains per-user/core-boundary behavior.

Across owners, every state tile and useful/padded output range has exactly one
writer. Each state tile also has exactly one reader. No cross-core reduction,
additional tensor gather, persistent scratch, or state-layout change is added.
BF16/FP32 state uses the original state byte size and compute configuration;
output remains FP32 with the existing padded tile layout.

The host implementation is a private compiled snapshot of the current
`_local12_recurrent`, changed only for partition selection, three source strings
and boundary runtime arguments. The probe reuses the recurrent fixtures/gates
in a private namespace. It warms both operations before either capture and
preallocates every input, output and state; allocation-tracker exemptions are
not used. Reports include hashes of the actual reader/compute/writer triples,
the generated host/main snapshots, baseline partition and candidate partition.

## Parent-run sequence

Run under the parent's serialized hardware wrapper. CPU checks first:

```bash
python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_column_recurrent --cpu-check
```

First focused B32 pairs smoke:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_column_recurrent \
  --variants pairs --batches 32 --state-dtypes bf16 --modes normal --low-cores 0 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/recurrent_pairs_smoke.json
```

Compare both grains at B1 and B32 if the direction merits continuing:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_column_recurrent \
  --variants pairs columns --batches 1 32 --state-dtypes bf16 --modes normal --low-cores 0 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/recurrent_columns_compare.json
```

Broader exactness and wrap controls for a promising configuration:

```bash
TT_METAL_TRACE_ALLOC_TRACKING=1 python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_column_recurrent \
  --variants pairs columns --batches 1 2 3 17 31 32 --state-dtypes bf16 fp32 \
  --modes normal zero_qkv zero_state \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/recurrent_columns_suite.json
TT_METAL_TRACE_ALLOC_TRACKING=1 python -m models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_column_recurrent \
  --variants pairs columns --batches 3 32 --cores 7 --low-cores 0 --state-dtypes bf16 fp32 \
  --modes normal zero_qkv zero_state \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/recurrent_columns_wrap7.json
```

The default B3/one-core extra case exercises ordinary full-head wraps and user
boundaries with identical ownership on both sides. Seven-core cases additionally
combine partial boundary heads with multiple full heads and repeated FIFO wraps.
`--grouped-qk off` provides an independent grouping control. Existing
`--poison-dead-padding` poisons raw V/gate padding and disables timing; it does
not separately poison unowned state-CB slots.

All cases require exact finite output **and state** after each eager step,
changed-input replay equality against saved eager values, and equality after
matched timed replay counts. Timings cover the complete recurrence operation,
not a selected subkernel. Watcher and uninstrumented timing remain separate
parent runs. A focused win must still pass full-layer gates and profiling
before a production route is proposed.

## CPU evidence and remaining uncertainty

- 480 B1–32/core-limit/grain ownership cases cover every state page exactly
  once and prove disjoint 64-byte useful/padded output write ranges.
- Four grouping/poison source combinations restore the complete baseline
  compute text exactly after removing the column-bound changes.
- 36 descriptor-only executions of the real host snapshot check all reader,
  writer and compute runtime arguments, source selection, preallocated output,
  state byte size and unchanged CB capacities for BF16/FP32 and both grains.
- Private baseline binding survives mutation of the original test namespace;
  the two candidate functions retain independent fixed grain bindings.
- Python compile/import, CLI preparation, Black/isort checks pass without
  importing TTNN. No RISC/compute compilation or device test was performed by
  this worker. Unowned-slot assumptions and parallel NoC behavior require the
  exactness/watcher gates above; no achieved gain is claimed.
