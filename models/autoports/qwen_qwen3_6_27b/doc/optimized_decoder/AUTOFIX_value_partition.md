# Disjoint value-column recurrence partition

This is an independent source-only hypothesis after accumulator fusion was
queued. It does not include accumulator fusion, alter existing frozen files, or
claim device correctness/performance. Its purpose is to measure whether finer
work distribution and state-page traffic scheduling outweigh duplicated vector
preparation in the retained FP32 recurrence.

## Artifacts

- `candidates/gdn_sfpu_value_partition.py`: standalone recurrence helper, default
  two value-column partitions, existing public runtime signature.
- `candidates/valuepart2_on73.py` and `valuepart4_on73.py`: frozen combo73-based
  candidates with two/four partitions.
- `candidates/make_value_partition_candidate.py`: creates immutable controls
  with `--partitions 1|2|4`.
- `candidates/check_value_partition_cpu.py` and `value_partition_cpu.json`:
  source, ownership, partition-boundary, and mathematical schedule checks.

The source constant `_GDN_VALUE_PARTITIONS` becomes a compile-time kernel
argument. All registered tensor buffers, state dtype, output dtype, and public
shapes remain unchanged. There is no mutable host phase or device host fallback.

## Ownership and unchanged arithmetic

The baseline assigns all four value-column tiles of each head to one core. This
candidate makes one virtual task own two or one value-column tiles, but still
retains all four K-row state tiles for each owned value column. It never splits
the K reduction or introduces a cross-core reduction.

Virtual task order is `(global K-head group, column partition, value head within
the group of three)`. For a task index `t` and partition count P:

```text
group = t // (3*P)
partition = (t // 3) % P
head = group*3 + t%3
first_value_column = partition*(4//P)
```

Each grouped core assignment starts and ends on three-task boundaries. The
existing Q/K normalization reuse therefore still serves three value heads with
the same Q/K and column partition. B1 preserves the original ungrouped setting.

The reader loads only the owned state tiles. The compute kernel loops only over
owned value columns; prediction, delta, state update, and readout use the same
four-K order and separate FP32 multiply/add operations as combo73. BF16 persistent
state is rounded only after that column's current FP32 readout. The writer
commits only owned state tiles and output-column rows. No other core reads or
writes those state tiles, so no global barrier is needed inside the operation.
The generic-op completion still covers all tasks before the next decoder step.

State, output, and raw input CBs retain the original two-head capacities and
per-task metadata counts. Inactive state/output slots in a task's CB region are
never read or written. This intentionally avoids additional local layout logic
in the first experiment. TensorAccessor compile-time offsets are adjusted for
the new partition argument, and compute receives the virtual starting task as a
second runtime argument. All external tensor addresses remain registered.

## Expected tradeoff

At B32 with grouped Q/K on 110 cores, the original maximum work is 60 value
columns per core. Two-way partitioning still has a 60-column maximum; four-way
partitioning reduces it to 57, only about a 5% balance improvement. Finer tasks
may change NoC bank contention and overlap, but that is unmeasured. Q/K, V, and
gate input loads and Q/K preparation are repeated for each partition; state
DRAM traffic is unchanged. Thus this experiment is not expected from source
alone to reduce a roughly 630-microsecond recurrence to 400 microseconds.
It should be kept only if measured traced decode beats the best correct
incumbent. B1 may benefit more from increased active cores, while B32 pays a
larger duplication cost. These are hypotheses, not a rejection without a run.

## Validation and commands

CPU ownership checks cover B1/B3/B32, P1/P2/P4, and core caps 64/80/110. Every
state page and output-column interval has exactly one owner, including tasks
that cross user boundaries. Grouped triples preserve one K-head and partition.
Eight mathematical recurrence steps with BF16 cache produce equal output/state
for one, two, and four partitions under identical per-column arithmetic order.
Python AST, inline braces, and changed TensorAccessor offsets pass. These checks
do not compile or execute device code or emulate SFPU approximations.

```sh
python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/check_value_partition_cpu.py
```

Recommended first device probe through the parent's serialized queue:

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_cache_precision_probe --candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/valuepart2_on73.py --rm --state-bf16 --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/valuepart2_cache_probe.json
```

Repeat with `valuepart4_on73.py`, and compare normalized inputs if needed.
Then run short decoder correctness and traced decode at B32 and B1. A winning
candidate needs the unchanged 512-step sequence and watcher before adoption.
The generator can create a P1 control with identical task-mapping machinery:

```sh
python3 models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/make_value_partition_candidate.py --base combo73.py --partitions 1 --output valuepart1_control_on73.py
```
