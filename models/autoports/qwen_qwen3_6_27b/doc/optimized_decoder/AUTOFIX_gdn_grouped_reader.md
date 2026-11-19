# Remove raw Q/K work already eliminated from grouped computation

After the parent's recurrence122 exact comparison passed, source inspection of
state packets and pipeline scheduling found an explicit redundant reader path:
compute prepares Q/K only once per aligned triple of value heads, but the reader
still fetches, widens, and publishes duplicate raw Q/K for both sibling heads.
Compute immediately discards those duplicates. This candidate removes that
producer/consumer work without changing any arithmetic or state traffic.

## Immutable artifacts

- `candidates/gdn_sfpu_grouped_reader.py`: standalone recurrence helper based on
  the update/readout fusion candidate.
- `candidates/make_gdn_grouped_reader_candidate.py`: integration generator.
- `candidates/grouped_reader_on_update_readout.py`: cumulative frozen integration.
- `candidates/check_gdn_grouped_reader_cpu.py` and `gdn_grouped_reader_cpu.json`:
  source, task provenance and abstract queue checks.

Main/shared source and existing candidates remain unchanged. The integrated
helper uses `_grouped_reader*` names and preserves all policies/defaults.

## Precise change and protocol

The host's existing `_head_partition` enables grouped Q/K only for B>=3 and
assigns complete aligned triples to cores. Within each triple, value heads
`hv=3k,3k+1,3k+2` all use exactly the same `(user,k)` raw query/key vectors.
Prepared CB13/14 already remain live through those three heads.

The reader now receives the existing `group_qk` selection as one additional
compile-time argument. TensorAccessor argument offsets shift consistently from
8 to 9. For each head it computes
`prepare_qk = !GROUP_QK || (h-start)%3==0`. When false, it skips only:

- reserving CB0/1 entries;
- q/k NoC reads or tiled face-row gathers;
- q/k BF16 widening/copies and CB0/1 publication.

State, V, g and beta reads, barriers, widening, ring reservation/publication and
all output writes remain as before. Compute removes the corresponding sibling
raw_q/raw_k wait/pop block. First-head `prepare_vector` still consumes each
published raw Q/K vector, and prepared Q/K lifetime/retirement is unchanged.
There is no changed numerical operation or reordered math instruction.

Raw Q/K rings retain their existing two-vector capacity. Both physical slots
are cleared at startup; only first-head vectors advance their pointers.
Reader and compute therefore each advance exactly once per triple, with no
unmatched publications or acknowledgments. State/value/gate rings still
advance once per head. Ungrouped mode, including the existing B1 policy,
continues to publish and consume Q/K every head.

For B32, grouped heads fall from 1536 raw Q/K preparations to 512. In the packed
ROW_MAJOR serving path this eliminates 2048 reads of 256 bytes and 262,144
scalar BF16-to-FP32 widening iterations. This is a bounded CPU/dataflow cost;
the work may already overlap SFPU compute, so no speedup is assumed.

## Host checks completed

Python syntax compilation passes. The checker proves writer-source identity
and complete compute-source identity except deletion of the redundant discard
block. It validates `(user, key-head)` provenance across all grouped triples,
including core boundaries and B1 ungrouped fallback. An abstract reader/compute/
writer queue model checks progress with the existing two-head input/output
capacities and two-vector raw Q/K rings. Cases cover B1/B3/B32, requested core
counts 1/8/32/110 and grouped enable/disable controls.

```sh
python3 models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/check_gdn_grouped_reader_cpu.py > models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/gdn_grouped_reader_cpu.json
```

The model is an abstract queue/provenance proof, not a device timing model. No
device was opened and no hardware result is attributed to this new variant.

## Parent-owned exact and performance checks

The existing frozen strict comparator supports both helper paths:

```sh
python -m models.autoports.qwen_qwen3_6_27b.tests.run_recurrence_exact_comparison --candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/gdn_sfpu_grouped_reader.py --reference models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/gdn_sfpu_update_readout.py --state-dtypes bf16 fp32 --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/grouped_reader_probe.json
```

Also use `--cores 1 --batches 3` to force repeated grouped ring wraps. Require
zero output/cache bit differences, then unchanged whole-decoder gate, watcher,
trace/cache behavior and measured latency. Keep this variant only if measured
throughput improves the accepted cumulative incumbent.

## Other packet/pipeline observations

State reader and writer already have two-head CB capacity and can overlap one
head's compute with the next read and previous write. Publishing four state
tiles earlier inside a head might shorten startup/drain, but does not remove
steady-state bytes and is not supported as a large throughput gain by source
alone. Prior value-column partition experiments are deliberately not repeated.

The 16 state pages/head are interleaved DRAM TILE pages. Treating them as one
contiguous 32 KiB NoC read would be incorrect: consecutive logical pages can
reside on different banks. Coalescing physical same-bank pages would require
bank-aware gather/scatter and proof of the accessor's mapping or a different
persistent state representation. No such change is justified merely by the
approximately 197 us compulsory state-I/O lower bound, so no speculative state
layout or packet-coalescing modification is included.
