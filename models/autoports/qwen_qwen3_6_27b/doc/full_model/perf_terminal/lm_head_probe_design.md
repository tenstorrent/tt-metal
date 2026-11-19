# Same-precision DRAM-sharded LM-head probe

Prepared `tests/probe_full_lm_head.py` without device execution or production
changes. CPU `py_compile` passes. Parent-owned B1 followups below now provide
a measured disposition: retain the current head because two correct adapted
DRAM-sharded candidates are slower. This is not a production policy change
or a B32 performance claim.

The current head is TP4 vocabulary-sharded: each rank computes
`[1,B,5120] @ [5120,62080]`, with real BF16 weights, BF16 activations/output,
HiFi4, FP32 destination accumulation, packer accumulation, and no approximate
math. The probe loads the exact reduced `Model(layer_indices=[0,3])` to obtain
those head weights, final norm and unchanged common sampler. It does not run
the two decoder layers in the timed path or load all64 layers. Its baseline is
the actual `Model.terminal(hidden, decode=True)`.

## Existing implementation and shape contract

Read `/home/ttuser/qwen36-27b/tt-transformers/src/tt_transformers/modules/lm_head/lm_head_1d.py`.
`LMHead1D.forward` supports caller-provided DRAM-sharded programs, explicit L1
width-sharded inputs, width-sharded outputs converted back to the configured
interleaved memory, per-split logical-width trimming, and concatenation.
`_validate_lm_head_program_configs` explicitly separates output storage cores
from activation shards and DRAM readers. This is the appropriate operation
sequence/template for the experiment.

The simple LMHead1D defaults are not this model's selected contract: its default
weight/output datatype isBF8 and its default compute recipe isHiFi2 without
FP32 destination accumulation. A direct module integration must explicitly
supply BF16 and the existing `model.io.ckc`, DRAM weight configs, input config,
programs, and logical split widths. The probe mirrors the explicit operation
sequence using already resident TT weights; it does not introduce LazyWeight
checkpoint ownership or claim that the module API is unsupported. A successful
candidate could later be expressed through that module's explicit config.

The native reference is
`ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`:

- Around124–196, multi-reader operation is Blackhole-specific; each bank's
  width in tiles must be divisible by readers, and reader widths must exactly
  partition the physical matrix.
- Around280–310, weight circular buffers use three K-block slots. Blindly
  copying the decoder's block2 config to the full wide head can exhaust L1.
- Around695 onward, `per_core_N_storage` sets independently distributed output
  storage; it need not equal a reader's full width.

Existing `MultichipDecoder._prepare_matmuls` supplies the same eight-bank
weight sharding, eight activation shards, width padding and program family.
`tests/ttnn/unit_tests/operations/matmul/test_matmul_dram_sharded_multicore.py`
and `test_matmul_dram_sharded_multishard.py` exercise independent activation /
output layouts, BF16 and FP32-destination variants. These sources establish a
plausible legal contract, not device success for this new shape.

## Bounded candidate set

All candidates use eight input storage cores with shard `[32,640]`,
`per_core_M=1`, output-storage `per_core_N=32` tiles, row-major DRAM width
shards over eight banks, BF16 output, and unchanged `model.io.ckc`.
`use_multicore_pipeline=False` matches the selected established DRAM-reader
family; this is not a simultaneous pipeline/dtype/fidelity sweep.

| Name | Local splits | Logical/physical N per split | Readers/bank | K block tiles | Output storage cores/split |
| --- | ---: | ---: | ---: | ---: | ---: |
| whole_r2_k1 |1 |62080/62464 |2 |1 |61 |
| split2_r1_k1 |2 |31040/31232 |1 |1 |31 |
| split2_r2_k2 |2 |31040/31232 |2 |2 |31 |

62080 is tile-aligned but not divisible by `8*32` or `16*32`; a literal
unadjusted bank partition is illegal. Each candidate pads *within each mesh
rank's existing vocabulary shard*. It never changes the assignment of global
token IDs to ranks. Split weights are native device slices of the actual
resident BF16 head, with zero padding used only for storage legality.

Every split is converted to DRAM interleaved output and trimmed back to its
logical width **before** local concatenation and sampling. This removal is
mandatory: retaining zero-logit padded vocabulary columns could select invalid
IDs when legitimate logits are negative. The final local width remains62080,
so the current common sampler's offsets and negative-value65536 padding remain
unchanged. There is no full-logit gather inside the measured path.

`per_core_N=32` spreads output storage over61 or31 cores, avoiding a roughly
half-megabyte output shard on only eight cores. K blocks were deliberately
kept small: the largest proposed per-reader weight triple-buffer term is
`122*1*3*2048 = 749568` bytes (or `61*2*3*2048` for the last candidate), before
accumulation/output/input/kernel requirements. This is only a source-derived
term; it was not a fit guarantee. The later parent-owned admission failure
and successful adapted candidates are recorded below. Whole-head block2 was excluded from the initial three because
its weight triple-buffer alone would double that term.

## What the probe measures

The four activation cases are actual checkpoint embedding rows and three
independent deterministic BF16 random activation tensors, all passed through
the actual final norm. B1 is the default; `--batch 32` checks the same logical
batch/padded-tile contract.

All inputs, candidate weights, graphs and common sampler programs are warmed
before any trace capture. The probe retains one head/staging trace per arm and
one unchanged common sample trace. All cases run through these traces. It
checks source identity, unchanged program-cache count and trace IDs, per-row
logit PCC>=0.9999, exact greedy-ID agreement with the baseline, top5/top100
containment of the baseline greedy ID, and actual common-sampler output against
both baseline sampled IDs and each candidate's maximum logit. Host logits are
read only for diagnostic comparisons and never fed into sampling or timing.

Two timing scopes are reported across rotating candidate order and repeated
runs:

1. Final norm/gather, input reshard if required, projection(s), output movement,
   trim/concat, and the same pad/copy into stable common-sampler logits.
2. That entire boundary plus the common sample trace and token readback.

No baseline pre-normalized input or candidate-excluded layout conversion is
used to manufacture a projection-only win. Timed loops have no full-logit
readback. These are terminal microbenchmarks, not autonomous complete-model
throughput. Integration would still require the real autoregressive accuracy,
trace ownership, capacity, B32 and output-quality checks after any production
change. A failure records its candidate/phase and aborts the process; it is not
a fallback or permission to continue device experiments after a hardware fault.

Parent-owned suggested first command, after the normal environment and device
ownership checks:

```
source python_env/bin/activate
source /home/ttuser/qwen36-27b/lane-env.sh
TT_METAL_DEVICE_PROFILER=0 TT_METAL_TRACE_ALLOC_TRACKING=1 \
TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
python -u -m models.autoports.qwen_qwen3_6_27b.tests.probe_full_lm_head \
  --candidates whole_r2_k1 split2_r1_k1 split2_r2_k2 \
  --output models/autoports/qwen_qwen3_6_27b/doc/full_model/perf_terminal/lm_head_candidates_b1.json
```

The parent subsequently ran isolated candidates after the first admission
failure; this author ran no hardware. The successful adaptations establish
legal output sharding and numerical/greedy correctness on this native build,
but regress complete terminal time. No candidate is promoted. B32 candidate
performance remains unmeasured.

## Admission failure and source-grounded adaptation

Parent-owned `lm_head_candidates_b1.json` records normal compile/admission
failure of `whole_r2_k1`: statically allocated dataflow buffers extend to
1628096 bytes, above1572864-byte maximum L1. This first error is not a rejection
of the DRAM-sharded family. It is a specific wide-reader accumulation footprint.

`matmul_utilities.cpp:339`,
`ttnn::prim::dram_sharded_helpers::validate_num_workers_per_dram_bank`, permits
**only1..3 readers**. Four readers are not a legal current-API candidate. No
native change to extend that limit is proposed here.

The factory accounting explains the failed allocation. `per_core_N=32` controls
the tensor's output storage distribution; it does not constrain internal
per-reader compute/output/partial buffers. For BF16 output with FP32 destination,
`per_core_M=1` disables `compact_output_alias`, and differing data formats disable
ordinary output/intermediate aliasing (`factory.cpp:471–475`). Consequently both
BF16 output and FP32 partial circular buffers remain present. The subblock
selection around199–231 rounds reader compute columns up to a four-tile span.

For these shapes (`B*num_blocks>1`, no bias), the principal static terms are:

```
input double buffer = 2 * K_block_tiles * 2048
weight triple buffer = 3 * reader_N_tiles * K_block_tiles * 2048
output buffer = compute_N_tiles * 2048
FP32 partial buffer = compute_N_tiles * 4096
```

| Candidate | Reader/compute N tiles | Input | Weights | Output | FP32 partials | CB sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| whole_r2_k1 |122/124 |4096 |749568 |253952 |507904 |1515520 |
| whole_r3_k1 |81/84 |4096 |497664 |172032 |344064 |1017856 |
| split2_r1_k1 |122/124 |4096 |749568 |253952 |507904 |1515520 |
| split2_r2_k2 |61/64 |8192 |749568 |131072 |262144 |1150976 |

The observed failed extent1628096 minus the whole_r2 CB sum1515520 leaves
112576 bytes of static starting-region/alignment overhead. Reusing that
observed offset as an estimate predicts1130432 bytes for whole_r3 and1263552
for split2_r2. This offset reconciliation is not an independent full L1
inventory or a successful allocation claim: tensor L1 storage and exact core
placement must still be admitted by the actual program. It does explain why
simply spreading output storage did not rescue whole_r2. Changing
`per_core_N_storage` alone cannot remove its FP32 compute/partial demand.
`split2_r1_k1` has the same per-reader footprint as the failed whole_r2; no
separate hardware attempt is needed to rediscover that geometry limit.

Added exactly one new candidate, **whole_r3_k1**, using existing supported
three-reader placement. It pads local62080 to62208, splits the eight banks'
243 tiles each over three81-tile readers, uses Kblock1, and retains the same
BF16/HiFi4/FP32/packer contract. There is one projection and no two-piece concat.
Its remaining output trim is necessary to preserve62080 real local token IDs.
The parent executed the source-backed whole3 followup:

```
python -u -m models.autoports.qwen_qwen3_6_27b.tests.probe_full_lm_head \
  --candidates whole_r3_k1 \
  --output models/autoports/qwen_qwen3_6_27b/doc/full_model/perf_terminal/lm_head_whole3_b1.json
```

No device execution was performed by this author. Parent-owned adapted
`lm_head_split2_b1.json` already passed strict tracing and all four activation
cases with exact logits (`max_abs_error=0`, top1 agreement1.0), common-sampler
agreement, and unchanged program/trace identities. Its timing is slower:
current versus split2_r2 head/staging1.664973→1.810702ms; terminal plus sample
and readback2.202379→2.348599ms. This establishes a correct but slower adapted
packing result, not merely an uninvestigated API failure. Whole_r3 was the bounded followup testing whether removal of split/concat
changed that decision. It also passed and was slower, as recorded below.
No production model, sampler, decoder fidelity or datatype was changed.


## Final B1 decision after the whole3 run

`lm_head_whole3_b1.json` passes with strict allocation tracking enabled,
skip-program-cache0, profiler disabled, and unchanged91 program-cache entries
and trace IDs. Four activation cases pass the PCC>=0.9999 and exact greedy-ID
gates, including actual common-sampler agreement. Its maximum absolute logit
difference from current is0.03125; do not call this case bitwise exact.

Whole3 head/staging is2.145382ms versus its current1.665316ms; head plus common
sampling/readback is2.681253ms versus current2.203284ms. The preceding split2
run measured1.810702ms versus1.664973ms for head/staging and2.348599ms versus
2.202379ms with sampling/readback, with exact logits across all cases. These
are each same-process B1 comparisons at identical BF16/HiFi4/FP32-destination
precision and allocation-tracking regime. Layout conversion and trim/concat
costs are included.

**Retain the unchanged current head.** Both legal adapted candidates regress
terminal latency; the family received real adaptation and successful execution
after the initial L1 failure. This rejects the measured configurations, not
every conceivable DRAM-sharded geometry. No B32 candidate timing or performance
claim is made, and no precision relaxation or production change was used.
`lm_head_decision.json` gives machine-readable evidence paths, exact timings,
accuracy scope and native source references.
