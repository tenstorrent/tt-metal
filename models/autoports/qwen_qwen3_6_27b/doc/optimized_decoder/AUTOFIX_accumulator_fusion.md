# Retained-state accumulator fusion and delta broadcast reuse

This is an independent source-only performance hypothesis after the half-DST
candidate regressed. The parent reported halfdst74 around 2.4529 ms versus the
pipeline62 legacy path around 2.3566 ms, and the newer passing combo73 at
2.181616 ms compact decode with minimum 512-step PCC 0.9958221643. The recurrence
still contributes approximately 0.65–0.73 ms. The candidate below keeps the
retained-DST schedule and removes avoidable movement inside its arithmetic.
It is not yet device-validated or known to improve latency.

## Artifacts

- `candidates/gdn_sfpu_accumulator_fused.py`: standalone recurrence helper with
  the existing `_recurrent_sfpu_compact` API.
- `candidates/accumulator_on73.py`: frozen passing-combo baseline plus overrides.
- `candidates/accumulator_on78.py`: corresponding full78-based candidate.
- `candidates/make_accumulator_candidate.py`: immutable candidate generator with
  `--variant both|accumulator|hoist|control` for independent measurements.
- `candidates/check_accumulator_fused_cpu.py` and `accumulator_fused_cpu.json`:
  source, aliasing, arithmetic-order, and state-boundary checks.

The recurrence pieces overridden by the helper are AST-identical in combo73 and
full78. Thus accumulator_on73 isolates this recurrence change without inheriting
other full78 changes. No existing frozen decoder, main decoder, or shared source
was edited.

## Two avoidable operations

For each of four value-column tiles, the original prediction contracts four
state K-row tiles with K. For m=1,2,3 it writes a full FP32 product into temporary
DST4, then runs a separate full-tile add into accumulator DST6. Readout repeats
the same pattern with Q. The candidate calls the existing
`gdn_col_mul_add_to(m,5,6,6)` for those later K rows. Its established low-level
implementation uses separate SFPMUL then SFPADD, writing directly into DST6.
It does not introduce FMA or reorder the four K contributions.

This removes 3 products × 2 contractions × 4 value columns = 24 temporary-tile
store/reload pairs and separate add passes per head. Counting 4 KiB per FP32 tile,
the removed temporary writes and reads total 192 KiB of internal DST traffic per
head, or 288 MiB across 1536 heads at B32. This is a logical traffic count inside
the compute engine, not DRAM bandwidth or a measured saving.

The original update also calls `gdn_copy_row_to(6,4)` four times per value column.
The following update primitive reads DST4, loads K into DST5, and writes only the
corresponding state DST0–3. DST4 is invariant. The candidate broadcasts delta once
before the K-row loop and reuses it, reducing 16 broadcasts to four per head.
The associated broadcast setup and full-tile stores are avoided 12 times/head.

## Precision and alias safety

No low-level SFPU instruction body changed. The existing `_gdn_col_mul_add_`
already supports an output that aliases its residual: each band loads all
residual values before its first store. Prediction/readout use data DSTm,
column DST5, and residual/output DST6. These are distinct from each other except
the intended accumulator alias. The m=0 product initializes DST6 before any
accumulating call. The K-row accumulation order remains m=0,1,2,3, followed by
the same SFPU reduction.

The primitive adds product plus accumulator, whereas the previous generic add
used accumulator plus product. Finite FP32 addition is commutative; no grouping
or multiply/add fusion changes. Device own-input comparison remains necessary
because CPU algebra does not establish actual SFPU execution equivalence.

All BF16 persistent-state behavior is unchanged: incoming state widens to FP32,
current readout uses the unrounded FP32 updated state, and only afterward does
the existing explicit RNE conversion feed BF16 persistence. Q/K normalization,
grouped Q/K reuse, full-DST synchronization, two-head CB capacities, exact FP32
unpack policy, reader, writer, and runtime tensor registration are unchanged.
DRAM state traffic and memory capacity therefore remain unchanged.

## Checks and measurement controls

CPU/source checks pass for unchanged reader/writer/config/partition, unchanged
low-level SFPU primitive bodies, read-before-write accumulator aliasing, and
absence of FMA. CPU retained recurrence comparisons pass for normal, zero-KV,
and zero-state cases with FP32 and BF16 persistent state over eight steps. The
test explicitly checks that the hoisted delta tile remains unchanged through
all four updates. Python AST and inline-kernel braces pass. This does not claim
device compilation, accuracy, determinism, watcher cleanliness, or performance.

```sh
python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/check_accumulator_fused_cpu.py
```

Recommended first device check, through the parent's serialized queue:

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_cache_precision_probe --candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/accumulator_on73.py --rm --state-bf16 --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/accumulator_cache_probe.json
```

Then run normalized own-input control, short decoder correctness, unchanged
512-step sequence, and traced warmed decode. If combined performance is unclear,
freeze isolated controls instead of assuming fewer operations win:

```sh
python3 models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/make_accumulator_candidate.py --base combo73.py --variant accumulator --output accumulator_only_on73.py
python3 models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/make_accumulator_candidate.py --base combo73.py --variant hoist --output delta_hoist_on73.py
python3 models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/make_accumulator_candidate.py --base combo73.py --variant control --output accumulator_control_on73.py
```

The helper's two source flags become compile-time kernel arguments, so controls
do not introduce runtime host phase mutation or change the public API. A result
must beat the best correct incumbent to be kept. Four value-column workers per
head remain a distinct untried alternative: B32 already fills the device, and
that split duplicates Q/K/gate setup. It is not rejected by this source analysis;
it is a later hypothesis if the lower-movement candidate is insufficient.
