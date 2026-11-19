# Isolated recurrent Q/K live-band preparation

Source-only preparation for item3 of `remaining_bottleneck_audit25.md`. New files are `tt/recurrent_live_row_candidate.py` and `tests/probe_recurrent_live_row.py`; no production/default/native source is edited. No TTNN/torch imports, build, hardware execution or performance claims were made during preparation.

## Exact bounded change

Only the six pointwise operations inside the current `prepare_vector` change their traversal: square, ordered four-tile fold, epsilon add, accurate rsqrt, scalar normalization, and separate Q scale. The underlying native arithmetic primitives, fold order, epsilon bits0x358637bd, scale bits0x3db504f3, direct FP32 unpack, FP32 pack boundaries, REDUCE_ROW implementation, transpose and column expansion remain unchanged. All recurrence prediction/update/readout, state persistence, writer, CB sizes and head partitioning remain unchanged. The CPU check reverses the helper insertion and six substitutions and requires exact recovery of the entire original compute source.

The candidate deliberately processes one **four-row hardware band**, not a claimed single logical row. Source proof:

- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_sfpu_common.h` makes `VectorMode::R` invoke its callable on faces0/1. It does not reduce the callable's loop count.
- `ckernel_sfpu_binary_bcast.h` describes each SFPLOAD as4rows×8columns, addresses0/2 in face0 and16/18 in face1. Those four vectors cover all32 columns of logical row0 and three extra hardware rows.
- The native binary multiply/add and unary scalar primitives iterate `ITERATIONS` and advance `dst_reg` once per vector. The candidate uses ITERATIONS2 with VectorMode::R for square/fold/Qscale, preserving their first two vectors in each top face. This covers128 elements rather than1024, without predication or manual arithmetic reimplementation.
- Epsilon and rsqrt need only row0/column0, subsequently extracted by the existing scalar broadcast. Their native primitives run ITERATIONS1, VectorMode::None, covering the first32-lane vector. Accurate rsqrt uses unchanged APPROX=false and FAST_APPROX=false.
- Scalar normalization copies the incumbent helper with only half/band loop bounds changed from2×4 to1×1. Its scalar extraction, four loads/multiplies/stores and operand order remain identical.
- Native REDUCE_ROW's merged tree mixes accumulator storage but guarantees each row's result in its own column0; dead result columns are not normalized scalars. The unchanged transpose maps only original row0 into consumed column0; the existing bitwise-mask broadcast discards all other transposed columns. Poisoned dead rows explicitly test this dependency rather than assuming it.

The maximum pointwise traversal reduction is8× for full-width operations and32× for scalar epsilon/rsqrt. This is **not** a recurrence speed prediction: packs, transpose, reduction, state movement and recurrence arithmetic are unchanged. No persistent memory or context contract changes are introduced.

## Focused probe and gates

The standalone diagnostic uploads real BF16-exact FP32 row0 values in `[B*4,32,128]` with nonzero finite or NaN dead rows, invokes the unchanged full `prepare_vector` prefix versus transformed prefix, and exposes complete `[B*4,32,32]` prepared Q/K tiles. All prepared bits must match. A rebound owner simultaneously holds the opposite changed-input bank; its results must match that bank's baseline bits. Independent FP64 normalization checks each group/rank with PCC>=.995, and zero controls require exact zero. The corrected golden mapping is `normalized.reshape(tasks,4,32).transpose(1,2)[..., segment_map]`, with `segment_map[c] = 2*(c//16) + c%2`. Physical slots select even/odd logical columns within each face; see the preserved smoke diagnosis below.

The complete recurrence uses the current exact `cumulative_exp_scalar` source as control, checked against the frozen accepted source identity. B1/3/17/32, BF16/FP32 persistence, normal/zero-QKV/zero-state/epsilon-small/large/NaN-dead-padding modes are exposed. Small and large cases scale only Q/K by2^-18 or2^12; V, decay and state remain bounded. Every output and persistent-state bit must match the incumbent. All32 batches can be requested, plus grouped on/off and one-core B3 user/ring wraps. Minimum queued count50; `--queued512` supplies long changed-input/state coverage.

Inputs, outputs, initial/rebound states and all banks exist before each complete-recurrence trace pair. Every program/copy path is warmed. Tracking1/skip0 requires the exact unsafe-buffer positive control, then rejects no released buffers. There are no exemptions. Changed inputs/states, queued trace reuse and final eager interleaving followed by replay are checked. Standalone prepared outputs are eager/rebound diagnostics; complete recurrence traces exercise the same preparation inline. All traces are released before the next case's allocations. Instrumented and poisoned runs suppress timings.

Seven alternating synchronized trace samples report complete recurrence latency; these are not isolated-preparation-only savings or whole-decoder performance. Any candidate win still requires parent whole-decoder integration and final-default gates.

## CPU checks and initial parent command

`python3 -m py_compile` passes. `python3 -m models.autoports.qwen_qwen3_6_27b.tests.probe_recurrent_live_row --cpu-check` passes with no TTNN/torch module imports; saved result `recurrent_live_row25.cpu.json` contains four grouped/poison source transformations,192 head-partition cases and native primitive hashes. This does not compile SFPU or prove its execution.

```bash
D=models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder
env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  bash "$D/run_job.sh" recurrent_live_row_b1_smoke25 \
  python -m models.autoports.qwen_qwen3_6_27b.tests.probe_recurrent_live_row \
  --batches 1 --state-dtypes bf16 --modes normal --low-cores 0 --queued 50 \
  --output "$D/recurrent_live_row_b1_smoke25.json"
```

After bounded smoke, run B1/3/17/32 both states and all modes with one-core B3; then grouped-off, all32 batches and queued512 according to risk. Separate watcher from performance. Performance must unset watcher/profiler, disable tracking, and exclude poisoned cases; the default modes include poison and intentionally disable timing. The first compiler/API or bit mismatch is a diagnostic, not enough to dismiss the family: inspect actual vector addressing/rounding at exposed prepared Q/K before changing the unchanged recurrence.


## Preserved first smoke and corrected oracle (v2)

Parent's `recurrent_live_row_b1_smoke25.json`/`.log` failed normally and closed the mesh. The complete Q prepared tile matched baseline/candidate exactly on all4 ranks (zero bit mismatches), then the first baseline independent-oracle check returned0.444985–0.539055 over4 groups. No recurrence or timing gate was reached, and the original failed artifact is unchanged.

The harness incorrectly interpreted physical slot addresses0/2 as contiguous8-column logical blocks. Current Blackhole `ckernel_sfpu_reduce.h:100,528,544` explicitly identifies offsets0/2 as **even/odd logical columns** of a16-column face. The misleading contiguous-block wording in `ckernel_sfpu_binary_bcast.h` is insufficient for a host tensor mapping. `_gdn_expand_column_` maps segment s to physical offset `(s/2)*16 + (s%2)*2`; therefore logical column c consumes segment `2*(c//16) + c%2`. The wrong contiguous8 mapping matches half of the logical columns, consistent with the approximately0.5 observed correlations. This is a source-derived diagnosis; fresh hardware must verify it.

Only the new probe's independent oracle mapping and failure diagnostics changed. The candidate helper/kernel source, exact prepared-bit gate, >=.995 per-group/rank oracle threshold, recurrence/state exactness and trace gates are unchanged. The corrected probe also collects all oracle owner/rank failures before raising and saves actual/expected first rows when they fail. `recurrent_live_row25_v2.cpu.json` records the one-to-one inverse slot mapping and native reduction-source hash. Initial fresh parent command is the same as above with run/output stem `recurrent_live_row_b1_smoke25_v2`. No device execution by this agent.
