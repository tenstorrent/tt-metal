# Half-DST BF16-cache recurrence candidate

This is a source-only performance hypothesis, not a device pass. The parent
reported that the earlier FP32 retained-DST candidate took approximately 2.43 ms
per decoder step versus 2.29 ms for boundary32. The current faithful BF16-state
recurrence still occupies approximately 0.65–0.73 ms per step. These observations
justify testing whether half-DST synchronization regains math/pack overlap even
though it performs more L1 intermediate transfers. They do not predict a win:
BF16 persistence and current reader buffering differ from the earlier comparison.

## Artifacts and integration

- `candidates/gdn_sfpu_bf16_half_dst.py`: self-contained replacement recurrence
  helper with the same `_recurrent_sfpu_compact` API.
- `candidates/halfdst_on62.py`: frozen pipeline62 decoder plus these overrides.
- `candidates/make_half_dst_candidate.py`: creates a new frozen copy from another
  baseline and appends the helper; refuses to overwrite an existing file.
- `candidates/check_half_dst_cpu.py` and `half_dst_cpu.json`: CPU-only scheduling,
  grouping, precision-boundary, and source checks.

No main decoder, existing frozen candidate, or shared TTNN source was changed.
The helper inherits pipeline62's two-head raw-input, state-input, persistent
state-output, and readout buffers. It retains aligned three-value-head Q/K reuse
for batches that use it. All tensor addresses remain runtime arguments registered
with generic_op; state allocation stays stable and output is separate.

The helper supports both BF16 and FP32 state. BF16 is the intended performance
candidate. FP32 is an own-input control, not an advertised speed improvement.
All Q/K normalization, exponent, prediction, delta, update, and readout arithmetic
comes from the accurate boundary32 half-DST topology, using separate multiply and
add rather than FMA. Full-DST synchronization is explicitly disabled.

## Precision and buffer ownership

| CB | Meaning | Format and ownership |
| --- | --- | --- |
| 5 | Persistent input state | BF16 for the candidate; reader produces, compute consumes |
| 8 | Decayed state | FP32; compute produces and consumes |
| 9 | K prediction | FP32; compute produces and consumes |
| 10 | Value delta | FP32; compute produces and consumes |
| 11 | Updated state for current readout | FP32; compute produces and consumes |
| 12 | Current readout | FP32; compute produces, writer consumes |
| 19 | State to persist | BF16 for the candidate; compute produces, writer consumes |

Every consumed FP32 CB selects `UnpackToDestFp32`, including CB11. Only BF16
CB5 and CB19 use Default when persistent state is BF16. The state reader/writer
use 2048-byte tiles for BF16 and 4096 bytes for FP32. No conversion of the whole
state tensor is performed through DRAM.

Three independent input-state tiles and one decay scalar occupy the four FP32
tiles in half-DST during decay. This amortizes format switches without changing
the per-element arithmetic. The decay loop switches SrcA to BF16 CB5, loads at
most three tiles, restores FP32 SrcA/copy mode, and loads the scalar. All later
arithmetic uses FP32 CBs.

After update, readout contracts FP32 CB11 with Q. Only after that contraction is
complete does the compute kernel reload CB11 in groups of four tiles, apply the
explicit TTNN FP32-to-BF16 RNE operation, and pack CB19. Writer waits on CB19
before waiting for/consuming CB12, so it cannot publish a state/output pair before
the persistence phase is available. Compute then pops CB11. This avoids racing
the writer against the unrounded state used for readout and ensures that current
output does not use the rounded cache value.

CB19, CB5, and CB12 are two-head rings with capacities divisible by the unchanged
per-head push/pop counts. CB11 remains a single-head internal ring. Total CB
allocation is 512000 bytes (500 KiB) per core with BF16 state, within the intended
Blackhole L1 budget; actual allocation/compilation still needs device validation.
The arithmetic uses at most four live FP32 DST tiles. The extra FP32 intermediates
and final persistence pack are explicit performance costs, so this candidate must
beat the retained implementation in measured traced decode to be kept.

## Checks and recommended device probe

CPU checks pass for FP32 and BF16 persistent state, normal/zero-KV/zero-state
cases, and eight successive mathematical recurrence steps. The spilled and
retained schedules produce equal CPU results under the same operation order.
The test also demonstrates that rounding state before readout changes output,
so it checks the critical boundary rather than assuming it harmless. These tests
do not emulate SFPU approximations or establish device equality.

Source checks confirm half-DST config, FP32 direct unpack for intermediates,
readout-before-persistence order, writer ownership, and maximum four live DST
tiles. Group partition checks cover B1/B3/B32 at core caps 64/80/110, with complete
three-head groups where enabled. Python AST and inline-kernel brace checks pass.

```sh
python_env/bin/python models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/check_half_dst_cpu.py
```

The existing cache precision probe accepts the BF16 state and RM controls. Run
this through the parent's serialized device queue, then compare with pipeline62:

```sh
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_cache_precision_probe --candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/halfdst_on62.py --rm --state-bf16 --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/halfdst_cache_probe.json
```

Also run normalized input (`--state-bf16` without `--rm`) to separate raw Q/K
normalization from recurrence arithmetic. After own-input output/state checks,
the required evidence is short correctness, unchanged 512-step sequence,
traced warmed decode, and watcher. A compile/API failure needs repair and does
not itself reject the schedule.
