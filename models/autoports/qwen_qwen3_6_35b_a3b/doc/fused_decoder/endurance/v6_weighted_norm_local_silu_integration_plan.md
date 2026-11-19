# Weighted normalization and local SiLU integration candidate

The aligned local-SiLU component has now passed **all 30 cases with zero FP32
gate and BF16 output bit differences**, including all 65,536 BF16 codes. Parent
evidence is `v6_silu_local_lookup_aligned_device.json` (clean close) and
`v6_silu_local_lookup_aligned_timing_device.json`. The clean timing run measured
0.484951590188 ms for existing lookup/multiply/concat and 0.042996001430 ms for
local lookup/multiply/packing, with identical output memory configuration and
first/last replay values. This is component evidence, not full decoder timing.

## New weighted interface and combined adapter gate

`tests/v6_weighted_norm_local_silu_probe.py`, SHA
`886c8c60d1df1a3e19e63284477f033e21143d125b16c4a5feaf8fb70e155b90`,
introduces `WeightedNormTail`. It preserves the exact native sum input and the
frozen helper's numerical prefix through normalization BF16 rounding, gamma
multiplication, and weighted BF16 rounding. The output carrier is FP32 with
those BF16 values. Only the gate input/copy, final gate multiplication and final
BF16 rounding are removed; they are handled by the already-tested local SiLU
component. The weighted output remains single-core height-sharded. The local
reader accesses it directly via TensorAccessor, without a placement copy.

`v6_weighted_norm_local_silu_cpu.json` records CPU setup for the same six actual
positions and eight random fixtures, arithmetic-prefix identity, source hashes,
and runtime host-conversion audit. The source and weighted compute kernel are
snapshotted alongside it. No hardware ran during preparation.

The parent-owned component gate compares the new weighted output and debug
normalized/inverse values bit-for-bit against the frozen `GatedNormTail` debug
values. It then compares the complete normalization/local-SiLU adapter with the
explicit native adapter. All checks are strict bit comparisons. Optional timing
compares explicit, current fused-tail, and new weighted/local adapters with five
samples of 100 warmed trace replays; every placement operation is included, and
output/determinism gates precede and follow timing.

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_weighted_norm_local_silu_probe.py --run-device --time --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_weighted_norm_local_silu_device.json
```

Omit `--time` for a watcher correctness run. Timing must run without watcher,
profiler, or competing device/host postprocessing work.

## Standalone decoder snapshot

`tests/v6_build_local_silu_snapshot.py` generated:

`v6_corrected_fma_full_v5_streaming_scan_local_silu_integration.py.snapshot`

SHA: `bd8ab8bd637eb028aef52a6856fdbb3316c34ba15bfdb7954697114f5b143932`.

Its base is the requested streaming-scan/norm-tail snapshot, SHA
`700c10db90b23fac1d40427c9bbcd82a6924c7704699d54768a0160c2b641e05`.
Only cache-boundary decode's normalization tail and final packing change. The
new branch computes weighted carriers, invokes the local lookup/packing helper,
then immediately invokes the existing streaming output projection. The original
public singleton branch remains unchanged. Multi-row prefill, chunk64/recurrent
state logic, full-attention/V5 paths, and public interfaces remain unchanged.

The builder restores the exact changed setup block, cache-boundary branch, two
helper nodes, and four appended lookup nodes, then proves the entire AST equals
the base. It also proves the local reader/helper class matches the device-tested
source apart from renamed static constants, and the weighted helper class
matches the component probe. All compute/reader strings are static constants;
there are no test imports or runtime source-building helpers in the snapshot.

The old gated helper is removed, rather than retained as an unused allocation.
The existing DRAM SiLU table remains necessary for prefill. Only linear layers
allocate the new 256 KiB/core replicated table; full-attention's early-return
setup is unchanged. The 8 MiB aggregate L1 cost remains subject to full-decoder
capacity testing and does not justify reducing the advertised context contract.

Exact build evidence: `v6_local_silu_integration_build.json`. Reproduction uses
the builder with fresh `--snapshot` and `--output` paths; it refuses to overwrite
evidence. No production source or measured helper was edited. The snapshot is
**not promoted**: weighted-component/adapter correctness, full decoder
correctness/stress/watcher, context capacity, and traced performance must still
pass before selection.

## Measured weighted gate and subsequent integration capacity failure

The parent subsequently completed
`v6_weighted_norm_local_silu_device.json`: all 14 cases passed exact weighted
FP32 output, normalized/inverse debug, and complete-adapter comparisons. The
device closed cleanly. Five samples of 100 warmed trace replays measured:

| Complete adapter | Median ms |
| --- | ---: |
| Explicit native | 0.528205567971 |
| Existing fused tail | 0.507356622256 |
| Weighted tail plus local lookup/packing | 0.064613223076 |

These are the same adapter boundaries and include placement copies. First/last
outputs were exact. The new weighted interface is therefore device-tested;
full decoder promotion remains a separate gate.

The parent then tested the bd8 standalone integration. It failed during
prefill grouped expert projection, before decode, because static CB storage
ended at address 1,174,208 while an allocated L1 buffer began at 1,060,800.
The 113,408-byte overlap is recorded in
`v6_local_silu_layer34_seed154.log`; the device closed cleanly. This confirms
that the full persistent LUT cannot be selected with the existing prefill
configuration. Neither MoE arithmetic/configuration nor the 262,144 context
contract is changed to accommodate it. The failed snapshot stays frozen.

`v6_local_silu_memory_audit.json` records source-derived allocation arithmetic.
The incremental LUT is 256 KiB on each of 32 workers. Weighted gamma remains
16 KiB on core 0; the weighted intermediate is 16 KiB, core/sum placement is
16/4 KiB, local reader rings total 18 KiB/worker, and final output is 8
KiB/worker. These are phase-specific payloads, not a measured peak or a complete
decoder allocation sum. Blackhole has 1.5 MiB physical L1/core before firmware,
small-L1 and other reservations. `BankManager::allocate_buffer` computes a
per-bank common-address allocation and checks allocator dependencies, so an
8 MiB physical payload can constrain more than 8 MiB of aggregate interleaved
address space. Source arithmetic alone cannot establish capacity.

Cache storage remains unchanged: 49,152-byte BF16 ROW_MAJOR convolution history
and 2,097,152-byte FP32 TILE recurrent state in DRAM. Full-attention setup, KV
dtype/page geometry, and absolute-position logic are unchanged. The new table
is context-independent, but its interference with bounded prefill scratch was
real; long-context validation remains mandatory after choosing the fix.

An inherited diagnostic retention was also found: `CorrectedFma.last_inputs`
holds its last three L1 operands, and the standalone snapshot never reads this
attribute. Its final prefill output matmul is `[1,32,64,128]`, retaining 3 MiB
aggregate, 48 KiB on each of 64 workers. This predates the bd8 change. Removing
that unused diagnostic ownership is a separate cleanup candidate, but alone
would not bridge the observed 113,408-byte overlap. It was not changed here.

## Remaining conversion and dispatch opportunities

The BF16 rounding followed by FP32 widening at the recurrent-core boundary is
required by the tested numerical contract. The reshape wrapper's request for
TILE layout is a no-op when already TILE (`to_layout_op.cpp:62` returns the
input). The two actual placement calls move core and sum to the current
single-core weighted kernel; they are required by that kernel binding. The
local lookup consumes its weighted shard directly and performs no tensor
conversion, reshard, or host fallback. The existing ordered output projection
still relocates to DRAM, untilizes, gathers swapped adjacent coefficients, and
retiles; that is a separate possible reader-fusion target.

A later isolated fused-tail experiment could retain the external native sum
and use one worker/head to fetch its core row, sum scalar, gamma row and Z codes.
Each worker would run the same stored FP32 variance/epsilon, native rsqrt and
column-multiply prefix, BF16-normalized/gamma/weighted rounding, then the same
lookup/final multiply/pack. This could remove the weighted intermediate, both
placement calls and one dispatch. It changes operand placement and lane
assignment, so its exactness is a hypothesis requiring the existing normalized,
inverse, weighted and final bit checks. No implementation or speedup is claimed
for this future experiment; memory capacity takes priority.

## Capacity-preserving lookup alternatives

`tests/v6_silu_dram_lookup_probe.py`, SHA
`162c061d2a00edb171662ad781823907436b24fdb418d6613c1a3ab48d0a3418`,
offers two zero-persistent-L1 alternatives. Both borrow the existing prefill
DRAM table, preserve the original compute kernel exactly, and use explicit
64-byte-aligned scratch reads. The parent measured both cleanly; all 30 cases
and all 65,536 codes have zero gate and final-output bit differences:

| Lookup/multiply/packing component | Existing median ms | Candidate median ms |
| --- | ---: | ---: |
| Stream full LUT into temporary CB | 0.485114711337 | 0.174429710023 |
| Fetch aligned 64-byte blocks | 0.485020079650 | 0.084588169120 |

Exact artifacts: `v6_silu_dram_stream_device.json` and
`v6_silu_dram_blocks_device.json`, plus their logs/source snapshots and CPU
setup reports. Stream mode reserves 260 KiB temporary CB/core during its kernel
and reads 8 MiB/decode. Block mode reserves 4 KiB temporary CB/core and reads
256 KiB/decode. Neither consumes LUT L1 while prefill executes. Blocks is the
measured fallback pending full decoder integration and capacity validation.

A third new probe, `tests/v6_silu_compact_lookup_probe.py`, SHA
`c546d954f88308658cf4af2b0f9ff481397e583b8d5e86e3cd30899d0a7b271c`,
stores the full-domain table exactly in 29,952 bytes/core. CPU derivation finds
positive exception codes `0x33c1..0x4185` (3,525 entries) and negative exceptions
`0xb440..0xc2b1` (3,698 entries), plus 256 explicit nonfinite entries. Other
outputs are constructed by raw integer FP32 `x/2`, positive identity, or signed
negative zero. Nine unused padding entries make 7,488 entries. Thresholds are
derived and verified against the frozen full table during setup, not guessed.

`v6_silu_compact_lookup_cpu.json` includes an independently compiled portable
C++ reconstruction proof: all 65,536 FP32 payloads match, including subnormals,
signed zeros and NaN payloads. Full-table SHA is
`f2cd8b89a24cfd6c18b45e49a86ef092baa81b08b7cb4730abe1f7b0eebe344c`;
padded compact-table SHA is
`658b28d6d40a2501f55854f610eabc6e3b5288b3980d7ac65b50ea29d97304fe`.
The compute kernel is unchanged. This saves 232,192 L1 bytes/worker relative to
bd8, more than the observed overlap. A same-order address-shift estimate leaves
118,784 bytes beyond that CB end, but this is explicitly **not** a hardware
capacity proof. Compact device correctness/timing and full decoder capacity
remain pending at this memo update. Parent command:

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_silu_compact_lookup_probe.py --run-device --time --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_silu_compact_lookup_device.json
```

## Compact component passed; new complete adapters and integrations

The parent completed `v6_silu_compact_lookup_device.json` with a clean close at
10:33:47 UTC. Every one of the 30 cases has **zero gate FP32 and output BF16 bit
differences**, including all 65,536 codes and nonfinite payloads. Clean warmed
trace timing measured existing lookup/multiply/concat at 0.484883780591 ms and
compact lookup/multiply/packing at 0.043936916627 ms. The 29,952-byte/core table
therefore preserves both the measured exactness and approximately the speed of
the larger table. It still needs full-decoder capacity and correctness gates.

The new complete-adapter probe is
`tests/v6_weighted_norm_compact_silu_probe.py`, SHA
`a4b77e7f22e8f0e7bd4122ca1458400ea1ef5efd02b135945b0e5d304b975cd6`.
It uses the already-tested weighted helper and substitutes only the frozen
compact lookup class. That class performs the exhaustive full-domain FP32
reconstruction verification in setup before uploading its compact table.
`v6_weighted_norm_compact_silu_cpu.json` records the 14-fixture CPU setup. The
parent-owned complete-adapter command is:

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_weighted_norm_compact_silu_probe.py --run-device --time --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_weighted_norm_compact_silu_device.json
```

`tests/v6_build_compact_silu_snapshot.py`, SHA
`6a2339e73b47508b5090b245aa8ca59615b4ac7156a1192566a3e02604bc4e5c`,
generated two frozen standalone candidates:

| Candidate | Base SHA prefix | Candidate SHA |
| --- | --- | --- |
| `v6_corrected_fma_full_v5_compact_silu_fused_matmul_integration.py.snapshot` | `8cda999c` | `951391e44ed08d1646f857091d74c011b3c93e0c8735f7a0d941890291fd5579` |
| `v6_corrected_fma_full_v5_direct_coeff_compact_silu_fused_matmul_integration.py.snapshot` | `c823e6f6` | `5ff93fd27348d9e49b6c09da6fd77a362243c7c575b4b92431232638247de242` |

The first preserves the verified fused-corrected-matmul base. The second also
preserves the other agent's direct-coefficient projection changes. Reports
`v6_compact_silu_integration_build.json` and
`v6_direct_coeff_compact_silu_integration_build.json` prove exact AST restoration
to each respective base outside the specific weighted-tail setup/decode/helper
regions. Prefill, public singleton, full-V5 paths and projection changes are
unchanged. Static reader/compute strings and setup-only compact verification
functions are embedded; no test imports or runtime source generation are used.

The memory audit's 3 MiB `CorrectedFma.last_inputs` retention described the bd8
source. The newer 8cda/c823 bases instantiate `FusedCorrectedMatmul` directly;
they do not instantiate that old diagnostic owner. Thus that retained-memory
finding must not be added to the new candidates' live allocation estimate.
No frozen source was changed to remove it. Both new integrations still require
actual prefill/full-context capacity testing; the context contract is unchanged.
