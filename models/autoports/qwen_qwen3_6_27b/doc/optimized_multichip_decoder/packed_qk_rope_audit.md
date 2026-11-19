# Direct packed QKVG split, normalization, rotary and placement candidate

2026-11-13. Independent CPU-only implementation. **No native build, TTNN import,
device execution, accuracy result or performance claim.** Parent owns device
validation. Main decoder, existing QK/RoPE helper and native sources were not
modified by this task.

## Actual layout and scope

The requested packed layout agrees with source. `tt/multichip_decoder.py` builds
`packed_qkvg` by concatenating rank-local Q, K, V and gate weight columns, in that
order (full-attention weight preparation, approximately lines148–164). The
compact attention decode path projects `[1,B,5120]` to BF16 DRAM TILE
`[1,B,3584]`, slices gate columns2048 onward and QKV columns0–2047, then runs
`_local_qkv_split`. The inherited `_linear_raw` returns BF16 DRAM for this
projection; correction handling also returns a BF16 combined result.

This is **one physical user-row tile**, padded to `[1,32,3584]`; it is not an
array of per-user head tiles. The 112 tile columns contain:

| Quantity | Logical columns | Tile columns |
|---|---:|---:|
| Q, six heads | 0–1535 | 0–47 |
| K, one head | 1536–1791 | 48–55 |
| V, one head | 1792–2047 | 56–63 |
| Gate, six heads | 2048–3583 | 64–111 |

The existing `_LOCAL_QKV_STAGE` proves the tiled BF16 face-row address:
`(user//16)*1024 + (user%16)*32 + horizontal_face*512` bytes. A live row fragment
is32 bytes. The candidate reads an aligned64-byte fragment into distinct
64-byte scratch slots and scatters only the live32-byte half into the target
head row. An odd final user's neighboring padded row may be fetched into
scratch, but is never copied into compute inputs. This is explicitly poisoned
and checked by the device probe.

## Candidate contract

New file `tt/packed_qk_rope_candidate.py` exposes:

```python
q, k, v, gate = packed_qk_rope(
    projected, q_weight, k_weight, cos, sin,
    epsilon=epsilon, outputs=None,  # or four caller-owned buffers
)
```

Only BF16 TILE DRAM packed inputs with exact logical/padded shapes above and
public B1–32 are accepted. Norm weights remain FP32 or BF16 DRAM
`[1,1,256]`; trig remains the actual BF16 DRAM `[B,1,1,64]` used by the prior
candidate. Device and coordinate-set equality, output nonaliasing, exact output
logical/padded shapes and memory layouts are checked. It does not silently
reshard, change precision, choose a fallback or require aligned public B.

Outputs are:

- Q: `[1,B,6,256]`, BF16 TILE DRAM.
- K: `[1,B,1,256]`, BF16 HEIGHT_SHARDED L1 on cores32 through32+B−1,
  eight-column row-major, shard `[32,256]`.
- V: `[1,B,1,256]`, same format on cores0 throughB−1. K and V preserve the
  disjoint-owner contract required by native fused paged cache update.
- Gate: `[1,B,1536]`, BF16 TILE DRAM, ready for the existing sigmoid multiply.

There are2B workers. Each gathers only its own Q or K row fragments directly
into compute CB0; Q workers also gather their V row into CB11. These complete
physical head buffers are zeroed before live-row copies, with an explicit
`write_zeros_l1_barrier`. Scratch slots are aligned64 bytes, with an extra slot
to own alignment slack. Every scratch reuse follows a read barrier. No
cross-worker semaphore or global intermediate Q/K tensor is needed.

**The COMPUTE string is imported unchanged from the previously validated
`parallel_qk_rope_candidate.py`.** This preserves FP32 sum/rsqrt/affine work,
the explicit BF16 normalized boundary, native FPU rotate-half products, packed
BF16 sine/cosine products and final packed addition. Q writes directly to DRAM;
K writes directly to its owning cache-update shard. Writer waits on the same
CB3/CB10 dependencies and acknowledges writes before releasing buffers.

Q writers also place their V shard and copy complete gate tiles. Gate tile
`t` belongs uniquely to Q worker `t % B`; copies are batched by at most8 tiles.
This copies each of the48 gate tiles exactly once, preserving raw physical
padding bits. **Gate dead rows are intentionally copied from source, not
zeroed.** Later per-element sigmoid/multiply does not mix users. Probe validates
logical gate bits plus all physical padding bits, and Q/K/V dead rows must be
finite zeros. V and gate copies are included in both correctness and timing.

This removes the material QKV slice, gate slice launch, standalone split and
its Q/K input tensors from the selected fused-QK/RoPE family. It still reads
packed DRAM (no claim of direct width-sharded projection consumption), writes
all four final outputs and copies gate physical tiles. Reader/compute/writer
scheduling and extra CB storage can affect latency; only the paired full-family
measurement can establish a win.

## Probe and unchanged quality gates

New `tests/probe_packed_qk_rope.py` compares three complete families, starting
from identical rank-distinct packed DRAM inputs and ending with Q/K/V/gate:

1. Native material slices and `_local_qkv_split`, selected parallel norm,
   native partial RoPE, K placement, V and gate outputs.
2. Same slices/split, validated fused QK norm/RoPE, V and gate outputs.
3. Direct packed candidate, including V/gate movement.

The first family uses the previously selected parallel norm; “native” within
its numerical report denotes the native rotary chain, not native RMSNorm.
The baseline for the new direct optimization is family2. Every family creates
its own four output buffers in eager allocation timing and trace capture.
Additional persistent-output correctness traces are separate from the matched
latency comparison. Warmed eager latency includes allocation and dispatch;
matched warmed trace timing includes all device movement but not host allocation.
Projection, cache update and SDPA are outside this isolated interval; final
adoption still requires decoder-level PCC, state and latency validation.

Coverage includes:

- B1/3/17/32 by default or every public B1–32 via `--all-batches`.
- Three changing inputs, actual HF trig with distinct per-rank/per-user
  positions, nonconstant weights, and one zero Q head.
- Q/K output PCC≥.995 against both prior families and a Torch reference with
  explicit BF16 norm and native rotary cast boundaries; per-user PCC≥.995.
  **Every Q/K/V/gate logical bit must equal family2**, including Q/K tail bits.
  V/gate bits must additionally equal host packed input.
- Exact IEEE BF16 NaNs in all invalid packed user rows. Host construction with
  `preserve_nan_values=True` prevents the earlier device tilize NaN→Inf
  canonicalization. A setup-only raw112-tile copy transfers full physical
  storage into the logically nonaligned tensor. Upload, logical storage, live
  updates and post-invocation inputs are checked bit-for-bit. No reshape/slice
  is used to create poisoned input padding.
- Input/weight/trig immutability; every direct output's physical padding;
  all-address rebinding with changed packed/trig/weights and retained old
  buffers, exact outputs and unchanged program-cache entry count.
- 50 queued direct launches with changed inputs and persistent outputs;50
  queued trace replays for each family; changed-input exact replay in both
  forward/reverse family order; separate preallocated-output trace replay.
- Allocation-tracking positive control deliberately allocates after capture,
  requires the unsafe-allocation exception, frees that allocation, then
  validates ordinary replay. Watcher/tracker runs are excluded from timings.
- Per-case progress/failure JSON, commands, source hashes, actual positions,
  exact-bit diagnostics and latency samples. Existing artifact files are
  never overwritten. No relaxed PCC or hidden runtime fallback.

## CPU evidence and frozen provenance

Commands executed without TTNN/Torch imports or device use:

```bash
python models/autoports/qwen_qwen3_6_27b/tt/packed_qk_rope_candidate.py --cpu-check
black --check --target-version py312 --line-length 120 \
  models/autoports/qwen_qwen3_6_27b/tt/packed_qk_rope_candidate.py \
  models/autoports/qwen_qwen3_6_27b/tests/probe_packed_qk_rope.py
# Both files also parsed with Python ast.parse.
```

CPU geometry/source check passed all32 batches. It independently decodes tiled
source/destination face addresses for every live user/head/channel fragment,
checks aligned64-byte read bounds, proves disjoint2B owners and exact once-only
coverage of all48 gate tiles. This does not compile or prove device NoC behavior.

SHA256 at handoff:

- Candidate: `644967b7929ab37b6815f2360da045bbbbab0b98f55047badc191d987d239cea`
- Probe: `b0fe7366928c87fc1770811145c11b511c288af419461ec4aa9e274de93d7b12`
- Unchanged shared QK/RoPE helper:
  `23d22234e9eb4dbfcd79c92476575d19e1fe12d0974cf28f942dedeb565d0f43`

## Parent-owned serial device commands

Use the existing lease/watchdog wrapper, after the current job is closed.
These commands are proposed, not executed by this task.

```bash
M=models/autoports/qwen_qwen3_6_27b
D=$M/doc/optimized_multichip_decoder
TT_METAL_WATCHER=10 MULTICHIP_JOB_TIMEOUT=900 \
  bash "$D/run_job.sh" packed_qkrope_smoke_v1 \
  env -u TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_ALLOC_TRACKING=1 \
  TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python "$M/tests/probe_packed_qk_rope.py" --batches 1 --steps 3 \
  --output "$D/packed_qkrope_smoke_v1.json"

TT_METAL_WATCHER=10 MULTICHIP_JOB_TIMEOUT=2400 \
  bash "$D/run_job.sh" packed_qkrope_wide_v1 \
  env -u TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_ALLOC_TRACKING=1 \
  TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python "$M/tests/probe_packed_qk_rope.py" --all-batches --steps 3 \
  --output "$D/packed_qkrope_wide_v1.json"

MULTICHIP_JOB_TIMEOUT=1200 bash "$D/run_job.sh" packed_qkrope_perf_v1 \
  env -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER \
  TT_METAL_TRACE_ALLOC_TRACKING=0 \
  python "$M/tests/probe_packed_qk_rope.py" --batches 1 3 17 32 \
  --steps 3 --samples 7 --replays 50 --output "$D/packed_qkrope_perf_v1.json"
```

First native/API/shape/NoC failure is an implementation bug to diagnose, not
performance rejection evidence. Preserve logs, adapt the exact tile/CB/address
contract and retry under AutoFix. If this family is correct but slower, compare
the full split+fused baseline, not only the already-isolated QK/RoPE helper.
A profiler rerun with watcher/tracker disabled should verify removal of the
slice/split launches and include the complete direct generic-op interval.
Final decoder-level integration/default selection remains parent-owned.
