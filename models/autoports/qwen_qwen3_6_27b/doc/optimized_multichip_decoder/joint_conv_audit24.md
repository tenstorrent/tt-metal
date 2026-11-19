# Joint correction combine and local convolution fusion, audit24

2026-11-13. CPU-only isolated candidate/probe, no main or shared-helper changes,
TTNN/Torch imports, device execution or native build. Not a default selection,
performance result, numerical pass or stage-completion claim.

Owned new files:

- `tt/joint_conv_candidate.py`
- `tests/probe_joint_conv.py`
- This audit.

## Actual boundary and arithmetic

The selected compact joint projection produces FP32 (optionally BF16) tiled
`[1,B,N]`, logical N at least6784, normally padded to7168. Primary columns are
0..4159 and correction columns4160..6783. The current compact matmul returns
width-sharded L1 directly to `PackedCorrection._combine`; a DRAM restore is not
part of the selected native control. `packed_correction_candidate.py` lines234–253
and316–397 establish the dispatch/format contract.

Current boundary is `_combine(joint)` -> `packed[:,:,:2560]` ->
`local_conv(qkv,history,taps_rm)`, whose first operation is TILE-to-ROW_MAJOR.
`multichip_decoder.py::_gdn`/`_local_batched_conv` and
`recurrent_next_candidate.py::local_conv` establish these material operations.
The convolution baseline in the probe is that exact helper, not a reconstructed
Torch formula or the slower unfused Python loop.

For output tile-column c in0..129, primary tile is c. Correction is tile130+c
for c<80, absent for80<=c<128, and tile130+c-48 for c>=128. The absent correction
is still **FP32 add with positive zero followed by RNE BF16**, including Z.
The candidate preserves this operation even for BF16 joint input. FP32 inputs
use `UnpackToDestFp32` on CB0/1; BF16 inputs use Default unpack and the same
FP32 DST/add/typecast. Joint input dtype is not an invitation to change matmul
accumulation, weights or coefficient precision.

For each QKV value the preserved order is:

```text
raw = RNE_BF16(FP32(primary) + FP32(correction))
p0 = FP32(old_history[0]) * FP32(tap[0])
p1 = FP32(old_history[1]) * FP32(tap[1])
p2 = FP32(old_history[2]) * FP32(tap[2])
p3 = FP32(raw)            * FP32(tap[3])
sum = ((p0 + p1) + p2) + p3
out = native_BF16_SiLU(RNE_BF16(sum))
new_history = [old_history[1], old_history[2], raw]
```

The first BF16 boundary is physically packed to CB2 and reloaded before the
current-token product. It cannot be optimized away. Native
`mul_binary_tile`, `add_binary_tile`, explicit Float32->Float16_b typecast,
and the FP32-DST `calculate_silu(false,8)` specialization are copied from
`optimized_decoder.py::_conv107__CONV_RM_PIPE2_COMPUTE` (line4230). This is SiLU,
including its native sigmoid approximation/rounding; a standalone sigmoid or
FP32 SiLU substitution would change arithmetic. Beta/alpha outputs here are
raw corrected logits, as in the original combine boundary. Their later
sigmoid/decay and Z normalization/gating remain downstream and unchanged.

## Candidate topology and storage

Signature:

```python
conv_qkv, corrected = joint_conv(
    joint, history, taps_rm,
    output=prepared_row_major_qkv,
    rest=prepared_corrected,
    rest_only=False, cores=80,
)
```

`conv_qkv` is owning BF16 ROW_MAJOR DRAM `[B,1,2560]`, directly usable by the
packed-QKV recurrence. `history` remains owning BF16 ROW_MAJOR DRAM
`[B,3,2560]` and mutates in place. Four BF16 ROW_MAJOR `[1,1,2560]` taps are
read-only, on the same mesh. No intermediate DRAM restore of the joint or QKV
is added. Both FP32/BF16 TILE DRAM and WIDTH_SHARDED L1 joint inputs are admitted.
Only the compact B1..32 Blackhole path is implemented; no prefill temporal
fusion or public alignment restriction is introduced.

The default corrected output is owning tiled BF16 `[1,B,4160]`: this preserves
all existing Z/beta/alpha slice offsets and optionally exposes raw corrected
QKV for debugging. `rest_only=True` returns `[1,B,1600]`: Z1536 followed by
beta32 and alpha32. It avoids writing the80 raw-QKV tiles; this option requires
explicit downstream offset adaptation before model integration. Its beta and
alpha live widths are12, preserving the20 padding columns of each32-column
region. It is not silently wired into the existing full-packed consumers.

Each work item owns one32-channel column for all users. Strided assignment
puts the first80 QKV columns on80 workers before assigning cheap Z/tail work.
No core reduction, reshard, or cross-core history dependency is required.
History/tap reads are aligned64-byte channel strips, gathered into private
scratch then copied with integers into proper BF16 tile coordinates. Source
input tiles are read through TensorAccessor directly from their actual memory.
The row coordinate is public user, so nonaligned B needs only internal zeros.
All padded users are zeroed before arithmetic. Unused physical N columns are
never read. Inputs and tap weights are not mutated.

The writer consumes old history1/2 and packed raw combined QKV only after
convolution output publication proves compute consumed the old inputs. It
writes aligned64-byte rows and waits for the NoC write barrier before publishing
a done token. Compute is the sole input-CB pop owner and waits for that token;
reader cannot overwrite old history/current-token CB storage prematurely.
Every channel has one owner. Writer and reader scratch CBs are distinct.

Per worker: FP32-source CB storage43,392 bytes, BF16-source39,296 bytes; reader
scratch6,464 and writer scratch8,256 bytes explicitly include up-to63-byte
alignment slack. The optional full packed output is266,240 bytes/device versus
102,400 bytes for rest-only, independent of logical B because of tile padding.
Convolved output adds B*5,120 bytes; history retains its original size/layout.
All source/output/history addresses must be distinct; malformed shapes,
dtypes, placements, mesh mismatches and unsupported worker counts fail on host.
No persistent owner cache is introduced.

## CPU evidence and limits

`python -m models.autoports.qwen_qwen3_6_27b.tests.probe_joint_conv --cpu-check`
passes2,196,480 element-address checks,24,960 column-ownership checks and seven
negative geometry cases, covering every B1..32 and workers1/8/48/80/110/130.
It verifies native arithmetic source fragments, residual mapping, history
selection, all tile coordinate permutations, scratch alignment and generated
FP32/BF16 sources. Both Python ASTs parse, Black formatting passes, and a
`sys.modules` assertion verified neither TTNN nor Torch was imported.
These are source/geometry checks, not compiled-kernel or exact-device evidence.

Potential performance limitation: convolution computes a padded32-user tile
per channel. B1 therefore performs80 convolution tiles, compared with three
packed raw tiles in the existing convolution. At B32 the native convolution
processes96 packed tiles; the new80-tile formulation also reads each tap only
once per channel instead of once per user. This predicts different batch
crossovers; it does not justify a blanket rejection or a batch-wide default.
Full/rest-only and actual L1 input must be compared as complete families.

If small-B padding dominates, the shortest subsequent adaptation is a
user/chunk-packed gather: combine1024 channel elements for one user into a
raw tile, then reuse the existing three-chunk convolution math/packing. Give
that family compact rest-only output to avoid concurrent partial writes into
a shared full-packed output tile. It changes ownership, not rounding. If the
first JIT failure concerns mixed source unpack/pack state or a partial-copy API,
first isolate combine output/history on the existing descriptor formats and
adapt the explicit `copy_init`/accessor calls; do not reject the fusion on that
first error. The current tile-column version already uses integer layout
conversion and native BF16 pack boundaries, so no full-tensor restore is needed.

## Prepared hardware probe and next integration step

The probe uses distinct per-rank/user synthetic joint values in the actual
7168 physical-width contract and actual eight-core `[32,896]` L1 shards, plus
independent histories and distinct four-tap coefficients. It is not a real
model-weight matmul or HF reference. It compares native complete boundary to
full-packed and compact-rest fused outputs, including exact per-user history
bits, all rest values, finite checks and unchanged PCC>=.995. It includes
cancellation/tie inputs, physical NaN padding with bitwise transport checks,
all-input/output address rebinding, changed inputs,50 accumulating queued
steps, traced changed inputs, stable outputs and allocation-positive-control.
All candidate inputs/banks/histories/outputs are allocated before capture.
The original combine/untilize APIs allocate native intermediates in capture;
the probe preserves and times that actual behavior rather than pretending they
support output preallocation. Replay tracking must still pass for both families.

Parent-only commands, using the existing device environment/serial queue:

```bash
TT_METAL_WATCHER=1 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_joint_conv \
  --batches 1 3 --dtypes float32 --placements l1 --cores 80 --no-timing \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/joint_conv_smoke24.json

TT_METAL_WATCHER=1 TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_joint_conv \
  --all-batches --dtypes float32 bfloat16 --placements l1 dram --cores 80 --no-timing \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/joint_conv_wide24.json

env -u TT_METAL_WATCHER -u TT_METAL_TRACE_ALLOC_TRACKING -u TT_METAL_DEVICE_PROFILER \
  python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.probe_joint_conv \
  --batches 1 3 17 32 --dtypes float32 --placements l1 --cores 48 80 110 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/joint_conv_perf24.json
```

No timing is collected with watcher, profiler or allocation tracking enabled.
JSON timing covers the entire boundary in warmed traces, rotates family order,
and includes native allocation/operations and fused full/rest writes. Captured
history is allowed to evolve; a native accumulating oracle is computed before
capture, including the timing steady-state after four identical tokens.

Only after focused gates: expose raw compact joint projection output before
`PackedCorrection._finish_projection`, preserving the existing default finish.
In `_gdn` decode, consume that raw output with this helper and pass returned
convolved QKV directly to packed recurrence; do not convolve twice. Initially
retain full corrected output so existing beta/alpha/Z consumers work unchanged.
Select compact-rest only with explicit offset adaptation and complete downstream
comparison. Both changes require separate parent authorization/source ownership;
neither is applied here. Whole-layer PCC, realistic text/context stress,
resource ownership, warmed before/after timing and profiles remain required.

Frozen initial source SHA256:

- helper: `7e6a953a4e7aa6bc5a762486ca4867c4b0adc0f410ff937b8ca2fa76bd10d02b`
- probe: `90651b265861ef36156723dfda0a3d74ba5eaeff521de06dcd9abb5dd1843a5e`

## First device feedback and diagnostic retry

Parent's `joint_conv_smoke25.json`/`.log` records a normally closed first smoke
failure at B1/FP32/L1. Native eager0 output, history and corrected rest are
finite and self-exact on all four ranks. The original probe called PCC before
saving a nonfinite candidate's diagnostics; that exception prevented locating
which candidate tensor first diverged. This is an unresolved candidate failure,
not a precision/layout rejection and not a reason to relax finite/exact gates.

Probe diagnostic v2 (`b4f115810a04eb3a0e867b1b04b4979e763110b299fe869c96ade8b151623ee8`)
checks both sides' finiteness before PCC. On the first failed comparison it
saves all families' output/history/rest tensors, finite masks, reference,
joint input and tap values to a sibling `.first_failure.pt`, plus per-rank
counts, first64 mismatch coordinates and raw BF16 bits in JSON. The original
failed artifacts remain unchanged. Helper source is unchanged pending the
first discriminating retry; native baseline finiteness already narrows the
failure to candidate execution/transport. CPU checks and formatting still pass.

Diagnostic v2 is invalid as a quality gate: source review found its finiteness
block outside the per-user loop and its record append conditional on finiteness.
It could omit nonfinite users. Parent preserved its failed B3 unequal-constant
PCC artifact; no v2 pass is accepted. Diagnostic v3 restores unconditional
per-user records, asserts exactly3 tensors *4 ranks *B comparisons, and adds
a CPU AST guard proving the append/finiteness work belongs to the user loop.
Unequal constant tensors receive diagnostic PCC0; any other metric assertion
also yields diagnostic0 without bypassing exact/finiteness failure. This makes
first-failure dumps reachable rather than losing evidence in PCC calculation.
Helper unchanged. V3 probe SHA256:
`59c5f228b562bd16e38c5e40297ed30ef7a0f9311dd0e7e0b4fa036bd1d1e2ff`.

## AutoFix: missing FP32/BF16 unpack transition

V3's preserved first-failure tensor/JSON artifact is discriminating: for B1 on
rank0 both candidate families have exact history and corrected rest, but1,920
of2,560 output values differ and240 are nonfinite. First-face odd channels
survive while even channels corrupt; the next face reads zeros. This rules out
joint correction offsets/raw history transfer as the first divergence and
focuses the failure on mixed-format convolution compute.

Source diagnosis: `tt_metal/hw/inc/api/compute/tile_move_copy.h:25–30` states
that `copy_init` selects the copy operation but **does not reconfigure unpacker
data types**. Startup initialized CB0's FP32 format. The new fused compute
then used only `copy_init` to read BF16 history/taps/combined CBs, leaving the
physical unpack format stale. The existing standalone convolution starts with
BF16 CB0 and never crosses formats, so it does not have this problem. Existing
parallel QK/norm helpers explicitly use `reconfig_data_format_srca` at such
crossings.

The minimal v4 helper change adds two explicit source-format transitions:
`reconfig_data_format_srca(0)` before every joint combine, and
`reconfig_data_format_srca(4)` before the BF16 convolution reads. The former
also restores FP32 for a subsequent column after BF16 convolution. CB0/1 are
the same joint format; every convolution source CB is BF16, so no per-tap
transition is needed. Arithmetic, pack boundaries, transfer geometry, CBs,
outputs and native control remain unchanged. CPU checks now require both
transitions in the correct source order; all prior checks and Black pass.
No device result is claimed until the parent's fresh v4 smoke.

V4 source SHA256:

- helper `88e0e28d425e39e730d20d04ae8566468ee1c692b6166d7fe97195cf3d671727`
- probe `9a32e08777390c55fbba1704dd18bbb9a2f3f7287b71391cc8c1389d61162a78`

Parent verification: `joint_conv_smoke25_v4.json` passed B1 and B3 after the
explicit unpack transitions. Both full-packed and compact-rest families are
finite and bit-exact against the unchanged native control, including history,
changed inputs, rebinding,50 queued steps, watcher and allocation tracking.
V3 failed on the same family with exact rest/history but corrupted convolution
output; v4 changes only the two source-format transitions. This before/after
supports the missing reconfiguration diagnosis. Helper/probe are frozen at the
v4 SHAs above. Wide B/dtype/placement coverage and uninstrumented family timing
remain pending; this is not a default selection or whole-layer performance pass.
