# Opt-in solo attention candidate implementation

2026-11-17, source implementation only. No hardware execution, correctness
result or speedup is claimed by this note. Parent owns device validation.

Construct the candidate with:

```python
model = QwenModel(
    mesh,
    suppress_inactive_moe_routes=True,
    specialize_solo_slot0_attention=True,
    # Same checkpoint, context, precision and layer set as the control.
)
generator = QwenGenerator(model, max_batch_size=32, decode_buckets=(2, 4, 32))
```

Both model flags defaultFalse. The specialization requires route masking and
a retained physicalB2 bucket; invalid construction rejects. Adapter/plugin
code is unchanged, so this opt-in is not yet the serving default.

The constructor allocates shared immutable zero partials in DRAM TILE:
FP32[1,1,2048] for linear attention, BF16[1,1,1024] for full attention, both
padded to32 rows. Payload is327680bytes/chip before allocator alignment.
Their addresses appear in the specialized capture signature. No zero tensor
is a destination, and no request-time constant allocation occurs.

`model_solo0` uses the same physicalB2 hidden/logit carriers. Full attention
retains batched B2 input norm, packed projection, output projection, reduction
and residual; only row1's independent head/cache/SDPA chain is omitted.
Linear attention retains both input norms and row0's unchanged attention,
replacing row1's independent attention partial with the FP32 zero. B2
post-attention norm/router/shared/grouped-expert/terminal and inactive masks
remain. Dormant cache/state receives no omitted writes; position advancement
still reads device activity. The native grouped expert ACK drain is unchanged.

Selection uses `active_slots == (0,)` and current bucket selection containing2.
Owner1-only, high owners, empty active sets and multirow sets select ordinary
buckets. This is a generator-owned prepared signature, not an external plugin
bucket. Temporarily setting `generator.decode_buckets=(32,)` in a diagnostic
disables solo selection and forces the retained fixed32 reference without
recapture; restoring(2,4,32) restores normal selection. Tests must not call
preparation while using a temporary selection subset.

Expected retained traces for capacity32/buckets2/4/32:

| Key | Physical rows | Independent attention rows |
| --- | ---: | --- |
| model_2 | 2 | 0,1 |
| model_4 | 4 | 0..3 |
| model_fixed | 32 | 0..31 |
| model_solo0 | 2 | 0 |
| sampling | 32 | native draw32; unchanged |
| sampling_64 | 32 | native draw64; unchanged |

Capture signature for model_solo0 adds `attention_slots:[0]` and both immutable
zero buffer addresses; its physical_rows remains2. Six captures, zero workload
releases, and six teardown releases are expected if every preparation/capture
succeeds. Actual trace-region capacity remains an unverified requirement.

## Two-generator diagnostic preparation

Existing phase1 warmup was extracted into idempotent
`generator.prepare_without_capture(prefill_cases=((32,0),))`. The normal
`prepare()` still performs both phases for ordinary callers. This extraction
is needed to prevent a second diagnostic generator from compiling/allocating
helper state while the first already owns live traces:

```python
# Construct both complete generators and warm diagnostic observation paths
# first. Each owns its cache, activity, sampling, leases and trace inputs.
control.prepare_without_capture()
candidate.prepare_without_capture()
control.prepare()
candidate.prepare()
```

All ordinary bucket variants and the optional solo graph are warmed before
sampling/helper warmup completes, then request/cache/RNG state is restored.
`prepare()` consumes that prepared phase without rewarming. Changing the
prepared bucket set before capture rejects. Caller must prepare every
generator sharing a mesh before either captures; this method cannot discover
an unrelated generator automatically. Allocation tracking remains required.

## Required next tests

First retain the route-only exactness result, then enable this combined flag
in the parent's reduced discriminator. Use real layers0/3, nonzero dormant
slot1/high-owner state, equal forced tokens/positions, active full logits and
all-rank cache/state snapshots. Cover `(0,)→(0,1)→(0,)→(1,)→(0,1,3)→(0,17)→(0,)`,
actual remap, changed/unchanged required page mappings, native32/native64
sampling, two pending output leases and stale-input sentinels. Direct output
for inactive logits is not an exactness contract; active output and all dormant
state are. All32-active ordinary graph remains available and must match.

The parent should add its explicit test selector rather than infer a CLI flag
from this note; no test files were edited here. Run tracking1/skip0 with the
existing positive control. Only after reduced and all40 exactness should the
adapter opt in and serving benchmarks measure any gain. Nonblocking replay,
separate canonical sampling and deferred lease reads are untouched.

Source pre-commit initially reformatted model.py; the final rerun passes all
applicable hooks. No TTNN import or server/device command was executed by the
implementer. Numerical risk remains that changing inactive intermediate rows
exposes cross-row packing/rounding despite preserved B2 shapes. The exactness
discriminator must resolve that risk, not a tolerance relaxation.
