# KDA tail-padding prototypes (`kda_pad`)

## Verdict

Use **early crop** if `length` is fixed when the trace is captured and enough of
the 5,120-token input is padding. It is the smallest implementation, exactly
matches a physically trimmed production-shape run, and was 8.62% faster with
1,024 padded tokens. With only 224 padded tokens, however, every prototype was
slower than the existing 5,120-token path; the slice and dynamic-shape overhead
did not amortize. Keep the existing path when `length == hidden.shape[1]`.

The fixed-shape mask is the option to revisit if a single 5,120-token trace must
accept changing lengths, but this prototype did no less work and cost about
6.7% at both measured lengths. Late crop is dominated by early crop once the
tail is large enough to matter.

## Contract and scope

The prototype extends the layer call to:

```python
output, new_state = layer.forward(hidden, state, length)
```

- `0 < length <= hidden.shape[1]`.
- `length` is a multiple of the KDA compute chunk, 32 tokens.
- `output`, recurrent state, and convolution state must equal a call whose
  `hidden` input was physically trimmed to `hidden[:, :length]`.
- Tail values must not affect any returned tensor.
- Passing no `length` preserves the PR7 API and full-sequence path.
- This investigation covers a 5,120-token physical input on SP1 x TP8.
  Sequence-parallel length mapping and offsets are out of scope.

All branches have exactly one prototype commit directly on PR7 base
`68f40f4228844574cbe5fadd14ee79e4af610ec4`:

| Idea | Branch | Commit |
| --- | --- | --- |
| Fixed-shape identity mask | `kda_pad_mask` | `5a3264f3c2f` |
| Crop after input projection | `kda_pad_late_crop` | `77b26368d2a` |
| Crop before input projection | `kda_pad_early_crop` | `53f9dcdd8eb` |

## Ideas and prototype walkthrough

### 1. Fixed-shape identity mask

Keep every tensor at 5,120 tokens. Cache a device tail mask and multiply the
decay gate and beta by it. An invalid chunk then performs the recurrence
identity update (`gate = 0`, `beta = 0`). Select convolution state at `length`
from `old_state || projected_qkv`, and slice only the final output.

Expected: fixed shapes are friendly to one captured trace, but no expensive
projection, convolution, recurrence, normalization, or output-projection work
is skipped. Two mask multiplies and state/output slicing should make it slower,
independently of tail size.

Measured: about 6.7% slower at both tail sizes, matching that expectation. The
state is highly correlated but not bit-exact with trimmed execution because
the full grouped recurrence has different numerical grouping.

### 2. Late crop

Run the fused input projection over all 5,120 tokens, then slice its four
outputs (`qkv`, decay rank, output gate, beta) to `length`. Convolution and all
later stages see the true logical sequence.

Expected: exact trimmed semantics and reduced downstream work, while retaining
the full cost of the large fused input projection and paying for four slices.
It should beat masking but only win overall for a sufficiently large tail.

Measured: bit-exact at both production sizes. It was 3.56% slower for 224
padded tokens and 1.23% faster for 1,024 padded tokens.

### 3. Early crop

Slice `hidden` once at layer entry, then run the unchanged KDA pipeline on the
logical sequence. This is semantically the direct implementation of the
contract.

Expected: exact trimmed semantics and the greatest compute reduction because
every KDA stage scales with `length`. Its tradeoff is a distinct captured graph
and program set per supported length, plus one input slice.

Measured: bit-exact at both production sizes. It tied late crop within 0.002 ms
for 224 padded tokens, but was 8.62% faster for 1,024 padded tokens. This
confirms that avoiding the input projection matters once the saved work is
large enough.

## Hardware results

Environment: one 8-chip Blackhole mesh, Fabric1D, SP1 x TP8, firmware 19.5.0,
KMD 2.4.1, IOMMU enabled. Each number is the median of five samples; each sample
is the host time for ten non-blocking warm trace replays followed by a device
synchronization. Every prototype run measured its own unmodified 5,120-token
baseline in the same pytest process.

| Valid / physical tokens | Padding | Idea | Baseline median | Prototype median | Change |
| ---: | ---: | --- | ---: | ---: | ---: |
| 4,896 / 5,120 | 224 (4.375%) | Mask | 9.582 ms | 10.234 ms | +6.81% |
| 4,896 / 5,120 | 224 (4.375%) | Late crop | 9.591 ms | 9.932 ms | +3.56% |
| 4,896 / 5,120 | 224 (4.375%) | Early crop | 9.593 ms | 9.934 ms | +3.56% |
| 4,096 / 5,120 | 1,024 (20%) | Mask | 9.589 ms | 10.229 ms | +6.68% |
| 4,096 / 5,120 | 1,024 (20%) | Late crop | 9.593 ms | 9.475 ms | -1.23% |
| 4,096 / 5,120 | 1,024 (20%) | Early crop | 9.592 ms | 8.766 ms | **-8.62%** |

The sample ranges were 9.575--9.607 ms baseline and 10.028--10.630 ms mask at
4,896; 9.587--9.606 ms baseline and 9.752--10.315 ms late crop; and
9.581--9.605 ms baseline and 9.752--10.298 ms early crop. At 4,096 they were
9.586--9.609 / 10.028--10.636 ms, 9.585--9.615 / 9.290--9.816 ms, and
9.584--9.611 / 8.633--9.037 ms, respectively. The upward drift within prototype
samples makes these directional prototype results, not a production perf signoff.

Theoretical useful-token reductions are 4.375% and 20%. Wall time cannot scale
one-for-one because collectives, launch/synchronization work, slicing, and other
fixed costs remain. The observed early-crop results show the expected crossover:
overhead dominates the small tail, while a 20% tail recovers 8.62% wall time.

## Correctness

The production harness compares against a separate invocation on a physically
trimmed tensor and checks zero-filled versus random inactive tails.

| Length | Idea | Output PCC | Recurrent PCC | Convolution PCC |
| ---: | --- | ---: | ---: | ---: |
| 4,896 | Mask | 0.99999982 | 0.99999993 | 1.0 |
| 4,096 | Mask | 0.99999984 | 0.99999196 | 1.0 |
| 4,896 and 4,096 | Late crop | 1.0 (bit-exact) | 1.0 (bit-exact) | 1.0 (bit-exact) |
| 4,896 and 4,096 | Early crop | 1.0 (bit-exact) | 1.0 (bit-exact) | 1.0 (bit-exact) |

All variants were bit-identical between zero and random inactive tails. The
small single-device reference test also passed for output, recurrent state, and
convolution state with PCC 0.999958, 0.999835, and 0.999997, respectively. The
existing no-`length` forward-contract test passed unchanged.

## Sequence-length constraint found during measurement

The requested multiple-of-32 rule is necessary but not sufficient for every
length in the current production grouped-scan configuration. A 4,800-token
trimmed oracle has 150 chunks; its effective group divisor produces 10 groups
per local head and 120 summary owners, exceeding Blackhole's supported 110.
That run fails before it can be a correctness oracle. Therefore 4,896 is used
as the nearest supported below-5,120 comparison point (153 chunks, 108 owners).
This is an existing recurrence program-selection constraint, not a padding
semantic failure. A production implementation should either reject unsupported
lengths clearly or improve grouped-scan program selection.

## Commands and validation

The isolated worktree and environment were built successfully (1,552 build
steps):

```bash
./create_venv.sh --python-version 3.10
./build_metal.sh --enable-ccache --build-ttnn-tests
```

Correctness and compatibility:

```bash
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/layer/test_padding_prototype.py -sv
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/layer/test_contract.py::test_forward_contract -sv
```

Both passed. Performance/correctness commands were run once per branch and
length, substituting `IDEA` and `LENGTH`:

```bash
KDA_PAD_IDEA=IDEA KDA_PAD_LENGTH=LENGTH scripts/run_safe_pytest.sh \
  models/demos/deepseek_v3_d_p/tests/kda/perf/test_padding_prototype_perf.py -sv
```

All six supported-shape runs passed. The attempted `LENGTH=4800` mask run
failed with the 120-versus-110 grouped-scan owner error described above. No
offset or sequence-parallel padding behavior was implemented or validated.

## Recommendation

Advance early crop as the semantic implementation, but dispatch directly to
the existing full-sequence path when no tail is present and benchmark expected
serving length buckets before productionizing. If serving requires one trace
whose `length` changes at replay time, early crop cannot provide that by itself;
the fixed-shape mask establishes correctness for that design point but needs a
fused, chunk-aware recurrence path that actually skips invalid chunks to avoid
its measured regression.
