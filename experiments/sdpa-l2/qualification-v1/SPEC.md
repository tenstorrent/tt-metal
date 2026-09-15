# SDPA qualification v1: frozen numerical contract and execution scope

This is qualification, not tuning. Operator numerical kernels are unchanged.
Performance and end-to-end model evaluations (items 4 and 5 of the proposal)
are explicitly excluded. Model-derived operator inputs remain in scope but
require captures; their absence is an unqualified coverage gap, not a pass.

## Gates

All gates apply to each seed, batch item, and head independently. All use the
original BF16 Q/K/V and an FP64 reference, before query preprocessing.

| Gate | Fast | Accurate |
|---|---:|---:|
| Normal/model relative L2 | <=3.5% | <=0.5% |
| Stress relative L2 | <=5% | <=0.5% |
| Normal/model PCC | >=0.999 | >=0.99998 |
| Stress PCC | >=0.998 | >=0.99998 |
| Row p99 relative L2 | <=2 times applicable L2 limit | <=1% |
| Worst tested row relative L2 | <=4 times applicable L2 limit | <=2% |

Row error is RMS(error)/max(RMS(reference row),0.01*RMS(reference head)).
Unfloored row summaries are also retained. The percentile uses torch.quantile
with its default linear interpolation. Ordinary head L2 is not denominator-
floored. The proposal explicitly exempts undefined PCC for constant outputs.
The online scorer initially handled a constant reference but incorrectly
failed constant actual outputs with a nonconstant reference. adjudicate.py
corrects that scoring mistake from the retained raw metrics, removing only
`pcc_undefined_actual_constant` failures. It changes no thresholds and waives
no defined PCC failures. Raw records/scores are preserved in results.jsonl;
accepted-results.jsonl applies the agreed exception. The same adjudication
is applied to the ideal-rounding oracle. Defined PCC can still fail even for
ideal BF16-rounded outputs when the residual signal is below output resolution;
those cases are identified separately, not silently waived.

Zero V requires exact zero. Constant V=1 and single-key attention require <=1
BF16 ULP per element, defining ULP at a power-of-two boundary as the larger
of its two adjacent representable spacings. For the explicitly zero-reference
cancellation construction, the absolute max-error budget is T*RMS(V)/sqrt(K),
where T is the applicable relative L2 limit as a fraction. This fixes the
previous proposal's unspecified absolute tolerance; it is not an epsilon in
the L2 denominator. Cancellation uses Q=0 and exactly paired V,-V.

For common V=c+residual additionally require:
norm(actual-reference) <= T*norm(reference-c) + 2*norm(BF16(reference)-reference).

For common K, both device outputs are checked against their respective FP64
references. When centering is exactly BF16-representable, the references agree
and pairwise difference must be <=2*T*norm(reference). Otherwise the paired
comparison subtracts the reference shift caused by BF16 requantization, with
budget T*(norm(shifted_reference)+norm(centered_reference)). This is the
triangle-inequality consequence of the individual gates, not a relaxed gate.
The FP64 centered/shifted identity before requantization is checked on the
first seed for each geometry/offset. Q common modes are not assumed invariant.

The initial runner incorrectly asserted that every centered K was exactly
BF16-representable. It stopped on common K=-8, seed 1234, H=5, length 32768.
That was a harness error, not a device failure. The v1.1 runner handles the
quantization difference explicitly and retries that case; prior valid records
are retained. The original runner is qualify-initial.py and the interrupted
log is run-full.log. Numerical kernels and thresholds were not changed.

Physical tile-padding invariance changes from_torch's padding value 0 to +32
without changing logical data or execution geometry, and compares hashes of
the full outputs. Trace replay likewise compares all output elements. Full
output finiteness is checked on every device invocation, including replays.
Inputs are hashed before/after each case to check caller-data preservation.

## Fixed synthetic matrix

Noncausal, B=1, D=128, H in {5,10}; five seeds {1234,1235,1236,1237,1238}.

- Normal: lengths 2048,8192,25920,32768,65536,75600,131072,262144 (80 inputs).
- Stress: lengths 32768 and 262144, both H, all five seeds; Q/K scale 0.5 or 2,
  sparse outliers, and independent common Q/K/V offsets -32,-8,+8,+32
  (300 inputs). This is not a full distribution-by-length Cartesian sweep.
- Structural: lengths 2048,32768,262144, both H, all five seeds; zero V,
  constant V, uniform attention, cancellation (120 inputs).
- Boundary: lengths 31,32,33,127,128,129,511,512,513,1023,1024,1025,32767,
  32769,33280, both H, all five seeds (150 inputs).
- Single key: Q=128,K=1, both H, all five seeds (10 inputs).

Total: 660 input cases, 1320 requested mode cases. The manifest is
manifest-final.json. Seeds 1237/1238 were not used to tune the retained exp
coefficients. This run does not claim an additional release holdout suite.

Each device invocation runs the full operator, not sampled-device attention.
All query rows are checked at Q<=2048. Otherwise exactly 512 distinct rows per
head are selected: first/last/middle ranges, chunk edges, evenly spread rows,
and seed-determined random fill. Raw positions and input/output hashes are
recorded. Long-context accuracy and worst-row claims are sampled, not exhaustive.
Small references use stable FP64 attention; uniform/constant/zero/single-key
cases use exact analytical references. The original FP64 reference self-test
compares blocked and dense softmax and checks K/V common-mode identities.

## Tested implementation and no-fallback policy

The retained improved source remains unchanged locally. A measurement-only
host factory patch:

1. Removes only the BF16 compensation minimum-K-chunk-count gate, enabling
   compensation at short contexts as requested in the preceding measurements.
2. Rejects every FP32 request that cannot use the specialized streaming path.
3. Rejects every BF16 request that cannot enable compensation.
4. Logs the selected implementation and Q/K lengths.

See qualification-host.patch. Q chunks stay 128. Fast K chunks stay 512;
input double buffering is unchanged. Accurate uses K=1024 when aligned and
K=512 when that permits streaming (e.g. length 33280). It retains its existing
Q bit-ceiling/scale-compensation preprocessing; Fast has no Q preprocessing.

Short or padded FP32 inputs are recorded UNSUPPORTED, never executed on a
fallback. Three actual negative probes (2048,25920,32769) verify host rejection;
see guard-probes.log. Prechecks avoid repeating rejected allocation attempts.
Dense explicit masks and fully masked rows are not qualified by this matrix;
the current specializations exclude user masks. Single-key attention exercises
the generated-padding path for Fast; it is unsupported for FP32 streaming.

## Coverage gaps and reporting

- Model captures from two families have been requested but not supplied.
- Current run hardware is a single Blackhole P100A, not Galaxy. At discovery,
  the listed Blackhole Galaxy had no available cards. Available Wormhole
  Galaxies cannot execute these Blackhole-only specializations unchanged.
- No causal/GQA/D!=128/multibatch claim is made; those were outside v1 scope.
- No performance or end-to-end model-score acceptance is attempted.

PASS/FAIL are per executed mode case, combining all tested heads and gates.
UNSUPPORTED and ERROR are distinct from FAIL. Unexecuted cases are MISSING.
No aggregate mean, unsupported result, or missing hardware counts as a pass.
