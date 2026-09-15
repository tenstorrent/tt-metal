# Matched SDPA Pareto comparison

The inline chart includes all 12 distinct retained configurations, numeric ablations,
and principal FAST scheduling revisions with matched resident-input measurements.
`measurements.json` records exact values and source paths relative to `../`.
It does not include duplicate reruns, invalid experiments, or every intermediate
scheduling edit. Historical main FP32/non-streaming and full-chip measurements
are excluded because they do not have this matched measurement contract.

Blackhole P100A, one core at 1350 MHz; noncausal Q256/K512/D128; original BF16
normal inputs, seed 1236, no preprocessing. K/V repeat 512 times and Q repeats
16 times. Useful FLOPs: 549,755,813,888. Twenty warmups and ten blocking trace
measurements; plotted throughput uses median latency. All plotted paths stream.
BF16 retains two K/V slots, FP32 one; neither has recurring input transfers in
the timed loop. Internal score and recurrent-state traffic is included.

L2 is percent relative to FP64 attention on the original BF16 resident inputs.
Repeating identical K/V leaves the mathematical normalized result unchanged.
This is a numerical diagnostic, not distinct-input qualification, full-chip
throughput, or a prediction of Galaxy/model-level performance. The experimental
wrapper explicitly exercises Q256; production dispatch support is not implied.

Strict dominance means another measured point has no higher L2 and no lower
throughput, with at least one strict improvement. No tolerance or qualification
score is applied. The plotted line joins nondominated discrete observations;
interpolated points are not measured or necessarily implementable.

Strict frontier: G, F, E, D, C, B, A (ascending L2).
Practical choices highlighted: A main BF16, B FAST, E QK4/PV2, G ACCURATE.
D remains technically nondominated because it is slightly faster than E.
C remains technically nondominated because its resident L2 is slightly lower
than B, despite its poor speed/accuracy exchange. Neither conclusion implies
that these points remain on the frontier for other distributions.

The 0.5% line is a reference goal, not a claim of qualification acceptance.
Source reports: `../hybrid-mixed-v1/REPORT.md`,
`../hifi2-fp32-resident-v1/REPORT.md`, `../bf16-sfpu-v2/REPORT.md`.
