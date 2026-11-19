# Expert dtype gate: single-layer evidence (design section 0 "Expert dtype" / section 9)

Question: can the routed experts' gate/up weights (W13) run in `bfp4_b` instead of `bfp8_b` without changing
accuracy? The design adopts bfp4 for W13 only if the eval-set accuracy is unchanged (Phase 2 gate). This page is the
single-chip, single-layer half of that evidence: the PCC and per-row distributions per policy on both MoE layer kinds
with real weights and realistic activations, and the traced-decode device time that the policy buys. **The default
stays `bfp8_b` for gate/up and down.** The final decision needs the eval set (teacher-forced top-1/top-5 on the
standard text set, design section 9 "Acceptance"); a per-layer PCC cannot settle it because the error compounds
over 42 MoE layers.

Knobs (`tt/moe.py::TtRoutedExperts`, threaded through `TtMoE`, `TtDecoderLayer`, `FunctionalDecoder.from_state_dict`):
`expert_dtype` (W13 `[1,E,H,2I]`, gate|up columns) and `expert_down_dtype` (W2 `[1,E,I,H]`), both `ttnn.bfloat8_b`
by default. The shared expert and the dense MLP stay on `weight_dtype` (bfp8). Math fidelity is HiFi2 with bf16
activations for every policy (unchanged), so the tables isolate the weight format; LoFi with bfp4 weights is a
separate perf knob that only makes sense if the gate passes.

Policies: `bfp8` = bfp8 / bfp8 (baseline), `bfp4_gu` = bfp4 gate/up + bfp8 down (the design's candidate),
`bfp4_all` = bfp4 gate/up/down (lower bound).

## Method

`tests/test_expert_dtype.py` (one chip, `TT_VISIBLE_DEVICES=5`, tt-metal main 8cfcaf754d3 + `arg/glm53_dtype`):

* MoE sublayer alone at T=256 on `tests/test_moe.py::realistic_mlp_input` (the layer's own
  `post_attention_layernorm(ffn_hc(streams))` from the T256 dump) vs the HF `Glm5NextTextMoE` with the strict-fp32
  router bias; the TT top-8 sets vs HF are reported (the router is fp32 and identical for every policy).
* Full layer on the random-token dump `R4097` through `FunctionalDecoder`: prefill S=4097 (layer 3: dense rows
  < 2048, sparse beyond), PCC on the first 1024 rows (the dense region) and on all rows, the per-row PCC
  distribution (`per_row_pcc`, one row = 4 streams x 4096), then the dump's 4 decode steps 4097..4100 (layer 3 on
  the sparse path, eager).
* Device time: `tests/test_perf_layer.py` layer 4 (KDA+MoE, no SDPA kernel so the Tracy device profiler works),
  `GLM53_PERF_PHASE=decode GLM53_PERF_DECODE_STEPS=1` under `python -m tracy -r -p -v`, one signposted traced replay
  (B=1, position 1024), `tt-perf-report --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END`; the
  `Device Time` column summed over the window (tables under `tracy/dtype_gate/`).

Commands:

```bash
TT_VISIBLE_DEVICES=5 pytest models/demos/glm53_flash/tests/test_expert_dtype.py -s           # 6 cases, ~6 min
TT_VISIBLE_DEVICES=5 GLM53_PERF_LAYER=4 GLM53_PERF_PHASE=decode GLM53_PERF_DECODE_STEPS=1 \
  GLM53_PERF_EXPERT_DTYPE=bfp4 GLM53_PERF_EXPERT_DOWN_DTYPE=bfp8 \
  python -m tracy -r -p -v -m pytest models/demos/glm53_flash/tests/test_perf_layer.py -s
tt-perf-report generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv \
  --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END --no-advice --csv out.csv
```

## Accuracy (PCC vs HF, real weights; logs `~/ashai/glm53/_logs/dtype_sweep_layer{3,4}.log`)

Layer 3 (DSA + MoE):

| policy (gate/up, down) | MoE T=256 PCC | MoE per-row p5 / min | router sets | prefill PCC rows < 1024 | prefill PCC all 4097 | per-row p5 / min (rows < 1024) | per-row p5 / min (all), rows < 0.997 | decode PCC 4097..4100 |
|---|---|---|---|---|---|---|---|---|
| bfp8 (baseline) | 0.99995 | 0.99994 / 0.99961 | 255/256 | 0.99998 | 0.99994 | 0.99990 / 0.98783 | 0.99981 / 0.98783, 15 | 0.99997 0.99997 0.99996 0.99996 |
| bfp4_gu | 0.99968 | 0.99743 / 0.99579 | 255/256 | 0.99954 | 0.99945 | 0.99848 / 0.98554 | 0.99844 / 0.98554, 24 | 0.99980 0.99980 0.99951 0.99947 |
| bfp4_all | 0.99808 | 0.99302 / 0.99158 | 255/256 | 0.99744 | 0.99767 | 0.99631 / 0.98369 | 0.99625 / 0.98369, 745 | 0.99792 0.99810 0.99775 0.99809 |

Layer 4 (KDA + MoE):

| policy (gate/up, down) | MoE T=256 PCC | MoE per-row p5 / min | router sets | prefill PCC rows < 1024 | prefill PCC all 4097 | per-row p5 / min (rows < 1024) | per-row p5 / min (all), rows < 0.997 | decode PCC 4097..4100 |
|---|---|---|---|---|---|---|---|---|
| bfp8 (baseline) | 0.99995 | 0.99991 / 0.99127 | 255/256 | 1.00000 | 1.00000 | 0.99997 / 0.99652 | 0.99995 / 0.99652, 1 | 0.99996 0.99999 1.00000 0.99999 |
| bfp4_gu | 0.99504 | 0.99256 / 0.98726 | 255/256 | 0.99996 | 0.99994 | 0.99948 / 0.99617 | 0.99956 / 0.99607, 5 | 0.99747 0.99987 0.99997 0.99998 |
| bfp4_all | 0.99141 | 0.98797 / 0.98472 | 255/256 | 0.99992 | 0.99990 | 0.99920 / 0.99532 | 0.99934 / 0.99522, 9 | 0.99628 0.99980 0.99995 0.99997 |

Reading the tables:

* The router is identical across policies (fp32 composite; the one differing token of 256 is the known tie-break vs
  HF on both layers), so every difference is the expert matmuls.
* MoE sublayer error (1 - PCC): bfp8 5.5e-5 on both layers. bfp4 gate/up: 3.2e-4 on layer 3 (6x the bfp8 error) and
  5.0e-3 on layer 4 (90x; the layer-4 MoE alone sits exactly at the 0.995 bar). bfp4 down on top of that: 1.9e-3 /
  8.6e-3 (layer 4 below the bar). The sensitivity differs per layer: the layer-4 experts' outputs are small relative to
  their inputs, so the same weight noise is a larger fraction of the output.
* At the layer level the mHC post-scale and the residual dilute the sublayer error: bfp4_gu keeps the full-layer
  prefill at 0.99945 (layer 3) / 0.99994 (layer 4), decode worst step 0.99947 / 0.99747, all above the 0.995 bar.
  The per-row tail widens (rows < 0.997 on the 4097 random-prompt rows: 15 -> 24 on layer 3, 1 -> 5 on layer 4; the
  global min row of layer 3, 0.988, is the attention path, present in the baseline too).
* bfp4_all is visibly worse: layer-3 prefill 0.99767 with 745/4097 rows below 0.997, layer-4 MoE 0.9914.
* A single layer passing the PCC bar is not "accuracy unchanged": 42 MoE layers compound this. The eval set decides.

## Device time (layer 4 traced decode, B=1, one replay, Tracy device profiler, `Device Time` summed over the window)

Same chip (5), same build and session, one signposted traced replay per policy (471 ops each, duplicate ids 0;
tables `tracy/dtype_gate/layer4_decode_<policy>.{txt,csv}`, stacked summaries `*_summary.png`). The DRAM columns
are tt-perf-report's model for 8 active experts (`--active-experts 8`; 512 GB/s peak).

| policy (gate/up, down) | W13 sparse matmul (8/288 x 32 x 4096 x 4096) | W2 sparse matmul (8/288 x 32 x 2048 x 4096) | experts total | rest of the step | step device time | wall / step |
|---|---|---|---|---|---|---|
| bfp8 (baseline) | 765 us, 181 GB/s (35 %) | 529 us, 133 GB/s (26 %) | 1.294 ms | 9.64 ms | **10.93 ms** | 11.36 ms |
| bfp4_gu | 624 us, 114 GB/s (22 %) | 530 us, 133 GB/s (26 %) | 1.154 ms | 9.65 ms | **10.80 ms** | 11.25 ms |
| bfp4_all | 624 us, 114 GB/s (22 %) | 465 us, 79 GB/s (15 %) | 1.089 ms | 9.63 ms | **10.72 ms** | 11.30 ms |

Noise floor: a second bfp8 run on the same chip in the same session (`layer4_decode_bfp8_rep.txt`) gives 10.92 ms
device with W13 765 us and W2 529 us, identical to the microsecond, so the 141 us / 64 us deltas are real device-time
differences (the wall time of a single replay is host-noisy: 11.4 vs 17.2 ms; device time is the metric). The earlier
chip-0 table in `tracy/layer4_kda_moe/` (643 / 468 us, 10.76 ms) differs from this chip's baseline by ~19 % on the
sparse matmuls, so policies are only comparable on one chip in one session, as here.

Expert weight bytes per decode step (8 active experts of 288): W13 142.6 MB bfp8 -> 75.5 MB bfp4, W2 71.3 MB bfp8 ->
37.7 MB bfp4 (bfp8 = 1.0625 B/element, bfp4 = 0.5625).

Reading the table:

* Only the two sparse matmuls change; every other op is identical (rest of the step 9.64 / 9.65 / 9.63 ms).
* bfp4 gate/up saves 141 us on the W13 matmul (-18 % of that op, -1.2 % of the layer step: 10.93 -> 10.80 ms);
  bfp4 down saves another 64 us (-12 % of W2).
* The bytes halve but the time does not: at this shape (B=1 padded to 32 rows, 8 of 288 experts, the 64-core
  `mcast_in0` 1D config of the per-block composite) the sparse matmul reaches only 35 % of DRAM peak in bfp8 and
  22 % in bfp4, i.e. it is not streaming-bound on one chip; the design's "bfp4 halves the dominant expert-streaming
  term" (section 7) assumes an expert kernel at the DRAM roofline. The expert term is also only 12 % of this
  single-chip decode step, which the KDA composites dominate (Phase 2/3 work).
* So the perf side of the gate has to be re-measured on the Phase-2 layout (72 experts x 256-wide slices per chip,
  TP8 over the intermediate, 4-32 users per row) and on an expert kernel that reaches the roofline; the single-chip
  number above is a lower bound on the saving, not the design's 2x.

## Recommendation for the Phase-2 gate

* Keep `bfp8_b` for gate/up and down as the default (unchanged).
* Carry **only `bfp4_gu`** (bfp4 gate/up, bfp8 down) into the eval-set gate: it stays above the per-layer bar on both
  layer kinds and buys the expert-streaming reduction the design counts on. Drop `bfp4_all`: the down projection in
  bfp4 costs another 2-4x in sublayer error for only one third of the remaining expert bytes.
* Gate criterion (design section 9 "Acceptance"): teacher-forced top-1/top-5 on the standard text set with the full
  model, bfp4_gu vs bfp8, within run-to-run noise; the final decision needs that eval set, not these PCCs. If it
  passes, re-measure the sparse matmuls at LoFi (bfp4 weights do not need HiFi2) as a second perf step.
