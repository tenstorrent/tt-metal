---
name: tt-moe-correctness
description: Correctness checklist for MoE modules and chains on Blackhole Galaxy - router and bias semantics, expert-to-chip mapping, dispatch capacity overflow, real-token skew, bfp8/bfp4/mxfp4 checks, ops that silently corrupt data, PCC thresholds, chained-PCC drift, near-tie flips, determinism and faulty-host detection. Use when writing or validating a router, expert, dispatch/combine, layer, chain, decode or prefill test, and whenever a PCC is lower than expected.
---

# MoE correctness

Last validated against commit: <none yet> (section J only: 0d3b876e60f, MiMo attention probe on .30; other sections not exercised)
Evidence and commits for every item: `bringup/GUIDE_dsv4_lessons.md` (ids L-xx in brackets). Log every result (`tt-bringup-log`).

## A. Router (most divergence starts here)
1. Implement exactly the HF semantics, write them in LOG: score function (sigmoid vs softmax vs sqrtsoftplus), bias used for
   SELECTION ONLY (aux-free / noaux_tc: top-k of score+bias, weights from the raw score), weight normalisation (sum to 1 over top-k?),
   routed scaling factor, group-limited routing, hash-routed layers, router dtype. Then compare the selected set per token vs the fp32 reference.
2. Bias magnitude: DSV4's bias reached 10-14 while the gate op takes bf16: only 12-39% of tokens got the reference expert set. A hi/lo bias split (fp32 emulation) brought it to 92-93% [L-router].
   Check the bias dynamic range of YOUR model before trusting a bf16 gate.
3. Pass: expert-SET equality >= 99% on real tokens (order may differ), top-k weights PCC >= 0.9999; remaining mismatches are near-ties (gap in score+bias < bf16 resolution). Report the flip rate per layer.
4. Check the swiglu clamp / activation limits: `moe_compute` had no clamp while the model has one (|up|<=10) [L-clamp].
5. Score scaling: apply the scale on the right side of normalisation; test with a layer whose scale != 1.

## B. Experts and mapping
* Map: expert e -> (chip, local slot) must be one function used by weights loader, dispatch and combine; test with a torch reference that
  computes the dense MoE and compare per expert, not only the sum. 256 experts / 32 chips = 8 per chip (check op limits).
* Weights per expert are DRAM-bound; two copies of expert weights (decode + prefill layouts) do not fit 40 layers: plan ONE shared copy [L-onecopy].
* Shared expert (if any) is separate; MiMo has none: make sure the combine does not add a zero-weight path that costs time.

## C. Dispatch capacity and skew
* Capacity factor 2 silently dropped tokens (chip load 33,408 rows vs capacity 33,120) -> first-token PCC 0.953, 13/16 argmax; CF=4 -> 16/16 [L-cf].
  Always add a capacity check that FAILS LOUDLY (count rows per chip vs capacity) in the test and in a debug flag of the model.
* Test skew with REAL tokens: one expert received ~63x the mean; random-token MoE profiles are unreliable in either direction [L-skew]. Always profile MoE with real prompt tokens.
* Check every mesh row: failures appeared only in the last mesh row.

## D. Quantisation
* Default to the checkpoint-faithful dtype for routed experts (bfp8 in DSV4). bfp4/mxfp4 is an ACCURACY TRADE-OFF: ask the human before enabling by default.
  DSV4 measured: bfp4 logits PCC -0.05..-0.08, TF argmax 13/11/9 of 16 vs 15/15/13, GSM8K n=512 95.3% vs 96.1% (p=0.42, not resolvable), ~3-4.6 ms/token faster [L-bfp4].
  A claim "bfp4 costs 0.02-0.03" from one prompt was wrong (retracted).
* If the checkpoint is mxfp4/fp8: check which dtype the device op can actually consume (and whether it is lossless vs the checkpoint) in a single-expert test first.
* LoFi vs HiFi: smaller activation tiles do NOT make LoFi matmul faster (17 cycles/tile at 1..32 rows) [L-tile]; re-measure before claiming.
* KV cache dtype fp8: flushes subnormals (|x|<2^-6 read back 0): keep opt-in; test with real data.

## E. Ops that silently corrupt on a build (add each as a standing unit test)
* `reduce_scatter_minimal_async` over the TOKEN dim of a big partial-sum tensor: corrupts random 32-row blocks (nondeterministic). Use reduce_scatter over hidden + all_to_all [L-rs].
* Narrow/odd-shaped all-gathers and fused kernels with T not a multiple of the tile group (mHC mixes at T=12, 20 returned wrong values: pad) [L-pad].
* Zero-length slices / S_pad==C single-chunk shapes corrupted state for half the users [L-single-chunk]; U=8 grouped MoE hung [L-u8].
* Out-of-range positions of finished users in a batch corrupt routing and hang dispatch/reduce_scatter: clamp positions [L-oob].
* Rule: every new op/shape: compare against torch on >= 3 shapes incl. odd sizes, 5 repeated runs (determinism), per mesh row.

## F. PCC thresholds and chains
* Single module vs dump on real activations: >= 0.999 (layer), >= 0.9999 (router weights). Below: bisect by stage (q, scores, selected ids, output).
* Full-depth chain has NO CPU reference (policy): per-layer teacher-forced PCC + short CPU chain + section I gates. Device chain numbers below are DSV4 priors:
* Chained 40 layers: logits PCC 0.97-0.99 on real prompts is normal for DSV4 (0.972 first token, 0.983/0.975/0.980 TF decode steps); it drifts with depth and
  tokens near ties. NEVER set a chain gate from a single module threshold, and never judge a chain on random or 9-token prompts (PCC 0.74-0.87) [L-chain].
  Gate on: first-token argmax on real prompts, teacher-forced argmax >= ~13/16, per-layer stage PCC, then GSM8K-type accuracy vs reference.
* Always compute the reference's own bf16-vs-fp32 noise floor and compare the device to both.
* Near-tie flip vs real divergence: a flip is OK if top1-top2 gap in the reference logits is small (< ~0.5 with logits O(15-30)), it reproduces under teacher forcing,
  all users identical or random. Real divergence: gap large (> 1-2), systematic by user/row/position/length. Ragged lengths and position ~245 cases stayed unexplained in DSV4 [L-open].
* Plain decode itself is not bit-reproducible across builds/hosts: do not require bit equality across builds, require it within one build and flag set.

## G. Determinism test and faulty-host detection
* Run the same config twice, then on a second host: first tokens and layer outputs must be bit-identical in-process on one host, and PCC 1.0 / same argmax across hosts.
* A host whose first tokens differ only in some mesh rows, or differ between identical runs, is faulty: stop, log it, add to bad-host list (`tt-lab-ops`), discard its numbers [L-host31].
* Wide-shape ladder before "prefill works": batch/users 1..max, 1 and many chunks, chunk sizes from 128 to the maximum, two ISLs in one process, short and long ISL, 4 and all layers.

## I. End-to-end acceptance without a CPU full-model run (all REQUIRED)
* Per layer: teacher-forced device layer vs CPU oracle on the device's own hidden state (no compounding); short CPU chain for drift. Prefill and decode checked as separate phases.
* GSM8K accuracy on the device vs the published/reference accuracy (n and threshold from the scorecard; DSV4 n=512 resolved only >~2 pp, see L-bfp4), determinism (identical runs bit-identical), first-token sanity.
* Hosted reference when one exists (MiMo: OpenRouter `xiaomi/mimo-v2.6-flash`, the same model): qualitative/behavioural oracle only, not bitwise; log prompts and outputs in LOG.
* Residual risk: compounding error past the CPU prefix is caught only by these gates. Never waive them; escalate if they fail.

## H. Warm-up and measurement hygiene (correctness-adjacent)
First call in a process compiles (82 ms readback step 0, TTFT first call > warm): discard it. Paired in-process A/B only for perf claims. Never compare numbers across hosts/commits.

Escalate to the human: any proposal to lower precision, relax a PCC gate, drop tokens, or ship with an unexplained divergence. Stop and report if a gate fails twice with the same cause.

## J. Attention op conventions on Blackhole (measured by the MiMo attention probe, `models/demos/blackhole/mimo_v26/tests/probe_sdpa_ops.py`, LOG track `attn-probe`)
* Sink input: every SDPA op (decode, paged decode, MLA decode, prefill, chunked prefill) multiplies the sink by `scale` inside the kernel. Pass `sink/scale` (HF adds the RAW sink logit after scaling).
  Real MiMo sinks are 0.56..1.14 and a wrong convention still shows PCC 0.998: validate with sink magnitude x5 (raw convention then gives 0.82 decode / 0.96 prefill) or a needle test.
* Sliding window W: the ops attend to exactly W keys including the current one (keys p-W+1..p), same as HF. Check with a needle test (one attractive key with a huge V, queries/users on both sides of the boundary), not with PCC on random data.
* `chunked_scaled_dot_product_attention(scale=<python float>)` is rejected by the binding (noconvert); omit `scale`, the default is Q_dim**-0.5. K and V must have equal width there; `flash_mla_prefill` with an explicit V tensor accepts V narrower than K but has no window/sink/paged arguments.
* Decode: `paged_flash_multi_latent_attention_decode(q, k, v, head_dim_v=...)` accepts an independent V tensor (V width == head_dim_v) with sink and window; it has no `cache_position_modulo` (ring cache); the non-MLA paged op has it.
* Always run 5 repeats and compare bitwise (determinism), and one case per mesh-relevant shape.
