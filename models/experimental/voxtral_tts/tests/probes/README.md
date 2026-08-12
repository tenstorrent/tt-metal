# Measurement probes — not pytest tests

These need a device, take minutes, and print tables rather than asserting. None is named
`test_*.py`, so pytest does not collect them. Run them by hand:

```bash
export TT_METAL_HOME=$PWD
export PYTHONPATH=$PWD/ttnn:$PWD/tools:$PWD
python models/experimental/voxtral_tts/tests/probes/<name>.py
```

They are kept **because the numbers in `STATUS.md` are only as good as the instrument that
produced them**, and three times on this branch a conclusion had to be retracted after the
harness turned out to be measuring the harness (§6.37's aliasing trap, and two timing-loop errors
in §6.41/§6.43). A probe in `/tmp` cannot be re-run by whoever inherits this.

| probe | question it answers | key result |
|---|---|---|
| `ceilings.py` | what are this chip's two ceilings, and what fraction do we use? | 367 GB/s and 85.6 TFLOP/s; a decode frame uses **49% / 0.37%** (§6.53) |
| `sweep_mm.py` | which matmul program config is fastest per shape, in isolation? | the ttnn heuristic collapses on deep reductions: 144–147 GB/s at Kt=128/288 vs 352 at Kt=96 (§6.52) |
| `silu_fusion.py` | is `activation="silu"` actually fused? | **no** — it costs the same as a separate `ttnn.silu`; only `fused_activation` fuses (§6.52) |
| `mm_block_ab.py` | do `sweep_mm`'s isolated wins survive in the whole block? | mostly **no**, and the direction of the error reverses — the decisive experiment (§6.52) |
| `codes_real.py` | is the codes gate's 29.5% real, or an artefact of its synthetic input? | **artefact** — real prompts read 3.9% and are 100% off-by-one (§6.54) |
| `ref_vs_ref.py` | does the **reference** flip codes too, and are synthetic inputs near FSQ boundaries? | **no to both** — fp32 vs fp64 is 0/288, and the margins are identical (§6.54) |
| `device_err.py` | is the codes gate's 29.5% coming from Block 2 / FSQ? | **no** — with `h` fixed, Block 2 flips 2.4%; it is Block 1, 22× worse off-manifold (§6.54) |
| `prefill_precision.py` | can prefill accuracy be improved, and is the synthetic gate a canary? | **no to both** — real error is pinned at 0.70% across the whole weight ladder, and the synthetic number is non-monotonic in precision (§6.55) |
| `prefill_fp32.py` | is higher-precision prefill worth it (it runs once per utterance)? | **no** — fp32 weights buy nothing; fp32 activations work but the cache dtype blocks decode (§6.56) |
| `prefill_f32_act.py` | does fp32 prefill (typecast at the cache boundary) help past frame 0? | **no** — gain is gone by step 1; also shows decode is O(1) in position (§6.56) |
| `fp32_cache_handrolled.py` | can an fp32 KV cache be had by hand-rolling around `sdpa_decode`? | **no** — 44.7× slower; the op-count model fails on ops that materialise tensors (§6.57) |
| `bf16_decode.py` | what do bf16 weights through decode cost and buy? | **+29% for nothing** — worst-sample error is non-monotonic (§6.57) |
| `mos_batch.py` | automated MOS over a run's utterances — **this is the one `quality_report.py` calls** | wired into `--tier audio`; prints `MOS_MEAN` / `MOS_LONGFORM` / `MOS_MIN` for the harness to parse |
| `run_mos.py` / `mos_percase.py` | the one-off MOS investigations `mos_batch.py` grew out of — is the device perceptually worse than fp32? | **no** — delta −0.017/−0.027; long-form mean **4.63** at §6.59, **4.61** on the current build (§6.67) |
| `tail_probe.py` | does a change make rare BAD utterances likelier? counts failures, not means | the right shape for tail risk: many seeds on the three prompts that actually score low (§6.62) |
| `click_origin.py` | case 6 clicks — ours or the model's? | **the model's** — the fp32 reference clicks MORE on the same seed (69 vs 60) |
| `make_ref_ab.py` / `make_sampler.py` | build fp32-reference-vs-device pairs and a listening sampler from the current build | the inputs to every listening pass; §3 is explicit that a developer saying "ok" is not an eval |
| `perceptual.py` | MCD / F0 / codec transparency vs the fp32 reference | codec SNR 42.9 dB, LSD 0.62 dB; **MCD failed its self-test and is not reported** (§6.59) |
| `frame_ab.py` | does a block A/B predict the frame? | **no** — −2.124 ms on the blocks, 0 on the frame (§6.63) |
| `rtf_repeat.sh` | how repeatable is the generator? | **0.390 ms** over three identical runs, so it can decide (§6.63) |
| `opmap.py` | where does each block's time go, per op? | eager map -- ranks by LAUNCH cost, and got concat vs rms_norm backwards (§6.67) |
| `traced_cost.py` / `traced_ops.py` | what does an op cost INSIDE the trace? | the one that decides: concat 2.6 µs, rms_norm 63.5, heads 6.2, sdpa 22.4 (§6.67, §6.68) |
| `norm_traced.py` | is the sharded norm faster once traced? | **yes, +5.4 ms/frame** — reverses §6.39/§6.40 (§6.67) |
| `trace_probe.py` | is the ~68 µs per-op floor device time or host dispatch? | dispatch is **2.8–3.9%**, not 0% and not the ~100 µs others assumed (§6.49) |
| `seq_len_limits.py` | how long a prompt, how long an utterance, and what does length cost? | prefill has **no ceiling** (clean to 4096; the "~1024" claim was never measured); utterance length is `max_seq_len`, and it costs DRAM, not RTF (§6.69) |
| `xref_audit.py` | do the `[gpt-26]` / `§6.x` pointers still resolve? **run after any doc edit** | found a `codec-22` pointer cited twice and defined nowhere. Cannot catch a pointer to a REVERSED section — that stays manual. (Written WITHOUT brackets on purpose: this file is one of the four the audit scans, so naming a dead pointer in prose re-breaks it — which is exactly what the commit adding this row did.) |

**Read `frame_ab.py` before trusting any block A/B.** A tight loop measures device time with
dispatch overlapped; the real loop drains at 10 host crossings per frame and can absorb a device
saving whole. Block A/Bs screen; `--tier audio`'s `ms_per_frame` decides.

**Read `mm_block_ab.py` before trusting any isolated sweep.** It is the counterexample: `w2` and
`wo` posted 2.4× isolated wins and delivered exactly 0.00 ms in the block, while `w1` posted 1.03×
and delivered 2.42 ms. A tight loop of identical ops pipelines; a real block does not.

**`run_mos.py` must run from `/tmp/mosvenv`, never the main venv** — DistillMOS pulls `torchaudio`,
which STATUS §2 records as breaking `transformers` and taking `score_quality_set.py` with it.
`mos_setup.sh` builds that venv; the main one was verified `torchaudio`-free afterwards.

`trace_probe.py` opens the device with `trace_region_size`, which shifts the allocator enough to
move a free-running trajectory (`95dc26363f`). It therefore only ever **times**; it never
generates audio, and its numbers are not comparable to a run opened without the region.

**Idle the box before timing anything here.** A traced frame still does real host work every frame
(embed lookup, host↔device copies, argmax, FSQ quantize), so CPU contention inflates every frame
*uniformly and independently of position* — which looks exactly like a hardware effect. §6.69 got
43 ms/frame instead of 27.7 and had a thermal-droop write-up half drafted before finding the cause
was a stray `find /` in another shell.
