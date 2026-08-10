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
| `run_mos.py` / `mos_percase.py` | automated MOS (DistillMOS) — is the device perceptually worse than fp32? | **no** — delta −0.017/−0.027; long-form mean **4.63** (§6.59). Needs `/tmp/mosvenv`, see `mos_setup.sh` |
| `perceptual.py` | MCD / F0 / codec transparency vs the fp32 reference | codec SNR 42.9 dB, LSD 0.62 dB; **MCD failed its self-test and is not reported** (§6.59) |
| `frame_ab.py` | does a block A/B predict the frame? | **no** — −2.124 ms on the blocks, 0 on the frame (§6.63) |
| `rtf_repeat.sh` | how repeatable is the generator? | **0.390 ms** over three identical runs, so it can decide (§6.63) |
| `trace_probe.py` | is the ~68 µs per-op floor device time or host dispatch? | dispatch is **2.8–3.9%**, not 0% and not the ~100 µs others assumed (§6.49) |

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
