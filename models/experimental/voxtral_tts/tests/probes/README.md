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
| `trace_probe.py` | is the ~68 µs per-op floor device time or host dispatch? | dispatch is **2.8–3.9%**, not 0% and not the ~100 µs others assumed (§6.49) |

**Read `mm_block_ab.py` before trusting any isolated sweep.** It is the counterexample: `w2` and
`wo` posted 2.4× isolated wins and delivered exactly 0.00 ms in the block, while `w1` posted 1.03×
and delivered 2.42 ms. A tight loop of identical ops pipelines; a real block does not.

`trace_probe.py` opens the device with `trace_region_size`, which shifts the allocator enough to
move a free-running trajectory (`95dc26363f`). It therefore only ever **times**; it never
generates audio, and its numbers are not comparable to a run opened without the region.
