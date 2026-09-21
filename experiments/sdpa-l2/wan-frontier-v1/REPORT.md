# Wan2.2 480p stock smoke

Completed 2026-09-16 on bh-lb-08, eight Blackhole chips, IRD 221619.
Reservation extension succeeded with a 14-hour remaining allocation at the
time of extension (IRD capped the requested 24 hours).

## Result

**1 passed, 31 deselected**. The existing stock performance test and its
performance gates passed. The wrapper changes only checkpoint resolution to
a pinned local snapshot and host Torch threads to 16; no attention math or
model weights were changed for this smoke.

- Wan2.2-T2V-A14B-Diffusers, revision
  `5be7df9619b54f4e2667b2755bc6a756675b5cd7`.
- 832x480, 81 frames, 40 steps, seed 42, CFG 4.0/3.0; SP4/TP2 on 2x4 mesh.
- Butterfly prompt from [PLAN.md](PLAN.md).
- 32,760 logical video tokens, padded to 32,768; 40 heads, D128.
- Stock self-attention: HiFi2, BF16 destination, accurate exp setting
  (`exp_approx_mode=False`), ring communication. This is a separate control,
  not frozen frontier A.

| Timed phase | Seconds |
|---|---:|
| Text encoding | 0.0585 |
| Denoising, including expert reloads | 163.1376 |
| VAE decoding | 1.1305 |
| Total pipeline | 164.3348 |

The pipeline constructor ran a two-step allocation warmup before timing the
40-step video. The timed pipeline was untraced, matching the existing test's
8-chip configuration. Total excludes MP4 encoding, model construction,
initial weight conversion and allocation/JIT warmup. This is one timed sample,
not a steady-state multi-round performance qualification.

Cold pytest wall time was 622.76 seconds. The separate checkpoint download
took 406.7 seconds using host networking, verified all 41 files' sizes and
all weight LFS SHA256 hashes. Container download attempts failed due to DNS/
network errors; no system networking or TLS verification settings were changed.

## Cache and hardware

Converted-weight caching was enabled with
`TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0`. The first four component loads
were expected cold misses and created caches; the subsequent three expert
reloads were hits. The timed run's expert reloads took approximately 2.4 and
3.9 seconds. This is evidence of functioning cache reloads in this run, not
a separate fresh-process cold/warm equivalence qualification.

During generation the sampled device temperatures were 65.1–70.2 C and
clocks 975–1156 MHz. These are observed allocation timings, not peak-frequency
performance claims. Telemetry and raw logs are saved alongside this report.

## Output sanity

Output shape was `(1,81,480,832,3)`, uint8, range 0–255. The exported MP4
contains 81 frames at 832x480 and 16 fps (5.0625 seconds). First/middle/last
frames show the intended butterfly and flower with no obvious blank/corrupt
output. This is a smoke check, not quantitative or temporal quality qualification.

- [Video](stock-smoke-01/wan_output_video_t2v.mp4)
- [First/middle/last frames](stock-smoke-01/contact.png)
- [Raw run log](stock-smoke-01/stock-smoke-01.log)
- [JUnit result](stock-smoke-01/stock-smoke.xml)

## Proposed suite and runtime estimate

Stock + D/C/B/E/F/G, two prompts, seed 42: **14 videos**. Prompts are the
butterfly and the woman at an outdoor café with visible face/hands, as recorded
in PLAN.md. The human prompt replaces the initially proposed boxing cats.

At stock speed alone, 14 generations would take **38.3 minutes**. That is a
stock-equivalent reference calculation, not a prediction for every variant.
Budget **2–3 hours** for generation, per-variant warmups, sampled-frame CLIP,
representative block timing and bounded real-QKV accuracy checks, once the
Wan attention adapter is qualified. This allowance is an engineering estimate:
none of the non-stock variants has yet been measured in Wan.

Before that run, Wan needs an adapter for SP4/TP2 and correct handling of its
eight padded tokens. The existing FLUX adapter rejects these conditions and
must not be used with masking silently omitted. Allow another **1–3 hours**
for initial adaptation/qualification if no new kernel/runtime issue appears:
**3–6 hours total planning budget**, not a guaranteed completion time.

The broader suite has since completed: 14/14 videos, 48m27s for the suite
including pilots/setup, plus a short scoring/validation pass. See
[suite results](suite-01/REPORT.md), [findings](FINDINGS.md) and
[STATUS.md](STATUS.md). The estimates above are the original pre-integration
planning budget, not the measured runtime. All non-self-attention components
remained fixed; compressed KV transport was verified. The results disclose
the stock ring versus experimental all-gather scheduling difference.
