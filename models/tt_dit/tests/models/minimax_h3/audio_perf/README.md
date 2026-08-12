# audio_perf

Speed and accuracy measurement for the MiniMax-H3 audio decode. Two scripts; the suite's own
correctness gates live in `../test_audio_minimax_h3.py`.

## Acceptance: latency and PSNR against the CPU reference

    CVD_MESH=4x8 CVD_T_FACTOR=8 CVD_MESH_AXIS=1 CVD_TRACED=1 python cpu_vs_device.py

Encodes four real clips (speech and music) with the torch/diffusers reference, decodes each on CPU and
on device, and scores device against CPU. Prints PASS/FAIL on both criteria and writes
`_0_source` / `_1_cpu` / `_2_device` WAVs to `/data/rshirvani/audio_compare/clips/` so the difference can
be heard. Non-default configurations tag their WAV filenames, so a sharded decode cannot overwrite a
single-device one.

Current result at that configuration — **281.6 ms, 49.45 dB mean**, per-clip 47.87 / 47.82 / 52.83 /
49.28, which is identical to the single-device baseline, so T-sharding and tracing cost no accuracy:

    config: mesh 4x8, t_factor=8 axis=1, traced=True
    mean              2.269     0.282    8.06x     49.45
    ACCEPTANCE  psnr 49.45 dB vs 49.45 baseline (no degradation): PASS  |  latency 281.6 ms: PASS

Drop the `CVD_*` variables for the single-device baseline (0.9304 s at the same 49.45 dB).

| variable | meaning |
|---|---|
| `CVD_MESH` | mesh shape, e.g. `4x8`. Default `1x1`. |
| `CVD_T_FACTOR` | T-shard factor. **Must equal the length of the axis it shards** — on a 4x8 mesh only `(4, axis 0)` and `(8, axis 1)` work; anything else dies in `_partition_t` on a non-tile-aligned slice. |
| `CVD_MESH_AXIS` | which mesh axis carries the T shard. |
| `CVD_TRACED` | replay a captured trace. Worth ~3x on a sharded mesh (and only ~1.04x on a single device, because a mesh has to dispatch to every chip). |

## Timing only

    BENCH_N=5 python decode_bench.py <label>

Builds the decoder once, warms it, then reports a median over N decodes. Time one decode per process and
you measure build and JIT noise instead — that is where an earlier ~1% run-to-run spread came from.

## Two things that will waste your time otherwise

* **`pytest.ini` sets `timeout = 300`**, and `test_audio_decode_t_parallel` legitimately takes ~9 minutes
  on a 4x8 mesh. Pass `--timeout 2400` or it is killed and looks like a hang.
* **A wedged card is indistinguishable from a kernel hang.** A bare `ttnn.add` tells them apart in
  seconds; run that before debugging anything else.
