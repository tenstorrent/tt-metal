# audio_perf

Latency and accuracy measurement for the MiniMax-H3 audio decode. Two scripts; the suite's correctness
gates live in `../test_audio_minimax_h3.py`.

## Acceptance: latency and PSNR against the CPU reference

    # 1. baseline, single device
    python cpu_vs_device.py

    # 2. the sharded + traced configuration, with the baseline's mean PSNR passed back in
    CVD_MESH=4x8 CVD_T_FACTOR=8 CVD_MESH_AXIS=1 CVD_TRACED=1 CVD_BASELINE_PSNR=<mean from step 1> \
      python cpu_vs_device.py

Encodes four real clips (speech and music) with the torch/diffusers reference, decodes each on CPU and on
device, and scores device against CPU. Prints latency PASS/FAIL against the 300 ms target, and an accuracy
verdict only when `CVD_BASELINE_PSNR` is given -- the accuracy criterion is "no worse than the same levers
unsharded", which has to be measured rather than hard-coded. WAVs land in `$CVD_OUT_DIR` so the difference
can be heard; non-default configurations tag their filenames, so a sharded decode cannot overwrite a
single-device one.

**No numbers are quoted here on purpose.** The figures from the previous base (281.6 ms at 49.45 dB) were
measured with the *fast* levers, and the decoder's constructed defaults on main are accurate mode
(`split_mode="full"`, `tap_matmul=True`, `prefer_mac=True`) -- a slower, much more accurate configuration.
Re-measure before quoting anything.

| variable | meaning |
|---|---|
| `CVD_MESH` | mesh shape, e.g. `4x8`. Default `1x1`. |
| `CVD_T_FACTOR` | T-shard factor. **Must equal the length of the axis it shards** -- on a 4x8 mesh only `(4, axis 0)` and `(8, axis 1)` work; anything else dies in `_partition_t` on a non-tile-aligned slice. |
| `CVD_MESH_AXIS` | which mesh axis carries the T shard. |
| `CVD_TRACED` | replay a captured trace. On the previous base this was worth ~3x on a sharded mesh and only ~1.04x on a single device, because a mesh dispatches to every chip. |
| `CVD_BASELINE_PSNR` | mean PSNR to compare against; unset means report only. |
| `CVD_OUT_DIR` | where the WAVs go. |

## Timing only

    BENCH_N=5 python decode_bench.py <label>

Builds the decoder once, warms it, then reports a median over N decodes. Time one decode per process and
you measure build and JIT noise instead -- that is where an earlier ~1% run-to-run spread came from.

Both scripts read the checkpoint from `MINIMAX_H3_MODEL_PATH` (the variable the test suite uses);
`MINIMAX_H3_DIFFUSERS_DIR` is still accepted.

## Two things that will waste your time otherwise

* **`pytest.ini` sets `timeout = 300`**, and `test_audio_decode_t_parallel` takes several minutes on a 4x8
  mesh. Pass `--timeout 2400` or it is killed and looks like a hang.
* **A wedged card is indistinguishable from a kernel hang.** A bare `ttnn.add` tells them apart in
  seconds; run that before debugging anything else.
