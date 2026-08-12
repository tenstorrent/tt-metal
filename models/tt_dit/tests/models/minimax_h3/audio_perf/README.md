# audio_perf

Measurement and diagnosis for the MiniMax-H3 audio decode. Read **`ITEM1_RESULT.md`** and
**`ITEM2_RESULT.md`** first — they carry the conclusions and every number quoted elsewhere.

The shipping result: **281.6 ms at 49.45 dB PSNR vs the CPU reference**, on a 4x8 mesh with
`t_factor=8 axis=1` and tracing, from 0.9304 s single-device.

    CVD_MESH=4x8 CVD_T_FACTOR=8 CVD_MESH_AXIS=1 CVD_TRACED=1 python cpu_vs_device.py

## What each script is for

**Acceptance and timing**

| script | use |
|---|---|
| `cpu_vs_device.py` | **the acceptance measurement** — PSNR vs the CPU reference, .wav files, PASS/FAIL on both criteria. `CVD_MESH`/`CVD_T_FACTOR`/`CVD_MESH_AXIS`/`CVD_TRACED` select the config; no env vars = single device. |
| `decode_bench.py` | timing medians (`BENCH_N=5 python decode_bench.py <label>`). Every median in the docs came from here. |
| `trace_on_mesh.py` | trace vs untraced on the mesh — the 3.06x that produced the result. |
| `factor_scan.py` | latency vs shard factor, and the +343 ms cost of merely opening 32 chips. |

**Correctness, if a sharded decode ever goes wrong again**

| script | use |
|---|---|
| `stage_bisect.py` | instruments the real forward and localizes divergence to a stage. This is what found the `conv_pre` bug — start here. |
| `conv_sharded_probe.py` | sharded-vs-unsharded conv correctness across shapes. Showed only 2048->1024 k=7 was wrong. |
| `halo_correctness.py` | proves `_partition_t` and `_t_neighbor_pad` exact, using a ramp whose value is its global row index. |
| `divergence_probe.py` | separates the T-padding fault from a structural one (run T=256 against T=207). The `t_pad` residual is still open. |
| `fusion_on_mesh.py` | cheap regression for the fusion-plus-mesh crash (two bare `to_torch` calls on replicated weights). |
| `snake_fused_verify.py`, `run_snake_verify.sh` | fused snake vs a float64 golden; the shell wrapper is device health -> regression -> golden -> full gate. |

**Cost models and analysis tools**

| script | use |
|---|---|
| `t_sweep.py` | splits the decode into fixed and data-proportional cost by sweeping T (op count is T-independent). |
| `row_model.py` | rows per stage; sizes a lever before spending device time on it. CPU only. |
| `occupancy.py` | re-reads a profiler CSV *with a denominator* (span, gaps, dispatch) instead of summing FW durations. |
| `op_pipeline.py` | standing counter-evidence that the old "180 us/op device floor" was host issue cost. |
| `mesh32.py` | the two-axis 32-way config. Cannot beat 8-way at 207 latents; keep for clips past ~256 latents. |

**Future work (item 3, the depthwise channel multiplier)**

| script | use |
|---|---|
| `dw_layout_check.py` | host-only weight-layout check, the gate before touching the C++. Passes bit-exact. |
| `band_grouped_multiplier.py` | device acceptance test for the multiplier (2.94x, needs ~5e-08). |

## Reading rules that cost real time to learn

* **Do not quote absolute Tracy totals for this stage.** Two are on record, 224 ms and 1401 ms, and the
  second exceeds wall clock. Per-op *ranking* only. Use `occupancy.py` if you need a total.
* **Trace is 1.04x on one chip and 3.06x on a sharded mesh.** Any untraced multi-chip number understates
  its configuration by ~3x. Always state which configuration a number came from.
* **`pytest.ini` sets `timeout = 300`**, and `test_audio_decode_t_parallel` legitimately takes ~9 min.
  Pass `--timeout 2400` or it dies looking like a hang.
* **The T-shard factor must equal the mesh axis length** — only `(4, axis 0)` and `(8, axis 1)` run on a
  4x8 mesh.
* **18 `DRAM Auto slice` criticals per decode are pre-existing** and identical sharded or not.
* **A wedged card looks exactly like a kernel hang.** A bare `ttnn.add` tells them apart in seconds.

## What was removed

46 one-shot probe scripts and 6 exploration documents (`START_HERE_FUSION.md`, `AUDIO_FUSION_PLAN.md`,
`FUSED_BAND_DESIGN.md`, `AUDIO_KERNELS_BENCH.md`, `AUDIO_RESULTS.md`, `UPSTREAM_TRANSPOSE_TF32.md`) once
their conclusions were captured in `ITEM1_RESULT.md`, `ITEM2_RESULT.md` and `goal.md`'s dead-ends list.
Recover any of them with:

    git log --diff-filter=D --name-only -- models/tt_dit/tests/models/minimax_h3/audio_perf
    git show <commit>^:<path>

Those documents carried the design rationale for the fused-band / snake C++ work, and the source
comments that pointed at them were reworded to state the reason inline rather than cite a deleted file.
If you are working on that C++ and want the fuller derivation, it is in git history.

`PROFILE_2026_08_06.txt` is kept deliberately: it is the raw artifact behind "the profiler's 1401 ms is
not device time", and a retraction whose evidence has been deleted is just an assertion.
