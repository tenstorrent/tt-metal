# Denoise matmul_decode MLP — M=32 vs M=16 (32×32 vs 16×32 tile)

Single-chip (P150, 120 cores) denoise step, **L1 weights + matmul_decode MLP only**
(`PI0_MD_DENOISE=1`), 2 layers, 2-cam prefix=768. Driven by **M** (suffix/action token
count): M=32 is the production tile; M=16 is the tiny-tile experiment.

All numbers below: device 9, `DEVICE KERNEL DURATION` from tracy (not wall-clock).

## Tests

- Full denoise step:   `models/experimental/pi0_5/tests/perf/test_denoise_md_by_m.py`
- MLP matmul tile gain: `models/experimental/pi0_5/tests/perf/test_md_tile_gain.py`
- Op-level matmul:      `tests/ttnn/unit_tests/operations/matmul/test_matmul_decode.py`

## 1. Run the full-denoise M=32 / M=16 tests

```bash
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate

# M=32 — passes. Device-kernel total ~0.33 ms (327 us).
python -m tracy -p -r -n md_m32 -m pytest \
    models/experimental/pi0_5/tests/perf/test_denoise_md_by_m.py -k M32 --device-id 9

# M=16 — expected to FAIL (see §3).
python -m pytest \
    models/experimental/pi0_5/tests/perf/test_denoise_md_by_m.py -k M16 --device-id 9
```

## 2. MD32 full-denoise — captured data

Device-kernel total (2 layers): **327 us ≈ 0.33 ms**, 60 ops.

The 3 MLP matmuls run **full-width** (`partial_width_sharded=False` for gate/up **and**
down — see `ttnn_gemma.py:950-952`; the "partial K-split" code comments are stale).
gate/up N=4096 width-shards over `_md_n_cores(4096)=64` (128 halved to fit 120 cores);
down N=1024 over 32. Profiler reports the compute footprint as 72 / 36 cores.

| MLP matmul (K→N)        | mode       | cores | kernel (M=32) |
|-------------------------|------------|-------|---------------|
| gate  1024→4096         | full-width | 72    | ~5.1 us       |
| up    1024→4096         | full-width | 72    | ~5.1 us       |
| down  4096→1024         | full-width | 36    | ~11.4 us      |
| **MLP matmul / layer**  |            |       | **~21.6 us**  |

Other ops (same for any M): SDPA ~51 us×2 (31%), RoPE ~45 us, attn matmuls, layernorm, etc.

## 3. MD16 full-denoise — currently fails

M=16 reaches the matmul_decode MLP genuinely (tile=16, no pad-to-32) and raises:

```
TT_FATAL: Physical shard shape (16, 512) must be tile {32, 32} sized!
  at _md_shard_a -> ttnn::interleaved_to_sharded   (ttnn_gemma.py:884)
```

Root cause: matmul_decode width-shards activation A, and tt-metal requires tile-aligned
(32-row) shards. `_md_shard_a` uses the default 32×32 tile, so a 16-row A is rejected.
The **op itself supports 16×32** — the unit test sets `tile=ttnn.Tile((16,32))` on A, which
makes the (16,512) shard valid. Wiring that tiny tile into `_md_shard_a` would unlock the
in-model M=16 path.

## 4. 16×32-tile gain over 32×32 for the MLP matmuls (op-level)

Test: **`test_md_tile_gain.py`** — runs each MLP matmul (full-width, model core counts:
gate/up 64-core B-shard / 72-core footprint, down 32 / 36) under the device profiler at
M=32 and M=16, reads `DEVICE KERNEL DURATION`, and asserts 16×32 beats 32×32.

```bash
# asserts the gain — run as plain pytest; run_device_perf wraps the inner op with
# `python -m tracy` internally (per shape, M=16 and M=32):
pytest models/experimental/pi0_5/tests/perf/test_md_tile_gain.py::test_md_tile_gain -v -s --device-id 9

# the inner op alone, profiled — needs tracy to emit device-kernel durations:
python -m tracy -p -r -m pytest \
    models/experimental/pi0_5/tests/perf/test_md_tile_gain.py::test_md_op --device-id 9
python models/experimental/pi0_5/tests/perf/_parse_ops_perop.py   # MatmulDecode durations
```

Measured (device 9, per-op device-kernel):

| MLP matmul (K→N)   | cores | M=32 (32×32) | M=16 (16×32) | gain  | saved   |
|--------------------|-------|--------------|--------------|-------|---------|
| gate/up 1024→4096  | 72    | ~5100 ns     | ~4490 ns     | ~1.13–1.19× | ~700 ns |
| down   4096→1024   | 36    | ~11300 ns    | ~9000 ns     | ~1.25× | ~2300 ns |
| **per layer** (2×gate/up + down) | | **~21.6 us** | **~18.0 us** | **~1.2×** | **~3.6 us** |

So the 16×32 tile saves ~**3.6 us/layer** of MLP matmul time (~1.2×) — **not** 2× despite
half the rows: fixed per-op cost (weight streaming, mcast, K-loop setup) dominates at tiny M.
gate/up has run-to-run variance (~1.13–1.19×); the test asserts `gain >= 1.05`.
