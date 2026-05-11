# Integrating `planar_concat_cpp` into the Wan VAE→ffmpeg pipeline

The C++/AVX2 planar concat extension lands ~2× faster than the
`torch_threaded` reference in the bench harness:

| variant           | CTHW (ms) | CHWT (ms) |
|-------------------|----------:|----------:|
| `torch_threaded`  |     8.02  |    11.01  |
| `cpp`             |    12.15  |     9.52  |
| **`cpp_reused`**  |    **4.15** |    **5.43** |

The 8 ms gap between `cpp` and `cpp_reused` on CTHW is the per-call
`np.empty((T, row_stride), uint8)` first-touch page-fault tax — ~28K 4 KiB
pages × ~250 ns each. Production realises the `cpp_reused` numbers iff the
output buffer is held across calls.

## Test coverage gaps

Today's correctness coverage hits 720p / 4×8 mesh / single seed, plus four
extra shape variants in the standalone C++ test. Before flipping the
production switch, add:

1. **Odd-c UV shape** — every benchmark shape lands all UV writes through
   the AVX2-aligned streaming path; the SSE2-aligned branch in
   `stream_copy_n` (`planar_concat.cpp`) isn't exercised by any test.
2. **Non-720p resolution** — at minimum one smaller frame to confirm the
   shape-resize branch in the singleton-buffer pattern.
3. **Stale-buffer regression test** — call `cpp_reused` twice with
   *different* inputs and assert the second result matches its own
   independent naive reference. Catches "left-over bytes from previous
   call" failures that a single-input two-config test won't see.

## Integration steps

Lowest-risk rollout — bundle steps 1, 2, 5, 6 in one PR; 3, 4, 7 are
housekeeping that can land separately.

1. **Add the parameterized correctness tests** described above to
   `models/tt_dit/tests/unit/test_fast_device_to_host.py`. ~30 min.

2. **Wire `_yuv_planar_d2h`** in `models/tt_dit/utils/tensor.py:666-751`
   to call `planar_concat_cpp` with a module-level persistent output
   buffer, falling back to the existing `torch_threaded` path when
   `HAS_CPP_PLANAR_CONCAT is False`:

   ```python
   _PLANAR_OUT_BUF = {"buf": None, "shape": None}

   def _yuv_planar_d2h(...):
       if HAS_CPP_PLANAR_CONCAT:
           shape = (T, row_stride)
           if _PLANAR_OUT_BUF["shape"] != shape:
               _PLANAR_OUT_BUF["buf"] = np.empty(shape, dtype=np.uint8)
               _PLANAR_OUT_BUF["shape"] = shape
           return planar_concat_cpp(..., out=_PLANAR_OUT_BUF["buf"])
       # existing torch_threaded path
   ```

3. **Build orchestration.** Decide between:
   - **Build-at-install** — invoke `models/tt_dit/utils/cpp/build.sh` from
     the package's install hook (cleanest, but slows `pip install`).
   - **CI-built artefact** — keep manual `build.sh` for devs, add a CI job
     that builds + caches the `.so`. Risk: silent skip on dev boxes that
     never ran build.sh.
   - **Manual + warning** — leave as-is, log a one-line warning at
     `tensor.py` import when `HAS_CPP_PLANAR_CONCAT is False` pointing at
     `cpp/README.md`.

4. **CI coverage.** Add `test_yuv_planar_concat_speed` to the regular test
   suite so regressions show up. Note that without step 3 done, the
   `cpp*` variants silently skip — add an explicit
   `assert HAS_CPP_PLANAR_CONCAT, "C++ planar concat not built"` in a
   `cpp`-only sub-test once the build is wired in.

5. **Runtime CPU feature gate.** The `.so` is compiled with
   `-march=x86-64-v3` (AVX2 + FMA + BMI2). On pre-Haswell hosts the
   binary loads fine and SIGILLs on first call. Add a `__builtin_cpu_supports`
   check in `NB_MODULE(_planar_concat, m)` (bindings.cpp) that throws
   `ImportError` when AVX2 is unavailable, so the Python wrapper's existing
   `try: import` fallback kicks in.

6. **End-to-end pipeline byte-equality check.** Run a full Wan T2V
   generation with the C++ path enabled and disabled; the emitted `.mp4`
   bytes must be identical (host concat is bit-exact, no FP involved).

7. **Perf gate.** Once the end-to-end numbers are trusted, add a
   `_yuv_planar_d2h`-level perf test with a wall-clock assertion so
   accidental regressions (e.g. someone disabling streaming stores) get
   caught at PR time.

## Sfence / visibility note

The C++ kernel emits `_mm_sfence()` at the end of each worker task
(`scatter_one_cthw`), so by the time `pool.run()` returns, all non-temporal
stores are globally visible. Consumers reading the buffer from the same or
any other thread see committed data — no extra fencing needed in Python.

The current pipeline consumer is `ffmpeg` via
`subprocess.Popen.stdin.write(buf.tobytes())`, a sequential read on the
caller thread. Safe. If a future consumer threads the reads, double-check
that no reader runs concurrently with `pool.run()` (currently impossible
because `pool.run()` is blocking).
