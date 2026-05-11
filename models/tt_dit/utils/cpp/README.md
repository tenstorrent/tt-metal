# `_planar_concat` — C++/AVX2 YUV planar concat

Standalone nanobind extension that replaces the host-side
`planar_concat_torch_threaded` reference (~14 ms CHWT, ~8 ms CTHW for 720p
× 81 frames on a 4×8 mesh) with a `std::thread`-pooled implementation that
uses AVX2 byte-tile transposes for the CHWT scatter.

## Build

```sh
# from this directory:
./build.sh
```

The script installs `nanobind` into the active Python (if missing), runs
`cmake -B build`, and builds. The resulting shared object lands at:

```
build/_planar_concat.cpython-310-x86_64-linux-gnu.so
```

The Python loader stub at `models/tt_dit/utils/planar_concat.py` finds it
via `importlib.util` — no `sys.path` change needed.

### Pointing at a specific Python

```sh
PYTHON=/home/$USER/venv/bin/python ./build.sh
```

### Debug build

```sh
CMAKE_BUILD_TYPE=Debug ./build.sh
```

## Verify

After building:

```sh
pytest models/tt_dit/tests/unit/test_fast_device_to_host.py::test_yuv_planar_concat_speed -s
```

The header line should read `cpp = True`. The `cpp` row should appear in the
table for both CHWT and CTHW dim_orders. Correctness is checked
byte-for-byte against `planar_concat_naive` at test start.

## Layout

| file | purpose |
|------|---------|
| `bindings.cpp` | nanobind `NB_MODULE` entrypoint, arg validation, output buffer allocation |
| `planar_concat.cpp` | static `std::thread` pool, per-shard CHWT/CTHW dispatch |
| `transpose_avx2.cpp` | 16×16 SSE2 byte transpose; 32×32 / 32×N composed from it |
| `CMakeLists.txt` | `nanobind_add_module`, `-march=x86-64-v3 -O3` |

The 32×32 byte transpose is composed from four independent 16×16 transposes
with the standard TR↔BL block swap. The trailing T tile (`T % 32`) and W
tail (UV plane has `w_per_uv = 80 = 2·32 + 16`) use stack temp buffers to
avoid out-of-bounds source reads.

## Expected performance

For 720p × 81 frames × 4×8 mesh, on a host with AVX2 + 8+ cores:

| variant         | torch_threaded | C++ (target) |
|-----------------|---------------:|-------------:|
| CTHW            | 8.18 ms        | ≤ 3 ms       |
| CHWT            | 14.47 ms       | ≤ 5 ms       |

Targets are loose because the actual cost is bandwidth-bound (~225 MB
read+write per call). At ~30 GB/s aggregate DDR throughput across 8 cores,
the floor is ~7–8 ms for 225 MB. CTHW gets close because its inner loop is
already a clean memcpy; CHWT closes the gap by replacing the strided
torch/numpy iterator with a SIMD tile transpose.

## Knobs

```python
from models.tt_dit.utils.planar_concat import set_thread_pool_size
set_thread_pool_size(8)   # must be called before first scatter
```

The pool is created lazily on first scatter and capped at 8 by default
(matches the Python `_DEFAULT_REASSEMBLE_POOL` in `tensor.py`).
