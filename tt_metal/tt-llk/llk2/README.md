# LLK 2.0

A layered rewrite of the compute stack, built one op at a time. Design and op list:
[LLK 2.0 matmul kernel: layout and op list](https://claude.ai/artifact/B6gUhxaMbCzqyx3d3cJhFv).

Calls only go down one layer: Compute API → llk-lib (thread split, `configure` / `execute`) → HAL. Tensor and
Resolver are used by more than one layer. Everything is compile time except addresses.

## Layout

```
inc/common/                 arch-independent, always on the include path
    kernel_settings.h       Dest mode and math fidelity for the kernel
    tensor/tensor.h         TensorShape, TensorSpec, Tensor
    compute_api/            one op per file, on Tensors, taking the op's modifier struct
    resolver/               plan structs and the driver (a chain of one call)
    llk_lib/thread.h        current_thread and the per-thread call pattern
inc/blackhole/              one arch; the build adds inc/common and one inc/<arch>
    resolver/               each op's OpApi: accepts, options, work, plan
    llk_lib/                one header per thread or op: unpack.h (every unpack op), matmul.h (math),
                            pack.h (pack), dest.h (Dest sync)
tests/                      an out-of-tree consumer of the tt-llk test harness
```

The HAL is the typed Blackhole HAL in `tt_llk_blackhole/llk_lib/hal/`, which the harness already puts on the
include path.

## Ops

| Op | Compute API | Blackhole llk-lib | Tests |
|---|---|---|---|
| `matmul` | `compute_api/matmul.h` | `llk_lib/unpack.h`, `llk_lib/matmul.h` | `test_llk2_matmul.py` |
| `pack` (`relu`, `l1_acc`) | `compute_api/pack.h` | `llk_lib/pack.h` | `test_llk2_pack.py`, `test_llk2_matmul.py` |
| Dest sync | `compute_api/dest.h` | `llk_lib/dest.h` | every test |

A test kernel takes a Dest section, runs ops into it and packs it with the Dest sync API (`compute_api/dest.h`):
`dest_acquire`, `dest_commit`, `pack`, `dest_release`. No init call.

## Running the tests

Set up the harness venv once, in `tt-llk/tests` (`./setup_testing_env.sh` or `setup_external_testing_env.sh`).
Then, from `tt-llk/llk2`:

```
source ../tests/.venv/bin/activate
cd tests
export TT_METAL_SIMULATOR=<dir>/libttsim_bh.so   # soc_descriptor.yaml next to it; see tests/TTSIM.md
export TT_METAL_DISABLE_SFPLOADMACRO=1
pytest --run-simulator --timeout=600 -n 24
```

Drop `--run-simulator` and the two exports to run on a Blackhole device.
