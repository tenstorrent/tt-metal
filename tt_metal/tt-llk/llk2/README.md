# LLK 2.0

A layered rewrite of the compute stack, built one op at a time. Design and op list:
[LLK 2.0 matmul kernel: layout and op list](https://claude.ai/artifact/B6gUhxaMbCzqyx3d3cJhFv).

Calls only go down one layer: Compute API → llk-lib (thread split, `configure` / `execute`) → HAL. Tensor and
Resolver are used by more than one layer. Everything is compile time except addresses.

## Layout

```
inc/common/                   arch-independent, always on the include path
    kernel_settings.h         Dest mode and math fidelity for the kernel
    tensor/tensor.h           TensorShape, TensorSpec, Tensor
    compute_api/              one op per file, on Tensors, taking the op's modifier struct (modifiers.h);
                              eltwise_binary.h has add and sub; sync.h (owners: complete,
                              migrate), dest.h (Dest rings), dataflow.h (Tensors from dataflow buffers)
    resolver/                 plan structs (plan.h), the driver (resolver.h, a chain of one call) and the SFPU
                              activations' names (vec_function.h)
    llk_lib/thread.h          current_thread and the per-thread call pattern
inc/blackhole/                one arch; the build adds inc/common and one inc/<arch>
    resolver/                 each op's OpApi: accepts, options, work, plan; formats.h has the format rules they share
    llk_lib/                  one header per thread or op: unpack.h (every unpack op), matmul.h, eltwise_binary.h,
                              copy.h (math), vec.h (math or pack), pack.h (pack), sync.h (owner hand-offs),
                              dest.h (the Dest ring), math_common.h (what every math configure shares),
                              semaphores.h (the semaphores the threads hand work over with)
kernels/                      the target kernel, bmm_large_block_zm_fused_bias_activation.cpp, on LLK 2.0: one
                              .cpp, as tt-metal builds it
tests/                        an out-of-tree consumer of the tt-llk test harness; tests/helpers/ has the
                              test-only stand-ins for tt-metal's DataflowBuffer and kernel build (compile-time args)
```

The HAL is the typed Blackhole HAL in `tt_llk_blackhole/llk_lib/hal/`, which the harness already puts on the
include path.

## Ops

| Op | Compute API | Blackhole Resolver | Blackhole llk-lib | Tests |
|---|---|---|---|---|
| `matmul` | `compute_api/matmul.h` | `resolver/matmul.h` | `llk_lib/unpack.h`, `llk_lib/matmul.h` | `test_llk2_matmul.py` |
| `add`, `sub` (`Broadcast::Row`) | `compute_api/eltwise_binary.h` | `resolver/eltwise_binary.h` | `llk_lib/unpack.h`, `llk_lib/eltwise_binary.h` | `test_llk2_add.py` |
| `copy` | `compute_api/copy.h` | `resolver/copy.h` | `llk_lib/unpack.h` (incl. unpack to Dest), `llk_lib/copy.h` | `test_llk2_copy.py` |
| `vec_silu`, `vec_tanh`, `vec_gelu`, `vec_gelu_tanh`, `vec_relu_max`, `vec_sigmoid`, `vec_hardsigmoid`, `vec_hardtanh`, `vec_selu`, `vec_softplus` | `compute_api/vec.h` | `resolver/vec.h` | `llk_lib/vec.h` | `test_llk2_vec.py` |
| `pack` (`relu`, `l1_acc`) | `compute_api/pack.h` | `resolver/pack.h` | `llk_lib/pack.h` | `test_llk2_pack.py`, `test_llk2_matmul.py` |
| owners: `complete`, `migrate` (read after write is inside every op) | `compute_api/sync.h` | none | `llk_lib/sync.h` | `test_llk2_sync.py`, every test |
| Dest ring: `DestRing`, `reserve_back<Ring>()`, `release` | `compute_api/dest.h` | none | `llk_lib/dest.h` | every test |
| Tensors from dataflow buffers: `wait_front`, `peek_front`, `pop_front`, `reserve_back`, `release` | `compute_api/dataflow.h` | none | none (`tests/helpers/test_dataflow_buffer.h` emulates tt-metal's DataflowBuffer, test only) | `test_llk2_dfb.py` |
| target kernel: bmm_large_block_zm_fused_bias_activation | `kernels/bmm_large_block_zm_fused_bias_activation.cpp` (the test builds the same file) | all of the above | all of the above | `test_llk2_bmm_fused.py` |

Synchronization follows tt-llk-lab llk-tensor (reports 2 and 4): every op returns what it wrote as a Tensor owned by
its (thread, unit), and the next op hands it over, so a kernel writes no math -> pack or FPU -> SFPU sync, and moving
an SFPU call to the pack thread changes only its modifier:

```
const auto x   = llk2::wait_front<x_spec>(in0);                           // dataflow buffer -> Tensor
const auto y   = llk2::matmul(x, w, llk2::reserve_back<AccRing>());       // a Dest ring section, math's
const auto z   = llk2::vec_silu<{.thread = llk2::Thread::Pack}>(y);       // math hands y to the pack thread
const auto out = llk2::pack(z, llk2::reserve_back<out_spec>(out_buf));    // a STALLWAIT for the SFPU
llk2::release(out); llk2::release(z); llk2::release(x);                   // push, free the section, pop
```

Dataflow buffers stay tt-metal's: `compute_api/dataflow.h` calls their `wait_front` / `pop_front` / `reserve_back` /
`push_back` and returns the Tensor at the pointer, with the pop or push in its type. A pointer, and so an L1 Tensor's
address, is valid only on the thread that owns it (read: unpack, write: pack); llk2 reads L1 addresses only there.
No init call anywhere.

## Running the tests

Set up the harness venv once, in `tt-llk/tests` (`./setup_testing_env.sh` or `setup_external_testing_env.sh`).
Then, from `tt-llk/llk2`:

```
source ../tests/.venv/bin/activate
cd tests
export TT_METAL_SIMULATOR=<dir>/libttsim_bh.so   # soc_descriptor.yaml next to it; see tt-llk/tests/TTSIM.md
export TT_METAL_DISABLE_SFPLOADMACRO=1
pytest --run-simulator --timeout=600 -n 24
```

Drop `--run-simulator` and the two exports to run on a Blackhole device.
