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
                              eltwise_binary.h is what add.h and sub.h share; dest.h is the Dest sync API
    resolver/                 plan structs (plan.h), the driver (resolver.h, a chain of one call) and the SFPU
                              activations' names (vec_function.h)
    llk_lib/thread.h          current_thread and the per-thread call pattern
inc/blackhole/                one arch; the build adds inc/common and one inc/<arch>
    resolver/                 each op's OpApi: accepts, options, work, plan; formats.h has the format rules they share
    llk_lib/                  one header per thread or op: unpack.h (every unpack op), matmul.h, eltwise_binary.h,
                              copy.h (math), vec.h (math or pack), pack.h (pack), dest.h (Dest sync),
                              math_common.h (what every math configure shares), semaphores.h
                              (the semaphores the threads hand work over with)
kernels/                      the target kernel, bmm_large_block_zm_fused_bias_activation, on LLK 2.0, and
                              dataflow_tensor.h (an L1 Tensor at a dataflow buffer's pointer)
tests/                        an out-of-tree consumer of the tt-llk test harness; tests/helpers/ has the
                              test-only stand-in for tt-metal's DataflowBuffer
```

The HAL is the typed Blackhole HAL in `tt_llk_blackhole/llk_lib/hal/`, which the harness already puts on the
include path.

## Ops

| Op | Compute API | Blackhole Resolver | Blackhole llk-lib | Tests |
|---|---|---|---|---|
| `matmul` | `compute_api/matmul.h` | `resolver/matmul.h` | `llk_lib/unpack.h`, `llk_lib/matmul.h` | `test_llk2_matmul.py` |
| `add`, `sub` (`Broadcast::Row`) | `compute_api/add.h`, `compute_api/sub.h` | `resolver/eltwise_binary.h` | `llk_lib/unpack.h`, `llk_lib/eltwise_binary.h` | `test_llk2_add.py` |
| `copy` | `compute_api/copy.h` | `resolver/copy.h` | `llk_lib/unpack.h` (incl. unpack to Dest), `llk_lib/copy.h` | `test_llk2_copy.py` |
| `vec_silu`, `vec_tanh`, `vec_gelu`, `vec_gelu_tanh`, `vec_relu_max`, `vec_sigmoid`, `vec_hardsigmoid`, `vec_hardtanh`, `vec_selu`, `vec_softplus` | `compute_api/vec.h` | `resolver/vec.h` | `llk_lib/vec.h` | `test_llk2_vec.py` |
| `pack` (`relu`, `l1_acc`) | `compute_api/pack.h` | `resolver/pack.h` | `llk_lib/pack.h` | `test_llk2_pack.py`, `test_llk2_matmul.py` |
| Dest sync | `compute_api/dest.h` | none | `llk_lib/dest.h` | every test |
| Tensors from dataflow buffers (kernel side; llk2 takes only Tensors) | `kernels/dataflow_tensor.h` | none | none (`tests/helpers/test_dataflow_buffer.h` emulates tt-metal's DataflowBuffer, test only) | `test_llk2_dfb.py` |
| target kernel: bmm_large_block_zm_fused_bias_activation | `kernels/bmm_large_block_zm_fused_bias_activation.h` (body), `.cpp` (as tt-metal would build it) | all of the above | all of the above | `test_llk2_bmm_fused.py` |

Dataflow buffers stay in tt-metal, as in 1.0: a kernel calls `wait_front` / `pop_front` / `reserve_back` /
`push_back` itself and makes an L1 Tensor at the buffer's pointer (`kernels/dataflow_tensor.h`). A pointer, and so an
L1 Tensor's address, is valid only on the thread that owns it (read: unpack, write: pack); llk2 reads L1 addresses
only there.

A test kernel takes a Dest section, runs ops into it and packs it with the Dest sync API (`compute_api/dest.h`):
`dest_acquire`, `dest_commit`, `pack`, `dest_release`. No init call.

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
