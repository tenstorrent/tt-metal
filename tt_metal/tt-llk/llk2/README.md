# LLK 2.0

A layered rewrite of the compute stack, built one op at a time. Design and op list:
[LLK 2.0 matmul kernel: layout and op list](https://claude.ai/artifact/B6gUhxaMbCzqyx3d3cJhFv).

Calls only go down one layer: Kernel → Compute API → llk-lib (thread split, `configure` / `execute`) → HAL. Tensor,
Resolver and the `ct` / `rt` wrapper are used by more than one layer.

## Reading the code

**One body, three threads.** Every TRISC compiles the same kernel body, so every thread meets every Compute API
call. A call resolves itself at compile time (the Resolver) and then calls llk-lib's per-thread entries,
`llk_lib::<op><On, Plan>(...)`. An entry keeps its body only in thread `On`'s build: `if constexpr (runs_on<On>)`
(`inc/common/llk_lib/thread.h`). TRISC0 is `Thread::Unpack`, TRISC1 `Thread::Math`, TRISC2 `Thread::Pack`. So to
follow one call, read its Compute API function, then the entries it names, each for its own thread.

**Every file of a layer has the same form**, written down once per layer. Read the form, then any one file:

| Layer | The form is in | Then read |
|---|---|---|
| Kernel | `kernels/bmm_large_block_zm_fused_bias_activation.cpp` (the reference kernel) | `kernels/sdpa.cpp` |
| Compute API | `inc/common/compute_api/modifiers.h` ("The form of an op") | `compute_api/matmul.h` |
| Resolver | `inc/common/resolver/resolver.h` ("The form of an op's file") | `inc/blackhole/resolver/matmul.h` |
| llk-lib | `inc/common/llk_lib/thread.h` ("The shape of an llk-lib op") | `inc/blackhole/llk_lib/matmul.h` |

**Words.**

| Word | What it is | Where |
|---|---|---|
| spec | everything about a tensor that is known at compile time: shape, strides, format, storage (L1 or Dest), owner. A constant, passed as a template argument (`Spec`) | `tensor/tensor.h` |
| Tensor | a spec (in its type), an address, and what `release` does with it | `tensor/tensor.h` |
| view | a Tensor of part of another's tile grid (`rows`, `cols`, `cols_up_to`, ...): a new spec, an address computed from the start | `tensor/tensor.h` |
| owner | the (thread, unit) that last wrote a tensor; an op that reads it elsewhere waits for that owner first (the hand-off) | `compute_api/sync.h` |
| modifiers | what a call asks of an op beyond its tensors: one struct per op, a template argument | `resolver/modifiers.h` |
| accepts | the Resolver's first step: whether the op can run on these tensors, as named parts (`Accepts`). The Compute API asserts each part with one sentence | `resolver/resolver.h` |
| plan | the Resolver's result, a `Plan`: what each unit does for the call. With the `OpConfig` and the `Work` it is the whole compile-time side of a call (`resolved<...>`) | `resolver/plan.h` |
| params | one llk-lib op's part of the Plan, as that op's `configure` and `execute` read it (`<op>_params`, the template argument `P`) | each `llk_lib/<op>.h` |
| work | the `Work`: the call's loop counts (`tile_rows`, `tile_cols`, `k_tiles`) and tile geometry | `resolver/plan.h` |
| size, bound, count | a size is a number of tile rows or columns in a spec. A size marked run time is a bound (the most it can be); the count is its value at run time, held by the Tensor | `tensor/tensor.h`, "run-time sizes" |
| counts | the `WorkCounts` of a call: the Work's three counts as `execute` takes them, each a `ct` (a constant in its type) or an `rt` (a run-time value) | `resolver/work_counts.h`, `wrapper/wrapper.h` |
| walk, step, run | the order the unpackers go through their operands: `steps` x `runs` x `tiles_per_run`, one operand held per run and the other streamed | `resolver/plan.h`, `UnpackWalk` |
| configure, execute | the two halves of an llk-lib op: program the unit for the params (compile-time values only), then do the call's work | `llk_lib/thread.h` |

There is no init call: `configure` runs on every call and assumes nothing about what ran before.
(`vec_<function>_init` is a part of `vec_configure`.) Two things are per kernel launch, not per call, because they
are counts in use across calls (`llk_lib/thread.h`, "What is not an op's configure"): a semaphore is set up by its
first post, and the first reserve of a Dest ring section relies on Dest being zero at kernel start. The departures
of single ops from "configure writes all config, execute none" are listed in `llk_lib/thread.h` and in each op's
file comment (unpack, copy to Dest, pack, reduce).

**What llk-lib calls.** The typed Blackhole HAL (`tt_llk_blackhole/llk_lib/hal/`) as it is, for every instruction,
config register, counter and semaphore it offers. What the HAL does not offer lives in
`inc/blackhole/llk_lib/platform/` and is the list of HAL gaps: `sfpu.h` (the SFPU instructions llk-lib issues as
encoded words), `mailbox.h` (the thread mailboxes), `format.h` (format ids and classes), `l1.h` (L1 word addresses
and loads), `gpr_map.h` (the GPR assignment), `limits.h` (the MOP loop-count limit and the unpack config context)
and `issue.h` (how those words are issued, and the scoped wait). These wrap ckernel's and are the only files of
`llk2/` that name it. The one exception is the same for every
SFPU function: the arithmetic in `llk_lib/vec/<function>.h` is SFPI code, which is `sfpi::` C++ where the compiler
schedules the body and the SFPI compiler's instruction builtins (`TTI_SFP*`, with `p_sfpu` and `InstrModLoadStore`
operand names) where the body is a hand-scheduled sequence (the SFPLOADMACRO pipelines and replayed bodies of the
exps and the reciprocals). Besides the HAL, llk-lib uses `LLK_ASSERT` (`llk_assert.h`) and the Dest size constants
of `tensix_types.h`; each file names them under "Not HAL". No LLK 1.0 op (`_llk_*`, `llk_*`) is called from `llk2/`, and
no `ckernel` function or global outside `llk_lib/platform/`.

What is compile time and what is run time:

| Compile time (in types and template arguments) | Run time (in objects and function arguments) |
|---|---|
| a tensor's spec: shape, strides, format, storage, owner; an op's modifiers; the kernel's Dest mode and fidelity | a tensor's address |
| the Resolver's four steps and their results (config, Work, Plan); llk-lib's params; everything `configure` writes | one tile-grid size per tensor, if its spec marks it: the spec holds the upper bound, the Tensor the count ("Run-time sizes", below) |
| a view's length (`rows<N>`, `cols<N>`), or its upper bound (`cols_up_to<N>`) | a view's start, and the length under a bound |
| which thread and unit owns a tensor, so every hand-off | modifiers that are `Toggle::Runtime` (pack's `l1_acc`), an SFPU function's FP32 scalars, a `when` of a buffer call |

## Layout

```
inc/common/                   arch-independent, always on the include path
    kernel_settings.h         Dest mode and math fidelity for the kernel
    tensor/tensor.h           TensorShape, TensorSpec, Tensor, run-time sizes (runtime_rows, runtime_cols), and the
                              views of an L1 tensor (narrow, narrow_up_to, select; row, rows, rows_up_to, cols, cols_up_to, tile)
    wrapper/wrapper.h         ct<V>, rt<T>, ctrt: a value whose kind, compile time or run time, is in its type. What
                              llk-lib's execute takes its loop counts as
    compute_api/              one op per file, on Tensors; modifiers.h has the form every op has (start here);
                              eltwise_binary.h has add, sub and mul; sync.h (owners: complete,
                              migrate; read_value, a scalar on every thread), dest.h (Dest rings), dataflow.h (Tensors from dataflow buffers), counts.h
                              (a call's run-time counts, from its Tensors to llk-lib)
    resolver/                 the modifier structs of every op (modifiers.h), plan structs (plan.h), the driver
                              and the form of an op's Resolver file (resolver.h, a chain of one call), the names
                              of the SFPU functions, their forms and regions (vec_function.h) and of the reductions
                              (reduce_function.h), read_value's accepts (read_value.h),
                              and the counts of a Work as llk-lib takes them (work_counts.h: WorkCounts, plan_count)
    llk_lib/thread.h          current_thread, the per-thread call pattern, and the shape every llk-lib op has
                              (params, params_from, configure, execute, the per-thread entry)
inc/blackhole/                one arch; the build adds inc/common and one inc/<arch>
    resolver/                 each op's OpApi: accepts, options, work, plan; formats.h has the format rules they share
    llk_lib/                  one header per op, all in the one shape: unpack.h (the unpackers, for every op that
                              reads L1), matmul.h, eltwise_binary.h, reduce.h, copy.h (math), vec.h (math or pack),
                              pack.h (pack); and what is not an op: sync.h (owner hand-offs), dest.h (the Dest ring),
                              math_common.h (what the math ops share), formats.h, semaphores.h (the semaphores the
                              threads hand work over with)
    llk_lib/vec/              the SFPU functions' bodies, SFPI code: one file per function (silu.h, tanh.h, gelu.h,
                              exp.h, fast_exp.h, recip.h, ...) and common.h (what they share); vec.h includes them
kernels/                      tt-metal kernels on LLK 2.0, each one .cpp as tt-metal builds it:
                              bmm_large_block_zm_fused_bias_activation.cpp (the reference form), sdpa.cpp,
                              eltwise_binary_no_bcast.cpp
tests/                        an out-of-tree consumer of the tt-llk test harness; tests/helpers/ has the
                              test-only stand-ins for tt-metal's DataflowBuffer and kernel build (compile-time args);
                              tests/host/ has the tests that need no device (the wrapper, the views, which part
                              of accepts refuses a call), which
                              tt-llk/tests/hardware_free/kernels/test_llk2_{wrapper,views,accepts}.py build and run
```

The HAL is the typed Blackhole HAL in `tt_llk_blackhole/llk_lib/hal/`, which the harness already puts on the
include path.

## Ops

| Op | Compute API | Blackhole Resolver | Blackhole llk-lib | Tests |
|---|---|---|---|---|
| `matmul` (`transpose_in1`; `.accumulate`, its only form: it adds into acc) | `compute_api/matmul.h` | `resolver/matmul.h` | `llk_lib/unpack.h`, `llk_lib/matmul.h` | `test_llk2_matmul.py`; Int8 and UInt8 operands: `test_llk2_unpack_signedness.py` |
| `add`, `sub`, `mul` (`Broadcast::Row`, `Col`, `Scalar`; mul in the kernel's fidelity, and with `.accumulate`) | `compute_api/eltwise_binary.h` | `resolver/eltwise_binary.h` | `llk_lib/unpack.h`, `llk_lib/eltwise_binary.h` | `test_llk2_add.py`, `test_llk2_mul.py` |
| `copy` | `compute_api/copy.h` | `resolver/copy.h` | `llk_lib/unpack.h` (incl. unpack to Dest), `llk_lib/copy.h` | `test_llk2_copy.py` |
| the SFPU functions, one family with one modifier struct (`vec_modifiers`): `vec_silu`, `vec_tanh`, `vec_gelu`, `vec_gelu_tanh`, `vec_relu_max`, `vec_sigmoid`, `vec_hardsigmoid`, `vec_hardtanh`, `vec_selu`, `vec_softplus`, `vec_exp`, `vec_recip`. `.form`: `Exact`, `Fast` (tanh, gelu, gelu_tanh, sigmoid, exp, recip), `Polynomial` (exp). `.region`, the datums of each tile a call rewrites: `All`, `LeftFaces` (tile columns 0..15; every form but the fast exp), `LeftFacesEvenColumns` (exp's `Exact` and `Polynomial`). `.clamp_input = false`: the fast exp, which has no input clamping. `.input_scale`: exp. `.thread` | `compute_api/vec.h` | `resolver/vec.h` (`vec_has_form`, `vec_has_region`, `vec_has_clamping`) | `llk_lib/vec.h`, `llk_lib/vec/` | `test_llk2_vec.py` (`test_llk2_vec`: the activations; `test_llk2_vec_exp_recip`: exp and recip in every form and region) |
| `reduce` (row `Max` / `Sum`; `.accumulate`: folded into a seed; in a 32-bit Dest a max is exact in the 19 bits its result holds, and a sum is a product with the scaler) | `compute_api/reduce.h` | `resolver/reduce.h` | `llk_lib/unpack.h`, `llk_lib/reduce.h` | `test_llk2_reduce.py` |
| `pack` (`relu`; `l1_acc`: added into what L1 holds; `sum_tiles`: every tile of the Dest tensor added up into one L1 tile) | `compute_api/pack.h` | `resolver/pack.h` | `llk_lib/pack.h` | `test_llk2_pack.py` (behind a matmul, and `.l1_acc` behind a copy), `test_llk2_matmul.py` |
| owners: `complete`, `migrate` (read after write is inside every op) | `compute_api/sync.h` | none | `llk_lib/sync.h` | `test_llk2_sync.py`, every test |
| a scalar on every thread: `read_value` (a word of a Tensor at a buffer's front; unpack reads it and sends it to math and pack) | `compute_api/sync.h` | `resolver/read_value.h` (an accepts only: no unit runs) | `llk_lib/sync.h` | `test_llk2_read_value.py` |
| Dest ring: `DestRing` (views of the kernel's one ring, any spec that fits a section; `LLK2_DEST_SECTIONS`), `reserve_back<Ring>()`, `release` | `compute_api/dest.h` | none | `llk_lib/dest.h` | `test_llk2_dest.py`, every test |
| Tensors from dataflow buffers: `wait_front`, `peek_front`, `peek_front_rows` (a tensor read as its tile rows arrive), `pop_front`, `reserve_back`, `release`; `buffer_if` for a buffer only some builds of a kernel have (every call on it gives a Tensor that is not there); `words_spec` for a page of scalars | `compute_api/dataflow.h` | none | none (`tests/helpers/test_dataflow_buffer.h` emulates tt-metal's DataflowBuffer, test only) | `test_llk2_dfb.py` |
| target kernel: bmm_large_block_zm_fused_bias_activation | `kernels/bmm_large_block_zm_fused_bias_activation.cpp` (the test builds the same file) | all of the above | all of the above | `test_llk2_bmm_fused.py` |
| kernel: scaled dot product attention | `kernels/sdpa.cpp` (the test builds the same file) | all of the above | all of the above | `test_llk2_sdpa.py` |
| kernel: eltwise_binary_no_bcast (add, sub, mul; blocks of a run-time size) | `kernels/eltwise_binary_no_bcast.cpp` (the test builds the same file) | all of the above | all of the above | `test_llk2_runtime_size.py` (`test_llk2_eltwise_kernel`) |
| a run-time size through each op that takes one | the ops' own files, `compute_api/counts.h` | each op's `accepts` / `work`, `resolver/work_counts.h` | each op's `execute` | `test_llk2_runtime_size.py` |
| checks of what the library relies on (not ops): reconfiguration between ops, the unpackers' context, step and signedness rules, the config words pack shares with math, when the packer and the FPU latch config | none | none | `llk_lib/unpack.h`, `llk_lib/pack.h`, `llk_lib/reduce.h`, `llk_lib/math_common.h` | `test_llk2_reconfig.py`, `test_llk2_unpack_boundary.py`, `test_llk2_unpack_step.py`, `test_llk2_unpack_signedness.py`, `test_llk2_unpack_latch_probe.py`, `test_llk2_pack_shared_config.py`, `test_llk2_pack_l1_acc_latch.py`, `test_llk2_fpu_zero_flag_latch_probe.py`, `test_llk2_reduce_zero_flag_probe.py` |

A form, a region, an unclamped input or an input scale that a function does not have is refused at compile time
(`resolver/vec.h`). The functions written as SFPLOADMACRO schedules (the fast exp, the exact fp32 exp, the
reciprocals) have a second, plain body for the simulator build (`vec_has_loadmacro`, `llk_lib/vec/common.h`): the
board runs only the macro bodies and the simulator only the plain ones.

**One name for one thing across the ops.** `.accumulate`: an FPU op folds its result into what acc holds in Dest
(matmul always; mul; reduce, into a seed; add and sub: not ported). `.l1_acc`: the packer adds into what L1 holds.
`.form`, `.region`: the SFPU functions. `.transpose` (copy) and `.transpose_in1` (matmul): each tile transposed.
`.bcast`: how in1 is spread over in0.

The SFPU is one thread's at a time: a kernel's `vec_` calls that may overlap in time must all name the same thread
(`vec_modifiers::thread`), because SFPLOADMACRO delays count SFPU instructions from any thread (`compute_api/vec.h`).
Nothing checks this at compile time.

Synchronization follows tt-llk-lab llk-tensor (reports 2 and 4): every op returns what it wrote as a Tensor owned by
its (thread, unit), and the next op hands it over, so a kernel writes no math -> pack or FPU -> SFPU sync, and moving
an SFPU call to the pack thread changes only its modifier:

```
// Specs: constants at namespace scope. [2, 4] x [4, 3] tiles of 32x32 -> [2, 3].
inline constexpr auto& x_spec   = llk2::l1_spec<2 * 32, 4 * 32, DataFormat::Float16_b>;      // rows, columns, format
inline constexpr auto& w_spec   = llk2::l1_spec<4 * 32, 3 * 32, DataFormat::Float16_b>;
inline constexpr auto& out_spec = llk2::l1_spec<2 * 32, 3 * 32, DataFormat::Float16_b>;
inline constexpr auto& acc_spec = llk2::dst_spec<2 * 32, 3 * 32, llk2::kernel_dest_format>;  // the result, in Dest
inline constexpr llk2::vec_modifiers on_pack {.thread = llk2::Thread::Pack};
using AccRing = llk2::DestRing<acc_spec>;

// The kernel body, compiled for all three threads. x, w and out are L1 Tensors (from wait_front / reserve_back).
const auto y = llk2::matmul(x, w, llk2::reserve_back<AccRing>()); // y = x w in a Dest ring section, math's
const auto z = llk2::vec_silu<on_pack>(y);                        // math hands y to the pack thread's SFPU
llk2::pack(z, out);                                               // the packer waits for the SFPU
llk2::release(z);                                                 // frees the section
```

In a tt-metal kernel `x`, `w` and `out` come from dataflow buffers and are released back to them:
`const auto x = llk2::wait_front<x_spec>(in0);` ... `llk2::release(x);` (pop), and
`const auto out = llk2::reserve_back<out_spec>(out_buf);` ... `llk2::release(out);` (push).

Dataflow buffers stay tt-metal's: `compute_api/dataflow.h` calls their `wait_front` / `pop_front` / `reserve_back` /
`push_back` and returns the Tensor at the pointer, with the pop or push in its type. A pointer, and so an L1 Tensor's
address, is valid only on the thread that owns it (read: unpack, write: pack); llk2 reads L1 addresses only there.
No init call anywhere (the two per-launch exceptions: "Reading the code").

## Run-time sizes

One tile-grid size of a tensor (its tile rows, or its tile columns) may be known only at run time. The spec then
holds that size's upper bound and a mark, and the Tensor object holds the count (`tensor/tensor.h`, "run-time
sizes"). This is the design, not a placeholder: specs, Plans and llk-lib's params are structural values (template
arguments), and a structural value cannot hold a run-time number. Every compile-time decision is made on the bound;
only loop counts, buffer page counts and pack's MOP count follow the value.

```
const auto a = llk2::wait_front<llk2::runtime_cols<row_spec>>(in0, n);   // n tiles of a row of at most row_spec's
const auto b = llk2::cols_up_to<8>(llk2::row(s, i), c0, n);              // a view: n (1..8) tiles of row i from c0
const auto y = llk2::add(a, b, llk2::reserve_back<AccRing>());           // y holds n: the next op takes it from y
llk2::pack(y, llk2::reserve_back<llk2::runtime_cols<out_spec>>(out, n));
```

How it goes down the stack: the Resolver's `work` names where each of the Work's three counts comes from
(`CountSource`), the Compute API reads the counts from the call's Tensors (`compute_api/counts.h`), and llk-lib's
`execute` takes each loop count as a `ct` (a constant in its type) or an `rt` (`wrapper/wrapper.h`,
`resolver/work_counts.h`), so a call with compile-time sizes compiles to what it did before.

What takes a run-time size: add / sub / mul (also with a row or column broadcast), matmul (K; or N when in0 is one
tile row), reduce (the row length), the SFPU functions and pack (the tile count; pack also with `sum_tiles`),
`wait_front` / `reserve_back`. Limits: one run-time size per tensor; run-time tile columns only where the tiles follow
each other (one tile row; a view may have several); no view of a tensor with a run-time size; copy takes none; a Dest
tensor is always the bound's size.

## Not done

- The Dest mode is compile time; production has to change it mid-kernel (see "Dest mode" in the design doc,
  https://claude.ai/artifact/B6gUhxaMbCzqyx3d3cJhFv).
- The SFPU's one-thread rule (above) is not checked at compile time.
- Wormhole and Quasar: `inc/blackhole/` is the only arch.
- What each Resolver refuses as "not ported" (its file comment's last paragraph, and the sentence of the `accepts`
  part that refuses the call):
  - every op: tile sizes other than 32x32
  - matmul: a matmul that overwrites acc (`.accumulate = false`). It accepts integer formats in a 32-bit Dest: Int8
    and UInt8 operands are tested (`test_llk2_unpack_signedness.py`, exact), Int32 operands are accepted and not tested
  - add, sub, mul: integer formats; `.accumulate` for add and sub
  - copy: integer formats; a run-time size; a window of a wider tile grid as src
  - reduce: column and scalar reduce (`ReduceDim::Col`, `Scalar`); FP16-family formats
  - the SFPU functions: a Dest tensor in another format than the kernel's (FP16, the integers); `Fast` on silu,
    relu_max, hardtanh, hardsigmoid, selu and softplus (they have one form); the fast exp with input clamping or on
    a region; `Polynomial` for anything but exp on `LeftFacesEvenColumns`; `LeftFacesEvenColumns` for a function other
    than exp. `LeftFaces` is the same walk for every function and is tested with exp and recip
  - pack: untilize; a window of a wider tile grid as dst; no test packs to Int8 or UInt8
- Row-major (tilize) input: `Layout::RowMajor` exists in the Plan, no Resolver plans it and llk-lib's unpack refuses it.
- Tensors: no view of a Dest tensor, and none of a tensor with a run-time size (`tensor/tensor.h`).
- Dataflow buffers: no peek and no conditional wait for a tensor with a run-time size (`compute_api/dataflow.h`).
- Owners: a hand-off exists only between math and pack (`llk_lib/sync.h`); a tensor this kernel packed cannot be
  read back through its owner (`compute_api/sync.h`).
- `read_value` sends one word per call through the thread mailboxes. Flow control is the hardware's (a write to a
  full mailbox waits: tt-isa-documentation, `BabyRISCV/Mailboxes.md`); no test holds unpack a known number of words
  ahead.
- The SDPA kernel and the 1.0 dataflow kernels. `kernels/sdpa.cpp` runs with tt-metal's 1.0 reader and writer, which
  do not change. Two things the kernel therefore repeats instead of sharing a definition with them: the order of the
  writer's mask tiles (`generate_lightweight_mask_tiles`), and the reader's K range and keep-K/V arithmetic, which is
  inline in `reader_interleaved.cpp`. The kernel names both where it repeats them.
- The harness build of `kernels/sdpa.cpp` (`test_llk2_sdpa.py`) runs the kernel without causal, a sliding window,
  chunks, sinks, the zigzag order or a sent K range: those need the writer's mask tiles and the reader's pages
  emulated in the test stand-in. tt-metal's SDPA tests run them.
- `TT_SDPA_LLK2`, an environment variable read when the program is built, selects the 2.0 SDPA kernel in tt-metal's
  SDPA op. It is the demo's switch, temporary: a product route would be the op's own choice.
- llk-lib's departures from its own shape, each forced by the hardware or by a measured cost (`llk_lib/thread.h`).

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

The host-only tests (the wrapper, the views, the parts of accepts: compile-time rules, calls that must not compile)
need no device:
`pytest ../../tests/hardware_free/kernels/test_llk2_wrapper.py ../../tests/hardware_free/kernels/test_llk2_views.py ../../tests/hardware_free/kernels/test_llk2_accepts.py`
from `tt-llk/tests/python_tests` with `LLK_HARDWARE_FREE=1`, or all of `tt-llk/tests/run_hardware_free.sh`.

### Compiler

The pinned SFPI (`tt_metal/sfpi-version`, 7.93.0), which `setup_testing_env.sh` installs, builds and passes everything.

Ivan's config-folding SFPI build (`0b10`, a rebuild of `ivan/sfpi-gcc` `ivan/llk-0b10` on the current base) builds
the stack and is the second compiler everything here is checked with. With LLK asserts on, the Bfp8_b tests do not
build with it because of an internal compiler error that is a base GCC 15.1 RISC-V bug (the two L1 address asserts
in `pack_execute`). One wrong fold is known: in a loop it can fold a trailing Dest counter clear into a replay record
and then hoist the unmodified record, which breaks `kernels/eltwise_binary_no_bcast.cpp` at one tile per cycle
through tt-metal (`-mno-tt-tensix-hoist-config` avoids it; the SDPA kernel does not have the pattern). The harness
does not see root-function or link-time folding: a kernel is checked with this compiler through tt-metal as well.
