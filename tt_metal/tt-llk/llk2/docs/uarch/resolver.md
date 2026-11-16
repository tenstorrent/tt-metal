# LLK 2.0 Resolver

Status: **draft for review**. Conventions: [README.md](README.md).

For each Compute API call, the Resolver decides at compile time how the hardware does the work.
It reads the operands' specs ([tensor.md](tensor.md)), the op's modifiers and the kernel settings, and returns a `Plan`: register formats, routing to SrcA, SrcB or Dest, loop counts and address steps.
llk-lib executes the `Plan` without knowing which op produced it. [res/plan.h:14-16]

```
 common/resolver/          resolver.h   OpApi, resolve, accepted<>, resolved<>
                           plan.h       Rep, OpConfig, Work, UnpackPlan, UnpackWalk, MathPlan, PackPlan, Plan
                           vec_function.h  VecFunction, vec_call
 blackhole/resolver/       formats.h    shared format rules
                           matmul.h  eltwise_binary.h  copy.h  vec.h  pack.h   one OpApi per op
```

## Contract

### Driver

- **RES-DRV-1** · An op plugs in with an `OpApi<Mods>` of four function pointers: `accepts`, `options`, `work`, `plan`. [res/resolver.h:30-43]
- **RES-DRV-2** · The four are plain `constexpr` functions, not templates; everything they read arrives as a `constexpr` argument. [res/resolver.h:19-20]
- **RES-DRV-3** · `resolve` runs `options`, then `work`, then `plan`, and does not check `accepts`. [res/resolver.h:53-61]
- **RES-DRV-4** · `accepted<Api, A, B, Mods>` and `resolved<Api, A, B, Mods>` are variable templates over `kernel_settings`, so the results are named compile-time objects. [res/resolver.h:63-69]
- **RES-DRV-5** · The Compute API `static_assert`s `accepted<…>` before it reads `resolved<…>`, so a call that cannot run fails the build at the call with a message. [res/resolver.h:12-14] [api/matmul.h:43-46]
- **RES-DRV-6** · The Plan is passed to llk-lib as a class-type template argument, which in C++17 relies on the device compiler's `-ftt-nttp`. [api/matmul.h:54-55] [api/modifiers.h:7]
- **RES-DRV-7** · Each call resolves only itself (a chain of one). [res/resolver.h:5-7]
- **RES-DRV-8** · Every op has two operand slots; single-operand ops pass the same spec twice and ignore `in1`. [bh-res/copy.h:6] [bh-res/vec.h:14]
- **RES-DRV-9** · The SFPU activations share one `vec_op`; each call hands over a `vec_call{VecFunction, vec_modifiers}` as `Mods`. [res/resolver.h:16-19] [res/vec_function.h:34-39]

### Results

- **RES-OUT-1** · `OpConfig{unit, in0, in1, math_fidelity, result}`, where each `Rep` is `{where, format}` and `where` is `None | SrcA | SrcB | Dest`. [res/plan.h:27-34] [res/plan.h:75-90]
- **RES-OUT-2** · `Work{rt_dim, ct_dim, kt_dim, face_r_dim, num_faces}`; `kt_dim` is `no_k_steps` (1) for every op but matmul. [res/plan.h:72-73] [res/plan.h:92-100]
- **RES-OUT-3** · `Plan{op, in0, in1, walk, math, pack, work}`; nothing branches on `Plan::op`. [res/plan.h:171-181]
- **RES-OUT-4** · `UnpackPlan` steps (`tile_step`, `run_step`, `step_step`) are in tiles of that operand. [res/plan.h:102-118]
- **RES-OUT-5** · `MathPlan.function` is op-specific (`BinaryFunction` or `VecFunction`), 0 when unused; `fast` selects the SFPU's approximate mode. [res/plan.h:134-153]
- **RES-OUT-6** · `PackPlan.l1_acc` is a `Toggle`; with `Runtime` the bit comes from the pack call's argument. [res/plan.h:56-63] [res/plan.h:155-169]

### Unpack walk

- **RES-WALK-1** · One unpack implementation serves every op by running this loop nest over at most two streams. [res/plan.h:14-16] [res/plan.h:120-132]

```
for step in [0, walk.steps):                    // matmul: K steps
  for run in [0, walk.runs):                    // matmul: one per reused tile
    unpack the held stream once, if any         // matmul: the reused tile
    repeat walk.tiles_per_run:                  // one MOP iteration each
      unpack every streamed operand; its address += tile_step
```

- **RES-WALK-2** · Each op's `plan()` sets the counts and steps:

| Op | steps | runs | tiles/run | held | in0 → (tile, run, step) | in1 → (tile, run, step) | Source |
|----|------:|-----:|----------:|------|-------------------------|-------------------------|--------|
| matmul, `ct ≥ rt` | kt | rt | ct | SrcB (x) | x → SrcB (0, kt, 1) | w → SrcA (1, 0, ct) | [bh-res/matmul.h:72-113] |
| matmul, `ct < rt` | kt | ct | rt | SrcA (w) | x → SrcB (kt, 0, 1) | w → SrcA (0, 1, ct) | [bh-res/matmul.h:72-113] |
| add / sub | 1 | rt | ct | none | a → SrcA (1, ct, 0) | b → SrcB (1, ct, 0); `Row`: (1, 0, 0) | [bh-res/eltwise_binary.h:87-122] |
| copy | 1 | 1 | rt·ct | none | src → SrcA or Dest (1, 0, 0) | none | [bh-res/copy.h:82-100] |
| vec_*, pack | empty | empty | empty | none | none | none | [bh-res/vec.h:65] [bh-res/pack.h:187] |

- **RES-WALK-3** · Matmul holds the operand on the shorter output side and streams the longer one: when `ct ≥ rt`, x is held in SrcB; otherwise w is held in SrcA. [bh-res/matmul.h:61-69]

### Per-op decisions on Blackhole

- **RES-MM-1** · matmul accepts: both in L1; K matches; 32 × 32 tiles; both formats unpackable to Src; both float or both integer; integers need a 32-bit Dest; no `transpose_in1`. [bh-res/matmul.h:29-40]
- **RES-MM-2** · matmul options: x → SrcB, w → SrcA (as tt-llk), unit `Fpu`, the kernel's fidelity. [bh-res/matmul.h:42-52]
- **RES-MM-3** · matmul result: `[M, N]` in Dest, `Int32` for integer inputs, else `Float32` or `Float16_b` by Dest mode. [bh-res/matmul.h:17-25] [bh-res/matmul.h:50]
- **RES-MM-4** · matmul's math format is w's (in1's) register format. [bh-res/matmul.h:114-121]
- **RES-ELT-1** · add/sub accept 32 × 32 float tiles in L1, of one exponent width; `in1` has `in0`'s sizes, or with `Broadcast::Row` is one tile row of `in0`'s tile columns. [bh-res/eltwise_binary.h:39-64]
- **RES-ELT-2** · add/sub route a → SrcA, b → SrcB on the `Fpu`, always at LoFi (ELWADD/ELWSUB run one phase). [bh-res/eltwise_binary.h:66-77]
- **RES-ELT-3** · add/sub result: `a`'s shape in Dest; `Float32` with 32-bit Dest, else `Float16` for 5-bit-exponent inputs, else `Float16_b`. [bh-res/eltwise_binary.h:29-37]
- **RES-ELT-4** · add and sub differ only in `BinaryFunction`. [bh-res/eltwise_binary.h:8-10] [bh-res/eltwise_binary.h:134-148]
- **RES-CPY-1** · copy accepts 32 × 32 float tiles in L1 that the unpacker can write to Src, or straight to Dest. [bh-res/copy.h:51-59]
- **RES-CPY-2** · copy goes straight to Dest (unit `Unpacker`, math only waits) exactly when the format is 32-bit, Dest is 32-bit and there is no transpose; otherwise src → SrcA and the `Fpu` moves it (MOVA2D). [bh-res/copy.h:8-16] [bh-res/copy.h:31-35] [bh-res/copy.h:61-72]
- **RES-CPY-3** · copy result: the L1 format unchanged when straight, `Float32` with 32-bit Dest, else `Float16` or `Float16_b` by exponent width. [bh-res/copy.h:37-49]
- **RES-VEC-1** · vec_* accepts a Dest tensor in the kernel's Dest format, 32 × 32 tiles, a named function, driven from the math or the pack thread. [bh-res/vec.h:26-33]
- **RES-VEC-2** · vec_* works in place: unit `Sfpu`, result is the operand itself. [bh-res/vec.h:35-45]
- **RES-PACK-1** · pack accepts Dest → L1 with the same sizes, Dest strides on the Dest side, a dense L1 side, offsets 0, and 32 × 32 tiles. [bh-res/pack.h:139-146]
- **RES-PACK-2** · pack accepts only a Dest format that fits the Dest mode, an L1 format with `datum_bits ≠ 0`, and a conversion the packer can do. [bh-res/pack.h:147-148]
- **RES-PACK-3** · `l1_acc` (On or Runtime) only into `Float32`, `Float16_b`, `Float16`; no untilize. [bh-res/pack.h:133-137] [bh-res/pack.h:149]
- **RES-PACK-4** · pack's result is the L1 tensor owned by `(Pack, Packer)`, with fidelity 0. [bh-res/pack.h:153-163]
- **RES-API-1** · For matmul, add, sub and copy, the Compute API checks the Resolver's result spec against the Dest tensor the kernel passed. [api/matmul.h:46-49] [api/eltwise_binary.h:47-48] [api/copy.h:39-40]

### Format rules

- **RES-FMT-1** · `src_register_format(l1, dest_fp32)` gives the Src register format:

| L1 format | Register format | Source |
|-----------|-----------------|--------|
| `Float32`, `Tf32` | `Tf32` with 32-bit Dest, else `Float16_b` | [bh-res/formats.h:23-28] |
| `Float16_b`, `Bfp8_b`, `Bfp4_b`, `Bfp2_b` | same id | [bh-res/formats.h:29-33] |
| `Float16`, `Bfp8`, `Bfp4`, `Bfp2`, `Lf8`, `Fp8_e4m3` | same id | [bh-res/formats.h:34-40] |
| `UInt16`, `Int8`, `UInt8` | same id | [bh-res/formats.h:41-46] |
| `Int32`, `UInt32`, anything else | `Invalid` (Dest only) | [bh-res/formats.h:47-51] |

- **RES-FMT-2** · `pack_in_format(dest, out, dest_fp32)` picks the packer's input format; the first match wins: [bh-res/pack.h:42-67]
  1. `out` is `Bfp4_b` or `Bfp2_b` → `Float32` (32-bit Dest) or `Float16_b`.
  2. 32-bit Dest → `out`.
  3. Dest `Float16` and `out` `Bfp8_b` → `Bfp8`.
  4. B-format Dest and `out` `Float16` → `Float16`.
  5. Otherwise → the Dest format.
- **RES-FMT-3** · The packer converts if either the early or the late stage supports the pair; both tables are ported from `cpack_common.h`. [bh-res/pack.h:69-131]
- **RES-FMT-4** · Unlike the test harness, pack starts from the Dest tensor's format, not the unpacker's output; they differ only for block-float input with a 16-bit Dest. [bh-res/pack.h:13-19]

### Kernel settings

- **RES-SET-1** · `KernelSettings{dest_fp32, math_fidelity}` comes from `LLK2_DEST_FP32` (default 0) and `LLK2_MATH_FIDELITY` (default 0), defined before any LLK 2.0 header. [settings.h:8-37]
- **RES-SET-2** · `kernel_dest_format` is `Float32` with 32-bit Dest, else `Float16_b`. [settings.h:39-46]
- **RES-SET-3** · The fused kernel maps tt-metal's `DST_ACCUM_MODE` to `LLK2_DEST_FP32`. [kernel.cpp:62]
- **RES-SET-4** · The Dest mode is fixed for the whole kernel. [settings.h:5-6]

### Plan to llk-lib

- **RES-LIB-1** · Each llk-lib op has a `constexpr <op>_params_from(const Plan&)`. [lib/unpack.h:22] [lib/matmul.h:69] [lib/eltwise_binary.h:72] [lib/copy.h:75] [lib/vec.h:90] [lib/pack.h:111]
- **RES-LIB-2** · `unpack_params_from` gives unpacker 0 the operand going to SrcA or Dest and unpacker 1 the one going to SrcB, whatever its slot. [lib/unpack.h:156-166]
- **RES-LIB-3** · Unpack steps become 16-byte L1 words: `step · tile_bytes / 16`. [lib/unpack.h:144-151]
- **RES-LIB-4** · `matmul_params_from` sets `reuse_in0 = (walk.held == SrcB)`. [lib/matmul.h:69-75]

### Failure

- **RES-ERR-1** · Every rejection is a `static_assert` at the Compute API call, with one message per op that lists all of its rules. [api/matmul.h:43-44] [api/eltwise_binary.h:42-43] [api/copy.h:37] [api/vec.h:43-44] [api/pack.h:49-50]
- **RES-ERR-2** · A `Toggle::Runtime` modifier is accepted only where `On` would be. [api/modifiers.h:11-13]

## Rationale

- The Plan describes hardware work, not the op, so one unpack implementation replaces the op-specific unpack families. [res/plan.h:14-16]
  The 1.0 families this replaces are `unpack_A`, `unpack_AB`, `unpack_AB_matmul` and `tilize`. [derived: `tt_llk_blackhole/llk_lib/llk_unpack_*.h` on `main`]
- Matmul holds the shorter side so the longer output dimension gets the reuse (RES-WALK-3). [bh-res/matmul.h:61-65]
- `Float32` goes to Src as `Float16_b` with a 16-bit Dest to match the test harness; the code records `Tf32` as an open alternative. [bh-res/formats.h:23-25]
- copy goes straight to Dest for 32-bit data because the trip through 19-bit SrcA would lose bits. [bh-res/copy.h:11-13]
- A transposed 32-bit copy goes through SrcA because UNPACR to Dest with Haloize_mode is undefined. [bh-res/copy.h:14-16]
- add/sub always run LoFi because ELWADD and ELWSUB run one fidelity phase. [bh-res/eltwise_binary.h:74]

## Examples

### Driver code in a Compute API call

```cpp
static_assert(accepted<matmul_op, A, B, Mods>, "matmul takes 32x32 tiles in L1 with matching K, ...");
constexpr const Resolved& r = resolved<matmul_op, A, B, Mods>;
static_assert(equal(r.config.result.shape.sizes, D.shape.sizes) && r.config.result.format == D.format,
              "acc must have matmul's output shape [M, N] and the kernel's Dest format");
llk_lib::unpack<Thread::Unpack, r.plan>(in0.address, in1.address);
llk_lib::matmul<Thread::Math, r.plan>(acc.address);
```

[api/matmul.h:42-55]

### Matmul walk, rt = 2, ct = 4, kt = 3

`ct ≥ rt`, so x is held in SrcB and w streams through SrcA (RES-WALK-3). [derived]

```
step k=0:  run 0: hold x(0,0)  stream w(0,0) w(0,1) w(0,2) w(0,3)
           run 1: hold x(1,0)  stream w(0,0) w(0,1) w(0,2) w(0,3)
step k=1:  run 0: hold x(0,1)  stream w(1,0) … w(1,3)
           …
x offsets (tiles): run_step = kt = 3, step_step = 1   → x(r,k) at 3r + k
w offsets (tiles): tile_step = 1,     step_step = ct = 4 → w(k,c) at 4k + c
```

[derived] from RES-WALK-2.

### A rejected call

A kernel that gives a `Float16_b` accumulator to a matmul under 32-bit Dest fails with "acc must have matmul's output shape [M, N] and the kernel's Dest format". [derived from RES-MM-3, RES-API-1]

## Open items

- **RES-OPEN-1** · A chain of one: no op can pick its formats with the next op in view. [res/resolver.h:5-7]
  The tt-llk-lab llk-tensor driver negotiates across a chain. [PR]
- **RES-OPEN-2** · Every `accepts` requires 32 × 32 tiles, though the Tensor describes others (TEN-OPEN-2). [bh-res/matmul.h:27] [bh-res/eltwise_binary.h:26] [bh-res/copy.h:29] [bh-res/vec.h:24] [bh-res/pack.h:30]
- **RES-OPEN-3** · Not ported: matmul `transpose_in1`, pack untilize, integer eltwise and copy, `mul`, `Col`/`Scalar` broadcast, lossless 32-bit transpose in Dest. [bh-res/matmul.h:39] [api/pack.h:48] [bh-res/eltwise_binary.h:59-62] [bh-res/copy.h:55] [bh-res/eltwise_binary.h:9-10] [bh-res/copy.h:18]
- **RES-OPEN-4** · Address steps assume dense, row-major operands; subblocks need steps from a view's strides (TEN-OPEN-1). [bh-res/matmul.h:64-65] [PR]
- **RES-OPEN-5** · One Dest mode per kernel; production needs to change it mid-kernel, as the design doc linked from `kernel_settings.h` covers. [settings.h:5-6]
- **RES-OPEN-6** · `Float32` → Src with a 16-bit Dest: `Float16_b` (today) or `Tf32`? [bh-res/formats.h:24-25]
- **RES-OPEN-7** · `accepts` returns one bool, so the error message lists every rule rather than the one that failed. [derived from RES-ERR-1] Is a per-rule diagnostic wanted?

<!-- Links below are generated by gen_links.py; do not edit by hand. -->

[PR]: https://github.com/tenstorrent/tt-metal/pull/62961
[api/copy.h:37]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/compute_api/copy.h#L37
[api/copy.h:39-40]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/compute_api/copy.h#L39-L40
[api/eltwise_binary.h:42-43]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/compute_api/eltwise_binary.h#L42-L43
[api/eltwise_binary.h:47-48]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/compute_api/eltwise_binary.h#L47-L48
[api/matmul.h:42-55]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/compute_api/matmul.h#L42-L55
[api/matmul.h:43-44]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/compute_api/matmul.h#L43-L44
[api/matmul.h:43-46]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/compute_api/matmul.h#L43-L46
[api/matmul.h:46-49]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/compute_api/matmul.h#L46-L49
[api/matmul.h:54-55]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/compute_api/matmul.h#L54-L55
[api/modifiers.h:11-13]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/compute_api/modifiers.h#L11-L13
[api/modifiers.h:7]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/compute_api/modifiers.h#L7
[api/pack.h:48]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/compute_api/pack.h#L48
[api/pack.h:49-50]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/compute_api/pack.h#L49-L50
[api/vec.h:43-44]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/compute_api/vec.h#L43-L44
[bh-res/copy.h:11-13]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/copy.h#L11-L13
[bh-res/copy.h:14-16]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/copy.h#L14-L16
[bh-res/copy.h:18]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/copy.h#L18
[bh-res/copy.h:29]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/copy.h#L29
[bh-res/copy.h:31-35]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/copy.h#L31-L35
[bh-res/copy.h:37-49]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/copy.h#L37-L49
[bh-res/copy.h:51-59]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/copy.h#L51-L59
[bh-res/copy.h:55]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/copy.h#L55
[bh-res/copy.h:6]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/copy.h#L6
[bh-res/copy.h:61-72]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/copy.h#L61-L72
[bh-res/copy.h:8-16]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/copy.h#L8-L16
[bh-res/copy.h:82-100]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/copy.h#L82-L100
[bh-res/eltwise_binary.h:134-148]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/eltwise_binary.h#L134-L148
[bh-res/eltwise_binary.h:26]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/eltwise_binary.h#L26
[bh-res/eltwise_binary.h:29-37]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/eltwise_binary.h#L29-L37
[bh-res/eltwise_binary.h:39-64]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/eltwise_binary.h#L39-L64
[bh-res/eltwise_binary.h:59-62]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/eltwise_binary.h#L59-L62
[bh-res/eltwise_binary.h:66-77]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/eltwise_binary.h#L66-L77
[bh-res/eltwise_binary.h:74]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/eltwise_binary.h#L74
[bh-res/eltwise_binary.h:8-10]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/eltwise_binary.h#L8-L10
[bh-res/eltwise_binary.h:87-122]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/eltwise_binary.h#L87-L122
[bh-res/eltwise_binary.h:9-10]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/eltwise_binary.h#L9-L10
[bh-res/formats.h:23-25]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/formats.h#L23-L25
[bh-res/formats.h:23-28]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/formats.h#L23-L28
[bh-res/formats.h:24-25]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/formats.h#L24-L25
[bh-res/formats.h:29-33]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/formats.h#L29-L33
[bh-res/formats.h:34-40]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/formats.h#L34-L40
[bh-res/formats.h:41-46]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/formats.h#L41-L46
[bh-res/formats.h:47-51]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/formats.h#L47-L51
[bh-res/matmul.h:114-121]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/matmul.h#L114-L121
[bh-res/matmul.h:17-25]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/matmul.h#L17-L25
[bh-res/matmul.h:27]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/matmul.h#L27
[bh-res/matmul.h:29-40]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/matmul.h#L29-L40
[bh-res/matmul.h:39]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/matmul.h#L39
[bh-res/matmul.h:42-52]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/matmul.h#L42-L52
[bh-res/matmul.h:50]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/matmul.h#L50
[bh-res/matmul.h:61-65]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/matmul.h#L61-L65
[bh-res/matmul.h:61-69]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/matmul.h#L61-L69
[bh-res/matmul.h:64-65]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/matmul.h#L64-L65
[bh-res/matmul.h:72-113]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/matmul.h#L72-L113
[bh-res/pack.h:13-19]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/pack.h#L13-L19
[bh-res/pack.h:133-137]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/pack.h#L133-L137
[bh-res/pack.h:139-146]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/pack.h#L139-L146
[bh-res/pack.h:147-148]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/pack.h#L147-L148
[bh-res/pack.h:149]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/pack.h#L149
[bh-res/pack.h:153-163]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/pack.h#L153-L163
[bh-res/pack.h:187]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/pack.h#L187
[bh-res/pack.h:30]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/pack.h#L30
[bh-res/pack.h:42-67]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/pack.h#L42-L67
[bh-res/pack.h:69-131]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/pack.h#L69-L131
[bh-res/vec.h:14]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/vec.h#L14
[bh-res/vec.h:24]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/vec.h#L24
[bh-res/vec.h:26-33]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/vec.h#L26-L33
[bh-res/vec.h:35-45]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/vec.h#L35-L45
[bh-res/vec.h:65]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/resolver/vec.h#L65
[kernel.cpp:62]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/kernels/bmm_large_block_zm_fused_bias_activation.cpp#L62
[lib/copy.h:75]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/llk_lib/copy.h#L75
[lib/eltwise_binary.h:72]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/llk_lib/eltwise_binary.h#L72
[lib/matmul.h:69]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/llk_lib/matmul.h#L69
[lib/matmul.h:69-75]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/llk_lib/matmul.h#L69-L75
[lib/pack.h:111]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/llk_lib/pack.h#L111
[lib/unpack.h:144-151]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/llk_lib/unpack.h#L144-L151
[lib/unpack.h:156-166]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/llk_lib/unpack.h#L156-L166
[lib/unpack.h:22]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/llk_lib/unpack.h#L22
[lib/vec.h:90]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/blackhole/llk_lib/vec.h#L90
[res/plan.h:102-118]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/resolver/plan.h#L102-L118
[res/plan.h:120-132]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/resolver/plan.h#L120-L132
[res/plan.h:134-153]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/resolver/plan.h#L134-L153
[res/plan.h:14-16]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/resolver/plan.h#L14-L16
[res/plan.h:155-169]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/resolver/plan.h#L155-L169
[res/plan.h:171-181]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/resolver/plan.h#L171-L181
[res/plan.h:27-34]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/resolver/plan.h#L27-L34
[res/plan.h:56-63]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/resolver/plan.h#L56-L63
[res/plan.h:72-73]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/resolver/plan.h#L72-L73
[res/plan.h:75-90]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/resolver/plan.h#L75-L90
[res/plan.h:92-100]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/resolver/plan.h#L92-L100
[res/resolver.h:12-14]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/resolver/resolver.h#L12-L14
[res/resolver.h:16-19]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/resolver/resolver.h#L16-L19
[res/resolver.h:19-20]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/resolver/resolver.h#L19-L20
[res/resolver.h:30-43]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/resolver/resolver.h#L30-L43
[res/resolver.h:5-7]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/resolver/resolver.h#L5-L7
[res/resolver.h:53-61]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/resolver/resolver.h#L53-L61
[res/resolver.h:63-69]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/resolver/resolver.h#L63-L69
[res/vec_function.h:34-39]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/resolver/vec_function.h#L34-L39
[settings.h:39-46]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/kernel_settings.h#L39-L46
[settings.h:5-6]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/kernel_settings.h#L5-L6
[settings.h:8-37]: https://github.com/tenstorrent/tt-metal/blob/972eb4de3392481e56c7509f58870c754019ed01/tt_metal/tt-llk/llk2/inc/common/kernel_settings.h#L8-L37
