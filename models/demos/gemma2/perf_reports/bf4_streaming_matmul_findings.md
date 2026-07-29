# The bf4 decode matmul ceiling, and the one small thing blocking a fix

Investigation notes for gemma2-9B on 1xP150. Everything below is measured, not
projected, unless it says "projected".

## The problem

Decode on gemma2-9B is weight-bandwidth bound, so bfp4 weights should be roughly
2x faster than bfp8. They are not. Profiling the production `dram_sharded` matmul:

| weight dtype | achieved BW | MBU (of 550 GB/s) |
| --- | --- | --- |
| bfp8 | ~490 GB/s | 89% |
| bfp4 | ~308 GB/s | 55% |

bfp4 halves the bytes but does not halve the time, so most of the theoretical
gain is lost. Because bfp8 reaches 89% MBU on the same kernel and same shapes,
this is not a DRAM limit.

## Cause: batch-1 is padded to a 32-row tile

`tt_transformers` decode carries the activation as `[1, 1, 32, dim]` in standard
`[32,32]` tiles: one real row and 31 rows of padding. The matmul does the MAC
work for all 32. That waste is invisible when DRAM time dominates (bfp8) and
becomes the bottleneck when DRAM time is halved (bfp4).

DeepSeek's `dram_streaming_matmul` avoids it by using `[1,32]` tiny tiles, doing
1 row of MAC work instead of 32. Benchmarked at gemma2 FF1/FF3 shapes
(K=3584, N=14336, bfp4 weights, LoFi):

| kernel | m | time | achieved BW |
| --- | --- | --- | --- |
| production `dram_sharded` | 32 | 93.9 us | 308 GB/s |
| `dram_streaming_matmul` | 1 | 63.2 us | 453-500 GB/s |

PCC 0.99 against golden. So the 308 GB/s ceiling is a property of this kernel at
m=32, not of bfp4 and not of the hardware.

Sanity check that isolated numbers mean anything here: the same isolated harness
measures the production kernel at 93.9 us and the in-model profiler measures it
at 97.7 us, a 4% gap.

## What it would be worth

Substituting the streaming kernel for the DRAM-bound decode matmuls, projected:

- FF1/FF3 only: +9.5% end-to-end
- naive bridging on all decode matmuls: +23%
- chained bridging through the MLP: +30%

## How to feed it: CB tile aliasing (resolved)

The streaming kernel needs `in0` as a `[1,32]`-tiled row, replicated one copy per
compute core, height-sharded. **This is solved and needs no new ttnn API.** Skip
to "The bridge that works" for the answer; the dead end below is recorded because
it is the obvious first thing to try.

### The dead end

Trying to produce a `[1,32]`-tiled *tensor* from a decode activation with device
ops only:

| step | result |
| --- | --- |
| `untilize_with_unpadding` -> extract row 0 | works |
| `ttnn.repeat` -> replicate to 8 cores | works |
| `ttnn.tilize` -> make it `[1,32]`-tiled | **"Physical shard shape (1, 3584) must be tile {32,32} sized"** |
| `ttnn.copy` into a preallocated `[1,32]` buffer | **"Input tensor layout (ROW_MAJOR) must equal output tensor layout (TILE)"** |
| reshard, then `ttnn.copy` | same |
| `ttnn.assign` | same |
| `to_layout(TILE)`, then `ttnn.copy` | **"Input and output tensors must have the same tile shape"** |

No device op can produce a `[1,32]`-tiled tensor, and every copy path enforces
tile-shape equality. Host-side `from_torch(tile=...)` is the only constructor,
which would mean a host round-trip per token. Repro:
`deepseek_v3_b1/tests/unit_tests/test_stream_mm_bridge.py` and
`test_stream_mm_bridge2.py`.

That is a real limitation, but it is the wrong thing to want. A tensor-level
retile is unnecessary.

## The bridge that works

`dram_streaming_matmul` already demonstrates the answer internally: CB4, CB6 and
CB7 view a `[1,32]`-tiled `mm_out_tensor`'s memory as `[16,16]` by building a CB
descriptor from the tensor and then overwriting one field:

```python
cb7 = ttnn.cb_descriptor_from_sharded_tensor(cb_id, mm_out_tensor)
cb7.format_descriptors[0].tile = ttnn.TileDescriptor(ttnn.Tile([16, 16]))
cb7.format_descriptors[0].page_size = mul_tile_size
```

Tile aliasing is therefore already available in Python inside any `generic_op`,
independent of what the backing tensor declares. Applying it to CB0 lets the op
accept a plain ROW_MAJOR activation:

```python
row = ttnn.untilize_with_unpadding(x, [0, 0, 0, K - 1])   # [1,1,1,K] row-major
rep = ttnn.repeat(row, ttnn.Shape([1, 1, num_cores, 1]))  # replicate per core
in0 = ttnn.to_memory_config(rep, height_sharded_1xK)      # [1,K] per core
out = DRAMStreamingMatmul.op(in0, w, out_t, ..., in0_tile=ttnn.Tile([1, 32]))
```

Measured PCC 0.993 against golden, every step on device, no host round-trip and
no retile op. Repro: `test_stream_mm_cb_alias.py`. This needed a 3-line
`in0_tile` override in `micro_ops/dram_streaming_matmul/op.py`, defaulting to the
previous behaviour.

### Why the aliasing is legal

A `[1,32]` tile is 2 faces of `[1,16]`, i.e. 32 contiguous values, which is
exactly a row-major row. Measured on device for `[1,1,8,3584]` bf16 height-sharded
`[1,3584]` (`test_tile_alias_equiv.py`):

| | row-major | `[1,32]`-tiled |
| --- | --- | --- |
| total bytes | 57344 | 57344 |
| shard shape | (1, 3584) | (1, 3584) |
| pages x page size | 8 x 7168 | 896 x 64 |
| values via `to_torch` | identical | identical |

Same bytes in the same order. Only the pagination metadata differs, which is why
a CB can read one as the other.

## Measured: forward bridge is nearly free, and wins

Device times under tracy at FF1/FF3 shapes, bfp4 weights, LoFi
(`bench_bridged_ff.py`, medians over 52 iterations):

| op | device time |
| --- | --- |
| `dram_streaming_matmul` | 63.15 us |
| `repeat` (replicate per core) | 2.14 us |
| `untilize_with_unpadding` (extract row 0) | 2.07 us |
| `interleaved_to_sharded` (reshard) | 0.82 us |
| **forward chain total** | **68.19 us** |
| production `dram_sharded` baseline | 93.90 us |
| **saved per matmul** | **+25.7 us (27% faster)** |

The bridge costs 5.03 us to save 30.75 us, so the economics are decisively
positive. PCC 0.993. FF1 and FF3 share an input, so one bridge serves both.

## MEASURED IN-MODEL RESULT

gemma2-9B, 1x P150, ISL 128 / OSL 200, FF1 and FF3 on the streaming path
(`TT_STREAM_MM=1`), everything else unchanged:

| run | baseline | streaming | gain |
| --- | --- | --- | --- |
| performance mode, run 1 | 40.20 t/s/u | 43.09 t/s/u | +7.2% |
| performance mode, run 2 | 40.30 | 43.23 | +7.3% |
| performance mode, run 3 | 40.37 | 43.14 | +6.9% |
| accuracy mode | 22.21 | 24.79 | +11.6% |

Per token 24.8 ms -> 23.2 ms. The accuracy test passes in both configurations.

Predicted was +9.5%; actual is +7.2%, i.e. about 75% of the per-op arithmetic
survives into the model. Only FF1/FF3 are converted so far; FF2, QKV and WO are
still on the production kernel.

## No reverse bridge needed

An earlier revision of this document claimed integration was blocked on a reverse
retile micro-op. It is not. The streaming matmul writes straight into an ordinary
`[32,32]`-tiled output tensor: DST is physically 32x32 whatever the logical tile,
so with m=1 the result lands in row 0 and pack emits a full tile whose other rows
are junk. That is already the batch-1 padding contract, so stock eltwise ops
consume the result unchanged. Measured PCC 0.993, and it costs nothing (63.12 us
with a standard output vs 63.15 us with a tiny-tile one).

The dead end below is kept because it is the obvious thing to try first.

## Dead end: converting the output tensor

The streaming matmul emits `[1,32]`-tiled output. Stock eltwise ops (the `mul`
between FF1/FF3, the residual add) need standard `[32,32]` tiles, and nothing can
convert back:

| attempt | result |
| --- | --- |
| `to_layout(ROW_MAJOR)` on `[1,32]`-tiled output | `tensor_height % TILE_HEIGHT == 0` -- untilize hardcodes 32 |
| allocate output ROW_MAJOR with `tile=[1,32]` | `Configuring a ROW MAJOR page config with a custom tile configuration is not supported` |
| allocate output with height 32 so untilize's assert passes | assert passes, but `Cannot set circular buffer size to 3670016 ... bank size 114688` |
| same at FF2's narrower N=3584 | same failure, `917504` vs `28672` |

The last two are the same root cause: untilize sizes its circular buffer assuming
32-row tiles, so it over-allocates by exactly 32x. Forward direction works
because CB aliasing happens inside a `generic_op` we control; the reverse has to
go through stock ops we do not.

## How it is wired up

- `models/tt_transformers/tt/stream_mm.py` -- weight shuffle, shared device
  buffers, the bridge, and the matmul wrapper.
- `models/tt_transformers/tt/mlp.py` -- streaming branch for FF1/FF3 in decode.
- `models/tt_transformers/tt/model_config.py` -- `stream_mm_ctx()`, one set of
  buffers for the whole model rather than per layer.

Gated on `TT_STREAM_MM=1`, single device, no prefetcher. Shuffled weights get
their own cache name so they cannot collide with the standard copy.

## Next

FF2, QKV and WO are still on the production kernel. FF2 is the same size as
FF1/FF3, so converting it should give a similar increment; QKV and WO are
smaller. Extending to all of them is the obvious next step.

## Caveats

- The +23-30% assumes the isolated matmul numbers translate in-model. The 4%
  agreement on the production kernel supports that but does not prove it. Only
  the in-model prototype settles it.
- The bridge adds three ops per matmul (unpad, repeat, reshard). Measured cheap
  in isolation (1.4-8.7 us device time), but they eat into the 30 us/matmul
  saving and the real cost in-model is not yet measured.
- The streaming kernel's LLK (`kernel_includes/.../custom_mm.h`) restricts `in0`
  tile height to {1,2,4,8}. m=32 is not merely slow there, it produces garbage,
  so "just use the streaming kernel at m=32" is not an option.
- CB aliasing bypasses tensor-level spec checking. It is correct here only
  because row-major and `[1,32]`-tiled are provably byte-identical; it is not a
  general-purpose escape hatch.
- Weights need the one-time column-major shuffle (`shuffle_tensor_tiles`) and
  DRAM width-sharding at load time, which changes weight caching.

## Reproducing

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole
export TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto

pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_stream_mm_cb_alias.py -s    # the bridge that works
pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_tile_alias_equiv.py -s       # byte equivalence
python models/demos/deepseek_v3_b1/tests/unit_tests/bench_stream_mm.py               # 453-500 GB/s

# the dead end, kept for the record
pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_stream_mm_bridge.py -s       # tilize blocker
pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_stream_mm_bridge2.py -s      # copy-path blockers
```

---

## 2026-07-28: matmul_decode (tiny-tile) integrated + measured

The `smanoj/pi0_tiny_tile` branch's first-class decode matmul,
`ttnn.experimental.matmul_decode`, was surgically ported into this branch
(C++ op + nanobind + CMake), namespace-fixed for the `tt::tt_metal::TensorSpec`
migration, and validated after a clean `./build_metal` rebuild.

### Op is live and correct
- `tests/ttnn/unit_tests/operations/experimental/test_matmul_decode.py`
  - `test_matmul_decode` (full width-sharded): 5/5 PASS (m=1,4,8,16,32)
  - `test_matmul_decode_partial_width_sharded`: 6/6 PASS (m=1..64)

### Measured at gemma2-9B MLP shapes (bf4 weights, batch-1)
Bench: `tests/ttnn/unit_tests/operations/experimental/bench_matmul_decode_gemma2.py`
(P150, 11x10=110 core grid, trace-replay median over 50 reps)

| matmul | shape (k x n) | path | cores | PCC | time | vs prod ~98us |
|--------|---------------|------|-------|-----|------|---------------|
| FF1/FF3 | 3584 x 14336 | partial (k_blk=2, n_blk=32) | 64 | 0.993 | **68.0 us** | -30% |
| FF2     | 14336 x 3584 | partial | 56 | - | L1 clash (kc=7168 too big; needs k_blk=4 tune) |

### The catch that decides everything
`matmul_decode` requires **L1-resident, width-sharded weights**. Production
gemma2 MLP weights are **DRAM-sharded** (`create_dram_sharded_mem_config`,
mlp.py:54-55) and streamed fused with the matmul. So the 68 us is *compute-only,
weights already in L1* -- it does NOT include the DRAM->L1 load.

- A 9B layer's FF weights are ~87 MB bf4; total L1 is ~165 MB. Cannot keep all
  42 layers resident -> weights must come from DRAM every decode step.
- Naive use: DRAM->L1 load (~28.9 MB/matmul) + 68 us compute would be *slower*
  than the 98 us fused production path.
- The only way this wins in-model is an **overlapped DRAM->L1 prefetch**
  (double-buffer layer N+1's weights during layer N compute), i.e. combine
  matmul_decode with the DRAM prefetcher. Then steady-state = max(load, compute).
  Whether that beats 98 us depends on P150 DRAM peak BW -- must be measured.
- Note full-width tiny-tile (m=1, real compute reduction) needs n/64 cores:
  FF1/FF3 n=14336 -> 224 cores > 110, impossible on 1xP150. Only the partial
  path fits, and it pads m to a full 32-row tile (no tiny-tile compute saving).

### Verdict
matmul_decode is now a real, tested tool in the tree. But on a single P150 it
is **not a drop-in win** for FF1/FF3: the shapes force the partial path (m padded
to 32) and the weights must be prefetched DRAM->L1. Next experiment to settle it:
wire FF1 with an overlapped weight prefetch + matmul_decode and measure t/s/u.

### Reproduce
```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole
export TT_VISIBLE_DEVICES=0 MESH_DEVICE=P150
export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
pytest tests/ttnn/unit_tests/operations/experimental/test_matmul_decode.py -s -q
pytest tests/ttnn/unit_tests/operations/experimental/bench_matmul_decode_gemma2.py -s -q
```

### 2026-07-28 (cont): weights CANNOT be DRAM-resident (confirmed at op level)
Tried width-sharding the matmul_decode weight in DRAM (`BENCH_W_DRAM=1`):
`TT_FATAL Logical DRAM core 0-3 outside valid range (num_views=8)`. DRAM has only
8 banks; matmul_decode weights are width-sharded one-shard-per-compute-core and read
from each core's local L1. So the weight is fundamentally **L1-resident** -- there is
no DRAM-weight mode. matmul_decode is therefore NOT a drop-in for the DRAM-streaming
`ttnn.linear`; in-model use REQUIRES staging weights DRAM->L1 every decode step (and
overlapping that stage to have any chance of beating the 98 us fused path).

### 2026-07-28 (final): VERDICT from the 2xP150 profiler — matmul_decode is not a win for gemma2 MLP

Compared matmul_decode against the ACTUAL production decode matmuls (from
`gemma2-9B_2xp150_decode_ops.csv`), not the stale 1xP150 98us figure.

Production decode matmuls (DRAM-sharded, per chip, TP=2):
- Run on **12 cores**, at **~40% DRAM BW (~210 GB/s)** and **~40% FLOPs**,
  op-to-op gap ~0.6us (NOT latency-bound).
- Per-op device times cluster at ~35us (BF16xBFP4, = FF1/FF3 3584x7168 per chip),
  ~60-62us (larger), ~17us (attn).
- Matmuls are 61% of decode time (8498us/180 ops); CCL only ~16%.

matmul_decode at the same per-chip shapes (weights already in L1, 64 cores):
- FF1/FF3 3584x7168: **46.7us**  (vs production ~35us -> SLOWER)
- FF2 7168x3584: 52.3us

Why it loses: the production DRAM-sharded matmul is well tuned - it streams bf4
weights from DRAM fused with compute. matmul_decode (a) can't hold weights in
DRAM at all, so any in-model use adds a per-step DRAM->L1 stage, and (b) even
with weights pre-staged in L1 its gather_in0 + partial-K-reduce across 64 cores
is *slower per op* than the 12-core DRAM-sharded matmul. There is no config on a
110-core P150/P300 that lets the fast full-width tiny-tile path fit (needs
n/64 = 112-224 cores).

**Bottom line:** matmul_decode is now integrated + fully tested (reusable for
models whose weights fit L1, or future HW), but it does NOT speed up gemma2-9B/27B
dense MLP on P150 or P300. Stop pursuing it for this model.

**The real lever the profiler shows:** decode matmuls sit at ~40%/40% on 12 cores.
The upside is in the DRAM-sharded matmul program-config / core-count tuning
(sweeps), not in swapping the op. That is the direction worth the next effort.
