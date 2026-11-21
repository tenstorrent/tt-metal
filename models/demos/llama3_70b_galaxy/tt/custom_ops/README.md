# Llama3-70B Galaxy (Qwen3-32B) model-local custom ops

`ttnn.generic_op` kernels and their Python wrappers. Building the program from Python means a change
to a descriptor or a kernel needs no `_ttnn.so` rebuild. Conventions follow
`models/demos/wormhole/bge_m3/tt/custom_ops/README.md`.

## Layout

    custom_ops/
        __init__.py              package surface
        README.md                this file
        qwen_q_prep/             Qwen long-prefill Q prep (typecast, q_norm, RoPE, typecast)
            __init__.py, op.py
            kernels/             qprep_common.hpp, norm_{reader,compute,handoff_writer}.cpp,
                                 rope_{reader,compute,writer}.cpp
        qwen_width_norm/         Qwen context-parallel local RMSNorm, width split with a partial-sum exchange
            __init__.py, op.py
            kernels/             wnorm_common.hpp, reader.cpp, compute.cpp, writer.cpp
        qwen_cp_sdpa/            Qwen context-parallel causal SDPA, both ring slices of a column in one program
            __init__.py, op.py
            kernels/             cp_{reader,compute,writer}.cpp (the op), cp_step.hpp (its K-chunk step),
                                 stock_{reader,compute,writer}.cpp (verbatim copies of the stock SDPA kernels,
                                 used by chunked_sdpa_port)
        qwen_ag_matmul/          Qwen context-parallel row all-gather along M (fabric, from the op) + QKV matmul
            __init__.py, op.py
            kernels/             mm_{in0,in1_out,compute}.cpp (minimal_matmul kernels with block lists),
                                 ag_sender.cpp (fabric multicast sender), ag_hub.cpp (barrier + block release),
                                 v2_in0.cpp (7x10 form: in0 relay with the fabric sender folded in)

## Conventions

- A kernel path in `op.py` is relative to `TT_METAL_HOME`.
- Each sub-package exposes a callable that takes and returns ttnn tensors.
- An op validates its inputs and raises `ValueError` when they do not match. It does not fall back
  to the stock ops.

## Status

`qwen_q_prep` runs in the model on the Qwen Wormhole long-prefill path (batch 1, S > 8192, prefetcher):
`TtLlamaAttention._qwen_qkv_prep_direct` calls it twice per layer, for Q (this column's two ring slices)
and for K (`all_rows=True`), both reading their heads straight from the fused QKV buffer (`src=`), so
`nlp_create_qkv_heads` does not run there. `test_qwen_q_prep_op.py` (Q, in place),
`test_qwen_k_prep_op.py` (K, in place) and `test_qwen_qkv_direct_op.py` (Q/K/V from the fused buffer)
pass `torch.equal` on all 32 devices at S=16384 and 40960 against the stock-op methods
(`_qwen_local_q_prep_stock`, `_k_prep_stock`, `nlp_create_qkv_heads`), including a second call on
fresh buffers (program-cache hit path), and the 40k gate numbers are unchanged.
Measured: isolated wall 1.278 -> 0.889 ms at S=40960 (0.590 -> 0.984 ms at 16384, host-dispatch
bound); b1 40k TTFT 3655.0 -> 3596.3 ms.

## qwen_q_prep

Replaces `TtLlamaAttention._qwen_local_q_prep` (`tt/llama_attention.py`, around line 1108), which runs
these ops on each device: 2 x `ttnn.slice` + `ttnn.concat`, `ttnn.typecast` bf8 -> bf16,
`q_norm` (`ttnn.rms_norm`), `ttnn.experimental.rotary_embedding_llama`, `ttnn.typecast`
bf16 -> bf8, and `ttnn.experimental.paged_fill_cache`. The op runs one program per device and aims to
give the same bits.

### Contract

`qwen_q_prep(q, cos, sin, trans_mat, gamma, *, eps, grid=None, handoff_depth=2) -> q`

Per device of a `(rows, cols)` mesh, column `c` owns ring slices `c` and `2*cols-1-c` of `2*cols`:

| Tensor | Shape, dtype, layout | Notes |
| --- | --- | --- |
| `q` | `[1, H, S, 128]` bf8, TILE, DRAM interleaved | Updated in place on the owned slices. Other slices are not touched. Returned. |
| `cos`, `sin` | `[1, 1, S_rot, 128]` bf16, TILE, interleaved, `S_rot >= S` | The full prefill rot mats. Row `t` is position `t`. No per-layer slice. |
| `trans_mat` | `[1, 1, 32, 32]` bf16, TILE, interleaved | `transformation_mats["prefill"]` |
| `gamma` | `[1, 1, 4, 32]` bf16, ROW_MAJOR, interleaved | `q_norm.weight` |
| `eps` | float | `q_norm.eps`. The reader truncates it to bf16, as the stock reader does. |

`S % (2 * cols * 32) == 0`. For the 8x4 Galaxy that means `S % 256 == 0`, so every multiple of 8192 works.

Options (keyword-only):

- `all_rows=True`: process every position (two "slices" of S/2 rows on every device) instead of the
  column's ring slices. Used for K (`H = 1`), with `k_norm`'s gamma and eps.
- `src=xqkv, src_col=c`: read the input rows from a fused `[1, 1, S, W]` bf8 buffer, head `h` at columns
  `c + 128 * h`, instead of from `q`; `q` is then only written (rows the op does not own stay as they
  were, uninitialised for a fresh buffer). The N reader computes the source page as
  `src_base + t * src_row_stride + h * src_head_stride` (runtime args 11-13); without `src` these are
  `(0, 4, S/32 * 4)`, i.e. the `q` buffer itself.
- `q_out1=q1` (with `src`, not `all_rows`): split output. `q` and `q1` are `[1, H, S/(2*cols), 128]` bf8
  buffers; the R writer stores local slice 0 (ring slice `c`) in `q` and local slice 1 (ring slice
  `2*cols-1-c`) in `q1`, each row at page `(h * sl_t + tl) * 4` of its slice (`RowWalker::slice_page`, writer
  runtime args 10-11). The values are those of the natural form; the context-parallel prefill hands the two
  buffers straight to its two chunked SDPA calls instead of slicing a natural `[1, H, S, 128]` Q.

### Program

The compute grid (7x10 here) is split into 35 pairs: N cores are rows 0-4, and R core `(x, y+5)`
pairs with N core `(x, y)`. The owned tile rows (`2 * S/8/32 * H`, 2560 at S=40960) are dealt out to
the pairs in contiguous ranges, head index fastest, so consecutive rows share one position.

| Core | Reader | Compute | Writer |
| --- | --- | --- | --- |
| N | q bf8 tile rows from the full buffer, plus the scaler, eps and gamma | typecast + RMSNorm. HiFi2, fp32 DEST, the q_norm config. | Writes each bf16 row into the paired R core's `CB_NORMED`. |
| R | cos/sin once per position, trans_mat once. Receives the rows from N. | RoPE + typecast. HiFi4, 16-bit DEST, `bfp8_pack_precise`. | Writes bf8 rows to the same q pages. |

Why two core groups: the stock RMSNorm runs at HiFi2 with fp32 DEST, and the stock RoPE runs at
HiFi4 with 16-bit DEST. In 16-bit DEST, each HiFi4 fidelity phase of the RoPE multiplies accumulates
into a bf16 DEST, so RoPE in fp32 DEST would give different bits. A compute kernel has one fidelity
and one DEST width.

The handoff works per row `r` with two semaphores:

1. The R reader reserves `CB_NORMED` slot `r % depth` and increments `SEM_CREDIT` on the N core.
2. The N writer waits until `SEM_CREDIT >= r+1`, writes 8 KB to the R core's slot, and waits for the
   write acks.
3. The N writer increments `SEM_READY` on the R core.
4. The R reader waits until `SEM_READY >= r+1`, then pushes the slot to its compute kernel.

`CB_NORMED` is declared on every core, so it has one L1 address on all of them.

Writing in place is safe. Each owned tile row is read by exactly one N core. The paired R core
writes that row only after it has received the row from the N core, so the read is complete before
the write. No other core reads or writes the row. Ring SDPA runs after the op.

### Bit-identity argument, stage by stage

| Stage | Mirrors | Confidence |
| --- | --- | --- |
| Slice + concat + paged_fill_cache | Tile-aligned page copies of bf8 tiles. The op reads and writes the same pages directly. The chain's page table maps block `2h+j` to `8h+slice`, `llama_attention.py` `_qwen_local_q_setup`. | High |
| Typecast bf8 -> bf16 | `typecast_tile<Bfp8_b, Float16_b>` has no SFPU work (`tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h:313`), so the stock op is copy + pack. Every bf8 value fits exactly in bf16, so the copy through fp32 DEST in `norm_compute.cpp` `typecast_row` packs the same bf16 tile. | High |
| RMSNorm config | `RMSNorm.compute_kernel_config_hifi2` (`models/common/rmsnorm.py`): HiFi2, approx off, fp32 DEST. `ttnn.rms_norm` without a program config gives `LayerNormDefaultProgramConfig{use_welford=false}`, so the tile-reduction backend (`layernorm_stats_selector.hpp`) runs `layernorm.cpp` with RMSNORM + FUSE_GAMMA, Wt=4, block 4, and Float32 intermediates (`layernorm_op_multi_core.cpp:485-489, 1016-1024`). The gamma read is `reader_unary_interleaved_ln_rm_gb.cpp`. | High |
| x^2 | `layernorm.cpp:207-208` reconfig, `:243` `ckl::square`, which is `mul_init(x, x, false)` + `mul_tiles` (`kernel_lib/eltwise/core/chain.inl:1184, 1244`). | High |
| mean(x^2) | `layernorm.cpp:256` reconfig, `:261` `numeric::row_wise_mean` (`normalization/kernel_util/compute/numeric.h`): the swapped reconfig, `reduce_init<SUM, REDUCE_ROW>`, `reduce_tile` on tiles 0..3 into DST 0, then `mul_unary_tile(1/128)` on the SFPU. Same order, scaler tile and epsilon tile (`reduce_helpers_dataflow.inl`, `generate_bcast_scalar_metal2.hpp`). | High. Device confirmation needed (reduce order). |
| rsqrt(var + eps) | `layernorm.cpp:268-275`: `add_init(ex2, eps, false)`, `add_tiles`, `rsqrt_tile<RsqrtMode::Default>` (ckl `Rsqrt<Exact>`). The init order follows bge `balanced_layernorm`, which is bitwise against stock. | High |
| x * rstd * gamma | `layernorm.cpp:283-345`: col-bcast mul packed to the fp32 FUSION CB, then unpacked and row-bcast multiplied by gamma, then packed bf16. The fp32 round trip is kept, unlike the bge op. | High |
| RoPE config | `rotary_embedding_llama_device_operation.cpp:246-247`: default HiFi4, fp32 DEST off. The interleaved factory has bf16 intermediates and `RELOAD_IMPL=1` at this shape (`rotary_embedding_llama_multi_core_program_factory.cpp`). | High |
| RoPE sequence | `rotary_embedding_llama.cpp:85-89` (two hw_startups), `:122-130` matmul, `:132-142` `* sin` (`mul_init` with default `acc_to_dest=true`), `:144-158` `* cos` (no re-init), `:160-164` add. `rope_compute.cpp` makes the same calls in the same order. | High |
| cos/sin reuse | The op keeps a cos/sin row for up to H consecutive rows instead of re-reading it per head. The L1 data is the same. | High |
| Typecast bf16 -> bf8 | `typecast.cpp`: `fp32_dest_acc_en=false`, `bfp8_pack_precise=true`, and `typecast_tile<Float16_b, Bfp8_b>` is a no-op. The stock path packs the RoPE add from Float16_b DEST to bf16 (exact), copies it back to Float16_b DEST, and packs to bf8 with `pack_src_format=Float16_b` (`jit_build/data_format.cpp:329-331`). The op packs the add's DEST straight to bf8 with the same pack-source format. `bfp8_pack_precise` is kernel-wide but only changes bfp-format output CBs, and `CB_QOUT` is the only one. | Medium-high. Device confirmation needed. |

`test_qwen_q_prep_op.py` checks all of the above at once with `torch.equal` on every device.

### Expected time

Per device at S=40960. The chain takes about 1.26 ms per layer:
`keeps/08-qwen-local-q-prep.md`, 1.782 ms measured minus a 0.523 ms clone. The k04-2l profile shows
each op in the chain running at 175-205 GB/s of DRAM traffic, so the chain is bound by DRAM bandwidth.

| | Chain | qwen_q_prep |
| --- | --- | --- |
| DRAM bytes | about 256 MB | about 28 MB |
| Ops / programs | 8 | 1 |

The chain's 256 MB break down as follows:

| Ops | MB |
| --- | --- |
| slices | 22 |
| concat | 22 |
| typecast in | 32 |
| norm | 42 |
| RoPE (x, plus cos/sin re-read per row) | 84 |
| typecast out | 32 |
| paged fill | 22 |

The op's 28 MB are 11 MB of q read, 11 MB of q write, and 6 MB of cos/sin read. On top of that it
moves 21 MB L1-to-L1 on chip.

- The op is compute-bound, because each stage gets only 35 cores. The stock RMSNorm takes at most
  5.8 µs per row and the stock RoPE at most 11.2 µs per row, and both of those are DRAM-bound. Each
  pair handles 73 or 74 rows. That gives about 0.25-0.45 ms for the slower stage plus a short
  pipeline fill.
- Estimate: 0.3-0.5 ms against about 1.26 ms, a saving of about 0.8-1.0 ms per layer. Over 64
  layers that is about 50-60 ms of b1 40k TTFT, or about 1.2-1.5%.

These are estimates, not measurements. The test prints wall time for both paths.

### llama_attention.py integration (applied)

```diff
--- a/models/demos/llama3_70b_galaxy/tt/llama_attention.py
+++ b/models/demos/llama3_70b_galaxy/tt/llama_attention.py
@@ imports
 import torch
 import ttnn
 from models.common.lightweightmodule import LightweightModule
 from models.common.rmsnorm import RMSNorm
+from models.demos.llama3_70b_galaxy.tt.custom_ops.qwen_q_prep import qwen_q_prep
@@ def _qwen_local_q_prep(self, q_pre_bf8, rot_mats, seq_len):
     def _qwen_local_q_prep(self, q_pre_bf8, rot_mats, seq_len):
         """Q prep on this column's two ring-SDPA slices only; returns the full bf8 Q buffer (other slices unused)."""
+        if self.qk_norm:
+            # One generic_op program per device (custom_ops/qwen_q_prep): reads the owned slices and the
+            # positions' cos/sin straight from the full buffers and rewrites the slices in place.
+            return qwen_q_prep(
+                q_pre_bf8,
+                rot_mats[0],
+                rot_mats[1],
+                self.transformation_mats["prefill"],
+                self.q_norm.weight,
+                eps=self.q_norm.eps,
+            )
         starts, ends_q, page_table, (cos_local, sin_local) = self._qwen_local_q_setup(rot_mats, seq_len)
```

`_qwen_local_q_setup` is then used only when `qk_norm` is off, which is not the case for Qwen.
For an A/B run, gate the new branch on an environment variable until the device test passes.

### Open risks

1. Nothing has been compiled. API names and signatures were checked against the headers in this
   tree (bge custom ops use the same APIs), but expect a compile-fix round.
2. The handoff protocol (semaphores and the remote `CB_NORMED` address) can hang if it is wrong. Two
   things to check: that `get_write_ptr(CB_NORMED)` on the N core equals the R core's ring base, and
   that the semaphores are initialised before any core starts.
3. The 16-bit-DEST bf8 pack in `rope_compute.cpp` is the bit-identity step argued from the pack
   formats alone. If it differs, add a bf16 CB and a separate copy + pack stage to mirror the
   typecast exactly.
4. The `MeshProgramDescriptor` has 4 column entries. On a program-cache hit, `generic_op` must take
   the new buffer addresses from the runtime args, as the bge ops rely on. The test reuses buffers,
   so a new address only shows up across layers in the model.
5. If prefill runs under a sub-device manager whose worker cores exclude part of the grid, pass
   `grid=`.
6. The N:R split is fixed at 1:1. If one stage is much slower than the other, pairing 2 N cores to
   1 R core (or the reverse) is the next lever. So is reading bf8 straight into the norm, which drops
   the explicit typecast copy.
7. Pre-existing and out of scope: `q_norm` is built without `eps`, so it uses the RMSNorm default
   1e-5, not Qwen's 1e-6. The op takes `q_norm.eps`, so it stays identical to the chain either way.

### Test

    source python_env/bin/activate
    pytest models/demos/llama3_70b_galaxy/tests/test_qwen_q_prep_op.py -s

`test_qwen_q_prep_bit_exact[16384|40960]` needs the 8x4 mesh. `test_qwen_q_prep_plan_covers_owned_rows`
checks the work split on the host only.

## qwen_width_norm

Replaces the stock 2-slice form of `qwen_cp_prefill.local_norm` (2 width slices, `rms_norm_pre_all_gather` x2, stats
concat, `rms_norm_post_all_gather` x2, output concat: 8 ops) on the layout-B residual `[1, 1, S/32, 5120]`
(bf16, or bf8 for the first layer) -> bf8. `QWEN_CP_WIDTH_NORM=0` keeps the stock form.

Program (one per device, all devices the same): core `(cx, cy)` of the 7x10 grid owns width chunk `cx` (whole blocks
of 4 tiles: 24 or 20 tiles) of the tile rows of row group `cy`. Per tile row: partial `sum(x^2) / W` over the chunk
(fp32), written to slot `cx` of `CB_GATH` on the 7 cores of the grid row (two row-halves, cumulative `SEM_RECV`),
the 7 partials added in slot order, `rsqrt(sum + eps)`, `y = x * rstd * gamma` packed to bf8. x is read once. The
per-core work comes from the logical coords in the kernels, so the descriptor has only common runtime args (the
per-core-args first version cost 0.58 ms of host time per call; now 0.16 ms).

Not bit-identical to the stock form (fp32 partials instead of bf16 stats; the model calls it at HiFi4). Measured:
cpb_sweep job211 S=40960: 0.580 -> 0.296 ms per norm, rel_l2 vs fp32 1.329e-2 -> 1.230e-2;
`tests/test_qwen_width_norm_op.py` passes at S = 16k/24k/32k/40k for bf16 and bf8 inputs (closer to fp32 than the
stock form at every bucket); b1 40k TTFT 2129.9 -> 2070.0 ms; 40k gate PASS (pcc40k 0.9993625 -> 0.9995179).

## qwen_cp_sdpa

Replaces the attention core of the context-parallel prefill (`tt/qwen_cp_prefill.py`): two stock
`chunked_scaled_dot_product_attention` calls (one per owned ring slice, per-device chunk start), `ttnn.concat` and
`nlp_concat_heads`. `qwen_cp_sdpa(q0, q1, k, v, page_table, *, scale, compute_kernel_config, program_config)` takes
the two per-slice Q buffers of `qwen_q_prep(q_out1=...)`, the scratch K / V `[S/64, 1, 64, 128]` and the identity
page table, and returns `[1, 1, S/4, H * 128]` in the column's local order.

Program (one per mesh column): the work items are every (slice, head, q chunk) of both slices; each core runs an
explicit list of them (runtime args), balanced over all 7x10 cores by K-chunk cost (LPT + seeded annealing,
`schedule`). The reader / compute / writer are the stock streaming SDPA kernels reduced to the path the prefill runs
(causal, paged K / V, lightweight mask); the compute is `sdpa_standard_v2` with only the Q loop changed, so each Q
chunk is computed exactly as its stock call computes it (same K-chunk order and inner step). The writer stores each
chunk at its local row in the concat-heads layout.

Checks (`tests/test_qwen_cp_sdpa_op.py`): `chunked_sdpa_port` (the stock factory in Python, stock or local kernel
files) `torch.equal` to the stock call; `qwen_cp_sdpa` `torch.equal` to the pair + `nlp_concat_heads` on all 32
devices at S = 16384 / 32768 / 40960. 40k pipelined: 13.364 -> 12.476 ms per layer. The gain is bounded by the
q320 item granularity (busiest core 120-123 K iterations vs 130 for the pair); smaller q chunks change bits.
`QWEN_CP_LOCAL_SDPA=0` runs the stock pair.

The K-chunk step is `cp_inner_loop_step` (`kernels/cp_step.hpp`): the stock streaming `sdpa_inner_loop_step`
specialised to this configuration, with two packer-stall cuts that leave every value unchanged (row-sum packs
grouped to one L1-acc switch per sub_exp block; no mask stamp for row groups wholly below the diagonal). 40k
pipelined: 12.490 -> 12.230 ms per call; output `torch.equal` to the stock pair. `QWEN_CP_SDPA_DEFINES` passes
extra compile-time defines to the kernels (measurement knobs listed in `cp_step.hpp`; unset in the model).

## qwen_ag_matmul

Replaces `_row_ag` (all_gather_async along dim 2 over the mesh rows, FABRIC_1D_RING) + the QKV
`minimal_matmul` of `qwen_cp_prefill.attention_prefill` with one `ttnn.generic_op` program per device
(on by default for S >= 32768; `QWEN_CP_FUSED_AG=0` keeps the stock pair). The FF1 pair of
`mlp_prefill` uses `qwen_ag_matmul_v2` below (on by default for S >= 32768; `QWEN_CP_FUSED_AG_FF1=0`
keeps the stock pair). The fabric connections come from `ttnn.setup_fabric_connection` on the program
descriptor, so no C++ op is needed. Both ops need the 8 mesh rows as one FABRIC_1D_RING (`check_ring`
raises `ValueError` otherwise). The semaphore words are freed at the end of every prefill, also when
it raises (`release_fused_ag` in a `finally`).

`qwen_ag_matmul(x_local, w, *, config, compute_kernel_config, dtype, copy_local=False) -> (out, gathered)`:
`x_local` [1, 1, S/32, 5120] bf8 (this row's slot), `w` [1, 1, 5120, N] (N = 10 * N_block). `gathered`
[1, 1, S/4, 5120] gets every other row's slot (and the own slot with `copy_local`).

| Cores | Kernels |
| --- | --- |
| (0..5, 0..9) | the stock minimal_matmul program for the transposed grid (M over x, N over y), reduced to one N block per core, with an explicit M-block list per column (common runtime args only; a core finds its neighbours from its logical coordinates). With W_qkv's K x N_block (680 KiB) resident in L1, in1 is relayed only for each column's first block, then re-pushed from L1, so the columns run independently. |
| (6, 0..3) | fabric senders of links 0..3: BRISC fwd (half A 4 hops, half B 3), NCRISC bwd (3 / 4). Each link carries 40 of the 160 K-tile columns; rows go in waves of 8; each wave-half ends with an atomic inc of that (slot, wave)'s arrival word on the receivers' hub. Senders start once every peer has launched the call (barrier). |
| (6, 4) | hub: barrier, then releases each column's blocks in list order (own slot at once, remote blocks once their arrival word reaches 8 = 4 links x 2 halves). |

Bits: a tile's K order depends only on K_block and the N-block parity, never on the M split, so the
output equals the stock pair's; the gather copies bytes. The barrier and arrival words live in a
zero-filled L1 buffer on the hub core, created at the first call of a prefill (with a device sync, so no
peer's barrier inc can be wiped by a late zero-fill) and freed at the end of the prefill
(`release_semaphores`): L1 kept across the prefill -> decode switch moves the decode global CB, and the
decode trace capture then misses its program cache.

Checks (`tests/test_qwen_ag_matmul_op.py`): `matmul_port` (the 6x10 matmul with the block lists, no
fabric) and the fused op are `torch.equal` to the stock pair on all 32 devices at S = 16k / 24k / 32k /
40k, twice on fresh zero-filled buffers, every gathered slot equal; the plan covers every row once.
Pipelined ms per pair, stock -> fused (job367): 16k 0.895 -> 1.026, 24k 1.123 -> 1.381, 32k 1.507 ->
1.431, 40k 1.783 -> 1.603 (hence S >= 32768). At 40k the all-gather alone takes 1.13-1.15 ms and the
unwaited concurrent bound is 1.48 ms. The FF1 shape (W1 1.74 MB per core, not resident) runs in lockstep
and is slower than its stock pair (2.96-3.51 vs 2.84-2.89 ms, job366), so the MLP uses the 7x10 form below.

### qwen_ag_matmul_v2 (FF1)

Same contract and bits, no sender column: the matmul runs on the full 7x10 grid and the eight fabric
streams (links 0-3 fwd at (0..3, 9), bwd at (4, 9), (5, 9), (6, 9), (6, 8)) run inside those in0 receiver
cores (`kernels/v2_in0.cpp`): every wait of the in0 relay loop calls a non-blocking sender pump that sends
at most one packet or one inc. Column x takes one piece of the own slot (the slot split over the 7
columns), then every wave of remote slot (r + 1 + x) % 8, so each wave is one lockstep step; the in0
injectors wait on per-wave words (each sender increments wave word j on all 7 injectors of every receiver:
56 incs = wave j of every remote slot). Barrier word on the sender cores, both in one L1 buffer created and
freed per prefill like v1's. Waves (`v2_waves`): 12,16,12 at 40k. FF1 40k pair 2.894 -> 2.678 ms, 32k
2.451 -> 2.268 (job384); QKV in this form 1.57-1.59 ms, same as v1 (AG-bound), so QKV keeps v1.
Making the pump non-blocking (job381: 2.98 -> 2.86 at 8-row waves) mattered: a blocking flush per packet
batch or per inc stalls the in0 relay of the whole column.
