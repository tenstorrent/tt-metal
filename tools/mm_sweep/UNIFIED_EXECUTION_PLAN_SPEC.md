# Unified (Ns, Pk, Sm) Regime-A Matmul — Execution Plan Spec

Blueprint for a clean rewrite of the `--unified` code path in the Regime-A DRAM-BW-optimal
matmul prototype. **Read-only extraction — no behavior is proposed here, only documented.**

Computes `out[M,N] = in0[M,K] @ in1[K,N]` (M < N), bf16 in/out, HiFi2 math, fp32 dest
accumulation. `in0` is DRAM-interleaved, `in1` is DRAM width-sharded across 8 banks, `out` is
DRAM-interleaved.

## Files in the unified path
- Harness: `tests/tt_metal/tt_metal/perf_microbenchmark/regime_a_mm/test_regime_a_mm.cpp`
- in1 reader == consumer (BRISC or NCRISC depending on NoC group):
  `tests/tt_metal/tt_metal/perf_microbenchmark/regime_a_mm/kernels/reader_ring.cpp`
- in0 ring all-gather + split-K reduction + output write:
  `tests/tt_metal/tt_metal/perf_microbenchmark/regime_a_mm/kernels/in0_ring_writer.cpp`
- Compute (reused from minimal_matmul):
  `ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels/compute.cpp`
  (CreateKernel points here explicitly, harness line 773-775 — NOT a local copy.)

## How `--unified` is activated
`test_regime_a_mm.cpp:112-113` parses `--unified`. When set (`:127-132`):
- `ring = true`
- if `ksplit == 0` then `ksplit = 1` (so `Pk >= 1`)

Then `:133-136`: because `ksplit > 0`, `sharded = true` and `preaders = ksplit * mfac`, where
`mfac = (msplit>1?msplit:1) * (nslice>1?nslice:1)` (`:126`). Downstream, `sharded && KP`
(`KP = (ksplit || nsring)`, `:173`) selects the split-NOC K-split branch that builds the
`reader_ring` / `in0_ring_writer` kernels (`:550-692`).

Config knobs and their canonical variables:
| CLI | var | meaning |
|-----|-----|---------|
| `--nslice` | `Ns` (`nslice`) | N-slices per bank-band |
| `--ksplit` | `Pk` (`ksplit`) | K-slices (split-K depth), forced to ≥1 |
| `--msplit` | `Sm` (`msplit`) | M-split factor |
| `--kb` | `kb` | K-block depth (tiles) fed to compute |
| `--nsb` | `nsb` | N-subblock width (tiles); 0 ⇒ full `N_own` |

`Pk = ksplit?ksplit:1`, `Ns = nslice?nslice:1`, `Sm = msplit?msplit:1` (`:179`).
Total cores: `num_cores = 8 * preaders = 8 * Pk * Ns * Sm` (`:172`).

Note `deepk_u = unified && (in0direct||mshard||moverlap)` (`:182`) — **all three are legacy
flags being deleted**, so for the clean unified path `deepk_u == false` and the code takes the
ring branch everywhere. This spec documents the `deepk_u == false` path only.

---

## 1. Padding (three-factor)

Logical tile counts (`:171`): `Mt = M/32`, `Kt = K/32`, `Nt = N/32`.
Helpers (`:169-170`): `cdiv(x,y) = (x+y-1)/y`; `rup(x,y) = cdiv(x,y)*y` (round up to multiple).

Unified padding block (`:183-208`, non-deepk branch `:195-208`):

```
Kt_local = rup(cdiv(Kt, Pk), kb*8)      // per-core k-slice, padded to a multiple of kb*8
Kt_s     = Pk * Kt_local                 // padded K (in0/in1 buffer row stride, tiles)
M_block  = cdiv(Mt, Sm)                  // per-core M rows (tiles)
Mt_s     = Sm * M_block                  // padded M (buffer row count)
N_band   = cdiv(Nt, 8u)                  // per-bank N width, Nt padded to 8 banks
N_own    = cdiv(N_band, Ns)              // per-core N (before nsb sub-blocking)
N_sub    = nsb ? nsb : N_own             // compute/cb N-block width
N_bpc    = cdiv(N_own, N_sub)            // N-sub-blocks per core
N_own_s  = N_bpc * N_sub                 // per-core N padded to N_sub multiple
N_band_s = Ns * N_own_s                  // per-bank N padded (buffer col stride within a bank)
Nt_s     = 8u * N_band_s                 // padded N (out/in1 col stride, tiles)
N_block  = N_sub
```

Derived (`:235-236`):
```
K_num_blocks_eff = Kt_local / kb           // in0/in1 blocks per k-slice (multiple of 8, since Kt_local | kb*8)
K_num_blocks     = K_num_blocks_eff        // for unified (KP||unified true)
```

Ring shard width `Wsh = K_num_blocks_eff / ring_G` (`:570`), `ring_G = 8` for unified
(`global_ring == false`, `:174`). Requires `K_num_blocks_eff % 8 == 0` (`:574-576`) — always
satisfied by the `kb*8` rounding of `Kt_local`.

**Rounding rationale (comments `:175-178`, `:196`):** pad `Kt` so each k-slice is a multiple of
`kb*8` (ring shards × kb), pad `Mt` to an `Sm` multiple, pad `Nt` to `8*Ns*nsb`. Padding is the
identity when already divisible.

**Buffer sizes** (`:372-374`): `in0 = Mt_s * Kt_s`, `in1 = Kt_s * Nt_s`, `out = Mt_s * Nt_s`
tiles, all DRAM-interleaved, page = `tb` (bf16 tile bytes).

**Real vs pad representation** (`:376-397`, verify `:1227-1252`):
- `in1` filled entirely with `1.0` bf16 (`0x3F803F80`), including pad region.
- `in0`: real tiles `[m<Mt, k<Kt]` set to `1.0`, **all pad tiles = 0** (`fill_in0`, `:383-394`).
  Tile layout is row-major with padded row stride: `tile(m,k)` at word offset
  `(m*Kt_s + k) * words_per_tile`.
- Because in0 pad-K tiles are 0, products over the padded K range vanish, so every real output
  element `out[m<Mt, n<Nt] == K` regardless of layout/padding.
- Correctness check restricted to real region `[m<Mt, n<Nt]` (`:1233-1251`), reading with the
  **padded** column stride `Nt_s` (`off = (m*Nt_s + n)*words_per_tile`). Expected value `K`;
  pass iff no element exceeds 1% relative error. Padded output tiles are never written or checked.

---

## 2. Worker placement (split-NOC bank-adjacent)

Guarded by `if (sharded)` (`:267`). Uses the optimal DRAM-bank→logical-worker assignments
(`:270-272`):
```
opt0 = device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_0)   // 8 cores
opt1 = device->get_optimal_dram_bank_to_logical_worker_assignment(NOC::NOC_1)   // 8 cores
```
`grid2 = device->compute_with_storage_grid_size()` (BH = 11×10).

`find_near(t)` (`:275-293`): spiral search outward (diamond radius `d`, `dx∈[-d,d]`,
`dy=d-|dx|`, both `±dy`) from logical core `t` for the nearest **unused** core inside the grid;
marks it used. This is how grid gaps / already-taken cores are skipped — there is no explicit
11×10 gap handling in the unified path; `find_near` just walks to the next free logical core.

Placement loop (`:330-355`), for `b in 0..8`, `p in 0..preaders`:
```
noc = nosplit ? 0
    : (msplit>1 ? ((p/msplit)&1u)          // Sm>1: alternate NoC per k/n-slice group
                : (p & 1u));                // Sm==1: alternate NoC per slice index p
c   = find_near(noc ? opt1[b] : opt0[b]);   // basic unified: rectplace & in1mcast both false
cores.push_back(c);
core_bank.push_back(b);                       // bank id
core_suboff.push_back(p * N_block);           // (only used by legacy non-KP path)
core_noc.push_back(noc);
```
So **core index `i = b*preaders + p`**, i.e. bank-major, slice-minor.

Per-core assignment decode (`:877-903`), with `slice = i % preaders`, `bank = core_bank[i]`:
```
kk  = slice / mfac        // k-slice index (0..Pk-1)          mfac = Ns*Sm
sub = slice % mfac
mm  = sub % Sm            // M-block index (0..Sm-1)          — m is innermost
nn  = sub / Sm            // N-slice index (0..Ns-1)
```
This matches the header comment (`:123-125`): `g = k*(Ns*Sm) + n*Sm + m`, m innermost so M-slaves
are adjacent and the reduction over k has stride `Ns*Sm`.

Offsets (`:887-906`, non-nsring):
```
k_start = kk * Kt_local                 // first K tile of this k-slice (tiles)
m0v     = mm * M_block                  // M-block row offset (tiles)
nn_off  = nn * N_own_s                  // N-sub-band col offset within the bank (tiles)
bank_n0 = bank * N_band_s + nn_off      // this core's absolute N offset (tiles)
base_off= k_start * N_band_s * tb       // byte offset of the k-slice start within the bank
```

NoC group → kernel handles (`:904-905`): `core_noc[i]` selects `readerB/writerB` (NoC1 group)
vs `readerA/writerA` (NoC0 group). Kernel creation (`:689-692`):
```
readerA = reader_ring on g0, RISCV_0 / NOC_0
readerB = reader_ring on g1, RISCV_1 / NOC_1
writerA = in0_ring_writer on g0, RISCV_1 / NOC_1   // in0-read+reduce+write on OTHER NoC
writerB = in0_ring_writer on g1, RISCV_0 / NOC_0
```
So on each core, the in1 reader and the in0/out mover run on opposite NoCs (split-NOC).

---

## 3. Ring membership + ordering

Ring construction (`:826-876`, `KP && (fwd||ring)` branch — global_ring is false for unified).
For each slice index `j in 0..preaders`, the ring is the **8 cores sharing slice `j` across the
8 banks**: core `order[pos]*preaders + j`. There are `preaders = Pk*Ns*Sm` independent 8-wide
rings, one per (kk, nn, mm) group.

Chain order (`--chain`, default `"bank"`):
- `"bank"` (`:850-853`): `order = [0,1,2,...,7]` (bank order).
- `"nn"` (`:831-849`): greedy nearest-neighbor Hamiltonian over the 8 cores' **physical**
  coords (`worker_core_from_logical_core`), minimizing Manhattan hops per step.

Per-core ring data (`:855-873`, ring branch):
```
ring_pos[ci] = pos                                      // 0..7, this core's position
ring_nx/ny[ci] = phys(order[(pos+1)%8]*preaders + j)    // cyclic NEXT (forward target)
ring_px/py[ci] = phys(order[(pos+7)%8]*preaders + j)    // cyclic PREV (reduction reverse-credit)
```

**in0 all-gather** (`in0_ring_writer.cpp:194-241`, ring else-branch): cb0 holds the full
k-slice as `G` slots of `W` blocks. Every core is an injector:
- step 0: read own shard (shard index = `ring_pos`, blocks `ring_pos*W + wb`) from DRAM into
  cb0 slot 0 (`:207-217`), addressing in0 tile `(m0+m)*Kt + (k_start + sb*K_block + k)` with
  `Kt = Kt_s`.
- step `s` (1..G-1): `noc_semaphore_wait_min(fwd_ptr, step)` (`:220`) — wait prev forwarded a
  shard into slot `step`.
- if `step+1 < G`: forward current slot to next core's slot `step+1`
  (`noc_async_write` + `noc_semaphore_inc` on `fwd_sem`, `:222-226`).
- `cb_push_back(in0_cb, W*in0_blk)` each step (`:238`) so compute consumes shard-by-shard.

Result (header `:5-9`): each core forwards `G-1` times cyclically (no head/tail). Slot `s` of
core `c` ends up holding shard `(c-s)`.

**in1 read** (`reader_ring.cpp`): runs on the SAME core's in1 RISC (the reader==consumer). For
unified, `force_strided == 1` (`:605`, `(nsring||nslice>1||deepk||unified)`), so it takes the
strided sub-band branch (`reader_ring.cpp:188-219`), NOT the contiguous branch. For each
N-sub-band `nb in 0..N_bpc`, and `step in 0..G`, it reads shard `s = (ring_pos + G - step) % G`
(rotated to match the ring in0 delivery, `:197`), then `wb in 0..W` blocks. Each block reads
`kb` k-rows of `N_sub` tiles each, at in1 offset
`(ktile*N_band + n_base + nb*N_block)*tile_bytes` (`:207-211`) with `ktile = k_start +
kblk*K_block + kr`, `N_band = N_band_s`, `N_block = N_sub`, `n_base = nn_off`, read via
`get_noc_addr_from_bank_id<true>(bank_id, in1_addr + off)`. The rotated shard order guarantees
in0[k] pairs with in1[k]; the matmul sum is commutative so any consistent pairing is correct
(header `:1-5`).

in0 is read by the **writer RISC** (the OTHER NoC / `in0_ring_writer`), while in1 is read on the
reader RISC; for unified `in0_on_reader == 0` and `in0_same == false` (`in0risc` default
`"other"`, `:597`), so the two run on opposite NoCs concurrently.

---

## 4. Split-K reduction

When `Pk > 1`, the `kk` slices of a fixed `(bank, nn, mm)` form a linear reduction chain. Chain
neighbors (`:898-914`):
```
is_bottom = (kk == 0) ? 1 : 0        // :895 (noreduce/nsring legacy forced-1 ignored)
is_top    = (kk == Pk-1) ? 1 : 0     // :896
red_next  = cores[i + mfac]          // next k-slice, same (b,nn,mm); used iff !is_top (:899-903)
red_prev  = cores[i - mfac]          // prev k-slice; used iff !is_bottom (:911-914; is_bottom uses i)
```
Reduction stride in the core index is `mfac = Ns*Sm` (`:900`, `:911`).

Compute side (`compute.cpp:492-504`, `REDUCE_K` defined): after producing its matmul partial in
`intermediate_cb`, the bottom band `copy_block(intermediate→out_cb)`; every other band
`cb_wait_front(cb_reduce)` then `reduce_add_block(intermediate, cb_reduce → out_cb)` and pops
cb_reduce. `cb_reduce = c_7` (`:377`).

Data-mover side (`in0_ring_writer.cpp:244-295`, non-noreduce loop over `nb in 0..N_bpc`):
- `reduce_base = get_write_ptr(cb_reduce)` captured ONCE at `:81` **before any cb_reduce use**
  (comment: the write ptr drifts after receives; the base-capture-once fix). Slot for block `nb`
  = `reduce_base + (nb%2)*out_blk_bytes` (double-buffered, 2 slots).
- Non-bottom core (`:271-276`): `cb_reserve_back(cb_reduce)` (wait compute freed slot nb-2),
  `noc_semaphore_inc(prev_redfree, 1)` (tell prev slot nb%2 free), `noc_semaphore_wait_min(
  red_ptr, nb+1)` (prev delivered block nb), `cb_push_back(cb_reduce)` (compute reduce-adds it).
- After compute produces reduced block nb into out_cb (`cb_wait_front(out_cb)`, `:277`):
  - **non-top** (`:279-284`): `noc_semaphore_wait_min(redfree_ptr, nb+1)` (next freed its slot),
    `noc_async_write` the block to `red_next`'s `reduce_base + (nb%2)*out_blk_bytes`, barrier,
    `noc_semaphore_inc(next_recv, 1)` (deliver block nb).
  - **top** (`:285-292`): writes the block to DRAM `out`, tile `(m0+m)*Nt + (n_off+n)` with
    `Nt = Nt_s`, `n_off = n0 + nb*N_block`.
- `cb_pop_front(out_cb)` (`:294`).

Semaphores (`:474-491`): `recv_sem` (reduction recv, `red_sem_id`), `fwd_sem` (in0 ring recv),
`redfree_sem` (reverse credit for cb_reduce slot reuse across N-sub-blocks). `redfree_sem` is
created whenever `ring` (`:490-491`).

The final (top) core is the only one that writes `out` to DRAM per output block; when `Pk==1`
every core is both bottom and top, so `is_bottom==1 && is_top==1` and it writes directly with no
reduction traffic.

---

## 5. Circular buffers

`tb = tile_size(Float16_b)` (2048 B), `tf = tile_size(Float32)` (4096 B) (`:167`).
Block tile counts (`:402`): `in0_blk = M_block*kb`, `in1_blk = kb*N_sub`, `out_blk =
M_block*N_sub`. (`N_block == N_sub` for unified.) `Ktl` below = `Kt_local`; note
`K_num_blocks_eff = Ktl/kb`.

For the unified path (not moverlap/bstream/bcast/nsring):

| CB | index | source line | tiles | depth | df | notes |
|----|-------|-------------|-------|-------|----|----|
| cb0 (in0 k-slice resident) | c_0 | `:456-462` (`KP && (fwd\|\|ring)`) | `K_num_blocks_eff * in0_blk = (Ktl/kb)*M_block*kb = M_block*Ktl` | 1× (holds full k-slice, `G*W` blocks) | bf16 | ring fills all G slots, kept resident (`IN0_KSLICE_RESIDENT`) |
| cb1 (in1) | c_1 | `:469-470` | `in1_depth * in1_blk` | `in1_depth = 4` (sharded, non-deepk) | bf16 | `in1_blk = kb*N_sub` |
| cb2 (out) | c_2 | `:471` | `2 * out_blk` | 2 | bf16 | `out_blk = M_block*N_sub` |
| cb3 (fp32 intermediate) | c_3 | `:472` | `out_blk` | 1 (single-buffered) | fp32 | accumulator; matches minimal_matmul factory |
| cb7 (reduce) | c_7 | `:482` (`KP`, non-nsring) | `2 * out_blk` | 2 | bf16 | split-K running sum; only used when `Pk>1` but always allocated |

As functions of `(Ktl, Mblk=M_block, Nsub=N_sub, kb)`:
```
cb0 = Mblk * Ktl                     tiles bf16   (== K_num_blocks_eff * Mblk * kb)
cb1 = 4 * kb * Nsub                  tiles bf16
cb2 = 2 * Mblk * Nsub                tiles bf16
cb3 = Mblk * Nsub                    tiles fp32
cb7 = 2 * Mblk * Nsub                tiles bf16
```
Total L1 bytes ≈ `(Mblk*Ktl + 4*kb*Nsub + 2*Mblk*Nsub + 2*Mblk*Nsub)*tb + (Mblk*Nsub)*tf`.

**L1 budget check:** there is **no explicit total-L1 TT_FATAL** in the unified path. The only
size-related guards are the 16 KB-burst check (skipped for unified/nsb/nsring, `:561-563`) and
the `Kt_local`/`kb` divisibility from the `rup` rounding. CB allocation failure would surface at
`CreateCircularBuffer`/program-compile time, not via an explicit check here. (See Open Questions.)

Circular buffers are created on `all_cores` (`:439-483` via `mkcb(all_cores, ...)`).

---

## 6. Kernel arguments

### 6a. reader_ring.cpp (in1 reader == consumer)

Compile-time args (`test_regime_a_mm.cpp:591-611`, `ring` branch; `reader_ring.cpp:10-35`):
| idx | value (unified) | kernel name | meaning |
|-----|-----------------|-------------|---------|
| 0 | `kb` | `K_block` | K-block depth (tiles) |
| 1 | `N_sub` | `N_block` | N-sub-band width fed per block |
| 2 | `Wsh` = `K_num_blocks_eff/8` | `W` | in1 blocks per shard |
| 3 | `ring_G` = 8 | `G` | ring size |
| 4 | `tb` | `tile_bytes` | bf16 tile bytes |
| 5 | `in0_same?1:0` = 0 | `read_in0` | 0 ⇒ reader does NOT read in0 |
| 6 | `M_block` | `M_block` | M rows (tiles) |
| 7 | `Kt_s` | `Kt` | padded K row stride |
| 8 | `in0ready_sem` | `in0ready_sem_id` | (unused when read_in0=0) |
| 9 | `in0ord` = 0 (`in0order` default "before") | `in0_order` | in0 read order (unused, read_in0=0) |
| 10 | `N_bpc` | `N_bpc` | N-sub-blocks per core |
| 11 | `N_band_s` | `N_band` | per-bank width; stride for strided sub-band reads |
| 12 | `nsbcontig?1:0` = 0 | `contig_nsb` | diagnostic; 0 = real strided layout |
| 13 | 1 (unified) | `force_strided` | forces strided sub-band branch |
| 14 | `skipin1?1:0` = 0 | `skip_in1` | ablation |
| 15 | `in1valid_sem` | `in1valid_sem` | M-split reader→slaves (only used if Sm>1) |
| 16 | `in1ready_sem` | `in1ready_sem` | M-split slaves→reader |
| 17 | `in1mcast?1:0` = 0 | `in1mcast` | forward via mcast vs unicast |
| 18 | `deepk?1:0` = 0 | `in0_direct` | 0 for unified ring path |
| 19+ | `TensorAccessorArgs(in0)` | `in0_args` | in0 accessor (for the in0 shard read, unused here) |

Runtime args (`:918-968`, `ra`; `reader_ring.cpp:37-48`):
| idx | value | kernel name | meaning |
|-----|-------|-------------|---------|
| 0 | `in1_buf->address()` | `in1_addr` | in1 base DRAM addr |
| 1 | `bank` | `bank_id` | this core's DRAM bank |
| 2 | `bank & 0x3` | `vc` | NoC VC (unused in strided branch) |
| 3 | `base_off` = `k_start*N_band_s*tb` | `slice_off` | k-slice byte offset (unused in strided branch) |
| 4 | `ring_pos[i]` | `ring_pos` | ring position (rotates shard read order) |
| 5 | `in0_buf->address()` | `in0_addr` | (used only if read_in0) |
| 6 | `m0v` = `mm*M_block` | `m0` | M-block row offset (tiles) |
| 7 | `k_start` = `kk*Kt_local` | `k_start` | first K tile of this slice |
| 8 | `nn_off` = `nn*N_own_s` | `n_base` | sub-band N-offset within bank (tiles) |
| 9 | `2` (solo) when Sm==1; `1`/`0` (reader/slave) when Sm>1 | `mrole` | M-split role |
| 10 | `0` (solo) / `Sm-1` (reader) / `1` (slave) | `mpeers` | forward peer count |
| 11.. | M-split peer coords (only when Sm>1) | — | reader: `(Sm-1)×(x,y)`; slave: reader `(x,y)` |

For the canonical `Sm==1` case, `ra` has exactly 11 entries: 9 base + `{2, 0}` solo
(`:964-967`). `mrole==2` ⇒ reader reads in1 from DRAM normally (no slave recv, no forward).

### 6b. in0_ring_writer.cpp (in0 all-gather + reduce + write)

Compile-time args (`:636-668`, `ring` branch; `in0_ring_writer.cpp:15-46`):
| idx | value (unified) | kernel name | meaning |
|-----|-----------------|-------------|---------|
| 0 | `M_block` | `M_block` | M rows |
| 1 | `kb` | `K_block` | K-block depth |
| 2 | `N_sub` | `N_block` | output/in1 N-sub-band width |
| 3 | `K_num_blocks_eff` | `K_num_blocks` | G*W (full k-slice) |
| 4 | `tb` | `tile_bytes` | |
| 5 | `Kt_s` | `Kt` | padded K row stride (in0 addressing) |
| 6 | `Nt_s` | `Nt` | padded N col stride (out addressing) |
| 7 | `Wsh` | `W` | blocks per shard |
| 8 | `8u` | `G` | ring size |
| 9 | `fwd_sem` | `fwd_sem_id` | in0 ring recv semaphore |
| 10 | `recv_sem` | `red_sem_id` | reduction recv semaphore |
| 11 | `in0_same?1:0` = 0 | `in0_on_reader` | 0 ⇒ writer reads own in0 shard |
| 12 | `in0ready_sem` | `in0ready_sem_id` | (unused, in0_on_reader=0) |
| 13 | `N_bpc` | `N_bpc` | reduction loops over these |
| 14 | `redfree_sem` | `redfree_sem_id` | cb_reduce reverse credit |
| 15 | `skipin0?1:0` = 0 | `skip_in0` | ablation |
| 16 | `in0direct?1:0` = 0 | `in0_direct` | 0 for ring path |
| 17 | `skipfwd?1:0` = 0 | `skip_fwd` | ablation |
| 18 | `noreduce?1:0` = 0 | `noreduce` | ablation |
| 19 | `mshard?1:0` = 0 | `mshard` | legacy |
| 20 | `in0share_act?1:0` = 0 | `in0share` | legacy (separate flag) |
| 21 | `share_valid_sem` | `share_valid_sem` | (unused unless in0share) |
| 22 | `share_ready_sem` | `share_ready_sem` | (unused unless in0share) |
| 23 | `in0scatter_act?1:0` = 0 | `in0scatter` | legacy |
| 24+ | `TensorAccessorArgs(in0)` then `TensorAccessorArgs(out)` | `in0_args`,`out_args` | accessors |

Runtime args (`:1000-1046`, `wa`; `in0_ring_writer.cpp:48-68`):
| idx | value | kernel name | meaning |
|-----|-------|-------------|---------|
| 0 | `in0_buf->address()` | `in0_addr` | |
| 1 | `out_buf->address()` | `out_addr` | |
| 2 | `m0v` = `mm*M_block` | `m0` | M-block row offset |
| 3 | `bank_n0` = `bank*N_band_s + nn_off` | `n0` | absolute output N offset (tiles) |
| 4 | `k_start` = `kk*Kt_local` | `k_start` | first K tile of this slice |
| 5 | `ring_pos[i]` | `ring_pos` | 0..7 |
| 6 | `ring_nx[i]` | `fwd_next_x` | ring next (forward target) physical x |
| 7 | `ring_ny[i]` | `fwd_next_y` | ring next y |
| 8 | `nx` | `red_next_x` | reduction next (`cores[i+mfac]`, 0 if top) |
| 9 | `ny` | `red_next_y` | |
| 10 | `is_bottom` | `is_bottom` | `kk==0` |
| 11 | `is_top` | `is_top` | `kk==Pk-1` |
| 12 | `px` | `red_prev_x` | reduction prev (`cores[i-mfac]`, self if bottom) |
| 13 | `py` | `red_prev_y` | |
| 14+ | (in0share/in0scatter peers — NOT appended for basic unified) | — | legacy |

### 6c. compute.cpp

Compile-time args (`:757-758`, with `ring_nsb == true` for unified; `compute.cpp:343-350`):
`cct = {K_num_blocks_eff, M_block, kb, N_sub, 1u, N_bpc, sbh, sbw_c}`:
| idx | value | kernel name | meaning |
|-----|-------|-------------|---------|
| 0 | `K_num_blocks_eff` (`ksplit` set) | `K_num_blocks` | k-blocks in the resident slice |
| 1 | `M_block` | `M_block_tiles` | |
| 2 | `kb` | `K_block_tiles` | |
| 3 | `N_sub` (`nblk_c`) | `N_block_tiles` | |
| 4 | `1u` | `M_blocks_per_core` | |
| 5 | `N_bpc` (`nbpc_c`) | `N_blocks_per_core` | |
| 6 | `sbh` = `largest_div(M_block, 2)` | `subblock_h` | |
| 7 | `sbw_c` = `largest_div(N_sub, 4/sbh)` | `subblock_w` | (`:754`) |

Defines (`:765-772`): `REDUCE_K=1` (since `ksplit>=1`), `IN0_KSLICE_RESIDENT=1` (since
`ring_nsb`). `ComputeConfig`: `math_fidelity = HiFi2` (unless `--lofi`), `fp32_dest_acc_en =
true`, `dst_full_sync_en = false`, `math_approx_mode = false` (`:777-783`).

Runtime args (`:1050`; `compute.cpp:352-358`): `{0u, M_block, 0u, N_bpc*N_sub, is_bottom}`:
| idx | value | kernel name | meaning |
|-----|-------|-------------|---------|
| 0 | `0` | `M_start_tile` | |
| 1 | `M_block` | `M_end_tile` | |
| 2 | `0` | `N_start_tile` | |
| 3 | `N_bpc * N_sub` | `N_end_tile` | **must span all N_bpc sub-blocks** (see §7) |
| 4 | `is_bottom` | `is_reduce_bottom` | 1 for `kk==0` |

---

## 7. Per-core valid/tail extents (CURRENT padding-based behavior)

The current code carries **no per-core short-tail logic**; instead everything is padded up to
uniform block multiples (§1) so every core runs identical, fully-sized blocks. This is the
baseline the rewrite replaces with balanced floor/ceil ownership.

- **Uneven K / last short kb block:** `Kt_local = rup(cdiv(Kt,Pk), kb*8)` guarantees each
  k-slice is an exact multiple of `kb*8`, so `K_num_blocks_eff = Kt_local/kb` blocks are all full
  `kb`-deep. The extra padded K tiles are 0 in in0 (§1), contributing nothing.
- **Uneven N ownership / last nsb block:** `N_own_s = N_bpc*N_sub` and `N_band_s = Ns*N_own_s`
  pad N so every core owns exactly `N_bpc` sub-blocks of exactly `N_sub` tiles. Padded N columns
  are computed and written to padded output regions but never checked.
- **Uneven M / M-split:** `M_block = cdiv(Mt,Sm)`, `Mt_s = Sm*M_block`; every M-slave owns a full
  `M_block` rows; the last one may include padded rows (0 in in0).
- **compute tail clamps are effectively no-ops:** `compute.cpp:412-422` computes
  `current_M_block_tiles = min(m_tile+M_block, M_end_tile) - m_tile` and likewise for N. Because
  `M_blocks_per_core=1` with `M_end_tile=M_block`, and `N_end_tile = N_bpc*N_sub` exactly spans
  all `N_bpc` blocks, these clamps never truncate — every block is full-sized. The comment at
  `:1048-1049` warns that passing `N_block` (rather than `N_bpc*N_sub`) as `N_end_tile` would
  clamp sub-blocks `1..N_bpc-1` to empty and deadlock (reader/writer/compute block-count
  mismatch). So `N_end_tile` MUST be the full padded span.
- **Reader/writer counts:** reader_ring loops `N_bpc * G * W` in1 blocks (`:193-198`); the in0
  ring pushes `G` shards of `W` blocks (`:238`); the reduction loops `N_bpc` output blocks
  (`:270`). All counts derive from the padded dims, so all three kernels agree by construction.

Net: correctness holds because in0 padding is 0 and only the real `[Mt,Nt]` region is verified;
performance carries the padding overhead (wasted compute/reads on pad tiles), which the balanced
floor/ceil rewrite is meant to remove.

---

## 8. compute.cpp flags in the unified path

Defines set: `REDUCE_K` and `IN0_KSLICE_RESIDENT` (`:766-771`). (`IN1_RESIDENT` and
`FUSE_BIAS`/`FUSE_TERNARY`/`SFPU_OP_*` are NOT set for unified.)

**`IN0_KSLICE_RESIDENT`** (`compute.cpp:399-404, 435-482, 530-532`):
- `cb_wait_front(in0_cb, K_num_blocks * in0_block_num_tiles)` ONCE up front (`:403`) — the full
  k-slice (all `K_num_blocks_eff` blocks, block-major) is resident.
- Inside the k-loop, `cb_wait_front(in0_cb,...)` and `cb_pop_front(in0_cb,...)` per-block are
  **compiled out** (`#ifndef IN0_KSLICE_RESIDENT`, `:436-438, 474-478`).
- `matmul_blocks` is called with `in0_base = k_block * in0_block_num_tiles` (`:453-456`) — the
  k-loop indexes block `k_block` via a tile offset into the resident cb0 instead of popping. This
  lets cb0 be reused across the `N_blocks_per_core` (`N_bpc`) N-sub-blocks without re-reading in0.
- `cb_pop_front(in0_cb, K_num_blocks * in0_block_num_tiles)` once at the very end (`:531`).

**`REDUCE_K`** (`compute.cpp:492-504`): after the k-accumulation produces the fp32 partial in
`intermediate_cb` (with `llk_pack_reconfig_l1_acc` toggled on for `k_block>0`, `:483-489`):
- if `is_reduce_bottom`: `copy_block(intermediate → out_cb)` (this band has no incoming sum).
- else: `cb_wait_front(cb_reduce)`, `reduce_add_block(intermediate, cb_reduce → out_cb)`,
  `cb_pop_front(cb_reduce)` — add the running sum forwarded up from the band below.
- K-par never fuses bias/ternary (the `REDUCE_K` branch is exclusive of `FUSE_*`).

**`is_reduce_bottom`** (runtime arg 4 = harness `is_bottom` = `kk==0`): selects copy vs
reduce-add above. There is no compile-time `is_reduce_bottom`; it is per-core runtime.

**`N_end_tile` spanning `N_bpc*N_sub`** (runtime arg 3): with `N_blocks_per_core = N_bpc` and
`N_block_tiles = N_sub`, the compute N-loop (`:418-422`) walks `N_bpc` sub-blocks at
`n_tile = n_block_iter*N_sub`, clamping `n_tile_end = min(n_tile+N_sub, N_end_tile)`. Passing the
full `N_bpc*N_sub` keeps every sub-block full (see §7).

`M_blocks_per_core = 1`: the outer M-loop runs once; the whole `M_block` is one compute block.

The matmul itself (`matmul_blocks`, `:283-340`) iterates subblocks `subblock_h × subblock_w`
within `M_block × N_sub`, inner K over `K_block_tiles = kb`, accumulating into
`intermediate_cb` (fp32, L1-acc). `full_N_block_tiles = N_block_tiles = N_sub` sets the in1 row
stride within a block.

---

## Open questions / ambiguities

1. **No explicit L1 budget check.** The unified path never sums CB bytes against an L1 cap;
   oversubscription would fail at CB allocation / program compile, not via a harness TT_FATAL.
   The clean op should add an explicit budget check.
2. **`base_off` / `slice_off` (reader arg 3) and `vc` (arg 2) are dead in the unified strided
   branch.** The strided sub-band read (`reader_ring.cpp:188-219`) computes the in1 offset from
   `k_start` + tile arithmetic and does not use `slice_off`; `vc` is only used by the contiguous
   `noc_async_read_one_packet_set_state` branch. They are still passed. Confirm whether the
   rewrite keeps them.
3. **`in0_addr`/`in0ready_sem`/`in0_order` in reader_ring are inert** for unified
   (`read_in0 == 0`); passed but unused.
4. **cb7 (cb_reduce) is always allocated even when `Pk==1`** (`:482`, guarded only by `KP`,
   which is true whenever `ksplit>=1`, i.e. always for unified). For `Pk==1` it is wasted L1
   (2*out_blk bf16). The rewrite may want to allocate it only when `Pk>1`.
5. **NoC alternation formula differs by Sm.** `noc = msplit>1 ? ((p/Sm)&1) : (p&1)` (`:333`). For
   `Sm==1` this alternates per slice index `p` (which mixes kk and nn); the intended balance
   (equal NoC0/NOC1 load per bank) depends on `preaders` parity. Worth confirming the rewrite's
   NoC assignment matches the measured ~500 GB/s split-NOC behavior.
6. **`--sbh` override** (`sbh_ovr`) feeds `largest_div(M_block, sbh_ovr)`; default 0 ⇒
   `largest_div(M_block, 2)`. The fp32 DST budget is 4 tiles; `sbw_c = largest_div(N_sub,
   4/sbh)`. The known fp32 subblock>4 DST-overflow bug (project memory) means the rewrite must
   keep `sbh*sbw_c <= 4`. Verify the `4/sbh` integer division always yields a safe product.
7. **`worker_core_from_logical_core` vs grid gaps.** `find_near` operates purely in logical
   coordinates; the 11×10 BH grid's non-power-of-2 `gy` and any physical worker gap are handled
   only implicitly (physical coords are taken later for ring/reduction NoC targets). No explicit
   gap avoidance exists in the unified placement — confirm this is acceptable on the target grid.
