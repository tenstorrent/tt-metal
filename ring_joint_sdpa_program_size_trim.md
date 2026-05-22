# Ring Joint SDPA — Program Size Overflow & Trim Plan

Working document for a follow-up session. Captures the failure, the breakdown of where program-size goes, and a ranked, ordered list of trim approaches with byte estimates and perf risk.

---

## 1. The problem

**Failing test:**
```
tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_determinism[mla_100k-q160-k256]
```

**Error:**
```
TT_FATAL @ tt_metal/impl/program/program.cpp:2336: state.offset <= max_size
Program size (70848) too large for kernel config buffer (70656) on TENSIX
```

Overflow: **192 bytes** (0.27 % over the 70,656 B Tensix kernel-config ringbuffer on Blackhole).

**Reproduce:**
```bash
source python_env/bin/activate
scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_determinism -k "mla_100k-q160-k256"
```

**Perf constraint (hard):** any trim must not regress the `mla_100k-q160-k320` perf-check target of **62.9 % FPU utilization** on a 4-device QuietBox ring (see `RING_JOINT_PERF_CHECK_CONFIGS` at `test_ring_joint_sdpa.py:1175`).

---

## 2. Background — how program size is computed

`tt_metal/impl/program/program.cpp:2266 finalize_program_offsets` accumulates `state.offset` per programmable core type:

```
state.offset = rt_args + sems + CBs + DFBs + kernel_bins   (each aligned to L1_ALIGNMENT = 16 B on BH)
```

The TT_FATAL fires at `program.cpp:2331` when `state.offset` exceeds the ringbuffer max (70,656 B on BH Tensix).

Each kernel binary's contribution is `tt_memory::get_packed_size()` (≈ `.text + .data`) aligned to 16 B (`tt_metal/impl/program/dispatch.cpp:391-392`).

---

## 3. Where the bytes go — current breakdown

For the failing config (build cache root will change per build; locate via `find ~/.cache/tt-metal-cache -name "ring_joint_sdpa" -type d -newer <ref>` or just `ls -t ~/.cache/tt-metal-cache | head`):

### Per-RISC binary contribution (≈ 96.5 % of program size)

| RISC | Kernel | .text | .data | packed (16-aligned) |
|------|--------|------:|------:|--------------------:|
| brisc (writer)  | ring_joint_writer  |  8,160 |   0 |  **8,160** |
| ncrisc (reader) | ring_joint_reader  |  5,352 |   0 |  **5,360** |
| trisc0 (unpack) | ring_joint_sdpa    | 19,220 | 536 | **19,760** |
| trisc1 (math)   | ring_joint_sdpa    | 12,280 | 536 | **12,816** |
| trisc2 (pack)   | ring_joint_sdpa    | 22,108 | 128 | **22,240** |
| | | | | **68,336** |

### Non-binary contributions (≈ 2,500 B)
- CBs: factory uses indices `c_0..c_12, c_16, c_17, c_24..c_31` → `max_local_end_index = 32` → `32 × 4 × 4 = 512 B` (no remote CBs)
- RTAs + CRTAs + Semaphores + DFBs + alignment padding ≈ 2,000 B (6 sems × 16 B for MLA + fabric + alignment)

### Top compute-kernel symbols (demangled, by .text size)

**trisc2 (pack)** — 22,108 B total
| Bytes | Symbol |
|------:|---|
| **14,056** | `sdpa_inner_loop_step<…ring_mode=true, is_causal=true…>` [clone .constprop.0] [clone .isra.0] |
|  4,096 | `_start` |
|  1,792 | `normalize_row_streaming<false, 4, 8, false>` |
|  1,512 | `salad_correct_fused<2, 4, 8>` |

**trisc0 (unpack)** — 19,220 B total
| Bytes | Symbol |
|------:|---|
|  8,024 | `sdpa_inner_loop_step<…>` (same instantiation, smaller body on unpack) |
|  4,984 | `_start` |
|  1,712 | `sdpa_inner_loop_step::{lambda}::operator()` [clone .isra.0] |
|  1,704 | `normalize_row_streaming<…>` |
| ~1,500 | `llk_unpack_reconfig_data_format_src{a,b}` clones |

**trisc1 (math)** — 12,280 B total
| Bytes | Symbol |
|------:|---|
|  8,340 | `kernel_main()` (fully inlined `sdpa_ring_v2` + `sdpa_inner_loop_step`) |
|  2,288 | `normalize_row_streaming<…>` |
|    884 | `salad_correct_fused<…>` |
|    500 | `sub_exp_block_bcast_cols<false, scale_fp32>` |
|    268 | `_start` |

**ncrisc (reader)** — 5,352 B total
| Bytes | Symbol |
|------:|---|
|  4,096 | `kernel_main()` |
|    964 | `fetch_block<PaddedAddrGenerator<TensorAccessor<…>>>` |
|    292 | `_start` |

**brisc (writer)** — 8,160 B total
| Bytes | Symbol |
|------:|---|
|  4,164 | `kernel_main()` |
|  1,624 | `issue_restore_reads<TensorAccessor, TensorAccessor>` [clone .constprop.0] |
|  1,272 | `save_accumulators_with_trid<TensorAccessor, TensorAccessor>` [clone .constprop.0] |
|    808 | `write_block_row_grouped_trid<…>` [clone .constprop.0] |
|    292 | `_start` |

### Headline

A single template instantiation — `sdpa_inner_loop_step` defined at `compute_streaming.hpp:710` — accounts for ~22 KB across the three trisc binaries (14 KB on pack alone). Combined with `_start` (~9 KB across triscs, CB-init bulk), `normalize_row_streaming` (~5.8 KB), and `salad_correct_fused` (~2.4 KB), the compute kernel is the dominant story; reader + writer together are 13.5 KB.

---

## 4. Failing-config parameters (verified)

For `mla_100k-q160-k256` on 4-device QuietBox ring:

| Parameter | Value | Source |
|---|---|---|
| `is_causal` | `True` | test config |
| `is_balanced` | `True` | test config |
| `use_zigzag_balancing` | `True` | derived |
| `joint_seq_len` (L) | **0** | test (`test_ring_joint_sdpa.py:476`) |
| `num_joint_q_chunks` | **0** | derived from L=0 |
| `num_joint_k_chunks` | **0** | derived from L=0 |
| `Lt` | **0** | derived |
| `NHK` | 1 (MLA mode) | uses batch chain |
| `Sq_chunk_t` | 5 | q160 → 5 tiles |
| `Sk_chunk_t` | 8 | k256 → 8 tiles |
| `DHt` | 18 | MLA q/k head dim |
| `vDHt` | 4 | MLA v head dim |
| `qkt_subblock_h` × `qkt_subblock_w` | 1 × 8 | |
| `qktv_subblock_h` × `qktv_subblock_w` | 1 × 4 | |
| `q_per_core` | **4 or 5** | 20 q-chunks × 29 heads / ~130 cores |
| `q_per_core == 1` path | **NOT live** | rules out dead-code-stripping `issue_restore_reads`/`save_accumulators_with_trid` |
| Joint code paths | **dead at runtime** for this config | `if constexpr` opportunity |

---

## 5. Key files

### Compute kernel (touches all 3 triscs)
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/ring_joint_sdpa.cpp` — entry; CB index map at lines 92–116
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp` — `sdpa_ring_v2` (line 1480), `sdpa_inner_loop_step` (line 710), `normalize_row_streaming` (line 502), `salad_correct_fused` (line 421), `SDPA_NOINLINE` macro (lines 19–26 — empty on BH)
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp`

### Dataflow kernels
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_reader.cpp` — ncrisc
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/ring_joint_writer.cpp` — brisc
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/dataflow_common.hpp` — `fetch_block`, `issue_restore_reads`, `save_accumulators_with_trid`, `write_block_row_grouped_trid`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chain_link.hpp`

### Host
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp` — CBs at lines 647–768, semaphores at lines 478–499

### Investigation tools
- `/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-size -A <elf>` — section sizes
- `/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-nm --size-sort --print-size --reverse-sort --demangle <elf>` — top symbols
- `/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-objdump -drC <elf>` — disassembly

Build artifacts location (hash changes per build):
```
~/.cache/tt-metal-cache/<build-hash>/kernels/ring_joint_sdpa/<kernel-hash>/{trisc0,trisc1,trisc2}/*.elf
~/.cache/tt-metal-cache/<build-hash>/kernels/ring_joint_reader/<kernel-hash>/ncrisc/*.elf
~/.cache/tt-metal-cache/<build-hash>/kernels/ring_joint_writer/<kernel-hash>/brisc/*.elf
```

---

## 6. How to analyze program size

End-to-end workflow for reproducing the breakdown and verifying a trim. Run from the repo root.

### Step 1 — Reproduce the overflow and capture the error
```bash
source python_env/bin/activate
scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_determinism -k "mla_100k-q160-k256" 2>&1 | grep -A2 "Program size"
```
Expect: `Program size (70848) too large for kernel config buffer (70656)`. The two numbers tell you (a) current state.offset, (b) the ringbuffer limit. After a trim, the first number should drop.

### Step 2 — Locate the build cache for the failed run
The build cache root is hashed by device + build config. Find the most recent one that has `ring_joint_sdpa`:
```bash
find ~/.cache/tt-metal-cache -name "ring_joint_sdpa" -type d -printf '%T+ %p\n' 2>/dev/null | sort -r | head -1
```
Then descend into its single subdir to get the kernel build:
```bash
KERNELS_ROOT=$(find ~/.cache/tt-metal-cache -name "ring_joint_sdpa" -type d -printf '%T+ %p\n' 2>/dev/null | sort -r | head -1 | awk '{print $2}' | xargs -I{} dirname {})
COMPUTE_DIR=$(ls -d $KERNELS_ROOT/ring_joint_sdpa/*/)
READER_DIR=$(ls -d $KERNELS_ROOT/ring_joint_reader/*/)
WRITER_DIR=$(ls -d $KERNELS_ROOT/ring_joint_writer/*/)
```

### Step 3 — Per-RISC section sizes (kernel-binary contribution)
```bash
SIZE=/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-size
for elf in $COMPUTE_DIR/trisc0/trisc0.elf $COMPUTE_DIR/trisc1/trisc1.elf $COMPUTE_DIR/trisc2/trisc2.elf \
           $READER_DIR/ncrisc/ncrisc.elf $WRITER_DIR/brisc/brisc.elf; do
  echo "=== $elf ==="
  $SIZE $elf
done
```
What goes into program size: `packed_size = .text + .data` per binary, aligned up to 16 B (L1_ALIGNMENT on BH). `.bss` does not count — it's reserved separately in L1.

Compute the kernel_bins total:
```
trisc0_packed + trisc1_packed + trisc2_packed + ncrisc_packed + brisc_packed
```
This should be ~96 % of the failing 70,848 B.

### Step 4 — Top symbols per RISC (finds the offenders)
```bash
NM=/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-nm
for elf in $COMPUTE_DIR/trisc0/trisc0.elf $COMPUTE_DIR/trisc1/trisc1.elf $COMPUTE_DIR/trisc2/trisc2.elf \
           $READER_DIR/ncrisc/ncrisc.elf $WRITER_DIR/brisc/brisc.elf; do
  echo "=== $(basename $(dirname $elf)) ==="
  $NM --size-sort --print-size --reverse-sort --demangle $elf 2>/dev/null | head -15
done
```
Look for: single symbols >1 KB, multiple clones of the same template (`.constprop.0` / `.isra.0` / `.part.0`), and `_start` size (CB-init bulk on triscs).

### Step 5 — Disassemble a hot function to confirm a trim hypothesis
```bash
OBJDUMP=/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-objdump
$OBJDUMP -drC $COMPUTE_DIR/trisc2/trisc2.elf | less   # find sdpa_inner_loop_step, look for repeated patterns
```
For redundant reconfig audits (#3, #4): grep the disasm for `jal.*llk_unpack_reconfig` or `jal.*llk_pack_reconfig` and count call sites. Compare against the source `compute_streaming.hpp` lines to find the redundant ones.

### Step 6 — Account for non-binary contributions
After step 3 you have kernel_bins. The rest of state.offset is RTAs + sems + CBs + DFBs + alignment. CB region size is deterministic:
```
CB region = max_local_end_index × UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG × 4
          = max_local_end_index × 4 × 4
          = max_local_end_index × 16
```
For the failing test: `max_local_end_index = 32` (kernel uses up to `c_31`) → 512 B. Compacting CB indices to ≤22 (change #1) drops this to 352 B → 160 B saved deterministically.

The remaining ~2 KB (= 70,848 − kernel_bins − 512) is sems + RTAs + alignment padding; not easily decomposed without instrumenting `tt_metal/impl/program/dispatch.cpp:354` `finalize_kernel_bins`, but it's stable across small kernel changes.

### Step 7 — Verify a trim
After editing kernel source:
```bash
# Force rebuild — kernels are JIT, so just delete the cached binary for the affected kernel
rm -rf ~/.cache/tt-metal-cache/*/kernels/ring_joint_sdpa
rm -rf ~/.cache/tt-metal-cache/*/kernels/ring_joint_reader   # if dataflow changed
rm -rf ~/.cache/tt-metal-cache/*/kernels/ring_joint_writer   # if dataflow changed

# Re-run the test
scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_determinism -k "mla_100k-q160-k256" 2>&1 | grep "Program size"
```
If it passes silently (no "Program size" line), the overflow is gone. Re-run Step 3 to confirm the per-RISC sizes dropped by what you predicted.

### Optional — full ELF section dump (`.text` vs `.data` vs debug)
```bash
$SIZE -A $COMPUTE_DIR/trisc2/trisc2.elf
```
The `.text` and `.data` rows are what count; `.debug_*` sections are not loaded onto device.

### Quick one-liner — total program size delta after a change
Save before:
```bash
BEFORE=$(for elf in $COMPUTE_DIR/{trisc0,trisc1,trisc2}/*.elf $READER_DIR/ncrisc/*.elf $WRITER_DIR/brisc/*.elf; do
  [[ $elf == *.xip.elf ]] && continue
  $SIZE $elf | awk 'NR==2 { print $1+$2 }'
done | paste -sd+ | bc)
echo "before = $BEFORE"
```
Re-run after the rebuild; the diff is your per-binary kernel_bins delta (modulo 16-byte alignment per binary).

---

## 7. Trim approaches — ranked

Estimates are program-size deltas (post 16-byte alignment per binary).

### Tier 1 — High confidence, zero perf risk

| # | Change | Low | Mid | High |
|---|---|---:|---:|---:|
| 1 | CB index compaction | 460 B | 960 B | 1,660 B |
| 2 | `if constexpr` joint-path gating | 500 B | 700 B | 850 B |
| 3 | Remove redundant `reconfig_data_format` | 250 B | 400 B | 560 B |
| 4 | Delete dead `pack_reconfig_data_format` at line 798 | 240 B | 320 B | 360 B |
| | **Total Tier 1** | **~1,450 B** | **~2,380 B** | **~3,400 B** |

### Tier 2 — Safe cold-path outlining

| # | Change | Save | Notes |
|---|---|---:|---|
| 5 | Outline brisc `prefetch_for`/`flush_deferred_save` lambdas | 400–700 B | only on brisc; lambdas fully expand at 3 jal sites each |
| 6 | `noinline` the `normalize_row` lambda (compute_streaming.hpp:1020) | 300–500 B per trisc | last-K-of-last-ring-iter only — cold |

### Tier 3 — Needs perf measurement

| # | Change | Save | Notes |
|---|---|---:|---|
| 7 | Selectively restore `SDPA_NOINLINE` on BH for `blocked_matmul_and_pack` | 500–1300 B across triscs | adds jal/ret per subblock call (kt_num × q_num = 5/chunk here); medium risk |

---

## 8. Per-change detail

### #1 — CB index compaction
**What:** kernel uses CB indices `{0..12, 16, 17, 24..31}` (23 CBs total, max index 31). Renumber the high indices `24..31` into `13..20` so `max_local_end_index ≤ 22`.

**Why it saves bytes:**
- CB region: `(32-22) × 4 × 4 = 160 B` (deterministic — see `finalize_cbs` in `tt_metal/impl/program/dispatch.cpp:298`)
- `_start` on trisc0 (~4.9 KB) and trisc2 (~4.1 KB): per-CB unpack/pack-interface init iterates set bits in `local_cb_mask`. Fewer set bits → shorter inlined init sequence. Speculated 200–600 B per affected trisc; needs build to confirm.

**Touch:**
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/ring_joint_sdpa.cpp:109-116`
- `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp:695-731` (CircularBufferConfig calls for c_24–c_31)
- Reader/writer kernels (they reference the same CB indices — check `dataflow_common.hpp` and the kernel files)

**Perf risk:** none — pure compile-time rename.
**Complexity:** trivial-to-moderate (coordinated rename across kernel + factory + reader + writer).

---

### #2 — `if constexpr` joint-path gating
**What:** for `mla_100k`, `L = 0`, `num_joint_q_chunks = 0`, `num_joint_k_chunks = 0` — all already compile-time args. But the comparisons against them are runtime (`L != 0`, `is_joint_q ? ... : ...`, `kv_chunk_is_joint`), so GCC keeps the dead joint code. Convert to `if constexpr`.

**Touch (reader, `ring_joint_reader.cpp`):**
- Lines 234, 244, 268–315 (joint Q slice setup; joint generator ctors)
- Lines 386–441 (joint vs local KV chunk fetch)
- Lines 472–477

**Touch (writer, `ring_joint_writer.cpp`):**
- Line 268 (`get_q_chunk_info`)
- Line 370 (`joint_out_generator` ctor)
- Line 410 (`do_joint_kv`)
- Lines 501, 537, 637 — three `qi.is_joint_q ? joint_out_generator : out_generator` ternaries
- Line 289 (`get_end_seq_tile`)

**Wins:** eliminates 3 `PaddedAddrGenerator` materializations + several ternaries on each RISC. Reader ~250–400 B, writer ~250–450 B.

**Perf risk:** none — path is statically dead for the failing config; change adds zero instructions to the hot loop.
**Complexity:** moderate — must keep `num_joint_k_chunks > 0` branch compilable for other configs (e.g. WAN).

---

### #3 — Remove redundant `reconfig_data_format` calls
**What:** the 4-arg form of `reconfig_data_format(new_srca, new_srcb, old_srca, old_srcb)` skips the reconfig if `old == current`. Several call sites in `compute_streaming.hpp` pass stale "old" hints (e.g. `cb_qkt_im, cb_qkt_im` when the actual state is `cb_kt_in, cb_q_in`), so the compare always fails → unconditional reconfig is emitted every q_sub × kt_sub iteration.

**Sites to delete or repair:**
- `compute_streaming.hpp:807-809` — redundant (line 772 already set the correct state)
- `compute_streaming.hpp:800-802` — line 785 already reconfigs before sub_exp; sub_exp's internal init may re-establish
- `compute_streaming.hpp:970-972, 996-998, 1090-1092, 1115-1117` — Phase-2 V matmul; all four use 4-arg with stale hints

**Perf bonus:** removes runtime THCON stalls in the matmul prologue — pure win.
**Perf risk:** low — needs to verify that `sub_exp_block_bcast_cols`'s internal `sub_bcast_cols_init_short_custom` actually re-establishes srca/srcb formats (check `sdpa_sub_custom.h`).
**Complexity:** trivial-to-moderate.

---

### #4 — Delete `pack_reconfig_data_format(cb_qkt_im)` at line 798
**What:** between the pack-reconfig at `compute_streaming.hpp:769` (already configured pack → `cb_qkt_im`) and the matmul at line 812, the only state change is `sub_exp`'s `configure_single_tile_pack(reduce_cb)`. For this factory, `cb_qkt_im` uses `im_df` and `reduce_cb` (`cur.sum`) uses `stats_df` — and both are `Float16_b` (see `ring_joint_sdpa_program_factory.cpp:711-731`). The pack-format never actually changes; line 798 emits a no-op reconfig per q_subblock (5 inlined copies).

**Touch:** delete `compute_streaming.hpp:797-799` (the `if constexpr (!uniform_pack_format) { pack_reconfig_data_format(cb_qkt_im); }` block).

**Perf bonus:** removes per-q_subblock pack THCON stall — runtime win.
**Perf risk:** none — packer state already correct.
**Complexity:** trivial.

---

### #5 — Outline brisc lambdas
**What:** `prefetch_for` (`ring_joint_writer.cpp:490-516`) and `flush_deferred_save` (`ring_joint_writer.cpp:534-556`) lambdas have 3 jal sites each into `issue_restore_reads` / `save_accumulators_with_trid`, but the lambda bodies themselves fully expand at every call site. Convert to `static __attribute__((noinline))` free functions taking a small context struct.

**Perf risk:** low — called on paths that already do multi-hundred-cycle DRAM stalls; one extra jal/ret ≈ 6 cycles.
**Complexity:** moderate — thread a ctx struct instead of `[&]` captures.

---

### #6 — `noinline` the `normalize_row` lambda
**What:** `compute_streaming.hpp:1020-1027` lambda has 6 call sites in math `kernel_main`. Each site inlines `cb_push_back(cur.sum, sbh)` + `cb_push_back(out_cb, sbh * vDHt)` plus arg setup. Move pushes inside `normalize_row_streaming` (already `noinline, noclone`) and consolidate the dispatch into a single helper.

**Perf risk:** low — runs only on `is_last_k_of_last_ring_iter` (cold). Per-call overhead amortized over `Sq_chunk_t × (vDHt + 1)` tile ops inside normalize.

---

### #7 — `SDPA_NOINLINE` selective restore (Tier 3)
**What:** the `SDPA_NOINLINE` macro at `compute_streaming.hpp:19-26` is **empty on Blackhole**:
```cpp
#ifdef ARCH_BLACKHOLE
#define SDPA_NOINLINE
#else
#define SDPA_NOINLINE __attribute__((noinline, noclone))
#endif
```
Several functions marked `SDPA_NOINLINE` (`blocked_matmul_and_pack:168`, `sub_exp_block_bcast_cols:279`, `sub_exp_first_col_blocks:375`, `apply_lightweight_mask_streaming:614`) fully inline on BH.

Selectively annotate `blocked_matmul_and_pack` with hardcoded `__attribute__((noinline, noclone))` — it has 2 template instantiations called from 3 sites (lines 812, 980, 1099).

**Perf risk:** **medium** — adds 1 jal per subblock call. For this config: `kt_num_full_subblocks × q_num_subblocks = 1 × 5 = 5` calls per K-chunk Q@KT. The inner per-tile `matmul_block_no_mop` stays inlined, so FPU dispatch density is preserved. Single-digit cycle overhead per call — likely within FPU-utilization noise but **must be measured against the 62.9 % target before merge.**

**Complexity:** trivial (one attribute change), high test burden (full perf check + accuracy + determinism).

---

## 9. Recommended landing order (for new session)

The four Tier-1 changes are independent and each safe individually. Recommended sequence — smallest blast radius first, biggest leverage last:

1. **Land #4 first** (delete pack_reconfig at line 798) — single-line change, zero risk, ~240–360 B. Confirms the build pipeline and gives a quick win.
2. **Land #2 next** (`if constexpr` joint-path gating) — mechanical, well-scoped to reader/writer, ~500–850 B. Independent from compute-kernel changes. **After this PR, re-run the failing test** — likely already passes.
3. **Land #3** (redundant `reconfig_data_format` cleanup) — code-size + runtime perf bonus, but touches the matmul prologue so re-run perf check. ~250–560 B.
4. **Land #1 last** (CB index compaction) — biggest leverage but touches the most files (kernel + factory + reader + writer). Save for after #2 unblocks the test. ~460–1,660 B.

If headroom remains insufficient after Tier 1: pursue #5 (brisc lambda outlining) before #6 or #7 — brisc has clear leverage and the lambdas have low call frequency.

**Always run before merging any change:**
```bash
# 1. Determinism test (the originally failing one)
scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_determinism -k "mla_100k-q160-k256"

# 2. Accuracy test
scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_sdpa_accuracy -k "mla_100k-q160-k256"

# 3. Perf check (62.9 % FPU util target)
SDPA_PERF_CHECKS=1 scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::test_ring_joint_attention_perf_check -k "mla_100k"
```

For #7 specifically: also run the wan2_2 perf check (`-k "wan2_2"`) since it shares the compute kernel.

---

## 10. What NOT to touch

Consistent recommendations from all five investigations:
- **`sub_exp_block_bcast_cols` / `sub_exp_first_col_blocks`** — SFPU-heavy, per-K-chunk × q_subblock hot path
- **`matmul_block_no_mop`** and `blocked_matmul_and_pack`'s inner per-tile call — FPU-critical
- **`salad_correct_fused`** clones — already outlined; little remaining gain
- **TensorAccessor template depth** on reader/writer — hot address-calc
- **The `scale_fp32` template param** on `sub_exp_*` — single-instantiation here; removing would cost SFPU cycles
- **Compute kernel splits** (separate causal vs non-causal kernels, etc.) — invasive, defers the underlying problem

---

## 11. PR scoping (per user preference)

This work bundles trims that the size overflow forces. Do **not** sweep pre-existing dead code, shadow variables, comments, or unrelated cleanups into the same PR. Each Tier-1 change is independently shippable; keep PRs focused.

---

## 12. Results — applied trims

Working log of the steps actually applied to clear the
`mla_100k-q160-k256` determinism overflow. Each step is independently
shippable. Target ringbuffer is **70,656 B** on Blackhole Tensix.

### State table

| Step | Commit | Program size | Δ vs prev | Δ vs baseline | Headroom |
|---|---|---:|---:|---:|---:|
| Baseline (overflow) | — | **70,848 B** | — | — | **−192 B** (fails) |
| #1 — CB index compaction | `15a9e5c9a13` | **70,704 B** | −144 B | −144 B | −48 B (still fails) |
| #2 — `if constexpr` joint-path gating | `1fd39d3a75c` | **≈ 70,240 B** | ≈ −464 B | ≈ −608 B | ≈ +416 B ✓ |
| #3 — Relocate Q@KT srca/srcb reconfig | `fbfbf962da3`, `85d2edc11be` | **≈ 70,076 B** | ≈ −164 B | ≈ −772 B | ≈ +580 B ✓ |
| #4 — Delete dead `pack_reconfig_data_format` | — (not landed) | — | — | — | — |
| #5 — `noinline` brisc lambdas | `b002a3ca0a2` | **≈ 69,820 B** | ≈ −256 B | ≈ −1,028 B | ≈ +836 B ✓ |
| #6 — `noinline` `normalize_row` lambda | — (rejected) | ≈ 70,040 B | ≈ +220 B | — | ≈ +616 B (worse) |
| #7 — `noinline` `blocked_matmul_and_pack` | — (rejected) | **71,136 B** | +1,316 B | — | **−480 B (overflow)** |
| #8 — `salad_correct_row` sbh split | — (rejected) | ≈ 70,020 B | ≈ +200 B | — | ≈ +636 B (worse) |

Headroom = `70,656 − program_size`. Negative = overflow.

The #2 figure rounds because the JIT re-rolled the compute kernel under
a fresh cache root when the dataflow sources changed (compute source
was untouched but its trisc binaries shifted: trisc0 +448, trisc1 −176,
trisc2 −736 → compute net −464 B). Dataflow contribution from #2
alone: ncrisc +80, brisc −80 → net 0 B. The −608 B total below
baseline is the combined effect (CB region + JIT re-roll + dataflow
rewrite).

### Step #1 — CB index compaction (`15a9e5c9a13`)

**What changed**
- Renumbered intermediate CBs `c_24..c_31` → `c_13, c_14, c_15, c_18, c_19, c_20, c_21, c_22` in:
  - `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/ring_joint_sdpa.cpp`
  - `ttnn/cpp/ttnn/operations/transformer/sdpa/device/ring_joint_sdpa_program_factory.cpp`
- `c_16`/`c_17` skipped (already used by output CBs).
- Pure compile-time rename — same data formats, same sizes, same roles.

**Why it saves bytes**

`finalize_program_offsets` reserves the CB config region as a dense
array sized by `max_local_end_index = (highest CB index used) + 1`:

```
CB region bytes = max_local_end_index × UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG × 4
                = max_local_end_index × 16
```

Before: highest index used was `c_31` → `max_local_end_index = 32` → 512 B reserved.
After: highest index is `c_22` → `max_local_end_index = 23` → 368 B reserved.
Δ = (32 − 23) × 16 = **−144 B**.

**Speculation that did NOT pay off**
The doc estimated 200–600 B per affected trisc from a shorter `_start`
CB-init loop. ELFs were **byte-identical** to baseline after this
change — the LLK `_start` init is bounded by `NUM_CIRCULAR_BUFFERS`
(32), not by set-bit count.

**Tests after #1**

| Test | Result |
|---|---|
| determinism `mla_100k-q160-k256` | **STILL FAILS** — overflow by 48 B |
| accuracy `mla_100k-q160-k256` | n/a (blocked by determinism) |
| perf-check `mla_100k-q160-k320` | n/a (blocked by determinism) |

### Step #2 — `if constexpr` joint-path gating (`1fd39d3a75c`)

**What changed**
- Added two compile-time flags in `ring_joint_reader.cpp` and `ring_joint_writer.cpp`:
  ```cpp
  constexpr bool has_joint_q = num_joint_q_chunks > 0;
  constexpr bool has_joint_k = num_joint_k_chunks > 0;
  ```
- Wrapped every joint-vs-local branch in `if constexpr (has_joint_*)`.
  Joint code is now statically dead in `mla_100k` (both flags `false`)
  and dropped by the compiler. WAN-style configs (both flags `true`)
  still compile the joint paths unchanged.
- Templated writer helpers `get_q_chunk_info` and `get_end_seq_tile` on
  `has_joint_q` so the gating propagates into the helpers.
- For each `qi.is_joint_q ? joint_out_generator : out_generator`
  ternary (3 in writer + Q/K/V picks in reader), pick the generator
  **once** into a `const auto&` via a small IIFE that `if constexpr`
  -gates the joint return, then issue a single function call:
  ```cpp
  const auto& gen = [&]() -> const auto& {
      if constexpr (has_joint_q) {
          if (qi.is_joint_q) return joint_out_generator;
      }
      return out_generator;
  }();
  issue_restore_reads(gen, /*…all the args…*/);
  ```
- For slice setup (`q_slice`, `k_slice`/`v_slice`), default to the local
  path and override for joint inside `if constexpr` — no duplicated
  calls.

**Why it saves bytes**

`if constexpr` differs from regular `if`:
- **regular `if`** — both branches are compiled, even when the
  condition is a compile-time constant. The optimizer may DCE the
  unreachable branch, but doesn't always.
- **`if constexpr`** — the false branch is **discarded by the
  compiler** before code is even generated.

For `mla_100k` with both flags `false`, all joint-side code (slice
math, generator selectors, the joint-Q/K read paths) is dropped. The
`joint_*_generator` declarations remain at outer scope but their
constructors become unused → standard DCE removes them.

**Tests after #2**

| Test | Result |
|---|---|
| determinism `mla_100k-q160-k256` | **PASS** (was failing on overflow) |
| accuracy `mla_100k-q160-k256` | **PASS** |
| perf-check `mla_100k-q160-k320` (62.9 % FPU util, band [62.59, 63.21]) | **PASS** — see below |

**Perf-check stability — 4 runs**

| Run | Duration | math_util |
|---|---|---|
| 1 | 4.824 ms | 62.70 % |
| 2 | 4.815 ms | 62.81 % |
| 3 | 4.832 ms | 62.59 % (at floor) |
| 4 | 4.826 ms | 62.67 % |
| **mean** | **4.824 ms** | **62.69 %** |

All four runs pass the band check, but run-to-run variance occasionally
grazes the lower bound. We are consistently ~0.2 pp below the band
midpoint (62.90 %); a small additional regression here could push a
single run below the floor. Worth keeping in mind if any future change
touches the dataflow kernels.

### Step #3 — Relocate Q@KT pre-matmul srca/srcb reconfig (`fbfbf962da3`, `85d2edc11be`)

**What changed**

In `sdpa_inner_loop_step`'s `q_subblock > 0` path inside
`compute_streaming.hpp`:

- Deleted the unconditional 2-arg `reconfig_data_format(cb_kt_in, cb_q_in)`
  that ran after `sub_exp_block_bcast_cols` and before
  `mm_no_mop_reinit_short` (commit `fbfbf962da3`).
- Moved the 4-arg `reconfig_data_format(cb_qkt_im, cb_kt_in, cb_qkt_im,
  cb_q_in)` out of the per-`kt_subblock` matmul prologue and into the
  `q_subblock > 0` guard, placed immediately before
  `mm_no_mop_reinit_short`. The old per-`kt_subblock` copy was dropped
  (commit `85d2edc11be`).

Net effect from `main`: the matmul-side srca/srcb reconfig now runs
**once per kt_subblock only when `q_subblock > 0`**, and is positioned to
satisfy the matmul-init unpacker assert.

**Why it saves bytes**

Two effects compound:
1. The redundant 2-arg + 4-arg pair (both touching srca after sub_exp)
   collapses to a single 4-arg call per kt_subblock in the
   `q_subblock > 0` path.
2. The `q_subblock == 0` path no longer runs the per-`kt_subblock`
   reconfig at all. The outer 2-arg `reconfig_data_format(cb_kt_in,
   cb_q_in)` at the top of `sdpa_inner_loop_step` already establishes
   the `cb_kt_in/cb_q_in` state needed by `blocked_matmul_and_pack`,
   so the per-iter reconfig at the matmul prologue was always wasted
   for that path.

**Correctness — matmul-init assert**

`mm_no_mop_reinit_short` calls `llk_unpack_AB_matmul_init`, which runs
`LLK_ASSERT_BLOCK(are_unpackers_AB_configured_correctly(...))` against
`unpack_src_format` / `unpack_dst_format` for `cb_kt_in` / `cb_q_in`.
The 4-arg reconfig that runs before `sub_exp_block_bcast_cols` leaves
srca/srcb at `cb_qkt_im` (Float16_b); without restoring
`cb_kt_in/cb_q_in` before the matmul reinit, the assert would trip
under `TT_METAL_LLK_ASSERTS=1`. Placing the 4-arg reconfig in front of
`mm_no_mop_reinit_short` restores the expected state in time.

**Trim breakdown**

- trisc0 `.text`: **19,220 B → 19,056 B (−164 B)**
- Program size: 70,240 → ≈ 70,076 B (headroom ≈ +580 B under the
  70,656 B cap)

**Tests after #3**

| Test | Result |
|---|---|
| determinism `mla_100k-q160-k256` | **PASS** |
| perf-check `mla_100k-q160-k320` over 5 back-to-back runs | **PASS** — math_util mean **62.94 %**, all in band [62.59, 63.21] |

Perf-check expected math util subsequently bumped 62.9 % → **63.0 %**
in commit `d63ec7fffe8` (band becomes [62.685, 63.315]; all five
measured runs still inside).

### Step #4 (not landed) — Delete `pack_reconfig_data_format(cb_qkt_im)` at line 798

Investigated per the trim plan in section 8; **empirically rejected — the
delete is a net regression on Blackhole**.

**Hypothesis (still valid):** for this factory `im_df == stats_df ==
Float16_b` (factory lines 624, 626), so `pack_reconfig_data_format(cb_qkt_im)`
after `sub_exp_block_bcast_cols`'s internal `configure_single_tile_pack(reduce_cb)`
is a runtime no-op. Deleting it should drop ~5 inlined call sites on trisc2.

**Measured outcome (delete + rebuild + run determinism):**

| ELF section        | Baseline (step #3) | After delete | Δ        |
|---|---:|---:|---:|
| trisc0 `.text`     | 19,056 B           | 19,056 B     | 0 B      |
| trisc1 `.text`     | 12,280 B           | 12,280 B     | 0 B      |
| trisc2 `.text`     | 22,108 B           | **22,404 B** | **+296 B** |
| `sdpa_inner_loop_step` symbol (trisc2) | 14,056 B | **14,464 B** | **+408 B** |

Net program size grew. Determinism still passed (the elided call really is a
no-op semantically — `mla_100k-q160-k256` produced bit-identical outputs across
10 runs), but the size goal is missed.

**Why it grows (best understanding):** with the explicit
`pack_reconfig_data_format(cb_qkt_im)` present at line 798, the compiler can
treat the pack-format state as known-good before `blocked_matmul_and_pack`'s
internal `configure_row_pack_width` runs. Removing it forces
`configure_row_pack_width` (→ `llk_pack_init<…>` with
`skip_addrmod_config=true, skip_packer_strides=true`) to emit a more general
init sequence — net +408 B on the inner-loop symbol. The previous call was
free at runtime but cheap at compile time; deleting it is the opposite trade.

**Verdict:** leave line 797–799 in place. Re-examining via a *different*
refactor (e.g. teach `blocked_matmul_and_pack` that the pack format is
already cb_qkt_im, or move the pack-format invariant out of the per-iter
configure) is still open — see "What's next" below.

### Step #5 — `noinline` brisc lambdas

**What changed**

Added `__attribute__((noinline))` to the two large `[&]`-capture lambdas
inside the `use_deferred_norm` block of `ring_joint_writer.cpp`
`kernel_main()`:

- `prefetch_for(pf_q_index, pf_trid, barrier_first)` — issues NOC reads to
  fill staging for the next Q chunk. 3 call sites
  (intra-ring via `prefetch_intra_ring`, plus a direct cross-ring call).
- `flush_deferred_save()` — drains the pending raw-accumulator save to
  DRAM. 2 call sites (early flush before prefetch, late flush in the
  K-loop window).

The lambdas are kept `[&]` (no ctx-struct refactor). GCC honours
`__attribute__((noinline))` on lambda call operators — the closure
captures stay reference-based, and the body becomes a single function
per kernel build instead of being duplicated at each call site.

**Why it saves bytes**

Before: each call site inlined the full body (≈ 500–700 B each), so 3 +
2 = 5 copies on brisc.
After: one copy of each lambda body + 5 jal/ret call sites.

**Measured outcome (3 runs each, rebuild between)**

| ELF section / metric | Pre-#5 | Post-#5 | Δ |
|---|---:|---:|---:|
| brisc `.text` | 7,880 B | **7,624 B** | **−256 B** |
| `kernel_main()` (brisc) | (inlined lambdas) | 3,520 B | — |
| outlined `prefetch_for` (brisc) | n/a | 1,680 B | — |
| outlined `flush_deferred_save` (brisc) | n/a | 1,324 B | — |

`.text` drop is smaller than the doc's mid estimate (400–700 B) but
solidly real. After 16-byte alignment the program-size delta is also
−256 B (both 7,880 and 7,624 round up by 8 B each).

**Perf check — 3 runs each, mla_100k-q160-k320, band [62.69, 63.31], target 63.00 %**

| Run | WITH #5 | WITHOUT #5 |
|---|---|---|
| 1 | 4.812 ms / 62.85 % | 4.809 ms / 62.90 % |
| 2 | 4.805 ms / 62.95 % | 4.807 ms / 62.92 % |
| 3 | 4.807 ms / 62.92 % | 4.806 ms / 62.94 % |
| **mean** | **4.808 ms / 62.91 %** | **4.807 ms / 62.92 %** |

Δ = +0.001 ms / −0.01 pp util. Perf-neutral within run-to-run noise; all
six runs comfortably inside the band. The doc's "lambdas sit on
multi-hundred-cycle DRAM stall paths, so one extra jal/ret is free"
prediction matches the measurement.

**Tests after #5**

| Test | Result |
|---|---|
| determinism `mla_100k-q160-k256` | **PASS** |
| accuracy `mla_100k-q160-k256` | **PASS** |
| perf-check `mla_100k-q160-k320` (3 runs) | **PASS** — mean 62.91 % (in band) |

### Step #6 (not landed) — `noinline` the `normalize_row` lambda

Investigated per the trim plan in section 8/§6; **empirically rejected — the
outline is a net regression across the three triscs.**

**What was tried**

Added `__attribute__((noinline))` to the `normalize_row` lambda at
`compute_streaming.hpp:1017`. The lambda body is small (2x `cb_push_back`
+ call to `normalize_row_streaming` + counter increment) but has 6 call
sites, all guarded by `is_last_iter` (last K of last ring iter — cold by
construction). `normalize_row_streaming` itself is already
`static __attribute__((noinline, noclone))`, so the lambda only wraps
arg setup + the call.

**Measured outcome (rebuild + run determinism)**

| ELF section        | Baseline (post #1+#2+#3) | After noinline | Δ        |
|---|---:|---:|---:|
| trisc0 `.text`     | 19,056 B          | 19,016 B  | **−40 B**  |
| trisc1 `.text`     | 12,280 B          | 12,376 B  | **+96 B**  |
| trisc2 `.text`     | 22,108 B          | 22,272 B  | **+164 B** |
| **net Δ (.text)**  |                   |           | **+220 B** |

Net program size grew by ~224 B after 16-byte packing. Determinism
passes — the outline is semantically transparent — but the size goal is
missed.

**Why it grows (best understanding):** the inlined lambda body let the
compiler propagate `sbh` (always `qktv_h` or `drain_h`, both compile-time
constants at call sites) into `cb_push_back`'s argument and into
`normalize_row_streaming`'s call-site setup, allowing constant-folding
and tighter codegen at each site. Outlining widens `sbh` to a runtime
parameter, so the function body has to accept arbitrary values — net
+380 B for the outlined `operator()` (visible in the symbol table) and
+56 B in `sdpa_inner_loop_step`'s residual call setup, against ~−100 B
total in the now-elided inline copies. Same compile-time-vs-runtime
tradeoff that killed #4.

**Verdict:** leave the lambda inline. A targeted refactor that
preserves `sbh` as a compile-time constant per call site (e.g. two
explicit `normalize_row_h` and `normalize_row_remainder_h` outlined
functions templated on `sbh`) might still recover the save, but is a
larger change than #6 contemplated. Defer.

### Files touched

| File | Step | Change |
|---|---|---|
| `kernels/compute/ring_joint_sdpa.cpp` | #1 | CB index constants renumbered |
| `ring_joint_sdpa_program_factory.cpp` | #1 | `CircularBufferConfig` calls renumbered |
| `kernels/dataflow/ring_joint_reader.cpp` | #2 | `has_joint_q/k` flags, `if constexpr` gating at slice setup + Q/K/V read sites |
| `kernels/dataflow/ring_joint_writer.cpp` | #2 | `has_joint_q/k` flags, templated `get_q_chunk_info`/`get_end_seq_tile`, `if constexpr` gating at all `joint_out_generator` selectors (deferred-norm + eager paths) |
| `kernels/compute/compute_streaming.hpp` | #3 | 2-arg reconfig removed; 4-arg reconfig moved from per-`kt_subblock` matmul prologue into `q_subblock > 0` guard, before `mm_no_mop_reinit_short` |
| `tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py` | follow-up | perf-check `mla_100k` expected math util 62.9 % → 63.0 % |
| `kernels/dataflow/ring_joint_writer.cpp` | #5 | `__attribute__((noinline))` on `prefetch_for` and `flush_deferred_save` lambdas |

### Step #7 (not landed) — Tier-3 `noinline` on `blocked_matmul_and_pack`

Investigated per the trim plan in section 8/§7; **empirically rejected —
the noinline is a large net regression on Blackhole.**

**What was tried**

Replaced `SDPA_NOINLINE` (no-op on BH) with hardcoded
`__attribute__((noinline, noclone))` on the `blocked_matmul_and_pack`
template at `compute_streaming.hpp:168`. The function has two
instantiations live in this config:
`<true,KT_stride,KT_stride>` (Q@KT, 1 site) and
`<false,vDHt,vDHt>` (QKT@V, 2 sites).

**Measured outcome (rebuild + run determinism)**

| Metric              | Baseline (post #5) | After noinline | Δ          |
|---|---:|---:|---:|
| program size        | ≈ 69,820 B         | **71,136 B**   | **+1,316 B** |
| determinism test    | PASS               | **FAIL (overflow)** | — |

The overflow message reappears at the failing test — net **+1,316 B
regression**, more than wiping out the cumulative headroom from
#1+#2+#3+#5.

**Why it grows (best understanding):** same compile-time-vs-runtime
trade as #4/#6. When `blocked_matmul_and_pack` was inlined, GCC
specialized its body for each constant-folded call site (e.g. the
`q_subblock == 0` branch's `skip_pack_configure` choice, the trigger
semantics, the subblock width sequence) — the per-site copies were
each tighter than the union. Outlining forces a single body that has
to accept all combinations, and the union widens fast.

**Verdict:** leave `SDPA_NOINLINE` as-is on Blackhole (no-op macro,
allow inlining). A more invasive restructure — e.g. *splitting*
`blocked_matmul_and_pack` into two free functions, one per `transpose`
template arg, each with a single instantiation per kernel — might be
the path forward, but is well beyond the size-trim scope.

### Step #8 (not landed) — `salad_correct_row` lambda split by `sbh`

Investigated independently in this session; **empirically rejected —
the split alone regresses, and the noinline-augmented variant nets
~0 bytes saved.**

**What was tried**

The `salad_correct_row` lambda at `compute_streaming.hpp:1029` takes
`sbh` as a runtime `uint32_t` even though every call site passes a
compile-time constant (`qktv_h` or `qktv_remainder_h`). Two variants
were measured:

1. **Lambda split only** — converted to a generic lambda taking
   `auto sbh_c` (a `std::integral_constant<uint32_t, N>` per site),
   replacing the runtime branch on `sbh == qktv_remainder_h` with
   per-instantiation `if constexpr`. Updated all four call sites.

2. **Lambda split + force outline** — same as (1) plus
   `static __attribute__((noinline, noclone))` on
   `salad_correct_fused` itself (it currently has plain inlining
   on BH; this forces it to remain outlined per instantiation across
   all triscs).

**Measured outcomes (each rebuilt from clean cache):**

| Variant | trisc0 .text | trisc1 .text | trisc2 .text | Total Δ vs baseline |
|---|---:|---:|---:|---:|
| Baseline (post #5)         | 19,056 | 12,280 | 22,108 | 0 |
| (1) lambda split only      | **18,068 (−988)** | 12,280 (0) | **23,296 (+1,188)** | **+200 B** |
| (2) split + noinline       | 18,372 (−684) | 12,420 (+140) | 22,624 (+516) | −28 B |
| (2′) noinline only         | 18,380 (−676) | 12,420 (+140) | 22,624 (+516) | −20 B |

**Why (1) regresses:** the original lambda had a runtime branch
between `salad_correct_fused<qktv_remainder_h,…>` and
`salad_correct_fused<qktv_h,…>`. With both reachable, GCC kept the
`<qktv_h,…>` clone outlined on trisc1/trisc2 (it had 2 callsites).
Splitting the lambda makes each specialization see only ONE
`salad_correct_fused` instantiation per call site — GCC then loses
the "shared by two sites" inlining heuristic on trisc2 and inlines the
`<qktv_h,…>` body straight into `sdpa_inner_loop_step`. Result: net
+1,188 B on trisc2, swamping the -988 B trisc0 win.

**Why (2)/(2′) only break even:** explicitly outlining
`salad_correct_fused` lets the inner loop shrink, but adds a second
outlined symbol (`<qktv_remainder_h,4,8>`) on every trisc — the new
symbol is ~700–1,000 B and roughly offsets the inner-loop savings.

**Verdict:** leave `salad_correct_row` and `salad_correct_fused`
unchanged. The pattern matches #4/#6/#7: any GCC outlining nudge that
either creates a new outlined instantiation OR drops a heuristic-driven
existing one pays its own ~1 KB tax. The current lambda body
representation is already a local optimum on Blackhole.

### What's next (if more headroom needed)

After #1+#2+#3+#5 we have ≈ 836 B of headroom; determinism +
accuracy + perf-check all pass with margin. Every Tier-1/2/3 trim
that was *not* landed has now been empirically tested:

| Tier | # | Verdict |
|---|---|---|
| 1 | #4 | **rejected** — pack-reconfig delete is a +296 B net regression on trisc2 (Step #4) |
| 2 | #6 | **rejected** — plain `noinline` on `normalize_row` is +220 B net (Step #6) |
| 2 | #8 (new) | **rejected** — `salad_correct_row` lambda split is +200 B net regression on trisc2 |
|   | #8b | **marginal** — split + `noinline salad_correct_fused` is −28 B net (perf risk not worth it) |
| 3 | #7 | **rejected** — `noinline` on `blocked_matmul_and_pack` is +1,316 B net regression |

**Unifying pattern (#4, #6, #7, #8):** the streaming compute kernel
sits at a local optimum where GCC's inlining/outlining heuristics
have produced the smallest viable mix for this template+call-site
arrangement. Any blunt push in either direction (force outline a
heretofore-inlined function, force inline a heretofore-outlined one)
breaks a per-site constant-fold and the union of the per-site
specializations is larger than the inlined diversity. Future trim
opportunities will need to either:

1. Change the *call shape* so fewer specializations are needed
   (e.g. unify Phase-2 V matmul subblock dispatch so only one
   `blocked_matmul_and_pack<false,…>` instantiation is live), or
2. Move bytes out of the compute-kernel template altogether
   (e.g. a non-templated runtime dispatcher).

Both are larger architectural changes than the size-overflow
problem warranted; the four landed Tier-1/2 trims are sufficient.

Other angles surveyed in this session that returned no win:

- **Joint generator DCE on reader/writer** — already happening
  (joint_*_generator symbols absent from ELFs for `mla_100k` after
  step #2).
- **`is_causal` / `is_balanced` runtime branches in dataflow** — these
  *are* runtime branches over `constexpr` flags, but GCC's constant
  folding already collapses them; converting to `if constexpr`
  produces byte-identical ELFs.
- **`_start` size on trisc (4–5 KB)** — dominated by
  `setup_local_cb_read_write_interfaces` being called twice on BH
  (lower 32 + upper 32 CB bits, NUM_CIRCULAR_BUFFERS=64). Shrinking
  this is a tt_metal/firmware change, out of scope for the SDPA
  kernel.
- **brisc kernel_main width (3,520 B)** — already factored after
  step #5; the two outlined lambdas + `write_block_row_grouped_trid`
  account for the rest. No further per-call-site pattern collapses.
