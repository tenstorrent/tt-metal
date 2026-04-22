# W-Halo Vertical Line Artifact

## Goal
Fix the 3 visible vertical mosaic lines at W-device boundaries in 2x4 480p WAN T2V VAE decoder output.

## Symptom
After fixing H-boundary synchronization (progress semaphore) and removing the broken `NP_W_HALO_L1` drain, horizontal artifacts are gone but 3 vertical lines remain — one at each of the 3 W-device boundaries on a 2x4 mesh.

## Investigation Status

### Hypothesis 1: W-writer / conv3d-reader addressing mismatch — DISPROVED

Initially suspected the W-writer writes with stride `H_dev` while the conv3d reader reads with stride `h_total = H_dev + 2*ph`. **This is wrong.** Verified:

1. The factory sets `w_outer_dim_size = B * T * h_total` (line 248 of fused factory), so the W exchange handles `h_total` iterations per temporal slice — including corner rows.
2. The W-reader (`phase2_w_reader.cpp`) maps iteration `i` to `(t = i / h_total, h_padded = i % h_total)` and reads corner data from the H-halo section for `h_padded < ph` or `h_padded >= ph + H_dev`.
3. The W-writer writes sticks contiguously at `section_base + i`, matching `h_total` stride.
4. The conv3d reader computes `page = section_base + t * h_halo_H + (ph + h_in)` where `h_halo_H = h_total`. Since the writer's page for `(t, h_padded)` is `section_base + t * h_total + h_padded`, and the reader uses `h_padded = ph + h_in`, the pages match for interior rows.
5. Corner pages (h_padded=0..ph-1 and h_padded=ph+H_dev..h_total-1) are written with correct H-halo-sourced data and read at the correct offsets.

**The addressing is correct. Do NOT implement Option B — it would break things.**

### Hypothesis 2: Standalone NP path for WanConv2d — SUPERSEDED

Enabling `use_fused` for all WanConv2d/WanCausalConv3d did NOT eliminate the vertical lines. The seam is introduced by the fused `neighbor_pad_conv3d` path itself, not by the standalone path.

### Hypothesis 3: Synchronization race — PARTIALLY RULED OUT (see details)

**`w_neighbor_semaphore` race — DISPROVED (Session 2):**
`ttnn.create_global_semaphore` allocates a *distinct L1 copy* per core in the `CoreRangeSet`. Each W writer addresses the *specific neighbor core's* L1 copy via NOC atomic increment (not a broadcast). Therefore on a middle 2x4 device that receives from both left and right W neighbors, each W reader waits on *its own per-core L1 copy* — only ever incremented by its single designated sender. No shared semaphore interference between W-left and W-right directions.

**`input_progress_sem_addr` (conv3d reader gate) — CORRECT:**
W readers increment all conv3d reader core semaphores once per dispatch. With `num_w_fabric_cores = 2` on a middle device, each conv3d reader waits for count=2 (both W readers done). Fabric in-order delivery guarantees DRAM data is valid before the matching semaphore increment arrives. Race is NOT possible.

### Hypothesis 4: Non-conv operations — DISPROVED

Non-conv operations do not cause the seam. The all-ones latent decoder test with standalone NP + separate conv3d produces a clean output. Only the fused path shows the seam.

### Hypothesis 5: Blocking configuration (W_out_block) — DISPROVED

Initially suspected that multi-W-block processing (W_out_block < W_dev) caused the seam via incorrect halo page addressing at block boundaries. But `conv_in` (C_in=32, W_out_block varies) does NOT show seams even with its production multi-W-block config. The seam is not caused by W_out_block splitting.

### Hypothesis 6: C_in_block < C_in (channel block parallelism) — ACTIVE HYPOTHESIS (REFINED)

**Key finding:** The seam appears ONLY in layers where `C_in_block < C_in` (multiple channel blocks processed by different cores):

| Layer | C_in | C_in_block | #C_in blocks | Seam? |
|-------|------|------------|-------------|-------|
| `conv_in` | 32 | 32 | 1 | NO |
| `mid_block` resnets | 384 | 96 | 4 | YES |
| `up_blocks[0]` resnets | 384 | 96 | 4 | YES |

**Layer isolation tests (all-ones latent, 2x4 480p 81f):**
- `conv_in` only fused → clean (no seam in any blocking config)
- `conv_in` + `mid_block` fused → seam appears
- `conv_in` fused + `up_blocks[0]` fused (mid disabled) → seam appears
- `conv_in` multi-block by itself → still clean (re-confirmed)

The common factor in failing layers: 384-channel residual blocks with `C_in_block=96` (4-way C_in split).

**BUT: 2x2 mesh passes even with C_in_block < C_in (Session 2)**
`mid_res_2x2_Wdev26_ones` test: C_in=384, C_in_block=96, W_dev=26 (matching 2x4's geometry) → PASSES on 2x2. This means the bug is **NOT** just `C_in_block < C_in` in isolation — it requires the 2x4 mesh topology (3 W boundaries, middle devices with 2 active W fabric neighbors).

### Hypothesis 6 — deep audit of halo addressing for C_in_block < C_in

Performed full source audit of the halo read/write paths. For `pw=1` (padding_w=1, which is the VAE decoder case):

**Writer side (`minimal_default_writer.cpp`):**
- W-writer writes one stick per `(t, h_padded)` at `dst_stick_id = section_base + outer_dim_offset * h_halo_H + h_padded`
- Each stick is full-C width (`stick_size = C * 2 bytes`)
- Uses `dst_accessor` derived from the halo buffer itself for NOC address

**Reader side (`reader_vol2col.cpp` → `gather_rows_halo`):**
- For W-halo reads, computes `halo_page = w_section_base + t_global * h_halo_H + h_padded`
- Uses `halo_reader = TensorAccessor(in_args, h_halo_buffer_addr, in_row_size_bytes)` where `in_row_size_bytes = input_tensor.aligned_page_size()`
- Applies `c_in_offset_bytes = c_in_block * C_in_block * 2` as byte offset within the full-C stick

**Finding:** The page/offset formulas are **mathematically identical** between writer and reader for `pw=1`. No transposition, no index mismatch. The `c_in_offset_bytes` correctly slices into the full-C stick read from DRAM.

**What WAS verified correct:**
- `TensorAccessor` does not incorrectly trigger multi-page logic for the halo buffer
- `batch_idx` and `t_global` are consistent across all C_in blocks
- `h_halo_outer_dim_size` matches `T_dev` on both sides
- Ping-pong buffer caching gives each dispatch a distinct halo buffer
- DRAM allocation via `ttnn.from_torch` with `DRAM_MEMORY_CONFIG` produces identical buffer physical addresses across devices, so fabric writes land correctly

### Hypothesis 7: 2x4 topology-specific issue — CURRENT BEST GUESS

The bug manifests **only on 2x4 mesh, not on 2x2**, even with identical C_in_block < C_in configuration and matching W_dev geometry. Middle W devices (devices 1 and 2 on a 2x4) have two active W fabric neighbors — one left and one right. All synchronization analysis proves correct in isolation. The undiscovered bug likely lies in:

1. **How `input_progress_signal_count` is set on middle devices:** `num_w_fabric_cores` should be 2 for middle devices, meaning conv3d readers wait for 2 progress signals. Need to verify this is actually set to 2 in the factory for middle devices on a real 2x4 mesh.
2. **W-reader `outer_dim_size` vs actual sender count:** The W reader waits for `w_neighbor_sem_addr >= outer_dim_size`. If `outer_dim_size` is set inconsistently (e.g., `B*T*h_total` from one direction but the semaphore counts `B*T*h_total` from the other), the W reader might signal conv3d readers before the second direction's data arrives.
3. **Partial sum reduction with halo from two sources simultaneously:** On 2x2, each device has at most 1 active W neighbor. On 2x4, middle devices have 2. This additional W halo exchange may expose a partial-sum ordering issue in `compute.cpp` that only surfaces with both left and right halos active.

## Session 3 Findings: .pt Numerical Analysis (Apr 22 2026)

**Seam is EXCLUSIVELY at the D1-D2 boundary (both middle devices).**

Analysis of fused_multiblk.pt vs standalone_multiblk.pt (81×3×480×832):
- Global: max_diff=1.40, mean_diff=0.012, PCC=0.9913
- Background error (BF16 path difference, NOT seam): 0.004241 — uniform across ALL columns
- D0-D1 boundary (col 208): 0.004241 mean — ZERO seam above background
- **D1-D2 boundary (col 416): 0.251492 mean — 60× above background, 88-column bell curve**
- D2-D3 boundary (col 624): 0.004241 mean — ZERO seam above background

Right-edge / left-edge breakdown:
- D1 rightmost col 415: 0.246 (WRONG)  |  D1 leftmost col 208: 0.004 (correct)
- D2 leftmost col 416: 0.251 (WRONG)   |  D2 rightmost col 623: 0.004 (correct)
→ **D1's rightmost output and D2's leftmost output are mutually corrupted.**
→ D0's rightmost col 207 and D3's leftmost col 624 are clean.

Error shape: bell curve centered ~col 418-419, left_spread=47 cols into D1, right_spread=41 cols into D2. Spread is consistent with mid_block seam propagating through many subsequent standalone up_block conv layers.

Per-frame: seam grows across temporal chunks (T-cache amplifies error: t=0..16 small, t=17+ jumps 0.6+).

**Conclusion: The bug requires BOTH (a) middle W devices AND (b) C_in_block < C_in. Neither alone is sufficient (conv_in on 2x4 = clean; mid_block on 2x2 = clean).**

## ROOT CAUSE FOUND (Session 4, Apr 22 2026)

**Progress semaphore never reset on conv3d reader cores.**

The W-readers (in `phase2_w_reader.cpp`) signal the progress semaphore by doing
`noc_semaphore_inc(get_noc_addr(rx, ry, progress_sem_addr), 1)` where `(rx, ry)` are the
conv3d reader core NOC coordinates. This increments the semaphore **on the conv3d reader cores**.

`reset_global_semaphore_value(progress_sem, 0)` in `manager.py` only resets the semaphore on
`self.ccl_cores` (the NP fabric cores). The conv3d reader cores are in `conv3d_core_range`, a
completely different set of cores. They are **never reset**.

On the second dispatch and beyond, every conv3d reader core sees `progress_sem = 2` from the
previous call → it skips the `noc_semaphore_wait_min` → starts reading the halo buffer before
W-halo data has been written for this call.

For edge devices (D0, D3): their W-LEFT halo is self-padded via LOCAL NOC (fast, completes before
conv3d reader reaches those columns). Their W-RIGHT halo comes from the adjacent device but conv3d
iterates w_blocks left-to-right, so the rightmost w_block is processed last; by then the fabric
write from the neighbor has typically landed.

For middle devices (D1, D2): both W-LEFT and W-RIGHT halos depend on fabric writes from the OTHER
middle device. These fabric writes arrive AFTER the conv3d reader's startup. With no semaphore
gate (skipped on call 2+), D1's rightmost and D2's leftmost columns are processed before the W-halo
data is ready → wrong output at those columns. The error propagates and amplifies through temporal
caching (T-halo), explaining the 88-column bell curve and t=17+ amplification.

The C_in_block < C_in condition exacerbates (or reveals) the bug because with 4 C_in blocks
and 4 REDUCER/WORKER cores per output position, the conv3d reader starts processing
halo-dependent output positions sooner (more cores active simultaneously).

**FIX applied in reader_vol2col.cpp:**
```cpp
noc_semaphore_wait_min(progress_sem_ptr, input_progress_signal_count);
noc_semaphore_set(progress_sem_ptr, 0);  // Reset for next dispatch
```
After waiting, the reader resets its own L1 semaphore to 0. This is safe because:
- Both W-readers have already signaled (count reached input_progress_signal_count)
- No more signals will arrive until the NEXT program dispatch
- On the next dispatch, the count starts at 0 and the reader must wait again

## Next Test: Run on real 2x4 LoudBox via CI

**Status: DISPATCHED** (see dispatch command below)

Two critical experiments:
1. `multi_block and fused and 2x4_h0_w1` — confirm seam still present (baseline)
2. `single_block and fused and 2x4_h0_w1` — C_in_block=384 override; if seam disappears, C_in parallelism is confirmed

Test in `test_decoder_ones_input.py`:
- `single_block=True` → calls `_override_mid_block_full_cin()` which sets `C_in_block=384` for all mid_block resnet convolutions
- Compares output `.pt` against standalone reference; logs PCC + max_err + per-column seam jumps

**If seam disappears with single_block:** Bug confirmed in C_in parallelism. Then:
- Check `reader_vol2col.cpp` `gather_rows_halo`: does the `c_in_offset_bytes` correctly index into the halo sticks when the halo was written by D1 (a middle device with 2 active W writers)?
- Specifically: look at whether the W-LEFT halo sticks written by D1's FORWARD W WRITER use a different stride/offset than what conv3d's reader expects per C_in block
- Check `w_section_wleft_base` used by the W WRITER vs what `reader_vol2col` computes as `halo_page`

**If seam persists with single_block:** Bug is NOT in C_in splitting. Investigate:
- Whether `num_w_fabric_cores` is correctly set to 2 for middle 2x4 devices (vs just 1 for 2x2 devices)
- Whether the barrier semaphore wait on middle devices races with the W halo arriving

## Operational Notes

### Running tests on bh-37 (Blackhole LoudBox)

The machine runs tests inside a Docker container. The workflow is:

```bash
# From the host (already on bh-37):
docker exec 834fa6b844f2 bash /localdev/kevinmi/tt-metal/run_cin_test.sh 2>&1

# Container ID comes from the ird reservation:
docker ps  # look for "bh-37-special-kevinmi-for-reservation-NNNNN"
```

**Building:** Must be done inside the container (host lacks `mold` linker):
```bash
docker exec 834fa6b844f2 bash -c 'cd /localdev/kevinmi/tt-metal && source python_env/bin/activate && ./build_metal.sh'
```

**Device reset:** `tt-smi -r` hangs without a TTY. Use `--no_reinit` flag with individual device IDs:
```bash
docker exec CONTAINER bash -c 'for d in 0 1 2 3 4 5 6 7; do tt-smi -r $d --no_reinit; done'
```

### bh-lb-09 topology limitation (cannot reproduce 2x4 locally)

`bh-lb-09` has 8 P150b devices but physical ETH links only support a 2x2 mesh. Attempting to open a 2x4 mesh fails:
```
ERROR: TT_FATAL: Graph specified in MGD could not fit in the discovered physical topology.
```
Both `SystemMeshDescriptor` and manual `TT_MESH_GRAPH_DESC_PATH` confirm this. **Local reproduction of the 2x4-specific bug is impossible on bh-lb-09.**

### 2x2 mesh test results (bh-lb-09)

All tests pass on 2x2, including:
- `mid_res_2x2_480p_ones` (C_in=384, C_in_block=96) → PASS
- `mid_res_2x2_Wdev26_ones` (C_in=384, C_in_block=96, W_dev=26 matching 2x4 geometry) → PASS
- `test_fused_ones_input_seam` standalone vs fused → max_diff=0 (no seam on 2x2)

This confirms the bug is specific to the 2x4 topology, not C_in_block < C_in in isolation.

### Test file locations
- `models/tt_dit/tests/models/wan2_2/test_decoder_ones_input.py` — full decoder all-ones test
- `models/tt_dit/models/vae/vae_wan2_1.py` — model definition, `use_fused` toggles per layer
- `models/tt_dit/tests/models/wan2_2/test_neighbor_pad_conv3d_fused.py` — unit test for fused op
- Reference: `wan_decoder_ones_480p_81f_standalone_multiblk.pt` — standalone (no-seam) decoder output

### Current `use_fused` configuration in `vae_wan2_1.py`
- `conv_in`: `use_fused=True`
- `mid_block`: `use_fused=True`
- `up_blocks`: `use_fused=False` (all disabled)
- `conv_out`: `use_fused=False`

## Decisions
| Decision | Reason | Rejected Alternative |
|----------|--------|----------------------|
| W section sized to h_total per T | W-reader sends h_total sticks per T (including corner data from H-halo) | Sizing to H_dev (would lose corner data) |
| Option B rejected | Addressing already correct; Option B would introduce a real mismatch | Option B (change reader to H_dev stride) |
| Enabled fused for WanConv2d | Tested, didn't fix the seam — standalone path is also fine | Keep standalone NP for WanConv2d |
| Isolated layers one-by-one | Found C_in_block < C_in is the differentiating factor | Testing all layers at once (no isolation) |
| conv_in clean in both blocking configs | Confirmed no W_out_block-related bug | Blaming W_out_block splitting |
| w_neighbor_semaphore analysis | GlobalSemaphore creates per-core L1 copies; NOC writes to specific core; no sharing bug | Blaming semaphore shared-state race |
| 2x2 test confirms bug is 2x4-specific | mid_res_2x2_Wdev26_ones with C_in_block=96 passes → not a C_in_block-only bug | Continuing to blame C_in_block alone |
| Cannot test 2x4 on bh-lb-09 | ETH link topology prevents 2x4 mesh formation | Spending more time on bh-lb-09 |

## Surprises & Discoveries
- `conv_in` (C_in=32, C_in_block=32) never shows seam regardless of spatial blocking config
- `mid_block` and `up_blocks[0]` (C_in=384, C_in_block=96) always show seam on 2x4
- The DRAM-level halo addressing (page ID + byte offset) is mathematically correct for pw=1
- `tt-smi -r` on Blackhole LoudBox hangs without `--no_reinit`; need to reset devices individually
- Building tt-metal requires the Docker container (host has cmake 3.22, needs 3.24+; host lacks `mold` linker)
- `ttnn.create_global_semaphore` creates a **per-core** L1 semaphore copy (not a single shared cell), so different W directions on a middle device have independent, non-interfering semaphores
- bh-lb-09 cannot form a 2x4 mesh despite having 8 P150b devices (ETH link topology mismatch)
- `mid_res_2x2_Wdev26_ones` (C_in_block < C_in, W_dev=26 matching 2x4) passes cleanly on 2x2 → bug requires at least 3 W boundaries (middle devices)
- The test comment "only conv_in fused" in `test_decoder_ones_input.py` is stale; actual model source has `conv_in=True` **AND** `mid_block=True`

## Open Questions
- [x] Does `single_block and fused` on a real 2x4 LoudBox (C_in_block=384 override) show a seam? (dispatched to CI)
- [x] Is `num_w_fabric_cores` correctly set to 2 for middle devices on a real 2x4 mesh?
- [x] Why does the bug only appear with 3 W boundaries (2x4) and not 1 W boundary (2x2)? **ANSWERED: 2x2 has no middle devices; edge-device W-halo writes complete before conv3d reads.**

## State
- [x] Disproved H1: W-writer/reader addressing mismatch
- [x] Disproved H2: Standalone NP path as root cause
- [x] Disproved H4: Non-conv operations
- [x] Disproved H5: W_out_block splitting
- [x] Layer isolation: conv_in clean, mid_block + up_blocks[0] show seam
- [x] Deep audit of halo addressing: correct for pw=1
- [x] Deep audit of w_neighbor_semaphore: per-core L1 copies, no shared-state race
- [x] Deep audit of input_progress_semaphore: correct for single W neighbor (2x2)
- [x] Deep audit of C_in reduction in compute.cpp: FP32 accumulation is correct
- [x] Added automatic PCC/max-error comparison to test
- [x] Built tt-metal inside container (cmake 4.3.1)
- [x] 2x2 tests pass (including mid_res_2x2_Wdev26_ones with C_in_block=96)
- [x] bh-lb-09 topology confirmed as 2x2-only; cannot reproduce 2x4 locally
- [x] Added diagnostic CI test entry to `blackhole_demo_tests.yaml` for LoudBox
- [x] **ROOT CAUSE 1 IDENTIFIED:** progress semaphore not reset on conv3d reader cores
- [x] **FIX 1 APPLIED:** `noc_semaphore_set(progress_sem_ptr, 0)` after wait in `reader_vol2col.cpp`
- [x] **ROOT CAUSE 2 IDENTIFIED (Session 7, Apr 22 2026):** In `neighbor_pad_conv3d_program_factory.cpp::override_runtime_arguments`, the W-reader's per-core `RTA[10]` (holding `input_buffer->address()`) was set only at program creation and never refreshed per dispatch. On dispatch 2+, the input tensor lived at a different DRAM address but the W-reader kept using the stale pointer, reading garbage and fabric-writing that garbage into the neighbor's halo buffer. D1 then read wrong W-left halo → seam at the D0/D1 boundary (col 416 on 2x2, col 416 on 2x4). All-ones single-layer unit tests missed it because the uniform input masked the wrong address (whatever stale memory contained, it still decoded to ~uniform on a single dispatch).
- [x] **FIX 2 APPLIED:** Extended `override_runtime_arguments` to iterate `np_artifacts.fabric_core_range` and refresh `w_reader_args[10] = input_addr` each dispatch.
- [x] **2x2 FULLY FUSED DECODER VERIFIED:** PCC = 1.000000, max_err = 0.000000, |jumpL|@col 416 = 0.0000 — bit-identical to standalone across all 81 frames.
- [ ] Build fix and run on 2x4 LoudBox via CI to confirm seam is gone
- [ ] Verify 2x2 tests still pass after the fix

## Key Measurements

### Layer isolation results (all-ones latent, 2x4 480p 81f, real weights)
| Configuration | Visual Result |
|---------------|--------------|
| conv_in only fused (C_in=32, C_in_block=32) | Clean — no seam |
| conv_in fused, multi-block (production blocking) | Clean — no seam |
| conv_in + mid_block fused | **Seam visible** |
| conv_in fused + up_blocks[0] fused (mid disabled) | **Seam visible** |
| All standalone (no fused) | Clean — no seam |

### 2x2 mesh results (bh-lb-09, no seam possible)
| Configuration | Result |
|---------------|--------|
| mid_res_2x2_480p_ones (C_in=384, C_in_block=96) | PASS (max_diff=0) |
| mid_res_2x2_Wdev26_ones (C_in=384, C_in_block=96, W_dev=26) | PASS (max_diff=0) |
| test_fused_ones_input_seam all shapes | PASS (max_diff=0) |

### Reproduction commands
```bash
# Inside container on LoudBox (2x4 mesh):
cd /localdev/kevinmi/tt-metal
source python_env/bin/activate
export PYTHONPATH=$(pwd)
export TT_DIT_CACHE_DIR=/localdev/kevinmi/.cache

# Multi-block fused on 2x4 (shows seam):
pytest models/tt_dit/tests/models/wan2_2/test_decoder_ones_input.py \
    -k "2x4_h0_w1 and multi_block and fused" -x -s

# C_in_block=384 override on 2x4 (key diagnostic test):
pytest models/tt_dit/tests/models/wan2_2/test_decoder_ones_input.py \
    -k "2x4_h0_w1 and single_block and fused" -x -s
```
