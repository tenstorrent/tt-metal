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

### Hypothesis 3: Synchronization race — UNLIKELY BUT NOT FULLY RULED OUT

The progress semaphore flow appears correct. Not the primary cause.

### Hypothesis 4: Non-conv operations — DISPROVED

Non-conv operations do not cause the seam. The all-ones latent decoder test with standalone NP + separate conv3d produces a clean output. Only the fused path shows the seam.

### Hypothesis 5: Blocking configuration (W_out_block) — DISPROVED

Initially suspected that multi-W-block processing (W_out_block < W_dev) caused the seam via incorrect halo page addressing at block boundaries. But `conv_in` (C_in=32, W_out_block varies) does NOT show seams even with its production multi-W-block config. The seam is not caused by W_out_block splitting.

### Hypothesis 6: C_in_block < C_in (channel block parallelism) — ACTIVE HYPOTHESIS

**Key finding:** The seam appears ONLY in layers where `C_in_block < C_in` (multiple channel blocks processed by different cores):

| Layer | C_in | C_in_block | #C_in blocks | Seam? |
|-------|------|------------|-------------|-------|
| `conv_in` | 32 | 32 | 1 | NO |
| `mid_block` resnets | 384 | 96 | 4 | YES |
| `up_blocks[0]` resnets | 384 | 96 | 4 | YES |

**Layer isolation tests (all-ones latent, 2x4 480p 81f):**
- `conv_in` only fused → clean (no seam in any blocking config)
- `conv_in` + `mid_block` fused → seam appears
- `conv_in` fused + `up_blocks[0]` fused → seam appears
- `conv_in` multi-block by itself → still clean (re-confirmed)

The common factor in failing layers: 384-channel residual blocks with `C_in_block=96` (4-way C_in split).

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

**What this means:** The addressing at the DRAM level appears correct. The bug is likely in:
1. **L1 accumulation/reduction** — when multiple cores produce partial sums for different C_in blocks, the reduction to a single output may have an error at W-boundary output positions
2. **Synchronization between C_in block cores** — cores processing different C_in blocks for the same output position may have a race
3. **Weight loading / index offset** — the weight tile index for W-boundary halo input positions may be computed differently when C_in is split

## Next Test: Override C_in_block = 384 (full C_in, no split)

**Status: PREPARED, NOT YET RUN** (devices need reset, `tt-smi -r` requires `--no_reinit` flag on this machine)

Code is in place: `_override_mid_block_full_cin()` in `test_decoder_ones_input.py` sets `C_in_block=384` for mid_block resnets while keeping spatial blocking at production values (T=1, H=32, W=4).

Test also now has automatic PCC/max-error comparison against the standalone reference (`wan_decoder_ones_480p_81f_standalone.pt`).

**If seam disappears with C_in_block=384:** Bug is confirmed in C_in parallelism (L1 partial-sum reduction or synchronization). Next step: audit the compute kernel's partial sum accumulation and the writer's output assembly.

**If seam persists with C_in_block=384:** Bug is NOT in C_in splitting. Would need to investigate the conv3d compute kernel itself for W-boundary halo data processing errors.

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

### Test file locations
- `models/tt_dit/tests/models/wan2_2/test_decoder_ones_input.py` — full decoder all-ones test
- `models/tt_dit/models/vae/vae_wan2_1.py` — model definition, `use_fused` toggles per layer
- Reference: `wan_decoder_ones_480p_81f_standalone.pt` — standalone (no-seam) decoder output

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

## Surprises & Discoveries
- `conv_in` (C_in=32, C_in_block=32) never shows seam regardless of spatial blocking config
- `mid_block` and `up_blocks[0]` (C_in=384, C_in_block=96) always show seam
- The DRAM-level halo addressing (page ID + byte offset) is mathematically correct for pw=1 — the bug is likely in the L1-side processing (accumulation, reduction, or synchronization)
- `tt-smi -r` on Blackhole LoudBox hangs without `--no_reinit`; need to reset devices individually
- Building tt-metal requires the Docker container (host has cmake 3.22, needs 3.24+; host lacks `mold` linker)

## Open Questions
- [ ] Does overriding `C_in_block=384` (no channel split) on mid_block eliminate the seam? (test prepared, not yet run)
- [ ] If C_in splitting is confirmed as the cause, is the bug in the compute kernel's partial sum accumulation, or in how the reader dispatches halo data to different C_in cores?
- [ ] Is there a `c_in_block` synchronization barrier before the output reduction that might be missing?

## State
- [x] Disproved H1: W-writer/reader addressing mismatch
- [x] Disproved H2: Standalone NP path as root cause
- [x] Disproved H4: Non-conv operations
- [x] Disproved H5: W_out_block splitting
- [x] Layer isolation: conv_in clean, mid_block + up_blocks[0] show seam
- [x] Deep audit of halo addressing: correct for pw=1
- [x] Added automatic PCC/max-error comparison to test
- [x] Built tt-metal inside container (cmake 4.3.1)
- [ ] Run C_in_block=384 override test (devices need reset first)
- [ ] Based on result, either audit L1 accumulation or investigate compute kernel

## Key Measurements

### Layer isolation results (all-ones latent, 2x4 480p 81f, real weights)
| Configuration | Visual Result |
|---------------|--------------|
| conv_in only fused (C_in=32, C_in_block=32) | Clean — no seam |
| conv_in fused, multi-block (production blocking) | Clean — no seam |
| conv_in + mid_block fused | **Seam visible** |
| conv_in fused + up_blocks[0] fused (mid disabled) | **Seam visible** |
| All standalone (no fused) | Clean — no seam |

### Reproduction commands
```bash
# Inside container on bh-37:
cd /localdev/kevinmi/tt-metal
source python_env/bin/activate
export PYTHONPATH=$(pwd)
export TT_DIT_CACHE_DIR=/localdev/kevinmi/.cache

# Multi-block fused (shows seam with mid_block enabled):
pytest models/tt_dit/tests/models/wan2_2/test_decoder_ones_input.py \
    -k "multi_block and fused" -x -s

# C_in_block=384 override (pending test):
pytest models/tt_dit/tests/models/wan2_2/test_decoder_ones_input.py \
    -k "single_block and fused" -x -s
```
