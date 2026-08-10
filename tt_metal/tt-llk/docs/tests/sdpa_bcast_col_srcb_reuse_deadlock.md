# `sdpa_bcast_col_srcb_reuse` deadlocks an isolated compute kernel

**Status:** open — root cause identified and verified, fix not applied
**Arch:** Blackhole (verified on p300a)
**Affects:** `test_sdpa_bcast_col_srcb_reuse.py`, `test_unpack_A_sdpa.py`
**Header:** `models/demos/deepseek_v3_b1/kernel_includes/tt_llk/tt_llk_blackhole/llk_lib/llk_math_sdpa_bcast_col_srcb_reuse.h`
**Context:** advance-test scaffolds from `d83e72f`, first on-device run

---

## Summary

`_llk_math_sdpa_bcast_col_srcb_reuse_()` calls `ckernel_template::run()` **twice**
unconditionally. In an isolated compute-only kernel the second run never completes: Math stalls
forever and Packer stalls behind it (`TENSIX TIMED OUT ... waited 2 seconds for Math, Packer`,
with Unpack having completed).

There is **no test-side workaround**. The header's mop config hard-asserts `num_faces == 2`, and
that is the only value it permits — but that value deadlocks regardless of tile geometry, operand
supply, or the `dense` template flag. Every other instantiation either deadlocks or trips the
assert. Deleting the duplicated block fixes both tests immediately.

This is an LLK-side defect, not a test defect. The two tests cannot be made to pass by editing
the tests.

---

## 1. The duplicated block

`llk_math_sdpa_bcast_col_srcb_reuse.h`, lines 89-123. Lines 95-108 and 110-122 are the same
`if constexpr` chain, dispatching on the same compile-time `eltwise_binary_type`, back to back:

```cpp
inline void _llk_math_sdpa_bcast_col_srcb_reuse_(uint dst_index) {
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_BD);

    if constexpr ((eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB)) {
        ckernel_template::run();                                    // <-- run #1
    } else if constexpr (eltwise_binary_type == ELWMUL) {
        if constexpr (high_fidelity) {
            for (tile_num = 0; tile_num < num_tiles; tile_num++) { ckernel_template::run(); }
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 8, 0, p_setrwc::SET_BD);
        } else {
            ckernel_template::run();                                // <-- run #1
        }
    }

    if constexpr ((eltwise_binary_type == ELWADD) || (eltwise_binary_type == ELWSUB)) {
        ckernel_template::run();                                    // <-- run #2, lines 110-122
    } else if constexpr (eltwise_binary_type == ELWMUL) {
        if constexpr (high_fidelity) {
            for (tile_num = 0; tile_num < num_tiles; tile_num++) { ckernel_template::run(); }
        } else {
            ckernel_template::run();                                // <-- run #2
        }
    }
}
```

The only textual difference between the two copies is the `TTI_SETRWC` on line 104, present in the
first copy's high-fidelity branch and absent from the second. Every instantiation — ADD, SUB, MUL,
LoFi or HiFi — executes the MOP twice.

The MOP is programmed as `ckernel_template tmp(num_tiles, num_faces, ELWMUL(CLR_A, ...))`, so one
`run()` issues `num_tiles * num_faces` ELWMULs, each carrying `CLR_A`.

### Comparison A — the sibling op

`llk_math_sdpa_bcast_col_srca_srcb_reuse.h` is the same family, driven by a test with **identical**
tile geometry, and it does **not** hang. Its execute loops instead of duplicating:

```cpp
for (std::uint32_t tile_num = 0; tile_num < num_tiles; tile_num++) {
    ckernel_template::run();
    if constexpr (!skip_signalling) { t6_semaphore_post<p_stall::MATH>(semaphore::FPU_SFPU); }
}
```

Same family, same geometry, one loops and one duplicates — and only the duplicating one deadlocks.

### Comparison B — canonical tt-llk

`tt_llk_blackhole/llk_lib/llk_math_eltwise_binary.h:256-268` handles COL broadcast by looping over
face-rows with an explicit `CLR_B` between runs:

```cpp
for (std::uint32_t face_row = 0; face_row < num_faces_r_dim; face_row++) {
    ckernel_template::run();
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);
}
```

The forked header has neither the loop bound nor the inter-run `CLR_B`. Its hardcoded 2 runs are
consistent with `num_faces_r_dim == 2` being assumed rather than read.

---

## 2. Evidence

### 2.1 The block is where it hangs

Bisected the MATH thread by gating each call behind a `MATH_STAGE` define:

| Stage | Last call included | Timeouts |
|------:|--------------------|---------:|
| 1 | A2D datacopy seed | 0 |
| 2 | `..._init_` | 0 |
| 3 | `..._preamble_` | 0 |
| 4 | `..._` (execute) | **1** |
| 5 | `..._postamble_` | 1 |

The preamble completes, so the `STALLWAIT(SRCB_VLD)` / dummy-SrcB handshake is **not** the problem.
The execute is.

### 2.2 Deleting the duplicate fixes it

With lines 110-122 removed, both affected tests stop hanging and fall back to golden-only failures
like the other advance tests:

```
test_sdpa_bcast_col_srcb_reuse.py -> timeouts=0  1 failed
test_unpack_A_sdpa.py             -> timeouts=0  1 failed
```

The header was reverted after this experiment; it is currently unmodified.

### 2.3 Every test-side lever fails

All attempted against `test_sdpa_bcast_col_srcb_reuse.py` with the header untouched:

| Lever | Result |
|-------|--------|
| Extra operand SrcA unpacks, sweep 1→8 | Hangs. Math never consumes; Unpacker backs up behind full SrcA banks |
| Extra `_llk_unpack_A_sdpa_set_srcb_dummy_valid_()` | Hangs |
| `dense = true` template arg on `..._init_` | Hangs |
| `num_faces = 1` to the mop config | **LLK assertion fires** (see below) |
| 32×32 4-face tile, `num_faces = 2` per run | Hangs |
| Delete lines 110-122 | **Works** |

Feeding more operand data never helps, which rules out a simple SrcA deficit: the second run is not
waiting on an operand it could be given.

### 2.4 Why `num_faces = 1` is not the fix

`num_faces = 1` makes two runs cover exactly the 2 faces of the tiny tile, and the deadlock does
clear — but the run is not a pass. It trips:

```cpp
LLK_ASSERT(num_faces == 2, "num_faces must be 1, 2, or 4");   // configure_mop
```

The test then fails inside the harness's assert reporter rather than on a golden. Note the assert
is internally inconsistent: its message says 1, 2, or 4, its condition demands exactly 2, and the
enclosing `_llk_math_sdpa_bcast_col_srcb_reuse_init_` asserts the looser `{1, 2, 4}`. Worth fixing
regardless of the deadlock.

### 2.5 The test drives the primitive the same way the demo does

The advance tests deliberately pin the raw `_llk_*` primitive rather than the API wrapper, so it is
fair to ask whether they are simply calling it wrong. They are not:

```cpp
// llk_math_sdpa_bcast_col_srcb_reuse_api.h:18-21
const std::uint32_t num_faces = get_operand_num_faces(operand_id);
_llk_math_sdpa_bcast_col_srcb_reuse_init_<eltwise_binary_type, num_tiles, math_fidelity, dense>(
    num_faces, acc_to_dest);
```

The wrapper passes the operand's **total** face count, which the assert forces to 2. The test's
instantiation matches the demo's.

---

## 3. The open question

If the execute has always run the MOP twice, **why does the deepseek demo work?**

I could not answer this from the tree, and it should be settled before anyone edits code, because
it decides which fix below is correct.

The most likely explanation is that the op is never run in isolation in the demo. It sits in a
pipeline alongside a paired SFPU kernel (the `FPU_SFPU` / `UNPACK_MATH_DONE` signalling the sibling
op carries), and something in that context supplies whatever the second run consumes. If that is
right, the primitive is simply not isolation-safe, and the honest resolution is to document that
and skip the tests rather than reshape either side until the symptom disappears.

The competing explanation is that the duplication is a genuine copy-paste defect that the demo
tolerates because its geometry masks it. Comparisons A and B lean this way.

Whoever owns this op can likely answer in a minute; I would not guess.

---

## 4. Proposed solutions

Ranked. Each is contingent on §3.

### Option 1 — loop the run over face-rows (preferred, if the duplication is a defect)

Replace the second block with a proper bound, matching both the sibling op and canonical tt-llk:

```cpp
for (std::uint32_t face_row = 0; face_row < num_faces_r_dim; face_row++) {
    ckernel_template::run();
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, 0);   // canonical does this between runs
}
```

This requires `num_faces_r_dim` to be threaded into the execute, which it currently is not — the
execute takes only `dst_index` plus template params. That is the real cost of this option, and the
reason it is an LLK change rather than a test change.

Deleting lines 110-122 outright is the minimal variant. It is verified to fix both tests, but it
silently assumes one face-row, which is wrong for any 4-face operand. Prefer the loop.

**Blocked on:** confirming the intended face-row semantics, and on where the upstream of this
vendored file lives (see §6).

### Option 2 — skip the two tests, file the LLK bug

If the primitive is genuinely not isolation-safe (§3, first explanation), this is the *correct*
outcome, not a workaround:

```python
@pytest.mark.skip(reason="sdpa_bcast_col_srcb_reuse is not isolation-safe: its execute runs the "
                         "MOP twice and the second run deadlocks a compute-only kernel. See "
                         "docs/tests/sdpa_bcast_col_srcb_reuse_deadlock.md")
```

Cheap, honest, and unblocks CI immediately — a hang wedges the device and poisons every test after
it, so these two cost far more than their variant count suggests. Reversible once §3 is answered.

### Option 3 — drive the op in its real pipeline

If the second run is satisfied by a paired SFPU kernel, extend the test kernel to fake that side,
the way `sdpa_custom_mm_reuse_dest_srcb_test.cpp` already fakes `UNPACK_MATH_DONE` from the UNPACK
thread. This gives genuine coverage of the primitive as it is actually used.

Most work, best coverage, and only viable once someone states what the second run is waiting on. I
would not attempt it before §3 is answered — I already spent a full search of the parameter space
guessing, documented in §2.3, and it converged on nothing.

### Recommendation

**Option 2 now, Option 1 or 3 once §3 is answered.** Skipping stops the CI bleeding today without
committing to a theory of the bug, and neither of the real fixes is safe to pick before someone who
owns the op weighs in.

---

## 5. Related, not the same bug

### 5.1 `test_sdpa_custom_mm_reuse_dest_srcb` — third hang, separate cause, partially solved

8 variants, all hanging. Different header
(`llk_math_sdpa_custom_mm_reuse_dest_srcb.h`), different mechanism.

MATH seeds DEST with an A2D datacopy that consumes one SrcA the matmul stream never produces —
UNPACK supplies `kt_dim * nt_dim` tiles, MATH consumes `kt_dim * nt_dim + 1`. Adding a seed unpack
on the UNPACK thread (mirroring what `unpack_A_sdpa_test.cpp` already does) took it from 8 hangs to
2 genuine ones; the other 6 were cascade poisoning from the first.

Residual: `in0_rows >= 4` still hangs. The primitive MOVD2Bs a fixed 16 rows
(`4 x MOV_4_ROWS`) regardless of `in0_face_r_dim`, which is the obvious next thread to pull. The
partial fix was reverted rather than committed — a half-fixed hanging test is still a hanging test.

### 5.2 Harness bug in the assert reporter

Surfaced by §2.4, unrelated to the deadlock and worth fixing on its own:

```
AttributeError: 'CallstackEntry' object has no attribute 'file'
  helpers/device.py:316 in _print_callstack
```

`_print_callstack` reads `entry.file`, which the installed `ttexalens` version's `CallstackEntry`
does not expose. Any LLK assertion on a TRISC therefore crashes the reporter instead of printing
the assert — so an assert currently looks like a Python error, which cost real time here.

---

## 6. Before fixing: check the upstream

The header lives under `models/demos/deepseek_v3_b1/kernel_includes/`, a vendored copy. It is the
only copy in tt-metal, but I did not check tt-blaze. If the real upstream is there, the fix belongs
there and must flow down, or the next vendor sync silently reverts it.

---

## Appendix — reproduction

```bash
cd tt_metal/tt-llk/tests/python_tests

# Always reset between files. A hang wedges the device, and every later file then
# reports the unrelated "Polling brisc command timed out".
tt-smi -r
pytest -q --tb=no test_sdpa_bcast_col_srcb_reuse.py     # -> TENSIX TIMED OUT, Math + Packer

tt-smi -r
pytest -q --tb=no test_unpack_A_sdpa.py                 # -> same

tt-smi -r
pytest -q --tb=no test_sdpa_bcast_col_srca_srcb_reuse.py   # -> golden fail, NO hang (sibling op)
```

Full device status for all 8 advance-test files, and the two already-fixed
`compressed_custom_mm` bugs, are in the triage notes accompanying commit `37e5ccb`.
