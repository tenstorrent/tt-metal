# GDN Prefill Seq-Major Output — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate ~80 ms of full-pass kernel time in Qwen3.5-27B prefill by fusing per-head RMSNorm into the GDN prefill compute kernel and emitting the output in seq-major dense `[1, 1, seq, num_pairs*Dv]` layout — removing the `rms_norm → reshape → permute → reshape` chain that currently sits between the GDN kernel and the silu·z multiply in `gdn.py:forward_prefill`.

**Architecture:** A new prefill kernel pair (`gdn_prefill_with_norm.cpp` compute + `writer_gdn_prefill_seq_major.cpp` writer) lives alongside the existing pair. The reader and host op signature don't change. Selection happens through a new host entry point `gdn_prefill_fused_seq_major(...)` and an env-flagged switch in `gdn.py:forward_prefill`.

**Tech Stack:** TT-Metal kernels (C++), ttnn Python API, pytest + torch for parity testing, Qwen3.5-27B model code under `models/demos/qwen35_27b/`.

**Reference:** Spec at `docs/superpowers/specs/2026-04-28-gdn-prefill-seq-major-output.md`. Profile under analysis at `generated/profiler/reports/2026_04_28_00_27_47/`.

---

## Constants used throughout

These values come from the Qwen3.5-27B 4×P150 configuration and are stable across all tasks:

- `Nv_TP = 12` (value heads per device after TP=4)
- `Nk_TP = 4` (key heads per device after TP=4)
- `Dk = 128` (per-head key dim)
- `Dv = 128` (per-head value dim)
- `Vt = Dv / 32 = 4` (tiles per Dv vector)
- `Kt = Dk / 32 = 4`
- `value_dim_tp = Nv_TP * Dv = 1536`
- `num_pairs = B * Nv_TP` (= 12 in prefill since B=1)
- `num_tokens = seq` (chunked-prefill chunk size, 2048 in production)
- `BF16_TILE_BYTES = 2048` (32 × 32 × 2)

---

## File Structure

**New files:**
- `models/demos/qwen35_27b/tt/gdn_kernel/kernels/compute/gdn_prefill_with_norm.cpp` — copy of `gdn_prefill.cpp` with one extra RMSNorm phase before output is pushed to `cb_out`.
- `models/demos/qwen35_27b/tt/gdn_kernel/kernels/dataflow/writer_gdn_prefill_seq_major.cpp` — writer that L1-buffers 32 tokens of CB output and writes them into a dense seq-major destination buffer.
- `models/demos/qwen35_27b/tt/tests/test_gdn_prefill_seq_major.py` — unit test that drives the new entry point and compares against the existing path's output post-`ttnn.rms_norm` + `reshape → permute → reshape`.

**Modified files:**
- `models/demos/qwen35_27b/tt/gdn_kernel/gdn_kernel_op.py` — add path constants, update `_compute_kernel_hash` to include the new files, add `_build_prefill_seq_major_device_program(...)`, `_gdn_prefill_fused_seq_major(...)`, `gdn_prefill_fused_seq_major(...)`.
- `models/demos/qwen35_27b/tt/gdn.py` — env-flagged switch in `forward_prefill` (lines 909–950): allocate output as `[1, 1, seq, num_pairs*Dv]`, call `gdn_prefill_fused_seq_major`, drop the post-kernel `rms_norm → reshape → permute → reshape` chain.

**Untouched:**
- `gdn_prefill.cpp`, `writer_gdn_prefill.cpp`, `reader_gdn_prefill.cpp`, the existing `gdn_prefill_fused()` entry point, the `dv_split` variants. None of these change.

---

## Task 1: Scaffold paths + entry point + failing unit test

**Files:**
- Modify: `models/demos/qwen35_27b/tt/gdn_kernel/gdn_kernel_op.py:39-89` (path constants + hash)
- Create: `models/demos/qwen35_27b/tt/tests/test_gdn_prefill_seq_major.py`

This task wires the test framework BEFORE any kernel exists — "red" stage of TDD. After this task, importing the new entry point fails, and the test fails because the function doesn't exist. We don't add the entry point yet to keep the failure unambiguous.

- [ ] **Step 1.1: Add path constants for the new compute and writer kernels**

In `models/demos/qwen35_27b/tt/gdn_kernel/gdn_kernel_op.py`, after line 44 (which currently ends with `COMPUTE_PREFILL_SKELETON_PATH = ...`), add:

```python
COMPUTE_PREFILL_WITH_NORM_PATH = f"{_KERNEL_DIR}/compute/gdn_prefill_with_norm.cpp"
WRITER_PREFILL_SEQ_MAJOR_PATH = f"{_KERNEL_DIR}/dataflow/writer_gdn_prefill_seq_major.cpp"
```

In the same file, in `_compute_kernel_hash()` (lines 60-86), add the two new paths to the list iterated in the `for path in [...]` block:

```python
    for path in [
        READER_PATH,
        WRITER_PATH,
        # ... existing entries ...
        COMPUTE_PREFILL_SKELETON_PATH,
        COMPUTE_PREFILL_WITH_NORM_PATH,       # NEW
        WRITER_PREFILL_SEQ_MAJOR_PATH,        # NEW
    ]:
```

The hash is intentionally tolerant of missing files (`except FileNotFoundError: h.update(path.encode())`), so this works even before the kernel files exist.

- [ ] **Step 1.2: Create the unit test file with one parametrized test**

Create `models/demos/qwen35_27b/tt/tests/test_gdn_prefill_seq_major.py` with this complete content:

```python
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Correctness test for gdn_prefill_fused_seq_major.

Compares the new kernel pair (fused-norm compute + seq-major writer) against
the existing gdn_prefill_fused output post-applied with ttnn.rms_norm and the
reshape -> permute -> reshape chain that gdn.py:forward_prefill currently does
between the kernel and the silu*z multiply.

Pass criteria: PCC >= 0.999 on output, PCC >= 0.999 on final state.

Run:
    export TT_METAL_HOME=$(pwd)
    export HF_MODEL=/local/ttuser/atupe/Qwen27bFP8
    pytest models/demos/qwen35_27b/tt/tests/test_gdn_prefill_seq_major.py -v -s
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.qwen35_27b.tt.model import create_qwen35_model


def _get_model_path():
    return os.path.expanduser(os.environ.get("HF_MODEL", "~/models/Qwen3.5-27B-FP8"))


def _unshard(t):
    if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
    return t


def _to_mesh(t, mesh_device):
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _compute_pcc(ref, test):
    r = ref.float().flatten()
    t = test.float().flatten()
    if r.numel() == 0:
        return 1.0
    vr = r - r.mean()
    vt = t - t.mean()
    num = (vr * vt).sum()
    den = (vr.norm() * vt.norm()) + 1e-12
    return (num / den).item()


def _load_gdn_layer(mesh_device):
    model_path = _get_model_path()
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path
    model = create_qwen35_model(
        mesh_device,
        model_path=model_path,
        max_batch_size=32,
        max_seq_len=2048,
        dtype=ttnn.bfloat8_b,
        n_layers=3,
    )
    args = model.args
    gdn_layer_idx = None
    for i in range(args.n_layers):
        if args.layer_types[i] == "linear_attention":
            gdn_layer_idx = i
            break
    assert gdn_layer_idx is not None, "No GDN layer found in first 3 layers"
    gdn = model.layers[gdn_layer_idx].attention
    tw = gdn.tw
    return model, args, gdn, tw


def _reference_old_path_with_norm_and_permute(
    conv_out_3d, a_3d, b_3d, gdn, tw, mesh_device, num_pairs, num_tokens, Nv_TP, Nk_TP, repeat_factor, key_dim_tp
):
    """Drives the existing gdn_prefill_fused, then applies the post-kernel
    rms_norm + reshape + permute + reshape that gdn.py currently performs.
    Returns the final dense tensor in shape [1, 1, num_tokens, Nv_TP*Dv]."""
    from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_prefill_fused

    Dv = gdn.Dv
    rec_states = _to_mesh(torch.zeros(num_pairs, gdn.Dk, Dv, dtype=torch.bfloat16), mesh_device)
    flat_output = _to_mesh(torch.zeros(num_pairs * num_tokens, 1, Dv, dtype=torch.bfloat16), mesh_device)

    gdn_prefill_fused(
        conv_out_3d, a_3d, b_3d,
        gdn.neg_exp_A, tw["dt_bias"], tw["norm_w"],
        gdn.scale_tt, gdn.rms_scale_tt, gdn.rms_eps_tt,
        rec_states, flat_output,
        num_pairs=num_pairs, num_tokens=num_tokens,
        Nv_TP=Nv_TP, Nk_TP=Nk_TP, repeat_factor=repeat_factor, key_dim_tp=key_dim_tp,
    )

    out_n = ttnn.rms_norm(flat_output, weight=tw["norm_w"], epsilon=1e-6)
    ttnn.deallocate(flat_output)
    out_4d = ttnn.reshape(out_n, (1, num_pairs, num_tokens, Dv))
    ttnn.deallocate(out_n)
    out_4d = ttnn.permute(out_4d, (0, 2, 1, 3))
    out_dense = ttnn.reshape(out_4d, (1, 1, num_tokens, num_pairs * Dv))
    ttnn.deallocate(out_4d)

    ref_dense = ttnn.to_torch(out_dense, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
    ref_state = ttnn.to_torch(rec_states, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
        :num_pairs
    ].float()

    ttnn.deallocate(out_dense)
    ttnn.deallocate(rec_states)
    return ref_dense, ref_state


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "P150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 8))
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize("num_tokens", [64, 96, 2048])
def test_gdn_prefill_seq_major_correctness(mesh_device, reset_seeds, ensure_gc, num_tokens):
    """gdn_prefill_fused_seq_major must match (old kernel + ttnn.rms_norm + reshape/permute/reshape).

    num_tokens parametrization covers:
      - 64: small tile-aligned baseline.
      - 96: another tile-aligned case (catches off-by-one in the writer's L1 repacking).
      - 2048: production chunked-prefill chunk size.
    """
    if mesh_device.get_num_devices() < 4:
        pytest.skip("Full model requires TP>=4")

    # Lazy import — at Task 1, this function does not exist yet, so the test
    # fails at import. That is the "red" stage. Tasks 2+ make this import
    # succeed and the assertion pass.
    from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_prefill_fused_seq_major

    _, _args, gdn, tw = _load_gdn_layer(mesh_device)
    Nv_TP = gdn.Nv_TP
    Nk_TP = gdn.Nk_TP
    Dv = gdn.Dv
    qkv_dim_tp = gdn.qkv_dim_tp
    key_dim_tp = gdn.key_dim_tp
    num_pairs = 1 * Nv_TP
    repeat_factor = Nv_TP // Nk_TP

    logger.info(f"test_gdn_prefill_seq_major: N={num_tokens}, Nv_TP={Nv_TP}, Dv={Dv}")

    torch.manual_seed(42)
    conv_out_all = _to_mesh(torch.randn(1, 1, num_tokens, qkv_dim_tp, dtype=torch.bfloat16) * 0.1, mesh_device)
    a_all = _to_mesh(torch.randn(1, 1, num_tokens, Nv_TP, dtype=torch.bfloat16) * 0.1, mesh_device)
    b_all = _to_mesh(torch.randn(1, 1, num_tokens, Nv_TP, dtype=torch.bfloat16) * 0.1, mesh_device)

    conv_out_3d_a = _unshard(ttnn.reshape(conv_out_all, (1, num_tokens, qkv_dim_tp)))
    a_3d_a = _unshard(ttnn.reshape(a_all, (1, num_tokens, Nv_TP)))
    b_3d_a = _unshard(ttnn.reshape(b_all, (1, num_tokens, Nv_TP)))

    ref_dense, ref_state = _reference_old_path_with_norm_and_permute(
        conv_out_3d_a, a_3d_a, b_3d_a, gdn, tw, mesh_device,
        num_pairs=num_pairs, num_tokens=num_tokens,
        Nv_TP=Nv_TP, Nk_TP=Nk_TP, repeat_factor=repeat_factor, key_dim_tp=key_dim_tp,
    )

    ttnn.deallocate(conv_out_3d_a)
    ttnn.deallocate(a_3d_a)
    ttnn.deallocate(b_3d_a)

    # --- New kernel path ---
    conv_out_3d_b = _unshard(ttnn.reshape(conv_out_all, (1, num_tokens, qkv_dim_tp)))
    a_3d_b = _unshard(ttnn.reshape(a_all, (1, num_tokens, Nv_TP)))
    b_3d_b = _unshard(ttnn.reshape(b_all, (1, num_tokens, Nv_TP)))

    test_rec_states = _to_mesh(torch.zeros(num_pairs, gdn.Dk, Dv, dtype=torch.bfloat16), mesh_device)
    test_output = _to_mesh(
        torch.zeros(1, 1, num_tokens, num_pairs * Dv, dtype=torch.bfloat16), mesh_device
    )

    gdn_prefill_fused_seq_major(
        conv_out_3d_b, a_3d_b, b_3d_b,
        gdn.neg_exp_A, tw["dt_bias"], tw["norm_w"],
        gdn.scale_tt, gdn.rms_scale_tt, gdn.rms_eps_tt,
        test_rec_states, test_output,
        num_pairs=num_pairs, num_tokens=num_tokens,
        Nv_TP=Nv_TP, Nk_TP=Nk_TP, repeat_factor=repeat_factor, key_dim_tp=key_dim_tp,
    )

    test_dense = ttnn.to_torch(test_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
    test_state = ttnn.to_torch(test_rec_states, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
        :num_pairs
    ].float()

    ttnn.deallocate(conv_out_all)
    ttnn.deallocate(a_all)
    ttnn.deallocate(b_all)
    ttnn.deallocate(conv_out_3d_b)
    ttnn.deallocate(a_3d_b)
    ttnn.deallocate(b_3d_b)
    ttnn.deallocate(test_rec_states)
    ttnn.deallocate(test_output)

    output_pcc = _compute_pcc(ref_dense, test_dense)
    state_pcc = _compute_pcc(ref_state, test_state)

    logger.info(f"  num_tokens={num_tokens} Output PCC: {output_pcc:.6f}")
    logger.info(f"  num_tokens={num_tokens} State  PCC: {state_pcc:.6f}")

    assert output_pcc > 0.999, f"Output PCC {output_pcc:.6f} < 0.999 (num_tokens={num_tokens})"
    assert state_pcc > 0.999, f"State PCC {state_pcc:.6f} < 0.999 (num_tokens={num_tokens})"
```

- [ ] **Step 1.3: Run the test to confirm it fails with ImportError**

Run: `pytest models/demos/qwen35_27b/tt/tests/test_gdn_prefill_seq_major.py -v -s 2>&1 | tail -30`

Expected: `ImportError: cannot import name 'gdn_prefill_fused_seq_major' from 'models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op'` (or `AttributeError` from the import line).

This is the red stage. Confirm the failure mode is the missing function — not a typo, not a wrong path.

- [ ] **Step 1.4: Commit**

```bash
git add models/demos/qwen35_27b/tt/gdn_kernel/gdn_kernel_op.py \
        models/demos/qwen35_27b/tt/tests/test_gdn_prefill_seq_major.py
git commit -m "$(cat <<'EOF'
test(qwen35-gdn): scaffold seq-major prefill kernel test (red)

Adds the failing unit test that the new gdn_prefill_fused_seq_major
entry point will satisfy once the kernel pair is in place. Test compares
new path against (old kernel + ttnn.rms_norm + reshape/permute/reshape).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: New compute kernel (identical copy) + new entry point — green

**Files:**
- Create: `models/demos/qwen35_27b/tt/gdn_kernel/kernels/compute/gdn_prefill_with_norm.cpp` (initially identical to `gdn_prefill.cpp`)
- Modify: `models/demos/qwen35_27b/tt/gdn_kernel/gdn_kernel_op.py` (add builder + entry point)

This task gets the new entry point compiling and running, but with the **old writer** and a compute kernel that is byte-identical to the existing one. The unit test in Task 1 will FAIL with a PCC near zero (because the new path doesn't apply rms_norm yet) — that's expected and fixed in Task 3.

The point of this stage is to isolate kernel-wiring bugs from RMSNorm-math bugs.

- [ ] **Step 2.1: Create gdn_prefill_with_norm.cpp as an identical copy**

Run:

```bash
cp models/demos/qwen35_27b/tt/gdn_kernel/kernels/compute/gdn_prefill.cpp \
   models/demos/qwen35_27b/tt/gdn_kernel/kernels/compute/gdn_prefill_with_norm.cpp
```

Update the top header comment in the new file to:

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Prefill GDN compute kernel WITH FUSED RMSNORM (variant of gdn_prefill.cpp).
// Currently identical to the parent; RMSNorm phase is added in a follow-up step.
```

(Replace the existing top comment block — lines 1-17 of the original.)

- [ ] **Step 2.2: Add `_build_prefill_seq_major_device_program` next to the existing builder**

In `models/demos/qwen35_27b/tt/gdn_kernel/gdn_kernel_op.py`, immediately after `_build_prefill_device_program(...)` ends (after the `return ttnn.ProgramDescriptor(...)` block at ~line 970), add a new builder. It is a near-copy of `_build_prefill_device_program` but with three differences:

1. The compute kernel path is `COMPUTE_PREFILL_WITH_NORM_PATH`.
2. The writer kernel path is **still** `WRITER_PREFILL_PATH` (we swap to seq-major in Task 4).
3. The reader kernel path is unchanged (`READER_PREFILL_PATH`).

Concretely, append:

```python
def _build_prefill_seq_major_device_program(
    conv_out_dev,
    a_dev,
    b_dev,
    neg_exp_A_dev,
    dt_bias_dev,
    norm_w_dev,
    scale_dev,
    rms_scale_dev,
    rms_eps_dev,
    state_dev,
    output_dev,
    num_pairs_total,
    num_tokens,
    num_cores,
    grid,
    state_in_l1=False,
    Nv_TP=12,
    Nk_TP=4,
    repeat_factor=3,
    key_dim_tp=512,
):
    """ProgramDescriptor for the seq-major GDN prefill variant.

    Identical to _build_prefill_device_program except the compute kernel
    path is COMPUTE_PREFILL_WITH_NORM_PATH (which folds in per-head RMSNorm
    on the kernel's output before push to cb_out) and the writer path will
    move to WRITER_PREFILL_SEQ_MAJOR_PATH in a later task.
    """
    max_cores = grid.x * grid.y
    num_cores = min(num_cores, num_pairs_total, max_cores)
    pairs_per_core = num_pairs_total // num_cores
    remainder = num_pairs_total % num_cores

    core_coords = []
    for i in range(num_cores):
        core_coords.append(ttnn.CoreCoord(i % grid.x, i // grid.x))

    core_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in core_coords])

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    pair_offset = 0
    core_pair_counts = []

    for i, cc in enumerate(core_coords):
        n = pairs_per_core + (1 if i < remainder else 0)
        core_pair_counts.append(n)
        reader_rt_args[cc.x][cc.y] = [
            conv_out_dev.buffer_address(),  # 0
            a_dev.buffer_address(),  # 1
            b_dev.buffer_address(),  # 2
            neg_exp_A_dev.buffer_address(),  # 3
            dt_bias_dev.buffer_address(),  # 4
            norm_w_dev.buffer_address(),  # 5
            scale_dev.buffer_address(),  # 6
            rms_scale_dev.buffer_address(),  # 7
            state_dev.buffer_address(),  # 8
            rms_eps_dev.buffer_address(),  # 9
            pair_offset,  # 10
            n,  # 11
        ]
        writer_rt_args[cc.x][cc.y] = [
            output_dev.buffer_address(),
            state_dev.buffer_address(),
            pair_offset,
            n,
        ]
        pair_offset += n

    key_tile_off = key_dim_tp // 32
    v_tile_off = 2 * key_tile_off
    qkv_dim_tp = key_dim_tp * 2 + Nv_TP * (128)
    conv_tiles_per_row = qkv_dim_tp // 32
    ab_tiles_per_row = (Nv_TP + 31) // 32

    cb_descriptors = [
        _make_cb(0, Kt, core_ranges),  # cb_q_raw
        _make_cb(1, Kt, core_ranges),  # cb_k_raw
        _make_cb(2, Kt, core_ranges),  # cb_k_col
        _make_cb(3, Vt, core_ranges),  # cb_v
        _make_cb(4, 1, core_ranges),  # cb_g
        _make_cb(5, 1, core_ranges),  # cb_beta
        _make_cb(6, STATE_TILES, core_ranges),  # cb_state_in
        _make_cb(7, STATE_TILES, core_ranges),  # cb_state_b
        _make_cb(8, STATE_TILES, core_ranges),  # cb_state_out
        _make_cb(9, 1, core_ranges),  # cb_a
        _make_cb(10, 1, core_ranges),  # cb_b
        _make_cb(12, 1, core_ranges),  # cb_neg_exp_A
        _make_cb(13, 1, core_ranges),  # cb_dt_bias
        _make_cb(14, Vt, core_ranges),  # cb_norm_w
        _make_cb(15, 1, core_ranges),  # cb_scale
        _make_cb(16, Vt, core_ranges),  # cb_out
        _make_cb(17, Kt, core_ranges),  # cb_q
        _make_cb(18, Kt, core_ranges),  # cb_k_row
        _make_cb(21, 1, core_ranges),  # cb_scratch
        _make_cb(24, 1, core_ranges),  # cb_exp_g
        _make_cb(25, Vt, core_ranges),  # cb_kv_mem
        _make_cb(26, Vt, core_ranges),  # cb_delta
        _make_cb(27, Vt, core_ranges),  # cb_delta_s
        _make_cb(28, Kt, core_ranges),  # cb_sq_acc
        _make_cb(29, 1, core_ranges),  # cb_tmp
        _make_cb(31, 1, core_ranges),  # cb_rms_scale
        _make_cb(19, 1, core_ranges),  # cb_reduce_scaler
        _make_cb(20, 1, core_ranges),  # cb_rms_eps
    ]

    state_l1_flag = 1 if state_in_l1 else 0
    packed_reduce_scaler = 0x3F803F80

    groups = {}
    for i, cc in enumerate(core_coords):
        n = core_pair_counts[i]
        groups.setdefault(n, []).append(cc)

    all_kernels = []
    for n_pairs, cores in groups.items():
        if n_pairs == 0:
            continue

        group_ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

        group_reader_rt = ttnn.RuntimeArgs()
        group_writer_rt = ttnn.RuntimeArgs()
        for c in cores:
            group_reader_rt[c.x][c.y] = list(reader_rt_args[c.x][c.y])
            group_writer_rt[c.x][c.y] = list(writer_rt_args[c.x][c.y])

        reader_ct = [
            Kt, Vt, BF16_TILE_BYTES, state_l1_flag, packed_reduce_scaler,
            Nv_TP, Nk_TP, repeat_factor, key_tile_off, v_tile_off,
            num_tokens, conv_tiles_per_row, ab_tiles_per_row,
        ]
        # Writer compile-time args still match the OLD writer's layout for now.
        # Task 4 swaps in the seq-major writer with its own compile-time args.
        writer_ct = [Kt, Vt, BF16_TILE_BYTES, state_l1_flag, num_tokens]

        reader_kd = ttnn.KernelDescriptor(
            kernel_source=READER_PREFILL_PATH,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=group_ranges,
            compile_time_args=reader_ct,
            runtime_args=group_reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer_kd = ttnn.KernelDescriptor(
            kernel_source=WRITER_PREFILL_PATH,  # OLD writer for Task 2-3; swapped in Task 4
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=group_ranges,
            compile_time_args=writer_ct,
            runtime_args=group_writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        )
        compute_kd = ttnn.KernelDescriptor(
            kernel_source=COMPUTE_PREFILL_WITH_NORM_PATH,  # NEW compute path
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=group_ranges,
            compile_time_args=[Kt, Vt, n_pairs, num_tokens],
            runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=False,
            ),
        )
        all_kernels.extend([reader_kd, writer_kd, compute_kd])

    return ttnn.ProgramDescriptor(kernels=all_kernels, cbs=cb_descriptors)
```

- [ ] **Step 2.3: Add `_gdn_prefill_fused_seq_major` and `gdn_prefill_fused_seq_major` entry points**

In the same file, immediately after `gdn_prefill_fused(...)` ends (after line 1119), append:

```python
def _gdn_prefill_fused_seq_major(
    conv_out,
    a_fused,
    b_fused,
    neg_exp_A,
    dt_bias,
    norm_w,
    scale_tt,
    rms_scale_tt,
    rms_eps_tt,
    state,
    output,
    num_pairs_total,
    num_tokens,
    num_cores=12,
    Nv_TP=12,
    Nk_TP=4,
    repeat_factor=3,
    key_dim_tp=512,
):
    """Execute the seq-major prefill GDN kernel (fused-norm compute + seq-major writer)."""
    mesh_device = conv_out.device()
    mesh_shape = mesh_device.shape
    num_devices = mesh_shape[0] * mesh_shape[1]

    state_in_l1 = state.memory_config().buffer_type == ttnn.BufferType.L1

    all_tensors = [
        conv_out, a_fused, b_fused, neg_exp_A, dt_bias, norm_w,
        scale_tt, rms_scale_tt, rms_eps_tt, state, output,
    ]

    if num_devices == 1:
        devs = [ttnn.get_device_tensors(t)[0] for t in all_tensors]
        grid = devs[0].device().compute_with_storage_grid_size()
        program = _build_prefill_seq_major_device_program(
            *devs, num_pairs_total, num_tokens, num_cores, grid,
            state_in_l1=state_in_l1, Nv_TP=Nv_TP, Nk_TP=Nk_TP,
            repeat_factor=repeat_factor, key_dim_tp=key_dim_tp,
        )
        return ttnn.generic_op(all_tensors, program)

    per_device = [ttnn.get_device_tensors(t) for t in all_tensors]
    mesh_program = ttnn.MeshProgramDescriptor()
    for row in range(mesh_shape[0]):
        for col in range(mesh_shape[1]):
            device_idx = row * mesh_shape[1] + col
            coord = ttnn.MeshCoordinate(row, col)
            devs = [per_device[i][device_idx] for i in range(len(all_tensors))]
            grid = devs[0].device().compute_with_storage_grid_size()
            program = _build_prefill_seq_major_device_program(
                *devs, num_pairs_total, num_tokens, num_cores, grid,
                state_in_l1=state_in_l1, Nv_TP=Nv_TP, Nk_TP=Nk_TP,
                repeat_factor=repeat_factor, key_dim_tp=key_dim_tp,
            )
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program

    return ttnn.generic_op(all_tensors, mesh_program)


def gdn_prefill_fused_seq_major(
    conv_out,
    a_fused,
    b_fused,
    neg_exp_A,
    dt_bias,
    norm_w,
    scale_tt,
    rms_scale_tt,
    rms_eps_tt,
    state,
    output,
    num_pairs,
    num_tokens,
    num_cores=12,
    Nv_TP=12,
    Nk_TP=4,
    repeat_factor=3,
    key_dim_tp=512,
):
    """Prefill GDN with fused per-head RMSNorm and seq-major dense output.

    Args:
      conv_out:   [1, N, qkv_dim_tp]              TILE BFLOAT16 DRAM
      a_fused:    [1, N, Nv_TP]                   TILE BFLOAT16 DRAM
      b_fused:    [1, N, Nv_TP]                   TILE BFLOAT16 DRAM
      neg_exp_A:  [1, 1, Nv_TP]                   TILE BFLOAT16
      dt_bias:    [1, 1, Nv_TP]                   TILE BFLOAT16
      norm_w:     [1, 1, Dv]                      TILE BFLOAT16  (CONSUMED by the
                  compute kernel as the per-head RMSNorm weight — no longer
                  unused-for-API-compat as in gdn_prefill_fused.)
      scale_tt:   [1, 1, 1]                       TILE BFLOAT16
      rms_scale_tt:[1, 1, 1]                      TILE BFLOAT16  (consumed)
      rms_eps_tt: [1, 1, 1]                       TILE BFLOAT16  (consumed,
                  expected to equal Dv * 1e-6)
      state:      [num_pairs, Dk, Dv]             TILE BFLOAT16  (in-place)
      output:     [1, 1, num_tokens, num_pairs*Dv] TILE BFLOAT16 DRAM
                  (NOTE: shape differs from gdn_prefill_fused, which uses
                  [num_pairs*N, 1, Dv].)
    """
    logger.debug(f"GDN prefill seq-major: num_pairs={num_pairs}, num_tokens={num_tokens}")
    _gdn_prefill_fused_seq_major(
        conv_out, a_fused, b_fused, neg_exp_A, dt_bias, norm_w,
        scale_tt, rms_scale_tt, rms_eps_tt, state, output,
        num_pairs_total=num_pairs, num_tokens=num_tokens,
        num_cores=num_cores, Nv_TP=Nv_TP, Nk_TP=Nk_TP,
        repeat_factor=repeat_factor, key_dim_tp=key_dim_tp,
    )
```

- [ ] **Step 2.4: Run the unit test — expect PCC failure (not import failure)**

Run: `pytest models/demos/qwen35_27b/tt/tests/test_gdn_prefill_seq_major.py::test_gdn_prefill_seq_major_correctness -v -s -k "num_tokens=64" 2>&1 | tail -50`

Expected: the test now imports successfully and the kernel runs end-to-end, but it asserts a PCC failure on `output_pcc`. **Two distinct failure modes are acceptable here:**

1. The PCC is wildly wrong because the new path's output is the kernel's raw output (no rms_norm applied) but the test expects post-norm output. Output PCC will be far from 1.0.
2. `state_pcc > 0.999` should pass (recurrence math is unchanged).

Confirm:
- The kernel compiled (no `RuntimeError` about a kernel source file).
- The state PCC is very high (≥ 0.999).
- The output PCC is failing as expected.

If you see a kernel compile error, the new compute kernel file likely has a syntax issue from the comment update — re-read `gdn_prefill_with_norm.cpp` and fix.

- [ ] **Step 2.5: Commit**

```bash
git add models/demos/qwen35_27b/tt/gdn_kernel/kernels/compute/gdn_prefill_with_norm.cpp \
        models/demos/qwen35_27b/tt/gdn_kernel/gdn_kernel_op.py
git commit -m "$(cat <<'EOF'
feat(qwen35-gdn): add seq-major prefill kernel scaffolding

Adds _build_prefill_seq_major_device_program, _gdn_prefill_fused_seq_major,
and the public gdn_prefill_fused_seq_major entry point. Compute kernel
gdn_prefill_with_norm.cpp is currently an identical copy of gdn_prefill.cpp
(RMSNorm phase added in the next commit). Writer is still the existing one;
output buffer shape matches the seq-major target.

Unit test compiles and runs but fails on output PCC (expected — RMSNorm
not yet applied). State PCC passes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add fused per-head RMSNorm to the compute kernel

**Files:**
- Modify: `models/demos/qwen35_27b/tt/gdn_kernel/kernels/compute/gdn_prefill_with_norm.cpp`

This task adds one extra phase to the per-token compute path: between `Phase 5.6: output = q @ state_out` (which fills `cb_out` with `Vt` tiles) and the `cb_pop_front(cb_q, Kt)` cleanup, apply per-head RMSNorm to the `cb_out` tiles in place.

The math is `out = out * rsqrt(sum(out^2)/Dv + eps) * norm_w`. We already have:
- `cb_out` populated with `Vt` tiles holding the un-normalized Dv-vector.
- `cb_norm_w` (Vt tiles) — pinned to front via `cb_wait_front` before the main loop, never popped during the loop.
- `cb_rms_eps` (1 tile) — pinned scalar = `Dv * 1e-6`.

The existing `cb_sq_acc` and `cb_tmp` CBs (used for L2-norm of Q and K) are free again at this point (popped after Phase 2). We can reuse them.

After Task 3 the unit test should reach PCC ≥ 0.999 on output but the writer is still the OLD writer, so the test's assertion compares two `[num_pairs*seq, 1, Dv]` tensors — the test's reference path also still produces `[num_pairs*seq, 1, Dv]` because we haven't moved to seq-major writer yet.

Wait — re-read: the test expects output shape `[1, 1, num_tokens, num_pairs*Dv]`. With the OLD writer producing `[num_pairs*seq, 1, Dv]`, the test will fail at allocation/shape level.

To keep this task TDD-friendly, we need an interim test mode that asserts equivalence in the writer's native `[num_pairs*seq, 1, Dv]` layout BEFORE we swap writers in Task 4.

Add an interim assertion path to the test that bypasses the layout transformation when an env var is set.

- [ ] **Step 3.1: Add interim shape-check mode to the unit test**

Edit `models/demos/qwen35_27b/tt/tests/test_gdn_prefill_seq_major.py`. Inside `test_gdn_prefill_seq_major_correctness`, BEFORE the `test_output = _to_mesh(... shape (1,1,N,num_pairs*Dv))` line, insert this branch:

```python
    # During staging (Tasks 3-4), the new entry point may still be using the
    # OLD writer that emits [num_pairs*N, 1, Dv]. Set GDN_SEQ_MAJOR_INTERIM=1 to
    # exercise the kernel via the old writer's output shape and compare against
    # the reference's pre-permute intermediate (rms_norm output, NO permute).
    interim = os.environ.get("GDN_SEQ_MAJOR_INTERIM", "")
    if interim:
        # Re-run reference WITHOUT the post-rms_norm permute/reshape — just rms_norm.
        from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_prefill_fused as _old_pf
        ref_states = _to_mesh(torch.zeros(num_pairs, gdn.Dk, Dv, dtype=torch.bfloat16), mesh_device)
        ref_flat = _to_mesh(torch.zeros(num_pairs * num_tokens, 1, Dv, dtype=torch.bfloat16), mesh_device)
        conv_out_3d_r = _unshard(ttnn.reshape(conv_out_all, (1, num_tokens, qkv_dim_tp)))
        a_3d_r = _unshard(ttnn.reshape(a_all, (1, num_tokens, Nv_TP)))
        b_3d_r = _unshard(ttnn.reshape(b_all, (1, num_tokens, Nv_TP)))
        _old_pf(conv_out_3d_r, a_3d_r, b_3d_r, gdn.neg_exp_A, tw["dt_bias"], tw["norm_w"],
                gdn.scale_tt, gdn.rms_scale_tt, gdn.rms_eps_tt,
                ref_states, ref_flat, num_pairs=num_pairs, num_tokens=num_tokens,
                Nv_TP=Nv_TP, Nk_TP=Nk_TP, repeat_factor=repeat_factor, key_dim_tp=key_dim_tp)
        ref_normed = ttnn.rms_norm(ref_flat, weight=tw["norm_w"], epsilon=1e-6)
        ref_dense = ttnn.to_torch(ref_normed, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()
        ref_state = ttnn.to_torch(ref_states, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[
            :num_pairs
        ].float()
        ttnn.deallocate(ref_normed); ttnn.deallocate(ref_states)
        ttnn.deallocate(conv_out_3d_r); ttnn.deallocate(a_3d_r); ttnn.deallocate(b_3d_r)
        # Reshape test_output allocation to match old writer
        test_rec_states = _to_mesh(torch.zeros(num_pairs, gdn.Dk, Dv, dtype=torch.bfloat16), mesh_device)
        test_output = _to_mesh(torch.zeros(num_pairs * num_tokens, 1, Dv, dtype=torch.bfloat16), mesh_device)
        # ... existing kernel-call block runs as-is ...
```

This adds an interim mode to the test. After Task 3 completes you'll run the test with `GDN_SEQ_MAJOR_INTERIM=1` to validate the compute kernel change before Task 4 swaps in the new writer.

(The full layout-aware assertion at the end of the test still runs unchanged when the flag is unset — that case will fail until Task 4. The test parametrization stays as-is.)

- [ ] **Step 3.2: Add the RMSNorm phase to gdn_prefill_with_norm.cpp**

Open `models/demos/qwen35_27b/tt/gdn_kernel/kernels/compute/gdn_prefill_with_norm.cpp`. Locate the block immediately after `// 5.6: output = q @ state_out` (around line 393 in the original copy) — the loop that pushes `Vt` tiles into `cb_out`. **Before** the line `// Pop per-token inputs` (around line 409), insert the RMSNorm in-place phase. Concretely, replace this section (existing code from line ~393 to line ~408):

```cpp
            // 5.6: output = q @ state_out
            cb_wait_front(cb_state_out, state_tiles);
            cb_reserve_back(cb_out, Vt);
            mm_init(cb_q, cb_state_out, cb_out);
            for (uint32_t vt = 0; vt < Vt; vt++) {
                tile_regs_acquire();
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    matmul_tiles(cb_q, cb_state_out, kt, kt * Vt + vt, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out, vt);
                tile_regs_release();
            }
            cb_push_back(cb_out, Vt);

            // Pop per-token inputs
```

with the same body, plus the new RMSNorm phase before the cb_push_back/Pop block:

```cpp
            // 5.6: output = q @ state_out (writes into cb_out, Vt tiles)
            cb_wait_front(cb_state_out, state_tiles);
            cb_reserve_back(cb_out, Vt);
            mm_init(cb_q, cb_state_out, cb_out);
            for (uint32_t vt = 0; vt < Vt; vt++) {
                tile_regs_acquire();
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    matmul_tiles(cb_q, cb_state_out, kt, kt * Vt + vt, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out, vt);
                tile_regs_release();
            }
            cb_push_back(cb_out, Vt);

            // ============================================================
            // Phase 6: per-head RMSNorm (in-place on cb_out)
            //
            //   sq = transpose_wh(out)                    [Vt tiles]
            //   ssq = (out @ sq)                          [1 tile, scalar = sum(out^2)]
            //   inv = rsqrt(ssq + Dv * eps)               [1 tile, scalar]
            //   out = out * inv * norm_w                  [Vt tiles, in-place]
            //
            // Uses cb_sq_acc and cb_tmp (both free at this point — popped after
            // Phases 1 and 2). cb_norm_w stayed at front since the start of the
            // kernel; do NOT pop it.
            // ============================================================
            cb_wait_front(cb_out, Vt);

            // 6.1: transpose-WH each tile of cb_out into cb_sq_acc
            cb_reserve_back(cb_sq_acc, Vt);
            for (uint32_t vt = 0; vt < Vt; vt++) {
                tile_regs_acquire();
                transpose_wh_init_short(cb_out);
                transpose_wh_tile(cb_out, vt, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_sq_acc, vt);
                tile_regs_release();
            }
            cb_push_back(cb_sq_acc, Vt);

            // 6.2: ssq = sum_{vt} dot(cb_out[vt], cb_sq_acc[vt]) -> 1 tile in cb_tmp
            cb_wait_front(cb_sq_acc, Vt);
            cb_reserve_back(cb_tmp, 1);
            mm_init(cb_out, cb_sq_acc, cb_tmp);
            tile_regs_acquire();
            for (uint32_t vt = 0; vt < Vt; vt++) {
                matmul_tiles(cb_out, cb_sq_acc, vt, vt, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_tmp);
            tile_regs_release();
            cb_push_back(cb_tmp, 1);
            cb_pop_front(cb_sq_acc, Vt);

            // 6.3: inv = rsqrt(ssq + rms_eps)   -- rms_eps is precomputed Dv * eps
            cb_wait_front(cb_tmp, 1);
            tile_regs_acquire();
            add_tiles_init(cb_tmp, cb_rms_eps);
            add_tiles(cb_tmp, cb_rms_eps, 0, 0, 0);
            rsqrt_tile_init();
            rsqrt_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            cb_pop_front(cb_tmp, 1);
            cb_reserve_back(cb_tmp, 1);
            pack_tile(0, cb_tmp);
            tile_regs_release();
            cb_push_back(cb_tmp, 1);

            // 6.4: cb_out[vt] = cb_out[vt] * inv * cb_norm_w[vt]   (in-place)
            //
            // We rebuild cb_out's contents by popping the original Vt tiles,
            // multiplying by the inv scalar (broadcast) and the per-feature
            // norm_w tile, and re-pushing. cb_norm_w is read but not consumed.
            cb_wait_front(cb_tmp, 1);
            cb_reserve_back(cb_out, Vt);
            for (uint32_t vt = 0; vt < Vt; vt++) {
                // step a: tmp_vt = cb_out[vt] * inv (broadcast scalar)
                tile_regs_acquire();
                mul_tiles_bcast_scalar_init_short(cb_out, cb_tmp);
                mul_tiles_bcast_scalar(cb_out, cb_tmp, vt, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                // step b: result = tmp_vt * cb_norm_w[vt] (elementwise)
                tile_regs_acquire();
                mul_tiles_init(cb_out, cb_norm_w);
                mul_tiles(cb_out, cb_norm_w, vt, vt, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out, vt);
                tile_regs_release();
            }
            cb_pop_front(cb_out, Vt);
            cb_push_back(cb_out, Vt);
            cb_pop_front(cb_tmp, 1);

            // Pop per-token inputs
```

**Caveat to verify in implementation:** the in-place rebuild of `cb_out` (pop-then-re-push) must respect the writer's expectation that `cb_out` holds exactly `Vt` tiles at the front when it pops. Current writer pops `Vt` after `cb_wait_front(cb_out, Vt)`, which is consistent. If you see a deadlock here, the in-place pattern is wrong; switch to a "tmp CB then copy" pattern using `cb_reserve_back` on a different CB and rename.

- [ ] **Step 3.3: Run interim test mode and confirm parity on output**

Run:

```bash
GDN_SEQ_MAJOR_INTERIM=1 pytest models/demos/qwen35_27b/tt/tests/test_gdn_prefill_seq_major.py::test_gdn_prefill_seq_major_correctness -v -s -k "num_tokens=64" 2>&1 | tail -30
```

Expected: `Output PCC: 1.000000` (or ≥ 0.9999) and `State PCC: 1.000000`.

If output PCC is poor (< 0.999):
- Look at the rsqrt/mul/norm_w order in Phase 6.
- Check that `cb_rms_eps` was computed as `Dv * 1e-6` (verify in `gdn.py:172`: `self.rms_eps_tt = _scalar_to_mesh(self.Dv * 1e-6)`).
- If still off, write a print-tile debug version that emits ssq for one pair to confirm sum-of-squares matches the torch reference.

- [ ] **Step 3.4: Commit**

```bash
git add models/demos/qwen35_27b/tt/gdn_kernel/kernels/compute/gdn_prefill_with_norm.cpp \
        models/demos/qwen35_27b/tt/tests/test_gdn_prefill_seq_major.py
git commit -m "$(cat <<'EOF'
feat(qwen35-gdn): fuse per-head RMSNorm into prefill compute kernel

Adds Phase 6 to gdn_prefill_with_norm.cpp: in-place RMSNorm on cb_out
using the existing cb_norm_w (Vt tiles, per-feature weight) and
cb_rms_eps (Dv * 1e-6) inputs that were previously declared but unused.

Validated against (old kernel + ttnn.rms_norm) reference via interim
test mode (GDN_SEQ_MAJOR_INTERIM=1) at num_tokens=64.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: New seq-major writer kernel

**Files:**
- Create: `models/demos/qwen35_27b/tt/gdn_kernel/kernels/dataflow/writer_gdn_prefill_seq_major.cpp`
- Modify: `models/demos/qwen35_27b/tt/gdn_kernel/gdn_kernel_op.py:_build_prefill_seq_major_device_program` (swap writer path + extend compile-time args)

This task moves the new entry point from the old writer's `[num_pairs*seq, 1, Dv]` layout to the new writer's `[1, 1, seq, num_pairs*Dv]` dense layout. After this task the unit test in its default (non-interim) mode should pass.

- [ ] **Step 4.1: Create writer_gdn_prefill_seq_major.cpp**

Create `models/demos/qwen35_27b/tt/gdn_kernel/kernels/dataflow/writer_gdn_prefill_seq_major.cpp` with this content:

```cpp
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Prefill GDN writer kernel — seq-major dense output variant.
//
// Output layout: [1, 1, num_tokens, num_pairs * Dv]   (TILE)
//   In tile coords: Yt = ceil(num_tokens / 32),  Xt = num_pairs * Vt
//   Tile-id at (token_y_tile, pair, vt) =
//      token_y_tile * (num_pairs * Vt) + pair * Vt + vt
//
// State layout: [num_pairs, Dk, Dv]  (unchanged from writer_gdn_prefill.cpp)
//
// The compute kernel pushes Vt tiles per token into cb_out (each tile has its
// 32 rows holding 32 copies of the same Dv slice — the Y dim is logical-1
// padded to 32). This writer L1-buffers up to 32 tokens' Vt-tiles and
// accumulates them into Vt "dense" tiles where row r holds token r's slice.
// After 32 tokens (or end-of-sequence), the dense tiles are NoC-written to
// DRAM at the right (token_y_tile, pair) coordinate.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);    // [1,1,num_tokens, num_pairs*Dv]
    uint32_t state_addr = get_arg_val<uint32_t>(1);  // [num_pairs, Dk, Dv]
    uint32_t pair_start = get_arg_val<uint32_t>(2);
    uint32_t num_pairs_local = get_arg_val<uint32_t>(3);

    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t STATE_IN_L1 = get_compile_time_arg_val(3);
    constexpr uint32_t num_tokens = get_compile_time_arg_val(4);
    constexpr uint32_t num_pairs_total = get_compile_time_arg_val(5);  // NEW vs old writer
    constexpr uint32_t state_tiles = Kt * Vt;
    constexpr uint32_t face_bytes_per_row = 64;  // 32 cols × 2 bytes (BF16)
    constexpr uint32_t row_face_pairs_per_tile = 2; // tile is 2 face-pairs along Y
    constexpr uint32_t num_blocks = (num_tokens + 31) / 32;  // outer Y-tile count

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_state_out = tt::CBIndex::c_8;

    constexpr bool is_dram = true;
    const InterleavedAddrGenFast<is_dram> out_wr = {
        .bank_base_address = out_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};

    constexpr bool state_is_dram = (STATE_IN_L1 == 0);
    const InterleavedAddrGenFast<state_is_dram> state_wr = {
        .bank_base_address = state_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};

    // Static L1 scratch: Vt tiles for assembling one block's dense Y-tiles.
    // 4 tiles × 2 KB = 8 KB. Sized at compile time via a pinned CB slot.
    constexpr uint32_t cb_scratch = tt::CBIndex::c_30;
    cb_reserve_back(cb_scratch, Vt);
    uint32_t scratch_base = get_write_ptr(cb_scratch);

    for (uint32_t pp = 0; pp < num_pairs_local; pp++) {
        uint32_t pair = pair_start + pp;

        for (uint32_t block = 0; block < num_blocks; block++) {
            uint32_t tok_in_block = (block + 1 < num_blocks)
                                        ? 32
                                        : (num_tokens - block * 32);  // 1..32

            // Zero-init the scratch tiles for this block (covers tail-padding).
            uint32_t scratch_bytes = Vt * tile_bytes;
            volatile tt_l1_ptr uint32_t* sptr = (volatile tt_l1_ptr uint32_t*)scratch_base;
            for (uint32_t i = 0; i < scratch_bytes / 4; i++) {
                sptr[i] = 0;
            }

            for (uint32_t r = 0; r < tok_in_block; r++) {
                cb_wait_front(cb_out, Vt);
                uint32_t src_base = get_read_ptr(cb_out);

                // Each cb_out tile is 32x32 in TILE layout (4 faces of 16x16).
                // Row 0 of the logical Y dim sits in face 0/2 row 0 of the tile.
                // We copy that row (32 BF16 = 64 bytes) into row r of the dense
                // scratch tile (which has the same TILE layout). For r in 0..15
                // it goes into face 0; for r in 16..31 it goes into face 2.
                //
                // TILE layout face order (for 32x32 BF16): faces 0,1 are top
                // half (rows 0-15), faces 2,3 are bottom half (rows 16-31).
                // Each face is 16x16 in row-major.
                uint32_t face_offset = (r < 16) ? 0 : (2 * 16 * 16 * 2);  // bytes to bottom-half face 2
                uint32_t row_within_face = (r < 16) ? r : (r - 16);

                for (uint32_t vt = 0; vt < Vt; vt++) {
                    // Source: row 0 of source tile vt's face 0 (left half cols)
                    //         + row 0 of source tile vt's face 1 (right half cols)
                    uint32_t src_tile = src_base + vt * tile_bytes;
                    uint32_t dst_tile = scratch_base + vt * tile_bytes + face_offset;

                    // Face 0 (cols 0-15) row 0 -> dst face row r%16, cols 0-15
                    uint64_t* src_l = (uint64_t*)(src_tile + 0);
                    uint64_t* dst_l = (uint64_t*)(dst_tile + row_within_face * 16 * 2);
                    for (uint32_t i = 0; i < 4; i++) dst_l[i] = src_l[i];

                    // Face 1 (cols 16-31) row 0 -> dst face row r%16, cols 16-31
                    uint64_t* src_r = (uint64_t*)(src_tile + 16 * 16 * 2);
                    uint64_t* dst_r = (uint64_t*)(dst_tile + 16 * 16 * 2 + row_within_face * 16 * 2);
                    for (uint32_t i = 0; i < 4; i++) dst_r[i] = src_r[i];
                }

                cb_pop_front(cb_out, Vt);
            }

            // Write Vt scratch tiles to DRAM at this block's destination column strip.
            uint32_t dst_tile_base =
                block * (num_pairs_total * Vt) + pair * Vt;
            for (uint32_t vt = 0; vt < Vt; vt++) {
                noc_async_write_tile(dst_tile_base + vt, out_wr,
                                     scratch_base + vt * tile_bytes);
            }
            noc_async_write_barrier();
        }

        // State write — unchanged from old writer
        cb_wait_front(cb_state_out, state_tiles);
        uint32_t sp = get_read_ptr(cb_state_out);
        for (uint32_t s = 0; s < state_tiles; s++) {
            noc_async_write_tile(pair * state_tiles + s, state_wr, sp);
            sp += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_state_out, state_tiles);
    }
}
```

**Verify before implementation:** the L1 face-by-face copy pattern above uses the standard TT tile-face layout (4 × 16×16 faces in row-major within a 32×32 BF16 tile). If this assumption is wrong on Blackhole BH200, the per-tile inner loop needs to use the architecture-specific face layout. Check `tt_metal/hw/inc/blackhole/tile.h` (or the equivalent) for the canonical face order before the first build.

**Caveat on `cb_scratch`:** the writer uses `cb_index = 30` as a static L1 scratch region. That CB index isn't currently in `_make_cb` lists. Step 4.2 adds it.

- [ ] **Step 4.2: Wire the new writer + scratch CB into the builder**

In `_build_prefill_seq_major_device_program` (added in Task 2), make three edits:

1. Add a CB descriptor for `cb_scratch` (index 30) sized `Vt` tiles. Append to `cb_descriptors`:

   ```python
       _make_cb(30, Vt, core_ranges),  # cb_scratch (writer L1 repacking, seq-major variant)
   ```

2. Change the writer compile-time args from `[Kt, Vt, BF16_TILE_BYTES, state_l1_flag, num_tokens]` to `[Kt, Vt, BF16_TILE_BYTES, state_l1_flag, num_tokens, num_pairs_total]`:

   ```python
           writer_ct = [
               Kt, Vt, BF16_TILE_BYTES, state_l1_flag, num_tokens, num_pairs_total,
           ]
   ```

3. Change `kernel_source=WRITER_PREFILL_PATH` in the writer KernelDescriptor to `kernel_source=WRITER_PREFILL_SEQ_MAJOR_PATH`:

   ```python
           writer_kd = ttnn.KernelDescriptor(
               kernel_source=WRITER_PREFILL_SEQ_MAJOR_PATH,
               # ... rest unchanged ...
           )
   ```

- [ ] **Step 4.3: Run the unit test (default mode) — expect PCC ≥ 0.999**

Run all three parametrized cases:

```bash
pytest models/demos/qwen35_27b/tt/tests/test_gdn_prefill_seq_major.py::test_gdn_prefill_seq_major_correctness -v -s 2>&1 | tail -50
```

Expected: all three (`num_tokens=64`, `num_tokens=96`, `num_tokens=2048`) pass with `Output PCC ≥ 0.999`, `State PCC ≥ 0.999`.

If output PCC is bad on `num_tokens=2048` only, the bug is in the per-block addressing (off-by-one in `dst_tile_base`). Bisect by adding `num_tokens=128, 256, 1024` as parametrize values temporarily.

If output PCC is bad on `num_tokens=96` (not 64, not 2048), the bug is in the partial-block tail handling at end-of-sequence — verify `tok_in_block` and the zero-init loop.

If output PCC is bad uniformly, the bug is in the face-by-face L1 copy — re-check face layout vs. architecture documentation.

- [ ] **Step 4.4: Commit**

```bash
git add models/demos/qwen35_27b/tt/gdn_kernel/kernels/dataflow/writer_gdn_prefill_seq_major.cpp \
        models/demos/qwen35_27b/tt/gdn_kernel/gdn_kernel_op.py
git commit -m "$(cat <<'EOF'
feat(qwen35-gdn): seq-major writer for fused-norm prefill kernel

Adds writer_gdn_prefill_seq_major.cpp which L1-buffers 32 tokens of
cb_out at a time into Vt dense tiles and writes them into
[1, 1, num_tokens, num_pairs*Dv] DRAM destination. Eliminates the
post-kernel reshape -> permute -> reshape chain in gdn.py.

Wires the new writer + cb_scratch (CB 30, Vt tiles) into
_build_prefill_seq_major_device_program. Unit test at num_tokens
in {64, 96, 2048} passes PCC >= 0.999.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Wire into gdn.py:forward_prefill behind env flag

**Files:**
- Modify: `models/demos/qwen35_27b/tt/gdn.py:forward_prefill` (lines 909-950)

This task adds the `GDN_PREFILL_SEQ_MAJOR` env flag and uses the new entry point when it's set. Default is OFF. The change is contained to one block; the rest of `forward_prefill` is untouched.

- [ ] **Step 5.1: Modify forward_prefill to switch on the env flag**

In `models/demos/qwen35_27b/tt/gdn.py`, find the block from line 909 (where `prefill_output` is allocated) through line 950 (where `out_4d` is deallocated). Replace it with:

```python
        # ---- Allocate output buffer ----
        # Layout depends on which kernel variant we use:
        #   - Legacy (gdn_prefill_fused):     [num_pairs * N, 1, Dv]
        #   - Seq-major (gdn_prefill_fused_seq_major):
        #                                     [1, 1, N, num_pairs * Dv]   (RMSNorm + permute fused in)
        use_seq_major = bool(_os.environ.get("GDN_PREFILL_SEQ_MAJOR", ""))

        if use_seq_major:
            from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import (
                gdn_prefill_fused_seq_major,
            )

            prefill_output = ttnn.from_torch(
                torch.zeros(1, 1, seq_len, num_pairs * Dv, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

            gdn_prefill_fused_seq_major(
                conv_out_3d, a_3d, b_3d,
                self.neg_exp_A, tw["dt_bias"], tw["norm_w"],
                self.scale_tt, self.rms_scale_tt, self.rms_eps_tt,
                self._prefill_rec_states, prefill_output,
                num_pairs=num_pairs, num_tokens=seq_len,
                Nv_TP=Nv_TP, Nk_TP=Nk_TP,
                repeat_factor=repeat_factor, key_dim_tp=key_dim_tp,
            )
            ttnn.deallocate(conv_out_all)
            ttnn.deallocate(a_all)
            ttnn.deallocate(b_all)

            # RMSNorm + reshape + permute + reshape are all folded into the kernel.
            out_f = prefill_output
        else:
            prefill_output = ttnn.from_torch(
                torch.zeros(num_pairs * seq_len, 1, Dv, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

            gdn_prefill_fused(
                conv_out_3d, a_3d, b_3d,
                self.neg_exp_A, tw["dt_bias"], tw["norm_w"],
                self.scale_tt, self.rms_scale_tt, self.rms_eps_tt,
                self._prefill_rec_states, prefill_output,
                num_pairs=num_pairs, num_tokens=seq_len,
                Nv_TP=Nv_TP, Nk_TP=Nk_TP,
                repeat_factor=repeat_factor, key_dim_tp=key_dim_tp,
            )
            ttnn.deallocate(conv_out_all)
            ttnn.deallocate(a_all)
            ttnn.deallocate(b_all)

            # Legacy post-kernel chain: rms_norm + reshape + permute + reshape
            out_n = ttnn.rms_norm(prefill_output, weight=tw["norm_w"], epsilon=1e-6)
            ttnn.deallocate(prefill_output)
            out_4d = ttnn.reshape(out_n, (1, num_pairs, seq_len, Dv))
            ttnn.deallocate(out_n)
            out_4d = ttnn.permute(out_4d, (0, 2, 1, 3))
            out_f = ttnn.reshape(out_4d, (1, 1, seq_len, self.value_dim_tp))
            ttnn.deallocate(out_4d)
```

(Note: `gdn_prefill_fused` and `_os` are already imported at the top of the file: `from models.demos.qwen35_27b.tt.gdn_kernel.gdn_kernel_op import gdn_prefill_fused, ...` and `import os as _os`.)

- [ ] **Step 5.2: Confirm legacy path still works (flag unset)**

Run a focused module-level test (existing one, no need to add a new one):

```bash
pytest models/demos/qwen35_27b/tt/tests/test_gdn.py -v -s -k "prefill" 2>&1 | tail -30
```

Expected: all GDN prefill tests pass identically to before this task (`GDN_PREFILL_SEQ_MAJOR` is unset by default, so the `else` branch runs unchanged).

If a test fails with the flag unset, you've accidentally broken the legacy path — re-read the diff and check that the unflagged `else:` branch is byte-identical in behaviour to what was there before.

- [ ] **Step 5.3: Confirm new path works (flag set)**

```bash
GDN_PREFILL_SEQ_MAJOR=1 pytest models/demos/qwen35_27b/tt/tests/test_gdn.py -v -s -k "prefill" 2>&1 | tail -30
```

Expected: same tests pass, with PCC ≥ 0.99 vs reference (the same threshold the test uses today).

If the module-level PCC drops below 0.99 on the new path but the kernel-level unit test (Task 4) passes at ≥ 0.999, the divergence comes from accumulator-width differences between the in-kernel RMSNorm and `ttnn.rms_norm`. Document the new PCC and adjust the threshold ONLY if the e2e test (next task) also passes — never lower a threshold without a working e2e check.

- [ ] **Step 5.4: Commit**

```bash
git add models/demos/qwen35_27b/tt/gdn.py
git commit -m "$(cat <<'EOF'
feat(qwen35-gdn): wire seq-major prefill kernel into forward_prefill

Adds GDN_PREFILL_SEQ_MAJOR env flag (default off). When set, calls
gdn_prefill_fused_seq_major with output shape [1, 1, seq, num_pairs*Dv]
and skips the post-kernel rms_norm + reshape + permute + reshape chain
(everything is now inside the kernel).

Legacy path is preserved verbatim when the flag is unset.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: End-to-end + profile validation

**Files:** No code changes — this task verifies the spec's acceptance criteria using the open `test_e2e_generate.py`.

- [ ] **Step 6.1: Run the e2e generate test with the flag set**

```bash
GDN_PREFILL_SEQ_MAJOR=1 pytest models/demos/qwen35_27b/tt/tests/test_e2e_generate.py -v -s 2>&1 | tee /tmp/e2e_seq_major.log
```

Expected: same generated tokens as the baseline run (without the flag). Compare token-by-token against a baseline run captured beforehand:

```bash
pytest models/demos/qwen35_27b/tt/tests/test_e2e_generate.py -v -s 2>&1 | tee /tmp/e2e_baseline.log
diff <(grep "GENERATED:" /tmp/e2e_baseline.log) <(grep "GENERATED:" /tmp/e2e_seq_major.log)
```

Expected: empty diff (or only differences in timing-related log lines, not token IDs).

If tokens diverge, capture the divergence point (first token that differs and the prompt) and bisect — the most likely cause is residual numerical drift across the 48 GDN layers from the RMSNorm accumulator-width difference. Mitigation if needed: change the in-kernel RMSNorm to keep the rsqrt result in fp32 before the elementwise multiply.

- [ ] **Step 6.2: Capture a profile of the new path**

Re-run the e2e test under the profiler. The exact incantation depends on the project's profiler setup; in this branch the convention has been:

```bash
GDN_PREFILL_SEQ_MAJOR=1 \
TT_METAL_DEVICE_PROFILER=1 \
pytest models/demos/qwen35_27b/tt/tests/test_e2e_generate.py -v -s 2>&1 | tail -30
```

Locate the new profile dir under `generated/profiler/reports/<timestamp>/` and run the existing analysis script:

```bash
NEW_DIR=$(ls -td generated/profiler/reports/*/ | head -1)
mkdir -p ${NEW_DIR}analysis
python3 .claude/skills/profiler-report-analysis/scripts/process_profile.py \
    --csv ${NEW_DIR}ops_perf_results_*.csv \
    --output-dir ${NEW_DIR}analysis \
    --sdpas-per-iteration 1
```

- [ ] **Step 6.3: Verify the spec's acceptance criteria from the new profile**

In `${NEW_DIR}analysis/prefill.csv`:

1. **No `ReshapeViewDeviceOperation` rows** between the GDN `GenericOpDeviceOperation` and the silu·z `BinaryNgDeviceOperation`. Quick check:

   ```bash
   awk -F',' 'NR>1 && $1 ~ /Generic|Reshape|BinaryNg/ {print NR, $1}' ${NEW_DIR}analysis/prefill.csv | head -50
   ```

   You should see `GenericOpDeviceOperation` followed directly by `BinaryNgDeviceOperation` (silu*z), with no `ReshapeViewDeviceOperation` between them.

2. **No post-GDN `LayerNormDeviceOperation`.** The only LayerNorm rows that should remain are the pre/post-AG distributed-norm pairs.

   ```bash
   awk -F',' 'NR>1 && $1=="LayerNormDeviceOperation" {print NR, $1}' ${NEW_DIR}analysis/prefill.csv | wc -l
   ```

   Expected count: drops from current ~10 to ~0 of the small-`Y_PAD` post-GDN ones (the `LayerNormPreAllGather`/`LayerNormPostAllGather` rows are unchanged).

3. **Per-prefill kernel-time drop ≥ 8 ms.** Compare `summary.json → prefill.per_op` total kernel time vs. the baseline `2026_04_28_00_27_47` profile:

   ```bash
   python3 -c "
   import json
   new = json.load(open('${NEW_DIR}analysis/summary.json'.format(NEW_DIR='${NEW_DIR}')))
   old = json.load(open('generated/profiler/reports/2026_04_28_00_27_47/analysis/summary.json'))
   def total(p): return sum(v['kernel_total_ns'] for v in p['prefill']['per_op'].values()) / 1e6
   print(f'Old prefill: {total(old):.2f} ms; New: {total(new):.2f} ms; Drop: {total(old)-total(new):.2f} ms')
   "
   ```

   Expected: drop of ≥ 8 ms. Target ~10 ms for the 8 profiled layers, which extrapolates to ~80 ms for the full 64-layer model.

- [ ] **Step 6.4: If acceptance criteria pass, flip the env flag default**

In `models/demos/qwen35_27b/tt/gdn.py`, change:

```python
        use_seq_major = bool(_os.environ.get("GDN_PREFILL_SEQ_MAJOR", ""))
```

to:

```python
        # Default ON after correctness + perf validation; opt-out via env=0.
        use_seq_major = _os.environ.get("GDN_PREFILL_SEQ_MAJOR", "1") not in ("", "0", "false", "False")
```

Re-run the module-level test once with the flag explicitly unset to confirm the legacy path still works:

```bash
GDN_PREFILL_SEQ_MAJOR=0 pytest models/demos/qwen35_27b/tt/tests/test_gdn.py -v -s -k "prefill" 2>&1 | tail -10
```

- [ ] **Step 6.5: Commit and write up**

```bash
git add models/demos/qwen35_27b/tt/gdn.py
git commit -m "$(cat <<'EOF'
perf(qwen35-gdn): make seq-major prefill the default

Validated against e2e generate (token-exact match) and profiler-driven
acceptance criteria from the spec:
- No reshape pile between GDN kernel and silu*z multiply
- No post-GDN LayerNormDeviceOperation
- Per-prefill kernel-time drop of ~10 ms profiled (~80 ms full pass)

Legacy path preserved behind GDN_PREFILL_SEQ_MAJOR=0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Capture the before/after profile diff in a short note (does not need to be committed) so the next reviewer can see the win:

```
Before: <total non-GDN prefill ms from 2026_04_28_00_27_47>
After:  <total non-GDN prefill ms from new run>
Saving: <delta ms> profiled / ~8x extrapolated to full pass
```

---

## Self-Review

Spec coverage:
- "Compute kernel change" → Task 3 (the RMSNorm phase block).
- "Kernel writer change" → Task 4 (writer_gdn_prefill_seq_major.cpp + builder swap).
- "Host op wrapper" (`gdn_prefill_fused_seq_major`) → Task 2.
- "Host model changes" (env-flagged switch in `gdn.py:forward_prefill`) → Task 5.
- "Numerics + correctness" → Task 1 test + Task 3 interim mode + Task 4 final mode.
- "Testing" (kernel-level + module + e2e + profile) → Tasks 1, 5, 6.
- "Risk + rollback" → Task 5 env flag + Task 6 default-flip after validation.
- "Files touched" → matches the file list in this plan.
- "Acceptance criteria" → Task 6.3 verifies each one.

No spec gaps. The plan adds one staging detail (interim test mode in Task 3) that the spec didn't call out — that's an implementation aid, not a scope addition.

Type / API consistency: `gdn_prefill_fused_seq_major(...)` keeps the exact arg list of `gdn_prefill_fused(...)` except the docstring notes the changed `output` shape. The builder name `_build_prefill_seq_major_device_program` and the path constants `COMPUTE_PREFILL_WITH_NORM_PATH` / `WRITER_PREFILL_SEQ_MAJOR_PATH` are used consistently across all tasks.
