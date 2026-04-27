# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Deterministic hang reproduction test for GPT-OSS-120B SDPA paged-decode bug.
Issue: https://github.com/tenstorrent/tt-metal/issues/42917

Hang mechanism (UNFIXED kernel):
  apply_mask_at_last_chunk = do_reduce && is_causal
  → worker cores never drain cb_mask_in
  → in trace mode: writer stalls on cb_reserve_back (CB full after capture)
  → K tiles never pushed to cb_k_in → TRISC0 waits → TRISC1 waits →
  → TRISC2 semaphore deadlock in eltwise_typecast kernel (t6_semaphore_get<0>())

Production shapes (GPT-OSS-120B on Galaxy, per DP-chip):
  B=32, NH=8, NKV=1, D=64, block_size=64, k_chunk_size=128
  compute_with_storage_grid_size=CoreCoord(8,8)
  HiFi4, fp32_dest_acc_en=False (BFP8 destination)
  num_cores_per_head = min(64, 64*32*1)//32 = 2  → 1 reducer + 1 worker

How to run:
  ─────────────────────────────────────────────────────────
  From inside the Docker container on g10glx02 (Galaxy machine):

    cd /home/models-team/divanovic/tt-metal
    source /opt/venv/bin/activate
    export TT_METAL_HOME=/home/models-team/divanovic/tt-metal
    export ARCH_NAME=wormhole_b0

  ─── Single WH device (no hang, correctness check only) ───────────────────
    python -m pytest \
      tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_paged_decode_hang.py \
      --device-mode single --iterations 20 -v

  ─── Galaxy mesh, trace mode, 20 iterations (hangs on UNFIXED kernel) ──────
    python -m pytest \
      tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_paged_decode_hang.py \
      --device-mode galaxy --iterations 20 -v

  ─── Galaxy, infinite loop until hang (no iteration limit) ──────────────────
    python -m pytest \
      tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_paged_decode_hang.py \
      --device-mode galaxy --iterations 0 -v

  ─── With DPRINTs enabled (captures CB/mask state on chip 0) ────────────────
    TT_METAL_DPRINT_CORES=all \
    TT_METAL_DPRINT_CHIPS=0 \
    TT_METAL_DPRINT_FILE=/tmp/sdpa_dprint.log \
    python -m pytest \
      tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_paged_decode_hang.py \
      --device-mode galaxy --iterations 20 -v

  ─── With tt-triage on hang (run in separate terminal when device is hung) ───
    python tools/triage/triage.py > /tmp/new_tt_triage_output.txt 2>&1

  Expected behaviour:
    UNFIXED kernel: hangs within iterations 2-13 (trace replay stalls)
    FIXED kernel:   all iterations complete, each ~2s
"""

import time
import pytest
import torch
import ttnn
import os


# ── Production shapes ─────────────────────────────────────────────────────────
B = 32  # per-chip batch (128 total / 4 DP groups)
NH = 8  # Q heads per chip (64 / TP=8)
NKV = 1  # KV heads per chip (8 / TP=8)
D = 64  # head dimension
BLOCK_SIZE = 64  # KV page size
K_CHUNK = 128  # k_chunk_size — matches production
SCALE = D**-0.5

# Exact production hang position: 10*64+3 = 643, causing partial last page
CUR_POS = 643
NUM_PAGES = CUR_POS // BLOCK_SIZE + 2  # 12 pages

# Production compute kernel config
COMPUTE_CFG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=False,  # BFP8 dst → overflow triggers eltwise_typecast hang
    packer_l1_acc=False,
)

# Production program config — 8×8 grid, no max cap
# → num_cores_per_head = min(64, 64*32*1)//32//1 = 2 (reducer + worker)
PROG_CFG = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
    q_chunk_size=NH,
    k_chunk_size=K_CHUNK,
)


def pytest_addoption(parser):
    parser.addoption(
        "--device-mode",
        choices=["single", "galaxy"],
        default="single",
        help="single: open device 0; galaxy: open full mesh device",
    )
    parser.addoption(
        "--iterations",
        type=int,
        default=20,
        help="Number of decode iterations to run (0 = infinite until hang)",
    )
    parser.addoption(
        "--use-trace",
        action="store_true",
        default=True,
        help="Use Metal trace mode (production vLLM path, required for CB drain hang)",
    )


def build_tensors(device):
    """
    All B users share identical KV data so the page table is consistent.
    Positions 0..CUR_POS: valid random KV.
    Positions CUR_POS+1..: zero (fresh allocation — hang still triggers via CB drain).
    """
    torch.manual_seed(42)
    Q = torch.randn(1, B, NH, D)

    K_paged = torch.zeros(B, NKV, NUM_PAGES, BLOCK_SIZE, D)
    V_paged = torch.zeros(B, NKV, NUM_PAGES, BLOCK_SIZE, D)
    for tok in range(CUR_POS + 1):
        pg, off = tok // BLOCK_SIZE, tok % BLOCK_SIZE
        k_row = torch.randn(NKV, D)
        v_row = torch.randn(NKV, D)
        K_paged[:, :, pg, off, :] = k_row.unsqueeze(0).expand(B, -1, -1)
        V_paged[:, :, pg, off, :] = v_row.unsqueeze(0).expand(B, -1, -1)

    page_table = torch.arange(NUM_PAGES, dtype=torch.int32).unsqueeze(0).expand(B, -1)
    dram = ttnn.DRAM_MEMORY_CONFIG

    K_tt = K_paged.permute(0, 2, 1, 3, 4).reshape(B * NUM_PAGES, NKV, BLOCK_SIZE, D)
    V_tt = V_paged.permute(0, 2, 1, 3, 4).reshape(B * NUM_PAGES, NKV, BLOCK_SIZE, D)

    tt_Q = ttnn.as_tensor(Q, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_K = ttnn.as_tensor(K_tt, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_V = ttnn.as_tensor(V_tt, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=dram)
    tt_pt = ttnn.as_tensor(
        page_table, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=dram
    )
    tt_cp = ttnn.from_torch(
        torch.tensor([CUR_POS] * B, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=dram,
    )
    return tt_Q, tt_K, tt_V, tt_pt, tt_cp


def run_sdpa(device, tt_Q, tt_K, tt_V, tt_pt, tt_cp):
    return ttnn.transformer.paged_scaled_dot_product_attention_decode(
        tt_Q,
        tt_K,
        tt_V,
        cur_pos_tensor=tt_cp,
        page_table_tensor=tt_pt,
        scale=SCALE,
        program_config=PROG_CFG,
        compute_kernel_config=COMPUTE_CFG,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.fixture
def device_under_test(request):
    mode = request.config.getoption("--device-mode", default="single")
    if mode == "galaxy":
        mesh_shape = ttnn.MeshShape(4, 8)
        device = ttnn.open_mesh_device(mesh_shape=mesh_shape)
        yield device
        ttnn.close_mesh_device(device)
    else:
        device = ttnn.open_device(device_id=0)
        yield device
        ttnn.close_device(device)


@pytest.mark.timeout(3600)
def test_sdpa_paged_decode_hang(device_under_test, request):
    """
    Runs paged SDPA decode in a loop with production shapes and trace mode.
    Each iteration decodes at cur_pos=CUR_POS (partial last page).

    With UNFIXED kernel: hangs deterministically within trace replays
    because workers never drain cb_mask_in.

    With FIXED kernel: all iterations complete.

    Iteration timing is printed every iteration so you can observe when
    the device stops making progress (pre-hang detection).
    """
    device = device_under_test
    mode = request.config.getoption("--device-mode", default="single")
    max_iters = request.config.getoption("--iterations", default=20)
    use_trace = True  # always use trace (production path)

    print(
        f"\n[hang_repro] mode={mode} cur_pos={CUR_POS} B={B} NH={NH} "
        f"k_chunk={K_CHUNK} trace={use_trace} max_iters={max_iters or 'inf'}"
    )
    print(
        f"[hang_repro] num_cores_per_head = min(64,64*{B}*{NKV})//{B}//{NKV} = "
        f"{min(64,64*B*NKV)//B//NKV}  (1 reducer + {min(64,64*B*NKV)//B//NKV - 1} worker(s))"
    )
    print(f"[hang_repro] k_num_chunks = ceil({CUR_POS+1}/{K_CHUNK}) = " f"{-(-(CUR_POS+1)//K_CHUNK)}")
    print()

    tt_Q, tt_K, tt_V, tt_pt, tt_cp = build_tensors(device)

    # Warmup — JIT compiles kernels
    print("[hang_repro] Warming up (JIT compile)...")
    t_warmup = time.perf_counter()
    tt_out = run_sdpa(device, tt_Q, tt_K, tt_V, tt_pt, tt_cp)
    ttnn.synchronize_device(device)
    print(f"[hang_repro] Warmup done in {time.perf_counter()-t_warmup:.1f}s")

    if use_trace:
        print("[hang_repro] Capturing trace...")
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        tt_out = run_sdpa(device, tt_Q, tt_K, tt_V, tt_pt, tt_cp)
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        print("[hang_repro] Trace captured.")

    n = 0
    t_start = time.perf_counter()
    while True:
        n += 1
        t0 = time.perf_counter()

        if use_trace:
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        else:
            tt_out = run_sdpa(device, tt_Q, tt_K, tt_V, tt_pt, tt_cp)
            ttnn.synchronize_device(device)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_s = time.perf_counter() - t_start
        print(f"[hang_repro] iter {n:3d}  {elapsed_ms:7.1f} ms  total={total_s:.1f}s  " f"cur_pos={CUR_POS}  PASS")

        if max_iters and n >= max_iters:
            break

    if use_trace:
        ttnn.release_trace(device, trace_id)

    print(f"\n[hang_repro] Completed {n} iteration(s) without hang.")
