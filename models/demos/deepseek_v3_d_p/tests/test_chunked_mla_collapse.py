# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import subprocess
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

HEADS = 8
Q_LEN = 640
KVPE_DIM = 576
KV_LORA_RANK = 512
NUM_BLOCKS = 4000
BLOCK_SIZE = 64
CHUNK_START = 254080
SCALE = 0.14467962580268923
BANKS = 8
PAD_PAGE_BYTES = 64
MAX_ABS = 1e6

REPO = Path(__file__).resolve().parents[4]
OUT_DIR = Path(os.environ.get("REPRO_OUT_DIR", REPO / "repro_logs" / "hits"))
# Triage pins its own tt-umd/tt-exalens wheels, so it runs from a separate venv to keep them off the ttnn import path.
TRIAGE_PYTHON = os.environ.get("REPRO_TRIAGE_PYTHON", str(REPO / "triage_env" / "bin" / "python"))


def run_dram_triage(chip, tag):
    out = OUT_DIR / f"triage_{tag}.csv"
    cmd = [
        TRIAGE_PYTHON,
        str(REPO / "tools" / "tt-triage.py"),
        "--run=device_info",
        "--run=check_dram_health",
        f"--dev={chip}",
        "--llm-output",
        f"--llm-output-path={out}",
    ]
    try:
        res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        logger.error(f"triage {tag}: timed out")
        return
    (OUT_DIR / f"triage_{tag}.log").write_text(res.stdout + res.stderr)
    nonzero = []
    if out.exists():
        section = out.read_text().split("check_dram_health.py:")[-1].strip().split("\n\n")[0]
        nonzero = [r for r in section.splitlines()[1:] if any(v != "0" for v in r.split(",")[3:7])]
    logger.info(f"triage {tag}: exit {res.returncode}, {len(nonzero)} GDDR rows with nonzero EDC -> {out}")
    for r in nonzero:
        logger.warning(f"triage {tag}: {r}")


def bits(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def diff_summary(name, ref, now):
    neq = (bits(ref) != bits(now)).nonzero()
    if not neq.numel():
        return f"{name}: bit-identical"
    return f"{name}: {neq.shape[0]} elements differ, first at {neq[0].tolist()}, last at {neq[-1].tolist()}"


@pytest.mark.timeout(0)
def test_chunked_mla_collapse():
    chip = int(os.environ.get("REPRO_CHIP", 26))
    k_addr = int(os.environ.get("REPRO_K_ADDR", "0x4003880"), 0)
    iters = int(os.environ.get("REPRO_ITERS", 1000))
    post_hit_iters = int(os.environ.get("REPRO_POST_HIT_ITERS", 20))
    triage = os.environ.get("REPRO_TRIAGE", "1") == "1"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if triage:
        run_dram_triage(chip, "before_open")
    device = ttnn.open_device(device_id=chip)
    hit = None
    try:
        torch.manual_seed(0)
        tt_q = ttnn.from_torch(
            torch.randn(1, HEADS, Q_LEN, KVPE_DIM),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_page_table = ttnn.from_torch(
            torch.zeros(1, NUM_BLOCKS, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        probe = ttnn.from_torch(
            torch.zeros(1, BANKS * PAD_PAGE_BYTES // 4, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        free_addr = probe.buffer_address()
        ttnn.deallocate(probe)
        pad = None
        if k_addr > free_addr:
            pad = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, (k_addr - free_addr) // PAD_PAGE_BYTES * BANKS, PAD_PAGE_BYTES // 4]),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
                device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
        tt_k = ttnn.from_torch(
            torch.randn(1, 1, BLOCK_SIZE, KVPE_DIM),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logger.info(f"chip {chip}: single K block at {hex(tt_k.buffer_address())}")
        assert pad is None or tt_k.buffer_address() == k_addr, f"K landed at {hex(tt_k.buffer_address())}"
        # Host readback goes over PCIe, not the Tensix NoC read path the op uses, so it shows what DRAM actually holds.
        k_ref = ttnn.to_torch(tt_k)
        q_ref = ttnn.to_torch(tt_q)
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(3, 1),
            q_chunk_size=32,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        def run_op():
            out = ttnn.transformer.chunked_flash_mla_prefill(
                tt_q,
                tt_k,
                KV_LORA_RANK,
                tt_page_table,
                chunk_start_idx=CHUNK_START,
                scale=SCALE,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            host = ttnn.to_torch(out)
            ttnn.deallocate(out)
            return host

        out_ref = run_op()
        assert out_ref.float().abs().max() < MAX_ABS, "first output already blown up"

        t0 = time.perf_counter()
        for it in range(iters):
            out = run_op()
            rows = out.float().abs().reshape(-1, KV_LORA_RANK).amax(dim=-1)
            bad = (rows > MAX_ABS).nonzero().flatten()
            mismatch = not torch.equal(bits(out), bits(out_ref))
            if bad.numel() or mismatch:
                hit = it
                diff_rows = (bits(out) != bits(out_ref)).reshape(-1, KV_LORA_RANK).any(-1)
                diff_rows = diff_rows.nonzero().flatten()
                logger.error(
                    f"HIT iter {it} at {time.perf_counter() - t0:.0f}s: {bad.numel()} rows > {MAX_ABS:g}, "
                    f"{diff_rows.numel()} rows differ from iter-0 output "
                    f"(rows {diff_rows[0].item()}..{diff_rows[-1].item()}), max {rows.max().item():.4g}"
                )
                # Device stays open here so the checks see the state right after the bad read.
                logger.info(diff_summary("K in DRAM vs initial readback", k_ref, ttnn.to_torch(tt_k)))
                logger.info(diff_summary("Q in DRAM vs initial readback", q_ref, ttnn.to_torch(tt_q)))
                if triage:
                    run_dram_triage(chip, f"hit_iter{it}_live")
                post = [torch.equal(bits(run_op()), bits(out_ref)) for _ in range(post_hit_iters)]
                logger.info(f"post-hit reruns matching iter-0 output: {sum(post)}/{post_hit_iters} {post}")
                torch.save(
                    {"out_ref": out_ref, "out_bad": out, "k_ref": k_ref, "diff_rows": diff_rows},
                    OUT_DIR / f"hit_chip{chip}_iter{it}.pt",
                )
                break
            if (it + 1) % 200 == 0:
                logger.info(f"{it + 1}/{iters} iters clean at {time.perf_counter() - t0:.0f}s")
        if pad is not None:
            ttnn.deallocate(pad)
    finally:
        ttnn.close_device(device)
    if triage:
        run_dram_triage(chip, "after_close")

    assert hit is None, f"chunked_flash_mla_prefill output diverged at iteration {hit}"
