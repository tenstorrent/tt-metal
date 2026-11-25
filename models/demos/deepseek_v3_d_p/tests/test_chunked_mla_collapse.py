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
ETH_COLS = 1024

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


class EthLoad:
    """Full-mesh fabric all-gathers on every Tensix core but the SDPA ones, so the ethernet links carry traffic while
    SDPA reads K. A program waits only for earlier programs of its own sub-device, so each SDPA launch runs next to the
    last gather queued before it."""

    def __init__(self, mesh, sdpa_cores, mb, gathers):
        grid = mesh.compute_with_storage_grid_size()
        x0, x1 = sdpa_cores
        assert x1 < grid.x - 1, f"SDPA cores {sdpa_cores} leave no gather cores in row 0 of a {grid.x}x{grid.y} grid"
        self.mesh = mesh
        self.gathers = gathers
        self.sdpa_sd, self.ccl_sd = ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)
        sdpa_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, 0), ttnn.CoreCoord(x1, 0))])
        self.ccl_grid = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(x1 + 1, 0), ttnn.CoreCoord(grid.x - 1, 0)),
                ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(grid.x - 1, grid.y - 1)),
            ]
        )
        self.manager = mesh.create_sub_device_manager([ttnn.SubDevice([sdpa_grid]), ttnn.SubDevice([self.ccl_grid])], 0)
        mesh.load_sub_device_manager(self.manager)
        mesh.set_sub_device_stall_group([self.sdpa_sd])
        self.rows = mb * 1024 * 1024 // (2 * ETH_COLS) // 32 * 32
        self.mb = mb
        self.src = self.outs = None

    def allocate(self):
        """Called after K is placed, so the gather buffers never shift K off its address."""
        mesh = self.mesh
        self.src = ttnn.from_torch(
            torch.randn(1, 1, self.rows, ETH_COLS),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        shape = (mesh.shape[0], mesh.shape[1])
        self.outs = [
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, self.rows * shape[axis], ETH_COLS]),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                mesh,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            for axis in range(len(shape))
        ]
        self.issue()
        t0 = time.perf_counter()
        for _ in range(4):
            self.issue()
        self.sync()
        dt = (time.perf_counter() - t0) / 4
        moved = sum(self.mb * (n - 1) for n in shape)
        logger.info(
            f"eth load: {self.gathers} gathers/iter over axes {shape} of {self.mb} MB/chip, "
            f"{dt * 1000:.0f} ms per {len(shape)} gathers ({moved / 1024 / dt:.1f} GB/s received per chip)"
        )

    def issue(self):
        for g in range(self.gathers):
            axis = g % len(self.outs)
            ttnn.all_gather(
                self.src,
                2,
                cluster_axis=axis,
                output_tensor=self.outs[axis],
                subdevice_id=self.ccl_sd,
                sub_core_grids=self.ccl_grid,
            )

    def sync(self):
        ttnn.synchronize_device(self.mesh, sub_device_ids=[self.sdpa_sd, self.ccl_sd])

    def close(self):
        self.sync()
        for t in [self.src, *(self.outs or [])]:
            if t is not None:
                ttnn.deallocate(t)
        self.mesh.reset_sub_device_stall_group()
        self.mesh.clear_loaded_sub_device_manager()
        self.mesh.remove_sub_device_manager(self.manager)


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
    # REPRO_ETH=1 runs the op on every chip of the fabric mesh next to all-gathers, and checks chip `chip`.
    eth = os.environ.get("REPRO_ETH", "0") == "1"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if triage:
        run_dram_triage(chip, "before_open")
    load = None
    if eth:
        ttnn.set_fabric_config(getattr(ttnn.FabricConfig, os.environ.get("REPRO_FABRIC", "FABRIC_2D")))
        rows, cols = (int(v) for v in os.environ.get("REPRO_MESH", "8x4").split("x"))
        device = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
        device_ids = list(device.get_device_ids())
        mesh_idx = device_ids.index(chip)
        logger.info(f"mesh {rows}x{cols}, chip {chip} at mesh index {mesh_idx}, device ids {device_ids}")
    else:
        device = ttnn.open_device(device_id=chip)
    hit = None

    def to_dev(t, **kwargs):
        if eth:
            kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
        return ttnn.from_torch(t, device=device, **kwargs)

    def to_host(t):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[mesh_idx] if eth else t)

    try:
        if eth:
            load = EthLoad(
                device, (0, 2), int(os.environ.get("REPRO_ETH_MB", 128)), int(os.environ.get("REPRO_ETH_GATHERS", 4))
            )
        torch.manual_seed(0)
        tt_q = to_dev(
            torch.randn(1, HEADS, Q_LEN, KVPE_DIM),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_page_table = to_dev(
            torch.zeros(1, NUM_BLOCKS, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        probe = to_dev(
            torch.zeros(1, BANKS * PAD_PAGE_BYTES // 4, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
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
        tt_k = to_dev(
            torch.randn(1, 1, BLOCK_SIZE, KVPE_DIM),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logger.info(f"chip {chip}: single K block at {hex(tt_k.buffer_address())}")
        assert pad is None or tt_k.buffer_address() == k_addr, f"K landed at {hex(tt_k.buffer_address())}"
        # Host readback goes over PCIe, not the Tensix NoC read path the op uses, so it shows what DRAM actually holds.
        k_ref = to_host(tt_k)
        q_ref = to_host(tt_q)
        if load:
            load.allocate()
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
            if load:
                load.issue()
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
            host = to_host(out)
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
                logger.info(diff_summary("K in DRAM vs initial readback", k_ref, to_host(tt_k)))
                logger.info(diff_summary("Q in DRAM vs initial readback", q_ref, to_host(tt_q)))
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
        if eth:
            if load:
                load.close()
            ttnn.close_mesh_device(device)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        else:
            ttnn.close_device(device)
    if triage:
        run_dram_triage(chip, "after_close")

    assert hit is None, f"chunked_flash_mla_prefill output diverged at iteration {hit}"
