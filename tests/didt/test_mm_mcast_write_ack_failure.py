# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Screen every chip of a Blackhole galaxy for a multicast write-acknowledgement failure.

A heavy multicast matmul -- the MLA wkv_b1 projection from DeepSeek prefill, [1,64,5120,128] @
[1,64,128,512] over the full compute grid -- runs on one chip at a time. A healthy chip finishes in
seconds. On a defective one the in1 multicast sender parks in noc_async_write_barrier waiting for a
write acknowledgement that never arrives, its receivers block behind it, and the op never retires.

Such a chip is otherwise invisible: firmware and kernel images are byte-identical, GDDR BIST passes,
and run-to-run determinism is clean. Only a real multicast matmul exposes it. One chip at a time,
because on a whole mesh a wedged chip stalls its neighbours and the runtime blames whichever it
noticed first -- so this names the offending card outright instead.

Not di/dt, despite tests/didt: the failure did NOT react to throttling. It reproduced at every
TT_MM_THROTTLE_PERF level 0 through 5 (100% down to 33% of issue rate) and all six hung; it also
survives tt-smi -glx_reset_auto. tests/didt is simply where single-op hardware screens live.

Run this. Reset first, always -- and note the sweep is always a cold-cache run:

    tt-smi -glx_reset_auto
    TT_METAL_HOME=$PWD python_env/bin/python tests/didt/test_mm_mcast_write_ack_failure.py

Exits 1 if any chip hangs. Logs: /tmp/mm_mcast_write_ack_failure/chip<N>/pytest.log. Chips report as
they finish, so output is in completion order. Reset between runs too: this gate detects a hang but
cannot clear it, so a still-wedged chip makes its neighbours fail and look faulty.

One chip, cold:

    rm -rf /tmp/c19
    TT_METAL_HOME=$PWD TT_METAL_CACHE=/tmp/c19 TT_VISIBLE_DEVICES=19 \
      python_env/bin/pytest -svv --timeout=0 tests/didt/test_mm_mcast_write_ack_failure.py

The rm -rf is what makes it cold; a reused TT_METAL_CACHE silently gives a warm run. The same
command without it, run twice, is warm. Confirm which you got with:

    grep "JIT cache stats" <log>      # 0/30 hits = cold, 30/30 hits = warm

Knobs:

    TT_METAL_HOME=$PWD python_env/bin/python -c "
    from tests.didt.test_mm_mcast_write_ack_failure import scan_all_chips
    scan_all_chips(iters=200, watchdog=30)"

    iters     MATMUL_ITERS per chip, default 20 (~25ms each, so more soak is nearly free)
    watchdog  TT_METAL_OPERATION_TIMEOUT_SECONDS per chip in seconds, default 30
    work_dir  holds each chip's cache and pytest.log

These env vars override the shape and grid: MATMUL_HEADS (default 64), MATMUL_SEQ (default 5120),
MATMUL_GRID (default is the compute grid less its dispatch column, so 11x10 on a Blackhole galaxy).

The test arms the dispatch watchdog (TT_METAL_OPERATION_TIMEOUT_SECONDS) and auto-triage itself
before any device opens; exported values win. A
hang names the card by BusDeviceFunction, /dev/tenstorrent minor, tray:slot and ASIC_ID, since chip
ids are only sorted positions and TT_VISIBLE_DEVICES renumbers them.
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_wormhole_b0

# In the PCIe bus byte the high nibble selects the tray and the low nibble the slot within it,
# matching the T<tray>:N<slot> label tt-smi prints for the card.
TRAY_OF_BUS_NIBBLE = {0x00: 1, 0x40: 2, 0xC0: 3, 0x80: 4}
WATCHDOG_ENV = "TT_METAL_OPERATION_TIMEOUT_SECONDS"
TRIAGE_ENV = "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"


def device_params_for(fabric_config):
    return {
        "fabric_config": fabric_config,
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        "l1_small_size": 1152,
    }


@pytest.fixture(scope="session", autouse=True)
def arm_hang_watchdog():
    """Arm the dispatch watchdog (TT_METAL_OPERATION_TIMEOUT_SECONDS, the op-to-op no-progress gap)
    and auto-triage exactly as CI does
    (.github/actions/setup-job/action.yml), before the function-scoped mesh_device opens a device --
    the runtime reads these at device open, and without them a hang blocks forever unreported.
    Anything already exported wins, so a runner or CI keeps control."""
    out_dir = Path(os.environ.get("OUT_DIR", "generated/mm_mcast_write_ack_failure"))
    out_dir.mkdir(parents=True, exist_ok=True)
    sentinel = out_dir / ".triaged"
    sentinel.unlink(missing_ok=True)
    triage = (
        f"{sys.executable} {os.environ.get('TT_METAL_HOME', Path.cwd())}/tools/triage/triage.py --disable-progress "
        "--run=dump_running_operations --run=dump_op_window --run=dump_callstacks --llm-output"
    )
    os.environ.setdefault(WATCHDOG_ENV, "30")
    os.environ.setdefault("TT_TRIAGE_ENABLE_AGGREGATED_CALLSTACKS", "1")
    # First hang only: triage halts the cores, so anything running after it is on a poisoned device.
    os.environ.setdefault(
        TRIAGE_ENV,
        f"if [ -e {sentinel} ]; then exit 0; fi; touch {sentinel}; {triage} > {out_dir}/triage.log 2>&1",
    )
    logger.info(f"dispatch watchdog: {WATCHDOG_ENV}={os.environ[WATCHDOG_ENV]}s, triage -> {out_dir}/triage.log")


def chip_identity():
    """chip id -> (bdf, minor). Chip ids are the BusDeviceFunction-sorted position of each /dev/tenstorrent node."""
    minor_to_bdf = {
        int(node.name.split("!")[1]): Path(node, "device").resolve().name
        for node in Path("/sys/class/tenstorrent").glob("tenstorrent!*")
    }
    return dict(enumerate(sorted((bdf, minor) for minor, bdf in minor_to_bdf.items())))


def asic_ids():
    """chip id -> ASIC_ID. tt-smi -s is BusDeviceFunction-sorted, so its index is the chip id."""
    try:
        out = subprocess.run(["tt-smi", "-s"], capture_output=True, text=True, timeout=240).stdout
        devices = json.loads(out)["device_info"]
    except Exception as e:  # a wedged box may refuse telemetry; BusDeviceFunction alone still identifies the card
        logger.warning(f"could not read ASIC_IDs from tt-smi: {e!r}")
        return {}
    ids = {}
    for chip, dev in enumerate(devices):
        hi, lo = (dev.get("smbus_telem", {}).get(k) for k in ("ASIC_ID_HIGH", "ASIC_ID_LOW"))
        if hi and lo:
            ids[chip] = f"0x{int(hi, 16):08x}{int(lo, 16):08x}"
    return ids


def to_physical(runtime_id):
    """TT_VISIBLE_DEVICES renumbers visible chips to 0..N-1 by sorted chip id, so a runtime id is an
    index into that sorted set, not the physical chip. Without the filter the two coincide."""
    visible = os.environ.get("TT_VISIBLE_DEVICES", "").strip()
    if not visible:
        return runtime_id
    chips = sorted(int(v) for v in visible.split(",") if v.strip())
    return chips[runtime_id] if runtime_id < len(chips) else runtime_id


def describe(runtime_id, identity, asics=None):
    chip = to_physical(runtime_id)
    bdf, minor = identity.get(chip, ("?", "?"))
    bus = int(bdf.split(":")[1], 16) if bdf != "?" else None
    slot = f"T{TRAY_OF_BUS_NIBBLE.get(bus & 0xF0, '?')}:N{bus & 0x0F}" if bus is not None else "?"
    asic = f" asic_id={asics[chip]}" if asics and chip in asics else ""
    return f"chip {chip} (runtime id {runtime_id}): BusDeviceFunction={bdf} /dev/tenstorrent/{minor} {slot}{asic}"


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param((1, 1), device_params_for(ttnn.FabricConfig.FABRIC_2D), id="fabric2d-1x1"),
    ],
    indirect=["mesh_device", "device_params"],
)
@skip_for_wormhole_b0("shapes and program config are tuned for Blackhole")
@pytest.mark.timeout(0)
def test_mm_mcast_write_ack_failure_scan(mesh_device):
    if float(os.environ.get(WATCHDOG_ENV, 0)) <= 0:
        pytest.fail(f"{WATCHDOG_ENV} is non-positive, which disables hang detection entirely")

    grid = mesh_device.compute_with_storage_grid_size()
    gx, gy = (int(v) for v in os.environ.get("MATMUL_GRID", f"{grid.x - 1}x{grid.y}").split("x"))
    heads = int(os.environ.get("MATMUL_HEADS", 64))
    seq = int(os.environ.get("MATMUL_SEQ", 5120))
    m_tiles = seq // 32
    per_core_m = math.ceil(m_tiles / (gx * gy))
    while m_tiles % per_core_m:
        per_core_m += 1

    identity = chip_identity()
    for runtime_id in mesh_device.get_device_ids():
        logger.info(f"participating {describe(runtime_id, identity)}")
    logger.info(f"matmul [1,{heads},{seq},128] @ [1,{heads},128,512] on {gx}x{gy} cores, per_core_M={per_core_m}")

    dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    to_mesh = dict(
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=dram,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    in0 = ttnn.from_torch(torch.randn([1, heads, seq, 128]), dtype=ttnn.bfloat16, **to_mesh)
    in1 = ttnn.from_torch(torch.randn([1, heads, 128, 512]), dtype=ttnn.bfloat8_b, **to_mesh)

    try:
        for _ in range(int(os.environ.get("MATMUL_ITERS", 20))):
            out = ttnn.linear(
                in0,
                in1,
                memory_config=dram,
                dtype=ttnn.bfloat16,
                program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(gx, gy),
                    in0_block_w=4,
                    out_subblock_h=1,
                    out_subblock_w=8,
                    per_core_M=per_core_m,
                    per_core_N=16,
                    fuse_batch=False,
                    fused_activation=None,
                    mcast_in0=False,
                ),
                compute_kernel_config=ttnn.init_device_compute_kernel_config(
                    mesh_device.arch(),
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=True,
                ),
            )
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(out)
    except Exception:
        # The runtime blames a renumbered runtime id, not the physical chip, so name the card here.
        asics = asic_ids()
        for runtime_id in mesh_device.get_device_ids():
            logger.error(f"HUNG {describe(runtime_id, identity, asics)}")
        raise
    logger.success(f"no hang on chips {[to_physical(r) for r in mesh_device.get_device_ids()]}")


def scan_all_chips(iters=None, watchdog=30, work_dir=None):
    """Run this test once per chip of the host, one process each with its own cold cache, and report
    which chips hang. Returns the list of hung chip ids.

    Triage is disabled in the children because it halts cores and would disturb the other
    processes; re-run a failing chip on its own to triage it. This gate detects a hang but cannot
    clear one, so the failing chip stays wedged: reset the box (tt-smi -glx_reset_auto) before any
    re-run, or the wedged chip will fail its neighbours and they will look faulty too."""
    work_dir = Path(work_dir or Path(tempfile.gettempdir()) / "mm_mcast_write_ack_failure")
    shutil.rmtree(work_dir, ignore_errors=True)
    chips = sorted(chip_identity())
    running = {}
    for chip in chips:
        out = work_dir / f"chip{chip}"
        out.mkdir(parents=True)
        env = os.environ | {
            "TT_VISIBLE_DEVICES": str(chip),
            "TT_METAL_CACHE": str(out / "cache"),
            "OUT_DIR": str(out),
            WATCHDOG_ENV: str(watchdog),
            TRIAGE_ENV: "true",
        }
        if iters:
            env["MATMUL_ITERS"] = str(iters)
        log = (out / "pytest.log").open("w")
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--timeout=0", __file__, "-k", "1x1"]
        running[chip] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=log), log)
    logger.info(f"scanning {len(chips)} chips, dispatch watchdog={watchdog}s, logs under {work_dir}")

    # Poll rather than wait() in spawn order: a hung chip runs several times longer than a healthy
    # one, and a sequential wait holds back every result queued behind it. Chips report as they
    # finish, so the order here is completion order rather than chip order.
    hung = []
    pending = dict(running)
    while pending:
        for chip in [c for c, (proc, _) in pending.items() if proc.poll() is not None]:
            _, log = pending.pop(chip)
            log.close()
            text = (work_dir / f"chip{chip}" / "pytest.log").read_text(errors="replace")
            if "1 passed" in text:
                logger.success(f"chip {chip} PASS")
                continue
            # A killed child (Bus error, OOM) writes no diagnostic at all, so report that rather
            # than an empty reason. Ordered by usefulness: only the first names the card, and the
            # runtime's own line says "Device 0" for every chip because TT_VISIBLE_DEVICES renumbers.
            why = "no diagnostic line -- process killed (Bus error / OOM); check dmesg"
            for pattern in (
                r"HUNG chip \d+.*",
                r"Device \d+: Timeout[^.]*\.",
                r"Read 0x[0-9a-f]+ over PCIe[^\n]*",
                r"TT_THROW: [^(]*",
                r"RuntimeError: [^\n]*",
            ):
                found = re.search(pattern, text)
                if found:
                    why = found.group(0)
                    break
            logger.error(f"chip {chip} FAIL: {why}")
            logger.error(f"  cat {work_dir / f'chip{chip}' / 'pytest.log'}")
            hung.append(chip)
        if pending:
            time.sleep(0.5)

    hung.sort()
    if hung:
        logger.error(f"hung chips: {hung} -- reset before re-running: tt-smi -glx_reset_auto")
    else:
        logger.success(f"all {len(chips)} chips passed")
    return hung


if __name__ == "__main__":
    sys.exit(1 if scan_all_chips() else 0)
