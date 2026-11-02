# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_DRAFT_1ROW: the draft rows on the MTP layer's 1-row decode body against the verify-form rows, on the real model.

Starts the chat server (--mtp 4) with QWEN38_DR2_COMPARE=1: its eager MTP warm rounds run both draft bodies on the same state at
the positions QWEN38_DR2_POSITIONS (P % 4 = 0..3 and P % 32 in {0, 1, 29, 30, 31}, so draft rows cross 4- and 32-blocks) and print
one ``DR2 {json}`` line per round (tools/dr2_probe.py): drafted ids of both bodies, per draft row the PCC / max abs of the final
mixer's hidden, and whether the committed cache rows (< P) and compressed blocks (< P // 4) are bitwise unchanged by each body.
Env: DR2_VISIBLE_DEVICES / DR2_DEVICE_NODES (default the 8 devices of group 1), DR2_OUT (evidence directory), DR2_RESULTS (jsonl)."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
M = Path(__file__).resolve().parents[1]
POSITIONS = os.environ.get("DR2_POSITIONS", "29,62,95,128,161,194")
IDLE_LIMIT_S = 15 * 60


def test_draft_row_1row_vs_verify_form_rows():
    out = (
        Path(os.environ.get("DR2_OUT", "/tmp/dr2_out")) / f"run-{int(time.time())}"
    )  # the server refuses a used evidence dir
    out.mkdir(parents=True, exist_ok=True)
    nodes = os.environ.get("DR2_VISIBLE_DEVICES", "8,9,10,11,15,14,13,12")  # PCIe ids (TT_VISIBLE_DEVICES)
    dev_nodes = os.environ.get("DR2_DEVICE_NODES", nodes)  # /dev/tenstorrent nodes of the partition, in mesh order
    cache = "/mnt/nvme5/tmp/qwen38fn/cache_W/caches"
    env = dict(os.environ)
    env.update(
        QWEN38_DRAFT_1ROW="probe",
        QWEN38_DR2_COMPARE="1",
        QWEN38_DR2_POSITIONS=POSITIONS,
        TT_VISIBLE_DEVICES=nodes,
        PYTHONPATH=str(ROOT),
        MESH_DEVICE="P150x8",
    )
    env.pop("TT_MESH_GRAPH_DESC_PATH", None)
    env["TT_MESH_GRAPH_DESC_PATH"] = str(M / "tools/p150_x8_1x8_ring_mesh_graph_descriptor.textproto")
    # A partition whose UMD indices (TT_VISIBLE_DEVICES) differ from its /dev nodes (group 3: 24..31 over nodes 16..23): the
    # server's profile takes the nodes as the indices, so the profile's visible_devices is set to the env's here.
    launcher = (
        "import os, sys\n"
        "from dataclasses import replace\n"
        "from models.demos.blackhole.qwen38_flash_next.tools import hardware_profiles as hp\n"
        "from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as srv\n"
        "orig = hp.ResidentHardwareProfile.with_device_nodes\n"
        "hp.ResidentHardwareProfile.with_device_nodes = lambda self, nodes, **k: replace(\n"
        "    orig(self, nodes, **k), visible_devices=os.environ['TT_VISIBLE_DEVICES'])\n"
        "sys.exit(srv.main())\n"
    )
    cmd = [
        sys.executable,
        "-c",
        launcher,
        "--hardware-profile",
        "p150-x8",
        "--device-nodes",
        dev_nodes,
        "--checkpoint",
        "/mnt/nvme5/tmp/qwen38fn/hf",
        "--component-cache-root",
        f"{cache}/c32768-vllm/components",
        "--routed-bf4-scratch-root",
        f"{cache}/bf4-experts",
        "--model-io-cache-root",
        f"{cache}/c32768-vllm/model-io",
        "--evidence",
        str(out),
        "--phase-log",
        str(out / "phases.jsonl"),
        "--mtp",
        "4",
        "--port",
        os.environ.get("DR2_PORT", "8147"),
    ]
    log = out / "server.log"
    expected = len(POSITIONS.split(","))
    lines = []
    with log.open("w") as sink:
        proc = subprocess.Popen(
            cmd, env=env, stdout=sink, stderr=subprocess.STDOUT, start_new_session=True, cwd=str(ROOT)
        )
        last_size, last_change = 0, time.time()
        try:
            while proc.poll() is None:
                time.sleep(5)
                size = log.stat().st_size
                if size != last_size:
                    last_size, last_change = size, time.time()
                elif time.time() - last_change > IDLE_LIMIT_S:
                    pytest.fail("no server log progress for 15 minutes")
                lines = [ln for ln in log.read_text(errors="replace").splitlines() if "DR2 {" in ln]
                if len(lines) >= expected:
                    break
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
    rounds = [json.loads(ln.split("DR2 ", 1)[1]) for ln in lines]
    results = Path(os.environ.get("DR2_RESULTS", str(out / "results.jsonl")))
    results.write_text("".join(json.dumps(r) + "\n" for r in rounds))
    assert len(rounds) == expected, f"{len(rounds)} DR2 rounds of {expected}; see {log}"
    equal = total = 0
    for r in rounds:
        assert r["committed_unchanged_old"] and r["committed_unchanged_new"], r
        for row in r["rows"]:
            total += 1
            equal += row["token_old"] == row["token_new"]
        assert r["rows"][0]["pcc"] > 0.98, r["rows"][0]
    print(f"DR2 top-1 equality {equal}/{total}")
