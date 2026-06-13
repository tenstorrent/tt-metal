# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Minimal repro for the tt_bh_glx trace-capture hang.

Captures a TTNN trace whose ops run on disjoint 1x1 submeshes (no weights, no
sockets, no pipeline). Isolates whether `begin/end_trace_capture` over per-chip
submeshes hangs by itself.

Env knobs:
  REPRO_ROOT     parent | compute   (trace root: full (8,4) or (7,4) submesh)  [default compute]
  REPRO_COORDS   semicolon list "r,c;r,c"  device coords of the 1x1 submeshes   [default "0,0;1,0;6,3"]
  REPRO_FULLMESH 1 -> also run one op on a full-root-mesh tensor (control)      [default 0]
  REPRO_SOCKET   1 -> add a fabric-socket send/recv hop (subs[0]->subs[1]) in   [default 0]
                      the traced body, mirroring the pipeline's SocketTransport

Run with a hard timeout so a hang is obvious:
  timeout 150 python_env/bin/python models/experimental/pi0_5/tests/perf/_trace_submesh_repro.py
Exit 124 = timed out (hung). Watch the printed markers for where.
"""

import os
import sys

import torch
import ttnn


def _coords():
    raw = os.environ.get("REPRO_COORDS", "0,0;1,0;6,3")
    out = []
    for tok in raw.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        r, c = (int(x) for x in tok.split(","))
        out.append((r, c))
    return out


def main():
    def log(m):
        print(f"[repro] {m}", flush=True)

    root_kind = os.environ.get("REPRO_ROOT", "compute").strip().lower()
    coords = _coords()
    fullmesh = os.environ.get("REPRO_FULLMESH", "0") == "1"
    log(f"root={root_kind} coords={coords} fullmesh={fullmesh}")

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=134_217_728)
    submeshes = []
    try:
        if root_kind == "parent":
            root = parent
        else:
            root = parent.create_submesh(ttnn.MeshShape(7, 4), ttnn.MeshCoordinate(0, 0))
            submeshes.append(root)
        log(f"opened root shape={root.shape}")

        subs = []
        for r, c in coords:
            sm = root.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(r, c))
            submeshes.append(sm)
            subs.append(sm)
        log(f"carved {len(subs)} 1x1 submeshes")

        def mk(sm):
            return ttnn.from_torch(
                torch.randn(1, 1, 32, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=sm,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        tens = [mk(sm) for sm in subs]
        full_t = mk(root) if fullmesh else None
        log("uploaded tensors")

        # Optional fabric-socket hop subs[0] -> subs[1], mirroring SocketTransport.
        use_socket = os.environ.get("REPRO_SOCKET", "0") == "1"
        send_sock = recv_sock = sock_out = None
        if use_socket:
            if len(subs) < 2:
                raise RuntimeError("REPRO_SOCKET needs >=2 submeshes")
            conn = ttnn.SocketConnection(
                ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
                ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 1)),
            )
            mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 4096 * 4)
            cfg = ttnn.SocketConfig([conn], mem)
            send_sock, recv_sock = ttnn.create_socket_pair(subs[0], subs[1], cfg)
            sock_out = ttnn.allocate_tensor_on_device(tens[0].spec, subs[1])
            log("created socket pair + recv buffer")

        def step():
            outs = [ttnn.add(t, t, memory_config=ttnn.L1_MEMORY_CONFIG) for t in tens]
            if full_t is not None:
                outs.append(ttnn.add(full_t, full_t, memory_config=ttnn.L1_MEMORY_CONFIG))
            if use_socket:
                ttnn.experimental.send_direct_async(tens[0], send_sock)
                ttnn.experimental.recv_direct_async(sock_out, recv_sock)
            return outs

        log("warmup step (JIT)...")
        _ = step()
        if os.environ.get("REPRO_NOSYNC", "0") != "1":
            ttnn.synchronize_device(root)
            log("warmup done (synced)")
        else:
            log("warmup done (NO sync — mimics pipeline capture_trace)")

        log("begin_trace_capture")
        tid = ttnn.begin_trace_capture(root, cq_id=0)
        _ = step()
        log("captured body; calling end_trace_capture (HANG POINT)")
        ttnn.end_trace_capture(root, tid, cq_id=0)
        log("END_TRACE_CAPTURE OK")

        log("execute_trace")
        ttnn.execute_trace(root, tid, cq_id=0, blocking=True)
        log("EXECUTE_TRACE OK")
        log("SUCCESS")
    finally:
        for sm in reversed(submeshes):
            try:
                ttnn.close_mesh_device(sm)
            except Exception:
                pass
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
    sys.exit(0)
