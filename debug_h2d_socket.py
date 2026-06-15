# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone debug harness for the cross-process H2D/D2H socket test."""
import os
import sys
import time
import traceback
import uuid
import multiprocessing as mp

import torch

_WEIGHT_SEED = 0xB1A5E


def _log(tag, msg):
    print(f"[{time.time():.2f}] [{tag}] {msg}", flush=True)


def _make_weight(K, N):
    g = torch.Generator().manual_seed(_WEIGHT_SEED)
    return torch.randn(K, N, generator=g, dtype=torch.float32)


def worker(worker_index, visible_device, run_id, M, K, N, h2d_mode_name, num_iterations, ready_event, done_event, q):
    os.environ["TT_VISIBLE_DEVICES"] = str(visible_device)
    import ttnn

    mesh_device = None
    h2d_socket = None
    d2h_socket = None
    try:
        bpe = 2
        in_page = K * bpe
        out_page = N * bpe
        h2d_fifo = max(2048, in_page * 4)
        d2h_fifo = max(2048, out_page * 4)
        h2d_id = f"{run_id}_h2d_{worker_index}"
        d2h_id = f"{run_id}_d2h_{worker_index}"
        h2d_mode = getattr(ttnn.H2DMode, h2d_mode_name)

        _log(f"W{worker_index}", f"opening mesh device (TT_VISIBLE_DEVICES={visible_device})")
        mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
        _log(f"W{worker_index}", "mesh device opened")

        dc = ttnn.MeshCoordinate(0, 0)
        h2d_core = ttnn.MeshCoreCoord(dc, ttnn.CoreCoord(0, 0))
        d2h_core = ttnn.MeshCoreCoord(dc, ttnn.CoreCoord(1, 0))

        h2d_socket = ttnn.H2DSocket(mesh_device, h2d_core, ttnn.BufferType.L1, h2d_fifo, h2d_mode)
        h2d_socket.set_page_size(in_page)
        _log(f"W{worker_index}", "h2d socket created")
        d2h_socket = ttnn.D2HSocket(mesh_device, d2h_core, d2h_fifo)
        d2h_socket.set_page_size(out_page)
        _log(f"W{worker_index}", "d2h socket created")

        torch_weight = _make_weight(K, N)
        weight_tensor = ttnn.from_torch(
            torch_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _log(f"W{worker_index}", "weight on device")

        h2d_socket.export_descriptor(h2d_id)
        d2h_socket.export_descriptor(d2h_id)
        _log(f"W{worker_index}", "exported descriptors; setting ready")
        ready_event.set()

        for it in range(num_iterations):
            _log(f"W{worker_index}", f"iter {it}: alloc input + recv_async_h2d")
            input_rm = ttnn.from_torch(
                torch.zeros(M, K, dtype=torch.float32),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.experimental.recv_async_h2d(input_rm, h2d_socket)
            _log(f"W{worker_index}", f"iter {it}: recv enqueued; tilize+matmul")
            in_tile = ttnn.to_layout(input_rm, ttnn.TILE_LAYOUT)
            out_tile = ttnn.matmul(in_tile, weight_tensor)
            out_rm = ttnn.to_layout(out_tile, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.experimental.send_async_d2h(out_rm, d2h_socket)
            _log(f"W{worker_index}", f"iter {it}: send enqueued; synchronize")
            ttnn.synchronize_device(mesh_device)
            _log(f"W{worker_index}", f"iter {it}: synchronized")

        _log(f"W{worker_index}", "loop done; waiting for done_event")
        done_event.wait(timeout=120)
        _log(f"W{worker_index}", "done_event received")
        q.put((worker_index, "ok", None))
    except Exception:
        tb = traceback.format_exc()
        _log(f"W{worker_index}", f"ERROR:\n{tb}")
        q.put((worker_index, "error", tb))
        ready_event.set()
    finally:
        del h2d_socket
        del d2h_socket
        if mesh_device is not None:
            try:
                ttnn.close_mesh_device(mesh_device)
            except Exception:
                pass


def _num_physical_devices():
    d = "/dev/tenstorrent"
    if not os.path.isdir(d):
        return 0
    return sum(1 for e in os.listdir(d) if e.isdigit())


def main():
    import ttnn

    M, K, N, num_iterations, num_workers = 32, 32, 32, 1, 1
    h2d_mode_name = "HOST_PUSH"
    _log("MAIN", f"physical devices = {_num_physical_devices()}")

    run_id = uuid.uuid4().hex[:8]
    ctx = mp.get_context("spawn")
    ready = [ctx.Event() for _ in range(num_workers)]
    done = ctx.Event()
    q = ctx.Queue()
    procs = []
    for i in range(num_workers):
        p = ctx.Process(target=worker, args=(i, i, run_id, M, K, N, h2d_mode_name, num_iterations, ready[i], done, q))
        p.start()
        procs.append(p)

    conns = []
    try:
        for i, ev in enumerate(ready):
            _log("MAIN", f"waiting ready {i}")
            assert ev.wait(timeout=180), f"worker {i} not ready"
            _log("MAIN", f"worker {i} ready (alive={procs[i].is_alive()})")

        weight = _make_weight(K, N)
        for i in range(num_workers):
            h2d_id = f"{run_id}_h2d_{i}"
            d2h_id = f"{run_id}_d2h_{i}"
            _log("MAIN", f"connecting h2d {i}")
            h2d = ttnn.H2DSocket.connect(h2d_id, 60000)
            _log("MAIN", f"connecting d2h {i}")
            d2h = ttnn.D2HSocket.connect(d2h_id, 60000)
            conns.append((h2d, d2h))
            _log("MAIN", f"connected {i}")

        for it in range(num_iterations):
            for i, (h2d, d2h) in enumerate(conns):
                ti = torch.randn(M, K, dtype=torch.float32)
                ib = ti.to(torch.bfloat16).contiguous()
                _log("MAIN", f"iter {it} w{i}: write_tensor")
                h2d.write_tensor(ib)
                _log("MAIN", f"iter {it} w{i}: read_tensor")
                res = torch.zeros(M, N, dtype=torch.bfloat16)
                d2h.read_tensor(res)
                _log("MAIN", f"iter {it} w{i}: read done")
                exp = ti @ weight
                # quick pcc
                a = exp.flatten().to(torch.float32)
                b = res.flatten().to(torch.float32)
                pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
                _log("MAIN", f"iter {it} w{i}: pcc={pcc:.5f}")

        conns.clear()
        _log("MAIN", "setting done")
        done.set()
        for _ in range(num_workers):
            _log("MAIN", f"result: {q.get(timeout=120)}")
    finally:
        conns.clear()
        done.set()
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                _log("MAIN", f"terminating stuck proc {p.name}")
                p.terminate()
                p.join(timeout=30)
    _log("MAIN", "DONE")


if __name__ == "__main__":
    main()
