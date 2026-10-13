# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Run: scripts/tt-probe.sh bringup_knowledge < models/demos/common/bringup/knowledge/probes/<this file>
# Known issue "ttnn.as_tensor cache hit can spin on a big ROW_MAJOR sharded tensor" (Xing M.1, mmapped safetensors).
import os
import threading
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import ttnn

d = Path("/localdev/dnijemcevic/bringup/probe_tmp")
d.mkdir(parents=True, exist_ok=True)
src = d / "w.safetensors"
if not src.exists():
    save_file({"w": torch.randn(131072, 3584).to(torch.bfloat16)}, str(src))
for f in d.glob("as_tensor_cache_cols*"):
    f.unlink()
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))
CAP = 600
for call in (1, 2):
    with safe_open(str(src), framework="pt") as fh:
        w = fh.get_tensor("w")  # mmap-backed
    done = threading.Event()
    threading.Thread(
        target=lambda: done.wait(CAP)
        or (print(f"AS_TENSOR call {call}: still running after {CAP} s (spin)", flush=True), os._exit(3)),
        daemon=True,
    ).start()
    t0 = time.time()
    t = ttnn.as_tensor(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(2, 2), dims=(None, 1)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=str(d / "as_tensor_cache_cols"),
    )
    done.set()
    print(
        f"AS_TENSOR call {call} ({'miss' if call == 1 else 'hit'}): {time.time() - t0:.1f} s, shape {list(t.shape)}",
        flush=True,
    )
    ttnn.deallocate(t)
ttnn.close_mesh_device(mesh)
