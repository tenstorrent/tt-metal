# SPDX-License-Identifier: Apache-2.0
"""Weight conversion + on-disk cache of converted (sharded, bfp8) device tensors for MiMo-V2.6-Flash on the 4x8 mesh.

Cache = ttnn flatbuffer of the HOST-sharded tensor (same mechanism as ttnn.as_tensor(cache_file_name=...)), one file per tensor under
$MIMO_WCACHE (default /mnt/tt-data/ssinghal/mimo-wcache)/<tag>/layer_<L>/<name>.tensorbin. A hit loads straight to the mesh without touching the
checkpoint. The key includes dtype and layout so a precision change cannot silently reuse a stale file.
Mesh mapping: rows (axis 0, 4) = DP, columns (axis 1, 8) = TP. Experts follow tt/moe_prefill.py (global_expert(row, col, local)).
"""
import os
import time

import ttnn
from models.demos.blackhole.mimo_v26.reference import loader as Ld

CACHE_ROOT = os.environ.get("MIMO_WCACHE", "/mnt/tt-data/ssinghal/mimo-wcache")
TAG = "v1"
HIDDEN, DENSE_INTER, VOCAB = 4096, 16384, 152576


def _dtype_name(dt):
    return str(dt).split(".")[-1].lower()


class WeightCache:
    """get(layer, name, make, dtype, layout, mapper, squeeze) -> device tensor. `make` is only called on a miss (lazy checkpoint read)."""

    def __init__(self, mesh, root=CACHE_ROOT, tag=TAG, build_only=False):
        self.mesh, self.root, self.tag, self.build_only = mesh, root, tag, build_only
        self.stats = {"hit": 0, "miss": 0, "convert_s": 0.0, "load_s": 0.0}

    def path(self, layer, name, dtype, layout):
        d = os.path.join(self.root, self.tag, f"layer_{layer}" if layer is not None else "model")
        os.makedirs(d, exist_ok=True)
        lay = "tile" if layout == ttnn.TILE_LAYOUT else "rm"
        return os.path.join(d, f"{name}.{_dtype_name(dtype)}.{lay}.tensorbin")

    def get(self, layer, name, make, dtype, layout, mapper, squeeze=0, mem=ttnn.DRAM_MEMORY_CONFIG):
        f = self.path(layer, name, dtype, layout)
        if os.path.isfile(f):
            if self.build_only:
                self.stats["hit"] += 1
                return None
            t0 = time.time()
            try:
                t = ttnn._ttnn.tensor.load_tensor_flatbuffer(f, device=self.mesh)
                self.stats["hit"] += 1
                self.stats["load_s"] += time.time() - t0
                return t
            except RuntimeError as e:  # corrupt/partial file: rebuild
                print(f"WCACHE bad file {f}: {e}; rebuilding", flush=True)
        t0 = time.time()
        host = make()
        t = ttnn.from_torch(host, dtype=dtype, layout=layout, mesh_mapper=mapper)
        for _ in range(squeeze):
            t = ttnn.squeeze(t, dim=0)
        tmp = f + f".tmp{os.getpid()}"
        ttnn._ttnn.tensor.dump_tensor_flatbuffer(tmp, t)
        os.replace(tmp, f)  # atomic: a killed builder never leaves a truncated cache file
        self.stats["miss"] += 1
        self.stats["convert_s"] += time.time() - t0
        if self.build_only:
            return None
        return ttnn.to_device(t, self.mesh, memory_config=mem)


def col_mapper(mesh, dim):
    return ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(None, dim))


def rep_mapper(mesh):
    return ttnn.ReplicateTensorToMesh(mesh)


def row_col_mapper(mesh):
    return ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(0, 1))


def load_dense_ffn(ck, L):
    """Layer-0 dense SwiGLU, fp8 block dequant -> [K,N] torch bf16 (gate, up) and [N,K] (down)."""
    p = f"model.layers.{L}.mlp."
    out = []
    for n in ("gate_proj", "up_proj", "down_proj"):
        w = Ld.dequant_fp8_block(ck.get(p + n + ".weight"), ck.get(p + n + ".weight_scale_inv"))
        out.append(w.t().contiguous().bfloat16())
    return out


def norm_weight(ck, name):
    """RMSNorm weight -> [1,1,4096/32,32] row-major bf16 (ttnn.rms_norm weight layout)."""
    return ck.get(name).bfloat16().reshape(1, 1, HIDDEN // 32, 32)
