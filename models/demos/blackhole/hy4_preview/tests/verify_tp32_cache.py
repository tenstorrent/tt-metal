# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Round-trip check of the TP32 cache on a 1x8 slot: layer VL (default 1), chips VCHIP0 .. +7.
(a) bytes: the dump of the in-memory pack (hy4_moe_tp32.pack_chips -> chip_host_tensor) has the manifest sha256 of the cache file;
(b) device: mesh tensors loaded from the cache == mesh tensors made by the in-process pack (bank_tiles_tensor), read back from every chip, exact.
Env: HY4_TP32_CACHE_DIR, VL, VCHIP0, HY4_CKPT."""
import os

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import p
from models.demos.blackhole.hy4_preview.tt import hy4_tp32_cache as C
from models.demos.blackhole.hy4_preview.tt.hy4_moe_tp32 import bank_tiles_tensor, pack_chips
from models.demos.blackhole.hy4_preview.tt.hy4_moe_weights import load_moe_layer

L = int(os.environ.get("VL", "1"))
C0 = int(os.environ.get("VCHIP0", "0"))
root = C.cache_root()
d = C.check_layer(root, L, verify=True)
assert d, "no cache"
p("cache layer ok (meta + sha256 manifest verified)", d)
W = load_moe_layer(L, experts=list(range(256)))
w0, w1, w2 = (W[k].reshape(256, *W[k].shape[-2:]) for k in ("w0", "w1", "w2"))
import json
import tempfile

man = json.load(open(os.path.join(d, "moe_manifest_tp32.json")))["files"]
chips = list(range(C0, C0 + 8))
t01, t2 = pack_chips(w0, w1, w2, chips)
bad = 0
with tempfile.TemporaryDirectory() as td:
    for i, c in enumerate(chips):
        for kind, t in (("w01", t01[i]), ("w2", t2[i])):
            fp = os.path.join(td, C.fname(kind, c))
            ttnn.dump_tensor(fp, C.chip_host_tensor(t))
            ok = C.sha256_file(fp) == man[C.fname(kind, c)]["sha256"]
            bad += not ok
p("(a) bytes: in-memory pack dump vs cache file sha256, mismatches =", bad, "of 16")
md = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=16384)
try:
    a01, a2 = bank_tiles_tensor(md, t01), bank_tiles_tensor(md, t2)
    b01, b2 = C.load_layer(md, L, chip0=C0)
    nd = 0
    for ta, tb in ((a01, b01), (a2, b2)):
        for i, (x, y) in enumerate(zip(ttnn.get_device_tensors(ta), ttnn.get_device_tensors(tb))):
            nd += not torch.equal(ttnn.to_torch(x), ttnn.to_torch(y))
    p("(b) device readback in-memory vs cache, differing chip tensors =", nd, "of 16")
    ok = bad == 0 and nd == 0
    p("ROUNDTRIP", "PASS" if ok else "FAIL")
finally:
    ttnn.close_mesh_device(md)
