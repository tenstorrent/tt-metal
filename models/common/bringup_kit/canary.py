#!/usr/bin/env python3
"""Device canary: a tiny deterministic matmul replicated on every chip; flags chips/hosts that compute wrong results.

    canary.py [--mesh 4x8] [--record]      (run under the device lock, with the worktree's python_env; see host_health.py --canary)

Every chip multiplies the same fixed bf16 matrices. Checks: (1) all chips return bit-identical output (a host like .31, wrong outputs in mesh
rows 0-1, fails here with no stored reference at all); (2) the sha256 equals canary_ref.json (--record writes it; the reference is keyed by
arch + mesh shape, record it once on a known-good host). Prints CANARY_OK / CANARY_FAIL <details>. Exit 0 only on CANARY_OK.
HARDWARE STATUS: written against the ttnn Python API (open_mesh_device, ReplicateTensorToMesh, get_device_tensors); NOT yet run on a device.
"""
import argparse
import hashlib
import json
import os
import sys

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canary_ref.json")


def make_inputs():
    import torch

    g = torch.Generator().manual_seed(1234)
    return torch.randn(1, 1, 256, 512, generator=g).bfloat16(), torch.randn(1, 1, 512, 256, generator=g).bfloat16()


def digest(t):
    return hashlib.sha256(t.float().contiguous().numpy().tobytes()).hexdigest()[:16]


def evaluate(hashes, ref_hash):
    """hashes: per-chip digests. -> (ok, message). Pure function (unit tested)."""
    uniq = {}
    for i, h in enumerate(hashes):
        uniq.setdefault(h, []).append(i)
    if len(uniq) > 1:
        majority = max(uniq.values(), key=len)
        bad = sorted(i for v in uniq.values() if v is not majority for i in v)
        return False, f"chips disagree: {len(uniq)} distinct outputs, deviating chips {bad}"
    h = next(iter(uniq))
    if ref_hash is not None and h != ref_hash:
        return False, f"all chips agree ({h}) but differ from the stored reference {ref_hash}"
    return True, f"all {len(hashes)} chips agree ({h})" + ("" if ref_hash else "; no stored reference")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", default="4x8")
    p.add_argument("--record", action="store_true")
    a = p.parse_args(argv)
    import ttnn

    r, c = (int(x) for x in a.mesh.split("x"))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(r, c))
    try:
        x, w = make_inputs()
        tx = ttnn.from_torch(
            x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
        )
        tw = ttnn.from_torch(
            w, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
        )
        out = ttnn.matmul(tx, tw)
        hashes = [digest(ttnn.to_torch(t)) for t in ttnn.get_device_tensors(out)]
    finally:
        ttnn.close_mesh_device(mesh)
    refs = json.load(open(REF)) if os.path.exists(REF) else {}
    key = f"mesh{a.mesh}"
    ok, msg = evaluate(hashes, refs.get(key))
    if ok and a.record:
        refs[key] = hashes[0]
        json.dump(refs, open(REF, "w"), indent=1)
        msg += f"; recorded to {REF}"
    print(("CANARY_OK " if ok else "CANARY_FAIL ") + msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
