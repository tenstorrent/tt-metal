# SPDX-License-Identifier: Apache-2.0
"""Introspect the real ce_cache + moe_io tensors to resolve mesh/expert layout."""
import os, sys
from pathlib import Path
import torch

THIS_DIR = Path(__file__).resolve().parent
os.chdir(THIS_DIR)
sys.path.insert(0, str(THIS_DIR))
import ttnn  # noqa
import utils  # noqa

MESH = (4, 8)
CE = THIS_DIR / "moe_io" / "ce_cache"
IO = THIS_DIR / "moe_io"

dev = utils.DeviceGetter.get_device(MESH)


def info(name, t):
    try:
        print(f"{name:28s} shape={tuple(t.shape)} dtype={t.dtype} layout={t.layout}")
    except Exception as e:
        print(f"{name:28s} ERR {e}")


def per_device(name, path, dim_try=(0, 1)):
    t = ttnn.load_tensor(str(path), device=dev)
    info(name, t)
    return t


print("==== ce_cache tensors (as loaded, mesh) ====")
for k in [
    "main_const_eval_gate_up",
    "main_const_eval_39",
    "main_const_eval_37",
    "main_const_eval_9",
    "main_const_eval_24",
    "main_const_eval_25",
    "main_const_eval_16",
    "main_const_eval_19",
    "main_const_eval_29",
    "main_const_eval_35",
    "main_const_eval_6",
    "main_const_eval_17",
]:
    p = CE / f"{k}.tensorbin"
    if p.exists():
        per_device(k, p)

print("\n==== moe_io activation tensors ====")
for k in ["in_ttnn_add_22", "in_ttnn_reshape_117", "in_ttnn_rms_in_4d_3", "out_ttnn_add_27", "out_ttnn_rms_in_4d_4"]:
    per_device(k, IO / f"{k}.tensorbin")

print("\n==== expert_mapping (const_eval_37) values ====")
em = ttnn.load_tensor(str(CE / "main_const_eval_37.tensorbin"), device=dev)
# Try to view per-device. It's replicated or sharded; pull device 0 and a few.
try:
    shards = ttnn.get_device_tensors(em)
    print(f"num device shards: {len(shards)}")
    em0 = ttnn.to_torch(shards[0]).to(torch.int64)
    print(f"device0 expert_mapping shape={tuple(em0.shape)}")
    flat = em0.flatten()
    print(f"  first 32 entries: {flat[:32].tolist()}")
    print(f"  unique values: {sorted(set(flat.tolist()))[:40]} (n_unique={len(set(flat.tolist()))})")
    print(f"  min={flat.min().item()} max={flat.max().item()}")
    # Check if all device shards identical (replicated)
    em1 = ttnn.to_torch(shards[1]).to(torch.int64).flatten()
    print(f"  device1 == device0 ? {torch.equal(flat, em1)}")
except Exception as e:
    print("em introspection err:", e)

print("\n==== gate_up / w2 per-device shard shapes ====")
for k in ["main_const_eval_gate_up", "main_const_eval_39"]:
    t = ttnn.load_tensor(str(CE / f"{k}.tensorbin"), device=dev)
    shards = ttnn.get_device_tensors(t)
    print(f"{k}: global shape={tuple(t.shape)}  n_shards={len(shards)}  shard0 shape={tuple(shards[0].shape)}")

# router outputs live-in? No — recomputed in block. Inspect input reshape_117 per-device.
print("\n==== reshape_117 / rms_in_4d_3 per device ====")
for k in ["in_ttnn_reshape_117", "in_ttnn_rms_in_4d_3", "in_ttnn_add_22"]:
    t = ttnn.load_tensor(str(IO / f"{k}.tensorbin"), device=dev)
    shards = ttnn.get_device_tensors(t)
    print(f"{k}: global={tuple(t.shape)} n_shards={len(shards)} shard0={tuple(shards[0].shape)}")

if utils.DeviceGetter._instance is not None:
    ttnn.close_mesh_device(utils.DeviceGetter._instance)
    utils.DeviceGetter._instance = None
