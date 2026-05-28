#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Mechanical tests for ``Attention.update`` (no forward pass).

Two tests, both side-step the attention formula entirely:

1. **Read-back.**  Build a known constant-valued tensor matching the live
   ``wqkv`` / ``wo`` buffers (shape, dtype, layout, memory config, mesh
   sharding).  Push it through ``Attention.update``, read the live buffers
   back to torch, and assert byte equality.

2. **Buffer-address preservation.**  Snapshot ``buffer_address()`` for each
   weight before the update, do another update with a different constant,
   and snapshot again.  Addresses must be identical -- ``ttnn.copy`` is
   expected to overwrite the existing on-device buffer rather than allocate
   a new one, which is what keeps captured traces and the DRAM prefetcher's
   stashed addresses valid.

We initialize the model with ``dummy_weights=True`` so the test doesn't need
HF auth or weight downloads.
"""

from __future__ import annotations

import os

os.environ.setdefault("TT_LOGGER_LEVEL", "Error")

import gc
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRPO_SPEEDUP = HERE.parent  # .../grpo_speedup
REPO_ROOT = HERE.parents[4]  # .../tt-metal
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(GRPO_SPEEDUP))
sys.path.insert(0, str(REPO_ROOT))

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TTML_DEVICE_CONFIG_REL = "tt-train/configs/training_configs/grpo_boolq_llama_1dev.yaml"
MAX_SEQ_LEN = 2048

# Bf16-exact constants so torch.equal is meaningful round-trip.
WQKV_CONST_1 = 0.5
WO_CONST_1 = 0.25
WQKV_CONST_2 = 0.125
WO_CONST_2 = 0.0625


def _wqkv_mesh_mapper(attn):
    """Mirror the mesh mapping used in Attention.__init__ for ``self.wqkv``."""
    import ttnn

    return ttnn.ShardTensor2dMesh(
        attn.mesh_device,
        dims=(3, 2) if attn.TG else (2, 3),
        mesh_shape=attn.args.cluster_shape,
    )


def _wo_mesh_mapper(attn):
    """Mirror the mesh mapping used in Attention.__init__ for ``self.wo``."""
    import ttnn

    if attn.use_fused_all_gather_matmul or attn.TG:
        return ttnn.ShardTensor2dMesh(
            attn.mesh_device,
            dims=(2, 3),
            mesh_shape=attn.args.cluster_shape,
        )
    return ttnn.ShardTensorToMesh(attn.mesh_device, dim=2)


def _build_constant_like(attn, template, value, mesh_mapper):
    """Build a constant-valued ``ttnn.Tensor`` matching ``template``'s shape /
    dtype / layout / memory config, sharded with ``mesh_mapper``."""
    import torch
    import ttnn

    shape = tuple(template.shape)
    torch_t = torch.full(shape, float(value), dtype=torch.bfloat16)
    return ttnn.from_torch(
        torch_t,
        dtype=template.dtype,
        layout=template.layout,
        device=attn.mesh_device,
        memory_config=template.memory_config(),
        mesh_mapper=mesh_mapper,
    )


def _read_to_torch(tensor):
    """Read a (sharded) ttnn weight back to torch.

    The grpo_speedup configs target a 1-device mesh (``mesh_shape: [1, 1]``)
    so ``ttnn.to_torch`` with no composer is sufficient; this mirrors the
    pattern used in test_collapse_embedding_forward.py.
    """
    import ttnn

    return ttnn.to_torch(tensor)


def _buffer_addr(tensor):
    """``buffer_address()`` for a possibly multi-device ttnn.Tensor."""
    import ttnn

    try:
        return tensor.buffer_address()
    except Exception:
        return tuple(t.buffer_address() for t in ttnn.get_device_tensors(tensor))


def _all_equal_constant(got, value):
    """Return ``(ok, max_abs_diff)``: True iff every element of ``got`` equals
    ``value`` exactly (in bf16, the chosen constants are exact)."""
    import torch

    expected = torch.full(tuple(got.shape), float(value), dtype=got.dtype)
    diff = (got.float() - float(value)).abs()
    return torch.equal(got, expected), float(diff.max())


def _print_attn_shapes(attn):
    print(f">>> attn.wqkv: shape={tuple(attn.wqkv.shape)} dtype={attn.wqkv.dtype} layout={attn.wqkv.layout}")
    print(f">>> attn.wo:   shape={tuple(attn.wo.shape)} dtype={attn.wo.dtype} layout={attn.wo.layout}")
    wo_sharded_ring = getattr(attn, "wo_sharded_ring", None)
    if wo_sharded_ring is not None:
        print(
            f">>> attn.wo_sharded_ring: shape={tuple(wo_sharded_ring.shape)} "
            f"dtype={wo_sharded_ring.dtype} layout={wo_sharded_ring.layout}"
        )


def test_readback(attn) -> bool:
    """Test 1: update wqkv/wo with constants, read back, assert equality."""
    print()
    print("=== Test 1: read-back ===")
    print(f">>> updating: wqkv <- {WQKV_CONST_1},  wo <- {WO_CONST_1}")

    target_wqkv = _build_constant_like(attn, attn.wqkv, WQKV_CONST_1, _wqkv_mesh_mapper(attn))
    target_wo = _build_constant_like(attn, attn.wo, WO_CONST_1, _wo_mesh_mapper(attn))
    attn.update(wqkv=target_wqkv, wo=target_wo)

    got_wqkv = _read_to_torch(attn.wqkv)
    got_wo = _read_to_torch(attn.wo)

    wqkv_ok, wqkv_max = _all_equal_constant(got_wqkv, WQKV_CONST_1)
    wo_ok, wo_max = _all_equal_constant(got_wo, WO_CONST_1)

    print(f"  wqkv read-back == {WQKV_CONST_1}: {wqkv_ok}   (max|diff|={wqkv_max:.6g})")
    print(f"  wo   read-back == {WO_CONST_1}:   {wo_ok}   (max|diff|={wo_max:.6g})")

    ok = wqkv_ok and wo_ok

    # If the prefetched mirror exists, ``_update_wo`` also writes it; check it.
    wo_sharded_ring = getattr(attn, "wo_sharded_ring", None)
    if wo_sharded_ring is not None:
        got_ring = _read_to_torch(wo_sharded_ring)
        ring_ok, ring_max = _all_equal_constant(got_ring, WO_CONST_1)
        print(f"  wo_sharded_ring read-back == {WO_CONST_1}: {ring_ok}   (max|diff|={ring_max:.6g})")
        ok = ok and ring_ok

    return ok


def test_addresses_preserved(attn) -> bool:
    """Test 2: update again with different constants; addresses must not change."""
    print()
    print("=== Test 2: buffer-address preservation ===")

    addrs_before = {
        "wqkv": _buffer_addr(attn.wqkv),
        "wo": _buffer_addr(attn.wo),
    }
    wo_sharded_ring = getattr(attn, "wo_sharded_ring", None)
    if wo_sharded_ring is not None:
        addrs_before["wo_sharded_ring"] = _buffer_addr(wo_sharded_ring)
    print(f"  addrs before update: {addrs_before}")

    print(f">>> updating: wqkv <- {WQKV_CONST_2},  wo <- {WO_CONST_2}")
    target_wqkv = _build_constant_like(attn, attn.wqkv, WQKV_CONST_2, _wqkv_mesh_mapper(attn))
    target_wo = _build_constant_like(attn, attn.wo, WO_CONST_2, _wo_mesh_mapper(attn))
    attn.update(wqkv=target_wqkv, wo=target_wo)

    addrs_after = {
        "wqkv": _buffer_addr(attn.wqkv),
        "wo": _buffer_addr(attn.wo),
    }
    if wo_sharded_ring is not None:
        addrs_after["wo_sharded_ring"] = _buffer_addr(wo_sharded_ring)
    print(f"  addrs after  update: {addrs_after}")

    ok = addrs_before == addrs_after
    if not ok:
        for k in addrs_before:
            if addrs_before[k] != addrs_after[k]:
                print(f"  MISMATCH on '{k}': {addrs_before[k]} -> {addrs_after[k]}")

    return ok


def main() -> None:
    import ttnn
    from ttml.common.config import DeviceConfig, load_config

    from utils.llama_completer_ttt import LlamaGRPOCompleter

    print(">>> set_fabric_config(FABRIC_2D)")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    raw = load_config(os.path.join(REPO_ROOT, TTML_DEVICE_CONFIG_REL))
    device_config = DeviceConfig(raw)

    print(f">>> building LlamaGRPOCompleter ({MODEL_ID}, dummy_weights=True)")
    completer = LlamaGRPOCompleter(
        device_config=device_config,
        model_source=MODEL_ID,
        max_batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
        dummy_weights=True,
    )

    attn = completer.model.layers[0].attention
    _print_attn_shapes(attn)

    readback_ok = test_readback(attn)
    addresses_ok = test_addresses_preserved(attn)

    print()
    print("=== summary ===")
    print(f"  Test 1 read-back:               {'PASS' if readback_ok else 'FAIL'}")
    print(f"  Test 2 address preservation:    {'PASS' if addresses_ok else 'FAIL'}")
    print()
    print(f"  RESULT: {'PASS' if (readback_ok and addresses_ok) else 'FAIL'}")

    import ttml

    del completer
    gc.collect()
    ttml.autograd.AutoContext.get_instance().close_device()


if __name__ == "__main__":
    main()
