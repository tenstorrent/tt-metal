#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Zero-weights forward test for ``RMSNorm.update``.

RMSNorm computes ``y = (x / sqrt(mean(x^2) + eps)) * gamma``. After
``update`` zeros out ``gamma`` the multiplication collapses the entire
output to 0 regardless of ``x`` or ``eps``. We still set a small ``eps``
explicitly so a pathological zero-mean-zero-input wouldn't blow up the
intermediate division (paranoia; the random ``x`` we feed has nonzero
RMS in practice).

We exercise ``DistributedNorm.update`` -- the one-line passthrough into
``RMSNorm.update`` that the rest of the model actually calls -- so this
also covers the wrapper.
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

SEQ_LEN = 32  # one decode tile

# Small but nonzero eps so the 1/sqrt(...) in RMSNorm never sees a
# literal zero denominator even if the input happens to be all-zeros.
SMALL_EPS = 1e-12


def _build_zero_like(rms, template, mesh_device):
    """Build an all-zeros ttnn tensor shaped like ``rms.weight``.

    We mirror the constructor's ``ReplicateTensorToMesh`` -- the
    non-distributed RMSNorm path uses a replicated weight on every
    device in the mesh.
    """
    import torch
    import ttnn

    shape = tuple(template.shape)
    torch_t = torch.zeros(shape, dtype=torch.bfloat16)

    is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh_device else None

    return ttnn.from_torch(
        torch_t,
        dtype=template.dtype,
        layout=template.layout,
        device=mesh_device,
        memory_config=template.memory_config(),
        mesh_mapper=mesh_mapper,
    )


def _build_random_rms_input(completer):
    """Construct a synthetic ``(1, 1, SEQ_LEN, dim)`` RMSNorm input."""
    import torch
    import ttnn

    dim = completer.model_args.dim
    torch_x = torch.randn(1, 1, SEQ_LEN, dim, dtype=torch.bfloat16)
    return ttnn.from_torch(
        torch_x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=completer.mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(completer.mesh_device),
    )


def main() -> None:
    import torch
    import ttnn
    from ttml.common.config import DeviceConfig, load_config

    from models.tt_transformers.tt.common import Mode
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

    distributed_norm = completer.model.layers[0].attention_norm
    rms = distributed_norm.norm  # the underlying RMSNorm
    print(f">>> rms.weight shape={tuple(rms.weight.shape)} dtype={rms.weight.dtype} eps={rms.eps}")

    # ---- Step 1: zero out RMSNorm gamma via the DistributedNorm passthrough ----
    print()
    print("=== Step 1: DistributedNorm.update(gamma=0) ===")
    zeros_t = _build_zero_like(rms, rms.weight, completer.mesh_device)
    distributed_norm.update(zeros_t)

    # Lock eps to a small value, see module docstring for rationale.
    rms.eps = SMALL_EPS
    print(f">>> rms.eps now = {rms.eps}")

    # ---- Step 2: forward random input through RMSNorm ----
    print()
    print(
        f"=== Step 2: RMSNorm.forward(random, mode=PREFILL) on shape (1, 1, {SEQ_LEN}, {completer.model_args.dim}) ==="
    )
    x = _build_random_rms_input(completer)
    out = rms.forward(x, Mode.PREFILL)
    out_torch = ttnn.to_torch(out)
    print(f"  out shape  = {tuple(out_torch.shape)}")
    print(f"  out dtype  = {out_torch.dtype}")
    print(f"  out max|.| = {float(out_torch.abs().max()):.6g}")
    print(f"  out mean|.|= {float(out_torch.abs().mean()):.6g}")

    # ---- Assertion ----
    print()
    print("=== assertion ===")
    expected = torch.zeros_like(out_torch)
    ok = torch.isclose(out_torch, expected, atol=1e-6).all()
    print(f"  RMSNorm.forward(x) == 0 elementwise: {ok}   [must be True]")

    print()
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")

    import ttml

    del completer
    gc.collect()
    ttml.autograd.AutoContext.get_instance().close_device()


if __name__ == "__main__":
    main()
