#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Zero-weights forward test for ``MLP.update``.

Sanity test that an end-to-end ``MLP.forward`` collapses to all-zeros
once every projection has been overwritten with zeros:

* gate = x @ w1  -> 0 because w1 = 0
* up   = x @ w3  -> 0 because w3 = 0
* act  = silu(gate) * up = silu(0) * 0 = 0
* out  = act @ w2 = 0 @ w2 = 0

This exercises every ``ttnn.copy`` path in ``MLP._update_w{1,2,3}``
without needing real model weights -- ``dummy_weights=True`` skips the
HF auth + safetensors download entirely.
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

# Prefill seq_len for the synthetic MLP input. 128 stays well below
# args.prefill_len_cutoff (512 on WH / 1024 on BH) so MLP.forward does
# not take the chunked-prefill reshape branch.
SEQ_LEN = 128


def _w1_w3_mesh_mapper(mlp):
    """Mirror the mesh mapper used in ``MLP.__init__`` for ``self.w1`` and ``self.w3``."""
    import ttnn

    dims = (-1, -2) if mlp.args.is_galaxy else (-2, -1)
    return ttnn.ShardTensor2dMesh(mlp.mesh_device, dims=dims, mesh_shape=mlp.args.cluster_shape)


def _w2_mesh_mapper(mlp):
    """Mirror the mesh mapper used in ``MLP.__init__`` for ``self.w2``."""
    import ttnn

    dims = (-2, -1) if mlp.args.is_galaxy else (-1, -2)
    return ttnn.ShardTensor2dMesh(mlp.mesh_device, dims=dims, mesh_shape=mlp.args.cluster_shape)


def _build_zero_like(mlp, template, mesh_mapper):
    """Build an all-zeros ``ttnn.Tensor`` shaped like ``template``."""
    import torch
    import ttnn

    shape = tuple(template.shape)
    torch_t = torch.zeros(shape, dtype=torch.bfloat16)
    return ttnn.from_torch(
        torch_t,
        dtype=template.dtype,
        layout=template.layout,
        device=mlp.mesh_device,
        memory_config=template.memory_config(),
        mesh_mapper=mesh_mapper,
    )


def zero_mlp(mlp) -> None:
    """Splat zeros over ``w1``/``w2``/``w3`` via ``MLP.update``."""
    target_w1 = _build_zero_like(mlp, mlp.w1, _w1_w3_mesh_mapper(mlp))
    target_w2 = _build_zero_like(mlp, mlp.w2, _w2_mesh_mapper(mlp))
    target_w3 = _build_zero_like(mlp, mlp.w3, _w1_w3_mesh_mapper(mlp))
    mlp.update(w1=target_w1, w2=target_w2, w3=target_w3)


def _build_random_mlp_input(completer):
    """Construct a synthetic ``(1, 1, SEQ_LEN, dim)`` MLP input.

    Random values exercise the "activations multiply against weights"
    path properly -- if w1/w3 are exactly zero the product is exactly
    zero regardless of the activation magnitudes.
    """
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

    mlp = completer.model.layers[0].feed_forward
    print(f">>> mlp.w1 shape={tuple(mlp.w1.shape)} dtype={mlp.w1.dtype}")
    print(f">>> mlp.w2 shape={tuple(mlp.w2.shape)} dtype={mlp.w2.dtype}")
    print(f">>> mlp.w3 shape={tuple(mlp.w3.shape)} dtype={mlp.w3.dtype}")

    # ---- Step 1: zero out every MLP projection ----
    print()
    print("=== Step 1: MLP.update(w1=0, w2=0, w3=0) ===")
    zero_mlp(mlp)

    # ---- Step 2: build a synthetic input and run MLP.forward ----
    print()
    print(f"=== Step 2: forward synthetic random input shape (1, 1, {SEQ_LEN}, {completer.model_args.dim}) ===")
    x = _build_random_mlp_input(completer)
    out = mlp.forward(x, Mode.PREFILL)
    out_torch = ttnn.to_torch(out)
    print(f"  out shape  = {tuple(out_torch.shape)}")
    print(f"  out dtype  = {out_torch.dtype}")
    print(f"  out  max|.|= {float(out_torch.abs().max()):.6g}")
    print(f"  out  mean|.|={float(out_torch.abs().mean()):.6g}")

    # ---- Assertion ----
    print()
    print("=== assertion ===")
    expected = torch.zeros_like(out_torch)
    ok = torch.equal(out_torch, expected)
    print(f"  MLP.forward(x) == 0 elementwise: {ok}   [must be True]")
    if not ok:
        diff = out_torch.float().abs()
        print(f"  max |out| = {float(diff.max()):.6g}")
        print(f"  mean|out| = {float(diff.mean()):.6g}")

    print()
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")

    import ttml

    del completer
    gc.collect()
    ttml.autograd.AutoContext.get_instance().close_device()


if __name__ == "__main__":
    main()
