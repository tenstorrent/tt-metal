#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Construction-equivalence test for ``MLP.update``.

Round-trip on every MLP layer:

    1.  Generate greedily with the real, ``__init__``-loaded weights -> ``tokens_A``.
    2.  Snapshot every layer's ``w1``/``w2``/``w3`` to torch.
    3.  Overwrite every layer with a constant (deliberately break the model).
    4.  Generate again with the broken weights -> ``tokens_broken`` (sanity:
        must differ from ``tokens_A``, otherwise the overwrite was a no-op
        and the rest of the test is meaningless).
    5.  Restore every layer via ``MLP.update(w1=..., w2=..., w3=...)``.
    6.  Generate again -> ``tokens_B``.
    7.  Assert ``tokens_A == tokens_B`` byte-for-byte.

The claim: after ``update()``, the on-device MLP state is
indistinguishable from the state ``__init__`` produced for the same
weights -- "construction equivalence". A greedy generation through the
full Llama-3.2-1B-Instruct stack is the cheapest way to exercise every
consumer of the buffers (prefill matmul, decode matmul, captured traces).

We use real Llama-3.2-1B-Instruct weights (HF auth required) and the same
prompt as ``test_attention_update_construction.py``.
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
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.0  # greedy decoding -> deterministic, byte-comparable

# Same prompt as test_attention_update_construction.py.
PROMPT = "Explain a tensor in a paragraph."

# Bf16-exact constant we splat over w1/w2/w3 to deliberately break the
# model in between the snapshot and restore.
OVERWRITE_VALUE = 0.0


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


def _ttnn_like(template, torch_t, mesh_device, mesh_mapper):
    """Push a torch tensor onto device with the same dtype/layout/memcfg as ``template``."""
    import ttnn

    return ttnn.from_torch(
        torch_t,
        dtype=template.dtype,
        layout=template.layout,
        device=mesh_device,
        memory_config=template.memory_config(),
        mesh_mapper=mesh_mapper,
    )


def _build_constant_like(mlp, template, value, mesh_mapper):
    """Build a constant-valued ``ttnn.Tensor`` shaped like ``template``."""
    import torch
    import ttnn

    shape = tuple(template.shape)
    torch_t = torch.full(shape, float(value), dtype=torch.bfloat16)
    return ttnn.from_torch(
        torch_t,
        dtype=template.dtype,
        layout=template.layout,
        device=mlp.mesh_device,
        memory_config=template.memory_config(),
        mesh_mapper=mesh_mapper,
    )


def snapshot_mlp(mlp):
    """Read ``w1``/``w2``/``w3`` back to torch."""
    import ttnn

    return {
        "w1": ttnn.to_torch(mlp.w1),
        "w2": ttnn.to_torch(mlp.w2),
        "w3": ttnn.to_torch(mlp.w3),
    }


def restore_mlp(mlp, snap) -> None:
    """Push the snapshot back through ``MLP.update`` -- the operation under test."""
    w1_t = _ttnn_like(mlp.w1, snap["w1"], mlp.mesh_device, _w1_w3_mesh_mapper(mlp))
    w2_t = _ttnn_like(mlp.w2, snap["w2"], mlp.mesh_device, _w2_mesh_mapper(mlp))
    w3_t = _ttnn_like(mlp.w3, snap["w3"], mlp.mesh_device, _w1_w3_mesh_mapper(mlp))
    mlp.update(w1=w1_t, w2=w2_t, w3=w3_t)


def overwrite_mlp(mlp, value: float) -> None:
    """Splat ``value`` over ``w1``/``w2``/``w3`` via ``MLP.update``."""
    target_w1 = _build_constant_like(mlp, mlp.w1, value, _w1_w3_mesh_mapper(mlp))
    target_w2 = _build_constant_like(mlp, mlp.w2, value, _w2_mesh_mapper(mlp))
    target_w3 = _build_constant_like(mlp, mlp.w3, value, _w1_w3_mesh_mapper(mlp))
    mlp.update(w1=target_w1, w2=target_w2, w3=target_w3)


def _generate(completer, prompt_ids):
    """Greedy single-prompt completion -> list[int]."""
    completions = completer.generate(
        [prompt_ids],
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
    return completions[0]


def main() -> None:
    import ttnn
    from ttml.common.config import DeviceConfig, load_config

    from utils.llama_completer_ttt import LlamaGRPOCompleter

    print(">>> set_fabric_config(FABRIC_2D)")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    raw = load_config(os.path.join(REPO_ROOT, TTML_DEVICE_CONFIG_REL))
    device_config = DeviceConfig(raw)

    print(f">>> building LlamaGRPOCompleter ({MODEL_ID}, real weights)")
    completer = LlamaGRPOCompleter(
        device_config=device_config,
        model_source=MODEL_ID,
        max_batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
    )

    n_layers = len(completer.model.layers)
    print(f">>> {n_layers} MLP layers")

    prompt_ids = completer.tokenizer.encode(PROMPT, add_special_tokens=True)
    print(f">>> prompt   = {PROMPT!r}")
    print(f">>> prompt_ids = {prompt_ids}")

    # ---- Phase A: reference generation with constructed weights ----
    print()
    print("=== Phase A: greedy generate with __init__-loaded weights ===")
    tokens_A = _generate(completer, prompt_ids)
    text_A = completer.tokenizer.decode(tokens_A, skip_special_tokens=True)
    print(f"  tokens_A = {tokens_A}")
    print(f"  text_A   = {text_A!r}")

    # ---- Phase B: snapshot every layer's w1/w2/w3 ----
    print()
    print(f"=== Phase B: snapshot w1/w2/w3 of all {n_layers} MLP layers ===")
    snapshots = [snapshot_mlp(layer.feed_forward) for layer in completer.model.layers]
    print(f"  snapshot[0]['w1'] shape={tuple(snapshots[0]['w1'].shape)} dtype={snapshots[0]['w1'].dtype}")
    print(f"  snapshot[0]['w2'] shape={tuple(snapshots[0]['w2'].shape)} dtype={snapshots[0]['w2'].dtype}")
    print(f"  snapshot[0]['w3'] shape={tuple(snapshots[0]['w3'].shape)} dtype={snapshots[0]['w3'].dtype}")

    # ---- Phase C: deliberately break every layer ----
    print()
    print(f"=== Phase C: overwrite every layer's w1/w2/w3 with constant {OVERWRITE_VALUE} ===")
    for layer in completer.model.layers:
        overwrite_mlp(layer.feed_forward, OVERWRITE_VALUE)

    print(">>> greedy generate with broken (constant) weights")
    tokens_broken = _generate(completer, prompt_ids)
    text_broken = completer.tokenizer.decode(tokens_broken, skip_special_tokens=True)
    print(f"  tokens_broken = {tokens_broken}")
    print(f"  text_broken   = {text_broken!r}")

    # ---- Phase D: restore every layer via update(snapshot) ----
    print()
    print(f"=== Phase D: restore every layer via MLP.update(snapshot) ===")
    for layer, snap in zip(completer.model.layers, snapshots):
        restore_mlp(layer.feed_forward, snap)

    # ---- Phase E: generate again with restored weights ----
    print()
    print("=== Phase E: greedy generate with update()-restored weights ===")
    tokens_B = _generate(completer, prompt_ids)
    text_B = completer.tokenizer.decode(tokens_B, skip_special_tokens=True)
    print(f"  tokens_B = {tokens_B}")
    print(f"  text_B   = {text_B!r}")

    # ---- Assertions ----
    print()
    print("=== assertions ===")
    broken_differs = tokens_broken != tokens_A
    equivalence_ok = tokens_B == tokens_A
    print(f"  tokens_broken != tokens_A   (overwrite was effective):  {broken_differs}   [must be True]")
    print(f"  tokens_B == tokens_A        (construction equivalence): {equivalence_ok}   [must be True]")

    print()
    all_pass = broken_differs and equivalence_ok
    print(f"  RESULT: {'PASS' if all_pass else 'FAIL'}")

    import ttml

    del completer
    gc.collect()
    ttml.autograd.AutoContext.get_instance().close_device()


if __name__ == "__main__":
    main()
