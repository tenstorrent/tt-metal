#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Construction-equivalence test for ``Attention.update``.

Round-trip on every attention layer:

    1.  Generate greedily with the real, ``__init__``-loaded weights -> ``tokens_A``.
    2.  Snapshot every layer's ``wqkv``/``wo`` to torch.
    3.  Overwrite every layer with a constant (deliberately break the model).
    4.  Generate again with the broken weights -> ``tokens_broken`` (sanity:
        must differ from ``tokens_A``, otherwise the overwrite was a no-op
        and the rest of the test is meaningless).
    5.  Restore every layer via ``Attention.update(snapshot)``.
    6.  Generate again -> ``tokens_B``.
    7.  Assert ``tokens_A == tokens_B`` byte-for-byte.

The claim: after ``update()``, the on-device state is indistinguishable
from the state ``__init__`` produced for the same weights -- "construction
equivalence". A greedy generation through the full Llama-3.2-1B-Instruct
stack is the cheapest way to exercise every consumer of the buffers
(prefill matmul, decode matmul, fused all-gather matmul if used, KV cache
attention, captured traces).

We use real Llama-3.2-1B-Instruct weights (HF auth required) and the same
prompt as ``test_collapse_embedding_forward.py``.
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

# Same prompt as test_collapse_embedding_forward.py.
PROMPT = "Explain a tensor in a paragraph."

# Bf16-exact constant we splat over wqkv/wo to deliberately break the model
# in between the snapshot and restore. The exact value doesn't matter -- it
# just has to be different from any real weight, so the model produces a
# different generation under broken weights.
OVERWRITE_VALUE = 0.0


def _wqkv_mesh_mapper(attn):
    """Mirror the mesh mapper used in ``Attention.__init__`` for ``self.wqkv``."""
    import ttnn

    return ttnn.ShardTensor2dMesh(
        attn.mesh_device,
        dims=(3, 2) if attn.TG else (2, 3),
        mesh_shape=attn.args.cluster_shape,
    )


def _wo_mesh_mapper(attn):
    """Mirror the mesh mapper used in ``Attention.__init__`` for ``self.wo``."""
    import ttnn

    if attn.use_fused_all_gather_matmul or attn.TG:
        return ttnn.ShardTensor2dMesh(
            attn.mesh_device,
            dims=(2, 3),
            mesh_shape=attn.args.cluster_shape,
        )
    return ttnn.ShardTensorToMesh(attn.mesh_device, dim=2)


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


def _build_constant_like(attn, template, value, mesh_mapper):
    """Build a constant-valued ``ttnn.Tensor`` shaped like ``template``."""
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


def snapshot_attention(attn):
    """Read ``wqkv`` and ``wo`` back to torch.

    We only snapshot what ``Attention.update`` is responsible for. The
    prefetcher mirror ``wo_sharded_ring`` is rederived from ``wo`` inside
    ``_update_wo``, so it doesn't need to be snapshotted independently.
    """
    import ttnn

    return {
        "wqkv": ttnn.to_torch(attn.wqkv),
        "wo": ttnn.to_torch(attn.wo),
    }


def restore_attention(attn, snap) -> None:
    """Push the snapshot back through ``Attention.update`` -- the operation under test."""
    wqkv_t = _ttnn_like(attn.wqkv, snap["wqkv"], attn.mesh_device, _wqkv_mesh_mapper(attn))
    wo_t = _ttnn_like(attn.wo, snap["wo"], attn.mesh_device, _wo_mesh_mapper(attn))
    attn.update(wqkv=wqkv_t, wo=wo_t)


def overwrite_attention(attn, value: float) -> None:
    """Splat ``value`` over ``wqkv`` and ``wo`` via ``Attention.update``."""
    target_wqkv = _build_constant_like(attn, attn.wqkv, value, _wqkv_mesh_mapper(attn))
    target_wo = _build_constant_like(attn, attn.wo, value, _wo_mesh_mapper(attn))
    attn.update(wqkv=target_wqkv, wo=target_wo)


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
    print(f">>> {n_layers} attention layers")

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

    # ---- Phase B: snapshot every layer's wqkv/wo ----
    print()
    print(f"=== Phase B: snapshot wqkv/wo of all {n_layers} attention layers ===")
    snapshots = [snapshot_attention(layer.attention) for layer in completer.model.layers]
    print(f"  snapshot[0]['wqkv'] shape={tuple(snapshots[0]['wqkv'].shape)} dtype={snapshots[0]['wqkv'].dtype}")
    print(f"  snapshot[0]['wo']   shape={tuple(snapshots[0]['wo'].shape)} dtype={snapshots[0]['wo'].dtype}")

    # ---- Phase C: deliberately break every layer ----
    print()
    print(f"=== Phase C: overwrite every layer's wqkv/wo with constant {OVERWRITE_VALUE} ===")
    for layer in completer.model.layers:
        overwrite_attention(layer.attention, OVERWRITE_VALUE)

    print(">>> greedy generate with broken (constant) weights")
    tokens_broken = _generate(completer, prompt_ids)
    text_broken = completer.tokenizer.decode(tokens_broken, skip_special_tokens=True)
    print(f"  tokens_broken = {tokens_broken}")
    print(f"  text_broken   = {text_broken!r}")

    # ---- Phase D: restore every layer via update(snapshot) ----
    print()
    print(f"=== Phase D: restore every layer via Attention.update(snapshot) ===")
    for layer, snap in zip(completer.model.layers, snapshots):
        restore_attention(layer.attention, snap)

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
