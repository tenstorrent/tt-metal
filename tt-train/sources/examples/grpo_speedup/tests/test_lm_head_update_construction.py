#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Construction-equivalence test for ``LMHead.update``.

Round-trip on the model's single ``LMHead``:

    1.  Generate greedily with the real, ``__init__``-loaded weights -> ``tokens_A``.
    2.  Snapshot the dram_sharded chunks back to a torch ``(V, H)`` tensor,
        i.e. the exact shape of ``state_dict["output.weight"]``.
    3.  Overwrite with a constant via ``LMHead.update``.
    4.  Generate again with the broken weights -> ``tokens_broken`` (sanity:
        must differ from ``tokens_A``, otherwise the overwrite was a no-op).
    5.  Restore via ``LMHead.update(snapshot)``.
    6.  Generate again -> ``tokens_B``.
    7.  Assert ``tokens_A == tokens_B`` byte-for-byte.

The snapshot path also doubles as a regression check on the
``_permute_and_pad_output_weights`` / ``_build_dram_sharded_output_weights``
factoring -- if those transformations are inconsistent with the inverse
we apply here, the restored model will not match the original.
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

# Same prompt as the attention/MLP construction tests.
PROMPT = "Explain a tensor in a paragraph."

# Constant we splat over the LM head to deliberately break it.
OVERWRITE_VALUE = 0.0


def snapshot_lm_head(lm_head):
    """Read ``output_weights_dram_sharded`` back into a torch ``(vocab_size, dim)`` tensor.

    Inverse of ``__init__``:
        chunks -> concat along vocab axis -> slice off padding -> permute (dim, V) -> (V, dim)

    For a multi-device mesh, ``ttnn.to_torch`` of a ``ShardTensorToMesh(dim=-1)``
    tensor returns the concatenated form across the mesh, so the
    column-concat below works regardless of ``num_devices``.
    """
    import torch
    import ttnn

    # Each chunk has logical shape (dim, n_cols_after_full_mesh_concat).
    chunk_torches = [ttnn.to_torch(chunk) for chunk in lm_head.output_weights_dram_sharded]

    permuted_padded = torch.cat(chunk_torches, dim=-1)  # (dim, padded_vocab_size)

    # Slice off the right-pad applied in _permute_and_pad_output_weights.
    permuted = permuted_padded[:, : lm_head.vocab_size]  # (dim, vocab_size)

    return permuted.permute(1, 0).contiguous().to(torch.bfloat16)  # (vocab_size, dim)


def overwrite_lm_head(lm_head, value: float) -> None:
    """Splat ``value`` over the LM head via ``LMHead.update``."""
    import torch

    V = lm_head.vocab_size
    H = lm_head.args.dim
    const_weight = torch.full((V, H), float(value), dtype=torch.bfloat16)
    lm_head.update(const_weight)


def restore_lm_head(lm_head, snap_torch) -> None:
    """Push the torch snapshot back through ``LMHead.update``."""
    lm_head.update(snap_torch)


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

    lm_head = completer.model.lm_head
    n_chunks = len(lm_head.output_weights_dram_sharded)
    print(
        f">>> lm_head: vocab={lm_head.vocab_size} padded={lm_head.padded_vocab_size} "
        f"dim={lm_head.args.dim} dram_sharded_chunks={n_chunks}"
    )
    for i, chunk in enumerate(lm_head.output_weights_dram_sharded):
        print(f"  chunk[{i}] shape={tuple(chunk.shape)} dtype={chunk.dtype}")

    prompt_ids = completer.tokenizer.encode(PROMPT, add_special_tokens=True)
    print(f">>> prompt   = {PROMPT!r}")
    print(f">>> prompt_ids = {prompt_ids}")

    # ---- Phase A: reference generation ----
    print()
    print("=== Phase A: greedy generate with __init__-loaded weights ===")
    tokens_A = _generate(completer, prompt_ids)
    text_A = completer.tokenizer.decode(tokens_A, skip_special_tokens=True)
    print(f"  tokens_A = {tokens_A}")
    print(f"  text_A   = {text_A!r}")

    # ---- Phase B: snapshot LM head as a torch (V, H) tensor ----
    print()
    print("=== Phase B: snapshot LMHead output weights to torch ===")
    snap_torch = snapshot_lm_head(lm_head)
    print(f"  snapshot shape={tuple(snap_torch.shape)} dtype={snap_torch.dtype}")

    # ---- Phase C: deliberately break the LM head ----
    print()
    print(f"=== Phase C: overwrite LMHead with constant {OVERWRITE_VALUE} ===")
    overwrite_lm_head(lm_head, OVERWRITE_VALUE)

    print(">>> greedy generate with broken (constant) LM head")
    tokens_broken = _generate(completer, prompt_ids)
    text_broken = completer.tokenizer.decode(tokens_broken, skip_special_tokens=True)
    print(f"  tokens_broken = {tokens_broken}")
    print(f"  text_broken   = {text_broken!r}")

    # ---- Phase D: restore via LMHead.update(snapshot) ----
    print()
    print("=== Phase D: restore LMHead via update(snapshot_torch) ===")
    restore_lm_head(lm_head, snap_torch)

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
