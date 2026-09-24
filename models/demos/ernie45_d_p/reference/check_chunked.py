# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gate P1.3: reference chunked prefill == one-shot prefill (hidden states and KV cache)."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from models.demos.ernie45_d_p.bringup import metrics  # noqa: E402
from models.demos.ernie45_d_p.reference.ernie_ref import ErnieReference, load_book_tokens, pcc  # noqa: E402

TASK = os.environ.get("ERNIE_BRINGUP_TASK", "P1.3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=None, help="limit to first N layers (debug)")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    from transformers import AutoTokenizer

    layers = list(range(a.layers)) if a.layers else None
    ref = ErnieReference(dtype=torch.float32, layers=layers)
    tokens = load_book_tokens(AutoTokenizer.from_pretrained(ref.model_path), a.seq)

    full_h, _, full_cache = ref.prefill(tokens, chunk_size=a.seq, logits_last_n=0)
    ch_h, _, ch_cache = ref.prefill(tokens, chunk_size=a.chunk, logits_last_n=0)

    p_h = pcc(full_h, ch_h)
    p_k = min(pcc(full_cache.k[i], ch_cache.k[i]) for i in ref.layer_ids)
    p_v = min(pcc(full_cache.v[i], ch_cache.v[i]) for i in ref.layer_ids)
    print(
        f"hidden pcc={p_h:.8f} maxabs={(full_h - ch_h).abs().max():.3e}  kv_k min pcc={p_k:.8f}  kv_v min pcc={p_v:.8f}"
    )
    metrics.record(TASK, "pcc_hidden", p_h)
    metrics.record(TASK, "pcc_kv_k", p_k)
    metrics.record(TASK, "pcc_kv_v", p_v)


if __name__ == "__main__":
    main()
