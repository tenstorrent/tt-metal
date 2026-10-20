# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A larger fp32 golden for accuracy comparisons: many positions instead of make_golden.py's 8 short prompts.

Sequences of ``--seq-len`` tokens are cut from C4 documents, skipping the first ``--skip`` documents (the expert GPTQ
calibrates on the start of the same shard). The fp32 reference runs them layer by layer (attention per sequence, the
FFN over every token at once) and the last layer's hidden states are saved; the scorer applies the final norm and LM
head (tests/test_model_error_tp8_dp4.py with MIMO_GOLDEN pointing here), which keeps the file small.

    python -m models.experimental.mimo_v2_6_flash.reference.make_golden_large --calib c4-train.00000-of-01024.json.gz \
        --out golden_large.pt
"""

import argparse
import gzip
import json
import time

import torch
from loguru import logger
from transformers import AutoTokenizer

from . import model as ref
from .checkpoint import MiMoCheckpoint, default_model_path
from .config import MiMoConfig


def sequences(tok, path, num, length, skip):
    out = []
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            ids = tok(json.loads(line)["text"])["input_ids"]
            if len(ids) >= length:
                out.append(torch.tensor(ids[:length]))
                if len(out) == num:
                    break
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", required=True, help="C4 json.gz shard")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seqs", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--skip", type=int, default=20000, help="documents to skip (calibration uses the first ones)")
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    path = default_model_path()
    cfg = MiMoConfig.from_model_path(path)
    ckpt = MiMoCheckpoint(path, cfg)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    seqs = sequences(tok, args.calib, args.seqs, args.seq_len, args.skip)
    embedding = ckpt.embedding()
    hidden = [embedding[t].float() for t in seqs]
    del embedding
    for i in range(cfg.num_hidden_layers):
        t0 = time.time()
        layer = ref.ReferenceDecoderLayer(cfg, i, ckpt=ckpt)
        w = layer.w
        with torch.no_grad():
            post = []
            for s, t in enumerate(seqs):
                hn = ref.rms_norm(hidden[s], w["input_layernorm"], cfg.layernorm_epsilon)
                hidden[s] = hidden[s] + ref.attention(
                    hn, w["attn"], layer.dims, cfg.attention_value_scale, torch.arange(t.numel())
                )
                post.append(ref.rms_norm(hidden[s], w["post_attention_layernorm"], cfg.layernorm_epsilon))
            h = torch.cat(post)
            ffn = ref.moe(h, w["router"], w["experts"], cfg) if layer.is_moe else ref.dense_mlp(h, w["mlp"])
            for s, part in enumerate(torch.split(ffn, [t.numel() for t in seqs])):
                hidden[s] = hidden[s] + part
        del layer
        logger.info(f"layer {i}: {time.time() - t0:.0f}s")
    torch.save(
        {"token_ids": seqs, "final_hidden": hidden, "num_layers": cfg.num_hidden_layers, "source": args.calib},
        args.out,
    )
    logger.info(f"{len(seqs)} sequences x {args.seq_len} tokens -> {args.out}")


if __name__ == "__main__":
    main()
