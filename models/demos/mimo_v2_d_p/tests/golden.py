# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF-chain golden for the first ``n_layers`` MiMo-V2 layers on a real prompt (fp32), cached to disk.

{"ids": [S], "hidden": [n_layers+1, S, H] bf16 (0 = embeddings), "kv": {layer: (K [1,nkv,S,192] post-rope, V [1,nkv,S,128] scaled)}}
"""

from pathlib import Path

import torch
from loguru import logger

from models.demos.mimo_v2_d_p.reference import hf
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.reference.remote_st import LOCAL
from models.demos.mimo_v2_d_p.reference.weights import global_state, layer_state

GOLDEN_DIR = LOCAL / "golden"


@torch.no_grad()
def golden(n_layers: int, seq: int, with_kv: bool = True):
    path = GOLDEN_DIR / f"chain_L{n_layers}_S{seq}.pt"
    if path.exists():
        return torch.load(path)
    cfg = MiMoTextConfig.from_json()
    hcfg = hf.hf_config()
    hf.use_blocked_attention()
    ids = hf.tokenize_prompt(seq)
    x = global_state()["embed_tokens.weight"][ids][None].float()
    hidden, kv = [x[0].bfloat16()], {}
    for i in range(n_layers):
        spec = cfg.layer_attn(i)
        layer = hf.decoder_layer(i, layer_state(i, cfg), hcfg)
        x = hf.run_layer(layer, x, spec.window is not None, hcfg, window=spec.window)
        del layer
        hidden.append(x[0].bfloat16())
        if with_kv:
            k, v = hf.KV_CAPTURE.pop(i)
            kv[i] = (k.bfloat16(), v.bfloat16())
        logger.info(f"golden layer {i} ({spec.kind}) done, |x|={x.std().item():.4f}")
    out = {"ids": ids, "hidden": torch.stack(hidden), "kv": kv}
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return out


if __name__ == "__main__":
    import sys

    golden(int(sys.argv[1]), int(sys.argv[2]))
