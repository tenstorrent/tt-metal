# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Check the streaming reference against HF ``HYV4ForCausalLM`` (transformers >= 5.23, which ships ``hy_v4``).

The tt-metal env pins an older transformers without ``hy_v4``, so this runs in any env that has it::

    python -m models.demos.deepseek_v3_d_p.reference.hy4.hf_check                      # random tiny checkpoint
    python -m models.demos.deepseek_v3_d_p.reference.hy4.hf_check --model-dir /mnt/tt-data/Hy4-Preview \\
        --num-layers 2 --seq-len 96                                                   # real weights, truncated

HF runs eager in fp32 and densely (a full [S, S] mask), so keep ``--seq-len`` small. It compares, per
layer: the KVPE line, the index key, the top-k selection (as sets, over causally valid positions), and the
iHC streams; and the next-token logits when the full stack runs. Exit status is non-zero on mismatch.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.reference.hy4.checkpoint import Hy4Checkpoint, save_random_checkpoint
from models.demos.deepseek_v3_d_p.reference.hy4.config import tiny_random_config
from models.demos.deepseek_v3_d_p.reference.hy4.golden import pcc, topk_agreement
from models.demos.deepseek_v3_d_p.reference.hy4.model import Hy4PrefillReference


def hf_hy_v4_available() -> bool:
    try:
        from transformers import HYV4ForCausalLM  # noqa: F401
    except Exception:
        return False
    return True


class _Capture:
    def __init__(self):
        self.layers = {}
        self.logits = None

    def on_layer(self, layer, *, kvpe, index_k=None, topk=None, streams=None):
        self.layers[layer] = dict(
            kvpe=kvpe.clone(),
            index_k=None if index_k is None else index_k.clone(),
            topk=None if topk is None else topk.clone(),
            streams=streams.clone(),
        )

    def on_final(self, *, logits, position):
        self.logits = logits.clone()


def run_hf(model_dir: Path, token_ids: torch.Tensor, num_layers: int | None):
    from transformers import AutoConfig, HYV4ForCausalLM

    config = AutoConfig.from_pretrained(str(model_dir))
    if num_layers is not None and num_layers < config.num_hidden_layers:
        config.num_hidden_layers = num_layers
        config.indexer_types = config.indexer_types[:num_layers]
        config.mlp_layer_types = config.mlp_layer_types[:num_layers]
        config.layer_types = config.layer_types[:num_layers]
    model = HYV4ForCausalLM.from_pretrained(
        str(model_dir), config=config, dtype=torch.float32, attn_implementation="eager"
    ).eval()

    topk, streams = {}, {}
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.register_forward_hook(lambda m, a, o, i=i: topk.__setitem__(i, o[2][0].clone())))
        hooks.append(layer.register_forward_hook(lambda m, a, o, i=i: streams.__setitem__(i, o[0][0].clone())))
    with torch.no_grad():
        out = model(token_ids.view(1, -1), use_cache=True)
    for h in hooks:
        h.remove()
    cache = out.past_key_values
    layers = {}
    for i in range(config.num_hidden_layers):
        cl = cache.layers[i]
        kvpe = torch.cat([cl.keys[0, 0], cl.values[0, 0]], dim=-1)
        idx_k = cl.indexer_keys[0] if config.indexer_types[i] == "full" else None
        layers[i] = dict(kvpe=kvpe, index_k=idx_k, topk=topk[i], streams=streams[i])
    return layers, out.logits[0, -1].float(), config.num_hidden_layers


def compare(ours: _Capture, hf_layers: dict, hf_logits, full_stack: bool, *, verbose=True) -> list[str]:
    failures = []
    for i in sorted(hf_layers):
        h, o = hf_layers[i], ours.layers[i]
        p_kv = pcc(h["kvpe"], o["kvpe"])
        d_kv = (h["kvpe"] - o["kvpe"]).abs().max().item()
        p_st = pcc(h["streams"], o["streams"])
        line = f"layer {i}: kvpe pcc={p_kv:.7f} maxdiff={d_kv:.2e} streams pcc={p_st:.7f}"
        if h["index_k"] is not None:
            p_ik = pcc(h["index_k"], o["index_k"])
            S = h["topk"].shape[0]
            hf_tk = h["topk"].long()
            # HF keeps causally masked picks for rows with fewer than top-k valid keys; drop them.
            hf_tk = torch.where(hf_tk <= torch.arange(S).unsqueeze(1), hf_tk, torch.full_like(hf_tk, -1))
            agree = topk_agreement(hf_tk, o["topk"]).min().item()
            line += f" index_k pcc={p_ik:.7f} topk agreement(min row)={agree:.4f}"
            if p_ik < 0.99999 or agree < 1.0:
                failures.append(f"layer {i} indexer: pcc={p_ik} topk={agree}")
        if p_kv < 0.99999 or p_st < 0.99999:
            failures.append(f"layer {i}: kvpe pcc={p_kv} streams pcc={p_st}")
        if verbose:
            print(line, flush=True)
    if full_stack and ours.logits is not None:
        p_lg = pcc(hf_logits, ours.logits)
        same = int(hf_logits.argmax()) == int(ours.logits.argmax())
        print(f"final logits pcc={p_lg:.7f} argmax match={same}", flush=True)
        if p_lg < 0.9999 or not same:
            failures.append(f"logits pcc={p_lg} argmax match={same}")
    return failures


def check(model_dir: Path, seq_len: int, num_layers: int | None, seed: int = 0, block: int | None = None) -> list[str]:
    ckpt = Hy4Checkpoint(model_dir)
    cfg = ckpt.config
    n = cfg.num_hidden_layers if num_layers is None else min(num_layers, cfg.num_hidden_layers)
    cfg_run = cfg.truncated(n)
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, cfg.vocab_size, (seq_len,), generator=g)
    ours = _Capture()
    kw = {} if block is None else dict(attn_block=block, proj_block=block)
    Hy4PrefillReference(ckpt, cfg_run, prefetch=False, **kw).run(ids, ours, num_layers=n)
    hf_layers, hf_logits, _ = run_hf(model_dir, ids, n)
    full_stack = n == cfg.num_hidden_layers
    if not full_stack:
        ours.logits = None
    return compare(ours, hf_layers, hf_logits, full_stack)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=None, help="checkpoint dir (default: a random tiny checkpoint)")
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if not hf_hy_v4_available():
        print("transformers in this env has no hy_v4 (needs >= 5.23)", file=sys.stderr)
        return 2
    torch.manual_seed(args.seed)
    if args.model_dir is None:
        with tempfile.TemporaryDirectory() as tmp:
            save_random_checkpoint(tmp, tiny_random_config(), seed=args.seed)
            failures = check(Path(tmp), args.seq_len, args.num_layers, args.seed)
    else:
        failures = check(Path(args.model_dir), args.seq_len, args.num_layers, args.seed)
    print("PASS" if not failures else "FAIL:\n  " + "\n  ".join(failures), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
