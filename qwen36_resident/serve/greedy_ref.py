# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct (no vLLM) greedy decode of the serving prompts (prompts.py), the reference for chat_client.py --ref.

The model is built as e0_bringup.run_real (and generator_vllm) builds it. Per prompt, two decodes of max_tokens:
host: the pipeline-parallel prefill of all but the last prompt token, then one step per token on the host-written
    token state, each next token the argmax of that step's full logits (the reference). Each step's top-1 - top-2
    logit gap is kept, to tell a near-tie divergence from a real one.
device: the same prefill and first token, then the hub draws greedy tokens on device and feeds them back (the
    served path for temperature 0); the host reads each step's fed (token, position).

usage: python -m qwen36_resident.serve.greedy_ref --out ref.json [--max-tokens 128] [--max-len 32768]
"""
import argparse
import faulthandler
import json
import sys
import time

import torch
from transformers import AutoTokenizer

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.model import Dims, ResidentModel, State, is_attn
from qwen36_resident.prefill.pp_prefill import PPPrefill
from qwen36_resident.serve.prompts import PROMPTS, chat_ids
from qwen36_resident.tests.e0_bringup import PREFILL_CHUNKS, close_mesh, open_mesh


def build(mesh, ck, max_seq_len):
    cfg = ck.config
    layers, interval, vocab = cfg["num_hidden_layers"], cfg["full_attention_interval"], cfg["vocab_size"]
    d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x, vocab=vocab)
    C = max(PREFILL_CHUNKS)
    max_len = -(-max_seq_len // C) * C
    n_attn = sum(is_attn(l, interval) for l in range(layers))
    st = State(d, layers - n_attn, n_attn, 0, max_pos=max_len + 64, zero=True)
    pp = PPPrefill(mesh, ck, layers, interval, chunk=list(PREFILL_CHUNKS), max_len=max_len)
    w = Q.load(ck, d, layers, interval)
    emb = Q.embedding(ck)
    model = ResidentModel(mesh, d, w, st, layers, interval, lm_head=True, embed=pp.embed)
    del w
    model.step(model.token_state(emb[0].float(), 0))
    ttnn.synchronize_device(mesh)
    model.reset()
    model.capture_trace()
    pp.capture()
    pp.handoff(model, 1)
    ttnn.synchronize_device(mesh)
    return d, emb, pp, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=32768)
    a = ap.parse_args()
    faulthandler.dump_traceback_later(900, repeat=True, file=sys.stderr)
    path = Q.checkpoint_dir()
    ck = Q.Checkpoint(path)
    tok = AutoTokenizer.from_pretrained(path)
    vocab = ck.config["vocab_size"]
    mesh = open_mesh()
    rows = []
    try:
        t0 = time.time()
        d, emb, pp, model = build(mesh, ck, a.max_len)
        print(f"built in {time.time() - t0:.0f} s", flush=True)

        def logits_at(token, pos):
            model.step(model.token_state(emb[int(token)].float(), int(pos)))
            ttnn.synchronize_device(mesh)
            return model.logits()[:, : d.vocab_chip].reshape(-1)[:vocab].float()

        def prefill(ids):
            model.reset()
            pp.reset()
            T = len(ids)
            if T > 1:
                pp.run(torch.tensor(ids[: T - 1]))
                pp.handoff(model, T - 1)
            return logits_at(ids[T - 1], T - 1)

        def pick(lg):
            # argmax takes the first index on ties, as the hub and vLLM's greedy sampler do (topk's order is not)
            v = lg.topk(2).values
            return int(lg.argmax()), float(v[0] - v[1])

        for p in PROMPTS:
            ids = chat_ids(tok, p)
            T, N = len(ids), a.max_tokens
            # host: argmax of each step's logits
            t, gap = pick(prefill(ids))
            host, gaps = [t], [gap]
            for k in range(1, N):
                t, gap = pick(logits_at(host[-1], T - 1 + k))
                host.append(t)
                gaps.append(gap)
            # device: the hub's greedy draw, fed back on device
            t1, _ = pick(prefill(ids))
            model.set_sampling(0.0, 1, 1.0, 0)
            model.step(model.token_state(emb[t1].float(), T))
            dev, dev_pos = [t1], []
            for k in range(1, N):
                if k > 1:
                    model.step(None)
                ttnn.synchronize_device(mesh)
                t, q = model.fed_token()
                dev.append(t)
                dev_pos.append(q)
            first_diff = next((k for k in range(N) if host[k] != dev[k]), None)
            rows.append(
                dict(
                    name=p["name"],
                    thinking=p["thinking"],
                    prompt_ids=ids,
                    host_tokens=host,
                    host_gaps=gaps,
                    device_tokens=dev,
                    device_positions=dev_pos,
                    host_device_first_diff=first_diff,
                    host_text=tok.decode(host, skip_special_tokens=True),
                    device_text=tok.decode(dev, skip_special_tokens=True),
                )
            )
            pos_ok = dev_pos == list(range(T + 1, T + N))
            print(
                f"{p['name']}: T={T} host/device first diff {first_diff} (positions {'ok' if pos_ok else 'WRONG'}), "
                f"min gap {min(gaps):.4f}\n  {rows[-1]['host_text'][:200]!r}",
                flush=True,
            )
    finally:
        close_mesh(mesh)
    json.dump(dict(model=path, max_tokens=a.max_tokens, prompts=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
