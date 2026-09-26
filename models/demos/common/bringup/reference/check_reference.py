# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Reference gate, part 2: the reference is self-consistent.

    python -m models.demos.common.bringup.reference.check_reference --spec S --seq 4096 --chunk 2048

1. Chunked prefill equals one-shot prefill: final hidden and every state tensor of every layer (pcc_hidden,
   pcc_state_min, maxabs_hidden).
2. Every block type's graph is valid (graph_errors == 0) and truthful: replaying the representative layer's block
   through run_block, from the recorded block input and a state holding the recorded prefix, reproduces every
   recorded boundary of the last chunk exactly (graph_replay_maxabs, boundaries_missing).
"""

from __future__ import annotations

import argparse

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.core.metrics import cpu_threads
from models.demos.common.bringup.reference.golden import load_spec, text_tokens
from models.demos.common.bringup.reference.interface import boundary_names, run_block, validate_graph


def prefill(ref, tokens, chunk, rec_for_chunk=None):
    state = ref.new_state(tokens.shape[0])
    outs = []
    for c, s in enumerate(range(0, tokens.shape[0], chunk)):
        rec = rec_for_chunk(c) if rec_for_chunk else (lambda n, t: None)
        h, _ = ref.forward_chunk(tokens[s : s + chunk], s, state, rec, logits_last_n=0)
        outs.append(h)
    return torch.cat(outs), state


def write_graphs(ref, reps: dict) -> None:
    """results/block_graphs.json: each block type's steps, for the dashboard's model graph."""
    import json

    out = {
        bt: [
            {"name": s.name, "inputs": list(s.inputs), "output": s.output, "kind": s.kind, "stateful": s.stateful}
            for s in ref.block_graph(li)
        ]
        for bt, li in reps.items()
    }
    p = metrics.results_dir() / "block_graphs.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=2048)
    a = ap.parse_args(argv)
    assert a.seq % a.chunk == 0 and a.seq // a.chunk >= 2, "need at least two chunks"
    torch.set_num_threads(cpu_threads())
    spec = load_spec(a.spec)
    layers = spec.layers()
    ref = spec.hooks().reference(spec, layers=layers, dtype=torch.float32)
    tokens = text_tokens(spec, a.seq)
    last = a.seq // a.chunk - 1

    reps = {bt: spec.representative_layer(bt) for bt in spec.data["block_types"]}
    rep_layers = set(reps.values())
    recorded = {}

    def rec_for_chunk(c):
        def rec(name, t):
            if c == last and name.startswith("L") and int(name[1:].split(".")[0]) in rep_layers:
                recorded[name] = t.detach().clone()

        return rec

    full_h, full_state = prefill(ref, tokens, a.seq)
    ch_h, ch_state = prefill(ref, tokens, a.chunk, rec_for_chunk)
    p_h = metrics.pcc(full_h, ch_h)
    p_state = min(
        metrics.pcc(x, ch_t[n])
        for i in layers
        for ch_t in [ref.state_tensors(ch_state, i, a.seq)]
        for n, x in ref.state_tensors(full_state, i, a.seq).items()
    )
    metrics.record("pcc_hidden", p_h)
    metrics.record("pcc_state_min", p_state)
    metrics.record("maxabs_hidden", float((full_h - ch_h).abs().max()))
    print(f"chunked vs one-shot: hidden pcc={p_h:.9f} state min pcc={p_state:.9f}")

    graph_errs, worst, missing = 0, 0.0, 0
    start = last * a.chunk
    for bt, li in reps.items():
        steps = ref.block_graph(li)
        errs = validate_graph(steps)
        graph_errs += len(errs)
        for e in errs:
            print(f"graph {bt}: {e}")
        if errs:
            continue
        names = [f"L{li}.{b}" for b in boundary_names(steps)]
        miss = [n for n in names if n not in recorded]
        missing += len(miss)
        if miss:
            print(f"graph {bt} (layer {li}): forward_chunk did not record {miss}")
            continue
        # Fresh state holding only the prefix [0, start) of this layer, then replay the last chunk's block.
        state = ref.new_state(a.seq)
        prefix = {n: t for n, t in ref.state_tensors(ch_state, li, start).items()}
        ref.load_state(state, li, prefix, start)
        ctx = ref.chunk_context(li, start, a.chunk, state)
        replayed = {}
        run_block(
            steps,
            lambda name: ref.component(li, name),
            ctx,
            recorded[f"L{li}.in"],
            rec=lambda n, t: replayed.__setitem__(n, t),
            prefix=f"L{li}.",
        )
        for n in names:
            d = float((replayed[n].float() - recorded[n].float()).abs().max())
            worst = max(worst, d)
            if d > 0:
                print(f"graph {bt} (layer {li}): replayed {n} differs by {d:.3e}")
        print(f"graph {bt} (layer {li}): {len(steps)} steps replayed, max abs diff {worst:.3e}")
    write_graphs(ref, reps)
    metrics.record("graph_errors", graph_errs)
    metrics.record("boundaries_missing", missing)
    metrics.record("graph_replay_maxabs", worst)


if __name__ == "__main__":
    main()
