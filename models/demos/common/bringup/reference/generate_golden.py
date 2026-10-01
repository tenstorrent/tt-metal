# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Goldens step: run the CPU reference once over one ladder rung and store every boundary (format in golden.py).

    python -m models.demos.common.bringup.reference.generate_golden --spec S --rung s4096

The reference runs every layer up to the last selected one, because a layer's input depends on all layers before it;
layers after it are skipped (F40), and with them the model-level outputs (final norm, top-32, logits), which the ladder
only compares when the stack ends at the model's last layer. With a layer subset in the spec, only the selected layers'
boundaries and state are stored, and every chunk also stores the
block input of the first layer of each contiguous run of selected layers, so the device can restart there.
Boundaries are stored for every chunk if the rung sets ``full_dumps``, else only for the last chunk.
Rungs with ``golden: <other>`` reuse that rung's golden and are skipped here.

State: each layer stores its own names (``spec.state_names``). Tensors in ``state.fixed`` (recurrent state, conv tail)
do not grow along the sequence, so the state at a chunk start cannot be sliced from the final one: they are also stored
as the state before every dumped chunk (``kv_cache/layer_{i}_at_{start}``) and, on the serving-contract rung, after
``seq - tests.contract_tail_pad`` tokens (one extra partial chunk from the last chunk's snapshot).

Records golden_layers, golden_chunks, golden_hash_ok, cpu_seconds, text_top5_acc_last_chunk.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict

import torch
from safetensors.torch import save_file

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.core.metrics import cpu_threads
from models.demos.common.bringup.reference.golden import (
    LOGITS_TAIL,
    TOPK,
    content_hash,
    load_spec,
    rung_dir,
    store_dtype,
    text_tokens,
)


def run_starts(selected: list[int]) -> list[int]:
    """First layer of each contiguous run: [0, 1, 5, 6, 9] -> [0, 5, 9]."""
    return [layer for k, layer in enumerate(selected) if k == 0 or selected[k - 1] != layer - 1]


def reuse(spec, rung: dict, out) -> None:
    """A spec with a prior bring-up shares its goldens (same checkpoint and reference): check that this one is for the
    same rung and layers and is intact, and record the gate's metrics instead of regenerating it."""
    from models.demos.common.bringup.reference.golden import Golden

    m = json.loads((out / "manifest.json").read_text())
    want = {"seq": rung["seq"], "chunk": rung["chunk"], "layers": spec.layers(), "model": spec.data["hf_id"]}
    bad = {k: (m.get(k), v) for k, v in want.items() if m.get(k) != v}
    if bad:
        raise SystemExit(f"{out}: the prior's golden does not match this rung (got, want): {bad}")
    metrics.record("golden_layers", len(m["layers"]))
    metrics.record("golden_chunks", m["n_chunks"])
    metrics.record("golden_hash_ok", int(Golden(out).verify()))
    metrics.record("golden_reused", 1)
    print(f"reused the prior bring-up's golden {out} (content_hash {m.get('content_hash')})")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--rung", required=True)
    a = ap.parse_args(argv)
    torch.set_num_threads(cpu_threads())
    spec = load_spec(a.spec)
    rung = spec.rung(a.rung)
    if rung.get("golden"):
        print(f"rung {a.rung} reuses the golden of rung {rung['golden']}; nothing to generate")
        return
    seq, chunk = rung["seq"], rung["chunk"]
    out = rung_dir(spec, rung)
    if (out / "manifest.json").exists():
        if spec.prior:
            return reuse(spec, rung, out)
        raise SystemExit(f"{out} exists; goldens are generated once. Delete it to regenerate.")
    tmp = out.with_name(out.name + ".partial")
    (tmp / "kv_cache").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    selected = spec.layers()
    full_stack = max(selected) == spec.num_layers - 1
    run_layers = None if full_stack else list(range(max(selected) + 1))
    ref = spec.hooks().reference(spec, layers=run_layers, dtype=torch.float32)
    starts = set(run_starts(selected))
    keep_fp32 = tuple(spec.get("golden.keep_fp32", []))
    from models.demos.common.bringup.reference import prompt

    prompt_rec = prompt.load(spec)
    if prompt_rec is None:
        raise SystemExit(f"no canonical prompt at {prompt.prompt_path(spec)}; the intake step (R.1) builds it")
    tokens = text_tokens(spec, seq)
    n_chunks = seq // chunk
    dumped = list(range(n_chunks)) if rung.get("full_dumps") else [n_chunks - 1]
    print(f"reference loaded in {time.time() - t0:.0f}s; seq={seq} chunk={chunk}; storing layers {selected}")

    state = ref.new_state(seq)
    fixed = spec.state_fixed
    tail_at = None
    if fixed:
        from models.demos.common.bringup.testing.contract import contract_rung

        if contract_rung(spec) == a.rung:
            tail_at = seq - int(spec.get("tests.contract_tail_pad"))

    def snapshot(at: int) -> None:
        for i in selected:
            names = [n for n in spec.state_names(i) if n in fixed]
            if names:
                st = ref.state_tensors(state, i, at)
                save_file(
                    {f"{n}_cache_layer_{i}": store_dtype(st[n].clone(), keep_fp32, n) for n in names},
                    str(tmp / "kv_cache" / f"layer_{i}_at_{at}.safetensors"),
                )

    snapshots, before_last = [], None
    chunk_times, top1, top5 = [], [], []
    for c in range(n_chunks):
        per_layer, model_t = defaultdict(dict), {}
        dump = c in dumped
        if fixed and dump:
            snapshot(c * chunk)
            snapshots.append(c * chunk)
        if tail_at is not None and c == n_chunks - 1:
            before_last = copy.deepcopy(state)

        def rec(name, t, per_layer=per_layer, model_t=model_t, dump=dump):
            if name.startswith("L"):
                li, key = name[1:].split(".", 1)
                li = int(li)
                if li in selected and (dump or (key == "in" and li in starts)):
                    per_layer[li][key] = store_dtype(t.detach().clone(), keep_fp32, key)
            elif name != "logits" and (full_stack or name != "final_norm"):
                model_t[name] = store_dtype(t.detach().clone(), keep_fp32, name)

        s = c * chunk
        tc = time.time()
        _, logits = ref.forward_chunk(tokens[s : s + chunk], s, state, rec, logits_last_n=chunk if full_stack else 0)
        chunk_times.append(time.time() - tc)
        model_t["tokens"] = tokens[s : s + chunk].to(torch.int32).contiguous()
        if full_stack:
            logits = logits.float()
            vals, ids = torch.topk(logits, TOPK, dim=-1)
            model_t.update(
                top32_values=vals.contiguous(),
                top32_ids=ids.to(torch.int32).contiguous(),
                logits_tail=logits[-LOGITS_TAIL:].contiguous(),
            )
            nxt = tokens[s + 1 : s + chunk + 1]
            top1.append((ids[: nxt.shape[0], 0] == nxt).float().mean().item())
            top5.append((ids[: nxt.shape[0], :5] == nxt[:, None]).any(-1).float().mean().item())
        cdir = tmp / f"chunk_{c:02d}"
        cdir.mkdir(exist_ok=True)
        save_file(model_t, str(cdir / "model.safetensors"))
        for li, tensors in per_layer.items():
            save_file(tensors, str(cdir / f"layer_{li:02d}.safetensors"))
        print(
            f"chunk {c + 1}/{n_chunks} [{s},{s + chunk}) {chunk_times[-1]:.0f}s "
            + (f"top1={top1[-1]:.3f} top5={top5[-1]:.3f} " if full_stack else "")
            + f"layers stored={len(per_layer)}",
            flush=True,
        )

    for i in selected:
        st = ref.state_tensors(state, i, seq)
        save_file(
            {f"{n}_cache_layer_{i}": store_dtype(st[n], keep_fp32, n) for n in spec.state_names(i)},
            str(tmp / "kv_cache" / f"layer_{i}.safetensors"),
        )
    if before_last is not None:  # the contract's last chunk ends early: fixed state after seq - tail tokens
        s = (n_chunks - 1) * chunk
        state = before_last
        ref.forward_chunk(tokens[s:tail_at], s, state, lambda n, t: None, logits_last_n=0)
        snapshot(tail_at)
        snapshots.append(tail_at)

    (tmp / "metadata.json").write_text(
        json.dumps(
            {
                "model": spec.data["hf_id"],
                "token_ids": tokens.tolist(),
                "num_layers": len(selected),
                "layers": selected,
                "state_tensors": list(spec.get("state.tensors") or []),
                "state_by_layer": {str(i): spec.state_names(i) for i in selected},
                "state_fixed": fixed,
                "state_snapshots": snapshots,
                "kv_cache_format": spec.get("state.format", "separate_k_v"),
                "k_rope_layout": spec.get("state.k_rope_layout"),
                "seq_len": seq,
            }
        )
    )
    manifest = {
        "model": spec.data["hf_id"],
        "rung": a.rung,
        "seq": seq,
        "chunk": chunk,
        "n_chunks": n_chunks,
        "layers": selected,
        "subset": selected != list(range(spec.num_layers)),
        "run_starts": sorted(starts),
        "full_dumps": bool(rung.get("full_dumps")),
        "dumped_chunks": dumped,
        "compute_dtype": "float32",
        "text": prompt_rec["source"],
        "prompt_wrap": prompt_rec["wrap"],
        "prompt_sha256": prompt_rec["sha256"],
        "chunk_times_s": [round(t, 1) for t in chunk_times],
        "text_top1_acc": top1,
        "text_top5_acc": top5,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest["content_hash"] = content_hash(tmp)
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
    tmp.rename(out)  # atomic: an interrupted run never leaves a half-written golden under the real name

    from models.demos.common.bringup.reference.golden import Golden

    metrics.record("golden_layers", len(selected))
    metrics.record("golden_chunks", n_chunks)
    metrics.record("golden_hash_ok", int(Golden(out).verify()))
    metrics.record("cpu_seconds", round(time.time() - t0, 1))
    if top5:
        metrics.record("text_top5_acc_last_chunk", top5[-1])
    print(f"done in {time.time() - t0:.0f}s -> {out}\ncontent_hash {manifest['content_hash']}")


if __name__ == "__main__":
    main()
