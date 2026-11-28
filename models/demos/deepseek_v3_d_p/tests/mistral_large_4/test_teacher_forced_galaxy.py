# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Mistral-Large-4-1T-A52B on a Blackhole Galaxy: teacher-forced per-layer check against the CPU reference golden.

Each target layer i runs alone on the device with the golden's inputs, chunk by chunk: its hidden-state input is the
golden's output of layer i - 1 (``L<i-1>``), and every cross-layer value it reads is the golden's, not the device's:

  * its own KV / index-key prefix [0, chunk start) (the earlier chunk's rows, golden kv_cache) is written into its cache
    regions before the chunk (State.load_prefix), replacing what the device wrote on the earlier chunk;
  * a shared-KV layer's source layer (35) never runs: its cache rows [0, chunk end) and its bf16 rows of the chunk
    (TtAttention.chunk_kv) are the golden's;
  * a shared-index layer's top-k is the golden's top-k of its source full layer (``L<src>.topk``), put where the source
    indexer leaves it (TtModel._topk).

Compared per (layer, chunk): the output (PCC; ``delta``: the block's contribution out - in, against the golden's), the
rows the layer writes into its own caches (kv_latent, k_rope, index_key vs the golden kv_cache) and, on a full-indexer
layer, its top-k against the golden's (per-row recall). The device runs with its normal settings (BF4 experts).

    ML4_TF_GALAXY=1 ML4_TF_GOLDENS=fp=<dir>,bf4=<dir> [ML4_TF_LAYERS=36,40,...] \
    python -m pytest models/demos/deepseek_v3_d_p/tests/mistral_large_4/test_teacher_forced_galaxy.py

Goldens: cpu_golden.py --state --tf-targets <layers> (the layers default to the first golden's tf_targets); one pass
per golden, in the order given, each forced with that golden's inputs. Results go to ML4_TF_OUT (JSON), rewritten after
every layer. Skipped unless ML4_TF_GALAXY=1.
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

from models.demos.deepseek_v3_d_p.tests.mistral_large_4.test_e2e_galaxy import (  # noqa: F401  (mesh_device: fixture)
    CAPACITY,
    CKPT,
    _cmp,
    _pcc,
    _Spec,
    mesh_device,
)
from models.demos.deepseek_v3_d_p.tt.mistral_large_4 import hooks
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.hooks import device_model

OUT = Path(os.environ.get("ML4_TF_OUT", "generated/ml4_tf/result.json"))
PCC_MIN = float(os.environ.get("ML4_TF_PCC_MIN", "0.95"))  # every layer's output, every golden: a gross-error floor


def _goldens() -> list[tuple[str, Path]]:
    out = []
    for item in os.environ["ML4_TF_GOLDENS"].split(","):
        name, _, path = item.partition("=")
        out.append((name, Path(path)))
    return out


def _wait_meta(gdir: Path) -> dict:
    """The golden's meta.json (cpu_golden.py writes it last), waited for up to ML4_GOLDEN_WAIT_S."""
    p, t0 = gdir / "meta.json", time.time()
    while not p.exists():
        assert time.time() - t0 < float(os.environ.get("ML4_GOLDEN_WAIT_S", "36000")), f"no golden at {gdir}"
        time.sleep(30)
    return json.loads(p.read_text())


def _recall(dev: torch.Tensor, gold: torch.Tensor, end: int) -> dict:
    """Top-k agreement per query row: |device keys & golden keys| / |golden keys| (-1 pads ignored)."""

    def mask(t):
        t = t.long()
        t = torch.where((t < 0) | (t >= end), torch.full_like(t, end), t)
        m = torch.zeros(t.shape[0], end + 1, dtype=torch.bool)
        m.scatter_(1, t, True)
        return m[:, :end]

    a, b = mask(dev), mask(gold)
    r = (a & b).sum(-1).float() / b.sum(-1).clamp_min(1).float()
    return {
        "mean": round(float(r.mean()), 6),
        "min": round(float(r.min()), 6),
        "rows_equal": int((a == b).all(-1).sum()),
        "rows": int(b.shape[0]),
    }


class _States:
    """The golden's kv_cache/layer_<i>.safetensors as {kv_latent, k_rope, index_key}, read once."""

    def __init__(self, gdir: Path):
        self.gdir, self.got = gdir, {}

    def __getitem__(self, i: int) -> dict:
        if i not in self.got:
            from safetensors.torch import load_file

            st = load_file(str(self.gdir / "kv_cache" / f"layer_{i}.safetensors"))
            self.got[i] = {n: st[f"{n}_cache_layer_{i}"] for n in ("kv_latent", "k_rope", "index_key")}
        return self.got[i]


def _inject_kv_source(model, state, gst: _States, src: int, start: int, end: int) -> None:
    """A shared-KV layer's source layer as if it had run on this chunk with the golden's values: its cache rows
    [0, end) and its bf16 kvpe rows of the chunk (what its write_kv hands on). The source layer itself never runs."""
    from models.demos.deepseek_v3_d_p.tt.mistral_large_4.attention import kvpe_rows_to_device

    st = gst[src]
    state.load_prefix(src, st, end)
    built = src in model.mods
    mod = model._modules(src)["attention"]
    if not built:
        hooks._WEIGHTS.pop((str(Path(CKPT)), src), None)  # its host weights, as the model's own first call drops them
    if mod.chunk_kv is not None:
        ttnn.deallocate(mod.chunk_kv)
    mod.chunk_kv = kvpe_rows_to_device(mod, st["kv_latent"][start:end], st["k_rope"][start:end], model.mesh)
    mod.chunk_kv_start = start


def _inject_topk(model, gold: dict, src: int, start: int) -> None:
    """A shared-index layer's source full layer's top-k of this chunk: the golden's, where its indexer leaves it."""
    from models.demos.deepseek_v3_d_p.tt.mistral_large_4.attention import topk_to_device

    key = f"L{src}.topk"
    assert key in gold, f"the golden has no {key} (cpu_golden.py --tf-targets / --dump-topk)"
    old = model._topk.pop(src, None)
    if old is not None:
        ttnn.deallocate(old[1])
    model._topk[src] = (start, topk_to_device(gold[key], model.mesh))


def _own_kv(model, state, gst: _States, i: int, start: int, end: int) -> dict:
    """The rows [start, end) layer i wrote into the caches it owns, against the golden's."""
    from models.demos.deepseek_v3_d_p.tt.mistral_large_4.indexer import hadamard_matrix
    from models.demos.deepseek_v3_d_p.tt.mistral_large_4.settings import settings

    got, g, out = state.to_torch(i, end), gst[i], {}
    for n in ("kv_latent", "k_rope", "index_key"):
        if not got[n].shape[-1] or not g[n].shape[-1]:
            continue
        gt = g[n][start:end].float()
        if n == "index_key" and settings.get("INDEXER_HADAMARD"):
            gt = gt @ hadamard_matrix(gt.shape[-1])  # the cache holds the keys rotated
        out[n] = _cmp(got[n][start:end].float(), gt)
    return out


def _pass(model, name: str, gdir: Path, targets: list[int], res: dict, dump) -> None:
    """Every target layer on every chunk of one golden, forced with its values."""
    from safetensors.torch import load_file

    meta = _wait_meta(gdir)
    seq, chunk = meta["seq"], meta["chunk"]
    assert chunk == model.chunk, (chunk, model.chunk)
    cfg = model.cfg
    gst = _States(gdir)
    state = model.new_state(seq)
    per = res["goldens"].setdefault(name, {"dir": str(gdir), "meta": meta, "layers": {}})["layers"]
    for c in range(seq // chunk):
        start, end = c * chunk, (c + 1) * chunk
        gold = load_file(str(gdir / f"chunk_{c:02d}.safetensors"))
        for i in targets:
            assert f"L{i - 1}" in gold and f"L{i}" in gold, f"golden {gdir} lacks L{i - 1} / L{i} (--tf-targets)"
            t0 = time.time()
            if start:
                state.load_prefix(i, gst[i], start)  # the earlier chunks' rows of its own caches: the golden's
            if not cfg.owns_kv(i):
                _inject_kv_source(model, state, gst, cfg.kv_source(i), start, end)
            if not cfg.is_full(i):
                _inject_topk(model, gold, cfg.topk_source(i), start)
            x = gold[f"L{i - 1}"].float()
            h = model.from_host(x)
            out = model.layer(i, h, start, state, end=end)
            model.free(h)
            y = model.to_host(out)[:chunk]
            model.free(out)
            g = gold[f"L{i}"].float()
            r = {"out": _cmp(y, g), "delta_pcc": round(_pcc(y - x, g - x), 6)}
            r["kv"] = _own_kv(model, state, gst, i, start, end)
            if cfg.is_full(i) and f"L{i}.topk" in gold:
                from models.demos.deepseek_v3_d_p.tt.mistral_large_4.layout import topk_to_host

                r["topk_recall"] = _recall(topk_to_host(model._topk[i][1], model.mesh), gold[f"L{i}.topk"], end)
            r["seconds"] = round(time.time() - t0, 1)
            per.setdefault(str(i), {})[str(c)] = r
            kv = " ".join(f"{n} {v['pcc']}" for n, v in r["kv"].items())
            tk = f" topk_recall {r['topk_recall']['mean']}" if "topk_recall" in r else ""
            print(
                f"TF {name} L{i} chunk {c}: out {r['out']['pcc']} delta {r['delta_pcc']} {kv}{tk} ({r['seconds']} s)",
                flush=True,
            )
            dump()


@pytest.mark.timeout(43200)  # the weight load, plus waiting on goldens still being written
@pytest.mark.skipif(os.environ.get("ML4_TF_GALAXY") != "1", reason="needs a Blackhole Galaxy (ML4_TF_GALAXY=1)")
def test_teacher_forced(mesh_device):
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4.model import Config

    goldens = _goldens()
    meta0 = _wait_meta(goldens[0][1])
    env = os.environ.get("ML4_TF_LAYERS")
    targets = sorted(int(x) for x in env.split(",")) if env else list(meta0.get("tf_targets") or [])
    assert targets, "no target layers: ML4_TF_LAYERS, or a golden written with --tf-targets"
    cfg = Config(CKPT)
    # the targets, and the KV source layers the shared-KV ones read (built, never run: their state is the golden's)
    layers = sorted(set(targets) | {cfg.kv_source(i) for i in targets})
    res = {
        "node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
        "commit": os.environ.get("ML4_COMMIT", ""),
        "targets": targets,
        "model_layers": layers,
        "settings": {k: v for k, v in os.environ.items() if k.startswith(("MISTRAL_LARGE_4_", "ML4_FABRIC"))},
        "goldens": {},
    }

    def dump():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(res, indent=1) + "\n")

    t0 = time.time()
    model = device_model(mesh_device, _Spec(CKPT, max(meta0["seq"], CAPACITY), meta0["chunk"]), layers, lm_head=False)
    res["build_seconds"] = round(time.time() - t0, 1)
    for name, gdir in goldens:
        _pass(model, name, gdir, targets, res, dump)
    # per golden and layer: the worst chunk
    res["summary"] = {
        name: {
            i: {
                "out_pcc": min(v["out"]["pcc"] for v in d.values()),
                "delta_pcc": min(v["delta_pcc"] for v in d.values()),
                **{
                    f"{n}_pcc": min(v["kv"][n]["pcc"] for v in d.values() if n in v["kv"])
                    for n in ("kv_latent", "k_rope", "index_key")
                    if any(n in v["kv"] for v in d.values())
                },
            }
            for i, d in g["layers"].items()
        }
        for name, g in res["goldens"].items()
    }
    dump()
    for name, s in res["summary"].items():
        for i, v in s.items():
            print(f"TF SUMMARY {name} L{i}: " + " ".join(f"{k} {x}" for k, x in v.items()), flush=True)
    every = [v for g in res["goldens"].values() for d in g["layers"].values() for v in d.values()]
    assert all(v["out"]["finite"] for v in every), "non-finite device output"
    worst = min(v["out"]["pcc"] for v in every)
    assert worst >= PCC_MIN, f"a layer's teacher-forced output PCC {worst} < {PCC_MIN}"
