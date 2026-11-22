# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 layers 0-3 run free on device, scored per position against the FP64 chain.

The teacher-fed runs give each device layer the chain's exact input, so they measure each layer's own error. Here
the device runs layers 0-3 chained (``test_device_chain.py``: the production stage runtime from the token ids, KDA
carry and DSA caches across chunks), so each layer also carries the error of the layers before it. Whether that
error compounds with context is read off the same per-window curves the teacher-fed runs give:

- the device run writes one run directory per scored layer in its teacher feed's format, so the KDA
  (``kda_chain_feed.py score``), MoE (``moe_teacher_feed.py score``) and DSA (``dsa_teacher_feed.py score``)
  scorers score it against the chain unchanged;
- ``score`` here scores what the teacher feeds take as given: each layer's input and the live residual each layer
  leaves (``STREAM_KEYS``) against the chain's FP64 values, and the layer-0 KDA carry against the chain's state.

The FP64 references of the stream: the MoE inputs and the DSA layer's AttnRes read are the chain's products; the
live residual after layers 0 and 1 is the KDA rebuild's ``live_1`` / ``live_2`` (fp32), after layer 2
``live_2 + kda_output_2 + moe_output_2``; the KDA inputs of layers 1-2 are recomputed in FP64 from the rebuild's
live residual and the embedding, as the chain computes them.

    python -m models.demos.deepseek_v3_d_p.tests.kimi_k3.device_chain score --run /mnt/.../k31-1m-device-chain
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.layer import KDAReferenceState, kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kda.weights import normalize_kda_state_dict, required_kda_weight_names
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import (
    MANIFEST,
    atomic_save,
    segment_path,
    token_ids_sha256,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.chain_vs_trace import key_sums, pcc_nrmse
from models.demos.deepseek_v3_d_p.tests.kimi_k3.fp64_chain import (
    F64,
    ChainModel,
    attn_res64,
    rms64,
    state_path,
    tensor_digest,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.kda_chain_feed import rebuilt_manifest, rebuilt_path, rel
from models.demos.deepseek_v3_d_p.tests.kimi_k3.teacher_feed import (
    Trace,
    checkpoint_tensors,
    chunk_path,
    complete_run,
    quantile,
)

GOLDEN = Path(os.environ.get("K31_GOLDEN_DIR", "/mnt/weka/model-cache/scratch/mvasilijevic/k31/golden"))
CHAIN = GOLDEN / "k31-l0-3-fp64-1m"
REBUILT = GOLDEN / "k31-kda-rebuild-1m"
TRACE = Path("/mnt/weka/model-weights/llm/ref-data/debug_traces/kimi_k31_pro_1m/vllm-b2eba995-1048576tok")
# Each teacher-fed layer's chain input: the scorers read the reference (the chain) from its manifest.
INPUTS = {
    "kda-l1": (1, GOLDEN / "k31-1m-kda-l1-input"),
    "kda-l2": (2, GOLDEN / "k31-1m-kda-l2-input"),
    "moe-l1": (1, GOLDEN / "k31-1m-moe-l1-input"),
    "moe-l2": (2, GOLDEN / "k31-1m-moe-l2-input"),
    "dsa-l3": (3, GOLDEN / "k31-1m-dsa-l3-input"),
}


@dataclass(frozen=True)
class Taps:
    """What a device-chain run of layers ``0..num_layers-1`` taps, by the reference chain's products: the KDA layers
    (carry; output from layer 1), the MoE layers (output, routing) and the DSA layers (output, top-k, caches) the chain
    holds, and the ``stream``: each tapped layer's input and the live residual after every layer but the last."""

    num_layers: int
    kda: tuple[int, ...]
    moe: tuple[int, ...]
    dsa: tuple[int, ...]

    @property
    def stream_keys(self) -> tuple[str, ...]:
        keys = [f"kda_input_{l}" for l in self.kda if l] + [f"moe_input_{l}" for l in self.moe]
        return (*keys, *(f"dsa_input_{l}" for l in self.dsa), *(f"live_{l}" for l in range(self.num_layers - 1)))

    def dirs(self, out: Path) -> dict[str, Path]:
        """The run directory of each tap in ``out``."""
        names = [f"kda-l{l}" for l in self.kda] + [f"moe-l{l}" for l in self.moe] + [f"dsa-l{l}" for l in self.dsa]
        return {name: out / name for name in (*names, "stream")}


LAYERS_0_3 = Taps(num_layers=4, kda=(0, 1, 2), moe=(1, 2), dsa=(3,))  # ``fp64_chain``: layer 3's MoE is not stored
STREAM_KEYS = LAYERS_0_3.stream_keys


def chain_taps(chain: Path) -> Taps:
    """The taps a run scored against ``chain`` takes: ``fp64_chain`` (layers 0-3) or ``fp64_deep_chain`` (every layer
    of its ``kinds``)."""
    m = json.loads((chain / MANIFEST).read_text())
    if m["producer"] == "fp64_chain":
        return LAYERS_0_3
    if m["producer"] != "fp64_deep_chain":
        raise ValueError(f"{chain}: no device-chain taps for producer {m['producer']}")
    kinds = {int(l): k for l, k in m["kinds"].items()}
    pick = lambda site, kind: tuple(l for l in sorted(kinds) if kinds[l][site] == kind)  # noqa: E731
    return Taps(num_layers=len(kinds), kda=pick(0, "kda"), moe=pick(1, "moe"), dsa=pick(0, "dsa"))


def load_token_ids(trace: Path, count: int) -> torch.Tensor:
    """The chain's tokens: the trace's prompt ids ``[:count]``, as ``fp64_chain`` takes them."""
    return torch.tensor(Trace(trace).metadata["token_ids"][:count], dtype=torch.int64)


def stream_references(model: ChainModel, chain: Path, rebuilt: Path, index: int, emb: torch.Tensor) -> dict:
    """The FP64 value of every ``STREAM_KEYS`` tap over chain segment ``index`` (module docstring)."""
    seg = torch.load(segment_path(chain, index), weights_only=True, mmap=True)
    rb = torch.load(rebuilt_path(rebuilt, index), weights_only=True, mmap=True)
    sealed = emb.unsqueeze(1)
    want = {f"moe_input_{l}": seg[f"moe_input_{l}"] for l in (1, 2)}
    want["dsa_input_3"] = seg["dsa_input_3"]
    for l in (1, 2):
        L = model.layers[l]
        read = attn_res64(rb[f"live_{l}"].to(F64), sealed, L["attn_res"], model.eps)
        want[f"kda_input_{l}"] = rms64(read, L["input_layernorm"], model.eps)
    want["live_0"], want["live_1"] = rb["live_1"], rb["live_2"]
    want["live_2"] = rb["live_2"].double() + rb["kda_output_2"].double() + seg["moe_output_2"].double()
    return want


def score(*, run: Path, rebuilt: Path, trace: Path, model: ChainModel, window: int, log=print) -> dict:
    """Score the stream and the layer-0 carry of a device-chain run in ``run``; writes ``stream/metrics.json``."""
    stream = complete_run(run / "stream")
    count, chunk = stream["count"], stream["chunk"]
    rb = rebuilt_manifest(rebuilt, count)
    chain = Path(rb["chain"]["chain"])
    seg_rows = rb["segment_rows"]
    segment = seg_rows[0][1] - seg_rows[0][0]
    token_ids = load_token_ids(trace, count)
    sums = {k: [] for k in STREAM_KEYS}
    row_nrmse = {k: [] for k in STREAM_KEYS}
    for index in range(-(-count // segment)):
        lo, hi = seg_rows[index][0], min(seg_rows[index][1], count)
        got = {k: [] for k in STREAM_KEYS}
        for c in range(lo // chunk, -(-hi // chunk)):
            payload = torch.load(chunk_path(run / "stream", c), weights_only=True)
            a, b = max(lo, payload["start"]) - payload["start"], min(hi, payload["end"]) - payload["start"]
            for k in STREAM_KEYS:
                got[k].append(payload[k][a:b])
        emb = model.embedding[token_ids[lo:hi]].to(F64)
        want = stream_references(model, chain, rebuilt, index, emb)
        for k in STREAM_KEYS:
            g, w = torch.cat(got[k]).double(), want[k][: hi - lo].double()
            sums[k].append(key_sums(g, w))
            row_nrmse[k].append(((g - w).norm(dim=1) / w.norm(dim=1)).float())
        if index % 32 == 0:
            log(f"scored stream segment {index + 1} rows [{lo}, {hi})")
    sums = {k: torch.cat(v) for k, v in sums.items()}
    row_nrmse = {k: torch.cat(v) for k, v in row_nrmse.items()}

    def measures(lo: int, hi: int) -> dict:
        result = {"start": lo, "end": hi}
        for k in STREAM_KEYS:
            pcc, nrmse = pcc_nrmse(sums[k][lo:hi])
            result[f"{k}_pcc"], result[f"{k}_nrmse"] = pcc, nrmse
            result[f"{k}_row_nrmse_p99"] = quantile(row_nrmse[k][lo:hi], 0.99)
        return result

    states = []
    for entry in complete_run(run / "kda-l0").get("states", []):
        row = entry["row"]
        if row % segment:
            continue
        want = torch.load(state_path(chain, row // segment - 1), weights_only=True, mmap=True)["kda"][0]
        got = torch.load(run / "kda-l0" / entry["file"], weights_only=True)
        conv = max(rel(got[f"{c}_convolution"], want[f"{c}_convolution"]) for c in "qkv")
        states.append({"row": row, "recurrent_rel": rel(got["recurrent"], want["recurrent"]), "conv_rel": conv})
    result = {
        "count": count,
        "window": window,
        "chain": str(chain),
        "rebuilt": str(rebuilt),
        "run": str(run),
        "overall": measures(0, count),
        "windows": [measures(lo, min(lo + window, count)) for lo in range(0, count, window)],
        "kda_l0_states": states,
    }
    (run / "stream" / "metrics.json").write_text(json.dumps(result, indent=1))
    return result


# --- layers 0-7: the FP64 deep chain (``fp64_deep_chain``, layer-major) and its KDA rebuild ---------------------------

DEEP = GOLDEN / "k31-l0-7-fp64-256k"
DEEP_REBUILT = GOLDEN / "k31-kda-rebuild-256k-l4-6"
DEEP_KDA = (4, 5, 6)  # the KDA layers after the first DSA layer: the deep chain stores neither their output nor state
DEEP_REBUILD_VERSION = 1
# The rebuild's live sum is assembled from fp32-stored products (the 1M rebuild's live and layer-2 KDA output, the deep
# chain's FFN and DSA outputs) instead of the chain's FP64 sums: a ~1e-7 relative perturbation of every read.
DEEP_PRODUCT_BAR = 1e-5


def deep_products(chain: Path, layer: int, site: str, names: list[str], start: int, end: int) -> dict:
    """Products ``names`` of the deep chain's pass (``layer``, ``site``) for rows ``[start, end)``; the pass's
    segments are ``<chain>/L<layer>_<site>/segment_<i>.pt`` of ``manifest["segment"]`` rows each."""
    seg = json.loads((chain / MANIFEST).read_text())["segment"]
    parts: dict[str, list[torch.Tensor]] = {n: [] for n in names}
    for index in range(start // seg, -(-end // seg)):
        d = torch.load(chain / f"L{layer}_{site}" / f"segment_{index:06d}.pt", weights_only=True, mmap=True)
        a, b = max(start, d["start"]) - d["start"], min(end, d["end"]) - d["start"]
        for n in names:
            parts[n].append(d[n][a:b])
    return {n: torch.cat(v) for n, v in parts.items()}


def deep_complete(chain: Path, layers: int | None = None) -> dict:
    """The deep chain's manifest; refuses a chain whose passes (attention, then FFN) of layers ``0..layers-1``
    (default: every layer) are not all done."""
    m = json.loads((chain / MANIFEST).read_text())
    need = 2 * (len(m["kinds"]) if layers is None else layers)
    if len(m["segments"]) < need:
        raise ValueError(f"{chain}: {len(m['segments'])} passes complete, {need} needed")
    return m


def block_weights(checkpoint: Path, layer: int, kda_config: KDAConfig | None) -> dict:
    """Layer ``layer``'s AttnRes queries and norms, and its KDA weights when ``kda_config`` is given, in FP64
    (``ChainModel``'s per-layer form)."""
    names = ["input_layernorm.weight", "post_attention_layernorm.weight", "self_attention_res_norm.weight"]
    names += ["self_attention_res_proj.weight", "mlp_res_norm.weight", "mlp_res_proj.weight"]
    if kda_config is not None:
        names += [f"self_attn.{n}" for n in required_kda_weight_names(kda_config)]
    t = checkpoint_tensors(checkpoint, layer, names)
    L = {
        "input_layernorm": t["input_layernorm.weight"].to(F64),
        "attn_res": (t["self_attention_res_norm.weight"].to(F64), t["self_attention_res_proj.weight"].to(F64)),
        "mlp_res": (t["mlp_res_norm.weight"].to(F64), t["mlp_res_proj.weight"].to(F64)),
        "post_attention_layernorm": t["post_attention_layernorm.weight"].to(F64),
    }
    if kda_config is not None:
        kda = {n: t[f"self_attn.{n}"] for n in required_kda_weight_names(kda_config)}
        L["kda"] = {n: v.to(F64) for n, v in normalize_kda_state_dict(kda, kda_config).items()}
    return L


def deep_state_path(out: Path, index: int) -> Path:
    return out / f"state_{index:06d}.pt"


def rebuild_deep(
    *,
    chain: Path,
    rebuilt_1m: Path,
    out: Path,
    embedding: torch.Tensor,
    weights: dict,
    kda_config: KDAConfig,
    eps: float,
    token_ids: torch.Tensor,
    log=print,
) -> None:
    """The FP64 KDA output and state of layers ``DEEP_KDA`` over every row of the deep chain, into ``out``.

    Segment-major and serial (the KDA state carries): per deep-chain segment, the live AttnRes sum after layer 2 is
    the 1M rebuild's ``live_2 + kda_output_2`` plus the deep chain's ``moe_output_2`` (layers 0-3 of the two chains
    are the same computation); layer 3 adds the deep chain's DSA and MoE outputs; layers 4-6 run the FP64 KDA
    reference on their AttnRes read and add the stored MoE output. Each segment checks itself against what the deep
    chain stored: the recomputed read ``dsa_input_3`` and the MoE inputs of layers 3-6 (relative
    L2 <= ``DEEP_PRODUCT_BAR``). Writes ``rebuilt_<i>.pt`` (``kda_output_{4,5,6}`` fp32, ``checks``) and
    ``state_<i>.pt`` (the three FP64 states after segment i, the 1M chain's ``state_<i>.pt`` form); a rerun resumes
    after the last segment written. Needs the deep chain's passes through layer 6 only."""
    m = deep_complete(chain, layers=max(DEEP_KDA) + 1)
    rb = rebuilt_manifest(rebuilt_1m, m["count"])
    if rb["segment_rows"][0] != [0, m["segment"]]:
        raise ValueError(f"the 1M rebuild's segments {rb['segment_rows'][0]} are not the deep chain's {m['segment']}")
    identity = {
        "producer": "device_chain.rebuild_deep",
        "rebuild_version": DEEP_REBUILD_VERSION,
        "chain": str(chain),
        "chain_token_ids_sha256": m["token_ids_sha256"],
        "chain_passes": [p["digests"] for p in m["segments"][: 2 * (max(DEEP_KDA) + 1)]],
        "rebuilt_1m": str(rebuilt_1m),
        "layers": list(DEEP_KDA),
        "count": m["count"],
        "segment": m["segment"],
        "segment_rows": [[lo, min(lo + m["segment"], m["count"])] for lo in range(0, m["count"], m["segment"])],
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / MANIFEST
    if path.is_file() and json.loads(path.read_text()) != identity:
        raise ValueError(f"{out} holds a rebuild of something else")
    path.write_text(json.dumps(identity, indent=1))
    rows = identity["segment_rows"]
    done = 0
    while done < len(rows) and rebuilt_path(out, done).is_file() and deep_state_path(out, done).is_file():
        done += 1
    states: dict[int, KDAReferenceState | None] = {l: None for l in DEEP_KDA}
    if done:
        saved = torch.load(deep_state_path(out, done - 1), weights_only=True)["kda"]
        states = {l: KDAReferenceState(**saved[l]) for l in DEEP_KDA}
        log(f"resuming after segment {done}/{len(rows)}")
    for index in range(done, len(rows)):
        began = time.perf_counter()
        lo, hi = rows[index]
        sealed = embedding[token_ids[lo:hi]].to(F64).unsqueeze(1)
        r1m = torch.load(rebuilt_path(rebuilt_1m, index), weights_only=True, mmap=True)
        assert (r1m["start"], r1m["end"]) == (lo, hi), f"1M rebuild segment {index} is not rows [{lo}, {hi})"

        def ffn(l: int) -> dict:
            return deep_products(chain, l, "ffn", [f"moe_input_{l}", f"moe_output_{l}"], lo, hi)

        live = r1m["live_2"].double() + r1m["kda_output_2"].double() + ffn(2)["moe_output_2"].double()
        checks, payload = {}, {"start": lo, "end": hi}
        for l in range(3, max(DEEP_KDA) + 1):
            L = weights[l]
            read = attn_res64(live, sealed, L["attn_res"], eps)
            if l in DEEP_KDA:
                hidden = rms64(read, L["input_layernorm"], eps)
                attn, states[l] = kda_forward_reference(hidden.unsqueeze(0), L["kda"], kda_config, states[l], dtype=F64)
                attn = attn[0]
                payload[f"kda_output_{l}"] = attn.float()
            else:
                d = deep_products(chain, l, "attn", [f"dsa_input_{l}", f"attn_output_{l}"], lo, hi)
                checks[f"dsa_input_{l}_rel"] = rel(read, d[f"dsa_input_{l}"])
                attn = d[f"attn_output_{l}"].double()
            live = live + attn
            normed = rms64(attn_res64(live, sealed, L["mlp_res"], eps), L["post_attention_layernorm"], eps)
            f = ffn(l)
            checks[f"moe_input_{l}_rel"] = rel(normed, f[f"moe_input_{l}"])
            live = live + f[f"moe_output_{l}"].double()
        failed = {k: v for k, v in checks.items() if not v <= DEEP_PRODUCT_BAR}
        if failed:
            raise ValueError(f"segment {index}: the rebuild does not reproduce the deep chain: {failed}")
        payload["checks"] = checks
        state = {"start": lo, "end": hi, "kda": {l: vars(states[l]) for l in DEEP_KDA}}
        atomic_save(state, deep_state_path(out, index))
        atomic_save(payload, rebuilt_path(out, index))
        log(
            f"rebuilt segment {index + 1}/{len(rows)} rows [{lo}, {hi}) {time.perf_counter() - began:.1f} s "
            + json.dumps({k: float(f"{v:.3g}") for k, v in checks.items()})
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["score", "rebuild-deep"])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--rebuilt", type=Path, default=REBUILT)
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--chain", type=Path, default=DEEP, help="rebuild-deep: the deep chain")
    parser.add_argument("--out", type=Path, default=DEEP_REBUILT, help="rebuild-deep: the rebuild's directory")
    parser.add_argument("--window", type=int, default=65536)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    def log(msg: str) -> None:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)

    if args.command == "rebuild-deep":
        m = json.loads((args.chain / MANIFEST).read_text())
        checkpoint = Path(m["identity"]["checkpoint"])
        config = json.loads((checkpoint / "config.json").read_text())
        kda_config = KDAConfig.from_model_config(config)
        embedding = checkpoint_tensors(checkpoint, None, ["embed_tokens.weight"])["embed_tokens.weight"]
        if tensor_digest({"w": embedding}) != m["embedding_sha256"]:
            raise ValueError("the checkpoint's embedding is not the deep chain's")
        token_ids = load_token_ids(args.trace, m["count"])
        if token_ids_sha256(token_ids.tolist()) != m["token_ids_sha256"]:
            raise ValueError("the trace's tokens are not the deep chain's")
        weights = {l: block_weights(checkpoint, l, kda_config if l in DEEP_KDA else None) for l in range(3, 7)}
        log(f"loaded layers 3-6 from {checkpoint}")
        rebuild_deep(
            chain=args.chain,
            rebuilt_1m=args.rebuilt,
            out=args.out,
            embedding=embedding,
            weights=weights,
            kda_config=kda_config,
            eps=float(config.get("text_config", config)["rms_norm_eps"]),
            token_ids=token_ids,
            log=log,
        )
        return

    chain = Path(json.loads((args.rebuilt / MANIFEST).read_text())["chain"]["chain"])
    checkpoint = Path(json.loads((chain / MANIFEST).read_text())["identity"]["checkpoint"])
    log(f"loading weights (no routed experts) from {checkpoint}")
    model = ChainModel.from_checkpoint(checkpoint, args.threads, log, routed_experts=False)
    result = score(run=args.run, rebuilt=args.rebuilt, trace=args.trace, model=model, window=args.window, log=log)
    print(json.dumps({"overall": result["overall"], "kda_l0_states": result["kda_l0_states"][-3:]}, indent=1))


if __name__ == "__main__":
    main()
