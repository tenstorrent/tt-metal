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
``live_2 + kda_output_2 + moe_output_2``; the KDA inputs of layers 1-2 are the rebuild's (its FP64 KDA input rounded
once to bf16, which moves an NRMSE of 0.02-0.06 by under 1e-4). Every reference is a stored product.

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
from models.demos.deepseek_v3_d_p.tests.accuracy.golden_registry import token_ids_sha256
from models.demos.deepseek_v3_d_p.tests.accuracy.kda import state_errors
from models.demos.deepseek_v3_d_p.tests.accuracy.moe import routing_rows
from models.demos.deepseek_v3_d_p.tests.accuracy.topk import missed_keys, topk_recall
from models.demos.deepseek_v3_d_p.tests.accuracy.windowed import (
    COUNT,
    SUM_EE,
    SUM_YY,
    RowSeries,
    errors,
    quantile,
    row_nrmse,
    row_sums,
)
from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import golden_floor
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import MANIFEST, atomic_save, segment_path
from models.demos.deepseek_v3_d_p.tests.kimi_k3.chain_vs_trace import CUT_GAP_RANKS
from models.demos.deepseek_v3_d_p.tests.kimi_k3.fp64_chain import (
    F64,
    attn_res64,
    load_layer,
    open_chain,
    rms64,
    state_path,
    tensor_digest,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.fp64_deep_chain import GOLDEN_256K, open_deep_chain
from models.demos.deepseek_v3_d_p.tests.kimi_k3.kda_chain_feed import rebuilt_manifest, rebuilt_path, rel
from models.demos.deepseek_v3_d_p.tests.kimi_k3.teacher_feed import Trace, checkpoint_tensors, chunk_path, complete_run

# Derived artifacts (KDA rebuilds, teacher-fed inputs) stay here; the FP64 chains are registry goldens, opened by id.
GOLDEN = Path(os.environ.get("K31_GOLDEN_DIR", "/mnt/weka/model-cache/scratch/mvasilijevic/k31/golden"))
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


def stream_references(chain: Path, rebuilt: Path, index: int) -> dict:
    """The FP64 value of every ``STREAM_KEYS`` tap over chain segment ``index`` (module docstring)."""
    seg = torch.load(segment_path(chain, index), weights_only=True, mmap=True)
    rb = torch.load(rebuilt_path(rebuilt, index), weights_only=True, mmap=True)
    want = {f"moe_input_{l}": seg[f"moe_input_{l}"] for l in (1, 2)}
    want["dsa_input_3"] = seg["dsa_input_3"]
    want.update({f"kda_input_{l}": rb[f"kda_input_{l}"] for l in (1, 2)})
    want["live_0"], want["live_1"] = rb["live_1"], rb["live_2"]
    want["live_2"] = rb["live_2"].double() + rb["kda_output_2"].double() + seg["moe_output_2"].double()
    return want


def score(*, run: Path, rebuilt: Path, window: int, log=print) -> dict:
    """Score the stream and the layer-0 carry of a device-chain run in ``run``; returns the metrics
    (``write_score`` stores them as ``stream/metrics.json``)."""
    stream = complete_run(run / "stream")
    count, chunk = stream["count"], stream["chunk"]
    rb = rebuilt_manifest(rebuilt, count)
    chain = open_chain(rb["chain"]["chain"]).path
    seg_rows = rb["segment_rows"]
    segment = seg_rows[0][1] - seg_rows[0][0]
    sums = {k: [] for k in STREAM_KEYS}
    nrmse_rows = {k: [] for k in STREAM_KEYS}
    for index in range(-(-count // segment)):
        lo, hi = seg_rows[index][0], min(seg_rows[index][1], count)
        got = {k: [] for k in STREAM_KEYS}
        for c in range(lo // chunk, -(-hi // chunk)):
            payload = torch.load(chunk_path(run / "stream", c), weights_only=True)
            a, b = max(lo, payload["start"]) - payload["start"], min(hi, payload["end"]) - payload["start"]
            for k in STREAM_KEYS:
                got[k].append(payload[k][a:b])
        want = stream_references(chain, rebuilt, index)
        for k in STREAM_KEYS:
            sums[k].append(row_sums(want[k][: hi - lo], torch.cat(got[k])))
            nrmse_rows[k].append(row_nrmse(sums[k][-1], 0.0).float())
        if index % 32 == 0:
            log(f"scored stream segment {index + 1} rows [{lo}, {hi})")
    series = RowSeries()
    series.append(
        0,
        **{f"{k}_sums": torch.cat(v) for k, v in sums.items()},
        **{f"{k}_nrmse": torch.cat(v) for k, v in nrmse_rows.items()},
    )

    def measures(w) -> dict:
        return {n: v for k in STREAM_KEYS for n, v in stream_errors(w[f"{k}_sums"], w[f"{k}_nrmse"], f"{k}_").items()}

    states = []
    for entry in complete_run(run / "kda-l0").get("states", []):
        row = entry["row"]
        if row % segment:
            continue
        want = torch.load(state_path(chain, row // segment - 1), weights_only=True, mmap=True)["kda"][0]
        got = torch.load(run / "kda-l0" / entry["file"], weights_only=True)
        states.append({"row": row, **state_errors(got, want)})
    result = {
        "count": count,
        "window": window,
        "chain": str(chain),
        "rebuilt": str(rebuilt),
        "run": str(run),
        "overall": series.whole(measures),
        "windows": series.windows(window, measures),
        "kda_l0_states": states,
    }
    return result


def write_score(run: Path, result: dict) -> None:
    (run / "stream" / "metrics.json").write_text(json.dumps(result, indent=1))


# --- layers 0-7: the FP64 deep chain (``fp64_deep_chain``, layer-major) and its KDA rebuild ---------------------------

DEEP_REBUILT = GOLDEN / "k31-kda-rebuild-256k-l4-6"
DEEP_KDA = (4, 5, 6)  # the KDA layers after the first DSA layer: the deep chain stores neither their output nor state
DEEP_REBUILD_VERSION = 1
# The rebuild's live sum is assembled from fp32-stored products (the 1M rebuild's live and layer-2 KDA output, the deep
# chain's FFN and DSA outputs) instead of the chain's FP64 sums: a ~1e-7 relative perturbation of every read.
DEEP_PRODUCT_BAR = 1e-5


def segment_rows(path_of, segment: int, names: list[str], start: int, end: int) -> dict:
    """Rows ``[start, end)`` of ``names`` from per-segment files ``path_of(i)`` of ``segment`` rows each, each holding
    its own ``start`` / ``end``."""
    parts: dict[str, list[torch.Tensor]] = {n: [] for n in names}
    for index in range(start // segment, -(-end // segment)):
        d = torch.load(path_of(index), weights_only=True, mmap=True)
        a, b = max(start, d["start"]) - d["start"], min(end, d["end"]) - d["start"]
        for n in names:
            parts[n].append(d[n][a:b])
    return {n: torch.cat(v) for n, v in parts.items()}


def deep_products(chain: Path, layer: int, site: str, names: list[str], start: int, end: int) -> dict:
    """Products ``names`` of the deep chain's pass (``layer``, ``site``) for rows ``[start, end)``; the pass's
    segments are ``<chain>/L<layer>_<site>/segment_<i>.pt`` of ``manifest["segment"]`` rows each."""
    seg = json.loads((chain / MANIFEST).read_text())["segment"]  # (the chain's row segment)
    return segment_rows(lambda i: chain / f"L{layer}_{site}" / f"segment_{i:06d}.pt", seg, names, start, end)


def deep_complete(chain: Path, layers: int | None = None) -> dict:
    """The deep chain's manifest; refuses a chain whose passes (attention, then FFN) of layers ``0..layers-1``
    (default: every layer) are not all done."""
    m = json.loads((chain / MANIFEST).read_text())
    need = 2 * (len(m["kinds"]) if layers is None else layers)
    if len(m["passes"]) < need:
        raise ValueError(f"{chain}: {len(m['passes'])} passes complete, {need} needed")
    return m


def block_weights(checkpoint: Path, layer: int) -> dict:
    """Layer ``layer``'s AttnRes queries and norms in FP64 (``ChainModel``'s per-layer form), for ``score-deep``'s
    replay of each layer's input. No chain digest covers this subset of a layer (the chains digest each site's full
    weight set), so these are not checked against the chain; ``rebuild-deep`` loads whole sites and is."""
    names = ["input_layernorm.weight", "post_attention_layernorm.weight", "self_attention_res_norm.weight"]
    names += ["self_attention_res_proj.weight", "mlp_res_norm.weight", "mlp_res_proj.weight"]
    t = checkpoint_tensors(checkpoint, layer, names)
    return {
        "input_layernorm": t["input_layernorm.weight"].to(F64),
        "attn_res": (t["self_attention_res_norm.weight"].to(F64), t["self_attention_res_proj.weight"].to(F64)),
        "mlp_res": (t["mlp_res_norm.weight"].to(F64), t["mlp_res_proj.weight"].to(F64)),
        "post_attention_layernorm": t["post_attention_layernorm.weight"].to(F64),
    }


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
    rb = rebuilt_manifest(rebuilt_1m, m["rows"])
    if rb["segment_rows"][0] != [0, m["segment"]]:
        raise ValueError(f"the 1M rebuild's segments {rb['segment_rows'][0]} are not the deep chain's {m['segment']}")
    identity = {
        "producer": "device_chain.rebuild_deep",
        "rebuild_version": DEEP_REBUILD_VERSION,
        "chain": str(chain),
        "chain_token_ids_sha256": m["token_ids_sha256"],
        "chain_weights_sha256": {
            k: v
            for k, v in m["identity"]["weights_sha256"].items()
            if k == "embed_tokens" or int(k.split(".")[2]) <= max(DEEP_KDA)
        },
        "rebuilt_1m": str(rebuilt_1m),
        "layers": list(DEEP_KDA),
        "count": m["rows"],
        "segment": m["segment"],
        "segment_rows": [[lo, min(lo + m["segment"], m["rows"])] for lo in range(0, m["rows"], m["segment"])],
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


def rows_of(directory: Path, names: list[str], start: int, end: int) -> dict:
    """Rows ``[start, end)`` of ``names`` from a KDA rebuild's per-segment ``rebuilt_<i>.pt``."""
    seg_rows = json.loads((directory / MANIFEST).read_text())["segment_rows"]
    segment = seg_rows[0][1] - seg_rows[0][0]
    return segment_rows(lambda i: rebuilt_path(directory, i), segment, names, start, end)


def row_rms_medians(sums: torch.Tensor) -> dict:
    """The median over rows of each row's RMS error and RMS reference value (per element, as ``windowed.rms_err``)."""
    n = sums[:, COUNT]
    return {
        "row_err_median": quantile((sums[:, SUM_EE] / n).sqrt(), 0.5),
        "row_ref_median": quantile((sums[:, SUM_YY] / n).sqrt(), 0.5),
    }


def stream_errors(sums: torch.Tensor, rows: torch.Tensor, prefix: str) -> dict:
    """A continuous tap's window measures: absolute and relative error, per-row NRMSE p99, per-row RMS medians."""
    out = errors(sums, prefix)
    out[f"{prefix}row_nrmse_p99"] = quantile(rows, 0.99)
    out.update({f"{prefix}{n}": v for n, v in row_rms_medians(sums).items()})
    return out


def input_pair_keys(spec: Taps) -> tuple[str, ...]:
    """The half of each layer's input the device does not tap, derived on both sides so every layer has both: the KDA
    layers' pre-norm AttnRes read (``kda_read_l``) and the DSA layers' input after ``input_layernorm``
    (``dsa_normed_l``)."""
    return (*(f"kda_read_{l}" for l in spec.kda if l), *(f"dsa_normed_{l}" for l in spec.dsa))


def input_pairs(got: dict, weights: dict, eps: float, emb: torch.Tensor, spec: Taps) -> dict:
    """``input_pair_keys`` from one side's taps (the device's or the reference's): the KDA read in FP64 from the live
    sum before the layer, the DSA normed input in FP64 from its tapped read (as ``rebuild_deep`` / the chains compute
    them)."""
    out = {}
    for l in spec.kda:
        if l:
            out[f"kda_read_{l}"] = attn_res64(
                got[f"live_{l - 1}"].double(), emb.unsqueeze(1), weights[l]["attn_res"], eps
            )
    for l in spec.dsa:
        out[f"dsa_normed_{l}"] = rms64(got[f"dsa_input_{l}"].double(), weights[l]["input_layernorm"], eps)
    return out


def deep_references(
    *, chain: Path, rebuilt_1m: Path, rebuilt: Path, weights: dict, eps: float, emb: torch.Tensor, lo: int, hi: int
) -> dict:
    """The FP64 value of every tap of a layers 0-7 run over rows ``[lo, hi)``.

    The live residual after layer l (``live_l``): after layers 0-1 the 1M rebuild's live sums before layers 1-2, then
    each later layer's attention and FFN outputs added (KDA: the 1M rebuild for layer 2, ``rebuild_deep`` for layers
    4-6; DSA and MoE: the deep chain). The KDA inputs are recomputed from it as the chains compute them; the MoE
    inputs, DSA reads and every MoE / DSA product are the deep chain's."""
    r1 = rows_of(rebuilt_1m, ["live_1", "live_2", "kda_output_1", "kda_output_2"], lo, hi)
    want = {"kda_output_1": r1["kda_output_1"], "kda_output_2": r1["kda_output_2"]}
    want.update(rows_of(rebuilt, [f"kda_output_{l}" for l in DEEP_KDA], lo, hi))
    sealed = emb.unsqueeze(1)
    live = {0: r1["live_1"].double(), 1: r1["live_2"].double()}
    for l in range(1, 8):
        L = weights[l]
        if f"kda_output_{l}" in want:
            read = attn_res64(live[l - 1], sealed, L["attn_res"], eps)
            want[f"kda_input_{l}"] = rms64(read, L["input_layernorm"], eps)
            attn = want[f"kda_output_{l}"]
        else:
            names = [f"dsa_input_{l}", f"attn_output_{l}", f"topk_indices_{l}", f"topk_scores_{l}"]
            d = deep_products(chain, l, "attn", [*names, "latent", "kpe", "index_key"], lo, hi)
            want.update({n: d[n] for n in names})
            want.update({f"{k}_{l}": d[k] for k in ("latent", "kpe", "index_key")})
            attn = d[f"attn_output_{l}"]
        names = [f"moe_input_{l}", f"moe_output_{l}", f"routing_ids_{l}", f"routing_weights_{l}"]
        want.update(deep_products(chain, l, "ffn", names, lo, hi))
        if l >= 2:
            live[l] = live[l - 1] + attn.double() + want[f"moe_output_{l}"].double()
    want.update({f"live_{l}": v for l, v in live.items() if l < 7})
    return want


def score_deep(
    *,
    run: Path,
    chain: Path,
    rebuilt_1m: Path,
    rebuilt: Path,
    trace: Trace,
    weights: dict,
    embedding: torch.Tensor,
    eps: float,
    window: int,
    metrics_dir: Path,
    log=print,
) -> dict:
    """Score a layers 0-7 device-chain run in ``run`` against the deep chain per ``window`` rows; writes each tap's
    ``<metrics_dir>/<tap>/metrics.json`` and returns ``{tap: metrics}``.

    The per-row definitions are the teacher-feed scorers': the stream and the KDA outputs by absolute and relative
    error (``windowed.errors``: NRMSE falls when a layer adds a large accurate contribution to the live sum, while the
    error in it still grows), per-row NRMSE and per-row RMS medians, the stream with both halves of every layer input
    (``input_pairs``); the MoE output by absolute and relative error and per-row NRMSE floored at 1% of the median
    reference row norm, and its routing by ``moe.routing_rows`` against the FP64 ids, best first (so a miss's rank is
    its FP64 rank); the DSA attention output (every row: the deep chain attends every row) as the MoE output, its top-k
    by recall on rows >= k with each miss's FP64 rank (``missed_keys``) and the FP64 cut gap
    (``chain_vs_trace.CUT_GAP_RANKS``), its caches by absolute and relative error; the KDA carry by ``kda.state_errors``
    against the 1M chain's state (layers 0-2) or ``rebuild_deep``'s (layers 4-6). The GPU trace's routing and top-k
    recall of the same FP64 reference sit beside the device's."""
    spec = chain_taps(chain)
    stream = complete_run(run / "stream")
    count, chunk = stream["count"], stream["chunk"]
    m = deep_complete(chain)
    topk = int(m["dsa_dims"]["topk"])
    chain_1m = open_chain(rebuilt_manifest(rebuilt_1m, count)["chain"]["chain"]).path
    token_ids = load_token_ids(trace.path, count)
    kda_fed = tuple(l for l in spec.kda if l)
    stream_keys = (*spec.stream_keys, *input_pair_keys(spec))
    pooled = [*stream_keys, *(f"kda_{l}" for l in kda_fed)]  # scored by row sums and per-row NRMSE
    sums, rows = {k: [] for k in pooled}, {k: [] for k in pooled}
    moe_err = {l: [] for l in spec.moe}  # row sums
    dsa_err = {l: [] for l in spec.dsa}
    per_row = {f"{k}_{l}": [] for l in spec.moe for k in ("recall", "dropped", "weight_err", "gpu")}
    per_row.update({f"{k}_{l}": [] for l in spec.dsa for k in ("topk", "topk_gpu", "cut_gap")})
    misses = {f"moe_{l}": [] for l in spec.moe} | {f"dsa_{l}": [] for l in spec.dsa}

    def keep_misses(name: str, miss: dict, lo: int) -> None:
        miss["row"] = miss["row"] + lo
        misses[name].append({k: v if k == "spread_share" else v.int() for k, v in miss.items()})

    for c in range(-(-count // chunk)):
        began = time.perf_counter()
        s = torch.load(chunk_path(run / "stream", c), weights_only=True)
        lo, hi = s["start"], s["end"]
        emb = embedding[token_ids[lo:hi]].to(F64)
        want = deep_references(
            chain=chain, rebuilt_1m=rebuilt_1m, rebuilt=rebuilt, weights=weights, eps=eps, emb=emb, lo=lo, hi=hi
        )
        got = {k: s[k] for k in spec.stream_keys}
        got.update(input_pairs(got, weights, eps, emb, spec))
        want.update(input_pairs(want, weights, eps, emb, spec))
        for l in kda_fed:
            got[f"kda_{l}"] = torch.load(chunk_path(run / f"kda-l{l}", c), weights_only=True)["attn_out"]
            want[f"kda_{l}"] = want[f"kda_output_{l}"]
        for k in pooled:
            sums[k].append(row_sums(want[k], got[k]))
            rows[k].append(row_nrmse(sums[k][-1], 0.0).float())
        for l in spec.moe:
            p = torch.load(chunk_path(run / f"moe-l{l}", c), weights_only=True)
            moe_err[l].append(row_sums(want[f"moe_output_{l}"], p["moe_out"]))
            ids = want[f"routing_ids_{l}"].long()
            fp64_order = -torch.arange(ids.shape[1], dtype=torch.float32).expand(ids.shape)
            r = routing_rows(p["expert_ids"].long(), p["expert_weights"], ids, want[f"routing_weights_{l}"], fp64_order)
            for k in ("recall", "dropped", "weight_err"):
                per_row[f"{k}_{l}"].append(r[k].float())
            per_row[f"gpu_{l}"].append(topk_recall(trace.rows(f"expert_ids_layer_{l}", lo, hi).long(), ids).float())
            keep_misses(f"moe_{l}", r["misses"], lo)
        for l in spec.dsa:
            p = torch.load(chunk_path(run / f"dsa-l{l}", c), weights_only=True)
            dsa_err[l].append(row_sums(want[f"attn_output_{l}"], p["attn_out"]))
            sel, scores = want[f"topk_indices_{l}"].long(), want[f"topk_scores_{l}"].float()
            per_row[f"topk_{l}"].append(topk_recall(p["topk"].long(), sel).float())
            gpu = trace.rows(f"dsa_topk_indices_layer_{l}", lo, hi).long()
            per_row[f"topk_gpu_{l}"].append(topk_recall(gpu, sel).float())
            ordered = scores.double().sort(dim=1, descending=True).values
            gap = ordered[:, topk - 1 - min(CUT_GAP_RANKS, topk - 1)] - ordered[:, topk - 1]
            per_row[f"cut_gap_{l}"].append(gap.float())
            keep_misses(f"dsa_{l}", missed_keys(p["topk"].long(), sel, scores), lo)
        log(f"scored chunk {c + 1} rows [{lo}, {hi}) {time.perf_counter() - began:.1f} s")
    cut = lambda v: torch.cat(v)[:count]  # noqa: E731  (the chunks may run past the scored rows)
    moe_err = {l: cut(v) for l, v in moe_err.items()}
    dsa_err = {l: cut(v) for l, v in dsa_err.items()}
    series = RowSeries()  # every tap's per-row columns; misses as events
    columns = {f"{k}|sums": cut(v) for k, v in sums.items()} | {f"{k}|nrmse": cut(v) for k, v in rows.items()}
    columns |= {k: cut(v) for k, v in per_row.items()}
    for l, e in moe_err.items():
        columns |= {f"moe_{l}|sums": e, f"moe_{l}|nrmse": row_nrmse(e, golden_floor(e[:, SUM_YY].sqrt()))}
    for l, e in dsa_err.items():
        columns |= {f"dsa_{l}|sums": e, f"dsa_{l}|nrmse": row_nrmse(e, golden_floor(e[:, SUM_YY].sqrt()))}
    for l in spec.dsa:  # the device's caches against the FP64 keys
        caches = torch.load(run / f"dsa-l{l}" / "caches.pt", weights_only=True)
        k = deep_products(chain, l, "attn", ["latent", "kpe", "index_key"], 0, count)
        lora = k["latent"].shape[1]
        columns[f"kv_latent_{l}|sums"] = row_sums(k["latent"], caches["kvpe"][:count, :lora])
        columns[f"kv_rope_{l}|sums"] = row_sums(k["kpe"], caches["kvpe"][:count, lora:])
        columns[f"index_key_{l}|sums"] = row_sums(k["index_key"], caches["index"][:count])
    series.append(0, **columns)
    for name, v in misses.items():
        for x in v:
            series.add_events(name, **x)

    def miss_measures(rank: torch.Tensor, selected: torch.Tensor) -> dict:
        rank, selected = rank.float(), selected.float()
        return {
            "misses": int(rank.numel()),
            "miss_rank_median": quantile(rank, 0.5) if rank.numel() else None,
            # no miss in the window: no miss in the top half, not undefined
            "miss_top_half_fraction": float((2 * rank < selected).float().mean()) if rank.numel() else 0.0,
        }

    def window_measures(tap: str, w) -> dict:
        if tap == "stream":
            return {
                n: v for k in stream_keys for n, v in stream_errors(w[f"{k}|sums"], w[f"{k}|nrmse"], f"{k}_").items()
            }
        l = int(tap.split("-l")[1])
        if tap.startswith("kda"):
            if not l:
                return {}
            return {
                **stream_errors(w[f"kda_{l}|sums"], w[f"kda_{l}|nrmse"], "kda_"),
                "kda_row_nrmse_median": quantile(w[f"kda_{l}|nrmse"], 0.5),
            }
        if tap.startswith("moe"):
            recall, nr, mm = w[f"recall_{l}"], w[f"moe_{l}|nrmse"], w[f"moe_{l}"]
            return {
                **errors(w[f"moe_{l}|sums"], "moe_"),
                "moe_nrmse_median": quantile(nr, 0.5),
                "moe_nrmse_p99": quantile(nr, 0.99),
                "recall_vs_reference_mean": float(recall.mean()),
                "set_equal_vs_reference": float((recall == 1).float().mean()),
                "dropped_reference_weight_mean": float(w[f"dropped_{l}"].mean()),
                "dropped_reference_weight_p99": quantile(w[f"dropped_{l}"], 0.99),
                "weight_err_median": quantile(w[f"weight_err_{l}"], 0.5),
                "recall_gpu_vs_reference_mean": float(w[f"gpu_{l}"].mean()),
                **miss_measures(mm["rank"], mm["selected"]),
            }
        nr, mm = w[f"dsa_{l}|nrmse"], w[f"dsa_{l}"]
        gated = slice(max(w.start, topk) - w.start, None)  # rows < top-k select every causal key on both sides
        out = {
            **errors(w[f"dsa_{l}|sums"], "attn_"),
            "attn_nrmse_median": quantile(nr, 0.5),
            "attn_nrmse_p99": quantile(nr, 0.99),
            **{
                k: v
                for name in ("kv_latent", "kv_rope", "index_key")
                for k, v in errors(w[f"{name}_{l}|sums"], f"{name}_").items()
            },
        }
        if w.end > topk:
            out.update(
                recall_mean=float(w[f"topk_{l}"][gated].mean()),
                recall_p05=quantile(w[f"topk_{l}"][gated], 0.05),
                recall_gpu_vs_reference_mean=float(w[f"topk_gpu_{l}"][gated].mean()),
                cut_gap_median=quantile(w[f"cut_gap_{l}"][gated], 0.5),
            )
        return {**out, **miss_measures(mm["rank"], mm["selected"])}

    def states(tap: str) -> list[dict]:
        l, segment, out = int(tap.split("-l")[1]), m["segment"], []
        for entry in complete_run(run / tap).get("states", []):
            row = entry["row"]
            if row % segment:
                continue
            source = (
                deep_state_path(rebuilt, row // segment - 1)
                if l in DEEP_KDA
                else state_path(chain_1m, row // segment - 1)
            )
            ref = torch.load(source, weights_only=True, mmap=True)["kda"][l]
            dev = torch.load(run / tap / entry["file"], weights_only=True)
            out.append({"row": row, **state_errors(dev, ref)})
        return out

    results = {}
    for tap in spec.dirs(run):
        result = {
            "tap": tap,
            "count": count,
            "window": window,
            "reference": {"chain": str(chain), "rebuilt_1m": str(rebuilt_1m), "rebuilt": str(rebuilt)},
            "run": str(run),
            "overall": series.whole(lambda w, tap=tap: window_measures(tap, w)),
            "windows": series.windows(window, lambda w, tap=tap: window_measures(tap, w)),
        }
        if tap.startswith("kda"):
            result["states"] = states(tap)
        (metrics_dir / tap).mkdir(parents=True, exist_ok=True)
        (metrics_dir / tap / "metrics.json").write_text(json.dumps(result, indent=1))
        results[tap] = result
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["score", "rebuild-deep", "score-deep"])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--rebuilt", type=Path, default=REBUILT)
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--chain", default=GOLDEN_256K, help="rebuild-deep / score-deep: the deep chain's golden id")
    parser.add_argument("--out", type=Path, default=DEEP_REBUILT, help="rebuild-deep: the rebuild's directory")
    parser.add_argument("--deep-rebuilt", type=Path, default=DEEP_REBUILT, help="score-deep: rebuild_deep's directory")
    parser.add_argument("--window", type=int, default=65536)
    parser.add_argument("--metrics", type=Path, help="score-deep: where each tap's metrics.json goes (default: --run)")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    def log(msg: str) -> None:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)

    if args.command in ("rebuild-deep", "score-deep"):
        args.chain = open_deep_chain(args.chain).path
    if args.command == "rebuild-deep":
        m = json.loads((args.chain / MANIFEST).read_text())
        checkpoint = Path(m["identity"]["checkpoint"])
        config = json.loads((checkpoint / "config.json").read_text())
        kda_config = KDAConfig.from_model_config(config)
        embedding = checkpoint_tensors(checkpoint, None, ["embed_tokens.weight"])["embed_tokens.weight"]
        if tensor_digest({"w": embedding}) != m["identity"]["weights_sha256"]["embed_tokens"]:
            raise ValueError("the checkpoint's embedding is not the deep chain's")
        token_ids = load_token_ids(args.trace, m["rows"])
        if token_ids_sha256(token_ids) != m["token_ids_sha256"]:
            raise ValueError("the trace's tokens are not the deep chain's")
        # layers 3-6's attention sites as the deep chain loaded them, so their digests are its attn.layers.<l>
        weights, digests = {}, {}
        for l in range(3, 7):
            weights[l], d = load_layer(
                checkpoint, l, kda=l in DEEP_KDA, dense=False, moe=False, dsa_attn=l not in DEEP_KDA, threads=1, log=log
            )
            digests.update({f"attn.{k}": v for k, v in d.items()})
        open_deep_chain(args.chain.name, weights=digests)
        log(f"loaded layers 3-6 from {checkpoint}; digests match the deep chain's")
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
    if args.command == "score-deep":
        m = deep_complete(args.chain)
        checkpoint = Path(m["identity"]["checkpoint"])
        config = json.loads((checkpoint / "config.json").read_text())
        embedding = checkpoint_tensors(checkpoint, None, ["embed_tokens.weight"])["embed_tokens.weight"]
        weights = {l: block_weights(checkpoint, l) for l in range(1, len(m["kinds"]))}
        log(f"loaded the norms and AttnRes queries of layers 1-{len(m['kinds']) - 1}; scoring {args.run}")
        results = score_deep(
            run=args.run,
            chain=args.chain,
            rebuilt_1m=args.rebuilt,
            rebuilt=args.deep_rebuilt,
            trace=Trace(args.trace),
            weights=weights,
            embedding=embedding,
            eps=float(config.get("text_config", config)["rms_norm_eps"]),
            window=args.window,
            metrics_dir=args.metrics or args.run,
            log=log,
        )
        print(json.dumps({tap: r["overall"] for tap, r in results.items()}, indent=1))
        return

    result = score(run=args.run, rebuilt=args.rebuilt, window=args.window, log=log)
    write_score(args.run, result)
    print(json.dumps({"overall": result["overall"], "kda_l0_states": result["kda_l0_states"][-3:]}, indent=1))


if __name__ == "__main__":
    main()
