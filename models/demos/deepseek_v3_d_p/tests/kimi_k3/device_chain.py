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
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import MANIFEST, segment_path
from models.demos.deepseek_v3_d_p.tests.kimi_k3.chain_vs_trace import key_sums, pcc_nrmse
from models.demos.deepseek_v3_d_p.tests.kimi_k3.fp64_chain import F64, ChainModel, attn_res64, rms64, state_path
from models.demos.deepseek_v3_d_p.tests.kimi_k3.kda_chain_feed import rebuilt_manifest, rebuilt_path, rel
from models.demos.deepseek_v3_d_p.tests.kimi_k3.teacher_feed import Trace, chunk_path, complete_run, quantile

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
TAPS = ("kda-l0", *INPUTS, "stream")
STREAM_KEYS = ("kda_input_1", "kda_input_2", "moe_input_1", "moe_input_2", "dsa_input_3", "live_0", "live_1", "live_2")


def tap_dirs(out: Path) -> dict[str, Path]:
    """The run directory of each tap of a device-chain run in ``out``."""
    return {name: out / name for name in TAPS}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["score"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rebuilt", type=Path, default=REBUILT)
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--window", type=int, default=65536)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    def log(msg: str) -> None:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)

    chain = Path(json.loads((args.rebuilt / MANIFEST).read_text())["chain"]["chain"])
    checkpoint = Path(json.loads((chain / MANIFEST).read_text())["identity"]["checkpoint"])
    log(f"loading weights (no routed experts) from {checkpoint}")
    model = ChainModel.from_checkpoint(checkpoint, args.threads, log, routed_experts=False)
    result = score(run=args.run, rebuilt=args.rebuilt, trace=args.trace, model=model, window=args.window, log=log)
    print(json.dumps({"overall": result["overall"], "kda_l0_states": result["kda_l0_states"][-3:]}, indent=1))


if __name__ == "__main__":
    main()
