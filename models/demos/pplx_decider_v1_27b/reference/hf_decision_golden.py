# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-6 HF golden for end-to-end decision agreement, streamed one decoder layer at a time on CPU.

For the 25 rows of ``decision_prompts.build_prompt_set`` this reproduces
``DecisionModel.predict`` (snapshot ``source/src/autojev/model.py``) with batch 1 and no padding:
embed -> 64 x ``Qwen3_5DecoderLayer`` (each built from safetensors with ``strict=True``, 1D RoPE
positions ``arange(S)``, causal, no padding mask) -> final norm -> LAST token -> ``readout`` ->
``.float()`` -> mask options >= count with -1e9 -> ``/ temperature`` -> fp32 softmax.
Weights and activations use ``--dtype`` (default bf16, as the app's CUDA inference); RoPE cos/sin
are cast to that dtype as ``Qwen3_5TextRotaryEmbedding`` does inside the HF model.

Never builds the full model: one layer (~0.8 GB bf16) plus all prompts' activations live in RAM.

Usage::

    # glue check against the HF Qwen3_5TextModel on a tiny random config (seconds)
    python -m models.demos.pplx_decider_v1_27b.reference.hf_decision_golden --tiny-check
    # harness identity vs the stage-1 fp32 goldens (S=8192 prompt)
    python -m models.demos.pplx_decider_v1_27b.reference.hf_decision_golden --identity
    # cost estimate on the first N layers, then the full run
    python -m models.demos.pplx_decider_v1_27b.reference.hf_decision_golden --max-layers 2
    python -m models.demos.pplx_decider_v1_27b.reference.hf_decision_golden --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import time
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import save_file
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

from models.demos.pplx_decider_v1_27b.reference.decision_prompts import DEFAULT_SNAPSHOT, AppTokenizer, build_prompt_set
from models.demos.pplx_decider_v1_27b.reference.hf_reference import (
    NUM_OPTIONS,
    SnapshotReader,
    build_decoder_layer,
    build_final_norm,
    build_readout,
    layer_forward,
    rotary_cos_sin,
)

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
GOLDEN_ROOT = Path("/local/ttuser/gtobar/artifacts/pplx_decider/goldens")
DEFAULT_OUT = GOLDEN_ROOT / "decisions"
# Rows whose last-token hidden after every layer is the designated layer-wise trace
# (short, ~2k, ~6k tokens). The trace is saved for every row; these three are flagged.
TRACE_IDS = ("s01_ticket_routing", "m06_server_500", "x01_log_most_errors")


def rss_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    return float("nan")


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def position_embeddings(config, seq_len: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin as the HF text model computes them: 1D positions arange(S), cast to the hidden dtype."""
    rope = Qwen3_5TextRotaryEmbedding(config)
    return rope(torch.empty(1, seq_len, 1, dtype=dtype), torch.arange(seq_len)[None])


@torch.no_grad()
def stream_layers(hs: list[torch.Tensor], pes: list, layer_fn, layers, on_layer=None) -> list[torch.Tensor]:
    """Push every prompt through each layer in ``layers``; one layer is resident at a time."""
    for i in layers:
        start = time.time()
        layer = layer_fn(i)
        built = time.time() - start
        for p in range(len(hs)):
            hs[p] = layer_forward(layer, hs[p], pes[p])
        del layer
        if on_layer is not None:
            on_layer(i, hs, built, time.time() - start)
    return hs


@torch.no_grad()
def decision_head(h_last: torch.Tensor, readout, count: int, temperature: float):
    """``DecisionModel.forward`` + ``predict`` on one last-token hidden [1, 5120]."""
    logits = readout(h_last).float()
    mask = torch.arange(NUM_OPTIONS)[None] >= torch.tensor([count])[:, None]
    masked = logits.masked_fill(mask, -1e9)
    probs = (masked / temperature).softmax(-1)
    return logits[0], probs[0, :count]


# ----------------------------------------------------------------------------------------------
# Glue check: streamed pipeline == HF Qwen3_5TextModel forward on a tiny random config
# ----------------------------------------------------------------------------------------------


@torch.no_grad()
def tiny_check(reader: SnapshotReader, dtype: torch.dtype) -> dict:
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    base = reader.text_config.to_dict()
    base.update(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        vocab_size=512,
        layer_types=reader.text_config.layer_types[:4],
    )
    for key in ("model_type", "transformers_version", "architectures"):
        base.pop(key, None)
    cfg = Qwen3_5TextConfig(**base)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = Qwen3_5TextModel(cfg).eval()
    for name, param in model.named_parameters():  # make norms non-trivial
        if "norm" in name:
            param.add_(0.1 * torch.randn_like(param))
    # Load the reference the way the app does (from_pretrained(dtype=...)): this keeps the rotary
    # inv_freq buffer in fp32, whereas module.to(bf16) would round it (measured: 0.0156 max diff).
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        model = Qwen3_5TextModel.from_pretrained(tmp, dtype=dtype, attn_implementation="sdpa").eval()
    seq = 77
    ids = torch.randint(0, cfg.vocab_size, (1, seq))
    ref = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state[:, -1]

    def layer_fn(i):
        layer = Qwen3_5DecoderLayer(cfg, i)
        layer.load_state_dict(model.layers[i].state_dict(), strict=True)
        return layer.to(dtype).eval()

    hs = stream_layers([model.embed_tokens(ids)], [position_embeddings(cfg, seq, dtype)], layer_fn, range(4))
    mine = model.norm(hs[0])[:, -1]
    result = {
        "dtype": str(dtype),
        "max_abs_diff": float((mine.float() - ref.float()).abs().max()),
        "pcc": pcc(mine, ref),
    }
    logger.info(f"tiny glue check {result}")
    return result


# ----------------------------------------------------------------------------------------------
# Sanity (a): harness identity against the stage-1 fp32 goldens (S=8192 prompt)
# ----------------------------------------------------------------------------------------------


def _load(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    return obj if isinstance(obj, torch.Tensor) else obj["x"]


@torch.no_grad()
def identity_check(reader: SnapshotReader, out_dir: Path) -> dict:
    d = GOLDEN_ROOT / "S8192"
    cfg = reader.text_config
    ids = _load(d / "ids.pt")
    seq = ids.shape[1]
    res: dict = {"seq_len": seq}

    def cmp(name, mine, gold):
        res[name] = {
            "pcc": pcc(mine, gold),
            "max_abs_diff": float((mine.float() - gold.float()).abs().max()),
            "last_token_pcc": pcc(mine[:, -1], gold[:, -1]),
        }
        logger.info(f"identity {name}: {res[name]}  rss={rss_gb():.1f} GB")

    fp32 = torch.float32
    pe32 = position_embeddings(cfg, seq, fp32)
    emb = embed_prompts(reader, [ids[0].tolist()], fp32)
    gold_l0 = _load(d / "L0_input.pt")
    cmp("embedding_vs_L0_input", emb[0], gold_l0)
    # layers 0..2 (Gated DeltaNet) from the embedding -> stage-1 L3_input
    l0_out = {}
    hs = stream_layers(
        emb,
        [pe32],
        lambda i: build_decoder_layer(reader, i, fp32),
        range(3),
        on_layer=lambda i, h, b, t: l0_out.setdefault(i, h[0].clone()) if i == 0 else None,
    )
    gold_l3 = _load(d / "L3_input.pt")
    cmp("layers0-2_vs_L3_input", hs[0], gold_l3)
    del hs
    # layer 3 (full attention) from L3_input: this harness vs the stage-1 code path (fp32 cos/sin helper)
    mine3 = stream_layers([gold_l3.clone()], [pe32], lambda i: build_decoder_layer(reader, i, fp32), [3])[0]
    stage1_3 = layer_forward(build_decoder_layer(reader, 3, fp32), gold_l3, rotary_cos_sin(cfg, seq))
    cmp("layer3_vs_stage1_codepath", mine3, stage1_3)
    del stage1_3
    # layer 63 (full attention) from L63_input -> stage-1 final_input (a saved file)
    gold_63 = _load(d / "L63_input.pt")
    mine63 = stream_layers([gold_63], [pe32], lambda i: build_decoder_layer(reader, i, fp32), [63])[0]
    cmp("layer63_vs_final_input", mine63, _load(d / "final_input.pt"))
    del gold_63, mine63
    # information: bf16 vs fp32 on single layers 0 and 3 from the same fp32 inputs
    bf = torch.bfloat16
    pebf = position_embeddings(cfg, seq, bf)
    b0 = stream_layers([gold_l0.to(bf)], [pebf], lambda i: build_decoder_layer(reader, i, bf), [0])[0]
    cmp("bf16_vs_fp32_layer0", b0, l0_out[0])
    b3 = stream_layers([gold_l3.to(bf)], [pebf], lambda i: build_decoder_layer(reader, i, bf), [3])[0]
    cmp("bf16_vs_fp32_layer3", b3, mine3)
    res["peak_rss_gb"] = peak_rss_gb()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "identity_check.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


# ----------------------------------------------------------------------------------------------
# Main golden
# ----------------------------------------------------------------------------------------------


def embed_prompts(reader: SnapshotReader, ids_list: list[list[int]], dtype: torch.dtype) -> list[torch.Tensor]:
    table = reader.embedding_weight()  # bf16 [248320, 5120]; gathered rows only are kept
    out = [table[torch.tensor(ids)][None].to(dtype) for ids in ids_list]
    del table
    return out


def save_state(path: Path, next_layer: int, hs, trace) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"next_layer": next_layer, "hs": hs, "trace": trace}, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def run_golden(args) -> None:
    dtype = DTYPES[args.dtype]
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = AppTokenizer(args.snapshot)
    code_check = tok.verify_answer_codes()
    rows = [r for r in build_prompt_set(tok) if r["id"] not in set(args.drop)]
    logger.info(f"{len(rows)} rows, {sum(r['seq_len'] for r in rows)} tokens; answer codes {code_check}")
    reader = SnapshotReader(args.snapshot)
    cfg = reader.text_config
    n_layers = min(args.max_layers, cfg.num_hidden_layers)
    full = n_layers == cfg.num_hidden_layers

    if full:
        with open(out_dir / "prompts.jsonl", "w") as f:
            for idx, r in enumerate(rows):
                f.write(
                    json.dumps(
                        {
                            "idx": idx,
                            **{
                                k: r[k]
                                for k in (
                                    "id",
                                    "type",
                                    "band",
                                    "n",
                                    "count",
                                    "seq_len",
                                    "bucket",
                                    "row",
                                    "meta",
                                    "input_ids",
                                )
                            },
                            "text": tok.text(r["row"]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    state_path = out_dir / f"state_{args.dtype}.pt"
    start_layer = 0
    trace = torch.zeros(len(rows), cfg.num_hidden_layers, cfg.hidden_size, dtype=torch.float32)
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        hs, trace, start_layer = state["hs"], state["trace"], state["next_layer"]
        assert [h.shape[1] for h in hs] == [r["seq_len"] for r in rows], "state does not match the prompt set"
        logger.info(f"resumed at layer {start_layer}")
    else:
        hs = embed_prompts(reader, [r["input_ids"] for r in rows], dtype)
    pes = [position_embeddings(cfg, r["seq_len"], dtype) for r in rows]
    timings: list[dict] = []
    timing_path = out_dir / f"layer_times_{args.dtype}.jsonl"

    def on_layer(i, hs_, built, total):
        for p, h in enumerate(hs_):
            trace[p, i] = h[0, -1].float()
        rec = {
            "layer": i,
            "kind": cfg.layer_types[i],
            "build_s": round(built, 2),
            "total_s": round(total, 2),
            "rss_gb": round(rss_gb(), 2),
            "peak_rss_gb": round(peak_rss_gb(), 2),
        }
        timings.append(rec)
        with open(timing_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        done = len(timings)
        eta_h = (cfg.num_hidden_layers - i - 1) * (time.time() - t_stream) / done / 3600
        logger.info(
            f"layer {i:2d} {cfg.layer_types[i]:<16} build {built:5.1f}s total {total:6.1f}s "
            f"rss {rec['rss_gb']:.1f} GB peak {rec['peak_rss_gb']:.1f} GB  ETA(64) {eta_h:.2f} h"
        )
        if full and (i + 1) % args.save_every == 0:
            save_state(state_path, i + 1, hs_, trace)

    t_stream = time.time()
    stream_layers(hs, pes, lambda i: build_decoder_layer(reader, i, dtype), range(start_layer, n_layers), on_layer)
    if not full:
        per = [t["total_s"] for t in timings]
        logger.info(
            f"estimate: {per} s/layer -> 64 layers ~ {64 * sum(per) / len(per) / 3600:.2f} h "
            f"(peak RSS {peak_rss_gb():.1f} GB)"
        )
        return
    save_state(state_path, n_layers, hs, trace)

    final_norm = build_final_norm(reader, dtype)
    readout = build_readout(reader, dtype)
    temperature = reader.temperature
    assert abs(temperature - tok.temperature) < 1e-12
    tensors: dict[str, torch.Tensor] = {}
    summary_rows = []
    for idx, (r, h) in enumerate(zip(rows, hs)):
        h_last = final_norm(h)[:, -1]
        logits, probs = decision_head(h_last, readout, r["count"], temperature)
        keys = tok.autojev.options(r["row"]["question"])[0]
        top = torch.topk(probs, min(2, r["count"]))
        gap = float(top.values[0] - top.values[1]) if r["count"] > 1 else 1.0
        ans = tok.autojev.answer(r["row"]["question"], probs.tolist())
        rid = r["id"]
        tensors[f"{rid}.input_ids"] = torch.tensor(r["input_ids"], dtype=torch.int64)
        tensors[f"{rid}.final_hidden"] = h_last[0].float().contiguous()
        tensors[f"{rid}.logits"] = logits.contiguous()
        tensors[f"{rid}.probs"] = probs.contiguous()
        tensors[f"{rid}.layer_last_hidden"] = trace[idx].contiguous()
        summary_rows.append(
            {
                "idx": idx,
                "id": rid,
                "type": r["type"],
                "band": r["band"],
                "seq_len": r["seq_len"],
                "bucket": r["bucket"],
                "count": r["count"],
                "argmax": int(top.indices[0]),
                "choice": keys[int(top.indices[0])],
                "max_prob": float(top.values[0]),
                "top2_gap": gap,
                "expected": r["meta"]["expected"],
                "trace_designated": rid in TRACE_IDS,
                "answer": ans,
            }
        )
        logger.info(
            f"{idx:2d} {rid:<26} S={r['seq_len']:<5} -> {keys[int(top.indices[0])]!r} "
            f"p={float(top.values[0]):.3f} gap={gap:.3f} (expected {r['meta']['expected']})"
        )
    save_file(
        tensors,
        str(out_dir / f"golden_{args.dtype}.safetensors"),
        metadata={
            "dtype": args.dtype,
            "temperature": repr(temperature),
            "logits": "raw readout(hidden).float(), unmasked",
            "probs": "softmax(masked_logits / temperature)[:count], fp32",
            "layer_last_hidden": "last-token residual after each of the 64 layers (pre final norm), fp32",
        },
    )
    runtime = time.time() - t0
    summary = {
        "dtype": args.dtype,
        "snapshot": str(args.snapshot),
        "temperature": temperature,
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "answer_code_check": code_check,
        "n_rows": len(rows),
        "total_tokens": sum(r["seq_len"] for r in rows),
        "runtime_s": round(runtime, 1),
        "peak_rss_gb": round(peak_rss_gb(), 2),
        "trace_designated": list(TRACE_IDS),
        "layer_seconds_total": round(sum(t["total_s"] for t in timings), 1),
        "resumed_from_layer": start_layer,
        "rows": summary_rows,
    }
    (out_dir / f"summary_{args.dtype}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    cols = [
        "idx",
        "id",
        "type",
        "band",
        "seq_len",
        "bucket",
        "count",
        "argmax",
        "choice",
        "max_prob",
        "top2_gap",
        "expected",
        "trace_designated",
    ]
    with open(out_dir / f"summary_{args.dtype}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)
    logger.info(f"done in {runtime / 3600:.2f} h, peak RSS {peak_rss_gb():.1f} GB -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--max-layers", type=int, default=64, help="< 64: timing estimate only, nothing saved")
    parser.add_argument("--drop", nargs="*", default=[], help="row ids to leave out (cost reduction)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=4, help="checkpoint activations every N layers")
    parser.add_argument("--identity", action="store_true", help="sanity (a) vs the stage-1 fp32 goldens")
    parser.add_argument("--tiny-check", action="store_true", help="glue check vs HF Qwen3_5TextModel (tiny)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.tiny_check:
        reader = SnapshotReader(args.snapshot)
        for dt in DTYPES.values():
            tiny_check(reader, dt)
        return
    if args.identity:
        identity_check(SnapshotReader(args.snapshot), args.out)
        return
    run_golden(args)


if __name__ == "__main__":
    main()
