# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-12 HF golden for image decisions, streamed one text decoder layer at a time on CPU (bf16).

Reproduces ``DecisionModel.predict`` (snapshot ``source/src/autojev/model.py``) for the 8 rows of
``image_decision_prompts``, batch 1, no padding, following ``Qwen3_5Model.forward``
(``modeling_qwen3_5.py`` 1530-1600):

1. text embeddings ``embed_tokens(input_ids)``;
2. vision tower (``hf_vision_reference.build_vision_tower``) ``pooler_output`` cast to the embedding
   dtype and spliced with ``inputs_embeds.masked_scatter(input_ids == image_token_id, ...)``;
3. 3D position ids from HF ``Qwen3_5Model.get_rope_index`` (called unbound; text-only rows get
   ``arange`` on all 3 streams, as ``Qwen3_5TextModel`` builds them) -> ``Qwen3_5TextRotaryEmbedding``;
4. 64 x ``Qwen3_5DecoderLayer`` streamed (``hf_decision_golden.stream_layers``) -> final norm ->
   last token -> ``readout(h).float()`` -> mask -> ``/ temperature`` -> softmax.

Everything runs in bf16 (the app's CUDA dtype); the only fp32 is what HF/the app do internally.

Usage::

    python -m models.demos.pplx_decider_v1_27b.reference.hf_image_decision_golden --harness
    python -m models.demos.pplx_decider_v1_27b.reference.hf_image_decision_golden --max-layers 2
    python -m models.demos.pplx_decider_v1_27b.reference.hf_image_decision_golden
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import types
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file, save_file
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5TextRotaryEmbedding

from models.demos.pplx_decider_v1_27b.reference.hf_decision_golden import DEFAULT_OUT as TEXT_GOLDEN_DIR
from models.demos.pplx_decider_v1_27b.reference.hf_decision_golden import (
    decision_head,
    embed_prompts,
    pcc,
    peak_rss_gb,
    position_embeddings,
    rss_gb,
    stream_layers,
)
from models.demos.pplx_decider_v1_27b.reference.hf_reference import build_decoder_layer, build_final_norm, build_readout
from models.demos.pplx_decider_v1_27b.reference.hf_vision_reference import (
    GOLDEN_DIR,
    VisionSnapshotReader,
    build_vision_tower,
    default_dtype,
)
from models.demos.pplx_decider_v1_27b.reference.image_decision_prompts import (
    build_image_prompt_set,
    encode,
    image_tokenizer,
)

DTYPE = torch.bfloat16
DEFAULT_OUT = GOLDEN_DIR / "e2e"
HARNESS_ROW = "v02_count_circles"  # 936 patches (not a multiple of 128), 234 image tokens
TEXT_ROW = "s01_ticket_routing"  # shortest-but-one stage-6 row (178 tokens)


def rope_index(config, enc: dict) -> torch.Tensor:
    """3D position ids ``[3, 1, S]``: HF ``Qwen3_5Model.get_rope_index`` for image rows, ``arange``
    on all three streams for text-only rows (``Qwen3_5TextModel.forward`` with ``position_ids=None``)."""
    ids = enc["input_ids"]
    if "image_grid_thw" not in enc:
        return torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1)
    shim = types.SimpleNamespace(config=config)
    shim.get_vision_position_ids = types.MethodType(Qwen3_5Model.get_vision_position_ids, shim)
    pos, _ = Qwen3_5Model.get_rope_index(
        shim,
        ids,
        enc["mm_token_type_ids"],
        image_grid_thw=enc["image_grid_thw"],
        attention_mask=enc["attention_mask"],
    )
    return pos


def mrope_position_embeddings(text_config, position_ids: torch.Tensor, dtype=DTYPE):
    """cos/sin ``[1, S, 64]`` from 3D ids, cast to the hidden dtype as inside ``Qwen3_5TextModel``."""
    rope = Qwen3_5TextRotaryEmbedding(text_config)
    return rope(torch.empty(1, position_ids.shape[-1], 1, dtype=dtype), position_ids)


@torch.no_grad()
def prepare(reader: VisionSnapshotReader, tower, table: torch.Tensor, enc: dict, dtype=DTYPE):
    """``inputs_embeds`` (spliced), 3D ``position_ids`` and the rotary cos/sin for one encoded row."""
    ids = enc["input_ids"]
    embeds = table[ids[0]][None].to(dtype)  # == embed_tokens(input_ids) (bf16 table)
    if "pixel_values" in enc:
        feats = tower(enc["pixel_values"].type(tower.dtype), enc["image_grid_thw"]).pooler_output
        feats = feats.to(embeds.dtype)
        mask = (ids == reader.config.image_token_id).unsqueeze(-1)
        assert int(mask.sum()) * embeds.shape[-1] == feats.numel()
        embeds = embeds.masked_scatter(mask, feats)
    pos = rope_index(reader.config, enc)
    return embeds, pos, mrope_position_embeddings(reader.text_config, pos, dtype)


# ----------------------------------------------------------------------------------------------
# Harness proofs
# ----------------------------------------------------------------------------------------------


def build_hf_model_prefix(reader: VisionSnapshotReader, n_layers: int, dtype=DTYPE) -> Qwen3_5Model:
    """A real ``Qwen3_5Model`` with the vision tower, embeddings, text layers ``0..n-1`` and the final
    norm from the snapshot (strict load; snapshot keys already match ``Qwen3_5Model`` names)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(reader.path, local_files_only=True)
    cfg.text_config.num_hidden_layers = n_layers
    cfg.text_config.layer_types = cfg.text_config.layer_types[:n_layers]
    with default_dtype(dtype):
        model = Qwen3_5Model._from_config(cfg, dtype=dtype, attn_implementation="sdpa")
    wanted = set(model.state_dict())
    sd = {}
    for prefix in ["visual.", "language_model.embed_tokens.", "language_model.norm."] + [
        f"language_model.layers.{i}." for i in range(n_layers)
    ]:
        sd.update({prefix + k: v for k, v in reader.tensors_with_prefix(prefix).items()})
    assert set(sd) == wanted, (sorted(set(sd) ^ wanted))[:10]
    model.load_state_dict(sd, strict=True)
    return model.eval()


def cmp(a: torch.Tensor, b: torch.Tensor) -> dict:
    return {
        "bit_exact": bool(torch.equal(a, b)),
        "pcc": pcc(a, b),
        "max_abs_diff": float((a.float() - b.float()).abs().max()),
    }


@torch.no_grad()
def harness(args) -> dict:
    reader = VisionSnapshotReader(args.snapshot)
    tok = image_tokenizer(args.snapshot)
    res: dict = {"dtype": "bf16", "layers": [0, 1, 2, 3]}
    n = 4
    # (1) image row: streamed glue vs HF Qwen3_5Model.forward non-streamed (4 real layers + final norm)
    row = next(r for r in build_image_prompt_set(tok, args.out.parent / "images") if r["id"] == HARNESS_ROW)
    enc = row["enc"]
    tower = build_vision_tower(reader, DTYPE)
    table = reader.embedding_weight()
    embeds, pos, pe = prepare(reader, tower, table, enc)
    del table, tower
    hs = stream_layers([embeds.clone()], [pe], lambda i: build_decoder_layer(reader, i, DTYPE), range(n))
    mine = build_final_norm(reader, DTYPE)(hs[0])
    logger.info(f"streamed {n} layers, rss {rss_gb():.1f} GB")
    model = build_hf_model_prefix(reader, n)
    logger.info(f"HF {n}-layer Qwen3_5Model built, rss {rss_gb():.1f} GB")
    seen = {}

    def grab(_mod, _args, kwargs):
        seen["inputs_embeds"] = kwargs["inputs_embeds"].clone()
        seen["position_ids"] = kwargs["position_ids"].clone()

    model.language_model.register_forward_pre_hook(grab, with_kwargs=True)
    out = model(**enc, use_cache=False).last_hidden_state
    res["image_row"] = {
        "id": HARNESS_ROW,
        "seq_len": int(enc["input_ids"].shape[1]),
        "patches": row["patches"],
        "inputs_embeds_vs_hf": cmp(embeds, seen["inputs_embeds"]),
        "position_ids_equal_hf": bool(torch.equal(pos, seen["position_ids"])),
        "final_norm_hidden_vs_hf": cmp(mine, out),
        "last_token_vs_hf": cmp(mine[:, -1], out[:, -1]),
    }
    logger.info(f"harness image row: {res['image_row']}")
    del model, out, seen
    # (2) text-only row through the image-capable path == stage-6 path == stored stage-6 trace
    text_row = next(
        json.loads(line) for line in open(TEXT_GOLDEN_DIR / "prompts.jsonl") if json.loads(line)["id"] == TEXT_ROW
    )
    enc_t = encode(tok, text_row["row"])
    assert enc_t["input_ids"][0].tolist() == text_row["input_ids"] and "pixel_values" not in enc_t
    table = reader.embedding_weight()
    emb_new, pos_t, pe_new = prepare(reader, None, table, enc_t)
    del table
    emb_old = embed_prompts(reader, [text_row["input_ids"]], DTYPE)[0]
    pe_old = position_embeddings(reader.text_config, emb_old.shape[1], DTYPE)
    trace_new, trace_old = [], []
    stream_layers(
        [emb_new],
        [pe_new],
        lambda i: build_decoder_layer(reader, i, DTYPE),
        range(n),
        on_layer=lambda i, h, b, t: trace_new.append(h[0][0, -1].clone()),
    )
    stream_layers(
        [emb_old],
        [pe_old],
        lambda i: build_decoder_layer(reader, i, DTYPE),
        range(n),
        on_layer=lambda i, h, b, t: trace_old.append(h[0][0, -1].clone()),
    )
    stored = load_file(str(TEXT_GOLDEN_DIR / "golden_bf16.safetensors"))[f"{TEXT_ROW}.layer_last_hidden"][:n]
    new = torch.stack(trace_new)
    res["text_row"] = {
        "id": TEXT_ROW,
        "seq_len": len(text_row["input_ids"]),
        "embeds_equal": bool(torch.equal(emb_new, emb_old)),
        "cos_sin_equal": bool(torch.equal(pe_new[0], pe_old[0]) and torch.equal(pe_new[1], pe_old[1])),
        "layers0_3_last_token_vs_stage6_code": cmp(new, torch.stack(trace_old)),
        "layers0_3_last_token_vs_stored_stage6_golden": cmp(new.float(), stored),
    }
    logger.info(f"harness text row: {res['text_row']}")
    res["peak_rss_gb"] = round(peak_rss_gb(), 2)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "harness_check.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


# ----------------------------------------------------------------------------------------------
# Main golden
# ----------------------------------------------------------------------------------------------


@torch.no_grad()
def run_golden(args) -> None:
    t0 = time.time()
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = VisionSnapshotReader(args.snapshot)
    tok = image_tokenizer(args.snapshot)
    code_check = tok.verify_answer_codes()
    rows = build_image_prompt_set(tok, out_dir.parent / "images")
    cfg = reader.text_config
    n_layers = min(args.max_layers, cfg.num_hidden_layers)
    full = n_layers == cfg.num_hidden_layers
    tower = build_vision_tower(reader, DTYPE)
    table = reader.embedding_weight()
    prepped = [prepare(reader, tower, table, r["enc"]) for r in rows]
    del table, tower
    t_vision = time.time() - t0
    hs = [p[0].clone() for p in prepped]
    pes = [p[2] for p in prepped]
    trace = torch.zeros(len(rows), cfg.num_hidden_layers, cfg.hidden_size, dtype=DTYPE)
    timings: list[dict] = []

    def on_layer(i, hs_, built, total):
        for p, h in enumerate(hs_):
            trace[p, i] = h[0, -1]
        timings.append({"layer": i, "kind": cfg.layer_types[i], "build_s": round(built, 2), "total_s": round(total, 2)})
        logger.info(
            f"layer {i:2d} {cfg.layer_types[i]:<16} build {built:5.1f}s total {total:6.1f}s "
            f"rss {rss_gb():.1f} GB peak {peak_rss_gb():.1f} GB"
        )

    stream_layers(hs, pes, lambda i: build_decoder_layer(reader, i, DTYPE), range(n_layers), on_layer)
    if not full:
        per = [t["total_s"] for t in timings]
        logger.info(f"estimate: {per} s/layer -> 64 layers ~ {64 * sum(per) / len(per) / 60:.1f} min")
        return
    final_norm = build_final_norm(reader, DTYPE)
    readout = build_readout(reader, DTYPE)
    temperature = reader.temperature
    tensors: dict[str, torch.Tensor] = {}
    summary_rows = []
    with open(out_dir / "prompts.jsonl", "w") as f:
        for idx, r in enumerate(rows):
            keep = ("id", "type", "count", "seq_len", "bucket", "image_size", "grid_thw", "patches", "image_tokens")
            rec = {"idx": idx, **{k: r[k] for k in keep}, "row": r["row"], "meta": r["meta"]}
            f.write(json.dumps({**rec, "input_ids": r["input_ids"], "text": tok.text(r["row"])}) + "\n")
    for idx, (r, h, (embeds, pos, _)) in enumerate(zip(rows, hs, prepped)):
        h_last = final_norm(h)[:, -1]
        logits, probs = decision_head(h_last, readout, r["count"], temperature)
        keys = tok.autojev.options(r["row"]["question"])[0]
        top = torch.topk(probs, min(2, r["count"]))
        gap = float(top.values[0] - top.values[1]) if r["count"] > 1 else 1.0
        rid = r["id"]
        tensors[f"{rid}.input_ids"] = torch.tensor(r["input_ids"], dtype=torch.int64)
        tensors[f"{rid}.position_ids"] = pos[:, 0].contiguous()
        tensors[f"{rid}.inputs_embeds"] = embeds[0].contiguous()
        tensors[f"{rid}.final_hidden"] = h_last[0].contiguous()
        tensors[f"{rid}.logits"] = logits.contiguous()
        tensors[f"{rid}.probs"] = probs.contiguous()
        tensors[f"{rid}.layer_last_hidden"] = trace[idx].contiguous()
        choice = keys[int(top.indices[0])]
        summary_rows.append(
            {
                "idx": idx,
                "id": rid,
                "type": r["type"],
                "seq_len": r["seq_len"],
                "bucket": r["bucket"],
                "count": r["count"],
                "image_size": r["image_size"],
                "grid_thw": r["grid_thw"],
                "patches": r["patches"],
                "image_tokens": r["image_tokens"],
                "argmax": int(top.indices[0]),
                "choice": choice,
                "max_prob": float(top.values[0]),
                "top2_gap": gap,
                "expected": r["meta"]["expected"],
                "correct": choice == r["meta"]["expected"],
                "answer": tok.autojev.answer(r["row"]["question"], probs.tolist()),
            }
        )
        logger.info(
            f"{idx} {rid:<22} S={r['seq_len']} -> {choice!r} p={float(top.values[0]):.3f} gap={gap:.3f} "
            f"(expected {r['meta']['expected']})"
        )
    save_file(
        tensors,
        str(out_dir / "golden_bf16.safetensors"),
        metadata={
            "dtype": "bf16",
            "temperature": repr(temperature),
            "logits": "raw readout(hidden).float(), unmasked",
            "probs": "softmax(masked_logits / temperature)[:count], fp32 (app)",
            "layer_last_hidden": "last-token residual after each of the 64 layers (pre final norm), bf16",
            "position_ids": "[3, S] T/H/W mRoPE ids from get_rope_index",
            "inputs_embeds": "[S, 5120] text embeddings with image rows replaced by the merger output",
        },
    )
    runtime = time.time() - t0
    summary = {
        "dtype": "bf16",
        "snapshot": str(reader.path),
        "temperature": temperature,
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "answer_code_check": code_check,
        "n_rows": len(rows),
        "total_tokens": sum(r["seq_len"] for r in rows),
        "runtime_s": round(runtime, 1),
        "vision_and_embed_s": round(t_vision, 1),
        "peak_rss_gb": round(peak_rss_gb(), 2),
        "layer_seconds_total": round(sum(t["total_s"] for t in timings), 1),
        "rows": summary_rows,
    }
    (out_dir / "summary_bf16.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    cols = [
        "idx",
        "id",
        "type",
        "seq_len",
        "bucket",
        "count",
        "patches",
        "image_tokens",
        "argmax",
        "choice",
        "max_prob",
        "top2_gap",
        "expected",
        "correct",
    ]
    with open(out_dir / "summary_bf16.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)
    with open(out_dir / "layer_times_bf16.jsonl", "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in timings)
    logger.info(f"done in {runtime / 60:.1f} min, peak RSS {peak_rss_gb():.1f} GB -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--max-layers", type=int, default=64, help="< 64: timing estimate only, nothing saved")
    parser.add_argument("--harness", action="store_true", help="streamed-vs-HF and text-row identity checks")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.snapshot is None:
        from models.demos.pplx_decider_v1_27b.reference.decision_prompts import DEFAULT_SNAPSHOT

        args.snapshot = DEFAULT_SNAPSHOT
    if args.harness:
        harness(args)
        return
    run_golden(args)


if __name__ == "__main__":
    main()
