# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generate Hy4-preview prefill golden traces (host only, no device).

One run prefills a real prompt of the longest requested length through all 78 layers and writes one
trace per requested length. Prefill is causal, so the KV, index keys and top-k of the first N rows of a
long prompt ARE the golden for its N-token prefix: a 4K/32K/128K set costs one 128K pass, and every
trace's tokens are a prefix of the next one's.

Examples::

    # BF16, 4K + 32K + 128K from one LongBench-v2 document, into $HY4_GOLDEN_ROOT/hy4-bf16-longbench-*
    python -m models.demos.deepseek_v3_d_p.reference.hy4.generate_golden --seq-lens 4096,32768,131072 \\
        --longbench code_repo_qa

    # the official MXFP8 checkpoint, same prompt, for the BF16-vs-MXFP8 comparison
    python -m models.demos.deepseek_v3_d_p.reference.hy4.generate_golden --precision mxfp8 \\
        --seq-lens 4096,32768 --longbench code_repo_qa

    # smoke: random tiny model, no checkpoint needed
    python -m models.demos.deepseek_v3_d_p.reference.hy4.generate_golden --random-tiny --seq-lens 64 \\
        --out-root /tmp/hy4_smoke

``--num-layers`` truncates the stack (a partial trace: ``complete: false`` in metadata). ``--state-every``
saves the iHC streams every N layers so ``--resume`` can continue a killed long run.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import sys
import time
from pathlib import Path

import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.hy4.checkpoint import Hy4Checkpoint, save_random_checkpoint
from models.demos.deepseek_v3_d_p.reference.hy4.config import tiny_random_config
from models.demos.deepseek_v3_d_p.reference.hy4.golden import GoldenTraceWriter
from models.demos.deepseek_v3_d_p.reference.hy4.model import Hy4PrefillReference
from models.demos.deepseek_v3_d_p.reference.hy4.paths import golden_root, golden_trace_dir, resolve_hf_model

STATE_FILE = "_hy4_ref_state.pt"


# ------------------------------------------------------------------------------------------ prompts
def _hf_cache_roots() -> list[Path]:
    roots = [os.environ.get("HF_HUB_CACHE"), os.environ.get("HF_HOME") and f"{os.environ['HF_HOME']}/hub"]
    roots.append(str(Path.home() / ".cache" / "huggingface" / "hub"))
    return [Path(r) for r in roots if r]


def longbench_parquets(subset: str) -> list[Path]:
    found = []
    for root in _hf_cache_roots():
        found += glob.glob(str(root / "datasets--*longbench-v2" / "snapshots" / "*" / "data" / subset / "*.parquet"))
    return [Path(p) for p in sorted(set(found))]


def _parquet_texts(path: Path, field: str):
    import pyarrow.parquet as pq

    table = pq.read_table(str(path), columns=[field])
    for i, v in enumerate(table.column(field).to_pylist()):
        yield f"{path}#row{i}", v


def load_tokenizer(model_dir: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(model_dir / "tokenizer.json"))


def tokenize(tok, text: str, bos: int | None) -> list[int]:
    ids = tok.encode(text, add_special_tokens=False).ids
    return ([bos] if bos is not None else []) + ids


def pick_prompt(args, tok, bos, need: int):
    """(token_ids[:need], source description). LongBench: the first row with at least ``need`` tokens."""
    if args.prompt_file:
        p = Path(args.prompt_file)
        if p.suffix == ".parquet":
            cands = _parquet_texts(p, args.prompt_field)
        elif p.suffix == ".json":
            data = json.loads(p.read_text())
            data = data.get("prompts", data) if isinstance(data, dict) else data
            data = data if isinstance(data, list) else [data]
            cands = ((f"{p}#{i}", d["prompt"] if isinstance(d, dict) else d) for i, d in enumerate(data))
        else:
            cands = [(str(p), p.read_text())]
    elif args.longbench:
        files = longbench_parquets(args.longbench)
        if not files:
            raise SystemExit(f"LongBench-v2 subset {args.longbench!r} not found in the HF cache ({_hf_cache_roots()})")
        cands = (c for f in files for c in _parquet_texts(f, args.prompt_field))
    else:
        raise SystemExit("give --prompt-file, --longbench, or --random-tiny (real prompts, not random tokens)")
    best = (0, None)
    for src, text in cands:
        if not isinstance(text, str) or not text.strip():
            continue
        ids = tokenize(tok, text, bos)
        if len(ids) >= need:
            return ids[:need], src
        best = max(best, (len(ids), src), key=lambda t: t[0])
    raise SystemExit(f"no prompt reaches {need} tokens (longest: {best[0]} tokens from {best[1]})")


# ------------------------------------------------------------------------------------------- sinks
class PrefixTraceSink:
    """Fans each layer out to one ``GoldenTraceWriter`` per requested length (row prefix of the long run)."""

    def __init__(self, writers: dict):
        self.writers = writers  # {length: writer}

    def on_layer(self, layer, *, kvpe, index_k=None, topk=None, streams=None):
        for n, w in self.writers.items():
            if layer in w.meta["layers_written"]:
                continue  # resumed run
            w.on_layer(
                layer,
                kvpe=kvpe[:n],
                index_k=None if index_k is None else index_k[:n],
                topk=None if topk is None else topk[:n],
            )

    def on_final(self, *, logits, position):
        for n, w in self.writers.items():
            if position == n - 1:
                w.on_final(logits=logits, position=position)


# -------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seq-lens", default="4096", help="comma-separated prompt lengths, e.g. 4096,32768,131072")
    p.add_argument("--precision", choices=("bf16", "mxfp8"), default="bf16")
    p.add_argument("--model-dir", default=None, help="checkpoint dir (default: paths.resolve_hf_model(precision))")
    p.add_argument("--random-tiny", action="store_true", help="smoke test on a random tiny checkpoint")
    p.add_argument("--prompt-file", default=None, help=".txt, .json ([{prompt}] / {prompts}) or .parquet")
    p.add_argument("--longbench", default=None, help="LongBench-v2 subset from the local HF cache, e.g. code_repo_qa")
    p.add_argument("--prompt-field", default="context", help="parquet column holding the text")
    p.add_argument(
        "--prompt-min-tokens",
        type=int,
        default=None,
        help="pick the first document with at least this many tokens (default: the longest --seq-lens); "
        "pass 131072 on a 4K-only run to get the same document, hence a true prefix of the 128K golden",
    )
    p.add_argument("--no-bos", action="store_true", help="do not prepend bos_token_id")
    p.add_argument("--num-layers", type=int, default=None, help="truncate the stack (partial trace)")
    p.add_argument("--out-root", default=None, help=f"parent dir of the traces (default {golden_root()})")
    p.add_argument("--out", default=None, help="exact output dir (single --seq-lens only)")
    p.add_argument("--kv-dtype", choices=("float32", "bfloat16"), default="float32")
    p.add_argument("--index-k-frame", choices=("device", "hf"), default="device")
    p.add_argument("--no-topk", action="store_true", help="skip the top-k index streams (21 x S x 2048 int32)")
    p.add_argument("--rows-per-shard", type=int, default=5120)
    p.add_argument("--attn-block", type=int, default=128)
    p.add_argument("--proj-block", type=int, default=2048)
    p.add_argument("--threads", type=int, default=0, help="torch threads (0 = all cores)")
    p.add_argument("--state-every", type=int, default=0, help="save resumable state every N layers (0 = never)")
    p.add_argument("--resume", action="store_true", help="continue from <longest trace>/" + STATE_FILE)
    args = p.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    lens = sorted({int(x) for x in args.seq_lens.split(",") if x})
    if args.out and len(lens) != 1:
        raise SystemExit("--out takes a single --seq-lens; use --out-root for several")

    tmp_ckpt = None
    if args.random_tiny:
        import tempfile

        tmp_ckpt = tempfile.TemporaryDirectory()
        model_dir = save_random_checkpoint(tmp_ckpt.name, tiny_random_config())
        precision = "random-tiny"
    else:
        model_dir = Path(args.model_dir) if args.model_dir else resolve_hf_model(args.precision)
        precision = args.precision
    ckpt = Hy4Checkpoint(model_dir)
    cfg = ckpt.config
    if not args.random_tiny and ckpt.precision != args.precision:
        raise SystemExit(f"{model_dir} holds a {ckpt.precision} checkpoint, not {args.precision}")
    n_layers = cfg.num_hidden_layers if args.num_layers is None else min(args.num_layers, cfg.num_hidden_layers)
    bos = None if (args.no_bos or args.random_tiny) else cfg.bos_token_id

    if args.random_tiny:
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(3, cfg.vocab_size, (lens[-1],), generator=g).tolist()
        source = "random tokens (smoke test)"
    else:
        tok = load_tokenizer(model_dir)
        need = max(lens[-1], args.prompt_min_tokens or 0)
        ids, source = pick_prompt(args, tok, bos, need)
        ids = ids[: lens[-1]]
    logger.info(f"[hy4-golden] {precision} {model_dir}: {len(ids)} tokens from {source}; layers [0,{n_layers})")

    if args.random_tiny and not (args.out or args.out_root):
        raise SystemExit("--random-tiny needs --out or --out-root (keep smoke traces out of the shared golden root)")

    def out_dir(n):
        if args.out:
            return Path(args.out)
        name = golden_trace_dir(args.precision, n).name
        if args.random_tiny:
            name = name.replace(args.precision, "random-tiny")
        return (Path(args.out_root) if args.out_root else golden_root()) / name

    dtype = torch.float32 if args.kv_dtype == "float32" else torch.bfloat16
    provenance = {
        "checkpoint": str(model_dir) if not args.random_tiny else "random-tiny",
        "precision": precision,
        "prompt_source": source,
        "bos_prepended": bos is not None,
        "generator": "models.demos.deepseek_v3_d_p.reference.hy4.generate_golden",
        "reference": "Hy4PrefillReference fp32 (HF hy_v4 semantics, RoPE paired interleaved; rope angles fp64)",
        "norm_eps": {"rms_norm_eps": cfg.rms_norm_eps, "low_rank_norm_eps": cfg.low_rank_norm_eps},
        "host": socket.gethostname(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prefix_of_longest": len(lens) > 1,
    }
    rope_pairing = "interleaved"  # Hy4's checkpoint pairing (reference/hy4/model.py)
    writers = {}
    for n in lens:
        d = out_dir(n)
        prev = None
        if args.resume and (d / "metadata.json").exists():
            prev = json.loads((d / "metadata.json").read_text())
            prev_rope = prev.get("frames", {}).get("kvpe", {}).get("pe_rope", "half_split")
            if prev_rope != rope_pairing:
                raise SystemExit(f"--resume: {d} was written with {prev_rope} RoPE, this reference is {rope_pairing}")
        w = GoldenTraceWriter(
            d,
            cfg.truncated(n_layers) if n_layers < cfg.num_hidden_layers else cfg,
            ids[:n],
            rows_per_shard=args.rows_per_shard,
            kv_dtype=dtype,
            index_k_frame=args.index_k_frame,
            save_topk=not args.no_topk,
            extra_metadata=provenance,
            rope_pairing=rope_pairing,
        )
        if prev is not None:
            w.meta["layers_written"] = list(prev.get("layers_written", []))
            w._flush_meta()
        writers[n] = w

    state_path = out_dir(lens[-1]) / STATE_FILE
    start_state = None
    if args.resume:
        if not state_path.exists():
            raise SystemExit(f"--resume: no state at {state_path}")
        start_state = torch.load(state_path, weights_only=True)
        logger.info(f"[hy4-golden] resuming after layer {start_state['layer']}")

    def on_state(state):
        if args.state_every and (state["layer"] + 1) % args.state_every == 0:
            tmp = state_path.with_suffix(".tmp")
            torch.save({"layer": state["layer"], "streams": state["streams"], "topk": state["topk"]}, tmp)
            tmp.replace(state_path)

    ref = Hy4PrefillReference(
        ckpt, cfg, attn_block=args.attn_block, proj_block=args.proj_block, rope_pairing=rope_pairing
    )
    t0 = time.time()
    ref.run(
        torch.tensor(ids),
        PrefixTraceSink(writers),
        num_layers=n_layers,
        final_positions=[n - 1 for n in lens],
        start_state=start_state,
        on_state=on_state,
    )
    for n, w in writers.items():
        out = w.finalize(wall_seconds=round(time.time() - t0, 1))
        logger.success(f"[hy4-golden] {n}-token trace: {out} (complete={w.meta['complete']})")
        print(out)
    if state_path.exists() and all(w.meta["complete"] for w in writers.values()):
        state_path.unlink()
    if tmp_ckpt is not None:
        tmp_ckpt.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
