# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared pieces of the GPT-OSS batch-1 prefill TTFT harness and accuracy gate.

The model is built the way models/demos/gpt_oss/demo/text_demo.py builds it (prepare_gpt_oss_generator_args,
paged attention, block size 64, one user) and prefill is timed through the same entry point the demo times:
Generator.prefill_forward_text, which returns last-token logits on host.
"""

import os
import subprocess
import time
from pathlib import Path

import torch
from loguru import logger

PREFILL_LENGTHS = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
FAST_GATE_LENGTHS = [128, 1024, 8192]
RUNS_DIR = Path(os.getenv("PREFILL_RUNS_DIR", "/home/ttuser/gtobar/tt-ops/runs/gpt-oss-20b-prefill"))
GOLDENS_DIR = Path(os.getenv("PREFILL_GOLDENS_DIR", str(RUNS_DIR / "baseline" / "goldens")))
PAGE_BLOCK_SIZE = 64
# Real-text corpus for the accuracy gate: the repo's tech reports (prose plus code), tokenized once and frozen
# into GOLDENS_DIR/prompt_tokens.pt so later edits to these files never change the gate's prompts.
CORPUS_GLOB = "tech_reports/*.md"
CORPUS_EXCLUDE = ("/results/",)
VOCAB_RANDOM_HIGH = 199998  # ids >= this are o200k_harmony special tokens


def parse_lengths(spec):
    """'all' | 'fast' | '128,1k,8192' -> sorted list of ints."""
    if spec is None or spec == "all":
        return list(PREFILL_LENGTHS)
    if spec == "fast":
        return list(FAST_GATE_LENGTHS)
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        out.append(int(tok[:-1]) * 1024 if tok.endswith("k") else int(tok))
    return sorted(set(out))


def default_n(length):
    # One warm timed call per length after the compile pass (person's instruction, 2026-10-14). The clean-tree
    # baseline with N=5/3/2 showed within-run spread <= 0.06% at every length, so N=1 is inside the noise floor.
    return 1


def repo_root():
    return Path(__file__).resolve().parents[5]


def git_info():
    root = repo_root()

    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
        except Exception as e:  # noqa: BLE001
            return f"<error: {e}>"

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": [l for l in run("status", "--porcelain").splitlines() if l],
    }


# Env vars recorded by env_snapshot(): every variable with one of these prefixes (model levers such as
# GPT_OSS_ROUTED_PREFILL*, runtime knobs such as TT_METAL_*), plus the named ones below. A lever that is not in
# the snapshot of a run was not set for that run.
ENV_PREFIXES = ("GPT_OSS_", "TT_", "PREFILL_")
ENV_KEYS = ("HF_MODEL", "ARCH_NAME")
PROMPT_MODES = ("random", "text")


def prompt_mode():
    """PREFILL_PROMPT: 'random' (default; the iteration-0 baseline's fixed-seed uniform ids) or 'text' (the
    frozen real-text stream the accuracy gate uses). Routed-MoE timing depends on the token -> expert
    distribution, so MoE keeps must be measured with 'text'."""
    mode = os.getenv("PREFILL_PROMPT", "random").strip().lower()
    assert mode in PROMPT_MODES, f"PREFILL_PROMPT={mode!r}; expected one of {PROMPT_MODES}"
    return mode


def env_snapshot():
    snap = {k: v for k, v in os.environ.items() if k.startswith(ENV_PREFIXES) or k in ENV_KEYS}
    for k in ENV_KEYS:
        snap.setdefault(k, None)
    snap = dict(sorted(snap.items()))
    snap["_prompt_mode"] = prompt_mode()
    return snap


def random_prompt_tokens(length, seed=1234):
    """Fixed-seed random prompt of exactly `length` non-special tokens (TTFT harness input)."""
    g = torch.Generator().manual_seed(seed + length)
    return torch.randint(0, VOCAB_RANDOM_HIGH, (1, length), generator=g, dtype=torch.long)


def text_prompt_tokens(length):
    """First `length` tokens of the frozen real-text stream (GOLDENS_DIR/prompt_tokens.pt), shape (1, length).

    The frozen stream holds exactly max(PREFILL_LENGTHS) = 131072 tokens, so every harness length is a plain
    prefix (the same prompt the gate uses). A longer request repeats the stream end to end (deterministic);
    no current length needs that."""
    stream = load_or_build_prompt_tokens()
    reps = -(-length // stream.numel())
    return stream.repeat(reps)[:length].unsqueeze(0)


def prompt_tokens(length, mode=None):
    mode = mode or prompt_mode()
    return text_prompt_tokens(length) if mode == "text" else random_prompt_tokens(length)


def build_corpus_tokens(tokenizer, n_tokens=max(PREFILL_LENGTHS)):
    root = repo_root()
    files = sorted(
        p for p in root.glob(CORPUS_GLOB.split("/")[0] + "/**/*.md") if not any(x in str(p) for x in CORPUS_EXCLUDE)
    )
    ids = []
    used = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        ids.extend(tokenizer.encode(text + "\n\n", add_special_tokens=False))
        used.append(str(p.relative_to(root)))
        if len(ids) >= n_tokens:
            break
    assert len(ids) >= n_tokens, f"corpus too short: {len(ids)} < {n_tokens}"
    return torch.tensor(ids[:n_tokens], dtype=torch.long), used


def load_or_build_prompt_tokens(tokenizer=None, create=False):
    """Frozen real-text token stream; prompt for length L is the first L tokens."""
    path = GOLDENS_DIR / "prompt_tokens.pt"
    if path.exists():
        return torch.load(path)["tokens"]
    assert create, f"{path} missing; create it once with the clean tree (prefill_gate.py --save-goldens)"
    assert tokenizer is not None
    tokens, used = build_corpus_tokens(tokenizer)
    GOLDENS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"tokens": tokens, "files": used, "git": git_info()}, path)
    logger.info(f"Wrote frozen prompt tokens ({len(tokens)}) from {len(used)} files to {path}")
    return tokens


def build_prefill_generator(mesh_device, max_seq_len):
    """Build the 24-layer model + Generator on the mesh exactly as the demo's batch-1 path does."""
    from models.demos.gpt_oss.demo.text_demo import prepare_gpt_oss_generator_args
    from models.demos.gpt_oss.tests.test_factory import TestFactory
    from models.tt_transformers.tt.generator import Generator

    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    page_params = {"page_block_size": PAGE_BLOCK_SIZE, "page_max_num_blocks_per_dp": max_seq_len // PAGE_BLOCK_SIZE}
    torch.manual_seed(0)  # create_tt_page_table shuffles blocks with torch.randperm
    (model_args, model, page_table, tt_kv_cache, tokenizer, processor, _pac) = prepare_gpt_oss_generator_args(
        num_devices=mesh_device.get_num_devices(),
        data_parallel=1,
        mesh_device=mesh_device,
        global_batch_size=1,
        optimizations=None,
        max_seq_len=max_seq_len,
        page_params=page_params,
        paged_attention=True,
        mesh_config=setup["mesh_config"],
        state_dict=None,  # same as the demo's state_dict fixture with HF_MODEL set: warm cache skips the HF load
        users_row_sharded=False,
        long_context_mode=False,
    )
    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)
    return {
        "generator": generator,
        "model": model,
        "model_args": model_args,
        "page_table": page_table,
        "kv_cache": tt_kv_cache,
        "tokenizer": tokenizer,
    }


def run_prefill(ctx, tokens, enable_trace=False):
    """One timed prefill through Generator.prefill_forward_text; returns (seconds, last-token logits [vocab])."""
    length = tokens.shape[-1]
    t0 = time.perf_counter()
    logits = ctx["generator"].prefill_forward_text(
        tokens,
        page_table=ctx["page_table"],
        kv_cache=ctx["kv_cache"],
        prompt_lens=[length],
        enable_trace=enable_trace,
        warmup_prefill=False,
    )
    dt = time.perf_counter() - t0
    return dt, logits.reshape(-1).float()


def compare_logits(cur, ref, k=5):
    """Metrics of current last-token logits vs a reference vector (both [vocab])."""
    n = min(cur.numel(), ref.numel())
    a = cur[:n].double()
    b = ref[:n].double()
    a = a - a.mean()
    b = b - b.mean()
    pcc = float((a * b).sum() / (a.norm() * b.norm() + 1e-30))
    ref_top = torch.topk(ref[:n], k).indices.tolist()
    cur_top = torch.topk(cur[:n], k).indices.tolist()
    ref_sorted = torch.topk(ref[:n].float(), 2).values
    return {
        "pcc": pcc,
        "top1_match": ref_top[0] == cur_top[0],
        "ref_top1_in_cur_top5": ref_top[0] in cur_top,
        "top5_overlap": len(set(ref_top) & set(cur_top)) / k,
        "ref_top1": ref_top[0],
        "cur_top1": cur_top[0],
        "ref_top1_margin": float(ref_sorted[0] - ref_sorted[1]),
        "max_abs_diff": float((cur[:n].float() - ref[:n].float()).abs().max()),
    }
