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
# (the run records and goldens: PREFILL_RUNS_DIR, default under the repo's git-ignored generated/)
RUNS_DIR = Path(
    os.getenv("PREFILL_RUNS_DIR", str(Path(__file__).resolve().parents[5] / "generated" / "gpt_oss_prefill" / "runs"))
)
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


def trace_mode():
    """PREFILL_TRACE=1: traced prefill, called exactly as the demo calls it (enable_trace=True,
    warmup_prefill=False). Default 0: untraced, the frozen headline regime."""
    value = os.getenv("PREFILL_TRACE", "0").strip()
    assert value in ("0", "1"), f"PREFILL_TRACE={value!r}; expected 0 or 1"
    return value == "1"


def padded_prefill_len(model_args, length):
    """The bucket the generator pads a length-`length` prompt to (the model's hook, else the shared rule)."""
    from models.tt_transformers.tt.common import get_padded_prefill_len

    if isinstance(model_args, (list, tuple)):  # prepare_gpt_oss_generator_args returns one args per data-parallel group
        model_args = model_args[0]
    pad = getattr(model_args, "get_padded_prefill_len", None)
    return pad(length) if callable(pad) else get_padded_prefill_len(length)


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


# ---- batch-32 (prefill_ttft_b32.py / prefill_gate_b32.py) -----------------------------------------------------------
B32_BATCH = 32
# User u's text prompt starts at token u * B32_PROMPT_STRIDE of the frozen stream (131072 / 32 = 4096), wrapping
# around the stream end. For L <= 4096 the 32 prompts are disjoint; for longer L they are distinct shifts. User 0 is
# the batch-1 text prompt (first L tokens).
B32_PROMPT_STRIDE = 4096
# KV cache: bf8 (attention/kv_cache.py init_kv_cache default), shape per device [blocks, 8 kv heads / 4 = 2, 64, 64].
KV_BYTES_PER_BLOCK_PER_TENSOR = 2 * 64 * 64 // 1024 * 1088  # 8 bf8_b tiles of 1088 B
N_LAYERS = 24


def kv_bytes_per_device(num_blocks):
    """K+V bytes per device for `num_blocks` 64-token pages in all 24 layers (bf8_b, 2 local kv heads)."""
    return num_blocks * N_LAYERS * 2 * KV_BYTES_PER_BLOCK_PER_TENSOR


def b32_prompt_tokens(length, mode, batch=B32_BATCH, stride=B32_PROMPT_STRIDE):
    """[batch, length] prompts, one distinct prompt per user.

    text: user u = tokens [u * stride, u * stride + length) of the frozen stream, wrapping around its end.
    random: user u = random_prompt_tokens(length, seed=1234 + 7919 * u)."""
    if mode == "text":
        stream = load_or_build_prompt_tokens()
        need = stride * (batch - 1) + length
        ext = stream.repeat(-(-need // stream.numel()))
        out = torch.stack([ext[u * stride : u * stride + length] for u in range(batch)])
    else:
        out = torch.cat([random_prompt_tokens(length, seed=1234 + 7919 * u) for u in range(batch)])
    assert len({tuple(r[:64].tolist()) + tuple(r[-64:].tolist()) for r in out}) == batch, "prompts not distinct"
    return out


def dram_view(mesh_device):
    """DRAM allocator view (per bank) as a dict; ttnn.get_memory_view reports the same numbers for every device."""
    import ttnn

    v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    out = {k: getattr(v, k) for k in dir(v) if not k.startswith("_") and isinstance(getattr(v, k), (int, float))}
    if "num_banks" in out:
        for k in ("total_bytes_per_bank", "total_bytes_allocated_per_bank", "total_bytes_free_per_bank"):
            if k in out:
                out[k.replace("_per_bank", "_per_device_gb")] = out[k] * out["num_banks"] / 1e9
    return out


class _no_kv_file_cache:
    """Skip the on-disk tensor cache for the KV cache tensors only (k_cache_* / v_cache_*).

    init_kv_cache builds each layer's K and V with ttnn.as_tensor(torch.zeros(shape), cache_file_name=...), which
    writes one file per layer, tensor and shape into the model's tensor cache (32 users x 64K = 285 MB per file, 48
    files, 13.7 GB; the host disk has ~49 GB free). The device tensors are identical (zeros, bf8_b, TILE, DRAM,
    replicated); only the file write/read is skipped. Weights keep using the cache."""

    def __enter__(self):
        from models.demos.gpt_oss.tt.attention import kv_cache as kvmod

        self._mod, self._orig = kvmod, kvmod.get_cache_file_name

        def patched(path, name, _orig=self._orig):
            return None if name.startswith(("k_cache_", "v_cache_")) else _orig(path, name)

        kvmod.get_cache_file_name = patched
        return self

    def __exit__(self, *exc):
        self._mod.get_cache_file_name = self._orig
        return False


def build_prefill_generator_b32(mesh_device, kv_len_per_user, batch=B32_BATCH, max_seq_len=max(PREFILL_LENGTHS)):
    """Build the 24-layer model + Generator for global batch `batch` on the mesh the way the demo's non-row-sharded
    path does (prepare_gpt_oss_generator_args with global_batch_size=batch, max_batch_size=batch,
    users_row_sharded=False, paged attention, block 64), with a paged KV cache of batch x kv_len_per_user tokens
    (page table [batch, kv_len_per_user / 64], blocks shuffled by create_tt_page_table under torch.manual_seed(0)).
    max_seq_len (rope / context limit) stays 128K as in the batch-1 harness."""
    from models.demos.gpt_oss.demo.text_demo import prepare_gpt_oss_generator_args
    from models.demos.gpt_oss.tests.test_factory import TestFactory
    from models.tt_transformers.tt.generator import Generator

    assert kv_len_per_user % PAGE_BLOCK_SIZE == 0
    setup = TestFactory.setup_test(mesh_device, use_real_weights=False)
    num_blocks = batch * kv_len_per_user // PAGE_BLOCK_SIZE
    page_params = {"page_block_size": PAGE_BLOCK_SIZE, "page_max_num_blocks_per_dp": num_blocks}
    torch.manual_seed(0)  # create_tt_page_table shuffles blocks with torch.randperm
    with _no_kv_file_cache():
        (model_args, model, page_table, tt_kv_cache, tokenizer, processor, _pac) = prepare_gpt_oss_generator_args(
            num_devices=mesh_device.get_num_devices(),
            data_parallel=1,
            mesh_device=mesh_device,
            global_batch_size=batch,
            optimizations=None,
            max_seq_len=max_seq_len,
            page_params=page_params,
            paged_attention=True,
            mesh_config=setup["mesh_config"],
            state_dict=None,
            users_row_sharded=False,
            long_context_mode=False,
        )
    assert tuple(page_table.shape) == (batch, kv_len_per_user // PAGE_BLOCK_SIZE), page_table.shape
    assert model_args[0].max_batch_size == batch, model_args[0].max_batch_size
    generator = Generator(model, model_args, mesh_device, processor=processor, tokenizer=tokenizer)
    return {
        "generator": generator,
        "model": model,
        "model_args": model_args,
        "page_table": page_table,
        "kv_cache": tt_kv_cache,
        "tokenizer": tokenizer,
        "batch": batch,
        "kv_len_per_user": kv_len_per_user,
        "kv_num_blocks": num_blocks,
        "kv_dtype": str(tt_kv_cache[0][0][0].dtype),
        "kv_shape_per_device": list(tt_kv_cache[0][0][0].shape),
        "kv_bytes_per_device": kv_bytes_per_device(num_blocks),
    }


def run_prefill_batch(ctx, tokens, page_table=None):
    """One Generator.prefill_forward_text call for all users in `tokens` [B, L] (untraced, logits to host).
    Returns (seconds, logits [B, vocab] float)."""
    B, L = tokens.shape
    t0 = time.perf_counter()
    logits = ctx["generator"].prefill_forward_text(
        tokens,
        page_table=ctx["page_table"] if page_table is None else page_table,
        kv_cache=ctx["kv_cache"],
        prompt_lens=[L] * B,
        enable_trace=False,
        warmup_prefill=False,
    )
    dt = time.perf_counter() - t0
    return dt, logits.reshape(B, -1).float()


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
