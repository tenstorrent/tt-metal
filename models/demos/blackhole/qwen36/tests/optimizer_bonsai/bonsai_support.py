# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared setup for the Bonsai optimizer harness (test_bonsai_perf.py, test_bonsai_pcc.py, gen_reference.py).

The model is Ternary-Bonsai-2-27B converted to a dense Qwen3.8-27B checkpoint (HF_MODEL), run on the qwen36 path on
one P150 (mesh 1x1), batch 1, plain decode (no MTP head). The build follows demo/text_demo.py's traced single-device
path: Qwen36Model.from_pretrained, a paged KV cache with 64-token blocks and an identity page table, the chunk-outer
prefill trace (2048-token chunks), and the Generator decode trace primed by prime_decode_trace.

The prompt is exactly ISL = 4096 tokens, two full prefill chunks, so prefill is two replays of the chunk trace and
no masked tail bucket (and no masked-bucket warmup) runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import torch
from loguru import logger

ISL = 4096  # prompt tokens: two full 2048-token prefill chunks
GEN = 100  # gate positions: the prefill's last position + 99 teacher-forced decode steps
CHUNK = 2048
BLOCK_SIZE = 64
DECODE_CALLS = 129  # perf: 1 untimed call (compiles the argmax) + 128 timed decode steps
PREFILL_SAMPLES = 3  # perf: timed traced prefills after one untimed one
# KV budget: prompt + every decode position of the perf test or the gate, in 64-token blocks, a multiple of 8.
NUM_BLOCKS = ((-(-(ISL + max(DECODE_CALLS + 1, GEN)) // BLOCK_SIZE) + 7) // 8) * 8
MAX_SEQ_LEN = NUM_BLOCKS * BLOCK_SIZE
DEVICE_PARAMS = {"l1_small_size": 24576, "num_command_queues": 2}  # the demo's single-device params
TEXT_URL = "https://www.gutenberg.org/cache/epub/84/pg84.txt"

MODEL_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[6]
REFERENCE = REPO_ROOT / "generated" / "optimizer_reference" / f"bonsai2-27b-isl{ISL}.pt"
BASELINE_PATH = REPO_ROOT / "generated" / f"optimizer_accuracy_baseline_bonsai2-27b-isl{ISL}.json"


def load_reference() -> dict:
    assert REFERENCE.is_file(), f"missing {REFERENCE}; produce it with tests/optimizer_bonsai/gen_reference.py"
    ref = torch.load(REFERENCE)
    assert len(ref["prompt_ids"]) == ISL and len(ref["continuation_ids"]) == GEN
    return ref


def build_model(device, n_layers=None):
    """(model, page_table): the demo's single-device build, KV cache allocated, plain decode (no MTP).

    n_layers builds the first n checkpoint layers only (the profiler's depth); None builds all of them."""
    import ttnn
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    t0 = time.time()
    model = Qwen36Model.from_pretrained(
        device, max_batch_size=1, max_seq_len=MAX_SEQ_LEN, enable_mtp=False, n_layers=n_layers
    )
    print(f"BONSAI_MODEL_LOAD_S={time.time() - t0:.1f} layers={model.args.n_layers}", flush=True)
    kv_cache_shape = [NUM_BLOCKS, model.args.n_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)
    if os.environ.get("BONSAI_TERNARY_MLP") == "1":
        # One-token decode MLPs in Bonsai's own ternary weights, read from the GGUF (bonsai2/tt/ternary/mlp.py).
        # Packed on the host here; swapped in for the dense gate/up/down after the prefill (after_prefill).
        from models.demos.blackhole.bonsai2.tt.ternary.mlp import prepare_ternary_mlp

        prepare_ternary_mlp(
            model, os.environ.get("BONSAI_GGUF", "/local/ttuser/jerrywang/bonsai/gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf")
        )
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    return model, page_table


def after_prefill(model) -> None:
    """Between the prefill and the decode steps: swap in the ternary decode MLPs when BONSAI_TERNARY_MLP=1."""
    if os.environ.get("BONSAI_TERNARY_MLP") == "1":
        from models.demos.blackhole.bonsai2.tt.ternary.mlp import swap_in_ternary_mlp

        swap_in_ternary_mlp(model)


def prefill_traced(model, prompt_ids, page_table) -> torch.Tensor:
    """Next-token logits [vocab] (host, float) of the traced chunk-outer prefill, as the demo runs it."""
    import ttnn

    logits = model.prefill_traced_chunked(torch.tensor([prompt_ids]), page_table, actual_len=len(prompt_ids))
    return ttnn.to_torch(logits).reshape(-1).float()


def prefill_eager(model, prompt_ids, page_table) -> torch.Tensor:
    """The same prefill with each chunk-trace replay replaced by the eager forward the trace was captured from.

    For the profiler, which runs with trace off. prefill_traced_chunked stages each chunk's inputs into the
    persistent chunk buffers and then calls ttnn.execute_trace; here that call runs model._forward_prefill_chunk on
    those buffers instead (the forward record_prefill_trace_chunked captures), so the device runs the same programs
    in the same order. Needs prepare_prefill_trace_chunked first."""
    import ttnn

    assert model._chunked_prepared_key is not None, "call prepare_prefill_trace_chunked first"
    real_execute_trace = ttnn.execute_trace

    def run_chunk_eagerly(device, trace_id, cq_id=0, blocking=False):
        previous = model._chunked_trace_output
        if previous is not None:
            ttnn.deallocate(previous)
        model._chunked_trace_output = model._forward_prefill_chunk(
            model._chunk_token_buf,
            model._chunk_cos_buf,
            model._chunk_sin_buf,
            model._chunk_start_idx_tensor,
            model._chunk_full_page_table_buf,
            model._chunk_page_table_buf,
        )

    saved_trace_id = model._chunked_trace_id
    ttnn.execute_trace = run_chunk_eagerly
    model._chunked_trace_id = saved_trace_id if saved_trace_id is not None else "eager"
    try:
        logits = model.prefill_traced_chunked(torch.tensor([prompt_ids]), page_table, actual_len=len(prompt_ids))
    finally:
        ttnn.execute_trace = real_execute_trace
        model._chunked_trace_id = saved_trace_id
    return ttnn.to_torch(logits).reshape(-1).float()


def make_generator(model, device):
    from models.tt_transformers.tt.generator import Generator

    return Generator([model], [model.args], device)


def prime_decode(generator, model, token: int, pos: int, page_table) -> None:
    """Capture the decode trace at the post-prefill state without advancing the GDN state (the demo's step)."""
    from models.demos.blackhole.qwen36.tt.generator_interface import prime_decode_trace

    prime_decode_trace(generator, model, torch.tensor([[token]], dtype=torch.long), torch.tensor([pos]), page_table)


def decode_step(generator, token: int, pos: int, page_table, *, trace: bool, read: bool):
    """One batch-1 decode call: host logits [vocab] when read, else the device output."""
    out = generator.decode_forward(
        torch.tensor([[token]], dtype=torch.long),
        torch.tensor([pos]),
        page_table=page_table,
        kv_cache=None,
        enable_trace=trace,
        read_from_device=read,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )
    if read:
        return (out[0] if isinstance(out, tuple) else out).reshape(-1).float()
    return out


def device_argmax(tt_output, vocab_size: int) -> int:
    """Greedy token of a decode step whose logits stayed on the device: argmax on the device, one id read back."""
    import ttnn

    while isinstance(tt_output, (list, tuple)):  # [per data-parallel group] of (logits, log-probs)
        tt_output = tt_output[0]
    rank = len(tt_output.shape)
    logits = ttnn.slice(tt_output, [0] * rank, [1] * (rank - 1) + [vocab_size])  # row 0 (user 0), real vocab
    logits = ttnn.reshape(logits, (1, 1, 1, vocab_size))
    logits = ttnn.to_layout(ttnn.typecast(logits, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
    return int(ttnn.to_torch(ttnn.argmax(logits, dim=-1)).reshape(-1)[0])


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(MODEL_DIR), capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def tree_diff_hash() -> str:
    """Hash of the uncommitted diff, so candidates measured on the same HEAD can be told apart."""
    try:
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        return ""
    if not diff:
        return "clean"
    import hashlib

    return hashlib.sha1(diff).hexdigest()[:12]


def append_history(env_name: str, record: dict) -> None:
    """One JSON line per run in the file named by env_name (unset: no history)."""
    path = os.environ.get(env_name)
    if not path:
        return
    record = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tree": git_head(),
        "diff": tree_diff_hash(),
        **record,
    }
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError as error:
        logger.warning(f"history {path} not written: {error}")
