# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decode-step op profile for the roofline split (matmul / CCL / other / gaps).

Builds the generator, prefills a short prompt, warms the traced decode path,
then wraps ``N_STEPS`` traced decode steps in Tracy ``start`` / ``stop``
signposts so ``ops_perf_results_*.csv`` carries exactly those steps. The
companion ``decode_roofline_report.py`` splits the device time per step into
matmuls, collectives, everything else, and op-to-op gaps, and reports the
step against the DRAM-bandwidth floor.

    export TT_METAL_DEVICE_PROFILER=1
    HF_MODEL=google/gemma-4-31B-it MESH_DEVICE=P150x4 TT_VISIBLE_DEVICES=0,1,2,3 \\
    python -m tracy -p -r -v -m pytest \\
        models/demos/gemma4/tests/unit/test_decode_roofline.py -k "batch_1" -s
    python tools/tracy/process_ops_logs.py --date
    python models/demos/gemma4/tests/unit/decode_roofline_report.py <ops_perf_results.csv> --steps 10
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.demo.text_demo import (
    _batch_page_params,
    _create_tt_page_table,
    _install_hybrid_page_tables,
    _resolve_bounded_sliding,
)
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric, skip_if_config_only_checkpoint
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.tt_transformers.tt.common import PagedAttentionConfig

# Prefill (context) length before the measured decode steps. ``GEMMA4_ROOFLINE_PREFILL_LEN`` overrides it so the
# step can be measured at served context sizes (128 / 2048 / 8192 / 32768); the page pool, the per-layer tables and
# ``max_seq_len`` (``GEMMA4_ROOFLINE_MAX_SEQ_LEN``, default the power of two above prompt + steps, at least 4096)
# follow it.
PREFILL_LEN = int(os.environ.get("GEMMA4_ROOFLINE_PREFILL_LEN", "128"))
WARM_STEPS = 4
N_STEPS = int(os.environ.get("GEMMA4_ROOFLINE_STEPS", "3"))


def _roofline_max_seq_len() -> int:
    override = int(os.environ.get("GEMMA4_ROOFLINE_MAX_SEQ_LEN", "0"))
    if override > 0:
        return override
    needed = PREFILL_LEN + WARM_STEPS + N_STEPS + 2
    return max(4096, 1 << (needed - 1).bit_length())


# Eager by default: the device profiler records each op of an eager step with its
# host op record (name, type), which the split needs. Traced replays are not
# attributed per op by the Tracy post-processing on this build, so the measured
# step wall time (TPOT from the benchmarks) is used for the gap estimate instead.
TRACED = os.environ.get("GEMMA4_ROOFLINE_TRACE", "0") == "1"
# GEMMA4_ROOFLINE_DEVICE_ARGMAX=1: the next token comes from the model's on-device sampling module (the serving
# path: greedy SamplingParams, force-argmax over the TP-sharded logits, multi-core) and only the ``[batch]`` ids are
# read back, so the timed loop measures the device step. The default path reads the full logits to the host every
# step ([batch, 262144] bf16 = 16.8 MB at batch 32 inside the timed loop), which bounds the batch-32 wall time and
# hides the device step; it is kept as the baseline of earlier runs. (A plain ``ttnn.argmax`` over the 262144-wide
# replicated logits took 140 ms per step on a P150: not usable for timing.)
DEVICE_ARGMAX = os.environ.get("GEMMA4_ROOFLINE_DEVICE_ARGMAX", "0") == "1"


def _device_ops_in_order(captured):
    """Every ``function_start`` of a graph capture in execution order with its
    nesting depth and the top-level ttnn op that issued it. The report picks the
    device-launching level from this (names and counts differ per build), so
    nothing is filtered here."""
    ops = []
    stack = []  # names
    for n in captured:
        if n["node_type"] == "function_start":
            name = n["params"].get("name", "?")
            ops.append({"name": name, "parent": stack[0] if stack else None, "depth": len(stack)})
            stack.append(name)
        elif n["node_type"] == "function_end" and stack:
            stack.pop()
    return ops


def _signpost(name):
    try:
        from tracy import signpost
    except ImportError:  # not under python -m tracy: still a valid timing run
        return
    signpost(name)


@pytest.fixture
def model_path():
    return os.environ.get("HF_MODEL", "google/gemma-4-31B-it")


# The default line shapes plus the (2,4) block: an 8-chip engine opened as the 2-D mesh (plain TP over the columns,
# EP over the rows), the orientation of one DP engine carved from the Galaxy (test_geglu_routed_expert selects
# the same block).
@parametrize_mesh_with_fabric(
    mesh_shapes=[(1, 1), (1, 2), (1, 4), (1, 8), (2, 4), (1, 32)],
    device_params_extra={"trace_region_size": 512_000_000},
)
@pytest.mark.parametrize("batch_size", [1, 32], ids=["batch_1", "batch_32"])
def test_decode_roofline(mesh_device, model_path, batch_size):
    skip_if_config_only_checkpoint()
    max_seq_len = _roofline_max_seq_len()
    page_params = _batch_page_params(batch_size, PREFILL_LEN, WARM_STEPS + N_STEPS + 2)
    paged_attention_config = PagedAttentionConfig(
        block_size=int(page_params["page_block_size"]),
        max_num_blocks=int(page_params["page_max_num_blocks"]),
    )
    page_table = _create_tt_page_table(batch_size, paged_attention_config)
    bounded_sliding = _resolve_bounded_sliding(max_seq_len, mesh_device, model_path)
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=bounded_sliding,
    )
    if bounded_sliding:
        _install_hybrid_page_tables(
            generator.model[0],
            generator.model_args[0],
            batch_size=batch_size,
            block_size=int(page_params["page_block_size"]),
            max_seq_len=max_seq_len,
        )

    # Natural repeated text long enough for any PREFILL_LEN (~14 tokens per repetition).
    prompt = tokenizer.encode(
        "It was the best of times, it was the worst of times, " * max(12, PREFILL_LEN // 8), add_special_tokens=True
    )
    assert len(prompt) >= PREFILL_LEN, f"prompt has {len(prompt)} tokens < PREFILL_LEN={PREFILL_LEN}"
    tokens = torch.tensor(prompt[:PREFILL_LEN]).unsqueeze(0).repeat(batch_size, 1)
    if os.environ.get("GEMMA4_ROOFLINE_DISTINCT_ROWS", "0") == "1" and batch_size > 1:
        # Distinct random prompts per user: with the same prompt on every row all users route to the same
        # top-k, so a batch-32 step reads ~8 experts per layer instead of the ~100 a served batch touches
        # (the MoE weight traffic is then 10x under-represented). Seeded for run-to-run comparability.
        gen = torch.Generator().manual_seed(1234)
        tokens = torch.randint(1000, 200000, (batch_size, PREFILL_LEN), generator=gen)
        tokens[:, 0] = prompt[0]  # keep the BOS token
    t_pf = time.perf_counter()
    out = generator.prefill_forward_text(
        tokens,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=[PREFILL_LEN] * batch_size,
        warmup_prefill=False,
        enable_trace=False,
    )
    tok = torch.argmax(out.reshape(batch_size, -1).float(), dim=-1, keepdim=True)
    # The harness's own (untraced, first-call) prefill wall: the TTFT of this run, reported next to its decode.
    logger.info(
        f"TTFT_WALL batch={batch_size} prefill_len={PREFILL_LEN} eager first call: "
        f"{(time.perf_counter() - t_pf) * 1e3:.1f} ms (all {batch_size} users)"
    )
    current_pos = torch.full((batch_size,), PREFILL_LEN, dtype=torch.int32)

    greedy = None
    if DEVICE_ARGMAX:
        from models.common.sampling.sampling_params import SamplingParams

        greedy = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    first_call = [True]

    def step(t):
        o = generator.decode_forward(
            t,
            current_pos,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            enable_trace=TRACED,
            sampling_params=greedy,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=bool(DEVICE_ARGMAX and first_call[0]),
            reset_sampling_state=bool(DEVICE_ARGMAX and first_call[0]),
        )
        first_call[0] = False
        if DEVICE_ARGMAX:
            # on-device greedy sampling: the host sees the [batch] token ids only
            toks = o[0] if isinstance(o, (list, tuple)) else o
            return toks.reshape(-1)[:batch_size].to(torch.int64).unsqueeze(-1)
        logits = o[0] if isinstance(o, tuple) else o
        return torch.argmax(logits.reshape(batch_size, -1).float(), dim=-1, keepdim=True)

    for _ in range(WARM_STEPS):  # trace capture + warm replays
        tok = step(tok)
        current_pos += 1
    ttnn.synchronize_device(mesh_device)

    # Op names in execution order for ONE step (graph capture, normal run mode):
    # the device profiler rows carry no names on this build, and the Tracy
    # post-processing asserts on traced/eager mixes, so the report joins the
    # device rows of the signposted steps with this sequence by order.
    ops_json = os.path.join(
        os.environ.get("TT_METAL_HOME", "."), "generated", "profiler", f"roofline_ops_{batch_size}.json"
    )
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    tok = step(tok)
    current_pos += 1
    ttnn.synchronize_device(mesh_device)
    captured = ttnn.graph.end_graph_capture()
    ops = _device_ops_in_order(captured)
    os.makedirs(os.path.dirname(ops_json), exist_ok=True)
    with open(ops_json, "w") as f:
        json.dump({"batch_size": batch_size, "ops": ops}, f)
    logger.info(f"captured {len(ops)} device ops for one decode step -> {ops_json}")

    def _flush():
        # The device-side profiler buffers hold a bounded number of op records per
        # core; without a read-back between steps the earlier ops are lost and the
        # per-step split comes out short (seen: 1990 rows for 10 x 3618 ops).
        try:
            ttnn.ReadDeviceProfiler(mesh_device)
        except Exception:
            pass

    profiler_on = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    _flush()
    row0_tokens = []
    _signpost("start")
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        tok = step(tok)
        row0_tokens.append(int(tok[0, 0]))
        current_pos += 1
        ttnn.synchronize_device(mesh_device)
        if profiler_on:
            _flush()
    wall_ms = (time.perf_counter() - t0) / N_STEPS * 1e3
    _signpost("stop")
    # Host wall time per step (synchronized): the serving-relevant number when traced and the profiler is off.
    logger.info(
        f"WALL batch={batch_size} {'traced' if TRACED else 'eager'} ctx={PREFILL_LEN} max_seq_len={max_seq_len} "
        f"bounded={bounded_sliding} argmax={'device' if DEVICE_ARGMAX else 'host'}: {wall_ms:.2f} ms/step = "
        f"{1e3 / wall_ms:.2f} tok/s/user, {batch_size * 1e3 / wall_ms:.1f} tok/s aggregate ({N_STEPS} steps)"
    )
    # The greedy continuation of the repeated prompt (row 0): a coherent repetition says the long-context KV and the
    # traced replay are sane; garbage here would void the timing.
    logger.info(f"ROW0_TOKENS ctx={PREFILL_LEN}: {tokenizer.decode(row0_tokens)!r}")
    try:
        ttnn.ReadDeviceProfiler(mesh_device)
    except Exception:  # profiler off: nothing to flush
        pass
    logger.info(
        f"{os.path.basename(model_path)} batch={batch_size}: {N_STEPS} {'traced' if TRACED else 'eager'} decode steps between signposts"
    )
