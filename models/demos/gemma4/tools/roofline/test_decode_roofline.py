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
        models/demos/gemma4/tools/roofline/test_decode_roofline.py -k "batch_1" -s
    python tools/tracy/process_ops_logs.py --date
    python models/demos/gemma4/tools/roofline/decode_roofline_report.py <ops_perf_results.csv> --steps 10
"""

import csv
import json
import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.demo.text_demo import _resolve_bounded_sliding
from models.demos.gemma4.tests.test_factory import (
    GEMMA4_TEST_L1_SMALL_SIZE,
    lanes_fabric_config,
    parametrize_mesh_with_fabric,
    skip_if_config_only_checkpoint,
)
from models.demos.gemma4.tools.roofline import decode_roofline_report
from models.demos.gemma4.tt.attention.kv_cache_hybrid import build_hybrid_page_tables
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


def _roofline_pool(batch_size, block_size, max_seq_len, prefill_len, new_tokens):
    """The served engine's pool and page layout for ``batch_size`` users of ``prefill_len`` tokens plus
    ``new_tokens`` decode steps: ``(paged_attention_config, page_table [batch, cols], prompt_blocks)``.

    vLLM allocates every request the blocks of its prompt plus its output reservation from ONE pool (block 0 is
    the null block) and sizes that pool by the users it holds, not by ``batch x max_seq_len``; the decode table
    is ``max_seq_len / block_size`` columns wide with the unused columns 0 (the fill stops at valid_seq_len, so
    they are never written). An identity ``batch x cols`` layout instead doubled the pool past the served one at
    32 users x 64K (max_seq_len rounds up to 131072) and the weights no longer fit (DRAM OOM at layer 28).
    """
    cols = math.ceil(max_seq_len / block_size)  # the decode table width the generator installs
    prompt_blocks = math.ceil(prefill_len / block_size)  # what the plugin's prefill table carries per request
    blocks_per_user = math.ceil((prefill_len + new_tokens) / block_size)  # prompt + output reservation
    # paged_update_cache needs the pool larger than the table width; plus a spare tile row of blocks.
    max_num_blocks = max(1 + batch_size * blocks_per_user, cols) + 32
    page_table = torch.zeros(batch_size, cols, dtype=torch.int32)
    for u in range(batch_size):
        page_table[u, :blocks_per_user] = torch.arange(
            1 + u * blocks_per_user, 1 + (u + 1) * blocks_per_user, dtype=torch.int32
        )
    return PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks), page_table, prompt_blocks


def _roofline_per_layer_tables(model_args, page_table, prompt_blocks, *, block_size, max_seq_len):
    """Per-layer tables under bounded sliding: ``(for the prefill, for the decode)``.

    Sliding layers keep their window-sized rings (``build_hybrid_page_tables``: user u owns ring blocks
    ``[u*W, (u+1)*W)`` of the layer's own pool); every full-attention layer shares ``page_table`` (one host
    table per kv-cache group). For the prefill the full-attention rows carry what the shared per-user loop hands
    the model as the user's legacy slice: the prompt's blocks, then -1 up to the padded bucket (256000 tokens ->
    4096 blocks), the table every layer of the stock non-hybrid path runs on. The sequential-prefill guard compares
    that slice against the per-layer row over the narrower of the two widths, so a row whose tail is 0 or the decode
    blocks fails it (devq 0565), and the chunked SDPA reads the layer's own table and needs it to span the bucket
    (``kv_length >= chunk end``), so a prompt-width row fails the chunk past 256000 (devq 1030). Serving matches the
    guard on the sliding group's table, which the plugin passes as the legacy table; the demo path slices the
    full-attention chunk fill from the legacy table, so here it has to be the full-attention one.
    """
    batch_size = int(page_table.shape[0])
    n_layers = model_args.num_hidden_layers
    sliding = [model_args.layer_types[i] == "sliding_attention" for i in range(n_layers)]
    rings = build_hybrid_page_tables(
        n_layers,
        sliding,
        num_users=batch_size,
        block_size=block_size,
        max_seq_len=max_seq_len,
        sliding_window=model_args.sliding_window,
    )
    ring = next((rings[i] for i in range(n_layers) if sliding[i]), None)
    prefill_rows = page_table.clone()
    prefill_rows[:, prompt_blocks:] = -1
    per_layer_prefill = [ring if sliding[i] else prefill_rows for i in range(n_layers)]
    per_layer_decode = [ring if sliding[i] else page_table for i in range(n_layers)]
    return per_layer_prefill, per_layer_decode


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
# Traced steps under the device profiler follow the recipe of the traced prefill tracy session
# (tracy_prefill_common.py): no graph capture of a step (the capture of a replay names nothing) and no profiler
# read-back between the measured steps (one before the start signpost, one after stop), so the tracy report
# attributes every replayed program. GEMMA4_ROOFLINE_SKIP_CAPTURE / GEMMA4_ROOFLINE_FLUSH_PER_STEP override.
_PROFILED_TRACE = TRACED and os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
SKIP_CAPTURE = os.environ.get("GEMMA4_ROOFLINE_SKIP_CAPTURE", "1" if _PROFILED_TRACE else "0") == "1"
FLUSH_PER_STEP = os.environ.get("GEMMA4_ROOFLINE_FLUSH_PER_STEP", "0" if _PROFILED_TRACE else "1") == "1"


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
# batch_8: the served 256K floor of the 26B QB2 column (8 users at 256K fit its DRAM; 32 x 256K does not), so the
# long-context decode step can be profiled at the users the pool really holds (GEMMA4_ROOFLINE_PREFILL_LEN=256000).
# The pool and the page tables follow the served layout (_roofline_pool / _roofline_per_layer_tables).
@pytest.mark.parametrize("batch_size", [1, 8, 32], ids=["batch_1", "batch_8", "batch_32"])
def test_decode_roofline(mesh_device, model_path, batch_size):
    skip_if_config_only_checkpoint()
    max_seq_len = _roofline_max_seq_len()
    block_size = int(os.environ.get("GEMMA4_PAGE_BLOCK_SIZE", "64"))
    paged_attention_config, page_table, prompt_blocks = _roofline_pool(
        batch_size, block_size, max_seq_len, PREFILL_LEN, WARM_STEPS + N_STEPS + 2
    )
    logger.info(
        f"POOL batch={batch_size} block={block_size} max_num_blocks={paged_attention_config.max_num_blocks} "
        f"({paged_attention_config.max_num_blocks * block_size} tokens) prompt_blocks={prompt_blocks} "
        f"decode_cols={page_table.shape[1]} (served layout: per-user blocks from one pool, null block 0)"
    )
    bounded_sliding = _resolve_bounded_sliding(max_seq_len, mesh_device, model_path)
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=bounded_sliding,
    )
    model = generator.model[0]
    per_layer_decode = None
    if bounded_sliding:
        per_layer_prefill, per_layer_decode = _roofline_per_layer_tables(
            generator.model_args[0], page_table, prompt_blocks, block_size=block_size, max_seq_len=max_seq_len
        )
        # The persistent device buffers at the decode width first (a wider install later would reallocate
        # them), then the prompt-width tables the prefill runs on.
        model._active_page_tables_per_layer = per_layer_decode
        generator._install_per_layer_page_tables(model, per_layer_decode, writer="prefill")
        model._active_page_tables_per_layer = per_layer_prefill
        logger.info(
            f"Bounded sliding: {len(per_layer_prefill)} per-layer tables, full-attention rows = the prompt's "
            f"{prompt_blocks} blocks then -1 for the prefill, the decode blocks then 0 for the decode "
            f"({page_table.shape[1]} columns)"
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
    if per_layer_decode is not None:
        # The sequential loop left its 1-row stash: put the full-batch tables back, then install the decode-width
        # ones (the decode bucket re-uploads its per-layer tables after a prefill, as the serving path does).
        generator._clear_sequential_batch_page_tables()
        model._active_page_tables_per_layer = per_layer_decode
        generator._install_per_layer_page_tables(model, per_layer_decode, writer="prefill")
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
    if not SKIP_CAPTURE:
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
        if profiler_on and FLUSH_PER_STEP:
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


_roofline_body = test_decode_roofline  # the block-ring and lane variants run the same body under their own mesh


# ===== merged from tests/unit/test_decode_roofline_block_ring.py: Traced decode step of one 8-chip engine on the (2,4) block with the mesh opened on the RING fabric.
_FABRIC = (
    getattr(ttnn.FabricConfig, os.environ["ROOFLINE_FABRIC"])
    if os.environ.get("ROOFLINE_FABRIC")
    else lanes_fabric_config()
)


MESHES = [
    pytest.param(
        (2, 4),
        {
            "fabric_config": _FABRIC,
            "trace_region_size": 512_000_000,
            "l1_small_size": int(os.environ.get("ROOFLINE_L1_SMALL", str(GEMMA4_TEST_L1_SMALL_SIZE))),
        },
        id="2x4_ring",
    ),
]


@pytest.mark.parametrize("mesh_device, device_params", MESHES, indirect=True)
@pytest.mark.parametrize("batch_size", [1, 32], ids=["batch_1", "batch_32"])
def test_decode_roofline_block_ring(mesh_device, model_path, batch_size):
    _roofline_body(mesh_device, model_path, batch_size)


# ===== merged from tests/unit/test_decode_roofline_lanes.py: Decode-step op profile and traced step time of the Galaxy lane engine (8x4 mesh: fractured dense weights, four
SLOTS_PER_LANE = 32


BLOCKS_PER_USER = 4  # 256 tokens at block 64: the 128-token prompt plus the warm and profiled steps


@parametrize_mesh_with_fabric(
    mesh_shapes=[(2, 4), (8, 4)],
    device_params_extra={
        "fabric_config": lanes_fabric_config(),
        "l1_small_size": 65536,
        "trace_region_size": 512_000_000,
    },
)
@pytest.mark.parametrize("users_per_lane", [1, 32], ids=["upl_1", "upl_32"])
def test_decode_roofline_lanes(mesh_device, users_per_lane, monkeypatch):
    skip_if_config_only_checkpoint()
    monkeypatch.setenv("GEMMA4_GALAXY_FRACTURE", "1")
    monkeypatch.setenv("GEMMA4_GALAXY_LANES", "1")
    monkeypatch.setenv("GEMMA4_MOE_DECODE_KERNEL", os.environ.get("GEMMA4_MOE_DECODE_KERNEL", "1"))
    shape = tuple(mesh_device.shape)
    if shape == (2, 4):
        # 4 lanes x 32 rows over 2 mesh rows: the kernel frame must hold 128 rows (64 tokens per row-chip)
        monkeypatch.setenv("GEMMA4_MOE_DECODE_FRAME", os.environ.get("GEMMA4_MOE_DECODE_FRAME", "128"))

    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig

    model_path = os.environ.get("HF_MODEL", "google/gemma-4-26B-A4B-it")
    # Per-column pool: block 0 is the lane scratch, then 32 slots per lane (see test_lanes_e2e.py).
    pool_blocks = 1 + SLOTS_PER_LANE * BLOCKS_PER_USER
    paged_cfg = PagedAttentionConfig(block_size=64, max_num_blocks=pool_blocks)
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device,
        model_path,
        max_batch_size=SLOTS_PER_LANE,
        max_seq_len=4096,
        paged_attention_config=paged_cfg,
    )
    mesh_cfg = generator.model[0].mesh_config
    assert getattr(mesh_cfg, "lane_sharded", False), "lanes gate did not engage"
    # The caches hold the heads the attention writes (init_kv_cache from mesh_config.tp): the paged ops then address
    # them at the declared block size and the per-user page rows below hold BLOCKS_PER_USER x 64 tokens.
    model_kv = tt_kv_cache if hasattr(tt_kv_cache[0][0], "shape") else tt_kv_cache[0]  # [model][layer][k, v] nesting
    eff_bs = generator._effective_paged_block_size(model_kv)
    assert eff_bs == 64, f"effective paged block size {eff_bs} != declared 64 (cache heads vs the attention's view)"
    lanes = mesh_cfg.lanes
    B_g = lanes * SLOTS_PER_LANE
    rows = [lane * SLOTS_PER_LANE + s for lane in range(lanes) for s in range(users_per_lane)]
    page_rows = {
        row: torch.arange(
            1 + (row % SLOTS_PER_LANE) * BLOCKS_PER_USER,
            1 + (row % SLOTS_PER_LANE + 1) * BLOCKS_PER_USER,
            dtype=torch.int32,
        )
        for row in rows
    }

    prompt = tokenizer.encode("It was the best of times, it was the worst of times, " * 12, add_special_tokens=True)
    tok1 = torch.tensor(prompt[:PREFILL_LEN], dtype=torch.long).unsqueeze(0)
    cur_tok, positions = {}, {}
    prefill_ms = []
    for row in rows:
        t0 = time.perf_counter()
        out = generator.prefill_forward_text(
            tok1,
            page_table=page_rows[row].unsqueeze(0),
            kv_cache=tt_kv_cache,
            prompt_lens=torch.tensor([PREFILL_LEN]),
            empty_slots=[row],
            enable_trace=False,
            sampling_params=None,
            warmup_prefill=False,
        )
        lg = out[0] if isinstance(out, (list, tuple)) else out
        cur_tok[row] = int(torch.argmax(lg.reshape(-1)).item())  # the host read-back ends the call, as serving's does
        prefill_ms.append((time.perf_counter() - t0) * 1e3)
        positions[row] = PREFILL_LEN
    logger.info(f"prefilled {len(rows)} users ({users_per_lane} per lane x {lanes} lanes) of the {B_g}-row batch")
    # Same-run TTFT component: the eager single-user 128-token prefill call (first call = cold JIT), host logits
    # read back; the served c1 TTFT adds the scheduler and tokenizer around this.
    warm = sorted(prefill_ms[1:]) or prefill_ms
    logger.info(
        f"PREFILL ctx={PREFILL_LEN} eager single-user lanes={lanes} mesh={shape[0]}x{shape[1]}: first {prefill_ms[0]:.1f} ms, "
        f"warm median {warm[len(warm) // 2]:.1f} ms, warm min {warm[0]:.1f} ms over {len(prefill_ms)} calls"
    )
    # Distinct tokens per active row for the timed traced steps (the same token on every row routes every user to
    # the same experts; the served step touches ~100 experts per layer): seeded, so runs are comparable.
    rand_tokens = torch.Generator().manual_seed(1234)

    def _inputs():
        tokens_g = torch.zeros(B_g, dtype=torch.long)
        pos_g = torch.full((B_g,), -1, dtype=torch.long)
        pt_g = torch.zeros(B_g, BLOCKS_PER_USER, dtype=torch.int32)
        for row in rows:
            tokens_g[row] = cur_tok[row]
            pos_g[row] = positions[row]
            pt_g[row] = page_rows[row]
        return tokens_g, pos_g, pt_g

    def step(read_back=True, reload_page_table=True):
        tokens_g, pos_g, pt_g = _inputs()
        logits = generator.decode_forward(
            tokens_g.unsqueeze(-1),
            pos_g,
            page_table=pt_g,
            kv_cache=tt_kv_cache,
            enable_trace=TRACED,
            read_from_device=read_back,
            sampling_params=None,
            reload_inputs=True,
            reload_page_table=reload_page_table,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        if read_back:
            lg = logits[0] if isinstance(logits, (list, tuple)) else logits
            lg = lg.reshape(B_g, -1)
            for row in rows:
                cur_tok[row] = int(torch.argmax(lg[row]).item())
                positions[row] += 1
        else:
            for row in rows:
                cur_tok[row] = int(torch.randint(1000, 200000, (1,), generator=rand_tokens))
                positions[row] += 1
        return logits

    for i in range(WARM_STEPS):  # eager warm-up, or trace capture + warm replays
        step(read_back=True, reload_page_table=(i == 0 or not TRACED))
    ttnn.synchronize_device(mesh_device)

    if TRACED:
        # The served number: traced replays, device logits left on device (serving samples on device), the page
        # table unchanged between steps. Distinct tokens per row per step.
        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            step(read_back=False, reload_page_table=False)
            ttnn.synchronize_device(mesh_device)
        wall_ms = (time.perf_counter() - t0) / N_STEPS * 1e3
        users = len(rows)
        logger.info(
            f"WALL batch={users} traced lanes={lanes} upl={users_per_lane} mesh={shape[0]}x{shape[1]} "
            f"ctx={PREFILL_LEN}: {wall_ms:.2f} ms/step = {1e3 / wall_ms:.2f} tok/s/user, "
            f"{users * 1e3 / wall_ms:.1f} tok/s aggregate ({N_STEPS} steps)"
        )
        # Coherence of the replayed graph: continue row 0 greedily for a few steps from the prompt's last token.
        cur_tok[rows[0]] = int(prompt[PREFILL_LEN - 1])
        row0 = []
        for _ in range(6):
            step(read_back=True, reload_page_table=False)
            row0.append(cur_tok[rows[0]])
        logger.info(f"ROW0_TOKENS ctx={PREFILL_LEN}: {tokenizer.decode(row0)!r}")
        return

    # Op names in execution order for ONE step (see test_decode_roofline.py: the device rows carry no names).
    ops_json = os.path.join(
        os.environ.get("TT_METAL_HOME", "."), "generated", "profiler", f"roofline_ops_lanes_{users_per_lane}.json"
    )
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    step()
    ttnn.synchronize_device(mesh_device)
    captured = ttnn.graph.end_graph_capture()
    ops = _device_ops_in_order(captured)
    os.makedirs(os.path.dirname(ops_json), exist_ok=True)
    with open(ops_json, "w") as f:
        json.dump({"batch_size": B_g, "users_per_lane": users_per_lane, "ops": ops}, f)
    n_dev = sum(1 for o in ops if o["name"].endswith("DeviceOperation"))
    logger.info(f"captured {len(ops)} graph entries ({n_dev} device operations) for one lane decode step -> {ops_json}")

    def _flush():
        try:
            ttnn.ReadDeviceProfiler(mesh_device)
        except Exception:
            pass

    _flush()
    # Bound the device log: everything so far (load, 128 prefills, warm steps) is flushed and not needed; keep
    # only the two header lines so the profiled steps' rows are the file (the 8x4 mesh writes ~33 GB otherwise).
    dev_log = os.path.join(
        os.environ.get("TT_METAL_HOME", "."), "generated", "profiler", ".logs", "profile_log_device.csv"
    )
    if os.path.exists(dev_log):
        with open(dev_log) as f:
            header = [next(f), next(f)]
        with open(dev_log, "w") as f:
            f.writelines(header)
        logger.info(f"device log truncated to its header before the profiled steps: {dev_log}")
    _signpost("start")
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        step()
        ttnn.synchronize_device(mesh_device)
        _flush()
    wall_ms = (time.perf_counter() - t0) / N_STEPS * 1e3
    _signpost("stop")
    _flush()
    logger.info(
        f"WALL batch={len(rows)} eager lanes={lanes} upl={users_per_lane} mesh={shape[0]}x{shape[1]}: "
        f"{wall_ms:.2f} ms/step (profiler reads included; not a serving number)"
    )
    logger.info(
        f"{os.path.basename(model_path)} lanes={lanes} users_per_lane={users_per_lane}: {N_STEPS} eager decode steps between signposts"
    )


# ===== merged from tests/unit/test_decode_roofline_report.py: The roofline report reads only the signposted steps and sums the idle gaps as the CSV reports them.
COLS = ["OP CODE", "OP TYPE", "DEVICE ID", "DEVICE FW DURATION [ns]", "OP TO OP LATENCY [ns]"]


def _row(code, op_type="tt_dnn_device", device=0, dur_ns=0, gap_ns=""):
    return dict(zip(COLS, [code, op_type, device, dur_ns, gap_ns]))


def _write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def test_report_keeps_only_the_signposted_steps_and_sums_reported_gaps(tmp_path, capsys, monkeypatch):
    path = tmp_path / "ops_perf_results.csv"
    _write(
        path,
        [
            _row("Matmul", dur_ns=100_000_000, gap_ns=0),  # warmup: outside the signposts
            _row("start", op_type="signpost"),
            _row("Matmul", dur_ns=600_000, gap_ns=100_000),
            _row("AllGatherAsync", dur_ns=400_000, gap_ns=50_000),
            _row("Matmul", op_type="tt_dnn_device", device=1, dur_ns=5_000_000, gap_ns=0),  # another chip
            _row("stop", op_type="signpost"),
            _row("Matmul", dur_ns=100_000_000, gap_ns=0),  # after the steps
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        ["decode_roofline_report.py", str(path), "--steps", "1", "--params-b", "1", "--chips", "1"],
    )
    decode_roofline_report.main()
    out = capsys.readouterr().out
    assert "rows between signposts (device 0): 2" in out
    assert "device time/step: 1.00 ms  = matmul 0.60 + ccl 0.40 + other 0.00" in out
    assert "op-to-op gaps/step: 0.15 ms" in out
