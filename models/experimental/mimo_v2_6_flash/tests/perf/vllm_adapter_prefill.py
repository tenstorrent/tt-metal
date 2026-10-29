# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Long-prompt prefill through the vLLM adapter (generator_vllm.MiMoV2ForCausalLM.prefill_forward) on TP=8 x DP=4,
without vLLM: the steps a busy lane gets under the plugin's lane-DP scheduler, one request per lane prefilling with
``--max-num-batched-tokens`` (``--budget``) tokens per step, every busy lane in each step (one host thread, as the
plugin's worker). Runs the plugin's warmup order (warmup_model_prefill, the eager then the traced
warmup_model_decode), then for 1 and for 4 busy lanes one ``--context``-token prompt per lane, eager and with the
traced later chunks (MIMO_PREFILL_CHUNK_TRACE, default on; the eager runs hide the traces), and compares the last
logits. After the warmup it reports each replica's traces and how much of the trace region they use
(``MIMO_PREFILL_BATCH=1`` adds the prefill-batch traces, as in serving).

``--users``: requests per busy lane (each its own slot, blocks and prompt). ``--offset``: each lane's first request
takes ``--budget - offset`` tokens in its first step, so its later steps start off the chunk alignment; prefill_forward
moves such a start back, and with the budget at the model's max_prefill_chunk the row is longer than one chunk (a
pass that would mix a replay and an eager tail, which runs all eager: TtMiMoModel.chunk_traced), next to the lane's
other requests' traced rows. The chunk replays are counted per run.

The KV cache is vLLM's layout (one global, five SWA groups; tests/test_spec_decode_tp8_dp4.vllm_kv_layout) with the
KV pair split by default; each lane's request has its own global blocks and a ring of SWA blocks (vLLM frees the
SWA blocks a window no longer reads, so the window and the chunk are all a step needs).

Every line to grep starts with ``MIMO_VA``.
"""

import argparse
import os
import time
from types import SimpleNamespace

os.environ.setdefault("MIMO_PREFILL_BATCH", "0")  # no batch traces: these prompts are long

import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.reference.checkpoint import default_model_path
from models.experimental.mimo_v2_6_flash.tt import generator_vllm
from models.experimental.mimo_v2_6_flash.tt.generator_vllm import BLOCK_SIZE, MiMoV2ForCausalLM

SWA_RING = 64  # SWA blocks per request: a window and a chunk of up to 4096 + 255 positions fit many times


def say(msg):
    logger.info(msg)
    print(f"MIMO_VA {msg}", flush=True)


def vllm_kv_layout(cfg):
    """As tests/test_spec_decode_tp8_dp4.vllm_kv_layout."""
    kinds = {}
    for i in range(cfg.num_hidden_layers):
        kinds.setdefault(cfg.is_swa(i), []).append(i)
    size = min(len(layers) for layers in kinds.values())
    groups, buffers = [None] * cfg.num_hidden_layers, [None] * cfg.num_hidden_layers
    first = 0
    for layers in kinds.values():
        n = -(-len(layers) // size)
        for k in range(n):
            for t, i in enumerate(layers[k::n]):
                groups[i], buffers[i] = first + k, t
        first += n
    return groups, buffers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--budget", type=int, default=2048, help="tokens per lane per step")
    parser.add_argument("--per-lane", type=int, default=4, help="--max-num-seqs (decode slots per lane)")
    parser.add_argument("--lanes", default="1,4", help="busy-lane counts to run")
    parser.add_argument("--modes", default="eager,traced")
    parser.add_argument("--trace-region-mb", type=int, default=1000)
    parser.add_argument("--users", type=int, default=1, help="requests per busy lane (at most --per-lane)")
    parser.add_argument("--offset", type=int, default=0, help="the first step is --budget - offset tokens")
    args = parser.parse_args()
    assert 1 <= args.users <= args.per_lane and 0 <= args.offset < args.budget
    dp, n = 4, args.context
    path = default_model_path()
    cfg = MiMoConfig.from_model_path(path)
    pair = generator_vllm.KV_PAIR_SPLIT
    groups, buffers = vllm_kv_layout(cfg)
    MiMoV2ForCausalLM._layer_groups = groups
    MiMoV2ForCausalLM.prefill_token_budget = args.budget
    max_seq_len = n + 1024

    # Blocks per lane: each request's global blocks (128 positions each with the pair split) and a SWA ring per SWA
    # group, and at least as many as a SWA table is wide (the adapter's tables are as wide as the pool allows).
    global_span = 2 * BLOCK_SIZE if pair else BLOCK_SIZE
    global_blocks = -(-max_seq_len // global_span)
    num_groups = max(groups) + 1
    swa_groups = sorted({g for i, g in enumerate(groups) if cfg.is_swa(i)})
    per_request = global_blocks + len(swa_groups) * SWA_RING
    num_blocks = max(1 + args.users * per_request, -(-max_seq_len // BLOCK_SIZE) + 1)
    rows = dp * args.per_lane
    width = -(-max_seq_len // BLOCK_SIZE)
    group_rows = {}  # (request of the lane, group) -> its page-table row
    for u in range(args.users):
        for g in range(num_groups):
            row = torch.zeros(width, dtype=torch.int32)
            if g in swa_groups:
                base = 1 + u * per_request + global_blocks + swa_groups.index(g) * SWA_RING
                row[:] = base + torch.arange(width, dtype=torch.int32) % SWA_RING
            else:
                row[:global_blocks] = 1 + u * per_request + torch.arange(global_blocks, dtype=torch.int32)
            group_rows[(u, g)] = row

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(4, 8), l1_small_size=65536, trace_region_size=args.trace_region_mb << 20
    )
    try:
        t0 = time.time()
        hf = SimpleNamespace(num_hidden_layers=cfg.num_hidden_layers, _name_or_path=path)
        model = MiMoV2ForCausalLM.initialize_vllm_model(hf, mesh, rows, max_seq_len, tt_data_parallel=dp)
        specs = []
        for i in range(cfg.num_hidden_layers):
            dims = cfg.attention_dims(i)
            block = 2 * BLOCK_SIZE if pair and not cfg.is_swa(i) else BLOCK_SIZE
            specs.append(((num_blocks, 1, block, dims.head_dim), torch.bfloat16, buffers[i]))
        kv = model.allocate_kv_cache_per_layer(specs)
        say(f"built in {time.time() - t0:.0f} s: {num_blocks} blocks per lane, pair split {pair}")
        t0 = time.time()
        model.warmup_model_prefill(kv)
        model.warmup_model_decode(kv, enable_trace=False)
        model.warmup_model_decode(kv, enable_trace=True)
        say(
            f"warmed up in {time.time() - t0:.0f} s; chunk traces per lane: "
            f"{[list(m.chunk_traces) for m in model.models][0]}"
        )
        for lane, m in enumerate(model.models):
            used, total = m.trace_region_bytes()
            say(
                f"TRACES replica {lane}: decode {m.decode_trace is not None}, verify {m.verify_trace is not None}, "
                f"{sum(t['id'] is not None for t in m.prefill_traces.values())} prefill-batch, "
                f"{sum(t['id'] is not None for t in m.chunk_traces.values())} later-chunk; trace region "
                f"{used / 2**20:.1f} MB of {total / 2**20:.1f} MB used per device "
                f"({(total - used) / 2**20:.1f} MB free)"
            )
        prompts = [
            torch.randint(
                0, cfg.vocab_size, (n,), generator=torch.Generator().manual_seed(100 + lane), dtype=torch.int32
            )
            for lane in range(dp * args.users)
        ]
        saved = [m.chunk_traces for m in model.models]
        last = {}
        passes = {}
        for m in model.models:  # count each chunk prefill_enqueue replays (the rest run eagerly)
            replay = m._replay_prefill_chunk

            def counted(*a, _replay=replay, **k):
                passes["replays"] = passes.get("replays", 0) + 1
                return _replay(*a, **k)

            m._replay_prefill_chunk = counted
        for busy in [int(x) for x in args.lanes.split(",")]:
            for mode in args.modes.split(","):
                for m, traces in zip(model.models, saved):
                    m.chunk_traces = traces if mode == "traced" else {}
                passes.clear()
                lanes = list(range(busy))
                slots = [lane * args.per_lane + u for lane in lanes for u in range(args.users)]
                tokens = torch.stack([prompts[lane * args.users + u] for lane in lanes for u in range(args.users)])
                users = [u for _ in lanes for u in range(args.users)]
                t_run = time.time()
                step_times = []
                pos = [0] * len(slots)
                logits = [None] * len(slots)
                while min(pos) < n:
                    live = [r for r in range(len(slots)) if pos[r] < n]
                    ends = [
                        min(pos[r] + args.budget - (args.offset if pos[r] == 0 and users[r] == 0 else 0), n)
                        for r in live
                    ]
                    t1 = time.perf_counter()
                    width_rows = {g: torch.stack([group_rows[(users[r], g)] for r in live]) for g in range(num_groups)}
                    out = model.prefill_forward(
                        tokens[live],
                        torch.tensor([pos[r] for r in live], dtype=torch.int32),
                        torch.tensor(ends, dtype=torch.int32),
                        width_rows[0],
                        kv,
                        empty_slots=[slots[r] for r in live],
                        page_tables_per_layer=[width_rows[g] for g in groups],
                    )
                    step_times.append(time.perf_counter() - t1)
                    for k, r in enumerate(live):
                        pos[r], logits[r] = ends[k], out[k]
                    if len(step_times) % 64 == 0:
                        say(
                            f"{busy} lanes {mode}: position {min(pos)}, last 64 steps {sum(step_times[-64:]):.1f} s "
                            f"({1e3 * sum(step_times[-64:]) / 64:.0f} ms per step)"
                        )
                logits = torch.stack(logits)
                total = time.time() - t_run
                last[(busy, mode)] = logits
                cmp = ""
                if mode == "traced" and (busy, "eager") in last:
                    a, b = last[(busy, "eager")].float(), logits.float()
                    cmp = f"; vs eager max|d| {float((a - b).abs().max()):.4g}, top1 equal {bool((a.argmax(-1) == b.argmax(-1)).all())}"
                say(
                    f"RESULT {busy} lanes {mode}: {n} tokens per lane in {total:.1f} s ({n / total:.0f} tok/s per lane, "
                    f"{busy * n / total:.0f} for the Galaxy), {len(step_times)} steps, median step "
                    f"{1e3 * sorted(step_times)[len(step_times) // 2]:.0f} ms, {passes.get('replays', 0)} chunk replays"
                    f"{cmp}"
                )
        say(f"decode recaptures {model.decode_recaptures}")
    finally:
        for submesh in mesh.get_submeshes():
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
