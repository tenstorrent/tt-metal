# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The multi-user rows profile's shape on the Qwen3.8 checkpoint (1 x 4 mesh): step time by active users and context,
and the DRAM budget that sizes its KV pool.

Built as the served rows profile builds it (generator_vllm.Qwen36ResidentRowsForCausalLM): the PP prefill at max_len,
then a rows model of 8 slots (embed = the prefill's table: the hubs draw and feed every row's next token) on a paged
KV pool, the handoff / clear_slot warm-ups, then the traces. The pool is sized from the DRAM left after the prefill
model (--pool-blocks 0: everything but the rows model's other buffers and --margin-gb).

Step time: traced, fed (step(None) back to back, no host work between replays), each row sampling on device
(temperature 0.6, top-k 20, top-p 0.95):
  rows build, 1 / 2 / 4 / 8 active rows (rows 0 .. a-1 at position T - 1, the others empty) at each context T;
  a B = 1 build (fed, same sampling) at each T.
The KV pages hold zeros (device zero-fill); a step's time does not depend on them. Every row's block table holds its
own blocks, in shuffled pool order. After each point the fed positions are checked and the chips' AICLK is logged.

Smoke at the pool's far end: a prompt goes through the prefill into slot 5 on the pool's highest blocks, then greedy
fed steps; the B = 1 build decodes the same prompt (tokens compared; B = 1 runs another rmsnorm path, so a near-tie
may flip them).

Row orders (--orders): the a active rows are an order's first a. A B = 8 build splits each GDN value head over G head
cores (core g takes rows 8 g / G ..), so the rows a step activates decide how much of the GDN work runs in parallel:
"0,4,1,5,2,6,3,7" spreads users over G = 2 groups; "auto" (the default) is model.row_order(), the served fill order.
--no-pp: the embedding table alone (no prefill model, smoke, B = 1),
for quick probes, e.g. of RESIDENT_ATTN_WORKERS / RESIDENT_HEAD_GROUPS.

usage: python -m qwen36_resident.tests.rows_grid --corpus corpus_ids.pt [--max-len 262144] [--contexts ...]
           [--active 1,2,4,8] [--orders "0,1,2,3,4,5,6,7;0,4,1,5,2,6,3,7"] [--no-pp] [--out rows_grid.json]
"""

import argparse
import faulthandler
import gc
import json
import sys
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.model import Dims, ResidentModel, State, is_attn
from qwen36_resident.prefill.pp_prefill import PPPrefill
from qwen36_resident.tests.e0_bringup import PREFILL_CHUNKS, close_mesh, open_mesh

B = 8
BLOCK = 64
SAMPLING = (0.6, 20, 0.95)


def dram(mesh):
    """per chip, GB: total, allocated, free, and the largest contiguous free range times the banks"""
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    g = lambda x: x * v.num_banks / 1e9
    return dict(
        total=g(v.total_bytes_per_bank),
        allocated=g(v.total_bytes_allocated_per_bank),
        free=g(v.total_bytes_free_per_bank),
        largest_free=g(v.largest_contiguous_bytes_free_per_bank),
    )


def aiclk():
    out = []
    for i in range(4):
        try:
            out.append(int(open(f"/sys/class/tenstorrent/tenstorrent!{i}/tt_aiclk").read()))
        except OSError:
            out.append(None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--max-len", type=int, default=262144)
    ap.add_argument("--contexts", default="128,4096,32768,131072")
    ap.add_argument("--active", default="1,2,4,8")
    ap.add_argument("--time-steps", type=int, default=64)
    ap.add_argument("--warm-steps", type=int, default=8)
    ap.add_argument("--gap", type=float, default=3.0, help="idle seconds between grid points (chip clocks)")
    ap.add_argument("--pool-blocks", type=int, default=0, help="0: size the pool from the DRAM left")
    ap.add_argument("--margin-gb", type=float, default=0.5)
    ap.add_argument("--prompt-len", type=int, default=2000)
    ap.add_argument("--smoke-steps", type=int, default=32)
    ap.add_argument("--no-b1", action="store_true")
    ap.add_argument(
        "--orders",
        default="auto",
        help="';'-separated row orders: a active rows are an order's first a (later orders skip active sets measured); "
        "auto: model.row_order()",
    )
    ap.add_argument("--no-pp", action="store_true", help="no prefill model: embedding table only, no smoke, no B=1")
    ap.add_argument("--out")
    a = ap.parse_args()
    faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
    ids = torch.load(a.corpus)["ids"]
    contexts = [int(t) for t in a.contexts.split(",")]
    actives = [int(x) for x in a.active.split(",")]
    a.no_b1 |= a.no_pp
    t_start = time.time()
    out = dict(args=vars(a), dram_gb={}, step_ms={}, aiclk={}, fed_ok={})

    def phase(name):
        print(f"[{time.time() - t_start:7.1f} s] {name}", flush=True)

    def log_dram(tag):
        out["dram_gb"][tag] = v = dram(mesh)
        phase(f"DRAM/chip {tag}: " + ", ".join(f"{k} {x:.3f}" for k, x in v.items()) + " GB")
        return v

    mesh = open_mesh()
    try:
        ck = Q.Checkpoint(Q.checkpoint_dir())
        cfg = ck.config
        layers, interval, vocab = cfg["num_hidden_layers"], cfg["full_attention_interval"], cfg["vocab_size"]
        n = mesh.get_num_devices()
        d = Dims(n, mesh.dram_grid_size().x, vocab=vocab)
        emb = Q.embedding(ck)
        C = max(PREFILL_CHUNKS)
        max_len = -(-a.max_len // C) * C
        max_pos = max_len + 64
        n_attn = sum(is_attn(l, interval) for l in range(layers))
        empty = lambda: State(d, layers - n_attn, n_attn, 0, max_pos=max_pos, zero=True, kv=False)
        log_dram("open")
        if a.no_pp:
            pp = None
            embed = ttnn.from_torch(
                emb.bfloat16(),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            after_pp = log_dram("after embedding table")
        else:
            pp = PPPrefill(mesh, ck, layers, interval, chunk=list(PREFILL_CHUNKS), max_len=max_len)
            embed = pp.embed
            phase(f"prefill model (max_len {max_len})")
            after_pp = log_dram("after prefill model")

        # pool: the DRAM left after the prefill model, less the rows model's other buffers (decode weights, GDN states,
        # conv histories, rope table, token states: measured below) and the margin
        block_bytes = 2 * n_attn * (BLOCK // 32) * 8 * 1088  # K and V, every attention layer, 8 bf8 tiles a page
        need = -(-(max(contexts) + 1 + a.warm_steps + a.time_steps + 64) // BLOCK)  # blocks per row the grid writes
        smoke_need = -(-(a.prompt_len + a.smoke_steps + 64) // BLOCK)
        if a.pool_blocks:
            nb = a.pool_blocks
        elif a.no_pp:
            nb = B * need + smoke_need
        else:
            other_gb = 6.45  # decode weights 5.97 + 8 GDN states 0.30 + conv histories 0.08 + rope 0.03 + small
            nb = int((after_pp["largest_free"] - other_gb - a.margin_gb) * 1e9 // block_bytes)
        assert nb >= B * need + smoke_need, (nb, B * need + smoke_need)
        perm = torch.randperm(nb - smoke_need, generator=torch.Generator().manual_seed(11)).tolist()
        tables = [perm[u * need : (u + 1) * need] for u in range(B)]
        smoke_table = list(range(nb - smoke_need, nb))  # the pool's highest blocks
        out["pool"] = dict(blocks=nb, block_bytes=block_bytes, gb=nb * block_bytes / 1e9, tokens=nb * BLOCK)
        phase(f"KV pool {nb} blocks = {nb * block_bytes / 1e9:.3f} GB/chip = {nb * BLOCK} tokens")

        m = ResidentModel(
            mesh,
            d,
            Q.load(ck, d, layers, interval),
            [empty() for _ in range(B)],
            layers,
            interval,
            lm_head=True,
            embed=embed,
            kv_pool_blocks=nb,
            block_tables=[[0]] * B,
            rows=True,
        )
        gc.collect()
        phase(f"rows decode model (GDN head groups G={m.head_groups}: core g of a head takes rows {B} g / G ..)")
        out["head_groups"] = m.head_groups
        after_rows = log_dram("after rows model")
        out["rows_model_other_gb"] = after_rows["allocated"] - after_pp["allocated"] - nb * block_bytes / 1e9
        # the blocks that would still fit: the largest free range, less the margin
        room = int(max(after_rows["largest_free"] - a.margin_gb, 0) * 1e9 // block_bytes)
        out["pool_max_blocks"] = dict(blocks=nb + room, tokens=(nb + room) * BLOCK, margin_gb=a.margin_gb)
        phase(
            f"rows model's buffers besides the pool: {out['rows_model_other_gb']:.3f} GB/chip; the pool could take "
            f"{room} more blocks: max {nb + room} blocks = {(nb + room) * BLOCK} tokens (margin {a.margin_gb} GB)"
        )
        # warm-ups before the captures (contract 6a), as the served profile does
        if pp is not None:
            pp.handoff(m, 1, block_table=[0], slot=0)
        m.clear_slot(0)
        m.capture_trace()
        if pp is not None:
            pp.capture()
        m.reset(state=False)
        log_dram("after traces")
        m.set_page_table(tables)
        m.set_sampling(rows=[(*SAMPLING, 1000 + u) for u in range(B)])
        phase("traces")

        def time_fed(model, tok, check):
            """ms per traced fed step: tok first, then step(None) back to back; check(fed_token(), steps)"""
            model.reset(state=False)
            model.step(tok)
            for _ in range(a.warm_steps):
                model.step(None)
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            for _ in range(a.time_steps):
                model.step(None)
            ttnn.synchronize_device(mesh)
            ms = (time.time() - t0) / a.time_steps * 1e3
            ok = check(model.fed_token(), 1 + a.warm_steps + a.time_steps)
            return ms, ok

        # (order index, active rows): an order's first a rows, each active set once
        orders = [m.row_order() if o == "auto" else [int(x) for x in o.split(",")] for o in a.orders.split(";")]
        assert all(sorted(o) == list(range(B)) for o in orders)
        out["orders"] = orders
        points = []
        for oi, order in enumerate(orders):
            for act in actives:
                on = sorted(order[:act])
                if all(on != p for _, p in points):
                    points.append((oi, on))
        for T in contexts:
            for _, on in points:
                act = len(on)
                x0 = torch.zeros(B, 5120)
                for k, s in enumerate(on):
                    x0[s] = emb[int(ids[T - 1 + k])].float()
                active = [s in on for s in range(B)]
                tok = m.token_state(x0, [T - 1 if active[s] else 0 for s in range(B)], active=active)

                def check(fed, steps):
                    return all(
                        (f[1] == T - 1 + steps and f[2] and 0 <= f[0] < vocab) if active[s] else f == (0, 0, False)
                        for s, f in enumerate(fed)
                    )

                key = f"rows, {act} active, T={T}, rows {''.join(map(str, on))}"
                ms, ok = time_fed(m, tok, check)
                out["step_ms"][key], out["fed_ok"][key], out["aiclk"][key] = ms, ok, clk = ms, ok, aiclk()
                phase(f"{key}: step {ms:.3f} ms ({act / ms * 1e3:.1f} tok/s total); fed ok {ok}; AICLK {clk}")
                time.sleep(a.gap)

        # greedy at one point: the draw costs the same
        m.set_sampling(rows=[(0.0, 1, 1.0, 0)] * B)
        x0 = torch.stack([emb[int(ids[127 + s])].float() for s in range(B)])
        ms, ok = time_fed(
            m, m.token_state(x0, [127] * B, active=[True] * B), lambda fed, k: all(f[1] == 127 + k for f in fed)
        )
        out["step_ms"]["rows, 8 active, T=128, greedy"] = ms
        phase(f"rows, 8 active, T=128, greedy: step {ms:.3f} ms; fed ok {ok}")

        # smoke at the pool's far end: slot 5 on the highest blocks, the other rows idle
        if pp is not None:
            prompt = ids[: a.prompt_len]
            P = len(prompt)
            tables[5] = smoke_table
            m.set_page_table(tables)
            m.reset(state=False)
            pp.reset()
            pp.run(prompt[: P - 1])
            pp.handoff(m, P - 1, block_table=smoke_table, slot=5)
            x0 = torch.zeros(B, 5120)
            x0[5] = emb[int(prompt[-1])].float()
            m.step(m.token_state(x0, [P - 1 if s == 5 else 0 for s in range(B)], active=[s == 5 for s in range(B)]))
            rows_toks = []
            for k in range(a.smoke_steps):
                if k:
                    m.step(None)
                ttnn.synchronize_device(mesh)
                rows_toks.append(m.fed_token()[5][0])
            phase(f"smoke: slot 5 on blocks {smoke_table[0]}..{smoke_table[-1]}: {rows_toks}")
            out["smoke"] = dict(rows=rows_toks, blocks=[smoke_table[0], smoke_table[-1]])
            log_dram("after smoke")

        m.release_trace()
        if pp is not None:
            pp._ho.pop("model", None)
        del m
        gc.collect()
        ttnn.synchronize_device(mesh)

        if not a.no_b1:
            b1_table = list(range(need + smoke_need))
            m1 = ResidentModel(
                mesh,
                d,
                Q.load(ck, d, layers, interval),
                empty(),
                layers,
                interval,
                lm_head=True,
                embed=pp.embed,
                kv_pool_blocks=need + smoke_need,
                block_tables=[b1_table],
            )
            gc.collect()
            phase("B=1 decode model")
            pp.handoff(m1, 1, block_table=[0])
            m1.step(m1.token_state(emb[0].float(), 0))  # compile before the trace
            ttnn.synchronize_device(mesh)
            m1.capture_trace()
            pp.capture()
            m1.reset(state=False)
            m1.set_sampling(*SAMPLING, 1000)
            for T in contexts:
                key = f"B1, T={T}"
                tok = m1.token_state(emb[int(ids[T - 1])].float(), T - 1)
                ms, ok = time_fed(m1, tok, lambda fed, k: fed[1] == T - 1 + k and 0 <= fed[0] < vocab)
                out["step_ms"][key], out["fed_ok"][key], out["aiclk"][key] = ms, ok, clk = ms, ok, aiclk()
                phase(f"{key}: step {ms:.3f} ms ({1 / ms * 1e3:.1f} tok/s); fed ok {ok}; AICLK {clk}")
                time.sleep(a.gap)
            m1.set_sampling(0.0, 1, 1.0, 0)
            m1.reset()
            pp.reset()
            pp.run(prompt[: P - 1])
            pp.handoff(m1, P - 1, block_table=b1_table)
            m1.step(m1.token_state(emb[int(prompt[-1])].float(), P - 1))
            b1_toks = []
            for k in range(a.smoke_steps):
                if k:
                    m1.step(None)
                ttnn.synchronize_device(mesh)
                b1_toks.append(m1.fed_token()[0])
            lead = next((k for k, (x, y) in enumerate(zip(rows_toks, b1_toks)) if x != y), len(b1_toks))
            phase(f"smoke B=1: {b1_toks}; rows == B=1 for the first {lead}/{len(b1_toks)} tokens")
            out["smoke"].update(b1=b1_toks, leading_equal=lead)
            m1.release_trace()
            pp._ho.pop("model", None)
            del m1
    finally:
        close_mesh(mesh)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    phase("done")


if __name__ == "__main__":
    main()
