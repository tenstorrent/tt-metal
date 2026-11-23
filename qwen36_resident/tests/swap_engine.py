# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Strata S3 swap round trip on the real engine (vllm_rows.RowsEngine.swap_out / swap_restore, prefix_cache.SwapStore):
slot 0 decodes a prompt uninterrupted (the reference); slot 1 decodes the same prompt for --k1 steps, is swapped out to
host (its state at the q consumed tokens + the next token), dirtied by another prompt, and the state is restored into
slot 2 (other KV blocks), which decodes on. Slot 2's tokens after the swap must equal the reference's, and its logits
bit-identical (the same state, the same kernels). The Qwen3.8 checkpoint, the PP prefill, a rows build of 8.
usage: python -m qwen36_resident.tests.swap_engine [--prompt 3000] [--k1 24] [--k2 32] [--max-len 4096]
"""

import argparse
import faulthandler
import sys
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.generator_vllm import PREFILL_CHUNKS
from qwen36_resident.model import Dims, ResidentModel, State, is_attn
from qwen36_resident.prefill.pp_prefill import PPPrefill
from qwen36_resident.prefix_cache import SwapStore
from qwen36_resident.tests.e0_bringup import shuffled_tables
from qwen36_resident.tests.long_ctx import close_mesh, open_mesh
from qwen36_resident.vllm_rows import RowsEngine

B = 8


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/ttuser/experiments/qwen38/results/native/corpus_ids_40k.pt")
    ap.add_argument("--prompt", type=int, default=3000)
    ap.add_argument("--k1", type=int, default=24)
    ap.add_argument("--k2", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=4096)
    a = ap.parse_args()
    ids = torch.load(a.corpus)["ids"]
    prompt = ids[5000 : 5000 + a.prompt].tolist()
    other = ids[20000 : 20000 + a.prompt // 2].tolist()
    t_start = time.time()

    def phase(name):
        print(f"[{time.time() - t_start:7.1f} s] {name}", flush=True)

    mesh = open_mesh()
    ok = False
    try:
        ck = Q.Checkpoint(Q.checkpoint_dir())
        cfg = ck.config
        layers, interval, vocab = cfg["num_hidden_layers"], cfg["full_attention_interval"], cfg["vocab_size"]
        d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x, vocab=vocab)
        emb = Q.embedding(ck)
        C = max(PREFILL_CHUNKS)
        max_len = -(-a.max_len // C) * C
        max_pos = max_len + 64
        n_attn = sum(is_attn(l, interval) for l in range(layers))
        pp = PPPrefill(mesh, ck, layers, interval, chunk=list(PREFILL_CHUNKS), max_len=max_len)
        blocks, tables = shuffled_tables(B, max_pos, seed=37)
        m = ResidentModel(
            mesh,
            d,
            Q.load(ck, d, layers, interval),
            [State(d, layers - n_attn, n_attn, 0, max_pos=max_pos, zero=True, kv=False) for _ in range(B)],
            layers,
            interval,
            lm_head=True,
            kv_pool_blocks=blocks,
            block_tables=tables,
            rows=True,
        )
        m.capture_trace()
        pp.capture()
        m.reset()
        eng = RowsEngine(m, pp, emb, vocab, d.vocab_chip, B, 1024, mesh=mesh)
        eng.swap = SwapStore(int(40e9))
        phase("models")

        def step(slot, tok, pos):
            x0 = torch.zeros(B, 5120)
            x0[slot] = emb[int(tok)].float()
            p = [0] * B
            p[slot] = pos
            m.step(m.token_state(x0, p, active=[s == slot for s in range(B)], preserve=True))
            ttnn.synchronize_device(mesh)
            return m.logits()[slot, :, : d.vocab_chip].reshape(-1)[:vocab].float().clone()

        def prefill(slot, toks):
            eng.prefill_whole(toks[:-1], slot, tables[slot], eng.pick_chunk(len(toks) - 1))
            ttnn.synchronize_device(mesh)

        P = len(prompt)
        # the reference: slot 0, uninterrupted
        prefill(0, prompt)
        ref_lg, seq = [], list(prompt)
        lg = step(0, seq[-1], P - 1)
        for k in range(a.k1 + a.k2):
            seq.append(int(lg.argmax()))
            lg = step(0, seq[-1], len(seq) - 1)
            ref_lg.append(lg)
        ref = seq[P:]
        phase(f"slot 0: reference, {len(ref)} tokens")
        # slot 1: k1 steps, then out to host
        prefill(1, prompt)
        got = list(prompt)
        lg = step(1, got[-1], P - 1)
        for k in range(a.k1):
            got.append(int(lg.argmax()))
            lg = step(1, got[-1], len(got) - 1)
        got.append(int(lg.argmax()))  # (the host saw the next token: tokens = q + 1)
        q = len(got) - 1
        t0 = time.perf_counter()
        kept = eng.swap_out(1, tables[1], q, got, device_tok=False)
        t_out = time.perf_counter() - t0
        phase(f"slot 1: swapped out at q = {q} (kept {kept}, {1e3 * t_out:.0f} ms, {eng.swap.bytes / 1e6:.0f} MB host)")
        prefill(1, other)  # (slot 1 reused by another prompt)
        step(1, other[-1], len(other) - 1)
        m_ = eng.swap.match(got)
        assert m_ is not None and m_[1] == q, m_
        t0 = time.perf_counter()
        eng.swap_restore(2, m_[0], tables[2])
        ttnn.synchronize_device(mesh)
        t_in = time.perf_counter() - t0
        phase(f"slot 2: restored ({1e3 * t_in:.0f} ms), decoding on")
        lgs = []
        lg = step(2, got[q], q)
        lgs.append(lg)
        for k in range(a.k2 - 1):
            got.append(int(lg.argmax()))
            lg = step(2, got[-1], len(got) - 1)
            lgs.append(lg)
        got.append(int(lg.argmax()))
        after = got[P:]
        agree = sum(int(x == y) for x, y in zip(after, ref))
        n = min(len(after), len(ref))
        bit = all(torch.equal(x, y) for x, y in zip(lgs, ref_lg[a.k1 :]))
        diff = max(float((x - y).abs().max()) for x, y in zip(lgs, ref_lg[a.k1 :]))
        ok = kept and agree == n and bit
        print(
            f"swap round trip: tokens {agree}/{n} equal to the uninterrupted decode, logits after the swap bit-identical "
            f"{bit} (max |diff| {diff:.4f}); swap-out {1e3 * t_out:.0f} ms, restore {1e3 * t_in:.0f} ms for {q} tokens",
            flush=True,
        )
        print(f"swap engine: {'PASS' if ok else 'FAIL'}", flush=True)
    finally:
        close_mesh(mesh)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
