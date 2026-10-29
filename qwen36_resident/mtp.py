# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Greedy speculative decode of Qwen3.6-27B with its MTP layer (one draft token per step), host driven.

The main model runs the verify step (ResidentModel(verify=True)): rows t at p and the draft d at p + 1.
Its greedy tokens a0, a1 accept the draft when a0 == d: the step then commits d and a1, else a0. The
MTP model is one full-attention decoder layer with its own KV cache and the main model's lm_head (folded
with mtp.norm), run as a verify step too: its entry at position q takes x0 = fc([norm_e(embed(tok[q + 1])),
norm_h(h_q)]) for the main model's final hidden h_q and predicts tok[q + 2]. After a main step at p it
adds the entries p (and p + 1 when the draft was accepted) and its row on the kept path drafts the next
step's d. The fc runs on device (ResidentModel fc) on the main model's last x: norm_h(rmsnorm(x)) without
the final norm's weight in between (drafts accepted on the reference text: 84.7% instead of 86.4%).
generate() accepts drafts on the host; generate_fed() lets the two hubs write each other's token state
(mlp_hub.cpp feed kinds 1 and 2), so steps run back to back and the host reads each step's record late.
"""
import time

import torch
from loguru import logger

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.model import Dims, ResidentModel, State
from qwen36_resident.prefill.pp_prefill import PPPrefill

MTP = "mtp."
PREFILL_CHUNKS = (128, 256, 512, 1024)  # as served (generator_vllm.py)


class MtpCheckpoint:
    """the MTP layer as layer 0 of a checkpoint, its norm as the final norm"""

    def __init__(self, ck):
        self.ck, self.config = ck, ck.config

    def get(self, name):
        if name == f"{Q.PREFIX}norm.weight":
            name = f"{MTP}norm.weight"
        return self.ck.get(name)

    def layer(self, i, name):
        assert i == 0
        return self.ck.get(f"{MTP}layers.0.{name}")


def mtp_weights(ck, d):
    """ResidentModel weights of the MTP decoder layer (an attention layer) and its lm_head"""
    m = MtpCheckpoint(ck)
    w = dict(gdn=[], attn=[Q.attn_layer(m, 0, d)], mlp=[Q.mlp(m, 0, d)], out=[Q.out_proj(m, 0, d, True)])
    w["head"], w["vocab_per_chip"] = Q.lm_head(m, d)
    return w


def mtp_fc(ck, d):
    """per chip its K slice of the fc (see streamer_common.hpp): [2 H / n, H], the pre-fc norm weights folded"""
    w = ck.get(f"{MTP}fc.weight").float().T  # [2 H, H]: rows for embedding | hidden
    H = w.shape[1]
    scale = 1 + torch.cat([ck.get(f"{MTP}pre_fc_norm_{v}.weight").float() for v in ("embedding", "hidden")])
    w = (scale[:, None] * w).bfloat16()
    k = 2 * H // d.n
    return [w[c * k : (c + 1) * k].contiguous() for c in range(d.n)]


class SpecDecoder:
    def __init__(self, mesh, ck, layers, max_pos, trace=True, prefill=False):
        """prefill: also the pipeline-parallel prefill (generate_prefilled); max_pos then rounds up to its
        largest chunk"""
        n = mesh.get_num_devices()
        self.d = d = Dims(n, mesh.dram_grid_size().x, vocab=ck.config["vocab_size"])
        interval = ck.config["full_attention_interval"]
        n_attn = sum((l + 1) % interval == 0 for l in range(layers))
        self.emb = Q.embedding(ck).float()
        self.pp = None
        if prefill:
            C = max(PREFILL_CHUNKS)
            max_len = -(-max_pos // C) * C
            max_pos = max_len + 64
            # its embedding table is also the one the hubs feed from
            self.pp = PPPrefill(mesh, ck, layers, interval, chunk=list(PREFILL_CHUNKS), max_len=max_len)
            self.emb_t = self.pp.embed
            logger.info("prefill weights placed")
        else:
            self.emb_t = ttnn.from_torch(
                self.emb.bfloat16(),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
        w = Q.load(ck, d, layers, interval)
        main_tok, draft_tok = ResidentModel.tok_tensor(mesh, d, 2), ResidentModel.tok_tensor(mesh, d, 2, fc=True)
        self.main = ResidentModel(
            mesh,
            d,
            w,
            State(d, layers - n_attn, n_attn, 0, max_pos, zero=True),
            layers,
            interval,
            True,
            verify=True,
            embed=self.emb_t,
            tok=main_tok,
            feed_peer=draft_tok,
        )
        del w
        logger.info("main model placed")
        self.draft = ResidentModel(
            mesh,
            d,
            mtp_weights(ck, d),
            State(d, 0, 1, 0, max_pos, zero=True),
            1,
            1,
            True,
            verify=True,
            fc=mtp_fc(ck, d),
            embed=self.emb_t,
            tok=draft_tok,
            feed_peer=main_tok,
        )
        self.mesh = mesh
        if trace:
            for m in (self.main, self.draft):
                width = (2 if m.fc else 1) * self.emb.shape[1]
                m.step(m.token_state(torch.zeros(2, width), [0, 1], ring=0, slot=0))
                m.reset()
                m.capture_trace()
        if self.pp is not None:
            self.pp.capture()
            self.pp.handoff(self.main, 1)  # compiles the handoff kernel

    def _main(self, toks, p, ring, slot):
        m = self.main
        m.step(m.token_state(self.emb[toks], [p, p + 1], ring=ring, slot=slot, words={5: toks[1], 6: toks[0]}))
        return m.argmax(), m.x()

    draft_t = []

    def _draft(self, toks, h, q, ring, slot):
        """MTP entries at q, q + 1 for (toks[i], the main model's x of row i); its argmax per row"""
        t0 = time.perf_counter()
        m = self.draft
        tok = m.token_state(torch.cat([self.emb[toks], h], dim=1), [q, q + 1], ring=ring, slot=slot)
        t1 = time.perf_counter()
        m.step(tok)
        a = m.argmax()
        self.draft_t.append((t1 - t0, time.perf_counter() - t1))
        return a

    def generate(self, prompt, n_new):
        """greedy: the prompt's tokens, then n_new generated ones; returns (tokens, stats)"""
        seq = [int(t) for t in prompt]
        P = len(seq)
        assert P >= 2
        p = ring = slot = 0
        # the prompt in pairs; the pair holding position P - 2 continues from that row, so the main model has
        # seen positions < P - 1 and the draft row on the kept path predicts position P
        while p <= P - 2:
            accept = int(p + 1 <= P - 2)
            a, h = self._main([seq[p], seq[p + 1]], p, ring, slot)
            nxt = [seq[p + 1], seq[p + 2] if p + 2 < P else seq[p + 1]]
            d = self._draft(nxt, h, p, ring, slot)[accept]
            p, ring, slot = (p + 2, ring + 2, slot ^ 1) if accept else (p + 1, ring + 1, slot)
        assert p == P - 1
        steps = accepted = 0
        t0 = time.perf_counter()
        times, main_t = [], []
        self.draft_t = []
        while len(seq) < P + n_new:
            ts = time.perf_counter()
            a, h = self._main([seq[p], d], p, ring, slot)
            main_t.append(time.perf_counter() - ts)
            accept = int(a[0] == d)
            new = [d, a[1]] if accept else [a[0]]
            seq += new
            d = self._draft([a[0], a[1]], h, p, ring, slot)[accept]
            p, ring, slot = (p + 2, ring + 2, slot ^ 1) if accept else (p + 1, ring + 1, slot)
            steps += 1
            accepted += accept
            times.append((time.perf_counter() - ts, len(new)))
        total = time.perf_counter() - t0
        out = seq[: P + n_new]
        stats = dict(
            steps=steps,
            accept_rate=accepted / steps,
            tokens_per_step=(len(seq) - P) / steps,
            ms_per_step=1e3 * sorted(t for t, _ in times)[len(times) // 2],
            main_ms=1e3 * sorted(main_t)[len(main_t) // 2],
            draft_host_ms=1e3 * sorted(a for a, _ in self.draft_t)[len(self.draft_t) // 2],
            draft_device_ms=1e3 * sorted(b for _, b in self.draft_t)[len(self.draft_t) // 2],
            tok_s=(len(seq) - P) / total,
        )
        return out, stats

    def _prompt(self, seq):
        """the prompt through both models, host driven (see generate); returns (p, ring, slot, draft)"""
        P = len(seq)
        p = ring = slot = 0
        while p <= P - 2:
            accept = int(p + 1 <= P - 2)
            a, h = self._main([seq[p], seq[p + 1]], p, ring, slot)
            nxt = [seq[p + 1], seq[p + 2] if p + 2 < P else seq[p + 1]]
            d = self._draft(nxt, h, p, ring, slot)[accept]
            p, ring, slot = (p + 2, ring + 2, slot ^ 1) if accept else (p + 1, ring + 1, slot)
            if p % 4096 < 2:
                logger.info(f"prompt: {p} / {P} positions")
        return p, ring, slot, d

    def generate_fed(self, prompt, n_new):
        """generate() with the hubs feeding each other: the host writes only the first step's token state and
        reads each step's (accept, a0, a1) record while the next steps run"""
        for m in (self.main, self.draft):
            m.reset()
        seq = [int(t) for t in prompt]
        p, ring, slot, d = self._prompt(seq)
        return self._fed(seq, n_new, p, ring, slot, d, base=0, t0=time.perf_counter())

    def generate_prefilled(self, prompt, n_new, timings=None):
        """generate_fed() after the pipeline-parallel prefill of all but the last prompt token, handed off to
        the main model. The draft model gets no prompt entries: its positions start at the last prompt
        token (base), and the first step's draft is a placeholder (so that step decodes one token)"""
        for m in (self.main, self.draft):
            m.reset()
        self.pp.reset()
        seq = [int(t) for t in prompt]
        p = len(seq) - 1
        t0 = time.perf_counter()
        if p > 0:
            self.pp.run(torch.tensor(seq[:p]))
            if timings is not None:
                ttnn.synchronize_device(self.mesh)
                timings["prefill"] = time.perf_counter() - t0
            self.pp.handoff(self.main, p, timings)
        return self._fed(seq, n_new, p, 0, 0, seq[p], base=p, t0=t0)

    def _fed(self, seq, n_new, p, ring, slot, d, base, t0):
        """the hub-fed loop from the main step at p with draft d; t0: when the request started"""
        P = len(seq)
        m = self.main
        words = {5: d, 6: seq[p], 10: base}
        m.step(m.token_state(self.emb[[seq[p], d]], [p, p + 1], ring=ring, slot=slot, words=words))
        self.draft.step(None)
        view = ttnn.get_device_tensors(m.out_t)[0]
        pending = []
        steps = accepted = 0
        ttft = None
        first = True
        while len(seq) < P + n_new:
            if not first:
                m.step(None)
                self.draft.step(None)
            # the main step's record was written before the draft step that follows it
            pending.append((ttnn.from_device(view, blocking=False), ttnn.record_event(self.mesh, 0)))
            first = False
            if len(pending) < 2:
                continue
            host, ev = pending.pop(0)
            ttnn.event_synchronize(ev)
            if ttft is None:
                ttft = time.perf_counter() - t0
            rec = ttnn.to_torch(host).reshape(-1)[:4].tolist()
            accept, a0, a1 = rec[0], rec[1], rec[2]
            seq += [a0, a1] if accept else [a0]
            steps += 1
            accepted += accept
        ttnn.synchronize_device(self.mesh)
        total = time.perf_counter() - t0
        return seq[: P + n_new], dict(
            ttft_s=ttft,
            decode_tok_s=(len(seq) - P - 1) / (total - ttft),  # after the first token
            fed_steps=steps,
            fed_accept_rate=accepted / steps,
            fed_tokens_per_step=(len(seq) - P) / steps,
            fed_ms_per_step=1e3 * total / steps,
            fed_tok_s=(len(seq) - P) / total,
        )

    # ---- streaming, for the vLLM block adapter: start() per request, then next_block() per decode step. The
    # device runs `depth` main + draft steps ahead of the host; a step's record is read once the host needs
    # its tokens, and tokens past a block wait for the next one, so the device keeps decoding while the
    # host is away (the steps it ran past the end of a request are discarded by stop()).

    def start(self, prompt):
        """the prefill (prompt but its last token) and the first verify step: returns that step's row-0 logits
        [vocab] (the first token is their argmax); a second token it may commit waits for next_block"""
        self.stop()
        seq = [int(t) for t in prompt]
        p = len(seq) - 1
        # the handoff writes every state the main model reads (KV past the prompt is masked), and the draft
        # model writes each KV row before it attends to it: only a one-token prompt needs the initial state
        self.main.reset(state=p == 0)
        self.draft.reset(state=False)
        self.pp.reset()
        if p > 0:
            self.pp.run(torch.tensor(seq[:p]))
            self.pp.handoff(self.main, p)
        m = self.main
        words = {5: seq[p], 6: seq[p], 10: p}  # the placeholder draft repeats the token
        m.step(m.token_state(self.emb[[seq[p], seq[p]]], [p, p + 1], ring=0, slot=0, words=words))
        logits = m.logits()[0][:, : self.d.vocab_chip].reshape(-1)[: self.emb.shape[0]]
        accept, a0, a1, _ = self._record(ttnn.from_device(self._out_view(), blocking=True))
        self.draft.step(None)
        self._carry = [a1] if accept else []
        self._last_pos = p + 1 + accept  # position of the last token the device committed
        return logits

    def next_block(self, n, depth, limit):
        """the next n tokens, or fewer once the device would pass position `limit`"""
        while len(self._carry) < n:
            self._fill(depth, limit)
            if not self._pending:
                break
            host, ev = self._pending.pop(0)
            ttnn.event_synchronize(ev)
            accept, a0, a1, p = self._record(host)
            self._carry += [a0, a1] if accept else [a0]
            self._last_pos = p + 1 + accept
        self._fill(depth, limit)
        block, self._carry = self._carry[:n], self._carry[n:]
        return block

    def stop(self):
        """end the request: wait for the steps run ahead and drop them"""
        if getattr(self, "_pending", None):
            ttnn.synchronize_device(self.mesh)
        self._pending, self._carry, self._last_pos = [], [], 0

    def _fill(self, depth, limit):
        # each queued step commits up to 2 positions past the last one read
        while len(self._pending) < depth and self._last_pos + 2 * (len(self._pending) + 1) + 1 < limit:
            self.main.step(None)
            # the main step's record, read before the draft step (which does not write it) and the next main step
            read = ttnn.from_device(self._out_view(), blocking=False)
            self._pending.append((read, ttnn.record_event(self.mesh, 0)))
            self.draft.step(None)

    def _out_view(self):
        return ttnn.get_device_tensors(self.main.out_t)[0]

    @staticmethod
    def _record(host):
        """(accept, a0, a1, p) of a main step (mlp_hub.cpp feed kind 1)"""
        return ttnn.to_torch(host).reshape(-1)[:4].tolist()

    def greedy(self, prompt, n_new, prefill=False):
        """plain greedy on the main model's verify step (row 1 repeats row 0's token and is never kept): the
        tokens speculative decode must reproduce (prefill: after the prefill, as generate_prefilled)"""
        seq = [int(t) for t in prompt]
        p = ring = slot = 0
        if prefill:
            self.main.reset()
            self.pp.reset()
            p = len(seq) - 1
            if p > 0:
                self.pp.run(torch.tensor(seq[:p]))
                self.pp.handoff(self.main, p)
        while len(seq) < len(prompt) + n_new:
            a, _ = self._main([seq[p], seq[p]], p, ring, slot)
            if p + 1 >= len(seq):
                seq.append(a[0])
            p, ring = p + 1, ring + 1
        return seq
