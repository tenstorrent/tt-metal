# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4Engine (task product-integ): the model side of the product server, one object that owns the 78-layer chain, the sequence-sharded SCALED_FP8 pool, the direct-to-pool prefill and the decode loop.

  * prefill: ``Hy4PoolPrefill`` (chunked, any prompt length up to the pool, 4 users in lock step, the pool writes of the rows that are NOT being prefilled are masked: live decode users keep their pages). With the
    MTP layer built, the MTP prompt KV rows are written per chunk (``mtp_prompt_chunk``) and a SEED round writes the pair of the last prompt position and produces the first K drafts.
  * decode, SPEC mode (K >= 1, default): ``Hy4SpecRunner`` (one traced round = verify S = K + 1 rows -> accept -> MTP draft, all state resident), a GREEDY trace and a lossless SAMPLED trace
    (``hy4_spec_sample``, temperature only) sharing every persistent tensor. The host reads (a_0..a_k, m, P, drafts, next tokens) per round.
  * decode, PLAIN mode (K = 0): ``Hy4PoolDecoder`` S = 1 + ``DeviceLoop`` (device-resident loop and sampler, one trace per sampler variant).
Persistent tensors are allocated in ``__init__`` / ``warm`` BEFORE any trace capture (D75-1: a tensor allocated after a capture can be clobbered by the replays).

State mirror (host, per PHYSICAL mesh row): pending token, its position P, the K drafts, live flag. The device state of a round is authoritative; the mirror is refreshed from the read-back of every round and
re-uploaded (``Hy4SpecRunner.load``) only on a transition (prefill, a row freed, an idle row present, a mismatch with the plugin's block).
"""
import os
import time

import torch
from loguru import logger

import ttnn

PAD = 120002
ROWS = 4
PAGE = 512


class _Seed:
    """host sampling of the FIRST token of a request from its last-prompt-position logits row (fp32 [V])."""

    @staticmethod
    def sample(row, temperature, top_k, top_p, seed, pos):
        if temperature is None or temperature <= 0 or top_k == 1:
            return int(row.argmax())
        g = torch.Generator().manual_seed((int(seed) * 1000003 + int(pos)) & 0x7FFFFFFFFFFF)
        x = row.double() / float(temperature)
        if top_k is not None and 0 < top_k < x.numel():
            kth = torch.topk(x, int(top_k)).values[-1]
            x = torch.where(x >= kth, x, torch.full_like(x, float("-inf")))
        p = torch.softmax(x, dim=-1)
        if top_p is not None and 0 < top_p < 1.0:
            sp, si = torch.sort(p, descending=True)
            cs = torch.cumsum(sp, 0)
            keep = (cs - sp) < top_p
            p = torch.zeros_like(p).scatter_(0, si[keep], sp[keep])
            p = p / p.sum()
        return int(torch.multinomial(p.float(), 1, generator=g))


class Hy4Engine:
    def __init__(self, md, max_num_seqs, max_seq_len, k=None, log=None):
        from models.demos.blackhole.hy4_preview.tt import hy4_pool_decode as pool_decode
        from models.demos.blackhole.hy4_preview.tt.hy4_attn_opt import Opt
        from models.demos.blackhole.hy4_preview.tt.hy4_ccl_safe import hy4_fabric_selftest
        from models.demos.blackhole.hy4_preview.tt.hy4_chain import Hy4Chain, dram_view
        from models.demos.blackhole.hy4_preview.tt.hy4_pool_model import Hy4PoolDecoder
        from models.demos.blackhole.hy4_preview.tt.hy4_pool_prefill_model import Hy4PoolPrefill
        from models.demos.gpt_oss.config import mesh_4x8
        from models.demos.gpt_oss.tt.ccl import CCLManager

        assert (
            os.environ.get("HY4_POOL", "0") == "1" and os.environ.get("HY4_POOL_PF", "0") == "1"
        ), "the product engine needs HY4_POOL=1 HY4_POOL_PF=1 HY4_KV_FMT=scaled_fp8 (direct-to-pool prefill)"
        self.md, self.max_num_seqs, self.max_seq_len = md, max_num_seqs, max_seq_len
        self.K = int(os.environ.get("HY4_SPEC_K", "3")) if k is None else int(k)
        self.spec = self.K > 0
        self.sampled = self.spec and os.environ.get("HY4_SPEC_SAMPLED", "1") == "1"
        self.S = self.K + 1 if self.spec else 1
        self.pf_rows = int(os.environ.get("HY4_SERVE_PF_ROWS", "512"))
        ckpt = os.environ.get("HY4_CKPT", "/ckpt")
        nl = int(os.environ.get("HY4_SERVE_LAYERS", "78"))
        t0 = time.time()
        links = hy4_fabric_selftest(md)
        pool_decode._POOL_AG_LINKS = links.pool_ag
        logger.info(
            f"Hy4 fabric self-test: num_links={links.num_links} pool_ag={links.pool_ag} ({links.seconds:.1f} s)"
        )
        mc, ccl = mesh_4x8(), CCLManager(md, num_links=links.num_links, topology=ttnn.Topology.Ring)
        self.ch = Hy4Chain(md, mc, ccl, ckpt=ckpt)
        self.ch.add_layers(0, nl, log=lambda s: logger.info(s))
        if self.spec:
            self.ch.add_layers(78, 79, log=lambda s: logger.info(s))
        os.environ["HY4_HEAD_FP32"] = "1"  # fp32 logits out of the lm_head (D12-1)
        self.ch.build_head()
        self.vocab = int(self.ch.cfg.vocab_size)
        dv = dram_view(md)
        logger.info(f"Hy4 model built: {nl} layers (+MTP {self.spec}), {time.time() - t0:.0f} s, dram {dv}")
        # pool pages per mesh row: the model length + the verify rows beyond the committed length (K + 1) + the pad rows of a lock-step prefill chunk, rounded to the prefill chunk
        need = max_seq_len + self.K + 2
        self.pages = (-(-need // self.pf_rows)) * self.pf_rows // PAGE
        if os.environ.get("HY4_IDX_BLK", "0") == "1" and os.environ.get("HY4_B37_PAGES8", "1") == "1":
            # B37: the blocked indexer score layout needs n_local = pages * 64 to be a multiple of 32 * B (B = 16): pages % 8 == 0 (2049 pages at the 1M pool broke it: block width 8196, see RESULT_B37)
            self.pages = -(-self.pages // 8) * 8
        pool_bytes = (
            self.pages * 3_750_912 // 8
        )  # per bank: 78 (+MTP) kv pools + 22 key pools, 3,750,912 B per page per chip (CAPACITY_MEASURED.md)
        if pool_bytes > dv["free_per_bank"] - (96 << 20):
            raise RuntimeError(
                f"pool of {self.pages} pages/row needs {pool_bytes >> 20} MiB/bank, only {dv['free_per_bank'] >> 20} MiB/bank free after the weights: lower --max-model-len"
            )
        self.dec = Hy4PoolDecoder(
            self.ch, self.pages, mode="topk", S=self.S, bucket_tokens=self.pages * PAGE
        )  # S = 1 for plain
        self.pp = Hy4PoolPrefill(self.dec, pf_rows=self.pf_rows)
        self.runner = self.dl = self.mtp = None
        if self.spec:
            from models.demos.blackhole.hy4_preview.tt.hy4_mtp import Hy4MTP, greedy_tokens
            from models.demos.blackhole.hy4_preview.tt.hy4_spec_trace import Hy4SpecRunner

            ly = self.ch.mtp_layers[0]
            self.mtp = Hy4MTP(self.ch, ly, self.ch.rd.layer_sd(-1), Opt.from_env(), moe_fn=None)
            smp = None
            if self.sampled:
                from models.demos.blackhole.hy4_preview.tt.hy4_sampler import Hy4DeviceSampler
                from models.demos.blackhole.hy4_preview.tt.hy4_spec_sample import SpecSampling

                smp = SpecSampling(Hy4DeviceSampler(md, mc, ccl, users_per_row=self.S), self.K, invT=1.0, seed=0)
            self.runner = Hy4SpecRunner(
                self.dec, self.mtp, self.K, lambda lg: greedy_tokens(self.ch, lg), self.pages, sampling=smp
            )
            self.runner.default_variant = "G"
            self.samp = smp
        else:
            from models.demos.blackhole.hy4_preview.tt.hy4_device_loop import DeviceLoop

            self.dl = DeviceLoop(self.dec)
            self.loop_variants = os.environ.get("HY4_LOOP_VARIANTS", "G,Fp,F").split(",")
        dv = dram_view(md)
        logger.info(
            f"Hy4 pool built: pages/row={self.pages} (max_model_len {max_seq_len}), spec K={self.K} sampled={self.sampled}, pf_rows={self.pf_rows}, dram {dv}"
        )
        # host mirror of the per-row decode state (spec)
        K = max(self.K, 1)
        self.pend = [PAD] * ROWS
        self.pos = [0] * ROWS
        self.drafts = [[-1] * K for _ in range(ROWS)]
        self.live = [False] * ROWS
        self.dirty = True  # the device state differs from the mirror: ``load`` before the next round
        self.q_valid = False  # the device q rows / draft column describe the drafts of the mirror (sampled trace)
        self.params_dirty = True
        self.temp = [0.0] * ROWS
        self.seeds = [0] * ROWS
        self.rounds = 0
        self.t_round = 0.0
        self.warm_done = False
        # round pipelining (HY4_SPEC_PIPE=1): the device state of a round is resident, so round N+1 is submitted right behind the read-back of round N (speculatively: same caps / variant / params, no transition).
        # The host work between two ``round`` calls (plugin accept walk, proposal, scheduling, streaming) then overlaps the device time of the next round. A mismatch (finished request, admission, other caps / variant,
        # mirror != plugin block) drains the speculative round, discards it and reloads the mirror (the junk KV rows it wrote lie at positions >= the pending position and are overwritten before being read).
        self.pipe = os.environ.get("HY4_SPEC_PIPE", "0") == "1"
        self.inflight = None
        self.pipe_idle = (
            os.environ.get("HY4_SPEC_PIPE_IDLE", "1") == "1"
        )  # speculate also with idle (parked) rows: their device state drifts instead of being re-parked every round
        self.pipe_discards = 0
        self.pipe_hits = 0
        self.pipe_resync = int(os.environ.get("HY4_SPEC_PIPE_RESYNC", "256"))
        self._since_load = 0
        self.tl = [] if os.environ.get("HY4_ROUND_TL", "0") == "1" else None

    def drain(self):
        """wait for an in-flight speculative round and drop it (state on device advanced past the mirror: reload before the next round). Called before anything else touches the device (prefill, seed)."""
        if self.inflight is not None:
            ttnn.synchronize_device(self.md)
            self.inflight = None
            self.dirty = True
            self.q_valid = False
            self.params_dirty = True
            self.pipe_discards += 1

    # ------------------------------------------------------------------------------------------------ prefill
    def prefill(self, prompts_by_row, sampling=None):
        """prompts_by_row {mesh row: list[int]} -> {row: first token}. ``sampling`` {row: (temperature, top_k, top_p, seed)}. Direct-to-pool chunked prefill (+ MTP prompt KV + seed round in spec mode).
        The other mesh rows keep their pool pages (masked writes) and their decode state."""
        t0 = time.perf_counter()
        self.drain()
        rows = sorted(prompts_by_row)
        prompts = [prompts_by_row.get(r, [PAD]) for r in range(ROWS)]
        res = self.pp.prefill(
            prompts, keep_hidden=self.spec, keep_logits=True, mtp=self.mtp if self.spec else None, keep_rows=rows
        )
        first = {}
        for r in rows:
            sp = (sampling or {}).get(r, (0.0, -1, 1.0, 0))
            first[r] = _Seed.sample(res.logits[r], sp[0], sp[1], sp[2], sp[3], len(prompts_by_row[r]))
        t1 = time.perf_counter()
        if self.spec:
            self._seed_rows(rows, first, res, prompts_by_row, sampling or {})
        logger.info(
            f"HY4 prefill call: rows={rows} lens={[len(prompts_by_row[r]) for r in rows]} chunks={len(res.chunk_sec)} prefill={t1 - t0:.2f}s "
            f"(chunk median {sorted(res.chunk_sec)[len(res.chunk_sec) // 2]:.2f}s) chunks_s={[round(x, 1) for x in res.chunk_sec[:6]]}..{[round(x, 1) for x in res.chunk_sec[-2:]]} seed={time.perf_counter() - t1:.2f}s"
        )
        return first

    def _upload_rows(self, host, dt, lay, tgt):
        mp = ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=tuple(self.md.shape))
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=dt, layout=lay, mesh_mapper=mp), tgt)

    def _seed_rows(self, rows, first, res, prompts_by_row, sampling):
        """MTP seed round for the new rows: pair (h_{L-1}, x_L) at position L-1 gives the first K drafts (greedy). Live rows of the batch are protected: their state is re-loaded from the mirror afterwards and the junk
        the seed run writes lies at positions >= their pending position (overwritten by their next round before being read).
        """
        from models.demos.blackhole.hy4_preview.tt.hy4_chain import HID
        from models.demos.blackhole.hy4_preview.tt.hy4_common import upload

        md, rn, S, K = self.md, self.runner, self.S, self.K
        pend, pos = list(self.pend), list(self.pos)
        for r in range(ROWS):
            if r in rows:
                pend[r], pos[r] = first[r], len(prompts_by_row[r]) - 1
            elif not self.live[r]:
                pend[r], pos[r] = PAD, 0
        rn.load(pend, pos, [[-1] * K for _ in range(ROWS)], cap=[0] * ROWS)
        rn.set_park([0] * ROWS)  # the seed tail must not park the rows being seeded
        a_seed = ttnn.from_torch(
            torch.tensor([[pend[r]] * S for r in range(ROWS)], dtype=torch.int32).reshape(ROWS * S, 1),
            device=md,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape)),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        h = torch.zeros(ROWS, HID)
        for r in rows:
            h[r] = res.hidden[r]
        h_seed = upload(
            md, h.reshape(ROWS, 1, 1, HID).repeat(1, 1, S, 1).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0
        )
        rn._set_variant("G")  # the seed draft is greedy (the last capture left the sampled variant armed)
        rn.sd.derive(rn.tail.pos)
        rn.tail.run(a_seed, h_seed, derived=True)
        ttnn.synchronize_device(md)
        ttnn.deallocate(a_seed)
        ttnn.deallocate(h_seed)
        st = rn.read_state()
        for r in rows:
            self.pend[r], self.pos[r] = st[r]["tok"][0], st[r]["pos"]
            self.drafts[r] = list(st[r]["drafts"])
            self.live[r] = True
            sp = sampling.get(r, (0.0, -1, 1.0, 0))
            self.temp[r], self.seeds[r] = float(sp[0] or 0.0), int(sp[3]) & 0x7FFFFFFF
            assert self.pend[r] == first[r], (r, self.pend[r], first[r])
        self.dirty = True  # live rows' device state was overwritten by the seed run: reload the mirror
        self.q_valid = False
        self.params_dirty = True

    def release(self, row):
        """request on this mesh row finished: park the row (pos 0, pad token) at the next round."""
        self.live[row] = False
        self.pend[row], self.pos[row] = PAD, 0
        self.drafts[row] = [-1] * max(self.K, 1)
        self.dirty = True

    # ------------------------------------------------------------------------------------------------ spec rounds
    def round(self, caps, want_sampled, temps=None, seeds=None, plugin_block=None):
        """one speculative round for all four mesh rows. caps[r] = max drafts that may be accepted (0 for idle / draftless rows); want_sampled -> the lossless sampled trace (temps / seeds per row);
        plugin_block {row: (tokens [S], pos)} overrides the mirror when the plugin's candidate block differs. Returns per row dict(a, m, pos, drafts, tok).
        """
        rn = self.runner
        tA = time.perf_counter()
        pf = os.environ.get("HY4_SPEC_PIPE_FILE")
        if pf and self.rounds % 16 == 0:  # live A/B switch (a file holding 0 / 1) without a reboot
            try:
                with open(pf) as fh:
                    want = fh.read().strip() == "1"
                if want != self.pipe:
                    self.drain()
                    self.pipe = want
            except OSError:
                pass
        for r in range(ROWS):
            if not self.live[r]:
                caps[r] = 0
        idle = not all(self.live)
        if plugin_block:
            for r, (tk, p) in plugin_block.items():
                dr = [int(x) for x in tk[1 : 1 + self.K]]
                c = caps[r]
                if self.pend[r] != int(tk[0]) or self.pos[r] != int(p) or self.drafts[r][:c] != dr[:c]:
                    logger.warning(
                        f"HY4 round: row {r} mirror (tok {self.pend[r]}, pos {self.pos[r]}, drafts {self.drafts[r]}) != plugin block (tok {int(tk[0])}, pos {int(p)}, drafts {dr}): reload from the plugin's block"
                    )
                    self.pend[r], self.pos[r] = int(tk[0]), int(p)
                    self.drafts[r] = dr[:c] + [-1] * (self.K - c)
                    self.dirty = True
                    self.q_valid = False
        v = "S" if (want_sampled and self.sampled) else "G"
        park = tuple(0.0 if lv else 1.0 for lv in self.live)
        key = (
            park,
            tuple(caps),
            v,
            tuple(temps) if (v == "S" and temps is not None) else None,
            tuple(seeds) if (v == "S" and seeds is not None) else None,
        )
        hit = self.inflight is not None and not self.dirty and self.inflight == key
        if self.inflight is not None and not hit:
            self.drain()
        tB = time.perf_counter()
        if hit:
            self.pipe_hits += 1
        else:
            if self.dirty or (idle and not self.pipe) or (self.pipe and self._since_load >= self.pipe_resync):
                rn.load(self.pend, self.pos, self.drafts, cap=caps)
                self.dirty = False
                self._since_load = 0
                if idle:
                    for r in range(ROWS):
                        if not self.live[r]:
                            self.pend[r], self.pos[r] = PAD, 0
            else:
                rn.set_cap(caps)
            rn.set_park(park)
            if v == "S":
                if temps is not None:
                    new = (list(temps), list(seeds))
                    if new != (self.temp, self.seeds) or self.params_dirty:
                        self.temp, self.seeds = new
                        self.params_dirty = True
                if self.params_dirty:
                    self.samp.set_params([1e4 if t <= 0 else 1.0 / t for t in self.temp], self.seeds)
                    self.params_dirty = False
                if not self.q_valid:
                    self.samp.load_onehot(self.drafts)
                    self.q_valid = True
            else:
                self.q_valid = False
            rn.submit(v)
        tC = time.perf_counter()
        rd = rn.read_enqueue()  # behind the (possibly speculative) round, before the next one
        self.inflight = None
        if self.pipe and self._since_load < self.pipe_resync and (not idle or self.pipe_idle):
            rn.submit(v)  # speculative next round: same caps / variant / params
            self.inflight = key
            self._since_load += 1
        tD = time.perf_counter()
        st = rn.read_finish(rd)
        tE = time.perf_counter()
        self.t_round += tE - tB
        self.rounds += 1
        if self.tl is not None:
            self.tl.append(
                dict(
                    t=tA, prep=tB - tA, upload_submit=tC - tB, read_enq_spec_submit=tD - tC, wait_read=tE - tD, hit=hit
                )
            )
        for r in range(ROWS):
            if self.live[r]:
                self.pend[r], self.pos[r] = st[r]["tok"][0], st[r]["pos"]
                self.drafts[r] = list(st[r]["drafts"])
        if idle and not self.pipe:
            for r in range(ROWS):
                if not self.live[r]:
                    self.pend[r], self.pos[r] = PAD, 0
        return st

    # ------------------------------------------------------------------------------------------------ warm-up (before any request: all persistent tensors exist before the traces)
    def _dram(self, tag):
        from models.demos.blackhole.hy4_preview.tt.hy4_chain import dram_view

        v = dram_view(self.md)
        logger.info(
            f"HY4 dram[{tag}]: alloc {v['alloc_per_bank'] / 1e6:.0f} MB/bank, free {v['free_per_bank'] / 1e6:.0f} MB/bank, largest free {v['largest_free_per_bank'] / 1e6:.0f} MB"
        )

    def warm(self):
        """synthetic 4 x 3072-token prefill (compiles every PF / MTP chunk program, creates every lazily cached persistent tensor: PF constants, dense-id tables) -> seed -> eager round -> trace capture(s)."""
        if self.warm_done:
            return
        self.warm_done = True
        t0 = time.time()
        g = torch.Generator().manual_seed(0)
        n = min(3072, self.max_seq_len - 8)
        prompts = {r: torch.randint(1000, 100000, (n,), generator=g).tolist() for r in range(ROWS)}
        self._dram("before warm prefill")
        first = self.prefill(prompts)
        self._dram("after warm prefill")
        t1 = time.time()
        if self.spec:
            # D75-1: EVERY lazily created persistent tensor of EVERY variant must exist before the FIRST capture (one allocated between two captures can sit in the first trace's scratch and is clobbered by its
            # replays): run both round bodies eagerly first, then capture. (p4 / p5: the first sampled round hung at 78 layers after greedy rounds had replayed.)
            self.runner.body("G")
            ttnn.synchronize_device(self.md)
            if self.sampled:
                self.runner.body("S")
                ttnn.synchronize_device(self.md)
            self._dram("after eager bodies")
            self.runner.capture("G")
            self._dram("after G capture")
            if self.sampled:
                self.runner.capture("S")
                self._dram("after S capture")
            logger.info(f"HY4 warm-up: spec traces captured ({list(self.runner.traces)})")
        else:
            self.dl.capture(self.loop_variants)
        for r in range(ROWS):
            self.release(r)
        self.pp.set_keep(None)
        from models.demos.blackhole.hy4_preview.tt.hy4_chain import dram_view

        logger.info(
            f"HY4 warm-up: prefill 4 x {n} {t1 - t0:.1f} s (compile incl.), captures {time.time() - t1:.1f} s, dram {dram_view(self.md)}; first tokens {first}"
        )
