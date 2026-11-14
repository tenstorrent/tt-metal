# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM (tt-vllm-plugin) adapter of Hy4-Preview on the 4x8 Blackhole galaxy, v1 (task serve-long): the INTEG model (``Hy4PoolDecoder``: SCALED_FP8 sequence-sharded pool, split-kernel sparse
decode with the decode indexer, fused iHC, XMerge, skinny kernels) instead of the 2048-key chain decoder of v0 (serve-hy4).

Registered with the plugin as ``TTHYV4ForCausalLM`` through ``../vllm_metadata.json`` (``EXTRA_MODELS_DIR`` = ``models/demos/blackhole``). Slot protocol / host-sampling return convention
as v0 and as the DSV4.1 adapter. Flags are read from the environment of the server process (integ/INTERFACE.md): HY4_POOL=1 HY4_KV_FMT=scaled_fp8 HY4_OPT=all HY4_IHC_FUSED=1
HY4_FUSE=sparse,xchg HY4_XM_NSPLIT=8 HY4_SKINNY=... HY4_PF_OPT=all,moeall MOE_RING_FAST=1.

CONTRACT (every limit explicit)
  * Context: ``max_model_len`` up to 1,048,576 tokens per request (pool pages per mesh row = ceil(max_model_len / 512); one user per mesh row, identity page table). Decode = ONE traced
    step (top-k sparse mode, bucket = the whole pool); the ids of rows with pos < 2048 are filtered on device, so short and long contexts share the trace.
  * PROMPT CAP: a prompt is prefilled by the chain prefill, whose per-layer SCALED_FP8 cache is REPLICATED on the 8 chips of the mesh row (60 KB per token per chip) and lives until the hand-off
    into the pool. The pool + decode trace leave a fixed DRAM budget, so prompts are limited to ``prefill_cap`` tokens (printed at boot, measured by a synthetic 4 x cap warm-up prefill; override
    HY4_SERVE_PF_CAP). Longer prompts raise TTRequestError. Generation continues in the pool up to max_model_len. A direct-to-pool prefill (needed for ~1M-token prompts) is NOT part of this adapter.
  * Batch: max_num_seqs <= 4, one user per mesh row; a free row computes a dummy token at position 0 (garbage written to its own pool page 0 only).
  * Prefill: whole prompts per call (no chunked prefill / no interleaving), chunks of 512 rows on all four mesh rows (new prompts on their physical rows, PAD elsewhere), then a MASKED hand-off
    (``Hy4ServePool.handoff_rows``) so the live decode rows keep their pool pages.
  * Sampling: HOST (vLLM sampler on the fp32 logits [W,1,V]); ``supports_sample_on_device`` False (device sampling branch not merged: its RESULT was not pass at launch). No spec decode, no APC.
"""
import os
import time

import torch
from loguru import logger

PAD_TOKEN = 120002  # <｜hy_pad:opensource｜>
ROWS = 4
CHUNK = 512
MAX_MODEL_LEN = 1048576
PAGE = 512

try:
    from vllm_tt_plugin.platform import TTRequestError
except Exception:  # noqa: BLE001  (demo / unit tests without the plugin)

    class TTRequestError(ValueError):
        pass


class _Slots:
    """logical slot (the plugin's state slot) -> physical mesh row. ``remap`` applies the plugin's permutation: new slot i reads old slot remap[i]."""

    def __init__(self, n):
        self.n = n
        self.phys = list(range(n))

    def remap(self, remap):
        r = [int(x) for x in remap]
        assert sorted(r) == list(range(self.n)), r
        self.phys = [self.phys[x] for x in r]


class HYV4ForCausalLM:
    """tt-vllm-plugin model class (see the module docstring)."""

    model_capabilities = {
        "supports_prefix_caching": False,  # APC off for this task
        "supports_chunked_prefill": False,  # whole-prompt prefill calls (v0)
        "prefill_chunk_alignment": CHUNK,  # forced to 0 by the platform while chunked prefill is off; kept for the integ track
        "supports_async_decode": False,
        "supports_sample_on_device": False,  # host sampling of the fp32 logits (TODO: in-trace sampler)
        "supports_spec_decode": False,
        "fabric_config": None,  # opened by additional_config tt.fabric_config (FABRIC_1D_RING)
    }

    # ---- construction ------------------------------------------------------------------------------------------------------------------------
    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None, **kwargs
    ):
        mesh = tuple(mesh_device.shape)
        if mesh != (ROWS, 8):
            raise TTRequestError(f"Hy4-Preview needs the (4, 8) Blackhole galaxy mesh, got {mesh}")
        if tt_data_parallel != 1:
            raise TTRequestError("Hy4-Preview does not support data parallelism")
        if max_batch_size > ROWS:
            raise TTRequestError(
                f"Hy4 serving v0 supports max_num_seqs <= {ROWS} (one user per mesh row), got {max_batch_size}"
            )
        if max_seq_len > MAX_MODEL_LEN or max_seq_len < 1:
            raise TTRequestError(f"Hy4 serving supports max_model_len <= {MAX_MODEL_LEN}, got {max_seq_len}")
        return cls(mesh_device, max_batch_size, max_seq_len)

    def __init__(
        self, mesh_device, max_num_seqs, max_seq_len, *, vllm_config=None
    ):  # vllm_config: vLLM's text-generation protocol check wants the keyword
        import ttnn
        from models.demos.blackhole.hy4_preview.tt import hy4_pool_decode as pool_decode
        from models.demos.blackhole.hy4_preview.tt.hy4_ccl_safe import hy4_fabric_selftest
        from models.demos.blackhole.hy4_preview.tt.hy4_chain import Hy4Chain, dram_view
        from models.demos.blackhole.hy4_preview.tt.hy4_serve_pool import Hy4ServePool
        from models.demos.gpt_oss.config import mesh_4x8
        from models.demos.gpt_oss.tt.ccl import CCLManager

        self.ttnn = ttnn
        self.md = mesh_device
        self.max_num_seqs, self.max_seq_len = max_num_seqs, max_seq_len
        ckpt = os.environ.get("HY4_CKPT", "/ckpt")
        nl = int(os.environ.get("HY4_SERVE_LAYERS", "78"))  # < 78 only for adapter debugging
        t0 = time.time()
        links = hy4_fabric_selftest(mesh_device)  # integ INTERFACE.md: before the CCLManager and the pool context
        pool_decode._POOL_AG_LINKS = links.pool_ag
        logger.info(
            f"Hy4 fabric self-test: num_links={links.num_links} pool_ag={links.pool_ag} per_op={links.per_op} ({links.seconds:.1f} s)"
        )
        mc, ccl = mesh_4x8(), CCLManager(mesh_device, num_links=links.num_links, topology=ttnn.Topology.Ring)
        self.ch = Hy4Chain(mesh_device, mc, ccl, ckpt=ckpt)
        self.ch.add_layers(0, nl, log=lambda s: logger.info(s))
        if os.environ.get("HY4_SERVE_MTP", "0") == "1":
            self.ch.add_layers(78, 79, log=lambda s: logger.info(s))
        self.ch.build_head()
        self.vocab = int(self.ch.cfg.vocab_size)
        dv = dram_view(mesh_device)
        logger.info(f"Hy4 model built: {nl} layers, {time.time() - t0:.0f} s, dram {dv}")
        self.pages = -(-max_seq_len // PAGE)
        pool_bytes = (
            self.pages * 3_750_912 // 8
        )  # per bank: 79 kv pools + 22 key pools, 3,750,912 B per page per chip (CAPACITY_MEASURED.md)
        if pool_bytes > dv["free_per_bank"] - (96 << 20):
            raise TTRequestError(
                f"pool of {self.pages} pages/row needs {pool_bytes >> 20} MiB/bank, only {dv['free_per_bank'] >> 20} MiB/bank free after the weights"
            )
        self.dec = Hy4ServePool(self.ch, self.pages, mode="topk", bucket_tokens=self.pages * PAGE)
        self.pf_cap = 0  # set by warm_serving (measured)
        dv = dram_view(mesh_device)
        logger.info(
            f"Hy4 pool built: pages/row={self.pages} (max_model_len {max_seq_len}), bucket={self.dec.ctx.bucket_tokens}, dram {dv}"
        )
        self.slots = _Slots(max_num_seqs)
        self.warm = False
        self.steps = 0
        self.live = set()

    # vLLM inspects this protocol (``is_text_generation_model``) while building the ModelConfig, BEFORE the plugin loads the model.
    def embed_input_ids(self, input_ids):
        raise NotImplementedError("Use the TT plugin prefill_forward/decode_forward interface")

    def forward(self, input_ids, positions):
        raise NotImplementedError("Use the TT plugin prefill_forward/decode_forward interface")

    def compute_logits(self, hidden_states):
        raise NotImplementedError("The Hy4 generator owns the LM head")

    @classmethod
    def get_max_tokens_all_users(
        cls, model_name="", num_devices=1, tt_data_parallel=1, max_model_len=None, max_num_seqs=None, **kwargs
    ):
        return int(max_model_len or MAX_MODEL_LEN) * int(max_num_seqs or 1)

    # ---- KV cache (accounting only) ---------------------------------------------------------------------------------------------------------
    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        logger.info(
            f"Hy4 vLLM: kv_cache_shape {tuple(kv_cache_shape)} x {num_layers} layers is accounting only; the model owns the SCALED_FP8 pool "
            f"({self.pages} pages of {PAGE} tokens per mesh row)"
        )
        self.kv_cache = [(torch.empty(0), torch.empty(0)) for _ in range(num_layers)]
        return self.kv_cache

    # ---- prefill -----------------------------------------------------------------------------------------------------------------------------
    def _last_logits(self, x4, local_pos_by_phys):
        """x4 [1,4,512,D] of one chunk -> {phys row: fp32 logits [V]} for the rows whose last prompt token sits at local position p of this chunk."""
        ttnn = self.ttnn
        out = {}
        for phys, p in local_pos_by_phys.items():
            xs = ttnn.slice(x4, [0, 0, p, 0], [1, x4.shape[1], p + 1, x4.shape[3]])
            hn, lg = self.ch.head(xs)
            ttnn.deallocate(xs)
            ttnn.deallocate(hn)
            t = ttnn.to_torch(
                lg, mesh_composer=ttnn.ConcatMesh2dToTensor(self.md, dims=(0, 3), mesh_shape=tuple(self.md.shape))
            )
            ttnn.deallocate(lg)
            out[phys] = t.reshape(ROWS, -1)[phys].float()
        return out

    def _prefill_rows(self, prompts_by_phys):
        """prompts_by_phys {phys row: list[int] tokens} -> ({phys: logits[V]}, seconds). Leaves the decode caches of the OTHER rows untouched."""
        ttnn = self.ttnn
        lens = {p: len(t) for p, t in prompts_by_phys.items()}
        nch = max(-(-n // CHUNK) for n in lens.values())
        ids = torch.full((ROWS, nch * CHUNK), PAD_TOKEN, dtype=torch.long)
        for p, t in prompts_by_phys.items():
            ids[p, : len(t)] = torch.tensor(t, dtype=torch.long)
        logits = {}
        self.ch.reset()
        t0 = time.perf_counter()
        for c in range(nch):
            x4, ids_by = self.ch.prefill(ids[:, c * CHUNK : (c + 1) * CHUNK], start=c * CHUNK)
            fin = {p: (n - 1) - c * CHUNK for p, n in lens.items() if c * CHUNK <= n - 1 < (c + 1) * CHUNK}
            if fin:
                logits.update(self._last_logits(x4, fin))
            ttnn.deallocate(x4)
            for v in ids_by.values():
                ttnn.deallocate(v)
            ttnn.synchronize_device(self.md)
        t_pf = time.perf_counter() - t0
        t1 = time.perf_counter()
        self.dec.handoff_rows(nch * CHUNK, sorted(prompts_by_phys))
        self.ch.reset()  # free the concat-grown prefill caches and key caches
        ttnn.synchronize_device(self.md)
        logger.info(
            f"HY4 prefill call: rows={len(prompts_by_phys)} lens={sorted(lens.values())} chunks={nch} total={time.perf_counter() - t0:.2f}s "
            f"chunks_s={t_pf:.2f} handoff_s={time.perf_counter() - t1:.2f}"
        )
        return logits

    def prefill_forward(
        self,
        tokens,
        start_pos=None,
        prompt_lens=None,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        empty_slots=None,
        sampling_params=None,
        **kwargs,
    ):
        """tokens [B, T] (prompt from position 0), prompt_lens [B] = prompt length, empty_slots [B] = destination logical slot. Returns fp32 logits [B, 1, V] of the last prompt token."""
        B = int(tokens.shape[0])
        starts = [0] * B if start_pos is None else [int(x) for x in start_pos]
        lens = [int(tokens.shape[1])] * B if prompt_lens is None else [int(x) for x in prompt_lens]
        if any(s != 0 for s in starts):
            raise TTRequestError(
                "Hy4 serving v0 prefills whole prompts (start_pos must be 0: no chunked prefill, no prefix cache)"
            )
        if any(n < 1 for n in lens) or any(n > self.max_seq_len for n in lens):
            raise TTRequestError(f"prompt length {max(lens)} outside 1..max_model_len {self.max_seq_len}")
        if self.pf_cap and any(n > self.pf_cap for n in lens):
            raise TTRequestError(
                f"prompt length {max(lens)} exceeds the {self.pf_cap}-token prefill cap of this server (the chain prefill cache is replicated per mesh row and must fit next to the {self.pages}-page pool)"
            )
        slots = list(range(B)) if empty_slots is None else [int(s) for s in empty_slots]
        if len(set(slots)) != B:
            raise RuntimeError(f"duplicate prefill slots {slots}")
        phys = [self.slots.phys[s] for s in slots]
        prompts = {p: [int(x) for x in tokens[i, : lens[i]].tolist()] for i, p in enumerate(phys)}
        lg = self._prefill_rows(prompts)
        out = torch.zeros(B, 1, self.vocab, dtype=torch.float32)
        for i, p in enumerate(phys):
            out[i, 0] = lg[p]
        return out

    # ---- decode ------------------------------------------------------------------------------------------------------------------------------
    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        slot_remap=None,
        **kwargs,
    ):
        """tokens [W, 1], start_pos [W] (position of the fed token, -1 = padding row), slot_remap [max_num_seqs] optional. Returns fp32 logits [W, 1, V]."""
        if slot_remap is not None:
            self.slots.remap(slot_remap)
        W = int(tokens.shape[0])
        tok = [PAD_TOKEN] * ROWS
        pos = [0] * ROWS
        rows = []
        for i in range(W):
            p = int(start_pos[i])
            if p < 0:
                continue
            if p >= self.pages * PAGE:
                raise TTRequestError(f"position {p} beyond the {self.pages * PAGE} tokens of the Hy4 pool")
            ph = self.slots.phys[i]
            tok[ph], pos[ph] = int(tokens[i, 0]), p
            rows.append((i, ph))
        t0 = time.perf_counter()
        self.dec.set_inputs(tok, pos)
        out = self.dec.run()
        _, lgs = self.dec.read_argmax(out)
        res = torch.zeros(W, 1, self.vocab, dtype=torch.float32)
        for i, ph in rows:
            res[i, 0] = lgs[ph]
        self.steps += 1
        if self.steps % 50 == 1:
            logger.info(f"HY4 decode step {self.steps}: rows={len(rows)} ms={(time.perf_counter() - t0) * 1e3:.1f}")
        return res

    def release_request(self, slot):
        """Nothing to free: the cache rows of a mesh row are rewritten by the next prefill into it."""
        return

    # ---- warm-up -----------------------------------------------------------------------------------------------------------------------------
    def warm_serving(self):
        """Allocate everything and compile everything: short synthetic prefill + hand-off, decode trace capture, then a synthetic 4 x pf_cap prefill (compiles every chunk shape and PROVES the cap fits).
        Nothing persistent is allocated afterwards."""
        if self.warm:
            return
        self.warm = True
        from models.demos.blackhole.hy4_preview.tt.hy4_chain import dram_view

        t0 = time.perf_counter()
        g = torch.Generator().manual_seed(0)
        mk = lambda n: {p: torch.randint(1000, 100000, (n,), generator=g).tolist() for p in range(ROWS)}
        n0 = min(1024, self.pages * PAGE)
        self._prefill_rows(mk(n0))
        t1 = time.perf_counter()
        self.dec.set_inputs([PAD_TOKEN] * ROWS, [n0] * ROWS)
        self.dec.capture()
        t2 = time.perf_counter()
        dv = dram_view(self.md)
        logger.info(
            f"HY4 warm-up: short prefill {t1 - t0:.1f} s (compile incl.), decode trace capture {t2 - t1:.1f} s, dram after trace {dv}"
        )
        # prefill cap from the free DRAM left after pool + trace: 60.3 KB per token per chip, minus a reserve for the chunk transients
        reserve = float(os.environ.get("HY4_SERVE_PF_RESERVE_GIB", "1.5")) * (1 << 30)
        free = dv["free_per_bank"] * 8
        cap = int(max(0, free - reserve) // 60_300 // PAGE * PAGE)
        cap = int(os.environ.get("HY4_SERVE_PF_CAP", str(cap)))
        cap = max(PAGE, min(cap, self.max_seq_len, self.pages * PAGE))
        logger.info(
            f"HY4 prefill cap: {cap} tokens (free {free / (1 << 30):.2f} GiB per chip, reserve {reserve / (1 << 30):.2f} GiB)"
        )
        t3 = time.perf_counter()
        self._prefill_rows(mk(cap))
        logger.info(
            f"HY4 warm-up: synthetic 4 x {cap} prefill {time.perf_counter() - t3:.1f} s, dram after {dram_view(self.md)}"
        )
        self.pf_cap = cap
        # decode steps on the warm pool: trace sanity and the steady step time
        ts = []
        for i in range(6):
            self.dec.set_inputs([PAD_TOKEN] * ROWS, [cap] * ROWS)
            t4 = time.perf_counter()
            self.dec.read_argmax(self.dec.run())
            ts.append(time.perf_counter() - t4)
        logger.info(
            f"HY4 warm-up: decode step (host loop, readback incl.) {sorted(ts)[len(ts) // 2] * 1e3:.1f} ms; total warm {time.perf_counter() - t0:.0f} s"
        )
        self.slots = _Slots(self.max_num_seqs)

    def warmup_model_prefill(self, kv_cache=None, enable_trace=True, *args, **kwargs):
        self.warm_serving()

    def warmup_model_decode(self, *args, **kwargs):
        self.warm_serving()
