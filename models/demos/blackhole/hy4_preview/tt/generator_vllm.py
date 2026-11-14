# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM (tt-vllm-plugin) adapter of Hy4-Preview on the 4x8 Blackhole galaxy: TIER 0 serving (early quality signal), built on the chain model (``hy4_chain`` + ``hy4_decode``).

Registered with the plugin as ``TTHYV4ForCausalLM`` through ``../vllm_metadata.json`` (``arch`` HYV4ForCausalLM; the plugin prefixes ``TT``; ``EXTRA_MODELS_DIR`` must point at
``models/demos/blackhole``). Contract: vllm-tt-plugin ``docs/MODEL_*.md`` (a8962b0). Lifted from the DSV4.1 adapter (``deepseek_v41_flash/tt/generator_vllm.py``): same slot protocol,
same host-sampling return convention, none of its machinery (no APC, no spec decode, no interleaved prefill, no device sampler).

SCOPE (v0, deliberately small, every limit explicit)
  * Context: <= ``HY4_SERVE_TCAP`` (default and maximum 2048) KEYS per request, prompt + generated: the chain decode path (``hy4_decode.Hy4Decoder``) is dense attention over a persistent
    TILE bf16 cache of Tcap rows per layer (ids = arange(pos+1) built on the host, stock sparse_sdpa; no decode indexer / top-k, no pool, no SCALED_FP8 rows). ``max_model_len`` MUST be
    <= 2048 (checked at load). GSM8K chain-of-thought fits, GPQA thinking-ON (tens of thousands of tokens) does NOT.
  * Batch: max_num_seqs <= 4. The decode trace always runs 4 users, ONE PER MESH ROW (a free row computes a dummy token at position 0). The mesh row of a request is its PHYSICAL slot.
  * Prefill: whole prompts in one ``prefill_forward`` call (no chunked prefill, no interleaving: ``supports_chunked_prefill`` False, so ``max_num_batched_tokens`` must be >= max_model_len).
    The call runs the chain prefill in chunks of 512 rows on all four mesh rows (the new prompts on their physical rows, PAD tokens elsewhere), reads the last-token logits of every new
    row, then hands the concat-grown prefill caches over into the persistent decode caches for the NEW rows only (a per-row blend: the decode caches of running requests are untouched).
    ``prefill_chunk_alignment`` 512 is declared for the day chunked prefill is wired (inactive while chunked prefill is off). Prefill is eager (not traced): seconds per chunk.
  * Decode: ``Hy4Decoder`` traced step (T=1 token per device, 78 layers), host loop: set inputs, replay, full-vocabulary logits read back (201 ms/step measured in the chain task).
  * Sampling: HOST. ``supports_sample_on_device`` False: the plugin runs vLLM's own sampler on the fp32 logits [B, 1, V] returned by prefill and decode (temperature / top_p / top_k /
    penalties / logprobs all work; generation_config of the checkpoint: temperature 0.9, top_p 1, top_k -1). TODO(Tier 1): in-trace sampler, device argmax.
  * KV bookkeeping: vLLM blocks are accounting only (``allocate_kv_cache`` returns placeholders; the model owns the persistent caches); ``get_max_tokens_all_users`` = max_model_len x max_num_seqs.
  * Slots: ``empty_slots`` (prefill) and ``slot_remap`` (decode) are handled on the HOST: a logical slot maps to a physical mesh row, a remap permutes that map, no device state moves.
  * No prefix caching (APC is OFF for this task), no speculative decoding (MTP layer not even loaded), no structured-output device grammar, no images (text-only model).
"""
import os
import time

import torch
from loguru import logger

PAD_TOKEN = 120002  # <｜hy_pad:opensource｜>
ROWS = 4
CHUNK = 512
TCAP_MAX = 2048

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
            raise TTRequestError(f"Hy4 serving v0 supports max_num_seqs <= {ROWS} (one user per mesh row), got {max_batch_size}")
        tcap = int(os.environ.get("HY4_SERVE_TCAP", str(TCAP_MAX)))
        if tcap % 32 or tcap > TCAP_MAX or tcap % CHUNK:
            raise TTRequestError(f"HY4_SERVE_TCAP={tcap} must be a multiple of {CHUNK} and <= {TCAP_MAX}")
        if max_seq_len > tcap:
            raise TTRequestError(
                f"Hy4 serving v0 decodes at most {tcap} keys per request (chain decode path: dense attention, no decode indexer): "
                f"max_model_len {max_seq_len} is too large; launch with --max-model-len <= {tcap}"
            )
        return cls(mesh_device, max_batch_size, max_seq_len, tcap)

    def __init__(self, mesh_device, max_num_seqs, max_seq_len, tcap, *, vllm_config=None):  # vllm_config: vLLM's text-generation protocol check wants the keyword
        from models.demos.blackhole.hy4_preview.tt.hy4_chain import Hy4Chain, dram_view
        from models.demos.blackhole.hy4_preview.tt.hy4_decode import Hy4Decoder
        from models.demos.gpt_oss.config import mesh_4x8
        from models.demos.gpt_oss.tt.ccl import CCLManager

        import ttnn

        self.ttnn = ttnn
        self.md = mesh_device
        self.max_num_seqs, self.max_seq_len, self.tcap = max_num_seqs, max_seq_len, tcap
        ckpt = os.environ.get("HY4_CKPT", "/ckpt")
        nl = int(os.environ.get("HY4_SERVE_LAYERS", "78"))  # < 78 only for adapter debugging
        t0 = time.time()
        mc, ccl = mesh_4x8(), CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Ring)
        self.ch = Hy4Chain(mesh_device, mc, ccl, ckpt=ckpt)
        self.ch.add_layers(0, nl, log=lambda s: logger.info(s))
        self.ch.build_head()
        self.vocab = int(self.ch.cfg.vocab_size)
        logger.info(f"Hy4 model built: {nl} layers, {time.time() - t0:.0f} s, dram {dram_view(mesh_device)}")
        self.dec = Hy4Decoder(self.ch, tcap)
        self.slots = _Slots(max_num_seqs)
        self.warm = False
        self.steps = 0
        self._ones = None

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
        return int(max_model_len or TCAP_MAX) * int(max_num_seqs or 1)

    # ---- KV cache (accounting only) ---------------------------------------------------------------------------------------------------------
    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        logger.info(
            f"Hy4 vLLM: kv_cache_shape {tuple(kv_cache_shape)} x {num_layers} layers is accounting only; the model owns the persistent decode caches "
            f"(Tcap {self.tcap} rows per layer and mesh row)"
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
        self.dec.load_prefill_cache_rows(sorted(prompts_by_phys))
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
            raise TTRequestError("Hy4 serving v0 prefills whole prompts (start_pos must be 0: no chunked prefill, no prefix cache)")
        if any(n > self.tcap for n in lens) or any(n < 1 for n in lens):
            raise TTRequestError(f"prompt length {max(lens)} exceeds the {self.tcap} keys the Hy4 decode path holds")
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
            if p >= self.tcap:
                raise TTRequestError(f"position {p} beyond the {self.tcap} keys of the Hy4 decode path")
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
        """Compile every prefill chunk shape (starts 0..Tcap-512) with a synthetic 4 x Tcap prompt, hand over, capture the decode trace. Allocates nothing persistent afterwards."""
        if self.warm:
            return
        self.warm = True
        t0 = time.perf_counter()
        g = torch.Generator().manual_seed(0)
        prompts = {p: torch.randint(1000, 100000, (self.tcap,), generator=g).tolist() for p in range(ROWS)}
        self._prefill_rows(prompts)
        t1 = time.perf_counter()
        self.dec.set_inputs([PAD_TOKEN] * ROWS, [self.tcap // 2] * ROWS)
        self.dec.capture()
        logger.info(
            f"HY4 warm-up: prefill {t1 - t0:.1f} s (compile incl.), decode trace capture {time.perf_counter() - t1:.1f} s; "
            f"second prefill call timing follows"
        )
        # one more (compiled) prefill call: its timing is the steady prefill cost; rows are overwritten by real requests later
        self._prefill_rows({p: prompts[p][:1024] for p in range(ROWS)})
        self.slots = _Slots(self.max_num_seqs)

    def warmup_model_prefill(self, kv_cache=None, enable_trace=True, *args, **kwargs):
        self.warm_serving()

    def warmup_model_decode(self, *args, **kwargs):
        self.warm_serving()
