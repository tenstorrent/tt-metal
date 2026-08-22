# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Readiness/serving generator for poolside/Laguna-XS-2.1 on Blackhole P150 ASIC meshes.

Implements the standard Metal readiness ``Generator`` contract
(``models.common.readiness_check.contract.Generator``): low-level ``prefill_forward`` /
``decode_forward`` (the vLLM-shared level where the caller owns cache + page table) and the
high-level ``generate`` (owns cache/page table internally, loops deterministically).

The measured token-out decode path is **fully on-device traced split sampling**:

  * one captured decode trace does token-embedding -> 40-layer stack -> final norm -> column-sharded
    LM head -> ``Sampling1D`` greedy top-k(k=1) -> writes the sampled token back into the persistent
    decode token buffer (``tt_out_tok``), and advances ``cur_pos``/``rope_idx`` on device
    (``ttnn.plus_one``). Between replays nothing is rebuilt on the host except the page table when it
    actually changes; free-running decode feeds the sampled token back with zero host token/position
    work.
  * greedy = ``Sampling1D`` top-k with k=1 over each device's local vocab shard, all-gathering only
    the D×32 candidate set (NOT the full 100352-wide vocab). This is the canonical split-sampling
    contract and avoids a full-vocab all-gather / host argmax / full-logits readback on the hot path.

An explicit **host-sampling compatibility mode** (``host_sampling=True``) gathers full logits and
argmaxes on host — used only for tests that require host sampling; it never replaces the measured
on-device traced path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional

import torch

import ttnn
from models.common.modules.sampling.sampling_1d import Sampling1D
from models.common.readiness_check.contract import Generator as ReadinessGenerator

# The readiness runner loads this file as a standalone module (not as a package member), so relative
# imports fail there — fall back to the absolute package path.
try:
    from .model import MODEL_ID, LagunaModel
except ImportError:  # loaded via importlib.spec_from_file_location
    from models.autoports.poolside_laguna_xs_2_1.tt.model import MODEL_ID, LagunaModel

BLOCK_SIZE = 32


def _replicate(mesh):
    return ttnn.ReplicateTensorToMesh(mesh)


class LagunaGenerator(ReadinessGenerator):
    def __init__(self, mesh_device, model: LagunaModel, tokenizer, *, max_seq_len, host_sampling=False):
        self.mesh_device = mesh_device
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.host_sampling = host_sampling
        self.vocab = model.cfg.vocab
        self.hidden = model.cfg.hidden

        # Greedy split-sampling params (per batch size, lazily built).
        self._samplers: dict[int, Sampling1D] = {}
        self._greedy_params: dict[int, tuple] = {}

        # Owned decode/prefill state for generate().
        self._kv_cache = None
        self._kv_users = 0
        self._kv_seq = 0
        self._page_table = None

        # Persistent traced-decode tensors + captured trace id (keyed by batch).
        self._trace = {}  # batch -> dict(tid, tok, cur_pos, rope_idx, page_table, logits_out)
        self.counters = self._zero_counters()

    @staticmethod
    def _zero_counters():
        return dict(trace_replay=0, token_refresh=0, pos_refresh=0, page_table_refresh=0, sync=0, readback=0)

    # ---- factory ----------------------------------------------------------- #
    @classmethod
    def from_pretrained(
        cls,
        mesh_device,
        *,
        max_seq_len=16384,
        num_layers=None,
        host_sampling=False,
        hf_config=None,
        precision_policy=None,
        precision_config_path=None,
    ):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = LagunaModel.from_pretrained(
            mesh_device,
            hf_config=hf_config,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            precision_policy=precision_policy,
            precision_config_path=precision_config_path,
        )
        return cls(mesh_device, model, tokenizer, max_seq_len=max_seq_len, host_sampling=host_sampling)

    # ---- sampler / param builders ------------------------------------------ #
    def _sampler(self, batch):
        if batch not in self._samplers:
            s = Sampling1D(
                vocab_size=self.vocab,
                mesh_device=self.mesh_device,
                max_batch_size=batch,
                max_top_k=32,
                allow_force_argmax=True,
                pad_to_power_of_2=True,
            )
            s.load_device_buffers()
            self._samplers[batch] = s
            k = self._rep(torch.ones([batch], dtype=torch.int32), ttnn.uint32)
            p = self._rep(torch.ones([batch], dtype=torch.float32), ttnn.bfloat16)
            t = self._rep(torch.ones([batch], dtype=torch.float32), ttnn.bfloat16)
            self._greedy_params[batch] = (k, p, t)
        return self._samplers[batch]

    def _rep(self, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(
            t, dtype=dtype, layout=layout, device=self.mesh_device, mesh_mapper=_replicate(self.mesh_device)
        )

    # ---- KV cache / page table (owned by generate) ------------------------- #
    def _ensure_cache(self, users, seq_needed):
        seq_needed = min(seq_needed, self.max_seq_len)
        if self._kv_cache is not None and users <= self._kv_users and seq_needed <= self._kv_seq:
            return
        # (Re)allocate for the larger requirement.
        self._kv_cache = self.model.alloc_kv_cache(max_users=users, max_seq_len=seq_needed, block_size=BLOCK_SIZE)
        self._kv_users = users
        self._kv_seq = seq_needed
        self._page_table = self.model.make_page_table(users, self._kv_cache[0]["blocks_per_user"])
        self._trace = {}  # cache geometry changed -> invalidate captured decode traces

    # ---- token helpers ----------------------------------------------------- #
    def _tokens_to_device(self, token_ids_2d):
        """int list/tensor [1,S] -> uint32 device tensor [1,S] (replicated)."""
        t = torch.as_tensor(token_ids_2d, dtype=torch.int32).reshape(1, -1)
        return self._rep(t, ttnn.uint32)

    def _read_token(self, tok_buf, batch):
        """Read back only the sampled token id(s) (single-token readback, replicated across mesh)."""
        self.counters["readback"] += 1
        th = ttnn.to_torch(tok_buf, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))
        return th[0].flatten()[:batch].to(torch.int64).tolist()

    # ---- low-level prefill ------------------------------------------------- #
    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table=None,
        kv_cache=None,
        prompt_lens: Optional[List[int]] = None,
        return_all_logits: bool = False,
        user_id: int = 0,
        start_pos: int = 0,
        **kwargs: Any,
    ):
        """Low-level prefill. Fills ``kv_cache`` in place. Accepts any prompt length (the decoder
        chunks/pads internally; the public shapes here are the logical token count). Returns host
        logits: ``[batch, 1, vocab]`` (last position) or ``[batch, prompt_len, vocab]`` when
        ``return_all_logits``. On the device-sampling path callers may prefer sampled tokens, but the
        readiness prefill check wants logits, so prefill always returns logits."""
        tokens = torch.as_tensor(tokens, dtype=torch.int64)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        batch, seq = tokens.shape
        # Generator owns cache + page table when the caller does not supply a usable one.
        if kv_cache is None:
            self._ensure_cache(max(batch, self._kv_users or batch), start_pos + seq + 1)
            kv_cache = self._kv_cache
            page_table = self._page_table
        elif page_table is None:
            page_table = self._page_table

        all_logits = []
        last_logits = []
        for u in range(batch):
            tok_tt = self._tokens_to_device(tokens[u : u + 1])
            x = self.model.embed_prefill(tok_tt)
            h = self.model.prefill_layers(x, kv_cache, page_table, user_id=user_id + u, start_pos=start_pos)
            if return_all_logits:
                shards = self.model.lm_head_shards_prefill(h)  # [1,seq,V/D]
                logits = self.model.logits_to_host(shards).reshape(seq, self.vocab)
                all_logits.append(logits)
            else:
                last = ttnn.slice(h, [0, seq - 1, 0], [1, seq, self.hidden])
                shards = self.model.lm_head_shards_prefill(last)
                logits = self.model.logits_to_host(shards).reshape(1, self.vocab)
                last_logits.append(logits)
        if return_all_logits:
            return torch.stack(all_logits, dim=0)  # [batch, seq, vocab]
        return torch.stack(last_logits, dim=0)  # [batch, 1, vocab]

    # ---- low-level decode (single step) ------------------------------------ #
    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        *,
        page_table=None,
        kv_cache=None,
        return_logits: bool = False,
        **kwargs: Any,
    ):
        """Low-level single decode step (eager). ``tokens`` [batch,1], ``start_pos`` [batch].
        Returns sampled tokens ``[batch]`` (on-device greedy split sampling) or logits ``[batch,vocab]``
        when ``return_logits``. The high-level ``generate`` uses the traced path instead."""
        tokens = torch.as_tensor(tokens, dtype=torch.int64).reshape(-1, 1)
        batch = tokens.shape[0]
        pos = torch.as_tensor(start_pos, dtype=torch.int32).reshape(batch)
        if kv_cache is None:
            kv_cache = self._kv_cache
            page_table = self._page_table if page_table is None else page_table
        tok_tt = self._rep(tokens.reshape(1, batch).to(torch.int32), ttnn.uint32)
        cur = self._rep(pos, ttnn.int32)
        ridx = self._rep(pos.reshape(1, batch), ttnn.uint32)
        h = self.model.embed_decode(tok_tt)
        h = self.model.decode_layers(h, cur, ridx, page_table, kv_cache)
        shards = self.model.lm_head_shards_decode(h)
        if return_logits or self.host_sampling:
            logits = self.model.logits_to_host(shards).reshape(batch, self.vocab)
            if return_logits:
                return logits
            return torch.argmax(logits, dim=-1)
        tok_buf = self._rep(torch.zeros([1, 1, 1, batch], dtype=torch.int32), ttnn.uint32)
        self._greedy_sample(shards, batch, tok_buf)
        return torch.as_tensor(self._read_token(tok_buf, batch), dtype=torch.int64)

    def _greedy_sample(self, logit_shards, batch, tt_out_tok):
        sampler = self._sampler(batch)
        k, p, t = self._greedy_params[batch]
        sampler.decode_forward(logit_shards, k=k, p=p, temp=t, tt_out_tok=tt_out_tok)
        return tt_out_tok

    # ---- high-level generate ----------------------------------------------- #
    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        *,
        next_input: Optional[Callable[[int, int], int]] = None,
        enable_trace: bool = True,
        stop_on_eos: bool = False,
        **kwargs: Any,
    ) -> List[int]:
        """HF-style deterministic greedy loop over the low-level path. Returns the model's OWN
        predictions (argmax), regardless of teacher-forcing overrides. Teacher forcing (``next_input``
        returns the token to feed next) is honoured by overwriting the persistent decode token buffer
        from host between steps; free-running (``next_input is None``) feeds the on-device sampled
        token back with no host token work."""
        prompt = list(prompt_token_ids)
        P = len(prompt)
        batch = 1
        self._ensure_cache(1, P + max_new_tokens + 1)
        kv_cache, page_table = self._kv_cache, self._page_table

        # ---- prefill -> first prediction p0 (device greedy sample of the last position) ----
        tok_tt = self._tokens_to_device(torch.tensor(prompt))
        x = self.model.embed_prefill(tok_tt)
        h = self.model.prefill_layers(x, kv_cache, page_table, user_id=0, start_pos=0)
        last = ttnn.slice(h, [0, P - 1, 0], [1, P, self.hidden])
        shards = self.model.lm_head_shards_decode(ttnn.reshape(last, (1, 1, 1, self.hidden)))
        if self.host_sampling:
            logits = self.model.logits_to_host(shards).reshape(self.vocab)
            p0 = int(torch.argmax(logits))
        else:
            tok_buf0 = self._rep(torch.zeros([1, 1, 1, 1], dtype=torch.int32), ttnn.uint32)
            self._greedy_sample(shards, 1, tok_buf0)
            p0 = self._read_token(tok_buf0, 1)[0]

        preds = [p0]
        chosen = next_input(0, p0) if next_input is not None else p0

        if max_new_tokens <= 1:
            return preds

        if enable_trace and not self.host_sampling:
            self._generate_traced(preds, chosen, P, max_new_tokens, next_input, stop_on_eos)
        else:
            self._generate_eager(preds, chosen, P, max_new_tokens, next_input, stop_on_eos, enable_trace)
        return preds

    def _decode_trace_state(self, batch, page_table, init_pos, init_tok):
        """Build (once per batch) the persistent decode tensors and capture the decode trace over
        them: embed(tok) -> layers -> norm -> LM head -> greedy sample into tok -> plus_one(pos/rope).

        The compile + capture runs execute real decode steps (they write KV + advance positions), so
        they are staged at ``init_pos`` (the first real decode position) — never 0 — so their cache
        writes land at positions ``init_pos``/``init_pos+1`` (harmlessly overwritten by the first real
        replays) and NEVER corrupt the prefilled prompt cache at positions ``0..init_pos-1``."""
        st = self._trace.get(batch)
        if st is not None:
            return st
        tok = self._rep(torch.zeros([1, 1, 1, batch], dtype=torch.int32), ttnn.uint32)
        cur = self._rep(torch.zeros([batch], dtype=torch.int32), ttnn.int32)
        ridx = self._rep(torch.zeros([1, batch], dtype=torch.int32), ttnn.uint32)
        kv_cache = self._kv_cache

        def step():
            h = self.model.embed_decode(ttnn.reshape(tok, (1, batch)))
            h = self.model.decode_layers(h, cur, ridx, page_table, kv_cache)
            shards = self.model.lm_head_shards_decode(h)
            self._greedy_sample(shards, batch, tok)
            ttnn.plus_one(cur, skip_negative_entries=True)
            ttnn.plus_one(ridx)

        # Stage safe starting positions so the compile/capture writes never touch the prompt cache.
        ttnn.copy_host_to_device_tensor(self._host_rank4_tok(init_tok), tok)
        ttnn.copy_host_to_device_tensor(self._host_pos(init_pos), cur)
        ttnn.copy_host_to_device_tensor(self._host_ridx(init_pos), ridx)
        step()  # compile
        ttnn.synchronize_device(self.mesh_device)
        tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        step()  # capture
        ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        st = dict(tid=tid, tok=tok, cur=cur, ridx=ridx, batch=batch)
        self._trace[batch] = st
        return st

    def _generate_traced(self, preds, chosen, P, max_new_tokens, next_input, stop_on_eos):
        batch = 1
        st = self._decode_trace_state(batch, self._page_table, P, chosen)
        tok, cur, ridx, tid = st["tok"], st["cur"], st["ridx"], st["tid"]
        # Stage the first decode input/positions on device (one host refresh per prompt — a scheduler
        # state change, not per-token work).
        ttnn.copy_host_to_device_tensor(self._host_rank4_tok(chosen), tok)
        ttnn.copy_host_to_device_tensor(self._host_pos(P), cur)
        ttnn.copy_host_to_device_tensor(self._host_ridx(P), ridx)
        self.counters["token_refresh"] += 1
        self.counters["pos_refresh"] += 1
        eos = self._eos_id()
        for i in range(1, max_new_tokens):
            ttnn.execute_trace(self.mesh_device, tid, cq_id=0, blocking=True)
            self.counters["trace_replay"] += 1
            p_i = self._read_token(tok, batch)[0]  # model's own prediction (single-token readback)
            preds.append(p_i)
            if stop_on_eos and p_i == eos:
                break
            if next_input is not None:
                # Teacher forcing: overwrite the on-device token buffer with the forced next input.
                forced = next_input(i, p_i)
                ttnn.copy_host_to_device_tensor(self._host_rank4_tok(forced), tok)
                self.counters["token_refresh"] += 1
            # else free-running: tok already holds p_i on device (device feedback); cur/ridx advanced
            # on device inside the trace — no host token/position work.

    def _generate_eager(self, preds, chosen, P, max_new_tokens, next_input, stop_on_eos, enable_trace):
        batch = 1
        eos = self._eos_id()
        cur_in = chosen
        pos = P
        for i in range(1, max_new_tokens):
            out = self.decode_forward(
                torch.tensor([[cur_in]]), torch.tensor([pos]), page_table=self._page_table, kv_cache=self._kv_cache
            )
            p_i = int(out[0])
            preds.append(p_i)
            if stop_on_eos and p_i == eos:
                break
            cur_in = next_input(i, p_i) if next_input is not None else p_i
            pos += 1

    def _host(self, t, dtype):
        """Host-side multi-device tensor (no device=) — valid source for copy_host_to_device_tensor."""
        return ttnn.from_torch(t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=_replicate(self.mesh_device))

    def _host_rank4_tok(self, token_id):
        return self._host(torch.tensor([[[[int(token_id)]]]], dtype=torch.int32), ttnn.uint32)

    def _host_pos(self, pos):
        return self._host(torch.tensor([int(pos)], dtype=torch.int32), ttnn.int32)

    def _host_ridx(self, pos):
        return self._host(torch.tensor([[int(pos)]], dtype=torch.int32), ttnn.uint32)

    def _eos_id(self):
        eid = getattr(self.tokenizer, "eos_token_id", None)
        return int(eid) if isinstance(eid, int) else -1

    # ---- reset ------------------------------------------------------------- #
    def reset(self) -> None:
        """Zero the KV cache and clear per-prompt decode state. Keeps device buffers, weights, and
        captured decode traces alive so subsequent prompts are cheap."""
        if self._kv_cache is not None:
            self.model.reset_kv_cache(self._kv_cache)
        self.counters = self._zero_counters()

    def teardown(self) -> None:
        for st in self._trace.values():
            try:
                ttnn.release_trace(self.mesh_device, st["tid"])
            except Exception:
                pass
        self._trace = {}


# --- factory convention ---------------------------------------------------- #
def build_generator(model_dir, mesh_device, **kwargs) -> ReadinessGenerator:
    """Required readiness factory. ``kwargs``: max_seq_len, num_layers, host_sampling,
    precision_policy, precision_config_path. When neither precision kwarg is given, the
    datatype-sweep-selected config artifact is consumed by default (see
    ``LagunaModel.load_selected_precision_policy``) so build_generator and the vLLM adapter
    build the exact selected weight/activation/CCL/KV/compute-fidelity policy."""
    _ = Path(model_dir)
    return LagunaGenerator.from_pretrained(
        mesh_device,
        max_seq_len=int(kwargs.get("max_seq_len", 16384)),
        num_layers=kwargs.get("num_layers"),
        host_sampling=bool(kwargs.get("host_sampling", False)),
        precision_policy=kwargs.get("precision_policy"),
        precision_config_path=kwargs.get("precision_config_path"),
    )
