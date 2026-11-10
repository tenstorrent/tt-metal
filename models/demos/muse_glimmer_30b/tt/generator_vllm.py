# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""vLLM (vllm-tt-plugin) speculative-decode contract adapter for Muse Glimmer 30B with the DFlash2 drafter.

Wraps `DFlash2BatchGenerator` (up to 8 sequences, TP4 or TP8) on the plugin's spec-decode contract (vllm-tt-plugin
docs/SPEC_DECODE_CONTRACT.md): the plugin owns scheduling, sampling and the accept walk; this class owns the device
engine, its KV cache and the drafter. K (drafts per row) follows the number of live rows (`_K_BY_ROWS`, published as
the plan's `k_by_rows`); the engine verifies R = K + 1 rows per sequence in one 32-row step.

Select it with (the checkpoint architecture is MuseGlimmerForConditionalGeneration; the plugin prefixes `TT`):

    TT_MODEL_CLASS_OVERRIDES="TTMuseGlimmerForConditionalGeneration=models.demos.muse_glimmer_30b.tt.generator_vllm:MuseGlimmerDFlash2ForCausalLM"

Model-owned KV: engine slot u owns target page row u of the model's own cache and drafter ring u. vLLM's block tables
and block ids are bookkeeping only and are ignored; `get_kv_cache_spec` and `get_max_tokens_all_users` only have to
make the scheduler admit max_num_seqs sequences of max_model_len tokens.

Slots: the plugin names a request's state by a slot (`empty_slots` at prefill, `slot_remap` when it compacts rows,
`release_request`). A request keeps its engine slot for life; `phys[plugin_slot]` maps to it, and a `slot_remap` is
applied to that map only (no KV moves), once, when the decode that carries it succeeds.

Engine state follows the plugin's call sequence (verify, accept walk on the host, propose). A verify is committed
lazily: `_outstanding` holds the engine slots whose last verify is not committed yet, and the next call that learns
how many tokens the plugin accepted for a row (`propose_draft_tokens` through `accepted_counts`, or `decode_forward`
through `start_pos`) commits it first.
"""

import functools
import os
import time

import torch
from loguru import logger

_BLOCK_SIZE = 32
_ROWS = 16  # anchor + 15 drafts, verified in one target pass
_MAX_DRAFTS = _ROWS - 1
_SEQ_ALIGN = 256
_MAX_CONTEXT = 131072
_SUPPORTED_TP = (4, 8)
_TIMING = os.environ.get("MUSE_VLLM_TIMING") == "1"  # read once; log one MUSE_TIMING line per adapter entry
_K_BY_ROWS = ((2, 15), (4, 7), (8, 3))  # (max live rows, drafts per row): the plugin's draft cap and the engine's R - 1
_MAX_SEQS = _K_BY_ROWS[-1][0]
_MAX_DEVICE_TOP_K = 32  # the engine's topk verify reads the top 32 candidates of every row (model_config max_top_k)
_GREEDY_TEMPERATURE = 1e-5  # at or below: the target argmax (vLLM's sampling epsilon)


def _round_up(x, m):
    return -(-x // m) * m


def _served_seq_len(max_model_len):
    """Model max_seq_len for a served max_model_len: the engine writes 16 rows past the last committed position."""
    return _round_up(int(max_model_len) + _ROWS, _SEQ_ALIGN)


def _check_served_seq_len(max_model_len):
    s = _served_seq_len(max_model_len)
    if s > _MAX_CONTEXT:
        raise ValueError(
            f"max_model_len={max_model_len} needs a {s}-token model context (max_model_len + {_ROWS} rounded up to "
            f"{_SEQ_ALIGN}), above the {_MAX_CONTEXT} the model supports; launch with "
            f"--max-model-len <= {_MAX_CONTEXT - _ROWS}"
        )
    return s


def build_dflash_engine(model, enable_trace=True):
    """DFlash2 batch engine on `model` for args.max_batch_size sequences (engine slot u owns target page row u)."""
    from models.demos.muse_glimmer_30b.tt.dflash.batch_generator import build_batch_generator

    return build_batch_generator(model, model.args.max_batch_size, enable_trace=enable_trace, k_by_users=_K_BY_ROWS)


class MuseGlimmerDFlash2ForCausalLM:
    # decode_forward is overridden and runs no reload commands (DECODE_RELOAD_CONTRACT legacy mode).
    decode_input_update_contract = 0
    # The scheduler's KV bookkeeping is one uniform group; the model owns the physical cache.
    _HYBRID_KV_CACHE_GROUPS_ENABLED = False

    # Prefill and host-sampled steps return host logits for the plugin's host sampler; a host-sampled verify returns
    # argmax ids (greedy) or logits (accept mode "logits"). On a device-sampled step (sample_on_device_mode
    # "decode_only", every sampled row with 1 <= top_k <= 32) the engine reads only its top-32 candidates per row and
    # the adapter draws the tokens: an ordinary decode returns them, and an argmax_ids verify returns the ids up to the
    # first mismatch (spec_ids_are_sampled). After a request's first sampled step, its drafts come from the drafter
    # distribution and the verify ids are the engine's lossless rejection sample; before it, one draw per column.
    model_capabilities = {
        "supports_spec_decode": True,
        "spec_requirements": ("device_propose",),
        "spec_hidden_handoff": ("on_device",),
        "output_tokens_per_step": 1,
        "supports_sample_on_device": True,
        "max_device_top_k": _MAX_DEVICE_TOP_K,
        "spec_ids_are_sampled": True,
        "supports_device_penalties": False,
        "supports_device_grammar": False,
        "supports_async_decode": False,
        "supports_async_spec_decode": False,
        "supports_prefix_caching": False,
        "supports_chunked_prefill": False,
    }

    def __init__(self, engine, kv_handles=None):
        self.engine = engine
        self.kv_handles = kv_handles
        self.n_slots = int(engine.max_users)
        self.phys = list(range(self.n_slots))  # plugin state slot -> engine slot (see the module docstring)
        self._outstanding = set()  # engine slots whose last verify is not committed yet
        self._last_drafts = {}  # engine slot -> (drafts, q) of the last propose (mismatch warning, rejection sampling)
        self._rng = {}  # engine slot -> torch.Generator of its request's device-sampled draws (seeded on first use)

    @classmethod
    def _from_engine(cls, engine, kv_handles=None):
        """Build the adapter around an existing engine (host tests build it around a fake one)."""
        return cls(engine, kv_handles)

    # vLLM's model-registry inspection needs the upstream interface; execution goes through the TT plugin.
    def embed_input_ids(self, input_ids):  # pragma: no cover - protocol shim
        raise RuntimeError("Use the TT plugin's prefill/decode interface")

    def forward(self, input_ids, positions, **kwargs):  # pragma: no cover - protocol shim
        raise RuntimeError("Use the TT plugin's prefill/decode interface")

    def compute_logits(self, hidden_states, **kwargs):  # pragma: no cover - protocol shim
        raise RuntimeError("The DFlash2 engine owns the LM head")

    # ---- config-time hooks
    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        """Config-time plan; never raises and never touches a device."""
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        n = int(max_num_seqs)
        if not 1 <= n <= _MAX_SEQS:
            return SpecReject(
                reason=f"the DFlash2 engine serves at most {_MAX_SEQS} sequences (K by live rows {_K_BY_ROWS}), "
                f"got max_num_seqs={n}",
                supported_k=(),
            )
        k = max(1, min(int(requested_k), _MAX_DRAFTS))
        return SpecPlan(
            effective_k=k,
            lanes_per_request=1,
            extra_bytes_per_seq=0,  # drafter KV and verify buffers are allocated by the engine, not budgeted here
            extra_bytes_per_token=0,
            accept_modes=("argmax_ids", "logits"),
            drafter_state="internal",
            supports_narrow_decode=True,
            k_by_rows=tuple((rows, min(kr, k)) for rows, kr in _K_BY_ROWS),
        )

    @classmethod
    def get_max_tokens_all_users(cls, *, max_model_len=None, max_num_seqs=1, **kwargs):
        """The model owns one cache row of max_model_len tokens per sequence, so the scheduler pool covers them all."""
        n = int(max_num_seqs)
        if not 1 <= n <= _MAX_SEQS:
            raise ValueError(f"{cls.__name__} supports max_num_seqs in [1, {_MAX_SEQS}], got {max_num_seqs}")
        if max_model_len is None:
            return n * _MAX_CONTEXT
        _check_served_seq_len(max_model_len)
        return n * int(max_model_len)

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """Bookkeeping-only uniform KV spec (every layer a FullAttentionSpec so vLLM keeps one group)."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        cache_config = vllm_config.cache_config
        if cache_config.block_size != _BLOCK_SIZE:
            raise ValueError(f"Muse uses {_BLOCK_SIZE}-token KV pages, got block_size={cache_config.block_size}")
        text_config = getattr(vllm_config.model_config.hf_config, "text_config", vllm_config.model_config.hf_config)
        common = dict(
            block_size=_BLOCK_SIZE,
            num_kv_heads=text_config.num_key_value_heads,
            head_size=text_config.head_dim,
            dtype=vllm_config.model_config.dtype,
        )
        return {f"model.layers.{i}.self_attn": FullAttentionSpec(**common) for i in range(len(text_config.layer_types))}

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len=None, tt_data_parallel=1, optimizations=None
    ):
        n = int(max_batch_size)
        if not 1 <= n <= _MAX_SEQS:
            raise ValueError(
                f"{cls.__name__} serves at most {_MAX_SEQS} sequences; launch with --max-num-seqs <= {_MAX_SEQS}"
            )
        if tt_data_parallel != 1:
            raise ValueError(
                f"{cls.__name__} does not support data parallelism, got tt_data_parallel={tt_data_parallel}"
            )
        if optimizations is not None:
            raise ValueError(f"{cls.__name__} has one precision policy; drop the `optimizations` setting")
        if max_seq_len is None:
            raise ValueError("max_seq_len (the vLLM max_model_len) is required")
        seq_len = _check_served_seq_len(max_seq_len)
        shape = tuple(int(x) for x in mesh_device.shape)
        if shape == (8, 4):
            # The plugin opened the full Galaxy system mesh (a partial 1x8 open picks chips whose fabric links do not
            # all train): carve one TP8 replica out of it, like the tt-metal tests (tp_submesh reshapes to (4, 8)).
            from models.demos.muse_glimmer_30b.tt import mesh_utils

            mesh_device = mesh_utils.tp_submesh(mesh_device, 8)
            shape = tuple(int(x) for x in mesh_device.shape)
        if shape == (8, 4):
            # The plugin opened the full Galaxy system mesh (a partial 1x8 open picks chips whose fabric links do not
            # all train): carve one TP8 replica out of it, like the tt-metal tests (tp_submesh reshapes to (4, 8)).
            from models.demos.muse_glimmer_30b.tt import mesh_utils

            mesh_device = mesh_utils.tp_submesh(mesh_device, 8)
            shape = tuple(int(x) for x in mesh_device.shape)
        if shape[0] != 1 or shape[1] not in _SUPPORTED_TP:
            raise ValueError(f"{cls.__name__} needs a 1xTP mesh with TP in {_SUPPORTED_TP} (TP4 or TP8), got {shape}")

        from models.demos.muse_glimmer_30b.tt.model import MuseGlimmerModel
        from models.demos.muse_glimmer_30b.tt.model_config import (
            MuseGlimmerModelArgs,
            checkpoint_dir,
            dflash_checkpoint_dir,
        )

        ckpt = checkpoint_dir()
        logger.info(
            f"[muse dflash2] target {ckpt}, drafter {dflash_checkpoint_dir()}, "
            f"max_model_len={max_seq_len} -> model max_seq_len={seq_len}, max_num_seqs={n}"
        )
        args = MuseGlimmerModelArgs.from_checkpoint(
            ckpt, mesh_device=mesh_device, max_batch_size=n, max_seq_len=seq_len, max_prefill_len=seq_len
        )
        text_config = getattr(hf_config, "text_config", hf_config)
        if int(text_config.vocab_size) != int(args.vocab_size):
            logger.warning(f"hf_config vocab_size {text_config.vocab_size} != model vocab_size {args.vocab_size}")
        model = MuseGlimmerModel(args, mesh_device)
        engine = build_dflash_engine(model, enable_trace=True)
        return cls._from_engine(engine, kv_handles=[list(kv) for kv in model.kv_caches])

    # ---- KV: the model owns it
    def allocate_kv_cache(self, *args, **kwargs):
        del args, kwargs
        return self.kv_handles

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        del per_layer_specs  # vLLM's sizing is bookkeeping only
        return self.kv_handles

    # ---- warmup: the engine compiles on first use and captures its traces on the second and later uses
    def warmup_model_prefill(self, kv_cache=None, enable_trace=False, can_sample_on_device=False, **kwargs):
        return None

    def warmup_model_decode(self, kv_cache=None, enable_trace=False, **kwargs):
        return None

    # ---- engine state reconciliation
    def _commit(self, e, count, last):
        count, last = int(count), int(last)
        if not 1 <= count <= _ROWS:
            raise RuntimeError(f"cannot commit {count} tokens in engine slot {e}; a verify runs at most {_ROWS} rows")
        self.engine.commit_slot(e, count, last)
        self._outstanding.discard(e)

    def _remapped(self, slot_remap):
        """phys after `slot_remap` (slot i now names the state that slot slot_remap[i] named). Bookkeeping only: each
        request keeps its engine slot, so no KV moves."""
        if slot_remap is None:
            return list(self.phys)
        idx = [int(j) for j in torch.as_tensor(slot_remap).reshape(-1).tolist()]
        if sorted(idx) != list(range(self.n_slots)):
            raise RuntimeError(f"slot_remap {idx} is not a permutation of the {self.n_slots} slots")
        return [self.phys[j] for j in idx]

    def _engine_slot(self, phys, row):
        if row >= self.n_slots:
            raise RuntimeError(f"live row {row} is beyond the {self.n_slots} engine slots")
        e = phys[row]
        if e not in self.engine.slots:
            raise RuntimeError(f"row {row} (engine slot {e}) has no prefilled request")
        return e

    def _settle(self, phys, live):
        """The decode is accepted: its slot_remap takes effect and every live row has an uncommitted verify."""
        self.phys = phys
        self._outstanding.update(live.values())

    def _sampler(self, e, sp, i):
        """(temperature, top_p, top_k, generator) of decode row i (engine slot e) from TTSamplingParams, or None for a
        greedy row. A seeded request's generator starts from its seed at its first draw; an unseeded one from a random
        seed. Draws advance it in row and column order. The first call also sets the engine slot's sampling, so the
        slot's later proposals draw their drafts from the drafter distribution (lossless rejection sampling)."""
        t, k = float(sp.temperature[i]), int(sp.top_k[i])
        if t <= _GREEDY_TEMPERATURE or k == 1:
            return None
        if not 1 <= k <= _MAX_DEVICE_TOP_K:
            raise RuntimeError(f"row {i}: device-sampled top_k={k} outside [1, {_MAX_DEVICE_TOP_K}]")
        gen = self._rng.get(e)
        if gen is None:
            gen = torch.Generator()
            seed = sp.seed[i]
            if seed is None:
                gen.seed()
            else:
                gen.manual_seed(int(seed))
            self._rng[e] = gen
        s = self.engine.slots[e]
        if s.sp is None:
            s.sp, s.gen = (t, float(sp.top_p[i]), k), gen
        return t, float(sp.top_p[i]), k, gen

    @staticmethod
    def _draws(cand, n, sampler, drafts=(), q=None):
        """Tokens for the first n rows of a slot's topk candidates (values [R, C], ids [R, C]). A greedy row (sampler
        None) gets the top candidate of each row. A sampled row whose n-1 drafts were drawn from the drafter
        probabilities q = (probs [n-1, kq], ids [n-1, kq]) gets the lossless rejection sample: the accepted drafts, then
        the residual or bonus draw (the plugin accepts while the drafts match and commits the first mismatch). Any other
        sampled row gets one draw per row."""
        from models.demos.muse_glimmer_30b.tt.dflash.sampling import rejection_sample
        from models.demos.muse_glimmer_30b.tt.sampling import draw, filtered_probs

        values, ids = (torch.as_tensor(x) for x in cand)
        if sampler is None:
            return [int(ids[j, int(values[j].float().argmax())]) for j in range(n)]
        t, top_p, k, gen = sampler
        probs, pids = filtered_probs(values[:n], ids[:n], t, top_p, k)
        if q is not None:
            acc, bonus = rejection_sample(list(drafts), q[0], q[1], probs, pids, gen)
            return list(drafts[:acc]) + [bonus]
        return [draw(probs[j], pids[j], gen) for j in range(n)]

    # ---- prefill / decode
    def prefill_forward(self, tokens, page_table=None, kv_cache=None, prompt_lens=None, **kwargs):
        tokens = torch.as_tensor(tokens)
        n_rows = int(tokens.shape[0])
        empty_slots = kwargs.get("empty_slots")
        slots = list(range(n_rows)) if empty_slots is None else [int(s) for s in empty_slots]
        if len(slots) != n_rows or any(not 0 <= s < self.n_slots for s in slots):
            raise ValueError(f"empty_slots {slots} do not name {n_rows} of the {self.n_slots} slots")
        start_pos = kwargs.get("start_pos")
        if start_pos is not None and any(int(x) != 0 for x in torch.as_tensor(start_pos).reshape(-1).tolist()):
            raise ValueError("chunked prefill and prefix caching are not supported (start_pos must be 0)")
        for name in ("pixel_values", "pixel_values_videos"):
            if kwargs.get(name) is not None and any(v is not None for v in kwargs[name]):
                raise ValueError(f"{type(self).__name__} is text-only; {name} is not supported")
        lens = [int(prompt_lens[r]) if prompt_lens is not None else int(tokens.shape[1]) for r in range(n_rows)]
        max_seq_len = int(self.engine.model.args.max_seq_len)
        for n in lens:
            if n + _ROWS > max_seq_len:
                raise ValueError(
                    f"prompt of {n} tokens does not fit the model context {max_seq_len} (needs {_ROWS} spare)"
                )
        out = []
        for r, (s, n) in enumerate(zip(slots, lens)):
            e = self.phys[s]
            self._outstanding.discard(e)
            self._last_drafts.pop(e, None)
            self._rng.pop(e, None)
            logits = self.engine.start_slot(e, [int(t) for t in tokens[r, :n].tolist()], return_logits=True)
            out.append(logits.float().reshape(1, -1))
        return torch.stack(out)  # host logits [N, 1, V]

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        *,
        num_valid_drafts=None,
        accepted_counts=None,
        spec_mode=None,
        slot_remap=None,
        **kwargs,
    ):
        from vllm_tt_plugin.spec_decode import VerifyOutput

        tokens = torch.as_tensor(tokens)
        rows = int(tokens.shape[0])
        tokens = tokens.reshape(rows, -1)
        start_pos = torch.as_tensor(start_pos).reshape(rows, -1)
        phys = self._remapped(slot_remap)
        eng = self.engine
        vocab = int(eng.model.args.vocab_size)
        sp = kwargs.get("sampling_params")
        live = {i: self._engine_slot(phys, i) for i in range(rows) if int(start_pos[i, 0]) >= 0}
        for i, e in live.items():
            q = int(start_pos[i, 0])
            if e in self._outstanding:  # the plugin's position and token say what it committed
                self._commit(e, q - eng.slots[e].p, tokens[i, 0])
            if q != eng.slots[e].p:
                raise RuntimeError(f"decode row {i} at position {q} but engine slot {e} is at {eng.slots[e].p}")

        if spec_mode is None:
            if sp is not None:  # device-sampled: the drawn ids, int32 [B]
                cand = eng.verify_slots({e: [int(tokens[i, 0])] for i, e in live.items()}, "topk") if live else {}
                ids = torch.zeros(rows, dtype=torch.int32)
                for i, e in live.items():
                    ids[i] = self._draws(cand[e], 1, self._sampler(e, sp, i))[0]
                    self._last_drafts.pop(e, None)
                self._settle(phys, live)
                return ids
            # Host-sampled ordinary decode: a one-row verify per live row, host logits [B, 1, V] for the host sampler.
            lg = eng.verify_slots({e: [int(tokens[i, 0])] for i, e in live.items()}, "logits") if live else {}
            out = torch.zeros(rows, 1, vocab, dtype=torch.float32)
            for i, e in live.items():
                out[i, 0] = lg[e][0]
                self._last_drafts.pop(e, None)
            self._settle(phys, live)
            return out

        width = int(tokens.shape[1])
        if width > _ROWS:
            raise ValueError(f"verify block of {width} columns exceeds the engine's {_ROWS} rows")
        nv = torch.as_tensor(num_valid_drafts).reshape(-1)
        req, qs = {}, {}
        for i, e in live.items():
            n = int(nv[i])
            if not 0 <= n <= width - 1:
                raise ValueError(f"num_valid_drafts[{i}]={n} outside [0, {width - 1}]")
            toks = [int(t) for t in tokens[i, : 1 + n].tolist()]
            offered, q = self._last_drafts.pop(e, (None, None))
            if offered is not None and toks[1:] != offered[:n]:
                logger.warning(f"row {i}: verify drafts {toks[1:]} differ from the proposed {offered[:n]}")
            elif q is not None:  # the drafts were drawn from q
                qs[e] = (q[0][:n], q[1][:n])
            req[e] = toks
        if spec_mode == "argmax_ids":
            samplers = {} if sp is None else {i: self._sampler(e, sp, i) for i, e in live.items()}
            ids = torch.zeros(rows, width, dtype=torch.int32)
            if any(s is not None for s in samplers.values()):
                # Device-sampled with a sampled row (spec_ids_are_sampled): the ids up to the first mismatch.
                cand = eng.verify_slots(req, "topk")
                for i, e in live.items():
                    out = self._draws(cand[e], len(req[e]), samplers[i], req[e][1:], qs.get(e))
                    ids[i, : len(out)] = torch.tensor(out, dtype=torch.int32)
            else:
                am = eng.verify_slots(req, "argmax") if req else {}
                for i, e in live.items():
                    c = min(len(am[e]), width)
                    ids[i, :c] = torch.tensor([int(a) for a in am[e][:c]], dtype=torch.int32)
            self._settle(phys, live)
            return VerifyOutput(spec_mode="argmax_ids", argmax_ids=ids)
        if spec_mode == "logits":
            lg = eng.verify_slots(req, "logits") if req else {}
            # The plugin reads live rows up to their bonus column only; the rest stays unwritten (a zeroed
            # [B, 1+K, V] float block is ~134 MB at 8 rows).
            out = torch.empty(rows, width, vocab, dtype=torch.float32)
            for i, e in live.items():
                c = min(lg[e].shape[0], width)
                out[i, :c] = lg[e][:c].float()
            self._settle(phys, live)
            return VerifyOutput(spec_mode="logits", logits=out)
        raise ValueError(f"unsupported spec_mode {spec_mode!r}")

    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        from vllm_tt_plugin.spec_decode import DraftOutput

        committed_tokens = torch.as_tensor(committed_tokens)
        rows = int(committed_tokens.shape[0])
        committed_tokens = committed_tokens.reshape(rows, -1)
        committed_positions = torch.as_tensor(committed_positions).reshape(rows, -1)
        accepted_counts = torch.as_tensor(accepted_counts).reshape(-1)
        k = int(num_drafts)
        if k > _MAX_DRAFTS:
            raise ValueError(f"num_drafts={k} exceeds the drafter's {_MAX_DRAFTS}")
        eng = self.engine
        live = {i: self._engine_slot(self.phys, i) for i in range(rows) if int(committed_positions[i, 0]) >= 0}
        for i, e in live.items():
            if e in self._outstanding:
                count = int(accepted_counts[i])
                if not 1 <= count <= committed_tokens.shape[1]:
                    raise RuntimeError(
                        f"accepted_counts[{i}]={count} outside the committed block {tuple(committed_tokens.shape)}"
                    )
                self._commit(e, count, committed_tokens[i, count - 1])
        ids = torch.zeros(rows, k, dtype=torch.int32)
        num_valid = torch.zeros(rows, dtype=torch.int32)
        # A row without an anchor (prefilled, not verified yet) or without room for a 16-row verify gets no drafts.
        limit = int(eng.model.args.max_seq_len) - _ROWS
        ready = {i: e for i, e in live.items() if eng.slots[e].anchor is not None and eng.slots[e].p <= limit}
        drafts = eng.propose_slots(sorted(ready.values())) if ready else {}
        for e in set(live.values()) - set(ready.values()):
            self._last_drafts.pop(e, None)
        for i, e in ready.items():
            d = [int(t) for t in drafts[e][0]][:k]
            q, qc = drafts[e][1], drafts[e][2]
            self._last_drafts[e] = (d, None if q is None else (q[: len(d)], qc[: len(d)]))
            ids[i, : len(d)] = torch.tensor(d, dtype=torch.int32)
            num_valid[i] = len(d)
        return DraftOutput(draft_token_ids=ids, num_valid=num_valid)

    # ---- lifecycle
    def release_request(self, slot):
        e = self.phys[int(slot)]
        self.engine.release_slot(e)
        self._outstanding.discard(e)
        self._last_drafts.pop(e, None)
        self._rng.pop(e, None)

    def note_state_slots_moved(self, moves):
        """decode_forward already applied the matching slot_remap to `phys`; nothing moves on the device."""
        del moves

    def release_persistent_capture(self):
        self.engine.release_traces()


def _install_timing():
    """Wrap the adapter entries with MUSE_TIMING logging (only when MUSE_VLLM_TIMING=1 at import)."""
    cls = MuseGlimmerDFlash2ForCausalLM
    state = {"last_return": None, "engine_s": 0.0}

    def wrap_engine(engine):
        if getattr(engine, "_muse_timed", False):
            return
        for name in ("start_slot", "commit_slot", "verify_slots", "propose_slots"):
            fn = getattr(engine, name, None)
            if fn is None:
                continue

            def timed(*a, _fn=fn, **kw):
                t0 = time.perf_counter()
                try:
                    return _fn(*a, **kw)
                finally:
                    state["engine_s"] += time.perf_counter() - t0

            setattr(engine, name, timed)
        engine._muse_timed = True

    def wrap(name, kind_of):
        orig = getattr(cls, name)

        @functools.wraps(orig)
        def inner(self, *args, **kwargs):
            wrap_engine(self.engine)
            t0 = time.perf_counter()
            gap = (t0 - state["last_return"]) * 1e3 if state["last_return"] is not None else float("nan")
            state["engine_s"] = 0.0
            try:
                return orig(self, *args, **kwargs)
            finally:
                t1 = time.perf_counter()
                state["last_return"] = t1
                logger.info(
                    f"MUSE_TIMING kind={kind_of(kwargs)} engine_ms={state['engine_s'] * 1e3:.2f} "
                    f"adapter_ms={(t1 - t0) * 1e3:.2f} gap_ms={gap:.2f} slots={len(getattr(self.engine, 'slots', {}))}"
                )

        setattr(cls, name, inner)

    def decode_kind(kw):
        m = kw.get("spec_mode")
        if m is None and kw.get("sampling_params") is not None:
            return "decode_sampled"
        return "decode" if m is None else ("verify_argmax" if m == "argmax_ids" else "verify_logits")

    wrap("prefill_forward", lambda kw: "prefill")
    wrap("decode_forward", decode_kind)
    wrap("propose_draft_tokens", lambda kw: "propose")


if _TIMING:
    _install_timing()
