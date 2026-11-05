# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""vLLM (vllm-tt-plugin) speculative-decode contract adapter for Muse Glimmer 30B with the DFlash2 drafter.

Wraps `DFlash2Generator` (batch 1, TP4) on the plugin's spec-decode contract (vllm-tt-plugin
docs/SPEC_DECODE_CONTRACT.md): the plugin owns scheduling, sampling and the accept walk; this class owns the device
engine, its KV cache and the drafter.

Select it with (the checkpoint architecture is MuseGlimmerForConditionalGeneration; the plugin prefixes `TT`):

    TT_MODEL_CLASS_OVERRIDES="TTMuseGlimmerForConditionalGeneration=models.demos.muse_glimmer_30b.tt.generator_vllm:MuseGlimmerDFlash2ForCausalLM"

Model-owned KV: the engine runs on the model's own identity-paged KV cache (user 0). vLLM's block tables and block ids
are bookkeeping only and are ignored; `get_kv_cache_spec` only has to make the scheduler admit `max_model_len` tokens.

Engine state follows the plugin's call sequence (verify, accept walk on the host, propose). The engine commits a
verify lazily: `_outstanding` says the last verify has not been committed yet, and the next call that learns how many
tokens the plugin accepted (`propose_draft_tokens` through `accepted_counts`, or `decode_forward` through
`start_pos`) commits it first.
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
_SUPPORTED_TP = (4,)
_TIMING = os.environ.get("MUSE_VLLM_TIMING") == "1"  # read once; log one MUSE_TIMING line per adapter entry


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


def build_dflash_engine(mesh, args, model, enable_trace=True):
    """DFlash2 drafter + selector + generator for batch 1 on `model` (user 0 owns the identity page-table row)."""
    import ttnn
    from models.demos.muse_glimmer_30b.tt.dflash.drafter import TtDFlash2Drafter
    from models.demos.muse_glimmer_30b.tt.dflash.generator import DFlash2Generator
    from models.demos.muse_glimmer_30b.tt.dflash.selector import HostCandidateSelector

    drafter = TtDFlash2Drafter(mesh, args, model.ccl, model.rope)
    selector = HostCandidateSelector.from_checkpoint()
    nb = args.max_num_blocks_per_user
    pt = torch.arange(nb, dtype=torch.int32).reshape(1, nb)
    pt_tt = ttnn.from_torch(
        pt,
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return DFlash2Generator(model, drafter, selector, pt, pt_tt, list(args.eos_token_ids), enable_trace=enable_trace)


class MuseGlimmerDFlash2ForCausalLM:
    # decode_forward is overridden and runs no reload commands (DECODE_RELOAD_CONTRACT legacy mode).
    decode_input_update_contract = 0
    # The scheduler's KV bookkeeping is one uniform group; the model owns the physical cache.
    _HYBRID_KV_CACHE_GROUPS_ENABLED = False

    # The plugin samples prefill and ordinary decode on the host from our logits, and rejection-samples verify rows
    # on the host (accept mode "logits") or walks argmax ids (greedy).
    model_capabilities = {
        "supports_spec_decode": True,
        "spec_requirements": ("device_propose",),
        "spec_hidden_handoff": ("on_device",),
        "output_tokens_per_step": 1,
        "supports_sample_on_device": False,
        "supports_async_decode": False,
        "supports_async_spec_decode": False,
        "supports_prefix_caching": False,
        "supports_chunked_prefill": False,
    }

    def __init__(self, engine, kv_handles=None):
        self.engine = engine
        self.kv_handles = kv_handles
        self._outstanding = False  # the engine's last verify is not committed yet
        self._last_drafts = None  # drafts the last propose offered (for the mismatch warning)

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

        if int(max_num_seqs) != 1:
            return SpecReject(
                reason=f"the DFlash2 engine serves one sequence (batch 1), got max_num_seqs={max_num_seqs}",
                supported_k=(),
            )
        return SpecPlan(
            effective_k=max(1, min(int(requested_k), _MAX_DRAFTS)),
            lanes_per_request=1,
            extra_bytes_per_seq=0,  # drafter KV and verify buffers are allocated by the engine, not budgeted here
            extra_bytes_per_token=0,
            accept_modes=("argmax_ids", "logits"),
            drafter_state="internal",
            supports_narrow_decode=True,
        )

    @classmethod
    def get_max_tokens_all_users(cls, *, max_model_len=None, max_num_seqs=1, **kwargs):
        """The model owns one identity-mapped cache for one sequence, so the scheduler pool covers max_model_len."""
        if int(max_num_seqs) != 1:
            raise ValueError(f"{cls.__name__} supports max_num_seqs=1, got {max_num_seqs}")
        if max_model_len is None:
            return _MAX_CONTEXT
        _check_served_seq_len(max_model_len)
        return int(max_model_len)

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
        if int(max_batch_size) != 1:
            raise ValueError(f"{cls.__name__} supports one sequence; launch with --max-num-seqs 1")
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
        if shape[0] != 1 or shape[1] not in _SUPPORTED_TP:
            raise ValueError(f"{cls.__name__} needs a 1x{_SUPPORTED_TP} mesh, got {shape}")

        from models.demos.muse_glimmer_30b.tt.model import MuseGlimmerModel
        from models.demos.muse_glimmer_30b.tt.model_config import (
            MuseGlimmerModelArgs,
            checkpoint_dir,
            dflash_checkpoint_dir,
        )

        ckpt = checkpoint_dir()
        logger.info(
            f"[muse dflash2] target {ckpt}, drafter {dflash_checkpoint_dir()}, "
            f"max_model_len={max_seq_len} -> model max_seq_len={seq_len}"
        )
        args = MuseGlimmerModelArgs.from_checkpoint(
            ckpt, mesh_device=mesh_device, max_batch_size=1, max_seq_len=seq_len, max_prefill_len=seq_len
        )
        text_config = getattr(hf_config, "text_config", hf_config)
        if int(text_config.vocab_size) != int(args.vocab_size):
            logger.warning(f"hf_config vocab_size {text_config.vocab_size} != model vocab_size {args.vocab_size}")
        model = MuseGlimmerModel(args, mesh_device)
        engine = build_dflash_engine(mesh_device, args, model, enable_trace=True)
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
    def _commit(self, count, last):
        count, last = int(count), int(last)
        if not 1 <= count <= _ROWS:
            raise RuntimeError(f"cannot commit {count} tokens; the last verify ran {_ROWS} rows")
        self.engine.commit(count, last)
        self._outstanding = False

    def _commit_from_decode(self, tokens, start_pos):
        """A decode call follows an uncommitted verify: the plugin's position and token say what it committed."""
        q = int(start_pos[0, 0])
        count = q - int(self.engine.p)
        if not 1 <= count <= _ROWS:
            raise RuntimeError(f"decode at position {q} implies {count} committed tokens (engine at {self.engine.p})")
        self._commit(count, tokens[0, 0])

    def _commit_from_propose(self, committed_tokens, accepted_counts):
        count = int(accepted_counts[0])
        if not 1 <= count <= committed_tokens.shape[1]:
            raise RuntimeError(
                f"accepted_counts[0]={count} outside the committed block {tuple(committed_tokens.shape)}"
            )
        self._commit(count, committed_tokens[0, count - 1])

    # ---- prefill / decode
    def prefill_forward(self, tokens, page_table=None, kv_cache=None, prompt_lens=None, **kwargs):
        if tokens.shape[0] != 1:
            raise ValueError(f"prefill serves one sequence, got {tokens.shape[0]}")
        start_pos = kwargs.get("start_pos")
        if start_pos is not None and int(torch.as_tensor(start_pos).flatten()[0]) != 0:
            raise ValueError("chunked prefill and prefix caching are not supported (start_pos must be 0)")
        for name in ("pixel_values", "pixel_values_videos"):
            if kwargs.get(name) is not None and any(v is not None for v in kwargs[name]):
                raise ValueError(f"{type(self).__name__} is text-only; {name} is not supported")
        n = int(prompt_lens[0]) if prompt_lens is not None else int(tokens.shape[1])
        max_seq_len = int(self.engine.model.args.max_seq_len)
        if n + _ROWS > max_seq_len:
            raise ValueError(f"prompt of {n} tokens does not fit the model context {max_seq_len} (needs {_ROWS} spare)")
        prompt_ids = [int(t) for t in tokens[0, :n].tolist()]
        logits = self.engine.start(prompt_ids, return_logits=True)
        self._outstanding = False
        self._last_drafts = None
        return logits.float().reshape(1, 1, -1)  # host logits [B=1, 1, V]

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
        **kwargs,
    ):
        from vllm_tt_plugin.spec_decode import VerifyOutput

        tokens = torch.as_tensor(tokens)
        start_pos = torch.as_tensor(start_pos)
        rows = int(tokens.shape[0])
        start_pos = start_pos.reshape(rows, -1)
        tokens = tokens.reshape(rows, -1)
        if int(start_pos[0, 0]) < 0:
            raise RuntimeError("decode row 0 is inactive; the engine serves one sequence in row 0")
        if self._outstanding:
            self._commit_from_decode(tokens, start_pos)
        if int(start_pos[0, 0]) != int(self.engine.p):
            raise RuntimeError(f"decode at position {int(start_pos[0, 0])} but the engine is at {self.engine.p}")
        eng = self.engine

        if spec_mode is None:
            # Ordinary decode: a one-row verify, host logits [B, 1, V] for the plugin's host sampler.
            lg = eng.verify([int(tokens[0, 0])], "logits")
            self._outstanding = True
            self._last_drafts = None
            out = torch.zeros(rows, 1, lg.shape[-1], dtype=torch.float32)
            out[0, 0] = lg[0]
            return out

        width = int(tokens.shape[1])
        if width > _ROWS:
            raise ValueError(f"verify block of {width} columns exceeds the engine's {_ROWS} rows")
        n = int(torch.as_tensor(num_valid_drafts)[0])
        if not 0 <= n <= width - 1:
            raise ValueError(f"num_valid_drafts[0]={n} outside [0, {width - 1}]")
        toks = [int(t) for t in tokens[0, : 1 + n].tolist()]
        if self._last_drafts is not None and toks[1:] != self._last_drafts[:n]:
            logger.warning(f"verify drafts {toks[1:]} differ from the proposed {self._last_drafts[:n]}")
        self._last_drafts = None
        if spec_mode == "argmax_ids":
            am = eng.verify(toks, "argmax")
            self._outstanding = True
            ids = torch.zeros(rows, width, dtype=torch.int32)
            ids[0] = torch.tensor([int(a) for a in am[:width]], dtype=torch.int32)
            return VerifyOutput(spec_mode="argmax_ids", argmax_ids=ids)
        if spec_mode == "logits":
            lg = eng.verify(toks, "logits")
            self._outstanding = True
            out = torch.zeros(rows, width, lg.shape[-1], dtype=torch.float32)
            out[0] = lg[:width].float()
            return VerifyOutput(spec_mode="logits", logits=out)
        raise ValueError(f"unsupported spec_mode {spec_mode!r}")

    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        from vllm_tt_plugin.spec_decode import DraftOutput

        committed_tokens = torch.as_tensor(committed_tokens)
        accepted_counts = torch.as_tensor(accepted_counts)
        rows = int(committed_tokens.shape[0])
        k = int(num_drafts)
        if k > _MAX_DRAFTS:
            raise ValueError(f"num_drafts={k} exceeds the drafter's {_MAX_DRAFTS}")
        if self._outstanding:
            self._commit_from_propose(committed_tokens, accepted_counts)
        eng = self.engine
        ids = torch.zeros(rows, k, dtype=torch.int32)
        num_valid = torch.zeros(rows, dtype=torch.int32)
        if eng.anchor is None or eng.p + _ROWS > int(eng.model.args.max_seq_len):
            self._last_drafts = None  # nothing to continue from, or no room for the 16-row verify
            return DraftOutput(draft_token_ids=ids, num_valid=num_valid)
        drafts = [int(t) for t in eng.propose()[0]][:k]
        self._last_drafts = drafts
        ids[0, : len(drafts)] = torch.tensor(drafts, dtype=torch.int32)
        num_valid[0] = len(drafts)
        return DraftOutput(draft_token_ids=ids, num_valid=num_valid)

    # ---- lifecycle
    def release_request(self, slot):
        eng = self.engine
        eng.p = 0
        eng.anchor = None
        eng.pending = None
        self._outstanding = False
        self._last_drafts = None

    def note_state_slots_moved(self, moves):
        """Batch 1: the only slot is 0, so a move other than the identity is a bug."""
        if any(int(old) != int(new) for old, new in dict(moves).items()):
            raise RuntimeError(f"single-slot adapter was asked to move state slots: {moves}")

    def release_persistent_capture(self):
        self.engine.release_traces()


def _install_timing():
    """Wrap the adapter entries with MUSE_TIMING logging (only when MUSE_VLLM_TIMING=1 at import)."""
    cls = MuseGlimmerDFlash2ForCausalLM
    state = {"last_return": None, "engine_s": 0.0}

    def wrap_engine(engine):
        if getattr(engine, "_muse_timed", False):
            return
        for name in ("start", "commit", "verify", "propose"):
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
                    f"adapter_ms={(t1 - t0) * 1e3:.2f} gap_ms={gap:.2f} p={getattr(self.engine, 'p', None)}"
                )

        setattr(cls, name, inner)

    def decode_kind(kw):
        m = kw.get("spec_mode")
        return "decode" if m is None else ("verify_argmax" if m == "argmax_ids" else "verify_logits")

    wrap("prefill_forward", lambda kw: "prefill")
    wrap("decode_forward", decode_kind)
    wrap("propose_draft_tokens", lambda kw: "propose")


if _TIMING:
    _install_timing()
