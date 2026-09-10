# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""CosyVoice2's LLM backbone: Qwen2-0.5B, adapted from tt_transformers.

Confirmed against upstream `cosyvoice/llm/llm.py` directly (`Qwen2Encoder`,
`Qwen2LM`), not assumed from the earlier planning discussion:

    self.llm = Qwen2ForCausalLM.from_pretrained(pretrain_path)   # stock HF Qwen2
    ...
    outs = self.model(inputs_embeds=xs, attention_mask=masks,
                       use_cache=True, past_key_values=cache, ...)

Confirmed against the real checkpoint's own `config.json`
(`CosyVoice2-0.5B/CosyVoice-BlankEN/config.json`, not the generic
`Qwen/Qwen2-0.5B` model card): `hidden_size=896` (matches `cosyvoice2.yaml`'s
`llm_input_size`/`llm_output_size` exactly, so no projection is needed at the
boundary), `num_hidden_layers=24`, `num_attention_heads=14`,
`num_key_value_heads=2`, `intermediate_size=4864`, `rope_theta=1e6`,
`vocab_size=151936` -- a stock, unmodified Qwen2 architecture. Nothing about
the transformer stack itself is CosyVoice-specific.

What IS CosyVoice-specific, confirmed from `Qwen2LM.__init__`/`.inference`:
Qwen2LM bypasses its own parent class's `__init__` (`torch.nn.Module.__init__(
self)`, not `TransformerLM.__init__(self, ...)`), so CosyVoice1's separate
Conformer text encoder and speaker-embedding injection do not exist on a real
Qwen2LM at all. The whole input sequence is
`concat([sos_emb, text_emb, task_id_emb, speech_token_emb], dim=1)`, where
`text_emb` comes from Qwen2's *own* embedding table
(`self.llm.model.model.embed_tokens`), `sos_emb`/`task_id_emb` are two rows of
a separate `nn.Embedding(2, 896)`, and `speech_token_emb` comes from a third,
separate `nn.Embedding(speech_token_size + 3, 896)` (`speech_token_size =
6561`). Output projects through a small `nn.Linear(896, speech_token_size + 3)`
(`= 6564`), NOT Qwen2's own (tied, 151936-wide) lm_head.

No CosyVoice2 checkpoint is available yet, so `speech_embedding`, `llm_embedding`
and the output head are randomly initialised here (matching the pattern used
throughout this package: TorchHiFTDecodeRef, TtSourceModuleHnNSF's linear).
The Qwen2 backbone itself -- embeddings and all 24 transformer layers -- uses
REAL downloaded `Qwen/Qwen2-0.5B-Instruct` weights (architecturally identical
to CosyVoice2's own fine-tuned checkpoint, confirmed above), giving a stronger
validation baseline than random init for that part: see
tests/pcc/test_qwen2lm.py.

Scope of this module: construction only (the "skeleton") -- 24
`TransformerBlock`s, `RotarySetup`, final norm, the three embedding tables, and
the output head, all built and individually checked against a real reference.
Sequence assembly, the prefill/decode driving loop, and RAS sampling are a
separate, later piece (matching the real seam in `Qwen2LM.inference_wrapper`:
one `forward_one_step` call over the full prefix with `cache=None`, then one
call per new token with a growing cache -- tt_transformers' existing
prefill/decode split, not a new pattern, but not wired up here yet).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.decoder import TransformerBlock
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.embedding import Embedding
from models.tt_transformers.tt.rope import HfRotarySetup, RotarySetup, get_rot_mats, get_rot_mats_hf

# Attention.forward_prefill hard-asserts `seq_len % 128 == 0`. CosyVoice2's assembled
# prefix (sos + text + task_id + prompt speech tokens) is essentially never a multiple of
# 128, so TtQwen2LM.prefill right-pads to this before calling into it. Padding at the tail
# is harmless under causal attention (see prefill's docstring): a padded position can only
# ever be attended *from*, never *to*, by any real position before it.
PREFILL_SEQ_MULTIPLE = 128


class TtSmallEmbedding(LightweightModule):
    """A small, CosyVoice-specific embedding table (speech tokens, or the
    2-row sos/task_id table) -- not tt_transformers' `Embedding`, which is
    sized and cached around a big (100K+) HF vocab. `ttnn.embedding` directly,
    nothing else."""

    def __init__(self, device, weight: torch.Tensor, dtype=ttnn.bfloat16):
        super().__init__()
        self.device = device
        self.weights = ttnn.from_torch(
            weight.detach().float().unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )

    def forward(self, ids: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.embedding(ids, self.weights, layout=ttnn.TILE_LAYOUT)


class TtLinearHead(LightweightModule):
    """CosyVoice2's `llm_decoder`: one small Linear(896, speech_token_size+3).

    Not tt_transformers' `LMHead` -- that class exists to shard a 100K+-wide
    vocab projection across DRAM banks/devices with a prefetcher ring-matmul
    path. This head is 6564-wide on one device; none of that machinery
    applies, and instantiating it would build sharding infrastructure for a
    problem that does not exist here.
    """

    def __init__(self, device, weight: torch.Tensor, bias: torch.Tensor, dtype=ttnn.bfloat16):
        super().__init__()
        self.device = device
        self.weight = ttnn.from_torch(
            weight.detach().float().t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.bias = ttnn.from_torch(
            bias.detach().float().reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(x, self.weight, bias=self.bias)


class TtQwen2LM:
    """CosyVoice2's Qwen2-0.5B backbone, construction only -- see module
    docstring for what is and is not wired up yet.

    `args` is a `models.tt_transformers.tt.model_config.ModelArgs` built with
    `HF_MODEL=Qwen/Qwen2-0.5B-Instruct` (no special-case registration needed --
    confirmed empirically: the generic HF config path resolves `dim`, `n_layers`,
    `n_heads`, `n_kv_heads`, `rope_theta`, etc. correctly with zero entries added
    to model_config.py's per-model dicts). `state_dict` is `args.load_state_dict()`
    -- the real, HF-to-meta-converted Qwen2-0.5B-Instruct weights.
    """

    def __init__(
        self,
        args,
        mesh_device,
        state_dict: dict,
        speech_token_size: int = 6561,
        dtype=ttnn.bfloat16,
        seed: int = 0,
    ):
        self.args = args
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.speech_token_size = speech_token_size
        self.head_out_features = speech_token_size + 3  # eos, task_id-adjacent, fill -- see module docstring

        self.tt_ccl = TT_CCL(mesh_device)

        # -- Qwen2's own text embedding table: real weights, tt_transformers'
        # existing Embedding class is a direct, unmodified fit (built exactly
        # for "HF-format tok_embeddings.weight, big vocab lookup").
        self.text_embedding = Embedding(
            mesh_device=mesh_device,
            args=args,
            weight_cache_path=args.weight_cache_path(dtype),
            state_dict=state_dict,
            dtype=ttnn.bfloat16,  # row-major embedding lookup requires bf16, matching Transformer.__init__
        )

        # -- CosyVoice-specific tables: no checkpoint yet, random-init (same
        # pattern as TorchHiFTDecodeRef / TtSourceModuleHnNSF's linear).
        g = torch.Generator().manual_seed(seed)
        speech_embedding_weight = torch.empty(self.head_out_features, args.dim).normal_(0, 0.02, generator=g)
        llm_embedding_weight = torch.empty(2, args.dim).normal_(0, 0.02, generator=g)
        head_weight = torch.empty(self.head_out_features, args.dim).normal_(0, 0.02, generator=g)
        head_bias = torch.zeros(self.head_out_features)

        self.speech_embedding = TtSmallEmbedding(mesh_device, speech_embedding_weight, dtype=dtype)
        self.llm_embedding = TtSmallEmbedding(mesh_device, llm_embedding_weight, dtype=dtype)
        self.llm_decoder = TtLinearHead(mesh_device, head_weight, head_bias, dtype=dtype)

        # -- RoPE: standard HF-style, matching Qwen2's own rope_theta/config.
        RopeSetupClass = HfRotarySetup if args.use_hf_rope else RotarySetup
        self.rope_setup = RopeSetupClass(
            mesh_device,
            args.max_batch_size,
            args.head_dim,
            args.max_seq_len,
            args.rope_theta,
            args.rope_scaling,
            args.use_qk_fused,
        )
        self.trans_mats_dict = self.rope_setup.get_both_trans_mats()

        # -- 24 unmodified TransformerBlocks, real Qwen2-0.5B-Instruct weights.
        # No Transformer() wrapper: that class's __init__ also builds a
        # 151936-wide Embedding/LMHead we would never call -- wasted weight
        # loading and device memory for a component CosyVoice2's LLM doesn't
        # use (it never generates text, only speech tokens).
        self.layers = [
            TransformerBlock(
                args=args,
                mesh_device=mesh_device,
                tt_ccl=self.tt_ccl,
                dtype=dtype,
                state_dict=state_dict,
                weight_cache_path=args.weight_cache_path(dtype),
                layer_num=i,
                transformation_mats=self.trans_mats_dict,
            )
            for i in range(args.n_layers)
        ]

        self.norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", None),
                weight_cache_path=None if args.dummy_weights else args.weight_cache_path(dtype),
                weight_dtype=ttnn.bfloat16,
                weight_key="norm",
                add_unit_offset=args.rms_norm_add_unit_offset,
                is_distributed=args.is_distributed_norm,
                ccl_topology=args.ccl_topology(),
                tt_ccl=self.tt_ccl,
            ),
            args,
            tt_ccl=self.tt_ccl,
            TG=args.is_galaxy,
        )

        self._prefill_rot_mats = None  # lazily built, see _rot_mats_prefill_table

    # ------------------------------------------------------------------
    # Host-side embedding helpers -- each round-trips through the SAME
    # device table `TtQwen2LM.__init__` built (text_embedding / llm_embedding /
    # speech_embedding), so a value returned here is bit-identical to what the
    # production per-token path would embed, not a host-side shortcut.
    # ------------------------------------------------------------------
    def embed_text_tokens_host(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: torch [1, N] -> torch [1, N, dim], via the real Qwen2 embed_tokens table."""
        ids_dev = ttnn.from_torch(
            ids.reshape(1, 1, 1, -1).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
        )
        out = ttnn.to_torch(self.text_embedding(ids_dev)).float()
        return out.reshape(1, ids.shape[-1], self.args.dim)

    def embed_llm_tokens_host(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: torch [1, N] (values in {0=sos, 1=task_id}) -> torch [1, N, dim]."""
        ids_dev = ttnn.from_torch(
            ids.reshape(1, 1, 1, -1).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
        )
        out = ttnn.to_torch(self.llm_embedding(ids_dev)).float()
        return out.reshape(1, ids.shape[-1], self.args.dim)

    def embed_speech_tokens_host(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: torch [1, N] speech token ids -> torch [1, N, dim]."""
        ids_dev = ttnn.from_torch(
            ids.reshape(1, 1, 1, -1).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
        )
        out = ttnn.to_torch(self.speech_embedding(ids_dev)).float()
        return out.reshape(1, ids.shape[-1], self.args.dim)

    # ------------------------------------------------------------------
    # Sequence assembly
    # ------------------------------------------------------------------
    def assemble_prefill_sequence(
        self, text_ids: torch.Tensor, prompt_speech_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Build the LLM's input sequence exactly as real `Qwen2LM.inference` does:

            lm_input = torch.concat([sos_emb, text_emb, task_id_emb, speech_token_emb], dim=1)

        `text_ids`: torch [1, T] real Qwen2 token ids. `prompt_speech_ids`: torch [1, S]
        speech token ids, or None/empty for no prompt (both mean "no speech segment" --
        matching upstream's own `if prompt_speech_token_len != 0` guard). `llm_embedding`'s
        two rows are sos=0, task_id=1, per this package's module docstring.

        Returns a torch [1, L, dim] tensor, ready for `prefill`.
        """
        sos_emb = self.embed_llm_tokens_host(torch.tensor([[0]]))
        task_emb = self.embed_llm_tokens_host(torch.tensor([[1]]))
        text_emb = self.embed_text_tokens_host(text_ids)
        parts = [sos_emb, text_emb, task_emb]
        if prompt_speech_ids is not None and prompt_speech_ids.numel() > 0:
            parts.append(self.embed_speech_tokens_host(prompt_speech_ids))
        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------
    # Prefill / decode
    # ------------------------------------------------------------------
    def _rot_mats_prefill_table(self):
        """The full-context RoPE table PREFILL mode uses (tt_transformers' own
        `get_rot_mats`/`get_rot_mats_hf`, not the `RotarySetup` class instance --
        confirmed from `test_decoder_prefill.py`, which is the established
        pattern this method mirrors). Built once and cached: it depends only on
        `args`, not on any per-call sequence."""
        if self._prefill_rot_mats is None:
            rot_mats_fn = get_rot_mats_hf if self.args.use_hf_rope else get_rot_mats
            self._prefill_rot_mats = rot_mats_fn(
                head_dim=self.args.head_dim,
                device=self.mesh_device,
                seq_len=self.args.max_seq_len,
                theta=self.args.rope_theta,
                rope_scaling=self.args.rope_scaling,
            )
        return self._prefill_rot_mats

    def prefill(self, sequence_bsh: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Run the whole assembled prefix through all `n_layers` in PREFILL mode.

        `Attention.forward_prefill` runs `ttnn.transformer.scaled_dot_product_attention(
        ..., is_causal=True)` -- a real causal mask built internally, no external mask
        tensor needed -- and writes the resulting K/V into each layer's persistent
        `Attention.layer_past` via `ttnn.fill_cache`, the SAME cache object `decode_step`
        below reads. That is the whole prefill/decode continuity contract; nothing else
        has to be threaded between the two calls.

        `sequence_bsh`: torch [1, L, dim], from `assemble_prefill_sequence`. Right-padded
        on the host to a multiple of 128 (`forward_prefill`'s hard requirement) -- harmless
        under causal attention, since the padding rows sit *after* every real position and
        causal masking only ever looks backward, so no real position's output is affected
        by what the pad rows contain. The pad rows' own K/V get written into the cache
        (`fill_cache` always writes the entire sequence it receives), but that is likewise
        harmless: `decode_step` bounds attention by `current_pos`, not by cache occupancy,
        so slots `>= L` are simply never read on the very next decode step -- the same
        property `test_device_backbone_matches_real_qwen2_multistep_decode` already
        exercises for cache slots beyond the tested position.

        Returns `(last_hidden, L)`: the post-norm hidden state at the real last position
        (torch [1, 1, dim], BEFORE padding), and `L` -- the position the first `decode_step`
        call must use.
        """
        L = sequence_bsh.shape[1]
        Lp = ((L + PREFILL_SEQ_MULTIPLE - 1) // PREFILL_SEQ_MULTIPLE) * PREFILL_SEQ_MULTIPLE
        if Lp != L:
            pad = torch.zeros(1, Lp - L, sequence_bsh.shape[-1], dtype=sequence_bsh.dtype)
            sequence_bsh = torch.cat([sequence_bsh, pad], dim=1)

        x_tt = self.args.prepare_residual_tensor_prefill(sequence_bsh)
        rot_mats = self._rot_mats_prefill_table()
        for layer in self.layers:
            x_tt = layer(x_tt, None, rot_mats_global=rot_mats, mode=Mode.PREFILL)
        norm_config = self.args.get_norm_config("lm_head", Mode.PREFILL, None)
        x_tt = self.norm(x_tt, mode=Mode.PREFILL, norm_config=norm_config)

        hidden = ttnn.to_torch(x_tt).float()
        last_hidden = hidden[:, :, L - 1 : L, : self.args.dim]
        return last_hidden, L

    def decode_step(self, token_embedding_bsh: torch.Tensor, current_pos_val: int) -> torch.Tensor:
        """One new token's embedding in, one post-norm hidden state out.

        `token_embedding_bsh`: torch [1, 1, dim] (e.g. from `embed_speech_tokens_host` on
        the previously sampled token). `current_pos_val`: the absolute position this token
        occupies -- `L` (prefill's return value) for the step right after prefill, then
        `L+1, L+2, ...` for each step after -- continuing the SAME persistent KV cache
        `prefill` populated, never reset to 0.

        Returns torch [1, 1, 1, dim].
        """
        decode_input = self.args.prepare_residual_tensor_decode(
            token_embedding_bsh, self.args.get_residual_mem_config(Mode.DECODE, None)
        )
        current_pos = torch.tensor([current_pos_val])
        current_pos_tensor = ttnn.from_torch(current_pos, device=self.mesh_device, dtype=ttnn.int32)
        rot_mats = self.rope_setup.get_rot_mats(current_pos)
        x_tt = decode_input
        for layer in self.layers:
            x_tt = layer(x_tt, current_pos_tensor, rot_mats_global=rot_mats, mode=Mode.DECODE)
        norm_config = self.args.get_norm_config("lm_head", Mode.DECODE, None)
        x_tt = self.norm(x_tt, mode=Mode.DECODE, norm_config=norm_config)

        hidden = ttnn.to_torch(x_tt).float()
        return hidden[:, :, :1, : self.args.dim]

    def logits_for_hidden(self, hidden_bsh: torch.Tensor) -> torch.Tensor:
        """torch [1, 1, dim] (or [1, 1, 1, dim]) -> torch [head_out_features] logits."""
        hidden_tt = ttnn.from_torch(
            hidden_bsh.reshape(1, 1, 1, self.args.dim),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
        )
        logits_tt = self.llm_decoder(hidden_tt)
        return ttnn.to_torch(logits_tt).float().reshape(-1)[: self.head_out_features]

    # ------------------------------------------------------------------
    # RAS sampling: on-device nucleus primary path, host fallback for the
    # repeat-window resample. See tt/llm/sampling.py's module docstring for
    # why the two paths split this way.
    # ------------------------------------------------------------------
    def device_nucleus_sampler(self) -> "TtDeviceNucleusSampler":
        if getattr(self, "_device_nucleus_sampler", None) is None:
            self._device_nucleus_sampler = TtDeviceNucleusSampler(self.mesh_device, self.tt_ccl, self.head_out_features)
        return self._device_nucleus_sampler

    def ras_sample_device(
        self,
        logits: torch.Tensor,
        decoded_tokens,
        top_p: float = 0.8,
        top_k: int = 25,
        win_size: int = 10,
        tau_r: float = 0.1,
    ) -> int:
        """RAS sampling with the primary nucleus draw on-device.

        Mirrors `sampling.ras_sampling`'s control flow exactly, but the first
        `nucleus_sampling` call is `TtDeviceNucleusSampler.sample` instead of the host
        implementation -- NOT bit-exact with it (see sampling.py's module docstring for the
        `< top_p` vs `<= top_p` boundary mismatch). The repetition-triggered resample stays
        on host: it needs `decoded_tokens` and rewrites one score before drawing again,
        which `sampling.random_sampling` already does correctly.
        """
        from .sampling import is_repetitive, random_sampling

        top_ids = self.device_nucleus_sampler().sample(logits, top_p=top_p, top_k=top_k)
        if is_repetitive(decoded_tokens, top_ids, win_size, tau_r):
            logits = logits.clone()
            logits[top_ids] = -float("inf")
            top_ids = random_sampling(logits)
        return top_ids

    def generate(
        self,
        text_ids: torch.Tensor,
        prompt_speech_ids: torch.Tensor | None = None,
        *,
        max_tokens: int = 200,
        min_tokens: int = 0,
        eos_id: int | None = None,
        sampler: str = "ras",
        seed: int | None = None,
        **sampling_kwargs,
    ) -> list[int]:
        """Text token ids -> semantic speech token ids, autoregressively.

        Matches upstream `Qwen2LM.inference`'s control flow: one prefill over
        `[sos, text_emb, task_id_emb, speech_emb]`, then one `decode_step` per new token,
        each fed its own previously-sampled token's embedding, until `eos_id` is sampled (at
        or past `min_tokens`) or `max_tokens` is reached. `sampler='greedy'` is the
        deterministic path this package's tests use; `'ras'` is real Repetition-Aware
        Sampling (`ras_sample_device`, on-device nucleus primary + host fallback).

        No CosyVoice2 checkpoint exists yet (see module docstring), so `eos_id` has no
        trained meaning here -- this method exercises the real control flow and cache
        continuity, not real generation quality.
        """
        from .sampling import greedy, ras_sampling

        if seed is not None:
            torch.manual_seed(seed)
        if eos_id is None:
            eos_id = self.speech_token_size

        sequence = self.assemble_prefill_sequence(text_ids, prompt_speech_ids)
        hidden, pos = self.prefill(sequence)
        logits = self.logits_for_hidden(hidden)

        out: list[int] = []
        for i in range(max_tokens):
            if sampler == "greedy":
                token = greedy(logits)
            elif sampler == "ras":
                token = ras_sampling(logits.clone(), out, **sampling_kwargs)
            elif sampler == "ras_device":
                token = self.ras_sample_device(logits, out, **sampling_kwargs)
            else:
                raise ValueError(f"unknown sampler {sampler!r}")
            if token == eos_id and i >= min_tokens:
                break
            out.append(token)
            token_emb = self.embed_speech_tokens_host(torch.tensor([[token]]))
            hidden = self.decode_step(token_emb, pos)
            logits = self.logits_for_hidden(hidden)
            pos += 1
        return out


class TtDeviceNucleusSampler:
    """On-device top-p/top-k draw for RAS's primary path, via tt_transformers' own
    `SamplingGenerator`/`TTSampling` -- NOT a bespoke sampling op.

    Constructed against a small `SimpleNamespace` shim rather than `TtQwen2LM`'s own
    `args`, deliberately: `TTSampling` reads `vocab_size`/`padded_vocab_size` off whatever
    `args` it is given, and `TtQwen2LM.args` is Qwen2's own ModelArgs, sized for its real
    151936-wide text vocabulary (needed for `text_embedding`) -- not for the 6564-wide
    CosyVoice-specific speech-token distribution this sampler actually draws over. The shim
    pattern (a lightweight namespace exposing exactly the attributes `TTSampling.__init__`
    reads via `getattr`) matches the one `models/common/tests/test_tt_sampling.py` and
    `test_sampling.py` themselves use for single-device tests, not an ad hoc simplification.

    NOT bit-exact with `sampling.nucleus_sampling` -- see that module's docstring for the
    `< top_p` (host, before adding) vs `<= top_p` (`ttnn.topk`-backed, after adding)
    boundary mismatch. Correctness here is checked against what both algorithms DO agree
    on: at `top_k=1`, both must return the row's maximum-probability token.
    """

    def __init__(self, mesh_device, tt_ccl, vocab_size: int):
        self.mesh_device = mesh_device
        self.vocab_size = vocab_size
        self.padded_vocab_size = ((vocab_size + 31) // 32) * 32
        sampling_args = SimpleNamespace(
            vocab_size=vocab_size,
            padded_vocab_size=self.padded_vocab_size,
            max_batch_size=32,
            max_top_k=32,
            cluster_shape=(1, 1),
            sub_core_grids=None,
            sub_core_grid_topk=None,
            start_core=ttnn.CoreCoord(0, 0),
        )
        self.generator = SamplingGenerator(args=sampling_args, mesh_device=mesh_device, tt_ccl=tt_ccl)

    def sample(self, logits: torch.Tensor, *, top_p: float, top_k: int, temperature: float = 1.0) -> int:
        """`logits`: torch, `vocab_size`-long (last dim used if higher-rank).

        The device batch is hard-tiled to (a multiple of) 32 lanes (`format_sampling_params`);
        every lane is given the same real distribution and only lane 0's result is used --
        there is exactly one logical sequence being sampled here, not 32.
        """
        flat = logits.reshape(-1).float()
        assert flat.shape[0] == self.vocab_size, (flat.shape, self.vocab_size)

        # format_sampling_params is not optional decoration here: it is what turns the
        # dataclass's scalar penalty defaults (presence_penalty=0.0, a bare float) into
        # full max_batch_size-length lists. Skipping it and calling reset_sampling_params
        # on the raw SamplingParams reaches TTPenalties._pad_params with a 0-d/empty
        # tensor and raises IndexError -- this is the exact call sequence
        # apply_prefill_state/apply_decode_state use internally.
        params = format_sampling_params(
            SamplingParams(temperature=[temperature] * 32, top_k=[top_k] * 32, top_p=[top_p] * 32), 32
        )
        self.generator.reset_sampling_params(params)

        padded = torch.full((1, 1, 32, self.padded_vocab_size), float("-inf"))
        padded[0, 0, :, : self.vocab_size] = flat
        logits_tt = ttnn.from_torch(
            padded,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sampled = self.generator.sample(logits_tt, enable_trace=False)
        tokens_tt = sampled[0] if isinstance(sampled, tuple) else sampled
        tokens = ttnn.to_torch(tokens_tt).reshape(-1)[:32].long()
        return int(tokens[0].item())
