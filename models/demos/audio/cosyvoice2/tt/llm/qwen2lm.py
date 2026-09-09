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

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.decoder import TransformerBlock
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.embedding import Embedding
from models.tt_transformers.tt.rope import HfRotarySetup, RotarySetup


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
