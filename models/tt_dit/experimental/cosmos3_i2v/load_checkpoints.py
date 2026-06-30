# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""HF safetensors → TT weight loader.

The transformer is 27 safetensors shards (~128 GB FP16). Loading must be
chunked to avoid host OOM. Reuse the pattern from
`models/tt_transformers/tt/load_checkpoints.py`.

Weight key remapping (per safetensors index):
    layers.<i>.self_attn.to_{q,k,v,out}         → text expert QKV-O
    layers.<i>.self_attn.add_{q,k,v}_proj       → diffusion expert QKV
    layers.<i>.self_attn.to_add_out             → diffusion expert O
    layers.<i>.self_attn.norm_{q,k}             → text QK-norm
    layers.<i>.self_attn.norm_added_{q,k}       → diffusion QK-norm
    layers.<i>.mlp.{gate,up,down}_proj          → text MLP
    layers.<i>.mlp_moe_gen.{gate,up,down}_proj  → diffusion MLP
    layers.<i>.input_layernorm[_moe_gen]        → pre-attn norm per expert
    layers.<i>.post_attention_layernorm[_moe_gen] → post-attn norm per expert
    embed_tokens.weight                          → shared embed table
    proj_in / proj_out                           → latent <-> token projections
    time_embedder.*                              → timestep embedding
    norm / norm_moe_gen                          → final pre-output norms
    lm_head.weight                               → text output head (unused for I2V)
"""

from __future__ import annotations


def load_cosmos3_weights(*args, **kwargs):
    """Stub — Phase 1."""
    raise NotImplementedError("Phase 1 stub — see plan Phase 1 step 6")
