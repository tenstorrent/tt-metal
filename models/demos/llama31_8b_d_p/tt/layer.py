# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B decoder layer.

::

    residual = x
    h = input_layernorm(x)          -> Attention(...) -> h
    x = residual + h
    residual = x
    h = post_attention_layernorm(x) -> MLP(h)         -> h
    x = residual + h

HF anchor: ``transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward``.
Template: ``models/demos/gpt_oss_d_p/tt/layer.py:46`` (class), ``:65``/``:72`` (the two norms with
``substate`` + a per-module cache path), ``:98`` (``AttentionConfig`` from ``hf_config``), ``:111``
(``Attention``), ``:126`` (``__call__``), ``:137-140`` (the ``ttnn.move`` guard), the flow at
``:143-175``, ``:19`` (``_DELTA_PROBE``), ``:22`` (``_delta_stats``).

**Deleted vs the template** (``03_OUTLINE.md`` §3.16):

* the MoE ``MLP`` kwargs (``layer.py:60-62``, ``:82-93``) — Llama's FFN is dense SwiGLU, so
  ``expert_weight_dtype`` / ``use_ep_moe`` / ``ep_seq_len_per_chip`` are dead vocabulary;
* ``layer_types`` / ``is_sliding`` (``:96``, ``:105``, ``:119``) — every Llama layer is identical
  full-causal attention, so there is no per-layer config copy and no layer-type branch;
* ``position_idx`` (``:130``) — unused on the prefill path;
* ``max_local_batch_size`` (``:59``) — never read by anything this layer builds.

**Kept, deliberately:**

* the ``ttnn.move(hidden_states)`` re-allocation guard above 32K tokens and every eager
  ``deallocate(True)`` — both are load-bearing for long-context DRAM pressure, not tidiness
  (``BRINGUP_RECIPE.md:751-752``);
* the delta probe, behind ``LLAMA31_8B_DELTA_PROBE`` (``DEC-041``). Appendix E measured the
  ``tt_transformers`` decoder oracle at **0.9999985** — *higher* than either of its own sublayers
  (attention 0.9996099, MLP 0.9995823) — because the residual stream dominates the correlation. A
  layer-level PCC therefore **partially launders a degraded sublayer**, and magnitude ratios are
  what localise it. In a 32-layer stack this probe is the difference between "the model is at 0.997,
  somewhere" and "layer 17's MLP delta is 6x its neighbours'".

Every intermediate on the residual path is **full width** under scheme A (``DEC-018``); nothing in
this file changes a tensor's width.
"""

from __future__ import annotations

import os

import ttnn
from models.demos.llama31_8b_d_p.tt.attention import Attention, ProgramConfig, attention_config_from_hf
from models.demos.llama31_8b_d_p.tt.mlp import MLP
from models.demos.llama31_8b_d_p.tt.model_config import ModelArgs
from models.demos.llama31_8b_d_p.tt.rms_norm import RMSNorm
from models.demos.llama31_8b_d_p.utils.general_utils import get_cache_file_name
from models.demos.llama31_8b_d_p.utils.substate import substate

# One env var, off by default (03_OUTLINE.md §1 convention 10 budgets exactly two for this package;
# this is the first). Documented in README.md — DEC-041.
_DELTA_PROBE = os.environ.get("LLAMA31_8B_DELTA_PROBE", "") != ""

# Above this sequence length the layer input is re-allocated before use. Carried verbatim from
# gpt_oss_d_p/tt/layer.py:138; 32K tokens is where DRAM fragmentation starts to bite on this arch.
_MOVE_GUARD_SEQ_LEN = 32 * 1024


def _delta_stats(tag, layer_idx, t):
    """Log L2 / mean-abs / signed-mean / max-abs of one residual delta, from device 0's shard.

    Behind ``LLAMA31_8B_DELTA_PROBE``. Copied in shape from
    ``models/demos/gpt_oss_d_p/tt/layer.py:22``, and it reads exactly the four statistics that
    separate the failure modes a residual-dominated PCC hides:

    * ``L2`` growing faster than its neighbours' -> that sublayer is the one drifting;
    * ``signed_mean`` growing monotonically -> a *directional* bias accumulating (the fingerprint of
      a per-layer logic error, e.g. a wrong RoPE offset, rather than of rounding);
    * ``max|x|`` spiking -> the massive-activation outliers that make a bf8_b residual unsafe.

    Wrapped in ``try/except`` on purpose: a probe must never be able to fail a run.
    """
    try:
        from loguru import logger

        d0 = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
        logger.info(
            f"[delta-probe L{layer_idx:>2}] {tag}: L2={d0.norm():.3f}  mean|x|={d0.abs().mean():.4f}  "
            f"signed_mean={d0.mean():.5f}  max|x|={d0.abs().max():.3f}"
        )
    except Exception as e:  # never let the probe break a run
        from loguru import logger

        logger.warning(f"[delta-probe] failed at L{layer_idx} {tag}: {e}")


class DecoderLayer:
    """One Llama decoder layer: two norms, GQA attention, dense SwiGLU, two residual adds."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        layer_idx,
        *,
        mesh_config,
        ccl_manager,
        program_config=None,
        transformation_mats=None,
        max_seq_len=1024,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
        sequence_parallel=False,
        scatter_output=False,
        compute_kernel_config=None,
    ):
        """
        Args:
            mesh_device: the ttnn mesh device.
            hf_config: a ``LlamaHFConfig`` (``tt/model_config.py``). An OBJECT, never a dict
                (``DEC-009``).
            state_dict: the already-stripped ``model.layers.<i>.*`` sub-dict in HF layout — i.e.
                keys ``input_layernorm.weight``, ``self_attn.q_proj.weight``, ``mlp.gate_proj.weight``,
                ... Splitting it further is this class's job. ``{}`` means cache-only mode, which
                requires ``tensor_cache_path`` (``DEC-038``).
            layer_idx: this layer's index. Used for the per-layer KV-cache slot, the cache path and
                the delta probe's label.
            mesh_config: ``MeshConfig``.
            ccl_manager: ``CCLManager``; unused at TP=1 and SP=1.
            program_config: :class:`~.attention.config.ProgramConfig`. ``None`` builds the default
                (SDPA grid pinned at 8x8 — ``DEC-012`` / Appendix F.8). **One instance is shared by
                every layer** when the caller passes one, which is what ``tt/model.py`` does: it is
                a pure value object and Llama has no per-layer variation.
            transformation_mats: ``{"prefill": tensor}`` from ``tt/rope.build_transformation_mat``.
            max_seq_len: the attention config's ``max_seq_len``.
            weight_dtype: on-device projection dtype (default ``bfloat8_b``). Norm gains stay bf16
                regardless (``tt/rms_norm.py``, convention 11).
            tensor_cache_path: directory for the tilized weight cache, or ``None``.
            sequence_parallel: SP prefill (P8). Forwarded into ``AttentionConfig``.
            scatter_output: residual scheme (``DEC-018``). ``False`` = scheme A everywhere.
            compute_kernel_config: matmul compute config for the MLP. ``None`` lets ``MLP`` build the
                package default (HiFi4 + ``fp32_dest_acc_en=True``). Attention builds its own from
                ``program_config``, whose default is the same, so leaving both ``None`` is correct
                and not a precision hole (``DEC-031``).
        """
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.layer_idx = layer_idx
        self.scatter_output = scatter_output

        # scheme B (scatter_output=True) needs the norms to run distributed on a TP-sharded
        # residual, which is the dormant branch tt/rms_norm.py owns and P8 exercises. Refuse the
        # half-wired combination loudly rather than silently norming a shard as if it were full
        # width, which produces a plausible tensor with the wrong scale (DEC-024 / Appendix F.5).
        assert not scatter_output, (
            "DecoderLayer(scatter_output=True) is residual scheme B, which additionally requires "
            "RMSNorm(is_distributed=True) on both norms and a sharded residual add. This package "
            "ships scheme A (DEC-018); scheme B is P8's, and wiring only the attention/MLP half "
            "would norm a hidden/tp shard as if it were full width — silently wrong, not an error."
        )

        norm_kwargs = dict(mesh_config=mesh_config, is_distributed=False)
        self.input_layernorm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "input_layernorm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "input_layernorm"),
            **norm_kwargs,
        )
        self.post_attention_layernorm = RMSNorm(
            mesh_device,
            hf_config,
            substate(state_dict, "post_attention_layernorm"),
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "post_attention_layernorm"),
            **norm_kwargs,
        )

        self.mlp = MLP(
            mesh_device,
            hf_config,
            substate(state_dict, "mlp"),
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "mlp"),
            weight_dtype=weight_dtype,
            scatter_output=scatter_output,
            compute_kernel_config=compute_kernel_config,
        )

        # attention_config_from_hf is the ONE place model dimensions cross into tt/attention/
        # (DEC-036), so this file never reads hf_config for the attention block itself.
        self.self_attn = Attention(
            mesh_device,
            attention_config_from_hf(hf_config, max_seq_len=max_seq_len, sequence_parallel=sequence_parallel),
            substate(state_dict, "self_attn"),
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            program_config=program_config if program_config is not None else ProgramConfig(),
            layer_idx=layer_idx,
            transformation_mats=transformation_mats,
            weight_dtype=weight_dtype,
            tensor_cache_path=get_cache_file_name(tensor_cache_path, "self_attn"),
            scatter_output=scatter_output,
        )

    @staticmethod
    def state_dict_prefix(layer_idx) -> str:
        """The HF key prefix whose ``substate`` this class expects — ``model.layers.<i>.``."""
        return ModelArgs.get_state_dict_prefix("layer", layer_idx)

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        kv_cache=None,
        user_id=0,
        batch_size=1,
        cached_len=0,
        indexed_rope=False,
    ):
        """``[1, 1, batch*S_loc, hidden]`` bf16 TILE -> the same shape.

        Args:
            hidden_states: the residual stream coming in. **Consumed** — this layer deallocates it
                and returns a new tensor; do not use the handle afterwards.
            position_embeddings: ``[cos, sin]``, Meta/interleaved, from
                ``tt/rope.build_prefill_rope`` (per chunk) or ``build_indexed_rope`` (whole cache,
                with ``indexed_rope=True``).
            kv_cache: a ``LlamaKVCache`` or ``None``.
            user_id: cache slot for the per-user write.
            batch_size: users packed on the sequence dim.
            cached_len: valid prefix already in the cache before this chunk.
            indexed_rope: use the on-device indexed RoPE.
        """
        assert position_embeddings is not None, (
            f"DecoderLayer L{self.layer_idx} got position_embeddings=None. Llama applies FULL rotary "
            f"to Q and K in every layer; a None here would run attention with no positional "
            f"information at all and still return a correctly-shaped tensor. Build the tables with "
            f"tt/rope.build_prefill_rope (per chunk) or build_indexed_rope (whole cache)."
        )

        seqlen = hidden_states.shape[-2]
        if seqlen > _MOVE_GUARD_SEQ_LEN:
            # Re-allocate the residual to a fresh DRAM block before the layer's own allocations
            # fragment around it. Load-bearing for long-context prefill (BRINGUP_RECIPE.md:751).
            hidden_states = ttnn.move(hidden_states)

        # ---- attention sublayer -------------------------------------------------------
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            normed,
            rope_mats=position_embeddings,
            kv_cache=kv_cache,
            user_id=user_id,
            batch_size=batch_size,
            cached_len=cached_len,
            indexed_rope=indexed_rope,
        )
        # attention_forward already frees its input; this is the idempotent second call the
        # template also makes (``models/demos/gpt_oss_d_p/tt/layer.py:156``) and is a no-op on a freed tensor.
        normed.deallocate(True)

        if _DELTA_PROBE:
            _delta_stats("attn_out", self.layer_idx, hidden_states)

        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)

        # ---- MLP sublayer -------------------------------------------------------------
        residual = hidden_states
        normed = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(normed)
        normed.deallocate(True)

        if _DELTA_PROBE:
            _delta_stats("mlp_out ", self.layer_idx, hidden_states)

        hidden_states = ttnn.add(residual, hidden_states, output_tensor=hidden_states)
        residual.deallocate(True)

        return hidden_states
