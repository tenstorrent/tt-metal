# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN wrapper for ``transformers.models.qwen3_5_moe.Qwen3_5MoeDecoderLayer``.

The HF decoder layer's forward looks like this (paraphrased)::

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.linear_attn / self_attn(hidden_states, ...)
    hidden_states = residual + hidden_states           # <- aten::add.Tensor
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states           # <- aten::add.Tensor
    return hidden_states

When the inner sub-modules (input_layernorm, attention, MoE,
post_attention_layernorm) have been replaced with their TTNN counterparts the
two ``residual + hidden_states`` ops are still ordinary ``aten::add.Tensor``
calls. With the project default CPU dispatcher (``cpu_dispatcher.py``,
``can_dispatch_to_ttnn`` always returns ``False``), every one of these adds
falls back to torch via ``DispatchManager.dispatch_to_torch_wrapper``, which in
turn calls ``unwrap_to_torch`` on every input tensor -- the dominant chunk of
the residual ``_unwrap_to_torch`` cost in the steady-state trace
(``aten::add.Tensor`` shows ~10,240 calls = 40 layers * 2 adds * 128 tokens in
the captured pivot CSV).

This wrapper sequences the same call chain as the HF forward but uses
``ttnn.add`` directly for the residuals. Sub-module calls run through the
already-installed TTNN replacements (and their own captured traces); the
wrapper itself is intentionally a plain ``nn.Module`` so it can live inside
the model's ``nn.ModuleList`` (which validates ``isinstance(child,
nn.Module)`` whenever ``self.layers[:N]`` is sliced).

Wiring
------

The wrapper relies on its sub-modules already being TTNN-replaced when the
wrapper is constructed. Concretely, run ``register_module_replacement_dict``
twice in the test:

1. First pass: register ``Qwen3_5MoeRMSNorm`` -> ``TTNNQwen3MoeRMSNorm``,
   attention classes -> their TTNN ports, and the MoE class -> ``TTNNQwen3MoE``.
   The recursion will find these inside each ``Qwen3_5MoeDecoderLayer`` and
   swap them in place.
2. Second pass: register ``Qwen3_5MoeDecoderLayer`` ->
   ``TTNNQwen3MoeDecoderLayer``. The recursion now replaces each decoder layer
   with this wrapper, which holds the (already-replaced) torch decoder layer
   on ``self._fallback_torch_layer`` and reaches into it via
   ``layer.input_layernorm`` / ``layer.linear_attn`` / etc.

If the second pass is skipped, the module behaves identically to the unwrapped
torch decoder layer (and the TTNN sub-modules continue to work) -- so this
optimization is fully opt-in.
"""

import os
from typing import Optional

import torch
from torch import nn

import ttnn
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor


class TTNNQwen3MoeDecoderLayer(nn.Module):
    """Drop-in replacement for ``Qwen3_5MoeDecoderLayer`` that runs residual adds on TTNN.

    The wrapper deliberately inherits from ``torch.nn.Module`` (not from
    :class:`TTNNModule`) because each decoder layer lives inside an
    ``nn.ModuleList`` on the parent ``Qwen3_5MoeTextModel``. Slicing that
    list (``self.layers[:N]``, used by the HF forward) builds a fresh
    ``ModuleList`` whose ``add_module`` enforces ``isinstance(child,
    nn.Module)``; a non-``Module`` replacement raises a ``TypeError`` at the
    very first ``__init__`` call. ``TTNNModule`` machinery is unnecessary for
    this wrapper anyway -- it owns no weights of its own; the inner sub-modules
    (``input_layernorm``, ``linear_attn`` / ``self_attn``, ``post_attention_layernorm``,
    ``mlp``) keep their original ``TTNNModule`` lifecycle (preprocess /
    move_weights / per-module trace capture).

    Set ``TT_QWEN_CPU_DECODER_RESIDUAL_ADD=1`` to revert each residual add to
    the torch ``+`` operator (useful for A/B comparison).
    """

    def __init__(self):
        super().__init__()
        # Populated in ``from_torch`` so the bare ``cls()`` call succeeds.
        self.layer_type = "full_attention"
        self._use_torch_residual = False

    @classmethod
    def from_torch(cls, torch_layer):
        instance = cls()
        # ``_fallback_torch_layer`` is itself an ``nn.Module`` so this assignment
        # registers it under ``self._modules['_fallback_torch_layer']`` via
        # ``nn.Module.__setattr__``. That keeps ``set_device`` recursion (which
        # walks ``_modules``) reaching the inner TTNN sub-modules so their
        # devices and weights still get configured normally.
        instance._fallback_torch_layer = torch_layer
        instance.layer_type = getattr(torch_layer, "layer_type", "full_attention")
        # Honour the same env-var pattern the other ports expose.
        instance._use_torch_residual = os.environ.get("TT_QWEN_CPU_DECODER_RESIDUAL_ADD", "0").lower() in (
            "1",
            "true",
            "yes",
        )
        if instance._use_torch_residual:
            print("[DEBUG] TT_QWEN_CPU_DECODER_RESIDUAL_ADD=1: using torch + for residual adds")
        return instance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_attn_output(out):
        """Return the activation tensor from an attention layer's output.

        HF self-attention returns ``(attn_output, attn_weights)`` and our TTNN
        ports mirror that contract. The dispatcher / module-run plumbing
        sometimes surfaces the pair as a Python ``list`` rather than a tuple
        (e.g. ``[TTNNTensor, None]``) -- accept both, plus the bare-tensor
        case (``TTNNQwen3LinearAttention.forward`` returns just
        ``output_ttnn``).
        """
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    @staticmethod
    def _extract_ttnn(t):
        """Return the underlying ``ttnn.Tensor`` from `t` if it has one, else None."""
        if isinstance(t, ttnn.Tensor):
            return t
        if isinstance(t, TorchTTNNTensor):
            inner = getattr(t, "ttnn_tensor", None)
            if inner is not None and inner.is_allocated():
                return inner
        return None

    @staticmethod
    def _extract_dist_config(t):
        """Return a TorchTTNNTensor's distributed-tensor config if present."""
        if isinstance(t, TorchTTNNTensor):
            return t.ttnn_distributed_tensor_config
        return None

    @staticmethod
    def _materialize_to_match(t, device, config):
        """Materialize ``t`` as a ttnn.Tensor on ``device`` using ``config``'s mesh layout.

        The mesh mapper from ``config`` matters because the side that already has
        TTNN backing may be col-sharded across the mesh (the steady-state output
        layout of ``TTNNQwen3MoE`` / ``TTNNQwen3*Attention``); replicating ``t``
        instead would yield a per-device shape mismatch on ``ttnn.add``.
        Returns ``None`` if no torch backing can be located (caller should fall
        back to the torch ``+`` operator in that case).
        """
        # If we already have a TTNN-backed wrapper, just hand back the inner tensor
        # (cheaper than going through `to_ttnn` which may re-stage from torch).
        if isinstance(t, TorchTTNNTensor):
            inner = getattr(t, "ttnn_tensor", None)
            if inner is not None and inner.is_allocated():
                return inner
            torch_backing = getattr(t, "elem", None)
        elif isinstance(t, torch.Tensor):
            torch_backing = t
        else:
            return None

        if torch_backing is None:
            return None

        # Pick the mesh_mapper from the matching side's config so the per-device
        # shape lines up with the other operand. Default to replication on a
        # multi-device mesh when no config is available (matches the typical
        # behavior of `_to_ttnn` helpers throughout the symbiote stack).
        if config is not None and getattr(config, "mesh_mapper", None) is not None:
            mesh_mapper = config.mesh_mapper
        elif device is not None and device.get_num_devices() > 1:
            mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        else:
            mesh_mapper = None

        return ttnn.from_torch(
            torch_backing.detach().to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _residual_add(self, residual, hidden_states):
        """Compute ``residual + hidden_states``, preferring native ``ttnn.add``.

        - **Both sides TTNN-backed** (the steady state for layers 1..N-1 once
          the chain is primed): a single ``ttnn.add`` on device, no host work.
        - **One side TTNN-backed** (notably layer 0 where ``residual`` is the
          raw torch embedding while ``hidden_states`` has come back through
          attention/MoE): we materialize the torch side onto the mesh using
          the matching distributed config so per-device shapes line up, then
          run ``ttnn.add``. This breaks the cascade where the torch fallback
          would otherwise produce a torch-backed result and force every later
          layer's residual to also fall back.
        - **Neither side TTNN-backed**: defer to torch ``+`` so the existing
          dispatcher path (and any wrappers) handles it as before.

        The escape hatch ``TT_QWEN_CPU_DECODER_RESIDUAL_ADD=1`` forces the
        torch ``+`` path on every call for A/B comparison.
        """
        if self._use_torch_residual:
            return residual + hidden_states

        a = self._extract_ttnn(residual)
        b = self._extract_ttnn(hidden_states)

        if a is None and b is None:
            # Pure torch on both sides; nothing to win by detouring to device.
            return residual + hidden_states

        # Materialize the side that doesn't have TTNN backing using the other
        # side's distributed config so the mesh-level shapes match.
        if a is None:
            cfg = self._extract_dist_config(hidden_states)
            a = self._materialize_to_match(residual, b.device(), cfg)
            if a is None:
                return residual + hidden_states
        if b is None:
            cfg = self._extract_dist_config(residual)
            b = self._materialize_to_match(hidden_states, a.device(), cfg)
            if b is None:
                return residual + hidden_states

        out_ttnn = ttnn.add(a, b)
        out_wrapped = TorchTTNNTensor(out_ttnn)
        # Propagate the distributed config so the NEXT layer's residual lookup
        # finds a TTNN-backed tensor with a proper logical shape (and thus also
        # takes the fast path here).
        cfg = self._extract_dist_config(hidden_states) or self._extract_dist_config(residual)
        if cfg is not None:
            out_wrapped.set_distributed_tensor_config(cfg)
        return out_wrapped

    # ------------------------------------------------------------------
    # Forward (mirrors transformers' Qwen3_5MoeDecoderLayer.forward)
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        **kwargs,
    ):
        layer = self._fallback_torch_layer

        # ------- block 1: norm -> token mixer -> residual add -------
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            mixer_out = layer.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
            )
            # The TTNN linear attention typically returns just the activation,
            # but be defensive in case it ever surfaces a (output, state) pair
            # (or list, e.g. from a wrapper that lists tuple outputs).
            hidden_states = self._unpack_attn_output(mixer_out)
        elif self.layer_type == "full_attention":
            attn_out = layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            # HF self-attn returns (attn_output, attn_weights); the TTNN ports
            # mirror that contract but the dispatcher / module-run plumbing
            # may surface it as a list (`[TTNNTensor, None]`) rather than a
            # tuple, so accept both via `_unpack_attn_output`.
            hidden_states = self._unpack_attn_output(attn_out)
        else:
            # Defensive: if a future layer_type appears, fall back to the
            # original torch forward verbatim.
            return layer.__class__.forward(
                layer,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self._residual_add(residual, hidden_states)

        # ------- block 2: norm -> MoE -> residual add -------
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        mlp_out = layer.mlp(hidden_states)
        # MoE returns either a tensor or (tensor, router_logits) -- match HF's
        # `if isinstance(..., tuple): hidden_states, _ = hidden_states`.
        hidden_states = self._unpack_attn_output(mlp_out)
        hidden_states = self._residual_add(residual, hidden_states)

        return hidden_states
