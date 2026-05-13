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
    def _per_device_shape(t):
        """Return the *per-device* (i.e. ttnn-native) shape of ``t`` as a tuple.

        ``TorchTTNNTensor.shape`` is the *logical* shape (composed across the
        mesh) -- comparing those to decide if ``ttnn.add`` is safe would let
        through a replicated [B, S, H] vs col-sharded logical [B, S, H]
        mismatch, which is exactly the bug the wrapper had originally.
        Reading from the inner ``ttnn.Tensor.shape`` keeps us honest.
        """
        if isinstance(t, ttnn.Tensor):
            return tuple(int(t.shape[i]) for i in range(len(t.shape)))
        if isinstance(t, TorchTTNNTensor) and t.ttnn_tensor is not None:
            inner = t.ttnn_tensor
            return tuple(int(inner.shape[i]) for i in range(len(inner.shape)))
        return None

    @staticmethod
    def _configs_compatible(cfg_a, cfg_b, per_device_shape):
        """Returns True when both configs describe the same per-device sharding.

        Both ``None``: no distributed sharding on either side -- trivially safe.
        One side ``None``: cannot confirm match.
        Both non-``None``: compare the *logical* shapes produced from the common
        per-device shape. On T3K (1, 8) all col-shard configs -- whether built
        with 1D ``shard_tensor_to_mesh_mapper`` or 2D ``ShardTensor2dMesh`` --
        produce the same logical shape (``mesh_rows=1`` makes the batch scaling
        trivial), and they place the same hidden-dim slice on the same physical
        device. So matching logical shapes is the correct compatibility signal
        for all (1, N) meshes in production use.
        """
        if cfg_a is None and cfg_b is None:
            return True
        if cfg_a is None or cfg_b is None:
            return False
        return cfg_a.get_logical_shape(per_device_shape) == cfg_b.get_logical_shape(per_device_shape)

    def _residual_add(self, residual, hidden_states, force_torch=False):
        """Compute ``residual + hidden_states`` with a layout-safe fast path.

        The original wrapper called ``ttnn.add`` whenever both sides could be
        unwrapped to a ``ttnn.Tensor``, but ``ttnn.add`` operates per-device
        and is blind to high-level distributed configs (``logical_shape_fn``,
        mesh mapper / composer choice). When the two operands disagree on
        per-device layout -- e.g. the layer-0 case where ``residual`` is the
        raw nn.Embedding output (replicated full-hidden when wrapped) and
        ``hidden_states`` is the attention output (col-sharded along the last
        dim with a ``logical_shape_fn``) -- the per-device shapes don't line
        up and the silent-wrong-add cascades into gibberish across all 40
        layers.

        The fast path here is strictly opt-in: we only call ``ttnn.add``
        when (a) both inputs are TTNN-backed, (b) both share the same device
        and the same per-device shape, and (c) both ``DistributedTensorConfig``s
        produce the same *logical* shape from that per-device shape (i.e.
        they describe the same composed layout across the mesh). The third
        check is structural rather than identity -- each module's
        ``set_output_tensors_config_impl`` builds fresh ttnn mesh-mapper /
        composer instances per forward, so an identity check would never let
        the fast path fire and we'd fall back to torch ``+`` for every
        residual, which is exactly the ~10240-call / ~20s
        ``_unwrap_to_torch`` overhead the wrapper exists to eliminate.
        Anything that fails any check falls through to the legacy
        ``residual + hidden_states`` path, which the dispatcher reconciles
        via ``unwrap_to_torch`` / per-operand ``mesh_composer`` and re-stages
        the result on device.

        The escape hatch ``TT_QWEN_CPU_DECODER_RESIDUAL_ADD=1`` keeps the torch
        ``+`` path on every call for A/B comparison.
        """
        _dbg = os.environ.get("TT_QWEN_DEBUG_RESIDUAL", "0") == "1"

        if self._use_torch_residual or force_torch:
            if _dbg:
                print(
                    f"[RESIDUAL] layer={self._layer_idx} force_torch={force_torch} use_torch={self._use_torch_residual} -> TORCH+"
                )
            return residual + hidden_states

        a = self._extract_ttnn(residual)
        b = self._extract_ttnn(hidden_states)

        # Both sides need an underlying ttnn.Tensor for the fast path. We
        # intentionally do NOT materialize one side from torch here (the
        # original wrapper did, but that's where the layer-0 layout mismatch
        # came from -- the torch embedding's mesh staging used a different
        # mesh_mapper than the attention output, and ttnn.add couldn't see the
        # disagreement).
        if a is None or b is None:
            if _dbg:
                print(
                    f"[RESIDUAL] layer={self._layer_idx} a={a is not None} b={b is not None} -> TORCH+ (no ttnn tensor)"
                )
            return residual + hidden_states

        # Same device.
        if a.device() != b.device():
            if _dbg:
                print(f"[RESIDUAL] layer={self._layer_idx} -> TORCH+ (device mismatch)")
            return residual + hidden_states

        # Same per-device shape -- guards against replicated-vs-sharded.
        per_dev_a = self._per_device_shape(a)
        per_dev_b = self._per_device_shape(b)
        if per_dev_a != per_dev_b:
            if _dbg:
                print(
                    f"[RESIDUAL] layer={self._layer_idx} shape_a={per_dev_a} shape_b={per_dev_b} -> TORCH+ (shape mismatch)"
                )
            return residual + hidden_states

        # Same logical shape under each config -- two configs that produce
        # different logical shapes from the same per-device shape are
        # describing different mesh layouts and ttnn.add would be wrong.
        # Identity comparison of mesh_mapper / mesh_composer is too strict
        # (each module builds fresh instances per forward), so we compare the
        # *behavior* (logical shape) instead.
        cfg_a = self._extract_dist_config(residual)
        cfg_b = self._extract_dist_config(hidden_states)
        if not self._configs_compatible(cfg_a, cfg_b, per_dev_a):
            if _dbg:
                print(f"[RESIDUAL] layer={self._layer_idx} -> TORCH+ (config incompatible)")
            return residual + hidden_states

        # Require matching TTNN layout (e.g. both TILE) -- ttnn.add on
        # mismatched layouts silently produces wrong results, and a ROW_MAJOR
        # residual (from the CPU-dispatch fallback path) must not be added
        # element-wise with a TILE attention/MoE output.
        if a.get_layout() != b.get_layout():
            if _dbg:
                print(
                    f"[RESIDUAL] layer={self._layer_idx} layout_a={a.get_layout()} layout_b={b.get_layout()} -> TORCH+ (layout mismatch)"
                )
            return residual + hidden_states

        if _dbg:
            print(f"[RESIDUAL] layer={self._layer_idx} shape={per_dev_a} -> FAST PATH (ttnn.add)")
        out_ttnn = ttnn.add(a, b)
        out_wrapped = TorchTTNNTensor(out_ttnn)
        # Both configs are equivalent here; pick either one to propagate so
        # the next layer's residual lookup also takes the fast path.
        if cfg_b is not None:
            out_wrapped.set_distributed_tensor_config(cfg_b)
        elif cfg_a is not None:
            out_wrapped.set_distributed_tensor_config(cfg_a)
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

        # Block-1 residual always goes through torch + regardless of layer.
        # The TTNN embedding output's per-device columns are not guaranteed to
        # be physically aligned with the attention output's columns even when
        # both carry the same distributed config, so ttnn.add silently produces
        # wrong values. This matches the "TT_QWEN_CPU_EMBEDDING=1" baseline
        # (which also lets block-1 bail to torch+) and is empirically correct.
        # Block-2 (post-attn residual + MoE output) is safe for the fast path
        # because both operands come from the same TTNN execution path.
        hidden_states = self._residual_add(residual, hidden_states, force_torch=True)

        # ------- block 2: norm -> MoE -> residual add -------
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        mlp_out = layer.mlp(hidden_states)
        # MoE returns either a tensor or (tensor, router_logits) -- match HF's
        # `if isinstance(..., tuple): hidden_states, _ = hidden_states`.
        hidden_states = self._unpack_attn_output(mlp_out)
        hidden_states = self._residual_add(residual, hidden_states)

        return hidden_states
