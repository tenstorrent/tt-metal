# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `encoder_stack` (Qwen3Model) for FLUX.2-klein-9B's text encoder.

The encoder stack is the transformer body: 36 `Qwen3DecoderLayer`s followed by a
final RMSNorm, taking `inputs_embeds` to encoded hidden states. (The token
embedding is `token_embed`'s component, and the harness feeds this one embeddings
directly, so the lookup is deliberately not repeated here.)

Tensor-parallel scheme, derived from `models/tt_transformers/tt/model.py`:

  * A stack adds NO parallelism of its own. Every layer already takes a
    replicated hidden state in and, thanks to the two row-parallel all_reduces
    inside it, puts a replicated hidden state back -- so the layers compose by
    simply chaining, with NO collective between them. That replicated residual
    stream is the invariant the whole stack rests on; the only weights here that
    aren't inside a layer are the final norm's gamma, which is per-element over
    the model dim and therefore REPLICATED.
  * So this file's real job is to reproduce what `Qwen3Model.forward` does
    AROUND the layers, exactly, on the sharded layers that already graduated:
      - RoPE tables: built on host from the rotary embedding's own `inv_freq`
        buffer (a lookup table -- replicated, never split), following
        `Qwen3RotaryEmbedding.forward`: emb = cat(freqs, freqs), cos/sin scaled
        by `attention_scaling`.
      - The causal mask: `Qwen3Model` builds one via `create_causal_mask` when
        the caller passes none, so the reference IS causal and this port must be
        too, or it would be computing a different function than the golden. It
        is applied as the additive 0 / -1e9 bias the attention port already takes.

The math is unchanged from the torch reference -- only placement differs -- so
the gathered output still matches the single-device golden.
"""
from __future__ import annotations

import torch

import ttnn

from .decoder_layer import TtDecoderLayer


class TtEncoderStack:
    def __init__(
        self,
        mesh_device,
        layers,
        final_norm_gamma,
        inv_freq,
        attention_scaling,
        hidden_size,
        norm_eps,
        num_devices,
    ) -> None:
        self.mesh_device = mesh_device
        self.layers = layers
        self.final_norm_gamma = final_norm_gamma
        self.inv_freq = inv_freq
        self.attention_scaling = attention_scaling
        self.hidden_size = hidden_size
        self.norm_eps = norm_eps
        self.num_devices = num_devices
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # ---------------------------------------------------------------- build

    @classmethod
    def build(cls, device, torch_module, layers=None):
        """`layers` caps the DEPTH built. None means every layer; it is never 0.

        A capped stack is still a MODEL -- the final norm, the RoPE table and every
        distinct op a full layer runs are all still there, just repeated fewer times.
        """
        if torch_module is None:
            raise RuntimeError("encoder_stack stub needs the torch module to source its weights")

        num_devices = _num_devices(device)
        cfg = torch_module.config
        hidden_size = cfg.hidden_size

        depth = cfg.num_hidden_layers if layers is None else int(layers)
        if depth < 1:
            raise RuntimeError(f"layers must be >= 1 (None = every layer); got {layers!r}")
        depth = min(depth, cfg.num_hidden_layers)

        # Each layer shards its own weights; the stack just chains them. Held as a
        # plain list of same-typed elements so the stack is discoverable by a walk.
        layers = [TtDecoderLayer.build(device, layer) for layer in torch_module.layers[:depth]]

        rotary = torch_module.rotary_emb
        return cls(
            mesh_device=device,
            layers=layers,
            final_norm_gamma=_norm_gamma(torch_module.norm.weight, hidden_size, device, num_devices),
            inv_freq=rotary.inv_freq.detach().to(torch.float32),
            attention_scaling=float(rotary.attention_scaling),
            hidden_size=hidden_size,
            norm_eps=torch_module.norm.variance_epsilon,
            num_devices=num_devices,
        )

    # -------------------------------------------------------------- forward

    def __call__(
        self,
        inputs_embeds,
        attention_mask=None,
        position_ids=None,
        position_embeddings=None,
        kv_caches=None,
        cur_pos=None,
        mode="prefill",
        is_causal=None,
        **kwargs,
    ):
        in_shape = list(inputs_embeds.shape)
        seq_len = int(in_shape[-2])
        rows = 1
        for d in in_shape[:-2]:
            rows *= int(d)

        x = inputs_embeds

        if position_embeddings is None:
            # Legacy path: no caller-supplied tables, so build them the way
            # `Qwen3RotaryEmbedding.forward` does and let the attention port upload
            # them. The e2e pipeline never takes this branch -- it hands DEVICE tables
            # produced by the graduated `rotary_embedding` component, because host
            # tensors here would be host compute inside the forward.
            position_embeddings = self._rope_tables(seq_len)

        if is_causal is None:
            # `Qwen3Model` builds a causal mask whenever the caller passes none, so an
            # unmasked port would not be computing the golden's function.
            is_causal = attention_mask is None
        mask = attention_mask if (attention_mask is not None or is_causal) else self._causal_bias(None, seq_len)

        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                position_embeddings=position_embeddings,
                attention_mask=mask,
                kv_cache=(kv_caches[i] if kv_caches is not None else None),
                cur_pos=cur_pos,
                mode=mode,
                is_causal=is_causal,
            )

        x = ttnn.reshape(x, (rows, 1, seq_len, self.hidden_size))
        x = ttnn.rms_norm(
            x,
            epsilon=self.norm_eps,
            weight=self.final_norm_gamma,
            compute_kernel_config=self.compute_kernel_config,
        )
        return ttnn.reshape(x, tuple(in_shape))

    def allocate_kv_caches(self, batch, capacity):
        """One resident (K, V) pair per layer, each already sharded by kv head."""
        return [layer.attention.allocate_kv_cache(batch, capacity) for layer in self.layers]

    # -------------------------------------------------------------- helpers

    def _rope_tables(self, seq_len):
        """(cos, sin) for positions 0..seq_len-1, as `Qwen3RotaryEmbedding` builds them.

        Returned as torch tensors: the attention port uploads them itself, replicated."""
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = (emb.cos() * self.attention_scaling).unsqueeze(0)
        sin = (emb.sin() * self.attention_scaling).unsqueeze(0)
        return cos, sin

    def _causal_bias(self, attention_mask, seq_len):
        """The additive [1, 1, S, S] causal bias, matching `create_causal_mask`.

        `Qwen3Model` builds a causal mask whenever the caller passes none, so an
        unmasked port would not be computing the golden's function. A caller-
        supplied mask is honoured as-is and left to the attention port."""
        if attention_mask is not None:
            return attention_mask
        allowed = torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
        return torch.where(allowed, 0.0, -1e9).reshape(1, 1, seq_len, seq_len)


# ------------------------------------------------------------------ helpers


def _num_devices(device):
    try:
        return int(device.get_num_devices())
    except AttributeError:
        return 1


def _replicate_mapper(device, num_devices):
    if num_devices <= 1:
        return None
    return ttnn.ReplicateTensorToMesh(device)


def _norm_gamma(weight, dim, device, num_devices):
    """ttnn.rms_norm wants gamma as [1, 1, dim // TILE, TILE] in ROW_MAJOR
    (see models/common/rmsnorm.py). Per-element over the full model dim, so
    it is REPLICATED on every chip."""
    return ttnn.from_torch(
        weight.detach().to(torch.bfloat16).reshape(1, 1, dim // ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_replicate_mapper(device, num_devices),
    )


# Module-level `build` — primary test entry point.
def build(device, torch_module=None, layers=None):
    return TtEncoderStack.build(device, torch_module, layers=layers)


# Module-level shim with the component's lowercase slug name, for legacy SMOKE/PCC tests.
def encoder_stack(device, torch_module=None):
    return TtEncoderStack.build(device, torch_module)
