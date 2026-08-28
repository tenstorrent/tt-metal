# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `decoder_head` (Qwen3 `lm_head`) for FLUX.2-klein-9B's text encoder.

The component is a single `nn.Linear(hidden=4096 -> vocab=151936, bias=False)`.

Tensor-parallel scheme, derived from `models/tt_transformers/tt/lm_head.py`:

  * The head is COLUMN-parallel. Its output axis is the vocabulary: every logit
    is an independent dot product of the (full) hidden vector with one vocab
    row, so a chip that owns a disjoint slice of vocab columns can compute
    those logits on its own with NO cross-chip traffic during the matmul. The
    K axis (hidden=4096) stays whole on every chip, so the activation is simply
    replicated -- there is nothing to reduce.
  * The collective therefore comes AFTER the matmul and is an `all_gather` over
    the vocab axis (the column-parallel rule), not an all_reduce: each chip
    holds a *slice* of the answer, not a *partial sum* of it.

Vocab padding (the reason this isn't a one-liner): TILE_LAYOUT needs each chip's
column count to be a multiple of 32, but 151936 / 8 = 18992 is not. So the vocab
is padded up to the next multiple of `TILE_SIZE * num_devices` (152064 at TP=8)
with ZERO rows before sharding -- exactly what `compute_padded_vocab_size` /
`LMHead.__init__` do upstream. Zero columns contribute zero logits, so the pad
cannot perturb the real ones; they are sliced off after the gather. At TP=1 the
padding is a no-op because 151936 is already tile-aligned.

Nothing else here is a big matmul weight, so nothing else is split. The math is
unchanged from the torch reference -- only placement differs -- so the gathered
output still matches the single-device golden.
"""
from __future__ import annotations

import torch

import ttnn


class TtDecoderHead:
    def __init__(self, mesh_device, weight, vocab_size, padded_vocab_size, hidden_size, num_devices) -> None:
        self.mesh_device = mesh_device
        self.weight = weight
        self.vocab_size = vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.num_devices = num_devices
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # ---------------------------------------------------------------- build

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("decoder_head stub needs the torch module to source its weights")

        weight_torch = _linear_weight(torch_module)  # [vocab, hidden]
        vocab_size, hidden_size = int(weight_torch.shape[0]), int(weight_torch.shape[1])

        num_devices = _num_devices(device)
        padded_vocab_size = _nearest_multiple(vocab_size, ttnn.TILE_SIZE * num_devices)

        # torch nn.Linear stores [out, in]; ttnn matmuls x @ W want [in, out].
        # Keep the host copy in bf16 (the on-device dtype) so padding this
        # 4096 x 152064 table doesn't need a gigabyte-scale fp32 temporary.
        w = weight_torch.detach().to(torch.bfloat16).t()
        if padded_vocab_size > vocab_size:
            # Zero columns -> zero logits: pure placement padding, no math change.
            w = torch.cat(
                [w, torch.zeros(hidden_size, padded_vocab_size - vocab_size, dtype=w.dtype)],
                dim=-1,
            )
        else:
            w = w.contiguous()

        weight = ttnn.from_torch(
            w,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # Column-parallel: chip d owns vocab columns [d*P/N, (d+1)*P/N).
            mesh_mapper=_shard_mapper(device, num_devices, dim=-1),
        )

        return cls(
            mesh_device=device,
            weight=weight,
            vocab_size=vocab_size,
            padded_vocab_size=padded_vocab_size,
            hidden_size=hidden_size,
            num_devices=num_devices,
        )

    # -------------------------------------------------------------- forward

    def __call__(self, hidden_states, *args, **kwargs):
        x = hidden_states
        in_shape = list(x.shape)
        seq_len = int(in_shape[-2]) if len(in_shape) >= 2 else 1
        batch = int(in_shape[-3]) if len(in_shape) >= 3 else 1
        x = ttnn.reshape(x, (1, batch, seq_len, self.hidden_size))

        # ---- column-parallel projection: this chip's own slice of the vocab.
        # K (hidden) is whole on every chip, so no pre-collective is needed.
        logits = ttnn.linear(
            x,
            self.weight,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        # ---- the column-parallel collective: concatenate the vocab slices so
        # every chip ends up holding the whole logit vector.
        if self.num_devices > 1:
            logits = ttnn.all_gather(logits, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # ---- drop the alignment padding (151936 is tile-aligned, so this slice
        # is legal on TILE_LAYOUT).
        if self.padded_vocab_size > self.vocab_size:
            logits = ttnn.slice(
                logits,
                (0, 0, 0, 0),
                (1, batch, seq_len, self.vocab_size),
            )

        out_shape = tuple(in_shape[:-1]) + (self.vocab_size,)
        return ttnn.reshape(logits, out_shape)


# ------------------------------------------------------------------ helpers


def _num_devices(device):
    try:
        return int(device.get_num_devices())
    except AttributeError:
        return 1


def _shard_mapper(device, num_devices, dim):
    if num_devices <= 1:
        return None
    return ttnn.ShardTensorToMesh(device, dim=dim)


def _nearest_multiple(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


def _linear_weight(torch_module):
    """The [out, in] weight of this head, however the state dict names it."""
    w = getattr(torch_module, "weight", None)
    if isinstance(w, torch.Tensor) and w.dim() == 2:
        return w
    sd = torch_module.state_dict()
    for key in ("weight", "<root>.weight"):
        if key in sd:
            return sd[key]
    for key, val in sd.items():
        if key.endswith("weight") and isinstance(val, torch.Tensor) and val.dim() == 2:
            return val
    raise RuntimeError(f"decoder_head: no 2-D linear weight in state dict keys {list(sd)}")


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtDecoderHead.build(device, torch_module)


# Module-level shim with the component's lowercase slug name, for legacy SMOKE/PCC tests.
def decoder_head(device, torch_module=None):
    return TtDecoderHead.build(device, torch_module)
