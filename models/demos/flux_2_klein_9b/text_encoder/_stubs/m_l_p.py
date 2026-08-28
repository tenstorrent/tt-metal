# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `m_l_p` (Qwen3MLP) for FLUX.2-klein-9B's text encoder.

The SwiGLU feed-forward block:  down_proj(silu(gate_proj(x)) * up_proj(x)).

Tensor-parallel scheme, derived from `models/tt_transformers/tt/mlp.py` — the
textbook column-then-row pair:

  * `gate_proj` and `up_proj` are COLUMN-parallel. Their outputs feed SiLU and
    the elementwise gate, both per-element, so each chip can compute its own
    1536-wide slice of the intermediate axis with no collective at all. Both are
    split the SAME way, so a chip's gate slice multiplies its own up slice.
  * `down_proj` is ROW-parallel. It reduces the intermediate axis back to the
    model dim, so each chip owns the rows matching the slice it just produced
    and emits a PARTIAL sum over the full 4096 outputs; one `all_reduce` makes
    the result whole and identical on every chip.

12288 / 8 = 1536 = 48 tiles, so the split is exact and tile-aligned, and there
are no biases or norms here to keep replicated.

The canonical `models/tt_transformers/tt/mlp.py::MLP` was not reusable directly:
it needs a fully-populated `ModelArgs` and a live `TT_CCL` collective manager,
neither of which exists in the per-component PCC harness (it hands the stub a
bare device plus the torch module). This is that class's scheme, expressed
against the harness's inputs.

This is the same scheme `decoder_layer.py` applies to its own inline MLP; both
are PCC-gated against the same reference on every run.

The math is unchanged from the torch reference -- only placement differs -- so
the gathered output still matches the single-device golden.
"""
from __future__ import annotations

import torch

import ttnn


class TtMLP:
    def __init__(self, mesh_device, w_gate, w_up, w_down, hidden_size, num_devices) -> None:
        self.mesh_device = mesh_device
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
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
            raise RuntimeError("m_l_p stub needs the torch module to source its weights")

        hidden_act = getattr(torch_module.config, "hidden_act", "silu")
        if hidden_act != "silu":
            raise RuntimeError(f"this MLP port implements the SwiGLU (silu) gate, not hidden_act={hidden_act!r}")

        num_devices = _num_devices(device)
        hidden_size = torch_module.hidden_size
        intermediate_size = torch_module.intermediate_size
        if intermediate_size % (ttnn.TILE_SIZE * num_devices):
            raise RuntimeError(
                f"MLP TP needs intermediate_size divisible by TILE_SIZE*devices: "
                f"intermediate_size={intermediate_size}, devices={num_devices}"
            )

        sd = {k: v.detach().to(torch.bfloat16) for k, v in torch_module.state_dict().items()}

        # torch nn.Linear stores [out, in]; ttnn matmuls x @ W want [in, out].
        return cls(
            mesh_device=device,
            # Column-parallel pair: split the INTERMEDIATE (output) axis.
            w_gate=_to_device(sd["gate_proj.weight"].t(), device, _shard_mapper(device, num_devices, dim=-1)),
            w_up=_to_device(sd["up_proj.weight"].t(), device, _shard_mapper(device, num_devices, dim=-1)),
            # Row-parallel: split the INTERMEDIATE (input) axis the same way.
            w_down=_to_device(sd["down_proj.weight"].t(), device, _shard_mapper(device, num_devices, dim=0)),
            hidden_size=hidden_size,
            num_devices=num_devices,
        )

    # -------------------------------------------------------------- forward

    def __call__(self, x, *args, **kwargs):
        in_shape = list(x.shape)
        seq_len = int(in_shape[-2])
        # Fold EVERY leading dim into the batch axis, not just in_shape[-3]: the block
        # is called with [B, S, H] by the stack, with [B, 1, S, H] from inside a decoder
        # layer and with [1, 1, B, H] on the decode step, and only a product over all
        # leading dims reshapes all three without changing the element count.
        batch = 1
        for d in in_shape[:-2]:
            batch *= int(d)
        h = ttnn.reshape(x, (batch, 1, seq_len, self.hidden_size))

        # ---- column-parallel: this chip's own slice of the intermediate axis.
        gate = ttnn.linear(
            h,
            self.w_gate,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        up = ttnn.linear(
            h,
            self.w_up,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        # SiLU and the gate are per-element, so they need no collective.
        act = ttnn.multiply(ttnn.silu(gate), up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        # ---- row-parallel: every chip produces a PARTIAL sum over the model dim...
        out = ttnn.linear(
            act,
            self.w_down,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(act)

        # ...and the all_reduce turns those partials into the whole answer.
        if self.num_devices > 1:
            out = ttnn.all_reduce(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return ttnn.reshape(out, tuple(in_shape))


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


def _to_device(weight, device, mesh_mapper):
    return ttnn.from_torch(
        weight.contiguous(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtMLP.build(device, torch_module)


# Module-level shim with the component's lowercase slug name, for legacy SMOKE/PCC tests.
def m_l_p(device, torch_module=None):
    return TtMLP.build(device, torch_module)
