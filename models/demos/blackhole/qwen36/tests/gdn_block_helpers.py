# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test helpers: the per-layer KV cache entries and the ``mamba_state`` that vllm-tt-plugin hands Qwen36ForCausalLM,
built without vLLM or the plugin for a model of any layer count. No ttnn import."""

from dataclasses import dataclass

import torch

from models.demos.blackhole.qwen36.tt.kv_cache_spec import choose_block_size, gdn_state_shapes


@dataclass(frozen=True)
class GroupState:
    """Same fields as vllm_tt_plugin.mamba_state.MambaGroupState."""

    layer_names: tuple
    windows: torch.Tensor
    start_blocks: torch.Tensor
    checkpoint_blocks: torch.Tensor | None = None


@dataclass(frozen=True)
class StepState:
    """Same fields as vllm_tt_plugin.mamba_state.MambaStepState."""

    groups: tuple
    checkpoint_positions: torch.Tensor | None = None


class PluginLayout:
    """KV cache groups as vLLM's ``_get_kv_cache_groups_uniform_page_size`` forms them for a hybrid model: the attention
    layers in one group, the GDN layers in groups of the same size (layers[g::num_groups]), and the layer at position j
    of every group in layer slot j (the plugin's ``tensor_idx``)."""

    def __init__(self, attention, kv_heads, head_dim, dims, kv_bf8, num_blocks, block_size=None):
        attn = [i for i, a in enumerate(attention) if a]
        gdn = [i for i, a in enumerate(attention) if not a]
        assert attn and len(gdn) % len(attn) == 0, "the GDN layers must fill whole groups of the attention-group size"
        n_groups = len(gdn) // len(attn)
        self.gdn_groups = [gdn[g::n_groups] for g in range(n_groups)]
        if block_size is None:
            block_size, _ = choose_block_size(kv_heads=kv_heads, head_dim=head_dim, kv_bf8=kv_bf8, dims=dims)
        self.block_size = block_size
        self.num_blocks = num_blocks
        shapes, dtypes = gdn_state_shapes(dims)
        specs = [None] * len(attention)
        for j, i in enumerate(attn):
            specs[i] = ((num_blocks, kv_heads, block_size, head_dim), torch.bfloat16, j)
        for group in self.gdn_groups:
            for j, i in enumerate(group):
                specs[i] = (shapes, dtypes, j)
        self.per_layer_specs = specs
        self.group_names = [tuple(f"model.layers.{i}.linear_attn" for i in g) for g in self.gdn_groups]

    @classmethod
    def for_model(cls, model, num_blocks, block_size=None, kv_bf8=False):
        from models.demos.blackhole.qwen36.tt.kv_cache_spec import GdnDims

        a = model.args
        dims = GdnDims(nk=a.gdn_nk_tp, nv=a.gdn_nv_tp, dk=a.gdn_dk, dv=a.gdn_dv, kconv=a.gdn_conv_kernel_size)
        attention = [layer.is_full_attention for layer in model.layers]
        return cls(attention, a.n_local_kv_heads, a.head_dim, dims, kv_bf8, num_blocks, block_size)

    @property
    def num_groups(self):
        return len(self.gdn_groups)

    def step_state(self, state_blocks, rows, start_blocks=None, checkpoints=None):
        """mamba_state for ``rows`` submitted rows; ``state_blocks[r]`` is row r's destination block per GDN group, or
        None for a padding row (block 0). ``start_blocks[r]``: row r's start block per group, None (block 0) for a row
        at start_pos 0, or the string "same" for its destination blocks (mamba_cache_mode "none" after position 0);
        all None when omitted. ``checkpoints[r]``: (token count, block per group) of row r's prefill checkpoint, or
        None."""
        start_blocks = start_blocks or [None] * len(state_blocks)
        checkpoints = checkpoints or [None] * len(state_blocks)
        has_checkpoint = any(c is not None for c in checkpoints)
        groups = []
        for g, names in enumerate(self.group_names):
            windows = torch.zeros(rows, 1, dtype=torch.int32)
            starts = torch.zeros(rows, dtype=torch.int32)
            ckpt = torch.zeros(rows, 1, dtype=torch.int32) if has_checkpoint else None
            for r, blocks in enumerate(state_blocks):
                if blocks is not None:
                    windows[r, 0] = blocks[g]
                start = blocks if start_blocks[r] == "same" else start_blocks[r]
                if start is not None:
                    starts[r] = start[g]
                if checkpoints[r] is not None:
                    ckpt[r, 0] = checkpoints[r][1][g]
            groups.append(GroupState(layer_names=names, windows=windows, start_blocks=starts, checkpoint_blocks=ckpt))
        positions = None
        if has_checkpoint:
            positions = torch.zeros(rows, 1, dtype=torch.int32)
            for r, c in enumerate(checkpoints):
                if c is not None:
                    positions[r, 0] = c[0]
        return StepState(groups=tuple(groups), checkpoint_positions=positions)
