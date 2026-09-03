# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B token embedding — a **replicated** table, no vocab sharding, no collective.

HF anchor: ``transformers.models.llama.modeling_llama.LlamaModel.embed_tokens``
(a plain ``nn.Embedding(vocab_size, hidden_size)``).
Template: ``models/demos/gpt_oss_d_p/tt/model.py:77`` (the ``substate`` + ``unsqueeze`` pair),
``:84`` (``as_tensor``), ``:88`` (``ROW_MAJOR_LAYOUT``), ``:315`` (the ``ttnn.embedding`` call with
``layout=TILE_LAYOUT, dtype=bfloat16``).

**Why replicated (``DEC-015``).** ``128256 * 4096 * 2 B = 1.00 GiB`` per chip, which fits, and
sharding the table costs an all-gather per chunk plus a second layout to debug for no correctness
gain. ``models/demos/gpt_oss_d_p/tt/model.py:82`` carries the same deferral as a TODO. The
replicated table is also what makes residual **scheme A** natural: the embedding output is already
full-width on every TP chip, so nothing has to be gathered before the first norm (``DEC-018``,
Appendix F.5).

**Output dtype is bfloat16, deliberately, not the model's ``weight_dtype``.** This tensor seeds the
residual stream, and ``bfloat8_b``'s per-tile shared exponent crushes small channels once the
massive-activation outliers appear later in the stack — the reasoning recorded verbatim at
``models/demos/gpt_oss_d_p/tt/model.py:308-310``. ``03_OUTLINE.md`` §3.14 fixes the table itself at
bf16 for the same reason: an embedding row *is* an activation, not a projection.

No collective anywhere in this file (``bringup_log/04_CCL_PLAN.md`` §7, row 1).
"""

from __future__ import annotations

import ttnn
from models.demos.llama31_8b_d_p.utils.general_utils import get_cache_file_name


class Embedding:
    """Replicated token-embedding lookup: ``[1, 1, 1, S_loc]`` uint32 -> ``[1, 1, S_loc, hidden]`` bf16."""

    def __init__(self, mesh_device, hf_config, state_dict, *, mesh_config, tensor_cache_path=None):
        """
        Args:
            mesh_device: the ttnn mesh device.
            hf_config: a ``LlamaHFConfig``. ``vocab_size`` and ``hidden_size`` are read, and only to
                assert the table's shape.
            state_dict: the already-stripped ``model.embed_tokens.*`` sub-dict, i.e.
                ``{"weight": [vocab_size, hidden_size]}``. ``{}`` means cache-only mode, which
                requires ``tensor_cache_path``.
            mesh_config: ``MeshConfig``. Accepted for interface uniformity and asserted-on: the
                table is replicated, so no mapper is built from it. Passing it keeps every module in
                this package to one constructor shape (``03_OUTLINE.md`` §1 convention 1) and makes
                the *absence* of a mapper an explicit statement rather than an omission.
            tensor_cache_path: directory for the tilized weight cache, or ``None``.
        """
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.vocab_size = hf_config.vocab_size
        self.hidden_size = hf_config.hidden_size

        if state_dict:
            assert "weight" in state_dict, (
                "Embedding needs the stripped sub-dict {'weight': ...}; pass "
                "substate(sd, 'model.embed_tokens'), not the whole state dict"
            )
            weight = state_dict["weight"]
            assert weight.shape == (self.vocab_size, self.hidden_size), (
                f"embedding table is {tuple(weight.shape)}, expected "
                f"{(self.vocab_size, self.hidden_size)} from hf_config"
            )
            weight = weight.unsqueeze(0).unsqueeze(0)
        else:
            assert tensor_cache_path, (
                "Embedding got an empty state_dict and no tensor_cache_path; there is nothing to "
                "load from (cache-only mode needs the cache, DEC-038)"
            )
            weight = None

        # ROW_MAJOR: ttnn.embedding gathers rows from an untilized table and tilizes the *output*.
        # mesh_mapper is deliberately absent -> ttnn replicates. DEC-015.
        self.weight = ttnn.as_tensor(
            weight,
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "model.embed_tokens.weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, tokens):
        """``tokens``: ``[1, 1, 1, S_loc]`` uint32 ROW_MAJOR -> ``[1, 1, S_loc, hidden]`` bf16 TILE.

        Does not deallocate ``tokens`` — the caller owns it (``tt/model.py`` frees it right after,
        as ``models/demos/gpt_oss_d_p/tt/model.py:316`` does).
        """
        out = ttnn.embedding(tokens, self.weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        if len(out.shape) == 3:
            out = ttnn.unsqueeze_to_4D(out)
        return out
