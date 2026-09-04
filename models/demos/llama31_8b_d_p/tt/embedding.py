# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Token embedding for Llama-3.1-8B: a **replicated** `[128256, 4096]` table, bf16 out.

**HF anchor:** `transformers.models.llama.modeling_llama.LlamaModel.embed_tokens`
(`torch.nn.Embedding`). **Template:** `models/demos/gpt_oss_d_p/tt/model.py:84-91` (the replicated
`ttnn.as_tensor`) and `:315-318` (the `ttnn.embedding` + `unsqueeze_to_4D` call), lifted out of
`Model.__init__` into its own module so it can own a test (`bringup_log/03_OUTLINE.md` `[DEV-5]`).

**Replicated, not TP-sharded** (`DEC-024`): it is the only option with no collective at all, and
under residual scheme A (`DEC-025`) the layer stack wants a full-width `[1, 1, S_loc, 4096]`
residual anyway. The alternative — `models/demos/minimax_m3/tt/parallel_embedding.py:80`'s
vocab-sharded table plus an all-gather — was evaluated and not taken; the cost is 1 GB of bf16
table on every chip.

**The output is bf16, never bf8_b** (`DEC-022`): this tensor *is* the residual stream, the one
tensor all 32 layers accumulate into, and bf8_b's per-tile shared exponent crushes small channels
once a few large activations appear (`models/demos/gpt_oss_d_p/tt/model.py:313-315` carries the
same note).
"""

import ttnn
from models.demos.gpt_oss_d_p.utils.general_utils import get_cache_file_name

from .config import MeshConfig


class Embedding:
    """`[1, 1, 1, S]` uint32 token ids -> `[1, 1, S, hidden]` bf16 TILE hidden states."""

    def __init__(
        self,
        mesh_device,
        hf,
        state_dict,
        *,
        mesh_config=None,
        ccl_manager=None,
        tensor_cache_path=None,
        activation_dtype=ttnn.bfloat16,
    ):
        """
        Args:
            mesh_device: the open mesh.
            hf: the normalised config **dict** (recipe P1 trap 2).
            state_dict: already stripped to this module's own keys, i.e. `{"weight": ...}` — the
                caller splits with `substate(state_dict, "model.embed_tokens")`. May be empty in
                cache-only mode, which requires `tensor_cache_path`.
            mesh_config: the model's `MeshConfig`; defaults to TP over the whole column axis.
            ccl_manager: unused — a replicated table needs no collective (`DEC-024`). Accepted so
                the module keeps the package's constructor convention
                (`bringup_log/03_OUTLINE.md` §5) and so a future vocab-sharded variant is a body
                change rather than a signature change.
            tensor_cache_path: where `ttnn.as_tensor` persists / reloads the table.
            activation_dtype: the residual stream's dtype, bf16 (`DEC-022`).
        """
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
        self.vocab_size = hf["vocab_size"]
        self.hidden_size = hf["hidden_size"]
        self.activation_dtype = activation_dtype

        if state_dict:
            weight = state_dict["weight"]
            assert tuple(weight.shape) == (self.vocab_size, self.hidden_size), (
                f"embed_tokens.weight is {tuple(weight.shape)}, expected " f"({self.vocab_size}, {self.hidden_size})"
            )
        elif not tensor_cache_path:
            # Fail loud rather than embed against a `None` table — Appendix B's "cache-only build
            # silently wrong" row, and the one weight whose absence would make *every* layer run on
            # garbage rather than one.
            raise ValueError("Embedding needs either a state_dict with 'weight' or a tensor_cache_path to load from")
        else:
            weight = None

        # ROW_MAJOR: `ttnn.embedding` gathers rows, so the table is not tilized. The `layout=`
        # argument on the call below is the *output* layout.
        self.weight = ttnn.as_tensor(
            weight,
            device=mesh_device,
            dtype=activation_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, token_ids):
        """`[1, 1, 1, S]` (or `[1, S]`) uint32 ROW_MAJOR -> `[1, 1, S, hidden]` bf16 TILE."""
        out = ttnn.embedding(token_ids, self.weight, layout=ttnn.TILE_LAYOUT, dtype=self.activation_dtype)
        if len(out.shape) == 3:
            out = ttnn.unsqueeze_to_4D(out)
        return out
