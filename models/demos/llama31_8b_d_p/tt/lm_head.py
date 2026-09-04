# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LM head for Llama-3.1-8B: a column-parallel `[4096, 128256]` projection, no bias.

**HF anchor:** `transformers.models.llama.modeling_llama.LlamaForCausalLM.lm_head`.
**Template:** `models/demos/gpt_oss_d_p/tt/model.py:123-142` (the column-parallel mapper and the
`as_tensor` load) and `:241` (the matmul), lifted into its own module so it can own a test
(`bringup_log/03_OUTLINE.md` `[DEV-5]`).

**Why this file exists at all.** Prefill's product is the KV cache, not logits — so the lm_head is
not on the deployment critical path. It exists because `G-MODEL` gates **100% top-1 token
agreement** against HF, and the recipe requires the model to be built `with_lm_head=True` by default
so that half of the gate is never conditional (`BRINGUP_RECIPE.md:1559-1562`).

**No vocab padding.** `models/demos/gpt_oss_d_p/tt/model.py:31` `compute_per_device_vocab` pads to a
tile-aligned power of two because gpt-oss's vocab does not divide its mesh and because
`ttnn.topk`'s multi-core path needs a power-of-two width. Neither applies here: `128256 / 8 = 16032
= 501 * 32` is exact and tile-aligned (`bringup_log/00_MODEL_CARD.md` §4.2), and this package does
no on-device sampling (decode is an explicit non-goal). The padding branch is therefore deleted
rather than left as a no-op that reads as if padding might be needed.

**`tie_word_embeddings` is false** (`bringup_log/00_MODEL_CARD.md` §2), so `lm_head.weight` is a
real, separate checkpoint tensor and this module never reuses the embedding table.
"""

import ttnn
from models.demos.gpt_oss_d_p.utils.general_utils import get_cache_file_name

from .config import MeshConfig, default_compute_kernel_config


class LMHead:
    """`[1, 1, S, hidden]` -> `[1, 1, S, vocab/TP]` per chip, optionally all-gathered to full vocab."""

    def __init__(
        self,
        mesh_device,
        hf,
        state_dict,
        *,
        mesh_config=None,
        ccl_manager=None,
        weight_dtype=ttnn.bfloat8_b,
        activation_dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        fp32_dest_acc_en=True,
    ):
        """
        Args:
            mesh_device: the open mesh.
            hf: the normalised config **dict** (recipe P1 trap 2).
            state_dict: already stripped, i.e. `{"weight": ...}` — the caller splits with
                `substate(state_dict, "lm_head")`. May be empty in cache-only mode, which requires
                `tensor_cache_path`.
            mesh_config: the model's `MeshConfig`; defaults to TP over the whole column axis.
            ccl_manager: the model's `CCLManager`. Required only when `tp > 1` **and** the caller
                asks for gathered logits.
            weight_dtype: on-device weight dtype, `bfloat8_b` (`DEC-022`).
            activation_dtype: matmul output dtype, `bfloat16` (`DEC-022`). Note the template emits
                bf8_b logits (`models/demos/gpt_oss_d_p/tt/model.py:241`); `G-MODEL`'s top-1 is
                decided by the *gaps between* the top logits, so the wider dtype is kept here
                (`DEC-049`).
            tensor_cache_path: where `ttnn.as_tensor` persists / reloads the tilized weight.
            fp32_dest_acc_en: exposed only so a gate can A/B recipe §2.4's flag, as every other
                module in this package does. The default is the correct value.
        """
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
        self.ccl_manager = ccl_manager
        self.vocab_size = hf["vocab_size"]
        self.hidden_size = hf["hidden_size"]
        self.activation_dtype = activation_dtype
        self.compute_kernel_config = default_compute_kernel_config(mesh_device, fp32_dest_acc_en=fp32_dest_acc_en)

        assert self.vocab_size % self.mesh_config.tp == 0, (
            f"vocab_size {self.vocab_size} is not divisible by tp {self.mesh_config.tp}; this "
            f"package deletes gpt-oss's vocab padding because 128256/8 = 16032 is exact"
        )
        assert (
            self.mesh_config.shard_size(self.vocab_size) % ttnn.TILE_SIZE == 0
        ), f"per-device vocab {self.mesh_config.shard_size(self.vocab_size)} is not tile-aligned"

        if state_dict:
            # HF `[vocab, hidden]` -> ttnn `[1, 1, hidden, vocab]`, transposed at LOAD time.
            weight = state_dict["weight"]
            assert tuple(weight.shape) == (self.vocab_size, self.hidden_size), (
                f"lm_head.weight is {tuple(weight.shape)}, expected ({self.vocab_size}, {self.hidden_size}) — "
                f"tie_word_embeddings is false, so this must be its own tensor"
            )
            weight = weight.transpose(-1, -2).unsqueeze(0).unsqueeze(0)
        elif not tensor_cache_path:
            raise ValueError("LMHead needs either a state_dict with 'weight' or a tensor_cache_path to load from")
        else:
            weight = None

        self.weight = ttnn.as_tensor(
            weight,
            device=mesh_device,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self.mesh_config.column_parallel(mesh_device),
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, x, *, gather=False):
        """`[1, 1, S, hidden]` -> `[1, 1, S, vocab/TP]`, or the full `[1, 1, S, vocab]` if `gather`.

        The vocab shard is a **column-parallel output**, i.e. each chip holds a *slice* of the
        logits rather than a partial sum, so the collective is an all-gather and not a reduce
        (`bringup_log/04_CCL_PLAN.md` §4, LM-head row). It is off by default because
        `Model.process_output_prefill` gathers on the **host** — one `to_torch` per column, which is
        what the templates do (`models/demos/gpt_oss_d_p/tt/model.py:326-328`) and what keeps the
        prefill path free of a collective it does not need.
        """
        logits = ttnn.linear(
            x, self.weight, dtype=self.activation_dtype, compute_kernel_config=self.compute_kernel_config
        )
        if not gather or self.mesh_config.tp <= 1:
            return logits
        if self.ccl_manager is None:
            raise ValueError("LMHead(gather=True) at tp>1 needs a ccl_manager for the vocab all-gather")
        gathered = self.mesh_config.allgather(logits, self.ccl_manager, axis=self.mesh_config.tp_axis, dim=3)
        logits.deallocate(True)
        return gathered
