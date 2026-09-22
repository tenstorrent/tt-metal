# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The output projection. Ported from the ``lm_head`` block of ``minimax_m3/tt/model.py``.

``[1, 1, tokens_local, 12288] @ [12288, 131072]`` -> ``[1, 1, tokens_local, 32768]`` per chip, the
vocab column-parallel across the TP cols. The input is the TP-replicated residual stream, so no
collective is needed before the matmul and none is needed after it either: the vocab shards *are*
the output, and gathering 131072 logits per token is the caller's problem only if it wants them on
host (see :meth:`LMHead.gather`).

**No vocab padding.** The source pads the per-device width to a power of two so ``ttnn.topk``'s
multi-core bitonic path can be used for on-device sampling. 131072 / 4 = 32768 is already both
tile-aligned and a power of two, and sampling is out of scope for this bring-up, so the padding
machinery has nothing to do here and is left out rather than carried dead. The divisibility is
asserted instead — a vocab that needed padding would silently mis-shard.

``tie_word_embeddings`` is False for this checkpoint, so this is a real weight and not a view of
the embedding table. :meth:`~...reference.checkpoint.CheckpointLoader.lm_head` asserts that on the
loading side; there is nothing to assert here beyond the state dict it is handed.
"""

import ttnn
from models.demos.mistral_medium_3_5_128b.tt.compute import matmul_compute_kernel_config
from models.demos.mistral_medium_3_5_128b.utils.general_utils import get_cache_file_name


class LMHead:
    """Column-parallel vocab projection over the TP cols."""

    def __init__(
        self,
        mesh_device,
        config,
        state_dict,
        mesh_config,
        ccl_manager=None,
        *,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
    ):
        """
        Args:
            mesh_device: the open mesh.
            config: a :class:`~...reference.model_config.MistralMediumConfig`.
            state_dict: ``{"weight": [vocab_size, hidden_size]}`` in HF orientation. Empty =>
                cache-only load.
            mesh_config: :class:`~...tt.config.MeshConfig`.
            ccl_manager: only needed by :meth:`gather`.
            weight_dtype: on-device weight dtype (the spec's ``dataformats.weights``).
            tensor_cache_path: directory for the tilized-weight cache, or None.
        """
        assert (
            config.vocab_size % mesh_config.tp == 0
        ), f"vocab_size ({config.vocab_size}) must divide TP ({mesh_config.tp}); this head does not pad"
        assert (
            config.vocab_size // mesh_config.tp
        ) % ttnn.TILE_SIZE == 0, f"per-device vocab ({config.vocab_size // mesh_config.tp}) must be tile-aligned"
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.vocab_size = config.vocab_size
        self.vocab_local = config.vocab_size // mesh_config.tp

        # HF stores Linear weight as [out, in]; ttnn.matmul wants [in, out].
        weight = state_dict["weight"].transpose(0, 1).unsqueeze(0).unsqueeze(0) if state_dict else None
        self.weight = ttnn.as_tensor(
            weight,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_config.column_parallel(mesh_device),
            cache_file_name=get_cache_file_name(tensor_cache_path, "lm_head"),
        )

    def __call__(self, x):
        """``x``: ``[1, 1, tokens_local, hidden_size]`` -> ``[1, 1, tokens_local, vocab_local]``.

        bfloat16 out: the logits feed a softmax or an argmax, and a bf8 exponent-shared block across
        32768 vocab columns would quantise the tail of the distribution far more than the matmul
        itself does.
        """
        return ttnn.matmul(x, self.weight, dtype=ttnn.bfloat16, compute_kernel_config=matmul_compute_kernel_config())

    def gather(self, logits):
        """TP all-gather the vocab shards into full ``[1, 1, tokens_local, vocab_size]``.

        Separate from :meth:`__call__` because it is never wanted inside a captured trace (an
        all-gather writes to device) and never wanted at all during a prefill measured on K/V.
        """
        if self.mesh_config.tp == 1:
            return logits
        assert self.ccl_manager is not None, "gather needs a CCLManager"
        return self.mesh_config.allgather(logits, self.ccl_manager, axis=self.mesh_config.tp_axis, dim=3)
