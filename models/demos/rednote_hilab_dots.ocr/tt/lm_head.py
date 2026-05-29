# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN implementation of the dots.ocr Qwen2 language-model LM head.

Reference: models/demos/rednote_hilab_dots.ocr/reference/functional.py
           :func:`lm_head_forward`

    logits = F.linear(x, weight, None)

i.e. an untied Linear projecting hidden_size (1536) -> vocab_size (151936),
no bias (tie_word_embeddings=False). The reference ``lm_head_forward`` is the
BARE linear; the final ``norm`` (RMSNorm) is applied earlier inside
``language_model_forward`` and is NOT part of this block (the golden
``lm_head.pt`` is the un-normed Linear). This block therefore implements only
the projection, matching the golden exactly.

The HF ``lm_head.weight`` is stored [vocab_size, hidden_size]; ttnn.linear
computes ``x @ W`` with W laid out [in, out], so the weight is transposed on the
host at load time to [hidden_size, vocab_size]. The 1536 -> 151936 projection is
a wide matmul (~233M params); the weight is kept in DRAM. fp32_dest_acc + HiFi4
accumulation keeps the wide reduction accurate.

The matmul is DRAM-bandwidth-bound on the weight read (~233M params dominates
the kernel), so the weight is stored as bfloat8_b: this halves the weight
footprint / DRAM read and the kernel drops from ~1463 us to ~933 us (-36%,
traced) with PCC essentially unchanged (0.99997 vs the 0.99996 bf16 baseline).
The activation stays bf16. bf8 weight is safe here because the logits feed an
argmax in the generation phase, where the small numerical change does not flip
the top token. The output dtype stays bf16.

Reference TTNN impl this follows: models/tt_transformers/tt/lm_head.py (untied
final projection) and the local tt/mlp.py linear-loading pattern.
"""
import ttnn
from models.common.lightweightmodule import LightweightModule


class TtLMHead(LightweightModule):
    """dots.ocr LM head: untied Linear hidden_size -> vocab_size, no bias.

    Args:
        device: ttnn Device or MeshDevice.
        weight: torch.Tensor [vocab_size, hidden_size] (the ``lm_head.weight``
            parameter, untied). Transposed on host to [hidden_size, vocab_size].
        dtype: weight storage dtype (bfloat8_b by default; the wide matmul is
            DRAM-bandwidth-bound on the weight read, so bf8 halves the read and
            is a -36% traced win while PCC stays at 0.99997).
        weight_memory_config: storage for the (large) weight; DRAM by default.
    """

    def __init__(
        self,
        device,
        weight,
        dtype=ttnn.bfloat8_b,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        super().__init__()
        self.device = device

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None

        # HF weight is [vocab, hidden]; ttnn.linear wants [hidden, vocab].
        self.weight = ttnn.as_tensor(
            weight.transpose(0, 1).contiguous(),  # [hidden, vocab]
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=mesh_mapper,
        )

        # fp32 accumulation for the wide (hidden -> vocab) reduction.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [seq, hidden] (TILE layout) -> logits [seq, vocab]."""
        return ttnn.linear(
            x,
            self.weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
        )
