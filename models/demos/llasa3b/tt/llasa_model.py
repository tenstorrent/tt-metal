# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Llasa-3B Transformer Subclass

Overrides the base Transformer's decode path to handle Llasa-3B's
large vocabulary (193,800 tokens), which causes L1 buffer clashes
with the standard all_gather/all_reduce CCL operations.

Key differences from the base Transformer:
  - LoFi math for LM head to reduce L1 memory pressure
  - Skips all_reduce in LM head (weights are column-sharded, not row-sharded)
  - Skips all_gather_async in decode (causes L1 buffer clashes with 193K vocab)
  - Concatenates vocab shards from all devices on host for correct sampling
"""

import ttnn
from models.tt_transformers.tt.model import Mode, Transformer


class LlasaTransformer(Transformer):
    """Llasa-3B Transformer with large-vocabulary decode path overrides."""

    def _apply_llasa_patches(self):
        """Apply Llasa-specific patches after model construction.

        Called from prepare_llasa_generator after create_tt_model, since
        create_tt_model constructs a base Transformer and we upgrade it
        to LlasaTransformer via __class__ assignment.
        """
        self.lm_head.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def forward(self, x, current_pos, **kwargs):
        """Override forward to skip all_reduce in LM head.

        The LM head weights are column-sharded: each device computes logits
        for a different slice of the vocabulary, not partial sums. Therefore
        all_reduce (element-wise sum) is incorrect for this architecture.
        We temporarily replace tt_all_reduce with an identity function.
        """
        import models.tt_transformers.tt.lm_head as lm_head_module

        original_all_reduce = lm_head_module.tt_all_reduce
        lm_head_module.tt_all_reduce = lambda input_tensor, *args, **kw: input_tensor
        try:
            result = super().forward(x, current_pos, **kwargs)
        finally:
            lm_head_module.tt_all_reduce = original_all_reduce
        return result

    def ttnn_decode_forward(
        self,
        x,
        current_pos,
        rot_mat_idxs=None,
        page_table=None,
        kv_cache=None,
        sampling_on_device=False,
        capture_sampling_trace=False,
    ):
        """Override decode forward to skip all_gather_async.

        The all_gather_async causes L1 buffer clashes ("Statically allocated
        circular buffers clash with L1 buffers") with Llasa's 193K vocab output.
        Instead, process_output_decode concatenates device outputs on host.
        """
        rot_mats_global = self.rope_setup.get_rot_mats(rot_mat_idxs)
        rot_mats_local = self.rope_local_setup.get_rot_mats(rot_mat_idxs) if hasattr(self, "rope_local_setup") else None

        x_embed = self._transform_decode_inputs_device(x)

        tt_logits = self.forward(
            x_embed,
            current_pos,
            rot_mats_global=rot_mats_global,
            rot_mats_local=rot_mats_local,
            mode=Mode.DECODE,
            page_table=page_table,
            kv_cache=kv_cache,
        )

        # NOTE: Deliberately skip all_gather_async here.
        # The standard path gathers vocab shards across devices on-device,
        # but this causes L1 buffer clashes with Llasa's large vocabulary.
        # process_output_decode handles the gathering on host instead.

        tt_logits = ttnn.untilize(tt_logits, use_multicore=True)

        if not self.args.is_galaxy:
            tt_logits = ttnn.to_memory_config(tt_logits, ttnn.DRAM_MEMORY_CONFIG)

        return tt_logits, None

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False, is_log_probs=False):
        """Override to concatenate vocab shards from all devices on host.

        Without all_gather, each device has logits for its vocab shard only.
        On N300, speech tokens (IDs 128264+) are all on device 1, so we must
        concatenate both devices' outputs to sample from the full vocabulary.
        """
        if is_tokens or is_log_probs:
            padded_batch_size = 32
            if not is_log_probs:
                tt_out = ttnn.reshape(tt_out, ttnn.Shape([1, 1, padded_batch_size, 1]))
            return self.concat_host_output(tt_out, is_log_probs)[0, 0, :B, 0]

        if self.args.num_devices > 1:
            # Concatenate vocab shards from all devices on host
            tt_out = self.concat_host_output(tt_out)
        else:
            tt_out = ttnn.to_torch(tt_out).float()

        tt_out = tt_out[:, :, :B, : self.vocab_size].view(B, S, -1)
        return tt_out
