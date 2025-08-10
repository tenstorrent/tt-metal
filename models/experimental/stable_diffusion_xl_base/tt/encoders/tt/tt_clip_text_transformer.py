from models.experimental.stable_diffusion_xl_base.tt.encoders.tt.tt_clip_encoder import TtClipEncoder
from models.experimental.stable_diffusion_xl_base.tt.encoders.tt.tt_clip_embeddings import TtClipEmbeddings
from models.experimental.stable_diffusion_xl_base.tt.encoders.encoder_utils import _create_tt_4d_causal_attention_mask

import torch.nn as nn
import torch
import ttnn


class TtClipTextTransformer(nn.Module):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        model_config,
        num_attention_heads,
        hidden_size,
        num_encoder_layers,
    ):
        super().__init__()
        self.device = device

        self.embeddings = TtClipEmbeddings(
            device,
            state_dict,
            f"{module_path}.embeddings",
            model_config,
        )
        self.encoder = TtClipEncoder(
            device,
            state_dict,
            f"{module_path}.encoder",
            model_config,
            num_attention_heads,
            hidden_size,
            num_encoder_layers,
        )

        # Final layernorm
        final_norm_weights = state_dict[f"{module_path}.final_layer_norm.weight"]
        final_norm_bias = state_dict[f"{module_path}.final_layer_norm.bias"]
        # print("final_norm_weights shape = ", final_norm_weights.shape)
        # print("final_norm_bias shape = ", final_norm_bias.shape)
        # print("Final norm weights = ", final_norm_weights)
        # print("Final norm bias = ", final_norm_bias)

        self.tt_final_norm_weights = ttnn.from_torch(
            final_norm_weights, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
        )
        self.tt_final_norm_bias = (
            ttnn.from_torch(final_norm_bias, ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
            if final_norm_bias is not None
            else None
        )
        self.ln_eps = 1e-05
        self.ln_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def _gather_eos(self, seq_emb: ttnn.Tensor, input_ids: ttnn.Tensor, eos_token_id: int) -> ttnn.Tensor:
        ids_t = ttnn.to_torch(ttnn.get_device_tensors(input_ids)[0])

        # from HF: if self.eos_token_id == 2: use argmax, else: search for eos_token_id
        if eos_token_id == 2:
            # use argmax (highest token ID position)
            eos_idx = ids_t.to(dtype=torch.int, device=ids_t.device).argmax(dim=-1)
        else:
            # search for specific eos_token_id
            eos_mask = (ids_t.to(dtype=torch.int, device=ids_t.device) == eos_token_id).int()
            eos_idx = eos_mask.argmax(dim=-1)

        seq_t = ttnn.to_torch(ttnn.get_device_tensors(seq_emb)[0])  # Shape: [1, 1, 77, 768]

        # Handle 4D tensor - squeeze unnecessary dimensions and get the sequence dimension
        if len(seq_t.shape) == 4:
            seq_t = seq_t.squeeze(1)  # Remove dim 1: [1, 77, 768]

        b = torch.arange(seq_t.size(0))
        pooled_t = seq_t[b, eos_idx]  # [B, H]

        return ttnn.from_torch(
            pooled_t,
            dtype=seq_emb.get_dtype(),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

    def forward(self, input_ids):
        batch_size, seq_length = input_ids.shape

        hidden_states = self.embeddings.forward(input_ids)
        causal_attention_mask = _create_tt_4d_causal_attention_mask(
            input_ids.shape, self.device, dtype=hidden_states.dtype
        )

        print("Attn mask shape = ", causal_attention_mask.shape)
        print("input ids shape = ", input_ids.shape)

        encoder_output = self.encoder.forward(
            hidden_states,
            causal_attention_mask,
        )

        normalized_final_state = ttnn.layer_norm(
            encoder_output,
            weight=self.tt_final_norm_weights,
            bias=self.tt_final_norm_bias,
            epsilon=self.ln_eps,
            compute_kernel_config=self.ln_compute_kernel_config,
        )

        eos_token_id = 2
        # print("normalized_final_state shape = ", normalized_final_state.shape)
        # print("input_ids shape = ", input_ids.shape)
        pooled_output = self._gather_eos(normalized_final_state, input_ids, eos_token_id)
        return pooled_output
