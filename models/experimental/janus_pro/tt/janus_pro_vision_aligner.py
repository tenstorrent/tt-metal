"""
Vision aligner for Janus-Pro-7B.
Projects vision encoder output (hidden_size) to text embedding dim (projection_dim).
HF reference: JanusVisionAlignerMLP in transformers.models.janus.modeling_janus
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule

# The aligner activation follows HF JanusVisionAlignerMLP, which applies config.hidden_act
# (resolved into vision_act_layer). GELU carries its APPROXIMATION_MODE parameter explicitly at
# False, which is what a standalone ttnn.gelu defaults to (unary.hpp:339), so fusing moves where
# the SFPU pass runs without changing what it computes.
_FUSED_ACT = {
    ttnn.UnaryOpType.GELU: (ttnn.UnaryOpType.GELU, False),
    ttnn.UnaryOpType.RELU: (ttnn.UnaryOpType.RELU,),
    ttnn.UnaryOpType.SILU: (ttnn.UnaryOpType.SILU,),
}

_STANDALONE_ACT = {
    ttnn.UnaryOpType.GELU: ttnn.gelu,
    ttnn.UnaryOpType.RELU: ttnn.relu,
    ttnn.UnaryOpType.SILU: ttnn.silu,
}


class TtJanusProVisionAligner(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        state_dict_prefix,  # "model.aligner."
        weight_cache_path,
        dtype,
    ):
        super().__init__()
        self.args = args
        self.fused_act = _FUSED_ACT[args.vision_act_layer]
        self.act_fn = _STANDALONE_ACT[args.vision_act_layer]

        # HF JanusVisionAlignerMLP: fc1 + (depth - 1) hidden layers
        num_hidden = max(0, args.vision_aligner_depth - 1)

        def load_linear(name):
            w = torch.transpose(state_dict[f"{state_dict_prefix}{name}.weight"], -2, -1)
            cache = None if weight_cache_path is None else weight_cache_path / f"{state_dict_prefix}{name}.weight"
            weight = ttnn.as_tensor(
                w,
                dtype=ttnn.bfloat8_b,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache,
            )
            bias = None
            if f"{state_dict_prefix}{name}.bias" in state_dict:
                b = state_dict[f"{state_dict_prefix}{name}.bias"]
                bias_cache = (
                    None if weight_cache_path is None else weight_cache_path / f"{state_dict_prefix}{name}.bias"
                )
                bias = ttnn.as_tensor(
                    b,
                    dtype=dtype,
                    device=mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_file_name=bias_cache,
                )
                bias = ttnn.reshape(bias, [1, -1])
            return weight, bias

        self.fc1_weight, self.fc1_bias = load_linear("fc1")
        self.hidden_layers = []
        for i in range(num_hidden):
            w, b = load_linear(f"hidden_layers.{i}")
            self.hidden_layers.append((w, b))

    def _linear(self, x, weight, bias, program_config=None, dtype=ttnn.bfloat16):
        # HiFi2 costs nothing on the weight and half the math cycles: ttnn.linear puts the weight in
        # SrcA (llk_unpack_AB_matmul.h:43-44), whose mantissa HiFi2 already consumes, while HiFi4's
        # extra passes read SrcB -- the activation -- at 64 cycles per tile against 32
        # (matrix_engine.md:68-71, GEMM_FLOPS.md:107). HiFi4 with fp32 accumulation also trips
        # Wormhole issue 38306. Nothing reduces before the bias, so it rides inside.
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            program_config=program_config,
            compute_kernel_config=self.args.compute_kernel_config_hifi2,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x: [1, B, seq, vision_dim] — output of TtJanusProVisionModel (ln_post)
        #
        # Every activation but the last rides inside the matmul that produces its input. Standing
        # alone it is one op reading the whole intermediate back from DRAM for a single SFPU pass,
        # which costs more than the pass. Falling back to a standalone op keeps shapes ttnn has no
        # 2D config for working.
        batch_size, seq_len = x.shape[0], x.shape[-2]
        layers = [(self.fc1_weight, self.fc1_bias), *self.hidden_layers]
        for i, (weight, bias) in enumerate(layers):
            k, n = x.shape[-1], weight.shape[-1]
            last = i == len(layers) - 1
            program_config = self.args.vision_aligner_program_config(
                batch_size, seq_len, k, n, fused_activation=None if last else self.fused_act
            )
            # An intermediate projection is read once, by the projection after it, so it carries no
            # more than that read needs. The last one feeds the language model and stays bfloat16.
            x = self._linear(
                x,
                weight,
                bias,
                program_config=program_config,
                dtype=ttnn.bfloat16 if last else ttnn.bfloat8_b,
            )
            if not last and program_config is None:
                x = self.act_fn(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x
