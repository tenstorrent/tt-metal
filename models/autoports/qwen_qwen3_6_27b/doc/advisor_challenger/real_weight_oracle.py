"""Real-weight differential oracle for the packed-QKV L1 candidate."""

from pathlib import Path
import types

import torch
from safetensors import safe_open
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID, MODEL_REVISION, _to_device
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder
from models.common.utility_functions import comp_pcc

LAYER = 3
SNAPSHOT = Path("/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots") / MODEL_REVISION
SHARDS = (
    "model-00004-of-00015.safetensors",
    "model-00006-of-00015.safetensors",
    "model-00007-of-00015.safetensors",
    "model-00008-of-00015.safetensors",
)


def main():
    prefix = f"model.language_model.layers.{LAYER}."
    state = {}
    for name in SHARDS:
        with safe_open(SNAPSHOT / name, framework="pt", device="cpu") as handle:
            state.update({key: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)})
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION).text_config
    torch.manual_seed(20260803)
    hidden = (torch.randn(1, 1, 32, config.hidden_size) * 0.2).bfloat16()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder = OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=LAYER,
            mesh_device=mesh,
            batch=32,
            max_context=64,
            page_size=64,
            optimization_policy="bfp4_all_dram_w8",
        )
        hidden_tt = _to_device(hidden, mesh_device=mesh)
        page_table = _to_device(
            torch.arange(32, dtype=torch.int32).reshape(32, 1),
            mesh_device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
        )
        positions = _to_device(
            torch.zeros(32, dtype=torch.uint32),
            mesh_device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
        )

        shipped_method = decoder._optimized_decode_linear

        def incumbent_linear(self, activation, weight, **kwargs):
            weight_name = next(name for name, value in self.weights.items() if value is weight)
            program_config = self.decode_program_configs.get(weight_name)
            if program_config is not None:
                activation = ttnn.to_memory_config(activation, self.decode_input_memory_configs[weight_name])
                kwargs["memory_config"] = self.decode_output_memory_configs[weight_name]
                kwargs["program_config"] = program_config
            kwargs["compute_kernel_config"] = self.compute_kernel_config
            kwargs["dtype"] = ttnn.bfloat16
            output = ttnn.linear(activation, weight, **kwargs)
            if weight_name == "packed_linear_inputs" and output.is_sharded():
                output = ttnn.to_memory_config(output, ttnn.L1_MEMORY_CONFIG)
            return output

        def run(candidate):
            decoder._optimized_decode_linear = (
                shipped_method if candidate else types.MethodType(incumbent_linear, decoder)
            )
            out = decoder.decode_forward(
                hidden_states=hidden_tt,
                page_table=page_table,
                current_positions=positions,
            )
            ttnn.synchronize_device(mesh)
            return ttnn.to_torch(ttnn.get_device_tensors(out)[0])

        incumbent = run(False)
        candidate = run(True)
        passed, message = comp_pcc(incumbent.float(), candidate.float(), 0.998843)
        print("REAL_WEIGHT_DIFFERENTIAL_PCC", message)
        assert passed, message
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
