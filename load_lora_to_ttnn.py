import sys
import time
from pathlib import Path

import torch
import ttnn
from diffusers import DiffusionPipeline


from models.experimental.stable_diffusion_xl_base.tt.sdxl_utility import prepare_linear_params
from models.experimental.stable_diffusion_xl_base.tt.lora_weights_logger import lora_logger

LORA_PATH = "lora_weights/VoxelXL_v1.safetensors"

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32,
    use_safetensors=True,
)
pipeline.load_lora_weights(LORA_PATH)

full_state_dict = dict(pipeline.unet.named_parameters())

lora_weights_state_dict = {}
attention_base_weights_state_dict = {}
for name, param in full_state_dict.items():
    if "attentions" not in name:
        continue
    p = param.detach()
    if "lora" in name.lower():
        lora_weights_state_dict[name] = p
    else:
        attention_base_weights_state_dict[name] = p

device = ttnn.open_device(device_id=0)
lora_on_device = {}
base_on_device = {}

for key, tensor in lora_weights_state_dict.items():
    if not tensor.is_floating_point():
        continue
    weight_4d = tensor.unsqueeze(0).unsqueeze(0) if tensor.dim() == 2 else tensor
    tt_weights_device, _, host_creation_ms, host_to_device_ms = prepare_linear_params(
        device, weight_4d, bias=None, dtype=ttnn.bfloat8_b, is_lora_impacted=True
    )
    lora_on_device[key] = tt_weights_device
    lora_logger.log_weight_creation(
        key,
        "tt_lora",
        tt_weights_device.shape,
        ttnn.bfloat8_b,
        device,
        tensor_obj=tt_weights_device,
        host_creation_time_ms=host_creation_ms,
        host_to_device_time_ms=host_to_device_ms,
    )

# for key, tensor in attention_base_weights_state_dict.items():
#     if not tensor.is_floating_point() or tensor.dim() != 2:
#         continue
#     weight_4d = tensor.unsqueeze(0).unsqueeze(0)
#     tt_weights_device, _, host_creation_ms, host_to_device_ms = prepare_linear_params(
#         device, weight_4d, bias=None, dtype=ttnn.bfloat8_b, is_lora_impacted=True
#     )
#     base_on_device[key] = tt_weights_device
#     lora_logger.log_weight_creation(
#         key, "tt_base", tt_weights_device.shape, ttnn.bfloat8_b, device,
#         tensor_obj=tt_weights_device,
#         host_creation_time_ms=host_creation_ms, host_to_device_time_ms=host_to_device_ms
#     )

ttnn.synchronize_device(device)

ttnn.dump_device_memory_state(device, prefix="lora_voxel_")

start = time.perf_counter()
for t in lora_on_device.values():
    ttnn.deallocate(t)
for t in base_on_device.values():
    ttnn.deallocate(t)
ttnn.synchronize_device(device)
lora_deallocation_time_ms = (time.perf_counter() - start) * 1000

print(f"Lora deallocation time: {lora_deallocation_time_ms} ms")

ttnn.close_device(device)
