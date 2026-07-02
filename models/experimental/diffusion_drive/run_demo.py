# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d, infer_ttnn_module_args, preprocess_model_parameters
from models.demos.ufld_v2.reference.ufld_v2_model import CustomResNet34, BasicBlock

from models.experimental.diffusion_drive.ttnn_diffusion_drive import TtnnDiffusionDrive

# --- Preprocessing Utils Adaption ---
def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor_resnet(model, name, mesh_mapper)
    return custom_mesh_preprocessor

def custom_preprocessor_resnet(model, name, mesh_mapper=None):
    parameters = {}
    if isinstance(model, CustomResNet34):
        # This matches the structure expected by TtnnResnet34 implementation
        # coped from ufld_v2 but removed 'res_model' prefix if we are passing CustomResNet directly
        # However, TtnnResnet34 keys match properties of the class.
        
        # Conv1
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        parameters["conv1"] = {}
        parameters["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)

        # Layers
        for i in range(1, 5):
            layer_name = f"layer{i}"
            layer = getattr(model, layer_name)
            for j, block in enumerate(layer):
                block_name = f"{layer_name}_{j}"
                parameters[block_name] = {}
                
                # Conv1
                weight, bias = fold_batch_norm2d_into_conv2d(block.conv1, block.bn1)
                parameters[block_name]["conv1"] = {}
                parameters[block_name]["conv1"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
                bias = bias.reshape((1, 1, 1, -1))
                parameters[block_name]["conv1"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
                
                # Conv2
                weight, bias = fold_batch_norm2d_into_conv2d(block.conv2, block.bn2)
                parameters[block_name]["conv2"] = {}
                parameters[block_name]["conv2"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
                bias = bias.reshape((1, 1, 1, -1))
                parameters[block_name]["conv2"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
                
                # Downsample
                if block.downsample is not None:
                    weight, bias = fold_batch_norm2d_into_conv2d(block.downsample[0], block.downsample[1])
                    parameters[block_name]["downsample"] = {}
                    parameters[block_name]["downsample"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
                    bias = bias.reshape((1, 1, 1, -1))
                    parameters[block_name]["downsample"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
                    
    return parameters

def run_diffusion_drive_demo():
    # Setup Device
    device_id = 0
    device = ttnn.open_device(device_id=device_id)
    ttnn.SetDefaultDevice(device)
    
    # Validation inputs
    batch_size = 1
    input_h, input_w = 640, 640 # Target resolution from bounty description
    # Check if we should use 320x800 as likely used in ufld_v2? 
    # Bounty says "start with 640x640x3 (or model default)".
    # ResNet usually adaptable.
    
    torch_input = torch.randn((batch_size, 3, input_h, input_w), dtype=torch.bfloat16)

    # Reference Model (Backbone Only for now for config)
    torch_model = CustomResNet34(BasicBlock, [3, 4, 6, 3])
    torch_model.eval()
    
    # Preprocess Parameters
    print("Preprocessing parameters...")
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(),
        device=device,
    )
    
    # Trace for Conv Args
    print("Tracing model for conv args...")
    # infer_ttnn_module_args traces the model. We need to wrap CustomResNet34 to return single output 
    # instead of list [x2, x3, x4] because infer expects tensor or tuple of tensors, but 
    # the mapping needs to match our TtnnResnet usage.
    # TtnnResnet34 uses layerX_Y names.
    
    class Wrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            # We just need to run it so trace happens. Output value doesn't strictly matter for trace 
            # as long as shapes are correct.
            # But infer_ttnn_module_args uses the hooks.
            return self.base(x)

    wrapped_model = Wrapper(torch_model)
    
    parameters.conv_args = infer_ttnn_module_args(
        model=wrapped_model, 
        run_model=lambda model: model(torch_input.float()), # ResNet expectation
        device=device
    )
    
    # Initialize TTNN Model
    print("Initializing TTNN DiffusionDrive...")
    ttnn_model = TtnnDiffusionDrive(conv_args=parameters.conv_args, conv_pth=parameters, device=device)
    
    # Prepare Input
    print("Preparing input...")
    # TtnnResnet34 expects input in specific layout or handles it.
    # Based on ufld_v2 calling convention:
    # ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    # ttnn_input_tensor = ttnn_input_tensor.to(device, ttnn.L1_MEMORY_CONFIG)
    
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_input = ttnn_input.to(device, ttnn.L1_MEMORY_CONFIG)
    
    # Run
    print("Running Inference...")
    output = ttnn_model(ttnn_input, batch_size=batch_size)
    
    print("Inference finished.")
    print(f"Output shape: {output.shape}")
    
    # Convert to torch
    output_torch = ttnn.to_torch(output)
    print("Output converted to torch.")
    
    ttnn.close_device(device)
    print("Device closed.")

if __name__ == "__main__":
    run_diffusion_drive_demo()
