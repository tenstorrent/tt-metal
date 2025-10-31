# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from collections import OrderedDict
from typing import Dict, Any, List

import ttnn
from loguru import logger

from models.experimental.transfuser.reference.config import GlobalConfig
from models.experimental.transfuser.reference.transfuser_backbone import TransfuserBackbone
from models.experimental.transfuser.tt.custom_preprocessing import create_custom_mesh_preprocessor

from models.experimental.transfuser.tests.test_gpt import create_gpt_preprocessor
from models.experimental.transfuser.tt.transfuser_backbone import TtTransfuserBackbone
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from tests.ttnn.utils_for_testing import check_with_pcc


def fix_and_filter_checkpoint_keys(
    checkpoint_path: str, target_prefix: str = "module._model.", state_dict_key: str = None
) -> Dict[str, Any]:
    """
    Loads a PyTorch checkpoint, filters for keys starting with the target_prefix,
    and then removes that prefix from the keys.

    Args:
        checkpoint_path: Path to the .pth or .pt checkpoint file.
        target_prefix: The prefix that identifies the weights you want to keep
                       AND remove from the key (Default is 'module._model.').
        state_dict_key: The key in the loaded checkpoint dict that holds the
                        actual model state_dict (e.g., 'state_dict').
                        If None, it assumes the checkpoint itself is the state_dict.

    Returns:
        The modified state dictionary (OrderedDict) containing only the filtered
        and renamed keys, ready for model loading.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    # 1. Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 2. Extract the state dictionary
    if state_dict_key and state_dict_key in checkpoint:
        checkpoint_state_dict = checkpoint[state_dict_key]
    else:
        checkpoint_state_dict = checkpoint

    # 3. Filter and Strip the keys
    new_state_dict = OrderedDict()
    removed_keys_count = 0

    for k, v in checkpoint_state_dict.items():
        if k.startswith(target_prefix):
            # KEEP AND RENAME: Strip the prefix to match your model's keys
            name = k[len(target_prefix) :]
            new_state_dict[name] = v
        else:
            # DISCARD: These keys are outside of the module._model scope
            removed_keys_count += 1

    print(f"✅ Filtered and kept {len(new_state_dict)} keys starting with '{target_prefix}'.")
    print(f"🗑️ Discarded {removed_keys_count} keys that did not match the prefix.")

    return new_state_dict


def delete_incompatible_keys(state_dict: Dict[str, Any], keys_to_delete: List[str]) -> Dict[str, Any]:
    """
    Removes specified keys from a state dictionary. This is used to delete
    weights for layers that are being re-initialized (e.g., changing input channels).
    """
    deleted_count = 0
    new_state_dict = OrderedDict(state_dict)  # Create a modifiable copy

    for k_del in keys_to_delete:
        if k_del in new_state_dict:
            del new_state_dict[k_del]
            deleted_count += 1
            print(f"🗑️ Deleted incompatible key: {k_del}")

    print(f"Successfully deleted {deleted_count} key(s) for strict=True loading.")
    return new_state_dict


class TransfuserBackboneInfra:
    def __init__(
        self,
        device,
        image_architecture,
        lidar_architecture,
        n_layer,
        use_velocity,
        use_target_point_image,
        img_input_shape,
        lidar_input_shape,
        model_config,
        use_fallback,
        use_optimized_self_attn,
    ):
        super().__init__()
        # self._init_seeds()
        self.device = device
        self.n_layer = n_layer
        self.image_arch = image_architecture
        self.lidar_arch = lidar_architecture
        self.use_velocity = use_velocity
        self.img_input_shape = img_input_shape
        self.lidar_input_shape = lidar_input_shape
        self.num_devices = device.get_num_devices()
        # self.batch_size = batch_size * self.num_devices
        self.inputs_mesh_mapper, self.weights_mesh_mapper, self.output_mesh_composer = self.get_mesh_mappers(device)
        # self.name = name

        # setting machine to avoid loading files
        self.config = GlobalConfig(setting="eval")
        self.config.n_layer = self.n_layer
        if use_target_point_image:
            self.config.use_target_point_image = use_target_point_image

        # Build reference torch model
        torch_model = TransfuserBackbone(
            self.config,
            image_architecture=self.image_arch,
            lidar_architecture=self.lidar_arch,
            use_velocity=self.use_velocity,
        )
        torch_model.eval()
        checkpoint_path = "model_ckpt/models_2022/transfuser/model_seed1_39.pth"
        modified_state_dict = fix_and_filter_checkpoint_keys(
            checkpoint_path=checkpoint_path,
            target_prefix="module._model.",  # This is the prefix to keep and remove
            state_dict_key=None,  # Adjust this if needed
        )
        modified_state_dict = delete_incompatible_keys(modified_state_dict, ["lidar_encoder._model.stem.conv.weight"])
        torch_model.load_state_dict(modified_state_dict, strict=True)

        # Preprocess parameters for TTNN
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model,
            custom_preprocessor=create_custom_mesh_preprocessor(self.weights_mesh_mapper),
            device=None,
        )
        gpt1_parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model.transformer1,
            custom_preprocessor=create_gpt_preprocessor(device, n_layer, ttnn.bfloat16, use_optimized_self_attn),
            device=device,
        )
        parameters["transformer1"] = gpt1_parameters
        gpt2_parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model.transformer2,
            custom_preprocessor=create_gpt_preprocessor(device, n_layer, ttnn.bfloat16, use_optimized_self_attn),
            device=device,
        )
        parameters["transformer2"] = gpt2_parameters
        gpt3_parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model.transformer3,
            custom_preprocessor=create_gpt_preprocessor(device, n_layer, ttnn.bfloat16, use_optimized_self_attn),
            device=device,
        )
        parameters["transformer3"] = gpt3_parameters
        gpt4_parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_model.transformer4,
            custom_preprocessor=create_gpt_preprocessor(device, n_layer, ttnn.bfloat16, use_optimized_self_attn),
            device=device,
        )
        parameters["transformer4"] = gpt4_parameters

        inputs = torch.load("transfuser_inputs_final.pt")
        self.torch_image_input = inputs["image"]  # RGB camera image tensor
        self.torch_lidar_input = inputs["lidar"]  # LiDAR BEV tensor
        self.torch_velocity_input = inputs["velocity"]  # Ego velocity tensor
        with torch.no_grad():
            self.torch_features, self.torch_image_grid, self.torch_fused = torch_model(
                self.torch_image_input,
                self.torch_lidar_input,
                self.torch_velocity_input,
            )

        # Convert input to TTNN format
        self.input_image_tensor = ttnn.from_torch(
            self.torch_image_input.permute(0, 2, 3, 1),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self.inputs_mesh_mapper,
        )
        self.input_lidar_tensor = ttnn.from_torch(
            self.torch_lidar_input.permute(0, 2, 3, 1),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self.inputs_mesh_mapper,
        )
        self.input_velocity_tensor = ttnn.from_torch(
            self.torch_velocity_input,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Build TTNN model
        self.ttnn_model = TtTransfuserBackbone(
            device,
            parameters=parameters,
            stride=2,
            model_config=model_config,
            config=self.config,
            torch_model=torch_model,
            use_fallback=use_fallback,
        )

        # Run + validate
        self.run()
        self.validate(model_config)

    def _init_seeds(self):
        if not hasattr(self, "_model_initialized"):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            self._model_initialized = True

    def get_mesh_mappers(self, device):
        if device.get_num_devices() != 1:
            return (
                ttnn.ShardTensorToMesh(device, dim=0),
                None,
                ttnn.ConcatMeshToTensor(device, dim=0),
            )
        return None, None, None

    def run(self):
        self.output_features, self.output_image_grid, self.output_fused = self.ttnn_model(
            self.input_image_tensor, self.input_lidar_tensor, self.input_velocity_tensor, self.device
        )
        return self.output_features, self.output_image_grid, self.output_fused

    def validate(self, model_config, output_tensor=None):
        # Validate image output
        tt_features_torch = []
        fpn_names = ["p2", "p3", "p4", "p5"]
        for i, (feature, name) in enumerate(zip(self.output_features, fpn_names)):
            tt_feat = ttnn.to_torch(
                feature,
                device=self.device,
                mesh_composer=self.output_mesh_composer,
            )

            # Permute NHWC -> NCHW
            tt_feat = tt_feat.permute(0, 3, 1, 2)
            tt_features_torch.append(tt_feat)

        # Validate output_image_grid
        tt_image_grid_torch = ttnn.to_torch(
            self.output_image_grid,
            device=self.device,
            mesh_composer=self.output_mesh_composer,
        )
        tt_image_grid_torch = tt_image_grid_torch.permute(0, 3, 1, 2)

        # Validate output_fused_tensor
        tt_fused_torch = ttnn.to_torch(
            self.output_fused,
            device=self.device,
            mesh_composer=self.output_mesh_composer,
        )

        # Deallocate output tensors
        for feature in self.output_features:
            ttnn.deallocate(feature)
        ttnn.deallocate(self.output_image_grid)
        ttnn.deallocate(self.output_fused)

        # Validate FPN features
        fpn_pcc_results = []
        for torch_feat, tt_feat, name in zip(self.torch_features, tt_features_torch, fpn_names):
            pcc_passed, pcc_msg = check_with_pcc(torch_feat, tt_feat, pcc=0.95)
            fpn_pcc_results.append((pcc_passed, pcc_msg))
            logger.info(f"{name} PCC: {pcc_msg}")

        # Validate image grid
        grid_pcc_passed, grid_pcc_msg = check_with_pcc(self.torch_image_grid, tt_image_grid_torch, pcc=0.95)
        logger.info(f"Image Grid PCC: {grid_pcc_msg}")

        # Validate fused features
        fused_pcc_passed, fused_pcc_msg = check_with_pcc(self.torch_fused, tt_fused_torch, pcc=0.95)
        logger.info(f"Fused Features PCC: {fused_pcc_msg}")

        # All outputs must pass
        all_fpn_passed = all(result[0] for result in fpn_pcc_results)
        overall_passed = all_fpn_passed and grid_pcc_passed and fused_pcc_passed

        assert overall_passed, logger.error(
            f"PCC check failed - FPN: {fpn_pcc_results}, Grid: {grid_pcc_msg}, Fused: {fused_pcc_msg}"
        )

        return overall_passed, f"FPN: {fpn_pcc_results}, Grid: {grid_pcc_msg}, Fused: {fused_pcc_msg}"


# High accuracy model config
model_config = {
    "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    "WEIGHTS_DTYPE": ttnn.bfloat16,
    "ACTIVATIONS_DTYPE": ttnn.bfloat16,
    "fp32_dest_acc_en": True,
    "packer_l1_acc": True,
    "math_approx_mode": False,
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "image_architecture, lidar_architecture, n_layer, use_velocity, use_target_point_image, img_input_shape, lidar_input_shape",
    [("regnety_032", "regnety_032", 4, False, True, (1, 3, 160, 704), (1, 3, 256, 256))],
)
@pytest.mark.parametrize("use_fallback", [True])
@pytest.mark.parametrize("use_optimized_self_attn", [True])
def test_stem(
    device,
    image_architecture,
    lidar_architecture,
    n_layer,
    use_velocity,
    use_target_point_image,
    img_input_shape,
    lidar_input_shape,
    use_fallback,
    use_optimized_self_attn,
):
    TransfuserBackboneInfra(
        device,
        image_architecture,
        lidar_architecture,
        n_layer,
        use_velocity,
        use_target_point_image,
        img_input_shape,
        lidar_input_shape,
        model_config,
        use_fallback,
        use_optimized_self_attn,
    )
