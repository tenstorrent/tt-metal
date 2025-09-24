# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import ttnn


from models.experimental.transfuser.reference.config import GlobalConfig
from models.experimental.transfuser.reference.transfuser_backbone import TransfuserBackbone


# def create_gpt_preprocessor(device, n_layer, weight_dtype=ttnn.bfloat16):
#     def custom_preprocessor(torch_model, name, ttnn_module_args):
#         parameters = {}
#         if hasattr(torch_model, "ln_f"):
#             parameters["ln_f_weight"] = preprocess_linear_weight(torch_model.ln_f.weight, dtype=weight_dtype)
#             parameters["ln_f_bias"] = preprocess_linear_weight(torch_model.ln_f.bias, dtype=weight_dtype)
#         if hasattr(torch_model, "blocks"):
#             for i in range(n_layer):
#                 parameters[f"blocks_{i}"] = preprocess_model_parameters(
#                     initialize_model=lambda: torch_model.blocks[i],
#                     custom_preprocessor=create_gpt_block_preprocessor(device, weight_dtype),
#                     device=device,
#                 )
#         return parameters

#     return custom_preprocessor


@pytest.mark.parametrize(
    "image_architecture, lidar_architecture, n_layer, use_velocity, use_target_point_image, img_input_shape, lidar_input_shape",
    [("regnety_032", "regnety_032", 4, False, True, (1, 3, 160, 704), (1, 3, 256, 256))],  # GPT-SelfAttention 1
)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat16])
def test_transfuser_backbone(
    device,
    image_architecture,
    lidar_architecture,
    n_layer,
    use_velocity,
    use_target_point_image,
    img_input_shape,
    lidar_input_shape,
    input_dtype,
    weight_dtype,
):
    image_input = torch.randn(img_input_shape)
    lidar_input = torch.randn(lidar_input_shape)
    velocity_input = torch.randn(1, 1)

    # setting machine to avoid loading files
    config = GlobalConfig(setting="eval")
    config.n_layer = n_layer
    if use_target_point_image:
        config.use_target_point_image = use_target_point_image

    ref_layer = TransfuserBackbone(
        config,
        image_architecture=image_architecture,
        lidar_architecture=lidar_architecture,
        use_velocity=use_velocity,
    ).eval()

    features, image_features_grid, fused_features = ref_layer(image_input, lidar_input, velocity_input)

    pytest.skip("Skipping test_transfuser_backbone")

    # parameters = preprocess_model_parameters(
    #     initialize_model=lambda: ref_layer,
    #     custom_preprocessor=create_gpt_preprocessor(device, n_layer, weight_dtype),
    #     device=device,
    # )
    # tt_layer = TTGpt(
    #     device,
    #     parameters,
    #     n_head,
    #     n_layer,
    #     use_velocity=use_velocity,
    #     img_vert_anchors=img_vert_anchors,
    #     img_horz_anchors=img_horz_anchors,
    #     lidar_vert_anchors=lidar_vert_anchors,
    #     lidar_horz_anchors=lidar_horz_anchors,
    #     seq_len=seq_len,
    #     n_embd=n_embed,
    #     dtype=weight_dtype,
    #     memory_config=ttnn.L1_MEMORY_CONFIG,
    # )
    # # tt_token_embeddings = ttnn.from_torch(
    # #     token_embeddings, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype, memory_config=ttnn.L1_MEMORY_CONFIG
    # # )
    # tt_velocity_input = ttnn.from_torch(
    #     velocity_input, device=device, layout=ttnn.TILE_LAYOUT, dtype=input_dtype, memory_config=ttnn.L1_MEMORY_CONFIG
    # )
    # tt_output = tt_layer(token_embeddings, tt_velocity_input)
    # tt_torch_output = tt2torch_tensor(tt_output)
    # tt_image_output, tt_lidar_output = post_process_output(
    #     tt_torch_output,
    #     bz,
    #     seq_len,
    #     img_vert_anchors,
    #     img_horz_anchors,
    #     lidar_vert_anchors,
    #     lidar_horz_anchors,
    #     n_embed,
    #     img_h,
    #     img_w,
    #     lidar_h,
    #     lidar_w,
    # )
    # does_pass, image_out_pcc_message = check_with_pcc(ref_image_output, tt_image_output, 0.95)

    # logger.info(f"Image Output PCC: {image_out_pcc_message}")
    # assert does_pass, f"PCC check failed: {image_out_pcc_message}"

    # does_pass, lidar_out_pcc_message = check_with_pcc(ref_lidar_output, tt_lidar_output, 0.95)
    # assert does_pass, f"PCC check failed: {lidar_out_pcc_message}"

    # logger.info(f"Lidar Output PCC: {lidar_out_pcc_message}")

    # if does_pass:
    #     logger.info("GPT Passed!")
    # else:
    #     logger.warning("GPT Failed!")
