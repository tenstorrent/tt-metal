import os
import torch
import ttnn
import pytest
import matplotlib.pyplot as plt
from loguru import logger

from models.experimental.oft.reference.bbox import visualize_objects
from models.experimental.oft.reference.encoder import ObjectEncoder
from models.experimental.oft.reference.oftnet import OftNet
from models.experimental.oft.reference.utils import (
    get_abs_and_relative_error,
    load_calib,
    load_image,
    make_grid,
    visualize_score,
)
from models.experimental.oft.tests.test_common import (
    GRID_HEIGHT,
    GRID_RES,
    GRID_SIZE,
    H_PADDED,
    NMS_THRESH,
    W_PADDED,
    Y_OFFSET,
    load_checkpoint,
)
from models.experimental.oft.tt.model_preprocessing import create_OFT_model_parameters
from tests.ttnn.utils_for_testing import check_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16 * 1024}], indirect=True)
@pytest.mark.parametrize(
    "input_image_path, calib_path",
    [
        # (
        #     os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/000022.jpg")),
        #     os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/000022.txt")),
        # ),
        (
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/000013.jpg")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/000013.txt")),
        ),
        # (
        #     os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/000009.jpg")),
        #     os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources/000009.txt")),
        # )
    ],
)
@pytest.mark.parametrize(
    "model_dtype, fallback_feedforward, fallback_lateral, fallback_oft, use_host_decoder, scale_features, pcc_scores_oft, pcc_positions_oft, pcc_dimensions_oft, pcc_angles_oft",
    # fmt: off
    [
       ( torch.float32, False, False, False, True,  False, 0.298, 0.717, 0.994, 0.745),
       ( torch.float32, False, False, False, True,   True, 0.970, 0.995, 0.999, 0.820),
       ( torch.float32,  True,  True, False, True,  False, 0.882, 0.963, 0.998, 0.897),
       ( torch.float32, False, False, True, True,   False, 0.916, 0.883, 0.997, 0.934),
    ],
    ids=[
        "use_host_decoder1",
        "use_host_decoder_scale_features",
        "fallback_feedforward_lateral",
        "fallback_oft",
    ],
    # fmt: on
)
@pytest.mark.parametrize("checkpoints_path", [r"/home/mbezulj/checkpoint-0600.pth"])
@torch.no_grad()
# def run_demo_inference(
def test_oftnet(
    device,
    checkpoints_path,
    input_image_path,
    calib_path,
    model_dtype,
    fallback_feedforward,
    fallback_lateral,
    fallback_oft,
    use_host_decoder,
    scale_features,
    pcc_scores_oft,
    pcc_positions_oft,
    pcc_dimensions_oft,
    pcc_angles_oft,
):
    assert use_host_decoder == True, "Only use_host_decoder=True is supported for now"
    # Create output directory for saving visualizations
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    basename = os.path.basename(input_image_path).split(".")[0]

    torch.manual_seed(42)

    # ========================================================
    # OFT model configuration based on real model parameters
    input_tensor = load_image(input_image_path, pad_hw=(H_PADDED, W_PADDED), dtype=model_dtype)[None].to(model_dtype)
    calib = load_calib(calib_path, dtype=model_dtype)[None].to(model_dtype)
    grid = make_grid(GRID_SIZE, (-GRID_SIZE[0] / 2.0, Y_OFFSET, 0.0), GRID_RES, dtype=model_dtype)[None].to(model_dtype)

    topdown_layers = 8
    ref_model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=topdown_layers,
        grid_res=GRID_RES,
        grid_height=GRID_HEIGHT,
        dtype=torch.float32,
        scale_features=scale_features,
    )

    ref_model = load_checkpoint(checkpoints_path, ref_model)

    # def initialize_weights_and_biases(model):
    #     """Initialize all weights to 1 and all biases to 0"""
    #     import torch
    #     import torch.nn as nn
    #     for name, param in model.named_parameters():
    #         if 'weight' in name:
    #             nn.init.ones_(param)
    #         elif 'bias' in name:
    #             nn.init.zeros_(param)
    #     return model
    # ref_model = initialize_weights_and_biases(ref_model)

    parameters = create_OFT_model_parameters(ref_model, (input_tensor, calib, grid), device=device)

    ref_encoder = ObjectEncoder(nms_thresh=NMS_THRESH, dtype=model_dtype)

    # ========================================================
    # Create lower precision model
    test_model = OftNet(
        num_classes=1,
        frontend="resnet18",
        topdown_layers=topdown_layers,
        grid_res=GRID_RES,
        grid_height=GRID_HEIGHT,
        dtype=torch.bfloat16,
        scale_features=scale_features,
    )
    test_model = load_checkpoint(checkpoints_path, test_model)

    # ========================================================
    # Run torch fp32 and bfp16 inference pass

    intermediates, scores, pos_offsets, dim_offsets, ang_offsets = ref_model(input_tensor, calib, grid)

    tt_intermediates, tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets = test_model.forward(
        input_tensor.to(torch.float32), calib, grid
    )
    grid = grid.squeeze(0)  # TODO(mbezulj) align all shapes to get rid of squeezing/unsqueezing at random places
    ref_pred_encoded = [t.squeeze(0) for t in (scores, pos_offsets, dim_offsets, ang_offsets)]
    ref_detections, peaks = ref_encoder.forward(*ref_pred_encoded, grid)
    layer_names = (
        "image",
        "feats8",
        "feats16",
        "feats32",
        "lat8",
        "lat16",
        "lat32",
        "integral_img8",
        "integral_img16",
        "integral_img32",
        "bbox_top_left8",
        "bbox_btm_right8",
        "bbox_top_right8",
        "bbox_btm_left8",
        "bbox_top_left16",
        "bbox_btm_right16",
        "bbox_top_right16",
        "bbox_btm_left16",
        "bbox_top_left32",
        "bbox_btm_right32",
        "bbox_top_right32",
        "bbox_btm_left32",
        "ortho8",
        "ortho16",
        "ortho32",
        "ortho",
        "calib",
        "grid",
        "td",
    )
    ttnn_pred_encoded = [
        t.squeeze(0)
        for t in (
            tt_scores.to(torch.float32),
            tt_pos_offsets.to(torch.float32),
            tt_dim_offsets.to(torch.float32),
            tt_ang_offsets.to(torch.float32),
        )
    ]
    ttnn_detections, peaks = ref_encoder.forward(*ttnn_pred_encoded, grid)

    # ========================================================
    # Compare results

    # Check PCC on intermediates
    all_passed = True
    PCC_THRESHOLD = 0.990
    for i, (ref_out, test_out, layer_name) in enumerate(zip(intermediates, tt_intermediates, layer_names)):
        if "bbox" in layer_name:
            # bbox layers have different shape in TTNN vs torch, so skip them for now
            logger.warning(f"Skipping PCC check for bbox layer {layer_name} due to different shape in TTNN vs torch")
            continue

        passed, pcc = check_with_pcc(ref_out, test_out, PCC_THRESHOLD)
        abs, rel = get_abs_and_relative_error(ref_out, test_out)

        all_passed = all_passed and passed
        special_char = "✅" if passed else "❌"
        logger.warning(f"{special_char} Intermediate {i} {layer_name}: {passed=}, {pcc=}, {abs=:.3f}, {rel=:.3f}")

        # save latent and integral image distributions
        if "integral" in layer_name or "lat" in layer_name or "feat" in layer_name:
            # Visualize and save tensor distributions for integral layers

            fig = visualize_tensor_distributions(ref_out, test_out, title1="Reference Integral", title2="TTNN Integral")

            # Create output filename with same naming pattern as other visualizations
            test_id = f"{'scaled_' if scale_features else ''}{'fallback_ff_' if fallback_feedforward else ''}{'fallback_lat_' if fallback_lateral else ''}{'fallback_oft_' if fallback_oft else ''}host_decoder_{use_host_decoder}"
            output_file = os.path.join(output_dir, f"oft_integral_{basename}_{layer_name}_{test_id}.png")
            fig.savefig(output_file, dpi=300, bbox_inches="tight")
            logger.info(f"Saved integral tensor distribution to {output_file}")
            plt.close(fig)

    # check PCC on the encoded outputs
    # tt_scores = ttnn.to_torch(tt_scores)
    # tt_pos_offsets = ttnn.to_torch(tt_pos_offsets)
    # tt_dim_offsets = ttnn.to_torch(tt_dim_offsets)
    # tt_ang_offsets = ttnn.to_torch(tt_ang_offsets)

    all_passed = []
    ref_outs = [scores, pos_offsets, dim_offsets, ang_offsets]
    tt_outs = [tt_scores, tt_pos_offsets, tt_dim_offsets, tt_ang_offsets]
    names = ["scores", "pos_offsets", "dim_offsets", "ang_offsets"]
    expected_pcc = [pcc_scores_oft, pcc_positions_oft, pcc_dimensions_oft, pcc_angles_oft]
    for i, (ref_out, tt_out, layer_name, exp_pcc) in enumerate(zip(ref_outs, tt_outs, names, expected_pcc)):
        test_out = tt_out.reshape(ref_out.shape)  # assume it's already a torch tensor in the right format
        passed, pcc = check_with_pcc(ref_out, test_out, exp_pcc)
        abs, rel = get_abs_and_relative_error(ref_out, test_out)

        all_passed.append(passed)
        special_char = "✅" if passed else "❌"
        logger.warning(f"{special_char} Output {i} {layer_name}: {passed=}, {pcc=}, {abs=:.3f}, {rel=:.3f}")

    # =======================================================
    # Visualization
    input_tensor = input_tensor.to(torch.float32)

    # Visualize scores/heatmaps
    visualize_score(scores, tt_scores, grid.unsqueeze(0))
    plt.suptitle(basename, fontsize=16)
    plt.tight_layout()
    # Create an ID from the test parameters
    test_id = f"{'scaled_' if scale_features else ''}{'fallback_ff_' if fallback_feedforward else ''}{'fallback_lat_' if fallback_lateral else ''}{'fallback_oft_' if fallback_oft else ''}host_decoder_{use_host_decoder}"

    output_file = os.path.join(output_dir, f"oft_scores_{basename}_{test_id}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved scores comparison visualization to {output_file}")

    # Visualize predictions
    _, (ax1, ax2) = plt.subplots(nrows=2)
    # Add a super title showing the basename of the image
    plt.suptitle(basename, fontsize=16)
    input_tensor = input_tensor.squeeze(
        0
    )  # TODO(mbezulj) align all shapes to get rid of squeezing/unsqueezing at random places
    visualize_objects(input_tensor, calib, ref_detections, ax=ax1)
    ax1.set_title("Ref detections")
    visualize_objects(input_tensor, calib, ttnn_detections, ax=ax2)
    ax2.set_title("TTNN detections")

    # Save the comparison plot to a file
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"oft_detection_comparison_{basename}_{test_id}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    logger.info(f"Saved detection comparison visualization to {output_file}")

    # =======================================================
    # Fail test based on PCC results
    assert all(all_passed), f"OFTnet outputs did not pass the PCC check {all_passed=}"


def visualize_tensor_distributions(tensor1, tensor2, title1="Tensor 1", title2="Tensor 2"):
    """
    Visualizes the distribution of values in two tensors.

    Args:
        tensor1: First tensor to visualize
        tensor2: Second tensor to visualize
        title1: Title for the first tensor's histogram
        title2: Title for the second tensor's histogram

    Returns:
        matplotlib.axes.Axes: Axes object containing the plots
    """
    if isinstance(tensor1, ttnn.Tensor):
        tensor1 = ttnn.to_torch(tensor1)
    if isinstance(tensor2, ttnn.Tensor):
        tensor2 = ttnn.to_torch(tensor2)

    # Flatten tensors to 1D
    t1_flat = tensor1.float().flatten().detach().cpu().numpy()
    t2_flat = tensor2.float().flatten().detach().cpu().numpy()

    # Calculate statistics
    t1_mean, t1_std = t1_flat.mean(), t1_flat.std()
    t2_mean, t2_std = t2_flat.mean(), t2_flat.std()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot histogram for tensor1
    ax1.hist(t1_flat, bins=50, alpha=0.7)
    ax1.axvline(t1_mean, color="r", linestyle="--", label=f"Mean: {t1_mean:.4f}")
    ax1.axvline(t1_mean + t1_std, color="g", linestyle=":", label=f"Std: {t1_std:.4f}")
    ax1.axvline(t1_mean - t1_std, color="g", linestyle=":")
    ax1.set_title(f"{title1}\nMean: {t1_mean:.4f}, Std: {t1_std:.4f}")
    ax1.legend()

    # Plot histogram for tensor2
    ax2.hist(t2_flat, bins=50, alpha=0.7)
    ax2.axvline(t2_mean, color="r", linestyle="--", label=f"Mean: {t2_mean:.4f}")
    ax2.axvline(t2_mean + t2_std, color="g", linestyle=":", label=f"Std: {t2_std:.4f}")
    ax2.axvline(t2_mean - t2_std, color="g", linestyle=":")
    ax2.set_title(f"{title2}\nMean: {t2_mean:.4f}, Std: {t2_std:.4f}")
    ax2.legend()

    plt.tight_layout()
    return fig
