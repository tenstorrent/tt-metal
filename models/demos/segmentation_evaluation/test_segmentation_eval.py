# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
import os

import cv2
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from loguru import logger
from skimage import io

import ttnn
from models.common.utility_functions import is_blackhole, is_wormhole_b0


def iou(y_true, y_pred):
    """Computes Intersection over Union (IoU)."""
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return intersection / union if union != 0 else 0


def dice_score(y_true, y_pred):
    """Computes Dice Score (F1 Score for segmentation)."""
    intersection = np.logical_and(y_true, y_pred).sum()
    return (2 * intersection) / (y_true.sum() + y_pred.sum()) if (y_true.sum() + y_pred.sum()) != 0 else 0


def pixel_accuracy(y_true, y_pred):
    """Computes Pixel Accuracy."""
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    """Computes Precision (Positive Predictive Value)."""
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    return tp / (tp + fp) if (tp + fp) != 0 else 0


def recall(y_true, y_pred):
    """Computes Recall (Sensitivity)."""
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()
    return tp / (tp + fn) if (tp + fn) != 0 else 0


def f1_score(y_true, y_pred):
    """Computes F1 Score (Harmonic Mean of Precision and Recall)."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (2 * p * r) / (p + r) if (p + r) != 0 else 0


def evaluation(
    device,
    res,
    model_type,
    model,
    input_dtype,
    input_memory_config=None,
    model_name=None,
    config=None,
    batch_size=1,
    model_location_generator=None,
):
    if model_name == "vgg_unet":
        if is_blackhole():
            from models.demos.blackhole.vgg_unet.demo.demo_utils import prediction, preprocess

            output_path = "models/demos/blackhole/"
        elif is_wormhole_b0():
            from models.demos.wormhole.vgg_unet.demo.demo_utils import prediction, preprocess

            output_path = "models/demos/wormhole/"
        else:
            raise RuntimeError("Unsupported device: Only Blackhole and Wormhole are supported for this test.")

        path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")
        for dirname, _, filenames in os.walk("/kaggle/input"):
            for filename in filenames:
                logger.info(f"{os.path.join(dirname, filename)}")

        sample_count = 500
        X_test = preprocess(path, mode="eval", max_samples=550)
        if model_type == "torch_model":
            df_pred = prediction(X_test, model, model_type, batch_size=batch_size)
        else:
            df_pred = prediction(X_test, model, model_type, batch_size=batch_size)

        df_pred = X_test.merge(df_pred, on="image_path")
        df_pred.head(10)

        # Define the output folder
        if model_type == "torch_model":
            if (device.get_num_devices()) > 1:
                output_folder = output_path + "vgg_unet/demo/output_images_dp"
            else:
                output_folder = output_path + "vgg_unet/demo/output_images"
        else:
            if (device.get_num_devices()) > 1:
                output_folder = output_path + "vgg_unet/demo/output_images_ttnn_dp"
            else:
                output_folder = output_path + "vgg_unet/demo/output_images_ttnn"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        iou_list, dice_list, acc_list, prec_list, recall_list, f1_list = [], [], [], [], [], []
        count = 0

        # Loop over the images in df_pred and save the plots as files
        for i in range(len(df_pred)):
            if df_pred.has_mask[i] == 1:
                pred_mask = np.array(df_pred.predicted_mask[i]).squeeze()
                pred_mask = (pred_mask > 0.5).astype(np.uint8)

                gt_mask = io.imread(df_pred.mask_path[i])
                gt_mask = (gt_mask > 127).astype(np.uint8)

                # Resize if needed
                if pred_mask.shape != gt_mask.shape:
                    pred_mask = cv2.resize(
                        pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST
                    )

                # Compute metrics
                iou_list.append(iou(gt_mask, pred_mask) * 100)
                dice_list.append(dice_score(gt_mask, pred_mask) * 100)
                acc_list.append(pixel_accuracy(gt_mask, pred_mask) * 100)
                prec_list.append(precision(gt_mask, pred_mask) * 100)
                recall_list.append(recall(gt_mask, pred_mask) * 100)
                f1_list.append(f1_score(gt_mask, pred_mask) * 100)

                if count < 15:
                    # Create a new figure for each image and save it
                    fig, axs = plt.subplots(1, 5, figsize=(30, 7))

                    # Read MRI image
                    img = io.imread(df_pred.image_path[i])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axs[0].imshow(img)
                    axs[0].title.set_text("Brain MRI")

                    # Read original mask
                    mask = io.imread(df_pred.mask_path[i])
                    axs[1].imshow(mask)
                    axs[1].title.set_text("Original Mask")

                    # Read predicted mask
                    pred = np.array(df_pred.predicted_mask[i]).squeeze().round()
                    axs[2].imshow(pred)
                    axs[2].title.set_text("AI Predicted Mask")

                    # Overlay original mask with MRI
                    img[mask == 255] = (255, 0, 0)
                    axs[3].imshow(img)
                    axs[3].title.set_text("MRI with Original Mask (Ground Truth)")

                    # Overlay predicted mask with MRI
                    img_ = io.imread(df_pred.image_path[i])
                    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
                    img_[pred == 1] = (0, 255, 150)
                    axs[4].imshow(img_)
                    axs[4].title.set_text("MRI with AI Predicted Mask")

                    # Save the figure as a PNG file in the output folder
                    output_file = os.path.join(output_folder, f"image_{count+1}.png")
                    fig.tight_layout()
                    plt.savefig(output_file)

                    # Close the figure to avoid memory issues when saving many images
                    plt.close(fig)

                    count += 1

            if count == sample_count:
                break

        logger.info(f"IoU: {np.mean(iou_list):.2f}%")
        logger.info(f"Dice Score: {np.mean(dice_list):.2f}%")
        logger.info(f"Pixel Accuracy: {np.mean(acc_list):.2f}%")
        logger.info(f"Precision: {np.mean(prec_list):.2f}%")
        logger.info(f"Recall: {np.mean(recall_list):.2f}%")
        logger.info(f"F1 Score: {np.mean(f1_list):.2f}%")
        logger.info(f"Results saved to {output_folder}")

    # Note: YOLOv9c evaluation code was removed pending verification of usage rights.
    # Previously, this file contained evaluation tests for YOLOv9c segmentation.

    if model_name == "yolov9c":
        raise NotImplementedError(
            "YOLOv9c evaluation has been temporarily removed pending verification of usage rights. "
            "Please use one of the other supported models: Vanilla UNet, VGG UNet, or Segformer."
        )

    if model_name == "segformer":
        from torch.utils.data import DataLoader
        from transformers import AutoImageProcessor

        from models.demos.segformer.demo.demo_for_semantic_segmentation import (
            SemanticSegmentationDataset,
            custom_collate_fn,
            shift_gt_indices,
        )

        image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

        image_folder = "models/demos/segformer/demo/validation_data_ade20k/images"
        mask_folder = "models/demos/segformer/demo/validation_data_ade20k/annotations"

        dataset = SemanticSegmentationDataset(image_folder, mask_folder, image_processor)
        data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn
        )
        logger.info(f"Evaluating on {len(dataset)} samples...")

        max_samples = 2000
        sample_count = 0
        iou_list, dice_list, acc_list, prec_list, recall_list, f1_list = [], [], [], [], [], []
        for batch in data_loader:
            masks = batch["gt_mask"]
            input = batch["pixel_values"]
            if model_type == "tt_model":
                ttnn_output = model.run(input)
                ttnn_output = ttnn.to_torch(ttnn_output, mesh_composer=model.test_infra.output_mesh_composer)
                ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))
                h = w = int(math.sqrt(ttnn_output.shape[-1]))
                ttnn_final_output = torch.reshape(ttnn_output, (ttnn_output.shape[0], ttnn_output.shape[1], h, w))
                for i in range(len(masks)):
                    mask = masks[i]
                    ttnn_up = torch.nn.functional.interpolate(
                        ttnn_final_output[i : i + 1], size=mask.shape, mode="bilinear", align_corners=False
                    )
                    ttnn_pred_mask = ttnn_up.argmax(dim=1).squeeze().cpu().numpy()
                    mask = shift_gt_indices(mask)
                    mask = np.array(mask)

                    iou_list.append(iou(mask, ttnn_pred_mask) * 100)
                    dice_list.append(dice_score(mask, ttnn_pred_mask) * 100)
                    acc_list.append(pixel_accuracy(mask, ttnn_pred_mask) * 100)
                    prec_list.append(precision(mask, ttnn_pred_mask) * 100)
                    recall_list.append(recall(mask, ttnn_pred_mask) * 100)
                    f1_list.append(f1_score(mask, ttnn_pred_mask) * 100)
                    sample_count += 1
                    logger.info(f"sample_count: {sample_count}")
                    if sample_count == max_samples:
                        break
            if model_type == "torch_model":
                ref_logits = model(input).logits
                for i in range(len(masks)):
                    mask = masks[i]
                    ref_up = torch.nn.functional.interpolate(
                        ref_logits[i : i + 1], size=mask.shape, mode="bilinear", align_corners=False
                    )
                    ref_pred_mask = ref_up.argmax(dim=1).squeeze().cpu().numpy()
                    mask = shift_gt_indices(mask)
                    mask = np.array(mask)
                    iou_list.append(iou(mask, ref_pred_mask) * 100)
                    dice_list.append(dice_score(mask, ref_pred_mask) * 100)
                    acc_list.append(pixel_accuracy(mask, ref_pred_mask) * 100)
                    prec_list.append(precision(mask, ref_pred_mask) * 100)
                    recall_list.append(recall(mask, ref_pred_mask) * 100)
                    f1_list.append(f1_score(mask, ref_pred_mask) * 100)
                    sample_count += 1
                    logger.info(f"sample_count: {sample_count}")
                    if sample_count == max_samples:
                        break
        logger.info(f"IoU: {np.mean(iou_list):.2f}%")
        logger.info(f"Dice Score: {np.mean(dice_list):.2f}%")
        logger.info(f"Pixel Accuracy: {np.mean(acc_list):.2f}%")
        logger.info(f"Precision: {np.mean(prec_list):.2f}%")
        logger.info(f"Recall: {np.mean(recall_list):.2f}%")
        logger.info(f"F1 Score: {np.mean(f1_list):.2f}%")


def run_vgg_unet(
    device, model_type, use_pretrained_weight, res, model_location_generator, reset_seeds, device_batch_size
):
    from models.demos.vgg_unet.common import load_torch_model
    from models.demos.vgg_unet.reference.vgg_unet import UNetVGG19
    from models.demos.vgg_unet.runner.performant_runner import VggUnetTrace2CQ

    model_seg = UNetVGG19()
    if use_pretrained_weight:
        model_seg = load_torch_model(model_seg, model_location_generator)
    model_seg.eval()
    batch_size = device_batch_size * device.get_num_devices()
    if model_type == "tt_model":
        vgg_unet_trace_2cq = VggUnetTrace2CQ()

        vgg_unet_trace_2cq.initialize_vgg_unet_trace_2cqs_inference(
            device,
            model_location_generator=model_location_generator,
            use_pretrained_weight=use_pretrained_weight,
            device_batch_size=device_batch_size,
        )

    model_name = "vgg_unet"
    input_dtype = ttnn.bfloat16
    input_memory_config = ttnn.L1_MEMORY_CONFIG
    evaluation(
        device=device,
        res=res,
        model_type=model_type,
        model=vgg_unet_trace_2cq if model_type == "tt_model" else model_seg,
        input_dtype=input_dtype,
        input_memory_config=input_memory_config,
        model_name=model_name,
        batch_size=batch_size,
    )


@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        True,
    ],
    ids=[
        "pretrained_weight_true",
    ],
)
@pytest.mark.parametrize(
    "device_batch_size",
    ((1),),
)
@pytest.mark.parametrize("res", [(256, 256)])
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
def test_vgg_unet_dp(
    mesh_device, model_type, use_pretrained_weight, res, model_location_generator, reset_seeds, device_batch_size
):
    return run_vgg_unet(
        mesh_device, model_type, use_pretrained_weight, res, model_location_generator, reset_seeds, device_batch_size
    )


@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        True,
    ],
    ids=[
        "pretrained_weight_true",
    ],
)
@pytest.mark.parametrize(
    "batch_size",
    ((1),),
)
@pytest.mark.parametrize("res", [(256, 256)])
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
def test_vgg_unet(device, model_type, use_pretrained_weight, res, model_location_generator, reset_seeds, batch_size):
    return run_vgg_unet(
        device, model_type, use_pretrained_weight, res, model_location_generator, reset_seeds, batch_size
    )


def run_segformer_eval(device, model_location_generator, model_type, res, device_batch_size):
    from models.demos.segformer.common import load_config, load_torch_model
    from models.demos.segformer.reference.segformer_for_semantic_segmentation import (
        SegformerForSemanticSegmentationReference,
    )
    from models.demos.segformer.runner.performant_runner import SegformerTrace2CQ

    config = load_config("configs/segformer_semantic_config.json")
    reference_model = SegformerForSemanticSegmentationReference(config)
    target_prefix = f""
    reference_model = load_torch_model(
        reference_model, target_prefix, module="semantic_sub", model_location_generator=model_location_generator
    )
    reference_model.eval()
    batch_size = device_batch_size * device.get_num_devices()
    segformer_trace_2cq = SegformerTrace2CQ()
    segformer_trace_2cq.initialize_segformer_trace_2cqs_inference(
        device, model_location_generator=model_location_generator, device_batch_size=device_batch_size
    )

    if not os.path.exists("models/demos/segformer/demo/validation_data_ade20k"):
        logger.info("downloading data")
        os.system("python models/demos/segmentation_evaluation/dataset_download.py segformer")

    model_name = "segformer"
    input_dtype = ttnn.bfloat16
    input_memory_config = ttnn.L1_MEMORY_CONFIG

    evaluation(
        device=device,
        res=res,
        model_type=model_type,
        model=segformer_trace_2cq if model_type == "tt_model" else reference_model,
        input_dtype=input_dtype,
        input_memory_config=input_memory_config,
        model_name=model_name,
        config=reference_model.config,
        batch_size=batch_size,
    )


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
@pytest.mark.parametrize(
    "batch_size",
    ((1),),
)
@pytest.mark.parametrize("res", [(512, 512)])
def test_segformer_eval(device, model_location_generator, model_type, res, batch_size):
    return run_segformer_eval(device, model_location_generator, model_type, res, batch_size)


@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 6434816, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
@pytest.mark.parametrize(
    "device_batch_size",
    ((1),),
)
@pytest.mark.parametrize("res", [(512, 512)])
def test_segformer_eval_dp(mesh_device, model_location_generator, model_type, res, device_batch_size):
    return run_segformer_eval(mesh_device, model_location_generator, model_type, res, device_batch_size)
