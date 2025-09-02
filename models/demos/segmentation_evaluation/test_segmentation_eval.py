# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import argparse
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
from skimage.io import imsave
from tqdm import tqdm

import ttnn
from models.demos.vanilla_unet.common import VANILLA_UNET_L1_SMALL_SIZE
from models.demos.yolov9c.common import YOLOV9C_L1_SMALL_SIZE
from models.utility_functions import disable_persistent_kernel_cache


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
    if model_name == "vanilla_unet":
        from collections import defaultdict

        from models.demos.vanilla_unet.demo import demo_utils

        root_dir = "models/demos/segmentation_evaluation/imageset"
        patient_folders = sorted(os.listdir(root_dir))
        if model_location_generator == None or "TT_GH_CI_INFRA" not in os.environ:
            weights_path = "models/demos/vanilla_unet/unet.pt"
        else:
            weights_path = (
                model_location_generator("vision-models/unet_vanilla", model_subdir="", download_if_ci_v2=True)
                / "unet.pt"
            )
        sample_count = 0
        max_samples = 500
        all_patient_metrics = defaultdict(list)
        for patient_id in patient_folders:
            if sample_count >= max_samples:
                break
            patient_path = os.path.join("models/demos/segmentation_evaluation/imageset", patient_id)
            patient_output_path = os.path.join("models/demos/segmentation_evaluation/pred_image_set", patient_id)
            args = argparse.Namespace(
                device="cpu",
                batch_size=batch_size,
                weights="models/demos/vanilla_unet/unet.pt",
                images=patient_path,
                image_size=res,
                predictions=patient_output_path,
            )
            loader = demo_utils.data_loader_imageset(args)
            os.makedirs(patient_output_path, exist_ok=True)

            input_list = []
            pred_list = []
            true_list = []
            for i, data in tqdm(enumerate(loader)):
                x, y_true = data
                if x.shape[0] < args.batch_size:
                    logger.info(f"Skipping incomplete batch at index {i}, size {x.shape[0]}")
                    continue
                if model_type == "torch_model":
                    y_pred = model(x)
                else:
                    y_pred = model.run(x)
                    y_pred = ttnn.to_torch(y_pred, mesh_composer=model.runner_infra.output_mesh_composer)
                    y_pred = y_pred.permute(0, 3, 1, 2).to(torch.float32)

                y_pred_np = y_pred.detach().cpu().numpy()
                pred_list.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])

                y_true_np = y_true.detach().cpu().numpy()
                true_list.extend([y_true_np[s] for s in range(y_true_np.shape[0])])

                x_np = x.detach().cpu().numpy()
                input_list.extend([x_np[s] for s in range(x_np.shape[0])])

                sample_count += y_pred_np.shape[0]
            volumes = demo_utils.postprocess_per_volume(
                input_list,
                pred_list,
                true_list,
                loader.dataset.patient_slice_index,
                loader.dataset.patients,
            )

            metrics_list = []
            for p in volumes:
                x = volumes[p][0]
                y_pred = volumes[p][1]
                y_true = volumes[p][2]
                y_true = (y_true == 255).astype(np.uint8)  # Convert 255 → 1
                y_pred = y_pred.astype(np.uint8)

                metrics = {
                    "IoU": iou(y_true, y_pred) * 100,
                    "Dice Score": dice_score(y_true, y_pred) * 100,
                    "Pixel Accuracy": pixel_accuracy(y_true, y_pred) * 100,
                    "Precision": precision(y_true, y_pred) * 100,
                    "Recall": recall(y_true, y_pred) * 100,
                    "F1 Score": f1_score(y_true, y_pred) * 100,
                }

                for key, value in metrics.items():
                    all_patient_metrics[key].append(value)

                for s in range(x.shape[0]):
                    image = demo_utils.gray2rgb(x[s, 1])  # channel 1 is for FLAIR
                    image = demo_utils.outline(image, y_pred[s, 0], color=[255, 0, 0])
                    image = demo_utils.outline(image, y_true[s, 0], color=[0, 255, 0])
                    filename = "{}-{}.png".format(p, str(s).zfill(2))
                    filepath = os.path.join(args.predictions, filename)
                    imsave(filepath, image)

        final_avg_metrics = {key: np.mean(vals) for key, vals in all_patient_metrics.items()}
        for key, val in final_avg_metrics.items():
            logger.info(f"{key}: {val:.2f}%")

    if model_name == "vgg_unet":
        from models.demos.vgg_unet.demo.demo_utils import prediction, preprocess

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
                output_folder = "models/demos/vgg_unet/demo/output_images_dp"
            else:
                output_folder = "models/demos/vgg_unet/demo/output_images"
        else:
            if (device.get_num_devices()) > 1:
                output_folder = "models/demos/vgg_unet/demo/output_images_ttnn_dp"
            else:
                output_folder = "models/demos/vgg_unet/demo/output_images_ttnn"
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

    if model_name == "yolov9c":
        import json
        from datetime import datetime

        import fiftyone

        from models.demos.utils.common_demo_utils import LoadImages, preprocess
        from models.demos.yolov9c.demo.demo_utils import get_consistent_color, postprocess
        from models.demos.yolov9c.runner.performant_runner import YOLOv9PerformantRunner

        dataset_name = "coco-2017"
        if model_type == "torch_model":
            dataset = fiftyone.zoo.load_zoo_dataset(
                dataset_name,
                split="validation",
                max_samples=18,
                label_types=["segmentations"],
                include_id=True,
            )
        performant_runner = None
        if model_type == "tt_model":
            dataset = fiftyone.zoo.load_zoo_dataset(
                dataset_name,
                split="validation",
                max_samples=200,
                label_types=["segmentations"],
                include_id=True,
            )
            performant_runner = YOLOv9PerformantRunner(
                device,
                1,
                ttnn.bfloat8_b,
                ttnn.bfloat8_b,
                model_task="segment",
                resolution=(640, 640),
                model_location_generator=model_location_generator,
            )

        def load_coco_gt_mask(sample):
            detections = sample["segmentations"].detections
            height = sample.metadata.height
            width = sample.metadata.width

            gt_mask = np.zeros((height, width), dtype=np.uint8)

            for det in detections:
                if det.mask is not None and det.bounding_box is not None:
                    mask = det.mask.astype(bool)  # (h, w)
                    ymin, xmin, box_h, box_w = det.bounding_box  # normalized
                    y = int(ymin * height)
                    x = int(xmin * width)
                    h = mask.shape[0]
                    w = mask.shape[1]
                    y2 = min(y + h, height)
                    x2 = min(x + w, width)
                    gt_mask[y:y2, x:x2] |= mask[: y2 - y, : x2 - x]

            return gt_mask.astype(np.uint8)

        source_list = [i["filepath"] for i in dataset]
        data_set = LoadImages(path=[i["filepath"] for i in dataset])
        save_dir = "models/demos/yolov9c/demo/runs"

        with open(os.path.expanduser("~") + "/fiftyone" + "/" + dataset_name + "/info.json", "r") as file:
            data = json.load(file)
            classes = data["classes"]

        model_save_dir = os.path.join(save_dir, model_type)
        os.makedirs(model_save_dir, exist_ok=True)
        index = 0
        sample_count = 0

        iou_list, dice_list, acc_list, prec_list, recall_list, f1_list = [], [], [], [], [], []
        for sample, batch in zip(dataset, data_set):
            paths, im0s, s = batch
            im = preprocess(im0s, res=(640, 640))

            if model_type == "torch_model":
                preds = model(im)
                results = postprocess(preds, im, im0s, batch)
                os.makedirs(os.path.join(save_dir, model_type), exist_ok=True)

                image = cv2.imread(source_list[index])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                masks = results[0].masks.data.cpu().detach().numpy()
                pred_mask = np.any(masks, axis=0).astype(np.uint8)

                gt_mask = load_coco_gt_mask(sample)
                pred_mask_resized = cv2.resize(
                    pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST
                )

                iou_list.append(iou(gt_mask, pred_mask) * 100)
                dice_list.append(dice_score(gt_mask, pred_mask) * 100)
                acc_list.append(pixel_accuracy(gt_mask, pred_mask) * 100)
                prec_list.append(precision(gt_mask, pred_mask) * 100)
                recall_list.append(recall(gt_mask, pred_mask) * 100)
                f1_list.append(f1_score(gt_mask, pred_mask) * 100)

                sample_count += 1
                logger.info(f"sample_count {sample_count}")
                if sample_count <= 10:
                    mask_h, mask_w = masks.shape[1], masks.shape[2]

                    image = cv2.resize(image, (mask_w, mask_h))
                    overlay = image.copy()

                    for i in range(len(masks)):
                        mask = masks[i]
                        color = get_consistent_color(i)
                        mask_rgb = np.zeros_like(image, dtype=np.uint8)
                        for c in range(3):
                            mask_rgb[:, :, c] = (mask * color[c]).astype(np.uint8)

                        mask_bool = mask.astype(bool)
                        overlay[mask_bool] = (0.5 * overlay[mask_bool] + 0.5 * mask_rgb[mask_bool]).astype(np.uint8)

                    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    out_path = os.path.join(save_dir, model_type, f"segmentation_{timestamp}.jpg")
                    cv2.imwrite(out_path, overlay_bgr)
                    logger.info(f"Saved to {out_path}")
                index += 1
            else:
                preds = performant_runner.run(torch_input_tensor=im)
                preds = [
                    ttnn.to_torch(preds[0], dtype=torch.float32),
                    [
                        [ttnn.to_torch(t, dtype=torch.float32) for t in preds[1][0]],
                        ttnn.to_torch(preds[1][1], dtype=torch.float32),
                        ttnn.to_torch(preds[1][2], dtype=torch.float32)
                        .reshape((1, 160, 160, 32))
                        .permute((0, 3, 1, 2)),
                    ],
                ]
                skipped_sample = 0
                try:
                    results = postprocess(preds, im, im0s, batch)
                    for i in range(len(results)):
                        os.makedirs(os.path.join(save_dir, "tt_model"), exist_ok=True)
                        image = cv2.imread(paths[i])
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        if results[i].masks is None:
                            logger.warning(f"Skipping sample {paths[i]} due to missing mask in results[{i}]")
                            continue
                        try:
                            masks = results[i].masks.data.cpu().detach().numpy()
                            pred_mask = np.any(masks, axis=0).astype(np.uint8)

                            gt_mask = load_coco_gt_mask(sample)
                            pred_mask_resized = cv2.resize(
                                pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST
                            )

                            iou_list.append(iou(gt_mask, pred_mask) * 100)
                            dice_list.append(dice_score(gt_mask, pred_mask) * 100)
                            acc_list.append(pixel_accuracy(gt_mask, pred_mask) * 100)
                            prec_list.append(precision(gt_mask, pred_mask) * 100)
                            recall_list.append(recall(gt_mask, pred_mask) * 100)
                            f1_list.append(f1_score(gt_mask, pred_mask) * 100)

                            sample_count += 1
                            logger.info(f"sample_count {sample_count}")
                            if sample_count <= 10:
                                mask_h, mask_w = masks.shape[1], masks.shape[2]

                                image = cv2.resize(image, (mask_w, mask_h))
                                overlay = image.copy()

                                for i in range(len(masks)):
                                    mask = masks[i]
                                    color = get_consistent_color(i)
                                    mask_rgb = np.zeros_like(image, dtype=np.uint8)
                                    for c in range(3):
                                        mask_rgb[:, :, c] = (mask * color[c]).astype(np.uint8)

                                    mask_bool = mask.astype(bool)
                                    overlay[mask_bool] = (0.5 * overlay[mask_bool] + 0.5 * mask_rgb[mask_bool]).astype(
                                        np.uint8
                                    )

                                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                out_path = os.path.join(save_dir, "tt_model", f"segmentation_{timestamp}.jpg")
                                cv2.imwrite(out_path, overlay_bgr)
                                logger.info(f"Saved to {out_path}")
                        except Exception as e:
                            logger.warning(f"Error processing result {i} in sample {paths[i]}: {e}")
                            continue
                except Exception as e:
                    logger.warning(f"Failed to postprocess sample {paths[i]}: {e}")
                    skipped_sample += 1
                    logger.info(f"skipped_sample: {skipped_sample}")

        if performant_runner is not None:
            performant_runner.release()

        logger.info(f"Sample Count: {sample_count}")
        logger.info(f"IoU: {np.mean(iou_list):.2f}%")
        logger.info(f"Dice Score: {np.mean(dice_list):.2f}%")
        logger.info(f"Pixel Accuracy: {np.mean(acc_list):.2f}%")
        logger.info(f"Precision: {np.mean(prec_list):.2f}%")
        logger.info(f"Recall: {np.mean(recall_list):.2f}%")
        logger.info(f"F1 Score: {np.mean(f1_list):.2f}%")

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


def run_vanilla_unet(device, model_type, res, model_location_generator, reset_seeds, batch_size):
    from models.demos.vanilla_unet.common import load_torch_model
    from models.demos.vanilla_unet.runner.performant_runner import VanillaUNetPerformantRunner

    total_batch_size = batch_size * device.get_num_devices()
    reference_model = load_torch_model(model_location_generator)
    ttnn_model = VanillaUNetPerformantRunner(
        device,
        batch_size,
        act_dtype=ttnn.bfloat8_b,
        weight_dtype=ttnn.bfloat8_b,
        model_location_generator=model_location_generator,
    )

    if not os.path.exists("models/demos/segmentation_evaluation/imageset"):
        os.system("python models/demos/segmentation_evaluation/dataset_download.py vanilla_unet")

    model_name = "vanilla_unet"
    input_dtype = ttnn.bfloat16
    input_memory_config = ttnn.L1_MEMORY_CONFIG
    evaluation(
        device=device,
        res=res,
        model_type=model_type,
        model=ttnn_model if model_type == "tt_model" else reference_model,
        input_dtype=input_dtype,
        input_memory_config=input_memory_config,
        model_name=model_name,
        batch_size=total_batch_size,
    )


def run_vgg_unet(
    device, model_type, use_pretrained_weight, res, model_location_generator, reset_seeds, device_batch_size
):
    from models.demos.vgg_unet.common import load_torch_model
    from models.demos.vgg_unet.reference.vgg_unet import UNetVGG19
    from models.demos.vgg_unet.runner.performant_runner import VggUnetTrace2CQ

    disable_persistent_kernel_cache()

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
    "batch_size",
    ((1),),
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": VANILLA_UNET_L1_SMALL_SIZE, "trace_region_size": 1605632, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("res", [(480, 640)])
def test_vanilla_unet(device, model_type, res, model_location_generator, reset_seeds, batch_size):
    return run_vanilla_unet(device, model_type, res, model_location_generator, reset_seeds, batch_size)


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
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": VANILLA_UNET_L1_SMALL_SIZE, "trace_region_size": 1605632, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize("res", [(480, 640)])
def test_vanilla_unet_dp(mesh_device, model_type, res, model_location_generator, reset_seeds, device_batch_size):
    return run_vanilla_unet(mesh_device, model_type, res, model_location_generator, reset_seeds, device_batch_size)


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


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV9C_L1_SMALL_SIZE, "trace_region_size": 23887872, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        "True",
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        ("tt_model"),
        ("torch_model"),
    ],
)
@pytest.mark.parametrize(
    "model_task",
    [
        "segment",
    ],
)
@pytest.mark.parametrize("res", [(640, 640)])
def test_yolov9c(
    device,
    model_type,
    model_task,
    use_weights_from_ultralytics,
    res,
    model_location_generator,
    reset_seeds,
):
    from models.demos.yolov9c.common import load_torch_model

    disable_persistent_kernel_cache()
    enable_segment = model_task == "segment"

    torch_model = load_torch_model(model_location_generator=model_location_generator, model_task=model_task)

    model_name = "yolov9c"
    input_dtype = ttnn.bfloat16
    input_memory_config = ttnn.L1_MEMORY_CONFIG

    evaluation(
        device=device,
        res=res,
        model_type=model_type,
        model=torch_model,
        input_dtype=input_dtype,
        input_memory_config=input_memory_config,
        model_name=model_name,
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
