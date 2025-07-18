# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from loguru import logger
from skimage import io
from sklearn.model_selection import train_test_split

import ttnn


def process_single_image(
    image_path, mask_path, model, output_dir, model_type="torch_model", mesh_composer=None, mesh_mapper=None
):
    """
    Process a single MRI image and its mask using the segmentation model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read and preprocess image
    img = io.imread(image_path)
    img_resized = cv2.resize(img, (256, 256))

    # Standardize the image
    img_processed = img_resized.astype(np.float64)
    img_processed -= img_processed.mean()
    img_processed /= img_processed.std()

    # Convert to tensor format (1, 3, 256, 256)
    X = torch.from_numpy(img_processed).permute(2, 0, 1).unsqueeze(0).float()

    # Make prediction
    if model_type == "torch_model":
        with torch.no_grad():
            predict = model(X)
    else:
        n, c, h, w = X.shape
        ttnn_input = ttnn.from_torch(X, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mesh_mapper)

        predict = model.execute_vgg_unet_trace_2cqs_inference(ttnn_input)
        predict = ttnn.to_torch(predict, mesh_composer=mesh_composer)
        predict = predict.permute(0, 3, 1, 2)
        predict = predict.reshape(1, 1, 256, 256)
        predict = predict.float()
    pred_mask = predict.squeeze().numpy().round()

    # Read original mask
    original_mask = io.imread(mask_path)
    original_mask_resized = cv2.resize(original_mask, (256, 256))

    # Create visualization
    fig, axs = plt.subplots(1, 5, figsize=(30, 7))

    # # 1. Original MRI
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    axs[0].imshow(img_resized)
    axs[0].title.set_text("Brain MRI")

    # 2. Original Mask
    axs[1].imshow(original_mask_resized)
    axs[1].title.set_text("Original Mask")

    # 3. Predicted Mask
    pred_mask = np.array(pred_mask).squeeze().round()
    axs[2].imshow(pred_mask)
    axs[2].title.set_text("AI Predicted Mask")

    # 4. MRI with Original Mask overlay
    img_gt = img_resized.copy()
    img_gt[original_mask_resized == 255] = (255, 0, 0)  # Red for ground truth
    axs[3].imshow(img_gt)
    axs[3].title.set_text("MRI with Ground Truth")

    # 5. MRI with Predicted Mask overlay
    img_pred = img_resized.copy()
    img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
    img_pred[pred_mask == 1] = (0, 255, 150)
    axs[4].imshow(img_pred)
    axs[4].title.set_text("MRI with Prediction")

    # Save and display results
    output_path = os.path.join(output_dir, "single_image_result.png")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Results saved to {output_path}")


def pos_neg_diagnosis(mask_path):
    value = np.max(cv2.imread(mask_path))
    if value > 0:
        return 1
    else:
        return 0


def prediction(test, model, model_type, mesh_mapper=None, mesh_composer=None, batch_size=1):
    from math import ceil

    # empty lists to store results
    mask_list, image_id_list, has_mask_list = [], [], []

    num_images = len(test.image_path)
    num_batches = ceil(num_images / batch_size)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_images)
        if (end_idx - start_idx) < batch_size:
            break
        batch_paths = test.image_path[start_idx:end_idx]

        # Preallocate batch input array
        X = np.empty((len(batch_paths), 256, 256, 3), dtype=np.float64)

        # Load and preprocess images
        for j, img_path in enumerate(batch_paths):
            img = io.imread(img_path)
            img = cv2.resize(img, (256, 256))
            img = np.array(img, dtype=np.float64)
            img -= img.mean()
            img /= img.std()
            X[j] = img

        if model_type == "torch_model":
            X_tensor = torch.from_numpy(X).permute(0, 3, 1, 2).float()
            with torch.no_grad():
                predictions = model(X_tensor)
        else:
            X_tensor = torch.from_numpy(X).permute(0, 3, 1, 2).float()
            ttnn_input = ttnn.from_torch(
                X_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mesh_mapper
            )
            predictions = model.execute_vgg_unet_trace_2cqs_inference(ttnn_input)
            predictions = ttnn.to_torch(predictions, mesh_composer=mesh_composer)
            predictions = predictions.permute(0, 3, 1, 2)
            predictions = predictions.reshape(batch_size, 1, 256, 256).float()

        predictions_np = predictions.squeeze().numpy()

        if batch_size == 1:
            predictions_np = [predictions_np]  # Ensure it's iterable for bs=1

        for idx_in_batch, pred_mask in enumerate(predictions_np):
            img_path = batch_paths.iloc[idx_in_batch]
            if pred_mask.round().astype(int).sum() == 0:
                has_mask_list.append(0)
                mask_list.append("No mask :)")
            else:
                has_mask_list.append(1)
                mask_list.append(pred_mask)
            image_id_list.append(img_path)

    return pd.DataFrame({"image_path": image_id_list, "predicted_mask": mask_list, "has_mask": has_mask_list})


def preprocess(path, mode="default", max_samples=None):
    base_path = path
    relative_path = "lgg-mri-segmentation/kaggle_3m/"

    full_path = os.path.join(base_path, relative_path)
    data_map = []
    for sub_dir_path in glob.glob(full_path + "*"):
        # if os.path.isdir(sub_path_dir):
        try:
            dir_name = sub_dir_path.split("/")[-1]
            for filename in os.listdir(sub_dir_path):
                image_path = sub_dir_path + "/" + filename
                data_map.extend([dir_name, image_path])
        except Exception as e:
            logger.info(e)

    df = pd.DataFrame({"patient_id": data_map[::2], "path": data_map[1::2]})
    df_imgs = df[~df["path"].str.contains("mask")]  # if have not mask
    df_masks = df[df["path"].str.contains("mask")]  # if have mask

    df_imgs = df[~df["path"].str.contains("mask")]  # if have not mask
    df_masks = df[df["path"].str.contains("mask")]  # if have mask

    # File path line length images for later sorting
    BASE_LEN = len(full_path) + 44
    END_IMG_LEN = 4
    END_MASK_LEN = 9

    # Data sorting
    imgs = sorted(df_imgs["path"].values, key=lambda x: int(x[BASE_LEN:-END_IMG_LEN]))
    masks = sorted(df_masks["path"].values, key=lambda x: int(x[BASE_LEN:-END_MASK_LEN]))

    # Final dataframe
    brain_df = pd.DataFrame({"patient_id": df_imgs.patient_id.values, "image_path": imgs, "mask_path": masks})

    brain_df["mask"] = brain_df["mask_path"].apply(lambda x: pos_neg_diagnosis(x))
    brain_df

    brain_df["mask"].value_counts()

    count = 0
    i = 0
    fig, axs = plt.subplots(20, 3, figsize=(20, 50))
    for mask in brain_df["mask"]:
        if mask == 1:
            img = io.imread(brain_df.image_path[i])
            axs[count][0].title.set_text("Brain MRI")
            axs[count][0].imshow(img)

            mask = io.imread(brain_df.mask_path[i])
            axs[count][1].title.set_text("Mask")
            axs[count][1].imshow(mask, cmap="gray")

            img[mask == 255] = (255, 0, 0)  # change pixel color at the position of mask
            axs[count][2].title.set_text("MRI with Mask")
            axs[count][2].imshow(img)
            count += 1
        i += 1
        if count == 20:
            break

    fig.tight_layout()

    brain_df_train = brain_df.drop(columns=["patient_id"])
    # Convert the data in mask column to string format, to use categorical mode in flow_from_dataframe
    brain_df_train["mask"] = brain_df_train["mask"].apply(lambda x: str(x))
    brain_df_train.info()

    brain_df_mask = brain_df[brain_df["mask"] == 1].reset_index(drop=True)
    brain_df_mask.shape

    if mode == "eval":
        if max_samples is not None:
            return brain_df_mask.iloc[:max_samples]
        return brain_df_mask
    else:
        X_train, x_val = train_test_split(brain_df_mask, test_size=0.15)
        x_test, x_val = train_test_split(x_val, test_size=0.5)
        if max_samples is not None:
            return x_test.iloc[:max_samples]
        return x_test


def postprocess(df_pred, x_test, model_type):
    # merging original and prediction df
    df_pred = x_test.merge(df_pred, on="image_path")
    df_pred.head(10)

    # Define the output folder
    if model_type == "torch_model":
        output_folder = "models/demos/vgg_unet/demo/output_images"
    else:
        output_folder = "models/demos/vgg_unet/demo/output_images_ttnn"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    count = 0

    # Loop over the images in df_pred and save the plots as files
    for i in range(len(df_pred)):
        if df_pred.has_mask[i] == 1 and count < 15:
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

        if count == 20:
            break
    logger.info(f"Results saved to {output_folder}")
