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


def process_single_image(image_path, mask_path, model, output_dir, model_type="torch_model"):
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
        # X = torch.from_numpy(X).float()

        n, c, h, w = X.shape
        X = X.permute(0, 2, 3, 1)
        X = X.reshape(1, 1, h * w * n, c)
        ttnn_input = ttnn.from_torch(X, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn_input = ttnn.pad(ttnn_input, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)

        predict = model.execute_vgg_unet_trace_2cqs_inference(ttnn_input)
        predict = ttnn.to_torch(predict)
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


def prediction(test, model, model_type):
    # empty list to store results
    mask, image_id, has_mask = [], [], []

    # itetrating through each image in test data
    for i in test.image_path:
        # Creating a empty array of shape 1,256,256,1
        X = np.empty((1, 256, 256, 3))
        # read the image
        img = io.imread(i)
        # resizing the image and coverting them to array of type float64
        img = cv2.resize(img, (256, 256))
        img = np.array(img, dtype=np.float64)

        # standardising the image
        img -= img.mean()
        img /= img.std()
        # converting the shape of image from 256,256,3 to 1,256,256,3
        X[0,] = img
        if model_type == "torch_model":
            X = torch.from_numpy(X).permute(0, 3, 1, 2).float()
            # make prediction of mask
            with torch.no_grad():
                predict = model(X)
        else:
            X = torch.from_numpy(X).float()
            n, h, w, c = X.shape
            X = X.reshape(1, 1, h * w * n, c)
            ttnn_input = ttnn.from_torch(X, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn_input = ttnn.pad(ttnn_input, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)
            predict = model.execute_vgg_unet_trace_2cqs_inference(ttnn_input)
            predict = ttnn.to_torch(predict)
            predict = predict.permute(0, 3, 1, 2)
            predict = predict.reshape(1, 1, 256, 256)
            predict = predict.float()

        predict = predict.squeeze().numpy()
        # if sum of predicted mask is 0 then there is not tumour
        if predict.round().astype(int).sum() == 0:
            image_id.append(i)
            has_mask.append(0)
            mask.append("No mask :)")
        else:
            # if the sum of pixel values are more than 0, then there is tumour
            image_id.append(i)
            has_mask.append(1)
            mask.append(predict)

    return pd.DataFrame({"image_path": image_id, "predicted_mask": mask, "has_mask": has_mask})


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
            print(e)

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
