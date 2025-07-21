# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import urllib
import numpy as np
from scipy import linalg
import torch
from torchvision import transforms as TF
from PIL import Image
from loguru import logger
from models.experimental.stable_diffusion_xl_base.utils.inception import InceptionV3

COCO_STATISTICS_DOWNLOAD_PATH = "https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/tools/val2014.npz"


def get_activations(files, model, batch_size=1, dims=2048, device="cpu", num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        logger.warning("Warning: batch size is bigger than the data size. Setting batch size to data size")
        batch_size = len(files)

    dataset = ImagesDataset(files, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in dataloader:
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_activation_statistics(files, model):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of images
    -- model       : Instance of inception model

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path):
    if not os.path.isfile(path):
        logger.info(f"File {path} not found. Downloading...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(COCO_STATISTICS_DOWNLOAD_PATH, path)
        logger.info("Download complete.")

    with np.load(path) as f:
        m, s = f["mu"][:], f["sigma"][:]

    return m, s


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        logger.info(f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_fid_score(images, coco_statistics_path, inception_dims=2048):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images                : List of images
    -- coco_statistics_path  : Path to the precomputed statistics of the COCO dataset
    -- inception_dims        : Dimensionality of features returned by Inception
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_dims]

    model = InceptionV3([block_idx])
    model.eval()

    m1, s1 = compute_statistics_of_path(coco_statistics_path)

    m2, s2 = calculate_activation_statistics(images, model)

    # Ensure dimensions match before calculating FID
    assert s1.shape == s2.shape, f"Covariance shapes mismatch: {s1.shape} vs {s2.shape}"
    assert m1.shape == m2.shape, f"Mean shapes mismatch: {m1.shape} vs {m2.shape}"

    return calculate_frechet_distance(m1, s1, m2, s2)


class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, transforms=None):
        self.imgs = imgs
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i]
        if not isinstance(img, Image.Image):
            raise TypeError(f"Unsupported image type: {type(img)}")
        if self.transforms is not None:
            img = self.transforms(img)
        return img
