"""
Script to compare generated images from local folders with reference images from HuggingFace dataset.
Computes PCC and SSIM metrics for each comparison.
"""

from pathlib import Path
import numpy as np
from PIL import Image
from datasets import load_dataset
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from tqdm import tqdm
import torch

from datasets import load_dataset


def load_hf_dataset_sample():
    print("Loading HuggingFace dataset...")
    from huggingface_hub import hf_hub_download

    # Download ONLY the parquet file, bypassing JSON completely
    print("Downloading parquet file...")
    parquet_path = hf_hub_download(repo_id="Jankoioi/nvidia-test", filename="index.parquet", repo_type="dataset")

    print(f"Loading dataset from: {parquet_path}")
    # Load using datasets library - this properly deserializes images as PIL objects
    dataset = load_dataset("parquet", data_files=parquet_path, split="train")

    print(f"Dataset loaded with {len(dataset)} rows")

    # Filter and return
    filtered = dataset.filter(lambda x: x["set"] == "accuracy" and x["model_type"] == "base").select(range(20))

    print(f"Filtered to {len(filtered)} rows")
    return filtered


def image_to_array(image):
    if isinstance(image, Image.Image):
        return np.array(image)
    return image


def compute_pcc(img1, img2):
    img1_flat = img1.flatten().astype(np.float64)
    img2_flat = img2.flatten().astype(np.float64)

    # Compute PCC
    pcc = np.corrcoef(img1_flat, img2_flat)[0, 1]
    return pcc


def compute_ssim(img1, img2):
    """
    Compute Structural Similarity Index between two images.

    Args:
        img1: First image as numpy array
        img2: Second image as numpy array

    Returns:
        SSIM value (float)
    """
    # Ensure images have the same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Images must have the same shape. Got {img1.shape} and {img2.shape}")

    # If RGB, compute SSIM for each channel and average
    if len(img1.shape) == 3:
        ssim_value = ssim(img1, img2, channel_axis=2, data_range=255)
    else:
        ssim_value = ssim(img1, img2, data_range=255)

    return ssim_value


def compute_lpips(img1, img2, lpips_model):
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity) between two images.

    Args:
        img1: First image as numpy array (H, W, C) in range [0, 255]
        img2: Second image as numpy array (H, W, C) in range [0, 255]
        lpips_model: Pre-loaded LPIPS model

    Returns:
        LPIPS value (float) - lower is better (more similar)
    """
    # Convert numpy arrays to torch tensors
    # LPIPS expects input in range [-1, 1] with shape (1, C, H, W)

    def prepare_image(img):
        # Convert to float and normalize to [-1, 1]
        img_tensor = torch.from_numpy(img).float() / 255.0 * 2.0 - 1.0
        # Rearrange from (H, W, C) to (C, H, W)
        img_tensor = img_tensor.permute(2, 0, 1)
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

    img1_tensor = prepare_image(img1)
    img2_tensor = prepare_image(img2)

    # Compute LPIPS distance
    with torch.no_grad():
        lpips_value = lpips_model(img1_tensor, img2_tensor)

    return lpips_value.item()


def load_local_image(folder_path, image_id):
    image_path = folder_path / f"output{image_id}.png"
    if not image_path.exists():
        return None
    return np.array(Image.open(image_path))


def compare_images(ref_img, test_img, lpips_model=None, metrics=["pcc", "ssim", "lpips"]):
    """
    Compare two images using specified metrics.

    Args:
        ref_img: Reference image (numpy array)
        test_img: Test image (numpy array)
        lpips_model: Pre-loaded LPIPS model (required if 'lpips' in metrics)
        metrics: List of metrics to compute ['pcc', 'ssim', 'lpips']

    Returns:
        Dictionary with metric results
    """
    results = {}

    # Ensure images are the same size
    if ref_img.shape != test_img.shape:
        # Resize test image to match reference
        test_img_pil = Image.fromarray(test_img)
        ref_shape = ref_img.shape[:2]  # height, width
        test_img_pil = test_img_pil.resize((ref_shape[1], ref_shape[0]), Image.LANCZOS)
        test_img = np.array(test_img_pil)

    if "pcc" in metrics:
        try:
            results["pcc"] = compute_pcc(ref_img, test_img)
        except Exception as e:
            print(f"Error computing PCC: {e}")
            results["pcc"] = None

    if "ssim" in metrics:
        try:
            results["ssim"] = compute_ssim(ref_img, test_img)
        except Exception as e:
            print(f"Error computing SSIM: {e}")
            results["ssim"] = None

    if "lpips" in metrics:
        try:
            if lpips_model is None:
                raise ValueError("LPIPS model is required for LPIPS computation")
            results["lpips"] = compute_lpips(ref_img, test_img, lpips_model)
        except Exception as e:
            print(f"Error computing LPIPS: {e}")
            results["lpips"] = None

    return results


def main():
    # Define paths
    base_path = Path("/localdev/jmitrovic/tt-metal/output_20_accuracy_4_comb")

    folders = {
        "device_encoders_device_vae": base_path / "___device_encoders___device_vae",
        "device_encoders_host_vae": base_path / "___device_encoders___host_vae",
        "host_encoders_device_vae": base_path / "___host_encoders___device_vae",
        "host_encoders_host_vae": base_path / "___host_encoders___host_vae",
    }

    # Initialize LPIPS model
    print("Loading LPIPS model...")
    import lpips

    lpips_model = lpips.LPIPS(net="alex")  # or 'vgg' for more accurate but slower
    lpips_model.eval()
    print("LPIPS model loaded!")

    # Load HuggingFace dataset
    hf_df = load_hf_dataset_sample()

    # Prepare results storage
    all_results = []

    for idx in tqdm(range(len(hf_df)), desc="Processing images"):
        # Get reference image from HuggingFace
        hf_row = hf_df[idx]
        sample_id = hf_row["sample_id"]
        ref_image = image_to_array(hf_row["image"])

        print(f"\n--- Sample ID: {sample_id} (index {idx}) ---")
        print(f"Reference image shape: {ref_image.shape}")

        # Compare with each folder's image
        for folder_name, folder_path in folders.items():
            # Load local image (assuming output1.png corresponds to sample_id=1)
            local_img = load_local_image(folder_path, sample_id)

            if local_img is None:
                print(f"  {folder_name}: Image not found")
                continue

            print(f"  {folder_name}: {local_img.shape}")

            # Compute metrics - PASS lpips_model here!
            metrics_result = compare_images(
                ref_image, local_img, lpips_model=lpips_model, metrics=["pcc", "ssim", "lpips"]  # <-- Add this line
            )

            # Store results
            result_row = {
                "sample_id": sample_id,
                "folder": folder_name,
                "pcc": metrics_result.get("pcc"),
                "ssim": metrics_result.get("ssim"),
                "lpips": metrics_result.get("lpips"),
            }
            all_results.append(result_row)

            print(
                f"    PCC: {metrics_result.get('pcc', 'N/A'):.6f}"
                if metrics_result.get("pcc") is not None
                else "    PCC: N/A"
            )
            print(
                f"    SSIM: {metrics_result.get('ssim', 'N/A'):.6f}"
                if metrics_result.get("ssim") is not None
                else "    SSIM: N/A"
            )
            print(
                f"    LPIPS: {metrics_result.get('lpips', 'N/A'):.6f}"
                if metrics_result.get("lpips") is not None
                else "    LPIPS: N/A"
            )

        # save results to csv
        pd.DataFrame(all_results).to_csv("results_metrics_20_images.csv", index=False)
        print(f"Results saved to results_metrics_20_images.csv")


if __name__ == "__main__":
    main()
