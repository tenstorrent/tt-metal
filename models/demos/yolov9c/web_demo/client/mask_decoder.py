# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Client-side utility for decoding compressed masks from the YOLOv9c segmentation server.
"""

import base64
import gzip
from typing import Dict, List

import numpy as np


def decode_binary_compressed_mask(mask_data: Dict) -> np.ndarray:
    """
    Decode a binary compressed mask.

    Args:
        mask_data: Dictionary containing compressed mask data with format:
                  {"data": base64_string, "shape": [height, width], "format": "binary_compressed"}

    Returns:
        numpy.ndarray: Decoded binary mask (height, width) with values 0 or 1
    """
    if mask_data["format"] != "binary_compressed":
        raise ValueError(f"Expected format 'binary_compressed', got {mask_data['format']}")

    # Decode base64
    compressed_bytes = base64.b64decode(mask_data["data"])

    # Decompress with gzip
    packed_bytes = gzip.decompress(compressed_bytes)

    # Convert back to numpy array
    packed_data = np.frombuffer(packed_bytes, dtype=np.uint8)

    # Unpack binary data (8 pixels per byte)
    height, width = mask_data["shape"]
    total_pixels = height * width
    binary_mask = np.unpackbits(packed_data)[:total_pixels]

    # Reshape to original dimensions
    mask = binary_mask.reshape(height, width).astype(np.uint8)

    return mask


def decode_rle_mask(mask_data: Dict) -> np.ndarray:
    """
    Decode a Run-Length Encoded mask.

    Args:
        mask_data: Dictionary containing RLE mask data with format:
                  {"data": rle_list, "shape": [height, width], "format": "rle"}

    Returns:
        numpy.ndarray: Decoded binary mask (height, width) with values 0 or 1
    """
    if mask_data["format"] != "rle":
        raise ValueError(f"Expected format 'rle', got {mask_data['format']}")

    height, width = mask_data["shape"]
    total_pixels = height * width

    # Decode RLE
    flat_mask = []
    current_val = 0  # Start with 0

    for count in mask_data["data"]:
        flat_mask.extend([current_val] * count)
        current_val = 1 - current_val  # Toggle between 0 and 1

    # Ensure we have the right number of pixels
    if len(flat_mask) != total_pixels:
        # Pad or truncate if necessary
        if len(flat_mask) < total_pixels:
            flat_mask.extend([0] * (total_pixels - len(flat_mask)))
        else:
            flat_mask = flat_mask[:total_pixels]

    # Reshape to original dimensions
    mask = np.array(flat_mask, dtype=np.uint8).reshape(height, width)

    return mask


def decode_raw_mask(mask_data: Dict) -> np.ndarray:
    """
    Decode a raw mask (no compression).

    Args:
        mask_data: Dictionary containing raw mask data with format:
                  {"data": list_of_lists, "format": "raw"}

    Returns:
        numpy.ndarray: Decoded mask with original float values
    """
    if mask_data["format"] != "raw":
        raise ValueError(f"Expected format 'raw', got {mask_data['format']}")

    return np.array(mask_data["data"])


def decode_mask(mask_data: Dict) -> np.ndarray:
    """
    Universal mask decoder that automatically detects format and decodes accordingly.

    Args:
        mask_data: Dictionary containing mask data with format information

    Returns:
        numpy.ndarray: Decoded mask
    """
    format_type = mask_data.get("format", "raw")

    if format_type == "binary_compressed":
        return decode_binary_compressed_mask(mask_data)
    elif format_type == "rle":
        return decode_rle_mask(mask_data)
    elif format_type == "raw":
        return decode_raw_mask(mask_data)
    else:
        raise ValueError(f"Unknown mask format: {format_type}")


def decode_segmentation_response(response: Dict) -> List[np.ndarray]:
    """
    Decode all masks from a segmentation response.

    Args:
        response: Dictionary containing segmentation response with masks

    Returns:
        List[numpy.ndarray]: List of decoded masks
    """
    masks = []

    for mask_data in response.get("masks", []):
        decoded_mask = decode_mask(mask_data)
        masks.append(decoded_mask)

    return masks


def visualize_mask(mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create a visualization of a binary mask with transparency.

    Args:
        mask: Binary mask array (0s and 1s)
        alpha: Transparency level (0.0 to 1.0)

    Returns:
        numpy.ndarray: RGBA visualization array
    """
    height, width = mask.shape

    # Create RGBA array
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    # Set alpha channel
    rgba[:, :, 3] = 255

    # Set RGB channels for mask pixels
    mask_pixels = mask > 0
    rgba[mask_pixels, 0] = 255  # Red
    rgba[mask_pixels, 1] = 0  # Green
    rgba[mask_pixels, 2] = 0  # Blue
    rgba[mask_pixels, 3] = int(255 * alpha)  # Alpha

    return rgba


def get_mask_statistics(masks: List[np.ndarray]) -> Dict:
    """
    Get statistics about a list of masks.

    Args:
        masks: List of mask arrays

    Returns:
        Dict: Statistics including count, total pixels, average size, etc.
    """
    if not masks:
        return {"count": 0, "total_pixels": 0, "average_size": 0}

    total_pixels = sum(mask.sum() for mask in masks)
    sizes = [mask.sum() for mask in masks]

    return {
        "count": len(masks),
        "total_pixels": int(total_pixels),
        "average_size": float(np.mean(sizes)),
        "min_size": int(np.min(sizes)),
        "max_size": int(np.max(sizes)),
        "shapes": [list(mask.shape) for mask in masks],
    }


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the decoder
    print("Mask Decoder Utility")
    print("===================")
    print("This utility provides functions to decode compressed masks from the YOLOv9c segmentation server.")
    print("\nAvailable functions:")
    print("- decode_mask(mask_data): Universal decoder for any format")
    print("- decode_binary_compressed_mask(mask_data): For binary compressed format")
    print("- decode_rle_mask(mask_data): For Run-Length Encoded format")
    print("- decode_raw_mask(mask_data): For raw format (no compression)")
    print("- decode_segmentation_response(response): Decode all masks from server response")
    print("- visualize_mask(mask): Create RGBA visualization of a mask")
    print("- get_mask_statistics(masks): Get statistics about mask list")
