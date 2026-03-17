# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Plot utilities: tensor distribution histograms and heatmaps.
"""

import os
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import ttnn

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend: save only, no display (avoids figure accumulation)
    import matplotlib.pyplot as plt

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def plot_tensor_distribution(
    tensor,
    title=None,
    bins=100,
    save_path=None,
    ax=None,
    density=True,
    show_values_plot=False,
    max_values_points=20000,
    **hist_kwargs,
):
    """
    Plot the distribution of tensor elements for a torch or ttnn tensor.

    Args:
        tensor: A torch.Tensor or ttnn tensor.
        title: Optional title for the plot.
        bins: Number of histogram bins (default 100).
        save_path: If set, save the figure. If a directory, saves as "{title}.png" there;
            otherwise treated as the full file path.
        ax: Optional matplotlib axes to plot on (otherwise creates new figure).
        density: If True, normalize histogram to form a probability density (default True).
        show_values_plot: If True and ax is None, add a second subplot showing actual
            element values (index vs value) instead of only the density histogram.
        max_values_points: When show_values_plot is True, subsample to this many points
            for the values plot if the tensor is larger (default 20000).
        **hist_kwargs: Passed through to plt.hist (e.g. alpha, color, label).

    Returns:
        The matplotlib Axes used for the histogram, or the figure when show_values_plot
        created a second subplot (ax is the first axes).
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plot_tensor_distribution. Install with: pip install matplotlib")

    # Normalize to torch and then numpy
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().float().cpu().numpy()
    else:
        # Assume ttnn tensor
        data = ttnn.to_torch(tensor, dtype=torch.float32).detach().cpu().numpy()

    flat = data.flatten()
    effective_title = title or "Tensor element distribution"
    created_figure = False
    fig = None

    if ax is None and show_values_plot:
        fig, (ax, ax_values) = plt.subplots(1, 2, figsize=(12, 5))
        created_figure = True
    elif ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        created_figure = True

    ax.hist(flat, bins=bins, density=density, edgecolor="black", alpha=0.7, **hist_kwargs)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title(effective_title + " (distribution)")

    # Add summary stats as text
    stats = (
        f"min={flat.min():.4g}  max={flat.max():.4g}\n"
        f"mean={flat.mean():.4g}  std={flat.std():.4g}\n"
        f"n={flat.size}"
    )
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=9, verticalalignment="top", family="monospace")

    if ax is None and show_values_plot:
        # Second plot: actual values (index vs value)
        n = flat.size
        if n <= max_values_points:
            indices = np.arange(n)
            values = flat
        else:
            step = max(1, n // max_values_points)
            indices = np.arange(0, n, step)
            values = flat[indices]
        ax_values.plot(indices, values, linewidth=0.5, alpha=0.8)
        ax_values.set_xlabel("Element index")
        ax_values.set_ylabel("Value")
        ax_values.set_title(effective_title + " (values)")
        ax_values.grid(True, alpha=0.3)

    if save_path:
        # When save_path is a directory (or has no extension), filename is derived from the plot title
        if os.path.isdir(save_path) or not os.path.splitext(save_path)[1]:
            os.makedirs(save_path, exist_ok=True)
            safe_title = re.sub(r"[^\w\-]", "_", effective_title).strip("_") or "distribution"
            save_path = os.path.join(save_path, f"{safe_title}.png")
        ax.get_figure().savefig(save_path, bbox_inches="tight")
    if created_figure and fig is not None:
        plt.close(fig)
    return ax


def tensor_to_heatmap(
    tensor: Union[np.ndarray, torch.Tensor],
    output_path: Optional[Union[str, Path]] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    dpi: int = 150,
    figsize: Optional[tuple] = None,
) -> np.ndarray:
    """
    Convert a 2D tensor to a heatmap image.

    Args:
        tensor: 2D array (numpy or torch). Will be converted to numpy if needed.
        output_path: If provided, save the heatmap to this path.
        cmap: Matplotlib colormap name (e.g. 'viridis', 'plasma', 'coolwarm', 'RdBu').
        vmin, vmax: Value range for the colormap. If None, uses data min/max.
        dpi: DPI for saved image.
        figsize: (width, height) in inches. If None, scales with tensor shape.

    Returns:
        RGB image as numpy array of shape (H, W, 3), uint8 [0-255].
    """
    if not _HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for tensor_to_heatmap. Install with: pip install matplotlib")
    if hasattr(tensor, "detach"):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {arr.shape}")

    h, w = arr.shape
    if figsize is None:
        figsize = (max(8, w / 80), max(4, h / 80))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
