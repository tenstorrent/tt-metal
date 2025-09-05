import torch
import ttnn

# from .. import utils

from collections import namedtuple

ObjectData = namedtuple("ObjectData", ["classname", "position", "dimensions", "angle", "score"])


def gaussian_kernel(sigma=1.0, trunc=2.0):
    width = round(trunc * sigma)
    x = torch.arange(-width, width + 1).float() / sigma
    kernel1d = torch.exp(-0.5 * x**2)
    kernel2d = kernel1d.view(1, -1) * kernel1d.view(-1, 1)

    return kernel2d / kernel2d.sum()


class TTObjectEncoder:
    def __init__(
        self,
        device,
        classnames=["Car"],
        pos_std=[0.5, 0.36, 0.5],
        log_dim_mean=[[0.42, 0.48, 1.35]],
        log_dim_std=[[0.085, 0.067, 0.115]],
        sigma=1.0,
        nms_thresh=0.05,
    ):
        self.classnames = classnames
        self.nclass = len(classnames)
        self.pos_std = torch.tensor(pos_std)
        self.log_dim_mean = torch.tensor(log_dim_mean)
        self.log_dim_std = torch.tensor(log_dim_std)

        self.pos_std_ttnn = ttnn.from_torch(
            self.pos_std,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=device,
        )
        self.log_dim_mean_ttnn = ttnn.from_torch(
            self.log_dim_mean,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=device,
        )
        self.log_dim_std_ttnn = ttnn.from_torch(
            self.log_dim_std,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=device,
        )

        self.sigma = sigma
        self.nms_thresh = nms_thresh

        # Create Gaussian kernel for NMS
        # moved to init phase to avoid recreating it every time
        self.kernel = gaussian_kernel(sigma)
        self.kernel = self.kernel.expand(self.nclass, self.nclass, -1, -1)
        self.ttnn_kernel = ttnn.from_torch(self.kernel, dtype=ttnn.bfloat16)
        print(f"ttnn {self.kernel.shape=}")

    def decode(self, device, heatmaps, pos_offsets, dim_offsets, ang_offsets, grid):
        positions = self._decode_positions(device, pos_offsets, grid)
        dimensions = self._decode_dimensions(device, dim_offsets)
        angles = self._decode_angles(device, ang_offsets)
        peaks_torch, max_inds, scores = self._decode_heatmaps(device, heatmaps)

        # fallback to torch
        classids = torch.nonzero(peaks_torch)[:, 0]

        scores_torch = ttnn.to_torch(scores, dtype=torch.float32)
        positions_torch = ttnn.to_torch(positions, dtype=torch.float32)
        dimensions_torch = ttnn.to_torch(dimensions, dtype=torch.float32)
        angles_torch = ttnn.to_torch(angles, dtype=torch.float32)
        # peaks = torch.load("peaks_torch.pt")

        return peaks_torch, max_inds, scores_torch, classids, positions_torch, dimensions_torch, angles_torch
        # THIS SHOULD BE ADDED BACK
        scores_torch = scores_torch[peaks_torch]
        positions_torch = positions_torch[peaks_torch]
        dimensions_torch = dimensions_torch[peaks_torch]
        angles_torch = angles_torch[peaks_torch]

        objects = list()
        for score, cid, pos, dim, ang in zip(scores_torch, classids, positions_torch, dimensions_torch, angles_torch):
            objects.append(ObjectData(self.classnames[cid], pos, dim, ang, score))

        return objects, peaks_torch

    def _decode_heatmaps(self, device, heatmaps):
        peaks, max_inds = ttnn_non_maximum_suppression_dbg(device, heatmaps, self.ttnn_kernel)
        scores = heatmaps
        # classids = torch.nonzero(peaks)[:, 0] #moved to level above
        return peaks, max_inds, scores  # , classids

    def _decode_positions(self, device, pos_offsets, grid):
        # Compute the center of each grid cell
        # perhaps could be moved to init block
        centers = grid[1:, 1:] + grid[:-1, :-1]
        centers = ttnn.div(centers, 2.0)
        # Un-normalize grid offsets
        pos_offsets = ttnn.permute(pos_offsets, (0, 2, 3, 1))
        positions = pos_offsets * self.pos_std_ttnn + centers
        return positions

    def _decode_dimensions(self, device, dim_offsets):
        dim_offsets = ttnn.permute(dim_offsets, (0, 2, 3, 1))
        coef = dim_offsets * self.log_dim_std_ttnn + self.log_dim_mean_ttnn
        dimensions = ttnn.exp(coef)
        return dimensions

    def _decode_angles(self, device, angle_offsets):
        cos = angle_offsets[:, 0, :, :]
        sin = angle_offsets[:, 1, :, :]
        atan2 = ttnn.atan2(sin, cos)
        return atan2


def ttnn_non_maximum_suppression_dbg(
    device, heatmaps, kernel, sigma=1.0, thresh=0.05, max_peaks=50, dtype=ttnn.bfloat16
):
    heatmaps_4d = ttnn.unsqueeze(heatmaps, 0)
    n, c, h, w = heatmaps_4d.shape
    heatmaps_4d = ttnn.permute(heatmaps_4d, (0, 2, 3, 1))

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_approx_mode=False,
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    conv_config = ttnn.Conv2dConfig(
        deallocate_activation=False,  # umesto deallocate_input
        reallocate_halo_output=True,
        in_place=False,  # umesto in_place_halo
        weights_dtype=ttnn.bfloat16,
    )
    smoothed, [out_h, out_w] = ttnn.conv2d(
        input_tensor=heatmaps_4d,
        weight_tensor=kernel,
        device=device,
        in_channels=c,
        out_channels=c,
        batch_size=1,
        input_height=h,
        input_width=w,
        kernel_size=[5, 5],
        stride=[1, 1],
        padding=[2, 2],
        dilation=[1, 1],
        groups=1,
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
    )

    smoothed = ttnn.to_memory_config(smoothed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    smoothed = ttnn.to_layout(smoothed, ttnn.ROW_MAJOR_LAYOUT)

    ttnn_output, indices = ttnn.max_pool2d(
        input_tensor=smoothed,
        batch_size=n,
        input_h=h,
        input_w=w,
        channels=c,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1],
        dilation=[1, 1],
        # applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ceil_mode=False,
        in_place_halo=False,
        deallocate_input=False,
        reallocate_halo_output=True,
        return_indices=True,
    )
    # fallback to torch to calculate max_peaks
    max_inds = ttnn.to_torch(indices, dtype=torch.int64).permute(0, 3, 1, 2).view(n, c, h, w)
    heatmaps_torch = ttnn.to_torch(heatmaps, dtype=torch.float32)
    max_inds = max_inds.squeeze(0)
    flat_inds = torch.arange(out_h * out_w).type_as(max_inds).view(out_h, out_w)
    peaks = flat_inds == max_inds
    print(f"PEAKS {peaks.long().sum()}")
    peaks = peaks & (heatmaps_torch > thresh)
    # Keep only the top N peaks
    if peaks.long().sum() > max_peaks:
        scores = heatmaps_torch[peaks]
        scores, _ = torch.sort(scores, descending=True)
        peaks = peaks & (heatmaps_torch > scores[max_peaks - 1])

    print(f"{peaks.shape=}, {peaks.dtype=}")
    print(f"TTNN: Final PEAKS value {peaks.long().sum()}")
    # max_inds return for max_pool2d validation
    return peaks, max_inds
