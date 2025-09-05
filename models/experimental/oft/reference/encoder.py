import torch
import torch.nn.functional as F


from collections import namedtuple

ObjectData = namedtuple("ObjectData", ["classname", "position", "dimensions", "angle", "score"])


def gaussian_kernel(sigma=1.0, trunc=2.0):
    width = round(trunc * sigma)
    x = torch.arange(-width, width + 1).float() / sigma
    kernel1d = torch.exp(-0.5 * x**2)
    kernel2d = kernel1d.view(1, -1) * kernel1d.view(-1, 1)

    return kernel2d / kernel2d.sum()


class ObjectEncoder(object):
    def __init__(
        self,
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

        self.sigma = sigma
        self.nms_thresh = nms_thresh

    def decode(self, heatmaps, pos_offsets, dim_offsets, ang_offsets, grid):
        # Apply NMS to find positive heatmap locations
        peaks, max_inds, scores, classids = self._decode_heatmaps(heatmaps)
        positions = self._decode_positions(pos_offsets, peaks, grid)
        dimensions = self._decode_dimensions(dim_offsets, peaks)
        angles = self._decode_angles(ang_offsets, peaks)
        return peaks, max_inds, scores, classids, positions, dimensions, angles
        # THIS SHOULD BE ADDED BACK
        objects = list()
        for score, cid, pos, dim, ang in zip(scores, classids, positions, dimensions, angles):
            objects.append(ObjectData(self.classnames[cid], pos, dim, ang, score))

        return objects, peaks

    def _decode_heatmaps(self, heatmaps):
        peaks, max_inds = non_maximum_suppression(heatmaps, self.sigma)
        scores = heatmaps[peaks]
        classids = torch.nonzero(peaks)[:, 0]
        return peaks, max_inds, scores, classids

    def _decode_positions(self, pos_offsets, peaks, grid):
        # Compute the center of each grid cell
        centers = (grid[1:, 1:] + grid[:-1, :-1]) / 2.0

        # Un-normalize grid offsets
        positions = pos_offsets.permute(0, 2, 3, 1) * self.pos_std.to(grid) + centers
        return positions[peaks]

    def _decode_dimensions(self, dim_offsets, peaks):
        dim_offsets = dim_offsets.permute(0, 2, 3, 1)
        dimensions = torch.exp(dim_offsets * self.log_dim_std.to(dim_offsets) + self.log_dim_mean.to(dim_offsets))
        return dimensions[peaks]

    def _decode_angles(self, angle_offsets, peaks):
        cos, sin = torch.unbind(angle_offsets, 1)
        return torch.atan2(sin, cos)[peaks]


def non_maximum_suppression(heatmaps, sigma=1.0, thresh=0.05, max_peaks=50):
    # Smooth with a Gaussian kernel
    num_class = heatmaps.size(0)
    kernel = gaussian_kernel(sigma)
    kernel = kernel.to(heatmaps)
    kernel = kernel.expand(num_class, num_class, -1, -1)
    smoothed = F.conv2d(heatmaps[None], kernel, padding=int((kernel.size(2) - 1) / 2))
    # Max pool over the heatmaps
    max_inds = F.max_pool2d(smoothed, 3, stride=1, padding=1, return_indices=True)[1].squeeze(0)

    _, height, width = heatmaps.size()
    flat_inds = torch.arange(height * width).type_as(max_inds).view(height, width)
    peaks = flat_inds == max_inds
    print(f"PEAKS {peaks.long().sum()}")
    peaks = peaks & (heatmaps > thresh)
    if peaks.long().sum() > max_peaks:
        scores = heatmaps[peaks]
        scores, _ = torch.sort(scores, descending=True)
        peaks = peaks & (heatmaps > scores[max_peaks - 1])

    print(f"TORCH: Final PEAKS value {peaks.long().sum()}")

    return peaks, max_inds
