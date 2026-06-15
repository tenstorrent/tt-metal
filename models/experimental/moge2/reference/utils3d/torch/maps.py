from typing import *
from typing_extensions import Unpack
import math
from numbers import Number

import torch
from torch import Tensor
import torch.nn.functional as F

from .mesh import remove_unused_vertices, triangulate_mesh
from .transforms import unproject_cv
from .utils import masked_max, masked_min, sliding_window
from .helpers import batched

__all__ = [
    'uv_map',
    'pixel_coord_map',
    'screen_coord_map',
    'build_mesh_from_map',
    'build_mesh_from_depth_map',
    'depth_map_edge',
    'depth_map_aliasing',
    'point_map_to_normal_map',
    'depth_map_to_point_map',
    'depth_map_to_normal_map',
    'chessboard',
    'bounding_rect_from_mask',
    'masked_nearest_resize',
    'masked_area_resize', 
    'flood_fill',
    'perlin_noise',
    'perlin_noise_map',
    'fractal_perlin_noise_map'
]


def uv_map(
    *size: Union[int, Tuple[int, int]],
    top: float = 0.,
    left: float = 0.,
    bottom: float = 1.,
    right: float = 1.,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None
) -> Tensor:
    """
    Get image UV coordinate map. By default, (0., 0.) is the top-left corner of the image, and (1., 1.) is the bottom-right corner of the image.

    ## Parameters
    - `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
    - `top`: `float` defaults to 0.
    - `left`: `float` defaults to 0.
    - `bottom`: `float` defaults to 1.
    - `right`: `float` defaults to 1.
    - `dtype`: `torch.dtype` data type of the output uv map. Defaults to torch.float32.
    - `device`: `torch.device`, device of the output uv map. Defaults to None.

    ## Returns
    - `uv (Tensor)`: shape `(height, width, 2)`

    ## Example Usage

    >>> uv_map(10, 10):
    [[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
     [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
      ...             ...                  ...
     [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]
    """
    if len(size) == 1 and isinstance(size[0], tuple):
        height, width = size[0]
    else:
        height, width = size
    u = torch.linspace(left + 0.5 / width * (right - left), right - 0.5 / width * (right - left), width, dtype=dtype, device=device)
    v = torch.linspace(top + 0.5 / height * (bottom - top), bottom - 0.5 / height * (bottom - top), height, dtype=dtype, device=device)
    uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1)
    return uv


def pixel_coord_map(
    *size: Union[int, Tuple[int, int]],
    top: int = 0,
    left: int = 0,
    convention: Literal['integer-center', 'integer-corner'] = 'integer-center',
    dtype: torch.dtype = torch.float32,
    device: torch.device = None
) -> Tensor:
    """
    Get image pixel coordinates map. Support two conventions: `'integer-center'` and `'integer-corner'`.

    ## Parameters
    - `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
    - `top`: `int`, optional top boundary of the pixel coord map. Defaults to 0.
    - `left`: `int`, optional left boundary of the pixel coord map. Defaults to 0.
    - `convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
        - `'integer-center'`: `pixel[i][j]` has integer coordinates `(j, i)` as its center, and occupies square area `[j - 0.5, j + 0.5) × [i - 0.5, i + 0.5)`. 
            The top-left corner of the top-left pixel is `(-0.5, -0.5)`, and the bottom-right corner of the bottom-right pixel is `(width - 0.5, height - 0.5)`.
        - `'integer-corner'`: `pixel[i][j]` has coordinates `(j + 0.5, i + 0.5)` as its center, and occupies square area `[j, j + 1) × [i, i + 1)`.
            The top-left corner of the top-left pixel is `(0, 0)`, and the bottom-right corner of the bottom-right pixel is `(width, height)`.
    - `dtype`: `torch.dtype`, optional data type of the output pixel coord map. Defaults to torch.float32.

    ## Returns
        Tensor: shape (height, width, 2)
    
    >>> pixel_coord_map(10, 10, convention='integer-center', dtype=torch.long):
    [[[0, 0], [1, 0], ..., [9, 0]],
     [[0, 1], [1, 1], ..., [9, 1]],
        ...      ...         ...
     [[0, 9], [1, 9], ..., [9, 9]]]

    >>> pixel_coord_map(10, 10, convention='integer-corner', dtype=torch.float32):
    [[[0.5, 0.5], [1.5, 0.5], ..., [9.5, 0.5]],
     [[0.5, 1.5], [1.5, 1.5], ..., [9.5, 1.5]],
      ...             ...                  ...
    [[0.5, 9.5], [1.5, 9.5], ..., [9.5, 9.5]]]
    """
    if len(size) == 1 and isinstance(size[0], tuple):
        height, width = size[0]
    else:
        height, width = size
    u = torch.arange(left, left + width, dtype=dtype, device=device)
    v = torch.arange(top, top + height, dtype=dtype, device=device)
    if convention == 'integer-corner':
        assert torch.is_floating_point(u), "dtype should be a floating point type when convention is 'integer-corner'"
        u = u + 0.5
        v = v + 0.5
    u, v = torch.meshgrid(u, v, indexing='xy')
    return torch.stack([u, v], dim=2)


def screen_coord_map(
    *size: Union[int, Tuple[int, int]],
    top: float = 1.,
    left: float = 0.,
    bottom: float = 0.,
    right: float = 1.,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None
) -> Tensor:
    """
    Get screen space coordinate map, where (0., 0.) is the bottom-left corner of the image, and (1., 1.) is the top-right corner of the image.
    This is commonly used in graphics APIs like OpenGL.

    ## Parameters
    - `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
    - `top`: `float`, optional top boundary in the screen space. Defaults to 1.
    - `left`: `float`, optional left boundary in the screen space. Defaults to 0.
    - `bottom`: `float`, optional bottom boundary in the screen space. Defaults to 0.
    - `right`: `float`, optional right boundary in the screen space. Defaults to 1.
    - `dtype`: `torch.dtype`, optional data type of the output map. Defaults to torch.float32.

    ## Returns
        (Tensor): shape (height, width, 2)
    """
    if len(size) == 1 and isinstance(size[0], tuple):
        height, width = size[0]
    else:
        height, width = size
    x = torch.linspace(left + 0.5 / width, right - 0.5 / width, width, dtype=dtype, device=device)
    y = torch.linspace(top - 0.5 / height, bottom - 0.5 / height, height, dtype=dtype, device=device)
    x, y = torch.meshgrid(x, y, indexing='xy')
    return torch.stack([x, y], dim=2)


def build_mesh_from_map(
    *maps: Tensor,
    mask: Optional[Tensor] = None,
    tri: bool = False,
) -> Tuple[Tensor, ...]:
    """
    Get a mesh regarding image pixel uv coordinates as vertices and image grid as faces.

    ## Parameters
        *maps (Tensor): attribute maps in shape (height, width, [channels])
        mask (Tensor, optional): binary mask of shape (height, width), dtype=bool. Defaults to None.

    ## Returns
        faces (Tensor): faces connecting neighboring pixels. shape (T, 4) if tri is False, else (T, 3)
        *attributes (Tensor): vertex attributes in corresponding order with itorchut maps
        indices (Tensor, optional): indices of vertices in the original mesh
    """
    assert (len(maps) > 0) or (mask is not None), "At least one of maps or mask should be provided"
    height, width = maps[0].shape[:2] if mask is None else mask.shape
    device = maps[0].device if len(maps) > 0 else mask.device
    assert all(x.shape[:2] == (height, width) for x in maps), "All maps should have the same shape"

    row_faces = torch.stack([
        torch.arange(0, width - 1, dtype=torch.int32, device=device), 
        torch.arange(width, 2 * width - 1, dtype=torch.int32, device=device), 
        torch.arange(1 + width, 2 * width, dtype=torch.int32, device=device), 
        torch.arange(1, width, dtype=torch.int32, device=device)], 
    dim=1)
    faces = (torch.arange(0, (height - 1) * width, width, dtype=torch.int32, device=device)[:, None, None] + row_faces[None, :, :]).reshape((-1, 4))
    attributes = tuple(x.reshape(-1, *x.shape[2:]) for x in maps)
    if mask is not None:
        quad_mask = (mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]).ravel()
        faces = faces[quad_mask]
        faces, *attributes = remove_unused_vertices(faces, *attributes)
    if tri:
        faces = triangulate_mesh(faces)
    return faces, *attributes


def build_mesh_from_depth_map(
    depth: Tensor,
    *other_maps: Tensor,
    intrinsics: Tensor,
    extrinsics: Optional[Tensor] = None,
    atol: Optional[float] = None,
    rtol: Optional[float] = 0.05,
    tri: bool = False,
) -> Tuple[Tensor, ...]:
    """
    Get a mesh by lifting depth map to 3D, while removing depths of large depth difference.

    ## Parameters
        depth (Tensor): [H, W] depth map
        extrinsics (Tensor, optional): [4, 4] extrinsics matrix. Defaults to None.
        intrinsics (Tensor, optional): [3, 3] intrinsics matrix. Defaults to None.
        *other_maps (Tensor): [H, W, C] vertex attributes. Defaults to None.
        atol (float, optional): absolute tolerance. Defaults to None.
        rtol (float, optional): relative tolerance. Defaults to None.
            triangles with vertices having depth difference larger than atol + rtol * depth will be marked.
        remove_by_depth (bool, optional): whether to remove triangles with large depth difference. Defaults to True.
        return_uv (bool, optional): whether to return uv coordinates. Defaults to False.
        return_indices (bool, optional): whether to return indices of vertices in the original mesh. Defaults to False.

    ## Returns
        faces (Tensor): [T, 3] faces
        vertices (Tensor): [N, 3] vertices
        *other_attrs (Tensor): [N, C] vertex attributes
    """
    height, width = depth.shape
    uv = uv_map(height, width, dtype=depth.dtype, device=depth.device)
    mask = torch.isfinite(depth)
    if atol is not None or rtol is not None:
        mask = mask & ~depth_map_edge(depth, atol=atol, rtol=rtol, kernel_size=3, mask=mask)
    uv, depth, other_attrs, faces = build_mesh_from_map(uv, depth, *other_maps, mask=mask)
    pts = unproject_cv(uv, depth, intrinsics, extrinsics)
    if tri:
        faces = triangulate_mesh(faces, vertices=pts, method='diagonal')
        faces, pts, *other_attrs = remove_unused_vertices(faces, pts, *other_attrs)
    return faces, pts, *other_attrs


@batched(2)
def depth_map_edge(depth: Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: Tensor = None) -> torch.BoolTensor:
    """
    Compute the edge mask of a depth map. The edge is defined as the pixels whose neighbors have a large difference in depth.
    
    ## Parameters
        depth (Tensor): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    ## Returns
        edge (Tensor): shape (..., height, width) of dtype torch.bool
    """
    if mask is None:
        diff = (F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2))
    else:
        diff = (F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(torch.where(mask, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2))

    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    return edge


@batched(2)
def depth_map_aliasing(depth: Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: Tensor = None) -> torch.BoolTensor:
    """
    Compute the map that indicates the aliasing of a depth map. The aliasing is defined as the pixels which neither close to the maximum nor the minimum of its neighbors.
    ## Parameters
        depth (Tensor): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    ## Returns
        edge (Tensor): shape (..., height, width) of dtype torch.bool
    """
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])

    if mask is None:
        diff_max = F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2) - depth
        diff_min = F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2) + depth
    else:
        diff_max = F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) - depth
        diff_min = F.max_pool2d(torch.where(mask, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) + depth
    diff = torch.minimum(diff_max, diff_min)

    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    edge = edge.reshape(*shape)
    return edge


@batched(3, 2)
def point_map_to_normal_map(point: Tensor, mask: Tensor = None) -> Tensor:
    """
    Calculate normal map from point map. Value range is [-1, 1].

    ## Parameters
        point (Tensor): shape (..., height, width, 3), point map
        mask (Tensor): shape (..., height, width), binary mask. Defaults to None.

    ## Returns
        normal (Tensor): shape (..., height, width, 3), normal map. 
    """
    has_mask = mask is not None

    if mask is None:
        mask = torch.ones_like(point[..., 0], dtype=torch.bool)
    mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)

    pts = F.pad(point.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='constant', value=1).permute(0, 2, 3, 1)
    up = pts[:, :-2, 1:-1, :] - pts[:, 1:-1, 1:-1, :]
    left = pts[:, 1:-1, :-2, :] - pts[:, 1:-1, 1:-1, :]
    down = pts[:, 2:, 1:-1, :] - pts[:, 1:-1, 1:-1, :]
    right = pts[:, 1:-1, 2:, :] - pts[:, 1:-1, 1:-1, :]
    normal = torch.stack([
        torch.cross(up, left, dim=-1),
        torch.cross(left, down, dim=-1),
        torch.cross(down, right, dim=-1),
        torch.cross(right, up, dim=-1),
    ])
    normal = F.normalize(normal, dim=-1)
    valid = torch.stack([
        mask[:, :-2, 1:-1] & mask[:, 1:-1, :-2],
        mask[:, 1:-1, :-2] & mask[:, 2:, 1:-1],
        mask[:, 2:, 1:-1] & mask[:, 1:-1, 2:],
        mask[:, 1:-1, 2:] & mask[:, :-2, 1:-1],
    ]) & mask[None, :, 1:-1, 1:-1]
    normal = (normal * valid[..., None]).sum(dim=0)
    normal = F.normalize(normal, dim=-1)
    
    if has_mask:
        return normal, valid.any(dim=0)
    else:
        return normal


@batched(2, 2)
def depth_map_to_normal_map(depth: Tensor, intrinsics: Tensor, mask: Tensor = None) -> Tensor:
    """
    Calculate normal map from depth map. Value range is [-1, 1]. Normal direction in OpenCV identity camera's coordinate system.

    ## Parameters
        depth (Tensor): shape (..., height, width), linear depth map
        intrinsics (Tensor): shape (..., 3, 3), intrinsics matrix
    ## Returns
        normal (Tensor): shape (..., height, width, 3), normal map. 
    """
    pts = depth_map_to_point_map(depth, intrinsics)
    return point_map_to_normal_map(pts, mask)


def depth_map_to_point_map(depth: Tensor, intrinsics: Tensor, extrinsics: Tensor = None):
    height, width = depth.shape[-2:]
    uv = uv_map(height, width, dtype=depth.dtype, device=depth.device)
    pts = unproject_cv(uv, depth, intrinsics=intrinsics[..., None, :, :], extrinsics=extrinsics[..., None, :, :] if extrinsics is not None else None)
    return pts


def bounding_rect_from_mask(mask: torch.BoolTensor):
    """Get bounding rectangle of a mask

    ## Parameters
        mask (Tensor): shape (..., height, width), mask

    ## Returns
        rect (Tensor): shape (..., 4), bounding rectangle (left, top, right, bottom)
    """
    height, width = mask.shape[-2:]
    mask = mask.flatten(-2).unsqueeze(-1)
    uv = uv_map(height, width).to(mask.device).reshape(-1, 2)
    left_top = masked_min(uv, mask, dim=-2)[0]
    right_bottom = masked_max(uv, mask, dim=-2)[0]
    return torch.cat([left_top, right_bottom], dim=-1)


def chessboard(*size: Union[int, Tuple[int, int]], grid_size: int, color_a: Tensor, color_b: Tensor) -> Tensor:
    """Get a chessboard image

    ## Parameters
    - `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
    - `grid_size`: `int`, size of chessboard grid
    - `color_a`: `Tensor`, shape (channels,), color of the grid at the top-left corner
    - `color_b`: `Tensor`, shape (channels,), color in complementary grids

    ## Returns
    - `image` (Tensor): shape (height, width, channels), chessboard image
    """
    if len(size) == 1 and isinstance(size[0], tuple):
        height, width = size[0]
    else:
        height, width = size
    x = torch.div(torch.arange(width), grid_size, rounding_mode='floor')
    y = torch.div(torch.arange(height), grid_size, rounding_mode='floor')
    mask = ((x[None, :] + y[:, None]) % 2).to(color_a)
    image = (1 - mask[..., None]) * color_a + mask[..., None] * color_b
    return image


def masked_nearest_resize(
    *image: Tensor,
    mask: Tensor, 
    size: Tuple[int, int], 
    return_index: bool = False
) -> Tuple[Unpack[Tuple[Tensor, ...]], Tensor, Tuple[Tensor, ...]]:
    """
    Resize image(s) by nearest sampling with mask awareness. Suitable for sparse maps. ![masked_nearest_resize.png](doc/masked_nearest_resize.png)
    - Downsampling: Assign the nearest valid pixel within the target pixel's receptive field.
    - Upsampling: Assign the valid pixel to only the nearest pixel in the resized map.


    ### Parameters
    - `*image`: Itorchut image(s) of shape `(..., H, W, C)` or `(... , H, W)` 
        - You can pass multiple images to be resized at the same time for efficiency.
    - `mask`: itorchut mask of shape `(..., H, W)`, dtype=bool
    - `size`: target size `(H', W')`
    - `return_index`: whether to return the nearest neighbor indices in the original map for each pixel in the resized map.
        Defaults to False.

    ### Returns
    - `*resized_image`: resized image(s) of shape `(..., H', W', C)`. or `(..., H', W')`
    - `resized_mask`: mask of the resized map of shape `(..., H', W')`
    - `nearest_indices`: tuple of shape `(..., H', W')`. The nearest neighbor indices of the resized map of each dimension.
    """
    device = mask.device
    height, width = mask.shape[-2:]
    target_height, target_width = size
    filter_h_f, filter_w_f = height / target_height, width / target_width
    filter_h_i, filter_w_i = math.ceil(filter_h_f), math.ceil(filter_w_f)
    filter_size = filter_h_i * filter_w_i
    filter_shape = (filter_h_i, filter_w_i)
    padding_h, padding_w = filter_h_i // 2 + 1, filter_w_i // 2 + 1
    padding_shape = ((padding_h, padding_h), (padding_w, padding_w))
    
    # Window the original mask and uv
    pixels = pixel_coord_map(height, width, convention='integer-corner', dtype=torch.float32, device=device)
    indices = torch.arange(height * width, dtype=torch.long, device=device).reshape(height, width)
    window_pixels = sliding_window(pixels, window_size=filter_shape, pad_size=padding_shape, dim=(0, 1))
    window_indices = sliding_window(indices, window_size=filter_shape, pad_size=padding_shape, dim=(0, 1))
    window_mask = sliding_window(mask, window_size=filter_shape, pad_size=padding_shape, dim=(-2, -1))

    # Gather the target pixels's local window
    target_centers = uv_map(target_height, target_width, dtype=torch.float32, device=device) * torch.tensor([width, height], dtype=torch.float32, device=device)
    target_lefttop = target_centers - torch.tensor((filter_w_f / 2, filter_h_f / 2), dtype=torch.float32, device=device)
    target_window = torch.round(target_lefttop).to(torch.long) + torch.tensor((padding_w, padding_h), dtype=torch.long, device=device)

    target_window_pixels = window_pixels[target_window[..., 1], target_window[..., 0], :, :, :].reshape(target_height, target_width, 2, filter_size)                  # (target_height, tgt_width, 2, filter_size)
    target_window_mask = window_mask[..., target_window[..., 1], target_window[..., 0], :, :].reshape(*mask.shape[:-2], target_height, target_width, filter_size)     # (..., target_height, tgt_width, filter_size)
    target_window_indices = window_indices[target_window[..., 1], target_window[..., 0], :, :].reshape(target_height, target_width, filter_size)                      # (target_height, tgt_width, filter_size)

    # Compute nearest neighbor in the local window for each pixel 
    delta = target_window_pixels - target_centers[..., None]
    eps = torch.finfo(torch.float32).eps * max(height, width) # Shift a small epsilon to avoid numerical issues when the pixel is exactly on the border
    target_window_mask &= (-filter_w_f / 2 + eps < delta[..., 0, :]) & (delta[..., 0, :] <= filter_w_f / 2 + eps) & (-filter_h_f / 2 + eps < delta[..., 1, :]) & (delta[..., 1, :] <= filter_h_f / 2 + eps)
    dist = torch.where(target_window_mask, torch.square(delta[..., 0, :]) + torch.square(delta[..., 1, :]), torch.inf)      # (..., target_height, tgt_width, filter_size)
    nearest_in_window = torch.argmin(dist, dim=-1, keepdim=True)                                                 # (..., target_height, tgt_width, 1)
    nearest_idx = torch.gather(
        target_window_indices.expand(dist.shape),
        dim=-1,
        index=nearest_in_window,
    ).squeeze(-1)     # (..., target_height, tgt_width)
    nearest_i, nearest_j = nearest_idx // width, nearest_idx % width
    target_mask = torch.any(target_window_mask, dim=-1)
    batch_indices = [torch.arange(n, device=device).reshape([1] * i + [n] + [1] * (mask.ndim - i - 1)) for i, n in enumerate(mask.shape[:-2])]

    nearest_indices = (*batch_indices, nearest_i, nearest_j)
    outputs = tuple(x[nearest_indices] for x in image)

    if return_index:
        ret = (*outputs, target_mask, nearest_indices)
    else:
        ret = (*outputs, target_mask)

    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def masked_area_resize(
    *image: Tensor,
    mask: Tensor, 
    size: Tuple[int, int]
) -> Tuple[Unpack[Tuple[Tensor, ...]], Tensor]:
    """
    Resize 2D map by area sampling with mask awareness.

    ### Parameters
    - `*image`: Itorchut image(s) of shape `(..., H, W, C)` or `(..., H, W)`
        - You can pass multiple images to be resized at the same time for efficiency.
    - `mask`: Itorchut mask of shape `(..., H, W)`
    - `size`: target image size `(H', W')`

    ### Returns
    - `*resized_image`: resized image(s) of shape `(..., H', W', C)`. or `(..., H', W')`
    - `resized_mask`: mask of the resized map of shape `(..., H', W')`
    """
    device = mask.device
    height, width = mask.shape[-2:]
    target_height, target_width = size

    filter_h_f, filter_w_f = max(1, height / target_height), max(1, width / target_width)
    filter_h_i, filter_w_i = math.ceil(filter_h_f), math.ceil(filter_w_f)
    filter_size = filter_h_i * filter_w_i
    filter_shape = (filter_h_i, filter_w_i)
    padding_h, padding_w = filter_h_i // 2 + 1, filter_w_i // 2 + 1
    padding_shape = ((padding_h, padding_h), (padding_w, padding_w))
    
    # Window the original mask and uv (non-copy)
    pixels = pixel_coord_map((height, width), convention='integer-corner', dtype=torch.float32, device=device)
    indices = torch.arange(height * width, dtype=torch.long, device=device).reshape(height, width)
    window_pixels = sliding_window(pixels, window_size=filter_shape, pad_size=padding_shape, dim=(0, 1))
    window_indices = sliding_window(indices, window_size=filter_shape, pad_size=padding_shape, dim=(0, 1))
    window_mask = sliding_window(mask, window_size=filter_shape, pad_size=padding_shape, dim=(-2, -1))

    # Gather the target pixels's local window
    target_center = uv_map((target_height, target_width), dtype=torch.float32, device=device) * torch.tensor([width, height], dtype=torch.float32, device=device)
    target_lefttop = target_center - torch.tensor((filter_w_f / 2, filter_h_f / 2), dtype=torch.float32, device=device)
    target_bottomright = target_center + torch.tensor((filter_w_f / 2, filter_h_f / 2), dtype=torch.float32, device=device)
    target_window = torch.floor(target_lefttop).to(torch.long) + torch.tensor((padding_w, padding_h), dtype=torch.long, device=device)

    target_window_centers = window_pixels[target_window[..., 1], target_window[..., 0], :, :, :].reshape(target_height, target_width, 2, filter_size)                 # (target_height, tgt_width, 2, filter_size)
    target_window_mask = window_mask[..., target_window[..., 1], target_window[..., 0], :, :].reshape(*mask.shape[:-2], target_height, target_width, filter_size)     # (..., target_height, tgt_width, filter_size)
    target_window_indices = window_indices[target_window[..., 1], target_window[..., 0], :, :].reshape(target_height, target_width, filter_size)                      # (target_height, tgt_width, filter_size)

    # Compute pixel area in the local windows
    # (..., target_height, tgt_width, filter_size)
    target_window_lefttop = torch.maximum(target_window_centers - 0.5, target_lefttop[..., None])
    target_window_rightbottom = torch.minimum(target_window_centers + 0.5, target_bottomright[..., None])
    target_window_area = (target_window_rightbottom - target_window_lefttop).clip(0, None)
    target_window_area = torch.where(target_window_mask, target_window_area[..., 0, :] * target_window_area[..., 1, :], 0)

    target_area = torch.sum(target_window_area, dim=-1)   # (..., target_height, tgt_width)
    target_mask = target_area >= 0

    # Weighted sum by area
    outputs = [] 
    for x in image:
        assert x.shape[:mask.ndim] == mask.shape, f"Image and mask should have the same batch shape and spatial shape. Got {x.shape[:mask.ndim]} vs {mask.shape}"
        expand_channels = (slice(None),) * (x.ndim - mask.ndim)
        x = torch.where(mask[(..., *expand_channels)], x, 0)
        x = x.reshape(*x.shape[:mask.ndim - 2], height * width, *x.shape[mask.ndim:])[(*((slice(None),) * (mask.ndim - 2)), target_window_indices)]                   # (..., target_height, tgt_width, filter_size, ...)
        x = (x * target_window_area[(..., *expand_channels)]).sum(dim=mask.ndim) / torch.maximum(target_area[(..., *expand_channels)], torch.tensor(torch.finfo(torch.float32).eps, device=device))  # (..., target_height, tgt_width, ...)
        outputs.append(x)

    ret = *outputs, target_mask

    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def flood_fill(*image: Tensor, mask: Tensor, return_index: bool = False) -> Tensor:
    """
    Flooding fill the holes in the image(s) according to the mask. ![img](doc/flood_fill.png)

    Parameters
    ----
    - `*image` (Tensor): shape (..., height, width, [C]), itorchut image(s)
    - `mask` (Tensor): shape (..., height, width), binary mask indicating valid regions

    Returns
    ----
    - `*filled_image` (Tensor): shape (..., height, width, [C]), flood filled map
    - `filled_indices` (Tuple[Tensor, ...], optional): tuple of shape (..., height, width). The nearest neighbor indices of each pixel in the original map.
        It satisfies `filled_image = image[filled_indices]`
    """
    device = mask.device
    *batch_shape, height, width = mask.shape

    self_i, self_j = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing='ij'
    )
    nearest_i, nearest_j = self_i.expand_as(mask), self_j.expand_as(mask)

    filled_mask = mask.clone()

    step = (max(height, width) + 1) // 2
    steps = []
    while step > 1:
        steps.append(step)
        step = (step + 1) // 2
    steps += [1, 1]
    for step in steps:
        sliding_window_i = sliding_window(nearest_i, window_size=3, dilation=step, pad_size=step, dim=(-2, -1)).reshape(*batch_shape, height, width, 9)
        sliding_window_j = sliding_window(nearest_j, window_size=3, dilation=step, pad_size=step, dim=(-2, -1)).reshape(*batch_shape, height, width, 9)
        sliding_window_mask = sliding_window(filled_mask, window_size=3, dilation=step, pad_size=step, dim=(-2, -1)).reshape(*batch_shape, height, width, 9)
        filled_mask |= sliding_window_mask.any(dim=-1)
        dist = torch.where(sliding_window_mask, torch.square(sliding_window_i - self_i[..., None]) + torch.square(sliding_window_j - self_j[..., None]), torch.inf)
        nearest_in_window = torch.argmin(dist, dim=-1)
        nearest_i = torch.gather(sliding_window_i, index=nearest_in_window[..., None], dim=-1)[..., 0]
        nearest_j = torch.gather(sliding_window_j, index=nearest_in_window[..., None], dim=-1)[..., 0]

    batch_indices = [torch.arange(n, device=device).reshape([1] * i + [n] + [1] * (mask.ndim - i - 1)) for i, n in enumerate(batch_shape)]
    nearest_i, nearest_j = nearest_i.round().to(torch.long), nearest_j.round().to(torch.long)
    nearest_indices = (*batch_indices, nearest_i, nearest_j)

    outputs = tuple(x[nearest_indices] for x in image)

    if return_index:
        ret = *outputs, nearest_indices
    else:
        ret = outputs
    
    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def perlin_noise(x: Tensor, seed: Optional[int] = None) -> Tensor:
    """Generate Perlin noise for the given coordinates.
    
    Parameters
    ----
    - `x` (Tensor): shape (*batch_shape, N_1, ..., N_D, D), coordinates to sample Perlin.
        If itorchut ndim is more than D + 1, the leading dimensions are treated as batch dimensions. Instances in the batch have different noise patterns.
    - `seed` (int, optional): random seed. The same seed will generate the same noise pattern(s). Defaults to None.

    Returns
    ----
    - `y` (Tensor): shape (*batch_shape, N_1, ..., N_D), Perlin noise value at the given coordinates and seed. Value range is approximately [-1, 1]
    """
    D = x.shape[-1]
    xi = x.to(int)
    xf = x - xi
    dtype, device = x.dtype, x.device

    # Prepare gradient set and hash permutation
    rng = torch.Generator(device=x.device)
    if seed is not None:
        rng.manual_seed(seed)
    hash_set_size = min(16 ** D, x.numel() // D)
    perm = torch.randperm(hash_set_size, generator=rng, device=device, dtype=torch.long)
    gradient_set = torch.randn((hash_set_size, D), generator=rng, device=device, dtype=dtype)
    gradient_set = gradient_set / (torch.linalg.norm(gradient_set, dim=-1, keepdim=True) + torch.finfo(dtype).tiny)

    # Get corner coordinates
    unit_grid = torch.stack(
        torch.meshgrid(*([torch.tensor([0, 1])] * D), indexing='ij'),
        dim=-1
    ).reshape(-1, D).to(device=device)
    x_corners = xi[..., None, :] + unit_grid

    # Hash index initialization. 
    if x.ndim > D + 1:
        # If batch dimensions exist, instances have different noise patterns
        hash_idx = torch.arange(math.prod(x.shape[:-1 - D]), device=device).reshape(*x.shape[:-1 - D], *([1] * (D + 1)))
        hash_idx = perm[hash_idx % hash_set_size]
    else:
        # No batch dimensions, start from 0
        hash_idx = 0
    
    # Hash corner coordinates
    for i in range(x.shape[-1]):
        hash_idx = perm[(hash_idx + x_corners[..., i]) % hash_set_size]

    # Compute dot products of each corner's gradient and the direction to the point
    gradients = gradient_set[hash_idx]
    direction = xf[..., None, :] - unit_grid.to(dtype)
    dot = torch.einsum('...i,...i->...', gradients, direction)

    # Smooth interpolation
    u = xf * xf * xf * (xf * (xf * 6 - 15) + 10)
    w = torch.prod(1 - torch.abs(u[..., None, :] - unit_grid.to(dtype)), dim=-1)
    y = torch.einsum('...i,...i->...', dot, w)

    return y


def perlin_noise_map(size: Tuple[int, ...], frequency: Union[float, Tensor], seed: Optional[int] = None, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> Tensor:
    """Generate Perlin noise map.

    Parameters
    ----
    - `size` (Tuple[int, ...]): size of the noise map (..., H, W)
    - `frequency` (float | Tensor): frequency relative to map's larger dimension max(H, W) of the Perlin noise. 
        Represents how many periods of noise fit in the larger dimension of the map.
    - `seed` (int, optional): random seed. The same seed will generate the same noise pattern(s). Defaults to None.

    Returns
    ----
    - `noise_map` (Tensor): shape (..., H, W), Perlin noise map. Value range is approximately [-1, 1]
    """
    height, width = size[-2:]
    if isinstance(frequency, torch.Tensor):
        dtype = frequency.dtype
        device = frequency.device
    else:
        if dtype is None:
            dtype = torch.float32    
        frequency = torch.tensor(frequency, dtype=dtype, device=device)
    uv = uv_map(size[-2:], dtype=dtype, device=device)
    frequency_uv = frequency[..., None] * (torch.tensor([width, height], dtype=dtype, device=device) / max(size[-2:]))
    x = torch.broadcast_to(uv * frequency_uv[..., None, None, :], (*size, 2))
    noise_map = perlin_noise(x, seed=seed)
    return noise_map


def fractal_perlin_noise_map(
    size: Tuple[int, ...], 
    base_frequency: Union[float, Tensor], 
    octaves: int = 4, 
    lacunarity: float = 2.0, 
    gain: float = 0.5, 
    seed: Optional[int] = None, 
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Generate fractal Perlin noise map. ![fractal_perlin_base_frequeny2_octaves7_gain0.7.png](doc/fractal_perlin_base_frequeny2_octaves7_gain0.7.png)

    Parameters
    ----
    - `size` (Tuple[int, ...]): size of the noise map (..., H, W)
    - `base_frequency` (float | Tensor): base frequency (relative to map's larger dimension max(H, W)) of the Perlin noise.
        Represents how many periods of noise fit in the larger dimension of the map.
    - `octaves` (int, optional): number of octaves. Defaults to 4.
    - `lacunarity` (float, optional): frequency multiplier between octaves. Defaults to 2.0.
    - `gain` (float, optional): amplitude multiplier between octaves. Defaults to 0.5.
    - `seed` (int, optional): random seed. The same seed will generate the same noise pattern. Defaults to None.

    Returns
    ----
    - `noise_map` (Tensor): shape (..., H, W), fractal Perlin noise map. Value range is approximately [-1, 1]
    """
    if isinstance(base_frequency, torch.Tensor):
        dtype = base_frequency.dtype
        device = base_frequency.device
    else:
        if dtype is None:
            dtype = torch.float32    
        base_frequency = torch.tensor(base_frequency, dtype=dtype, device=device)
    
    noise_map = torch.zeros(size, dtype=dtype, device=device)
    amplitude = 1.0
    frequency_uv = base_frequency.clone()[..., None] * (torch.tensor([size[-1], size[-2]], dtype=dtype, device=device) / max(size[-2:]))
    uv = uv_map(size[-2:], dtype=dtype, device=device)
    for octave in range(octaves):
        x = torch.broadcast_to(uv * frequency_uv[..., None, None, :], (*size, 2)) 
        noise_map += amplitude * perlin_noise(x, seed=seed + octave if seed is not None else None)
        amplitude *= gain
        frequency_uv *= lacunarity
    return noise_map