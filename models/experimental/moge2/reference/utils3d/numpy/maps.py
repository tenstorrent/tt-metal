
from typing import *
from typing_extensions import Unpack
import math
import numpy as np
from numpy import ndarray

from ..helpers import no_warnings, timeit
from .utils import max_pool_2d, sliding_window, pooling, lite_sum, lite_prod, lite_dot, lite_norm
from .transforms import angle_between, unproject_cv
from .mesh import triangulate_mesh, remove_unused_vertices

__all__ = [
    'uv_map',
    'pixel_coord_map',
    'screen_coord_map',
    'build_grid_mesh',
    'build_mesh_from_map',
    'build_mesh_from_depth_map',
    'depth_map_edge',
    'depth_map_aliasing',
    'normal_map_edge',
    'point_map_to_normal_map',
    'depth_map_to_point_map',
    'depth_map_to_normal_map',
    'chessboard',
    'masked_nearest_resize',
    'masked_area_resize',
    'colorize_depth_map',
    'colorize_normal_map',
    'colorize_segmentation_map',
    'colorize_probability_map',
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
    dtype: np.dtype = np.float32
) -> ndarray:
    """
    Get image UV space coordinate map, where (0., 0.) is the top-left corner of the image, and (1., 1.) is the bottom-right corner of the image.
    This is commonly used as normalized image coordinates in texture mapping (when image is not flipped vertically).

    ## Parameters
    - `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
    - `top`: `float`, optional top boundary in uv space. Defaults to 0.
    - `left`: `float`, optional left boundary in uv space. Defaults to 0.
    - `bottom`: `float`, optional bottom boundary in uv space. Defaults to 1.
    - `right`: `float`, optional right boundary in uv space. Defaults to 1.
    - `dtype`: `np.dtype`, optional data type of the output uv map. Defaults to np.float32.

    ## Returns
    - `uv (ndarray)`: shape `(height, width, 2)`

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
    u = np.linspace(left + 0.5 / width * (right - left), right - 0.5 / width * (right - left), width, dtype=dtype)
    v = np.linspace(top + 0.5 / height * (bottom - top), bottom - 0.5 / height * (bottom - top), height, dtype=dtype)
    u, v = np.meshgrid(u, v, indexing='xy')
    return np.stack([u, v], axis=-1)


def pixel_coord_map(
    *size: Union[int, Tuple[int, int]],
    top: int = 0,
    left: int = 0,
    convention: Literal['integer-center', 'integer-corner'] = 'integer-center',
    dtype: np.dtype = np.float32
) -> ndarray:
    """
    Get image pixel coordinates map, where (0, 0) is the top-left corner of the top-left pixel, and (width, height) is the bottom-right corner of the bottom-right pixel.

    ## Parameters
    - `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
    - `top`: `int`, optional top boundary of the pixel coord map. Defaults to 0.
    - `left`: `int`, optional left boundary of the pixel coord map. Defaults to 0.
    - `convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
        - `'integer-center'`: `pixel[i][j]` has integer coordinates `(j, i)` as its center, and occupies square area `[j - 0.5, j + 0.5) × [i - 0.5, i + 0.5)`. 
            The top-left corner of the top-left pixel is `(-0.5, -0.5)`, and the bottom-right corner of the bottom-right pixel is `(width - 0.5, height - 0.5)`.
        - `'integer-corner'`: `pixel[i][j]` has coordinates `(j + 0.5, i + 0.5)` as its center, and occupies square area `[j, j + 1) × [i, i + 1)`.
            The top-left corner of the top-left pixel is `(0, 0)`, and the bottom-right corner of the bottom-right pixel is `(width, height)`.
    - `dtype`: `np.dtype`, optional data type of the output pixel coord map. Defaults to np.float32.

    ## Returns
        ndarray: shape (height, width, 2)
    
    >>> pixel_coord_map(10, 10, convention='integer-center', dtype=int):
    [[[0, 0], [1, 0], ..., [9, 0]],
     [[0, 1], [1, 1], ..., [9, 1]],
        ...      ...         ...
     [[0, 9], [1, 9], ..., [9, 9]]]

    >>> pixel_coord_map(10, 10, convention='integer-corner', dtype=np.float32):
    [[[0.5, 0.5], [1.5, 0.5], ..., [9.5, 0.5]],
     [[0.5, 1.5], [1.5, 1.5], ..., [9.5, 1.5]],
      ...             ...                  ...
    [[0.5, 9.5], [1.5, 9.5], ..., [9.5, 9.5]]]
    """
    if len(size) == 1 and isinstance(size[0], tuple):
        height, width = size[0]
    else:
        height, width = size
    u = np.arange(left, left + width, dtype=dtype)
    v = np.arange(top, top + height, dtype=dtype)
    if convention == 'integer-corner':
        assert np.issubdtype(dtype, np.floating), "dtype should be a floating point type when convention is 'integer-corner'"
        u = u + 0.5
        v = v + 0.5
    u, v = np.meshgrid(u, v, indexing='xy')
    return np.stack([u, v], axis=2)


def screen_coord_map(
    *size: Union[int, Tuple[int, int]],
    top: float = 1.,
    left: float = 0.,
    bottom: float = 0.,
    right: float = 1.,
    dtype: np.dtype = np.float32
) -> ndarray:
    """
    Get screen space coordinate map, where (0., 0.) is the bottom-left corner of the image, and (1., 1.) is the top-right corner of the image.
    This is commonly used in graphics APIs like OpenGL.

    ## Parameters
        - `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
        - `top`: `float`, optional top boundary in the screen space. Defaults to 1.
        - `left`: `float`, optional left boundary in the screen space. Defaults to 0.
        - `bottom`: `float`, optional bottom boundary in the screen space. Defaults to 0.
        - `right`: `float`, optional right boundary in the screen space. Defaults to 1.
        - `dtype`: `np.dtype`, optional data type of the output map. Defaults to np.float32.

    ## Returns
        (ndarray): shape (height, width, 2)
    """
    if len(size) == 1 and isinstance(size[0], tuple):
        height, width = size[0]
    else:
        height, width = size
    x = np.linspace(left + 0.5 / width, right - 0.5 / width, width, dtype=dtype)
    y = np.linspace(top - 0.5 / height, bottom - 0.5 / height, height, dtype=dtype)
    x, y = np.meshgrid(x, y, indexing='xy')
    return np.stack([x, y], axis=2)


def build_grid_mesh(height: int, width: int, *, shared_vertices: bool = True) -> ndarray:
    """
    Get mesh of `height * width` faces arranged in a 2D grid.

    Parameters
    ----
        height (int): height of the grid
        width (int): width of the grid
        shared_vertices (bool, optional): whether the vertices are shared among faces. Defaults to True.

    Returns
    ----
        faces (ndarray): faces in shape (height, width, 4). 
        - If shared_vertices is `True`, the vertex indices are arranged from 0 to `(H + 1) * (W + 1) - 1` like
            ```
                0 ----------- 1 --- ...    -------- W-1 --------------- W
                |    (0,0)    |       (0,1)         |        (0,W-1)    |
                W+1 -------   W+2 --- ...  -------- 2*W --------------- 2*W+1
                |    (1,0)    |       (1,1)         |   ...             |
                                        ...
                |    (H-1,0)  |     (H-1,1)         |       (H-1,W-1)   |
                H*(W+1) ----- H*(W+1)+1 --- ... --- (H+1)*(W+1)-2 ----- (H+1)*(W+1)-1
            ```
        - If shared_vertices is `False`, each face has its own 4 vertices.
            The vertex indices are arranged from 0 to `H * W * 4 - 1`, like
            ```
                0 ------- 3  4 ------- 7  ...  (W-1)*4 ---- (W-1)*4+3
                |  (0,0)  |  |   (0,1)  |      |   (0,W-1)  |
                1 ------- 2  5 ------- 6  ...  (W-1)*4+1 -- (W-1)*4+2
                W*4 ----- W*4+3
                |  (1,0)  |     ....
                W*4+1 --- W*4+2
                                        ...
                (H-1)*W*4 ---- (H-1)*W*4+3  ...        H*W*4 - 4 -----  H*W*4 - 1
                |   (H-1,0)         |                  |    (H-1,W-1)   |
                (H-1)*W*4+1 -- (H-1)*W*4+2  ...        H*W*4 - 3 ------ H*W*4 - 2
            ```
    """
    if shared_vertices:
        single_row_faces = np.stack([
            np.arange(0, width, dtype=np.int32), 
            np.arange(width + 1, 2 * width + 1, dtype=np.int32), 
            np.arange(width + 2, 2 * width + 2, dtype=np.int32), 
            np.arange(1, width + 1, dtype=np.int32)
        ], axis=1)
        faces = (np.arange(0, height * (width + 1), width + 1, dtype=np.int32)[:, None, None] + single_row_faces[None, :, :])
    else:
        faces = np.arange(0, height * width * 4, dtype=np.int32).reshape((height, width, 4))
    return faces


def build_mesh_from_map(
    *maps: ndarray,
    mask: Optional[ndarray] = None,
    domain: Literal['vertex', 'face'] = 'vertex',
    tri: bool = False,
) -> Tuple[ndarray, ...]:
    """
    Get a mesh regarding image pixel uv coordinates as vertices and image grid as faces.

    ## Parameters
        *maps (ndarray): attribute maps in shape (height, width, [channels])
        mask (ndarray, optional): binary mask of shape (height, width), dtype=bool. Defaults to None.
        domain (Literal['vertex', 'face'], optional): whether the pixel attributes correspond to vertices or faces. Defaults to 'vertex'.
        tri (bool, optional): whether to triangulate the mesh. Defaults to False.

    ## Returns
        faces (ndarray): faces connecting neighboring pixels. shape (T, 4) if tri is False, else (T, 3)
        *attributes (ndarray): vertex or face attributes in corresponding order with input maps
    """
    assert (len(maps) > 0) or (mask is not None), "At least one of maps or mask should be provided"
    height, width = maps[0].shape[:2] if mask is None else mask.shape
    assert all(x.shape[:2] == (height, width) for x in maps), "All maps should have the same shape"

    if domain == 'vertex':
        faces = build_grid_mesh(height - 1, width - 1, shared_vertices=True).reshape(-1, 4)
        if mask is not None:
            quad_mask = (mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]).ravel()
            faces = faces[quad_mask]
        if tri:
            faces = triangulate_mesh(faces)
        vertex_attributes = tuple(x.reshape(-1, *x.shape[2:]) for x in maps)
        faces, *vertex_attributes = remove_unused_vertices(faces, *vertex_attributes)
        return faces, *vertex_attributes
    elif domain == 'face':
        faces = build_grid_mesh(height, width, shared_vertices=True).reshape(-1, 4)
        face_attributes = tuple(x.reshape(-1, *x.shape[2:]) for x in maps)
        if mask is not None:
            where_mask = np.where(mask.reshape(-1))
            faces = faces[where_mask]
            face_attributes = face_attributes[where_mask]
        if tri:
            faces, face_indices = triangulate_mesh(faces, return_face_indices=True)
            face_attributes = tuple(x[face_indices] for x in face_attributes)
        return faces, *face_attributes
    else:
        raise ValueError(f"Unknown domain type: {domain}. Must be 'vertex' or 'face'.")


def build_mesh_from_depth_map(
    depth: ndarray,
    *maps: ndarray,
    intrinsics: ndarray,
    extrinsics: Optional[ndarray] = None,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    domain: Literal['vertex', 'face'] = 'vertex',
    tri: bool = False,
) -> Tuple[ndarray, ...]:
    """
    Get a mesh by lifting depth map to 3D, while removing depths of large depth difference.

    ## Parameters
        depth (ndarray): [H, W] depth map
        extrinsics (ndarray, optional): [4, 4] extrinsics matrix. Defaults to None.
        intrinsics (ndarray, optional): [3, 3] intrinsics matrix. Defaults to None.
        *maps (ndarray): [H, W, C] vertex attributes. Defaults to None.
        atol (float, optional): absolute tolerance of difference. Defaults to None.
        rtol (float, optional): relative tolerance of difference. Defaults to None.
            triangles with vertices having depth difference larger than atol + rtol * depth will be marked.
        domain (Literal['vertex', 'face'], optional): whether the pixel attributes correspond to vertices or faces. Defaults to 'vertex'.
        tri (bool, optional): whether to triangulate the mesh. Defaults to False.

    ## Returns
        faces (ndarray): [T, 3] faces
        vertices (ndarray): [N, 3] vertices
        *attributes (ndarray): [N, C] vertex attributes or [T, C] face attributes
    """
    height, width = depth.shape
    
    mask = np.isfinite(depth)

    if atol is not None or rtol is not None:
        mask = mask & ~depth_map_edge(depth, atol=atol, rtol=rtol, kernel_size=3, mask=mask)

    if domain == 'vertex':
        uv = uv_map(height, width, dtype=depth.dtype)
        pts_map = unproject_cv(uv, depth, intrinsics, extrinsics)
        faces, vertices, *attributes = build_mesh_from_map(pts_map, *maps, mask=mask, domain='vertex', tri=False)
        faces, vertices, *attributes = remove_unused_vertices(faces, vertices, *attributes)

    elif domain == 'face':
        uv = np.stack(np.meshgrid(np.linspace(0, 1, width + 1), np.linspace(0, 1, height + 1), indexing='xy'), axis=-1)
        depth = pooling(np.where(mask, depth, np.nan), kernel_size=2, stride=1, padding=1, axis=(0, 1), mode='mean')
        vertices = unproject_cv(uv, depth, intrinsics, extrinsics).reshape(-1, 3)
        faces, *attributes = build_mesh_from_map(*maps, mask=mask, domain='face', tri=False)
        faces, vertices = remove_unused_vertices(faces, vertices)
        
    if tri:
        faces = triangulate_mesh(faces, vertices=vertices, method='diagonal')
    return faces, vertices, *attributes


@no_warnings(category=RuntimeWarning)
def depth_map_edge(
    depth: ndarray, 
    atol: Optional[float] = None, 
    rtol: Optional[float] = None, 
    ltol: Optional[float] = None,
    kernel_size: int = 3, 
    mask: ndarray = None
) -> ndarray:
    """
    Compute the edge mask from depth map. The edge is defined as the pixels whose neighbors have large difference in depth.
    
    ## Parameters
        depth (ndarray): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance
        ltol (float): relative tolerance of inverse depth laplacian

    ## Returns
        edge (ndarray): shape (..., height, width) of dtype torch.bool
    """
    if mask is not None:
        depth = np.where(mask, depth, np.nan)
    
    if atol is not None or rtol is not None:
        diff = (max_pool_2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + max_pool_2d(-depth, kernel_size, stride=1, padding=kernel_size // 2))
    
    edge = np.zeros_like(depth, dtype=bool)
    
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= diff / depth > rtol

    if ltol is not None:
        disp = 1 / depth
        disp_mean_pooling = pooling(disp, kernel_size, stride=1, padding=kernel_size // 2, axis=(-2, -1), mode='mean')
        laplacian = (disp - disp_mean_pooling) / disp
        laplacian_window_max = pooling(laplacian, kernel_size, stride=1, padding=kernel_size // 2, axis=(-2, -1), mode='max')
        laplacian_window_min = pooling(laplacian, kernel_size, stride=1, padding=kernel_size // 2, axis=(-2, -1), mode='min')
        edge |= (laplacian_window_max > ltol) & (laplacian_window_min < -ltol)

    return edge


@no_warnings(category=RuntimeWarning)
def depth_map_aliasing(depth: ndarray, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: ndarray = None) -> ndarray:
    """
    Compute the map that indicates the aliasing of x depth map, identifying pixels which neither close to the maximum nor the minimum of its neighbors.
    ## Parameters
        depth (ndarray): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    ## Returns
        edge (ndarray): shape (..., height, width) of dtype torch.bool
    """
    if mask is None:
        diff_max = max_pool_2d(depth, kernel_size, stride=1, padding=kernel_size // 2) - depth
        diff_min = max_pool_2d(-depth, kernel_size, stride=1, padding=kernel_size // 2) + depth
    else:
        diff_max = max_pool_2d(np.where(mask, depth, -np.inf), kernel_size, stride=1, padding=kernel_size // 2) - depth
        diff_min = max_pool_2d(np.where(mask, -depth, -np.inf), kernel_size, stride=1, padding=kernel_size // 2) + depth
    diff = np.minimum(diff_max, diff_min)

    edge = np.zeros_like(depth, dtype=bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= diff / depth > rtol
    return edge


@no_warnings(category=RuntimeWarning)
def normal_map_edge(normals: ndarray, tol: float, kernel_size: int = 3, mask: ndarray = None) -> ndarray:
    """
    Compute the edge mask from normal map.

    ## Parameters
        normal (ndarray): shape (..., height, width, 3), normal map
        tol (float): tolerance in degrees
   
    ## Returns
        edge (ndarray): shape (..., height, width) of dtype torch.bool
    """
    assert normals.ndim >= 3 and normals.shape[-1] == 3, "normal should be of shape (..., height, width, 3)"
    normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-12)
    
    padding = kernel_size // 2
    normals_window = sliding_window(
        np.pad(normals, (*([(0, 0)] * (normals.ndim - 3)), (padding, padding), (padding, padding), (0, 0)), mode='edge'), 
        window_size=kernel_size, 
        stride=1, 
        axis=(-3, -2)
    )
    if mask is None:
        angle_diff = np.arccos((normals[..., None, None] * normals_window).sum(axis=-3)).max(axis=(-2, -1))
    else:
        mask_window = sliding_window(
            np.pad(mask, (*([(0, 0)] * (mask.ndim - 3)), (padding, padding), (padding, padding)), mode='edge'), 
            window_size=kernel_size, 
            stride=1, 
            axis=(-3, -2)
        )
        angle_diff = np.where(mask_window, np.arccos((normals[..., None, None] * normals_window).sum(axis=-3)), 0).max(axis=(-2, -1))

    angle_diff = max_pool_2d(angle_diff, kernel_size, stride=1, padding=kernel_size // 2)
    edge = angle_diff > np.deg2rad(tol)
    return edge


@no_warnings(category=RuntimeWarning)
def point_map_to_normal_map(point: ndarray, mask: ndarray = None, edge_threshold: float = None) -> ndarray:
    """Calculate normal map from point map. Value range is [-1, 1]. 

    ## Parameters
        point (ndarray): shape (height, width, 3), point map
        mask (optional, ndarray): shape (height, width), dtype=bool. Mask of valid depth pixels. Defaults to None.
        edge_threshold (optional, float): threshold for the angle (in degrees) between the normal and the view direction. Defaults to None.

    ## Returns
        normal (ndarray): shape (height, width, 3), normal map. 
    """
    height, width = point.shape[-3:-1]
    has_mask = mask is not None

    if mask is None:
        mask = np.ones_like(point[..., 0], dtype=bool)
    mask_pad = np.zeros((height + 2, width + 2), dtype=bool)
    mask_pad[1:-1, 1:-1] = mask
    mask = mask_pad

    pts = np.zeros((height + 2, width + 2, 3), dtype=point.dtype)
    pts[1:-1, 1:-1, :] = point
    up = pts[:-2, 1:-1, :] - pts[1:-1, 1:-1, :]
    left = pts[1:-1, :-2, :] - pts[1:-1, 1:-1, :]
    down = pts[2:, 1:-1, :] - pts[1:-1, 1:-1, :]
    right = pts[1:-1, 2:, :] - pts[1:-1, 1:-1, :]
    normal = np.stack([
        np.cross(up, left, axis=-1),
        np.cross(left, down, axis=-1),
        np.cross(down, right, axis=-1),
        np.cross(right, up, axis=-1),
    ])
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)
    
    valid = np.stack([
        mask[:-2, 1:-1] & mask[1:-1, :-2],
        mask[1:-1, :-2] & mask[2:, 1:-1],
        mask[2:, 1:-1] & mask[1:-1, 2:],
        mask[1:-1, 2:] & mask[:-2, 1:-1],
    ]) & mask[None, 1:-1, 1:-1]
    if edge_threshold is not None:
        view_angle = angle_between(pts[None, 1:-1, 1:-1, :], normal)
        view_angle = np.minimum(view_angle, np.pi - view_angle)
        valid = valid & (view_angle < np.deg2rad(edge_threshold))
    
    normal = (normal * valid[..., None]).sum(axis=0)
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-12)
    
    if has_mask:
        normal_mask =  valid.any(axis=0)
        normal = np.where(normal_mask[..., None], normal, 0)
        return normal, normal_mask
    else:
        return normal


def depth_map_to_normal_map(depth: ndarray, intrinsics: ndarray, mask: ndarray = None, edge_threshold: float = None) -> ndarray:
    """Calculate normal map from depth map. Value range is [-1, 1]. Normal direction in OpenCV identity camera's coordinate system.

    ## Parameters
        depth (ndarray): shape (height, width), linear depth map
        intrinsics (ndarray): shape (3, 3), intrinsics matrix
        mask (optional, ndarray): shape (height, width), dtype=bool. Mask of valid depth pixels. Defaults to None.
        edge_threshold (optional, float): threshold for the angle (in degrees) between the normal and the view direction. Defaults to None.

    ## Returns
        normal (ndarray): shape (height, width, 3), normal map. 
    """
    points = depth_map_to_point_map(depth, intrinsics)
    normals = point_map_to_normal_map(points, mask, edge_threshold)
    return normals


def depth_map_to_point_map(
    depth: ndarray,
    intrinsics: ndarray,
    extrinsics: ndarray = None,
) -> ndarray:
    """Unproject depth map to 3D points.

    ## Parameters
        depth (ndarray): [..., H, W] depth value
        intrinsics ( ndarray): [..., 3, 3] intrinsics matrix
        extrinsics (optional, ndarray): [..., 4, 4] extrinsics matrix

    ## Returns
        points (ndarray): [..., N, 3] 3d points
    """
    assert intrinsics is not None, "intrinsics matrix is required"
    uv = uv_map(depth.shape[-2:], dtype=depth.dtype)
    points = unproject_cv(
        uv, 
        depth, 
        intrinsics=intrinsics[..., None, :, :], 
        extrinsics=extrinsics[..., None, :, :] if extrinsics is not None else None
    )
    return points


def chessboard(*size: Union[int, Tuple[int, int]], grid_size: int, color_a: ndarray, color_b: ndarray) -> ndarray:
    """Get a chessboard image

    ## Parameters
        - `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
        - `grid_size (int)`: size of chessboard grid
        - `color_a (ndarray)`: color of the grid at the top-left corner
        - `color_b (ndarray)`: color in complementary grid cells

    ## Returns
        image (ndarray): shape (height, width, channels), chessboard image
    """
    if len(size) == 1 and isinstance(size[0], tuple):
        height, width = size[0]
    else:
        height, width = size
    x = np.arange(width) // grid_size
    y = np.arange(height) // grid_size
    mask = (x[None, :] + y[:, None]) % 2
    image = np.where(mask[..., None], color_a, color_b)
    return image


def masked_nearest_resize(
    *image: ndarray,
    mask: ndarray, 
    size: Tuple[int, int], 
    return_index: bool = False
) -> Tuple[Unpack[Tuple[ndarray, ...]], ndarray, Tuple[ndarray, ...]]:
    """
    Resize image(s) by nearest sampling with mask awareness. Suitable for sparse maps. ![masked_nearest_resize.png](doc/masked_nearest_resize.png)
    - Downsampling: Assign the nearest valid pixel within the target pixel's receptive field.
    - Upsampling: Assign the valid pixel to only the nearest pixel in the resized map.

    ### Parameters
    - `*image`: Input image(s) of shape `(..., H, W, C)` or `(... , H, W)` 
        - You can pass multiple images to be resized at the same time for efficiency.
    - `mask`: input mask of shape `(..., H, W)`, dtype=bool
    - `size`: target size `(H', W')`
    - `return_index`: whether to return the nearest neighbor indices in the original map for each pixel in the resized map.
        Defaults to False.

    ### Returns
    - `*resized_image`: resized image(s) of shape `(..., H', W', C)`. or `(..., H', W')`
    - `resized_mask`: mask of the resized map of shape `(..., H', W')`
    - `nearest_indices`: tuple of shape `(..., H', W')`. The nearest neighbor indices of the resized map of each dimension.
    """
    height, width = mask.shape[-2:]
    target_height, target_width = size
    filter_h_f, filter_w_f = height / target_height, width / target_width
    filter_h_i, filter_w_i = math.ceil(filter_h_f), math.ceil(filter_w_f)
    filter_size = filter_h_i * filter_w_i
    filter_shape = (filter_h_i, filter_w_i)
    padding_h, padding_w = filter_h_i // 2 + 1, filter_w_i // 2 + 1
    padding_shape = ((padding_h, padding_h), (padding_w, padding_w))
    
    # Window the original mask and uv
    pixels = pixel_coord_map(height, width, convention='integer-corner', dtype=np.float32)
    indices = np.arange(height * width, dtype=np.int32).reshape(height, width)
    window_pixels = sliding_window(pixels, window_size=filter_shape, pad_size=padding_shape, axis=(0, 1))
    window_indices = sliding_window(indices, window_size=filter_shape, pad_size=padding_shape, axis=(0, 1))
    window_mask = sliding_window(mask, window_size=filter_shape, pad_size=padding_shape, axis=(-2, -1))

    # Gather the target pixels's local window
    target_centers = uv_map(target_height, target_width, dtype=np.float32) * np.array([width, height], dtype=np.float32)
    target_lefttop = target_centers - np.array((filter_w_f / 2, filter_h_f / 2), dtype=np.float32)
    target_window = np.round(target_lefttop).astype(np.int32) + np.array((padding_w, padding_h), dtype=np.int32)

    target_window_pixels = window_pixels[target_window[..., 1], target_window[..., 0], :, :, :].reshape(target_height, target_width, 2, filter_size)                  # (target_height, tgt_width, 2, filter_size)
    target_window_mask = window_mask[..., target_window[..., 1], target_window[..., 0], :, :].reshape(*mask.shape[:-2], target_height, target_width, filter_size)     # (..., target_height, tgt_width, filter_size)
    target_window_indices = window_indices[target_window[..., 1], target_window[..., 0], :, :].reshape(target_height, target_width, filter_size)                      # (target_height, tgt_width, filter_size)

    # Compute nearest neighbor in the local window for each pixel 
    delta = target_window_pixels - target_centers[..., None]
    eps = np.finfo(np.float32).eps * max(height, width) # Shift a small epsilon to avoid numerical issues when the pixel is exactly on the border
    target_window_mask &= (-filter_w_f / 2 + eps < delta[..., 0, :]) & (delta[..., 0, :] <= filter_w_f / 2 + eps) & (-filter_h_f / 2 + eps < delta[..., 1, :]) & (delta[..., 1, :] <= filter_h_f / 2 + eps)
    dist = np.where(target_window_mask, np.square(delta[..., 0, :]) + np.square(delta[..., 1, :]), np.inf)      # (..., target_height, tgt_width, filter_size)
    nearest_in_window = np.argmin(dist, axis=-1, keepdims=True)                                                 # (..., target_height, tgt_width, 1)
    nearest_idx = np.take_along_axis(
        np.broadcast_to(target_window_indices, dist.shape),
        nearest_in_window, 
        axis=-1
    ).squeeze(-1)     # (..., target_height, tgt_width)
    nearest_i, nearest_j = nearest_idx // width, nearest_idx % width
    target_mask = np.any(target_window_mask, axis=-1)
    batch_indices = [np.arange(n).reshape([1] * i + [n] + [1] * (mask.ndim - i - 1)) for i, n in enumerate(mask.shape[:-2])]

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
    *image: ndarray,
    mask: ndarray, 
    size: Tuple[int, int]
) -> Tuple[Unpack[Tuple[ndarray, ...]], ndarray]:
    """
    Resize 2D map by area sampling with mask awareness.

    ### Parameters
    - `*image`: Input image(s) of shape `(..., H, W, C)` or `(..., H, W)`
        - You can pass multiple images to be resized at the same time for efficiency.
    - `mask`: Input mask of shape `(..., H, W)`
    - `size`: target image size `(H', W')`

    ### Returns
    - `*resized_image`: resized image(s) of shape `(..., H', W', C)`. or `(..., H', W')`
    - `resized_mask`: mask of the resized map of shape `(..., H', W')`
    """
    height, width = mask.shape[-2:]
    target_height, target_width = size

    filter_h_f, filter_w_f = max(1, height / target_height), max(1, width / target_width)
    filter_h_i, filter_w_i = math.ceil(filter_h_f), math.ceil(filter_w_f)
    filter_size = filter_h_i * filter_w_i
    filter_shape = (filter_h_i, filter_w_i)
    padding_h, padding_w = filter_h_i // 2 + 1, filter_w_i // 2 + 1
    padding_shape = ((padding_h, padding_h), (padding_w, padding_w))
    
    # Window the original mask and uv (non-copy)
    pixels = pixel_coord_map((height, width), convention='integer-corner', dtype=np.float32)
    indices = np.arange(height * width, dtype=np.int32).reshape(height, width)
    window_pixels = sliding_window(pixels, window_size=filter_shape, pad_size=padding_shape, axis=(0, 1))
    window_indices = sliding_window(indices, window_size=filter_shape, pad_size=padding_shape, axis=(0, 1))
    window_mask = sliding_window(mask, window_size=filter_shape, pad_size=padding_shape, axis=(-2, -1))

    # Gather the target pixels's local window
    target_center = uv_map((target_height, target_width), dtype=np.float32) * np.array([width, height], dtype=np.float32)
    target_lefttop = target_center - np.array((filter_w_f / 2, filter_h_f / 2), dtype=np.float32)
    target_bottomright = target_center + np.array((filter_w_f / 2, filter_h_f / 2), dtype=np.float32)
    target_window = np.floor(target_lefttop).astype(np.int32) + np.array((padding_w, padding_h), dtype=np.int32)

    target_window_centers = window_pixels[target_window[..., 1], target_window[..., 0], :, :, :].reshape(target_height, target_width, 2, filter_size)                 # (target_height, tgt_width, 2, filter_size)
    target_window_mask = window_mask[..., target_window[..., 1], target_window[..., 0], :, :].reshape(*mask.shape[:-2], target_height, target_width, filter_size)     # (..., target_height, tgt_width, filter_size)
    target_window_indices = window_indices[target_window[..., 1], target_window[..., 0], :, :].reshape(target_height, target_width, filter_size)                      # (target_height, tgt_width, filter_size)

    # Compute pixel area in the local windows
    # (..., target_height, tgt_width, filter_size)
    target_window_lefttop = np.maximum(target_window_centers - 0.5, target_lefttop[..., None])
    target_window_rightbottom = np.minimum(target_window_centers + 0.5, target_bottomright[..., None])
    target_window_area = (target_window_rightbottom - target_window_lefttop).clip(0, None)
    target_window_area = np.where(target_window_mask, target_window_area[..., 0, :] * target_window_area[..., 1, :], 0)

    target_area = np.sum(target_window_area, axis=-1)   # (..., target_height, tgt_width)
    target_mask = target_area >= 0

    # Weighted sum by area
    outputs = [] 
    for x in image:
        assert x.shape[:mask.ndim] == mask.shape, "Image and mask should have the same batch shape and spatial shape"
        expand_channels = (slice(None),) * (x.ndim - mask.ndim)
        x = np.where(mask[(..., *expand_channels)], x, 0)
        x = x.reshape(*x.shape[:mask.ndim - 2], height * width, *x.shape[mask.ndim:])[(*((slice(None),) * (mask.ndim - 2)), target_window_indices)]                   # (..., target_height, tgt_width, filter_size, ...)
        x = (x * target_window_area[(..., *expand_channels)]).sum(axis=mask.ndim) / np.maximum(target_area[(..., *expand_channels)], np.finfo(np.float32).eps)          # (..., target_height, tgt_width, ...)
        outputs.append(x)
    
    ret = *outputs, target_mask

    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def colorize_depth_map(depth: ndarray, mask: ndarray = None, near: Optional[float] = None, far: Optional[float] = None, cmap: str = 'Spectral') -> ndarray:
    """Colorize depth map for visualization.

    ## Parameters
        - `depth` (ndarray): shape (..., H, W), linear depth map
        - `mask` (ndarray, optional): shape (..., H, W), dtype=bool. Mask of valid depth pixels. Defaults to None.
        - `near` (float, optional): near plane for depth normalization. If None, use the 0.1% quantile of valid depth values. Defaults to None.
        - `far` (float, optional): far plane for depth normalization. If None, use the 99.9% quantile of valid depth values. Defaults to None.
        - `cmap` (str, optional): colormap name in matplotlib. Defaults to 'Spectral'.
    
    ## Returns
        - `colored` (ndarray): shape (..., H, W, 3), dtype=uint8, RGB [0, 255]
    """
    try:
        import matplotlib
    except ImportError:
        raise ImportError("matplotlib is required for colorizing depth map")
    
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)
    if near is None:
        near = np.nanmin(depth)
    if far is None:
        far = np.nanmax(depth)
    
    disp = (1 / depth - 1 / far) / (1 / near - 1 / far)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def colorize_normal_map(normal: ndarray, mask: ndarray = None, flip_yz: bool = False, normalize: bool = True) -> np.ndarray:
    """Colorize normal map for visualization. Value range is [-1, 1].
    
    ## Parameters
        - `normal` (ndarray): shape (H, W, 3), normal
        - `mask` (ndarray, optional): shape (H, W), dtype=bool. Mask of valid depth pixels. Defaults to None.
        - `flip_yz` (bool, optional): whether to flip the y and z. 
            - This is useful when converting between OpenCV and OpenGL camera coordinate systems. Defaults to False.
        - `normalize` (bool, optional): whether to normalize the normal vectors. Defaults to True.

    ## Returns
        - `colored` (ndarray): shape (H, W, 3), dtype=uint8, RGB in [0, 255]
    """
    if normalize:
        normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + np.finfo(normal.dtype).tiny)
    if mask is not None:
        normal = np.where(mask[..., None], normal, 0)
    if flip_yz:
        normal = normal * [0.5, -0.5, -0.5] + 0.5
    else:
        normal = normal * 0.5 + 0.5
    normal = (normal.clip(0, 1) * 255).astype(np.uint8)
    return normal


def colorize_segmentation_map(segmentation: np.ndarray, mask: Optional[np.ndarray] = None, vdim: int = 0) -> np.ndarray:
    """Colorize segmentation map for visualization. The same value will be assigned with the same color.

    Parameters
    ----
    - `segmentation` (ndarray): shape (..., H, W, [...]), segmentation map. The last `ndim` dimensions are treated as value channels.
    - `mask` (ndarray, optional): shape (..., H, W), binary mask indicating valid regions. Defaults to None (all regions are valid).
    - `vdim` (int, optional): number of dimensions to treat as value channels. Defaults to 0 (scalars)

    Returns
    ----
    - `colored` (ndarray): shape (..., H, W, 3), dtype=uint8, RGB in [0, 255]
    """
    if not (0 <= vdim <= segmentation.ndim):
        raise ValueError(f"vdim should be in [0, segmentation.ndim], got vdim={vdim} and segmentation.ndim={segmentation.ndim}")

    if vdim == 0:
        map_shape = segmentation.shape
        value_shape = ()
    else:
        map_shape = segmentation.shape[:-vdim]
        value_shape = segmentation.shape[-vdim:]

    flat = np.ascontiguousarray(segmentation.reshape(-1, *value_shape))
    value_bytes = flat.view(np.uint8).reshape(flat.shape[0], -1)

    # Align each value/vector to 4 bytes before viewing as uint32 words.
    pad = (-value_bytes.shape[1]) % 4
    if pad:
        value_bytes = np.pad(value_bytes, ((0, 0), (0, pad)), mode='constant', constant_values=0)

    words = value_bytes.view(np.uint32).reshape(flat.shape[0], -1)

    # 32-bit FNV-1a style accumulation, then avalanche mix.
    hashed = np.full(words.shape[0], np.uint32(0x811C9DC5), dtype=np.uint32)
    for i in range(words.shape[1]):
        hashed ^= words[:, i]
        hashed *= np.uint32(0x01000193)

    hashed ^= hashed >> np.uint32(16)
    hashed *= np.uint32(0x7FEB352D)
    hashed ^= hashed >> np.uint32(15)
    hashed *= np.uint32(0x846CA68B)
    hashed ^= hashed >> np.uint32(16)

    hash_bytes = hashed.view(np.uint8).reshape(-1, 4)
    if np.little_endian:
        colored = hash_bytes[:, 2::-1]
    else:
        colored = hash_bytes[:, 1:4]
    colored = colored.reshape(*map_shape, 3)
    if mask is not None:
        colored = np.where(mask, colored, 0)
    return np.ascontiguousarray(colored)


def colorize_probability_map(probability: np.ndarray, mask: Optional[np.ndarray] = None, cmap: str = 'viridis', alpha: float = 1.0, beta: float = 1.0):
    """Colorize probability map for visualization.

    The remapping is:
    `p' = p^alpha / (p^alpha + (1 - p)^beta)`
    where larger `alpha` suppresses low-to-mid probabilities and larger `beta` suppresses high probabilities.

    Parameters
    -------
    - `probability` (ndarray): shape (..., H, W), probability map with values in [0, 1].
    - `mask` (ndarray, optional): shape (..., H, W), binary mask indicating valid regions. Defaults to None (all regions are valid).
    - `cmap` (str, optional): colormap name in matplotlib. Defaults to 'viridis'.
    - `alpha` (float, optional): exponent applied to `probability` in contrast remapping. Defaults to 1.0.
    - `beta` (float, optional): exponent applied to `(1 - probability)` in contrast remapping. Defaults to 1.0.

    Returns
    ------
    - `colored` (ndarray): shape (..., H, W, 3), dtype=uint8, RGB in [0, 255].

    Examples for tuning `alpha` and `beta`
    ------
    
    | Parameters | Curve shape | Effect |
    |---|---|---|
    | `alpha=2.5, beta=2.5` | Steep S-shape | Separate "possible" and "impossible" around 0.5 |
    | `alpha=0.3, beta=1.0` | Fast early rise, then gradual flattening | Emphasize tiny probabilities near zero |
    | `alpha=1.0, beta=0.3` | Flat early, then sharp rise near the end | Emphasize subtle differences near certainty (close to 1) |
    | `alpha=0.5, beta=0.5` | Hourglass-like (inverse-S) | Increase contrast near both extremes globally |
    """
    try:
        import matplotlib
    except ImportError:
        raise ImportError("matplotlib is required for colorizing probability map")
    a, b = probability ** alpha, (1 - probability) ** beta
    remapped_prob = a / (a + b)
    colored = matplotlib.colormaps[cmap](remapped_prob)[..., :3]
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    if mask is not None:
        colored = np.where(mask[..., None], colored, 0)
    return colored



def flood_fill(*image: ndarray, mask: ndarray, return_index: bool = False) -> ndarray:
    """
    Flooding fill the holes in the image(s) according to the mask. ![doc/flood_fill.png](doc/flood_fill.png)
    
    Parameters
    ----
    - `*image` (ndarray): shape (..., height, width, [C]), input image(s)
    - `mask` (ndarray): shape (..., height, width), binary mask indicating valid regions

    Returns
    ----
    - `*filled_image` (ndarray): shape (..., height, width, [C]), flood filled map
    - `filled_indices` (Tuple[ndarray, ...], optional): tuple of shape (..., height, width). The nearest neighbor indices of each pixel in the original map.
        It satisfies `filled_image = image[filled_indices]`
    """
    *batch_shape, height, width = mask.shape

    self_i, self_j = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing='ij'
    )
    nearest_i, nearest_j = np.broadcast_to(self_i, mask.shape), np.broadcast_to(self_j, mask.shape)

    filled_mask = mask.copy()

    step = (max(height, width) + 1) // 2
    steps = []
    while step > 1:
        steps.append(step)
        step = step // 2
    steps += [1, 1]
    for step in steps:
        sliding_window_i = sliding_window(nearest_i, window_size=3, dilation=step, pad_size=step, axis=(-2, -1)).reshape(*batch_shape, height, width, 9)
        sliding_window_j = sliding_window(nearest_j, window_size=3, dilation=step, pad_size=step, axis=(-2, -1)).reshape(*batch_shape, height, width, 9)
        sliding_window_mask = sliding_window(filled_mask, window_size=3, dilation=step, pad_size=step, axis=(-2, -1)).reshape(*batch_shape, height, width, 9)
        filled_mask |= sliding_window_mask.any(axis=-1)
        dist = np.where(sliding_window_mask, np.square(sliding_window_i - self_i[..., None]) + np.square(sliding_window_j - self_j[..., None]), np.inf)
        nearest_in_window = np.argmin(dist, axis=-1)
        nearest_i = np.take_along_axis(sliding_window_i, nearest_in_window[..., None], axis=-1)[..., 0]
        nearest_j = np.take_along_axis(sliding_window_j, nearest_in_window[..., None], axis=-1)[..., 0]

    batch_indices = [np.arange(n).reshape([1] * i + [n] + [1] * (mask.ndim - i - 1)) for i, n in enumerate(batch_shape)]
    nearest_i, nearest_j = nearest_i.round().astype(int), nearest_j.round().astype(int)
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


def perlin_noise(x: ndarray, seed: Optional[int] = None) -> ndarray:
    """Generate Perlin noise for the given coordinates.
    
    Parameters
    ----
    - `x` (ndarray): shape (*batch_shape, N_1, ..., N_D, D), coordinates to sample Perlin.
        If input ndim is more than D + 1, the leading dimensions are treated as batch dimensions. Instances in the batch have different noise patterns.
    - `seed` (int, optional): random seed. The same seed will generate the same noise pattern(s). Defaults to None.

    Returns
    ----
    - `y` (ndarray): shape (*batch_shape, N_1, ..., N_D), Perlin noise value at the given coordinates and seed. Value range is approximately [-1, 1]
    """
    D = x.shape[-1]
    xi = x.astype(int)
    xf = x - xi

    # Prepare gradient set and hash permutation
    rng = np.random.default_rng(seed)
    hash_set_size = min(16 ** D, x.size // D)
    perm = rng.permutation(hash_set_size)
    gradient_set = rng.standard_normal((hash_set_size, D)).astype(x.dtype)
    gradient_set = gradient_set / (lite_norm(gradient_set, axis=-1)[..., None] + np.finfo(x.dtype).tiny)
    
    # Get corner coordinates
    unit_grid = np.stack(
        np.meshgrid(*([[0, 1]] * D), indexing='ij'), 
        axis=-1
    ).reshape(-1, D)  # (2 ** D, D)
    x_corners = xi[..., None, :].repeat(unit_grid.shape[0], axis=-2)
    x_corners += unit_grid
    
    # Hash corner coordinates
    hash_idx = perm[x_corners[..., 0] % hash_set_size]
    for i in range(1, x.shape[-1]):
        hash_idx += x_corners[..., i]
        hash_idx %= hash_set_size
        hash_idx = perm[hash_idx]

    # Batch hashing
    if x.ndim > D + 1:
        # If batch dimensions exist, instances have different noise patterns
        batch_idx = np.arange(math.prod(x.shape[:x.ndim - D - 1]))
        batch_idx = batch_idx.reshape(*x.shape[:x.ndim - D - 1], *([1] * (D + 1)))
        hash_idx = perm[(hash_idx + batch_idx) % hash_set_size]

    # Compute dot products of each corner's gradient and the direction to the point
    gradients = gradient_set.view(f'<S{gradient_set.itemsize * D}').ravel()[hash_idx].view(x.dtype).reshape(*hash_idx.shape, D) # NOTE: equivalent to `gradients = gradient_set[hash_idx]`, but 6x faster. Flattened fancy indexing is faster.
    direction = xf[..., None, :].repeat(unit_grid.shape[0], axis=-2)
    direction -= unit_grid.astype(x.dtype)      # NOTE: equivalent to `direction = xf[..., None, :] - unit_grid.astype(x.dtype)`, but 3x faster. Repeating is faster than broadcasting here.
    dot = lite_dot(gradients, direction, axis=-1)

    # Smooth interpolation
    u = np.polyval([6, -15, 10, 0, 0, 0], xf)
    w = u[..., None, :].repeat(unit_grid.shape[0], axis=-2)
    w -= unit_grid.astype(x.dtype)
    np.abs(w, out=w)
    w *= -1
    w += 1          # NOTE: equivalent to `w = 1 - np.abs(u[..., None, :] - unit_grid.astype(x.dtype))`, but 3x faster
    w = lite_prod(w, axis=-1)
    y = lite_dot(dot, w, axis=-1)

    return y


def perlin_noise_map(size: Tuple[int, ...], frequency: Union[float, ndarray], seed: Optional[int] = None, dtype: Optional[np.dtype] = np.float32) -> ndarray:
    """Generate Perlin noise map.

    Parameters
    ----
    - `size` (Tuple[int, ...]): size of the noise map (..., H, W)
    - `frequency` (float | ndarray): frequency relative to map's larger dimension max(H, W) of the Perlin noise. 
        Represents how many periods of noise fit in the larger dimension of the map.
    - `seed` (int, optional): random seed. The same seed will generate the same noise pattern(s). Defaults to None.

    Returns
    ----
    - `noise_map` (ndarray): shape (..., H, W), Perlin noise map. Value range is approximately [-1, 1]
    """
    height, width = size[-2:]
    uv = uv_map(size[-2:], dtype=dtype) 
    frequency_uv = np.asarray(frequency, dtype=dtype)[..., None] * (np.array([width, height], dtype=dtype) / max(size[-2:]))
    x = np.broadcast_to(uv * frequency_uv[..., None, None, :], (*size, 2)) 
    noise_map = perlin_noise(x, seed=seed)
    return noise_map


def fractal_perlin_noise_map(
    size: Tuple[int, ...], 
    base_frequency: Union[float, ndarray], 
    octaves: int = 4, 
    lacunarity: float = 2.0, 
    gain: float = 0.5, 
    seed: Optional[int] = None, 
    dtype: Optional[np.dtype] = np.float32
) -> ndarray:
    """Generate fractal Perlin noise map. ![fractal_perlin_base_frequeny2_octaves7_gain0.7.png](doc/fractal_perlin_base_frequeny2_octaves7_gain0.7.png)

    Parameters
    ----
    - `size` (Tuple[int, ...]): size of the noise map (..., H, W)
    - `base_frequency` (float | ndarray): base frequency (relative to map's larger dimension max(H, W)) of the Perlin noise.
        Represents how many periods of noise fit in the larger dimension of the map.
    - `octaves` (int, optional): number of octaves. Defaults to 4.
    - `lacunarity` (float, optional): frequency multiplier between octaves. Defaults to 2.0.
    - `gain` (float, optional): amplitude multiplier between octaves. Defaults to 0.5.
    - `seed` (int, optional): random seed. The same seed will generate the same noise pattern. Defaults to None.

    Returns
    ----
    - `noise_map` (ndarray): shape (..., H, W), fractal Perlin noise map. Value range is approximately [-1, 1]
    """
    noise_map = np.zeros(size, dtype=dtype)
    amplitude = 1.0
    frequency_uv = np.asarray(base_frequency, dtype=dtype).copy()[..., None] * (np.array([size[-1], size[-2]], dtype=dtype) / max(size[-2:]))
    uv = uv_map(size[-2:], dtype=dtype) 
    for octave in range(octaves):
        x = np.broadcast_to(uv * frequency_uv[..., None, None, :], (*size, 2)) 
        noise_map += amplitude * perlin_noise(x, seed=seed + octave if seed is not None else None)
        amplitude *= gain
        frequency_uv *= lacunarity
    return noise_map