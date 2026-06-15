from typing import *
from numbers import Number 
import itertools

import torch
from torch import Tensor
import torch.nn.functional as F

from .helpers import batched, totensor


__all__ = [
    'perspective_from_fov',
    'perspective_from_window', 
    'intrinsics_from_fov',
    'intrinsics_from_focal_center',
    'focal_to_fov',
    'fov_to_focal',
    'intrinsics_to_fov',
    'view_look_at',
    'extrinsics_look_at',
    'perspective_to_intrinsics',
    'intrinsics_to_perspective',
    'extrinsics_to_view',
    'view_to_extrinsics',
    'normalize_intrinsics',
    'denormalize_intrinsics',
    'crop_intrinsics',
    'pixel_to_uv',
    'pixel_to_ndc',
    'uv_to_pixel',
    'depth_linear_to_buffer',
    'depth_buffer_to_linear',
    'project_gl',
    'project_cv',
    'unproject_gl',
    'unproject_cv',
    'project',
    'unproject',
    'skew_symmetric',
    'rotation_matrix_from_vectors',
    'euler_axis_angle_rotation',
    'euler_angles_to_matrix',
    'matrix_to_euler_angles',
    'matrix_to_quaternion',
    'quaternion_to_matrix',
    'quaternion_multiply',
    'quaternion_inverse',
    'quaternion_normalize',
    'matrix_to_axis_angle',
    'axis_angle_to_matrix',
    'axis_angle_to_quaternion',
    'quaternion_to_axis_angle',
    'make_affine_matrix',
    'random_rotation_matrix',
    'lerp',
    'slerp',
    'slerp_rotation_matrix',
    'interpolate_se3_matrix',
    'extrinsics_to_essential',
    'rotation_matrix_2d',
    'rotate_2d',
    'translate_2d',
    'scale_2d',
    'transform_points',
    'angle_between'
]



@totensor(_others=torch.float32)
@batched(_others=0)
def perspective_from_fov(
    *,
    fov_x: Optional[Union[float, Tensor]] = None,
    fov_y: Optional[Union[float, Tensor]] = None,
    fov_min: Optional[Union[float, Tensor]] = None,
    fov_max: Optional[Union[float, Tensor]] = None,
    aspect_ratio: Optional[Union[float, Tensor]] = None,
    near: Optional[Union[float, Tensor]],
    far: Optional[Union[float, Tensor]],
) -> Tensor:
    """
    Get OpenGL perspective matrix from field of view 

    ## Returns
        (Tensor): [..., 4, 4] perspective matrix
    """
    if fov_max is not None:
        fx = torch.maximum(1, 1 / aspect_ratio) / torch.tan(fov_max / 2)
        fy = torch.maximum(1, aspect_ratio) / torch.tan(fov_max / 2)
    elif fov_min is not None:
        fx = torch.minimum(1, 1 / aspect_ratio) / torch.tan(fov_min / 2)
        fy = torch.minimum(1, aspect_ratio) / torch.tan(fov_min / 2)
    elif fov_x is not None and fov_y is not None:
        fx = 1 / torch.tan(fov_x / 2)
        fy = 1 / torch.tan(fov_y / 2)
    elif fov_x is not None:
        fx = 1 / torch.tan(fov_x / 2)
        fy = fx * aspect_ratio
    elif fov_y is not None:
        fy = 1 / torch.tan(fov_y / 2)
        fx = fy / aspect_ratio
    zeros = torch.zeros_like(fx)
    ones = torch.ones_like(fx)
    perspective = torch.stack([
        fx, zeros, zeros, zeros,
        zeros, fy, zeros, zeros,
        zeros, zeros, (near / far + 1) / (near / far - 1), 2. * near / (near / far - 1),
        zeros, zeros, -ones, zeros
    ], dim=-1).unflatten(-1, (4, 4))
    return perspective


@totensor(_others=torch.float32)
@batched(_others=0)
def perspective_from_window(
    left: Union[float, Tensor],
    right: Union[float, Tensor],
    bottom: Union[float, Tensor],
    top: Union[float, Tensor],
    near: Union[float, Tensor],
    far: Union[float, Tensor]
) -> Tensor:
    """
    Get OpenGL perspective matrix from the window of z=-1 projection plane

    ## Returns
        (Tensor): [..., 4, 4] perspective matrix
    """
    zeros = torch.zeros_like(left)
    ones = torch.ones_like(left)
    perspective = torch.stack([
        2 / (right - left), zeros, (right + left) / (right - left), zeros,
        zeros, 2 / (top - bottom), (top + bottom) / (top - bottom), zeros,
        zeros, zeros, (near / far + 1) / (near / far - 1), 2. * near / (near / far - 1),
        zeros, zeros, -ones, zeros
    ], dim=-1).unflatten(-1, (4, 4))
    return perspective


@totensor(_others=torch.float32)
@batched(_others=0)
def intrinsics_from_focal_center(
    fx: Union[float, Tensor],
    fy: Union[float, Tensor],
    cx: Union[float, Tensor],
    cy: Union[float, Tensor]
) -> Tensor:
    """
    Get OpenCV intrinsics matrix

    ## Parameters
        focal_x (float | Tensor): focal length in x axis
        focal_y (float | Tensor): focal length in y axis
        cx (float | Tensor): principal point in x axis
        cy (float | Tensor): principal point in y axis

    ## Returns
        (Tensor): [..., 3, 3] OpenCV intrinsics matrix
    """
    zeros, ones = torch.zeros_like(fx), torch.ones_like(fx)
    ret = torch.stack([
        fx, zeros, cx, 
        zeros, fy, cy, 
        zeros, zeros, ones
    ], dim=-1).unflatten(-1, (3, 3))
    return ret



@totensor(_others=torch.float32)
@batched(_others=0)
def intrinsics_from_fov(
    *,
    fov_x: Optional[Union[float, Tensor]] = None,
    fov_y: Optional[Union[float, Tensor]] = None,
    fov_max: Optional[Union[float, Tensor]] = None,
    fov_min: Optional[Union[float, Tensor]] = None,
    cx: Union[float, Tensor] = 0.5,
    cy: Union[float, Tensor] = 0.5,
    aspect_ratio: Optional[Union[float, Tensor]] = None,
) -> Tensor:
    """
    Get normalized OpenCV intrinsics matrix from given field of view.
    You can provide either fov_x, fov_y, fov_max or fov_min and aspect_ratio

    Parameters
    ----
        fov_x (float | Tensor): field of view in x axis
        fov_y (float | Tensor): field of view in y axis
        fov_max (float | Tensor): field of view in largest dimension
        fov_min (float | Tensor): field of view in smallest dimension
        cx (float | Tensor): principal point x coordinate
        cy (float | Tensor): principal point y coordinate
        aspect_ratio (float | Tensor): aspect ratio of the image

    Returns
    ----
        (Tensor): [..., 3, 3] OpenCV intrinsics matrix
    """
    if fov_max is not None:
        fx = torch.maximum(1, 1 / aspect_ratio) / (2 * torch.tan(fov_max / 2))
        fy = torch.maximum(1, aspect_ratio) / (2 * torch.tan(fov_max / 2))
    elif fov_min is not None:
        fx = torch.minimum(1, 1 / aspect_ratio) / (2 * torch.tan(fov_min / 2))
        fy = torch.minimum(1, aspect_ratio) / (2 * torch.tan(fov_min / 2))
    elif fov_x is not None and fov_y is not None:
        fx = 1 / (2 * torch.tan(fov_x / 2))
        fy = 1 / (2 * torch.tan(fov_y / 2))
    elif fov_x is not None:
        fx = 1 / (2 * torch.tan(fov_x / 2))
        fy = fx * aspect_ratio
    elif fov_y is not None:
        fy = 1 / (2 * torch.tan(fov_y / 2))
        fx = fy / aspect_ratio
    ret = intrinsics_from_focal_center(fx, fy, cx, cy)
    return ret


def focal_to_fov(focal: Tensor):
    return 2 * torch.atan(0.5 / focal)


def fov_to_focal(fov: Tensor):
    return 0.5 / torch.tan(fov / 2)


def intrinsics_to_fov(intrinsics: Tensor) -> Tuple[Tensor, Tensor]:
    "NOTE: approximate FOV by assuming centered principal point"
    fov_x = focal_to_fov(intrinsics[..., 0, 0])
    fov_y = focal_to_fov(intrinsics[..., 1, 1])
    return fov_x, fov_y


@totensor(_others=torch.float32)
@batched(1, 1, 1)
def view_look_at(
    eye: Tensor,
    look_at: Tensor,
    up: Tensor
) -> Tensor:
    """
    Get OpenGL view matrix looking at something

    ## Parameters
        eye (Tensor): [..., 3] the eye position
        look_at (Tensor): [..., 3] the position to look at
        up (Tensor): [..., 3] head up direction (y axis in screen space). Not necessarily othogonal to view direction

    ## Returns
        (Tensor): [..., 4, 4], view matrix
    """
    z = eye - look_at
    x = torch.cross(up, z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    R = torch.stack([x, y, z], dim=-2)
    R = F.normalize(R, dim=-1)
    t = -torch.matmul(R, eye[..., None])
    return make_affine_matrix(R, t.squeeze(-1))


@totensor(_others=torch.float32)
@batched(1, 1, 1)
def extrinsics_look_at(
    eye: Tensor,
    look_at: Tensor,
    up: Tensor
) -> Tensor:
    """
    Get OpenCV extrinsics matrix looking at something

    ## Parameters
        eye (Tensor): [..., 3] the eye position
        look_at (Tensor): [..., 3] the position to look at
        up (Tensor): [..., 3] head up direction (-y axis in screen space). Not necessarily othogonal to view direction

    ## Returns
        (Tensor): [..., 4, 4], extrinsics matrix
    """
    return view_to_extrinsics(view_look_at(eye, look_at, up))


@batched(2)
def perspective_to_intrinsics(perspective: Tensor) -> Tensor:
    """
    OpenGL perspective matrix to OpenCV intrinsics

    ## Parameters
        perspective (Tensor): [..., 4, 4] OpenGL perspective matrix

    ## Returns
        (Tensor): shape [..., 3, 3] OpenCV intrinsics
    """
    assert torch.allclose(perspective[:, [0, 1, 3], 3], 0), "The matrix is not a perspective projection matrix"
    ret = torch.tensor([[0.5, 0., 0.5], [0., -0.5, 0.5], [0., 0., 1.]], dtype=perspective.dtype, device=perspective.device) \
        @ perspective[:, [0, 1, 3], :3] \
        @ torch.diag(torch.tensor([1, -1, -1], dtype=perspective.dtype, device=perspective.device))
    return ret / ret[:, 2, 2, None, None]


@totensor(None, _others='intrinsics')
@batched(2, 0, 0)
def intrinsics_to_perspective(
    intrinsics: Tensor,
    near: Union[float, Tensor],
    far: Union[float, Tensor],
) -> Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix
    NOTE: not work for tile-shifting intrinsics currently

    ## Parameters
        intrinsics (Tensor): [..., 3, 3] OpenCV intrinsics matrix
        near (float | Tensor): [...] near plane to clip
        far (float | Tensor): [...] far plane to clip
    ## Returns
        (Tensor): [..., 4, 4] OpenGL perspective matrix
    """
    device, dtype = intrinsics.device, intrinsics.dtype
    batch_shape = intrinsics.shape[:-2]
    m = torch.tensor([[2, 0, -1], [0, -2, 1], [0, 0, 1]], dtype=dtype, device=device) @ intrinsics @ torch.diag(torch.tensor([1, -1, -1], dtype=dtype, device=device))
    perspective = torch.cat([
        torch.cat([m[..., :2, :], torch.zeros((*batch_shape, 2, 1), dtype=dtype, device=device)], dim=-1),
        torch.cat([torch.zeros((*batch_shape, 1, 2), dtype=dtype, device=device), ((near / far + 1) / (near / far - 1))[..., None, None], (2. * near / (near / far - 1))[..., None, None]], dim=-1),
        torch.tensor([0., 0., -1, 0], dtype=dtype, device=device).expand(*batch_shape, 1, 4)
    ], dim=-2)
    return perspective


def extrinsics_to_view(extrinsics: Tensor) -> Tensor:
    """
    OpenCV camera extrinsics to OpenGL view matrix

    ## Parameters
        extrinsics (Tensor): [..., 4, 4] OpenCV camera extrinsics matrix

    ## Returns
        (Tensor): [..., 4, 4] OpenGL view matrix
    """
    return extrinsics * torch.tensor([1, -1, -1, 1], dtype=extrinsics.dtype, device=extrinsics.device)[:, None]


def view_to_extrinsics(view: Tensor) -> Tensor:
    """
    OpenGL view matrix to OpenCV camera extrinsics

    ## Parameters
        view (Tensor): [..., 4, 4] OpenGL view matrix

    ## Returns
        (Tensor): [..., 4, 4] OpenCV camera extrinsics matrix
    """
    return view  * torch.tensor([1, -1, -1, 1], dtype=view.dtype, device=view.device)[:, None]


@totensor(None, 'intrinsics')
@batched(2, 1)
def normalize_intrinsics(
    intrinsics: Tensor,
    size: Union[Tuple[Number, Number], Tensor],
    pixel_convention: Literal['integer-corner', 'integer-center'] = 'integer-center',
) -> Tensor:
    """
    Normalize camera intrinsics to uv space

    ## Parameters
    - `intrinsics` (Tensor): `(..., 3, 3)` camera intrinsics to normalize
    - `size` (tuple | Tensor): A tuple `(height, width)` of the image size,
        or an array of shape `(..., 2)` corresponding to the multiple image size(s)
    - `pixel_convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
        - For more definitions, please refer to `pixel_coord_map()`

    ## Returns
        (Tensor): [..., 3, 3] normalized camera intrinsics
    """
    if isinstance(size, tuple):
        size = torch.tensor(size, dtype=intrinsics.dtype, device=intrinsics.device)
        size = size.expand(*intrinsics.shape[:-2], 2)
    height, width = size.unbind(-1)
    zeros = torch.zeros_like(width)
    ones = torch.ones_like(width)
    if pixel_convention == 'integer-center':
        transform = torch.stack([
            1 / width, zeros, 0.5 / width,
            zeros, 1 / height, 0.5 / height,
            zeros, zeros, ones
        ], dim=-1).reshape(*zeros.shape, 3, 3)
    elif pixel_convention == 'integer-corner':
        transform = torch.stack([
            1 / width, zeros, zeros,
            zeros, 1 / height, zeros,
            zeros, zeros, ones
        ], dim=-1).reshape(*zeros.shape, 3, 3)
    return transform @ intrinsics


@totensor(None, 'intrinsics')
@batched(2, 1)
def denormalize_intrinsics(
    intrinsics: Tensor,
    size: Union[Tuple[Number, Number], Tensor],
    pixel_convention: Literal['integer-center', 'integer-corner'] = 'integer-center',
) -> Tensor:
    """
    Denormalize camera intrinsics(s) from uv space to pixel space

    ## Parameters
    - `intrinsics` (Tensor): `(..., 3, 3)` camera intrinsics
    - `size` (tuple | Tensor): A tuple `(height, width)` of the image size,
        or an array of shape `(..., 2)` corresponding to the multiple image size(s)
    - `pixel_convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
        - For more definitions, please refer to `pixel_coord_map()`

    ## Returns
        (Tensor): [..., 3, 3] denormalized camera intrinsics in pixel space
    """
    if isinstance(size, tuple):
        size = torch.tensor(size, dtype=intrinsics.dtype, device=intrinsics.device)
        size = size.expand(*intrinsics.shape[:-2], 2)
    height, width = size.unbind(-1)
    zeros = torch.zeros_like(width)
    ones = torch.ones_like(width)
    if pixel_convention == 'integer-center':
        transform = torch.stack([
            width, zeros, -0.5 * ones,
            zeros, height, -0.5 * ones,
            zeros, zeros, ones
        ], dim=-1).reshape(*zeros.shape, 3, 3)
    elif pixel_convention == 'integer-corner':
        transform = torch.stack([
            width, zeros, zeros,
            zeros, height, zeros,
            zeros, zeros, ones
        ], dim=-1).reshape(*zeros.shape, 3, 3)
    return transform @ intrinsics


@totensor(None, _others='intrinsics')
@batched(2, 1, _others=0)
def crop_intrinsics(
    intrinsics: Tensor,
    size: Union[Tuple[Number, Number], Tensor],
    cropped_top: Union[Number, Tensor],
    cropped_left: Union[Number, Tensor],
    cropped_height: Union[Number, Tensor],
    cropped_width: Union[Number, Tensor],
) -> Tensor:
    """
    Evaluate the new intrinsics after cropping the image

    ## Parameters
        intrinsics (Tensor): (..., 3, 3) camera intrinsics(s) to crop
        height (int | Tensor): (...) image height(s)
        width (int | Tensor): (...) image width(s)
        cropped_top (int | Tensor): (...) top pixel index of the cropped image(s)
        cropped_left (int | Tensor): (...) left pixel index of the cropped image(s)
        cropped_height (int | Tensor): (...) height of the cropped image(s)
        cropped_width (int | Tensor): (...) width of the cropped image(s)

    ## Returns
        (Tensor): (..., 3, 3) cropped camera intrinsics
    """
    height, width = size.unbind(-1)
    zeros = torch.zeros_like(height)
    ones = torch.ones_like(height)
    transform = torch.stack([
        width / cropped_width, zeros, -cropped_left / cropped_width,
        zeros, height / cropped_height, -cropped_top / cropped_height,
        zeros, zeros, ones
    ]).reshape(*zeros.shape, 3, 3)
    return transform @ intrinsics


def pixel_to_uv(
    pixel: Tensor,
    size: Union[Tuple[Number, Number], Tensor],
    pixel_convention: Literal['integer-corner', 'integer-center'] = 'integer-center',
) -> Tensor:
    """
    ## Parameters
    - `pixel` (Tensor): `(..., 2)` pixel coordinrates 
    - `size` (tuple | Tensor): A tuple `(height, width)` of the image size,
        or an array of shape `(..., 2)` corresponding to the multiple image size(s)
    - `pixel_convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
        - For more definitions, please refer to `pixel_coord_map()`

    ## Returns
        (Tensor): `(..., 2)` uv coordinrates
    """
    if not torch.is_floating_point(pixel):
        pixel = pixel.float()
    if pixel_convention == 'integer-center':
        pixel = pixel + 0.5
    uv = pixel / torch.as_tensor(size, device=pixel.device).flip(-1)
    return uv


def uv_to_pixel(
    uv: Tensor,
    size: Union[Tuple[Number, Number], Tensor],
    pixel_convention: Literal['integer-corner', 'integer-center'] = 'integer-center',
) -> Tensor:
    """
    Convert UV space coordinates to pixel space coordinates.

    ## Parameters
    - `uv` (Tensor): `(..., 2)` uv coordinrates.
    - `size` (tuple | Tensor): A tuple `(height, width)` of the image size,
        or an array of shape `(..., 2)` corresponding to the multiple image size(s)
    - `pixel_convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
        - For more definitions, please refer to `pixel_coord_map()`

    ## Returns
        (Tensor): `(..., 2)` pixel coordinrates
    """
    pixel = uv * torch.as_tensor(size, device=uv.device).flip(-1)
    if pixel_convention == 'integer-center':
        pixel = pixel - 0.5
    return pixel


def pixel_to_ndc(
    pixel: Tensor,
    size: Union[Tuple[Number, Number], Tensor],
    pixel_convention: Literal['integer-corner', 'integer-center'] = 'integer-center',
) -> Tensor:
    """
    Convert pixel coordinates to NDC (Normalized Device Coordinates).

    ## Parameters
    - `pixel` (Tensor): `(..., 2)` pixel coordinrates.
    - `size` (tuple | Tensor): A tuple `(height, width)` of the image size,
        or an array of shape `(..., 2)` corresponding to the multiple image size(s)
    - `pixel_convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
        - For more definitions, please refer to `pixel_coord_map()`

    ## Returns
        (Tensor): `(..., 2)` ndc coordinrates, the range is (-1, 1)
    """
    if not torch.is_floating_point(pixel):
        pixel = pixel.float()
    if pixel_convention == 'integer-center':
        pixel = pixel + 0.5
    ndc = pixel / (torch.as_tensor(size, device=pixel.device).flip(-1) * torch.tensor([2, -2], dtype=pixel.dtype, device=pixel.device)) \
        + torch.tensor([-1, 1], dtype=pixel.dtype, device=pixel.device)
    return ndc


def depth_linear_to_buffer(
    depth: Tensor,
    near: Union[float, Tensor],
    far: Union[float, Tensor]
) -> Tensor:
    """
    Project linear depth to depth value in screen space

    ## Parameters
        depth (Tensor): [...] depth value
        near (float | Tensor): [...] near plane to clip
        far (float | Tensor): [...] far plane to clip

    ## Returns
        (Tensor): [..., 1] depth value in screen space, value ranging in [0, 1]
    """
    return (1 - near / depth) / (1 - near / far)


def depth_buffer_to_linear(
    depth: Tensor,
    near: Union[float, Tensor],
    far: Union[float, Tensor]
) -> Tensor:
    """
    Linearize depth value to linear depth

    ## Parameters
        depth (Tensor): [...] screen depth value, ranging in [0, 1]
        near (float | Tensor): [...] near plane to clip
        far (float | Tensor): [...] far plane to clip

    ## Returns
        (Tensor): [...] linear depth
    """
    return near / (1 - (1 - near / far) * depth)


def project_gl(
    points: Tensor,
    projection: Tensor,
    view: Tensor = None,
) -> Tuple[Tensor, Tensor]:
    """
    Project 3D points to 2D following the OpenGL convention (except for row major matrices)

    ## Parameters
        points (Tensor): [..., N, 3] or [..., N, 4] 3D points to project, if the last 
            dimension is 4, the points are assumed to be in homogeneous coordinates
        view (Tensor): [..., 4, 4] view matrix
        projection (Tensor): [..., 4, 4] projection matrix

    ## Returns
        scr_coord (Tensor): [..., N, 3] screen space coordinates, value ranging in [0, 1].
            The origin (0., 0., 0.) is corresponding to the left & bottom & nearest
        linear_depth (Tensor): [..., N] linear depth
    """
    if points.shape[-1] == 3:
        points = torch.cat([points, torch.ones((*points.shape[:-1], 1), dtype=points.dtype, device=points.device)], dim=-1)
    transform = projection if view is None else projection @ view
    clip_coord = points @ transform.mT
    ndc_coord = clip_coord[..., :3] / clip_coord[..., 3:]
    scr_coord = ndc_coord * 0.5 + 0.5
    linear_depth = clip_coord[..., 3]
    return scr_coord, linear_depth


def project_cv(
    points: Tensor,
    intrinsics: Tensor,
    extrinsics: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Project 3D points to 2D following the OpenCV convention

    ## Parameters
        points (Tensor): [..., N, 3] 3D points
        intrinsics (Tensor): [..., 3, 3] intrinsics matrix
        extrinsics (Tensor): [..., 4, 4] extrinsics matrix

    ## Returns
        uv_coord (Tensor): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        linear_depth (Tensor): [..., N] linear depth
    """
    intrinsics = torch.cat([
        torch.cat([intrinsics, torch.zeros((*intrinsics.shape[:-2], 3, 1), dtype=intrinsics.dtype, device=intrinsics.device)], dim=-1),
        torch.tensor([[0, 0, 0, 1]], dtype=intrinsics.dtype, device=intrinsics.device).expand(*intrinsics.shape[:-2], 1, 4)
    ], dim=-2)
    transform = intrinsics @ extrinsics if extrinsics is not None else intrinsics
    points = torch.cat([points, torch.ones((*points.shape[:-1], 1), dtype=points.dtype, device=points.device)], dim=-1)
    points = points @ transform.mT
    uv_coord = points[..., :2] / points[..., 2:3]
    linear_depth = points[..., 2]
    return uv_coord, linear_depth


def unproject_gl(
    uv: Tensor,
    depth: Tensor,
    projection: Tensor,
    view: Optional[Tensor] = None,
) -> Tensor:
    """
    Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrices)

    ## Parameters
        uv (Tensor): (..., N, 2) screen space XY coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & bottom
        depth (Tensor): (..., N) linear depth values
        projection (Tensor): (..., 4, 4) projection  matrix
        view (Tensor): (..., 4, 4) view matrix
        
    ## Returns
        points (Tensor): (..., N, 3) 3d points
    """
    ndc_xy = uv * 2 - 1
    view_z = -depth
    clip_xy = torch.linalg.inv(projection[..., :2, :2] - ndc_xy[..., :, None] * projection[..., 3:, :2]) \
        @ ((ndc_xy[..., :, None] * projection[..., 3:, 2:] - projection[..., :2, 2:]) \
        @ torch.cat([view_z[..., None, None], torch.ones_like(view_z[..., None, None])], axis=-2))
    points = torch.cat([clip_xy.squeeze(-1), view_z[..., None], torch.ones_like(view_z)[..., None]], axis=-1)
    if view is not None:
        points = points @ torch.linalg.inv(view).mT
    return points[..., :3]
    

def unproject_cv(
    uv: Tensor,
    depth: Tensor,
    intrinsics: Tensor,
    extrinsics: Tensor = None,
) -> Tensor:
    """
    Unproject uv coordinates to 3D view space following the OpenCV convention

    ## Parameters
        uv (Tensor): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        depth (Tensor): [..., N] depth value
        extrinsics (Tensor): [..., 4, 4] extrinsics matrix
        intrinsics (Tensor): [..., 3, 3] intrinsics matrix

    ## Returns
        points (Tensor): [..., N, 3] 3d points
    """
    intrinsics = torch.cat([
        torch.cat([intrinsics, torch.zeros((*intrinsics.shape[:-2], 3, 1), dtype=intrinsics.dtype, device=intrinsics.device)], dim=-1),
        torch.tensor([[0, 0, 0, 1]], dtype=intrinsics.dtype, device=intrinsics.device).expand(*intrinsics.shape[:-2], 1, 4)
    ], dim=-2)
    transform = intrinsics @ extrinsics if extrinsics is not None else intrinsics
    points = torch.cat([uv, torch.ones((*uv.shape[:-1], 1), dtype=uv.dtype, device=uv.device)], dim=-1) * depth[..., None]
    points = torch.cat([points, torch.ones((*points.shape[:-1], 1), dtype=uv.dtype, device=uv.device)], dim=-1)
    points = points @ torch.linalg.inv(transform).mT
    points = points[..., :3]
    return points


def project(
    points: Tensor,
    *,
    intrinsics: Optional[Tensor] = None,
    extrinsics: Optional[Tensor] = None,
    view: Optional[Tensor] = None,
    projection: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    Calculate projection. 
    - For OpenCV convention, use `intrinsics` and `extrinsics` matrices. 
    - For OpenGL convention, use `view` and `projection` matrices.

    ## Parameters

    - `points`: (..., N, 3) 3D world-space points
    - `intrinsics`: (..., 3, 3) intrinsics matrix
    - `extrinsics`: (..., 4, 4) extrinsics matrix
    - `view`: (..., 4, 4) view matrix
    - `projection`: (..., 4, 4) projection matrix

    ## Returns

    - `uv`: (..., N, 2) 2D coordinates. 
        - For OpenCV convention, it is the normalized image coordinate where (0, 0) is the top left corner.
        - For OpenGL convention, it is the screen space XY coordinate where (0, 0) is the bottom left corner.
    - `depth`: (..., N) linear depth values, where `depth > 0` is visible.
        - For OpenCV convention, it is the Z coordinate in camera space.
        - For OpenGL convention, it is the -Z coordinate in camera space.
    """
    assert (intrinsics is not None or extrinsics is not None) ^ (view is not None or projection is not None), \
        "Either camera intrinsics (and extrinsics) or projection (and view) matrices must be provided."
    
    if intrinsics is not None:
        return project_cv(points, intrinsics, extrinsics)
    elif projection is not None:
        return project_gl(points, projection, view)
    else:
        raise ValueError("Invalid combination of input parameters.")


def unproject(
    uv: Tensor,
    depth: Optional[Tensor],
    *,
    intrinsics: Optional[Tensor] = None,
    extrinsics: Optional[Tensor] = None,
    projection: Optional[Tensor] = None,
    view: Optional[Tensor] = None,
) -> Tensor:
    """
    Calculate inverse projection. 
    - For OpenCV convention, use `intrinsics` and `extrinsics` matrices. 
    - For OpenGL convention, use `view` and `projection` matrices.

    ## Parameters

    - `uv`: (..., N, 2) 2D coordinates. 
        - For OpenCV convention, it is the normalized image coordinate where (0, 0) is the top left corner.
        - For OpenGL convention, it is the screen space XY coordinate where (0, 0) is the bottom left corner.
    - `depth`: (..., N) linear depth values, where `depth > 0` is visible.
        - For OpenCV convention, it is the Z coordinate in camera space.
        - For OpenGL convention, it is the -Z coordinate in camera space.
    - `intrinsics`: (..., 3, 3) intrinsics matrix
    - `extrinsics`: (..., 4, 4) extrinsics matrix
    - `view`: (..., 4, 4) view matrix
    - `projection`: (..., 4, 4) projection matrix

    ## Returns

    - `points`: (..., N, 3) 3D world-space points
    """
    assert (intrinsics is not None or extrinsics is not None) ^ (view is not None or projection is not None), \
        "Either camera intrinsics (and extrinsics) or projection (and view) matrices must be provided."
    
    if intrinsics is not None:
        return unproject_cv(uv, depth, intrinsics, extrinsics)
    elif projection is not None:
        return unproject_gl(uv, depth, projection, view)
    else:
        raise ValueError("Invalid combination of input parameters.")


def euler_axis_angle_rotation(axis: str, angle: Tensor) -> Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    ## Parameters
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    ## Returns
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: Tensor, convention: str = 'XYZ') -> Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    ## Parameters
        euler_angles: Euler angles in radians as tensor of shape (..., 3), XYZ
        convention: permutation of "X", "Y" or "Z", representing the order of Euler rotations to apply.

    ## Returns
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        euler_axis_angle_rotation(c, euler_angles[..., 'XYZ'.index(c)])
        for c in convention
    ]
    # return functools.reduce(torch.matmul, matrices)
    return matrices[2] @ matrices[1] @ matrices[0]


def skew_symmetric(v: Tensor):
    "Skew symmetric matrix from a 3D vector"
    assert v.shape[-1] == 3, "v must be 3D"
    x, y, z = v.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros,
    ], dim=-1).reshape(*v.shape[:-1], 3, 3)


def rotation_matrix_from_vectors(v1: Tensor, v2: Tensor):
    "Rotation matrix that rotates v1 to v2"
    I = torch.eye(3).to(v1)
    v1 = F.normalize(v1, dim=-1)
    v2 = F.normalize(v2, dim=-1)
    v = torch.cross(v1, v2, dim=-1)
    c = torch.sum(v1 * v2, dim=-1)
    K = skew_symmetric(v)
    R = I + K + (1 / (1 + c))[None, None] * (K @ K)
    return R


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    ## Parameters
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    ## Returns
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def matrix_to_euler_angles(matrix: Tensor, convention: str) -> Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.
    NOTE: The composition order eg. `XYZ` means `Rz * Ry * Rx` (like blender), instead of `Rx * Ry * Rz` (like pytorch3d)

    ## Parameters
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    ## Returns
        Euler angles in radians as tensor of shape (..., 3), in the order of XYZ (like blender), instead of convention (like pytorch3d)
    """
    if not all(c in 'XYZ' for c in convention) or not all(c in convention for c in 'XYZ'):
        raise ValueError(f"Invalid convention {convention}.")
    if not matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    
    i0 = 'XYZ'.index(convention[0])
    i2 = 'XYZ'.index(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(matrix[..., i2, i0] * (-1.0 if i2 - i0 in [-1, 2] else 1.0))
    else:
        central_angle = torch.acos(matrix[..., i2, i2])

    # Angles in composition order
    o = [
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2, :], True, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0], False, tait_bryan
        ),
    ]
    return torch.stack([o[convention.index(c)] for c in 'XYZ'], -1)


def axis_angle_to_matrix(axis_angle: Tensor, eps: float = 1e-12) -> Tensor:
    """Convert axis-angle representation (rotation vector) to rotation matrix, whose direction is the axis of rotation and length is the angle of rotation

    ## Parameters
        axis_angle (Tensor): shape (..., 3), axis-angle vcetors

    ## Returns
        Tensor: shape (..., 3, 3) The rotation matrices for the given axis-angle parameters
    """
    batch_shape = axis_angle.shape[:-1]
    device, dtype = axis_angle.device, axis_angle.dtype

    angle = torch.norm(axis_angle + eps, dim=-1, keepdim=True)
    axis = axis_angle / angle

    cos = torch.cos(angle)[..., None, :]
    sin = torch.sin(angle)[..., None, :]

    rx, ry, rz = axis.unbind(dim=-1)
    zeros = torch.zeros(batch_shape, dtype=dtype, device=device)
    K = torch.stack([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1).view((*batch_shape, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device)
    rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)
    return rot_mat


def matrix_to_axis_angle(rot_mat: Tensor, eps: float = 1e-12) -> Tensor:
    """Convert a batch of 3x3 rotation matrices to axis-angle representation (rotation vector)

    ## Parameters
        rot_mat (Tensor): shape (..., 3, 3), the rotation matrices to convert

    ## Returns
        Tensor: shape (..., 3), the axis-angle vectors corresponding to the given rotation matrices
    """
    quat = matrix_to_quaternion(rot_mat)
    axis_angle = quaternion_to_axis_angle(quat, eps=eps)
    return axis_angle


def quaternion_to_axis_angle(quaternion: Tensor, eps: float = 1e-12) -> Tensor:
    """Convert a batch of quaternions (w, x, y, z) to axis-angle representation (rotation vector)

    ## Parameters
        quaternion (Tensor): shape (..., 4), the quaternions to convert

    ## Returns
        Tensor: shape (..., 3), the axis-angle vectors corresponding to the given quaternions
    """
    assert quaternion.shape[-1] == 4
    norm = torch.norm(quaternion[..., 1:], dim=-1, keepdim=True)
    axis = quaternion[..., 1:] / norm.clamp(min=eps)
    angle = 2 * torch.atan2(norm, quaternion[..., 0:1])
    return angle * axis


def axis_angle_to_quaternion(axis_angle: Tensor, eps: float = 1e-12) -> Tensor:
    """Convert axis-angle representation (rotation vector) to quaternion (w, x, y, z)

    ## Parameters
        axis_angle (Tensor): shape (..., 3), axis-angle vcetors

    ## Returns
        Tensor: shape (..., 4) The quaternions for the given axis-angle parameters
    """
    axis = F.normalize(axis_angle, dim=-1, eps=eps)
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    quat = torch.cat([torch.cos(angle / 2), torch.sin(angle / 2) * axis], dim=-1)
    return quat


def matrix_to_quaternion(rot_mat: Tensor, eps: float = 1e-12) -> Tensor:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)

    ## Parameters
        rot_mat (Tensor): shape (..., 3, 3), the rotation matrices to convert

    ## Returns
        Tensor: shape (..., 4), the quaternions corresponding to the given rotation matrices
    """
    # Extract the diagonal and off-diagonal elements of the rotation matrix
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = rot_mat.flatten(-2).unbind(dim=-1)

    diag = torch.diagonal(rot_mat, dim1=-2, dim2=-1)
    M = torch.tensor([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=rot_mat.dtype, device=rot_mat.device)
    wxyz = (1 + diag @ M.transpose(-1, -2)).clamp_(0).sqrt().mul(0.5)
    _, max_idx = wxyz.max(dim=-1)
    xw = torch.sign(m21 - m12)
    yw = torch.sign(m02 - m20)
    zw = torch.sign(m10 - m01)
    yz = torch.sign(m21 + m12)
    xz = torch.sign(m02 + m20)
    xy = torch.sign(m01 + m10)
    ones = torch.ones_like(xw)
    sign = torch.where(
        max_idx[..., None] == 0,
        torch.stack([ones, xw, yw, zw], dim=-1),
        torch.where(
            max_idx[..., None] == 1,
            torch.stack([xw, ones, xy, xz], dim=-1),
            torch.where(
                max_idx[..., None] == 2,
                torch.stack([yw, xy, ones, yz], dim=-1),
                torch.stack([zw, xz, yz, ones], dim=-1)
            )
        )
    )
    quat = sign * wxyz
    quat = F.normalize(quat, dim=-1, eps=eps)
    return quat


def quaternion_to_matrix(quaternion: Tensor, eps: float = 1e-12) -> Tensor:
    """Converts a batch of quaternions (w, x, y, z) to rotation matrices
    
    ## Parameters
        quaternion (Tensor): shape (..., 4), the quaternions to convert
    
    ## Returns
        Tensor: shape (..., 3, 3), the rotation matrices corresponding to the given quaternions
    """
    assert quaternion.shape[-1] == 4
    quaternion = quaternion_normalize(quaternion, eps=eps)
    w, x, y, z = quaternion.unbind(dim=-1)
    zeros = torch.zeros_like(w)
    I = torch.eye(3, dtype=quaternion.dtype, device=quaternion.device)
    xyz = quaternion[..., 1:]
    A = xyz[..., :, None] * xyz[..., None, :] - I * torch.square(xyz).sum(dim=-1)[..., None, None]
    B = torch.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros
    ], dim=-1).unflatten(-1, (3, 3))
    rot_mat = I + 2 * (A + w[..., None, None] * B)
    return rot_mat


def quaternion_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Multiplies two quaternions (w, x, y, z)

    Parameters
    ----
        q1 (Tensor): shape (..., 4), the first quaternion
        q2 (Tensor): shape (..., 4), the second quaternion

    Returns
    ----
        Tensor: shape (..., 4), the product of the two quaternions
    """
    w1, v1 = q1[..., :1], q1[..., 1:]
    w2, v2 = q2[..., :1], q2[..., 1:]
    res_w = w1 * w2 - (v1 * v2).sum(dim=-1, keepdim=True)
    res_v = w1 * v2 + w2 * v1 + torch.cross(v1, v2, dim=-1)
    res_q = torch.cat([res_w, res_v], dim=-1)
    return res_q


def quaternion_inverse(quaternion: Tensor) -> Tensor:
    """Calculate the inverse of a batch of quaternions (w, x, y, z)

    Parameters
    ----
        quaternion (Tensor): shape (..., 4), the quaternions to invert

    Returns
    ----
        Tensor: shape (..., 4), no normalization applied. It depends on the input quaternion.
    """
    w, v = quaternion[..., 0:1], quaternion[..., 1:]
    inv_quat = torch.cat([w, -v], dim=-1)
    return inv_quat


def quaternion_normalize(quaternion: Tensor, eps: float = 1e-12) -> Tensor:
    """Normalize quaternions (w, x, y, z) to unit length and positive w component
    
    Parameters
    ----
        quaternion (Tensor): shape (..., 4), the quaternions to normalize

    Returns
    ----
        Tensor: shape (..., 4), the normalized quaternions with unit length and positive w component
    """
    w_sign = torch.where(quaternion[..., 0] >= 0, torch.tensor(1.0, device=quaternion.device, dtype=quaternion.dtype), torch.tensor(-1.0, device=quaternion.device, dtype=quaternion.dtype))
    quaternion = w_sign[..., None] * F.normalize(quaternion, dim=-1, eps=eps)
    return quaternion


def random_rotation_matrix(*size: int, dtype=torch.float32, device: torch.device = None) -> Tensor:
    """
    Generate random 3D rotation matrix.

    ## Parameters
        dtype: The data type of the output rotation matrix.

    ## Returns
        Tensor: `(*size, 3, 3)` random rotation matrix.
    """
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = size[0]
    rand_quat = torch.randn((*size, 4), dtype=dtype, device=device)
    return quaternion_to_matrix(rand_quat)


@totensor(_others=torch.float32)
@batched(1, 1, 1)
def lerp(v1: torch.Tensor, v2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation between two vectors.

    ## Parameters
    - `v1` (Tensor): `(..., D)` vector 1
    - `v2` (Tensor): `(..., D)` vector 2
    - `t` (Tensor): `(..., N)` interpolation parameter in [0, 1]

    ## Returns
        Tensor: `(..., N, D)` interpolated vector
    """
    return v1[..., None, :] + t[..., None] * (v2 - v1)[..., None, :]


@totensor(_others=torch.float32)
@batched(1, 1, 1)
def slerp(v1: Tensor, v2: Tensor, t: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Spherical linear interpolation between two (unit) vectors. 

    ## Parameters
        `v1` (Tensor): `(..., D)` (unit) vector 1
        `v2` (Tensor): `(..., D)` (unit) vector 2
        `t` (Tensor): `(..., N)` interpolation parameter in [0, 1]

    ## Returns
        Tensor: `(..., N, D)` interpolated unit vector
    """
    v1 = F.normalize(v1, dim=-1, eps=eps)
    v2 = F.normalize(v2, dim=-1, eps=eps)
    cos = torch.sum(v1 * v2, dim=-1)
    v_ortho1 = v2 - v1 * cos[..., None]
    v_ortho2 = v1 - v2 * cos[..., None]
    sin = torch.minimum(v_ortho1.norm(dim=-1), v_ortho2.norm(dim=-1))
    theta = torch.atan2(sin + eps, cos) * t
    v_ortho1 = F.normalize(v_ortho1, dim=-1, eps=eps)
    v = v1[..., None, :] * torch.cos(theta)[..., None] + v_ortho1[..., None, :] * torch.sin(theta)[..., None]
    return v


@totensor(_others=torch.float32)
@batched(2, 2, 1)
def slerp_rotation_matrix(R1: Tensor, R2: Tensor, t: Union[Number, Tensor]) -> Tensor:
    """Spherical linear interpolation between two 3D rotation matrices

    ## Parameters
        R1 (Tensor): shape (..., 3, 3), the first rotation matrix
        R2 (Tensor): shape (..., 3, 3), the second rotation matrix
        t (Tensor): scalar or shape (..., N), the interpolation factor

    ## Returns
        Tensor: shape (..., N, 3, 3), the interpolated rotation matrix
    """
    assert R1.shape[-2:] == (3, 3) and R2.shape[-2:] == (3, 3)
    quat1 = matrix_to_quaternion(R1)
    quat2 = matrix_to_quaternion(R2)
    slerped_quat = slerp(quat1, quat2, t)
    return quaternion_to_matrix(slerped_quat)


@totensor(_others=torch.float32)
@batched(2, 2, 1)
def interpolate_se3_matrix(T1: Tensor, T2: Tensor, t: Tensor):
    """Interpolate between two SE(3) transformation matrices.
    - Spherical linear interpolation (SLERP) is used for the rotational part.
    - Linear interpolation is used for the translational part.

    ## Parameters
    - `T1` (Tensor): (..., 4, 4) SE(3) matrix 1
    - `T2` (Tensor): (..., 4, 4) SE(3) matrix 2
    - `t` (Tensor): (..., N) interpolation parameter in [0, 1]

    ## Returns
        Tensor: (..., N, 4, 4) interpolated SE(3) matrix
    """
    assert T1.shape[-2:] == (4, 4) and T2.shape[-2:] == (4, 4)
    pos = lerp(T1[..., :3, 3], T2[..., :3, 3], t)
    rot = slerp_rotation_matrix(T1[..., :3, :3], T2[..., :3, :3], t)
    transform = make_affine_matrix(rot, pos)
    return transform


def extrinsics_to_essential(extrinsics: Tensor):
    """
    extrinsics matrix `[[R, t] [0, 0, 0, 1]]` such that `x' = R (x - t)` to essential matrix such that `x' E x = 0`

    ## Parameters
        extrinsics (Tensor): [..., 4, 4] extrinsics matrix

    ## Returns
        (Tensor): [..., 3, 3] essential matrix
    """
    assert extrinsics.shape[-2:] == (4, 4)
    R = extrinsics[..., :3, :3]
    t = extrinsics[..., :3, 3]
    zeros = torch.zeros_like(t)
    t_x = torch.stack([
        zeros, -t[..., 2], t[..., 1],
        t[..., 2], zeros, -t[..., 0],
        -t[..., 1], t[..., 0], zeros
    ]).reshape(*t.shape[:-1], 3, 3)
    return R @ t_x


@batched(2, 1)
def make_affine_matrix(M: Tensor, t: Tensor):
    """
    Make an affine transformation matrix from a linear matrix and a translation vector.

    ## Parameters
        M (Tensor): [..., D, D] linear matrix (rotation, scaling or general deformation)
        t (Tensor): [..., D] translation vector

    ## Returns
        Tensor: [..., D + 1, D + 1] affine transformation matrix
    """
    return torch.cat([
        torch.cat([M, t[..., None]], dim=-1),
        torch.tensor([0, 0, 0, 1], dtype=M.dtype, device=M.device).expand(*M.shape[:-2], 1, 4)
    ], dim=-2)


def rotation_matrix_2d(theta: Union[float, Tensor]):
    """
    2x2 matrix for 2D rotation

    ## Parameters
        theta (float | Tensor): rotation angle in radians, arbitrary shape (...,)

    ## Returns
        (Tensor): (..., 2, 2) rotation matrix
    """
    if isinstance(theta, float):
        theta = torch.tensor(theta)
    return torch.stack([
        torch.cos(theta), -torch.sin(theta),
        torch.sin(theta), torch.cos(theta),
    ], dim=-1).unflatten(-1, (2, 2))


def rotate_2d(theta: Union[float, Tensor], center: Tensor = None):
    """
    3x3 matrix for 2D rotation around a center
    ```
       [[Rxx, Rxy, tx],
        [Ryx, Ryy, ty],
        [0,     0,  1]]
    ```
    ## Parameters
        theta (float | Tensor): rotation angle in radians, arbitrary shape (...,)
        center (Tensor): rotation center, arbitrary shape (..., 2). Default to (0, 0)
        
    ## Returns
        (Tensor): (..., 3, 3) transformation matrix
    """
    if isinstance(theta, float):
        theta = torch.tensor(theta)
        if center is not None:
            theta = theta.to(center)
    if center is None:
        center = torch.zeros(2).to(theta).expand(*theta.shape, -1)
    R = rotation_matrix_2d(theta)
    return torch.cat([
        torch.cat([
            R, 
            center[..., :, None] - R @ center[..., :, None],
        ], dim=-1),
        torch.tensor([[0, 0, 1]], dtype=center.dtype, device=center.device).expand(*center.shape[:-1], -1, -1),
    ], dim=-2)


def translate_2d(translation: Tensor):
    """
    Translation matrix for 2D translation
    ```
       [[1, 0, tx],
        [0, 1, ty],
        [0, 0,  1]]
    ```
    ## Parameters
        translation (Tensor): translation vector, arbitrary shape (..., 2)
    
    ## Returns
        (Tensor): (..., 3, 3) transformation matrix
    """
    return torch.cat([
        torch.cat([
            torch.eye(2, dtype=translation.dtype, device=translation.device).expand(*translation.shape[:-1], -1, -1),
            translation[..., None],
        ], dim=-1),
        torch.tensor([[0, 0, 1]], dtype=translation.dtype, device=translation.device).expand(*translation.shape[:-1], -1, -1),
    ], dim=-2)


def scale_2d(scale: Union[float, Tensor], center: Tensor = None):
    """
    Scale matrix for 2D scaling
    ```
       [[s, 0, tx],
        [0, s, ty],
        [0, 0,  1]]
    ```
    ## Parameters
        scale (float | Tensor): scale factor, arbitrary shape (...,)
        center (Tensor): scale center, arbitrary shape (..., 2). Default to (0, 0)

    ## Returns
        (Tensor): (..., 3, 3) transformation matrix
    """
    if isinstance(scale, float):
        scale = torch.tensor(scale)
        if center is not None:
            scale = scale.to(center)
    if center is None:
        center = torch.zeros(2, dtype=scale.dtype, device=scale.device).expand(*scale.shape, -1)
    return torch.cat([
        torch.cat([
            scale * torch.eye(2, dtype=scale.dtype, device=scale.device).expand(*scale.shape[:-1], -1, -1),
            center[..., :, None] - center[..., :, None] * scale[..., None, None],
        ], dim=-1),
        torch.tensor([[0, 0, 1]], dtype=scale.dtype, device=scale.device).expand(*center.shape[:-1], -1, -1),
    ], dim=-2)


def transform_points(x: Tensor, *Ts: Tensor) -> Tensor:
    """
    Apply transformation(s) to a point or a set of points.
    It is like `(Tn @ ... @ T2 @ T1 @ x[:, None]).squeeze(0)`, but: 
    1. Automatically handle the homogeneous coordinate;
            - x will be padded with homogeneous coordinate 1.
            - Each T will be padded by identity matrix to match the dimension. 
    2. Using efficient contraction path when array sizes are large, based on `einsum`.

    ## Parameters
    - `x`: Tensor, shape `(..., D)`: the points to be transformed.
    - `Ts`: Tensor, shape `(..., D + 1, D + 1)`: the affine transformation matrix (matrices)
        If more than one transformation is given, they will be applied in corresponding order.
    ## Returns
    - `y`: Tensor, shape `(..., D)`: the transformed point or a set of points.

    ## Example Usage
    
    - Just linear transformation

        ```
        y = transform(x_3, mat_3x3) 
        ```

    - Affine transformation

        ```
        y = transform(x_3, mat_3x4)
        ```

    - Chain multiple transformations

        ```
        y = transform(x_3, T1_4x4, T2_3x4, T3_3x4)
        ```
    """
    input_dim = x.shape[-1]
    pad_dim = max(max(max(T.shape[-2:]) for T in Ts), x.shape[-1])
    x = torch.cat([x, torch.ones((*x.shape[:-1], pad_dim - x.shape[-1]), dtype=x.dtype, device=x.device)], dim=-1)
    I = torch.eye(pad_dim, dtype=x.dtype, device=x.device)
    Ts = [
        torch.cat([
            torch.cat([T, I[:T.shape[-2], T.shape[-1]:].expand(*T.shape[:-2], -1, -1)], dim=-1),
            I[T.shape[-2]:, :].expand(*T.shape[:-2], -1, -1)
        ], dim=-2)
        for T in Ts
    ]
    total_numel = sum(t.numel() for t in Ts) + x.numel()
    if total_numel > 1000:
        # Only use einsum when the total number of elements is large enough to benefit from optimized contraction path
        operands = [*reversed(Ts), x[..., None]]
        offset = len(operands) + 1
        batch_shape = torch.broadcast_shapes(*(m.shape[:-2] for m in operands))
        batch_subscripts = tuple(range(offset, offset + len(batch_shape)))
        # Broadcasted size 1 dimensions can be squeezed to avoid redundant broadcasting in einsum
        subscripts, squeezed_operands = [], []
        for i, m in enumerate(operands):
            squeezable = tuple(b_m == 1 and b > 1 for b_m, b in zip(m.shape[:-2], batch_shape[len(batch_shape) - (m.ndim - 2):]))
            squeezed_operands.append(
                m.squeeze(axis=tuple(j for j, s in enumerate(squeezable) if s))
            )
            subscripts.append(
                (*tuple(j for j, s in zip(batch_subscripts[len(batch_shape) - (m.ndim - 2):], squeezable) if not s), i, i + 1)
            )
        y = torch.einsum(
            *itertools.chain(*zip(squeezed_operands, subscripts)), 
            (*range(offset, offset + len(batch_shape)), 0, len(operands)), 
        )
        y = y.squeeze(-1)
    else:
        y = x[..., None]
        for T in Ts:
            y = T @ y
        y = y.squeeze(-1)
    return y[..., :input_dim]


def angle_between(v1: Tensor, v2: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Calculate the angle between two (batches of) vectors.
    Better precision than using the arccos dot product directly.

    ## Parameters
    - `v1`: Tensor, shape (..., D): the first vector.
    - `v2`: Tensor, shape (..., D): the second vector.
    - `eps`: float, optional: prevents zero angle difference (indifferentiable).

    ## Returns
    `angle`: Tensor, shape (...): the angle between the two vectors.
    """
    v1 = F.normalize(v1, dim=-1, eps=eps)
    v2 = F.normalize(v2, dim=-1, eps=eps)
    cos = (v1 * v2).sum(dim=-1)
    sin = torch.minimum((v2 - v1 * cos[..., None]).norm(dim=-1), (v1 - v2 * cos[..., None]).norm(dim=-1))
    return torch.atan2(sin + eps, cos)

    