import numpy as np
from numpy import ndarray
from typing import *
import itertools
from numbers import Number

from .helpers import toarray, batched
from ..helpers import no_warnings
from .utils import lite_dot, lite_norm, lite_sum


__all__ = [
    'perspective_from_fov',
    'perspective_from_window',
    'intrinsics_from_fov',
    'intrinsics_from_focal_center',
    'fov_to_focal',
    'focal_to_fov',
    'intrinsics_to_fov',
    'view_look_at',
    'extrinsics_look_at',
    'perspective_to_intrinsics',
    'perspective_to_near_far',
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
    'unproject_cv',
    'unproject_gl',
    'project_cv',
    'project_gl',
    'project',
    'unproject',
    'screen_coord_to_view_coord',
    'quaternion_to_matrix',
    'quaternion_multiply',
    'quaternion_inverse',
    'quaternion_normalize',
    'axis_angle_to_matrix',
    'matrix_to_quaternion',
    'extrinsics_to_essential',
    'axis_angle_to_quaternion',
    'euler_axis_angle_rotation',
    'euler_angles_to_matrix',
    'matrix_to_axis_angle',
    'matrix_to_euler_angles',
    'quaternion_to_axis_angle',
    'skew_symmetric',
    'rotation_matrix_from_vectors',
    'ray_intersection',
    'make_affine_matrix',
    'random_rotation_matrix',
    'lerp',
    'slerp',
    'slerp_rotation_matrix',
    'interpolate_se3_matrix',
    'piecewise_lerp',
    'piecewise_interpolate_se3_matrix',
    'transform_points',
    'angle_between',
]


@toarray(_others=np.float32)
@batched(_others=0)
def perspective_from_fov(
    *,
    fov_x: Optional[Union[float, ndarray]] = None,
    fov_y: Optional[Union[float, ndarray]] = None,
    fov_min: Optional[Union[float, ndarray]] = None,
    fov_max: Optional[Union[float, ndarray]] = None,
    aspect_ratio: Optional[Union[float, ndarray]] = None,
    near: Optional[Union[float, ndarray]],
    far: Optional[Union[float, ndarray]],
) -> ndarray:
    """
    Get OpenGL perspective matrix from field of view 

    ## Returns
        (ndarray): [..., 4, 4] perspective matrix
    """
    if fov_max is not None:
        fx = np.maximum(1, 1 / aspect_ratio) / np.tan(fov_max / 2)
        fy = np.maximum(1, aspect_ratio) / np.tan(fov_max / 2)
    elif fov_min is not None:
        fx = np.minimum(1, 1 / aspect_ratio) / np.tan(fov_min / 2)
        fy = np.minimum(1, aspect_ratio) / np.tan(fov_min / 2)
    elif fov_x is not None and fov_y is not None:
        fx = 1 / np.tan(fov_x / 2)
        fy = 1 / np.tan(fov_y / 2)
    elif fov_x is not None:
        fx = 1 / np.tan(fov_x / 2)
        fy = fx * aspect_ratio
    elif fov_y is not None:
        fy = 1 / np.tan(fov_y / 2)
        fx = fy / aspect_ratio
    perspective = np.zeros((fx.shape[0], 4, 4), dtype=fx.dtype)
    perspective[:, 0, 0] = fx
    perspective[:, 1, 1] = fy
    perspective[:, 2, 2] = (near / far + 1) / (near / far - 1)
    perspective[:, 2, 3] = 2. * near / (near / far - 1)
    perspective[:, 3, 2] = -1.
    return perspective


@toarray(_others=np.float32)
@batched(_others=0)
def perspective_from_window(
    left: Union[float, ndarray],
    right: Union[float, ndarray],
    bottom: Union[float, ndarray],
    top: Union[float, ndarray],
    near: Union[float, ndarray],
    far: Union[float, ndarray]
) -> ndarray:
    """
    Get OpenGL perspective matrix from the window of z=-1 projection plane

    ## Returns
        (ndarray): [..., 4, 4] perspective matrix
    """
    perspective = np.zeros((left.shape[0], 4, 4), dtype=left.dtype)
    perspective[:, 0, 0] = 2 / (right - left)
    perspective[:, 0, 2] = (right + left) / (right - left)
    perspective[:, 1, 1] = 2 / (top - bottom)
    perspective[:, 1, 2] = (top + bottom) / (top - bottom)
    perspective[:, 2, 2] = (near / far + 1) / (near / far - 1)
    perspective[:, 2, 3] = 2. * near / (near / far - 1)
    perspective[:, 3, 2] = -1.
    return perspective


@toarray(_others=np.float32)
@batched(_others=0)
def intrinsics_from_focal_center(
    fx: Union[float, ndarray],
    fy: Union[float, ndarray],
    cx: Union[float, ndarray],
    cy: Union[float, ndarray],
) -> ndarray:
    """
    Get OpenCV intrinsics matrix

    ## Returns
        (ndarray): [..., 3, 3] OpenCV intrinsics matrix
    """
    if any(isinstance(x, ndarray) for x in (fx, fy, cx, cy)):
        dtype = np.result_type(fx, fy, cx, cy)
    fx, fy, cx, cy = np.broadcast_arrays(fx, fy, cx, cy)
    ret = np.zeros((*fx.shape, 3, 3), dtype=dtype)
    ret[..., 0, 0] = fx
    ret[..., 1, 1] = fy
    ret[..., 0, 2] = cx
    ret[..., 1, 2] = cy
    ret[..., 2, 2] = 1.
    return ret


@toarray(_others=np.float32)
@batched(_others=0)
def intrinsics_from_fov(
    *,
    fov_x: Optional[Union[float, ndarray]] = None,
    fov_y: Optional[Union[float, ndarray]] = None,
    fov_max: Optional[Union[float, ndarray]] = None,
    fov_min: Optional[Union[float, ndarray]] = None,
    cx: Union[float, ndarray] = 0.5,
    cy: Union[float, ndarray] = 0.5,
    aspect_ratio: Optional[Union[float, ndarray]] = None,
) -> ndarray:
    """
    Get normalized OpenCV intrinsics matrix from given field of view.
    You can provide either fov_x, fov_y, fov_max or fov_min and aspect_ratio

    Parameters
    ----
        fov_x (float | ndarray): field of view in x axis
        fov_y (float | ndarray): field of view in y axis
        fov_max (float | ndarray): field of view in largest dimension
        fov_min (float | ndarray): field of view in smallest dimension
        cx (float | ndarray): principal point x coordinate
        cy (float | ndarray): principal point y coordinate
        aspect_ratio (float | ndarray): aspect ratio of the image

    Returns
    ----
        (ndarray): [..., 3, 3] OpenCV intrinsics matrix
    """
    if fov_max is not None:
        fx = np.maximum(1, 1 / aspect_ratio) / (2 * np.tan(fov_max / 2))
        fy = np.maximum(1, aspect_ratio) / (2 * np.tan(fov_max / 2))
    elif fov_min is not None:
        fx = np.minimum(1, 1 / aspect_ratio) / (2 * np.tan(fov_min / 2))
        fy = np.minimum(1, aspect_ratio) / (2 * np.tan(fov_min / 2))
    elif fov_x is not None and fov_y is not None:
        fx = 1 / (2 * np.tan(fov_x / 2))
        fy = 1 / (2 * np.tan(fov_y / 2))
    elif fov_x is not None:
        fx = 1 / (2 * np.tan(fov_x / 2))
        fy = fx * aspect_ratio
    elif fov_y is not None:
        fy = 1 / (2 * np.tan(fov_y / 2))
        fx = fy / aspect_ratio
    ret = intrinsics_from_focal_center(fx, fy, cx, cy)
    return ret


def focal_to_fov(focal: ndarray):
    return 2 * np.arctan(0.5 / focal)


def fov_to_focal(fov: ndarray):
    return 0.5 / np.tan(fov / 2)


def intrinsics_to_fov(intrinsics: ndarray) -> Tuple[ndarray, ndarray]:
    fov_x = focal_to_fov(intrinsics[..., 0, 0])
    fov_y = focal_to_fov(intrinsics[..., 1, 1])
    return fov_x, fov_y


@toarray(_others=np.float32)
@batched(_others=1)
def view_look_at(
    eye: ndarray,
    look_at: ndarray,
    up: ndarray
) -> ndarray:
    """
    Get OpenGL view matrix looking at something

    ## Parameters
        eye (ndarray): [..., 3] the eye position
        look_at (ndarray): [..., 3] the position to look at
        up (ndarray): [..., 3] head up direction (y axis in screen space). Not necessarily othogonal to view direction

    ## Returns
        (ndarray): [..., 4, 4], view matrix
    """
    z = eye - look_at
    x = np.cross(up, z)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=-2)
    R = R / lite_norm(R, axis=-1)[..., None]
    t = (-R @ eye[..., None]).squeeze(-1)
    return make_affine_matrix(R, t)


@toarray(_others=np.float32)
@batched(_others=1)
def extrinsics_look_at(
    eye: ndarray,
    look_at: ndarray,
    up: ndarray
) -> ndarray:
    """
    Get OpenCV extrinsics matrix looking at something

    ## Parameters
        eye (ndarray): [..., 3] the eye position
        look_at (ndarray): [..., 3] the position to look at
        up (ndarray): [..., 3] head up direction (-y axis in screen space). Not necessarily othogonal to view direction

    ## Returns
        (ndarray): [..., 4, 4], extrinsics matrix
    """
    z = look_at - eye
    x = np.cross(-up, z)
    y = np.cross(z, x)
    # x = np.cross(y, z)
    x = x / lite_norm(x, axis=-1)[..., None]
    y = y / lite_norm(y, axis=-1)[..., None]
    z = z / lite_norm(z, axis=-1)[..., None]
    R = np.stack([x, y, z], axis=-2)
    t = -np.matmul(R, eye[..., None])
    return np.concatenate([
        np.concatenate([R, t], axis=-1),
        np.array([[[0., 0., 0., 1.]]], dtype=eye.dtype).repeat(eye.shape[0], axis=0)
    ], axis=-2)


def perspective_to_intrinsics(perspective: ndarray) -> ndarray:
    """
    OpenGL perspective matrix to OpenCV intrinsics

    ## Parameters
        perspective (ndarray): [..., 4, 4] OpenGL perspective matrix

    ## Returns
        (ndarray): shape [..., 3, 3] OpenCV intrinsics
    """
    assert np.allclose(perspective[:, [0, 1, 3], 3], 0), "The matrix is not a perspective projection matrix"
    ret = np.array([[0.5, 0., 0.5], [0., -0.5, 0.5], [0., 0., 1.]], dtype=perspective.dtype) \
        @ perspective[..., [0, 1, 3], :3] \
        @ np.diag(np.array([1, -1, -1], dtype=perspective.dtype))
    return ret


def perspective_to_near_far(perspective: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Get near and far planes from OpenGL perspective matrix

    ## Parameters
    """
    a, b = perspective[..., 2, 2], perspective[..., 2, 3]
    near, far =  b / (a - 1), b / (a + 1)
    return near, far


@toarray(None, _others='intrinsics')
@batched(2, 0, 0)
def intrinsics_to_perspective(
    intrinsics: ndarray,
    near: Union[float, ndarray],
    far: Union[float, ndarray],
) -> ndarray:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    ## Parameters
        intrinsics (ndarray): [..., 3, 3] OpenCV intrinsics matrix
        near (float | ndarray): [...] near plane to clip
        far (float | ndarray): [...] far plane to clip
    ## Returns
        (ndarray): [..., 4, 4] OpenGL perspective matrix
    """
    perspective = np.zeros((intrinsics.shape[0], 4, 4), dtype=intrinsics.dtype)
    perspective[..., [0, 1, 3], :3] = np.array([[2, 0, -1], [0, -2, 1], [0, 0, 1]], dtype=intrinsics.dtype) \
        @ intrinsics \
        @ np.diag(np.array([1, -1, -1], dtype=intrinsics.dtype))
    perspective[:, 2, 2] = (near / far + 1) / (near / far - 1)
    perspective[:, 2, 3] = 2. * near / (near / far - 1)
    perspective[:, 3, 2] = -1.
    return perspective


def extrinsics_to_view(extrinsics: ndarray) -> ndarray:
    """
    OpenCV camera extrinsics to OpenGL view matrix

    ## Parameters
        extrinsics (ndarray): [..., 4, 4] OpenCV camera extrinsics matrix

    ## Returns
        (ndarray): [..., 4, 4] OpenGL view matrix
    """
    return extrinsics * np.array([1, -1, -1, 1], dtype=extrinsics.dtype)[:, None]


def view_to_extrinsics(view: ndarray) -> ndarray:
    """
    OpenGL view matrix to OpenCV camera extrinsics

    ## Parameters
        view (ndarray): [..., 4, 4] OpenGL view matrix

    ## Returns
        (ndarray): [..., 4, 4] OpenCV camera extrinsics matrix
    """
    return view * np.array([1, -1, -1, 1], dtype=view.dtype)[:, None]


@toarray(None, 'intrinsics')
@batched(2, 1)
def normalize_intrinsics(
    intrinsics: ndarray,
    size: Union[Tuple[Number, Number], ndarray],
    pixel_convention: Literal['integer-center', 'integer-corner'] = 'integer-center',
) -> ndarray:
    """
    Normalize intrinsics from pixel cooridnates to uv coordinates

    ## Parameters
    - `intrinsics` (ndarray): `(..., 3, 3)` camera intrinsics to normalize
    - `size` (tuple | ndarray): A tuple `(height, width)` of the image size,
        or an array of shape `(..., 2)` corresponding to the multiple image size(s)
    - `pixel_convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
        - For more definitions, please refer to `pixel_coord_map()`

    ## Returns
        `(ndarray)`: `(..., 3, 3)` normalized camera intrinsics(s)
    """
    if isinstance(size, tuple):
        size = np.array(size, dtype=intrinsics.dtype)
        size = np.broadcast_to(size, (*intrinsics.shape[:-2], 2))
    height, width = size[..., 0], size[..., 1]
    zeros = np.zeros_like(width)
    ones = np.ones_like(width)
    if pixel_convention == 'integer-center':
        transform = np.stack([
            1 / width, zeros, 0.5 / width,
            zeros, 1 / height, 0.5 / height,
            zeros, zeros, ones
        ], axis=-1).reshape(*zeros.shape, 3, 3)
    elif pixel_convention == 'integer-corner':
        transform = np.stack([
            1 / width, zeros, zeros,
            zeros, 1 / height, zeros,
            zeros, zeros, ones
        ], axis=-1).reshape(*zeros.shape, 3, 3)
    return transform @ intrinsics


@toarray(None, 'intrinsics')
@batched(2, 1)
def denormalize_intrinsics(
    intrinsics: ndarray,
    size: Union[Tuple[Number, Number], ndarray],
    pixel_convention: Literal['integer-center', 'integer-corner'] = 'integer-center',
) -> ndarray:
    """
    Denormalize intrinsics from uv cooridnates to pixel coordinates

    ## Parameters
    - `intrinsics` (ndarray): `(..., 3, 3)` camera intrinsics to denormalize
    - `size` (tuple | ndarray): A tuple `(height, width)` of the image size,
        or an array of shape `(..., 2)` corresponding to the multiple image size(s)
    - `pixel_convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
        - For more definitions, please refer to `pixel_coord_map()`

    ## Returns
        `(ndarray)`: `(..., 3, 3)` denormalized camera intrinsics in pixel coordinates
    """
    if isinstance(size, tuple):
        size = np.array(size, dtype=intrinsics.dtype)
        size = np.broadcast_to(size, (*intrinsics.shape[:-2], 2))
    height, width = size[..., 0], size[..., 1]
    zeros = np.zeros_like(width)
    ones = np.ones_like(width)
    if pixel_convention == 'integer-center':
        transform = np.stack([
            width, zeros, -0.5 * ones,
            zeros, height, -0.5 * ones,
            zeros, zeros, ones
        ], axis=-1).reshape(*zeros.shape, 3, 3)
    elif pixel_convention == 'integer-corner':
        transform = np.stack([
            width, zeros, zeros,
            zeros, height, zeros,
            zeros, zeros, ones
        ], axis=-1).reshape(*zeros.shape, 3, 3)
    return transform @ intrinsics


@toarray(None, _others='intrinsics')
@batched(2, 1, _others=0)
def crop_intrinsics(
    intrinsics: ndarray,
    size: Union[Tuple[Number, Number], ndarray],
    cropped_top: Union[Number, ndarray],
    cropped_left: Union[Number, ndarray],
    cropped_height: Union[Number, ndarray],
    cropped_width: Union[Number, ndarray],
) -> ndarray:
    """
    Evaluate the new intrinsics after cropping the image

    ## Parameters
    - `intrinsics` (ndarray): (..., 3, 3) camera intrinsics(s) to crop
    - `size` (tuple | ndarray): A tuple `(height, width)` of the image size,
        or an array of shape `(..., 2)` corresponding to the multiple image size(s)
    - `cropped_top` (int | ndarray): (...) top pixel index of the cropped image(s)
    - `cropped_left` (int | ndarray): (...) left pixel index of the cropped image(s)
    - `cropped_height` (int | ndarray): (...) height of the cropped image(s)
    - `cropped_width` (int | ndarray): (...) width of the cropped image(s)

    ## Returns
        (ndarray): (..., 3, 3) cropped camera intrinsics
    """
    height, width = size[..., 0], size[..., 1]
    zeros = np.zeros_like(width)
    ones = np.ones_like(width)
    transform = np.stack([
        width / cropped_width, zeros, -cropped_left / cropped_width,
        zeros, height / cropped_height, -cropped_top / cropped_height,
        zeros, zeros, ones
    ]).reshape(*zeros.shape, 3, 3)
    return transform @ intrinsics


def pixel_to_uv(
    pixel: ndarray,
    size: Union[Tuple[Number, Number], ndarray],
    pixel_convention: Literal['integer-center', 'integer-corner'] = 'integer-center',
) -> ndarray:
    """
    Convert pixel space coordiantes to UV space coordinates.

    ## Parameters
    - `pixel` (ndarray): `(..., 2)` pixel coordinrates 
    - `size` (tuple | ndarray): A tuple `(height, width)` of the image size,
        or an array of shape `(..., 2)` corresponding to the multiple image size(s)
    - `pixel_convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
        - For more definitions, please refer to `pixel_coord_map()`

    ## Returns
        (ndarray): `(..., 2)` uv coordinrates
    """
    if not np.issubdtype(pixel.dtype, np.floating):
        pixel = pixel.astype(np.float32)
    if pixel_convention == 'integer-center':
        pixel = pixel + 0.5
    uv = pixel / np.flip(size, axis=-1)
    return uv


@toarray(size='uv')
def uv_to_pixel(
    uv: ndarray,
    size: Union[Tuple[Number, Number], ndarray],
    pixel_convention: Literal['integer-center', 'integer-corner'] = 'integer-center',
) -> ndarray:
    """
    Convert UV space coordinates to pixel space coordinates.

    ## Parameters
    - `uv` (ndarray): `(..., 2)` uv coordinrates.
    - `size` (tuple | ndarray): A tuple `(height, width)` of the image size,
        or an array of shape `(..., 2)` corresponding to the multiple image size(s)
    - `pixel_convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
        - For more definitions, please refer to `pixel_coord_map()`

    ## Returns
        (ndarray): `(..., 2)` pixel coordinrates
    """
    pixel = uv * np.flip(size, axis=-1)
    if pixel_convention == 'integer-center':
        pixel = pixel - 0.5
    return pixel


@toarray(size=np.float32)
def pixel_to_ndc(
    pixel: ndarray,
    size: Union[Tuple[Number, Number], ndarray],
    pixel_convention: Literal['integer-center', 'integer-corner'] = 'integer-center',
) -> ndarray:
    """
    Convert pixel coordinates to NDC (Normalized Device Coordinates).

    ## Parameters
    - `pixel` (ndarray): `(..., 2)` pixel coordinrates.
    - `size` (tuple | ndarray): A tuple `(height, width)` of the image size,
        or an array of shape `(..., 2)` corresponding to the multiple image size(s)
    - `pixel_convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates represent pixel centers or corners. Defaults to 'integer-center'.
        - For more definitions, please refer to `pixel_coord_map()`

    ## Returns
        (ndarray): `(..., 2)` ndc coordinrates, the range is (-1, 1)
    """
    if not np.issubdtype(pixel.dtype, np.floating):
        pixel = pixel.astype(np.float32)
    dtype = pixel.dtype
    if pixel_convention == 'integer-center':
        pixel = pixel + 0.5
    ndc = pixel / (np.flip(size, axis=-1) * np.array([2, -2], dtype=dtype)) \
        + np.array([-1, 1], dtype=dtype)
    return ndc


@batched(0, 0, 0)
def depth_linear_to_buffer(
    depth: ndarray,
    near: Union[float, ndarray],
    far: Union[float, ndarray]
) -> ndarray:
    """
    Project linear depth to depth value in screen space

    ## Parameters
        depth (ndarray): [...] depth value
        near (float | ndarray): [...] near plane to clip
        far (float | ndarray): [...] far plane to clip

    ## Returns
        (ndarray): [..., 1] depth value in screen space, value ranging in [0, 1]
    """
    return (1 - near / depth) / (1 - near / far)


@batched(0, 0, 0)
def depth_buffer_to_linear(
    depth_buffer: ndarray,
    near: Union[float, ndarray],
    far: Union[float, ndarray]
) -> ndarray:
    """
    OpenGL depth buffer to linear depth

    ## Parameters
        depth_buffer (ndarray): [...] depth value
        near (float | ndarray): [...] near plane to clip
        far (float | ndarray): [...] far plane to clip

    ## Returns
        (ndarray): [..., 1] linear depth
    """
    return near / (1 - (1 - near / far) * depth_buffer)


def project_gl(
    points: ndarray,
    projection: ndarray,
    view: ndarray = None,
) -> Tuple[ndarray, ndarray]:
    """
    Project 3D points to 2D following the OpenGL convention (except for row major matrices)

    ## Parameters
        points (ndarray): [..., N, 3] or [..., N, 4] 3D points to project, if the last 
            dimension is 4, the points are assumed to be in homogeneous coordinates
        view (ndarray): [..., 4, 4] view matrix
        projection (ndarray): [..., 4, 4] projection matrix

    ## Returns
        scr_coord (ndarray): [..., N, 2] OpenGL screen space XY coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & bottom
        linear_depth (ndarray): [..., N] linear depth
    """
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones((*points.shape[:-1], 1), dtype=points.dtype)], axis=-1)
    transform = projection @ view if view is not None else projection
    clip_coord = points @ transform.swapaxes(-2, -1)
    ndc_coord = clip_coord[..., :3] / clip_coord[..., 3:]
    scr_coord = ndc_coord * 0.5 + 0.5
    linear_depth = clip_coord[..., 3]
    return scr_coord[..., :2], linear_depth


@no_warnings()
def project_cv(
    points: ndarray,
    intrinsics: ndarray,
    extrinsics: Optional[ndarray] = None,
) -> Tuple[ndarray, ndarray]:
    """
    Project 3D points to 2D following the OpenCV convention

    ## Parameters
        points (ndarray): [..., N, 3]
        extrinsics (ndarray): [..., 4, 4] extrinsics matrix
        intrinsics (ndarray): [..., 3, 3] intrinsics matrix

    ## Returns
        uv_coord (ndarray): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        linear_depth (ndarray): [..., N] linear depth
    """
    assert intrinsics is not None, "intrinsics matrix is required"
    points = np.concatenate([points, np.ones((*points.shape[:-1], 1), dtype=points.dtype)], axis=-1)
    intrinsics = np.block([
        [intrinsics, np.zeros((*intrinsics.shape[:-2], 3, 1), dtype=intrinsics.dtype)],
        [np.broadcast_to(np.array([0, 0, 0, 1], dtype=intrinsics.dtype), (*intrinsics.shape[:-2], 1, 4))]
    ])
    transform = intrinsics @ extrinsics if extrinsics is not None else intrinsics
    points = points @ transform.swapaxes(-2, -1)
    uv_coord = points[..., :2] / points[..., 2:3]
    linear_depth = points[..., 2]
    return uv_coord, linear_depth


def unproject_gl(
    uv: ndarray,
    depth: ndarray,
    projection: ndarray,
    view: Optional[ndarray] = None,
) -> ndarray:
    """
    Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrices)

    ## Parameters
        uv (ndarray): (..., N, 2) screen space XY coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & bottom
        depth (ndarray): (..., N) linear depth values
        projection (ndarray): (..., 4, 4) projection  matrix
        view (ndarray): (..., 4, 4) view matrix
        
    ## Returns
        points (ndarray): (..., N, 3) 3d points
    """
    ndc_xy = uv * 2 - 1
    view_z = -depth
    clip_xy = np.linalg.inv(projection[..., :2, :2] - ndc_xy[..., :, None] * projection[..., 3:, :2]) \
        @ ((ndc_xy[..., :, None] * projection[..., 3:, 2:] - projection[..., :2, 2:]) \
        @ np.concatenate([view_z[..., None, None], np.ones_like(view_z[..., None, None])], axis=-2))
    points = np.concatenate([clip_xy.squeeze(-1), view_z[..., None], np.ones_like(view_z)[..., None]], axis=-1)
    if view is not None:
        points = points @ np.linalg.inv(view).swapaxes(-2, -1)
    return points[..., :3]


@batched(2, 2)
def screen_coord_to_view_coord(screen_coord: ndarray, projection: ndarray) -> ndarray:
    """
    Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrices)

    ## Parameters
        screen_coord (ndarray): (..., N, 3) screen space XYZ coordinates, value ranging in [0, 1]
            The origin (0., 0.) is corresponding to the left & bottom
        projection (ndarray): (..., 4, 4) projection matrix

    ## Returns
        points (ndarray): [..., N, 3] 3d points in view space
    """
    assert projection is not None, "projection matrix is required"
    ndc_xy = screen_coord * 2 - 1
    clip_coord = np.concatenate([ndc_xy, np.ones_like(ndc_xy[..., :1])], axis=-1)
    points = clip_coord @ np.linalg.inv(projection).swapaxes(-1, -2)
    points = points[..., :3] / points[..., 3:]
    return points


@batched(2, 1, 2, 2)
def unproject_cv(
    uv: ndarray,
    depth: ndarray,
    intrinsics: ndarray,
    extrinsics: ndarray = None,
) -> ndarray:
    """
    Unproject uv coordinates to 3D view space following the OpenCV convention

    ## Parameters
        uv (ndarray): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        depth (ndarray): [..., N] depth value
        extrinsics (ndarray): [..., 4, 4] extrinsics matrix
        intrinsics (ndarray): [..., 3, 3] intrinsics matrix

    ## Returns
        points (ndarray): [..., N, 3] 3d points
    """
    intrinsics = np.block([
        [intrinsics, np.zeros((*intrinsics.shape[:-2], 3, 1), dtype=intrinsics.dtype)],
        [np.broadcast_to(np.array([0, 0, 0, 1], dtype=intrinsics.dtype), (*intrinsics.shape[:-2], 1, 4))]
    ])
    transform = intrinsics @ extrinsics if extrinsics is not None else intrinsics
    points = np.concatenate([uv, np.ones((*uv.shape[:-1], 1), dtype=uv.dtype)], axis=-1) * depth[..., None]
    points = np.concatenate([points, np.ones((*points.shape[:-1], 1), dtype=uv.dtype)], axis=-1)
    points = points @ np.linalg.inv(transform).swapaxes(-2, -1)
    points = points[..., :3]
    return points



def project(
    points: ndarray,
    *,
    intrinsics: Optional[ndarray] = None,
    extrinsics: Optional[ndarray] = None,
    view: Optional[ndarray] = None,
    projection: Optional[ndarray] = None
) -> Tuple[ndarray, ndarray]:
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
    uv: ndarray,
    depth: Optional[ndarray],
    *,
    intrinsics: Optional[ndarray] = None,
    extrinsics: Optional[ndarray] = None,
    projection: Optional[ndarray] = None,
    view: Optional[ndarray] = None,
) -> ndarray:
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


def quaternion_to_matrix(quaternion: ndarray) -> ndarray:
    """Converts a batch of quaternions (w, x, y, z) to rotation matrices
    
    ## Parameters
        quaternion (ndarray): shape (..., 4), the quaternions to convert
    
    ## Returns
        ndarray: shape (..., 3, 3), the rotation matrices corresponding to the given quaternions
    """
    assert quaternion.shape[-1] == 4
    quaternion = quaternion / np.maximum(lite_norm(quaternion, axis=-1), np.finfo(quaternion.dtype).tiny)[..., None]
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    zeros = np.zeros_like(w)
    I = np.eye(3, dtype=quaternion.dtype)
    xyz = quaternion[..., 1:]
    A = xyz[..., :, None] * xyz[..., None, :] - I * lite_sum(np.square(xyz), axis=-1)[..., None, None]
    B = np.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros
    ], axis=-1).reshape(*quaternion.shape[:-1], 3, 3)
    rot_mat = I + 2 * (A + w[..., None, None] * B)
    return rot_mat


def matrix_to_quaternion(rot_mat: ndarray) -> ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)

    ## Parameters
        rot_mat (ndarray): shape (..., 3, 3), the rotation matrices to convert

    ## Returns
        ndarray: shape (..., 4), the quaternions corresponding to the given rotation matrices
    """
    # Extract the diagonal and off-diagonal elements of the rotation matrix
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = [rot_mat[..., i, j] for i in range(3) for j in range(3)]

    diag = np.diagonal(rot_mat, axis1=-2, axis2=-1)
    M = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=rot_mat.dtype)
    wxyz = 0.5 * np.clip(1 + diag @ M.T, 0.0, None) ** 0.5
    max_idx = np.argmax(wxyz, axis=-1)
    xw = np.sign(m21 - m12)
    yw = np.sign(m02 - m20)
    zw = np.sign(m10 - m01)
    yz = np.sign(m21 + m12)
    xz = np.sign(m02 + m20)
    xy = np.sign(m01 + m10)
    ones = np.ones_like(xw)
    sign = np.where(
        max_idx[..., None] == 0,
        np.stack([ones, xw, yw, zw], axis=-1),
        np.where(
            max_idx[..., None] == 1,
            np.stack([xw, ones, xy, xz], axis=-1),
            np.where(
                max_idx[..., None] == 2,
                np.stack([yw, xy, ones, yz], axis=-1),
                np.stack([zw, xz, yz, ones], axis=-1)
            )
        )
    )
    quat = sign * wxyz
    quat = quat / np.maximum(lite_norm(quat, axis=-1), np.finfo(quat.dtype).tiny)[..., None]
    return quat


def quaternion_multiply(q1: ndarray, q2: ndarray) -> ndarray:
    """Multiplies two quaternions (w, x, y, z)

    Parameters
    ----
        q1 (ndarray): shape (..., 4), the first quaternion
        q2 (ndarray): shape (..., 4), the second quaternion

    Returns
    ----
        ndarray: shape (..., 4), the product of the two quaternions
    """
    w1, v1 = q1[..., :1], q1[..., 1:]
    w2, v2 = q2[..., :1], q2[..., 1:]
    res_w = w1 * w2 - lite_dot(v1, v2)[..., None]
    res_v = w1 * v2 + w2 * v1 + np.cross(v1, v2, axis=-1)
    res_q = np.concatenate([res_w, res_v], axis=-1)
    return res_q


def quaternion_normalize(quaternion: ndarray) -> ndarray:
    """Normalize quaternions (w, x, y, z) to unit length and positive w component
    
    Parameters
    ----
        quaternion (ndarray): shape (..., 4), the quaternions to normalize

    Returns
    ----
        ndarray: shape (..., 4), the normalized quaternions with unit length and positive w component
    """
    w_sign = np.where(quaternion[..., 0] >= 0, 1.0, -1.0).astype(quaternion.dtype)
    quat = quaternion * w_sign[..., None]
    quat = quat / np.maximum(lite_norm(quat, axis=-1)[..., None], np.finfo(quat.dtype).tiny)
    return quat


def quaternion_inverse(quaternion: ndarray) -> ndarray:
    """Calculate the inverse of a batch of quaternions (w, x, y, z)

    Parameters
    ----
        quaternion (ndarray): shape (..., 4), the quaternions to invert

    Returns
    ----
        ndarray: shape (..., 4)
    """
    w, v = quaternion[..., 0:1], quaternion[..., 1:]
    inv_quat = np.concatenate([w, -v], axis=-1)
    return inv_quat


def quaternion_to_axis_angle(quaternion: ndarray) -> ndarray:
    """Convert a batch of quaternions (w, x, y, z) to axis-angle representation (rotation vector)

    ## Parameters
        quaternion (ndarray): shape (..., 4), the quaternions to convert

    ## Returns
        ndarray: shape (..., 3), the axis-angle vectors corresponding to the given quaternions
    """
    assert quaternion.shape[-1] == 4
    norm = lite_norm(quaternion[..., 1:], axis=-1)
    axis = quaternion[..., 1:] / np.maximum(norm, np.finfo(quaternion.dtype).tiny)[..., None]
    angle = 2 * np.atan2(norm, quaternion[..., 0:1])
    return angle * axis


def matrix_to_axis_angle(rot_mat: ndarray) -> ndarray:
    """Convert a batch of 3x3 rotation matrices to axis-angle representation (rotation vector)

    ## Parameters
        rot_mat (ndarray): shape (..., 3, 3), the rotation matrices to convert

    ## Returns
        ndarray: shape (..., 3), the axis-angle vectors corresponding to the given rotation matrices
    """
    quat = matrix_to_quaternion(rot_mat)
    axis_angle = quaternion_to_axis_angle(quat)
    return axis_angle


def extrinsics_to_essential(extrinsics: ndarray):
    """
    extrinsics matrix `[[R, t] [0, 0, 0, 1]]` such that `x' = R (x - t)` to essential matrix such that `x' E x = 0`

    ## Parameters
        extrinsics (np.ndaray): [..., 4, 4] extrinsics matrix

    ## Returns
        (np.ndaray): [..., 3, 3] essential matrix
    """
    assert extrinsics.shape[-2:] == (4, 4)
    R = extrinsics[..., :3, :3]
    t = extrinsics[..., :3, 3]
    zeros = np.zeros_like(t[..., 0])
    t_x = np.stack([
        zeros, -t[..., 2], t[..., 1],
        t[..., 2], zeros, -t[..., 0],
        -t[..., 1], t[..., 0], zeros
    ]).reshape(*t.shape[:-1], 3, 3)
    return t_x @ R 


def euler_axis_angle_rotation(axis: str, angle: ndarray) -> ndarray:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    ## Parameters
        axis: Axis label "X" or "Y or "Z".
        angle: any shape ndarray of Euler angles in radians

    ## Returns
        Rotation matrices as ndarray of shape (..., 3, 3).
    """

    cos = np.cos(angle)
    sin = np.sin(angle)
    one = np.ones_like(angle)
    zero = np.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return np.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: ndarray, convention: str = 'XYZ') -> ndarray:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    ## Parameters
        euler_angles: Euler angles in radians as ndarray of shape (..., 3), XYZ
        convention: permutation of "X", "Y" or "Z", representing the order of Euler rotations to apply.

    ## Returns
        Rotation matrices as ndarray of shape (..., 3, 3).
    """
    if euler_angles.shape[-1] != 3:
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
    return matrices[2] @ matrices[1] @ matrices[0]


def skew_symmetric(v: ndarray):
    "Skew symmetric matrix from a 3D vector"
    assert v.shape[-1] == 3, "v must be 3D"
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zeros = np.zeros_like(x)
    return np.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros,
    ], axis=-1).reshape(*v.shape[:-1], 3, 3)


def rotation_matrix_from_vectors(v1: ndarray, v2: ndarray):
    "Rotation matrix that rotates v1 to v2"
    I = np.eye(3, dtype=v1.dtype)
    v1 = v1 / lite_norm(v1, axis=-1)[..., None]
    v2 = v2 / lite_norm(v2, axis=-1)[..., None]
    v = np.cross(v1, v2, axis=-1)
    c = lite_dot(v1, v2, axis=-1)
    K = skew_symmetric(v)
    R = I + K + (1 / (1 + c)).astype(v1.dtype)[None, None] * (K @ K)    # Avoid numpy's default type casting for scalars
    return R


def axis_angle_to_matrix(axis_angle: ndarray) -> ndarray:
    """Convert axis-angle representation (rotation vector) to rotation matrix, whose direction is the axis of rotation and length is the angle of rotation

    ## Parameters
        axis_angle (ndarray): shape (..., 3), axis-angle vcetors

    ## Returns
        ndarray: shape (..., 3, 3) The rotation matrices for the given axis-angle parameters
    """
    batch_shape = axis_angle.shape[:-1]
    dtype = axis_angle.dtype

    angle = lite_norm(axis_angle, axis=-1)[..., None]
    axis = axis_angle / np.maximum(angle, np.finfo(dtype).tiny)

    cos = np.cos(angle)[..., None, :]
    sin = np.sin(angle)[..., None, :]

    rx, ry, rz = np.split(axis, 3, axis=-1)
    zeros = np.zeros((*batch_shape, 1), dtype=dtype)
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=-1).reshape((*batch_shape, 3, 3))

    ident = np.eye(3, dtype=dtype)
    rot_mat = ident + sin * K + (1 - cos) * (K @ K)
    return rot_mat


def axis_angle_to_quaternion(axis_angle: ndarray) -> ndarray:
    """Convert axis-angle representation (rotation vector) to quaternion (w, x, y, z)

    ## Parameters
        axis_angle (ndarray): shape (..., 3), axis-angle vcetors

    ## Returns
        ndarray: shape (..., 4) The quaternions for the given axis-angle parameters
    """
    angle = lite_norm(axis_angle, axis=-1)[..., None]
    axis = axis_angle / np.maximum(angle, np.finfo(axis_angle.dtype).tiny)
    quat = np.concatenate([np.cos(angle / 2), np.sin(angle / 2) * axis], axis=-1)
    return quat


def _angle_from_tan(
    axis: str, other_axis: str, data: ndarray, horizontal: bool, tait_bryan: bool
) -> ndarray:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    ## Parameters
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as ndarray of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    ## Returns
        Euler Angles in radians for each matrix in data as a ndarray
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return np.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return np.atan2(-data[..., i2], data[..., i1])
    return np.atan2(data[..., i2], -data[..., i1])


def matrix_to_euler_angles(matrix: ndarray, convention: str) -> ndarray:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.
    NOTE: The composition order eg. `XYZ` means `Rz * Ry * Rx` (like blender), instead of `Rx * Ry * Rz` (like pytorch3d)

    ## Parameters
        matrix: Rotation matrices as ndarray of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    ## Returns
        Euler angles in radians as ndarray of shape (..., 3), in the order of XYZ (like blender), instead of convention (like pytorch3d)
    """
    if not all(c in 'XYZ' for c in convention) or not all(c in convention for c in 'XYZ'):
        raise ValueError(f"Invalid convention {convention}.")
    if not matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    
    i0 = 'XYZ'.index(convention[0])
    i2 = 'XYZ'.index(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = np.asin(matrix[..., i2, i0] * (-1.0 if i2 - i0 in [-1, 2] else 1.0))
    else:
        central_angle = np.acos(matrix[..., i2, i2])

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
    return np.stack([o[convention.index(c)] for c in 'XYZ'], -1)


def random_rotation_matrix(*size: int, dtype=np.float32) -> ndarray:
    """
    Generate random 3D rotation matrix.

    ## Parameters
        dtype: The data type of the output rotation matrix.

    ## Returns
        ndarray: `(*size, 3, 3)` random rotation matrix.
    """
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = size[0]
    rand_quat = np.random.randn(*size, 4).astype(dtype)
    return quaternion_to_matrix(rand_quat)


def ray_intersection(p1: ndarray, d1: ndarray, p2: ndarray, d2: ndarray):
    """
    Compute the intersection/closest point of two D-dimensional rays
    If the rays are intersecting, the closest point is the intersection point.

    ## Parameters
        p1 (ndarray): (..., D) origin of ray 1
        d1 (ndarray): (..., D) direction of ray 1
        p2 (ndarray): (..., D) origin of ray 2
        d2 (ndarray): (..., D) direction of ray 2

    ## Returns
        (ndarray): (..., N) intersection point
    """
    p1, d1, p2, d2 = np.broadcast_arrays(p1, d1, p2, d2)
    dtype = p1.dtype
    dim = p1.shape[-1]
    d = np.stack([d1, d2], axis=-2)     # (..., 2, D)
    p = np.stack([p1, p2], axis=-2)     # (..., 2, D)
    A = np.concatenate([
        (np.eye(dim, dtype=dtype) * np.ones((*p.shape[:-2], 2, 1, 1))).reshape(*d.shape[:-2], 2 * dim, dim),         # (..., 2 * D, D)
        -(np.eye(2, dtype=dtype)[..., None] * d[..., None, :]).swapaxes(-2, -1).reshape(*d.shape[:-2], 2 * dim, 2)    # (..., 2 * D, 2)
    ], axis=-1)                             # (..., 2 * D, D + 2)
    b = p.reshape(*p.shape[:-2], 2 * dim)   # (..., 2 * D)
    x = np.linalg.solve(A.swapaxes(-1, -2) @ A + 1e-12 * np.eye(dim + 2, dtype=dtype), (A.swapaxes(-1, -2) @ b[..., :, None]))[..., 0]
    return x[..., :dim], (x[..., dim], x[..., dim + 1])


@batched(2, 1)
def make_affine_matrix(M: ndarray, t: ndarray) -> ndarray:
    """
    Make an affine transformation matrix from a linear matrix and a translation vector.

    ## Parameters
        M (ndarray): [..., D, D] linear matrix (rotation, scaling or general deformation)
        t (ndarray): [..., D] translation vector

    ## Returns
        ndarray: [..., D + 1, D + 1] affine transformation matrix
    """
    x = np.block([
        [M, t[..., None]], 
        [np.zeros((*M.shape[:-2], 1, M.shape[-1]), dtype=M.dtype), np.ones((*M.shape[:-2], 1, 1), dtype=M.dtype)]
    ])
    return x


@toarray(_others=np.float32)
@batched(1, 1, 1)
def lerp(x1: ndarray, x2: ndarray, t: ndarray) -> ndarray:
    """
    Linear interpolation between two vectors.

    ## Parameters
        x1 (ndarray): [..., D] vector 1
        x2 (ndarray): [..., D] vector 2
        t (ndarray): [..., N] interpolation parameter. [0, 1] for interpolation between x1 and x2, otherwise for extrapolation.

    ## Returns
        ndarray: [..., N, D] interpolated vector
    """
    return x1[..., None, :] + t[..., None] * (x2 - x1)[..., None, :]



@toarray(_others=np.float32)
@batched(1, 1, 1)
def slerp(v1: ndarray, v2: ndarray, t: ndarray) -> ndarray:
    """
    Spherical linear interpolation between two (unit) vectors.

    ## Parameters
    - `v1` (ndarray): `(..., D)` (unit) vector 1
    - `v2` (ndarray): `(..., D)` (unit) vector 2
    - `t` (ndarray): `(..., N)` interpolation parameter in [0, 1]

    ## Returns
        ndarray: `(..., N, D)` interpolated unit vector
    """
    v1 = v1 / np.maximum(lite_norm(v1, axis=-1), np.finfo(v1.dtype).tiny)[..., None]
    v2 = v2 / np.maximum(lite_norm(v2, axis=-1), np.finfo(v2.dtype).tiny)[..., None]
    cos = lite_dot(v1, v2, axis=-1)
    v_ortho1 = v2 - v1 * cos[..., None]
    v_ortho2 = v1 - v2 * cos[..., None]
    sin = np.minimum(lite_norm(v_ortho1, axis=-1), lite_norm(v_ortho2, axis=-1))
    theta = np.atan2(sin, cos)[..., None] * t
    v_ortho1 = v_ortho1 / np.maximum(lite_norm(v_ortho1, axis=-1), np.finfo(v1.dtype).tiny)[..., None]
    v = v1[..., None, :] * np.cos(theta)[..., None] + v_ortho1[..., None, :] * np.sin(theta)[..., None]
    return v


@toarray(_others=np.float32)
@batched(2, 2, 1)
def slerp_rotation_matrix(R1: ndarray, R2: ndarray, t: ndarray) -> ndarray:
    """
    Spherical linear interpolation between two rotation matrices.

    ## Parameters
    - `R1` (ndarray): [..., 3, 3] rotation matrix 1
    - `R2` (ndarray): [..., 3, 3] rotation matrix 2
    - `t` (ndarray): [..., N] interpolation parameter in [0, 1]

    ## Returns
        ndarray: [...,N, 3, 3] interpolated rotation matrix
    """
    quat1 = matrix_to_quaternion(R1)
    quat2 = matrix_to_quaternion(R2)
    quat = slerp(quat1, quat2, t)
    return quaternion_to_matrix(quat)


@toarray(_others=np.float32)
@batched(2, 2, 1)
def interpolate_se3_matrix(T1: ndarray, T2: ndarray, t: ndarray) -> ndarray:
    """
    Linear interpolation between two SE(3) matrices.

    ## Parameters
    - `T1` (ndarray): [..., 4, 4] SE(3) matrix 1
    - `T2` (ndarray): [..., 4, 4] SE(3) matrix 2
    - `t` (ndarray): [..., N] interpolation parameter in [0, 1]

    ## Returns
        ndarray: [..., N, 4, 4] interpolated SE(3) matrix
    """
    assert T1.shape[-2:] == (4, 4) and T2.shape[-2:] == (4, 4)
    rot = slerp_rotation_matrix(T1[..., :3, :3], T2[..., :3, :3], t)
    pos = lerp(T1[..., :3, 3], T2[..., :3, 3], t)
    return make_affine_matrix(rot, pos)


def piecewise_lerp(x: ndarray, t: ndarray, s: ndarray, extrapolation_mode: Literal['constant', 'linear'] = 'constant') -> ndarray:
    """
    Linear spline interpolation.

    ## Parameters
    - `x`: ndarray, shape (n, d): the values of data points.
    - `t`: ndarray, shape (n,): the times of the data points.
    - `s`: ndarray, shape (m,): the times to be interpolated.
    - `extrapolation_mode`: str, the mode of extrapolation. 'constant' means extrapolate the boundary values, 'linear' means extrapolate linearly.
    
    ## Returns
    - `y`: ndarray, shape (..., m, d): the interpolated values.
    """
    i = np.searchsorted(t, s, side='left')
    if extrapolation_mode == 'constant':
        prev = np.clip(i - 1, 0, len(t) - 1)
        suc = np.clip(i, 0, len(t) - 1)
    elif extrapolation_mode == 'linear':
        prev = np.clip(i - 1, 0, len(t) - 2)
        suc = np.clip(i, 1, len(t) - 1)
    else:
        raise ValueError(f'Invalid extrapolation_mode: {extrapolation_mode}')
    
    u = (s - t[prev]) / np.maximum(t[suc] - t[prev], 1e-12)
    y = lerp(x[prev], x[suc], u)

    return y


def piecewise_interpolate_se3_matrix(T: ndarray, t: ndarray, s: ndarray, extrapolation_mode: Literal['constant', 'linear'] = 'constant') -> ndarray:
    """
    Linear spline interpolation for SE(3) matrices.

    ## Parameters
    - `T`: ndarray, shape (n, 4, 4): the SE(3) matrices.
    - `t`: ndarray, shape (n,): the times of the data points.
    - `s`: ndarray, shape (m,): the times to be interpolated.
    - `extrapolation_mode`: str, the mode of extrapolation. 'constant' means extrapolate the boundary values, 'linear' means extrapolate linearly.

    ## Returns
    - `T_interp`: ndarray, shape (..., m, 4, 4): the interpolated SE(3) matrices.
    """
    i = np.searchsorted(t, s, side='left')
    if extrapolation_mode == 'constant':
        prev = np.clip(i - 1, 0, len(t) - 1)
        suc = np.clip(i, 0, len(t) - 1)
    elif extrapolation_mode == 'linear':
        prev = np.clip(i - 1, 0, len(t) - 2)
        suc = np.clip(i, 1, len(t) - 1)
    else:
        raise ValueError(f'Invalid extrapolation_mode: {extrapolation_mode}')
    
    u = (s - t[prev]) / np.maximum(t[suc] - t[prev], 1e-12)
    T = interpolate_se3_matrix(T[prev], T[suc], u)

    return T


def transform_points(x: ndarray, *Ts: ndarray) -> ndarray:
    """
    Apply transformation(s) to a point or a set of points.
    It is like `(Tn @ ... @ T2 @ T1 @ x[:, None]).squeeze(0)`, but: 
    1. Automatically handle the homogeneous coordinate;
            - x will be padded with homogeneous coordinate 1.
            - Each T will be padded by identity matrix to match the dimension. 
    2. Using efficient contraction path when array sizes are large, based on `einsum`.
    
    ## Parameters
    - `x`: ndarray, shape `(..., D)`: the points to be transformed.
    - `Ts`: ndarray, shape `(..., D1, D2)`: the affine transformation matrix (matrices)
        If more than one transformation is given, they will be applied in corresponding order.
    ## Returns
    - `y`: ndarray, shape `(..., D)`: the transformed point or a set of points.

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
    x = np.concatenate([x, np.ones((*x.shape[:-1], pad_dim - x.shape[-1]), dtype=x.dtype)], axis=-1)
    I = np.eye(pad_dim, dtype=x.dtype)
    Ts = [
        np.concatenate([
            np.concatenate([T, np.broadcast_to(I[:T.shape[-2], T.shape[-1]:], (*T.shape[:-2], T.shape[-2], pad_dim - T.shape[-1]))], axis=-1),
            np.broadcast_to(I[T.shape[-2]:, :], (*T.shape[:-2], pad_dim - T.shape[-2], pad_dim))
        ], axis=-2)
        for T in Ts
    ]
    total_numel = sum(t.size for t in Ts) + x.size
    if total_numel > 1000:
        # Only use einsum when the total number of elements is large enough to benefit from optimized contraction path
        operands = [*reversed(Ts), x[..., None]]
        offset = len(operands) + 1
        batch_shape = np.broadcast_shapes(*(m.shape[:-2] for m in operands))
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
        y = np.einsum(
            *itertools.chain(*zip(squeezed_operands, subscripts)), 
            (*range(offset, offset + len(batch_shape)), 0, len(operands)), 
            optimize="optimal"
        )
        y = y.squeeze(-1)
    else:
        y = x[..., None]
        for T in Ts:
            y = T @ y
        y = y.squeeze(-1)
    return y[..., :input_dim]
    

def angle_between(v1: ndarray, v2: ndarray):
    """
    Calculate the angle between two (batches of) vectors.
    Better precision than using the arccos dot product directly.

    ## Parameters
    - `v1`: ndarray, shape (..., D): the first vector.
    - `v2`: ndarray, shape (..., D): the second vector.

    ## Returns
    `angle`: ndarray, shape (...): the angle between the two vectors.
    """
    n1 = lite_norm(v1, axis=-1)
    n2 = lite_norm(v2, axis=-1)
    v1 = v1 / np.where(n1 == 0, 1, n1)[..., None]
    v2 = v2 / np.where(n2 == 0, 1, n2)[..., None]
    cos = lite_dot(v1, v2)
    sin = np.minimum(lite_norm(v2 - v1 * cos[..., None], axis=-1), lite_norm(v1 - v2 * cos[..., None], axis=-1))
    return np.atan2(sin, cos)
