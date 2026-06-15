# Auto-generated implementation redirecting to numpy/torch implementations
import sys
from typing import TYPE_CHECKING
import utils3d
from ..helpers import suppress_traceback

__all__ = ["sliding_window", 
"pooling", 
"max_pool_2d", 
"lookup", 
"lookup_get", 
"lookup_set", 
"group", 
"csr_matrix_from_dense_indices", 
"reverse_permutation", 
"vector_outer", 
"perspective_from_fov", 
"perspective_from_window", 
"intrinsics_from_fov", 
"intrinsics_from_focal_center", 
"fov_to_focal", 
"focal_to_fov", 
"intrinsics_to_fov", 
"view_look_at", 
"extrinsics_look_at", 
"perspective_to_intrinsics", 
"perspective_to_near_far", 
"intrinsics_to_perspective", 
"extrinsics_to_view", 
"view_to_extrinsics", 
"normalize_intrinsics", 
"denormalize_intrinsics", 
"crop_intrinsics", 
"pixel_to_uv", 
"pixel_to_ndc", 
"uv_to_pixel", 
"depth_linear_to_buffer", 
"depth_buffer_to_linear", 
"unproject_cv", 
"unproject_gl", 
"project_cv", 
"project_gl", 
"project", 
"unproject", 
"screen_coord_to_view_coord", 
"quaternion_to_matrix", 
"quaternion_multiply", 
"quaternion_inverse", 
"quaternion_normalize", 
"axis_angle_to_matrix", 
"matrix_to_quaternion", 
"extrinsics_to_essential", 
"axis_angle_to_quaternion", 
"euler_axis_angle_rotation", 
"euler_angles_to_matrix", 
"matrix_to_axis_angle", 
"matrix_to_euler_angles", 
"quaternion_to_axis_angle", 
"skew_symmetric", 
"rotation_matrix_from_vectors", 
"ray_intersection", 
"make_affine_matrix", 
"random_rotation_matrix", 
"lerp", 
"slerp", 
"slerp_rotation_matrix", 
"interpolate_se3_matrix", 
"piecewise_lerp", 
"piecewise_interpolate_se3_matrix", 
"transform_points", 
"angle_between", 
"kabasch", 
"umeyama", 
"affine_umeyama", 
"solve_pose", 
"segment_solve_pose", 
"solve_poses_sequential", 
"segment_solve_poses_sequential", 
"segment_roll", 
"segment_take", 
"segment_argmax", 
"segment_argmin", 
"segment_concatenate", 
"segment_concat", 
"segment_chain", 
"group_as_segments", 
"triangulate_mesh", 
"compute_face_corner_angles", 
"compute_face_corner_normals", 
"compute_face_corner_tangents", 
"compute_face_normals", 
"compute_face_tangents", 
"compute_vertex_normals", 
"remove_corrupted_faces", 
"merge_duplicate_vertices", 
"remove_unused_vertices", 
"subdivide_mesh", 
"mesh_edges", 
"mesh_half_edges", 
"mesh_connected_components", 
"graph_connected_components", 
"mesh_adjacency_graph", 
"flatten_mesh_indices", 
"create_cube_mesh", 
"create_icosahedron_mesh", 
"create_square_mesh", 
"create_camera_frustum_mesh", 
"merge_meshes", 
"uv_map", 
"pixel_coord_map", 
"screen_coord_map", 
"build_grid_mesh", 
"build_mesh_from_map", 
"build_mesh_from_depth_map", 
"depth_map_edge", 
"depth_map_aliasing", 
"normal_map_edge", 
"point_map_to_normal_map", 
"depth_map_to_point_map", 
"depth_map_to_normal_map", 
"chessboard", 
"masked_nearest_resize", 
"masked_area_resize", 
"colorize_depth_map", 
"colorize_normal_map", 
"colorize_segmentation_map", 
"colorize_probability_map", 
"flood_fill", 
"perlin_noise", 
"perlin_noise_map", 
"fractal_perlin_noise_map", 
"RastContext", 
"rasterize_triangles", 
"rasterize_triangles_peeling", 
"rasterize_lines", 
"rasterize_point_cloud", 
"sample_texture", 
"test_rasterization", 
"read_extrinsics_from_colmap", 
"read_intrinsics_from_colmap", 
"write_extrinsics_as_colmap", 
"write_intrinsics_as_colmap", 
"read_obj", 
"write_obj", 
"read_ply", 
"write_ply", 
"masked_min", 
"masked_max", 
"csr_eliminate_zeros", 
"lexsort", 
"index_reduce", 
"index_reduce_", 
"scatter_argmax", 
"scatter_argmin", 
"large_multinomial", 
"matrix_trace", 
"rotation_matrix_2d", 
"rotate_2d", 
"translate_2d", 
"scale_2d", 
"pose_graph_optimization", 
"segment_median", 
"segment_sum", 
"segment_cumsum", 
"segment_sort", 
"segment_argsort", 
"segment_topk", 
"stack_segments", 
"segment_multinomial", 
"segment_combinations", 
"segment_searchsorted", 
"mesh_dual_graph", 
"compute_boundaries", 
"remove_isolated_pieces", 
"compute_mesh_laplacian", 
"laplacian_smooth_mesh", 
"taubin_smooth_mesh", 
"laplacian_hc_smooth_mesh", 
"bounding_rect_from_mask", 
"texture_composite"]

def _contains_tensor(obj):
    if isinstance(obj, (list, tuple)):
        return any(_contains_tensor(item) for item in obj)
    elif isinstance(obj, dict):
        return any(_contains_tensor(value) for value in obj.values())
    else:
        import torch
        return isinstance(obj, torch.Tensor)


@suppress_traceback
def _call_based_on_args(fname, args, kwargs):
    if 'torch' in sys.modules:
        if any(_contains_tensor(arg) for arg in args) or any(_contains_tensor(v) for v in kwargs.values()):
            fn = getattr(utils3d.torch, fname, None)
            if fn is None:
                raise NotImplementedError(f"Function {fname} has no torch implementation.")
            return fn(*args, **kwargs)
    fn = getattr(utils3d.numpy, fname, None)
    if fn is None:
        raise NotImplementedError(f"Function {fname} has no numpy implementation.") 
    return fn(*args, **kwargs)


@suppress_traceback
def sliding_window(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.sliding_window, utils3d.torch.sliding_window
    return _call_based_on_args('sliding_window', args, kwargs)

@suppress_traceback
def pooling(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.pooling, None
    return _call_based_on_args('pooling', args, kwargs)

@suppress_traceback
def max_pool_2d(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.max_pool_2d, None
    return _call_based_on_args('max_pool_2d', args, kwargs)

@suppress_traceback
def lookup(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.lookup, utils3d.torch.lookup
    return _call_based_on_args('lookup', args, kwargs)

@suppress_traceback
def lookup_get(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.lookup_get, utils3d.torch.lookup_get
    return _call_based_on_args('lookup_get', args, kwargs)

@suppress_traceback
def lookup_set(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.lookup_set, utils3d.torch.lookup_set
    return _call_based_on_args('lookup_set', args, kwargs)

@suppress_traceback
def group(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.group, utils3d.torch.group
    return _call_based_on_args('group', args, kwargs)

@suppress_traceback
def csr_matrix_from_dense_indices(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.csr_matrix_from_dense_indices, utils3d.torch.csr_matrix_from_dense_indices
    return _call_based_on_args('csr_matrix_from_dense_indices', args, kwargs)

@suppress_traceback
def reverse_permutation(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.reverse_permutation, utils3d.torch.reverse_permutation
    return _call_based_on_args('reverse_permutation', args, kwargs)

@suppress_traceback
def vector_outer(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.vector_outer, utils3d.torch.vector_outer
    return _call_based_on_args('vector_outer', args, kwargs)

@suppress_traceback
def perspective_from_fov(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.perspective_from_fov, utils3d.torch.perspective_from_fov
    return _call_based_on_args('perspective_from_fov', args, kwargs)

@suppress_traceback
def perspective_from_window(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.perspective_from_window, utils3d.torch.perspective_from_window
    return _call_based_on_args('perspective_from_window', args, kwargs)

@suppress_traceback
def intrinsics_from_fov(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.intrinsics_from_fov, utils3d.torch.intrinsics_from_fov
    return _call_based_on_args('intrinsics_from_fov', args, kwargs)

@suppress_traceback
def intrinsics_from_focal_center(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.intrinsics_from_focal_center, utils3d.torch.intrinsics_from_focal_center
    return _call_based_on_args('intrinsics_from_focal_center', args, kwargs)

@suppress_traceback
def fov_to_focal(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.fov_to_focal, utils3d.torch.fov_to_focal
    return _call_based_on_args('fov_to_focal', args, kwargs)

@suppress_traceback
def focal_to_fov(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.focal_to_fov, utils3d.torch.focal_to_fov
    return _call_based_on_args('focal_to_fov', args, kwargs)

@suppress_traceback
def intrinsics_to_fov(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.intrinsics_to_fov, utils3d.torch.intrinsics_to_fov
    return _call_based_on_args('intrinsics_to_fov', args, kwargs)

@suppress_traceback
def view_look_at(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.view_look_at, utils3d.torch.view_look_at
    return _call_based_on_args('view_look_at', args, kwargs)

@suppress_traceback
def extrinsics_look_at(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.extrinsics_look_at, utils3d.torch.extrinsics_look_at
    return _call_based_on_args('extrinsics_look_at', args, kwargs)

@suppress_traceback
def perspective_to_intrinsics(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.perspective_to_intrinsics, utils3d.torch.perspective_to_intrinsics
    return _call_based_on_args('perspective_to_intrinsics', args, kwargs)

@suppress_traceback
def perspective_to_near_far(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.perspective_to_near_far, None
    return _call_based_on_args('perspective_to_near_far', args, kwargs)

@suppress_traceback
def intrinsics_to_perspective(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.intrinsics_to_perspective, utils3d.torch.intrinsics_to_perspective
    return _call_based_on_args('intrinsics_to_perspective', args, kwargs)

@suppress_traceback
def extrinsics_to_view(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.extrinsics_to_view, utils3d.torch.extrinsics_to_view
    return _call_based_on_args('extrinsics_to_view', args, kwargs)

@suppress_traceback
def view_to_extrinsics(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.view_to_extrinsics, utils3d.torch.view_to_extrinsics
    return _call_based_on_args('view_to_extrinsics', args, kwargs)

@suppress_traceback
def normalize_intrinsics(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.normalize_intrinsics, utils3d.torch.normalize_intrinsics
    return _call_based_on_args('normalize_intrinsics', args, kwargs)

@suppress_traceback
def denormalize_intrinsics(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.denormalize_intrinsics, utils3d.torch.denormalize_intrinsics
    return _call_based_on_args('denormalize_intrinsics', args, kwargs)

@suppress_traceback
def crop_intrinsics(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.crop_intrinsics, utils3d.torch.crop_intrinsics
    return _call_based_on_args('crop_intrinsics', args, kwargs)

@suppress_traceback
def pixel_to_uv(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.pixel_to_uv, utils3d.torch.pixel_to_uv
    return _call_based_on_args('pixel_to_uv', args, kwargs)

@suppress_traceback
def pixel_to_ndc(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.pixel_to_ndc, utils3d.torch.pixel_to_ndc
    return _call_based_on_args('pixel_to_ndc', args, kwargs)

@suppress_traceback
def uv_to_pixel(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.uv_to_pixel, utils3d.torch.uv_to_pixel
    return _call_based_on_args('uv_to_pixel', args, kwargs)

@suppress_traceback
def depth_linear_to_buffer(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.depth_linear_to_buffer, utils3d.torch.depth_linear_to_buffer
    return _call_based_on_args('depth_linear_to_buffer', args, kwargs)

@suppress_traceback
def depth_buffer_to_linear(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.depth_buffer_to_linear, utils3d.torch.depth_buffer_to_linear
    return _call_based_on_args('depth_buffer_to_linear', args, kwargs)

@suppress_traceback
def unproject_cv(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.unproject_cv, utils3d.torch.unproject_cv
    return _call_based_on_args('unproject_cv', args, kwargs)

@suppress_traceback
def unproject_gl(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.unproject_gl, utils3d.torch.unproject_gl
    return _call_based_on_args('unproject_gl', args, kwargs)

@suppress_traceback
def project_cv(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.project_cv, utils3d.torch.project_cv
    return _call_based_on_args('project_cv', args, kwargs)

@suppress_traceback
def project_gl(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.project_gl, utils3d.torch.project_gl
    return _call_based_on_args('project_gl', args, kwargs)

@suppress_traceback
def project(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.project, utils3d.torch.project
    return _call_based_on_args('project', args, kwargs)

@suppress_traceback
def unproject(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.unproject, utils3d.torch.unproject
    return _call_based_on_args('unproject', args, kwargs)

@suppress_traceback
def screen_coord_to_view_coord(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.screen_coord_to_view_coord, None
    return _call_based_on_args('screen_coord_to_view_coord', args, kwargs)

@suppress_traceback
def quaternion_to_matrix(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.quaternion_to_matrix, utils3d.torch.quaternion_to_matrix
    return _call_based_on_args('quaternion_to_matrix', args, kwargs)

@suppress_traceback
def quaternion_multiply(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.quaternion_multiply, utils3d.torch.quaternion_multiply
    return _call_based_on_args('quaternion_multiply', args, kwargs)

@suppress_traceback
def quaternion_inverse(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.quaternion_inverse, utils3d.torch.quaternion_inverse
    return _call_based_on_args('quaternion_inverse', args, kwargs)

@suppress_traceback
def quaternion_normalize(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.quaternion_normalize, utils3d.torch.quaternion_normalize
    return _call_based_on_args('quaternion_normalize', args, kwargs)

@suppress_traceback
def axis_angle_to_matrix(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.axis_angle_to_matrix, utils3d.torch.axis_angle_to_matrix
    return _call_based_on_args('axis_angle_to_matrix', args, kwargs)

@suppress_traceback
def matrix_to_quaternion(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.matrix_to_quaternion, utils3d.torch.matrix_to_quaternion
    return _call_based_on_args('matrix_to_quaternion', args, kwargs)

@suppress_traceback
def extrinsics_to_essential(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.extrinsics_to_essential, utils3d.torch.extrinsics_to_essential
    return _call_based_on_args('extrinsics_to_essential', args, kwargs)

@suppress_traceback
def axis_angle_to_quaternion(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.axis_angle_to_quaternion, utils3d.torch.axis_angle_to_quaternion
    return _call_based_on_args('axis_angle_to_quaternion', args, kwargs)

@suppress_traceback
def euler_axis_angle_rotation(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.euler_axis_angle_rotation, utils3d.torch.euler_axis_angle_rotation
    return _call_based_on_args('euler_axis_angle_rotation', args, kwargs)

@suppress_traceback
def euler_angles_to_matrix(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.euler_angles_to_matrix, utils3d.torch.euler_angles_to_matrix
    return _call_based_on_args('euler_angles_to_matrix', args, kwargs)

@suppress_traceback
def matrix_to_axis_angle(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.matrix_to_axis_angle, utils3d.torch.matrix_to_axis_angle
    return _call_based_on_args('matrix_to_axis_angle', args, kwargs)

@suppress_traceback
def matrix_to_euler_angles(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.matrix_to_euler_angles, utils3d.torch.matrix_to_euler_angles
    return _call_based_on_args('matrix_to_euler_angles', args, kwargs)

@suppress_traceback
def quaternion_to_axis_angle(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.quaternion_to_axis_angle, utils3d.torch.quaternion_to_axis_angle
    return _call_based_on_args('quaternion_to_axis_angle', args, kwargs)

@suppress_traceback
def skew_symmetric(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.skew_symmetric, utils3d.torch.skew_symmetric
    return _call_based_on_args('skew_symmetric', args, kwargs)

@suppress_traceback
def rotation_matrix_from_vectors(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.rotation_matrix_from_vectors, utils3d.torch.rotation_matrix_from_vectors
    return _call_based_on_args('rotation_matrix_from_vectors', args, kwargs)

@suppress_traceback
def ray_intersection(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.ray_intersection, None
    return _call_based_on_args('ray_intersection', args, kwargs)

@suppress_traceback
def make_affine_matrix(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.make_affine_matrix, utils3d.torch.make_affine_matrix
    return _call_based_on_args('make_affine_matrix', args, kwargs)

@suppress_traceback
def random_rotation_matrix(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.random_rotation_matrix, utils3d.torch.random_rotation_matrix
    return _call_based_on_args('random_rotation_matrix', args, kwargs)

@suppress_traceback
def lerp(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.lerp, utils3d.torch.lerp
    return _call_based_on_args('lerp', args, kwargs)

@suppress_traceback
def slerp(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.slerp, utils3d.torch.slerp
    return _call_based_on_args('slerp', args, kwargs)

@suppress_traceback
def slerp_rotation_matrix(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.slerp_rotation_matrix, utils3d.torch.slerp_rotation_matrix
    return _call_based_on_args('slerp_rotation_matrix', args, kwargs)

@suppress_traceback
def interpolate_se3_matrix(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.interpolate_se3_matrix, utils3d.torch.interpolate_se3_matrix
    return _call_based_on_args('interpolate_se3_matrix', args, kwargs)

@suppress_traceback
def piecewise_lerp(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.piecewise_lerp, None
    return _call_based_on_args('piecewise_lerp', args, kwargs)

@suppress_traceback
def piecewise_interpolate_se3_matrix(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.piecewise_interpolate_se3_matrix, None
    return _call_based_on_args('piecewise_interpolate_se3_matrix', args, kwargs)

@suppress_traceback
def transform_points(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.transform_points, utils3d.torch.transform_points
    return _call_based_on_args('transform_points', args, kwargs)

@suppress_traceback
def angle_between(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.angle_between, utils3d.torch.angle_between
    return _call_based_on_args('angle_between', args, kwargs)

@suppress_traceback
def kabasch(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.kabasch, utils3d.torch.kabasch
    return _call_based_on_args('kabasch', args, kwargs)

@suppress_traceback
def umeyama(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.umeyama, utils3d.torch.umeyama
    return _call_based_on_args('umeyama', args, kwargs)

@suppress_traceback
def affine_umeyama(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.affine_umeyama, utils3d.torch.affine_umeyama
    return _call_based_on_args('affine_umeyama', args, kwargs)

@suppress_traceback
def solve_pose(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.solve_pose, utils3d.torch.solve_pose
    return _call_based_on_args('solve_pose', args, kwargs)

@suppress_traceback
def segment_solve_pose(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.segment_solve_pose, utils3d.torch.segment_solve_pose
    return _call_based_on_args('segment_solve_pose', args, kwargs)

@suppress_traceback
def solve_poses_sequential(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.solve_poses_sequential, None
    return _call_based_on_args('solve_poses_sequential', args, kwargs)

@suppress_traceback
def segment_solve_poses_sequential(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.segment_solve_poses_sequential, None
    return _call_based_on_args('segment_solve_poses_sequential', args, kwargs)

@suppress_traceback
def segment_roll(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.segment_roll, utils3d.torch.segment_roll
    return _call_based_on_args('segment_roll', args, kwargs)

@suppress_traceback
def segment_take(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.segment_take, utils3d.torch.segment_take
    return _call_based_on_args('segment_take', args, kwargs)

@suppress_traceback
def segment_argmax(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.segment_argmax, utils3d.torch.segment_argmax
    return _call_based_on_args('segment_argmax', args, kwargs)

@suppress_traceback
def segment_argmin(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.segment_argmin, utils3d.torch.segment_argmin
    return _call_based_on_args('segment_argmin', args, kwargs)

@suppress_traceback
def segment_concatenate(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.segment_concatenate, utils3d.torch.segment_concatenate
    return _call_based_on_args('segment_concatenate', args, kwargs)

@suppress_traceback
def segment_concat(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.segment_concat, utils3d.torch.segment_concat
    return _call_based_on_args('segment_concat', args, kwargs)

@suppress_traceback
def segment_chain(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.segment_chain, utils3d.torch.segment_chain
    return _call_based_on_args('segment_chain', args, kwargs)

@suppress_traceback
def group_as_segments(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.group_as_segments, utils3d.torch.group_as_segments
    return _call_based_on_args('group_as_segments', args, kwargs)

@suppress_traceback
def triangulate_mesh(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.triangulate_mesh, utils3d.torch.triangulate_mesh
    return _call_based_on_args('triangulate_mesh', args, kwargs)

@suppress_traceback
def compute_face_corner_angles(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.compute_face_corner_angles, utils3d.torch.compute_face_corner_angles
    return _call_based_on_args('compute_face_corner_angles', args, kwargs)

@suppress_traceback
def compute_face_corner_normals(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.compute_face_corner_normals, utils3d.torch.compute_face_corner_normals
    return _call_based_on_args('compute_face_corner_normals', args, kwargs)

@suppress_traceback
def compute_face_corner_tangents(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.compute_face_corner_tangents, utils3d.torch.compute_face_corner_tangents
    return _call_based_on_args('compute_face_corner_tangents', args, kwargs)

@suppress_traceback
def compute_face_normals(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.compute_face_normals, utils3d.torch.compute_face_normals
    return _call_based_on_args('compute_face_normals', args, kwargs)

@suppress_traceback
def compute_face_tangents(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.compute_face_tangents, utils3d.torch.compute_face_tangents
    return _call_based_on_args('compute_face_tangents', args, kwargs)

@suppress_traceback
def compute_vertex_normals(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.compute_vertex_normals, None
    return _call_based_on_args('compute_vertex_normals', args, kwargs)

@suppress_traceback
def remove_corrupted_faces(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.remove_corrupted_faces, utils3d.torch.remove_corrupted_faces
    return _call_based_on_args('remove_corrupted_faces', args, kwargs)

@suppress_traceback
def merge_duplicate_vertices(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.merge_duplicate_vertices, utils3d.torch.merge_duplicate_vertices
    return _call_based_on_args('merge_duplicate_vertices', args, kwargs)

@suppress_traceback
def remove_unused_vertices(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.remove_unused_vertices, utils3d.torch.remove_unused_vertices
    return _call_based_on_args('remove_unused_vertices', args, kwargs)

@suppress_traceback
def subdivide_mesh(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.subdivide_mesh, utils3d.torch.subdivide_mesh
    return _call_based_on_args('subdivide_mesh', args, kwargs)

@suppress_traceback
def mesh_edges(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.mesh_edges, utils3d.torch.mesh_edges
    return _call_based_on_args('mesh_edges', args, kwargs)

@suppress_traceback
def mesh_half_edges(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.mesh_half_edges, utils3d.torch.mesh_half_edges
    return _call_based_on_args('mesh_half_edges', args, kwargs)

@suppress_traceback
def mesh_connected_components(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.mesh_connected_components, utils3d.torch.mesh_connected_components
    return _call_based_on_args('mesh_connected_components', args, kwargs)

@suppress_traceback
def graph_connected_components(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.graph_connected_components, utils3d.torch.graph_connected_components
    return _call_based_on_args('graph_connected_components', args, kwargs)

@suppress_traceback
def mesh_adjacency_graph(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.mesh_adjacency_graph, None
    return _call_based_on_args('mesh_adjacency_graph', args, kwargs)

@suppress_traceback
def flatten_mesh_indices(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.flatten_mesh_indices, None
    return _call_based_on_args('flatten_mesh_indices', args, kwargs)

@suppress_traceback
def create_cube_mesh(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.create_cube_mesh, utils3d.torch.create_cube_mesh
    return _call_based_on_args('create_cube_mesh', args, kwargs)

@suppress_traceback
def create_icosahedron_mesh(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.create_icosahedron_mesh, utils3d.torch.create_icosahedron_mesh
    return _call_based_on_args('create_icosahedron_mesh', args, kwargs)

@suppress_traceback
def create_square_mesh(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.create_square_mesh, None
    return _call_based_on_args('create_square_mesh', args, kwargs)

@suppress_traceback
def create_camera_frustum_mesh(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.create_camera_frustum_mesh, utils3d.torch.create_camera_frustum_mesh
    return _call_based_on_args('create_camera_frustum_mesh', args, kwargs)

@suppress_traceback
def merge_meshes(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.merge_meshes, None
    return _call_based_on_args('merge_meshes', args, kwargs)

@suppress_traceback
def uv_map(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.uv_map, utils3d.torch.uv_map
    return _call_based_on_args('uv_map', args, kwargs)

@suppress_traceback
def pixel_coord_map(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.pixel_coord_map, utils3d.torch.pixel_coord_map
    return _call_based_on_args('pixel_coord_map', args, kwargs)

@suppress_traceback
def screen_coord_map(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.screen_coord_map, utils3d.torch.screen_coord_map
    return _call_based_on_args('screen_coord_map', args, kwargs)

@suppress_traceback
def build_grid_mesh(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.build_grid_mesh, None
    return _call_based_on_args('build_grid_mesh', args, kwargs)

@suppress_traceback
def build_mesh_from_map(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.build_mesh_from_map, utils3d.torch.build_mesh_from_map
    return _call_based_on_args('build_mesh_from_map', args, kwargs)

@suppress_traceback
def build_mesh_from_depth_map(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.build_mesh_from_depth_map, utils3d.torch.build_mesh_from_depth_map
    return _call_based_on_args('build_mesh_from_depth_map', args, kwargs)

@suppress_traceback
def depth_map_edge(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.depth_map_edge, utils3d.torch.depth_map_edge
    return _call_based_on_args('depth_map_edge', args, kwargs)

@suppress_traceback
def depth_map_aliasing(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.depth_map_aliasing, utils3d.torch.depth_map_aliasing
    return _call_based_on_args('depth_map_aliasing', args, kwargs)

@suppress_traceback
def normal_map_edge(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.normal_map_edge, None
    return _call_based_on_args('normal_map_edge', args, kwargs)

@suppress_traceback
def point_map_to_normal_map(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.point_map_to_normal_map, utils3d.torch.point_map_to_normal_map
    return _call_based_on_args('point_map_to_normal_map', args, kwargs)

@suppress_traceback
def depth_map_to_point_map(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.depth_map_to_point_map, utils3d.torch.depth_map_to_point_map
    return _call_based_on_args('depth_map_to_point_map', args, kwargs)

@suppress_traceback
def depth_map_to_normal_map(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.depth_map_to_normal_map, utils3d.torch.depth_map_to_normal_map
    return _call_based_on_args('depth_map_to_normal_map', args, kwargs)

@suppress_traceback
def chessboard(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.chessboard, utils3d.torch.chessboard
    return _call_based_on_args('chessboard', args, kwargs)

@suppress_traceback
def masked_nearest_resize(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.masked_nearest_resize, utils3d.torch.masked_nearest_resize
    return _call_based_on_args('masked_nearest_resize', args, kwargs)

@suppress_traceback
def masked_area_resize(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.masked_area_resize, utils3d.torch.masked_area_resize
    return _call_based_on_args('masked_area_resize', args, kwargs)

@suppress_traceback
def colorize_depth_map(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.colorize_depth_map, None
    return _call_based_on_args('colorize_depth_map', args, kwargs)

@suppress_traceback
def colorize_normal_map(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.colorize_normal_map, None
    return _call_based_on_args('colorize_normal_map', args, kwargs)

@suppress_traceback
def colorize_segmentation_map(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.colorize_segmentation_map, None
    return _call_based_on_args('colorize_segmentation_map', args, kwargs)

@suppress_traceback
def colorize_probability_map(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.colorize_probability_map, None
    return _call_based_on_args('colorize_probability_map', args, kwargs)

@suppress_traceback
def flood_fill(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.flood_fill, utils3d.torch.flood_fill
    return _call_based_on_args('flood_fill', args, kwargs)

@suppress_traceback
def perlin_noise(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.perlin_noise, utils3d.torch.perlin_noise
    return _call_based_on_args('perlin_noise', args, kwargs)

@suppress_traceback
def perlin_noise_map(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.perlin_noise_map, utils3d.torch.perlin_noise_map
    return _call_based_on_args('perlin_noise_map', args, kwargs)

@suppress_traceback
def fractal_perlin_noise_map(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.fractal_perlin_noise_map, utils3d.torch.fractal_perlin_noise_map
    return _call_based_on_args('fractal_perlin_noise_map', args, kwargs)

@suppress_traceback
def RastContext(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.RastContext, utils3d.torch.RastContext
    return _call_based_on_args('RastContext', args, kwargs)

@suppress_traceback
def rasterize_triangles(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.rasterize_triangles, utils3d.torch.rasterize_triangles
    return _call_based_on_args('rasterize_triangles', args, kwargs)

@suppress_traceback
def rasterize_triangles_peeling(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.rasterize_triangles_peeling, utils3d.torch.rasterize_triangles_peeling
    return _call_based_on_args('rasterize_triangles_peeling', args, kwargs)

@suppress_traceback
def rasterize_lines(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.rasterize_lines, None
    return _call_based_on_args('rasterize_lines', args, kwargs)

@suppress_traceback
def rasterize_point_cloud(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.rasterize_point_cloud, None
    return _call_based_on_args('rasterize_point_cloud', args, kwargs)

@suppress_traceback
def sample_texture(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.sample_texture, utils3d.torch.sample_texture
    return _call_based_on_args('sample_texture', args, kwargs)

@suppress_traceback
def test_rasterization(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.test_rasterization, None
    return _call_based_on_args('test_rasterization', args, kwargs)

@suppress_traceback
def read_extrinsics_from_colmap(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.read_extrinsics_from_colmap, None
    return _call_based_on_args('read_extrinsics_from_colmap', args, kwargs)

@suppress_traceback
def read_intrinsics_from_colmap(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.read_intrinsics_from_colmap, None
    return _call_based_on_args('read_intrinsics_from_colmap', args, kwargs)

@suppress_traceback
def write_extrinsics_as_colmap(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.write_extrinsics_as_colmap, None
    return _call_based_on_args('write_extrinsics_as_colmap', args, kwargs)

@suppress_traceback
def write_intrinsics_as_colmap(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.write_intrinsics_as_colmap, None
    return _call_based_on_args('write_intrinsics_as_colmap', args, kwargs)

@suppress_traceback
def read_obj(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.read_obj, None
    return _call_based_on_args('read_obj', args, kwargs)

@suppress_traceback
def write_obj(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.write_obj, None
    return _call_based_on_args('write_obj', args, kwargs)

@suppress_traceback
def read_ply(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.read_ply, None
    return _call_based_on_args('read_ply', args, kwargs)

@suppress_traceback
def write_ply(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        utils3d.numpy.write_ply, None
    return _call_based_on_args('write_ply', args, kwargs)

@suppress_traceback
def masked_min(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.masked_min
    return _call_based_on_args('masked_min', args, kwargs)

@suppress_traceback
def masked_max(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.masked_max
    return _call_based_on_args('masked_max', args, kwargs)

@suppress_traceback
def csr_eliminate_zeros(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.csr_eliminate_zeros
    return _call_based_on_args('csr_eliminate_zeros', args, kwargs)

@suppress_traceback
def lexsort(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.lexsort
    return _call_based_on_args('lexsort', args, kwargs)

@suppress_traceback
def index_reduce(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.index_reduce
    return _call_based_on_args('index_reduce', args, kwargs)

@suppress_traceback
def index_reduce_(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.index_reduce_
    return _call_based_on_args('index_reduce_', args, kwargs)

@suppress_traceback
def scatter_argmax(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.scatter_argmax
    return _call_based_on_args('scatter_argmax', args, kwargs)

@suppress_traceback
def scatter_argmin(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.scatter_argmin
    return _call_based_on_args('scatter_argmin', args, kwargs)

@suppress_traceback
def large_multinomial(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.large_multinomial
    return _call_based_on_args('large_multinomial', args, kwargs)

@suppress_traceback
def matrix_trace(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.matrix_trace
    return _call_based_on_args('matrix_trace', args, kwargs)

@suppress_traceback
def rotation_matrix_2d(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.rotation_matrix_2d
    return _call_based_on_args('rotation_matrix_2d', args, kwargs)

@suppress_traceback
def rotate_2d(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.rotate_2d
    return _call_based_on_args('rotate_2d', args, kwargs)

@suppress_traceback
def translate_2d(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.translate_2d
    return _call_based_on_args('translate_2d', args, kwargs)

@suppress_traceback
def scale_2d(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.scale_2d
    return _call_based_on_args('scale_2d', args, kwargs)

@suppress_traceback
def pose_graph_optimization(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.pose_graph_optimization
    return _call_based_on_args('pose_graph_optimization', args, kwargs)

@suppress_traceback
def segment_median(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.segment_median
    return _call_based_on_args('segment_median', args, kwargs)

@suppress_traceback
def segment_sum(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.segment_sum
    return _call_based_on_args('segment_sum', args, kwargs)

@suppress_traceback
def segment_cumsum(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.segment_cumsum
    return _call_based_on_args('segment_cumsum', args, kwargs)

@suppress_traceback
def segment_sort(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.segment_sort
    return _call_based_on_args('segment_sort', args, kwargs)

@suppress_traceback
def segment_argsort(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.segment_argsort
    return _call_based_on_args('segment_argsort', args, kwargs)

@suppress_traceback
def segment_topk(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.segment_topk
    return _call_based_on_args('segment_topk', args, kwargs)

@suppress_traceback
def stack_segments(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.stack_segments
    return _call_based_on_args('stack_segments', args, kwargs)

@suppress_traceback
def segment_multinomial(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.segment_multinomial
    return _call_based_on_args('segment_multinomial', args, kwargs)

@suppress_traceback
def segment_combinations(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.segment_combinations
    return _call_based_on_args('segment_combinations', args, kwargs)

@suppress_traceback
def segment_searchsorted(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.segment_searchsorted
    return _call_based_on_args('segment_searchsorted', args, kwargs)

@suppress_traceback
def mesh_dual_graph(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.mesh_dual_graph
    return _call_based_on_args('mesh_dual_graph', args, kwargs)

@suppress_traceback
def compute_boundaries(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.compute_boundaries
    return _call_based_on_args('compute_boundaries', args, kwargs)

@suppress_traceback
def remove_isolated_pieces(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.remove_isolated_pieces
    return _call_based_on_args('remove_isolated_pieces', args, kwargs)

@suppress_traceback
def compute_mesh_laplacian(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.compute_mesh_laplacian
    return _call_based_on_args('compute_mesh_laplacian', args, kwargs)

@suppress_traceback
def laplacian_smooth_mesh(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.laplacian_smooth_mesh
    return _call_based_on_args('laplacian_smooth_mesh', args, kwargs)

@suppress_traceback
def taubin_smooth_mesh(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.taubin_smooth_mesh
    return _call_based_on_args('taubin_smooth_mesh', args, kwargs)

@suppress_traceback
def laplacian_hc_smooth_mesh(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.laplacian_hc_smooth_mesh
    return _call_based_on_args('laplacian_hc_smooth_mesh', args, kwargs)

@suppress_traceback
def bounding_rect_from_mask(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.bounding_rect_from_mask
    return _call_based_on_args('bounding_rect_from_mask', args, kwargs)

@suppress_traceback
def texture_composite(*args, **kwargs):
    if TYPE_CHECKING:  # redirected to:
        None, utils3d.torch.texture_composite
    return _call_based_on_args('texture_composite', args, kwargs)

