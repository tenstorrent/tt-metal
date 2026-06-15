import numpy as np
from numpy import ndarray
from typing import *
if TYPE_CHECKING:
    import scipy.sparse as sp
    from scipy.sparse import csr_array

from .transforms import unproject_cv, angle_between
from .utils import lookup, csr_matrix_from_dense_indices
from .segment_ops import segment_roll


__all__ = [
    'triangulate_mesh',
    'compute_face_corner_angles',
    'compute_face_corner_normals',
    'compute_face_corner_tangents',
    'compute_face_normals',
    'compute_face_tangents',
    'compute_vertex_normals',
    'remove_corrupted_faces',
    'merge_duplicate_vertices',
    'remove_unused_vertices',
    'subdivide_mesh',
    'mesh_edges',
    'mesh_half_edges',
    'mesh_connected_components',
    'graph_connected_components',
    'mesh_adjacency_graph',
    'flatten_mesh_indices',
    'create_cube_mesh',
    'create_icosahedron_mesh',
    'create_square_mesh',
    'create_camera_frustum_mesh',
    'merge_meshes',
    # 'calc_quad_candidates',
    # 'calc_quad_distortion',
    # 'calc_quad_direction',
    # 'calc_quad_smoothness',
    # 'solve_quad',
    # 'solve_quad_qp',
    # 'tri_to_quad'
]


def triangulate_mesh(
    faces: ndarray,
    vertices: ndarray = None,
    method: Literal['fan', 'strip', 'diagonal'] = 'fan',
    return_face_indices: bool = False,
) -> ndarray:
    """
    Triangulate a polygonal mesh.

    ## Parameters
        faces (ndarray): [L, P] polygonal faces
        vertices (ndarray, optional): [N, 3] 3-dimensional vertices.
            If given, the triangulation is performed according to the distance
            between vertices. Defaults to None.
        method (str, optional): triangulation method. Defaults to 'fan'.
            - 'fan': connect the first vertex to all other vertex pairs
            - 'strip': create a triangle strip
            - 'diagonal': for quad faces only, split according to the shorter diagonal
        return_face_indices (bool, optional): whether to return the original face indices for each triangle. Defaults to False.

    ## Returns
        (ndarray): [L * (P - 2), 3] triangular faces
    """
    if faces.shape[-1] == 3:
        return faces
    P = faces.shape[-1]
    if method == 'fan':
        i = np.arange(P - 2, dtype=int)
        triangle_loop = np.stack([np.zeros_like(i), i + 1, i + 2], axis=1)  # (P - 2, 3)
        triangles = faces[:, triangle_loop].reshape((-1, 3))
        if return_face_indices:
            triangle_face_indices = np.repeat(np.arange(faces.shape[0], dtype=int), len(triangle_loop))
    elif method == 'strip':
        i = np.arange(P - 2, dtype=int)
        j = i // 2
        loop_indices = np.where(
            (i % 2 == 0)[:, None],
            np.stack([(P - j) % P, j + 1, P - j - 1], axis=1),
            np.stack([j + 1, j + 2, P - j - 1], axis=1)
        )
        triangles = faces[:, loop_indices].reshape((-1, 3))
        if return_face_indices:
            triangle_face_indices = np.repeat(np.arange(faces.shape[0], dtype=int), len(triangle_loop))
    elif method == 'diagonal':
        assert faces.shape[-1] == 4, "Diagonal-aware method is only supported for quad faces"
        assert vertices is not None, "Vertices must be provided for diagonal method"
        backslash = np.linalg.norm(vertices[faces[:, 0]] - vertices[faces[:, 2]], axis=-1) < \
                        np.linalg.norm(vertices[faces[:, 1]] - vertices[faces[:, 3]], axis=-1)
        triangles = np.where(
            backslash[:, None],
            faces[:, [0, 1, 2, 0, 2, 3]],
            faces[:, [0, 1, 3, 3, 1, 2]]
        ).reshape((-1, 3))
        if return_face_indices:
            triangle_face_indices = np.repeat(np.arange(faces.shape[0], dtype=int), 2)
    
    if return_face_indices:
        return triangles, triangle_face_indices
    else:
        return triangles


def compute_face_corner_angles(
    vertices: ndarray,
    faces: Optional[ndarray] = None,
) -> ndarray:
    """
    Compute face corner angles of a mesh

    ## Parameters
    - `vertices` (ndarray): `(..., N, 3)` vertices if `faces` is provided, or `(..., F, P, 3)` if `faces` is None
    - `faces` (ndarray, optional): `(F, P)` face vertex indices, where P is the number of vertices per face

    ## Returns
    - `angles` (ndarray): `(..., F, P)` face corner angles
    """
    if faces is not None:
        vertices = vertices[..., faces, :]              # (..., T, P, 3)
    edges = np.roll(vertices, -1, axis=-2) - vertices   # (..., T, P, 3)
    angles = angle_between(-np.roll(edges, 1, axis=-2), edges)
    return angles


def compute_face_corner_normals(
    vertices: ndarray,
    faces: Optional[ndarray] = None,
    normalize: bool = True
) -> ndarray:
    """
    Compute the face corner normals of a mesh

    ## Parameters
    - `vertices` (ndarray): `(..., N, 3)` vertices if `faces` is provided, or `(..., F, P, 3)` if `faces` is None
    - `faces` (ndarray, optional): `(F, P)` face vertex indices, where P is the number of vertices per face
    - `normalize` (bool): whether to normalize the normals to unit vectors. If not, the normals are the raw cross products.

    ## Returns
    - `normals` (ndarray): (..., F, P, 3) face corner normals
    """
    if faces is not None:
        vertices = vertices[..., faces, :]  # (..., T, P, 3)
    edges = np.roll(vertices, -1, axis=-2) - vertices  # (..., T, P, 3)
    normals = np.cross(np.roll(edges, 1, axis=-2), edges)
    if normalize:
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + np.finfo(vertices.dtype).eps
    return normals


def compute_face_corner_tangents(
    vertices: ndarray,
    uv: ndarray,
    faces_vertices: Optional[ndarray] = None,
    faces_uv: Optional[ndarray] = None,
    normalize: bool = True
) -> ndarray:
    """
    Compute the face corner tangent (and bitangent) vectors of a mesh

    ## Parameters
    - `vertices` (ndarray): `(..., N, 3)` if `faces` is provided, or `(..., F, P, 3)` if `faces_vertices` is None
    - `uv` (ndarray): `(..., N, 2)` if `faces` is provided, or `(..., F, P, 2)` if `faces_uv` is None
    - `faces_vertices` (ndarray, optional): `(F, P)` face vertex indices
    - `faces_uv` (ndarray, optional): `(F, P)` face UV indices
    - `normalize` (bool): whether to normalize the tangents to unit vectors. If not, the tangents (dX/du, dX/dv) matches the UV parameterized manifold.

    ## Returns
    - `tangents` (ndarray): `(..., F, P, 3, 2)` face corner tangents (and bitangents), 
        where the last dimension represents the tangent and bitangent vectors.
    """
    if faces_vertices is not None:
        vertices = vertices[..., faces_vertices, :]
    if faces_uv is not None:
        uv = uv[..., faces_uv, :]
    
    edge_xyz = np.roll(vertices, -1, axis=-2) - vertices  # (..., F, P, 3)
    edge_uv = np.roll(uv, -1, axis=-2) - uv  # (..., F, P, 2)
    
    tangents = np.stack([np.roll(edge_xyz, 1, axis=-2), edge_xyz], axis=-1) \
        @ np.linalg.inv(np.stack([np.roll(edge_uv, 1, axis=-2), edge_uv], axis=-1))
    if normalize:
        tangents /= np.linalg.norm(tangents, axis=-1, keepdims=True) + np.finfo(tangents.dtype).eps
    return tangents


def compute_face_normals(
    vertices: ndarray,
    faces: Optional[ndarray] = None,
) -> ndarray:
    """
    Compute face normals of a mesh

    ## Parameters
    - `vertices` (ndarray): `(..., N, 3)` vertices if `faces` is provided, or `(..., F, P, 3)` if `faces` is None
    - `faces` (ndarray, optional): `(F, P)` face vertex indices, where P is the number of vertices per face

    ## Returns
    - `normals` (ndarray): `(..., F, 3)` face normals. Always normalized.
    """
    if faces is not None:
        vertices = vertices[..., faces, :]  # (..., F, P, 3)
    if vertices.shape[-2] == 3:
        normals = np.cross(
            vertices[..., 1, :] - vertices[..., 0, :],
            vertices[..., 2, :] - vertices[..., 0, :]
        )
    else:
        normals = compute_face_corner_normals(vertices, normalize=False)
        normals = np.mean(normals, axis=-2)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True) + np.finfo(vertices.dtype).eps
    return normals


def compute_face_tangents(
    vertices: ndarray,
    uv: ndarray,
    faces_vertices: Optional[ndarray] = None,
    faces_uv: Optional[ndarray] = None,
    normalize: bool = True
) -> ndarray:
    """
    Compute the face corner tangent (and bitangent) vectors of a mesh

    ## Parameters
    - `vertices` (ndarray): `(..., N, 3)` if `faces` is provided, or `(..., F, P, 3)` if `faces_vertices` is None
    - `uv` (ndarray): `(..., N, 2)` if `faces` is provided, or `(..., F, P, 2)` if `faces_uv` is None
    - `faces_vertices` (ndarray, optional): `(F, P)` face vertex indices
    - `faces_uv` (ndarray, optional): `(F, P)` face UV indices

    ## Returns
    - `tangents` (ndarray): `(..., F, 3, 2)` face corner tangents (and bitangents), 
        where the last dimension represents the tangent and bitangent vectors.
    """
    if faces_vertices is not None:
        vertices = vertices[..., faces_vertices, :]  # (..., F, P, 3)
    if faces_uv is not None:
        uv = uv[..., faces_uv, :]  # (..., F, P, 2)
    if vertices.shape[-2] == 3:
        tangents = np.stack([vertices[..., 1, :] - vertices[..., 0, :], vertices[..., 2, :] - vertices[..., 0, :]], axis=-1) \
            @ np.linalg.inv(np.stack([uv[..., 1, :] - uv[..., 0, :], uv[..., 2, :] - uv[..., 0, :]], axis=-1))
    else:
        tangents = compute_face_corner_tangents(vertices, uv, normalize=False)
        tangents = np.mean(tangents, axis=-2)
    if normalize:
        tangents /= np.linalg.norm(tangents, axis=-1, keepdims=True) + np.finfo(vertices.dtype).eps
    return tangents



def compute_vertex_normals(
    vertices: ndarray,
    faces: ndarray,
    weighted: Literal['uniform', 'area', 'angle'] = 'uniform'
) -> ndarray:
    """
    Compute vertex normals of a triangular mesh by averaging neighboring face normals

    ## Parameters
        vertices (ndarray): [..., N, 3] 3-dimensional vertices
        faces (ndarray): [T, P] face vertex indices, where P is the number of vertices per face

    ## Returns
        normals (ndarray): [..., N, 3] vertex normals (already normalized to unit vectors)
    """
    face_corner_normals = compute_face_corner_normals(vertices, faces, normalize=False)
    if weighted == 'uniform':
        face_corner_normals /= np.linalg.norm(face_corner_normals, axis=-1, keepdims=True) + np.finfo(vertices.dtype).eps
    elif weighted == 'area':
        pass
    elif weighted == 'angle':
        face_corner_angle = compute_face_corner_angles(vertices, faces)
        face_corner_normals *= face_corner_angle[..., None]
    vertex_normals = np.zeros_like(vertices, dtype=vertices.dtype)
    np.add.at(
        vertex_normals, 
        (..., faces[..., None], np.arange(3)), 
        face_corner_normals
    )
    vertex_normals /= np.linalg.norm(vertex_normals, axis=-1, keepdims=True) + np.finfo(vertices.dtype).eps
    return vertex_normals



def remove_corrupted_faces(faces: ndarray) -> ndarray:
    """
    Remove corrupted faces (faces with duplicated vertices)

    ## Parameters
        faces (ndarray): [T, 3] triangular face indices

    ## Returns
        ndarray: [T_, 3] triangular face indices
    """
    corrupted = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
    return faces[~corrupted]


def merge_duplicate_vertices(
    vertices: ndarray, 
    faces: ndarray,
    tol: float = 1e-6
) -> Tuple[ndarray, ndarray]:
    """
    Merge duplicate vertices of a triangular mesh. 
    Duplicate vertices are merged by selecte one of them, and the face indices are updated accordingly.

    ## Parameters
        vertices (ndarray): [N, 3] 3-dimensional vertices
        faces (ndarray): [T, 3] triangular face indices
        tol (float, optional): tolerance for merging. Defaults to 1e-6.

    ## Returns
        vertices (ndarray): [N_, 3] 3-dimensional vertices
        faces (ndarray): [T, 3] triangular face indices
    """
    vertices_round = np.round(vertices / tol)
    _, uni_i, uni_inv = np.unique(vertices_round, return_index=True, return_inverse=True, axis=0)
    vertices = vertices[uni_i]
    faces = uni_inv[faces]
    return vertices, faces


def remove_unused_vertices(
    faces: ndarray,
    *vertice_attrs,
    return_indices: bool = False
) -> Tuple[ndarray, ...]:
    """
    Remove unreferenced vertices of a mesh. 
    Unreferenced vertices are removed, and the face indices are updated accordingly.

    ## Parameters
        faces (ndarray): [T, P] face indices
        *vertice_attrs: vertex attributes

    ## Returns
        faces (ndarray): [T, P] face indices
        *vertice_attrs: vertex attributes
        indices (ndarray, optional): [N] indices of vertices that are kept. Defaults to None.
    """
    P = faces.shape[-1]
    fewer_indices, inv_map = np.unique(faces, return_inverse=True)
    faces = inv_map.astype(np.int32).reshape(-1, P)
    ret = [faces]
    for attr in vertice_attrs:
        ret.append(attr[fewer_indices])
    if return_indices:
        ret.append(fewer_indices)
    return tuple(ret)


def subdivide_mesh(
    vertices: ndarray,
    faces: ndarray, 
    level: int = 1
) -> Tuple[ndarray, ndarray]:
    """
    Subdivide a triangular mesh by splitting each triangle into 4 smaller triangles.
    NOTE: All original vertices are kept, and new vertices are appended to the end of the vertex list.
    
    ## Parameters
        vertices (ndarray): [N, 3] 3-dimensional vertices
        faces (ndarray): [T, 3] triangular face indices
        level (int, optional): level of subdivisions. Defaults to 1.

    ## Returns
        vertices (ndarray): [N_, 3] subdivided 3-dimensional vertices
        faces (ndarray): [(4 ** level) * T, 3] subdivided triangular face indices
    """
    for _ in range(level):
        edges = np.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
        edges = np.sort(edges, axis=2)
        uni_edges, uni_inv = np.unique(edges.reshape(-1, 2), return_inverse=True, axis=0)
        uni_inv = uni_inv.reshape(3, -1)
        midpoints = (vertices[uni_edges[:, 0]] + vertices[uni_edges[:, 1]]) / 2

        n_vertices = vertices.shape[0]
        vertices = np.concatenate([vertices, midpoints], axis=0)
        faces = np.concatenate([
            np.stack([faces[:, 0], n_vertices + uni_inv[0], n_vertices + uni_inv[2]], axis=1),
            np.stack([faces[:, 1], n_vertices + uni_inv[1], n_vertices + uni_inv[0]], axis=1),
            np.stack([faces[:, 2], n_vertices + uni_inv[2], n_vertices + uni_inv[1]], axis=1),
            np.stack([n_vertices + uni_inv[0], n_vertices + uni_inv[1], n_vertices + uni_inv[2]], axis=1),
        ], axis=0)
    return vertices, faces


@overload
def flatten_mesh_indices(faces1: ndarray, attr1: ndarray, *more_faces_attrs_pairs: ndarray) -> Tuple[ndarray, ...]: 
    """
    Rearrange the indices of a mesh to a flattened version. Vertices will be no longer shared.

    ## Parameters:
    - `faces1`: [T, P] face indices of the first attribute
    - `attr1`: [N1, ...] attributes of the first mesh

    Optionally, more pairs of faces and attributes can be provided:
    - `faces2`: ...
    - `attr2`: ...
    - ...

    ## Returns
    - `faces`: [T, P] flattened face indices, contigous from 0 to T * P - 1
    - `attr1`: [T * P, ...] attributes of the first mesh, where every P values correspond to a face
    - `attr2`: ...
    - ...
    """
def flatten_mesh_indices(*args: ndarray) -> Tuple[ndarray, ...]:
    assert len(args) % 2 == 0, "The number of arguments must be even."
    T, P = args[0].shape
    assert all(arg.shape[0] == T and arg.shape[1] == P for arg in args[::2]), "The faces must have the same shape."
    attr_flat = []
    for faces_, attr_ in zip(args[::2], args[1::2]):
        attr_flat_ = attr_[faces_].reshape(-1, *attr_.shape[1:])
        attr_flat.append(attr_flat_)
    faces_flat = np.arange(T * P, dtype=np.int32).reshape(T, P)
    return faces_flat, *attr_flat



def create_square_mesh(tri: bool = False) -> Tuple[ndarray, ndarray]: 
    """
    Create a square mesh of area 1 centered at origin in the xy-plane.

    ## Returns
        vertices (ndarray): shape (4, 3)
        faces (ndarray): shape (1, 4)
    """
    vertices = np.array([
        [0.5, 0.5, 0],   [-0.5, 0.5, 0],   [-0.5, -0.5, 0],   [0.5, -0.5, 0] # v0-v1-v2-v3
    ], dtype=np.float32)
    if tri:
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    else:
        faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    return vertices, faces  


def create_cube_mesh(tri: bool = False) -> Tuple[ndarray, ndarray]:
    """
    Create a cube mesh of size 1 centered at origin.

    ### Parameters
        tri (bool, optional): return triangulated mesh. Defaults to False, which returns quad mesh.

    ### Returns
        vertices (ndarray): shape (8, 3) 
        faces (ndarray): shape (12, 3)
    """
    vertices = np.array([
        [0.5, 0.5, 0.5],   [-0.5, 0.5, 0.5],   [-0.5, -0.5, 0.5],   [0.5, -0.5, 0.5], # v0-v1-v2-v3
        [0.5, 0.5, -0.5],  [-0.5, 0.5, -0.5],  [-0.5, -0.5, -0.5],  [0.5, -0.5, -0.5] # v4-v5-v6-v7
    ], dtype=np.float32).reshape((-1, 3))

    faces = np.array([
        [0, 1, 2, 3], #  (front)
        [5, 4, 7, 6], #  (back)
        [4, 5, 1, 0], #  (top)
        [2, 6, 7, 3], #  (bottom)
        [1, 5, 6, 2], #  (left)
        [4, 0, 3, 7], #  (right)
    ], dtype=np.int32)

    if tri:
        faces = triangulate_mesh(faces, vertices=vertices)

    return vertices, faces


def create_camera_frustum_mesh(extrinsics: ndarray, intrinsics: ndarray, depth: float = 1.0) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Create a triangle mesh of camera frustum.
    """
    assert extrinsics.shape == (4, 4) and intrinsics.shape == (3, 3)
    vertices = unproject_cv(
        np.array([[0, 0], [0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32), 
        np.array([0] + [depth] * 4, dtype=np.float32), 
        intrinsics,
        extrinsics
    ).astype(np.float32)
    edges = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4], 
        [1, 2], [2, 3], [3, 4], [4, 1]
    ], dtype=np.int32)
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
        [1, 2, 3],
        [1, 3, 4]
    ], dtype=np.int32)
    return vertices, edges, faces


def create_icosahedron_mesh():
    """
    Create an icosahedron mesh of centered at origin.
    """
    A = (1 + 5 ** 0.5) / 2
    vertices = np.array([
        [0, 1, A], [0, -1, A], [0, 1, -A], [0, -1, -A],
        [1, A, 0], [-1, A, 0], [1, -A, 0], [-1, -A, 0],
        [A, 0, 1], [A, 0, -1], [-A, 0, 1], [-A, 0, -1]
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 8], [0, 8, 4], [0, 4, 5], [0, 5, 10], [0, 10, 1],
        [3, 2, 9], [3, 9, 6], [3, 6, 7], [3, 7, 11], [3, 11, 2],
        [1, 6, 8], [8, 9, 4], [4, 2, 5], [5, 11, 10], [10, 7, 1],
        [2, 4, 9], [9, 8, 6], [6, 1, 7], [7, 10, 11], [11, 5, 2]
    ], dtype=np.int32)
    return vertices, faces


def merge_meshes(meshes: List[Tuple[ndarray, ...]]) -> Tuple[ndarray, ...]:
    """
    Merge multiple meshes into one mesh. Vertices will be no longer shared.

    ## Parameters
        - `meshes`: a list of tuple (faces, vertices_attr1, vertices_attr2, ....)

    ## Returns
        - `faces`: [sum(T_i), P] merged face indices, contigous from 0 to sum(T_i) * P - 1
        - `*vertice_attrs`: [sum(T_i) * P, ...] merged vertex attributes, where every P values correspond to a face
    """
    faces_merged = []
    attrs_merged = [[] for _ in meshes[0][1:]]
    vertex_offset = 0
    for f, *attrs in meshes:
        faces_merged.append(f + vertex_offset)
        vertex_offset += len(attrs[0])
        for attr_merged, attr in zip(attrs_merged, attrs):
            attr_merged.append(attr)
    faces_merged = np.concatenate(faces_merged, axis=0)
    attrs_merged = [np.concatenate(attr_list, axis=0) for attr_list in attrs_merged]
    return (faces_merged, *attrs_merged)


def mesh_edges(
    faces: Union[ndarray, Tuple[ndarray, ndarray], 'csr_array'], 
    return_face2edge: bool = False, 
    return_edge2face: bool = False, 
    return_counts: bool = False
) -> Tuple[ndarray, Union[ndarray, 'csr_array'], 'csr_array', 'ndarray']:
    """Get undirected edges of a mesh. Optionally return additional mappings.

    ## Parameters
    - `faces` (ndarray): polygon faces, which can be in 3 formats:
        - Regular mesh in regular array: `(F, P)`, where each face has `P` vertices.
        - Irregular mesh in segmented array: tuple of `(vertex_indices, offsets)`, where vertex_indices[offsets[i]:offsets[i+1]] are the vertex indices of face i. 
        - Irregular mesh in CSR array: `(F, V)` binary CSR array of indices, each row corresponds to the vertices of a face.
      (Note that segmented array is almost equivalent to csr array: `vertex_indices` ~ `faces.indices`, `offsets` ~ `faces.indptr`.)
    - `return_face2edge` (bool): whether to return the face to edge mapping
    - `return_edge2face` (bool): whether to return the edge to face mapping
    - `return_counts` (bool): whether to return the counts of edges

    ## Returns
    - `edges` (ndarray): `(E, 2)` unique edges' vertex indices

    If `return_face2edge`, `return_edge2face`, `return_opposite_edge`, or `return_counts` is True, the corresponding outputs will be appended in order:

    - `face2edge` (ndarray | csr_array): mapping from faces to the indices of edges
        - `(F, P)` if input `faces` is a dense array
        - `(F, E)` if input `faces` is segmented array
    - `edge2face` (csr_array): `(E, F)` binary sparse CSR matrix of edge to face.
    - `counts` (ndarray): `(E,)` counts of each edge
    """
    from scipy.sparse import csr_array
    
    if isinstance(faces, (tuple, csr_array)):
        if isinstance(faces, csr_array):
            vertex_indices, offsets = faces.indices, faces.indptr
        else:
            vertex_indices, offsets = faces
        edges = np.stack([vertex_indices, segment_roll(vertex_indices, offsets, -1)], axis=-1) # (nzz, 2)
    else:
        edges = np.stack([faces, np.roll(faces, -1, axis=-1)], axis=-1).reshape(-1, 2)    # (F * P, 2)

    a, b = edges[:, 0], edges[:, 1]
    edges = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=-1)
    
    unique = np.unique(edges, return_inverse=return_face2edge or return_edge2face, return_counts=return_counts or return_edge2face, axis=0)
    
    edges: ndarray = unique[0] if isinstance(unique, tuple) else unique
    if return_face2edge or return_edge2face:
        inv_map: ndarray = unique[1]
    if return_counts or return_edge2face:
        counts: ndarray = unique[-1]

    ret = (edges,)
    if return_face2edge:
        if isinstance(faces, tuple):
            face2edge = csr_array((np.ones_like(inv_map, dtype=bool), inv_map, offsets), shape=(len(offsets) - 1, edges.shape[0]))
        else:
            face2edge = inv_map.reshape(faces.shape)
        ret += (face2edge,)
    if return_edge2face:
        if isinstance(faces, (tuple, csr_array)):
            lengths = np.diff(offsets)
            edge2face = csr_array((
                np.ones_like(inv_map, dtype=bool),
                np.repeat(np.arange(len(lengths)), lengths)[np.argsort(inv_map)],
                np.concatenate([np.array([0], dtype=counts.dtype), np.cumsum(counts)]),
            ), shape=(edges.shape[0], len(lengths)))
            ret += (edge2face,)
        else:
            edge2face = csr_array((
                np.ones_like(inv_map, dtype=bool),
                np.argsort(inv_map) // faces.shape[1],
                np.concatenate([np.array([0], dtype=counts.dtype), np.cumsum(counts)]),
            ), shape=(edges.shape[0], faces.shape[0]))
            ret += (edge2face,)
    if return_counts:
        ret += (counts,)
    return ret[0] if len(ret) == 1 else ret


def mesh_half_edges(
    faces: Union[ndarray, Tuple[ndarray, ndarray], 'csr_array'], 
    return_face2edge: bool = False, 
    return_edge2face: bool = False, 
    return_twin: bool = False,
    return_next: bool = False,
    return_prev: bool = False,
    return_counts: bool = False
) -> Tuple[ndarray, Union[ndarray, 'csr_array'], 'csr_array', ndarray, ndarray, ndarray, ndarray]:
    """Get half edges of a mesh. Optionally return additional mappings.

    ## Parameters
    - `faces` (ndarray): polygon faces, which can be in 3 formats:
    - `faces` (ndarray): polygon faces, which can be in 3 formats:
        - Regular mesh in regular array: `(F, P)`, where each face has `P` vertices.
        - Irregular mesh in segmented array: tuple of `(vertex_indices, offsets)`, where vertex_indices[offsets[i]:offsets[i+1]] are the vertex indices of face i. 
        - Irregular mesh in CSR array: `(F, V)` binary CSR array of indices, each row corresponds to the vertices of a face.
      (Note that segmented array is almost equivalent to csr array: `vertex_indices` ~ `faces.indices`, `offsets` ~ `faces.indptr`.)
    - `return_face2edge` (bool): whether to return the face to edge mapping
    - `return_edge2face` (bool): whether to return the edge to face mapping
    - `return_twin` (bool): whether to return the mapping from one edge to its opposite/twin edge
    - `return_next` (bool): whether to return the mapping from one edge to its next edge in the face loop
    - `return_prev` (bool): whether to return the mapping from one edge to its previous edge in the face loop
    - `return_counts` (bool): whether to return the counts of edges

    ## Returns
    - `edges` (ndarray): `(E, 2)` unique edges' vertex indices

    If `return_face2edge`, `return_edge2face`, `return_opposite_edge`, or `return_counts` is True, the corresponding outputs will be appended in order:

    - `face2edge` (ndarray | csr_array): mapping from faces to the indices of edges
        - `(F, P)` if input `faces` is a dense array
        - `(F, E)` if input `faces` is a sparse csr array
    - `edge2face` (csr_array): `(E, F)` binary sparse CSR matrix of edge to face.
    - `twin` (ndarray): `(E,)` mapping from edges to indices of opposite edges. -1 if not found. 
    - `next` (ndarray): `(E,)` mapping from edges to indices of next edges in the face loop.
    - `prev` (ndarray): `(E,)` mapping from edges to indices of previous edges in the face loop.
    - `counts` (ndarray): `(E,)` counts of each half edge

    NOTE: If the mesh is not manifold, `twin`, `next`, and `prev` can point to arbitrary one of the candidates.
    """
    from scipy.sparse import csr_array

    if isinstance(faces, (tuple, csr_array)):
        if isinstance(faces, csr_array):
            vertex_indices, offsets = faces.indices, faces.indptr
        else:
            vertex_indices, offsets = faces
        edges = np.stack([vertex_indices, segment_roll(vertex_indices, offsets, -1)], axis=-1) # (nzz, 2)
    else:
        edges = np.stack([faces, np.roll(faces, -1, axis=-1)], axis=-1).reshape(-1, 2)    # (F * P, 2)
    
    requires_inv_map = return_face2edge or return_edge2face or return_next or return_prev
    requires_counts = return_counts or return_edge2face

    unique = np.unique(edges, return_inverse=requires_inv_map, return_counts=requires_counts, axis=0)

    edges: ndarray = unique[0] if isinstance(unique, tuple) else unique
    if requires_inv_map:
        inv_map: ndarray = unique[1]
    if requires_counts:
        counts: ndarray = unique[-1]

    ret = (edges,)
    if return_face2edge or return_next or return_prev:
        if isinstance(faces, (tuple, csr_array)):
            face2edge = csr_array(
                (np.ones_like(inv_map, dtype=bool), inv_map, offsets), 
                shape=(len(offsets) - 1, edges.shape[0])
            )
        else:
            face2edge = inv_map.reshape(faces.shape)
        if return_face2edge:
            ret += (face2edge,)
    if return_edge2face:
        if isinstance(faces, (tuple, csr_array)):
            lengths = np.diff(offsets)
            edge2face = csr_array((
                np.ones_like(inv_map, dtype=bool),
                np.repeat(np.arange(len(lengths)), lengths)[np.argsort(inv_map)],
                np.concatenate([np.array([0], dtype=counts.dtype), np.cumsum(counts, axis=0)]),
            ), shape=(edges.shape[0], len(lengths)))
            ret += (edge2face,)
        else:
            edge2face = csr_array((
                np.ones_like(inv_map, dtype=bool),
                np.argsort(inv_map) // faces.shape[1],
                np.concatenate([np.array([0], dtype=counts.dtype), np.cumsum(counts, axis=0)]),
            ), shape=(edges.shape[0], faces.shape[0]))
            ret += (edge2face,)
    if return_twin:
        twin_edge = lookup(edges, np.flip(edges, -1))
        ret += (twin_edge,)
    if return_next or return_prev:
        if isinstance(face2edge, csr_array):
            face2edge_indices = face2edge.indices
            face2edge_indices_next = segment_roll(face2edge.indices, face2edge.indptr, -1)
        else:
            face2edge_indices = face2edge.reshape(-1)
            face2edge_indices_next = np.roll(face2edge, -1, axis=-1).reshape(-1)
    if return_next:
        next_edge = np.full(edges.shape[0], -1, dtype=np.int32)
        np.put(next_edge, face2edge_indices, face2edge_indices_next)
        ret += (next_edge,)
    if return_prev:
        prev_edge = np.full(edges.shape[0], -1, dtype=np.int32)
        np.put(prev_edge, face2edge_indices_next, face2edge_indices)
        ret += (prev_edge,)
    if return_counts:
        ret += (counts,)
    return ret[0] if len(ret) == 1 else ret


def mesh_connected_components(
    faces: Optional[Union[ndarray, Tuple[ndarray, ndarray], 'csr_array']] = None,
    num_vertices: Optional[int] = None
) -> Union[ndarray, Tuple[ndarray, ndarray]]:
    """
    Compute connected faces of a mesh.

    ## Parameters
    - `faces` (ndarray): polygon faces, which can be in 3 formats:
        - Regular mesh in regular array: `(F, P)`, where each face has `P` vertices.
        - Irregular mesh in segmented array: tuple of `(vertex_indices, offsets)`, where vertex_indices[offsets[i]:offsets[i+1]] are the vertex indices of face i. 
        - Irregular mesh in CSR array: `(F, V)` binary CSR array of indices, each row corresponds to the vertices of a face.
      (Note that segmented array is almost equivalent to csr array: `vertex_indices` ~ `faces.indices`, `offsets` ~ `faces.indptr`.)
    - `num_vertices` (int, optional): total number of vertices. If not given, only presented vertices in `faces` are considered.

    ## Returns

    If `num_vertices` is given, return:
    - `labels` (ndarray): (N,) component labels of each vertex

    If `num_vertices` is None, return:
    - `vertices_ids` (ndarray): (N,) vertex indices that are in the edges
    - `labels` (ndarray): (N,) int32 component labels corresponding to `vertices_ids`
    """
    edges = mesh_edges(faces, directed=False)
    return graph_connected_components(edges, num_vertices)


def graph_connected_components(
    edges: ndarray, 
    num_vertices: Optional[int] = None
) -> Union[ndarray, Tuple[ndarray, ndarray]]:
    """
    Compute connected components of an undirected graph.
    Using scipy.sparse.csgraph.connected_components as backend.

    ## Parameters
    - `edges` (ndarray): (E, 2) edge indices

    ## Returns

    If `num_vertices` is given, return:
    - `labels` (ndarray): (N,) component labels of each vertex

    If `num_vertices` is None, return:
    - `vertices_ids` (ndarray): (N,) vertex indices that are in the edges
    - `labels` (ndarray): (N,) int32 component labels corresponding to `vertices_ids`
    """
    from scipy.sparse.csgraph import connected_components
    if num_vertices is None:
        # Re-index edges
        vertices_ids, edges = np.unique(edges.reshape(-1), return_inverse=True)
        edges = edges.reshape(-1, 2)

    num_connected_components, labels = connected_components(
        sp.coo_array(
            (np.ones(edges.shape[0]), edges.T), 
            shape=(num_vertices or len(vertices_ids), num_vertices or len(vertices_ids))
        ), 
        directed=False
    )

    if num_vertices is None:
        return vertices_ids, labels
    else:
        return labels


def mesh_adjacency_graph(
    adjacency: Literal[
        'vertex2edge',
        'vertex2face',
        'edge2vertex',
        'edge2face',
        'face2edge',
        'face2vertex',
        'vertex2edge2vertex',
        'vertex2face2vertex',
        'edge2vertex2edge',
        'edge2face2edge',
        'face2edge2face',
        'face2vertex2face',
    ],
    faces: Optional[Union[ndarray, Tuple[ndarray, ndarray], 'csr_array']] = None,
    edges: Optional[ndarray] = None,
    num_vertices: Optional[int] = None,
    self_loop: bool = False,
) -> 'csr_array':
    """
    Get adjacency graph of a mesh.
    
    ## Parameters
    - `adjacency` (str): type of adjacency graph. Options:
        - `'vertex2edge'`: vertex to adjacent edges. Returns (V, E) csr
        - `'vertex2face'`: vertex to adjacent faces. Returns (V, F) csr
        - `'edge2vertex'`: edge to adjacent vertices. Returns (E, V) csr
        - `'edge2face'`: edge to its adjacent faces. Returns (E, F) csr
        - `'face2edge'`: face to its adjacent edges. Returns (F, E) csr
        - `'face2vertex'`: face to its adjacent vertices. Returns (F, V) csr
        - `'vertex2edge2vertex'`: vertex to adjacent vertices if they share an edge. Returns (V, V) csr
        - `'vertex2face2vertex'`: vertex to adjacent vertices if they share a face. Returns (V, V) csr
        - `'edge2vertex2edge'`: edge to adjacent edges if they share a vertex. Returns (E, E) csr
        - `'edge2face2edge'`: edge to adjacent edges if they share a face. Returns (E, E) csr
        - `'face2edge2face'`: face to adjacent faces if they share an edge. Returns (F, F) csr
        - `'face2vertex2face'`: face to adjacent faces if they share a vertex. Returns (F, F) csr
    - `faces` (ndarray): polygon faces
        - `(F, P)` dense array of indices, where each face has `P` vertices.
        - `(F, V)` binary sparse csr array of indices, each row corresponds to the vertices of a face.
    - `edges` (ndarray, optional): (E, 2) edge indices. NOTE: assumed to be undirected edges.
    - `num_vertices` (int, optional): total number of vertices.
    - `self_loop` (bool): whether to include self-loops in the adjacency graph. Defaults to False.

    ## Returns
    - `graph` (csr_array): adjacency graph in csr format
    """
    from scipy.sparse import csr_array

    if isinstance(faces, csr_array):
        if num_vertices is None:
            num_vertices = faces.shape[1]
        else:
            assert num_vertices == faces.shape[1], f'num_vertices ({num_vertices}) does not match csr array faces.shape[1] ({faces.shape[1]})'
    
    if adjacency == 'vertex2edge':
        assert edges is not None and num_vertices is not None
        return mesh_adjacency_graph('edge2vertex', edges=edges, num_vertices=num_vertices).transpose().tocsr()
    elif adjacency == 'vertex2face':
        return mesh_adjacency_graph('face2vertex', faces=faces, num_vertices=num_vertices).transpose().tocsr()
    elif adjacency == 'edge2vertex':
        assert edges is not None and num_vertices is not None
        return csr_matrix_from_dense_indices(edges, num_vertices)
    elif adjacency == 'edge2face':
        assert edges is not None and faces is not None
        return mesh_adjacency_graph('face2edge', faces=faces, edges=edges).transpose().tocsr() 
    elif adjacency == 'face2edge':
        assert edges is not None and faces is not None
        if isinstance(faces, (tuple, csr_array)):
            vertex_indices, offsets = faces if isinstance(faces, tuple) else (faces.indices, faces.indptr)
            face_edges = np.stack([vertex_indices, segment_roll(vertex_indices, offsets, -1)], axis=-1) # (nzz, 2)
        else:
            face_edges = np.stack([faces, np.roll(faces, -1, axis=-1)], axis=-1).reshape(-1, 2)    # (F * P, 2)
        a, b = face_edges[:, 0], face_edges[:, 1]
        face_edges = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=-1)
        indices = lookup(edges, face_edges)
        return csr_array((
            np.ones_like(indices, dtype=bool),
            indices, 
            offsets if isinstance(faces, (tuple, csr_array)) else np.arange(0, faces.size + 1, faces.shape[1])
        ), shape=(faces.shape[0], edges.shape[0])).tocsr()
    elif adjacency == 'face2vertex':
        assert faces is not None
        if isinstance(faces, csr_array):
            return faces
        elif isinstance(faces, tuple):
            assert num_vertices is not None
            return csr_array(
                (np.ones_like(faces[0], dtype=bool), vertex_indices, offsets), 
                shape=(len(offsets) - 1, num_vertices)
            )
        else:
            assert num_vertices is not None
            return csr_matrix_from_dense_indices(faces, num_vertices)
    elif adjacency == 'vertex2edge2vertex':
        e2v = mesh_adjacency_graph('edge2vertex', edges=edges, num_vertices=num_vertices)
        v2e2v = (e2v.transpose() @ e2v).tocsr()
        if not self_loop:
            v2e2v.setdiag(0)
            v2e2v.eliminate_zeros()
        return v2e2v
    elif adjacency == 'vertex2face2vertex':
        f2v = mesh_adjacency_graph('face2vertex', faces=faces, num_vertices=num_vertices)
        v2f2v = (f2v.transpose() @ f2v).tocsr()
        if not self_loop:
            v2f2v.setdiag(0)
            v2f2v.eliminate_zeros()
        return v2f2v
    elif adjacency == 'edge2vertex2edge':
        # num_vertices is optional here
        if num_vertices is None:    
            vertices_id, edges = np.unique(edges.reshape(-1), return_inverse=True)
            edges = edges.reshape(-1, 2)
            num_vertices = len(vertices_id)
        e2v = mesh_adjacency_graph('edge2vertex', edges=edges, num_vertices=num_vertices)
        e2v2e = (e2v @ e2v.transpose()).tocsr()
        if not self_loop:
            e2v2e.setdiag(0)
            e2v2e.eliminate_zeros()
        return e2v2e
    elif adjacency == 'edge2face2edge':
        e2f = mesh_adjacency_graph('edge2face', faces=faces, edges=edges)
        e2f2e = (e2f @ e2f.transpose()).tocsr()
        if not self_loop:
            e2f2e.setdiag(0)
            e2f2e.eliminate_zeros()
        return e2f2e
    elif adjacency == 'face2edge2face':
        if edges is None:
            _, f2e = mesh_edges(faces, directed=False, return_face2edge=True)
        else:
            f2e = mesh_adjacency_graph('face2edge', faces=faces, edges=edges)
        f2e2f = (f2e @ f2e.transpose()).tocsr()
        if not self_loop:
            f2e2f.setdiag(0)
            f2e2f.eliminate_zeros()
        return f2e2f
    elif adjacency == 'face2vertex2face':
        if isinstance(faces, csr_array):
            f2v = faces
        elif isinstance(faces, tuple):
            vertex_indices, offsets = faces
            unique_vertices, vertex_indices = np.unique(vertex_indices, return_inverse=True)
            f2v = csr_array(
                (np.ones_like(vertex_indices, dtype=bool), vertex_indices, offsets), 
                shape=(len(offsets) - 1, len(unique_vertices))
            )
        else:
            unique_vertices, inv = np.unique(faces.reshape(-1), return_inverse=True)
            faces = inv.reshape(-1, faces.shape[1])
            f2v = csr_matrix_from_dense_indices(faces, len(unique_vertices))
        f2v2f = (f2v @ f2v.transpose()).tocsr()
        if not self_loop:
            f2v2f.setdiag(0)
            f2v2f.eliminate_zeros()
        return f2v2f
    else:
        raise ValueError(f'Unknown adjacency type: {adjacency}')




# def calc_quad_candidates(
#     edges: ndarray,
#     face2edge: ndarray,
#     edge2face: ndarray,
# ):
#     """
#     Calculate the candidate quad faces.

#     ## Parameters
#         edges (ndarray): [E, 2] edge indices
#         face2edge (ndarray): [T, 3] face to edge relation
#         edge2face (ndarray): [E, 2] edge to face relation

#     ## Returns
#         quads (ndarray): [Q, 4] quad candidate indices
#         quad2edge (ndarray): [Q, 4] edge to quad candidate relation
#         quad2adj (ndarray): [Q, 8] adjacent quad candidates of each quad candidate
#         quads_valid (ndarray): [E] whether the quad corresponding to the edge is valid
#     """
#     E = edges.shape[0]
#     T = face2edge.shape[0]

#     quads_valid = edge2face[:, 1] != -1
#     Q = quads_valid.sum()
#     quad2face = edge2face[quads_valid]  # [Q, 2]
#     quad2edge = face2edge[quad2face]  # [Q, 2, 3]
#     flag = quad2edge == np.arange(E)[quads_valid][:, None, None] # [Q, 2, 3]
#     flag = flag.argmax(axis=-1)  # [Q, 2]
#     quad2edge = np.stack([
#         quad2edge[np.arange(Q)[:, None], np.arange(2)[None, :], (flag + 1) % 3],
#         quad2edge[np.arange(Q)[:, None], np.arange(2)[None, :], (flag + 2) % 3],
#     ], axis=-1).reshape(Q, 4)  # [Q, 4]

#     quads = np.concatenate([
#         np.where(
#             (edges[quad2edge[:, 0:1], 1:] == edges[quad2edge[:, 1:2], :]).any(axis=-1),
#             edges[quad2edge[:, 0:1], [[0, 1]]],
#             edges[quad2edge[:, 0:1], [[1, 0]]],
#         ),
#         np.where(
#             (edges[quad2edge[:, 2:3], 1:] == edges[quad2edge[:, 3:4], :]).any(axis=-1),
#             edges[quad2edge[:, 2:3], [[0, 1]]],
#             edges[quad2edge[:, 2:3], [[1, 0]]],
#         ),
#     ], axis=1)  # [Q, 4]

#     quad2adj = edge2face[quad2edge]  # [Q, 4, 2]
#     quad2adj = quad2adj[quad2adj != quad2face[:, [0,0,1,1], None]].reshape(Q, 4)  # [Q, 4]
#     quad2adj_valid = quad2adj != -1
#     quad2adj = face2edge[quad2adj]  # [Q, 4, 3]
#     quad2adj[~quad2adj_valid, 0] = quad2edge[~quad2adj_valid]
#     quad2adj[~quad2adj_valid, 1:] = -1
#     quad2adj = quad2adj[quad2adj != quad2edge[..., None]].reshape(Q, 8)  # [Q, 8]
#     edge_valid = -np.ones(E, dtype=np.int32)
#     edge_valid[quads_valid] = np.arange(Q)
#     quad2adj_valid = quad2adj != -1
#     quad2adj[quad2adj_valid] = edge_valid[quad2adj[quad2adj_valid]]  # [Q, 8]

#     return quads, quad2edge, quad2adj, quads_valid


# def calc_quad_distortion(
#     vertices: ndarray,
#     quads: ndarray,
# ):
#     """
#     Calculate the distortion of each candidate quad face.

#     ## Parameters
#         vertices (ndarray): [N, 3] 3-dimensional vertices
#         quads (ndarray): [Q, 4] quad face indices

#     ## Returns
#         distortion (ndarray): [Q] distortion of each quad face
#     """
#     edge0 = vertices[quads[:, 1]] - vertices[quads[:, 0]]  # [Q, 3]
#     edge1 = vertices[quads[:, 2]] - vertices[quads[:, 1]]  # [Q, 3]
#     edge2 = vertices[quads[:, 3]] - vertices[quads[:, 2]]  # [Q, 3]
#     edge3 = vertices[quads[:, 0]] - vertices[quads[:, 3]]  # [Q, 3]
#     cross = vertices[quads[:, 0]] - vertices[quads[:, 2]]  # [Q, 3]

#     len0 = np.maximum(np.linalg.norm(edge0, axis=-1), 1e-10)  # [Q]
#     len1 = np.maximum(np.linalg.norm(edge1, axis=-1), 1e-10)  # [Q]
#     len2 = np.maximum(np.linalg.norm(edge2, axis=-1), 1e-10)  # [Q]
#     len3 = np.maximum(np.linalg.norm(edge3, axis=-1), 1e-10)  # [Q]
#     len_cross = np.maximum(np.linalg.norm(cross, axis=-1), 1e-10)  # [Q]

#     angle0 = np.arccos(np.clip(np.sum(-edge0 * edge1, axis=-1) / (len0 * len1), -1, 1))  # [Q]
#     angle1 = np.arccos(np.clip(np.sum(-edge1 * cross, axis=-1) / (len1 * len_cross), -1, 1)) \
#            + np.arccos(np.clip(np.sum(cross * edge2, axis=-1) / (len_cross * len2), -1, 1))  # [Q]
#     angle2 = np.arccos(np.clip(np.sum(-edge2 * edge3, axis=-1) / (len2 * len3), -1, 1))  # [Q]
#     angle3 = np.arccos(np.clip(np.sum(-edge3 * -cross, axis=-1) / (len3 * len_cross), -1, 1)) \
#            + np.arccos(np.clip(np.sum(-cross * edge0, axis=-1) / (len_cross * len0), -1, 1))  # [Q]

#     normal0 = np.cross(edge0, edge1)  # [Q, 3]
#     normal1 = np.cross(edge2, edge3)  # [Q, 3]
#     normal0 = normal0 / np.maximum(np.linalg.norm(normal0, axis=-1, keepdims=True), 1e-10)  # [Q, 3]
#     normal1 = normal1 / np.maximum(np.linalg.norm(normal1, axis=-1, keepdims=True), 1e-10)  # [Q, 3]
#     angle_normal = np.arccos(np.clip(np.sum(normal0 * normal1, axis=-1), -1, 1))  # [Q]

#     D90 = np.pi / 2
#     D180 = np.pi
#     D360 = np.pi * 2
#     ang_eng = (np.abs(angle0 - D90)**2 + np.abs(angle1 - D90)**2 + np.abs(angle2 - D90)**2 + np.abs(angle3 - D90)**2) / 4  # [Q]
#     dist_eng = np.abs(angle0 - angle2)**2 / np.minimum(np.maximum(np.minimum(angle0, angle2), 1e-10), np.maximum(D180 - np.maximum(angle0, angle2), 1e-10)) \
#              + np.abs(angle1 - angle3)**2 / np.minimum(np.maximum(np.minimum(angle1, angle3), 1e-10), np.maximum(D180 - np.maximum(angle1, angle3), 1e-10))  # [Q]
#     plane_eng = np.where(angle_normal < D90/2, np.abs(angle_normal)**2, 1e10)  # [Q]
#     eng = ang_eng + 2 * dist_eng + 2 * plane_eng  # [Q]

#     return eng


# def calc_quad_direction(vertices: ndarray, quads: ndarray):
#     """
#     Calculate the direction of each candidate quad face.

#     ## Parameters
#         vertices (ndarray): [N, 3] 3-dimensional vertices
#         quads (ndarray): [Q, 4] quad face indices

#     ## Returns
#         direction (ndarray): [Q, 4] direction of each quad face.
#             Represented by the angle between the crossing and each edge.
#     """
#     mid0 = (vertices[quads[:, 0]] + vertices[quads[:, 1]]) / 2  # [Q, 3]
#     mid1 = (vertices[quads[:, 1]] + vertices[quads[:, 2]]) / 2  # [Q, 3]
#     mid2 = (vertices[quads[:, 2]] + vertices[quads[:, 3]]) / 2  # [Q, 3]
#     mid3 = (vertices[quads[:, 3]] + vertices[quads[:, 0]]) / 2  # [Q, 3]

#     cross0 = mid2 - mid0  # [Q, 3]
#     cross1 = mid3 - mid1  # [Q, 3]
#     cross0 = cross0 / np.maximum(np.linalg.norm(cross0, axis=-1, keepdims=True), 1e-10)  # [Q, 3]
#     cross1 = cross1 / np.maximum(np.linalg.norm(cross1, axis=-1, keepdims=True), 1e-10)  # [Q, 3]

#     edge0 = vertices[quads[:, 1]] - vertices[quads[:, 0]]  # [Q, 3]
#     edge1 = vertices[quads[:, 2]] - vertices[quads[:, 1]]  # [Q, 3]
#     edge2 = vertices[quads[:, 3]] - vertices[quads[:, 2]]  # [Q, 3]
#     edge3 = vertices[quads[:, 0]] - vertices[quads[:, 3]]  # [Q, 3]
#     edge0 = edge0 / np.maximum(np.linalg.norm(edge0, axis=-1, keepdims=True), 1e-10)  # [Q, 3]
#     edge1 = edge1 / np.maximum(np.linalg.norm(edge1, axis=-1, keepdims=True), 1e-10)  # [Q, 3]
#     edge2 = edge2 / np.maximum(np.linalg.norm(edge2, axis=-1, keepdims=True), 1e-10)  # [Q, 3]
#     edge3 = edge3 / np.maximum(np.linalg.norm(edge3, axis=-1, keepdims=True), 1e-10)  # [Q, 3]

#     direction = np.stack([
#         np.arccos(np.clip(np.sum(cross0 * edge0, axis=-1), -1, 1)),
#         np.arccos(np.clip(np.sum(cross1 * edge1, axis=-1), -1, 1)),
#         np.arccos(np.clip(np.sum(-cross0 * edge2, axis=-1), -1, 1)),
#         np.arccos(np.clip(np.sum(-cross1 * edge3, axis=-1), -1, 1)),
#     ], axis=-1)  # [Q, 4]

#     return direction


# def calc_quad_smoothness(
#     quad2edge: ndarray,
#     quad2adj: ndarray,
#     quads_direction: ndarray,
# ):
#     """
#     Calculate the smoothness of each candidate quad face connection.

#     ## Parameters
#         quad2adj (ndarray): [Q, 8] adjacent quad faces of each quad face
#         quads_direction (ndarray): [Q, 4] direction of each quad face

#     ## Returns
#         smoothness (ndarray): [Q, 8] smoothness of each quad face connection
#     """
#     Q = quad2adj.shape[0]
#     quad2adj_valid = quad2adj != -1
#     connections = np.stack([
#         np.arange(Q)[:, None].repeat(8, axis=1),
#         quad2adj,
#     ], axis=-1)[quad2adj_valid]  # [C, 2]
#     shared_edge_idx_0 = np.array([[0, 0, 1, 1, 2, 2, 3, 3]]).repeat(Q, axis=0)[quad2adj_valid]  # [C]
#     shared_edge_idx_1 = np.argmax(quad2edge[quad2adj][quad2adj_valid] == quad2edge[connections[:, 0], shared_edge_idx_0][:, None], axis=-1)  # [C]
#     valid_smoothness = np.abs(quads_direction[connections[:, 0], shared_edge_idx_0] - quads_direction[connections[:, 1], shared_edge_idx_1])**2  # [C]
#     smoothness = np.zeros([Q, 8], dtype=np.float32)
#     smoothness[quad2adj_valid] = valid_smoothness
#     return smoothness


# def solve_quad(
#     face2edge: ndarray,
#     edge2face: ndarray,
#     quad2adj: ndarray,
#     quads_distortion: ndarray,
#     quads_smoothness: ndarray,
#     quads_valid: ndarray,
# ):
#     """
#     Solve the quad mesh from the candidate quad faces.

#     ## Parameters
#         face2edge (ndarray): [T, 3] face to edge relation
#         edge2face (ndarray): [E, 2] edge to face relation
#         quad2adj (ndarray): [Q, 8] adjacent quad faces of each quad face
#         quads_distortion (ndarray): [Q] distortion of each quad face
#         quads_smoothness (ndarray): [Q, 8] smoothness of each quad face connection
#         quads_valid (ndarray): [E] whether the quad corresponding to the edge is valid

#     ## Returns
#         weights (ndarray): [Q] weight of each valid quad face
#     """
#     import scipy.optimize as opt

#     T = face2edge.shape[0]
#     E = edge2face.shape[0]
#     Q = quads_distortion.shape[0]
#     edge_valid = -np.ones(E, dtype=np.int32)
#     edge_valid[quads_valid] = np.arange(Q)

#     quads_connection = np.stack([
#         np.arange(Q)[:, None].repeat(8, axis=1),
#         quad2adj,
#     ], axis=-1)[quad2adj != -1]  # [C, 2]
#     quads_connection = np.sort(quads_connection, axis=-1)  # [C, 2]
#     quads_connection, quads_connection_idx = np.unique(quads_connection, axis=0, return_index=True)  # [C, 2], [C]
#     quads_smoothness = quads_smoothness[quad2adj != -1]  # [C]
#     quads_smoothness = quads_smoothness[quads_connection_idx]  # [C]
#     C = quads_connection.shape[0]

#     # Construct the linear programming problem

#     # Variables:
#     #   quads_weight: [Q] weight of each quad face
#     #   tri_min_weight: [T] minimum weight of each triangle face
#     #   conn_min_weight: [C] minimum weight of each quad face connection
#     #   conn_max_weight: [C] maximum weight of each quad face connection
#     # Objective:
#     #   mimi

#     c = np.concatenate([
#         quads_distortion - 3,
#         quads_smoothness*4 - 2,
#         quads_smoothness*4,
#     ], axis=0)  # [Q+C]

#     A_ub_triplet = np.concatenate([
#         np.stack([np.arange(T), edge_valid[face2edge[:, 0]], np.ones(T)], axis=1),  # [T, 3]
#         np.stack([np.arange(T), edge_valid[face2edge[:, 1]], np.ones(T)], axis=1),  # [T, 3]
#         np.stack([np.arange(T), edge_valid[face2edge[:, 2]], np.ones(T)], axis=1),  # [T, 3]
#         np.stack([np.arange(T, T+C), np.arange(Q, Q+C), np.ones(C)], axis=1),  # [C, 3]
#         np.stack([np.arange(T, T+C), quads_connection[:, 0], -np.ones(C)], axis=1),  # [C, 3]
#         np.stack([np.arange(T, T+C), quads_connection[:, 1], -np.ones(C)], axis=1),  # [C, 3]
#         np.stack([np.arange(T+C, T+2*C), np.arange(Q+C, Q+2*C), -np.ones(C)], axis=1),  # [C, 3]
#         np.stack([np.arange(T+C, T+2*C), quads_connection[:, 0], np.ones(C)], axis=1),  # [C, 3]
#         np.stack([np.arange(T+C, T+2*C), quads_connection[:, 1], np.ones(C)], axis=1),  # [C, 3]
#     ], axis=0)  # [3T+6C, 3]
#     A_ub_triplet = A_ub_triplet[A_ub_triplet[:, 1] != -1]  # [3T', 3]
#     A_ub = sp.coo_matrix((A_ub_triplet[:, 2], (A_ub_triplet[:, 0], A_ub_triplet[:, 1])), shape=[T+2*C, Q+2*C])  # [T, 
#     b_ub = np.concatenate([np.ones(T), -np.ones(C), np.ones(C)], axis=0)  # [T+2C]
#     bound = np.stack([
#         np.concatenate([np.zeros(Q), -np.ones(C), np.zeros(C)], axis=0),
#         np.concatenate([np.ones(Q), np.ones(C), np.ones(C)], axis=0),
#     ], axis=1)  # [Q+2C, 2]
#     A_eq = None
#     b_eq = None

#     print('Solver statistics:')
#     print(f'    #T = {T}')
#     print(f'    #Q = {Q}')
#     print(f'    #C = {C}')

#     # Solve the linear programming problem
#     last_num_valid = 0
#     for i in range(100):
#         res_ = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bound)
#         if not res_.success:
#             print(f'    Iter {i} | Failed with {res_.message}')
#             break
#         res = res_
#         weights = res.x[:Q]
#         valid = (weights > 0.5)
#         num_valid = valid.sum()
#         print(f'    Iter {i} | #Q_valid = {num_valid}')
#         if num_valid == last_num_valid:
#             break
#         last_num_valid = num_valid
#         A_eq_triplet = np.stack([
#             np.arange(num_valid),
#             np.arange(Q)[valid],
#             np.ones(num_valid),
#         ], axis=1)  # [num_valid, 3]
#         A_eq = sp.coo_matrix((A_eq_triplet[:, 2], (A_eq_triplet[:, 0], A_eq_triplet[:, 1])), shape=[num_valid, Q+2*C])  # [num_valid, Q+C]
#         b_eq = np.where(weights[valid] > 0.5, 1, 0)  # [num_valid]

#     # Return the result
#     quads_weight = res.x[:Q]
#     conn_min_weight = res.x[Q:Q+C]
#     conn_max_weight = res.x[Q+C:Q+2*C]
#     return quads_weight, conn_min_weight, conn_max_weight


# def solve_quad_qp(
#     face2edge: ndarray,
#     edge2face: ndarray,
#     quad2adj: ndarray,
#     quads_distortion: ndarray,
#     quads_smoothness: ndarray,
#     quads_valid: ndarray,
# ):
#     """
#     Solve the quad mesh from the candidate quad faces.

#     ## Parameters
#         face2edge (ndarray): [T, 3] face to edge relation
#         edge2face (ndarray): [E, 2] edge to face relation
#         quad2adj (ndarray): [Q, 8] adjacent quad faces of each quad face
#         quads_distortion (ndarray): [Q] distortion of each quad face
#         quads_smoothness (ndarray): [Q, 8] smoothness of each quad face connection
#         quads_valid (ndarray): [E] whether the quad corresponding to the edge is valid

#     ## Returns
#         weights (ndarray): [Q] weight of each valid quad face
#     """
#     import piqp

#     T = face2edge.shape[0]
#     E = edge2face.shape[0]
#     Q = quads_distortion.shape[0]
#     edge_valid = -np.ones(E, dtype=np.int32)
#     edge_valid[quads_valid] = np.arange(Q)

#     # Construct the quadratic programming problem
#     C_smoothness_triplet = np.stack([
#         np.arange(Q)[:, None].repeat(8, axis=1)[quad2adj != -1],
#         quad2adj[quad2adj != -1],
#         5 * quads_smoothness[quad2adj != -1],
#     ], axis=-1)  # [C, 3]
#     # C_smoothness_triplet = np.concatenate([
#     #     C_smoothness_triplet,
#     #     np.stack([np.arange(Q), np.arange(Q), 20*np.ones(Q)], axis=1),
#     # ], axis=0)  # [C+Q, 3]
#     C_smoothness = sp.coo_matrix((C_smoothness_triplet[:, 2], (C_smoothness_triplet[:, 0], C_smoothness_triplet[:, 1])), shape=[Q, Q])  # [Q, Q]
#     C_smoothness = C_smoothness.tocsc()
#     C_dist = quads_distortion - 20  # [Q]

#     A_eq = sp.coo_matrix((np.zeros(Q), (np.zeros(Q), np.arange(Q))), shape=[1, Q])  # [1, Q]\
#     A_eq = A_eq.tocsc()
#     b_eq = np.array([0])

#     A_ub_triplet = np.concatenate([
#         np.stack([np.arange(T), edge_valid[face2edge[:, 0]], np.ones(T)], axis=1),  # [T, 3]
#         np.stack([np.arange(T), edge_valid[face2edge[:, 1]], np.ones(T)], axis=1),  # [T, 3]
#         np.stack([np.arange(T), edge_valid[face2edge[:, 2]], np.ones(T)], axis=1),  # [T, 3]
#     ], axis=0)  # [3T, 3]
#     A_ub_triplet = A_ub_triplet[A_ub_triplet[:, 1] != -1]  # [3T', 3]
#     A_ub = sp.coo_matrix((A_ub_triplet[:, 2], (A_ub_triplet[:, 0], A_ub_triplet[:, 1])), shape=[T, Q])  # [T, Q]
#     A_ub = A_ub.tocsc()
#     b_ub = np.ones(T)

#     lb = np.zeros(Q)
#     ub = np.ones(Q)

#     solver = piqp.SparseSolver()
#     solver.settings.verbose = True
#     solver.settings.compute_timings = True
#     solver.setup(C_smoothness, C_dist, A_eq, b_eq, A_ub, b_ub, lb, ub)

#     status = solver.solve()

#     # x = cp.Variable(Q)
#     # prob = cp.Problem(
#     #     cp.Minimize(cp.quad_form(x, C_smoothness) + C_dist.T @ x),
#     #     [
#     #         A_ub @ x <= b_ub,
#     #         x >= 0, x <= 1,
#     #     ]
#     # )

#     # # Solve the quadratic programming problem
#     # prob.solve(solver=cp.PIQP, verbose=True)

#     # Return the result
#     weights = solver.result.x
#     return weights


# def tri_to_quad(
#     vertices: ndarray,
#     faces: ndarray, 
# ) -> Tuple[ndarray, ndarray]:
#     """
#     Convert a triangle mesh to a quad mesh.
#     NOTE: The input mesh must be a manifold mesh.

#     ## Parameters
#         vertices (ndarray): [N, 3] 3-dimensional vertices
#         faces (ndarray): [T, 3] triangular face indices

#     ## Returns
#         vertices (ndarray): [N_, 3] 3-dimensional vertices
#         faces (ndarray): [Q, 4] quad face indices
#     """
#     raise NotImplementedError