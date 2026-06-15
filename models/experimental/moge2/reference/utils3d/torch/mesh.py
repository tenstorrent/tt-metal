import torch
from torch import Tensor
import torch.nn.functional as F
from typing import *
from .transforms import angle_between, unproject_cv
from .utils import lookup, csr_eliminate_zeros, csr_matrix_from_dense_indices
from .segment_ops import segment_roll


__all__ = [
    'triangulate_mesh',
    'compute_face_corner_angles',
    'compute_face_corner_normals',
    'compute_face_corner_tangents',
    'compute_face_normals',
    'compute_face_tangents',
    'mesh_edges',
    'mesh_half_edges',
    'mesh_dual_graph',
    'mesh_connected_components',
    'graph_connected_components',
    'compute_boundaries',
    'remove_unused_vertices',
    'remove_corrupted_faces',
    'remove_isolated_pieces',
    'merge_duplicate_vertices',
    'subdivide_mesh',
    'compute_mesh_laplacian',
    'laplacian_smooth_mesh',
    'taubin_smooth_mesh',
    'laplacian_hc_smooth_mesh',
    "create_cube_mesh",
    "create_camera_frustum_mesh",
    "create_icosahedron_mesh",
]


def triangulate_mesh(
    faces: Tensor,
    vertices: Tensor = None,
    method: Literal['fan', 'strip', 'diagonal'] = 'fan'
) -> Tensor:
    """
    Triangulate a polygonal mesh.

    ## Parameters
    - `faces` (Tensor): [L, P] polygonal faces
    - `vertices` (Tensor, optional): [N, 3] 3-dimensional vertices.
        If given, the triangulation is performed according to the distance
        between vertices. Defaults to None.
    - `method`

    ## Returns
        (Tensor): [L * (P - 2), 3] triangular faces
    """
    if faces.shape[-1] == 3:
        return faces
    P = faces.shape[-1]

    if method == 'fan':
        i = torch.arange(P - 2, dtype=torch.int64, device=faces.device)
        loop_indices = torch.stack([torch.zeros_like(i), i + 1, i + 2], dim=1)
        return faces[:, loop_indices].reshape((-1, 3))
    elif method == 'strip':
        i = torch.arange(P - 2, dtype=torch.int64, device=faces.device)
        j = i // 2
        loop_indices = torch.where(
            (i % 2 == 0)[:, None],
            torch.stack([(P - j) % P, j + 1, P - j - 1], dim=1),
            torch.stack([j + 1, j + 2, P - j - 1], dim=1)
        )
        return faces[:, loop_indices].reshape((-1, 3))
    elif method == 'diagonal':
        assert faces.shape[-1] == 4, "Diagonal-aware method is only supported for quad faces"
        assert vertices is not None, "Vertices must be provided for diagonal method"
        backslash = torch.linalg.norm(vertices[faces[:, 0]] - vertices[faces[:, 2]], dim=-1) < \
                        torch.linalg.norm(vertices[faces[:, 1]] - vertices[faces[:, 3]], dim=-1)
        faces = torch.where(
            backslash[:, None],
            faces[:, [0, 1, 2, 0, 2, 3]],
            faces[:, [0, 1, 3, 3, 1, 2]]
        ).reshape((-1, 3))
        return faces


def compute_face_corner_angles(
    vertices: Tensor,
    faces: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute face corner angles of a mesh

    ## Parameters
    - `vertices` (Tensor): `(..., N, 3)` vertices if `faces` is provided, or `(..., F, P, 3)` if `faces` is None
    - `faces` (Tensor, optional): `(F, P)` face vertex indices, where P is the number of vertices per face

    ## Returns
    - `angles` (Tensor): `(..., F, P)` face corner angles
    """
    if faces is not None:
        vertices = vertices.index_select(-2, faces.view(-1)).view(*vertices.shape[:-2], *faces.shape, vertices.shape[-1])   # (..., F, P, 3)
    loop = torch.arange(faces.shape[1])
    edges = vertices[..., faces[:, torch.roll(loop, -1)], :] - vertices[..., faces[:, loop], :]
    angles = angle_between(-torch.roll(edges, 1, dims=-2), edges)
    return angles


def compute_face_corner_normals(
    vertices: Tensor,
    faces: Optional[Tensor] = None,
    normalize: bool = True
) -> Tensor:
    """
    Compute the face corner normals of a mesh

    ## Parameters
    - `vertices` (Tensor): `(..., N, 3)` vertices if `faces` is provided, or `(..., F, P, 3)` if `faces` is None
    - `faces` (Tensor, optional): `(F, P)` face vertex indices, where P is the number of vertices per face
    - `normalize` (bool): whether to normalize the normals to unit vectors. If not, the normals are the raw cross products.

    ## Returns
    - `normals` (Tensor): (..., F, P, 3) face corner normals
    """
    if faces is not None:
        vertices = vertices.index_select(-2, faces.view(-1)).view(*vertices.shape[:-2], *faces.shape, vertices.shape[-1])   # (..., F, P, 3)
    edges = torch.roll(vertices, -1, dim=-2) - vertices  # (..., T, P, 3)
    normals = torch.cross(torch.roll(edges, 1, dims=-2), edges)
    if normalize:
        normals /= torch.linalg.norm(normals, dim=-1, keepdim=True) + torch.finfo(vertices.dtype).eps
    return normals


def compute_face_corner_tangents(
    vertices: Tensor,
    uv: Tensor,
    faces_vertices: Optional[Tensor] = None,
    faces_uv: Optional[Tensor] = None,
    normalize: bool = True
) -> Tensor:
    """
    Compute the face corner tangent (and bitangent) vectors of a mesh

    ## Parameters
    - `vertices` (Tensor): `(..., N, 3)` if `faces` is provided, or `(..., F, P, 3)` if `faces_vertices` is None
    - `uv` (Tensor): `(..., N, 2)` if `faces` is provided, or `(..., F, P, 2)` if `faces_uv` is None
    - `faces_vertices` (Tensor, optional): `(F, P)` face vertex indices
    - `faces_uv` (Tensor, optional): `(F, P)` face UV indices
    - `normalize` (bool): whether to normalize the tangents to unit vectors. If not, the tangents (dX/du, dX/dv) matches the UV parameterized manifold.
s
    ## Returns
    - `tangents` (Tensor): `(..., F, P, 3, 2)` face corner tangents (and bitangents), 
        where the last dimension represents the tangent and bitangent vectors.
    """
    if faces_vertices is not None:
        vertices = vertices.index_select(-2, faces_vertices.view(-1)).view(*vertices.shape[:-2], *faces_vertices.shape, vertices.shape[-1])   # (..., F, P, 3)
    if faces_uv is not None:
        uv = uv.index_select(-2, faces_uv.view(-1)).view(*uv.shape[:-2], *faces_uv.shape, uv.shape[-1])   # (..., F, P, 2)
    
    edge_xyz = torch.roll(vertices, -1, dim=-2) - vertices  # (..., F, P, 3)
    edge_uv = torch.roll(uv, -1, dim=-2) - uv  # (..., F, P, 2)

    tangents = torch.stack([torch.roll(edge_xyz, 1, dim=-2), edge_xyz], dim=-1) \
        @ torch.linalg.inv(torch.stack([torch.roll(edge_uv, 1, dim=-2), edge_uv], dim=-1))
    if normalize:
        tangents /= torch.linalg.norm(tangents, dim=-1, keepdim=True) + torch.finfo(tangents.dtype).eps
    return tangents


def compute_face_normals(
    vertices: Tensor,
    faces: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute face normals of a mesh

    ## Parameters
    - `vertices` (Tensor): `(..., N, 3)` vertices if `faces` is provided, or `(..., F, P, 3)` if `faces` is None
    - `faces` (Tensor, optional): `(F, P)` face vertex indices, where P is the number of vertices per face

    ## Returns
    - `normals` (Tensor): `(..., F, 3)` face normals. Always normalized.
    """
    if faces is not None:
        vertices = vertices.index_select(-2, faces.view(-1)).view(*vertices.shape[:-2], *faces.shape, vertices.shape[-1])   # (..., F, P, 3)
    if vertices.shape[-2] == 3:
        normals = torch.cross(
            vertices[..., 1, :] - vertices[..., 0, :],
            vertices[..., 2, :] - vertices[..., 0, :]
        )
    else:
        normals = compute_face_corner_normals(vertices, normalize=False)
        normals = torch.mean(normals, dim=-2)
    normals /= torch.linalg.norm(normals, dim=-1, keepdim=True) + torch.finfo(vertices.dtype).eps
    return normals


def compute_face_tangents(
    vertices: Tensor,
    uv: Tensor,
    faces_vertices: Optional[Tensor] = None,
    faces_uv: Optional[Tensor] = None,
    normalize: bool = True
) -> Tensor:
    """
    Compute the face corner tangent (and bitangent) vectors of a mesh

    ## Parameters
    - `vertices` (Tensor): `(..., N, 3)` if `faces` is provided, or `(..., F, P, 3)` if `faces_vertices` is None
    - `uv` (Tensor): `(..., N, 2)` if `faces` is provided, or `(..., F, P, 2)` if `faces_uv` is None
    - `faces_vertices` (Tensor, optional): `(F, P)` face vertex indices
    - `faces_uv` (Tensor, optional): `(F, P)` face UV indices

    ## Returns
    - `tangents` (Tensor): `(..., F, 3, 2)` face corner tangents (and bitangents), 
        where the last dimension represents the tangent and bitangent vectors.
    """
    if faces_vertices is not None:
        vertices = vertices.index_select(-2, faces_vertices.view(-1)).view(*vertices.shape[:-2], *faces_vertices.shape, vertices.shape[-1])   # (..., F, P, 3)
    if faces_uv is not None:
        uv = uv.index_select(-2, faces_uv.view(-1)).view(*uv.shape[:-2], *faces_uv.shape, uv.shape[-1])   # (..., F, P, 2)
    if vertices.shape[-2] == 3:
        tangents = torch.stack([vertices[..., 1, :] - vertices[..., 0, :], vertices[..., 2, :] - vertices[..., 0, :]], dim=-1) \
            @ torch.linalg.inv(torch.stack([uv[..., 1, :] - uv[..., 0, :], uv[..., 2, :] - uv[..., 0, :]], dim=-1))
    else:
        tangents = compute_face_corner_tangents(vertices, uv, normalize=False)
        tangents = torch.mean(tangents, dim=-2)
    if normalize:
        tangents /= torch.linalg.norm(tangents, dim=-1, keepdim=True) + torch.finfo(vertices.dtype).eps
    return tangents


def compute_vertex_normals(
    vertices: Tensor,
    faces: Tensor,
    weighted: Literal['uniform', 'area', 'angle'] = 'uniform'
) -> Tensor:
    """Compute vertex normals of a polygon mesh by averaging neighboring face normals

    ## Parameters
        vertices (Tensor): [..., N, 3] 3-dimensional vertices
        faces (Tensor): [T, P] face vertex indices, where P is the number of vertices per face

    ## Returns
        normals (Tensor): [..., N, 3] vertex normals (already normalized to unit vectors)
    """
    face_corner_normals = compute_face_corner_normals(vertices, faces, normalize=False)
    if weighted == 'uniform':
        face_corner_normals = F.normalize(face_corner_normals, p=2, dim=-1)
    elif weighted == 'area':
        pass
    elif weighted == 'angle':
        face_corner_angle = compute_face_corner_angles(vertices, faces)
        face_corner_normals *= face_corner_angle[..., None]
    vertex_normals = torch.index_put(
        torch.zeros_like(vertices, dtype=vertices.dtype),
        (..., faces[..., None], torch.arange(3)),
        face_corner_normals,
        accumulate=True
    )
    vertex_normals = F.normalize(vertex_normals, p=2, dim=-1)
    return vertex_normals

    
def mesh_edges(
    faces: Tensor, 
    return_face2edge: bool = False, 
    return_edge2face: bool = False, 
    return_counts: bool = False
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Get undirected edges of a mesh. Optionally return additional mappings.

    ## Parameters
    - `faces` (Tensor): polygon faces
        - `(F, P)` dense array of indices, where each face has `P` vertices.
        - `(F, V)` binary sparse csr array of indices, each row corresponds to the vertices of a face.
    - `return_face2edge` (bool): whether to return the face to edge mapping
    - `return_edge2face` (bool): whether to return the edge to face mapping
    - `return_counts` (bool): whether to return the counts of edges

    ## Returns
    - `edges` (Tensor): `(E, 2)` unique edges' vertex indices

    If `return_face2edge`, `return_edge2face`, `return_opposite_edge`, or `return_counts` is True, the corresponding outputs will be appended in order:

    - `face2edge` (Tensor): mapping from faces to the indices of edges
        - `(F, P)` if input `faces` is a dense array
        - `(F, E)` if input `faces` is a sparse csr array
    - `edge2face` (Tensor): `(E, F)` binary sparse CSR matrix of edge to face.
    - `counts` (Tensor): `(E,)` counts of each edge
    """
    if faces.is_sparse_csr:
        edges = torch.stack([faces.col_indices(), segment_roll(faces.col_indices(), faces.crow_indices(), -1)], dim=-1) # (nzz, 2)
    else:
        edges = torch.stack([faces, torch.roll(faces, -1, dims=-1)], dim=-1).reshape(-1, 2)    # (F * P, 2)
    
    a, b = edges[:, 0], edges[:, 1]
    edges = torch.stack([torch.minimum(a, b), torch.maximum(a, b)], dim=-1)
    
    requires_inv_map = return_face2edge or return_edge2face
    requires_counts = return_counts or return_edge2face

    unique = torch.unique(edges, return_inverse=requires_inv_map, return_counts=requires_counts, axis=0)

    edges: Tensor = unique[0] if isinstance(unique, tuple) else unique
    if requires_inv_map:
        inv_map: Tensor = unique[1]
    if requires_counts:
        counts: Tensor = unique[-1]

    ret = (edges,)
    if return_face2edge:
        if faces.is_sparse_csr:
            face2edge = torch.sparse_csr_tensor(faces.crow_indices(), inv_map, torch.ones_like(inv_map, dtype=torch.bool), shape=(faces.shape[0], edges.shape[0]))
        else:
            face2edge = inv_map.reshape(faces.shape)
        ret += (face2edge,)
    if return_edge2face:
        if faces.is_sparse_csr:
            lengths = faces.crow_indices()[1:] - faces.crow_indices()[:-1]
            edge2face = Tensor((
                torch.ones_like(inv_map, dtype=torch.bool),
                torch.repeat_interleave(torch.arange(faces.shape[0]), lengths).index_select(0, torch.argsort(inv_map)),
                torch.cat([torch.tensor([0], dtype=counts.dtype, device=counts.device), torch.cumsum(counts, dim=0)]),
            ), shape=(edges.shape[0], faces.shape[0]))
            ret += (edge2face,)
        else:
            edge2face = Tensor((
                torch.ones_like(inv_map, dtype=torch.bool),
                torch.argsort(inv_map) // faces.shape[1],
                torch.cat([torch.tensor([0], dtype=counts.dtype, device=counts.device), torch.cumsum(counts, dim=0)]),
            ), shape=(edges.shape[0], faces.shape[0]))
            ret += (edge2face,)
    if return_counts:
        ret += (counts,)
    return ret[0] if len(ret) == 1 else ret


def mesh_half_edges(
    faces: Tensor, 
    return_face2edge: bool = False, 
    return_edge2face: bool = False, 
    return_twin: bool = False,
    return_next: bool = False,
    return_prev: bool = False,
    return_counts: bool = False
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Get half edges of a mesh. Optionally return additional mappings.

    ## Parameters
    - `faces` (Tensor): polygon faces
        - `(F, P)` dense array of indices, where each face has `P` vertices.
        - `(F, V)` binary sparse csr array of indices, each row corresponds to the vertices of a face.
    - `return_face2edge` (bool): whether to return the face to edge mapping
    - `return_edge2face` (bool): whether to return the edge to face mapping
    - `return_twin` (bool): whether to return the mapping from one edge to its opposite/twin edge
    - `return_next` (bool): whether to return the mapping from one edge to its next edge in the face loop
    - `return_prev` (bool): whether to return the mapping from one edge to its previous edge in the face loop
    - `return_counts` (bool): whether to return the counts of edges

    ## Returns
    - `edges` (Tensor): `(E, 2)` unique edges' vertex indices

    If `return_face2edge`, `return_edge2face`, `return_opposite_edge`, or `return_counts` is True, the corresponding outputs will be appended in order:

    - `face2edge` (Tensor | Tensor): mapping from faces to the indices of edges
        - `(F, P)` if input `faces` is a dense array
        - `(F, E)` if input `faces` is a sparse csr array
    - `edge2face` (Tensor): `(E, F)` binary sparse CSR matrix of edge to face.
    - `twin` (Tensor): `(E,)` mapping from edges to indices of opposite edges. -1 if not found. 
    - `next` (Tensor): `(E,)` mapping from edges to indices of next edges in the face loop.
    - `prev` (Tensor): `(E,)` mapping from edges to indices of previous edges in the face loop.
    - `counts` (Tensor): `(E,)` counts of each half edge

    NOTE: If the mesh is not manifold, `twin`, `next`, and `prev` can point to arbitrary one of the candidates.
    """
    if isinstance(faces, Tensor):
        edges = torch.stack([faces.indices, segment_roll(faces.indices, faces.crow_indices(), -1)], dim=-1) # (nzz, 2)
    else:
        edges = torch.stack([faces, torch.roll(faces, -1, dims=-1)], dim=-1).reshape(-1, 2)    # (F * P, 2)
    
    requires_inv_map = return_face2edge or return_edge2face or return_next or return_prev
    requires_counts = return_counts or return_edge2face

    unique = torch.unique(edges, return_inverse=requires_inv_map, return_counts=requires_counts, axis=0)

    edges: Tensor = unique[0] if isinstance(unique, tuple) else unique
    if requires_inv_map:
        inv_map: Tensor = unique[1]
    if requires_counts:
        counts: Tensor = unique[-1]

    ret = (edges,)
    if return_face2edge or return_next or return_prev:
        if faces.is_sparse_csr:
            face2edge = Tensor((torch.ones_like(inv_map, dtype=torch.bool), inv_map, faces.crow_indices()), shape=(faces.shape[0], edges.shape[0]))
        else:
            face2edge = inv_map.reshape(faces.shape)
        if return_face2edge:
            ret += (face2edge,)
    if return_edge2face:
        if faces.is_sparse_csr:
            lengths = faces.crow_indices()[1:] - faces.crow_indices()[:-1]
            edge2face = Tensor((
                torch.ones_like(inv_map, dtype=torch.bool),
                torch.repeat_interleave(torch.arange(faces.shape[0]), lengths).index_select(0, torch.argsort(inv_map)),
                torch.cat([torch.tensor([0], dtype=counts.dtype, device=counts.device), torch.cumsum(counts, dim=0)]),
            ), shape=(edges.shape[0], faces.shape[0]))
            ret += (edge2face,)
        else:
            edge2face = Tensor((
                torch.ones_like(inv_map, dtype=torch.bool),
                torch.argsort(inv_map) // faces.shape[1],
                torch.cat([torch.tensor([0], dtype=counts.dtype, device=counts.device), torch.cumsum(counts, dim=0)]),
            ), shape=(edges.shape[0], faces.shape[0]))
            ret += (edge2face,)
    if return_twin:
        twin_edge = lookup(edges, torch.flip(edges, [-1]))
        ret += (twin_edge,)
    if return_next or return_prev:
        if face2edge.is_sparse_csr:
            face2edge_indices = face2edge.col_indices()
            face2edge_indices_next = segment_roll(face2edge.col_indices(), face2edge.crow_indices(), -1)
        else:
            face2edge_indices = face2edge.reshape(-1)
            face2edge_indices_next = torch.roll(face2edge, -1, dims=-1).reshape(-1)
    if return_next:
        next_edge = torch.full(edges.shape[0], -1, dtype=torch.int32, device=edges.device)
        torch.index_put_(next_edge, face2edge_indices, face2edge_indices_next)
        ret += (next_edge,)
    if return_prev:
        prev_edge = torch.full(edges.shape[0], -1, dtype=torch.int32, device=edges.device)
        torch.index_put_(prev_edge, face2edge_indices_next, face2edge_indices)
        ret += (prev_edge,)
    if return_counts:
        ret += (counts,)
    return ret[0] if len(ret) == 1 else ret


def mesh_connected_components(faces: Tensor, num_vertices: Optional[int] = None) -> List[Tensor]:
    """
    Compute connected components of a mesh.

    ## Parameters
    - `faces` (Tensor): polygon faces
        - `(F, P)` dense tensor of indices, where each face has `P` vertices.
        - `(F, V)` binary sparse csr tensor of indices, each row corresponds to the vertices of a face.
    - `num_vertices` (int, optional): total number of vertices. If given, the returned components will include all vertices. Defaults to None.

    ## Returns

    If `num_vertices` is given, return:
    - `labels` (Tensor): (N,) component labels of each vertex

    If `num_vertices` is None, return:
    - `vertices_ids` (Tensor): (N,) vertex indices that are in the edges
    - `labels` (Tensor): (N,) component labels corresponding to `vertices_ids`
    """
    edges = mesh_edges(faces, directed=False)
    return graph_connected_components(edges, num_vertices)


def graph_connected_components(edges: Tensor, num_vertices: Optional[int] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Compute connected components of an undirected graph.

    ## Parameters
    - `edges` (Tensor): (E, 2) edge indices

    ## Returns

    If `num_vertices` is given, return:
    - `labels` (Tensor): (N,) component labels of each vertex

    If `num_vertices` is None, return:
    - `vertices_ids` (Tensor): (N,) vertex indices that are in the edges
    - `labels` (Tensor): (N,) component labels corresponding to `vertices_ids`
    """
    if num_vertices is None:
        # Re-index edges
        vertices_ids, edges = torch.unique(edges.flatten(), return_inverse=True)
        edges = edges.view(-1, 2)
        labels = torch.arange(vertices_ids.shape[0], dtype=torch.int32, device=edges.device)
    else:
        labels = torch.arange(num_vertices, dtype=torch.int32, device=edges.device)
    
    # Make edges undirected
    edges = torch.cat([edges, edges.flip(-1)], dim=0)
    src, dst = edges.unbind(-1)

    # Loop until convergence
    while True:
        labels = labels.index_select(0, labels)
        new_labels = labels.scatter_reduce(
            dim=0, 
            index=dst, 
            src=labels.index_select(0, src), 
            reduce='amin', 
            include_self=True
        )
        if torch.equal(labels, new_labels):
            break
        labels = new_labels

    if num_vertices is None:
        return vertices_ids, labels
    else:
        return labels



def compute_boundaries(
    faces: Tensor,
    edges: Tensor = None,
    face2edge: Tensor = None,
    edge_degrees: Tensor=None
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Compute boundary edges of a mesh.

    ## Parameters
        faces (Tensor): [T, 3] triangular face indices
        edges (Tensor): [E, 2] edge indices.
        face2edge (Tensor): [T, 3] mapping from face to edge.
        edge_degrees (Tensor): [E] degree of each edge.

    ## Returns
        boundary_edge_indices (List[Tensor]): list of boundary edge indices
        boundary_face_indices (List[Tensor]): list of boundary face indices
    """    
    # Map each edge to boundary edge index
    boundary_edges = edges[edge_degrees == 1]                                                                     # [BE, 2]
    boundary_edges_idx = torch.nonzero(edge_degrees == 1, as_tuple=False).flatten()                               # [BE]
    E = edges.shape[0]                                                                                             # Edge count
    BE = boundary_edges.shape[0]                                                                            # Boundary edge count
    map_to_boundary_edges = torch.full((E,), -1, dtype=torch.int32, device=faces.device)                    # [E]
    map_to_boundary_edges[boundary_edges_idx] = torch.arange(BE, dtype=torch.int32, device=faces.device)
    
    # Re-index boundary vertices
    boundary_vertices, boundary_edges = torch.unique(boundary_edges.flatten(), return_inverse=True)
    boundary_edges = boundary_edges.view(-1, 2)
    BV = boundary_vertices.shape[0]
    
    boundary_edge_labels = torch.arange(BE, dtype=torch.int32, device=faces.device)
    while True:
        boundary_vertex_labels = torch.scatter_reduce(
            torch.zeros(BV, dtype=torch.int32, device=faces.device),
            0,
            boundary_edges.flatten().long(),
            boundary_edge_labels.view(-1, 1).expand(-1, 2).flatten(),
            reduce='amin',
            include_self=False
        )
        new_boundary_edge_labels = torch.min(boundary_vertex_labels[boundary_edges], dim=-1).values
        if torch.equal(boundary_edge_labels, new_boundary_edge_labels):
            break
        boundary_edge_labels = new_boundary_edge_labels
        
    labels = torch.unique(boundary_edge_labels)
    boundary_edge_indices = [boundary_edges_idx[boundary_edge_labels == label] for label in labels]
    edge_labels = torch.full((E,), -1, dtype=torch.int32, device=faces.device)
    edge_labels[boundary_edges_idx] = boundary_edge_labels
    boundary_face_indices = [torch.nonzero((edge_labels[face2edge] == label).any(dim=-1), as_tuple=False).flatten() for label in labels]
    
    return boundary_edge_indices, boundary_face_indices


def mesh_dual_graph(faces: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Get dual graph of a mesh. (Mesh face as dual graph's vertex, adjacency by edge sharing)
    
    ## Parameters
    - `faces`: `Tensor` faces indices
        - `(F, P)` dense tensor 

    ## Returns
    - `dual_graph` (Tensor): `(F, F)` binary sparse CSR matrix. Adjacency matrix of the dual graph.
    """
    device = faces.device
    edges, face2edge = mesh_edges(faces, directed=False, return_face2edge=True)
    if not face2edge.is_sparse_csr:
        face2edge = csr_matrix_from_dense_indices(face2edge, n_cols=len(edges))
    dual_graph = (face2edge.float() @ face2edge.float().transpose(-2, -1)).bool()
    neg_diag = torch.sparse.spdiags(torch.full((dual_graph.shape[0],), -1, dtype=torch.float32, device=device), torch.tensor(0, device=device), dual_graph.shape, layout=torch.sparse_csr)
    dual_graph = dual_graph.float() + neg_diag
    dual_graph = csr_eliminate_zeros(dual_graph).bool()
    return dual_graph


def remove_unused_vertices(
    faces: Tensor,
    *vertice_attrs,
    return_indices: bool = False
) -> Tuple[Tensor, ...]:
    """
    Remove unreferenced vertices of a mesh. 
    Unreferenced vertices are removed, and the face indices are updated accordingly.

    ## Parameters
        faces (Tensor): [T, P] face indices
        *vertice_attrs: vertex attributes

    ## Returns
        faces (Tensor): [T, P] face indices
        *vertice_attrs: vertex attributes
        indices (Tensor, optional): [N] indices of vertices that are kept. Defaults to None.
    """
    P = faces.shape[-1]
    fewer_indices, inv_map = torch.unique(faces, return_inverse=True)
    faces = inv_map.to(torch.int32).reshape(-1, P)
    ret = [faces]
    for attr in vertice_attrs:
        ret.append(attr[fewer_indices])
    if return_indices:
        ret.append(fewer_indices)
    return tuple(ret)


def remove_corrupted_faces(
    faces: Tensor
) -> Tensor:
    """
    Remove corrupted faces (faces with duplicated vertices)

    ## Parameters
        faces (Tensor): [F, 3] face indices

    ## Returns
        Tensor: [F_reduced, 3] face indices
    """
    corrupted = (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 2] == faces[:, 0])
    return faces[~corrupted]


def merge_duplicate_vertices(
    vertices: Tensor,
    faces: Tensor,
    tol: float = 1e-6
) -> Tuple[Tensor, Tensor]:
    """
    Merge duplicate vertices of a triangular mesh. 
    Duplicate vertices are merged by selecte one of them, and the face indices are updated accordingly.

    ## Parameters
        vertices (Tensor): [N, 3] 3-dimensional vertices
        faces (Tensor): [T, 3] triangular face indices
        tol (float, optional): tolerance for merging. Defaults to 1e-6.

    ## Returns
        vertices (Tensor): [N_, 3] 3-dimensional vertices
        faces (Tensor): [T, 3] triangular face indices
    """
    vertices_round = torch.round(vertices / tol)
    uni, uni_inv = torch.unique(vertices_round, dim=0, return_inverse=True)
    uni[uni_inv] = vertices
    faces = uni_inv[faces]
    return uni, faces


def remove_isolated_pieces(
    vertices: Tensor,
    faces: Tensor,
    connected_components: List[Tensor] = None,
    thresh_num_faces: int = None,
    thresh_radius: float = None,
    thresh_boundary_ratio: float = None,
    remove_unreferenced: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Remove isolated pieces of a mesh. 
    Isolated pieces are removed, and the face indices are updated accordingly.
    If no face is left, will return the largest connected component.

    ## Parameters
        vertices (Tensor): [N, 3] 3-dimensional vertices
        faces (Tensor): [T, 3] triangular face indices
        connected_components (List[Tensor], optional): connected components of the mesh. If None, it will be computed. Defaults to None.
        thresh_num_faces (int, optional): threshold of number of faces for isolated pieces. Defaults to None.
        thresh_radius (float, optional): threshold of radius for isolated pieces. Defaults to None.
        remove_unreferenced (bool, optional): remove unreferenced vertices after removing isolated pieces. Defaults to True.

    ## Returns
        vertices (Tensor): [N_, 3] 3-dimensional vertices
        faces (Tensor): [T, 3] triangular face indices
    """
    if connected_components is None:
        connected_components = compute_connected_components(faces)
    connected_components = sorted(connected_components, key=lambda x: len(x), reverse=True)
    if thresh_num_faces is not None:
        removed = []
        for i in range(1, len(connected_components)):
            if len(connected_components[i]) < thresh_num_faces:
                removed.append(i)
        for i in removed[::-1]:
            connected_components.pop(i)
    if thresh_radius is not None:
        removed = []
        for i in range(1, len(connected_components)):
            comp_vertices = vertices[faces[connected_components[i]].flatten().unique()]
            comp_center = comp_vertices.mean(dim=0)
            comp_radius = (comp_vertices - comp_center).norm(p=2, dim=-1).max()
            if comp_radius < thresh_radius:
                removed.append(i)
        for i in removed[::-1]:
            connected_components.pop(i)
    if thresh_boundary_ratio is not None:
        removed = []
        for i in range(1, len(connected_components)):
            edges = torch.cat([faces[connected_components[i]][:, [0, 1]], faces[connected_components[i]][:, [1, 2]], faces[connected_components[i]][:, [2, 0]]], dim=0)
            edges = torch.sort(edges, dim=1).values
            edges, counts = torch.unique(edges, return_counts=True, dim=0)
            num_boundary_edges = (counts == 1).sum().item()
            num_faces = len(connected_components[i])
            if num_boundary_edges / num_faces > thresh_boundary_ratio:
                removed.append(i)
        for i in removed[::-1]:
            connected_components.pop(i)
    
    # post-process
    faces = torch.cat([faces[connected_components[i]] for i in range(len(connected_components))], dim=0)
    if remove_unreferenced:
        faces, vertices = remove_unused_vertices(faces, vertices)
    return vertices, faces


def subdivide_mesh(vertices: Tensor, faces: Tensor, n: int = 1) -> Tuple[Tensor, Tensor]:
    """
    Subdivide a triangular mesh by splitting each triangle into 4 smaller triangles.
    NOTE: All original vertices are kept, and new vertices are appended to the end of the vertex list.
    
    ## Parameters
        vertices (Tensor): [N, 3] 3-dimensional vertices
        faces (Tensor): [T, 3] triangular face indices
        n (int, optional): number of subdivisions. Defaults to 1.

    ## Returns
        vertices (Tensor): [N_, 3] subdivided 3-dimensional vertices
        faces (Tensor): [4 * T, 3] subdivided triangular face indices
    """
    for _ in range(n):
        edges = torch.stack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
        edges = torch.sort(edges, dim=2)
        uni_edges, uni_inv = torch.unique(edges, return_inverse=True, dim=0)
        midpoints = (vertices[uni_edges[:, 0]] + vertices[uni_edges[:, 1]]) / 2

        n_vertices = vertices.shape[0]
        vertices = torch.cat([vertices, midpoints], dim=0)
        faces = torch.cat([
            torch.stack([faces[:, 0], n_vertices + uni_inv[0], n_vertices + uni_inv[2]], dim=1),
            torch.stack([faces[:, 1], n_vertices + uni_inv[1], n_vertices + uni_inv[0]], dim=1),
            torch.stack([faces[:, 2], n_vertices + uni_inv[2], n_vertices + uni_inv[1]], dim=1),
            torch.stack([n_vertices + uni_inv[0], n_vertices + uni_inv[1], n_vertices + uni_inv[2]], dim=1),
        ], dim=0)
    return vertices, faces


def compute_mesh_laplacian(vertices: Tensor, faces: Tensor, weight: str = 'uniform') -> Tensor:
    """Laplacian smooth with cotangent weights

    ## Parameters
        vertices (Tensor): shape (..., N, 3)
        faces (Tensor): shape (T, 3)
        weight (str): 'uniform' or 'cotangent'
    """
    sum_verts = torch.zeros_like(vertices)                          # (..., N, 3)
    sum_weights = torch.zeros(*vertices.shape[:-1]).to(vertices)    # (..., N)
    face_verts = torch.index_select(vertices, -2, faces.view(-1)).view(*vertices.shape[:-2], *faces.shape, vertices.shape[-1])   # (..., T, 3)
    if weight == 'cotangent':
        for i in range(3):
            e1 = face_verts[..., (i + 1) % 3, :] - face_verts[..., i, :]
            e2 = face_verts[..., (i + 2) % 3, :] - face_verts[..., i, :]
            cot_angle = (e1 * e2).sum(dim=-1) / torch.cross(e1, e2, dim=-1).norm(p=2, dim=-1)   # (..., T, 3)
            sum_verts = torch.index_add(sum_verts, -2, faces[:, (i + 1) % 3], face_verts[..., (i + 2) % 3, :] * cot_angle[..., None])
            sum_weights = torch.index_add(sum_weights, -1, faces[:, (i + 1) % 3], cot_angle)
            sum_verts = torch.index_add(sum_verts, -2, faces[:, (i + 2) % 3], face_verts[..., (i + 1) % 3, :] * cot_angle[..., None])
            sum_weights = torch.index_add(sum_weights, -1, faces[:, (i + 2) % 3], cot_angle)
    elif weight == 'uniform':
        for i in range(3):
            sum_verts = torch.index_add(sum_verts, -2, faces[:, i], face_verts[..., (i + 1) % 3, :])
            sum_weights = torch.index_add(sum_weights, -1, faces[:, i], torch.ones_like(face_verts[..., i, 0]))
    else:
        raise NotImplementedError
    return sum_verts / (sum_weights[..., None] + 1e-7)


def laplacian_smooth_mesh(vertices: Tensor, faces: Tensor, weight: str = 'uniform', times: int = 5) -> Tensor:
    """Laplacian smooth with cotangent weights

    ## Parameters
        vertices (Tensor): shape (..., N, 3)
        faces (Tensor): shape (T, 3)
        weight (str): 'uniform' or 'cotangent'
    """
    for _ in range(times):
        vertices = laplacian(vertices, faces, weight)
    return vertices


def taubin_smooth_mesh(vertices: Tensor, faces: Tensor, lambda_: float = 0.5, mu_: float = -0.51) -> Tensor:
    """Taubin smooth mesh

    ## Parameters
        vertices (Tensor): _description_
        faces (Tensor): _description_
        lambda_ (float, optional): _description_. Defaults to 0.5.
        mu_ (float, optional): _description_. Defaults to -0.51.

    ## Returns
        Tensor: _description_
    """
    pt = vertices + lambda_ * laplacian_smooth_mesh(vertices, faces)
    p = pt + mu_ * laplacian_smooth_mesh(pt, faces)
    return p


def laplacian_hc_smooth_mesh(vertices: Tensor, faces: Tensor, times: int = 5, alpha: float = 0.5, beta: float = 0.5, weight: str = 'uniform'):
    """HC algorithm from Improved Laplacian Smoothing of Noisy Surface Meshes by J.Vollmer et al.
    """
    p = vertices
    for i in range(times):
        q = p
        p = laplacian_smooth_mesh(vertices, faces, weight)
        b = p - (alpha * vertices + (1 - alpha) * q)
        p = p - (beta * b + (1 - beta) * laplacian_smooth_mesh(b, faces, weight)) * 0.8
    return p


def create_cube_mesh(tri: bool = False, device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
    """
    Create a cube mesh of size 1 centered at origin.

    ### Parameters
        tri (bool, optional): return triangulated mesh. Defaults to False, which returns quad mesh.

    ### Returns
        vertices (Tensor): shape (8, 3) float32
        faces (Tensor): shape (12, 3) int32
    """
    vertices = torch.tensor([
        [0.5, 0.5, 0.5],   [-0.5, 0.5, 0.5],   [-0.5, -0.5, 0.5],   [0.5, -0.5, 0.5], # v0-v1-v2-v3
        [0.5, 0.5, -0.5],  [-0.5, 0.5, -0.5],  [-0.5, -0.5, -0.5],  [0.5, -0.5, -0.5] # v4-v5-v6-v7
    ], dtype=torch.float32, device=device).reshape((-1, 3))

    faces = torch.tensor([
        [0, 1, 2, 3], #  (front)
        [5, 4, 7, 6], #  (back)
        [4, 5, 1, 0], #  (top)
        [2, 6, 7, 3], #  (bottom)
        [1, 5, 6, 2], #  (left)
        [4, 0, 3, 7], #  (right)
    ], dtype=torch.int32, device=device)

    if tri:
        faces = triangulate_mesh(faces, vertices=vertices)

    return vertices, faces


def create_camera_frustum_mesh(extrinsics: Tensor, intrinsics: Tensor, depth: float = 1.0) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Create a triangle mesh of camera frustum.
    """
    assert extrinsics.shape == (4, 4) and intrinsics.shape == (3, 3)
    vertices = unproject_cv(
        torch.tensor([[0, 0], [0, 0], [0, 1], [1, 1], [1, 0]], dtype=extrinsics.dtype, device=extrinsics.device), 
        torch.tensor([0] + [depth] * 4, dtype=extrinsics.dtype, device=extrinsics.device), 
        intrinsics,
        extrinsics
    )
    edges = torch.tensor([
        [0, 1], [0, 2], [0, 3], [0, 4], 
        [1, 2], [2, 3], [3, 4], [4, 1]
    ], dtype=torch.int32, device=extrinsics.device)
    faces = torch.tensor([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
        [1, 2, 3],
        [1, 3, 4]
    ], dtype=torch.int32, device=extrinsics.device)
    return vertices, edges, faces


def create_icosahedron_mesh(device: Optional[torch.device] = None) -> Tuple[Tensor, Tensor]:
    """
    Create an icosahedron mesh of centered at origin.
    """
    A = (1 + 5 ** 0.5) / 2
    vertices = torch.tensor([
        [0, 1, A], [0, -1, A], [0, 1, -A], [0, -1, -A],
        [1, A, 0], [-1, A, 0], [1, -A, 0], [-1, -A, 0],
        [A, 0, 1], [A, 0, -1], [-A, 0, 1], [-A, 0, -1]
    ], dtype=torch.float32, device=device)
    faces = torch.tensor([
        [0, 1, 8], [0, 8, 4], [0, 4, 5], [0, 5, 10], [0, 10, 1],
        [3, 2, 9], [3, 9, 6], [3, 6, 7], [3, 7, 11], [3, 11, 2],
        [1, 6, 8], [8, 9, 4], [4, 2, 5], [5, 11, 10], [10, 7, 1],
        [2, 4, 9], [9, 8, 6], [6, 1, 7], [7, 10, 11], [11, 5, 2]
    ], dtype=torch.int32, device=device)
    return vertices, faces
