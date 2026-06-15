from io import TextIOWrapper
from typing import *
from itertools import chain, repeat
from pathlib import Path
import os

import numpy as np
from numpy import ndarray


__all__ = [
    'read_obj', 
    'write_obj', 
]


class WavefrontOBJDict(TypedDict):
    "Typing for wavefront .obj file content"
    mtllib: List[str]
    "Material library filenames"
    v: ndarray
    "Vertex coordinates"
    vt: ndarray
    "Vertex texture coordinates"
    vn: ndarray
    "Vertex normals"
    f: Union[ndarray, Tuple[ndarray, ndarray]]
    "Vertex indices of faces. NOTE: 0-based indexing"
    ft: Union[ndarray, Tuple[ndarray, ndarray]]
    "Vertex texture coordinate indices of faces. NOTE: 0-based indexing"
    fn: Union[ndarray, Tuple[ndarray, ndarray]]
    "Vertex normal indices of faces. NOTE: 0-based indexing"
    l: Union[ndarray, Tuple[ndarray, ndarray]]
    "Vertex indices of lines. NOTE: 0-based indexing"
    lt: Union[ndarray, Tuple[ndarray, ndarray]]
    "Vertex texture coordinate indices of lines. NOTE: 0-based indexing"
    p: ndarray
    "Vertex indices of points. NOTE: 0-based indexing"
    pt: ndarray
    "Vertex texture coordinate indices of points. NOTE: 0-based indexing"
    o: Dict[str, Dict[Literal['f', 'l', 'p'], slice]]
    "Object names and corresponding slices of faces/lines/points"
    g: Dict[str, Dict[Literal['f', 'l', 'p'], slice]]
    "Group names and corresponding slices of faces/lines/points"
    usemtl: Dict[str, Dict[Literal['f', 'l', 'p'], slice]]
    "Material names and corresponding slices of faces/lines/points"


def read_obj(
    file : Union[str, Path, TextIOWrapper],
    encoding: Union[str, None] = None,
    ignore_unknown: bool = False
) -> WavefrontOBJDict:
    """
    Read wavefront .obj file.

    Parameters
    ----
    - `file` (str, Path, TextIOWrapper): filepath or file object
    - `encoding` (str, optional): file encoding
    - `ignore_unknown` (bool): whether to ignore unknown keywords in .obj file. Default to False.

    Returns
    ----
    A dictionary maybe containing the following fields. Note that some fields may be absent if not present in the .obj file:

    Vertices and attributes data:
    - `v` (ndarray): (N_v, 3 or 4) vertex coordinates.
    - `vt` (ndarray): (N_vt, 2 or 3). vertex texture coordinates.
    - `vn` (ndarray): (N_vn, 3). vertex normals.

    Primitive definitions (face/line). NOTE: all indices are 0-based, unlike the 1-based indexing in .obj file.
    - `f` (ndarray): Vertex indices in each face.
        - If all faces have the same number of vertices: (M, K)
        - If faces have different numbers of vertices: a tuple of segmented array:
            - `data` (ndarray): flattened array of shape (sum(K_i),)
            - `offsets` (ndarray): shape (M + 1,), offsets of each face in the flattened array
    - `ft` (ndarray): Vertex texture coordinate indices of each face. The same format as `f`.
    - `fn` (ndarray): Vertex normal indices of each face. The same format as `f`.
    - `l` (ndarray): Vertex indices of each line segment.
        - If all lines have the same number of vertices: (L, K_line)
        - If lines have different numbers of vertices: a tuple of segmented array:
            - `data` (ndarray): flattened array of shape (sum(K_line_i),)
            - `offsets` (ndarray): shape (L + 1,), offsets of each line in the flattened array
    - `lt` (ndarray): Vertex texture coordinate indices of each line. The same format as `l`.
    - `p` (ndarray): Vertex indices of each point. 1D array of shape (N_p,)
    - `pt` (ndarray): Vertex texture coordinate indices of each point. The same format as `p`.

    Object/Group/Material info:
    - `o` (Dict[str, Dict[Literal['f', 'l', 'p'], slice]]). The objects and their corresponding faces/lines/points. E.g. `o['object_A']['f']` gives the slice of faces belonging to object_A.
    - `g` (Dict[str, Dict[Literal['f', 'l', 'p'], slice]]). The groups and their corresponding faces/lines/points. E.g. `g['group_A']['f']` gives the slice of faces belonging to group_A.
    - `usemtl` (Dict[str, Dict[Literal['f', 'l', 'p'], slice]]). The materials and their corresponding faces/lines/points. E.g. `usemtl['material_A']['f']` gives the slice of faces using material_A.

    Here, `names` is a list of names, and `offsets` is an array of shape (num_entities + 1,) indicating the start and end indices of faces belonging to each object/group/material.
    For example, the second material has name `material_names[1]`, and its faces are `f[material_offsets[1]:material_offsets[2]]`.

    Material library:
    - `mtllib` (List[str]): list of material library filenames - `mtllib` lines in .obj file.
    """
    if hasattr(file,'read'):
        lines = file.read().splitlines()
    else:
        with open(file, 'r', encoding=encoding) as fp:
            lines = fp.read().splitlines()
    mtllib = []
    v, vt, vn = [], [], []          # Vertex coordinates, Vertex texture coordinate, Vertex normal
    f, ft, fn = [], [], []          # Face indices, Face texture indices, Face normal indices
    l, lt = [], []                  # Line indices, Line texture indices
    p, pt = [], []                  # Point indices, Point texture indices
    o = {}
    o_names, o_offsets = [], []
    g = {}
    g_names, g_offsets = [], []
    usemtl = {}
    usemtl_names, usemtl_offsets = [], []

    int_ = lambda x: int(x) if x != '' else 0
    pad = lambda l, n, x = 0: l + [x] * (n - len(l))
    
    # Read the lines
    for i, line in enumerate(lines):
        keyword, *seq = line.strip().split()
        if len(seq) == 0: 
            continue
        if keyword == 'v':
            assert 3 <= len(seq) <= 4, f'Invalid format of line {i}: {line}'
            v.append([float(e) for e in seq])
        elif keyword == 'vt':
            assert 3 <= len(seq) <= 4, f'Invalid format of line {i}: {line}'
            vt.append([float(e) for e in seq])
        elif keyword == 'vn':
            assert len(seq) == 3, f'Invalid format of line {i}: {line}'
            vn.append([float(e) for e in seq])
        elif keyword == 'f':
            elems = [pad(list(map(int_, item.split('/'))), 3) for item in seq]
            f.append([e[0] for e in elems])
            ft.append([e[1] for e in elems])
            fn.append([e[2] for e in elems])
        elif keyword == 'l':
            elems = [pad(list(map(int_, item.split('/'))), 2) for item in seq]
            l.append([e[0] for e in elems])
            lt.append([e[1] for e in elems])
        elif keyword == 'p':
            elems = [pad(list(map(int_, item.split('/'))), 2) for item in seq]
            p.extend([e[0] for e in elems])
            pt.extend([e[1] for e in elems])
        elif keyword == 'o':
            assert len(seq) == 1, f'Invalid format of line {i}: {line}'
            o_names.append(seq[0])
            o_offsets.append((len(f), len(l), len(p)))
        elif keyword == 'g':
            assert len(seq) == 1, f'Invalid format of line {i}: {line}'
            g_names.append(seq[0])
            g_offsets.append((len(f), len(l), len(p)))
        elif keyword == 'usemtl':
            assert len(seq) == 1, f'Invalid format of line {i}: {line}'
            usemtl_names.append(seq[0])
            usemtl_offsets.append((len(f), len(l), len(p)))
        elif keyword == 'mtllib':
            mtllib.extend(seq)
        elif keyword[0] == '#':
            continue
        else:
            if not ignore_unknown:
                raise Exception(f'Unknown keyword {keyword} in line {i}: {line}')

    # Process vertex data
    if len(v) > 0:
        max_v_dim = max(map(len, v)) if len(v) > 0 else 0
        v = np.array([v_ + [1.0] * (max_v_dim - len(v_)) for v_ in v], dtype=np.float32)
    else:
        v = None
    
    if len(vt) > 0:
        max_vt_dim = max(map(len, vt)) if len(vt) > 0 else 0
        vt = np.array([vt_ + [0.0] * (max_vt_dim - len(vt_)) for vt_ in vt], dtype=np.float32)
    else:
        vt = None
    
    if len(vn) > 0:
        vn = np.array(vn, dtype=np.float32)
    else:
        vn = None

    # Process face data
    if len(f) > 0:          
        f_lengths = np.array(list(map(len, f)))  
        f = np.array(list(chain.from_iterable(f)), dtype=np.int32) - 1
        ft = np.array(list(chain.from_iterable(ft)), dtype=np.int32) - 1
        fn = np.array(list(chain.from_iterable(fn)), dtype=np.int32) - 1

        # If all indices are -1, set to None
        if (f == -1).all():
            f = None
        if (ft == -1).all():
            ft = None
        if (fn == -1).all():
            fn = None

        if (f_lengths == f_lengths[0]).all():
            f_len = f_lengths[0]
            f = f.reshape(-1, f_len) if f is not None else None
            ft = ft.reshape(-1, f_len) if ft is not None else None
            fn = fn.reshape(-1, f_len) if fn is not None else None
        else:
            f_offsets = np.concatenate([[0], np.cumsum(f_lengths)])
            f = (f, f_offsets) if f is not None else None
            ft = (ft, f_offsets) if ft is not None else None
            fn = (fn, f_offsets) if fn is not None else None

    else:
        f = None
        ft = None
        fn = None

    # Process line data
    if len(l) > 0:
        l_lengths = np.array(list(map(len, l)))
        l = np.array(list(chain.from_iterable(l)), dtype=np.int32) - 1
        lt = np.array(list(chain.from_iterable(lt)), dtype=np.int32) - 1
        
        if (l == -1).all():
            l = None
        if (lt == -1).all():
            lt = None

        if (l_lengths == l_lengths[0]).all():
            l_len = l_lengths[0]
            l = l.reshape(-1, l_len) if l is not None else None
            lt = lt.reshape(-1, l_len) if lt is not None else None
        else:
            l_offsets = np.concatenate([[0], np.cumsum(l_lengths)])
            l = (l, l_offsets) if l is not None else None
            lt = (lt, l_offsets) if lt is not None else None
    else:
        l = None
        lt = None

    # Process point data
    if len(p) > 0:
        p =  np.array(p, dtype=np.int32) - 1
        pt = np.array(pt, dtype=np.int32) - 1
        if (p == -1).all():
            p = None
        if (pt == -1).all():
            pt = None
    else:
        p = None
        pt = None

    # Process object/group/material info
    if len(o_names) > 0:
        o_offsets.append(len(f))
        for i in range(len(o_names)):
            start_f, start_l, start_p = o_offsets[i]
            end_f, end_l, end_p = o_offsets[i + 1]
            o_i = {}
            if f is not None and end_f > start_f:
                o_i['f'] = slice(start_f, end_f)
            if l is not None and end_l > start_l:
                o_i['l'] = slice(start_l, end_l)
            if p is not None and end_p > start_p:
                o_i['p'] = slice(start_p, end_p)
            o[o_names[i]] = o_i
    else:
        o = None
    
    if len(g_names) > 0:
        g_offsets.append(len(f))
        for i in range(len(g_names)):
            start_f, start_l, start_p = g_offsets[i]
            end_f, end_l, end_p = g_offsets[i + 1]
            g_i = {}
            if f is not None and end_f > start_f:
                g_i['f'] = slice(start_f, end_f)
            if l is not None and end_l > start_l:
                g_i['l'] = slice(start_l, end_l)
            if p is not None and end_p > start_p:
                g_i['p'] = slice(start_p, end_p)
            g[g_names[i]] = g_i
    else:  
        g = None

    if len(usemtl_names) > 0:
        usemtl_offsets.append(len(f))
        for i in range(len(usemtl_names)):
            start_f, start_l, start_p = usemtl_offsets[i]
            end_f, end_l, end_p = usemtl_offsets[i + 1]
            usemtl_i = {}
            if f is not None and end_f > start_f:
                usemtl_i['f'] = slice(start_f, end_f)
            if l is not None and end_l > start_l:
                usemtl_i['l'] = slice(start_l, end_l)
            if p is not None and end_p > start_p:
                usemtl_i['p'] = slice(start_p, end_p)
            usemtl[usemtl_names[i]] = usemtl_i
    else:
        usemtl = None

    output = {'mtllib': mtllib, 'v': v, 'vt': vt, 'vn': vn, 'f': f, 'ft': ft, 'fn': fn, 'l': l, 'lt': lt, 'p': p, 'pt': pt, 'o': o, 'g': g, 'usemtl': usemtl}
    output = {k: v for k, v in output.items() if v is not None}

    return output


def write_obj(
    file: Union[str, Path, os.PathLike],
    obj: WavefrontOBJDict,
    encoding: Union[str, None] = None
):
    if isinstance(file, (str, Path, os.PathLike)):
        fp = Path(file).open('w', encoding=encoding)
    else:
        fp = file
    
    with fp:
        # Write mtllib
        if 'mtllib' in obj:
            for mtllib in obj.get('mtllib', []):
                print('mtllib', mtllib, file=fp)
        
        # Write vertices and attributes
        for k in ['v', 'vt', 'vn', 'vp']:
            if k not in obj:
                continue
            for v in obj[k]:
                print(k, *map(float, v), file=fp)
        
        # Write faces
        if 'f' in obj or 'ft' in obj or 'fn' in obj:
            f = obj.get('f', None)
            if isinstance(f, tuple):
                f_data, f_offsets = f
                f = np.split(f_data, f_offsets[1:-1])
            ft = obj.get('ft', None)
            if isinstance(ft, tuple):
                ft_data, f_offsets = ft
                ft = np.split(ft_data, f_offsets[1:-1])
            fn = obj.get('fn', None)
            if isinstance(fn, tuple):
                fn_data, f_offsets = fn
                fn = np.split(fn_data, f_offsets[1:-1])
            for i, (f_i, ft_i, fn_i) in enumerate(zip(f if f is not None else repeat(None), ft if ft is not None else repeat(None), fn if fn is not None else repeat(None))):
                if f_i is not None and ft_i is None and fn_i is None:
                    print('f', ' '.join(f'{f_ij + 1}' for f_ij in f_i), file=fp)
                else:
                    print('f', ' '.join(f'{f_ij}/{ft_ij}/{fn_ij}' for f_ij, ft_ij, fn_ij in zip(f_i if f_i is not None else repeat(''), ft_i if ft_i is not None else repeat(''), fn_i if fn_i is not None else repeat(''))))

        # Write lines
        if 'l' in obj or 'lt' in obj:
            l = obj.get('l', None)
            if isinstance(l, tuple):
                l_data, l_offsets = l
                l = np.split(l_data, l_offsets[1:-1])
            lt = obj.get('lt', None)
            if isinstance(lt, tuple):
                lt_data, l_offsets = lt
                lt = np.split(lt_data, l_offsets[1:-1])
            for i, (l_i, lt_i) in enumerate(zip(l if l is not None else repeat(None), lt if lt is not None else repeat(None))):
                if l_i is not None and lt_i is None:
                    print('l', ' '.join(f'{l_ij + 1}' for l_ij in l_i), file=fp)
                else:
                    print('l', ' '.join(f'{l_ij + 1}/{lt_ij + 1}' for l_ij, lt_ij in zip(l_i if l_i is not None else repeat(''), lt_i if lt_i is not None else repeat(''))))

        # Write points
        if 'p' in obj or 'pt' in obj:
            p = obj.get('p', None)
            pt = obj.get('pt', None)
            for i, (p_i, pt_i) in enumerate(zip(p if p is not None else repeat(None), pt if pt is not None else repeat(None))):
                if p_i is not None and pt_i is None:
                    print('p', f'{p_i + 1}', file=fp)
                else:
                    print('p', f'{p_i + 1}/{pt_i + 1}', file=fp)