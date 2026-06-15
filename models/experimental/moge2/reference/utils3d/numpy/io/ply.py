import io
from typing import *
import os
import numpy as np
from numpy import ndarray
import struct
from pathlib import Path
from functools import partial
from itertools import chain

from ...helpers import timeit
from ..segment_ops import segment_concatenate


SUPPORTED_COUNT_TYPES = ['char', 'uchar', 'short', 'ushort', 'int', 'uint']
SUPPORTED_DATA_TYPES = ['char', 'uchar', 'short', 'ushort', 'int', 'uint', 'float', 'double']

__all__ = [
    'read_ply',
    'write_ply',
]


C_TYPE_TO_NP_TYPE = {
    'char': np.int8,
    'uchar': np.uint8,
    'short': np.int16,
    'ushort': np.uint16,
    'int': np.int32,
    'uint': np.uint32,
    'float': np.float32,
    'double': np.float64,
}

NP_TYPE_TO_C_TYPE = {
    'float32': 'float',
    'float64': 'double',
    'int8': 'char',
    'uint8': 'uchar',
    'int16': 'short',
    'uint16': 'ushort',
    'int32': 'int',
    'uint32': 'uint',
}


# 字节大小映射 (用于计算指针偏移)
C_TYPE_TO_SIZE = {
    'char': 1, 'uchar': 1,
    'short': 2, 'ushort': 2,
    'int': 4, 'uint': 4,
    'float': 4, 'double': 8
}

# Struct 格式字符 (用于 struct.unpack 读取 list count)
# 注意：实际使用时需要加上字节序前缀 ('<' 或 '>')
C_TYPE_TO_STRUCT_FMT = {
    'char': 'b', 'uchar': 'B',
    'short': 'h', 'ushort': 'H',
    'int': 'i', 'uint': 'I',
}

class PLYHeaderDict(TypedDict):
    format: str
    elements: Dict[str, 'PLYElementDict']


class PLYElementDict(TypedDict):
    count: int
    properties: List[Dict[str, 'PLYPropertyDict']]


class PLYPropertyDict(TypedDict):
    type: Literal['scalar', 'list']
    name: str
    data_type: Literal['char', 'uchar', 'short', 'ushort', 'int', 'uint', 'float', 'double']
    count_type: Optional[Literal['char', 'uchar', 'short', 'ushort', 'int', 'uint']]


def parse_ply_header(file: IO) -> Tuple[PLYHeaderDict, bytes]:
    """
    Return a JSON-like dictionary representing the PLY header.

    Returns
    ----
    - `header` (Dict): Parsed PLY header.
    Example

    ```python
    {
        "format": "ascii",
        "version": "1.0",
        "elements": {
            "vertex": {
                "count": 8,
                "properties": [
                    {"type": "scalar", "data_type": "float", "name": "x"},
                    {"type": "scalar", "data_type": "float", "name": "y"},
                    {"type": "scalar", "data_type": "float", "name": "z"}
                ]
            },
            "face": {
                "count": 6,
                "properties": [
                    {"type": "list", "count_type": "uchar", "data_type": "int", "name": "vertex_indices"}
                ]
            }
        }
        "comment": "This is a sample PLY file"
    }
    ```
    """
    line = file.readline()
    if isinstance(line, bytes):
        line = line.decode('ascii')
    line = line.strip()
    if line != 'ply':
        raise ValueError('Invalid PLY file: missing "ply" header')
    header = {}
    elements = {}
    curr_elem = None
    while True:
        line = file.readline()
        if isinstance(line, bytes):
            line = line.decode('ascii')
        line = line.strip()
        tokens = line.split()
        if line == 'end_header':
            break
        if tokens[0] == 'comment':
            if 'comment' not in header:
                header['comment'] = []
            header['comment'].append(line[8:])
        elif tokens[0] == 'format':
            format_, version = line.split()[1:3]
            if 'format' in header:
                raise ValueError('Multiple format declarations in PLY header')
            header['format'] = format_
            header['version'] = version
            if format_ not in ['ascii', 'binary_little_endian', 'binary_big_endian']:
                raise ValueError(f'Unsupported PLY format: {format_}')
        elif tokens[0] == 'element':
            curr_elem = tokens[1]
            elem_count = int(tokens[2])
            elements[curr_elem] = {
                'count': elem_count, 
                'properties': []
            }
        elif tokens[0] == 'property':
            if tokens[1] == 'list':
                count_type, data_type, name = tokens[2:]
                if count_type not in SUPPORTED_COUNT_TYPES or data_type not in SUPPORTED_DATA_TYPES:
                    raise ValueError(f'Unsupported PLY property type: {line}')
                prop = {
                    'type': 'list',
                    'count_type': count_type,
                    'data_type': data_type,
                    'name': name
                }
            else:
                data_type, name = tokens[1:]
                if data_type not in SUPPORTED_DATA_TYPES:
                    raise ValueError(f'Unsupported PLY property type: {line}')
                prop = {
                    'type': 'scalar',
                    'data_type': data_type,
                    'name': name
                }
            elements[curr_elem]['properties'].append(prop)
        elif tokens[0] == 'end_header':
            break

    header['elements'] = elements
    return header


def parse_ply_data_ascii(header: PLYHeaderDict, data: str) -> Dict[str, Dict[str, Union[ndarray, Tuple[ndarray, ndarray]]]]:
    result = {}
    tokens = data.split()
    token_ptr = 0

    for elem_name, elem_info in header['elements'].items():
        elem_count = elem_info['count']
        props = elem_info['properties']
        elem_result = result[elem_name] = {}
        
        # Check if this Element contains any 'list' type properties
        has_list = any(p['type'] == 'list' for p in props)
        
        if elem_count == 0:
            # Handle empty elements
            for p in props:
                dtype = C_TYPE_TO_NP_TYPE[p['data_type']]
                if p['type'] == 'scalar':
                    elem_result[p['name']] = np.array([], dtype=dtype)
                else:
                    elem_result[p['name']] = (np.array([], dtype=dtype), np.array([0], dtype=np.int32))
            continue

        if not has_list:    
            # --- Fast branch: scalar only ---
            num_props = len(props)
            for i, prop in enumerate(props):
                dtype = C_TYPE_TO_NP_TYPE[prop['data_type']]
                elem_result[prop['name']] = np.array(
                    tokens[token_ptr + i:token_ptr + elem_count * num_props:num_props], 
                    dtype=dtype
                )
            token_ptr += elem_count * num_props
        else:
            # --- Slow branch: list ---
            maybe_block_size = 0
            prop_is_scalar = []
            prop_maybe = [] # (maybe_offset_in_block, maybe_cnt)
            for p in props:
                if p['type'] == 'scalar':
                    maybe_offset_in_block = maybe_block_size
                    maybe_block_size += 1
                    prop_is_scalar.append(True)
                    prop_maybe.append((maybe_offset_in_block, None))
                else:
                    maybe_offset_in_block = maybe_block_size
                    maybe_cnt = int(tokens[token_ptr + maybe_block_size])
                    maybe_block_size += 1 + maybe_cnt
                    prop_is_scalar.append(False)
                    prop_maybe.append((maybe_offset_in_block, maybe_cnt))
                
            # Check if fixed-size block is possible
            if len(tokens) - token_ptr < maybe_block_size * elem_count:
                is_fixed_block = False
            else:
                is_fixed_block = True
                for is_scalar, (maybe_offset_in_block, maybe_cnt) in zip(prop_is_scalar, prop_maybe):
                    if is_scalar:
                        continue
                    try:
                        counts_arr = np.array(
                            tokens[token_ptr + maybe_offset_in_block:token_ptr + maybe_block_size * elem_count:maybe_block_size], 
                            dtype=np.int32
                        )
                    except ValueError:
                        is_fixed_block = False
                        break
                    if not (counts_arr == maybe_cnt).all():
                        is_fixed_block = False
                        break
            
            if is_fixed_block:
                # --- Fast branch: fixed-size block ---
                for p, is_scalar, (maybe_offset_in_block, maybe_cnt) in zip(props, prop_is_scalar, prop_maybe):
                    p_name = p['name']
                    if is_scalar:
                        dtype = C_TYPE_TO_NP_TYPE[p['data_type']]
                        elem_result[p_name] = np.array(
                            tokens[token_ptr + maybe_offset_in_block:token_ptr + maybe_block_size * elem_count:maybe_block_size], 
                            dtype=dtype
                        )
                    else:
                        dtype = C_TYPE_TO_NP_TYPE[p['data_type']]
                        data_arr = np.array([
                            tokens[token_ptr + maybe_offset_in_block + i:token_ptr + maybe_block_size * elem_count:maybe_block_size]
                            for i in range(1, maybe_cnt + 1)
                        ], dtype=dtype).reshape(maybe_cnt, elem_count).swapaxes(0, 1)
                        data_arr = np.ascontiguousarray(data_arr)
                        elem_result[p_name] = data_arr
                token_ptr += maybe_block_size * elem_count
            else:
                # --- Slow branch: variable-size block ---
                temp_data_container = []  # (data_list, offsets_list)
                for p in props:
                    if p['type'] == 'scalar':
                        temp_data_container.append(([None] * elem_count, None))
                    else:
                        temp_data_container.append(([None] * elem_count, [None] * elem_count))

                prop_infos = tuple((is_scalar, *container) for is_scalar, container in zip(prop_is_scalar, temp_data_container))
                for i in range(elem_count):
                    for is_scalar, chunks, counts in prop_infos:
                        if is_scalar:
                            chunks[i] = tokens[token_ptr]
                            token_ptr += 1
                        else:   # 'list'
                            cnt = int(tokens[token_ptr])
                            token_ptr += 1
                            
                            chunks[i] = tokens[token_ptr:token_ptr + cnt]
                            counts[i] = cnt

                            token_ptr += cnt

                # Convert to NumPy arrays
                for prop, (chunks, counts) in zip(props, temp_data_container):
                    p_name = prop['name']
                    dtype = C_TYPE_TO_NP_TYPE[prop['data_type']]
                    if prop['type'] == 'scalar':
                        data_arr = np.array(chunks, dtype=dtype)
                        elem_result[p_name] = data_arr
                    else:
                        chunks = list(chain.from_iterable(chunks))
                        data_arr = np.array(chunks, dtype=dtype)
                        counts_arr = np.array(counts, dtype=np.int32)
                        if np.all(counts_arr == counts_arr[0]):
                            data_arr = data_arr.reshape(elem_count, -1)
                            elem_result[p_name] = data_arr
                        else:
                            offsets_arr = np.concatenate((np.array([0], dtype=counts_arr.dtype), np.cumsum(counts_arr)))
                            elem_result[p_name] = (data_arr, offsets_arr)
    return result


def parse_ply_data_binary(header: PLYHeaderDict, data: Union[bytes, bytearray, np.memmap]) -> Dict[str, Dict[str, Union[ndarray, Tuple[ndarray, ndarray]]]]:
    endian_str = '<' if header['format'] == 'binary_little_endian' else '>'
    result = {}
    
    data_ptr = 0
    
    for elem_name, elem_info in header['elements'].items():
        elem_count = elem_info['count']
        props = elem_info['properties']
        elem_result = result[elem_name] = {}
        
        if elem_count == 0: 
            for p in props:
                dtype = C_TYPE_TO_NP_TYPE[p['data_type']]
                if p['type'] == 'scalar':
                    elem_result[p['name']] = np.array([], dtype=dtype)
                else:
                    elem_result[p['name']] = (np.array([], dtype=dtype), np.array([0], dtype=np.int32))
            continue

        has_list = any(p['type'] == 'list' for p in props)

        # --- Fast branch: scalar only ---
        if not has_list:
            dtype_list = []
            for p in props:
                np_type = C_TYPE_TO_NP_TYPE[p['data_type']]
                dt_str = endian_str + np.dtype(np_type).char
                dtype_list.append((p['name'], dt_str))
            block_size = sum(C_TYPE_TO_SIZE[p['data_type']] for p in props)
            total_bytes = elem_count * block_size
            structured_arr = np.frombuffer(data[data_ptr:data_ptr + total_bytes], dtype=dtype_list, count=elem_count)
            
            for p in props:
                elem_result[p['name']] = structured_arr[p['name']]
            data_ptr += total_bytes
        else:
            # --- mixed scalar and list ---
            # Prepare property configurations
            prop_configs = []       # (is_scalar, item_size, unpack_cnt, cnt_size)
            prop_maybe = []         # (maybe_offset_in_block, maybe_cnt)
            maybe_block_size = 0
            for p in props:
                if p['type'] == 'scalar':
                    maybe_offset_in_block = maybe_block_size
                    item_size = C_TYPE_TO_SIZE[p['data_type']]
                    prop_configs.append((True, item_size, None, None))
                    maybe_block_size += item_size
                else:
                    maybe_offset_in_block = maybe_block_size
                    cnt_type = p['count_type']
                    data_type = p['data_type']
                    cnt_size = C_TYPE_TO_SIZE[cnt_type]
                    item_size = C_TYPE_TO_SIZE[data_type]

                    cnt_fmt = endian_str + C_TYPE_TO_STRUCT_FMT[cnt_type]
                    unpack_cnt = struct.Struct(cnt_fmt).unpack_from

                    maybe_cnt = unpack_cnt(data, data_ptr + maybe_offset_in_block)[0]
                    maybe_block_size += cnt_size + maybe_cnt * item_size
                    
                    prop_configs.append((False, item_size, unpack_cnt, cnt_size))
                    prop_maybe.append((maybe_offset_in_block, maybe_cnt))

            # Check if fixed-size block is possible
            if len(data) - data_ptr < maybe_block_size * elem_count:
                is_fixed_block = False
            else:
                is_fixed_block = True
                byte_arr = np.frombuffer(data[data_ptr:data_ptr + maybe_block_size * elem_count], dtype=np.uint8).reshape(elem_count, maybe_block_size)
                for p, (is_scalar, item_size, unpack_cnt, cnt_size), (maybe_offset_in_block, maybe_cnt) in zip(props, prop_configs, prop_maybe):
                    if is_scalar:
                        continue
                    count_dtype = np.dtype(C_TYPE_TO_NP_TYPE[p['count_type']]).newbyteorder(endian_str)
                    count_arr = byte_arr[:, maybe_offset_in_block:maybe_offset_in_block + cnt_size].view(count_dtype).reshape(-1)
                    if not (count_arr == maybe_cnt).all():
                        is_fixed_block = False
                        break
            
            if is_fixed_block:
                # --- Fast branch: fixed-size block ---
                for p, (is_scalar, item_size, unpack_cnt, cnt_size), (maybe_offset_in_block, maybe_cnt) in zip(props, prop_configs, prop_maybe):
                    p_name = p['name']
                    if is_scalar:
                        dtype = np.dtype(C_TYPE_TO_NP_TYPE[p['data_type']]).newbyteorder(endian_str)
                        data_arr = byte_arr[:, maybe_offset_in_block:maybe_offset_in_block + item_size].view(dtype).reshape(-1)
                        elem_result[p_name] = data_arr
                    else:
                        dtype = np.dtype(C_TYPE_TO_NP_TYPE[p['data_type']]).newbyteorder(endian_str)
                        data_arr = byte_arr[:, maybe_offset_in_block + cnt_size:maybe_offset_in_block + cnt_size + maybe_cnt * item_size].view(dtype).reshape(elem_count, maybe_cnt)
                        elem_result[p_name] = data_arr
                data_ptr += maybe_block_size * elem_count
            else:
                # --- Slow branch: variable-size block ---
                if isinstance(data, np.memmap):
                    data = data.tobytes()   # Load all data into memory for faster access
                temp_containers = []
                for p in props:
                    if p['type'] == 'scalar':
                        temp_containers.append(([None] * elem_count, None))
                    else:
                        temp_containers.append(([None] * elem_count, [None] * elem_count))  # (data_chunks, counts), preallocate for speed
                
                # Collect data chunks one by one
                prop_config_containers = tuple((*prop_cfg, *container) for prop_cfg, container in zip(prop_configs, temp_containers))    
                for i in range(elem_count):
                    for p_is_scalar, item_size, unpack_cnt, cnt_size, chunks, counts in prop_config_containers:
                        if p_is_scalar:
                            chunks[i] = data[data_ptr:data_ptr + item_size]
                            data_ptr += item_size
                        else:
                            cnt = unpack_cnt(data, data_ptr)[0]
                            data_ptr += cnt_size
                            
                            chunks[i] = data[data_ptr:(data_ptr := data_ptr + cnt * item_size)]
                            counts[i] = cnt
                            
                # Merge chunks and convert to NumPy arrays
                for p, (chunks, counts) in zip(props, temp_containers):
                    p_name = p['name']
                    if p['type'] == 'scalar':
                        dtype = np.dtype(C_TYPE_TO_NP_TYPE[p['data_type']]).newbyteorder(endian_str)
                        full_data_bytes = bytearray().join(chunks)
                        data_arr = np.frombuffer(full_data_bytes, dtype=dtype)
                        elem_result[p_name] = data_arr
                    elif p['type'] == 'list':
                        full_data_bytes = bytearray().join(chunks)
                        dtype = np.dtype(C_TYPE_TO_NP_TYPE[p['data_type']]).newbyteorder(endian_str)
                        data_arr = np.frombuffer(full_data_bytes, dtype=dtype)
                        counts_arr = np.array(counts, dtype=np.int32)
                        if np.all(counts_arr == counts_arr[0]):
                            data_arr = data_arr.reshape(elem_count, -1)
                            elem_result[p_name] = data_arr
                        else:
                            offsets_arr = np.concatenate((np.array([0], dtype=counts_arr.dtype), np.cumsum(counts_arr)))
                            elem_result[p_name] = (data_arr, offsets_arr)

    return result


def read_ply(file: Union[str, os.PathLike, IO]) -> Dict[str, Dict[str, Union[ndarray, Tuple[ndarray, ndarray]]]]:
    """Read a PLY file. Supports arbitrary properties, polygonal meshes. Very fast.

    Parameters
    ----------
    - `file` (str | os.PathLike | IO): Path to the PLY file or a file-like object.

    Returns
    -------
    - `data` (Dict): Parsed PLY data. Example
        ```python
            {
                "vertex": {
                    "x": ndarray,
                    "y": ndarray,
                    "z": ndarray,
                    ... # other properties, like "nx", "ny", "nz", "red", "green", "blue", etc.
                },
                "face": {
                    "vertex_indices": ndarray for regular lists or (ndarray, offsets ndarray) for irregular lists,
                    ...
                },
                ...
            }
        ```

    Performance
    -------

    Tested on a few binary PLY files:

    | Content Type   |  `utils3d` | `Open3D` | `Trimesh` | `plyfile` | `meshio` |
    |-----------  |------------| -------- |-----------|-----------| ---------|
    | Point Cloud (V=921,600) | 26.3 ms | 132.8 ms | 36.8 ms | 23.1 ms | 25.8 ms |
    | Triangle Mesh (V=425,949, F=841,148) | 17.4 ms | 144.8 ms | 341.9 ms | 2655.5 ms | 366.8 ms |
    | Polygon Mesh (V=437,645, F=871,414) | 289.5 ms | x | x | 1999.5 ms | 3905.3 ms |
    """
    if isinstance(file, (str, os.PathLike)):
        fp = open(file, 'rb')
        fsize = os.fstat(fp.fileno()).st_size
        is_file = True
    else:
        fp = file

    # Parse header
    header = parse_ply_header(fp)

    # Parse data
    with fp:
        if header['format'] == 'ascii':
            data = fp.read()
            if isinstance(data, bytes):
                data = data.decode('ascii')
            result = parse_ply_data_ascii(header, data)
        else:   # 'binary'
            if is_file:
                data = np.memmap(fp, mode='c', offset=fp.tell(), dtype=np.uint8, shape=(fsize - fp.tell(),))
            else:
                data = bytearray(fp.read())
            result = parse_ply_data_binary(header, data)

    return result


def get_ply_header_from_data(
    data: Dict[str, Dict[str, Union[ndarray, Tuple[ndarray, ndarray]]]], 
    format_: Literal['ascii', 'binary_little_endian', 'binary_big_endian'] = 'binary_little_endian',
    version: Literal['1.0'] = '1.0'
) -> PLYHeaderDict:
    header = {}
    header['format'] = format_
    header['version'] = version
    header['elements'] = {}
    assert format_ in ['ascii', 'binary_little_endian', 'binary_big_endian'], f'Unsupported PLY format: {format_}'
    for elem_name, elem_data in data.items():
        elem_count = None
        # Check element count consistency
        for prop_name, prop_data in elem_data.items():
            cnt = prop_data.shape[0] if isinstance(prop_data, ndarray) else prop_data[1].shape[0] - 1
            if elem_count is None:
                elem_count = cnt
            else:
                assert elem_count == cnt, f'Inconsistent element counts for element {elem_name}'
        if elem_count is None:
            elem_count = 0

        # Define element
        property_list = []
        header['elements'][elem_name] = {'count': elem_count, 'properties': property_list}

        # Define properties
        for prop_name, prop_data in elem_data.items():
            if isinstance(prop_data, tuple):
                # irregular list property
                data_arr, offsets_arr = prop_data
                dtype = data_arr.dtype
                data_c_type = NP_TYPE_TO_C_TYPE[np.dtype(dtype).name]
                counts = np.diff(offsets_arr)
                assert np.all(counts >= 0), f'Negative counts found in property {prop_name}'
                max_cnt = np.max(counts)
                cnt_c_type = 'uchar' if max_cnt < 256 else 'ushort' if max_cnt < 65536 else 'uint'
                property_list.append({
                    'type': 'list',
                    'count_type': cnt_c_type,
                    'data_type': data_c_type,
                    'name': prop_name
                })
            elif prop_data.ndim == 2:
                # fixed-size list property
                cnt = prop_data.shape[1]
                cnt_c_type = 'uchar' if cnt < 256 else 'ushort' if cnt < 65536 else 'uint'
                data_c_type = NP_TYPE_TO_C_TYPE[np.dtype(prop_data.dtype).name]
                property_list.append({
                    'type': 'list',
                    'count_type': cnt_c_type,
                    'data_type': data_c_type,
                    'name': prop_name
                })
            else:
                # scalar property
                dtype = prop_data.dtype
                data_c_type = NP_TYPE_TO_C_TYPE[np.dtype(dtype).name]
                property_list.append({
                    'type': 'scalar',
                    'data_type': data_c_type,
                    'name': prop_name
                })
    return header
    

def dump_ply_header(header: PLYHeaderDict) -> str:
    "Dump PLY header from dictionary to string."
    header_lines = []
    header_lines.append('ply')
    header_lines.append(f'format {header["format"]} 1.0')
    for elem_name, elem_info in header['elements'].items():
        elem_count = elem_info['count']
        header_lines.append(f'element {elem_name} {elem_count}')
        for prop in elem_info['properties']:
            if prop['type'] == 'scalar':
                header_lines.append(f'property {prop["data_type"]} {prop["name"]}')
            else:
                header_lines.append(f'property list {prop["count_type"]} {prop["data_type"]} {prop["name"]}')
    header_lines.append('end_header')
    header_str = '\n'.join(header_lines) + '\n'
    return header_str


def dump_ply_data_binary(
    header: PLYHeaderDict,
    data: Dict[str, Dict[str, Union[ndarray, Tuple[ndarray, ndarray]]]]
) -> Generator[Union[bytes, memoryview], None, None]:
    "Dump PLY data to binary bytes."
    endian_str = '<' if header['format'] == 'binary_little_endian' else '>'
    for elem_name, elem_config in header['elements'].items():
        elem_data = data[elem_name]
        elem_count = header['elements'][elem_name]['count']
        if not any(isinstance(v, tuple) for v in elem_data.values()):
            # --- scalar & fixed-size list only ---
            prop_data_bytes_list: List[np.ndarray] = []
            for prop in elem_config['properties']:
                prop_name = prop['name']
                prop_data = elem_data[prop_name]
                data_c_type = prop['data_type']
                data_np_type = np.dtype(C_TYPE_TO_NP_TYPE[data_c_type]).newbyteorder(endian_str)
                if prop_data.ndim == 1:
                    # scalar property
                    prop_data_bytes_list.append(prop_data.astype(data_np_type).view(np.uint8).reshape(elem_count, data_np_type.itemsize))
                elif prop_data.ndim == 2:
                    # fixed-size list property
                    cnt = prop_data.shape[1]
                    cnt_c_type = prop['count_type']
                    cnt_np_type = np.dtype(C_TYPE_TO_NP_TYPE[cnt_c_type]).newbyteorder(endian_str)

                    prop_data_bytes_list.append(np.full(elem_count, cnt, dtype=cnt_np_type).view(np.uint8).reshape(elem_count, cnt * cnt_np_type.itemsize))
                    prop_data_bytes_list.append(prop_data.astype(data_np_type).view(np.uint8).reshape(elem_count, prop_data.shape[1] * data_np_type.itemsize))
                else:
                    raise ValueError(f'Unsupported property data shape for property "{prop_name}": {prop_data.shape}')
            if len(prop_data_bytes_list) == 1:
                yield prop_data_bytes_list[0].data
            else:
                elem_data_bytes = np.concatenate(prop_data_bytes_list, axis=1).reshape(-1)
                yield elem_data_bytes.data
        else:
            # --- mixed scalar and variable-size list ---
            prop_data_bytes_list: List[np.ndarray] = []
            for prop in elem_config['properties']:
                prop_name = prop['name']
                prop_data = elem_data[prop_name]
                data_c_type = prop['data_type']
                data_np_type = np.dtype(C_TYPE_TO_NP_TYPE[data_c_type]).newbyteorder(endian_str)
                item_size = np.dtype(data_np_type).itemsize
                if isinstance(prop_data, tuple):
                    # list property
                    prop_data_arr, offsets_arr = prop_data

                    # Append counts
                    counts = np.diff(offsets_arr).astype(np.int32)
                    count_c_type = prop['count_type']
                    counts_np_type = np.dtype(C_TYPE_TO_NP_TYPE[count_c_type]).newbyteorder(endian_str)
                    counts_bytes = counts.astype(counts_np_type).view(np.uint8).reshape(-1)
                    count_size = C_TYPE_TO_SIZE[count_c_type]
                    prop_data_bytes_list.append((counts_bytes, np.arange(0, counts_bytes.size + 1, count_size)))
                    
                    # Append data
                    prop_data_bytes = prop_data_arr.astype(data_np_type).view(np.uint8).reshape(-1)
                    prop_data_bytes_list.append((prop_data_bytes, offsets_arr * item_size))
                else:
                    # scalar property
                    prop_data_bytes = prop_data.astype(data_np_type).view(np.uint8).reshape(-1)
                    prop_data_bytes_list.append((prop_data_bytes, np.arange(0, prop_data_bytes.size + 1, item_size)))
            elem_data_bytes, _ = segment_concatenate(prop_data_bytes_list)
            yield elem_data_bytes.data


def write_ply(file: Union[str, os.PathLike, IO], data: Dict[str, Dict[str, Union[ndarray, Tuple[ndarray, ndarray]]]], format_: Literal['ascii', 'binary_little_endian', 'binary_big_endian'] = 'binary_little_endian') -> None:
    """Write a PLY file. Supports arbitrary properties, polygonal meshes.

    Parameters
    ----------
    - `file` (str | os.PathLike | IO): Path to the PLY file or a file-like object.
    - `data` (Dict): PLY data to write. The structure is like
        ```python
            {
                "vertex": {
                    "x": ndarray,
                    "y": ndarray,
                    "z": ndarray,
                    ... # other properties, like "nx", "ny", "nz", "red", "green", "blue", etc.
                },
                "face": {
                    "vertex_indices": ndarray for regular lists or (ndarray, offsets ndarray) for irregular lists,
                    ...
                },
                ...
            }
        ```
    - `format` (str): PLY format. Options are 'ascii', 'binary_little_endian', 'binary_big_endian'.

    Performance
    -------
    
    | Content Type   |  `utils3d` | `Open3D` | `Trimesh` | `plyfile` | `meshio` |
    |-----------  |------------| -------- |-----------|-----------|---------|
    | Point Cloud (V=921,600)| 45.1 ms | 175.1 ms | 47.7 ms | 9.2 ms | 43.9 ms |
    | Triangle Mesh (V=425,949, F=841,148) | 38.3 ms | 137.9 ms | 41.3 ms | 2063.1 ms | 46.9 ms |
    | Polygon Mesh (V=437,645, F=871,414) | 234.0 ms | x | x | 1653.2 ms | 12360.8 ms |
    """
    # --- Prepare header ---
    header = get_ply_header_from_data(data, format_=format_)
    header_str = dump_ply_header(header)

    # --- Prepare data bytes ---
    if isinstance(file, (str, os.PathLike)):
        fp = open(file, 'wb')
        is_file = True
    else:
        fp = file
        is_file = False

    with fp:
        fp.write(header_str.encode())
        if format_ in ['binary_little_endian', 'binary_big_endian']:
            # --- Binary format ---
            data_bytes = dump_ply_data_binary(header, data)
            for chunk in data_bytes:
                if not is_file and isinstance(chunk, memoryview):
                    chunk = chunk.tobytes() # In case general IO does not support memoryview
                fp.write(chunk)
        else:
            # --- ASCII format ---
            # TODO: implement ASCII writing
            raise NotImplementedError('ASCII PLY writing not implemented yet.')