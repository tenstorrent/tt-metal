import ttnn
import utils


def main_const_eval_rope_cos_sin_doubled(arg_0, device=None):
    """E47 iter2: precompute doubled cos/sin tables for `ttnn.experimental.rotary_embedding_llama`.

    Input: arg_0[0] = `model.transformer.freqs_cis`, BF16 ROW_MAJOR replicated, shape [16384, 32, 2].
           Trailing dim is (cos_k, sin_k) per freq k.
    Output: [cos_doubled, sin_doubled] each BF16 TILE [16384, 64] replicated across mesh.
           cos_doubled[p] = [c_0, c_0, c_1, c_1, ..., c_31, c_31].
    """
    import torch

    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    composer = ttnn.ConcatMeshToTensor(device, dim=0)
    freqs = ttnn.to_torch(arg_0[0], mesh_composer=composer)
    # tensor is mesh-replicated; ConcatMeshToTensor(dim=0) gives 32 stacked copies — slice one
    freqs = freqs[:16384].to(torch.float32)
    cos = freqs[..., 0]
    sin = freqs[..., 1]
    cos_doubled = cos.repeat_interleave(2, dim=-1)
    sin_doubled = sin.repeat_interleave(2, dim=-1)
    mesh_mapper = ttnn.ReplicateTensorToMesh(device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
    cos_t = ttnn.from_torch(
        cos_doubled.unsqueeze(0).unsqueeze(0),  # [1, 1, 16384, 64]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        mesh_mapper=mesh_mapper,
    )
    sin_t = ttnn.from_torch(
        sin_doubled.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        mesh_mapper=mesh_mapper,
    )
    return [cos_t, sin_t]


def main_const_eval_all_reduce_semaphores(device=None):
    """E49: persistent GlobalSemaphore POOLs for ttnn.experimental.all_reduce_async.

    Mirrors the rotating-pool pattern from
    `models/demos/deepseek_v3/tt/ccl.py` — 2 slots per type, rotated per call
    so consecutive CCL calls use different semaphores and don't contend.

    Per call signature: barrier=2 sems, rs=3 sems, ag=2 sems (each list).
    Total allocated: 2 slots × (2+3+2) = 14 semaphores.
    """
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
    SLOTS = 2
    pool_barrier = [[ttnn.create_global_semaphore(device, cores, 0) for _ in range(2)] for _ in range(SLOTS)]
    pool_rs = [[ttnn.create_global_semaphore(device, cores, 0) for _ in range(3)] for _ in range(SLOTS)]
    pool_ag = [[ttnn.create_global_semaphore(device, cores, 0) for _ in range(2)] for _ in range(SLOTS)]
    return [pool_barrier, pool_rs, pool_ag]


# E49: rotating counter for the all_reduce_async semaphore pool. Module-level
# state, incremented per call site, modulo SLOTS=2.
_ccl_pool_slot_counter = [0]


def _ccl_next_slot():
    slot = _ccl_pool_slot_counter[0]
    _ccl_pool_slot_counter[0] = (slot + 1) % 2
    return slot


def main_const_eval_rope_trans_mat(device=None):
    """E47: Build the 32x32 half-rotate transformation matrix for
    `ttnn.experimental.rotary_embedding_llama`. This matrix has -1 on the
    super-diagonal of each 2-row block and +1 on the sub-diagonal:
        M[i,   i+1] = -1   (i even)
        M[i+1, i  ] = +1   (i even)
    Matrix-multiplying [a0 b0 a1 b1 ...] @ M produces [-b0 a0 -b1 a1 ...],
    i.e. the half-rotate that the kernel needs internally. Single fixed
    32x32 tensor, BF16 TILE, INTERLEAVED DRAM. Replicated to mesh.
    """
    import torch

    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    dhead = 32
    M = torch.zeros(1, 1, dhead, dhead, dtype=torch.float32)
    # tt-transformers convention: M[2k, 2k+1] = +1, M[2k+1, 2k] = -1.
    # This makes (x @ M) compute rotate_half(x) where x is in interleaved-pair
    # format [r_0, i_0, r_1, i_1, ...] and rotate_half([r, i]) = [-i, r].
    for i in range(0, dhead, 2):
        M[..., i, i + 1] = 1.0
        M[..., i + 1, i] = -1.0
    mesh_mapper = ttnn.ReplicateTensorToMesh(device) if hasattr(ttnn, "ReplicateTensorToMesh") else None
    t = ttnn.from_torch(
        M,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        mesh_mapper=mesh_mapper,
    )
    return [t]


def main_const_eval_0(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_Tensor_0 = ttnn.Tensor(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            127,
        ],
        [1, 1, 128],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_Tensor_0]


def main_const_eval_1(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_zeros_0 = ttnn.zeros(
        shape=ttnn.Shape([32, 1, 128]),
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_all_gather_0 = ttnn.all_gather(
        input_tensor=ttnn_zeros_0,
        dim=0,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_zeros_0, False)
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_all_gather_0,
        [16384],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_all_gather_0, False)
    ttnn_slice_0 = ttnn.slice(
        ttnn_reshape_0,
        [0],
        [256],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_0 = ttnn.to_layout(ttnn_slice_0, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_0, False)
    ttnn_slice_1 = ttnn.slice(
        ttnn_reshape_0,
        [256],
        [512],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_1 = ttnn.to_layout(ttnn_slice_1, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_1, False)
    ttnn_slice_2 = ttnn.slice(
        ttnn_reshape_0,
        [512],
        [768],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_2 = ttnn.to_layout(ttnn_slice_2, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_2, False)
    ttnn_slice_3 = ttnn.slice(
        ttnn_reshape_0,
        [768],
        [1024],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_3 = ttnn.to_layout(ttnn_slice_3, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_3, False)
    ttnn_slice_4 = ttnn.slice(
        ttnn_reshape_0,
        [1024],
        [1280],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_4 = ttnn.to_layout(ttnn_slice_4, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_4, False)
    ttnn_slice_5 = ttnn.slice(
        ttnn_reshape_0,
        [1280],
        [1536],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_5 = ttnn.to_layout(ttnn_slice_5, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_5, False)
    ttnn_slice_6 = ttnn.slice(
        ttnn_reshape_0,
        [1536],
        [1792],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_6 = ttnn.to_layout(ttnn_slice_6, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_6, False)
    ttnn_slice_7 = ttnn.slice(
        ttnn_reshape_0,
        [1792],
        [2048],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_7 = ttnn.to_layout(ttnn_slice_7, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_7, False)
    ttnn_slice_8 = ttnn.slice(
        ttnn_reshape_0,
        [2048],
        [2304],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_8 = ttnn.to_layout(ttnn_slice_8, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_8, False)
    ttnn_slice_9 = ttnn.slice(
        ttnn_reshape_0,
        [2304],
        [2560],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_9 = ttnn.to_layout(ttnn_slice_9, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_9, False)
    ttnn_slice_10 = ttnn.slice(
        ttnn_reshape_0,
        [2560],
        [2816],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_10 = ttnn.to_layout(ttnn_slice_10, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_10, False)
    ttnn_slice_11 = ttnn.slice(
        ttnn_reshape_0,
        [2816],
        [3072],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_11 = ttnn.to_layout(ttnn_slice_11, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_11, False)
    ttnn_slice_12 = ttnn.slice(
        ttnn_reshape_0,
        [3072],
        [3328],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_12 = ttnn.to_layout(ttnn_slice_12, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_12, False)
    ttnn_slice_13 = ttnn.slice(
        ttnn_reshape_0,
        [3328],
        [3584],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_13 = ttnn.to_layout(ttnn_slice_13, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_13, False)
    ttnn_slice_14 = ttnn.slice(
        ttnn_reshape_0,
        [3584],
        [3840],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_14 = ttnn.to_layout(ttnn_slice_14, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_14, False)
    ttnn_slice_15 = ttnn.slice(
        ttnn_reshape_0,
        [3840],
        [4096],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_15 = ttnn.to_layout(ttnn_slice_15, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_15, False)
    ttnn_slice_16 = ttnn.slice(
        ttnn_reshape_0,
        [4096],
        [4352],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_16 = ttnn.to_layout(ttnn_slice_16, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_16, False)
    ttnn_slice_17 = ttnn.slice(
        ttnn_reshape_0,
        [4352],
        [4608],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_17 = ttnn.to_layout(ttnn_slice_17, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_17, False)
    ttnn_slice_18 = ttnn.slice(
        ttnn_reshape_0,
        [4608],
        [4864],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_18 = ttnn.to_layout(ttnn_slice_18, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_18, False)
    ttnn_slice_19 = ttnn.slice(
        ttnn_reshape_0,
        [4864],
        [5120],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_19 = ttnn.to_layout(ttnn_slice_19, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_19, False)
    ttnn_slice_20 = ttnn.slice(
        ttnn_reshape_0,
        [5120],
        [5376],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_20 = ttnn.to_layout(ttnn_slice_20, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_20, False)
    ttnn_slice_21 = ttnn.slice(
        ttnn_reshape_0,
        [5376],
        [5632],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_21 = ttnn.to_layout(ttnn_slice_21, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_21, False)
    ttnn_slice_22 = ttnn.slice(
        ttnn_reshape_0,
        [5632],
        [5888],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_22 = ttnn.to_layout(ttnn_slice_22, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_22, False)
    ttnn_slice_23 = ttnn.slice(
        ttnn_reshape_0,
        [5888],
        [6144],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_23 = ttnn.to_layout(ttnn_slice_23, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_23, False)
    ttnn_slice_24 = ttnn.slice(
        ttnn_reshape_0,
        [6144],
        [6400],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_24 = ttnn.to_layout(ttnn_slice_24, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_24, False)
    ttnn_slice_25 = ttnn.slice(
        ttnn_reshape_0,
        [6400],
        [6656],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_25 = ttnn.to_layout(ttnn_slice_25, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_25, False)
    ttnn_slice_26 = ttnn.slice(
        ttnn_reshape_0,
        [6656],
        [6912],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_26 = ttnn.to_layout(ttnn_slice_26, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_26, False)
    ttnn_slice_27 = ttnn.slice(
        ttnn_reshape_0,
        [6912],
        [7168],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_27 = ttnn.to_layout(ttnn_slice_27, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_27, False)
    ttnn_slice_28 = ttnn.slice(
        ttnn_reshape_0,
        [7168],
        [7424],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_28 = ttnn.to_layout(ttnn_slice_28, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_28, False)
    ttnn_slice_29 = ttnn.slice(
        ttnn_reshape_0,
        [7424],
        [7680],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_29 = ttnn.to_layout(ttnn_slice_29, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_29, False)
    ttnn_slice_30 = ttnn.slice(
        ttnn_reshape_0,
        [7680],
        [7936],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_30 = ttnn.to_layout(ttnn_slice_30, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_30, False)
    ttnn_slice_31 = ttnn.slice(
        ttnn_reshape_0,
        [7936],
        [8192],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_31 = ttnn.to_layout(ttnn_slice_31, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_31, False)
    ttnn_slice_32 = ttnn.slice(
        ttnn_reshape_0,
        [8192],
        [8448],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_32 = ttnn.to_layout(ttnn_slice_32, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_32, False)
    ttnn_slice_33 = ttnn.slice(
        ttnn_reshape_0,
        [8448],
        [8704],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_33 = ttnn.to_layout(ttnn_slice_33, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_33, False)
    ttnn_slice_34 = ttnn.slice(
        ttnn_reshape_0,
        [8704],
        [8960],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_34 = ttnn.to_layout(ttnn_slice_34, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_34, False)
    ttnn_slice_35 = ttnn.slice(
        ttnn_reshape_0,
        [8960],
        [9216],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_35 = ttnn.to_layout(ttnn_slice_35, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_35, False)
    ttnn_slice_36 = ttnn.slice(
        ttnn_reshape_0,
        [9216],
        [9472],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_36 = ttnn.to_layout(ttnn_slice_36, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_36, False)
    ttnn_slice_37 = ttnn.slice(
        ttnn_reshape_0,
        [9472],
        [9728],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_37 = ttnn.to_layout(ttnn_slice_37, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_37, False)
    ttnn_slice_38 = ttnn.slice(
        ttnn_reshape_0,
        [9728],
        [9984],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_38 = ttnn.to_layout(ttnn_slice_38, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_38, False)
    ttnn_slice_39 = ttnn.slice(
        ttnn_reshape_0,
        [9984],
        [10240],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_39 = ttnn.to_layout(ttnn_slice_39, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_39, False)
    ttnn_slice_40 = ttnn.slice(
        ttnn_reshape_0,
        [10240],
        [10496],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_40 = ttnn.to_layout(ttnn_slice_40, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_40, False)
    ttnn_slice_41 = ttnn.slice(
        ttnn_reshape_0,
        [10496],
        [10752],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_41 = ttnn.to_layout(ttnn_slice_41, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_41, False)
    ttnn_slice_42 = ttnn.slice(
        ttnn_reshape_0,
        [10752],
        [11008],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_42 = ttnn.to_layout(ttnn_slice_42, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_42, False)
    ttnn_slice_43 = ttnn.slice(
        ttnn_reshape_0,
        [11008],
        [11264],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_43 = ttnn.to_layout(ttnn_slice_43, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_43, False)
    ttnn_slice_44 = ttnn.slice(
        ttnn_reshape_0,
        [11264],
        [11520],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_44 = ttnn.to_layout(ttnn_slice_44, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_44, False)
    ttnn_slice_45 = ttnn.slice(
        ttnn_reshape_0,
        [11520],
        [11776],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_45 = ttnn.to_layout(ttnn_slice_45, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_45, False)
    ttnn_slice_46 = ttnn.slice(
        ttnn_reshape_0,
        [11776],
        [12032],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_46 = ttnn.to_layout(ttnn_slice_46, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_46, False)
    ttnn_slice_47 = ttnn.slice(
        ttnn_reshape_0,
        [12032],
        [12288],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_47 = ttnn.to_layout(ttnn_slice_47, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_47, False)
    ttnn_slice_48 = ttnn.slice(
        ttnn_reshape_0,
        [12288],
        [12544],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_48 = ttnn.to_layout(ttnn_slice_48, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_48, False)
    ttnn_slice_49 = ttnn.slice(
        ttnn_reshape_0,
        [12544],
        [12800],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_49 = ttnn.to_layout(ttnn_slice_49, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_49, False)
    ttnn_slice_50 = ttnn.slice(
        ttnn_reshape_0,
        [12800],
        [13056],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_50 = ttnn.to_layout(ttnn_slice_50, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_50, False)
    ttnn_slice_51 = ttnn.slice(
        ttnn_reshape_0,
        [13056],
        [13312],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_51 = ttnn.to_layout(ttnn_slice_51, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_51, False)
    ttnn_slice_52 = ttnn.slice(
        ttnn_reshape_0,
        [13312],
        [13568],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_52 = ttnn.to_layout(ttnn_slice_52, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_52, False)
    ttnn_slice_53 = ttnn.slice(
        ttnn_reshape_0,
        [13568],
        [13824],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_53 = ttnn.to_layout(ttnn_slice_53, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_53, False)
    ttnn_slice_54 = ttnn.slice(
        ttnn_reshape_0,
        [13824],
        [14080],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_54 = ttnn.to_layout(ttnn_slice_54, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_54, False)
    ttnn_slice_55 = ttnn.slice(
        ttnn_reshape_0,
        [14080],
        [14336],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_55 = ttnn.to_layout(ttnn_slice_55, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_55, False)
    ttnn_slice_56 = ttnn.slice(
        ttnn_reshape_0,
        [14336],
        [14592],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_56 = ttnn.to_layout(ttnn_slice_56, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_56, False)
    ttnn_slice_57 = ttnn.slice(
        ttnn_reshape_0,
        [14592],
        [14848],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_57 = ttnn.to_layout(ttnn_slice_57, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_57, False)
    ttnn_slice_58 = ttnn.slice(
        ttnn_reshape_0,
        [14848],
        [15104],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_58 = ttnn.to_layout(ttnn_slice_58, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_58, False)
    ttnn_slice_59 = ttnn.slice(
        ttnn_reshape_0,
        [15104],
        [15360],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_59 = ttnn.to_layout(ttnn_slice_59, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_59, False)
    ttnn_slice_60 = ttnn.slice(
        ttnn_reshape_0,
        [15360],
        [15616],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_60 = ttnn.to_layout(ttnn_slice_60, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_60, False)
    ttnn_slice_61 = ttnn.slice(
        ttnn_reshape_0,
        [15616],
        [15872],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_61 = ttnn.to_layout(ttnn_slice_61, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_61, False)
    ttnn_slice_62 = ttnn.slice(
        ttnn_reshape_0,
        [15872],
        [16128],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_62 = ttnn.to_layout(ttnn_slice_62, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_62, False)
    ttnn_slice_63 = ttnn.slice(
        ttnn_reshape_0,
        [16128],
        [16384],
        [1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_zeros_full = ttnn.to_layout(ttnn_reshape_0, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_reshape_0, False)
    ttnn_to_layout_63 = ttnn.to_layout(ttnn_slice_63, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_slice_63, False)
    return [
        ttnn_to_layout_0,
        ttnn_to_layout_1,
        ttnn_to_layout_2,
        ttnn_to_layout_3,
        ttnn_to_layout_4,
        ttnn_to_layout_5,
        ttnn_to_layout_6,
        ttnn_to_layout_7,
        ttnn_to_layout_8,
        ttnn_to_layout_9,
        ttnn_to_layout_10,
        ttnn_to_layout_11,
        ttnn_to_layout_12,
        ttnn_to_layout_13,
        ttnn_to_layout_14,
        ttnn_to_layout_15,
        ttnn_to_layout_16,
        ttnn_to_layout_17,
        ttnn_to_layout_18,
        ttnn_to_layout_19,
        ttnn_to_layout_20,
        ttnn_to_layout_21,
        ttnn_to_layout_22,
        ttnn_to_layout_23,
        ttnn_to_layout_24,
        ttnn_to_layout_25,
        ttnn_to_layout_26,
        ttnn_to_layout_27,
        ttnn_to_layout_28,
        ttnn_to_layout_29,
        ttnn_to_layout_30,
        ttnn_to_layout_31,
        ttnn_to_layout_32,
        ttnn_to_layout_33,
        ttnn_to_layout_34,
        ttnn_to_layout_35,
        ttnn_to_layout_36,
        ttnn_to_layout_37,
        ttnn_to_layout_38,
        ttnn_to_layout_39,
        ttnn_to_layout_40,
        ttnn_to_layout_41,
        ttnn_to_layout_42,
        ttnn_to_layout_43,
        ttnn_to_layout_44,
        ttnn_to_layout_45,
        ttnn_to_layout_46,
        ttnn_to_layout_47,
        ttnn_to_layout_48,
        ttnn_to_layout_49,
        ttnn_to_layout_50,
        ttnn_to_layout_51,
        ttnn_to_layout_52,
        ttnn_to_layout_53,
        ttnn_to_layout_54,
        ttnn_to_layout_55,
        ttnn_to_layout_56,
        ttnn_to_layout_57,
        ttnn_to_layout_58,
        ttnn_to_layout_59,
        ttnn_to_layout_60,
        ttnn_to_layout_61,
        ttnn_to_layout_62,
        ttnn_to_layout_63,
        ttnn_to_layout_zeros_full,
    ]


def main_const_eval_2(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_0 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_64 = ttnn.to_layout(ttnn_to_device_0, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_0, False)
    ttnn_permute_0 = ttnn.permute(
        ttnn_to_layout_64,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_64, False)
    ttnn_from_device_0 = ttnn.from_device(ttnn_permute_0)
    ttnn.deallocate(ttnn_permute_0, False)
    ttnn_typecast_0 = ttnn.typecast(ttnn_from_device_0, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_0, False)
    ttnn_to_device_1 = ttnn.to_device(
        ttnn_typecast_0,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_0, False)
    return [ttnn_to_device_1]


def main_const_eval_3(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_2 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_65 = ttnn.to_layout(ttnn_to_device_2, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_2, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_to_layout_65,
        [1, 8, 7168, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_65, False)
    ttnn_from_device_1 = ttnn.from_device(ttnn_reshape_1)
    ttnn.deallocate(ttnn_reshape_1, False)
    ttnn_typecast_1 = ttnn.typecast(ttnn_from_device_1, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_1, False)
    ttnn_to_device_3 = ttnn.to_device(
        ttnn_typecast_1,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_1, False)
    return [ttnn_to_device_3]


def main_const_eval_4(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_zeros_1 = ttnn.zeros(
        shape=ttnn.Shape([32, 1, 128, 1]),
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_zeros_1]


def main_const_eval_5(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_4 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_66 = ttnn.to_layout(ttnn_to_device_4, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_4, False)
    ttnn_permute_1 = ttnn.permute(
        ttnn_to_layout_66,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_66, False)
    ttnn_from_device_2 = ttnn.from_device(ttnn_permute_1)
    ttnn_typecast_2 = ttnn.typecast(ttnn_from_device_2, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_2, False)
    ttnn_to_device_5 = ttnn.to_device(
        ttnn_typecast_2,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_2, False)
    ttnn_from_device_3 = ttnn.from_device(ttnn_permute_1)
    ttnn.deallocate(ttnn_permute_1, False)
    ttnn_typecast_3 = ttnn.typecast(ttnn_from_device_3, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_3, False)
    ttnn_to_device_6 = ttnn.to_device(
        ttnn_typecast_3,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_3, False)
    return [ttnn_to_device_5, ttnn_to_device_6]


def main_const_eval_6(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_full_0 = ttnn.full(
        shape=ttnn.Shape([1, 1]),
        fill_value=9.9999999747524271e-07,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_full_0]


def main_const_eval_7(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_full_1 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=1,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_full_1]


def main_const_eval_8(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_full_2 = ttnn.full(
        shape=ttnn.Shape([1, 1]),
        fill_value=8,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_full_2]


def main_const_eval_9(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_7 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_67 = ttnn.to_layout(ttnn_to_device_7, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_7, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_to_layout_67,
        [1, 8, 32],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_67, False)
    ttnn_typecast_4 = ttnn.typecast(
        ttnn_reshape_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_2, False)
    ttnn_reshape_3 = ttnn.reshape(
        ttnn_typecast_4,
        [1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_4, False)
    return [ttnn_reshape_3]


def main_const_eval_10(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_full_3 = ttnn.full(
        shape=ttnn.Shape([1, 1]),
        fill_value=0.125,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_full_3]


def main_const_eval_11(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_full_4 = ttnn.full(
        shape=ttnn.Shape([1, 1]),
        fill_value=0.08837890625,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_full_4]


def main_const_eval_12(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_Tensor_1 = ttnn.Tensor(
        [256.0, 1.0],
        [2, 1],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_Tensor_1]


def main_const_eval_13(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_8 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_68 = ttnn.to_layout(ttnn_to_device_8, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_8, False)
    ttnn_reshape_4 = ttnn.reshape(
        ttnn_to_layout_68,
        [16384, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_68, False)
    ttnn_to_layout_69 = ttnn.to_layout(ttnn_reshape_4, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_reshape_4, False)
    return [ttnn_to_layout_69]


def main_const_eval_14(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_9 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_70 = ttnn.to_layout(ttnn_to_device_9, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_9, False)
    ttnn_permute_2 = ttnn.permute(
        ttnn_to_layout_70,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_70, False)
    ttnn_from_device_4 = ttnn.from_device(ttnn_permute_2)
    ttnn.deallocate(ttnn_permute_2, False)
    ttnn_typecast_5 = ttnn.typecast(ttnn_from_device_4, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_4, False)
    ttnn_to_device_10 = ttnn.to_device(
        ttnn_typecast_5,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_5, False)
    return [ttnn_to_device_10]


def main_const_eval_15(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_11 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_71 = ttnn.to_layout(ttnn_to_device_11, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_11, False)
    ttnn_permute_3 = ttnn.permute(
        ttnn_to_layout_71,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_71, False)
    ttnn_from_device_5 = ttnn.from_device(ttnn_permute_3)
    ttnn.deallocate(ttnn_permute_3, False)
    ttnn_typecast_6 = ttnn.typecast(ttnn_from_device_5, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_5, False)
    ttnn_to_device_12 = ttnn.to_device(
        ttnn_typecast_6,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_6, False)
    return [ttnn_to_device_12]


def main_const_eval_16(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_13 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_72 = ttnn.to_layout(ttnn_to_device_13, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_13, False)
    ttnn_permute_4 = ttnn.permute(
        ttnn_to_layout_72,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_72, False)
    ttnn_from_device_6 = ttnn.from_device(ttnn_permute_4)
    ttnn.deallocate(ttnn_permute_4, False)
    ttnn_typecast_7 = ttnn.typecast(ttnn_from_device_6, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_6, False)
    ttnn_to_device_14 = ttnn.to_device(
        ttnn_typecast_7,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_7, False)
    return [ttnn_to_device_14]


def main_const_eval_17(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_full_5 = ttnn.full(
        shape=ttnn.Shape([1, 1]),
        fill_value=0.00013950893480796367,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_full_5]


def main_const_eval_18(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_full_6 = ttnn.full(
        shape=ttnn.Shape([32, 1, 128]),
        fill_value=float("-inf"),
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_all_gather_1 = ttnn.all_gather(
        input_tensor=ttnn_full_6,
        dim=0,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_full_6, False)
    ttnn_reshape_5 = ttnn.reshape(
        ttnn_all_gather_1,
        [16384],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_all_gather_1, False)
    ttnn_to_layout_73 = ttnn.to_layout(ttnn_reshape_5, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_reshape_5, False)
    return [ttnn_to_layout_73]


def main_const_eval_19(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_15 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_74 = ttnn.to_layout(ttnn_to_device_15, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_15, False)
    ttnn_permute_5 = ttnn.permute(
        ttnn_to_layout_74,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_74, False)
    ttnn_from_device_7 = ttnn.from_device(ttnn_permute_5)
    ttnn.deallocate(ttnn_permute_5, False)
    ttnn_typecast_8 = ttnn.typecast(ttnn_from_device_7, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_7, False)
    ttnn_to_device_16 = ttnn.to_device(
        ttnn_typecast_8,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_8, False)
    return [ttnn_to_device_16]


def main_const_eval_20(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_17 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_75 = ttnn.to_layout(ttnn_to_device_17, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_17, False)
    ttnn_permute_6 = ttnn.permute(
        ttnn_to_layout_75,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_75, False)
    ttnn_from_device_8 = ttnn.from_device(ttnn_permute_6)
    ttnn_typecast_9 = ttnn.typecast(ttnn_from_device_8, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_8, False)
    ttnn_to_device_18 = ttnn.to_device(
        ttnn_typecast_9,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_9, False)
    ttnn_from_device_9 = ttnn.from_device(ttnn_permute_6)
    ttnn.deallocate(ttnn_permute_6, False)
    ttnn_typecast_10 = ttnn.typecast(ttnn_from_device_9, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_9, False)
    ttnn_to_device_19 = ttnn.to_device(
        ttnn_typecast_10,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_10, False)
    return [ttnn_to_device_18, ttnn_to_device_19]


def main_const_eval_21(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_20 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_76 = ttnn.to_layout(ttnn_to_device_20, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_20, False)
    ttnn_permute_7 = ttnn.permute(
        ttnn_to_layout_76,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_76, False)
    ttnn_from_device_10 = ttnn.from_device(ttnn_permute_7)
    ttnn.deallocate(ttnn_permute_7, False)
    ttnn_typecast_11 = ttnn.typecast(ttnn_from_device_10, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_10, False)
    ttnn_to_device_21 = ttnn.to_device(
        ttnn_typecast_11,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_11, False)
    return [ttnn_to_device_21]


def main_const_eval_22(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_22 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_77 = ttnn.to_layout(ttnn_to_device_22, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_22, False)
    ttnn_reshape_6 = ttnn.reshape(
        ttnn_to_layout_77,
        [16, 256, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_77, False)
    ttnn_slice_64 = ttnn.slice(
        ttnn_reshape_6,
        [0, 0, 0],
        [16, 128, 512],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_from_device_11 = ttnn.from_device(ttnn_slice_64)
    ttnn.deallocate(ttnn_slice_64, False)
    ttnn_typecast_12 = ttnn.typecast(ttnn_from_device_11, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_11, False)
    ttnn_to_device_23 = ttnn.to_device(
        ttnn_typecast_12,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_12, False)
    ttnn_permute_8 = ttnn.permute(
        ttnn_reshape_6,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_6, False)
    ttnn_slice_65 = ttnn.slice(
        ttnn_permute_8,
        [0, 0, 128],
        [16, 512, 256],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_8, False)
    ttnn_from_device_12 = ttnn.from_device(ttnn_slice_65)
    ttnn.deallocate(ttnn_slice_65, False)
    ttnn_typecast_13 = ttnn.typecast(ttnn_from_device_12, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_12, False)
    ttnn_to_device_24 = ttnn.to_device(
        ttnn_typecast_13,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_13, False)
    return [ttnn_to_device_23, ttnn_to_device_24]


def main_const_eval_23(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_full_7 = ttnn.full(
        shape=ttnn.Shape([1, 1, 1]),
        fill_value=0.134765625,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_full_7]


def main_const_eval_24(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_25 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_78 = ttnn.to_layout(ttnn_to_device_25, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_25, False)
    ttnn_reshape_7 = ttnn.reshape(
        ttnn_to_layout_78,
        [1, 1, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_78, False)
    ttnn_typecast_14 = ttnn.typecast(
        ttnn_reshape_7,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_7, False)
    return [ttnn_typecast_14]


def main_const_eval_25(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_26 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_79 = ttnn.to_layout(ttnn_to_device_26, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_26, False)
    ttnn_permute_9 = ttnn.permute(
        ttnn_to_layout_79,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_79, False)
    ttnn_typecast_15 = ttnn.typecast(
        ttnn_permute_9,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_9, False)
    ttnn_from_device_13 = ttnn.from_device(ttnn_typecast_15)
    ttnn.deallocate(ttnn_typecast_15, False)
    ttnn_typecast_16 = ttnn.typecast(ttnn_from_device_13, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_13, False)
    ttnn_to_device_27 = ttnn.to_device(
        ttnn_typecast_16,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_16, False)
    return [ttnn_to_device_27]


def main_const_eval_26(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_full_8 = ttnn.full(
        shape=ttnn.Shape([1, 1, 1]),
        fill_value=2.5,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_full_8]


def main_const_eval_27(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_28 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_80 = ttnn.to_layout(ttnn_to_device_28, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_28, False)
    ttnn_permute_10 = ttnn.permute(
        ttnn_to_layout_80,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_80, False)
    ttnn_from_device_14 = ttnn.from_device(ttnn_permute_10)
    ttnn.deallocate(ttnn_permute_10, False)
    ttnn_typecast_17 = ttnn.typecast(ttnn_from_device_14, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_14, False)
    ttnn_to_device_29 = ttnn.to_device(
        ttnn_typecast_17,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_17, False)
    return [ttnn_to_device_29]


def main_const_eval_28(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_30 = ttnn.to_device(
        arg_0[3],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_81 = ttnn.to_layout(ttnn_to_device_30, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_30, False)
    ttnn_to_device_31 = ttnn.to_device(
        arg_0[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_82 = ttnn.to_layout(ttnn_to_device_31, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_31, False)
    ttnn_to_device_32 = ttnn.to_device(
        arg_0[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_83 = ttnn.to_layout(ttnn_to_device_32, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_32, False)
    ttnn_to_device_33 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_84 = ttnn.to_layout(ttnn_to_device_33, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_33, False)
    ttnn_permute_11 = ttnn.permute(
        ttnn_to_layout_83,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_83, False)
    ttnn_permute_12 = ttnn.permute(
        ttnn_to_layout_82,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_82, False)
    ttnn_permute_13 = ttnn.permute(
        ttnn_to_layout_81,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_81, False)
    ttnn_permute_14 = ttnn.permute(
        ttnn_to_layout_84,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_84, False)
    ttnn_concat_0 = ttnn.concat(
        [ttnn_permute_11, ttnn_permute_12, ttnn_permute_13, ttnn_permute_14],
        1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_14, False)
    ttnn.deallocate(ttnn_permute_13, False)
    ttnn.deallocate(ttnn_permute_12, False)
    ttnn.deallocate(ttnn_permute_11, False)
    ttnn_from_device_15 = ttnn.from_device(ttnn_concat_0)
    ttnn.deallocate(ttnn_concat_0, False)
    ttnn_typecast_18 = ttnn.typecast(ttnn_from_device_15, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_15, False)
    ttnn_to_device_34 = ttnn.to_device(
        ttnn_typecast_18,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_18, False)
    return [ttnn_to_device_34]


def main_const_eval_29(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_35 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_85 = ttnn.to_layout(ttnn_to_device_35, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_35, False)
    ttnn_permute_15 = ttnn.permute(
        ttnn_to_layout_85,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_85, False)
    ttnn_from_device_16 = ttnn.from_device(ttnn_permute_15)
    ttnn.deallocate(ttnn_permute_15, False)
    ttnn_typecast_19 = ttnn.typecast(ttnn_from_device_16, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_16, False)
    ttnn_to_device_36 = ttnn.to_device(
        ttnn_typecast_19,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_19, False)
    return [ttnn_to_device_36]


def main_const_eval_30(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_37 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_86 = ttnn.to_layout(ttnn_to_device_37, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_37, False)
    ttnn_to_layout_87 = ttnn.to_layout(ttnn_to_layout_86, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_to_layout_86, False)
    return [ttnn_to_layout_87]


def main_const_eval_31(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_38 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_88 = ttnn.to_layout(ttnn_to_device_38, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_38, False)
    ttnn_reshape_8 = ttnn.reshape(
        ttnn_to_layout_88,
        [1, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_88, False)
    ttnn_typecast_20 = ttnn.typecast(
        ttnn_reshape_8,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_8, False)
    return [ttnn_typecast_20]


def main_const_eval_32(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_39 = ttnn.to_device(
        arg_0[3],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_89 = ttnn.to_layout(ttnn_to_device_39, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_39, False)
    ttnn_to_device_40 = ttnn.to_device(
        arg_0[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_90 = ttnn.to_layout(ttnn_to_device_40, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_40, False)
    ttnn_to_device_41 = ttnn.to_device(
        arg_0[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_91 = ttnn.to_layout(ttnn_to_device_41, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_41, False)
    ttnn_to_device_42 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_92 = ttnn.to_layout(ttnn_to_device_42, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_42, False)
    ttnn_permute_16 = ttnn.permute(
        ttnn_to_layout_91,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_91, False)
    ttnn_permute_17 = ttnn.permute(
        ttnn_to_layout_90,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_90, False)
    ttnn_permute_18 = ttnn.permute(
        ttnn_to_layout_89,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_89, False)
    ttnn_permute_19 = ttnn.permute(
        ttnn_to_layout_92,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_92, False)
    ttnn_concat_1 = ttnn.concat(
        [ttnn_permute_16, ttnn_permute_17, ttnn_permute_18, ttnn_permute_19],
        1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_19, False)
    ttnn.deallocate(ttnn_permute_18, False)
    ttnn.deallocate(ttnn_permute_17, False)
    ttnn.deallocate(ttnn_permute_16, False)
    ttnn_from_device_17 = ttnn.from_device(ttnn_concat_1)
    ttnn.deallocate(ttnn_concat_1, False)
    ttnn_typecast_21 = ttnn.typecast(ttnn_from_device_17, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_17, False)
    ttnn_to_device_43 = ttnn.to_device(
        ttnn_typecast_21,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_21, False)
    return [ttnn_to_device_43]


def main_const_eval_33(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_full_9 = ttnn.full(
        shape=ttnn.Shape([1, 1]),
        fill_value=128,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_full_9]


def main_const_eval_34(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_44 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_93 = ttnn.to_layout(ttnn_to_device_44, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_44, False)
    ttnn_reshape_9 = ttnn.reshape(
        ttnn_to_layout_93,
        [1, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_93, False)
    ttnn_typecast_22 = ttnn.typecast(
        ttnn_reshape_9,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_9, False)
    return [ttnn_typecast_22]


def main_const_eval_35(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_Tensor_2 = ttnn.Tensor(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            127,
            128,
            129,
            130,
            131,
            132,
            133,
            134,
            135,
            136,
            137,
            138,
            139,
            140,
            141,
            142,
            143,
            144,
            145,
            146,
            147,
            148,
            149,
            150,
            151,
            152,
            153,
            154,
            155,
            156,
            157,
            158,
            159,
            160,
            161,
            162,
            163,
            164,
            165,
            166,
            167,
            168,
            169,
            170,
            171,
            172,
            173,
            174,
            175,
            176,
            177,
            178,
            179,
            180,
            181,
            182,
            183,
            184,
            185,
            186,
            187,
            188,
            189,
            190,
            191,
            192,
            193,
            194,
            195,
            196,
            197,
            198,
            199,
            200,
            201,
            202,
            203,
            204,
            205,
            206,
            207,
            208,
            209,
            210,
            211,
            212,
            213,
            214,
            215,
            216,
            217,
            218,
            219,
            220,
            221,
            222,
            223,
            224,
            225,
            226,
            227,
            228,
            229,
            230,
            231,
            232,
            233,
            234,
            235,
            236,
            237,
            238,
            239,
            240,
            241,
            242,
            243,
            244,
            245,
            246,
            247,
            248,
            249,
            250,
            251,
            252,
            253,
            254,
            255,
        ],
        [1, 1, 256],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_Tensor_2]


def main_const_eval_36(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_arange_0 = ttnn.arange(
        0,
        32,
        1,
        dtype=ttnn.DataType.UINT32,
        device=device,
        layout=ttnn.Layout.TILE,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_10 = ttnn.reshape(
        ttnn_arange_0,
        [32, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_arange_0, False)
    ttnn_repeat_0 = ttnn.repeat(
        ttnn_reshape_10,
        ttnn.Shape([1, 8, 1]),
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_10, False)
    return [ttnn_repeat_0]


def main_const_eval_37(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_typecast_23 = ttnn.typecast(arg_0[0], ttnn.DataType.UINT16, memory_config=None)
    ttnn_to_device_45 = ttnn.to_device(
        ttnn_typecast_23,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_23, False)
    return [ttnn_to_device_45]


def main_const_eval_38(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_full_10 = ttnn.full(
        shape=ttnn.Shape([1, 1, 1, 1]),
        fill_value=float("-inf"),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_full_10]


def main_const_eval_39(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_46 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_94 = ttnn.to_layout(ttnn_to_device_46, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_46, False)
    ttnn_reshape_11 = ttnn.reshape(
        ttnn_to_layout_94,
        [1, 8, 2048, 7168],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_94, False)
    ttnn_from_device_18 = ttnn.from_device(ttnn_reshape_11)
    ttnn.deallocate(ttnn_reshape_11, False)
    ttnn_typecast_24 = ttnn.typecast(ttnn_from_device_18, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_18, False)
    ttnn_to_device_47 = ttnn.to_device(
        ttnn_typecast_24,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_24, False)
    return [ttnn_to_device_47]


def main_const_eval_40(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_48 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_95 = ttnn.to_layout(ttnn_to_device_48, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_48, False)
    ttnn_reshape_12 = ttnn.reshape(
        ttnn_to_layout_95,
        [1, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_95, False)
    ttnn_typecast_25 = ttnn.typecast(
        ttnn_reshape_12,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_12, False)
    return [ttnn_typecast_25]


def main_const_eval_41(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_49 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_96 = ttnn.to_layout(ttnn_to_device_49, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_49, False)
    ttnn_reshape_13 = ttnn.reshape(
        ttnn_to_layout_96,
        [1, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_96, False)
    ttnn_typecast_26 = ttnn.typecast(
        ttnn_reshape_13,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_13, False)
    return [ttnn_typecast_26]


def main_const_eval_42(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_50 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_97 = ttnn.to_layout(ttnn_to_device_50, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_50, False)
    ttnn_permute_20 = ttnn.permute(
        ttnn_to_layout_97,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_97, False)
    ttnn_from_device_19 = ttnn.from_device(ttnn_permute_20)
    ttnn.deallocate(ttnn_permute_20, False)
    ttnn_typecast_27 = ttnn.typecast(ttnn_from_device_19, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_19, False)
    ttnn_to_device_51 = ttnn.to_device(
        ttnn_typecast_27,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_27, False)
    return [ttnn_to_device_51]


def main_const_eval_43(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_52 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_98 = ttnn.to_layout(ttnn_to_device_52, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_52, False)
    ttnn_permute_21 = ttnn.permute(
        ttnn_to_layout_98,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_98, False)
    ttnn_from_device_20 = ttnn.from_device(ttnn_permute_21)
    ttnn.deallocate(ttnn_permute_21, False)
    ttnn_typecast_28 = ttnn.typecast(ttnn_from_device_20, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_20, False)
    ttnn_to_device_53 = ttnn.to_device(
        ttnn_typecast_28,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_28, False)
    return [ttnn_to_device_53]


def main_const_eval_44(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_full_11 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=0,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_full_11]


def main_const_eval_45(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_54 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_99 = ttnn.to_layout(ttnn_to_device_54, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_54, False)
    ttnn_permute_22 = ttnn.permute(
        ttnn_to_layout_99,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_99, False)
    ttnn_from_device_21 = ttnn.from_device(ttnn_permute_22)
    ttnn.deallocate(ttnn_permute_22, False)
    ttnn_typecast_29 = ttnn.typecast(ttnn_from_device_21, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_21, False)
    ttnn_to_device_55 = ttnn.to_device(
        ttnn_typecast_29,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_29, False)
    return [ttnn_to_device_55]


def main_const_eval_46(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_56 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_100 = ttnn.to_layout(ttnn_to_device_56, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_56, False)
    ttnn_permute_23 = ttnn.permute(
        ttnn_to_layout_100,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_100, False)
    ttnn_from_device_22 = ttnn.from_device(ttnn_permute_23)
    ttnn.deallocate(ttnn_permute_23, False)
    ttnn_typecast_30 = ttnn.typecast(ttnn_from_device_22, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_22, False)
    ttnn_to_device_57 = ttnn.to_device(
        ttnn_typecast_30,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_30, False)
    return [ttnn_to_device_57]


def main_const_eval_47(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_58 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_101 = ttnn.to_layout(ttnn_to_device_58, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_58, False)
    ttnn_reshape_14 = ttnn.reshape(
        ttnn_to_layout_101,
        [1, 8, 7168, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_101, False)
    ttnn_from_device_23 = ttnn.from_device(ttnn_reshape_14)
    ttnn.deallocate(ttnn_reshape_14, False)
    ttnn_typecast_31 = ttnn.typecast(ttnn_from_device_23, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_23, False)
    ttnn_to_device_59 = ttnn.to_device(
        ttnn_typecast_31,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_31, False)
    return [ttnn_to_device_59]


def main_const_eval_gate_up(arg_gate, arg_up, device=None):
    """E30: pre-packed [1, 8, 7168, 4096] BFP8_B weight = concat(gate_proj, up_proj) along N.

    Lets the MoE FFN run a single fused sparse_matmul covering both gate (W0) and
    up (W1) projections at once, with the same input + sparsity. Output is sliced
    in main into two halves before SwiGLU."""
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    g_dev = ttnn.to_device(
        arg_gate[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    g_tile = ttnn.to_layout(g_dev, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(g_dev, False)
    g_r = ttnn.reshape(
        g_tile,
        [1, 8, 7168, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(g_tile, False)
    u_dev = ttnn.to_device(
        arg_up[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    u_tile = ttnn.to_layout(u_dev, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(u_dev, False)
    u_r = ttnn.reshape(
        u_tile,
        [1, 8, 7168, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(u_tile, False)
    gu = ttnn.concat(
        [g_r, u_r],
        dim=-1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(g_r, False)
    ttnn.deallocate(u_r, False)
    gu_host = ttnn.from_device(gu)
    ttnn.deallocate(gu, False)
    gu_bfp8 = ttnn.typecast(gu_host, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(gu_host, False)
    gu_out = ttnn.to_device(
        gu_bfp8,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(gu_bfp8, False)
    return [gu_out]


def main_const_eval_48(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_60 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_102 = ttnn.to_layout(ttnn_to_device_60, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_60, False)
    ttnn_reshape_15 = ttnn.reshape(
        ttnn_to_layout_102,
        [16, 256, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_102, False)
    ttnn_slice_66 = ttnn.slice(
        ttnn_reshape_15,
        [0, 0, 0],
        [16, 128, 512],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_from_device_24 = ttnn.from_device(ttnn_slice_66)
    ttnn.deallocate(ttnn_slice_66, False)
    ttnn_typecast_32 = ttnn.typecast(ttnn_from_device_24, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_24, False)
    ttnn_to_device_61 = ttnn.to_device(
        ttnn_typecast_32,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_32, False)
    ttnn_permute_24 = ttnn.permute(
        ttnn_reshape_15,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_15, False)
    ttnn_slice_67 = ttnn.slice(
        ttnn_permute_24,
        [0, 0, 128],
        [16, 512, 256],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_24, False)
    ttnn_from_device_25 = ttnn.from_device(ttnn_slice_67)
    ttnn.deallocate(ttnn_slice_67, False)
    ttnn_typecast_33 = ttnn.typecast(ttnn_from_device_25, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_25, False)
    ttnn_to_device_62 = ttnn.to_device(
        ttnn_typecast_33,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_33, False)
    return [ttnn_to_device_61, ttnn_to_device_62]


def main_const_eval_49(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_full_12 = ttnn.full(
        shape=ttnn.Shape([1, 1, 1]),
        fill_value=float("-inf"),
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_full_12]


def main_const_eval_50(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_full_13 = ttnn.full(
        shape=ttnn.Shape([1, 1, 1]),
        fill_value=float("-inf"),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_full_13]


def main_const_eval_51(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_Tensor_3 = ttnn.Tensor(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            127,
        ],
        [1, 1, 1, 128],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_Tensor_3]


def main_const_eval_52(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_full_14 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=16384,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [ttnn_full_14]


def main_const_eval_53(device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_arange_1 = ttnn.arange(
        0,
        32,
        1,
        dtype=ttnn.DataType.INT32,
        device=device,
        layout=ttnn.Layout.TILE,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_16 = ttnn.reshape(
        ttnn_arange_1,
        [32, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_1 = ttnn.repeat(
        ttnn_reshape_16,
        ttnn.Shape([1, 1, 128, 1]),
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_16, False)
    ttnn_reshape_17 = ttnn.reshape(
        ttnn_arange_1,
        [32, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_arange_1, False)
    ttnn_repeat_2 = ttnn.repeat(
        ttnn_reshape_17,
        ttnn.Shape([1, 4, 1]),
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_17, False)
    return [ttnn_repeat_1, ttnn_repeat_2]


def main_const_eval_54(arg_0, device=None):
    if device is None:
        device = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_device_63 = ttnn.to_device(
        arg_0[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_103 = ttnn.to_layout(ttnn_to_device_63, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_to_device_63, False)
    ttnn_permute_25 = ttnn.permute(
        ttnn_to_layout_103,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_103, False)
    ttnn_from_device_26 = ttnn.from_device(ttnn_permute_25)
    ttnn.deallocate(ttnn_permute_25, False)
    ttnn_typecast_34 = ttnn.typecast(ttnn_from_device_26, ttnn.DataType.BFLOAT8_B, memory_config=None)
    ttnn.deallocate(ttnn_from_device_26, False)
    ttnn_to_device_64 = ttnn.to_device(
        ttnn_typecast_34,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_34, False)
    return [ttnn_to_device_64]


ce_cache__main = {}


def _main(activations, weights):
    import tracy as _tracy

    global ce_cache__main
    ce_cache__main = consteval__main(ce_cache__main, weights)
    # decode_1_start fires AFTER consteval so the signposted scope excludes
    # the one-shot device ops from main_const_eval_*().
    _tracy.signpost("decode_1_start")
    _tracy.signpost("prologue_start")
    args_0 = activations[0]
    args_1 = activations[1]
    model_transformer_layers_0_attn_indexer_k_cache = activations[2]
    L__past_key_values___layers_0_compressed_kv = activations[3]
    L__past_key_values___layers_0_k_pe = activations[4]
    model_transformer_layers_1_attn_indexer_k_cache = activations[5]
    L__past_key_values___layers_1_compressed_kv = activations[6]
    L__past_key_values___layers_1_k_pe = activations[7]
    activation_0 = activations[8]
    activation_1 = activations[9]
    var_0 = ce_cache__main["main_const_eval_1"]
    var_1 = var_0[0]
    var_2 = var_0[1]
    var_3 = var_0[2]
    var_4 = var_0[3]
    var_5 = var_0[4]
    var_6 = var_0[5]
    var_7 = var_0[6]
    var_8 = var_0[7]
    var_9 = var_0[8]
    var_10 = var_0[9]
    var_11 = var_0[10]
    var_12 = var_0[11]
    var_13 = var_0[12]
    var_14 = var_0[13]
    var_15 = var_0[14]
    var_16 = var_0[15]
    var_17 = var_0[16]
    var_18 = var_0[17]
    var_19 = var_0[18]
    var_20 = var_0[19]
    var_21 = var_0[20]
    var_22 = var_0[21]
    var_23 = var_0[22]
    var_24 = var_0[23]
    var_25 = var_0[24]
    var_26 = var_0[25]
    var_27 = var_0[26]
    var_28 = var_0[27]
    var_29 = var_0[28]
    var_30 = var_0[29]
    var_31 = var_0[30]
    var_32 = var_0[31]
    var_33 = var_0[32]
    var_34 = var_0[33]
    var_35 = var_0[34]
    var_36 = var_0[35]
    var_37 = var_0[36]
    var_38 = var_0[37]
    var_39 = var_0[38]
    var_40 = var_0[39]
    var_41 = var_0[40]
    var_42 = var_0[41]
    var_43 = var_0[42]
    var_44 = var_0[43]
    var_45 = var_0[44]
    var_46 = var_0[45]
    var_47 = var_0[46]
    var_48 = var_0[47]
    var_49 = var_0[48]
    var_50 = var_0[49]
    var_51 = var_0[50]
    var_52 = var_0[51]
    var_53 = var_0[52]
    var_54 = var_0[53]
    var_55 = var_0[54]
    var_56 = var_0[55]
    var_57 = var_0[56]
    var_58 = var_0[57]
    var_59 = var_0[58]
    var_60 = var_0[59]
    var_61 = var_0[60]
    var_62 = var_0[61]
    var_63 = var_0[62]
    var_64 = var_0[63]
    var_zeros_full = var_0[64]
    var_65 = ce_cache__main["main_const_eval_4"]
    var_66 = ce_cache__main["main_const_eval_5"]
    var_67 = ce_cache__main["main_const_eval_6"]
    var_68 = ce_cache__main["main_const_eval_10"]
    var_69 = ce_cache__main["main_const_eval_11"]
    var_70 = ce_cache__main["main_const_eval_17"]
    var_71 = ce_cache__main["main_const_eval_18"]
    var_72 = ce_cache__main["main_const_eval_20"]
    var_73 = ce_cache__main["main_const_eval_22"]
    var_74 = ce_cache__main["main_const_eval_23"]
    var_75 = ce_cache__main["main_const_eval_33"]
    var_76 = ce_cache__main["main_const_eval_37"]
    var_77 = ce_cache__main["main_const_eval_38"]
    var_78 = ce_cache__main["main_const_eval_48"]
    var_79 = ce_cache__main["main_const_eval_50"]
    var_80 = ce_cache__main["main_const_eval_53"]
    var_81 = var_80[0]
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((4, 8))
    ttnn_to_layout_104 = ttnn.to_layout(args_1, ttnn.Layout.TILE, None, memory_config=None)
    ttnn_gt_1 = ttnn.gt(
        ce_cache__main["main_const_eval_44"],
        ttnn_to_layout_104,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_35 = ttnn.typecast(
        args_0,
        ttnn.DataType.UINT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    # Path B: args_0 (input_ids) must survive _main so pcc.py --trace can
    # ttnn.copy_host_to_device_tensor a fresh replay value into the same
    # buffer between begin/end_trace_capture and execute_trace.
    # ttnn.deallocate(args_0, False)
    ttnn_reshape_19 = ttnn.reshape(
        ttnn_typecast_35,
        [32],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_35, False)
    ttnn_to_layout_105 = ttnn.to_layout(ttnn_reshape_19, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_reshape_19, False)
    ttnn_embedding_0 = ttnn.embedding(
        ttnn_to_layout_105,
        ce_cache__main["main_const_eval_30"],
        padding_idx=None,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_105, False)
    ttnn_typecast_36 = ttnn.typecast(
        ttnn_embedding_0,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_embedding_0, False)
    ttnn_rms_in_4d_0 = ttnn.reshape(
        ttnn_typecast_36,
        [1, 1, 32, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    _tracy.signpost("layer_0_start")
    ttnn_rms_stats_4d_0 = ttnn.rms_norm_pre_all_gather(
        ttnn_rms_in_4d_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_rms_sliced_0 = ttnn.slice(
        ttnn_rms_stats_4d_0,
        [0, 0, 0, 0],
        [1, 1, 32, 1],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_rms_stats_4d_0, False)
    ttnn_reshape_20 = ttnn.reshape(
        ttnn_rms_sliced_0,
        [1, 32, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_rms_sliced_0, False)
    ttnn_all_gather_2 = ttnn.all_gather(
        input_tensor=ttnn_reshape_20,
        dim=0,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_20, False)
    ttnn_sum_1 = ttnn.sum(
        ttnn_all_gather_2,
        [0],
        False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_all_gather_2, False)
    ttnn_multiply_0 = ttnn.multiply(
        ttnn_sum_1,
        var_70,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_sum_1, False)
    ttnn_add_0 = ttnn.add(
        ttnn_multiply_0,
        var_67,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_multiply_0, False)
    ttnn_rsqrt_0 = ttnn.rsqrt(
        ttnn_add_0,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_0, False)
    ttnn_multiply_1 = ttnn.multiply(
        ttnn_typecast_36,
        ttnn_rsqrt_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_rsqrt_0, False)
    ttnn_typecast_37 = ttnn.multiply(
        ce_cache__main["main_const_eval_40"],
        ttnn_multiply_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_multiply_1, False)
    ttnn_matmul_0 = ttnn.matmul(
        ttnn_typecast_37,
        ce_cache__main["main_const_eval_32"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_typecast_37, False)
    ttnn_slice_68 = ttnn.slice(
        ttnn_matmul_0,
        [0, 0],
        [32, 576],
        [1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_slice_70 = ttnn.slice(
        ttnn_matmul_0,
        [0, 640],
        [32, 2176],
        [1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_slice_71 = ttnn.slice(
        ttnn_matmul_0,
        [0, 2176],
        [32, 2304],
        [1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_0, False)
    ttnn_reshape_21 = ttnn.reshape(
        ttnn_slice_71,
        [1, 32, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_slice_71, False)
    ttnn_all_gather_3 = ttnn.all_gather(
        input_tensor=ttnn_reshape_21,
        dim=0,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_21, False)
    ttnn_sum_2 = ttnn.sum(
        ttnn_all_gather_3,
        [0],
        False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_all_gather_3, False)
    ttnn_typecast_38 = ttnn.typecast(
        ttnn_sum_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_sum_2, False)
    ttnn_reshape_22 = ttnn.reshape(
        ttnn_typecast_38,
        [32, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_38, False)
    ttnn_layer_norm_0 = ttnn.layer_norm(
        ttnn_reshape_22,
        epsilon=9.9999999747524271e-07,
        weight=weights["model.transformer.layers.0.attn.indexer.k_norm.weight"],
        bias=weights["model.transformer.layers.0.attn.indexer.k_norm.bias"],
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        program_config=None,
    )
    ttnn.deallocate(ttnn_reshape_22, False)
    ttnn_typecast_39 = ttnn.typecast(
        ttnn_layer_norm_0,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_layer_norm_0, False)
    ttnn_slice_72 = ttnn.slice(
        ttnn_typecast_39,
        [0, 0, 0],
        [32, 1, 64],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    # E47 iter3: RoPE site 1 (layer 0 indexer-K rope-part) — build kernel input
    # Convert half-concat [32,1,64] → interleaved-pair [1,1,32,64] BF16 for the kernel.
    _rope0_x_split = ttnn.reshape(
        ttnn_slice_72,
        [32, 1, 2, 32],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_slice_72, False)
    _rope0_x_perm = ttnn.permute(
        _rope0_x_split,
        [0, 1, 3, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(_rope0_x_split, False)
    _rope0_x = ttnn.reshape(
        _rope0_x_perm,
        [1, 1, 32, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_rope0_x_perm, False)
    ttnn_typecast_42 = ttnn.add(
        ttnn_to_layout_104,
        ce_cache__main["main_const_eval_52"],
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_41 = ttnn.typecast(
        ttnn_gt_1,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_gt_1, False)
    ttnn_to_layout_106 = ttnn.to_layout(args_1, ttnn.Layout.TILE, None, memory_config=None)
    # Path B: args_1 (cache_position) must survive _main so pcc.py --trace can
    # refill it via ttnn.copy_host_to_device_tensor before execute_trace.
    # ttnn.deallocate(args_1, False)
    ttnn_typecast_43 = ttnn.typecast(
        ttnn_to_layout_106,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_106, False)
    ttnn_where_0 = ttnn.where(
        ttnn_typecast_41,
        ttnn_typecast_42,
        ttnn_typecast_43,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_43, False)
    ttnn.deallocate(ttnn_typecast_42, False)
    ttnn.deallocate(ttnn_typecast_41, False)
    ttnn_typecast_44 = ttnn.typecast(
        ttnn_where_0,
        ttnn.DataType.UINT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_where_0, False)
    ttnn_to_layout_107 = ttnn.to_layout(ttnn_typecast_44, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_typecast_44, False)
    # E47 iter4: shared cos/sin precomputation (one position lookup, used by all RoPE sites).
    _rope_cos_pos = ttnn.embedding(
        ttnn_to_layout_107,
        ce_cache__main["main_const_eval_rope_cos_doubled"],
        padding_idx=None,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    _rope_sin_pos = ttnn.embedding(
        ttnn_to_layout_107,
        ce_cache__main["main_const_eval_rope_sin_doubled"],
        padding_idx=None,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    # Restore the original cos/sin source for the remaining RoPE sites that still
    # use the addcmul-style chain (sites 3-6 will be migrated incrementally).
    ttnn_embedding_1 = ttnn.embedding(
        ttnn_to_layout_107,
        ce_cache__main["main_const_eval_13"],
        padding_idx=None,
        layout=ttnn.Layout.TILE,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_107, False)
    ttnn_reshape_24 = ttnn.reshape(
        ttnn_embedding_1,
        [1, 1, 1, 32, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_embedding_1, False)
    ttnn_slice_74 = ttnn.slice(
        ttnn_reshape_24,
        [0, 0, 0, 0, 0],
        [1, 1, 1, 32, 1],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_45 = ttnn.typecast(
        ttnn_slice_74,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_slice_74, False)
    ttnn_slice_76 = ttnn.slice(
        ttnn_reshape_24,
        [0, 0, 0, 0, 1],
        [1, 1, 1, 32, 2],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_24, False)
    ttnn_typecast_46 = ttnn.typecast(
        ttnn_slice_76,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_slice_76, False)
    # [1, 64] → [1, 1, 1, 64] for per-site repeat to whatever seq_len each site needs.
    _rope_cos_pos_4d = ttnn.reshape(
        _rope_cos_pos,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_rope_cos_pos, False)
    _rope_sin_pos_4d = ttnn.reshape(
        _rope_sin_pos,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_rope_sin_pos, False)
    # Per-site repeat for site 1 (seq_len=32).
    _rope0_cos_tiled = ttnn.repeat(_rope_cos_pos_4d, ttnn.Shape([1, 1, 32, 1]))
    _rope0_sin_tiled = ttnn.repeat(_rope_sin_pos_4d, ttnn.Shape([1, 1, 32, 1]))
    # Run the kernel (prefill mode, DRAM-interleaved I/O).
    _rope0_out = ttnn.experimental.rotary_embedding_llama(
        _rope0_x,
        _rope0_cos_tiled,
        _rope0_sin_tiled,
        ce_cache__main["main_const_eval_rope_trans_mat"],
        is_decode_mode=False,
    )
    ttnn.deallocate(_rope0_x, False)
    ttnn.deallocate(_rope0_cos_tiled, False)
    ttnn.deallocate(_rope0_sin_tiled, False)
    # Output [1, 1, 32, 64] BF16 interleaved-pair → convert back to half-concat [32, 64].
    _rope0_out_5d = ttnn.reshape(
        _rope0_out,
        [32, 1, 1, 32, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_rope0_out, False)
    _rope0_out_perm = ttnn.permute(
        _rope0_out_5d,
        [0, 1, 2, 4, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(_rope0_out_5d, False)
    ttnn_typecast_47 = ttnn.reshape(
        _rope0_out_perm,
        [32, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_rope0_out_perm, False)
    ttnn_slice_79 = ttnn.slice(
        ttnn_typecast_39,
        [0, 0, 64],
        [32, 1, 128],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_39, False)
    ttnn_reshape_26 = ttnn.reshape(
        ttnn_slice_79,
        [32, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_slice_79, False)
    ttnn_concat_4 = ttnn.concat(
        [ttnn_typecast_47, ttnn_reshape_26],
        1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_26, False)
    ttnn.deallocate(ttnn_typecast_47, False)
    ttnn_matmul_1 = ttnn.matmul(
        ttnn_concat_4,
        var_72[0],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_concat_4, False)
    ttnn_reshape_27 = ttnn.reshape(
        ttnn_matmul_1,
        [1, 32, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_1, False)
    ttnn_repeat_3 = ttnn.repeat(
        ttnn_to_layout_104,
        ttnn.Shape([32]),
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_memory_config_0 = ttnn.to_memory_config(
        ttnn_reshape_27,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 1)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
                    ]
                ),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_reshape_27, False)
    ttnn_to_layout_108 = ttnn.to_layout(ttnn_repeat_3, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_repeat_3, False)
    ttnn.experimental.paged_update_cache(
        model_transformer_layers_0_attn_indexer_k_cache,
        ttnn_to_memory_config_0,
        update_idxs_tensor=ttnn_to_layout_108,
        share_cache=False,
        page_table=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_0, False)
    ttnn_reshape_29 = ttnn.reshape(
        ttnn_slice_70,
        [1, 1, 32, 1536],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_slice_70, False)
    ttnn_reduce_scatter_0 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_29,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_29, False)
    ttnn_reshape_30 = ttnn.reshape(
        ttnn_reduce_scatter_0,
        [32, 192],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reduce_scatter_0, False)
    ttnn_all_gather_4 = ttnn.all_gather(
        input_tensor=ttnn_reshape_30,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_30, False)
    ttnn_rms_norm_0 = ttnn.rms_norm(
        ttnn_all_gather_4,
        epsilon=9.9999999747524271e-07,
        weight=weights["model.transformer.layers.0.attn.q_norm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn.deallocate(ttnn_all_gather_4, False)
    ttnn_matmul_5 = ttnn.matmul(
        ttnn_rms_norm_0,
        ce_cache__main["main_const_eval_14"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_0, False)
    ttnn_reshape_42 = ttnn.reshape(
        ttnn_matmul_5,
        [32, 1, 16, 192],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_5, False)
    ttnn_slice_86 = ttnn.slice(
        ttnn_reshape_42,
        [0, 0, 0, 0],
        [32, 1, 16, 128],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_29 = ttnn.permute(
        ttnn_slice_86,
        [2, 0, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_slice_86, False)
    ttnn_reshape_43 = ttnn.reshape(
        ttnn_permute_29,
        [16, 32, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_29, False)
    ttnn_matmul_6 = ttnn.matmul(
        ttnn_reshape_43,
        var_73[0],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_reshape_43, False)
    ttnn_reshape_44 = ttnn.reshape(
        ttnn_matmul_6,
        [16, 32, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_6, False)
    ttnn_permute_30 = ttnn.permute(
        ttnn_reshape_44,
        [1, 2, 0, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_44, False)
    ttnn_reshape_45 = ttnn.reshape(
        ttnn_slice_68,
        [1, 32, 576],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_slice_68, False)
    ttnn_all_gather_6 = ttnn.all_gather(
        input_tensor=ttnn_reshape_45,
        dim=0,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_45, False)
    ttnn_sum_5 = ttnn.sum(
        ttnn_all_gather_6,
        [0],
        False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_all_gather_6, False)
    ttnn_slice_87 = ttnn.slice(
        ttnn_sum_5,
        [0, 0],
        [32, 512],
        [1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_46 = ttnn.reshape(
        ttnn_slice_87,
        [32, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_slice_87, False)
    ttnn_rms_norm_1 = ttnn.rms_norm(
        ttnn_reshape_46,
        epsilon=9.9999999747524271e-07,
        weight=weights["model.transformer.layers.0.attn.kv_norm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn.deallocate(ttnn_reshape_46, False)
    ttnn_reshape_47 = ttnn.reshape(
        ttnn_rms_norm_1,
        [1, 32, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_rms_norm_1, False)
    ttnn_to_memory_config_1 = ttnn.to_memory_config(
        ttnn_reshape_47,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 1)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
                    ]
                ),
                [32, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_reshape_47, False)
    ttnn.experimental.paged_update_cache(
        L__past_key_values___layers_0_compressed_kv,
        ttnn_to_memory_config_1,
        update_idxs_tensor=ttnn_to_layout_108,
        share_cache=False,
        page_table=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_1, False)
    ttnn_reshape_48 = ttnn.reshape(
        L__past_key_values___layers_0_compressed_kv,
        [32, 128, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_49 = ttnn.reshape(
        ttnn_permute_30,
        [32, 16, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_30, False)
    ttnn_slice_88 = ttnn.slice(
        ttnn_reshape_42,
        [0, 0, 0, 128],
        [32, 1, 16, 192],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_42, False)
    # E47 iter5: RoPE site 2 (layer 0 Q_rope q_b nope) via rotary_embedding_llama.
    # Input slice_88 [32, 1, 16, 64] BF16 interleaved-pair → reshape to [1, 1, 512, 64]
    # (treat batch*n_heads = 32*16 = 512 as seq_len, all share same args_1 position).
    _rope1_x = ttnn.reshape(
        ttnn_slice_88,
        [1, 1, 512, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_slice_88, False)
    _rope1_cos = ttnn.repeat(_rope_cos_pos_4d, ttnn.Shape([1, 1, 512, 1]))
    _rope1_sin = ttnn.repeat(_rope_sin_pos_4d, ttnn.Shape([1, 1, 512, 1]))
    _rope1_out = ttnn.experimental.rotary_embedding_llama(
        _rope1_x,
        _rope1_cos,
        _rope1_sin,
        ce_cache__main["main_const_eval_rope_trans_mat"],
        is_decode_mode=False,
    )
    ttnn.deallocate(_rope1_x, False)
    ttnn.deallocate(_rope1_cos, False)
    ttnn.deallocate(_rope1_sin, False)
    ttnn_reshape_51 = ttnn.reshape(
        _rope1_out,
        [32, 16, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_rope1_out, False)
    ttnn_slice_91 = ttnn.slice(
        ttnn_sum_5,
        [0, 512],
        [32, 576],
        [1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_sum_5, False)
    # E47 iter6: RoPE site 3 (layer 0 k_pe write) via rotary_embedding_llama.
    # slice_91 [32, 64] BF16 interleaved-pair → reshape [1, 1, 32, 64] for kernel.
    _rope2_x = ttnn.reshape(
        ttnn_slice_91,
        [1, 1, 32, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_slice_91, False)
    _rope2_cos = ttnn.repeat(_rope_cos_pos_4d, ttnn.Shape([1, 1, 32, 1]))
    _rope2_sin = ttnn.repeat(_rope_sin_pos_4d, ttnn.Shape([1, 1, 32, 1]))
    _rope2_out = ttnn.experimental.rotary_embedding_llama(
        _rope2_x,
        _rope2_cos,
        _rope2_sin,
        ce_cache__main["main_const_eval_rope_trans_mat"],
        is_decode_mode=False,
    )
    ttnn.deallocate(_rope2_x, False)
    ttnn.deallocate(_rope2_cos, False)
    ttnn.deallocate(_rope2_sin, False)
    ttnn_reshape_53 = ttnn.reshape(
        _rope2_out,
        [1, 32, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_rope2_out, False)
    ttnn_to_memory_config_2 = ttnn.to_memory_config(
        ttnn_reshape_53,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 1)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
                    ]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_reshape_53, False)
    ttnn.experimental.paged_update_cache(
        L__past_key_values___layers_0_k_pe,
        ttnn_to_memory_config_2,
        update_idxs_tensor=ttnn_to_layout_108,
        share_cache=False,
        page_table=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_2, False)
    ttnn_reshape_54 = ttnn.reshape(
        L__past_key_values___layers_0_k_pe,
        [32, 128, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    # === E_sdpa_layer0: flash_multi_latent_attention_decode replaces matmul_9 ===
    import math as _math

    _sdpa0_q_concat = ttnn.concat(
        [ttnn_reshape_49, ttnn_reshape_51],
        dim=-1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_49, False)
    ttnn.deallocate(ttnn_reshape_51, False)
    _sdpa0_q = ttnn.reshape(
        _sdpa0_q_concat,
        [1, 32, 16, 576],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_sdpa0_q_concat, False)
    _sdpa0_k_concat = ttnn.concat(
        [ttnn_reshape_48, ttnn_reshape_54],
        dim=-1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_48, False)
    ttnn.deallocate(ttnn_reshape_54, False)
    _sdpa0_k = ttnn.reshape(
        _sdpa0_k_concat,
        [32, 1, 128, 576],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_sdpa0_k_concat, False)
    _sdpa0_prog_cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=0,
        k_chunk_size=128,
        exp_approx_mode=False,
        max_cores_per_head_batch=1,
    )
    _sdpa0_out = ttnn.transformer.flash_multi_latent_attention_decode(
        _sdpa0_q,
        _sdpa0_k,
        None,
        head_dim_v=512,
        is_causal=True,
        cur_pos_tensor=ttnn_to_layout_108,
        scale=0.134765625,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        program_config=_sdpa0_prog_cfg,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(_sdpa0_q, False)
    ttnn.deallocate(_sdpa0_k, False)
    # E46: fold permute([1,0,2,3]) → permute([2,0,1,3]) into single permute([2,1,0,3])
    ttnn_permute_33 = ttnn.permute(
        _sdpa0_out,
        [2, 1, 0, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(_sdpa0_out, False)
    # === END E_sdpa_layer0 ===
    ttnn_reshape_61 = ttnn.reshape(
        ttnn_permute_33,
        [16, 32, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_33, False)
    ttnn_matmul_10 = ttnn.matmul(
        ttnn_reshape_61,
        var_73[1],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_reshape_61, False)
    ttnn_reshape_62 = ttnn.reshape(
        ttnn_matmul_10,
        [16, 32, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_10, False)
    ttnn_permute_34 = ttnn.permute(
        ttnn_reshape_62,
        [1, 2, 0, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_62, False)
    ttnn_reshape_63 = ttnn.reshape(
        ttnn_permute_34,
        [32, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_34, False)
    ttnn_matmul_11 = ttnn.matmul(
        ttnn_reshape_63,
        ce_cache__main["main_const_eval_21"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_reshape_63, False)
    ttnn_reshape_64 = ttnn.reshape(
        ttnn_matmul_11,
        [1, 1, 32, 7168],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_11, False)
    ttnn_reduce_scatter_2 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_64,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_64, False)
    ttnn_reshape_65 = ttnn.reshape(
        ttnn_reduce_scatter_2,
        [32, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reduce_scatter_2, False)
    ttnn_typecast_57 = ttnn.typecast(
        ttnn_reshape_65,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_65, False)
    ttnn_add_10 = ttnn.add(
        ttnn_typecast_57,
        ttnn_typecast_36,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_57, False)
    ttnn.deallocate(ttnn_typecast_36, False)
    ttnn_rms_in_4d_1 = ttnn.reshape(
        ttnn_add_10,
        [1, 1, 32, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    _tracy.signpost("attn_0_end")
    _tracy.signpost("mlp_0_start")
    ttnn_rms_stats_4d_1 = ttnn.rms_norm_pre_all_gather(
        ttnn_rms_in_4d_1,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_rms_sliced_1 = ttnn.slice(
        ttnn_rms_stats_4d_1,
        [0, 0, 0, 0],
        [1, 1, 32, 1],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_rms_stats_4d_1, False)
    ttnn_reshape_66 = ttnn.reshape(
        ttnn_rms_sliced_1,
        [1, 32, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_rms_sliced_1, False)
    ttnn_all_gather_8 = ttnn.all_gather(
        input_tensor=ttnn_reshape_66,
        dim=0,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_66, False)
    ttnn_sum_7 = ttnn.sum(
        ttnn_all_gather_8,
        [0],
        False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_all_gather_8, False)
    ttnn_multiply_25 = ttnn.multiply(
        ttnn_sum_7,
        var_70,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_sum_7, False)
    ttnn_add_11 = ttnn.add(
        ttnn_multiply_25,
        var_67,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_multiply_25, False)
    ttnn_rsqrt_1 = ttnn.rsqrt(
        ttnn_add_11,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_11, False)
    ttnn_multiply_26 = ttnn.multiply(
        ttnn_add_10,
        ttnn_rsqrt_1,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_rsqrt_1, False)
    ttnn_typecast_58 = ttnn.multiply(
        ce_cache__main["main_const_eval_41"],
        ttnn_multiply_26,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_multiply_26, False)
    ttnn_all_gather_9 = ttnn.all_gather(
        input_tensor=ttnn_typecast_58,
        dim=0,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_typecast_58, False)
    ttnn_matmul_12 = ttnn.matmul(
        ttnn_all_gather_9,
        ce_cache__main["main_const_eval_45"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn_reshape_67 = ttnn.reshape(
        ttnn_matmul_12,
        [1, 1, 128, 4608],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_12, False)
    ttnn_reduce_scatter_3 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_67,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_67, False)
    ttnn_reshape_68 = ttnn.reshape(
        ttnn_reduce_scatter_3,
        [128, 576],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reduce_scatter_3, False)
    ttnn_all_gather_10 = ttnn.all_gather(
        input_tensor=ttnn_reshape_68,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_68, False)
    ttnn_typecast_59 = ttnn.typecast(
        ttnn_all_gather_10,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_all_gather_10, False)
    ttnn_matmul_13 = ttnn.matmul(
        ttnn_all_gather_9,
        ce_cache__main["main_const_eval_2"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_all_gather_9, False)
    ttnn_reshape_69 = ttnn.reshape(
        ttnn_matmul_13,
        [1, 1, 128, 4608],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_13, False)
    ttnn_reduce_scatter_4 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_69,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_69, False)
    ttnn_reshape_70 = ttnn.reshape(
        ttnn_reduce_scatter_4,
        [128, 576],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reduce_scatter_4, False)
    ttnn_all_gather_11 = ttnn.all_gather(
        input_tensor=ttnn_reshape_70,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_70, False)
    ttnn_typecast_60 = ttnn.typecast(
        ttnn_all_gather_11,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_all_gather_11, False)
    ttnn_typecast_61 = ttnn.multiply(
        ttnn_typecast_59,
        ttnn_typecast_60,
        dtype=ttnn.DataType.BFLOAT16,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_60, False)
    ttnn.deallocate(ttnn_typecast_59, False)
    ttnn_matmul_14 = ttnn.matmul(
        ttnn_typecast_61,
        ce_cache__main["main_const_eval_43"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_typecast_61, False)
    ttnn_reshape_71 = ttnn.reshape(
        ttnn_matmul_14,
        [1, 1, 128, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_14, False)
    ttnn_reduce_scatter_5 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_71,
        dim=2,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_71, False)
    ttnn_reshape_72 = ttnn.reshape(
        ttnn_reduce_scatter_5,
        [32, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reduce_scatter_5, False)
    ttnn_typecast_62 = ttnn.typecast(
        ttnn_reshape_72,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_72, False)
    ttnn_add_12 = ttnn.add(
        ttnn_typecast_62,
        ttnn_add_10,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_62, False)
    ttnn.deallocate(ttnn_add_10, False)
    ttnn_rms_in_4d_2 = ttnn.reshape(
        ttnn_add_12,
        [1, 1, 32, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    _tracy.signpost("layer_0_end")
    _tracy.signpost("layer_1_start")
    ttnn_rms_stats_4d_2 = ttnn.rms_norm_pre_all_gather(
        ttnn_rms_in_4d_2,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_rms_sliced_2 = ttnn.slice(
        ttnn_rms_stats_4d_2,
        [0, 0, 0, 0],
        [1, 1, 32, 1],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_rms_stats_4d_2, False)
    ttnn_reshape_73 = ttnn.reshape(
        ttnn_rms_sliced_2,
        [1, 32, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_rms_sliced_2, False)
    ttnn_all_gather_12 = ttnn.all_gather(
        input_tensor=ttnn_reshape_73,
        dim=0,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_73, False)
    ttnn_sum_9 = ttnn.sum(
        ttnn_all_gather_12,
        [0],
        False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_all_gather_12, False)
    ttnn_multiply_29 = ttnn.multiply(
        ttnn_sum_9,
        var_70,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_sum_9, False)
    ttnn_add_13 = ttnn.add(
        ttnn_multiply_29,
        var_67,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_multiply_29, False)
    ttnn_rsqrt_2 = ttnn.rsqrt(
        ttnn_add_13,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_13, False)
    ttnn_multiply_30 = ttnn.multiply(
        ttnn_add_12,
        ttnn_rsqrt_2,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_rsqrt_2, False)
    ttnn_typecast_63 = ttnn.multiply(
        ce_cache__main["main_const_eval_34"],
        ttnn_multiply_30,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_multiply_30, False)
    ttnn_matmul_15 = ttnn.matmul(
        ttnn_typecast_63,
        ce_cache__main["main_const_eval_28"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_typecast_63, False)
    ttnn_slice_161 = ttnn.slice(
        ttnn_matmul_15,
        [0, 0],
        [32, 576],
        [1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_slice_163 = ttnn.slice(
        ttnn_matmul_15,
        [0, 640],
        [32, 2176],
        [1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_slice_164 = ttnn.slice(
        ttnn_matmul_15,
        [0, 2176],
        [32, 2304],
        [1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_15, False)
    ttnn_reshape_74 = ttnn.reshape(
        ttnn_slice_164,
        [1, 32, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_slice_164, False)
    ttnn_all_gather_13 = ttnn.all_gather(
        input_tensor=ttnn_reshape_74,
        dim=0,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_74, False)
    ttnn_sum_10 = ttnn.sum(
        ttnn_all_gather_13,
        [0],
        False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_all_gather_13, False)
    ttnn_typecast_64 = ttnn.typecast(
        ttnn_sum_10,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_sum_10, False)
    ttnn_reshape_75 = ttnn.reshape(
        ttnn_typecast_64,
        [32, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_64, False)
    ttnn_layer_norm_1 = ttnn.layer_norm(
        ttnn_reshape_75,
        epsilon=9.9999999747524271e-07,
        weight=weights["model.transformer.layers.1.attn.indexer.k_norm.weight"],
        bias=weights["model.transformer.layers.1.attn.indexer.k_norm.bias"],
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        program_config=None,
    )
    ttnn.deallocate(ttnn_reshape_75, False)
    ttnn_typecast_65 = ttnn.typecast(
        ttnn_layer_norm_1,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_layer_norm_1, False)
    ttnn_slice_165 = ttnn.slice(
        ttnn_typecast_65,
        [0, 0, 0],
        [32, 1, 64],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    # E47 iter7: RoPE site 4 (layer 1 indexer-K rope-part) — mirror of site 1.
    # slice_165 [32, 1, 64] BF16 half-concat → reshape+permute→ [1, 1, 32, 64] interleaved-pair.
    _rope3_x_split = ttnn.reshape(
        ttnn_slice_165,
        [32, 1, 2, 32],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_slice_165, False)
    _rope3_x_perm = ttnn.permute(
        _rope3_x_split,
        [0, 1, 3, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(_rope3_x_split, False)
    _rope3_x = ttnn.reshape(
        _rope3_x_perm,
        [1, 1, 32, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_rope3_x_perm, False)
    _rope3_cos = ttnn.repeat(_rope_cos_pos_4d, ttnn.Shape([1, 1, 32, 1]))
    _rope3_sin = ttnn.repeat(_rope_sin_pos_4d, ttnn.Shape([1, 1, 32, 1]))
    _rope3_out = ttnn.experimental.rotary_embedding_llama(
        _rope3_x,
        _rope3_cos,
        _rope3_sin,
        ce_cache__main["main_const_eval_rope_trans_mat"],
        is_decode_mode=False,
    )
    ttnn.deallocate(_rope3_x, False)
    ttnn.deallocate(_rope3_cos, False)
    ttnn.deallocate(_rope3_sin, False)
    # Output [1, 1, 32, 64] interleaved-pair → convert back to half-concat [32, 64].
    _rope3_out_5d = ttnn.reshape(
        _rope3_out,
        [32, 1, 1, 32, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_rope3_out, False)
    _rope3_out_perm = ttnn.permute(
        _rope3_out_5d,
        [0, 1, 2, 4, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(_rope3_out_5d, False)
    ttnn_typecast_67 = ttnn.reshape(
        _rope3_out_perm,
        [32, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_rope3_out_perm, False)
    ttnn_slice_170 = ttnn.slice(
        ttnn_typecast_65,
        [0, 0, 64],
        [32, 1, 128],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_65, False)
    ttnn_reshape_78 = ttnn.reshape(
        ttnn_slice_170,
        [32, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_slice_170, False)
    ttnn_concat_13 = ttnn.concat(
        [ttnn_typecast_67, ttnn_reshape_78],
        1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_78, False)
    ttnn.deallocate(ttnn_typecast_67, False)
    ttnn_matmul_16 = ttnn.matmul(
        ttnn_concat_13,
        var_66[0],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_concat_13, False)
    ttnn_reshape_79 = ttnn.reshape(
        ttnn_matmul_16,
        [1, 32, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_16, False)
    ttnn_to_memory_config_3 = ttnn.to_memory_config(
        ttnn_reshape_79,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 1)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
                    ]
                ),
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_reshape_79, False)
    ttnn.experimental.paged_update_cache(
        model_transformer_layers_1_attn_indexer_k_cache,
        ttnn_to_memory_config_3,
        update_idxs_tensor=ttnn_to_layout_108,
        share_cache=False,
        page_table=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_3, False)
    ttnn_reshape_81 = ttnn.reshape(
        ttnn_slice_163,
        [1, 1, 32, 1536],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_slice_163, False)
    ttnn_reduce_scatter_6 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_81,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_81, False)
    ttnn_reshape_82 = ttnn.reshape(
        ttnn_reduce_scatter_6,
        [32, 192],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reduce_scatter_6, False)
    ttnn_all_gather_14 = ttnn.all_gather(
        input_tensor=ttnn_reshape_82,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_82, False)
    ttnn_rms_norm_2 = ttnn.rms_norm(
        ttnn_all_gather_14,
        epsilon=9.9999999747524271e-07,
        weight=weights["model.transformer.layers.1.attn.q_norm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn.deallocate(ttnn_all_gather_14, False)
    ttnn_matmul_20 = ttnn.matmul(
        ttnn_rms_norm_2,
        ce_cache__main["main_const_eval_54"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_rms_norm_2, False)
    ttnn_reshape_93 = ttnn.reshape(
        ttnn_matmul_20,
        [32, 1, 16, 192],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_20, False)
    ttnn_slice_177 = ttnn.slice(
        ttnn_reshape_93,
        [0, 0, 0, 0],
        [32, 1, 16, 128],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_38 = ttnn.permute(
        ttnn_slice_177,
        [2, 0, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_slice_177, False)
    ttnn_reshape_94 = ttnn.reshape(
        ttnn_permute_38,
        [16, 32, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_38, False)
    ttnn_matmul_21 = ttnn.matmul(
        ttnn_reshape_94,
        var_78[0],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_reshape_94, False)
    ttnn_reshape_95 = ttnn.reshape(
        ttnn_matmul_21,
        [16, 32, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_21, False)
    ttnn_permute_39 = ttnn.permute(
        ttnn_reshape_95,
        [1, 2, 0, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_95, False)
    ttnn_reshape_96 = ttnn.reshape(
        ttnn_slice_161,
        [1, 32, 576],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_slice_161, False)
    ttnn_all_gather_16 = ttnn.all_gather(
        input_tensor=ttnn_reshape_96,
        dim=0,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_96, False)
    ttnn_sum_13 = ttnn.sum(
        ttnn_all_gather_16,
        [0],
        False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_all_gather_16, False)
    ttnn_slice_178 = ttnn.slice(
        ttnn_sum_13,
        [0, 0],
        [32, 512],
        [1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_97 = ttnn.reshape(
        ttnn_slice_178,
        [32, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_slice_178, False)
    ttnn_rms_norm_3 = ttnn.rms_norm(
        ttnn_reshape_97,
        epsilon=9.9999999747524271e-07,
        weight=weights["model.transformer.layers.1.attn.kv_norm.weight"],
        bias=None,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        program_config=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    ttnn.deallocate(ttnn_reshape_97, False)
    ttnn_reshape_98 = ttnn.reshape(
        ttnn_rms_norm_3,
        [1, 32, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_rms_norm_3, False)
    ttnn_to_memory_config_4 = ttnn.to_memory_config(
        ttnn_reshape_98,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 1)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
                    ]
                ),
                [32, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_reshape_98, False)
    ttnn.experimental.paged_update_cache(
        L__past_key_values___layers_1_compressed_kv,
        ttnn_to_memory_config_4,
        update_idxs_tensor=ttnn_to_layout_108,
        share_cache=False,
        page_table=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_4, False)
    ttnn_reshape_99 = ttnn.reshape(
        L__past_key_values___layers_1_compressed_kv,
        [32, 128, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_100 = ttnn.reshape(
        ttnn_permute_39,
        [32, 16, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_39, False)
    ttnn_slice_179 = ttnn.slice(
        ttnn_reshape_93,
        [0, 0, 0, 128],
        [32, 1, 16, 192],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_93, False)
    # E47 iter8: RoPE site 5 (layer 1 Q_rope q_b nope) — mirror of site 2.
    _rope4_x = ttnn.reshape(
        ttnn_slice_179,
        [1, 1, 512, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_slice_179, False)
    _rope4_cos = ttnn.repeat(_rope_cos_pos_4d, ttnn.Shape([1, 1, 512, 1]))
    _rope4_sin = ttnn.repeat(_rope_sin_pos_4d, ttnn.Shape([1, 1, 512, 1]))
    _rope4_out = ttnn.experimental.rotary_embedding_llama(
        _rope4_x,
        _rope4_cos,
        _rope4_sin,
        ce_cache__main["main_const_eval_rope_trans_mat"],
        is_decode_mode=False,
    )
    ttnn.deallocate(_rope4_x, False)
    ttnn.deallocate(_rope4_cos, False)
    ttnn.deallocate(_rope4_sin, False)
    ttnn_reshape_102 = ttnn.reshape(
        _rope4_out,
        [32, 16, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_rope4_out, False)
    ttnn_slice_182 = ttnn.slice(
        ttnn_sum_13,
        [0, 512],
        [32, 576],
        [1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_sum_13, False)
    # E47 iter9: RoPE site 6 (layer 1 k_pe write) — mirror of site 3.
    _rope5_x = ttnn.reshape(
        ttnn_slice_182,
        [1, 1, 32, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_slice_182, False)
    _rope5_cos = ttnn.repeat(_rope_cos_pos_4d, ttnn.Shape([1, 1, 32, 1]))
    _rope5_sin = ttnn.repeat(_rope_sin_pos_4d, ttnn.Shape([1, 1, 32, 1]))
    _rope5_out = ttnn.experimental.rotary_embedding_llama(
        _rope5_x,
        _rope5_cos,
        _rope5_sin,
        ce_cache__main["main_const_eval_rope_trans_mat"],
        is_decode_mode=False,
    )
    ttnn.deallocate(_rope5_x, False)
    ttnn.deallocate(_rope5_cos, False)
    ttnn.deallocate(_rope5_sin, False)
    ttnn_reshape_104 = ttnn.reshape(
        _rope5_out,
        [1, 32, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_rope5_out, False)
    # Final dealloc of the now-unused cos/sin source tensors (originally consumed
    # by the addcmul chain; with all 6 sites migrated, these can be released).
    ttnn.deallocate(ttnn_typecast_45, False)
    ttnn.deallocate(ttnn_typecast_46, False)
    ttnn_to_memory_config_5 = ttnn.to_memory_config(
        ttnn_reshape_104,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 1)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(7, 2)),
                    ]
                ),
                [32, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_reshape_104, False)
    ttnn.experimental.paged_update_cache(
        L__past_key_values___layers_1_k_pe,
        ttnn_to_memory_config_5,
        update_idxs_tensor=ttnn_to_layout_108,
        share_cache=False,
        page_table=None,
    )
    ttnn.deallocate(ttnn_to_memory_config_5, False)
    ttnn_reshape_105 = ttnn.reshape(
        L__past_key_values___layers_1_k_pe,
        [32, 128, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    # === E_sdpa_layer1: flash_multi_latent_attention_decode replaces matmul_24 ===
    import math as _math

    _sdpa1_q_concat = ttnn.concat(
        [ttnn_reshape_100, ttnn_reshape_102],
        dim=-1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_100, False)
    ttnn.deallocate(ttnn_reshape_102, False)
    _sdpa1_q = ttnn.reshape(
        _sdpa1_q_concat,
        [1, 32, 16, 576],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_sdpa1_q_concat, False)
    _sdpa1_k_concat = ttnn.concat(
        [ttnn_reshape_99, ttnn_reshape_105],
        dim=-1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_99, False)
    ttnn.deallocate(ttnn_reshape_105, False)
    _sdpa1_k = ttnn.reshape(
        _sdpa1_k_concat,
        [32, 1, 128, 576],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_sdpa1_k_concat, False)
    _sdpa1_prog_cfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=0,
        k_chunk_size=128,
        exp_approx_mode=False,
        max_cores_per_head_batch=1,
    )
    _sdpa1_out = ttnn.transformer.flash_multi_latent_attention_decode(
        _sdpa1_q,
        _sdpa1_k,
        None,
        head_dim_v=512,
        is_causal=True,
        cur_pos_tensor=ttnn_to_layout_108,
        scale=0.134765625,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        program_config=_sdpa1_prog_cfg,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(_sdpa1_q, False)
    ttnn.deallocate(_sdpa1_k, False)
    ttnn.deallocate(ttnn_to_layout_108, False)
    # E46: fold permute([1,0,2,3]) → permute([2,0,1,3]) into single permute([2,1,0,3])
    ttnn_permute_42 = ttnn.permute(
        _sdpa1_out,
        [2, 1, 0, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(_sdpa1_out, False)
    # === END E_sdpa_layer1 ===
    ttnn_reshape_112 = ttnn.reshape(
        ttnn_permute_42,
        [16, 32, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_42, False)
    ttnn_matmul_25 = ttnn.matmul(
        ttnn_reshape_112,
        var_78[1],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_reshape_112, False)
    ttnn_reshape_113 = ttnn.reshape(
        ttnn_matmul_25,
        [16, 32, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_25, False)
    ttnn_permute_43 = ttnn.permute(
        ttnn_reshape_113,
        [1, 2, 0, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_113, False)
    ttnn_reshape_114 = ttnn.reshape(
        ttnn_permute_43,
        [32, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_43, False)
    ttnn_matmul_26 = ttnn.matmul(
        ttnn_reshape_114,
        ce_cache__main["main_const_eval_15"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_reshape_114, False)
    ttnn_reshape_115 = ttnn.reshape(
        ttnn_matmul_26,
        [1, 1, 32, 7168],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_26, False)
    ttnn_reduce_scatter_8 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_115,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_115, False)
    ttnn_reshape_116 = ttnn.reshape(
        ttnn_reduce_scatter_8,
        [32, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reduce_scatter_8, False)
    ttnn_typecast_77 = ttnn.typecast(
        ttnn_reshape_116,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_116, False)
    ttnn_add_22 = ttnn.add(
        ttnn_typecast_77,
        ttnn_add_12,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_77, False)
    ttnn.deallocate(ttnn_add_12, False)
    ttnn_reshape_117 = ttnn.reshape(
        ttnn_add_22,
        [32, 1, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rms_in_4d_3 = ttnn.reshape(
        ttnn_reshape_117,
        [1, 1, 32, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    _tracy.signpost("attn_1_end")
    _tracy.signpost("moe_start")
    # === MoE-capture: dump the live-in activation tensors crossing the moe_start boundary ===
    import os as _os_cap

    _os_cap.makedirs("moe_io", exist_ok=True)
    ttnn.dump_tensor("moe_io/in_ttnn_add_22.tensorbin", ttnn_add_22)
    ttnn.dump_tensor("moe_io/in_ttnn_reshape_117.tensorbin", ttnn_reshape_117)
    ttnn.dump_tensor("moe_io/in_ttnn_rms_in_4d_3.tensorbin", ttnn_rms_in_4d_3)
    # === end MoE-capture inputs ===
    ttnn_rms_stats_4d_3 = ttnn.rms_norm_pre_all_gather(
        ttnn_rms_in_4d_3,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_rms_sliced_3 = ttnn.slice(
        ttnn_rms_stats_4d_3,
        [0, 0, 0, 0],
        [1, 1, 32, 1],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_rms_stats_4d_3, False)
    ttnn_reshape_118 = ttnn.reshape(
        ttnn_rms_sliced_3,
        [1, 32, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_rms_sliced_3, False)
    ttnn_all_gather_18 = ttnn.all_gather(
        input_tensor=ttnn_reshape_118,
        dim=0,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_118, False)
    ttnn_sum_15 = ttnn.sum(
        ttnn_all_gather_18,
        [0],
        False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_all_gather_18, False)
    ttnn_multiply_54 = ttnn.multiply(
        ttnn_sum_15,
        var_70,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_sum_15, False)
    ttnn_add_23 = ttnn.add(
        ttnn_multiply_54,
        var_67,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_multiply_54, False)
    ttnn_rsqrt_3 = ttnn.rsqrt(
        ttnn_add_23,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_23, False)
    ttnn_reshape_119 = ttnn.reshape(
        ttnn_rsqrt_3,
        [32, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_rsqrt_3, False)
    ttnn_multiply_55 = ttnn.multiply(
        ttnn_reshape_117,
        ttnn_reshape_119,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_119, False)
    ttnn.deallocate(ttnn_reshape_117, False)
    ttnn_typecast_78 = ttnn.multiply(
        ce_cache__main["main_const_eval_24"],
        ttnn_multiply_55,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_multiply_55, False)
    ttnn_reshape_120 = ttnn.reshape(
        ttnn_typecast_78,
        [32, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_79 = ttnn.typecast(
        ttnn_reshape_120,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_27 = ttnn.matmul(
        ttnn_typecast_79,
        ce_cache__main["main_const_eval_25"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.FLOAT32,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_typecast_79, False)
    ttnn_reshape_121 = ttnn.reshape(
        ttnn_matmul_27,
        [1, 1, 32, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_27, False)
    ttnn_reduce_scatter_9 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_121,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_121, False)
    ttnn_reshape_122 = ttnn.reshape(
        ttnn_reduce_scatter_9,
        [32, 32],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reduce_scatter_9, False)
    ttnn_all_gather_19 = ttnn.all_gather(
        input_tensor=ttnn_reshape_122,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_122, False)
    # === E_router_fuse: deepseek_grouped_gate replaces ~50-op router block ===
    # Inputs:
    #   ttnn_all_gather_19  - FP32 TILE [32, 256]   gate logits (pre-sigmoid)
    #   main_const_eval_9   - FP32 TILE [1,  256]   routing bias
    # Outputs (must keep these variable names for downstream rewiring):
    #   ttnn_multiply_58    - FP32       [32, 1, 8] normalized scaled weights
    #   ttnn_typecast_86    - INT32      [32, 8]    selected expert indices
    _dgg_scores_bf16 = ttnn.typecast(
        ttnn_all_gather_19,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_all_gather_19, False)
    _dgg_scores_4d = ttnn.reshape(
        _dgg_scores_bf16,
        [1, 1, 32, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_dgg_scores_bf16, False)
    _dgg_bias_bf16 = ttnn.typecast(
        ce_cache__main["main_const_eval_9"],
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    _dgg_bias_4d = ttnn.reshape(
        _dgg_bias_bf16,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_dgg_bias_bf16, False)
    _dgg_bias_bcast = ttnn.repeat(
        _dgg_bias_4d,
        ttnn.Shape([1, 1, 32, 1]),
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_dgg_bias_4d, False)
    _dgg_weights_bf16, _dgg_indices_u16 = ttnn.experimental.deepseek_grouped_gate(
        _dgg_scores_4d,
        _dgg_bias_bcast,
        n_groups=8,
        summed_experts_per_group=2,
        topk_groups=4,
        n_activated_experts=8,
        route_scale=2.5,
        epsilon=1e-20,
    )
    ttnn.deallocate(_dgg_scores_4d, False)
    ttnn.deallocate(_dgg_bias_bcast, False)
    # E45: drop the BF16→FP32 typecast on _dgg_weights — feed BF16 weights to matmul_29
    ttnn_multiply_58 = ttnn.reshape(
        _dgg_weights_bf16,
        [32, 1, 8],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_dgg_weights_bf16, False)
    _dgg_indices_i32 = ttnn.typecast(
        _dgg_indices_u16,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_dgg_indices_u16, False)
    ttnn_typecast_86 = ttnn.reshape(
        _dgg_indices_i32,
        [32, 8],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_dgg_indices_i32, False)
    # === END E_router_fuse ===
    ttnn_reshape_140 = ttnn.reshape(
        ttnn_typecast_86,
        [32, 8, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_eq_0 = ttnn.eq(
        ttnn_reshape_140,
        ce_cache__main["main_const_eval_35"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_140, False)
    ttnn_typecast_92 = ttnn.typecast(
        ttnn_eq_0,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    # E45: matmul_29 in BF16 — input A (weights) is now BF16, input B (eq mask) skips typecast
    ttnn_matmul_29 = ttnn.matmul(
        ttnn_multiply_58,
        ttnn_eq_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_eq_0, False)
    ttnn.deallocate(ttnn_multiply_58, False)
    ttnn_reshape_141 = ttnn.reshape(
        ttnn_matmul_29,
        [1, 32, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_concat_22 = ttnn.concat(
        [ttnn_reshape_141, ttnn_reshape_141, ttnn_reshape_141, ttnn_reshape_141],
        1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_141, False)
    ttnn_all_gather_24 = ttnn.all_gather(
        input_tensor=ttnn_concat_22,
        dim=1,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_concat_22, False)
    ttnn_reshape_142 = ttnn.reshape(
        ttnn_all_gather_24,
        [1, 1, 512, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_all_gather_24, False)
    ttnn_reshape_143 = ttnn.reshape(
        ttnn_typecast_78,
        [32, 1, 1, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_78, False)
    ttnn_reshape_144 = ttnn.reshape(
        ttnn_typecast_86,
        [32, 1, 1, 8],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_86, False)
    ttnn_all_gather_25 = ttnn.all_gather(
        input_tensor=ttnn_reshape_143,
        dim=0,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_143, False)
    ttnn_all_gather_26 = ttnn.all_gather(
        input_tensor=ttnn_all_gather_25,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_all_gather_25, False)
    ttnn_all_gather_27 = ttnn.all_gather(
        input_tensor=ttnn_reshape_144,
        dim=0,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_144, False)
    ttnn_to_layout_255 = ttnn.to_layout(ttnn_all_gather_26, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_all_gather_26, False)
    ttnn_typecast_93 = ttnn.typecast(
        ttnn_all_gather_27,
        ttnn.DataType.UINT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_all_gather_27, False)
    # E40 site 1: on-device to_layout(ROW_MAJOR) replaces from_device→to_layout→to_device round-trip
    ttnn_to_device_65 = ttnn.to_layout(
        ttnn_typecast_93,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_93, False)
    v_92, v_93 = ttnn.all_to_all_dispatch(
        input_tensor=ttnn_to_layout_255,
        expert_indices_tensor=ttnn_to_device_65,
        expert_mapping_tensor=var_76,
        cluster_axis=0,
        memory_config=None,
    )
    ttnn.deallocate(ttnn_to_device_65, False)
    ttnn.deallocate(ttnn_to_layout_255, False)
    ttnn_to_layout_257 = ttnn.to_layout(v_93, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(v_93, False)
    ttnn_typecast_94 = ttnn.typecast(
        ttnn_to_layout_257,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_257, False)
    ttnn_to_layout_258 = ttnn.to_layout(v_92, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(v_92, False)
    ttnn_reshape_145 = ttnn.reshape(
        ttnn_typecast_94,
        [1, 1, 512, 8],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_94, False)
    # E45: typecast_95 (FP32→BF16) eliminated — reshape_142 is already BF16 after matmul_29 fold
    ttnn_to_layout_259 = ttnn.to_layout(ttnn_reshape_142, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_reshape_142, False)
    ttnn_typecast_96 = ttnn.typecast(
        ttnn_reshape_145,
        ttnn.DataType.UINT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_145, False)
    # E40 site 2: on-device to_layout(ROW_MAJOR) replaces from_device→to_layout→to_device round-trip
    ttnn_to_device_66 = ttnn.to_layout(
        ttnn_typecast_96,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_96, False)
    v_94, v_95 = ttnn.moe_expert_token_remap(
        topk_tensor=ttnn_to_layout_259,
        expert_mapping_tensor=var_76,
        expert_metadata_tensor=ttnn_to_device_66,
        reduction_size=32,
        memory_config=None,
    )
    ttnn.deallocate(v_94, False)
    ttnn.deallocate(ttnn_to_layout_259, False)
    ttnn_to_layout_261 = ttnn.to_layout(v_95, ttnn.Layout.TILE, None, memory_config=None)
    ttnn_typecast_97 = ttnn.typecast(
        ttnn_to_layout_261,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_261, False)
    ttnn_reshape_146 = ttnn.reshape(
        ttnn_to_layout_258,
        [16, 1, 32, 7168],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_258, False)
    ttnn_reshape_147 = ttnn.reshape(
        ttnn_typecast_97,
        [16, 1, 1, 8],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_97, False)
    ttnn_typecast_98 = ttnn.typecast(
        ttnn_reshape_147,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_147, False)
    ttnn_to_layout_262 = ttnn.to_layout(ttnn_typecast_98, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_typecast_98, False)
    ttnn_sparse_matmul_gate_up = ttnn.sparse_matmul(
        input_tensor_a=ttnn_reshape_146,
        input_tensor_b=ce_cache__main["main_const_eval_gate_up"],
        sparsity=ttnn_to_layout_262,
        program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=2,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
            gather_in0=False,
            hop_cores=ttnn.CoreRangeSet([]),
            num_global_cb_receivers=0,
            untilize_out=False,
        ),
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        memory_config=None,
        dtype=None,
    )
    ttnn.deallocate(ttnn_to_layout_262, False)
    ttnn.deallocate(ttnn_reshape_146, False)
    ttnn_reshape_gate_up = ttnn.reshape(
        ttnn_sparse_matmul_gate_up,
        [16, 8, 32, 4096],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_sparse_matmul_gate_up, False)
    ttnn_reshape_148 = ttnn.slice(
        ttnn_reshape_gate_up,
        [0, 0, 0, 0],
        [16, 8, 32, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_149 = ttnn.slice(
        ttnn_reshape_gate_up,
        [0, 0, 0, 2048],
        [16, 8, 32, 4096],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_gate_up, False)
    ttnn_multiply_59 = ttnn.multiply(
        ttnn_reshape_148,
        ttnn_reshape_149,
        dtype=ttnn.DataType.BFLOAT16,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_149, False)
    ttnn.deallocate(ttnn_reshape_148, False)
    # E40 site 3: on-device typecast replaces from_device→typecast→to_device round-trip
    ttnn_to_device_67 = ttnn.typecast(
        v_95,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v_95, False)
    ttnn_sparse_matmul_2 = ttnn.sparse_matmul(
        input_tensor_a=ttnn_multiply_59,
        input_tensor_b=ce_cache__main["main_const_eval_39"],
        sparsity=ttnn_to_device_67,
        program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 7),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=4,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
            gather_in0=False,
            hop_cores=ttnn.CoreRangeSet([]),
            num_global_cb_receivers=0,
            untilize_out=False,
        ),
        is_input_a_sparse=True,
        is_input_b_sparse=False,
        memory_config=None,
        dtype=None,
    )
    ttnn.deallocate(ttnn_to_device_67, False)
    ttnn.deallocate(ttnn_multiply_59, False)
    ttnn_permute_44 = ttnn.permute(
        ttnn_sparse_matmul_2,
        [1, 0, 2, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_sparse_matmul_2, False)
    ttnn_reshape_150 = ttnn.reshape(
        ttnn_permute_44,
        [8, 1, 512, 7168],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_44, False)
    ttnn_to_layout_263 = ttnn.to_layout(ttnn_reshape_150, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_reshape_150, False)
    ttnn_all_to_all_combine_0 = ttnn.all_to_all_combine(
        input_tensor=ttnn_to_layout_263,
        expert_metadata_tensor=ttnn_to_device_66,
        expert_mapping_tensor=var_76,
        cluster_axis=0,
        output_shard_dim=2,
        memory_config=None,
    )
    ttnn.deallocate(ttnn_to_layout_263, False)
    ttnn.deallocate(ttnn_to_device_66, False)
    ttnn_post_combine_tilized = ttnn.experimental.deepseek_moe_post_combine_tilize(
        ttnn_all_to_all_combine_0,
        output_memory_config=ttnn.MemoryConfig(
            buffer_type=ttnn.BufferType.L1,
            nd_shard_spec=ttnn.NdShardSpec(
                shard_shape=ttnn.Shape([32, 3584]),
                grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_all_to_all_combine_0, False)
    ttnn_to_layout_264 = ttnn.to_memory_config(
        ttnn_post_combine_tilized,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_post_combine_tilized, False)
    # E49: fuse the post-MoE reduce_scatter_10 + all_gather_28 (both cluster_axis=1, dim=3)
    # into one ttnn.experimental.all_reduce_async using the rotating semaphore pool.
    _ar0_slot = _ccl_next_slot()
    ttnn_all_reduce_0 = ttnn.experimental.all_reduce_async(
        ttnn_to_layout_264,
        cluster_axis=1,
        mesh_device=utils.DeviceGetter.get_device((4, 8)),
        barrier_semaphores=ce_cache__main["main_const_eval_all_reduce_pool_barrier"][_ar0_slot],
        rs_global_semaphores=ce_cache__main["main_const_eval_all_reduce_pool_rs"][_ar0_slot],
        ag_global_semaphores=ce_cache__main["main_const_eval_all_reduce_pool_ag"][_ar0_slot],
        math_op=ttnn.ReduceType.Sum,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_264, False)
    ttnn_to_layout_265 = ttnn.to_layout(ttnn_all_reduce_0, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_all_reduce_0, False)
    ttnn_mesh_partition_3 = ttnn.mesh_partition(
        input_tensor=ttnn_to_layout_265,
        dim=2,
        cluster_axis=0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_265, False)
    ttnn_mesh_partition_4 = ttnn.mesh_partition(
        input_tensor=ttnn_mesh_partition_3,
        dim=3,
        cluster_axis=1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_mesh_partition_3, False)
    ttnn_to_layout_266 = ttnn.to_layout(ttnn_mesh_partition_4, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_mesh_partition_4, False)
    ttnn_typecast_100 = ttnn.typecast(
        ttnn_to_layout_266,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_266, False)
    ttnn_reshape_151 = ttnn.reshape(
        ttnn_matmul_29,
        [32, 256, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_29, False)
    # E45: matmul_30 still wants FP32 inputs — cast the BF16 matmul_29 result back to FP32
    ttnn_reshape_151_fp32 = ttnn.typecast(
        ttnn_reshape_151,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_151, False)
    ttnn_matmul_30 = ttnn.matmul(
        ttnn_typecast_92,
        ttnn_reshape_151_fp32,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.FLOAT32,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_reshape_151_fp32, False)
    ttnn.deallocate(ttnn_typecast_92, False)
    ttnn_reshape_152 = ttnn.reshape(
        ttnn_matmul_30,
        [32, 8],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_30, False)
    ttnn_permute_45 = ttnn.permute(
        ttnn_reshape_152,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_152, False)
    ttnn_reshape_153 = ttnn.reshape(
        ttnn_permute_45,
        [8, 1, 32, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_45, False)
    ttnn_multiply_60 = ttnn.multiply(
        ttnn_typecast_100,
        ttnn_reshape_153,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_153, False)
    ttnn.deallocate(ttnn_typecast_100, False)
    ttnn_sum_18 = ttnn.sum(
        ttnn_multiply_60,
        [0],
        False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_60, False)
    ttnn_typecast_101 = ttnn.typecast(
        ttnn_sum_18,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_sum_18, False)
    ttnn_matmul_31 = ttnn.matmul(
        ttnn_reshape_120,
        ce_cache__main["main_const_eval_19"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn_reshape_154 = ttnn.reshape(
        ttnn_matmul_31,
        [1, 1, 32, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_31, False)
    ttnn_reduce_scatter_11 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_154,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_154, False)
    ttnn_reshape_155 = ttnn.reshape(
        ttnn_reduce_scatter_11,
        [32, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reduce_scatter_11, False)
    ttnn_all_gather_29 = ttnn.all_gather(
        input_tensor=ttnn_reshape_155,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_155, False)
    ttnn_typecast_102 = ttnn.typecast(
        ttnn_all_gather_29,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_all_gather_29, False)
    ttnn_matmul_32 = ttnn.matmul(
        ttnn_reshape_120,
        ce_cache__main["main_const_eval_29"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_reshape_120, False)
    ttnn_reshape_156 = ttnn.reshape(
        ttnn_matmul_32,
        [1, 1, 32, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_32, False)
    ttnn_reduce_scatter_12 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_156,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_156, False)
    ttnn_reshape_157 = ttnn.reshape(
        ttnn_reduce_scatter_12,
        [32, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reduce_scatter_12, False)
    ttnn_all_gather_30 = ttnn.all_gather(
        input_tensor=ttnn_reshape_157,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_157, False)
    ttnn_typecast_103 = ttnn.typecast(
        ttnn_all_gather_30,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_all_gather_30, False)
    ttnn_typecast_104 = ttnn.multiply(
        ttnn_typecast_102,
        ttnn_typecast_103,
        dtype=ttnn.DataType.BFLOAT16,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_103, False)
    ttnn.deallocate(ttnn_typecast_102, False)
    ttnn_matmul_33 = ttnn.matmul(
        ttnn_typecast_104,
        ce_cache__main["main_const_eval_16"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_typecast_104, False)
    ttnn_typecast_105 = ttnn.add(
        ttnn_matmul_33,
        ttnn_typecast_101,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_33, False)
    ttnn.deallocate(ttnn_typecast_101, False)
    ttnn_reshape_158 = ttnn.reshape(
        ttnn_add_22,
        [1, 32, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_22, False)
    ttnn_add_27 = ttnn.add(
        ttnn_typecast_105,
        ttnn_reshape_158,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_158, False)
    ttnn.deallocate(ttnn_typecast_105, False)
    ttnn_rms_in_4d_4 = ttnn.reshape(
        ttnn_add_27,
        [1, 1, 32, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    # === MoE-capture: dump the live-out tensors produced by the MoE block ===
    ttnn.dump_tensor("moe_io/out_ttnn_add_27.tensorbin", ttnn_add_27)
    ttnn.dump_tensor("moe_io/out_ttnn_rms_in_4d_4.tensorbin", ttnn_rms_in_4d_4)
    # === end MoE-capture outputs ===
    _tracy.signpost("moe_end")
    _tracy.signpost("layer_1_end")
    _tracy.signpost("lm_head_start")
    ttnn_rms_stats_4d_4 = ttnn.rms_norm_pre_all_gather(
        ttnn_rms_in_4d_4,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_rms_sliced_4 = ttnn.slice(
        ttnn_rms_stats_4d_4,
        [0, 0, 0, 0],
        [1, 1, 32, 1],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_rms_stats_4d_4, False)
    ttnn_reshape_159 = ttnn.reshape(
        ttnn_rms_sliced_4,
        [1, 32, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_rms_sliced_4, False)
    ttnn_all_gather_31 = ttnn.all_gather(
        input_tensor=ttnn_reshape_159,
        dim=0,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_159, False)
    ttnn_sum_20 = ttnn.sum(
        ttnn_all_gather_31,
        [0],
        False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_all_gather_31, False)
    ttnn_multiply_62 = ttnn.multiply(
        ttnn_sum_20,
        var_70,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_sum_20, False)
    ttnn_add_28 = ttnn.add(
        ttnn_multiply_62,
        var_67,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_multiply_62, False)
    ttnn_rsqrt_4 = ttnn.rsqrt(
        ttnn_add_28,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_28, False)
    ttnn_reshape_160 = ttnn.reshape(
        ttnn_add_27,
        [32, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_27, False)
    ttnn_multiply_63 = ttnn.multiply(
        ttnn_reshape_160,
        ttnn_rsqrt_4,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_160, False)
    ttnn.deallocate(ttnn_rsqrt_4, False)
    ttnn_multiply_64 = ttnn.multiply(
        ce_cache__main["main_const_eval_31"],
        ttnn_multiply_63,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_multiply_63, False)
    ttnn_matmul_34 = ttnn.matmul(
        ttnn_multiply_64,
        ce_cache__main["main_const_eval_42"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.FLOAT32,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_64, False)
    ttnn_reshape_161 = ttnn.reshape(
        ttnn_matmul_34,
        [1, 1, 32, 129280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_34, False)
    ttnn_reduce_scatter_13 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_161,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_161, False)
    ttnn_reshape_162 = ttnn.reshape(
        ttnn_reduce_scatter_13,
        [32, 16160],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reduce_scatter_13, False)
    ttnn_all_gather_32 = ttnn.all_gather(
        input_tensor=ttnn_reshape_162,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_162, False)
    # Revert E44: argmax requires ROW_MAJOR — restore the Untilize→argmax→Tilize chain
    ttnn_to_layout_267 = ttnn.to_layout(ttnn_all_gather_32, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_all_gather_32, False)
    ttnn_argmax_0 = ttnn.argmax(
        ttnn_to_layout_267,
        1,
        True,
        sub_core_grids=None,
        use_multicore=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_268 = ttnn.to_layout(ttnn_argmax_0, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_argmax_0, False)
    ttnn_typecast_106 = ttnn.typecast(
        ttnn_to_layout_268,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_268, False)
    ttnn_all_gather_33 = ttnn.all_gather(
        input_tensor=ttnn_typecast_106,
        dim=0,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn_add_29 = ttnn.add(
        ttnn_to_layout_104,
        ce_cache__main["main_const_eval_7"],
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_104, False)
    _tracy.signpost("lm_head_end")
    _tracy.signpost("decode_1_end")
    return [
        model_transformer_layers_0_attn_indexer_k_cache,
        model_transformer_layers_1_attn_indexer_k_cache,
        L__past_key_values___layers_0_compressed_kv,
        L__past_key_values___layers_0_k_pe,
        L__past_key_values___layers_1_compressed_kv,
        L__past_key_values___layers_1_k_pe,
        ttnn_typecast_106,
        ttnn_all_gather_33,
        ttnn_add_29,
        ttnn_to_layout_267,
    ]


def consteval__main(ce_cache, weights):
    if not ce_cache:
        # E49: rotating semaphore pool for all_reduce_async (deepseek_v3 pattern)
        main_const_eval_all_reduce_semaphores_0 = main_const_eval_all_reduce_semaphores()
        ce_cache["main_const_eval_all_reduce_pool_barrier"] = main_const_eval_all_reduce_semaphores_0[0]
        ce_cache["main_const_eval_all_reduce_pool_rs"] = main_const_eval_all_reduce_semaphores_0[1]
        ce_cache["main_const_eval_all_reduce_pool_ag"] = main_const_eval_all_reduce_semaphores_0[2]
        # E47: rotary_embedding_llama trans_mat (32x32 BF16 TILE half-rotate)
        main_const_eval_rope_trans_mat_0 = main_const_eval_rope_trans_mat()
        ce_cache["main_const_eval_rope_trans_mat"] = main_const_eval_rope_trans_mat_0[0]
        # E47 iter2: doubled cos/sin tables for rotary_embedding_llama
        main_const_eval_rope_cos_sin_doubled_0 = main_const_eval_rope_cos_sin_doubled(
            [weights["model.transformer.freqs_cis"]]
        )
        ce_cache["main_const_eval_rope_cos_doubled"] = main_const_eval_rope_cos_sin_doubled_0[0]
        ce_cache["main_const_eval_rope_sin_doubled"] = main_const_eval_rope_cos_sin_doubled_0[1]
        main_const_eval_0_0 = main_const_eval_0()
        ce_cache["main_const_eval_0"] = main_const_eval_0_0[0]
        main_const_eval_1_0 = main_const_eval_1()
        ce_cache["main_const_eval_1"] = [
            main_const_eval_1_0[0],
            main_const_eval_1_0[1],
            main_const_eval_1_0[2],
            main_const_eval_1_0[3],
            main_const_eval_1_0[4],
            main_const_eval_1_0[5],
            main_const_eval_1_0[6],
            main_const_eval_1_0[7],
            main_const_eval_1_0[8],
            main_const_eval_1_0[9],
            main_const_eval_1_0[10],
            main_const_eval_1_0[11],
            main_const_eval_1_0[12],
            main_const_eval_1_0[13],
            main_const_eval_1_0[14],
            main_const_eval_1_0[15],
            main_const_eval_1_0[16],
            main_const_eval_1_0[17],
            main_const_eval_1_0[18],
            main_const_eval_1_0[19],
            main_const_eval_1_0[20],
            main_const_eval_1_0[21],
            main_const_eval_1_0[22],
            main_const_eval_1_0[23],
            main_const_eval_1_0[24],
            main_const_eval_1_0[25],
            main_const_eval_1_0[26],
            main_const_eval_1_0[27],
            main_const_eval_1_0[28],
            main_const_eval_1_0[29],
            main_const_eval_1_0[30],
            main_const_eval_1_0[31],
            main_const_eval_1_0[32],
            main_const_eval_1_0[33],
            main_const_eval_1_0[34],
            main_const_eval_1_0[35],
            main_const_eval_1_0[36],
            main_const_eval_1_0[37],
            main_const_eval_1_0[38],
            main_const_eval_1_0[39],
            main_const_eval_1_0[40],
            main_const_eval_1_0[41],
            main_const_eval_1_0[42],
            main_const_eval_1_0[43],
            main_const_eval_1_0[44],
            main_const_eval_1_0[45],
            main_const_eval_1_0[46],
            main_const_eval_1_0[47],
            main_const_eval_1_0[48],
            main_const_eval_1_0[49],
            main_const_eval_1_0[50],
            main_const_eval_1_0[51],
            main_const_eval_1_0[52],
            main_const_eval_1_0[53],
            main_const_eval_1_0[54],
            main_const_eval_1_0[55],
            main_const_eval_1_0[56],
            main_const_eval_1_0[57],
            main_const_eval_1_0[58],
            main_const_eval_1_0[59],
            main_const_eval_1_0[60],
            main_const_eval_1_0[61],
            main_const_eval_1_0[62],
            main_const_eval_1_0[63],
            main_const_eval_1_0[64],
        ]
        main_const_eval_2_0 = main_const_eval_2([weights["model.transformer.layers.0.ffn.w3.weight"]])
        ce_cache["main_const_eval_2"] = main_const_eval_2_0[0]
        main_const_eval_3_0 = main_const_eval_3([weights["model.transformer.layers.1.ffn.mlp.experts.gate_proj"]])
        ce_cache["main_const_eval_3"] = main_const_eval_3_0[0]
        main_const_eval_4_0 = main_const_eval_4()
        ce_cache["main_const_eval_4"] = main_const_eval_4_0[0]
        main_const_eval_5_0 = main_const_eval_5([weights["model.transformer.layers.1.attn.indexer.haddamard"]])
        ce_cache["main_const_eval_5"] = [main_const_eval_5_0[0], main_const_eval_5_0[1]]
        main_const_eval_6_0 = main_const_eval_6()
        ce_cache["main_const_eval_6"] = main_const_eval_6_0[0]
        main_const_eval_7_0 = main_const_eval_7()
        ce_cache["main_const_eval_7"] = main_const_eval_7_0[0]
        main_const_eval_8_0 = main_const_eval_8()
        ce_cache["main_const_eval_8"] = main_const_eval_8_0[0]
        main_const_eval_9_0 = main_const_eval_9([weights["model.transformer.layers.1.ffn.mlp.router.gate.bias"]])
        ce_cache["main_const_eval_9"] = main_const_eval_9_0[0]
        main_const_eval_10_0 = main_const_eval_10()
        ce_cache["main_const_eval_10"] = main_const_eval_10_0[0]
        main_const_eval_11_0 = main_const_eval_11()
        ce_cache["main_const_eval_11"] = main_const_eval_11_0[0]
        main_const_eval_12_0 = main_const_eval_12()
        ce_cache["main_const_eval_12"] = main_const_eval_12_0[0]
        main_const_eval_13_0 = main_const_eval_13([weights["model.transformer.freqs_cis"]])
        ce_cache["main_const_eval_13"] = main_const_eval_13_0[0]
        main_const_eval_14_0 = main_const_eval_14([weights["model.transformer.layers.0.attn.wq_b.weight"]])
        ce_cache["main_const_eval_14"] = main_const_eval_14_0[0]
        main_const_eval_15_0 = main_const_eval_15([weights["model.transformer.layers.1.attn.wo.weight"]])
        ce_cache["main_const_eval_15"] = main_const_eval_15_0[0]
        main_const_eval_16_0 = main_const_eval_16([weights["model.transformer.layers.1.ffn.shared_experts.w2.weight"]])
        ce_cache["main_const_eval_16"] = main_const_eval_16_0[0]
        main_const_eval_17_0 = main_const_eval_17()
        ce_cache["main_const_eval_17"] = main_const_eval_17_0[0]
        main_const_eval_18_0 = main_const_eval_18()
        ce_cache["main_const_eval_18"] = main_const_eval_18_0[0]
        main_const_eval_19_0 = main_const_eval_19([weights["model.transformer.layers.1.ffn.shared_experts.w1.weight"]])
        ce_cache["main_const_eval_19"] = main_const_eval_19_0[0]
        main_const_eval_20_0 = main_const_eval_20([weights["model.transformer.layers.0.attn.indexer.haddamard"]])
        ce_cache["main_const_eval_20"] = [
            main_const_eval_20_0[0],
            main_const_eval_20_0[1],
        ]
        main_const_eval_21_0 = main_const_eval_21([weights["model.transformer.layers.0.attn.wo.weight"]])
        ce_cache["main_const_eval_21"] = main_const_eval_21_0[0]
        main_const_eval_22_0 = main_const_eval_22([weights["model.transformer.layers.0.attn.wkv_b.weight"]])
        ce_cache["main_const_eval_22"] = [
            main_const_eval_22_0[0],
            main_const_eval_22_0[1],
        ]
        main_const_eval_23_0 = main_const_eval_23()
        ce_cache["main_const_eval_23"] = main_const_eval_23_0[0]
        main_const_eval_24_0 = main_const_eval_24([weights["model.transformer.layers.1.ffn_norm.weight"]])
        ce_cache["main_const_eval_24"] = main_const_eval_24_0[0]
        main_const_eval_25_0 = main_const_eval_25([weights["model.transformer.layers.1.ffn.mlp.router.gate.weight"]])
        ce_cache["main_const_eval_25"] = main_const_eval_25_0[0]
        main_const_eval_26_0 = main_const_eval_26()
        ce_cache["main_const_eval_26"] = main_const_eval_26_0[0]
        main_const_eval_27_0 = main_const_eval_27([weights["model.transformer.layers.0.attn.indexer.wq_b.weight"]])
        ce_cache["main_const_eval_27"] = main_const_eval_27_0[0]
        main_const_eval_28_0 = main_const_eval_28(
            [
                weights["model.transformer.layers.1.attn.indexer.wk.weight"],
                weights["model.transformer.layers.1.attn.wkv_a.weight"],
                weights["model.transformer.layers.1.attn.indexer.weights_proj.weight"],
                weights["model.transformer.layers.1.attn.wq_a.weight"],
            ]
        )
        ce_cache["main_const_eval_28"] = main_const_eval_28_0[0]
        main_const_eval_29_0 = main_const_eval_29([weights["model.transformer.layers.1.ffn.shared_experts.w3.weight"]])
        ce_cache["main_const_eval_29"] = main_const_eval_29_0[0]
        main_const_eval_30_0 = main_const_eval_30([weights["model.transformer.embed.weight"]])
        ce_cache["main_const_eval_30"] = main_const_eval_30_0[0]
        main_const_eval_31_0 = main_const_eval_31([weights["model.transformer.norm.weight"]])
        ce_cache["main_const_eval_31"] = main_const_eval_31_0[0]
        main_const_eval_32_0 = main_const_eval_32(
            [
                weights["model.transformer.layers.0.attn.indexer.wk.weight"],
                weights["model.transformer.layers.0.attn.wkv_a.weight"],
                weights["model.transformer.layers.0.attn.indexer.weights_proj.weight"],
                weights["model.transformer.layers.0.attn.wq_a.weight"],
            ]
        )
        ce_cache["main_const_eval_32"] = main_const_eval_32_0[0]
        main_const_eval_33_0 = main_const_eval_33()
        ce_cache["main_const_eval_33"] = main_const_eval_33_0[0]
        main_const_eval_34_0 = main_const_eval_34([weights["model.transformer.layers.1.attn_norm.weight"]])
        ce_cache["main_const_eval_34"] = main_const_eval_34_0[0]
        main_const_eval_35_0 = main_const_eval_35()
        ce_cache["main_const_eval_35"] = main_const_eval_35_0[0]
        main_const_eval_36_0 = main_const_eval_36()
        ce_cache["main_const_eval_36"] = main_const_eval_36_0[0]
        main_const_eval_37_0 = main_const_eval_37([weights["model.transformer.layers.1.ffn.mlp.expert_mapping"]])
        ce_cache["main_const_eval_37"] = main_const_eval_37_0[0]
        main_const_eval_38_0 = main_const_eval_38()
        ce_cache["main_const_eval_38"] = main_const_eval_38_0[0]
        main_const_eval_39_0 = main_const_eval_39([weights["model.transformer.layers.1.ffn.mlp.experts.down_proj"]])
        ce_cache["main_const_eval_39"] = main_const_eval_39_0[0]
        main_const_eval_40_0 = main_const_eval_40([weights["model.transformer.layers.0.attn_norm.weight"]])
        ce_cache["main_const_eval_40"] = main_const_eval_40_0[0]
        main_const_eval_41_0 = main_const_eval_41([weights["model.transformer.layers.0.ffn_norm.weight"]])
        ce_cache["main_const_eval_41"] = main_const_eval_41_0[0]
        main_const_eval_42_0 = main_const_eval_42([weights["model.transformer.head.weight"]])
        ce_cache["main_const_eval_42"] = main_const_eval_42_0[0]
        main_const_eval_43_0 = main_const_eval_43([weights["model.transformer.layers.0.ffn.w2.weight"]])
        ce_cache["main_const_eval_43"] = main_const_eval_43_0[0]
        main_const_eval_44_0 = main_const_eval_44()
        ce_cache["main_const_eval_44"] = main_const_eval_44_0[0]
        main_const_eval_45_0 = main_const_eval_45([weights["model.transformer.layers.0.ffn.w1.weight"]])
        ce_cache["main_const_eval_45"] = main_const_eval_45_0[0]
        main_const_eval_46_0 = main_const_eval_46([weights["model.transformer.layers.1.attn.indexer.wq_b.weight"]])
        ce_cache["main_const_eval_46"] = main_const_eval_46_0[0]
        main_const_eval_47_0 = main_const_eval_47([weights["model.transformer.layers.1.ffn.mlp.experts.up_proj"]])
        ce_cache["main_const_eval_47"] = main_const_eval_47_0[0]
        main_const_eval_gate_up_0 = main_const_eval_gate_up(
            [weights["model.transformer.layers.1.ffn.mlp.experts.gate_proj"]],
            [weights["model.transformer.layers.1.ffn.mlp.experts.up_proj"]],
        )
        ce_cache["main_const_eval_gate_up"] = main_const_eval_gate_up_0[0]
        main_const_eval_48_0 = main_const_eval_48([weights["model.transformer.layers.1.attn.wkv_b.weight"]])
        ce_cache["main_const_eval_48"] = [
            main_const_eval_48_0[0],
            main_const_eval_48_0[1],
        ]
        main_const_eval_49_0 = main_const_eval_49()
        ce_cache["main_const_eval_49"] = main_const_eval_49_0[0]
        main_const_eval_50_0 = main_const_eval_50()
        ce_cache["main_const_eval_50"] = main_const_eval_50_0[0]
        main_const_eval_51_0 = main_const_eval_51()
        ce_cache["main_const_eval_51"] = main_const_eval_51_0[0]
        main_const_eval_52_0 = main_const_eval_52()
        ce_cache["main_const_eval_52"] = main_const_eval_52_0[0]
        main_const_eval_53_0 = main_const_eval_53()
        ce_cache["main_const_eval_53"] = [
            main_const_eval_53_0[0],
            main_const_eval_53_0[1],
        ]
        main_const_eval_54_0 = main_const_eval_54([weights["model.transformer.layers.1.attn.wq_b.weight"]])
        ce_cache["main_const_eval_54"] = main_const_eval_54_0[0]
    return ce_cache


def load_activations_for__main(tensors_dir="./tensors"):
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((4, 8))
    utils_load_tensor_0 = utils.load_tensor(
        f"{tensors_dir}/arg4.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.INT32,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_1 = utils.load_tensor(
        f"{tensors_dir}/arg7.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_2 = utils.load_tensor(
        f"{tensors_dir}/arg9.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_3 = utils.load_tensor(
        f"{tensors_dir}/arg18.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_4 = utils.load_tensor(
        f"{tensors_dir}/arg23.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_5 = utils.load_tensor(
        f"{tensors_dir}/arg30.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_6 = utils.load_tensor(
        f"{tensors_dir}/arg33.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_7 = utils.load_tensor(
        f"{tensors_dir}/arg34.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_8 = utils.load_tensor(
        f"{tensors_dir}/arg49.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    utils_load_tensor_9 = utils.load_tensor(
        f"{tensors_dir}/arg50.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_1,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return [
        utils_load_tensor_0,
        utils_load_tensor_1,
        utils_load_tensor_2,
        utils_load_tensor_3,
        utils_load_tensor_4,
        utils_load_tensor_5,
        utils_load_tensor_6,
        utils_load_tensor_7,
        utils_load_tensor_8,
        utils_load_tensor_9,
    ]


_main_weights = {}


def load_weights_for__main(tensors_dir="./tensors"):
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device((4, 8))
    global _main_weights
    utils_load_tensor_10 = utils.load_tensor(
        f"{tensors_dir}/arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.0.attn.indexer.haddamard"] = utils_load_tensor_10
    utils_load_tensor_11 = utils.load_tensor(
        f"{tensors_dir}/arg1.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    _main_weights["model.transformer.layers.0.attn.indexer.k_norm.bias"] = utils_load_tensor_11
    utils_load_tensor_12 = utils.load_tensor(
        f"{tensors_dir}/arg2.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    _main_weights["model.transformer.layers.0.attn.indexer.k_norm.weight"] = utils_load_tensor_12
    utils_load_tensor_13 = utils.load_tensor(
        f"{tensors_dir}/arg3.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.0.attn.indexer.wk.weight"] = utils_load_tensor_13
    utils_load_tensor_14 = utils.load_tensor(
        f"{tensors_dir}/arg5.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.embed.weight"] = utils_load_tensor_14
    utils_load_tensor_15 = utils.load_tensor(
        f"{tensors_dir}/arg6.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.0.attn_norm.weight"] = utils_load_tensor_15
    utils_load_tensor_16 = utils.load_tensor(
        f"{tensors_dir}/arg8.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.freqs_cis"] = utils_load_tensor_16
    utils_load_tensor_17 = utils.load_tensor(
        f"{tensors_dir}/arg10.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.1.attn.indexer.haddamard"] = utils_load_tensor_17
    utils_load_tensor_18 = utils.load_tensor(
        f"{tensors_dir}/arg11.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    _main_weights["model.transformer.layers.1.attn.indexer.k_norm.bias"] = utils_load_tensor_18
    utils_load_tensor_19 = utils.load_tensor(
        f"{tensors_dir}/arg12.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    _main_weights["model.transformer.layers.1.attn.indexer.k_norm.weight"] = utils_load_tensor_19
    utils_load_tensor_20 = utils.load_tensor(
        f"{tensors_dir}/arg13.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.1.attn.indexer.wk.weight"] = utils_load_tensor_20
    utils_load_tensor_21 = utils.load_tensor(
        f"{tensors_dir}/arg14.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.0.attn.wo.weight"] = utils_load_tensor_21
    utils_load_tensor_22 = utils.load_tensor(
        f"{tensors_dir}/arg15.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.0.attn.wkv_b.weight"] = utils_load_tensor_22
    utils_load_tensor_23 = utils.load_tensor(
        f"{tensors_dir}/arg16.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.0.attn.wkv_a.weight"] = utils_load_tensor_23
    utils_load_tensor_24 = utils.load_tensor(
        f"{tensors_dir}/arg17.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    _main_weights["model.transformer.layers.0.attn.kv_norm.weight"] = utils_load_tensor_24
    utils_load_tensor_25 = utils.load_tensor(
        f"{tensors_dir}/arg19.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.0.attn.indexer.weights_proj.weight"] = utils_load_tensor_25
    utils_load_tensor_26 = utils.load_tensor(
        f"{tensors_dir}/arg20.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.0.attn.indexer.wq_b.weight"] = utils_load_tensor_26
    utils_load_tensor_27 = utils.load_tensor(
        f"{tensors_dir}/arg21.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.0.attn.wq_a.weight"] = utils_load_tensor_27
    utils_load_tensor_28 = utils.load_tensor(
        f"{tensors_dir}/arg22.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    _main_weights["model.transformer.layers.0.attn.q_norm.weight"] = utils_load_tensor_28
    utils_load_tensor_29 = utils.load_tensor(
        f"{tensors_dir}/arg24.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.0.attn.wq_b.weight"] = utils_load_tensor_29
    utils_load_tensor_30 = utils.load_tensor(
        f"{tensors_dir}/arg25.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.0.ffn.w2.weight"] = utils_load_tensor_30
    utils_load_tensor_31 = utils.load_tensor(
        f"{tensors_dir}/arg26.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.0.ffn.w3.weight"] = utils_load_tensor_31
    utils_load_tensor_32 = utils.load_tensor(
        f"{tensors_dir}/arg27.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.0.ffn_norm.weight"] = utils_load_tensor_32
    utils_load_tensor_33 = utils.load_tensor(
        f"{tensors_dir}/arg28.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.0.ffn.w1.weight"] = utils_load_tensor_33
    utils_load_tensor_34 = utils.load_tensor(
        f"{tensors_dir}/arg29.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.1.attn_norm.weight"] = utils_load_tensor_34
    utils_load_tensor_35 = utils.load_tensor(
        f"{tensors_dir}/arg31.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.1.attn.wkv_a.weight"] = utils_load_tensor_35
    utils_load_tensor_36 = utils.load_tensor(
        f"{tensors_dir}/arg32.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    _main_weights["model.transformer.layers.1.attn.kv_norm.weight"] = utils_load_tensor_36
    utils_load_tensor_37 = utils.load_tensor(
        f"{tensors_dir}/arg35.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.head.weight"] = utils_load_tensor_37
    utils_load_tensor_38 = utils.load_tensor(
        f"{tensors_dir}/arg36.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.1.attn.wo.weight"] = utils_load_tensor_38
    utils_load_tensor_39 = utils.load_tensor(
        f"{tensors_dir}/arg37.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.1.attn.wkv_b.weight"] = utils_load_tensor_39
    utils_load_tensor_40 = utils.load_tensor(
        f"{tensors_dir}/arg38.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.1.attn.indexer.weights_proj.weight"] = utils_load_tensor_40
    utils_load_tensor_41 = utils.load_tensor(
        f"{tensors_dir}/arg39.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.1.attn.indexer.wq_b.weight"] = utils_load_tensor_41
    utils_load_tensor_42 = utils.load_tensor(
        f"{tensors_dir}/arg40.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.1.attn.wq_a.weight"] = utils_load_tensor_42
    utils_load_tensor_43 = utils.load_tensor(
        f"{tensors_dir}/arg41.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    _main_weights["model.transformer.layers.1.attn.q_norm.weight"] = utils_load_tensor_43
    utils_load_tensor_44 = utils.load_tensor(
        f"{tensors_dir}/arg42.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.1.attn.wq_b.weight"] = utils_load_tensor_44
    utils_load_tensor_45 = utils.load_tensor(
        f"{tensors_dir}/arg43.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.1.ffn.shared_experts.w2.weight"] = utils_load_tensor_45
    utils_load_tensor_46 = utils.load_tensor(
        f"{tensors_dir}/arg44.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.1.ffn.shared_experts.w3.weight"] = utils_load_tensor_46
    utils_load_tensor_47 = utils.load_tensor(
        f"{tensors_dir}/arg45.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.1.ffn_norm.weight"] = utils_load_tensor_47
    utils_load_tensor_48 = utils.load_tensor(
        f"{tensors_dir}/arg46.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.1.ffn.shared_experts.w1.weight"] = utils_load_tensor_48
    utils_load_tensor_49 = utils.load_tensor(
        f"{tensors_dir}/arg47.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.1.ffn.mlp.router.gate.bias"] = utils_load_tensor_49
    utils_load_tensor_50 = utils.load_tensor(
        f"{tensors_dir}/arg48.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.1.ffn.mlp.router.gate.weight"] = utils_load_tensor_50
    utils_load_tensor_51 = utils.load_tensor(
        f"{tensors_dir}/arg51.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.INT32,
        None,
        None,
    )
    _main_weights["model.transformer.layers.1.ffn.mlp.expert_mapping"] = utils_load_tensor_51
    utils_load_tensor_52 = utils.load_tensor(
        f"{tensors_dir}/arg52.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.1.ffn.mlp.experts.down_proj"] = utils_load_tensor_52
    utils_load_tensor_53 = utils.load_tensor(
        f"{tensors_dir}/arg53.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.1.ffn.mlp.experts.up_proj"] = utils_load_tensor_53
    utils_load_tensor_54 = utils.load_tensor(
        f"{tensors_dir}/arg54.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.layers.1.ffn.mlp.experts.gate_proj"] = utils_load_tensor_54
    utils_load_tensor_55 = utils.load_tensor(
        f"{tensors_dir}/arg55.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        None,
        None,
    )
    _main_weights["model.transformer.norm.weight"] = utils_load_tensor_55
    return _main_weights


def main():
    import os
    import tracy as _tracy

    use_trace = os.environ.get("TTXLA_USE_TRACE", "") == "1"
    # TTXLA_TENSORS_DIR overrides the default activations dir. Useful for
    # measuring decode-step-K perf via tracy: point it at tensors_step{K}/.
    tensors_dir = os.environ.get("TTXLA_TENSORS_DIR", "./tensors")
    load_weights_for__main_0 = load_weights_for__main()
    if use_trace:
        # E48: program-trace path — capture _main into a device-side trace
        # so the measured run replays without host launch overhead between ops.
        device = utils.DeviceGetter.get_device((4, 8))
        # 1) Warmup — cold compile + populate ce_cache. _main deallocates args_0/args_1.
        warm_act = load_activations_for__main(tensors_dir=tensors_dir)
        _main(warm_act, load_weights_for__main_0)
        ttnn.synchronize_device(device)
        # 2) Re-load activations (warmup freed args_0/args_1).
        trace_act = load_activations_for__main(tensors_dir=tensors_dir)
        # 3) Capture trace.
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        _main(trace_act, load_weights_for__main_0)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        # 4) Execute trace — this is the tracy-measured run.
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        ttnn.release_trace(device, tid)
    else:
        load_activations_for__main_0 = load_activations_for__main(tensors_dir=tensors_dir)
        _main_0 = _main(load_activations_for__main_0, load_weights_for__main_0)
    if utils.DeviceGetter._instance is not None:
        ttnn.close_mesh_device(utils.DeviceGetter._instance)
        utils.DeviceGetter._instance = None
    return 0


if __name__ == "__main__":
    main()
