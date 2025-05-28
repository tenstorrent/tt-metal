import torch


def clip(n: int, lower: int, upper: int) -> int:
    n = n if n >= lower else lower
    return n if n < upper else upper


def multi_scale_kernel_attn_sampling(bottom_data, height, width, nheads, channels, h, w, m, c):
    w_stride = nheads * channels
    h_stride = width * w_stride

    base_ptr = m * channels + c

    h_ptr_offset = int(h_stride * h)
    w_ptr_offset = int(w_stride * w)

    offset = base_ptr + h_ptr_offset + w_ptr_offset
    if offset >= 0 and offset < len(bottom_data):
        val = bottom_data[base_ptr + h_ptr_offset + w_ptr_offset]
    else:
        val = 0

    return val


def multiscale_kernel_attn_forward_gpu_kernel(
    n,
    data_value,
    data_spatial_shapes,
    data_level_start_index,
    data_sampling_loc,
    data_attn_weight,
    batch_size,
    spatial_size,
    num_heads,
    channels,
    num_levels,
    num_query,
    num_point,
    data_col,
):
    for i in range(n):
        print(i)

        _temp = i
        c_col = _temp % channels
        _temp = _temp // channels
        sampling_index = _temp
        m_col = _temp % num_heads
        _temp = _temp // num_heads
        q_col = _temp % num_query
        _temp = _temp // num_query
        b_col = _temp

        data_col_ptr = data_col[i:]  # TODO Handle
        data_weight_ptr = sampling_index * num_levels * num_point
        # print("data_weight_ptr ", data_weight_ptr)
        data_loc_w_ptr = data_weight_ptr << 1
        # print("data_loc_w_ptr ", data_loc_w_ptr)

        qid_stride = num_heads * channels
        data_value_ptr_init_offset = b_col * spatial_size * qid_stride
        col = 0

        for l_col in range(num_levels):
            level_start_id = data_level_start_index[l_col]
            spatial_h_ptr = l_col << 1

            # print("spatial_h_ptr ", spatial_h_ptr)
            spatial_h = data_spatial_shapes[spatial_h_ptr]
            spatial_w = data_spatial_shapes[spatial_h_ptr + 1]

            offset = data_value_ptr_init_offset + level_start_id * qid_stride
            data_value_ptr = data_value[offset:]
            for p_col in range(num_point):
                loc_w = int(data_sampling_loc[data_loc_w_ptr])
                loc_h = int(data_sampling_loc[data_loc_w_ptr + 1])
                weight = data_attn_weight[data_weight_ptr]
                # print("spatial_h  ", spatial_h)
                # print("spatial_w ", spatial_w)
                # print("loc_h ", loc_h)
                # print("loc_w ", loc_w)

                loc_h_ = clip(loc_h, 0, spatial_h - 1)
                loc_w_ = clip(loc_w, 0, spatial_w - 1)
                # print("loc_h_ ", loc_h_)
                # print("loc_w_ ", loc_w_)
                # print("data_value_ptr shape ", data_value.shape)
                col += (
                    multi_scale_kernel_attn_sampling(
                        data_value_ptr, spatial_h, spatial_w, num_heads, channels, loc_h_, loc_w_, m_col, c_col
                    )
                    * weight
                )
                data_weight_ptr += 1
                data_loc_w_ptr += 2

        data_col_ptr[0] = col


def GeometricKernelAttentionFunc(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step):
    assert (value.is_contiguous(), "value tensor has to be contiguous")
    assert (spatial_shapes.is_contiguous(), "spatial_shapes tensor has to be contiguous")
    assert (level_start_index.is_contiguous(), "level_start_index tensor has to be contiguous")
    assert (sampling_loc.is_contiguous(), "sampling_loc tensor has to be contiguous")
    assert (attn_weight.is_contiguous(), "attn_weight tensor has to be contiguous")

    # assert(value.type().is_cuda(), "value must be a CUDA tensor")
    # assert(spatial_shapes.type().is_cuda(), "spatial_shapes must be a CUDA tensor")
    # assert(level_start_index.type().is_cuda(), "level_start_index must be a CUDA tensor")
    # assert(sampling_loc.type().is_cuda(), "sampling_loc must be a CUDA tensor")
    # assert(attn_weight.type().is_cuda(), "attn_weight must be a CUDA tensor")

    batch = value.shape[0]
    spatial_size = value.shape[1]
    num_heads = value.shape[2]
    channels = value.shape[3]

    num_levels = spatial_shapes.shape[0]
    num_query = sampling_loc.shape[1]
    num_point = sampling_loc.shape[4]

    im2col_step_ = min(batch, im2col_step)

    assert (batch % im2col_step_ == 0, f"Batch size {batch} must be divisible by im2col_step {im2col_step_}")

    output = torch.zeros((batch, num_query, num_heads, channels), dtype=value.dtype, device=value.device)
    batch_n = im2col_step_
    output_n = output.view(batch // im2col_step_, batch_n, num_query, num_heads, channels)  # passing reference

    per_value_size = spatial_size * num_heads * channels
    per_sample_loc_size = num_query * num_heads * num_levels * num_point * 2
    per_attn_weight_size = num_query * num_heads * num_levels * num_point

    value_flatten = value.flatten()
    sampling_loc_flatten = sampling_loc.flatten()
    attn_weight_flatten = attn_weight.flatten()
    spatial_shapes_flatten = spatial_shapes.flatten()
    level_start_index_flatten = level_start_index.flatten()

    num_kernel = batch_n * num_query * num_heads * channels
    for n in range(batch // im2col_step_):
        columns = output_n.select(0, n)

        start_idx = n * im2col_step_ * per_value_size
        value_ = value_flatten[start_idx:]

        start_idx = n * im2col_step_ * per_sample_loc_size
        sampling_loc_ = sampling_loc_flatten[start_idx:]

        start_idx = n * im2col_step_ * per_attn_weight_size
        attn_weight_ = attn_weight_flatten[start_idx:]

        multiscale_kernel_attn_forward_gpu_kernel(
            num_kernel,
            value_,
            spatial_shapes_flatten,
            level_start_index_flatten,
            sampling_loc_,
            attn_weight_,
            batch_n,
            spatial_size,
            num_heads,
            channels,
            num_levels,
            num_query,
            num_point,
            columns.flatten(),
        )
    # output = torch.zeros(batch, num_query, num_heads*channels)
    # print(output.shape)
    output = output.view(batch, num_query, num_heads * channels)
    return output
