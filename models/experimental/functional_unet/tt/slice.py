import ttnn


def chunk_batch_dim(x, memory_config, slice_length=2):
    assert len(x.shape) == 4, f"Expected input tensor to be rank 4 (was len(x.shape)"
    B, C, H, W = x.shape
    for batch_idx in range(0, B, slice_length):
        start, end = batch_idx, batch_idx + slice_length
        yield ttnn.slice(x, [start, 0, 0, 0], [end, C, H, W], memory_config=memory_config)


def run_by_slicing_inputs_from_dram(model, input_tensor, device, slice_memory_config, slice_length=2):
    assert len(input_tensor.shape) == 4, f"Expected input tensor to be rank 4 (was len(x.shape)"
    input_tensor = ttnn.to_device(input_tensor, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output = []
    for slice in chunk_batch_dim(input_tensor, memory_config=slice_memory_config, slice_length=slice_length):
        slice = model(slice, move_input_tensor_to_device=False)
        output.append(ttnn.to_memory_config(slice, ttnn.DRAM_MEMORY_CONFIG))
    return output
