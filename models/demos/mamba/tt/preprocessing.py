import ttnn


def split_sequence_length(x, chunk_size: int = 32):
    """
    Generator function to yield chunks of a tensor of shape (1, 1, L, E) into (1, 1, 32, E).

    Parameters:
    tensor (torch.Tensor): The input tensor of shape (1, 1, L, E).
    chunk_size (int): The size of each chunk along the third dimension. Default is 32.

    Yields:
    torch.Tensor: Chunks of the input tensor of shape (1, 1, chunk_size, E).
    """

    assert chunk_size % 32 == 0, "Chunk size must be multiple of 32"
    assert x.shape[2] % 32 == 0, "Sequence length size must be multiple of 32"

    _, _, L, E = x.shape

    for i in range(0, L, chunk_size):
        slice_start = (0, 0, i, 0)
        slice_end = (0, 0, i + chunk_size - 1, E - 1)
        yield ttnn.slice(x, ttnn.Shape(slice_start), ttnn.Shape(slice_end))
