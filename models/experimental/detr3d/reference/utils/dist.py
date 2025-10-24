# Copyright (c) Facebook, Inc. and its affiliates.
import pickle

import torch
import torch.distributed as dist


def is_distributed():
    if not dist.is_available() or not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_distributed():
        return 0
    return dist.get_rank()


def is_primary():
    return get_rank() == 0


def get_world_size():
    if not is_distributed():
        return 1
    return dist.get_world_size()


def barrier():
    if not is_distributed():
        return
    torch.distributed.barrier()


def setup_print_for_distributed(is_primary):
    """
    This function disables printing when not in primary process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_primary or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed(gpu_id, global_rank, world_size, dist_url, dist_backend):
    torch.cuda.set_device(gpu_id)
    print(
        f"| distributed init (rank {global_rank}) (world {world_size}): {dist_url}",
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=dist_backend,
        init_method=dist_url,
        world_size=world_size,
        rank=global_rank,
    )
    torch.distributed.barrier()
    setup_print_for_distributed(is_primary())


def all_reduce_sum(tensor):
    if not is_distributed():
        return tensor
    dim_squeeze = False
    if tensor.ndim == 0:
        tensor = tensor[None, ...]
        dim_squeeze = True
    torch.distributed.all_reduce(tensor)
    if dim_squeeze:
        tensor = tensor.squeeze(0)
    return tensor


def all_reduce_average(tensor):
    val = all_reduce_sum(tensor)
    return val / get_world_size()


# Function from DETR - https://github.com/facebookresearch/detr/blob/master/util/misc.py
def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


# Function from https://github.com/facebookresearch/detr/blob/master/util/misc.py
def all_gather_pickle(data, device):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device)

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device=device)
    size_list = [torch.tensor([0], device=device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device=device))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device=device)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def all_gather_dict(data):
    """
    Run all_gather on data which is a dictionary of Tensors
    """
    assert isinstance(data, dict)

    gathered_dict = {}
    for item_key in data:
        if isinstance(data[item_key], torch.Tensor):
            if is_distributed():
                data[item_key] = data[item_key].contiguous()
                tensor_list = [torch.empty_like(data[item_key]) for _ in range(get_world_size())]
                dist.all_gather(tensor_list, data[item_key])
                gathered_tensor = torch.cat(tensor_list, dim=0)
            else:
                gathered_tensor = data[item_key]
            gathered_dict[item_key] = gathered_tensor
    return gathered_dict
