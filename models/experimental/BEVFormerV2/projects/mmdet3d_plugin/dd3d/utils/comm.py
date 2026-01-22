# Copyright 2021 Toyota Research Institute.  All rights reserved.
import logging
from functools import wraps

import torch.distributed as dist

from models.experimental.BEVFormerV2.projects.mmdet3d_plugin.dependency import (
    get_world_size,
    get_rank,
    is_main_process,
    synchronize,
    gather,
    all_gather,
    shared_random_seed,
    reduce_dict,
    create_local_process_group,
    get_local_rank,
    get_local_size,
    get_local_process_group,
)


# Create a comm module-like object for compatibility
class CommModule:
    get_world_size = staticmethod(get_world_size)
    get_rank = staticmethod(get_rank)
    is_main_process = staticmethod(is_main_process)
    synchronize = staticmethod(synchronize)
    gather = staticmethod(gather)
    all_gather = staticmethod(all_gather)
    shared_random_seed = staticmethod(shared_random_seed)
    reduce_dict = staticmethod(reduce_dict)
    create_local_process_group = staticmethod(create_local_process_group)
    get_local_rank = staticmethod(get_local_rank)
    get_local_size = staticmethod(get_local_size)
    get_local_process_group = staticmethod(get_local_process_group)


d2_comm = CommModule()


# Copyright (c) Facebook, Inc. and its affiliates.
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import functools
import numpy as np
import torch
import torch.distributed as dist

_LOCAL_PROCESS_GROUP = None
_MISSING_LOCAL_PG_ERROR = (
    "Local process group is not yet created! Please use detectron2's `launch()` "
    "to start processes and initialize pytorch process group. If you need to start "
    "processes in other ways, please call comm.create_local_process_group("
    "num_workers_per_machine) after calling torch.distributed.init_process_group()."
)


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


@functools.lru_cache()
def create_local_process_group(num_workers_per_machine: int) -> None:
    """
    Create a process group that contains ranks within the same machine.

    Detectron2's launch() in engine/launch.py will call this function. If you start
    workers without launch(), you'll have to also call this. Otherwise utilities
    like `get_local_rank()` will not work.

    This function contains a barrier. All processes must call it together.

    Args:
        num_workers_per_machine: the number of worker processes per machine. Typically
          the number of GPUs.
    """
    global _LOCAL_PROCESS_GROUP
    assert _LOCAL_PROCESS_GROUP is None
    assert get_world_size() % num_workers_per_machine == 0
    num_machines = get_world_size() // num_workers_per_machine
    machine_rank = get_rank() // num_workers_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_workers_per_machine, (i + 1) * num_workers_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            _LOCAL_PROCESS_GROUP = pg


def get_local_process_group():
    """
    Returns:
        A torch process group which only includes processes that are on the same
        machine as the current process. This group can be useful for communication
        within a machine, e.g. a per-machine SyncBN.
    """
    assert _LOCAL_PROCESS_GROUP is not None, _MISSING_LOCAL_PG_ERROR
    return _LOCAL_PROCESS_GROUP


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None, _MISSING_LOCAL_PG_ERROR
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    assert _LOCAL_PROCESS_GROUP is not None, _MISSING_LOCAL_PG_ERROR
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    if dist.get_backend() == dist.Backend.NCCL:
        # This argument is needed to avoid warnings.
        # It's valid only for NCCL backend.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()  # use CPU group by default, to reduce GPU RAM usage.
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    return output


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return [data]
    rank = dist.get_rank(group=group)

    if rank == dst:
        output = [None for _ in range(world_size)]
        dist.gather_object(data, output, dst=dst, group=group)
        return output
    else:
        dist.gather_object(data, None, dst=dst, group=group)
        return []


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
        If workers need a shared RNG, they can use this shared seed to
        create one.

    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2**31)
    all_ints = all_gather(ints)
    return all_ints[0]


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.

    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum

    Returns:
        a dict with the same keys as input_dict, after reduction.
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
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


LOG = logging.getLogger(__name__)

_NESTED_BROADCAST_FROM_MASTER = False


def is_distributed():
    return d2_comm.get_world_size() > 1


def broadcast_from_master(fn):
    """If distributed, only the master executes the function and broadcast the results to other workers.

    Usage:
    @broadcast_from_master
    def foo(a, b): ...
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):  # pylint: disable=unused-argument
        global _NESTED_BROADCAST_FROM_MASTER

        if not is_distributed():
            return fn(*args, **kwargs)

        if _NESTED_BROADCAST_FROM_MASTER:
            assert d2_comm.is_main_process()
            LOG.warning(f"_NESTED_BROADCAST_FROM_MASTER = True, {fn.__name__}")
            return fn(*args, **kwargs)

        if d2_comm.is_main_process():
            _NESTED_BROADCAST_FROM_MASTER = True
            ret = [
                fn(*args, **kwargs),
            ]
            _NESTED_BROADCAST_FROM_MASTER = False
        else:
            ret = [
                None,
            ]
        if dist.is_initialized():
            dist.broadcast_object_list(ret)
        ret = ret[0]

        assert ret is not None
        return ret

    return wrapper


def master_only(fn):
    """If distributed, only the master executes the function.

    Usage:
    @master_only
    def foo(a, b): ...
    """

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if d2_comm.is_main_process():
            ret = fn(*args, **kwargs)
        d2_comm.synchronize()
        if d2_comm.is_main_process():
            return ret

    return wrapped_fn


def gather_dict(dikt):
    """Gather python dictionaries from all workers to the rank=0 worker.

    Assumption: the keys of `dikt` are disjoint across all workers.

    If rank = 0, then returned aggregated dict.
    If rank > 0, then return `None`.
    """
    dict_lst = d2_comm.gather(dikt, dst=0)
    if d2_comm.is_main_process():
        gathered_dict = {}
        for dic in dict_lst:
            for k in dic.keys():
                assert k not in gathered_dict, f"Dictionary key overlaps: {k}"
            gathered_dict.update(dic)
        return gathered_dict
    else:
        return None


def reduce_sum(tensor):
    """
    Adapted from AdelaiDet:
        https://github.com/aim-uofa/AdelaiDet/blob/master/adet/utils/comm.py
    """
    if not is_distributed():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor
