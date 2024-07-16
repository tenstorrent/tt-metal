import torch
import ttnn

log = []
tlog_device_mesh = None


def tlog(name, tensor, gather_dim=0, log_filename="tlog.pt"):
    if isinstance(tensor, ttnn.Tensor):
        if gather_dim is None:
            tensor = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(tlog_device_mesh, dim=0))[0]
        else:
            tensor = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(tlog_device_mesh, dim=gather_dim))
    log.append((name, tensor.detach()))
    torch.save(log, log_filename)


def prefix_pair(prefix_a, prefix_b, items):
    """Convert a log of (name, tensor) items to a log of (name, tensor_a, tensor_b) items
    by matching pairs that differ only by the prefix to the previous/next with that name"""
    items_a = [(n, t) for n, t in items if n.startswith(prefix_a)]
    items_b = [(n, t) for n, t in items if n.startswith(prefix_b)]

    output = []
    for (na, ta), (nb, tb) in zip(items_a, items_b):
        name = na[len(prefix_a) :]
        if name == nb[len(prefix_b) :]:
            output.append((name, ta, tb))
        else:
            print(f"prefix_pair: mismatched names {na} {nb}")

    return output
