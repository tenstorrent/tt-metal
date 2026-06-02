import torch

_REGISTRY = {}


def register(name):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn

    return deco


def apply_transform(name: str, weights: dict, order: list) -> dict:
    return _REGISTRY[name](weights, order)


@register("identity")
def _identity(weights, order):
    return weights[order[0]]


@register("concat_qkv")
def _concat_qkv(weights, order):
    w = torch.cat([weights[n]["weight"] for n in order], dim=0)
    biases = [weights[n].get("bias") for n in order]
    b = torch.cat(biases, dim=0) if all(x is not None for x in biases) else None
    return {"weight": w, "bias": b}
