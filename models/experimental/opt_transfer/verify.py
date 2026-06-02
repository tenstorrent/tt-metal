import torch


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def golden_outputs(reference_module, example_inputs: tuple) -> torch.Tensor:
    reference_module.eval()
    with torch.no_grad():
        return reference_module(*example_inputs)
