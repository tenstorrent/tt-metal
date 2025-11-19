try:
    from .minicpm_official.modeling_minicpmo import DVAE as PyTorchDVAE
except Exception:
    try:
        from reference.pytorch_dvae import PyTorchDVAE  # type: ignore
    except Exception:
        import torch.nn as nn

        class PyTorchDVAE(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def forward(self, *args, **kwargs):
                raise NotImplementedError("PyTorchDVAE shim used; real implementation missing.")


__all__ = ["PyTorchDVAE"]
