try:
    # Use MultiModalProjector from official modeling if available
    from .minicpm_official.modeling_minicpmo import MultiModalProjector as PyTorchAudioProjector
except Exception:
    try:
        from reference.pytorch_audio_projector import PyTorchAudioProjector  # type: ignore
    except Exception:
        import torch.nn as nn

        class PyTorchAudioProjector(nn.Module):
            def __init__(self, input_dim, output_dim, pool_step=2):
                super().__init__()
                self.linear1 = nn.Linear(input_dim, output_dim)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(output_dim, output_dim)

            def forward(self, x):
                x = self.relu(self.linear1(x))
                x = self.linear2(x)
                return x


__all__ = ["PyTorchAudioProjector"]
