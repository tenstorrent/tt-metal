try:
    # Prefer legacy reference_incorrect version which matches test API
    from reference.pytorch_chattts_decoder import PyTorchChatTTSDecoder  # type: ignore
except Exception:
    try:
        # Fallback to official implementation if legacy not available
        from minicpm_official.modeling_minicpmo import ConditionalChatTTS as PyTorchChatTTSDecoder
    except Exception:
        import torch.nn as nn

        class PyTorchChatTTSDecoder(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def forward(self, *args, **kwargs):
                raise NotImplementedError("PyTorchChatTTSDecoder shim used; real implementation missing.")


__all__ = ["PyTorchChatTTSDecoder"]
