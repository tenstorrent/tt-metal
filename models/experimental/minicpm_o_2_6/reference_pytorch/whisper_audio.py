try:
    # Prefer official components
    from minicpm_official.modeling_minicpmo import WhisperEncoder as WhisperAudioEncoder
    from minicpm_official.configuration_minicpm import WhisperConfig as WhisperAudioConfig
except Exception:
    try:
        from reference.whisper_audio import WhisperAudioEncoder, WhisperAudioConfig  # type: ignore
    except Exception:
        import torch.nn as nn

        class WhisperAudioConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class WhisperAudioEncoder(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def forward(self, *args, **kwargs):
                raise NotImplementedError("WhisperAudioEncoder shim used; real implementation missing.")


__all__ = ["WhisperAudioEncoder", "WhisperAudioConfig"]
