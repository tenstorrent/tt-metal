# Shim to expose MultimodalQwen2Model and MultimodalQwen2Config for tests.
try:
    # Prefer an official implementation if present in minicpm_official wrapper
    from .minicpm_official_wrapper import MiniCPMOConfig as MultimodalQwen2Config

    # Provide a thin wrapper class that uses the official wrapper for forward compatibility.
    class MultimodalQwen2Model:
        def __init__(self, config):
            # Use the official wrapper as a reference implementation where possible
            self._wrapped = __import__("minicpm_official_wrapper", fromlist=["MiniCPMOConfig"])
            self._cfg = config

        def load_weights(self, weights):
            # Not a real loader; rely on tests to use TTNN weights for comparison.
            pass

        def __call__(self, *args, **kwargs):
            raise NotImplementedError("MultimodalQwen2Model shim: use MiniCPMO wrapper for full functionality.")

except Exception:
    # Fallback to legacy reference_incorrect if present
    try:
        from reference.multimodal_qwen import MultimodalQwen2Model, MultimodalQwen2Config  # type: ignore
    except Exception:
        # Minimal placeholders
        class MultimodalQwen2Config:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class MultimodalQwen2Model:
            def __init__(self, config):
                pass

            def load_weights(self, weights):
                pass

            def __call__(self, *args, **kwargs):
                raise NotImplementedError("MultimodalQwen2Model shim used; real implementation missing.")


__all__ = ["MultimodalQwen2Model", "MultimodalQwen2Config"]
