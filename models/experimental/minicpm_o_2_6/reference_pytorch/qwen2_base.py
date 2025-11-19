try:
    # Prefer the real Qwen2Config from transformers if available
    from transformers import Qwen2ForCausalLM as Qwen2Model, Qwen2Config
except Exception:
    # Fallback placeholder for tests that only instantiate config/model
    class Qwen2Model:
        def __init__(self, config):
            pass

        def load_state_dict(self, *args, **kwargs):
            pass

    class Qwen2Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


__all__ = ["Qwen2Model", "Qwen2Config"]
