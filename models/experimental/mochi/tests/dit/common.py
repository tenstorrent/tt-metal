import os

from models.experimental.mochi.tt.common import get_mochi_dir

NUM_HEADS = 24
HEAD_DIM = 128


def load_model_weights(state_dict_prefix):
    """Load and prepare model weights."""
    weights_path = os.path.join(get_mochi_dir(), "dit.safetensors")
    from safetensors.torch import load_file

    state_dict = load_file(weights_path)
    return state_dict, {
        k[len(state_dict_prefix) + 1 :]: v for k, v in state_dict.items() if k.startswith(state_dict_prefix)
    }
