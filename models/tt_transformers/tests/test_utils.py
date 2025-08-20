# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from models.tt_transformers.tt.model_config import HfAttentionWrapper, HfDecoderWrapper, HfModelWrapper


def _extract_dtype_from_state_dict(model):
    """Helper to extract dtype from model's state_dict."""
    try:
        state_dict = model.state_dict()
        for key, param in state_dict.items():
            if "weight" in key:
                print(f"get_ref_model_dype: key={key}, dtype={param.dtype}")
                return param.dtype
    except Exception as e:
        pass
    return None


def get_ref_model_dype(ref_model, model_name):
    default_dype = torch.float32

    if ref_model is None and model_name is None:
        return default_dype

    try:
        models_to_check = []
        if isinstance(ref_model, HfAttentionWrapper):
            models_to_check.append(ref_model.attention)
        elif isinstance(ref_model, HfDecoderWrapper):
            models_to_check.append(ref_model.decoder)
        elif isinstance(ref_model, HfModelWrapper):
            models_to_check.append(ref_model.model)
        else:
            models_to_check = [ref_model]

        # Try all models until one works
        for model in models_to_check:
            if model is not None:
                dtype = _extract_dtype_from_state_dict(model)
                if dtype is not None:
                    return dtype

    except Exception as e:
        pass

    # try hardcoded dtypes
    if model_name and isinstance(model_name, str):
        model_name_lower = model_name.lower()
        if "mistral-7b" in model_name_lower:
            return torch.bfloat16
        if "llama" in model_name_lower:
            return torch.bfloat16

    return default_dype
