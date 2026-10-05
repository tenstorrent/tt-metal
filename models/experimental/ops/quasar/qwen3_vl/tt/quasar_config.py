# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar bring-up configuration for the Qwen3-VL copy: bf16, device-derived grids, truncation."""
import math


def vision_padded_seq_len(n: int) -> int:
    # Always a multiple of 128 (vision_attention). Above 1024 the MLP and WO reshape into 1024-row chunks and,
    # above 2048, the QKV matmul into 2048-row chunks, so anything over 1024 rounds up to a multiple of 2048.
    step = 128 if n <= 1024 else 2048
    return math.ceil(n / step) * step


def truncate_hf_config(config, vision_layers, text_layers, deepstack_at=None):
    config.vision_config.depth = vision_layers
    config.text_config.num_hidden_layers = text_layers
    if deepstack_at is not None:
        config.vision_config.deepstack_visual_indexes = [deepstack_at]
    return config


# --- Quasar model args -------------------------------------------------------------------------
from contextlib import contextmanager  # noqa: E402

import ttnn  # noqa: E402
from models.common.utility_functions import is_quasar  # noqa: E402
from models.experimental.ops.quasar.qwen3_vl.tt.model_config import VisionModelArgs  # noqa: E402
from models.tt_transformers.tt import model_config as _ttt_mc  # noqa: E402
from models.tt_transformers.tt.model_config import (  # noqa: E402
    DecodersPrecision,
    MathFidelitySetting,
    ModelArgs,
    ModelOptimizations,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
)

# ModelArgs keys tuning tables by device name; Quasar has none, so it borrows the single-chip WH entries.
QUASAR_DEVICE_NAME = "N150"


def bf16_decoders_precision(num_decoders, model_name):
    settings = {
        "TensorPrecision": {g: PrecisionSetting.BF16 for g in TensorGroup},
        # HiFi4 without fp32 dest accumulation: matmuls reload fp32 partials through srcA, which is undefined.
        "OpFidelity": {g: MathFidelitySetting.HIFI4_FP16 for g in OpGroup},
    }
    return DecodersPrecision(num_decoders, model_name, ModelOptimizations(settings))


def strip_fp32_dest_acc(args):
    """Turn off fp32 dest accumulation in every compute_kernel_config_* attribute; returns the names changed."""
    changed = []
    for name, cfg in list(vars(args).items()):
        if name.startswith("compute_kernel_config") and getattr(cfg, "fp32_dest_acc_en", False):
            setattr(
                args,
                name,
                ttnn.WormholeComputeKernelConfig(
                    math_fidelity=cfg.math_fidelity,
                    math_approx_mode=cfg.math_approx_mode,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=cfg.packer_l1_acc,
                ),
            )
            changed.append(name)
    return changed


@contextmanager
def _quasar_device_name():
    orig = _ttt_mc.determine_device_name

    def name(mesh_device):
        try:
            return orig(mesh_device)
        except ValueError:
            return QUASAR_DEVICE_NAME

    _ttt_mc.determine_device_name = name
    try:
        yield
    finally:
        _ttt_mc.determine_device_name = orig


class _QuasarArgsMixin:
    def _quasar_init(self, parent_init, *args, **kwargs):
        if kwargs.get("optimizations") is None:
            kwargs["optimizations"] = lambda a: bf16_decoders_precision(a.n_layers, a.model_name)
        with _quasar_device_name():
            parent_init(self, *args, **kwargs)
        self.lm_head_dtype = ttnn.bfloat16
        self.ccl_dtype = ttnn.bfloat16
        strip_fp32_dest_acc(self)


class QuasarModelArgs(_QuasarArgsMixin, ModelArgs):
    def __init__(self, *args, **kwargs):
        self._quasar_init(ModelArgs.__init__, *args, **kwargs)
        # ttnn.scatter is not ported to Quasar; the vision-token merge copies rows on the host instead.
        self.device_scatter = False


class QuasarVisionModelArgs(_QuasarArgsMixin, VisionModelArgs):
    def __init__(self, *args, **kwargs):
        self._quasar_init(VisionModelArgs.__init__, *args, **kwargs)
        self.vision_weight_dtype = ttnn.bfloat16
        self.vision_mlp_fc1_dtype = ttnn.bfloat16


def model_args_classes(force=False):
    if force or is_quasar():
        return QuasarModelArgs, QuasarVisionModelArgs
    return ModelArgs, VisionModelArgs
