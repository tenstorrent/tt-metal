# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Model optimisation configs for the Quasar port of the SDXL base UNet (1024x1024 only).

``ModelOptimisations1024x1024Quasar`` currently *inherits every Wormhole 1024x1024 config*
(8x8 worker grid: conv core grids, matmul program configs on 5x8 / 8x8, group norm on 8x8 /
4x8, layer norm on 5x8, SDPA on 8x8). Quasar has an 8x4 worker grid with 4 MB of L1 per
cluster, so these configs are a *starting point* that will be retuned op by op once the
Quasar ops exist. Keep all Quasar-specific overrides in this class so the diff against the
Wormhole config stays reviewable.

The only behavioural difference today is that the group-norm parameter tensors (gamma, beta,
masks) are uploaded through ``qsr`` (host tilize + ``quasar.to_device``) instead of the
mainline device path.
"""

import ttnn
from models.demos.stable_diffusion_xl_base.quasar.tt.sdxl_utility import (
    prepare_gn_beta_gamma,
    prepare_gn_mask,
    prepare_gn_mask_negative_mask,
)
from models.demos.stable_diffusion_xl_base.tt.model_configs.model_configs_1024x1024 import (
    ModelOptimisations1024x1024,
)

IMAGE_RESOLUTION = (1024, 1024)


class ModelOptimisations1024x1024Quasar(ModelOptimisations1024x1024):
    """Wormhole 1024x1024 configs, to be retuned for the Quasar 8x4 grid (see module docstring).

    Quasar has no block-float formats (``is_supported_quasar`` in tt_backend_api_types.cpp lists
    Float16/Float16_b/Float32/Fp8/Int*/MxFp*/MxInt*, no Bfp*), so the Wormhole bfloat8_b weight
    dtypes (attention, feed-forward, conv weights) are replaced with bfloat16 here. That doubles
    the weight footprint (SDXL UNet: ~5.2 GB in bf16); the MX formats are the eventual target.
    """

    def __init__(
        self,
        conv_act_dtype=ttnn.bfloat16,
        conv_w_dtype=ttnn.bfloat16,
        attention_weights_dtype=ttnn.bfloat16,
        ff_weights_dtype=ttnn.bfloat16,
        force_full_grid=False,
    ):
        super().__init__(
            conv_act_dtype=conv_act_dtype,
            conv_w_dtype=conv_w_dtype,
            attention_weights_dtype=attention_weights_dtype,
            ff_weights_dtype=ff_weights_dtype,
            force_full_grid=force_full_grid,
        )
        # the base class hardcodes conv_ws_dtype = bfloat8_b inside most Conv2dConfigs
        self.conv_ws_dtype = ttnn.bfloat16
        for conv_config in self.conv_configs.values():
            if conv_config.weights_dtype == ttnn.bfloat8_b:
                conv_config.weights_dtype = ttnn.bfloat16

    def _generate_groupnorm_params_quasar(self, config, weights, bias, groups, device):
        if config["memory_config"] != ttnn.DRAM_MEMORY_CONFIG:
            gamma, beta = prepare_gn_beta_gamma(device, weights, bias, config["op_config"]["core_grid"].x)
            mask = prepare_gn_mask(device, weights.shape[0], groups, config["op_config"]["core_grid"].x)
            negative_mask = (
                prepare_gn_mask_negative_mask(device, weights.shape[0], groups, config["op_config"]["core_grid"].x)
                if config["negative_mask"]
                else None
            )
        else:
            # Not used by the 1024x1024 config (every group norm is L1 block sharded); kept for parity.
            [gamma, beta], mask = ttnn.dram_group_norm_params_from_torch(
                [weights, bias],
                weights.shape[0],
                groups,
                device,
                core_grid=config["op_config"]["core_grid"],
                return_mask=True,
            )
            negative_mask = None

        return mask, negative_mask, gamma, beta

    def get_groupnorm_params(self, module_path, weights, bias, groups, device):
        config = self._get_groupnorm_config(module_path)
        mask, negative_mask, gamma, beta = self._generate_groupnorm_params_quasar(config, weights, bias, groups, device)
        return config["op_config"], config["memory_config"], mask, negative_mask, gamma, beta


def get_image_resolution_from_model_config(model_config):
    assert isinstance(model_config, ModelOptimisations1024x1024Quasar), type(model_config).__name__
    return IMAGE_RESOLUTION


def load_model_optimisations(
    image_resolution=IMAGE_RESOLUTION,
    conv_act_dtype=None,
    conv_w_dtype=None,
    attention_weights_dtype=None,
    ff_weights_dtype=None,
    force_full_grid=False,
):
    """Quasar counterpart of ``models.demos.stable_diffusion_xl_base.tt.model_configs.load_model_optimisations``.

    Only (1024, 1024) is supported. The mainline loader raises on Quasar because it keys the
    config on ``is_wormhole_b0()`` / ``is_blackhole()``.
    """
    if tuple(image_resolution) != IMAGE_RESOLUTION:
        raise ValueError(f"The Quasar SDXL port only supports {IMAGE_RESOLUTION}, got {image_resolution}")

    init_kwargs = {"force_full_grid": force_full_grid}
    if conv_act_dtype is not None:
        init_kwargs["conv_act_dtype"] = conv_act_dtype
    if conv_w_dtype is not None:
        init_kwargs["conv_w_dtype"] = conv_w_dtype
    if attention_weights_dtype is not None:
        init_kwargs["attention_weights_dtype"] = attention_weights_dtype
    if ff_weights_dtype is not None:
        init_kwargs["ff_weights_dtype"] = ff_weights_dtype
    return ModelOptimisations1024x1024Quasar(**init_kwargs)
