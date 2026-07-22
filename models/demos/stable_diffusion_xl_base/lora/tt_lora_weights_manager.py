# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import re
from collections import defaultdict

from loguru import logger

import ttnn


def _clip_text_model_is_flattened(text_encoder):
    """Whether this CLIP text encoder's attention module paths lack the
    ``text_model.`` prefix.

    transformers>=5 flattened ``CLIPTextModel`` (SDXL's first text encoder) so
    ``named_modules()`` now yields e.g. ``encoder.layers.0.self_attn.q_proj``
    instead of ``text_model.encoder.layers.0.self_attn.q_proj``. The projection
    text encoder (``CLIPTextModelWithProjection``, te2) still keeps the prefix.
    """
    for name, _ in text_encoder.named_modules():
        if name.endswith((".q_proj", ".k_proj", ".v_proj", ".out_proj", ".fc1", ".fc2")):
            return not name.startswith("text_model.")
    return False


def _strip_text_model_segment(mapping, prefix):
    """Drop the ``text_model.`` segment right after ``{prefix}.`` in each key."""
    seg = f"{prefix}.text_model."
    return {(f"{prefix}." + k[len(seg) :] if k.startswith(seg) else k): v for k, v in mapping.items()}


def load_lora_weights_te_compat(pipeline, lora_path, adapter_name=None):
    """``pipeline.load_lora_weights`` that also works under transformers>=5.

    diffusers' text-encoder LoRA loader builds its rank dict by matching
    ``{module}.lora_B.weight`` against the live ``text_encoder`` module names, but
    the converted LoRA keys always carry a ``text_model.`` prefix. Since
    transformers>=5 flattened ``CLIPTextModel`` (dropping that prefix from its
    module paths), nothing matches for the first text encoder, the rank dict comes
    out empty, and ``get_peft_kwargs`` raises ``IndexError: list index out of
    range``. When an encoder is flattened we strip ``text_model.`` from its LoRA
    keys (weights and network-alpha keys) and mirror diffusers' own
    unet -> te1 -> te2 load sequence with the remapped state dict.

    When no encoder is flattened (transformers<5, or after diffusers fixes this
    upstream) we defer to the stock loader untouched.

    TODO: remove once the upstream diffusers loader handles transformers>=5.
    """
    encoders = [
        (getattr(pipeline, "text_encoder", None), "text_encoder"),
        (getattr(pipeline, "text_encoder_2", None), "text_encoder_2"),
    ]
    if not any(te is not None and _clip_text_model_is_flattened(te) for te, _ in encoders):
        pipeline.load_lora_weights(lora_path)
        return

    result = pipeline.lora_state_dict(lora_path)
    state_dict = result[0]
    network_alphas = result[1] if len(result) > 1 else None

    for te, prefix in encoders:
        if te is not None and _clip_text_model_is_flattened(te):
            state_dict = _strip_text_model_segment(state_dict, prefix)
            if network_alphas:
                network_alphas = _strip_text_model_segment(network_alphas, prefix)

    pipeline.load_lora_into_unet(
        state_dict, network_alphas, pipeline.unet, adapter_name=adapter_name, _pipeline=pipeline
    )
    pipeline.load_lora_into_text_encoder(
        state_dict,
        network_alphas,
        pipeline.text_encoder,
        prefix="text_encoder",
        adapter_name=adapter_name,
        _pipeline=pipeline,
    )
    pipeline.load_lora_into_text_encoder(
        state_dict,
        network_alphas,
        pipeline.text_encoder_2,
        prefix="text_encoder_2",
        adapter_name=adapter_name,
        _pipeline=pipeline,
    )


class TtLoRAWeightsManager:
    def __init__(self, device, torch_pipeline):
        self._device = device
        self._torch_pipeline = torch_pipeline

        self._base_weights_host = {}
        self._base_weights_device = {}

        self._mm_compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        self._is_fused = False

        # Status tracking for the most recent load_lora_weights() call. These let
        # the pipeline/runner report back what was actually applied vs skipped.
        self._skipped_reason = None
        self._text_encoder_components = []

    def is_fused(self):
        return self._is_fused

    def skipped_reason(self):
        return self._skipped_reason

    def text_encoder_components(self):
        return list(self._text_encoder_components)

    def prepare_lora_linear_params(self, device, weights, bias, dtype, name, permute_weights=True):
        if permute_weights:
            weights = weights.movedim(-1, -2)
        tt_weights_host = ttnn.from_torch(weights, dtype, layout=ttnn.TILE_LAYOUT)
        tt_weights_device = ttnn.allocate_tensor_on_device(tt_weights_host.spec, device)
        ttnn.copy_host_to_device_tensor(tt_weights_host, tt_weights_device)
        tt_bias = ttnn.from_torch(bias, dtype, device=device, layout=ttnn.TILE_LAYOUT) if bias is not None else None

        self._base_weights_host[f"{name}.weight"] = tt_weights_host
        self._base_weights_device[f"{name}.weight"] = tt_weights_device
        return tt_weights_device, tt_bias

    def get_lora_adapters(self):
        return self._torch_pipeline.get_list_adapters()

    def has_lora_adapter(self):
        return any(self.get_lora_adapters().values())

    def _get_supported_lora_affecting_ops(self):
        return (".to_q", ".to_k", ".to_v", ".to_out.0", ".proj_in", ".proj_out", ".ff.net.2", ".ff.net.0.proj")

    def _uses_dora(self):
        adapters = self.get_lora_adapters()
        unet_adapters = adapters.get("unet", [])
        if not unet_adapters:
            return False

        adapter_name = unet_adapters[0]
        unet = self._torch_pipeline.unet

        for _, module in unet.named_modules():
            use_dora_dict = getattr(module, "use_dora", None)
            if use_dora_dict and use_dora_dict.get(adapter_name, False):
                return True

        return False

    def _text_encoder_components_present(self):
        adapters = self.get_lora_adapters()
        return [c for c in ("text_encoder", "text_encoder_2") if adapters.get(c)]

    def _affects_unsupported_ops(self):
        for key in self._get_lora_params():
            if not any(key.endswith(s) for s in self._get_supported_lora_affecting_ops()):
                return True
        return False

    def load_lora_weights(self, lora_path):
        self._skipped_reason = None
        self._text_encoder_components = []

        if self.has_lora_adapter():
            logger.info("LoRA weights already loaded, skipping.")
            return

        load_lora_weights_te_compat(self._torch_pipeline, lora_path)

        if self._uses_dora():
            logger.warning("DoRA is not supported, skipping loading LoRA weights.")
            self._torch_pipeline.unload_lora_weights()
            self._skipped_reason = "dora"
            return

        if self._affects_unsupported_ops():
            logger.warning("LoRA weights affect unsupported UNet operations, skipping loading LoRA weights.")
            self._torch_pipeline.unload_lora_weights()
            self._skipped_reason = "unsupported_ops"
            return

        # Text-encoder adapters are supported via host-side fuse + on-device
        # encoder reload (handled by TtSDXLPipeline). Record which TE components
        # are present so the caller can fuse and report them.
        self._text_encoder_components = self._text_encoder_components_present()

    def fuse_lora(self, lora_scale=1.0):
        if not self.has_lora_adapter():
            logger.warning("No LoRA weights loaded. Please load LoRA weights with load_lora_weights() before fusing.")
            return

        if self._is_fused:
            logger.info("LoRA weights already fused. Skipping fusion.")
            return

        lora_params = self._get_lora_params()
        if not lora_params:
            logger.info("No LoRA parameters affecting the UNet were found. LoRA will have no effect.")
            return

        # Process self-attention Q, K and V matrices separately
        self_attention_matrices = defaultdict(dict)

        for layer_path, (lora_a, lora_b, scaling) in lora_params.items():
            scale = scaling * lora_scale
            # (B@A)^T = A^T @ B^T
            # lora_a: [rank, in] -> lora_a_T: [in, rank]
            # lora_b: [out, rank] -> lora_b_T: [rank, out]
            lora_a_T = lora_a.movedim(-1, -2).unsqueeze(0).unsqueeze(0)
            lora_b_T = lora_b.movedim(-1, -2).unsqueeze(0).unsqueeze(0)

            # Collect self-attention QKV weights matrices
            if layer_path.endswith((".to_q", ".to_k", ".to_v")):
                attn_path, component = layer_path.rsplit(".", 1)
                if f"{attn_path}.to_qkv.weight" in self._base_weights_device:
                    lora_a_tt = ttnn.from_torch(
                        lora_a_T, dtype=ttnn.bfloat16, device=self._device, layout=ttnn.TILE_LAYOUT
                    )
                    lora_b_tt = ttnn.from_torch(
                        lora_b_T, dtype=ttnn.bfloat16, device=self._device, layout=ttnn.TILE_LAYOUT
                    )
                    delta_tt = ttnn.mul(
                        ttnn.matmul(lora_a_tt, lora_b_tt, compute_kernel_config=self._mm_compute_config), scale
                    )
                    self_attention_matrices[attn_path][component] = delta_tt
                    continue

            # Process GEGLU weights
            if (
                f"{layer_path}.linear_1.weight" in self._base_weights_device
                and f"{layer_path}.linear_2.weight" in self._base_weights_device
            ):
                lora_b1_T, lora_b2_T = lora_b_T.chunk(2, dim=-1)
                lora_a_tt = ttnn.from_torch(lora_a_T, dtype=ttnn.bfloat16, device=self._device, layout=ttnn.TILE_LAYOUT)
                lora_b1_tt = ttnn.from_torch(
                    lora_b1_T, dtype=ttnn.bfloat16, device=self._device, layout=ttnn.TILE_LAYOUT
                )
                lora_b2_tt = ttnn.from_torch(
                    lora_b2_T, dtype=ttnn.bfloat16, device=self._device, layout=ttnn.TILE_LAYOUT
                )
                delta_1 = ttnn.mul(
                    ttnn.matmul(lora_a_tt, lora_b1_tt, compute_kernel_config=self._mm_compute_config), scale
                )
                delta_2 = ttnn.mul(
                    ttnn.matmul(lora_a_tt, lora_b2_tt, compute_kernel_config=self._mm_compute_config), scale
                )
                self._apply_delta(f"{layer_path}.linear_1.weight", delta_1)
                self._apply_delta(f"{layer_path}.linear_2.weight", delta_2)
                continue

            # Process cross-attn Q/K/V, to_out, proj_in/out, ff.net.2 weights
            lora_a_tt = ttnn.from_torch(lora_a_T, dtype=ttnn.bfloat16, device=self._device, layout=ttnn.TILE_LAYOUT)
            lora_b_tt = ttnn.from_torch(lora_b_T, dtype=ttnn.bfloat16, device=self._device, layout=ttnn.TILE_LAYOUT)
            delta_tt = ttnn.mul(ttnn.matmul(lora_a_tt, lora_b_tt, compute_kernel_config=self._mm_compute_config), scale)
            self._apply_delta(f"{layer_path}.weight", delta_tt)

        self._apply_qkv_deltas(self_attention_matrices)

        ttnn.synchronize_device(self._device)
        self._is_fused = True

    def _get_lora_params(self):
        unet = self._torch_pipeline.unet
        lora_params = {}

        for name, module in unet.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue

            clean_name = re.sub(r"^base_model\.model\.", "", name)

            adapter_names = list(module.lora_A.keys())
            if not adapter_names:
                continue
            # TODO: Handle multiple adapters
            adapter_name = adapter_names[0]

            lora_a = module.lora_A[adapter_name].weight.data
            lora_b = module.lora_B[adapter_name].weight.data
            scaling = module.scaling.get(adapter_name, 1.0) if hasattr(module, "scaling") else 1.0

            lora_params[clean_name] = (lora_a, lora_b, scaling)

        return lora_params

    def _apply_delta(self, base_key, delta):
        if base_key not in self._base_weights_device:
            logger.debug(f"Base weight key {base_key} not found, skipping LoRA fusion for this layer.")
            return

        ttnn.add_(self._base_weights_device[base_key], delta)

    def _apply_qkv_deltas(self, qkv_pending):
        for attn_path, components in qkv_pending.items():
            qkv_key = f"{attn_path}.to_qkv.weight"
            if qkv_key not in self._base_weights_device:
                logger.debug(f"Base weight key {qkv_key} not found, skipping LoRA fusion for this layer.")
                continue

            to_q = components["to_q"]
            to_k = components["to_k"]
            to_v = components["to_v"]

            qkv_delta_tt = ttnn.concat([to_q, to_k, to_v], dim=-1)

            ttnn.add_(self._base_weights_device[qkv_key], qkv_delta_tt)

    def unload_lora_weights(self):
        # Restore original base weights to the device
        for key in self._base_weights_device.keys():
            host_tensor = self._base_weights_host[key]
            device_tensor = self._base_weights_device[key]
            ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)

        # Torch adapters may already have been stripped by the text-encoder fuse
        # path (which unloads them after merging); only unload when still attached.
        if self.has_lora_adapter():
            self._torch_pipeline.unload_lora_weights()
        self._is_fused = False
        self._skipped_reason = None
        self._text_encoder_components = []
