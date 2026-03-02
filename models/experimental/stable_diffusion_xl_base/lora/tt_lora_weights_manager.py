import re
from loguru import logger
from collections import defaultdict
import ttnn


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

    def load_lora_weights(self, lora_path):
        if self.has_lora_adapter():
            logger.info("LoRA weights already loaded, skipping.")
            return

        self._torch_pipeline.load_lora_weights(lora_path)

    def fuse_lora(self, lora_scale=1.0):
        if not self.has_lora_adapter():
            logger.warning("No LoRA weights loaded. Please load LoRA weights with load_lora_weights() before fusing.")
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
                        lora_a_T, dtype=ttnn.bfloat8_b, device=self._device, layout=ttnn.TILE_LAYOUT
                    )
                    lora_b_tt = ttnn.from_torch(
                        lora_b_T, dtype=ttnn.bfloat8_b, device=self._device, layout=ttnn.TILE_LAYOUT
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
                lora_a_tt = ttnn.from_torch(
                    lora_a_T, dtype=ttnn.bfloat8_b, device=self._device, layout=ttnn.TILE_LAYOUT
                )
                lora_b1_tt = ttnn.from_torch(
                    lora_b1_T, dtype=ttnn.bfloat8_b, device=self._device, layout=ttnn.TILE_LAYOUT
                )
                lora_b2_tt = ttnn.from_torch(
                    lora_b2_T, dtype=ttnn.bfloat8_b, device=self._device, layout=ttnn.TILE_LAYOUT
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
            lora_a_tt = ttnn.from_torch(lora_a_T, dtype=ttnn.bfloat8_b, device=self._device, layout=ttnn.TILE_LAYOUT)
            lora_b_tt = ttnn.from_torch(lora_b_T, dtype=ttnn.bfloat8_b, device=self._device, layout=ttnn.TILE_LAYOUT)
            delta_tt = ttnn.mul(ttnn.matmul(lora_a_tt, lora_b_tt, compute_kernel_config=self._mm_compute_config), scale)
            self._apply_delta(f"{layer_path}.weight", delta_tt)

        self._apply_qkv_deltas(self_attention_matrices)

        ttnn.synchronize_device(self._device)

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

        self._torch_pipeline.unload_lora_weights()
