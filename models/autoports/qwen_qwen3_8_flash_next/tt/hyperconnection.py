"""Four-stream gated residual from the target HF decoder."""
import ttnn


class HyperConnection:
    @classmethod
    def from_state_dict(cls, state, *, prefix, hf_config, mesh_device):
        obj = cls()
        obj.width = hf_config.hidden_size
        obj.count = hf_config.hc_count
        obj.eps = hf_config.rms_norm_eps
        obj.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        def upload(x):
            return ttnn.from_torch(
                x.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        norm = 1 + state[prefix + ".hc_norm.weight"].float()
        obj.norm = [upload(norm[i * obj.width : (i + 1) * obj.width].reshape(1, 1, 1, -1)) for i in range(obj.count)]
        obj.down = upload(state[prefix + ".input_mix_weight_down.weight"].T)
        obj.up = upload(state[prefix + ".input_mix_weight_up.weight"].T)
        obj.inject = upload(state[prefix + ".block_inject_weight.weight"].T)
        return obj

    def _part(self, x, i):
        return x[..., i * self.width : (i + 1) * self.width]

    def read(self, x):
        streams = [
            ttnn.rms_norm(self._part(x, i), weight=self.norm[i], epsilon=self.eps, compute_kernel_config=self.compute)
            for i in range(self.count)
        ]
        normalized = ttnn.concat(streams, dim=-1)
        mix = ttnn.linear(normalized, self.down, compute_kernel_config=self.compute)
        mix = ttnn.silu(ttnn.multiply(mix, 1 / self.count))
        mix = ttnn.sigmoid(ttnn.linear(mix, self.up, compute_kernel_config=self.compute))
        mixed = ttnn.multiply(self._part(mix, 0), streams[0])
        for i in range(1, self.count):
            mixed = ttnn.add(mixed, ttnn.multiply(self._part(mix, i), streams[i]))
        mixed = ttnn.multiply(mixed, 1 / self.count)
        inject = ttnn.linear(normalized, self.inject, compute_kernel_config=self.compute)
        inject = ttnn.multiply(ttnn.sigmoid(ttnn.multiply(inject, 1 / self.count)), 2)
        return mixed, inject

    def write(self, residual, update, inject):
        return ttnn.concat(
            [
                ttnn.add(self._part(residual, i), ttnn.multiply(update, inject[..., i : i + 1]))
                for i in range(self.count)
            ],
            dim=-1,
        )
