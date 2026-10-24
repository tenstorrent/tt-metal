# SPDX-License-Identifier: Apache-2.0
"""Decode-only temporal-window packing candidate. No cross-user convolution outputs retained."""
import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.multichip_decoder import fp32_recurrence
from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import reshape

def packed_delta(self, x, state):
    batch, length, _ = x.shape
    projected = self._linear(x, self.mix)
    assert length == 1 and batch > 1
    current = ttnn.to_layout(projected[..., :2560], ttnn.ROW_MAJOR_LAYOUT)
    windows = ttnn.concat([state.conv, current], dim=1)
    packed = ttnn.reshape(windows, [1, batch*4, 2560])
    physical = ((batch*4 + 31)//32)*32
    if physical != batch*4:
        packed = ttnn.pad(packed, [(0,0),(0,physical-batch*4),(0,0)], value=0)
    packed = ttnn.to_layout(packed, ttnn.TILE_LAYOUT)
    hist = state.conv[:1]
    outputs = ttnn.experimental.kda.qkv_causal_conv1d_silu(
        packed, hist, *self.taps, 512, 512, 1536,
        actual_start=self.conv_start, predecessor_carry=hist,
        program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=128),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False))
    qflat, kflat, vflat = [ttnn.to_layout(
        ttnn.reshape(ttnn.to_layout(t,ttnn.ROW_MAJOR_LAYOUT)[:,:batch*4], [batch,4,width])[:,3:4],
        ttnn.TILE_LAYOUT) for t,width in zip(outputs,(512,512,1536))]
    next_conv = windows[:,1:4]
    ttnn.copy(next_conv, state.conv)

    def l2(t):
        t = ttnn.typecast(t, ttnn.float32)
        return ttnn.rms_norm(t, weight=self.l2_weight, epsilon=1e-6/128, compute_kernel_config=self.ckc)

    a = ttnn.typecast(projected[..., 4096:4108], ttnn.float32)
    decay = ttnn.multiply(self.neg_A, ttnn.softplus(ttnn.add(a, self.dt)))
    beta = ttnn.sigmoid(ttnn.typecast(projected[..., 4128:4140], ttnn.float32))
    q = l2(reshape(qflat, [batch, length, 4, 128]))
    k = l2(reshape(kflat, [batch, length, 4, 128]))
    v = ttnn.typecast(reshape(vflat, [batch, length, 12, 128]), ttnn.float32)
    # This kernel assigns one value head to each core. Split its batch axis,
    # retaining batched projections and an explicit state for every request.
    grid = self.mesh_device.compute_with_storage_grid_size()
    kernel_batch = (grid.x * grid.y) // 12
    ys, states = [], []
    for begin in range(0, batch, kernel_batch):
        end = min(begin + kernel_batch, batch)
        y_part, state_part = fp32_recurrence(
            q[begin:end], k[begin:end], v[begin:end], decay[begin:end], beta[begin:end],
            initial_state=state.recurrent[begin:end], output_final_state=True, compute_kernel_config=self.ckc,
            inplace_state=self.policy.get("inplace_recurrence", False),
        )
        ys.append(y_part)
        states.append(state_part)
    y = ttnn.concat(ys, dim=0) if len(ys) > 1 else ys[0]
    new_state = ttnn.concat(states, dim=0) if len(states) > 1 else states[0]
    if new_state.buffer_address() != state.recurrent.buffer_address():
        ttnn.copy(new_state, state.recurrent)
    y = self._norm(ttnn.typecast(y, ttnn.bfloat16), self.out_norm)
    z = reshape(projected[..., 2560:4096], [batch, length, 12, 128])
    return self._linear(reshape(ttnn.multiply(z, y, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), [batch, length, 1536]), self.out)
