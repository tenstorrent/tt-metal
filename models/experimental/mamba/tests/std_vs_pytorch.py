#!/usr/bin/env python3
"""Standard ttnn ops vs custom SSM kernels vs Hugging Face Mamba.

Reference is transformers MambaMixer.slow_forward on state-spaces/mamba-2.8b,
pulled from Hugging Face. One layer, batch 1, sequence 32, bfloat16 on device.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from transformers import MambaModel

import ttnn

MODEL_ID = "state-spaces/mamba-2.8b"
SEQ = 32
LAYER = 0
PCC_MIN = 0.99
DTYPE = ttnn.bfloat16
FID = ttnn.MathFidelity.HiFi2


def pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.detach().float().reshape(-1)
    b = expected.detach().float().reshape(-1)
    n = min(int(a.numel()), int(b.numel()))
    a = a[:n]
    b = b[:n]
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom) == 0.0:
        return 1.0
    return float((a * b).sum() / denom)


def to_dev(tensor: torch.Tensor, device):
    return ttnn.from_torch(
        tensor.contiguous().float(),
        dtype=DTYPE,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def timed(device, fn, iters=3):
    out = fn()
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - start) / iters * 1e3, out


def report(name, actual, expected, ms):
    score = pcc(actual, expected)
    status = "PASS" if score >= PCC_MIN else "FAIL"
    print(f"{status:4} {name:48} pcc={score:.4f}  {ms:8.3f} ms", flush=True)
    return score


def pad_n(tensor, state):
    if tensor.shape[-1] == state:
        return F.pad(tensor, (0, 32 - state))
    return tensor


def torch_pieces(mixer, hidden):
    batch, seq_len, _ = hidden.shape
    projected = mixer.in_proj(hidden).transpose(1, 2)
    x, gate = projected.chunk(2, dim=1)
    x = mixer.act(mixer.conv1d(x)[..., :seq_len])
    params = mixer.x_proj(x.transpose(1, 2))
    time_step, B, C = torch.split(params, [mixer.time_step_rank, mixer.ssm_state_size, mixer.ssm_state_size], dim=-1)
    dt = F.softplus(mixer.dt_proj(time_step)).transpose(1, 2)
    A = -torch.exp(mixer.A_log.float())
    discrete_A = torch.exp(A[None, :, None, :] * dt[:, :, :, None])
    delta_b_u = dt[:, :, :, None] * B[:, None, :, :].float() * x[:, :, :, None].float()
    state = torch.zeros(batch, mixer.intermediate_size, mixer.ssm_state_size)
    ys = []
    for i in range(seq_len):
        state = discrete_A[:, :, i, :] * state + delta_b_u[:, :, i, :]
        ys.append((state * C[:, i, :].unsqueeze(1)).sum(-1))
    y = torch.stack(ys, dim=-1)
    y = y + x * mixer.D[None, :, None]
    y = y * mixer.act(gate)
    out = mixer.out_proj(y.transpose(1, 2))
    return {
        "B": B,
        "C": C,
        "dt": dt,
        "discrete_A": discrete_A,
        "delta_b_u": delta_b_u,
        "y": y,
        "out": out,
        "x": x,
        "gate": gate,
    }


def std_repeat_mul(a, b, device):
    hidden = b.shape[-1]
    latent = a.shape[-1]
    at = to_dev(a, device)
    bt = to_dev(b, device)
    a_rep = ttnn.repeat(at, ttnn.Shape([1, 1, 1, hidden]))
    b_rep = ttnn.repeat_interleave(bt, latent, dim=3)

    def fn():
        return ttnn.multiply(a_rep, b_rep, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return fn


def load_mixer():
    """The Hub checkpoint is the state-spaces format, not a Transformers config."""
    from safetensors import safe_open
    from transformers import MambaConfig

    path = (
        "/home/andy/.cache/huggingface/hub/models--state-spaces--mamba-2.8b/"
        "snapshots/e886be8192cbb383b01559a3877dfd5e6bfb3e55/model.safetensors"
    )
    config = MambaConfig(
        vocab_size=50280,
        hidden_size=2560,
        state_size=16,
        num_hidden_layers=1,
        expand=2,
        conv_kernel=4,
        intermediate_size=5120,
        time_step_rank=160,
        use_bias=False,
        use_conv_bias=True,
        use_mambapy=False,
        hidden_act="silu",
    )
    mixer = MambaModel(config).layers[0].mixer
    weights = {}
    with safe_open(path, framework="pt") as handle:
        prefix = f"backbone.layers.{LAYER}.mixer."
        for key in handle.keys():
            if key.startswith(prefix):
                weights[key[len(prefix) :]] = handle.get_tensor(key)
    missing, unexpected = mixer.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"mixer load missing={missing} unexpected={unexpected}")
    return mixer.eval()


def main():
    torch.manual_seed(0)
    print(f"loading {MODEL_ID} layer {LAYER}", flush=True)
    mixer = load_mixer()
    hidden = torch.randn(1, SEQ, mixer.hidden_size)
    with torch.no_grad():
        reference = mixer.slow_forward(hidden)
        pieces = torch_pieces(mixer, hidden)
    print(f"pytorch slow_forward vs formulas pcc={pcc(pieces['out'], reference):.4f}", flush=True)

    device = ttnn.CreateDevice(0, l1_small_size=16384)
    print(
        f"arch={device.arch()} layer={LAYER} seq={SEQ} d_inner={mixer.intermediate_size} d_state={mixer.ssm_state_size}",
        flush=True,
    )
    try:
        # B * dt from the real layer, padded 16 -> 32, outer dim 32 (kernel constraint).
        b = pad_n(pieces["B"][:, 0, :], mixer.ssm_state_size).reshape(1, 1, 1, 32).repeat(1, 1, 32, 1)
        dt = pieces["dt"][:, :, 0].reshape(1, 1, 1, mixer.intermediate_size).repeat(1, 1, 32, 1)
        torch_bd = b.repeat(1, 1, 1, mixer.intermediate_size) * dt.repeat_interleave(32, dim=-1)
        fn = std_repeat_mul(b, dt, device)
        ms, out = timed(device, fn)
        report("std repeat + multiply (B*dt)", ttnn.to_torch(out), torch_bd, ms)

        at, bt = to_dev(b, device), to_dev(dt, device)

        def custom_rep():
            return ttnn.experimental.repeat_and_interleave_eltwise_mul(
                at, bt, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=DTYPE, math_fidelity=FID
            )

        ms, out = timed(device, custom_rep)
        report("custom repeat_and_interleave_eltwise_mul", ttnn.to_torch(out), torch_bd, ms)

        # Sum over the state axis. Real width is d_inner * 32.
        packed = torch.randn(1, 1, 32, mixer.intermediate_size * 32)
        expected_sum = packed.reshape(1, 1, 32, mixer.intermediate_size, 32).sum(-1)
        xt = to_dev(packed, device)
        shaped = ttnn.reshape(xt, (1, 1, 32, mixer.intermediate_size, 32))

        def std_sum():
            return ttnn.sum(shaped, dim=-1)

        ms, out = timed(device, std_sum)
        report("std reshape + sum", ttnn.to_torch(out), expected_sum, ms)

        def custom_sum():
            return ttnn.experimental.hc_sum_reduce(
                xt, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=DTYPE, math_fidelity=FID
            )

        ms, out = timed(device, custom_sum)
        report("custom hc_sum_reduce", ttnn.to_torch(out), expected_sum, ms)

        # Recurrence with multiply and add. Compare the sequence output y to PyTorch.
        e = mixer.intermediate_size
        discrete_a = pad_n(pieces["discrete_A"], mixer.ssm_state_size)
        delta_b_u = pad_n(pieces["delta_b_u"], mixer.ssm_state_size)
        c_steps = [
            to_dev(pad_n(pieces["C"][:, i, :], mixer.ssm_state_size).reshape(1, 1, 1, 32).repeat(1, 1, e, 1), device)
            for i in range(SEQ)
        ]
        a_steps = [to_dev(discrete_a[:, :, i, :].reshape(1, 1, e, 32), device) for i in range(SEQ)]
        bu_steps = [to_dev(delta_b_u[:, :, i, :].reshape(1, 1, e, 32), device) for i in range(SEQ)]
        zeros = to_dev(torch.zeros(1, 1, e, 32), device)

        def std_scan():
            h = zeros
            ys = []
            for i in range(SEQ):
                h = ttnn.multiply(a_steps[i], h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                h = ttnn.add(bu_steps[i], h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                proj = ttnn.multiply(h, c_steps[i], memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ys.append(ttnn.sum(proj, dim=-1))
            return ttnn.concat([ttnn.reshape(y, (1, 1, 1, e)) for y in ys], dim=2)

        ms, out = timed(device, std_scan, iters=1)
        got = ttnn.to_torch(out).float().reshape(1, SEQ, e).transpose(1, 2)
        # PyTorch y before the D residual and the gate. Padding zeros do not change the sum.
        state = torch.zeros(1, e, mixer.ssm_state_size)
        ys = []
        for i in range(SEQ):
            state = pieces["discrete_A"][:, :, i, :] * state + pieces["delta_b_u"][:, :, i, :]
            ys.append((state * pieces["C"][:, i, :].unsqueeze(1)).sum(-1))
        pre = torch.stack(ys, dim=-1)
        report("std mul/add scan y vs pytorch", got, pre, ms)

        # D residual, SiLU gate, output projection. This is the mixer output.
        d = to_dev(mixer.D.detach().reshape(1, 1, 1, e), device)
        x = to_dev(pieces["x"].detach().transpose(1, 2).reshape(1, 1, SEQ, e), device)
        gate = to_dev(torch.nn.functional.silu(pieces["gate"].detach()).transpose(1, 2).reshape(1, 1, SEQ, e), device)
        w_out = to_dev(mixer.out_proj.weight.detach().t().contiguous(), device)
        bias = (
            to_dev(mixer.out_proj.bias.detach().reshape(1, 1, 1, mixer.hidden_size), device)
            if mixer.out_proj.bias is not None
            else None
        )

        def std_out():
            y = std_scan()
            y = ttnn.add(
                y, ttnn.multiply(x, d, memory_config=ttnn.DRAM_MEMORY_CONFIG), memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            y = ttnn.multiply(y, gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            y = ttnn.reshape(y, (1, SEQ, e))
            return ttnn.linear(y, w_out, bias=bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        ms, out = timed(device, std_out, iters=1)
        report("std mixer out vs pytorch slow_forward", ttnn.to_torch(out), reference, ms)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
