# SPDX-License-Identifier: Apache-2.0
"""One-chip check: chunked GDN prefill (gdn_prefill_chunk) vs the serial kernels on identical inputs.

Serial = the serving prefill recurrence (resident kernel for 128/256/512, broadcast local kernel
otherwise). Both consume the same raw conv-output q/k/v, gate inputs and initial state; a float64
torch recurrence is the arbiter. Also chains two calls to check the state carry across chunks.
Usage: python gdn_prefill_chunk_probe.py [--lengths 32,100,128,512] [--chain 512,100] [--time]
"""

import argparse
import glob
import json
import os
import time
from types import SimpleNamespace

import torch
from safetensors import safe_open

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt import gdn_prefill_chunk as chunk
from models.autoports.qwen_qwen3_8_flash_next.tt.gdn_prefill_resident import _resident_recurrence
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import _local_recurrence

p = argparse.ArgumentParser()
p.add_argument("--lengths", default="32,100,128,256,512")
p.add_argument("--chain", default="512,100")
p.add_argument("--layer", type=int, default=0)
p.add_argument("--time", action="store_true")
p.add_argument("--variant", default="default", help="default|horner|phased|phased_horner")
p.add_argument(
    "--bf16-ref", action="store_true", help="also score against a reference with bf16-rounded normalized q/k"
)
args = p.parse_args()


def layer_gates(layer):
    root = os.environ["QWEN_WEIGHTS"]
    index = json.load(open(glob.glob(f"{root}/model.safetensors.index.json")[0]))["weight_map"]
    out = {}
    for name in ("A_log", "dt_bias"):
        key = f"model.language_model.layers.{layer}.linear_attn.{name}"
        with safe_open(f"{root}/{index[key]}", "pt") as f:
            out[name] = f.get_tensor(key).float()[:12]  # chip 0's value heads
    return out


def reference(q, k, v, g, beta, state, bf16_qk=False):
    """float64 token recurrence; q/k raw [S,4,128] (l2-normalized here), v [S,12,128], g/beta [S,12]."""
    q, k, v, g, beta, state = (t.double() for t in (q, k, v, g, beta, state))
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() * 128**-0.5
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    if bf16_qk:
        q, k = q.bfloat16().double(), k.bfloat16().double()
    q, k = q.repeat_interleave(3, 1), k.repeat_interleave(3, 1)
    out = []
    for t in range(q.shape[0]):
        state = state * g[t].exp()[:, None, None]
        read = (state * k[t][:, :, None]).sum(-2)
        state = state + k[t][:, :, None] * ((v[t] - read) * beta[t][:, None])[:, None, :]
        out.append((state * q[t][:, :, None]).sum(-2))
    return torch.stack(out, 1), state  # [12,S,128], [12,128,128]


def main():
    torch.manual_seed(7)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    gates = layer_gates(args.layer)
    up = lambda t, dt=ttnn.float32: ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh)
    compute = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    a_neg = up(-gates["A_log"].exp().reshape(1, 1, 12))
    dt_bias = up(gates["dt_bias"].reshape(1, 1, 12))
    qscale, kscale = up(torch.full((1, 1, 1, 128), 1 / 128)), up(torch.full((1, 1, 1, 128), 128**-0.5))
    consts = chunk.const_tiles(SimpleNamespace(mesh=mesh, upload=lambda t, dt: up(t, dt)))
    fake = SimpleNamespace(
        device=mesh, recurrence_broadcast=True, prefill_qk_rm_signatures={}, prefill_resident_signatures={}
    )

    def heads(x, s, h):
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.to_layout(ttnn.reshape(x, [1, s, h, 128]), ttnn.TILE_LAYOUT)

    def inputs(s):
        raw = dict(
            q=torch.randn(1, s, 512).bfloat16(),
            k=torch.randn(1, s, 512).bfloat16(),
            v=torch.randn(1, s, 1536).bfloat16(),
            a=torch.randn(1, s, 12),
            b=torch.randn(1, s, 12),
        )
        raw["g"] = -gates["A_log"].exp() * torch.nn.functional.softplus(raw["a"] + gates["dt_bias"])
        raw["beta"] = torch.sigmoid(raw["b"].bfloat16()).float()
        return raw

    def device_gates(raw, log_space):
        a = up(raw["a"])
        g = ttnn.multiply(
            a_neg, ttnn.softplus(ttnn.add(a, dt_bias)), activations=[] if log_space else [ttnn.UnaryOpType.EXP]
        )
        return g, up(raw["beta"])

    def prepare(raw, log_space):
        s = raw["q"].shape[1]
        q, k, v = up(raw["q"], ttnn.bfloat16), up(raw["k"], ttnn.bfloat16), up(raw["v"], ttnn.bfloat16)
        g, beta = device_gates(raw, log_space)
        if log_space:
            return q, k, v, g, beta
        q = ttnn.rms_norm(
            ttnn.typecast(heads(q, s, 4), ttnn.float32),
            weight=qscale,
            epsilon=1e-06 / 128,
            compute_kernel_config=compute,
        )
        k = ttnn.rms_norm(
            ttnn.typecast(heads(k, s, 4), ttnn.float32),
            weight=kscale,
            epsilon=1e-06 / 128,
            compute_kernel_config=compute,
        )
        return q, k, ttnn.typecast(heads(v, s, 12), ttnn.float32), g, beta

    def serial(dev, state):
        fake.recurrent_state = state
        fn = _resident_recurrence if dev[0].shape[1] in (128, 256, 512) else _local_recurrence
        return fn(fake, *dev)

    def chunked(dev, state):
        options = {}
        if "horner" in args.variant:
            options["wy_inverse"] = ttnn.ChunkGdnWyInverse.HORNER
        if "phased" in args.variant:
            options["program_config"] = ttnn.ChunkGdnPhasedProgramConfig()
        o, final = chunk.chunk_recurrence(*dev, state, consts, **options)
        ttnn.copy(final, state)
        return o, state

    def run(seq, path):
        torch.manual_seed(11)
        state0 = torch.randn(1, 12, 128, 128) * 0.05
        state = up(state0)
        outs, ref_state, ref_outs, bref_state, bref_outs = [], state0[0].double(), [], state0[0].double(), []
        for s in seq:
            raw = inputs(s)
            o, st = path(prepare(raw, path is chunked), state)
            outs.append(ttnn.to_torch(o).float()[0])
            r, ref_state = reference(
                raw["q"][0].float().reshape(s, 4, 128),
                raw["k"][0].float().reshape(s, 4, 128),
                raw["v"][0].float().reshape(s, 12, 128),
                raw["g"][0],
                raw["beta"][0],
                ref_state,
            )
            ref_outs.append(r)
            if args.bf16_ref:
                r, bref_state = reference(
                    raw["q"][0].float().reshape(s, 4, 128),
                    raw["k"][0].float().reshape(s, 4, 128),
                    raw["v"][0].float().reshape(s, 12, 128),
                    raw["g"][0],
                    raw["beta"][0],
                    bref_state,
                    True,
                )
                bref_outs.append(r)
        run.bref = (torch.cat(bref_outs, 1), bref_state) if args.bf16_ref else None
        return torch.cat(outs, 1), ttnn.to_torch(st).float()[0], torch.cat(ref_outs, 1), ref_state

    def err(x, ref):
        d = (x.double() - ref).abs()
        return dict(max=float(d.max()), mean=float(d.mean()), rel=float(d.norm() / ref.norm()))

    cases = [[int(s)] for s in args.lengths.split(",") if s] + (
        [[int(s) for s in args.chain.split(",")]] if args.chain else []
    )
    try:
        for seq in cases:
            so, ss, ro, rs = run(seq, serial)
            co, cs, _, _ = run(seq, chunked)
            row = dict(
                seq=seq,
                serial_out=err(so, ro),
                chunk_out=err(co, ro),
                serial_state=err(ss, rs),
                chunk_state=err(cs, rs),
                chunk_vs_serial_out=float((co - so).abs().max()),
                chunk_vs_serial_state=float((cs - ss).abs().max()),
            )
            if run.bref is not None:
                row.update(chunk_out_vs_bf16qk=err(co, run.bref[0]), chunk_state_vs_bf16qk=err(cs, run.bref[1]))
            print("GDNCHUNK", json.dumps(row), flush=True)
            if args.time:
                for name, path in (("serial", serial), ("chunk", chunked)):
                    dev = prepare(inputs(seq[0]), path is chunked)
                    state = up(torch.zeros(1, 12, 128, 128))
                    path(dev, state)
                    ttnn.synchronize_device(mesh)
                    t = time.perf_counter()
                    for _ in range(5):
                        path(dev, state)
                    ttnn.synchronize_device(mesh)
                    print("GDNTIME", name, seq[0], f"{(time.perf_counter() - t) / 5 * 1e3:.2f} ms", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
