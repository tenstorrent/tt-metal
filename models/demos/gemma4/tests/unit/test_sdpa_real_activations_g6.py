# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""G6 developer harness (report only): the prefill SDPA op alone on REAL activations -- the chip-0 Q / K / V the model
handed the op (GEMMA4_SDPA_DUMP_DIR, written by tt/attention/prefill.py for the first sliding and the first global
layer) -- against a torch fp32 SDPA with the explicit causal (+ sliding window) mask, per position: band PCC / cliff
read, under (a) our program config, (b) the op defaults (no program config), (c) no sliding window (valid only when
the prompt does not reach the window). Isolates an op defect on this data from the model path around it.

    GEMMA4_SDPA_DUMP_DIR=<dir> pytest models/demos/gemma4/tests/unit/test_sdpa_real_activations_g6.py -k 1x1 -s
"""

import os

import pytest
import torch

import ttnn
from models.demos.gemma4.tt.attention.operations import prefill_sdpa_program_config

from ...tests.test_factory import parametrize_mesh_with_fabric


def _pcc(a, b):
    a = a.flatten().double() - a.flatten().double().mean()
    b = b.flatten().double() - b.flatten().double().mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def _torch_sdpa_fp32(q, k, v, window, scale):
    q, k, v = q.float(), k.float(), v.float()
    rep = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    S = q.shape[2]
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    idx = torch.arange(S)
    masked = idx.unsqueeze(0) > idx.unsqueeze(1)
    if window:
        masked |= idx.unsqueeze(0) < (idx.unsqueeze(1) - window + 1)
    scores = scores.masked_fill(masked, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def _cliff(out, ref):
    S = out.shape[2]
    per = [_pcc(out[0, :, i, :], ref[0, :, i, :]) for i in range(S)]
    bands = " ".join(
        f"[{s}:{min(s + 128, S)}) {sum(per[s:s + 128]) / len(per[s:s + 128]):.3f}" for s in range(0, S, 128)
    )
    first_bad = next((i for i in range(64, S - 16) if sum(per[i : i + 16]) / 16 < 0.9), None)
    return f"pcc_all {_pcc(out, ref):.4f} first_bad={first_bad} min {min(per):.3f}@{per.index(min(per))} bands {bands}"


def _arms(window, S, hd, nh):
    arms = [
        ("ours", prefill_sdpa_program_config(hd, S, sliding_window=window, num_heads=nh), window),
        ("opdefault", None, window),
    ]
    if (
        hd <= 256
    ):  # the 31B's wide default ((12,10) / q 128) and the 12B's former fixed config ((8,8) / q 256), explicitly
        kc = min(128, max(32, int(window) // 2)) if window else 128
        arms.append(
            (
                "wide12x10_q128",
                ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(12, 10),
                    q_chunk_size=128,
                    k_chunk_size=kc,
                    exp_approx_mode=False,
                ),
                window,
            )
        )
        arms.append(
            (
                "fixed8x8_q256",
                ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                    q_chunk_size=256,
                    k_chunk_size=kc,
                    exp_approx_mode=False,
                ),
                window,
            )
        )
    if window and S <= window:
        arms.append(("nowindow", prefill_sdpa_program_config(hd, S, sliding_window=None, num_heads=nh), None))
        arms.append(("nowindow_opdefault", None, None))
    return arms


def _run(mesh_device, path, ckc, label):
    d = torch.load(path, map_location="cpu")
    S, window, hd, scale = int(d["seq_len"]), d["sliding_window"], int(d["head_dim"]), float(d["scale"])
    out_path = path.replace("sdpa_in_", "sdpa_out_")
    inm = torch.load(out_path, map_location="cpu")["out"] if os.path.exists(out_path) else None
    if d["q"].dim() == 5:  # every chip's shard dumped: each chip's data through the op
        for c in range(d["q"].shape[0]):
            _run_one(
                mesh_device,
                d["q"][c],
                d["k"][c],
                d["v"][c],
                S,
                window,
                hd,
                scale,
                ckc,
                f"{label} chip={c}",
                inm[c].float() if inm is not None else None,
            )
    else:
        _run_one(
            mesh_device,
            d["q"],
            d["k"],
            d["v"],
            S,
            window,
            hd,
            scale,
            ckc,
            label,
            inm.float() if inm is not None else None,
        )
    return S, window, hd


def _run_one(mesh_device, q, k, v, S, window, hd, scale, ckc, label, inm):
    ref = _torch_sdpa_fp32(q, k, v, window, scale)
    to_dev = lambda t: ttnn.from_torch(t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    q_tt, k_tt, v_tt = to_dev(q), to_dev(k), to_dev(v)
    outs = {}
    for name, pc, win in _arms(window, S, hd, int(q.shape[1])):
        out = ttnn.transformer.scaled_dot_product_attention(
            q_tt,
            k_tt,
            v_tt,
            is_causal=True,
            scale=scale,
            sliding_window_size=win,
            program_config=pc,
            compute_kernel_config=ckc,
        )
        outs[name] = ttnn.to_torch(out).float()
        out.deallocate(True)
        cfg = (
            f"q{pc.q_chunk_size}/k{pc.k_chunk_size}/grid{pc.compute_with_storage_grid_size.x}x{pc.compute_with_storage_grid_size.y}"
            if pc
            else "op-defaults"
        )
        print(f"RESULT sdpa_real {label} arm={name} cfg={cfg} window={win} {_cliff(outs[name], ref)}")
        save_dir = os.environ.get("GEMMA4_SDPA_SAVE_OUT")  # per-arm outputs for an offline error-structure read
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {"out": outs[name].to(torch.bfloat16), "ref": ref},
                os.path.join(save_dir, f"{label.replace(' ', '_').replace('=', '')}_{name}.pt"),
            )
        err = (outs[name] - ref)[0]
        rel = " ".join(
            f"[{b}:{min(b + 128, S)}) {float(err[:, b:b + 128].norm() / ref[0, :, b:b + 128].norm()):.5f}"
            for b in range(0, S, 128)
        )
        print(f"RESULT sdpa_relerr {label} arm={name} cfg={cfg} rel_rms_per_band {rel}")
    for t in (q_tt, k_tt, v_tt):
        t.deallocate(True)
    if inm is not None:  # the in-model output of the same call (and chip)
        print(f"RESULT sdpa_real {label} arm=in_model_out vs_fp32 {_cliff(inm, ref)}")
        print(
            f"RESULT sdpa_real {label} arm=in_model_out vs_op_alone_ours pcc {_pcc(inm, outs['ours']):.4f} max|diff| {(inm - outs['ours']).abs().max():.4f}"
        )


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("kind", ["sliding", "global"])
def test_sdpa_real_activations(kind, mesh_device, reset_seeds, request):
    dump_dir = os.environ.get("GEMMA4_SDPA_DUMP_DIR")
    if not dump_dir or os.environ.get("CI") == "true":
        pytest.skip("developer harness: needs GEMMA4_SDPA_DUMP_DIR (the model path's SDPA input dump)")
    ckc = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    single = os.path.join(dump_dir, f"sdpa_in_{kind}.pt")
    numbered = sorted(
        p
        for p in os.listdir(dump_dir)
        if p.startswith("sdpa_in_") and p.endswith(f"_{kind}.pt") and p != f"sdpa_in_{kind}.pt"
    )
    if not numbered and not os.path.exists(single):
        pytest.skip(f"no {kind} dump in {dump_dir}")
    paths = [os.path.join(dump_dir, p) for p in numbered] or [single]
    for path in paths:
        label = f"kind={kind} call={os.path.basename(path)[8:11] if numbered else 'first'}"
        S, window, hd = _run(mesh_device, path, ckc, label)
    print(f"RESULT sdpa_real kind={kind} calls={len(paths)} S={S} window={window} hd={hd}")
