# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Optional TP4 prefill RoPE batching candidate; no TTNN import at module load.

Install on a constructed decoder before warmup. This preserves the original
method for B=1 and S=1 and folds only a multi-user, multi-token RoPE call.
Run with --cpu-check for a hardware-free transform check; otherwise arguments
are forwarded to the existing run_multichip_decoder harness.
"""

from types import MethodType


def batched_prefill_rope(layer, x, rope):
    """Rotate each user's own positions with one native prefill RoPE call.

    Input [B,H,S,D] and trig [B,1,S,64] become [1,H,B*T,64] and
    [1,1,B*T,64], where T=ceil(S/32)*32. Explicit per-user padding keeps
    tile reshape semantics unambiguous; no public alignment is required.
    """
    import ttnn

    batch, heads, seq, width = tuple(x.shape)
    if batch <= 1 or seq <= 1:
        return layer._prefill_rope_original(x, rope)
    if width < 64:
        raise ValueError(f"Qwen partial rotary width requires at least 64 channels: {x.shape}")
    cos, sin = rope
    expected = (batch, 1, seq, 64)
    if tuple(cos.shape) != expected or tuple(sin.shape) != expected:
        raise ValueError(f"RoPE trig must match each user's logical positions: expected {expected}")
    padded_seq = ((seq + 31) // 32) * 32

    def padded(t):
        t = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
        if seq != padded_seq:
            t = ttnn.pad(t, ((0, 0), (0, 0), (0, padded_seq - seq), (0, 0)), value=0.0)
        return t

    prefix = padded(x[:, :, :, :64])
    # H,B,T ordering repeats the complete B*T trig sequence once per head.
    prefix = ttnn.reshape(ttnn.permute(prefix, (1, 0, 2, 3)), (1, heads, batch * padded_seq, 64))
    trig = [ttnn.reshape(padded(t), (1, 1, batch * padded_seq, 64)) for t in (cos, sin)]
    rotated = ttnn.experimental.rotary_embedding_hf(
        prefix,
        *trig,
        compute_kernel_config=layer.ckc,
    )
    rotated = ttnn.permute(ttnn.reshape(rotated, (heads, batch, padded_seq, 64)), (1, 0, 2, 3))
    if seq != padded_seq:
        rotated = rotated[:, :, :seq, :]
    if width == 64:
        return rotated
    return ttnn.concat([rotated, x[:, :, :, 64:]], dim=-1)


def install(layer):
    """Install once on one full-attention decoder instance; return that instance."""
    if layer.kind != "full_attention":
        raise ValueError("The prefill RoPE candidate applies only to full_attention")
    if hasattr(layer, "_prefill_rope_original"):
        raise ValueError("The prefill RoPE candidate is already installed")
    layer._prefill_rope_original = layer._rope
    layer._rope = MethodType(batched_prefill_rope, layer)
    # Existing harness emits selected_policy, making the opt-in visible in JSON.
    layer.policy["batched_prefill_rope_candidate"] = True
    return layer


def cpu_check():
    """Exact index/pointwise checks, including per-user positions and padding.

    This validates the transform, not native TTNN reshape/permute or hardware
    numerics. It deliberately uses separate positions and dead-padding poison.
    """
    import json

    import torch
    import torch.nn.functional as F

    torch.set_num_threads(2)
    torch.manual_seed(731)
    cases = []
    for batch in (2, 3, 17, 32):
        for heads in (1, 6):
            for seq in (2, 31, 32, 33, 65, 513):
                x = torch.randn(batch, heads, seq, 256)
                positions = torch.arange(seq)[None, :] + 977 * torch.arange(batch)[:, None]
                angles = positions[:, None, :, None] / torch.arange(1, 33)[None, None, None, :]
                angles = angles.repeat(1, 1, 1, 2)
                cos, sin = angles.cos(), angles.sin()

                def rotate(t, c, s):
                    return t * c + torch.cat((-t[..., 32:], t[..., :32]), -1) * s

                expected = rotate(x[..., :64], cos, sin)
                padded_seq = ((seq + 31) // 32) * 32
                # NaNs in owned dead lanes must not appear in any retained row.
                prefix = F.pad(x[..., :64], (0, 0, 0, padded_seq - seq), value=float("nan"))
                prefix = prefix.permute(1, 0, 2, 3).reshape(1, heads, batch * padded_seq, 64)
                trig = [
                    F.pad(t, (0, 0, 0, padded_seq - seq), value=float("nan")).reshape(1, 1, batch * padded_seq, 64)
                    for t in (cos, sin)
                ]
                actual = rotate(prefix, *trig).reshape(heads, batch, padded_seq, 64).permute(1, 0, 2, 3)
                actual = actual[:, :, :seq, :]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                combined = torch.cat([actual, x[..., 64:]], -1)
                assert torch.isfinite(combined).all()
                assert torch.equal(combined[..., 64:], x[..., 64:])
                cases.append({"batch": batch, "heads": heads, "seq": seq, "padded_seq": padded_seq})
    result = {"pass": True, "kind": "cpu_transform_only", "cases": len(cases), "coverage": cases}
    print(json.dumps(result, indent=2))
    return result


def native_probe(argv):
    """Parent-serialized native correctness probe; no benchmark timing claims."""
    import argparse
    import hashlib
    import json
    import os
    from pathlib import Path
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(description=native_probe.__doc__)
    parser.add_argument("--cases", nargs="+", default=["2x3", "3x33", "17x31", "32x33", "3x513"])
    parser.add_argument("--heads", nargs="+", type=int, default=[1, 6])
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cases = [tuple(map(int, value.split("x"))) for value in args.cases]
    if any(len(case) != 2 or not 2 <= case[0] <= 32 or case[1] < 2 for case in cases):
        raise ValueError("Probe cases must be BxS with 2<=B<=32, S>=2")
    if args.steps < 2 or any(h not in (1, 6) for h in args.heads):
        raise ValueError("Require at least two changed-input steps and local heads 1 or 6")
    import torch
    import ttnn
    from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc

    torch.set_num_threads(4)
    report = {
        "status": "running",
        "pass": False,
        "mesh": [1, 4],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "watcher": os.environ.get("TT_METAL_WATCHER", "0"),
        "profiler": os.environ.get("TT_METAL_DEVICE_PROFILER", "0"),
        "cases": [],
        "positions": "distinct by rank, user and step",
        "dead_padding": "NaN poison",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024)
    try:
        ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        def baseline(x, rope):
            rotated = [
                ttnn.experimental.rotary_embedding_hf(
                    x[row : row + 1, :, :, :64],
                    rope[0][row : row + 1],
                    rope[1][row : row + 1],
                    compute_kernel_config=ckc,
                )
                for row in range(x.shape[0])
            ]
            return ttnn.concat([ttnn.concat(rotated, dim=0), x[:, :, :, 64:]], dim=-1)

        layer = SimpleNamespace(kind="full_attention", policy={}, ckc=ckc, _rope=baseline)
        install(layer)

        def read(t):
            return [ttnn.to_torch(v).contiguous() for v in ttnn.get_device_tensors(t)]

        def upload(values, seq):
            tensor = ttnn.from_torch(
                torch.cat(values, dim=-1).contiguous(),
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            )
            # Retain logical S while deliberately putting NaNs into the backing
            # padded sequence lanes. The candidate must own zero-padding them.
            return tensor[:, :, :seq, :]

        def compare(actual, expected, batch):
            records = []
            for rank, (a, b) in enumerate(zip(actual, expected)):
                for user in range(batch):
                    records.append(
                        {
                            "rank": rank,
                            "user": user,
                            "pcc": pcc(a[user], b[user]),
                            "finite": bool(torch.isfinite(a[user]).all() and torch.isfinite(b[user]).all()),
                            "bit_equal": torch.equal(a[user].view(torch.int16), b[user].view(torch.int16)),
                        }
                    )
            assert all(r["finite"] and r["pcc"] >= 0.995 for r in records), records
            return records

        for batch, seq in cases:
            for heads in args.heads:
                entry = {"batch": batch, "seq": seq, "heads": heads, "eager": [], "trace": [], "pass": False}
                report["cases"].append(entry)
                save()
                bank, reference = [], []
                padded_seq = (seq + 31) // 32 * 32
                for step in range(args.steps):
                    xs, cs, ss = [], [], []
                    for rank in range(4):
                        generator = torch.Generator().manual_seed(731 + 97 * rank + 7 * step + batch + seq)
                        x = torch.randn(batch, heads, seq, 256, generator=generator).to(torch.bfloat16)
                        positions = (
                            torch.arange(seq)[None, :] + 977 * torch.arange(batch)[:, None] + rank * 137 + step * 59
                        )
                        angles = positions[:, None, :, None] / torch.arange(1, 33)[None, None, None, :]
                        angles = angles.repeat(1, 1, 1, 2)
                        pad = (0, 0, 0, padded_seq - seq)
                        xs.append(torch.nn.functional.pad(x, pad, value=float("nan")))
                        cs.append(torch.nn.functional.pad(angles.cos().to(torch.bfloat16), pad, value=float("nan")))
                        ss.append(torch.nn.functional.pad(angles.sin().to(torch.bfloat16), pad, value=float("nan")))
                    bank.append(tuple(upload(values, seq) for values in (xs, cs, ss)))
                inputs = [ttnn.empty_like(t) for t in bank[0]]

                def set_input(step, bank, inputs):
                    for src, dst in zip(bank[step], inputs):
                        ttnn.copy(src, dst)

                for step in range(args.steps):
                    set_input(step, bank, inputs)
                    expected = read(baseline(inputs[0], inputs[1:]))
                    actual = read(layer._rope(inputs[0], inputs[1:]))
                    reference.append(expected)
                    entry["eager"].append({"step": step, "checks": compare(actual, expected, batch)})
                    save()
                set_input(0, bank, inputs)
                warm = layer._rope(inputs[0], inputs[1:])
                del warm
                ttnn.synchronize_device(mesh)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                traced = layer._rope(inputs[0], inputs[1:])
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                try:
                    for step in range(args.steps):
                        set_input(step, bank, inputs)
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                        entry["trace"].append({"step": step, "checks": compare(read(traced), reference[step], batch)})
                        save()
                finally:
                    ttnn.release_trace(mesh, tid)
                entry["pass"] = True
                save()
                print(f"ROPE_PROBE pass B{batch} S{seq} H{heads}", flush=True)
                del traced, inputs, bank, reference
        report["pass"] = True
        report["status"] = "pass"
        save()
    except Exception as exc:
        report["error"] = repr(exc)
        report["status"] = "failed"
        save()
        raise
    finally:
        ttnn.close_mesh_device(mesh)


def main():
    import hashlib
    import sys
    from pathlib import Path

    if sys.argv[1:2] == ["--probe"]:
        native_probe(sys.argv[2:])
        return
    if sys.argv[1:] == ["--cpu-check"]:
        cpu_check()
        return
    if "--baseline" in sys.argv:
        raise ValueError("The candidate CLI measures TP4 only; use the existing baseline fixture")
    from models.autoports.qwen_qwen3_6_27b.tests import run_multichip_decoder as runner

    class CandidateDecoder(runner.MultichipDecoder):
        @classmethod
        def from_state_dict(cls, *args, **kwargs):
            layer = install(super().from_state_dict(*args, **kwargs))
            layer.policy["batched_prefill_rope_candidate_sha256"] = hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest()
            return layer

    # Only this process's harness binding changes. No persistent source or
    # globally imported MultichipDecoder class is mutated.
    runner.MultichipDecoder = CandidateDecoder
    runner.main()


if __name__ == "__main__":
    main()
