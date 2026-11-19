# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU proof of double-single cumulative decay; no device import or execution."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

E = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def padded_chunks(values):
    n = values.shape[-2]
    padded = torch.nn.functional.pad(values, (0, 0, 0, (-n) % 64))
    return padded.reshape(*values.shape[:-2], -1, 64, values.shape[-1])


def double_single_cpu(values):
    """Full TwoSum residual, FP32 low accumulation, FP32 visible prefixes."""
    chunks = padded_chunks(values)
    hi = torch.zeros_like(chunks[..., 0, :])
    lo = torch.zeros_like(hi)
    output = []
    for i in range(64):
        x = chunks[..., i, :]
        s = hi + x
        b = s - hi
        e = (hi - (s - b)) + (x - b)
        lo = lo + e
        output.append(s + lo)
        hi = s
    return torch.stack(output, dim=-2)


def ttnn_double_single_chunk_cumsum(t, values, *, mesh_device, memory_config):
    """Device-only candidate, restarting at each internal chunk64 boundary.

    No Torch/host conversion and no unsupported FP32 ADD/SUB flags. Returns the
    internally padded length so callers can inspect the final chunk's g_last.
    CPU finite-fixture parity is not a universal floating-point proof.
    """
    n = values.shape[-2]
    assert n > 0 and values.dtype == t.float32
    pad = (-n) % 64
    if pad:
        values = t.pad(values, [(0, 0), (0, 0), (0, pad), (0, 0)], 0.0, memory_config=memory_config)
    shape = (*tuple(values.shape)[:-2], 1, values.shape[-1])
    zero = t.zeros(shape, dtype=t.float32, layout=t.TILE_LAYOUT, device=mesh_device, memory_config=memory_config)
    output = []

    def add(a, b):
        return t.add(a, b, dtype=t.float32, memory_config=memory_config)

    def sub(a, b):
        return t.subtract(a, b, dtype=t.float32, memory_config=memory_config)

    for start in range(0, n, 64):
        hi = lo = zero
        for i in range(start, start + 64):
            x = values[..., i : i + 1, :]
            s = add(hi, x)
            b = sub(s, hi)
            e = add(sub(hi, sub(s, b)), sub(x, b))
            lo = add(lo, e)
            output.append(add(s, lo))
            hi = s
    return t.concat(output, dim=-2, memory_config=memory_config)


def compare(values):
    chunks = padded_chunks(values)
    expected = chunks.double().cumsum(-2).float()
    hf = chunks.cumsum(-2)
    assert torch.equal(hf, expected), "The tested Torch CPU cumsum must equal FP64-accumulate/FP32-output"
    got = double_single_cpu(values)
    ec = expected.contiguous().view(torch.int32)
    gc = got.contiguous().view(torch.int32)
    mismatch = ec != gc
    positions = mismatch.nonzero()
    examples = []
    for idx in positions[:8]:
        ix = tuple(idx.tolist())
        row = ix[:-2]
        token = ix[-2]
        channel = ix[-1]
        examples.append(
            dict(
                output_index=list(ix),
                expected=float(expected[ix]),
                actual=float(got[ix]),
                expected_code=int(ec[ix]),
                actual_code=int(gc[ix]),
                input_prefix_codes=chunks[row][..., : token + 1, channel].contiguous().view(torch.int32).tolist(),
            )
        )
    return dict(
        shape=list(values.shape),
        logical_length=values.shape[-2],
        padding=(-values.shape[-2]) % 64,
        compared_prefixes=expected.numel(),
        fp32_code_differences=int(mismatch.sum()),
        max_abs=float((expected - got).abs().max()),
        mismatch_examples=examples,
    )


def run(args):
    torch.set_num_threads(4)
    assert not args.output.exists(), "Preserve previous evidence"
    payload = torch.load(args.matrices, map_location="cpu", weights_only=False)
    actual = payload["cases"][128]["operands"]["g"]
    assert tuple(actual.shape) == (1, 32, 128, 1) and actual.dtype == torch.float32
    rows = []
    for n in (1, 2, 31, 32, 33, 63, 64, 65, 127, 128):
        row = compare(actual[..., :n, :])
        rows.append(dict(family="actual_prefix", **row))
    for seed in (0, 1, 42, 154, 913):
        generator = torch.Generator().manual_seed(seed)
        for n in (1, 31, 33, 63, 64, 65, 127, 128, 129, 257):
            shape = (2, 32, n, 1)
            samples = {
                "negative_uniform": -torch.rand(shape, generator=generator) * 12,
                "negative_logspread": -torch.exp(torch.rand(shape, generator=generator) * 14 - 9),
                "negative_softplus_like": -torch.nn.functional.softplus(torch.randn(shape, generator=generator) * 3)
                * torch.exp(torch.randn((2, 32, 1, 1), generator=generator)),
            }
            for family, values in samples.items():
                assert torch.isfinite(values).all()
                rows.append(dict(family=family, seed=seed, **compare(values)))
    result = dict(
        classification="CPU finite-fixture precision proof; device not run",
        acceptance_eligible=False,
        status="cpu_pass" if all(r["fp32_code_differences"] == 0 for r in rows) else "cpu_mismatch",
        script_sha256=sha(__file__),
        input_matrices=str(args.matrices),
        input_matrices_sha256=sha(args.matrices),
        actual_g_sha256=hashlib.sha256(actual.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
        algorithm="s=hi+x; b=s-hi; e=(hi-(s-b))+(x-b); lo+=e; output=s+lo; hi=s; all intermediates FP32",
        target="Torch FP32 cumsum, independently matched to FP64 cumsum rounded to FP32 at every prefix",
        universal_exactness_claim=False,
        device_execution=False,
        torch_version=torch.__version__,
        compared_prefixes=sum(r["compared_prefixes"] for r in rows),
        fp32_code_differences=sum(r["fp32_code_differences"] for r in rows),
        cases=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = Path(str(args.output) + ".source.py.snapshot")
    snapshot.write_bytes(Path(__file__).read_bytes())
    result.update(source_snapshot=str(snapshot), source_snapshot_sha256=sha(snapshot))
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

    record(args.output)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    assert "ttnn" not in sys.modules
    print(json.dumps({k: result[k] for k in ("status", "compared_prefixes", "fp32_code_differences")}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrices", type=Path, default=E / "v6_chunk64_state_cpu_setup_final.json.cpu_matrices.pt")
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
