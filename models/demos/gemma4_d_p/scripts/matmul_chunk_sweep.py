# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated per-device sweep of the Gemma4-31B prefill projections over chunk size (issue #59851).

Each projection runs on one Blackhole chip with one CP8/TP4 device's shapes: C / 8 token rows, the TP shard of the
weight (bfp8), and the production input/output layouts, program config and compute config. The matmul goes through the
model's own code (attention.operations.apply_attn_projection and MLP._project), so the configs follow it as C changes.

Run under the device profiler; every measured variant is bracketed by signposts "v-<tag>-start/stop":

    python -m tracy -r -p -o <out_dir> models/demos/gemma4_d_p/scripts/matmul_chunk_sweep.py --meta <out_dir>/meta.jsonl

then summarize with matmul_chunk_sweep_report.py. --attn-fidelity pins the attention projections' math fidelity (the
model picks HiFi2 below 512 rows per device and LoFi from there); the MLP is LoFi at every chunk where it has an
explicit config.
"""

import argparse
import json

import torch
from tracy import signpost

import ttnn
from models.demos.gemma4_d_p.tt.attention import operations
from models.demos.gemma4_d_p.tt.matmul_config import (
    is_short_m,
    short_m_output_memcfg,
    short_m_sharded_memcfg,
    to_l1_width_sharded,
)
from models.demos.gemma4_d_p.tt.mlp import MLP

CP = 8
HIDDEN = 5376
# (K, N) of one TP4 device's projection weight, from the HF config (hidden 5376, intermediate 21504, 32 Q heads,
# sliding: 16 KV heads x 256, global: 4 tied KV heads x 512).
OPS = {
    "sliding_qkv": (HIDDEN, (32 * 256 + 2 * 16 * 256) // 4),
    "global_qk": (HIDDEN, (32 * 512 + 4 * 512) // 4),
    "sliding_o": (32 * 256 // 4, HIDDEN),
    "global_o": (32 * 512 // 4, HIDDEN),
    "mlp_gate": (HIDDEN, 21504 // 4),
    "mlp_up": (HIDDEN, 21504 // 4),
    "mlp_down": (21504 // 4, HIDDEN),
}
DEFAULT_CHUNKS = [256 * i for i in range(1, 17)] + [5120, 6144, 7168, 8192, 10240, 12288, 16384, 24576, 32768]
FIDELITY = {"prod": None, "lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunks", type=int, nargs="+", default=DEFAULT_CHUNKS, help="chunk sizes C (tokens per request)")
    p.add_argument("--ops", nargs="+", default=list(OPS), choices=list(OPS))
    p.add_argument("--attn-fidelity", nargs="+", default=["prod"], choices=list(FIDELITY))
    p.add_argument("--reps", type=int, default=10, help="timed calls per variant (after 2 warm-up calls)")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--weights", help="torch file of real TP-shard weights {op: [K, N] tensor} (default: random)")
    p.add_argument(
        "--fallback-lofi",
        action="store_true",
        help="experiment: where the model has no explicit program config (ttnn's default), still use LoFi with fp32 "
        "accumulation and packer L1 accumulation, as the explicit configs do",
    )
    p.add_argument("--tag-prefix", default="", help="prefix of every variant tag, to tell runs apart")
    p.add_argument("--meta", required=True, help="jsonl written with one line per variant (shapes, configs, PCC)")
    return p.parse_args()


def describe(obj):
    """Plain-text form of a ttnn config object (None stays None)."""
    return None if obj is None else repr(obj)


def compute_config_fields(ckc):
    if ckc is None:
        return None
    return {f: str(getattr(ckc, f)) for f in ("math_fidelity", "fp32_dest_acc_en", "packer_l1_acc", "math_approx_mode")}


def lofi_fp32_config(device, fidelity=ttnn.MathFidelity.LoFi):
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )


class Sweep:
    def __init__(self, device, args):
        self.dev = device
        self.args = args
        self.meta = open(args.meta, "a")
        self.real = torch.load(args.weights) if args.weights else {}
        self._w = {}
        # MLP._project / _matmul_configs only read these attributes; the weights are built here instead.
        self.mlp = MLP.__new__(MLP)
        grid = device.compute_with_storage_grid_size()
        self.mlp.mesh_device = device
        self.mlp.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)
        self.mlp.compute_kernel_config = None  # MLP's value for bfp8 weights
        if args.fallback_lofi:
            self.mlp.compute_kernel_config = lofi_fp32_config(device)
        self.grid = grid

    def weight(self, op):
        if op not in self._w:
            k, n = OPS[op]
            t = self.real.get(op)
            if t is None:
                gen = torch.Generator().manual_seed(hash(op) % 2**31)
                t = torch.randn(k, n, generator=gen) / k**0.5
            assert tuple(t.shape) == (k, n), f"{op}: weight {tuple(t.shape)} != {(k, n)}"
            w = ttnn.from_torch(
                t.reshape(1, 1, k, n).float(),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # Reference against the dequantized weight so PCC measures the matmul, not bfp8 rounding.
            self._w[op] = (ttnn.to_torch(w).float().reshape(k, n), w)
        return self._w[op]

    def input(self, op, rows):
        """Torch values and the device tensor in the layout the model hands this projection."""
        k, _ = OPS[op]
        x = torch.randn(1, 1, rows, k).bfloat16().float()
        short = rows // ttnn.TILE_SIZE <= 8
        if op in ("sliding_qkv", "global_qk", "mlp_gate", "mlp_up"):
            # TP row all-gather output (layer._gather_rows): short-M sharded in L1, DRAM otherwise.
            memcfg = short_m_sharded_memcfg(self.dev, rows, k // ttnn.TILE_SIZE) if short else ttnn.DRAM_MEMORY_CONFIG
        elif op in ("sliding_o", "global_o"):
            memcfg = ttnn.DRAM_MEMORY_CONFIG  # nlp_concat_heads output
        else:
            memcfg = operations.prefill_short_lived_memcfg()  # gate * up output
        return x, ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev, memory_config=memcfg
        )

    def case(self, op, rows):
        """(prepare, run, configs): prepare() does the production layout change ahead of the matmul (timed under its
        own tag), run(x) is the matmul exactly as the model issues it."""
        _, w = self.weight(op)
        if op.startswith("mlp"):
            gelu = op == "mlp_gate"
            per_core_n = 4 if op == "mlp_down" else None

            if op == "mlp_down":
                prepare = (lambda x: to_l1_width_sharded(x)) if rows // 32 <= 8 else None
            else:
                prepare = None

            def run(x):
                if op == "mlp_down":
                    memcfg = ttnn.DRAM_MEMORY_CONFIG
                else:
                    memcfg = short_m_output_memcfg(x, w) if is_short_m(x) else operations.prefill_short_lived_memcfg()
                return self.mlp._project(x, w, memcfg, gelu=gelu, per_core_n=per_core_n)

            def configs(x):
                fused = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH) if gelu else None
                pc, ckc = self.mlp._matmul_configs(x, w, fused, per_core_n)
                return pc, ckc

            return prepare, run, configs

        prepare = (lambda x: to_l1_width_sharded(x)) if op.endswith("_o") and rows // 32 <= 8 else None
        return (
            prepare,
            (lambda x: operations.apply_attn_projection(x, w)),
            (lambda x: operations.projection_matmul_configs(x, w)),
        )

    def measure(self, op, chunk, fidelity):
        rows = chunk // CP
        k, n = OPS[op]
        tag = f"{self.args.tag_prefix}{op}-C{chunk}-{fidelity}"
        record = dict(tag=tag, op=op, chunk=chunk, rows=rows, k=k, n=n, attn_fidelity=fidelity)
        try:
            xt, x = self.input(op, rows)
            wt, _ = self.weight(op)
            record["padded_rows"] = x.padded_shape[-2]
            record["in_memcfg"] = describe(x.memory_config())
            prepare, run, configs = self.case(op, rows)
            xp = prepare(x) if prepare else x
            pc, ckc = configs(xp)
            record.update(program_config=describe(pc), compute_config=compute_config_fields(ckc))
            record["matmul_in_memcfg"] = describe(xp.memory_config())

            out = run(xp)
            record["out_memcfg"] = describe(out.memory_config())
            got = ttnn.to_torch(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG) if out.is_sharded() else out)
            ref = xt.reshape(rows, k) @ wt
            if op == "mlp_gate":
                ref = torch.nn.functional.gelu(ref, approximate="tanh")
            got = got.float().reshape(rows, n)
            record["pcc"] = float(torch.corrcoef(torch.stack([ref.flatten(), got.flatten()]))[0, 1])
            record["rel_rmse"] = float((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())
            out.deallocate(True)
            run(xp).deallocate(True)  # second warm-up
            ttnn.synchronize_device(self.dev)
            if prepare:
                signpost(f"v-{tag}-prepare-start")
                for _ in range(self.args.reps):
                    prepare(x).deallocate(True)
                ttnn.synchronize_device(self.dev)
                signpost(f"v-{tag}-prepare-stop")
            signpost(f"v-{tag}-start")
            for _ in range(self.args.reps):
                run(xp).deallocate(True)
            ttnn.synchronize_device(self.dev)
            signpost(f"v-{tag}-stop")
            if xp is not x:
                xp.deallocate(True)
            x.deallocate(True)
            record["status"] = "ok"
        except Exception as e:  # noqa: BLE001 - one failed point must not end the sweep
            record["status"] = f"error: {str(e).splitlines()[0][:300]}"
        ttnn.ReadDeviceProfiler(self.dev)
        self.meta.write(json.dumps(record) + "\n")
        self.meta.flush()
        print(f"{record['status'][:80]:80s} {tag} pcc={record.get('pcc', float('nan')):.5f}", flush=True)


def main():
    args = parse_args()
    device = ttnn.open_device(device_id=args.device_id)
    try:
        sweep = Sweep(device, args)
        original = operations.projection_math_fidelity
        if args.fallback_lofi:
            configs = operations.projection_matmul_configs

            def with_fallback(hidden_states, weight):
                pc, ckc = configs(hidden_states, weight)
                return (pc, ckc) if pc is not None else (None, lofi_fp32_config(device))

            operations.projection_matmul_configs = with_fallback
        for fidelity in args.attn_fidelity:
            pinned = FIDELITY[fidelity]
            operations.projection_math_fidelity = original if pinned is None else (lambda rows, f=pinned: f)
            for op in args.ops:
                if op.startswith("mlp") and fidelity != args.attn_fidelity[0]:
                    continue  # the MLP's fidelity does not follow --attn-fidelity
                for chunk in args.chunks:
                    sweep.measure(op, chunk, fidelity if not op.startswith("mlp") else "prod")
        operations.projection_math_fidelity = original
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
