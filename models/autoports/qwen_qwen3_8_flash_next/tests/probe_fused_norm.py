# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Dedicated RMSNorm candidate on real target normalization weights."""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import row_pcc, runtime_audit
from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import WeightLoader, rms
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import fused_rms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(782)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    weights = load_weights(args.snapshot, 0, "attn_hyper_connection.")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    rows = []
    try:
        loader = WeightLoader(weights, mesh)
        weight = loader.norm("attn_hyper_connection.hc_norm.weight", groups=4)
        for n in (1, 31, 32):
            x = ttnn.from_torch((torch.randn(1, 4, n, 2560) * 0.02).bfloat16(), layout=ttnn.TILE_LAYOUT, device=mesh)
            values = {}
            timings = {}
            for name, fn in [
                ("baseline", lambda: rms(x, weight, config.rms_norm_eps)),
                ("fused", lambda: fused_rms(x, weight, config.rms_norm_eps, loader.compute)),
            ]:
                with runtime_audit():
                    output = fn()
                values[name] = ttnn.to_torch(output)
                del output
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                output = fn()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                start = time.perf_counter()
                for _ in range(20):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                timings[name] = (time.perf_counter() - start) * 1000 / 20
                ttnn.release_trace(mesh, trace)
                del output
            pcc = row_pcc(values["baseline"], values["fused"]).min().item()
            rows.append(
                dict(
                    tokens=n,
                    min_row_pcc=pcc,
                    exact=torch.equal(values["baseline"], values["fused"]),
                    traced_host_ms=timings,
                )
            )
            print(rows[-1], flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    args.output.write_text(
        json.dumps(dict(rows=rows, passed=all(r["min_row_pcc"] >= 0.995 for r in rows)), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
