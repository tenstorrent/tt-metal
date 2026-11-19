# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unmeasured AutoFix localization of native RMSNorm's discrete amplification.

Host readbacks deliberately instrument router inputs/logits and actual selected IDs.
This is diagnostic evidence, never runtime-audit or performance evidence.
"""
import argparse
import copy
import json
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import measure, row_pcc
from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedGatedResidual
from models.autoports.qwen_qwen3_8_flash_next.tt.router_topk import aten_router_topk


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(2071)
    cfg = AutoConfig.from_pretrained(a.snapshot, local_files_only=True).text_config
    weights = load_weights(a.snapshot, 0, "")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    rows = []
    try:
        base = FunctionalDecoder.from_state_dict(weights, hf_config=cfg, layer_idx=0, mesh_device=mesh)
        state = base.create_state()
        saved = [
            ttnn.from_torch(ttnn.to_torch(x), dtype=x.dtype, layout=x.layout) for x in (state.conv, state.recurrent)
        ]
        x = (torch.randn(1, 1, 33, 10240) * 0.02).bfloat16()
        device = ttnn.from_torch(x, device=mesh, layout=ttnn.TILE_LAYOUT)
        baseline = None
        for name, sites, merged in [
            ("baseline", (), False),
            ("attn_only", ("attn_mix",), False),
            ("mlp_only", ("mlp_mix",), False),
            ("both", ("attn_mix", "mlp_mix"), False),
            ("merged", ("attn_mix", "mlp_mix"), True),
        ]:
            model = copy.copy(base)
            for site in sites:
                gr = FusedGatedResidual.__new__(FusedGatedResidual)
                gr.__dict__.update(getattr(base, site).__dict__)
                gr.merge_rsqrt = merged
                setattr(model, site, gr)
            model.mlp = copy.copy(base.mlp)
            capture = []
            original = model.mlp.router_logits

            def router(value):
                logits = original(value)
                scores, indices = aten_router_topk(logits)
                capture.append(
                    (ttnn.to_torch(value).clone(), ttnn.to_torch(logits).clone(), ttnn.to_torch(indices).clone())
                )
                return logits

            model.mlp.router_logits = router
            for host, target in zip(saved, (state.conv, state.recurrent)):
                ttnn.copy_host_to_device_tensor(host, target)
            out = model.prefill_forward(device, state=state)
            actual = ttnn.to_torch(out)
            del out
            inputs = torch.cat([v[0] for v in capture], dim=2)
            logits = torch.cat([v[1] for v in capture], dim=2)
            ids = torch.cat([v[2] for v in capture], dim=2).long()
            if baseline is None:
                baseline = (actual, inputs, logits, ids)
            changed = (ids.sort(dim=-1).values != baseline[3].sort(dim=-1).values).any(-1).flatten()
            row = dict(
                variant=name,
                metrics=measure(baseline[0], actual, x),
                router_input_min_row_pcc=float(row_pcc(baseline[1], inputs).min()),
                changed_membership_rows=changed.nonzero().flatten().tolist(),
                row8_ids=ids[0, 0, 8].tolist(),
                row8_logits_max_abs_error=float((logits[0, 0, 8] - baseline[2][0, 0, 8]).abs().max()),
            )
            rows.append(row)
            print(name, row["metrics"]["contribution_min_row_pcc"], row["changed_membership_rows"], flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    a.output.write_text(json.dumps(dict(scope=__doc__, rows=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
