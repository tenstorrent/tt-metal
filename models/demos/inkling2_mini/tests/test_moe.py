# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Inkling-2-Mini MoE block on device vs HF ``Inkling2SparseMoeBlock`` (random weights, batch-1 decode).

Reference: transformers >= 5.24. If it is not importable here, the reference runs in a subprocess with
``$INKLING_REF_PYTHON`` (default /work/inkling-bringup/ref-venv/bin/python), which executes this file with
``--ref in.pt out.pt`` (no ttnn needed on that side).

Two references per expert dtype: A = the bf16 weights (truth), B = the bf16 of the BFP4/BFP8 round-trip.
Gated on B (kernel error only, review r3 #11); A (adds weight quantization error) is reported.
The input is the layer state before post_attention_layernorm (mlp_norm), which this block owns.

    flock /tmp/tt-card.lock pytest -s models/demos/inkling2_mini/tests/test_moe.py
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import time

import torch

try:
    import pytest

    import ttnn
except ImportError:  # the --ref side runs without ttnn
    ttnn = None

HIDDEN, INTER, NUM_EXPERTS, TOP_K = 2304, 1152, 128, 8
N_TOKENS = 16
REF_SOURCES = "/work/inkling-bringup/ref-sources"
REF_PYTHON = os.environ.get("INKLING_REF_PYTHON", "/work/inkling-bringup/ref-venv/bin/python")
TMP_DIR = os.environ.get("INKLING_T4_TMP", "/work/inkling-bringup/tmp-T4")
RESULTS = os.environ.get("INKLING_T4_RESULTS")  # optional JSON path for the measured numbers


# ---------------------------------------------------------------- HF reference (both sides)
def _hf_block():
    from transformers.models.inkling2.configuration_inkling2 import Inkling2TextConfig
    from transformers.models.inkling2.modeling_inkling2 import Inkling2SparseMoeBlock

    with open(os.path.join(REF_SOURCES, "config.json")) as f:
        tc = json.load(f)["text_config"]
    keys = ("hidden_size", "n_routed_experts", "num_experts_per_tok", "intermediate_size")
    keys += ("mlp_gate_softcap", "mlp_up_softcap", "route_scale", "n_shared_experts")
    config = Inkling2TextConfig(**{k: tc[k] for k in keys})
    with torch.device("meta"):
        block = Inkling2SparseMoeBlock(config)
    return block


def _hf_norm(norm_w):
    from transformers.models.inkling2.modeling_inkling2 import Inkling2RMSNorm

    with open(os.path.join(REF_SOURCES, "config.json")) as f:
        eps = json.load(f)["text_config"]["rms_norm_eps"]
    norm = Inkling2RMSNorm(HIDDEN, eps)
    norm.weight.data = norm_w.bfloat16()  # the model's norm weight is bf16
    return norm


def _hf_run(h, norm_w, router_w, bias, variants):
    """h [N,H] layer state; HF decoder: mlp(post_attention_layernorm(h).to(bf16)).
    variants {name: (gate_up [E,2I,H], down [E,H,I])} -> {name: out [N,H]}, indices [N,k]."""
    block = _hf_block()
    with torch.no_grad():
        x = _hf_norm(norm_w)(h).bfloat16()
    outs, indices = {}, None
    for name, (gate_up, down) in variants.items():
        state = {
            "gate.weight": router_w.float(),
            "gate.e_score_correction_bias": bias.float(),
            "experts.gate_up_proj": gate_up.bfloat16(),
            "experts.down_proj": down.bfloat16(),
        }
        block.load_state_dict(state, assign=True, strict=True)
        with torch.no_grad():
            outs[name] = block(x.unsqueeze(0))[0].float()
            _, _, idx = block.gate(x)
        indices = idx
    return outs, indices


def _reference(h, norm_w, router_w, bias, variants):
    try:
        import transformers.models.inkling2  # noqa: F401

        return _hf_run(h, norm_w, router_w, bias, variants)
    except ImportError:
        pass
    os.makedirs(TMP_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=TMP_DIR) as tmp:
        src, dst = os.path.join(tmp, "in.pt"), os.path.join(tmp, "out.pt")
        torch.save({"h": h, "norm_w": norm_w, "router_w": router_w, "bias": bias, "variants": variants}, src)
        subprocess.run([REF_PYTHON, "-I", os.path.abspath(__file__), "--ref", src, dst], check=True)
        res = torch.load(dst)
    return res["outs"], res["indices"]


# ---------------------------------------------------------------- inputs
def _random_inputs(seed=0):
    """Scales chosen so the router spread, both softcaps and the expert sum are all exercised."""
    g = torch.Generator().manual_seed(seed)
    h = (3.0 * torch.randn(N_TOKENS, HIDDEN, generator=g)).bfloat16()  # layer state, pre-norm
    norm_w = 1.0 + 0.1 * torch.randn(HIDDEN, generator=g)
    router_w = torch.randn(NUM_EXPERTS, HIDDEN, generator=g) / math.sqrt(HIDDEN)
    bias = 0.05 * torch.randn(NUM_EXPERTS, generator=g)
    gate = torch.randn(NUM_EXPERTS, INTER, HIDDEN, generator=g) * (3.0 / math.sqrt(HIDDEN))
    up = torch.randn(NUM_EXPERTS, INTER, HIDDEN, generator=g) * (8.0 / math.sqrt(HIDDEN))
    gate_up = torch.cat([gate, up], dim=1).bfloat16()
    del gate, up
    down = (torch.randn(NUM_EXPERTS, HIDDEN, INTER, generator=g) / math.sqrt(INTER)).bfloat16()
    return h, norm_w, router_w, bias, gate_up, down


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _to_dev(x_row, device):
    return ttnn.from_torch(x_row.reshape(1, 1, 1, HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


BYTES_PER_PARAM = {"bfp4": 576 / 1024, "bfp8": 1088 / 1024}

if ttnn is not None:

    @pytest.mark.parametrize("device_params", [{"trace_region_size": 64 * 1024 * 1024}], indirect=True)
    @pytest.mark.parametrize("dtype_name", ["bfp4", "bfp8"])
    def test_moe(device, dtype_name):
        from models.demos.inkling2_mini.tt.moe import Inkling2MoE, prepare_moe_weights

        expert_dtype = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}[dtype_name]
        x, norm_w, router_w, bias, gate_up, down = _random_inputs()
        weights = prepare_moe_weights(norm_w, router_w, bias, gate_up, down, expert_dtype)
        # B: the device weights read back (same quantized values the kernels see), in HF layout
        gate_up_b = ttnn.to_torch(weights["experts.w_gate_up"])[0].transpose(1, 2).bfloat16()
        down_b = ttnn.to_torch(weights["experts.w_down"])[0].transpose(1, 2).bfloat16()
        refs, ref_idx = _reference(x, norm_w, router_w, bias, {"A": (gate_up, down), "B": (gate_up_b, down_b)})
        del gate_up_b, down_b

        moe = Inkling2MoE(device, weights)
        del weights

        outs, overlaps = [], []
        for t in range(N_TOKENS):
            out, idx = moe(_to_dev(x[t], device))
            outs.append(ttnn.to_torch(out).reshape(-1)[:HIDDEN].float())
            got = set(ttnn.to_torch(idx).reshape(-1)[:TOP_K].long().tolist())
            overlaps.append(len(got & set(ref_idx[t].tolist())) / TOP_K)
        out_tt = torch.stack(outs)
        assert torch.isfinite(out_tt).all()
        pcc_a, pcc_b = _pcc(out_tt, refs["A"]), _pcc(out_tt, refs["B"])
        overlap = sum(overlaps) / len(overlaps)

        # Trace: one captured block, replayed twice on the same input -> identical outputs, equal to eager.
        x_dev = _to_dev(x[0], device)
        moe(x_dev)  # warm the program cache for this input tensor
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        out_trace, _ = moe(x_dev)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        run1 = ttnn.to_torch(out_trace).reshape(-1)[:HIDDEN].float()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
        run2 = ttnn.to_torch(out_trace).reshape(-1)[:HIDDEN].float()
        trace_same = torch.equal(run1, run2)
        trace_vs_eager = _pcc(run1, outs[0])
        ttnn.release_trace(device, tid)

        block_us = _time_traced(device, lambda: moe(x_dev))
        # Expert step alone (2 indexed sparse_matmuls + SwiGLU + combine), routing fixed.
        x_norm = moe.norm(x_dev)
        weights_k, ids_k = moe.route(x_norm)
        expert_us = _time_traced(device, lambda: moe.experts(x_norm, weights_k, ids_k))
        mm_us = _time_traced(device, lambda: moe.expert_matmuls(x_norm, ids_k))
        mm_gbps = TOP_K * (HIDDEN * 2 * INTER + INTER * HIDDEN) * BYTES_PER_PARAM[dtype_name] / (mm_us * 1e-6) / 1e9
        expert_bytes = TOP_K * (HIDDEN * 2 * INTER + INTER * HIDDEN) * BYTES_PER_PARAM[dtype_name]
        gbps = expert_bytes / (expert_us * 1e-6) / 1e9

        res = {
            "dtype": dtype_name,
            "pcc_vs_A_bf16": round(pcc_a, 6),
            "pcc_vs_B_roundtrip": round(pcc_b, 6),
            "top8_overlap": round(overlap, 4),
            "trace_same_twice": trace_same,
            "trace_vs_eager_pcc": round(trace_vs_eager, 6),
            "block_us": round(block_us, 1),
            "expert_step_us": round(expert_us, 1),
            "expert_MB": round(expert_bytes / 1e6, 2),
            "expert_GBps": round(gbps, 1),
            "sparse_matmuls_us": round(mm_us, 1),
            "sparse_matmuls_GBps": round(mm_gbps, 1),
            "gate_up_grid": str(moe.gate_up_config.compute_with_storage_grid_size),
            "down_grid": str(moe.down_config.compute_with_storage_grid_size),
        }
        print("T4-RESULT", json.dumps(res))
        if RESULTS:
            with open(RESULTS, "a") as f:
                f.write(json.dumps(res) + "\n")

        # Gate (review r3 #11): kernel error only, vs HF on the same round-tripped weights. pcc_vs_A (weight
        # quantization included) is reported, not gated.
        assert overlap >= 0.97, res
        assert pcc_b >= 0.999, res
        assert trace_same and trace_vs_eager > 0.9999, res


def _time_traced(device, fn, repeats=20, iters=20):
    """Device time per call of ``fn``: one trace holds ``repeats`` calls, so the ~50 us per-replay launch
    cost (PCIe x4 VM) is amortised; mean over ``iters`` replays."""
    fn()
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(repeats):
        fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    for _ in range(2):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(iters):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    ttnn.release_trace(device, tid)
    return (time.perf_counter() - start) / iters / repeats * 1e6


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--ref":
    inp = torch.load(sys.argv[2])
    outs, indices = _hf_run(inp["h"], inp["norm_w"], inp["router_w"], inp["bias"], inp["variants"])
    torch.save({"outs": outs, "indices": indices}, sys.argv[3])
