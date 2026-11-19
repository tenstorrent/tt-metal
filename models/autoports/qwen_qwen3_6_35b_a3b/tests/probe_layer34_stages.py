# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Eager layer-34 diagnosis with deliberate host reads between decoder operations.

This is neither a latency benchmark nor a runtime-fallback audit. Instance-bound
hooks copy tensors to CPU only at selected steps; the decoder consumes its original
device tensors. CPU sums are diagnostic reconstructions, never model inputs.

Run serially under an external bound, for example:
    timeout --signal=TERM --kill-after=30s 1800s python -m \
      models.autoports.qwen_qwen3_6_35b_a3b.tests.probe_layer34_stages \
      --output /tmp/layer34_stages_fresh

The output directory must not exist. Baseline evidence is saved before opening the
four-device mesh, so a subsequent failure does not discard the first run.
"""

import argparse
import json
import math
import types
from pathlib import Path

import torch

CAPTURE_STEPS = (0, 18, 41, 53, 70, 77)
DECODE_STEPS = 78
ROOT = Path(__file__).resolve().parents[1]
STAGE_ORDER = (
    "input_norm.input",
    "input_norm.output",
    "out_proj.input",
    "out_proj.output",
    "out_proj.reduced",
    "post_norm.input",
    "post_norm.output",
    "moe.input",
    "router.output",
    "moe.output",
    "moe.reduced",
    "final.output",
    "state.conv",
    "state.recurrent",
)


def _bitexact(a, b):
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    )


def _metrics(a, b):
    result = {"baseline_shape": list(a.shape), "mesh_shape": list(b.shape), "bitexact": _bitexact(a, b)}
    if a.shape != b.shape:
        return result
    x, y = a.double().flatten(), b.double().flatten()
    finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
    corr = None
    if finite:
        if torch.equal(x, y):
            corr = 1.0
        else:
            xc, yc = x - x.mean(), y - y.mean()
            denominator = xc.norm() * yc.norm()
            if denominator:
                value = (xc @ yc / denominator).item()
                corr = value if math.isfinite(value) else None
    result.update(
        baseline_dtype=str(a.dtype),
        mesh_dtype=str(b.dtype),
        all_finite=finite,
        pcc=corr,
        max_abs=(x - y).abs().max().item() if finite else None,
        rounded_to_baseline_dtype_bitexact=_bitexact(a, b.to(a.dtype)),
    )
    return result


def _compose(parts, mode):
    if mode == "replicated":
        return parts[0]
    if mode == "concat":
        return torch.cat(parts, dim=-1)
    if mode == "sum_fp32":
        return torch.stack([part.float() for part in parts], dim=0).sum(dim=0)
    if mode == "recurrent":
        return torch.cat(parts, dim=1)
    if mode == "conv":
        # Each TP rank stores [Q_rank(512), K_rank(512), V_rank(1024)].
        blocks = [part.split((512, 512, 1024), dim=-1) for part in parts]
        return torch.cat([torch.cat([block[k] for block in blocks], dim=-1) for k in range(3)], dim=-1)
    raise ValueError(mode)


class Recorder:
    def __init__(self, ttnn, multichip):
        self.ttnn = ttnn
        self.multichip = multichip
        self.step = None
        self.reductions = 0
        self.tensors = {}
        self.metadata = {}

    def activate(self, step):
        self.step = str(step)
        self.reductions = 0
        self.tensors[self.step] = {}
        self.metadata[self.step] = {}

    def capture(self, name, tensor, mode="replicated"):
        if self.step is None:
            return
        if name in self.tensors[self.step]:
            raise RuntimeError(f"duplicate diagnostic stage {self.step}/{name}")
        # Host collection is intentionally eager and outside every timed/audited
        # path. Conversion produces a temporary; tensor itself remains untouched.
        device_dtype, memory = str(tensor.dtype), str(tensor.memory_config())
        readable = self.ttnn.to_memory_config(tensor, self.ttnn.DRAM_MEMORY_CONFIG) if tensor.is_sharded() else tensor
        parts = [self.ttnn.to_torch(part).clone() for part in self.ttnn.get_device_tensors(readable)]
        expected = 4 if self.multichip else 1
        if len(parts) != expected:
            raise RuntimeError(f"{name}: expected {expected} ranks, got {len(parts)}")
        effective_mode = mode if self.multichip else "replicated"
        value = _compose(parts, effective_mode)
        rank_equal = all(_bitexact(parts[0], part) for part in parts[1:])
        self.tensors[self.step][name] = {"ranks": parts, "value": value}
        self.metadata[self.step][name] = {
            "composition": effective_mode,
            "device_dtype": device_dtype,
            "device_memory": memory,
            "shape": list(value.shape),
            "rank_shapes": [list(part.shape) for part in parts],
            "rank_bitexact": rank_equal,
            "replica_check_pass": rank_equal if effective_mode == "replicated" else None,
            "cpu_diagnostic_only": effective_mode == "sum_fp32",
        }

    def state(self, state):
        self.capture("state.conv", state["conv"], "conv")
        self.capture("state.recurrent", state["recurrent"], "recurrent")


def _instrument(dec, recorder):
    """Bind hooks to this instance; never mutate a class or the TTNN namespace."""
    norm_name = "_normalized_input" if recorder.multichip else "norm"
    original_norm = getattr(dec, norm_name)
    original_linear = dec.linear
    original_moe = dec._moe
    norm_keys = {"input_layernorm.weight": "input_norm", "post_attention_layernorm.weight": "post_norm"}

    def norm(self, x, key, *args, **kwargs):
        name = norm_keys.get(key)
        if name:
            recorder.capture(name + ".input", x)
        result = original_norm(x, key, *args, **kwargs)
        if name:
            recorder.capture(name + ".output", result)
        return result

    def linear(self, x, key, *args, **kwargs):
        out_proj = key == "linear_attn.out_proj.weight"
        if out_proj:
            recorder.capture("out_proj.input", x, "concat")
        result = original_linear(x, key, *args, **kwargs)
        if out_proj:
            recorder.capture("out_proj.output", result, "sum_fp32")
            if not recorder.multichip:
                recorder.capture("out_proj.reduced", result)
        elif key == "mlp.gate.weight":
            recorder.capture("router.output", result)
        return result

    def moe(self, x, *args, **kwargs):
        recorder.capture("moe.input", x)
        result = original_moe(x, *args, **kwargs)
        recorder.capture("moe.output", result, "sum_fp32")
        if not recorder.multichip:
            recorder.capture("moe.reduced", result)
        return result

    setattr(dec, norm_name, types.MethodType(norm, dec))
    dec.linear = types.MethodType(linear, dec)
    dec._moe = types.MethodType(moe, dec)
    if recorder.multichip:
        original_reduce = dec._reduce

        def reduce(self, x, *args, **kwargs):
            result = original_reduce(x, *args, **kwargs)
            if recorder.step is not None:
                # Current linear decoder has exactly attention and MoE residual
                # reductions. Record the actual device result as well as CPU sums.
                names = ("out_proj.reduced", "moe.reduced")
                if recorder.reductions >= len(names):
                    raise RuntimeError("unexpected extra residual reduction in selected decode step")
                recorder.capture(names[recorder.reductions], result)
                recorder.reductions += 1
            return result

        dec._reduce = types.MethodType(reduce, dec)


def _run_side(ttnn, c, sd, payload, *, multichip, baseline_router, mesh_build_kwargs):
    if multichip:
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import MultichipDecoder as Decoder

        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    else:
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.optimized_decoder import OptimizedDecoder as Decoder

        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0])
    recorder = Recorder(ttnn, multichip)
    try:
        build_kwargs = dict(mesh_build_kwargs) if multichip else {}
        repair = build_kwargs.pop("functional_repair", {})
        dec = Decoder.from_state_dict(sd, hf_config=c, layer_idx=34, mesh_device=mesh, **build_kwargs)
        if dec.kind != "linear_attention" or multichip and dec.residual_layout != "replicated":
            raise RuntimeError(
                "diagnostic requires layer-34 linear attention and the current replicated residual default"
            )
        if repair:
            from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_multichip_b1_repair import (
                _install_norm,
                _install_projection,
            )

            if repair.get("native_linear_norm"):
                _install_norm(dec, full_input_shard=False, native_linear_norm=True)
            if repair.get("native_linear_projection"):
                _install_projection(dec, sd)
            dec.precision_policy["stage_diagnostic_functional_repair"] = repair
        if multichip and baseline_router:
            from models.autoports.qwen_qwen3_6_35b_a3b.tt.mesh_baseline_router_probe import install

            # The baseline-policy proxy intercepts topk. Bypass the new native
            # selector branch explicitly so this flag actually selects that path.
            dec.decode_fp32_selector = False
            install(dec)
            dec.precision_policy["stage_diagnostic_decode_fp32_selector_override"] = False
        _instrument(dec, recorder)
        if multichip:
            dec.prepare_decode_batch(1)
        state = dec.new_recurrent_state()
        plan = dec.prepare_prefill(seq_len=128)

        def upload(t):
            return ttnn.from_torch(
                t,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        # No warmup/reset: this is the original saved sequence from zero state.
        prefill = dec.prefill_forward(upload(payload["inputs"]["x"]), state=state, plan=plan)
        recorder.activate("prefill")
        recorder.capture("final.output", prefill)
        recorder.state(state)
        recorder.step = None
        for step in range(DECODE_STEPS):
            if step in CAPTURE_STEPS:
                recorder.activate(step)
            output = dec.decode_forward(upload(payload["inputs"]["xd"][step]), state=state, positions=None)
            if recorder.step is not None:
                recorder.capture("final.output", output)
                recorder.state(state)
                if multichip and recorder.reductions != 2:
                    raise RuntimeError(f"step {step}: expected two residual reductions")
                missing = set(STAGE_ORDER) - recorder.tensors[str(step)].keys()
                if missing:
                    raise RuntimeError(f"step {step}: missing stages {sorted(missing)}")
                print(f"captured {Decoder.__name__} decode step {step}", flush=True)
            recorder.step = None
        return {
            "tensors": recorder.tensors,
            "metadata": recorder.metadata,
            "decoder": Decoder.__name__,
            "precision_policy": dec.precision_policy,
        }
    finally:
        ttnn.close_mesh_device(mesh)


def _compare(sides, payload):
    comparisons = {}
    for step, baseline in sides["baseline"]["tensors"].items():
        candidate = sides["multichip"]["tensors"][step]
        stages = {
            name: _metrics(baseline[name]["value"], candidate[name]["value"])
            for name in STAGE_ORDER
            if name in baseline and name in candidate
        }
        comparisons[step] = {
            "stages": stages,
            "first_non_bitexact_stage": next((name for name, row in stages.items() if not row["bitexact"]), None),
            "first_stage_below_0_995": next(
                (name for name, row in stages.items() if row.get("pcc") is not None and row["pcc"] < 0.995), None
            ),
        }
        for side in ("baseline", "multichip"):
            saved = payload[side]["prefill"] if step == "prefill" else payload[side]["decode"][int(step)]
            comparisons[step][side + "_saved_output_reproduction"] = _metrics(
                saved, sides[side]["tensors"][step]["final.output"]["value"]
            )
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=Path, default=ROOT / "doc/optimized_multichip_decoder/repair_default_sensitive_l34.pt"
    )
    parser.add_argument("--output", type=Path, required=True, help="fresh output directory; never overwritten")
    parser.add_argument(
        "--baseline-router-selector", action="store_true", help="install diagnostic BF16 selector on mesh only"
    )
    parser.add_argument(
        "--mesh-build-kwargs",
        default="{}",
        help="constructor JSON; optional functional_repair supports native_linear_norm/native_linear_projection",
    )
    args = parser.parse_args()
    mesh_build_kwargs = json.loads(args.mesh_build_kwargs)
    if not isinstance(mesh_build_kwargs, dict):
        raise ValueError("mesh-build-kwargs must be an object")
    repair = mesh_build_kwargs.get("functional_repair", {})
    supported = {"native_linear_norm", "native_linear_projection"}
    if (
        not isinstance(repair, dict)
        or set(repair) - supported
        or any(type(value) is not bool for value in repair.values())
    ):
        raise ValueError(
            "diagnostic functional_repair accepts only boolean native_linear_norm/native_linear_projection"
        )
    payload = torch.load(args.seed, map_location="cpu", weights_only=False)
    signature = payload["signature"]
    required = {"layer": 34, "length": 128, "batch": 1, "stack_depth": 1}
    if any(signature.get(key) != value for key, value in required.items()):
        raise ValueError(f"incompatible seed signature: {signature}")
    if tuple(payload["inputs"]["x"].shape) != (1, 128, 2048) or tuple(payload["inputs"]["xd"].shape[1:]) != (
        1,
        1,
        2048,
    ):
        raise ValueError("unexpected saved activation shape")
    if payload["inputs"]["xd"].shape[0] < DECODE_STEPS:
        raise ValueError("seed does not contain 78 decode steps")
    args.output.mkdir(parents=True, exist_ok=False)
    result_path = args.output / "result.json"
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import REVISION, config, weights

    record(result_path)
    provenance_path = Path(str(result_path) + ".provenance.json")
    provenance = json.loads(provenance_path.read_text())
    provenance.update(
        execution="eager diagnostic; intentional host reads between operations; not timed and not a fallback audit",
        device="serial MeshShape(1,1) device 0 / MeshShape(1,4), FABRIC_1D for mesh",
        seed=str(args.seed.resolve()),
        seed_signature=signature,
        checkpoint_revision=REVISION,
        decode_steps=DECODE_STEPS,
        capture_steps=CAPTURE_STEPS,
        mesh_build_kwargs=mesh_build_kwargs,
        baseline_build_kwargs={},
        repair_installation="reuse instance helpers; no wrapper/class monkeypatch and no implicit production-default overrides",
        baseline_router_selector=args.baseline_router_selector,
        composition="out_proj/moe CPU FP32 partial sums are diagnostic; .reduced stages are actual device reductions",
    )
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    import ttnn

    c, sd = config(), weights(34)
    sides = {}
    for side in ("baseline", "multichip"):
        sides[side] = _run_side(
            ttnn,
            c,
            sd,
            payload,
            multichip=side == "multichip",
            baseline_router=args.baseline_router_selector,
            mesh_build_kwargs=mesh_build_kwargs,
        )
        torch.save(sides[side], args.output / f"{side}.pt")
        print(f"saved {side} diagnostic tensors", flush=True)
    replica_failures = [
        f"{side}/{step}/{name}"
        for side, data in sides.items()
        for step, entries in data["metadata"].items()
        for name, entry in entries.items()
        if entry["replica_check_pass"] is False
    ]
    result = {
        "diagnostic_only": True,
        "capture_steps": CAPTURE_STEPS,
        "passing_control_step": 0,
        "replica_check_pass": not replica_failures,
        "replica_failures": replica_failures,
        "comparisons": _compare(sides, payload),
        "stage_metadata": {side: data["metadata"] for side, data in sides.items()},
        "precision_policy": {side: data["precision_policy"] for side, data in sides.items()},
        "tensor_artifacts": ["baseline.pt", "multichip.pt"],
        "interpretation": "PCC is whole-stage output. CPU partial sums do not model device reduction order/rounding; use .reduced for that comparison. No quality gate is changed.",
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(result_path), "replica_check_pass": not replica_failures}), flush=True)
    if replica_failures:
        raise RuntimeError(f"replicated stages disagree across ranks: {replica_failures}")


if __name__ == "__main__":
    main()
