# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""C04 CPU fixtures and numerical checks. No device imports or native certification.

The signed diagnostic follows standard_exp_test_utils, with distinct KV heads and
128-key chunks. The original random BF16 inputs and FP32 SDPA oracle are retained.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

SOURCE_PATHS = (
    "models/tt_dit/encoders/gemma/model_gemma.py",
    "models/tt_dit/tests/encoders/gemma/test_gemma_native_gqa.py",
    "models/tt_dit/tests/encoders/gemma/gqa_requalification.py",
    "ttnn/cpp/ttnn/operations/transformer/sdpa/sdpa.cpp",
    "ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_program_factory.cpp",
    "ttnn/cpp/ttnn/operations/transformer/sdpa/device/sdpa_device_operation.cpp",
    "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/sdpa.cpp",
    "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp",
    "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp",
    "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/reader_interleaved.cpp",
)

SEQ, HEAD_DIM = 1024, 256
ORIGINAL_CASES = tuple(f"tp{16 // h}-{m}" for h in (2, 4) for m in ("causal", "leftpad"))
DIAGNOSTIC_CASES = ("signed-unit-one-finite", "signed-scaled-multi-inf", "signed-unit-multi-finite", "one-visible-key")
ALL_CASES = ORIGINAL_CASES + DIAGNOSTIC_CASES
MODES = ("accurate", "approximate")
PRECISION = {
    "dtype": "BF16",
    "math_fidelity": "HiFi2",
    "math_approx_mode": False,
    "fp32_dest_acc_en": True,
    "packer_l1_acc": True,
    "q_chunk_size": 128,
    "k_chunk_size": 128,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


@dataclass(frozen=True)
class Recipe:
    case: str
    exp_mode: str = "accurate"

    def __post_init__(self):
        require(self.case in ALL_CASES and self.exp_mode in MODES, "unknown case or exp mode")

    @property
    def heads(self):
        return 2 if self.case.startswith("tp8") else 4

    @property
    def keys(self):
        return 128 if self.case == "signed-unit-one-finite" else SEQ

    @property
    def scale(self):
        return 1.0 if self.case in ("signed-unit-one-finite", "signed-unit-multi-finite") else 1 / 16

    @property
    def causal(self):
        return self.case.endswith("-causal")

    def identity(self):
        return {
            **asdict(self),
            **PRECISION,
            "exp_approx_mode": self.exp_mode == "approximate",
            "q_shape": [1, self.heads, SEQ, HEAD_DIM],
            "kv_shape": [1, self.heads // 2, self.keys, HEAD_DIM],
            "scale": self.scale,
            "is_causal": self.causal,
        }


def dense_oracle(q, k, v, mask, scale, causal):
    """Independent FP64 dense heads; no SDPA or repeat_interleave in this oracle.

    Empty masked rows produce zero, matching the existing FP32 SDPA definition.
    They remain subject to all-output finiteness, but not leftpad token scoring.
    """
    require(all(x.dtype == torch.bfloat16 for x in (q, k, v)), "oracle requires BF16 operands")
    require(mask is None or mask.dtype == torch.bfloat16, "oracle requires BF16 mask")
    outputs = []
    group = q.shape[1] // k.shape[1]
    for head in range(q.shape[1]):
        kv_head = head // group
        scores = (q[:, head].double() @ k[:, kv_head].double().transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask[:, 0].double()
        if causal:
            scores = scores.masked_fill(torch.ones(scores.shape[-2:], dtype=torch.bool).triu(1), -torch.inf)
        empty = torch.isneginf(scores).all(dim=-1, keepdim=True)
        probabilities = torch.softmax(scores.masked_fill(empty, 0), dim=-1).masked_fill(empty, 0)
        outputs.append(probabilities @ v[:, kv_head].double())
    return torch.stack(outputs, dim=1)


def fp32_oracle(q, k, v, mask, scale, causal):
    return torch.nn.functional.scaled_dot_product_attention(
        q.float(),
        k.float().repeat_interleave(2, dim=1),
        v.float().repeat_interleave(2, dim=1),
        attn_mask=None if mask is None else mask.float(),
        is_causal=causal,
        scale=scale,
    )


def make_sample(recipe, index):
    require(type(index) is int and index in (0, 1), "expected A or B fixture")
    seed, real_tokens = ((11, 257), (29, 613))[index]
    heads, keys = recipe.heads, recipe.keys
    if recipe.case in ORIGINAL_CASES:
        generator = torch.Generator().manual_seed(seed)
        q = torch.randn((1, heads, SEQ, HEAD_DIM), generator=generator).bfloat16()
        k = torch.randn((1, heads // 2, keys, HEAD_DIM), generator=generator).bfloat16()
        v = torch.randn((1, heads // 2, keys, HEAD_DIM), generator=generator).bfloat16()
        mask = None
        if not recipe.causal:
            causal = torch.full((SEQ, SEQ), -torch.inf).triu(1)[None, None]
            valid = torch.arange(SEQ) >= SEQ - real_tokens
            mask = (causal + torch.where(valid[None, None, None, :], 0.0, -torch.inf)).bfloat16()
    else:
        rows, key, dim = torch.arange(SEQ), torch.arange(keys), torch.arange(HEAD_DIM)
        q = torch.zeros((1, heads, SEQ, HEAD_DIM))
        k = torch.zeros((1, heads // 2, keys, HEAD_DIM))
        v = torch.empty_like(k)
        for head in range(heads):
            q[0, head, :, 0] = ((rows + head * 7 + index * 3) % 64 + 1).float() / 4
        for head in range(heads // 2):
            k[0, head, :, 0] = ((key % 64) - 63).float() / 16 + head / 4
            values = (((key[:, None] % 64) * 17 + dim[None, :] * 13 + head * 19 + index * 7) % 61) - 30
            v[0, head] = values.float() / 32
        penalty = -16.0 if recipe.case.endswith("finite") else -torch.inf
        mask = torch.zeros((1, 1, SEQ, keys))
        mask[..., :128:8] = penalty
        if keys > 128:
            k[:, :, 128:, 0] += 1
            v[:, :, 128:] *= 4
            mask[:, :, :128, 128:] = penalty
            mask[:, :, 128:, 256:] = penalty
        if recipe.case == "one-visible-key":
            mask.fill_(-torch.inf)
            mask[..., 127 + index] = 0
        q, k, v, mask = (x.bfloat16() for x in (q, k, v, mask))
    return {
        "seed": seed,
        "q": q,
        "k": k,
        "v": v,
        "mask": mask,
        "real_tokens": real_tokens if recipe.case.endswith("leftpad") else SEQ,
    }


def metrics(expected, actual):
    require(expected.shape == actual.shape, "metric shape mismatch")
    a, b = expected.double().flatten(), actual.double().flatten()
    require(a.numel() > 1 and torch.isfinite(a).all() and torch.isfinite(b).all(), "nonfinite or empty vectors")
    ac, bc = a - a.mean(), b - b.mean()
    require(ac.square().sum() > 0 and bc.square().sum() > 0, "constant vectors")
    gain = torch.dot(ac, bc) / torch.dot(ac, ac)
    return {
        "pcc": (torch.dot(ac, bc) / (ac.norm() * bc.norm())).item(),
        # Preserve original unbiased std normalization, rather than upstream NL2.
        "relative_rmse": ((b - a).square().mean().sqrt() / a.std()).item(),
        "gain": gain.item(),
        "bias": (b.mean() - gain * a.mean()).item(),
        "max_abs": (b - a).abs().max().item(),
    }


def gate(values):
    require(values["pcc"] >= 0.999 and values["relative_rmse"] <= 0.02, f"C04 oracle gate: {values}")


def score(sample, actual, reference_key="reference"):
    reference = sample[reference_key]
    require(actual.shape == reference.shape and torch.isfinite(actual).all(), "shape or all-output finiteness")
    first = SEQ - sample["real_tokens"]
    expected, observed = reference[..., first:, :], actual[..., first:, :]
    aggregate = metrics(expected, observed)
    heads = [metrics(expected[:, h], observed[:, h]) for h in range(expected.shape[1])]
    # Row diagnostics do not replace the original aggregate gate.
    rows = sorted(
        {
            r
            for r in (127, 128, 129, 256, 257, 612, 613, first, first + 127, first + 128, first + 129)
            if first <= r < SEQ
        }
    )
    return {
        **aggregate,
        "per_head": heads,
        "rows": {str(r): metrics(reference[..., r, :], actual[..., r, :]) for r in rows},
    }


def prepare_fixture(recipe):
    cases = []
    for index in (0, 1):
        sample = make_sample(recipe, index)
        args = [sample[k] for k in ("q", "k", "v", "mask")]
        sample["reference"] = fp32_oracle(*args, recipe.scale, recipe.causal)
        sample["reference_fp64"] = dense_oracle(*args, recipe.scale, recipe.causal)
        sample["oracle_comparison"] = score(sample, sample["reference"], "reference_fp64")
        gate(sample["oracle_comparison"])
        if recipe.case == "one-visible-key":
            expected = (
                sample["v"][:, :, 127 + index : 128 + index].repeat_interleave(2, dim=1).expand_as(sample["reference"])
            )
            require(torch.equal(sample["reference"], expected), "one-visible-key must return V exactly")
        cases.append(sample)
    # Fixture identity deliberately excludes mode: both modes consume identical bytes.
    return {"case": recipe.case, "cases": cases, "geometry": Recipe(recipe.case).identity()}


def validate_fixture(fixture, recipe):
    require(
        fixture["case"] == recipe.case and fixture["geometry"] == Recipe(recipe.case).identity(), "fixture geometry"
    )
    require(len(fixture["cases"]) == 2, "fixture must contain A and B")
    for index, sample in enumerate(fixture["cases"]):
        wanted = make_sample(recipe, index)
        for key, value in wanted.items():
            actual = sample[key]
            require(
                torch.equal(value, actual) and value.dtype == actual.dtype
                if isinstance(value, torch.Tensor)
                else value == actual,
                f"fixture input mismatch: {key}",
            )
        args = [sample[k] for k in ("q", "k", "v", "mask")]
        independent = dense_oracle(*args, recipe.scale, recipe.causal)
        require(torch.equal(independent, sample["reference_fp64"]), "stale FP64 oracle")
        require(torch.equal(fp32_oracle(*args, recipe.scale, recipe.causal), sample["reference"]), "stale FP32 oracle")
        gate(score(sample, sample["reference"], "reference_fp64"))


def verify_outputs(result, fixture, recipe):
    """Numerical CPU contract only, including when inputs are copied oracles."""
    require(result["recipe"] == recipe.identity(), "wrong mode/precision/geometry")
    replay = result["replay_outputs"]
    require([r["fixture_index"] for r in replay] == [0, 1, 0], "missing/duplicate A/B/A")
    require(set(result["eager_outputs"]) == {"expanded", "native"}, "missing/extra eager route")
    values = []
    for index, row in enumerate(replay):
        require(set(row) == {"fixture_index", "expanded", "native"}, "missing/extra replay route")
        sample = fixture["cases"][row["fixture_index"]]
        for route in ("expanded", "native"):
            value = score(sample, row[route])
            gate(value)
            values.append({"replay": index, "route": route, **value})
        require(torch.equal(row["expanded"], row["native"]), "route parity")
    for route in ("expanded", "native"):
        require(torch.equal(replay[0][route], result["eager_outputs"][route]), "eager/replay mismatch")
        require(torch.equal(replay[0][route], replay[2][route]), "restored A mismatch")
        require(not torch.equal(replay[0][route], replay[1][route]), "stale B")
    return {"status": "cpu_numerical_contract_pass", "native_qualified": False, "metrics": values}


def validate_native_binding(result, provenance, execution, recipe, result_path, fixture_path, root, artifacts):
    """C04 bindings on top of task38's validated manifest/request_provenance.

    This is not an attestation producer. Native owners supply independent executed
    recipe records and raw logs after completion; schema-valid JSON is not proof
    of its producer's correctness. Synthetic tests exercise rejection only.
    """
    require(result.get("status") == "collected" and result.get("provenance") == provenance, "execution provenance")
    require(
        execution.get("status") == "completed" and execution.get("kind") == "native_execution",
        "execution pending or synthetic",
    )
    require(execution.get("provenance") == provenance, "receipt provenance")
    require(result.get("case") == recipe.case, "result case identity")
    arithmetic = provenance["execution_configuration"]["arithmetic"]
    require(
        arithmetic["math_fidelity"] == "HiFi2"
        and arithmetic["approximate_math"] is False
        and all(arithmetic[k] is True for k in ("fp32_destination", "packer_requested", "packer_effective"))
        and all(arithmetic["dtypes"].get(k) == "BF16" for k in ("q", "k", "v", "mask")),
        "manifest precision differs from C04 recipe",
    )
    require(
        provenance["execution_configuration"]["chunk_dispatch"]["selected_paths"]
        == [Recipe(name, recipe.exp_mode).identity() for name in ALL_CASES],
        "manifest executed recipe matrix",
    )
    build, source = provenance["build"], provenance["source"]
    require(build["status"] == "attested", "native build unattested")
    require(
        all(
            build.get(k)
            for k in (
                "cpp_id",
                "data_id",
                "compiler",
                "python_environment",
                "tracy_revision",
                "runtime_firmware_driver",
                "binary_hashes",
                "attestation",
            )
        ),
        "incomplete native identity",
    )
    require(build["attestation"] in artifacts, "missing hashed attestation")
    require(
        result["commit"] == source["commit"] and result["tracked_diff_sha256"] == source["dirty_patch_sha256"],
        "source identity",
    )
    require(
        set(result["source_sha256"]) == set(SOURCE_PATHS)
        and all(source["source_files"].get(k) == v for k, v in result["source_sha256"].items()),
        "source file identity",
    )
    require(result["native_binary_sha256"] == build["binary_hashes"], "native binary identity")
    require(result["recipe"] == recipe.identity(), "executed recipe")
    require(result["fixture_sha256"] == digest(fixture_path), "fixture identity")
    samples = [{"request_id": result["request_id"], "index": i, "seed": seed} for i, seed in enumerate((11, 29, 11))]
    require(
        isinstance(result["request_id"], str) and result["request_id"] and result["samples"] == samples,
        "sample identity",
    )

    def linked(link):
        path = (root / link["name"]).resolve()
        require(path.is_relative_to(root.resolve()) and link["name"] in artifacts, "unbound receipt artifact")
        require(
            path.is_file() and path.stat().st_size == link["bytes"] > 0 and digest(path) == link["sha256"],
            "overwritten raw artifact",
        )
        require(
            artifacts[link["name"]]["sha256"] == link["sha256"] and artifacts[link["name"]]["bytes"] == link["bytes"],
            "receipt/manifest mismatch",
        )
        return path

    linked(execution["raw_log"])
    entries = execution["results"]
    names = [entry["recipe"]["case"] for entry in entries]
    require(len(names) == len(ALL_CASES) and set(names) == set(ALL_CASES), "incomplete/duplicate execution matrix")
    for entry in entries:
        require(
            entry["recipe"] == Recipe(entry["recipe"]["case"], recipe.exp_mode).identity(),
            "receipt recipe mode/precision",
        )
    entry = next(row for row in entries if row["recipe"]["case"] == recipe.case)
    require(linked(entry["output"]) == result_path.resolve(), "raw result binding")
    require(linked(entry["fixture"]) == fixture_path.resolve(), "raw fixture binding")
    require(entry["samples"] == samples and entry["routes"] == ["expanded", "native"], "executed sample/route scope")


def verify_suite(result_dir, fixture_dir, mode, manifest_path, execution_path):
    """Only this full-matrix entry point can emit an accurate component verdict.

    Import the combined task38 adapter at runtime; its absence fails closed. Do
    not vendor a weaker adapter on the pinned historical base. Both manifest and
    execution paths belong to the independently prepared native-owner package.
    """
    report_path = result_dir / f"qualification-{mode}.json"
    report_path.unlink(missing_ok=True)
    require(mode in MODES, "unknown mode")
    require(
        manifest_path is not None and execution_path is not None,
        "task38 manifest and native execution receipt required",
    )
    from models.tt_dit.utils.acceptance.adapters import request_provenance, validate_manifest

    manifest_bytes = manifest_path.read_bytes()
    manifest = strict_json(manifest_bytes)
    root = manifest_path.parent.resolve()
    validate_manifest(manifest, artifact_root=root)
    provenance = request_provenance(manifest)
    artifacts = {row["path"]: row for row in manifest["artifacts"]}
    execution_name = str(execution_path.resolve().relative_to(root))
    require(
        execution_name in artifacts and artifacts[execution_name]["sha256"] == digest(execution_path),
        "unbound execution receipt",
    )
    execution_bytes = execution_path.read_bytes()
    require(
        hashlib.sha256(execution_bytes).hexdigest() == artifacts[execution_name]["sha256"], "changed execution receipt"
    )
    execution = strict_json(execution_bytes)
    reports = []
    # Hash before loading and again before publication, so altered raw files cannot
    # leave a success report. Re-verification always removes an earlier report.
    hashes = {str(manifest_path): hashlib.sha256(manifest_bytes).hexdigest()}
    # Recheck every canonical manifest artifact, including attestation and raw
    # log, against its declared hash at the end, not a later untrusted snapshot.
    hashes.update({str(root / row["path"]): row["sha256"] for row in manifest["artifacts"]})
    bound_paths = []
    for name in ALL_CASES:
        bound_paths.extend((result_dir / mode / f"{name}.pt", fixture_dir / f"{name}.pt"))
    for path in bound_paths:
        require(str(path.resolve()) in hashes, "unbound fixture/result")
    for name in ALL_CASES:
        recipe = Recipe(name, mode)
        result_path, fixture_path = result_dir / mode / f"{name}.pt", fixture_dir / f"{name}.pt"
        result = torch.load(result_path, map_location="cpu", weights_only=True)
        fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
        validate_native_binding(result, provenance, execution, recipe, result_path, fixture_path, root, artifacts)
        validate_fixture(fixture, recipe)
        report = verify_outputs(result, fixture, recipe)
        reports.append({"case": name, "recipe": recipe.identity(), "metrics": report["metrics"]})
    require(
        all(digest(Path(path)) == value for path, value in hashes.items()),
        "raw evidence changed during verification",
    )
    report = {
        "status": "accurate_component_pass" if mode == "accurate" else "approximate_diagnostic_pass",
        "quality_pass": mode == "accurate",
        "scope": "C04 component only; independent native receipt producer review required",
        "pipeline_qualified": False,
        "provenance": provenance,
        "hashes": hashes,
        "cases": reports,
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
