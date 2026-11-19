# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Strict native-pack recurrence comparison, or separate full-layer precision test.

The focused path preserves all existing exact output/state gates and records a
failure on any mismatch. --full-layer forwards the normal runner's unchanged
0.995 PCC gates; it is a separate precision experiment, not an exactness waiver.
--cpu-check imports neither torch nor TTNN.
"""

import argparse
import ast
import hashlib
import json
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

from models.autoports.qwen_qwen3_6_27b.tests import probe_multichip_recurrent as original
from models.autoports.qwen_qwen3_6_27b.tt.recurrent_pack_candidate import (
    BASELINE,
    NATIVE_BODY,
    ORIGINAL_BODY,
    VARIANT,
    native_pack_sources,
    register_variant,
)
from models.autoports.qwen_qwen3_6_27b.tt.recurrent_variants import sources as original_sources


def replace_once(source, old, new):
    assert source.count(old) == 1, (source.count(old), old)
    return source.replace(old, new, 1)


def baseline_sources(reader, compute, *, grouped, poison_dead_padding=False):
    return original_sources(reader, compute, BASELINE, grouped=grouped, poison_dead_padding=poison_dead_padding)


def hashes():
    return {
        str(path.relative_to(original.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__).resolve(),
            Path(original.__file__).resolve(),
            original.ROOT / "tt/recurrent_pack_candidate.py",
            original.ROOT / "tt/recurrent_variants.py",
            original.ROOT / "tt/multichip_decoder.py",
            original.ROOT / "tt/optimized_decoder.py",
        )
    }


def adapted_main_source():
    text = Path(original.__file__).read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    source = ast.get_source_segment(text, node)
    source = replace_once(
        source, "choices=VARIANTS, default=list(VARIANTS[1:7])", f"choices=[{VARIANT!r}], default=[{VARIANT!r}]"
    )
    source = replace_once(
        source,
        "candidate = sources(reader, compute, variant, grouped=group, poison_dead_padding=args.poison_dead_padding)",
        "baseline = _baseline_sources(reader, compute, grouped=group, poison_dead_padding=args.poison_dead_padding)\n"
        "        candidate = _native_pack_sources(*baseline)",
    )
    source = replace_once(
        source,
        "source_overrides=(reader, compute) if index == 0 else candidate",
        "source_overrides=baseline if index == 0 else candidate",
    )
    source = replace_once(
        source,
        '"grouped_qk": group, "partition": partition, "candidate_sha256": [digest(s) for s in candidate],',
        '"grouped_qk": group, "partition": partition, "candidate_sha256": [digest(s) for s in candidate],\n'
        '                 "baseline_sha256": [digest(s) for s in baseline],',
    )
    source = replace_once(
        source,
        'report = {"status": "running",',
        'report = {"native_pack": _metadata, "comparison": "cumulative_exp_scalar versus cumulative_native_pack",\n'
        '              "status": "running",',
    )
    source = replace_once(
        source,
        "                ttnn.synchronize_device(mesh)\n                tid = ttnn.begin_trace_capture(mesh, cq_id=0)",
        "                ttnn.synchronize_device(mesh)\n"
        "            # Compile/warm both programs before any active trace exists.\n"
        "            for index in range(2):\n"
        "                tid = ttnn.begin_trace_capture(mesh, cq_id=0)",
    )
    return source


def build_main():
    source = adapted_main_source()
    namespace = dict(vars(original))
    namespace.update(
        __name__=__name__ + "._snapshot",
        __doc__=__doc__,
        _baseline_sources=baseline_sources,
        _native_pack_sources=native_pack_sources,
        _metadata={
            "classification": "unproven precision candidate; strict exact gates retained",
            "source_sha256": hashes(),
            "generated_main_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "lifetime": "all input/output allocation and both program warmups precede captures",
        },
    )
    exec(compile(source, __file__ + ":adapted_main", "exec"), namespace)
    return namespace["main"]


def full_layer(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy-json", default="{}")
    parser.add_argument("--policy-file", type=Path)
    args, forwarded = parser.parse_known_args(argv)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    assert "--baseline" not in forwarded and "--legacy-decode-layout" not in forwarded
    policy = json.loads(args.policy_file.read_text() if args.policy_file else args.policy_json)
    assert policy.get("recurrent_variant", VARIANT) in (BASELINE, VARIANT)
    assert policy.get("flat_decode", True) and policy.get("batched_conv", True)
    policy.update(recurrent_variant=VARIANT, batched_conv_b1=True)
    assert not args.output.exists(), "Use a fresh output path to prevent stale evidence"
    # Device imports and registration occur only at the actual full-layer entrypoint.
    from models.autoports.qwen_qwen3_6_27b.tests import run_multichip_decoder as runner

    failure, state = None, {}
    try:
        with (
            register_variant() as state,
            patch.object(
                sys,
                "argv",
                [sys.argv[0], *forwarded, "--policy-json", json.dumps(policy), "--output", str(args.output)],
            ),
        ):
            runner.main()
            assert state["candidate_source_calls"] > 0, "Candidate unused: choose a flat GDN layer"
    except BaseException:
        failure = traceback.format_exc()
        raise
    finally:
        report = json.loads(args.output.read_text()) if args.output.exists() else {"status": "fail"}
        report["native_pack"] = {
            "classification": "precision candidate; ordinary runner PCC/trace gates unchanged",
            "effective_policy": policy,
            "registration": state,
            "source_sha256": hashes(),
            "failure": failure,
            "forwarded_arguments": forwarded,
        }
        if failure:
            report["status"] = "fail"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


def cpu_check():
    from models.autoports.qwen_qwen3_6_27b.tt import recurrent_variants

    reader, compute, _ = original.frozen_sources(original.FROZEN)
    before = recurrent_variants.sources, recurrent_variants.VARIANTS
    for grouped in (False, True):
        for poison in (False, True):
            baseline = baseline_sources(reader, compute, grouped=grouped, poison_dead_padding=poison)
            candidate = native_pack_sources(*baseline)
            assert candidate[0] == baseline[0]
            assert candidate[1].replace(NATIVE_BODY, ORIGINAL_BODY, 1) == baseline[1]
            # Entire update/readout, FP32 control, pack, address and CB protocol
            # blocks follow the helper and must remain byte-identical.
            assert candidate[1].split("// Preserve update=", 1)[1] == baseline[1].split("// Preserve update=", 1)[1]
            try:
                with register_variant() as state:
                    assert (
                        recurrent_variants.sources(
                            reader, compute, VARIANT, grouped=grouped, poison_dead_padding=poison
                        )
                        == candidate
                    )
                    assert (
                        recurrent_variants.sources(
                            reader, compute, BASELINE, grouped=grouped, poison_dead_padding=poison
                        )
                        == baseline
                    )
                    assert state == {"candidate_source_calls": 1, "other_source_calls": 1}
                    raise RuntimeError("restoration control")
            except RuntimeError as error:
                assert str(error) == "restoration control"
            assert (recurrent_variants.sources, recurrent_variants.VARIANTS) == before
    build_main()
    # Exercise real CLI forwarding and setup registration using a stdlib-only
    # runner double; no device imports or calls occur in this CPU check.
    import tempfile
    from types import ModuleType

    module_name = "models.autoports.qwen_qwen3_6_27b.tests.run_multichip_decoder"
    fake = ModuleType(module_name)
    observed = []

    def fake_main():
        arguments = sys.argv[1:]
        policy = json.loads(arguments[arguments.index("--policy-json") + 1])
        assert policy["recurrent_variant"] == VARIANT and policy["batched_conv_b1"] is True
        assert policy["local_fused_conv"] is True
        observed.append(arguments)
        recurrent_variants.sources(reader, compute, VARIANT, grouped=False)
        target = Path(arguments[arguments.index("--output") + 1])
        target.write_text(json.dumps({"status": "pass"}))

    fake.main = fake_main
    with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {module_name: fake}):
        target = Path(directory) / "report.json"
        full_layer(["--output", str(target), "--policy-json", '{"local_fused_conv":true}', "--batch", "1"])
        report = json.loads(target.read_text())
        assert report["status"] == "pass"
        assert report["native_pack"]["registration"]["candidate_source_calls"] == 1
        assert "--batch" in observed[0]
    assert (recurrent_variants.sources, recurrent_variants.VARIANTS) == before
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    print(
        "PASS: four source controls, body-only delta, unchanged FP32/address/CB blocks, registry restoration, exact probe AST; no TTNN/torch import"
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--cpu-check"]:
        cpu_check()
    elif "--full-layer" in sys.argv[1:]:
        full_layer([arg for arg in sys.argv[1:] if arg != "--full-layer"])
    else:
        build_main()()
