# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU counterfactuals for the exact probe_ple_decoder RNG/long-prefill fixture.

No threshold changes or acceptance classification. Actual packed-cache lookup
and fixed six-projection FP64 controls retain real BF16 experts. Device BFP8
expert arithmetic is not modeled. Selected intermediate rows are diagnostic;
all output rows participate in the PCC summaries.
"""

import argparse
import ast
import contextlib
import hashlib
import inspect
import json
import math
import sys
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer

from .probe_ple_batch_quantization import CachedRows, tensor_sha
from .probe_ple_public_backend import PROJECTIONS
from .probe_ple_quantization import SelectedRows


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_contract():
    """Compile only three pure CPU definitions; never import the TTNN harness."""
    path = Path(__file__).with_name("probe_ple_decoder.py")
    tree = ast.parse(path.read_text())
    names = {"load_reference", "streaming_pcc", "metrics"}
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == names
    namespace = dict(
        torch=torch,
        json=json,
        math=math,
        safe_open=safe_open,
        Qwen4ExpTextDecoderLayer=Qwen4ExpTextDecoderLayer,
        SelectedRows=SelectedRows,
    )
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=3707)
    parser.add_argument("--row", type=int, default=55)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--device-output-cache", type=Path)
    parser.add_argument("--saved-controls", type=Path)
    args = parser.parse_args()
    assert "ttnn" not in sys.modules
    contract = cpu_contract()
    metrics = contract["metrics"]
    torch.set_num_threads(16)  # Match the untouched literal harness exactly.
    if not 0 <= args.row < args.seq_len:
        parser.error("Selected row must be within prefill")
    identity = {
        "seed": 1731,
        "seq_len": args.seq_len,
        "decode_steps": 3,
        "threads": 16,
        "snapshot": str(args.snapshot),
        "cache_dir": str(args.cache_dir),
        "config_sha256": file_sha(args.snapshot / "config.json"),
        "hf_source_sha256": file_sha(inspect.getfile(Qwen4ExpTextDecoderLayer)),
        "manifest_sha256": file_sha(args.cache_dir / "manifest.json"),
        "probe_ple_decoder_sha256": file_sha(Path(__file__).with_name("probe_ple_decoder.py")),
        "diagnostic_sha256": file_sha(__file__),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": __doc__,
        "identity": identity,
        "device_executed": False,
        "target_row": args.row,
        "policies": {},
        "accepted": False,
    }
    if args.saved_controls:
        saved = torch.load(args.saved_controls, map_location="cpu", weights_only=False)
        for key in ("seed", "seq_len", "config_sha256", "hf_source_sha256", "manifest_sha256"):
            assert saved["identity"][key] == identity[key]
        report = saved["report"]
    else:
        torch.manual_seed(1731)
        config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
        config._attn_implementation = "eager"
        literal, weights = contract["load_reference"](args.snapshot, config)
        prompt = (torch.randn(1, args.seq_len, 10240) * 0.02).bfloat16()
        decode = [(torch.randn(1, 1, 10240) * 0.02).bfloat16() for _ in range(3)]
        ids = torch.randint(0, config.vocab_size, (1, args.seq_len + 3))
        eos = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
        ids[0, max(args.seq_len - 2, 0)] = eos
        ids[0, args.seq_len + 1] = eos
        identity.update(
            prompt_sha256=tensor_sha(prompt),
            ids_sha256=tensor_sha(ids),
            decode_sha256=[tensor_sha(value) for value in decode],
        )
        outputs, intermediates, routes, hashes = {}, {}, {}, {}
        index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
        lookup = CachedRows(
            args.snapshot, index, "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.", args.cache_dir
        )
        for policy in ("literal", "packed", "packed_six_fp64", "literal_repeat"):
            if policy == "literal":
                reference = literal
            else:
                rng = torch.random.get_rng_state()
                with torch.device("meta"):
                    reference = Qwen4ExpTextDecoderLayer(config, 1)
                torch.random.set_rng_state(rng)
                missing = reference.load_state_dict(weights, strict=False, assign=True)
                assert (
                    missing.missing_keys == ["ple.ple_embedding.ngram_embedding.weight"] and not missing.unexpected_keys
                )
                reference.ple.ple_embedding.ngram_embedding = lookup
                lookup.use_packed = policy != "literal_repeat"
                reference.eval()
            captured, router = {}, {}
            current = {}
            handles = []

            def select(value):
                if isinstance(value, (tuple, list)):
                    return [select(item) for item in value]
                if value.ndim >= 3 and value.shape[1] == current["length"]:
                    return value[:, current["row"] : current["row"] + 1].clone()
                if value.ndim >= 2 and value.shape[0] == current["length"]:
                    return value[current["row"] : current["row"] + 1].clone()
                raise ValueError(f"Unexpected selected intermediate shape {tuple(value.shape)}")

            for name in (
                "ple",
                "attn_hyper_connection",
                "linear_attn",
                "mlp_hyper_connection",
                "mlp",
                *PROJECTIONS.values(),
            ):

                def hook(_module, _inputs, output, name=name):
                    captured[name] = select(output)

                handles.append(reference.get_submodule(name).register_forward_hook(hook))

            def routing(_module, _inputs, output):
                router["indices"] = output[2].clone()
                captured["router"] = select(output)

            handles.append(reference.mlp.gate.register_forward_hook(routing))
            cache = DynamicCache(config=config)
            outputs[policy], intermediates[policy], routes[policy] = {}, {}, {}
            try:
                with contextlib.ExitStack() as stack:
                    if policy == "packed_six_fp64":
                        for name in PROJECTIONS.values():
                            module = reference.get_submodule(name)

                            def project(value, module=module):
                                assert value.dtype == module.weight.dtype == torch.bfloat16 and module.bias is None
                                return F.linear(value.double(), module.weight.double()).to(value.dtype)

                            stack.enter_context(mock.patch.object(module, "forward", project))
                    for phase, hidden, tokens in [
                        ("prefill", prompt, ids[:, : args.seq_len]),
                        *[
                            (f"decode{i}", value, ids[:, args.seq_len + i : args.seq_len + i + 1])
                            for i, value in enumerate(decode)
                        ],
                    ]:
                        current.update(length=hidden.shape[1], row=args.row if phase == "prefill" else 0)
                        captured.clear()
                        output = reference(
                            hidden, position_embeddings=(None, None), past_key_values=cache, ple_input_ids=tokens
                        )
                        outputs[policy][phase] = output
                        intermediates[policy][phase] = dict(captured)
                        routes[policy][phase] = router["indices"].clone()
                        if policy != "literal":
                            score = metrics(outputs["literal"][phase], output, hidden)
                            report["policies"].setdefault(policy, {})[phase] = score
                            if policy == "literal_repeat":
                                assert torch.equal(
                                    outputs["literal"][phase], output
                                ), "Independent literal replay differs"
                        print("CPU_POLICY_PHASE", policy, phase, flush=True)
            finally:
                for handle in handles:
                    handle.remove()
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["independent_literal_repeat_exact"] = True
        report["cache_parity_checks"] = lookup.parity_checks
        report["selected_cache_rows"] = len(lookup.ids)
        report["selected_row"] = {}
        for policy in ("packed", "packed_six_fp64"):
            target = prompt[:, args.row : args.row + 1]
            row = metrics(
                outputs["literal"]["prefill"][:, args.row : args.row + 1],
                outputs[policy]["prefill"][:, args.row : args.row + 1],
                target,
            )
            left = routes["literal"]["prefill"].reshape(args.seq_len, -1)
            right = routes[policy]["prefill"].reshape_as(left)
            row.update(
                literal_router=left[args.row].tolist(),
                control_router=right[args.row].tolist(),
                router_changed_rows=int((left != right).any(-1).sum()),
            )
            report["selected_row"][policy] = row
        saved = {
            "identity": identity,
            "report": report,
            "prompt": prompt,
            "ids": ids,
            "decode_inputs": decode,
            "outputs": outputs,
            "selected_intermediates": intermediates,
            "router_indices": routes,
        }
        args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved, args.tensor_output)
    if args.device_output_cache:
        device = torch.load(args.device_output_cache, map_location="cpu", weights_only=False)
        assert torch.equal(device["prompt"], saved["prompt"]) and torch.equal(device["ids"], saved["ids"])
        assert torch.equal(device["expected_prefill"], saved["outputs"]["literal"]["prefill"])
        report["matched_device"] = {
            policy: metrics(saved["outputs"][policy]["prefill"], device["actual_prefill"], saved["prompt"])
            for policy in ("literal", "packed", "packed_six_fp64")
        }
        report["device_cache_identity"] = device["identity"]
    assert "ttnn" not in sys.modules
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "selected_row": report.get("selected_row"),
                "literal_repeat_exact": report.get("independent_literal_repeat_exact"),
                "device_executed": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
