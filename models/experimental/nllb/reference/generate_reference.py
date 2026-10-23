# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Generate an independent, local-assets-only CUDA FP32 portable NLLB fixture."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import socket
import sys
import time

import numpy as np

MANIFEST_SHA256 = "6e84dc434749506886c1720312df3c7dcebb3b6cab2b3acdeb96abdcc9d842cf"
HELPER_SHA256 = "fe4dccc800dd915f0964a76a85c0c095f6bfc6286950557390961d272bd00630"
TOKENIZER_FILES = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def load_helper(directory, pin):
    path = Path(directory) / "reference/envelope_regression.py"
    if sha(path) != pin:
        raise ValueError("portable helper SHA256 mismatch")
    spec = importlib.util.spec_from_file_location("_portable_nllb_envelope_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_assets(manifest, pin, model, checkpoint, tokenizer):
    """Verify every consumed official asset, including index and tokenizer extras."""
    if sha(manifest) != pin:
        raise ValueError("asset manifest SHA256 mismatch")
    document = json.loads(Path(manifest).read_text(), object_pairs_hook=unique)
    if document.get("schema") != "nllb-local-reference-assets-v1":
        raise ValueError("unknown asset manifest schema")
    spec = document["models"][model]
    if not spec["repo_id"].startswith("facebook/nllb-200") or not re.fullmatch("[0-9a-f]{40}", spec["revision"]):
        raise ValueError("invalid official repository identity")
    checkpoint, tokenizer = Path(checkpoint).resolve(), Path(tokenizer).resolve()
    if not checkpoint.is_dir() or not tokenizer.is_dir():
        raise ValueError("materialized local checkpoint/tokenizer directories required")
    weights = set(spec["weight_files"])
    required = weights | TOKENIZER_FILES | {"config.json", "generation_config.json"}
    if spec["weight_index"]:
        required.add(spec["weight_index"])
    if set(spec["files"]) != required or not weights:
        raise ValueError("manifest asset inventory mismatch")
    hashes = {}
    for name, expected in spec["files"].items():
        if Path(name).name != name or not re.fullmatch("[0-9a-f]{64}", expected["sha256"]):
            raise ValueError("invalid manifest filename/hash")
        root = tokenizer if name in TOKENIZER_FILES else checkpoint
        path = root / name
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError("asset missing or resolves outside directory: " + name)
        if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError("asset bytes/SHA256 mismatch: " + name)
        hashes[name] = expected["sha256"]
    # HF tokenizers can consume optional local files. Never silently consume an unpinned one.
    for root in {checkpoint, tokenizer}:
        extras = {
            p.name for p in root.iterdir() if p.suffix in {".json", ".bin", ".model", ".txt", ".safetensors"}
        } - required
        if extras:
            raise ValueError("untracked loader assets: " + ", ".join(sorted(extras)))
    if spec["weight_index"]:
        index = json.loads((checkpoint / spec["weight_index"]).read_text(), object_pairs_hook=unique)
        if not index.get("weight_map") or set(index["weight_map"].values()) != weights:
            raise ValueError("index shard inventory mismatch")
    elif weights != {"pytorch_model.bin"}:
        raise ValueError("single-file checkpoint inventory mismatch")
    return spec, hashes


def require_compute_host(execution, hostname, environ, cuda_available):
    host = hostname.lower().split(".")[0]
    if "login" in host or host.startswith("lo-"):
        raise RuntimeError("reference inference is forbidden on login nodes")
    if execution == "slurm" and not environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm compute allocation required")
    if execution not in ("slurm", "direct") or not cuda_available:
        raise RuntimeError("dedicated CUDA compute host required; no CPU fallback")


def generation_policy(generation, config):
    """Inspect runtime-inherited None defaults without modifying the loaded policy."""
    expected = dict(
        forced_eos_token_id=None,
        suppress_tokens=None,
        begin_suppress_tokens=None,
        bad_words_ids=None,
        force_words_ids=None,
        min_length=0,
        min_new_tokens=None,
        no_repeat_ngram_size=0,
        encoder_no_repeat_ngram_size=0,
        repetition_penalty=1.0,
    )
    raw = {key: getattr(generation, key, default) for key, default in expected.items()}
    provider = getattr(generation, "_get_default_generation_params", None)
    defaults = provider() if callable(provider) else {}
    if not isinstance(defaults, dict):
        raise ValueError("invalid generation default provider")
    effective = {key: defaults.get(key, value) if value is None else value for key, value in raw.items()}
    if effective != expected:
        raise ValueError("nondefault forcing/repetition policy: " + json.dumps(effective))
    for name in ("decoder_start_token_id", "eos_token_id", "pad_token_id"):
        if getattr(generation, name, None) != config[name]:
            raise ValueError("generation special token mismatch: " + name)
    return dict(
        raw=raw,
        effective=effective,
        inherited_defaults={k: effective[k] for k in raw if raw[k] is None},
        default_provider="_get_default_generation_params" if callable(provider) else None,
    )


def make_identity(helper, config_path, inputs, metadata, weights):
    return dict(
        config_sha256=sha(config_path),
        input_hashes={k: helper.array_hash(v) for k, v in inputs.items()},
        target_id=metadata["target_id"],
        weight_sha256=weights,
        requests={
            case: [r[0] for r in helper.requests(case, inputs, metadata["source_lengths"])] for case in helper.CASES
        },
    )


def qualify(helper, outputs, events, inputs, metadata, config):
    """Natural, observed coverage; early EOS never manufactures boundary coverage."""
    expected = {
        f"{case}__{name}": (len(ids), cap)
        for case in helper.CASES
        for name, ids, mask, cap in helper.requests(case, inputs, metadata["source_lengths"])
    }
    if set(outputs) != set(expected) or set(events) != set(expected):
        raise ValueError("missing or extra generation requests")
    rows = {}
    for key, (batch, cap) in expected.items():
        rows[key] = helper.canonical(outputs[key], batch, metadata["target_id"], cap, config)
        if events[key] != [[n, batch] for n in range(1, outputs[key].shape[1])]:
            raise ValueError("actual HF full-prefix forward observations disagree: " + key)
    base = rows["base__cap64"]
    covered = (
        all(len(base[i]) == 65 for i in (0, 1))
        and all([n, 4] in events["base__cap64"] for n in (63, 64))
        and any(row[-1] == config["eos_token_id"] for row in base[2:])
    )
    return dict(
        covered=covered,
        source_lengths=metadata["source_lengths"],
        actual_forward_prefixes=events,
        actual_new_tokens={k: [len(r) - 1 for r in v] for k, v in rows.items()},
        natural_eos=[r[-1] == config["eos_token_id"] for r in base],
    )


def make_fixture(identity, outputs, metadata):
    # Publication state belongs to the sidecar, not the independently pinned fixture.
    metadata = {key: value for key, value in metadata.items() if key not in ("completed", "fixture_sha256", "error")}
    return dict(
        schema="nllb-portable-envelope-fp32-v1",
        identity=identity,
        precision="fp32",
        tf32=False,
        outputs={key: value.tolist() for key, value in outputs.items()},
        metadata=metadata,
    )


def publish(path, fixture, qualified):
    """Never replace an earlier fixture or publish uncovered/partially generated output."""
    if not qualified:
        raise ValueError("natural decoder boundary/EOS coverage incomplete; fixture not published")
    payload = json.dumps(fixture, indent=2) + "\n"
    if len(payload.encode()) > 1024 * 1024:
        raise ValueError("fixture exceeds portable verifier size limit")
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    owned = False
    try:
        with temporary.open("x") as stream:
            owned = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # atomic, exclusive publication: never clobber a prior pin
    finally:
        if owned:
            temporary.unlink(missing_ok=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=("600m", "1.3b-distilled", "3.3b"), required=True)
    p.add_argument("--checkpoint", required=True, help="materialized local official HF checkpoint directory")
    p.add_argument("--tokenizer-directory", help="defaults to checkpoint directory")
    p.add_argument(
        "--package-directory", required=True, help="NLLB package root containing reference/envelope_regression.py"
    )
    p.add_argument("--helper-sha256", default=HELPER_SHA256)
    p.add_argument("--manifest", default=str(Path(__file__).with_name("official-assets.json")))
    p.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    p.add_argument("--execution", choices=("direct", "slurm"), required=True)
    p.add_argument("--device", type=int, default=0, help="visible CUDA device index")
    p.add_argument("--output", required=True, help="new fixture JSON path; existing files are never overwritten")
    return p


def load_tokenizer(factory, directory):
    return factory.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False, token=False, src_lang="eng_Latn", use_fast=True
    )


def loading_metadata(value):
    """Copy HF loading diagnostics to JSON; sets have deterministic list order."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: loading_metadata(value[key]) for key in sorted(value)}
    if type(value) in (list, tuple):
        return [loading_metadata(item) for item in value]
    if type(value) is set:
        return sorted((loading_metadata(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    raise TypeError("unsupported loading metadata type: " + type(value).__name__)


def load_fp32_model(factory, torch, checkpoint, device):
    model, loading = factory.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
        token=False,
        use_safetensors=False,
        weights_only=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("checkpoint did not load exactly: " + str(loading))
    model = model.to(torch.device("cuda", device)).eval()
    parameters = list(model.parameters())
    if not parameters or any(p.dtype != torch.float32 or p.device.type != "cuda" for p in parameters):
        raise ValueError("all learned parameters must be CUDA FP32")
    return model, loading_metadata(loading)


def run(args, report):
    require_compute_host(args.execution, socket.gethostname(), os.environ, True)
    if args.device < 0:
        raise ValueError("negative CUDA index")
    # Set before importing Torch or initializing CUDA. These are reproducibility settings, not model policy.
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in (":4096:8", ":16:8"):
        raise ValueError("unsupported deterministic CUBLAS_WORKSPACE_CONFIG")
    helper = load_helper(args.package_directory, args.helper_sha256)
    tokenizer_path = args.tokenizer_directory or args.checkpoint
    spec, hashes = validate_assets(args.manifest, args.manifest_sha256, args.model, args.checkpoint, tokenizer_path)
    config_path = Path(args.checkpoint) / "config.json"
    config = json.loads(config_path.read_text(), object_pairs_hook=unique)
    import torch
    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    require_compute_host(args.execution, socket.gethostname(), os.environ, torch.cuda.is_available())
    torch.cuda.set_device(args.device)
    torch.manual_seed(1729)
    np.random.seed(1729)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    tokenizer = load_tokenizer(AutoTokenizer, tokenizer_path)
    inputs, metadata = helper.build_inputs(tokenizer, config)
    identity = make_identity(helper, config_path, inputs, metadata, {n: hashes[n] for n in spec["weight_files"]})
    report.update(
        identity=identity,
        repo_id=spec["repo_id"],
        revision=spec["revision"],
        asset_sha256=hashes,
        helper_sha256=args.helper_sha256,
        manifest_sha256=args.manifest_sha256,
        generator_sha256=sha(__file__),
        python=platform.python_version(),
        torch=torch.__version__,
        transformers=transformers.__version__,
        cuda=torch.version.cuda,
        hostname=socket.gethostname(),
        gpu=torch.cuda.get_device_name(args.device),
        device=args.device,
        execution=args.execution,
        deterministic_algorithms=True,
        cublas_workspace_config=workspace,
        tf32=False,
        use_cache=False,
        precision="fp32",
        source_metadata=metadata,
    )
    started = time.monotonic()
    model, loading = load_fp32_model(AutoModelForSeq2SeqLM, torch, args.checkpoint, args.device)
    report["loading_info"] = loading
    report["generation_policy"] = generation_policy(model.generation_config, config)
    report["load_seconds"] = time.monotonic() - started
    report["generation_config"] = model.generation_config.to_dict()
    outputs, events, timings = {}, {}, {}
    with torch.inference_mode():
        for case in helper.CASES:
            for name, ids, mask, cap in helper.requests(case, inputs, metadata["source_lengths"]):
                key = case + "__" + name
                print(json.dumps(dict(phase="generate", request=key, cap=cap)), flush=True)
                observed = []

                def observe(module, positional, kwargs):
                    prefix = kwargs.get("decoder_input_ids")
                    if prefix is None or kwargs.get("past_key_values") is not None:
                        raise ValueError("missing full decoder prefix or unexpected cache")
                    observed.append([int(prefix.shape[1]), int(prefix.shape[0])])

                hook = model.register_forward_pre_hook(observe, with_kwargs=True)
                device_ids = torch.tensor(ids.copy(), dtype=torch.long, device="cuda")
                device_mask = torch.tensor(mask.copy(), dtype=torch.long, device="cuda")
                before = (device_ids.clone(), device_mask.clone())
                try:
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    tokens = model.generate(
                        input_ids=device_ids,
                        attention_mask=device_mask,
                        forced_bos_token_id=metadata["target_id"],
                        forced_eos_token_id=None,
                        max_new_tokens=cap,
                        do_sample=False,
                        num_beams=1,
                        use_cache=False,
                        num_return_sequences=1,
                        return_dict_in_generate=False,
                    )
                    torch.cuda.synchronize()
                    timings[key] = time.monotonic() - started
                    if not torch.equal(before[0], device_ids) or not torch.equal(before[1], device_mask):
                        raise ValueError("reference mutated inputs")
                    if tokens.dtype != torch.int64:
                        raise ValueError("reference generated non-int64 tokens")
                    outputs[key] = tokens.detach().cpu().numpy().copy()
                    events[key] = observed
                finally:
                    hook.remove()
    report["qualification"] = qualify(helper, outputs, events, inputs, metadata, config)
    report["request_seconds"] = timings
    report["generation_arguments"] = dict(
        forced_bos_token_id=metadata["target_id"],
        forced_eos_token_id=None,
        max_new_tokens="per request 63/64",
        do_sample=False,
        num_beams=1,
        use_cache=False,
        num_return_sequences=1,
        return_dict_in_generate=False,
    )
    validate_assets(args.manifest, args.manifest_sha256, args.model, args.checkpoint, tokenizer_path)
    fixture = make_fixture(identity, outputs, report)
    publish(args.output, fixture, report["qualification"]["covered"])
    report.update(completed=True, fixture_sha256=sha(args.output))


def main(argv=None):
    args = parser().parse_args(argv)
    path = Path(args.output)
    report_path = path.with_name(path.name + ".report.json")
    # Durable exclusive intent; a failed run requires a fresh output path, never a blind overwrite.
    report = dict(completed=False, scope="Independent FP32 reference only; not TT acceptance.")
    with report_path.open("x") as stream:
        stream.write(json.dumps(report) + "\n")
    try:
        if path.exists():
            raise FileExistsError("fixture already exists")
        run(args, report)
        return 0
    except Exception as error:
        report.update(completed=False, error=f"{type(error).__name__}: {error}")
        return 2
    finally:
        pending = report_path.with_name(report_path.name + ".pending")
        pending.write_text(json.dumps(report, indent=2) + "\n")
        pending.replace(report_path)
        print(json.dumps(dict(completed=report["completed"], report=str(report_path))), flush=True)


if __name__ == "__main__":
    sys.exit(main())
